"""Replay HNCO source-energy candidates and finite fixed-network evidence."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import pending_candidate as candidate
import pending_molecule_inputs as source
import pending_equilibrium_impact as impact
from curated_adoption_history import historical_source

SLUG = "hnco"
DATA = ROOT / f"tests/data/validation/{SLUG}_candidate/20261106"
IMPACT = ROOT / f"tests/data/validation/{SLUG}_equilibrium_impact/20261106"


def test_source_ledger_and_physical_anchor():
    model = json.loads((DATA / "candidate.json").read_text())
    assert model == source.build(SLUG)
    record = model["source_records"][SLUG]
    assert record["delta_GH_J_mol"] == pytest.approx(-17260.0, abs=1e-8, rel=0)
    assert model["production_change"] == "none"
    if SLUG == "p4s3":
        assert record["janaf_D0_J_mol"] is None and record["selected_D0_J_mol"] is None
    if SLUG == "cis_h2n2":
        assert not record["nasa_same_identity"]
        with pytest.raises(ValueError):
            source.nasa_spec(record)
    if SLUG == "nco":
        assert model["special_controls"]["dissociation_delta_GH_J_mol"] == -14724.0
        assert any(
            "nco_candidate/20261106/source/hnco_source_review/" in p
            for p in model["source_sha256"]
        )
        assert not any(
            "/validation/hnco_source_review/" in p for p in model["source_sha256"]
        )


def test_dense_single_potential_and_independent_source_calorics():
    model = json.loads((DATA / "candidate.json").read_text())
    report = candidate.validate_definition(model)
    assert report["temperature_count"] == 2752 and report["offgrid_count"] == 251
    temperatures = np.unique(
        np.r_[np.arange(1500.0, 4001.0), np.linspace(1500.137, 3999.813, 251)]
    )
    for variant in model["variants"] + model.get("selected_variants", []):
        for species, spec in variant["replacements"].items():
            record = next(
                r for r in model["source_records"].values() if r["species"] == species
            )
            extra = (
                spec.get("delta_GH_J_mol", record["delta_GH_J_mol"])
                - record["delta_GH_J_mol"]
            )
            actual = np.array(
                [
                    list(candidate.reaction_properties(t, spec).values())
                    for t in temperatures
                ]
            )
            expected = np.array(
                [
                    list(
                        source.independent_properties(
                            record, t, spec["kind"], extra
                        ).values()
                    )
                    for t in temperatures
                ]
            )
            np.testing.assert_allclose(
                actual[:, :2], expected[:, :2], atol=1e-8, rtol=0
            )
            np.testing.assert_allclose(
                actual[:, 2:], expected[:, 2:], atol=2e-10, rtol=0
            )
            with pytest.raises(ValueError):
                candidate.formation_h(1499.0, spec)
    bad = expected.copy()
    bad[0, 0] += 1e-5
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(bad[:, :2], expected[:, :2], atol=1e-8, rtol=0)


def test_fixed_network_archives_controls_and_counts():
    summary = impact.replay(DATA / "candidate.json", IMPACT)
    assert summary["new_endpoint_solves"] == 150
    model = json.loads((DATA / "candidate.json").read_text())
    for network in ("ions", "neutral"):
        with np.load(IMPACT / f"{network}.npz") as a:
            species = list(a["species"])
            elements = list(a["elements"])
            n = np.exp(a["ln_n"])
            x = n / n.sum(-1, keepdims=True)
            row = next(r for r in summary["networks"] if r["network"] == network)
            for vi, response in enumerate(row["responses"], 1):
                for target, metrics in response["targets"].items():
                    j = species.index(target)
                    ratio = x[vi, :, j] / x[0, :, j]
                    assert ratio.min() == pytest.approx(
                        metrics["x_ratio_min"], rel=2e-12, abs=1e-14
                    )
                    assert ratio.max() == pytest.approx(
                        metrics["x_ratio_max"], rel=2e-12, abs=1e-14
                    )
                    for element, values in metrics[
                        "maximum_element_share_before_after"
                    ].items():
                        k = elements.index(element)
                        share = n * a["formula_matrix"][k] / a["budget"][:, k, None]
                        np.testing.assert_allclose(
                            [share[0, :, j].max(), share[vi, :, j].max()],
                            values,
                            rtol=2e-12,
                            atol=1e-25,
                        )


def test_frozen_execution_receipts_match_inputs_and_outputs():
    receipt = json.loads((DATA / "receipt.json").read_text())
    for path, digest in receipt["selected_file_sha256"].items():
        assert (
            sha256(historical_source(ROOT, path, digest).read_bytes()).hexdigest()
            == digest
        ), path
    for path, digest in receipt["output_sha256"].items():
        assert sha256((DATA / path).read_bytes()).hexdigest() == digest, path
