"""Replay the actual 28-species production adoption, including coupled choices."""

import json
from pathlib import Path
import sys

import jax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
from curated_adoption_history import historical_source
from curated_gas_v2_adoption import BASELINE, DATA, response_summary, selected_specs
from gas_source_comparison import file_hash
from pending_candidate import candidate_setup, formation_h
from pending_equilibrium_impact import constraints, log_fractions


def test_execution_pins_and_counts():
    receipt = json.loads((DATA / "receipt.json").read_text())
    assert (receipt["result"], receipt["backend"], receipt["dtype"]) == ("passed", "cpu", "float64")
    assert receipt["command"][1] == "mceg/reviews/curated_gas_v2_adoption.py"
    for path, digest in receipt["selected_file_sha256"].items():
        assert file_hash(historical_source(ROOT, path, digest)) == digest, path
    for path, digest in receipt["output_sha256"].items():
        assert file_hash(DATA / path) == digest, path
    summary = json.loads((DATA / "summary.json").read_text())
    assert (summary["dataset"], summary["base_dataset"]) == ("curated-gas-v2", "curated-ch2-v1")
    assert (summary["new_production_solves"], summary["reused_baseline_cases"], summary["tighter_reference_solves"],
            summary["joint_path_solves"], summary["independent_fastchem_runs"]) == (150, 150, 6, 10, 2)
    assert summary["review_count"] == 27 and len(summary["targets"]) == 28
    for name, count in (("diagnostics.json", 150), ("controls.json", 16)):
        rows = json.loads((DATA / name).read_text())
        assert len(rows) == count and all(row["converged"] for row in rows)
    assert len(summary["independent_solver"]) == 2
    for row in summary["independent_solver"]:
        assert len(row["comparisons"]) == 1
        assert row["comparisons"][0]["dataset"] == "curated-gas-v2"
        assert row["comparisons"][0]["max_abs_delta_log10_x"] < 1e-4


def test_coupled_anchor_selection_and_changed_source_rejection(monkeypatch):
    specs, _ = selected_specs()
    shifts = {s: v["delta_GH_J_mol"] for s, v in specs.items() if v["kind"] == "retained_energy"}
    assert shifts["C1N1O1"]-shifts["C1H1N1O1"] == -14724
    assert shifts["N2O4"]-2*shifts["N1O2"] == -425
    assert shifts["H1N1O2cis"]-shifts["H1N1O2trans"] == -555
    assert shifts["N2O5"] == 3613 and shifts["P4S3"] == 56542
    assert specs["O2"]["parameters"]["retained"]["delta_GH_J_mol"] == -107
    source = ROOT / "tests/data/validation/n2o5_h298_candidate/20261106/candidate.json"
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: original(path)+(b"\n" if path == source else b""))
    with pytest.raises(AssertionError):
        selected_specs()


@pytest.mark.parametrize("network,count", [("ions", 537), ("neutral", 421)])
def test_archived_target_only_closure_controls_and_all_responses(network, count):
    specs, _ = selected_specs()
    summary = json.loads((DATA / "summary.json").read_text())["networks"][network == "neutral"]
    with np.load(DATA / f"{network}.npz") as data, np.load(BASELINE / f"{network}.npz") as base:
        assert data["ln_n"].shape == (75, count) and data["converged"].all()
        for key in ("species", "elements", "formula_matrix", "budget", "temperature_K", "pressure_bar", "C_over_O"):
            np.testing.assert_array_equal(data[key], base[key])
        species, elements = data["species"].tolist(), data["elements"].tolist()
        matrix, budget, t, p = (data[k] for k in ("formula_matrix", "budget", "temperature_K", "pressure_bar"))
        tracked = [e for e in elements if e != "e-"]
        targets = [species.index(s) for s in specs]
        expected = base["h"].copy()
        for s, spec in specs.items():
            expected[:, species.index(s)] = [formation_h(float(temp), spec) for temp in t]
        np.testing.assert_allclose(data["h"], expected, atol=2e-12, rtol=0)
        np.testing.assert_array_equal(np.delete(data["h"]-base["h"], targets, axis=-1), 0)
        pair_n, pair_h = np.stack((base["ln_n"], data["ln_n"])), np.stack((base["h"], data["h"]))
        checks = constraints(matrix, elements, species, budget, pair_n, pair_h, p, tracked)
        for key, value in checks.items():
            assert float(value.max()) == pytest.approx(summary["closure"][key], abs=1e-12, rel=1e-8)
        selected, tight = data["tight_case_indices"], data["tight_reference_ln_n"]
        constraints(matrix, elements, species, budget[selected], tight, data["h"][selected], p[selected], tracked)
        delta = log_fractions(tight)-log_fractions(data["ln_n"][selected])
        assert abs(delta).max() < 1e-4 and abs(delta[:, targets]).max() < 1e-8
        assert abs(np.exp(log_fractions(tight))-np.exp(log_fractions(data["ln_n"][selected]))).max() < 1e-9
        i = int(data["path_case_index"])
        direction = data["h"][i]-base["h"][i]
        ph = base["h"][i]+data["lambda_values"][:, None]*direction
        constraints(matrix, elements, species, np.broadcast_to(budget[i], (5, len(elements))), data["lambda_ln_n"], ph, np.repeat(p[i], 5), tracked)
        assert np.diff(np.exp(data["lambda_ln_n"])@direction).max() < 1e-10
        corrected, scale = data["delta_g_corrected"], data["gibbs_numerical_scale"]
        np.testing.assert_array_equal(data["gibbs_ordering_resolved"], (corrected[0] > scale) & (corrected[1] < -scale))
        response, _ = response_summary(species, elements, matrix, budget, t, p, data["C_over_O"], pair_n, specs)
        # Response records contain dimensional values and dimensionless fractions;
        # compare numbers with field-scale roundoff, retaining exact identities.
        def compare(left, right):
            if isinstance(left, dict):
                assert left.keys() == right.keys()
                for key in left:
                    compare(left[key], right[key])
            elif isinstance(left, list):
                assert len(left) == len(right)
                for a, b in zip(left, right):
                    compare(a, b)
            elif isinstance(left, float):
                assert left == pytest.approx(right, rel=2e-12, abs=1e-15)
            else:
                assert left == right
        compare(response, summary["combined_response"])
        changed = expected.copy()
        changed[:, species.index("C1N1O1")] += 1e-3
        with pytest.raises(AssertionError):
            constraints(matrix, elements, species, budget, data["ln_n"], changed, p, tracked)


@pytest.mark.parametrize("ions", [True, False])
def test_actual_factory_dense_identity(ions):
    from exogibbs.presets import curated_ch2, curated_v2

    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        specs, _ = selected_specs()
        actual, base = curated_v2.chemsetup(ions=ions), curated_ch2.chemsetup(ions=ions)
        reference = candidate_setup(base, specs)
        temperatures = np.unique(np.r_[np.arange(1500., 4001.), np.linspace(1500.137, 3999.813, 251)])
        np.testing.assert_allclose(actual.hvector_func(temperatures), reference.hvector_func(temperatures), atol=2e-12, rtol=0)
        row = json.loads((DATA / "summary.json").read_text())["networks"][not ions]["dense_identity"]
        assert (row["temperature_count"], row["offgrid_count"]) == (2752, 251)
        assert np.all(np.array(row["GHSCp_max_abs_errors"]) < [2e-8, 2e-8, 2e-10, 2e-10])
    finally:
        jax.config.update("jax_enable_x64", previous)
