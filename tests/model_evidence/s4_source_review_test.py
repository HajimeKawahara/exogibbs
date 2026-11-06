"""Check S4 source references, balanced thermochemistry and archive scope offline."""

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import s4_source_review as study
from gas_source_comparison import check_equal


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_official_records_and_pinned_replay(report):
    check_equal(report, json.loads((study.DATA / "comparison.json").read_text()))
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    for path, expected in receipt["input_and_artifact_sha256"].items():
        assert sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    assert report["official_nasa_matching_coefficients"] == 18
    assert report["official_janaf_frozen_matching_nodes"] == 4
    header = report["nasa_header_check"]
    assert header["H298_J_mol"] == pytest.approx(135632.333, abs=0.001)
    assert header["S298_J_mol_K"] == pytest.approx(293.564631, abs=0.000002)
    assert header["Cp298_J_mol_K"] == pytest.approx(65.944203, abs=0.000002)


def test_balanced_thermochemistry_and_energy_alignment(report):
    for row in report["source_samples"]:
        t = row["temperature_K"]
        terms = row["reaction_delta_G_components_J_mol"]
        delta = row["reaction_delta_G_J_mol"]
        molecule, atom = row["properties"]["S4"], row["properties"]["S"]
        assert delta == pytest.approx(molecule["delta_G_J_mol"]-4*atom["delta_G_J_mol"], abs=1e-8)
        assert row["atomic_delta_G_J_mol"] == pytest.approx(-4*atom["delta_G_J_mol"], abs=1e-8)
        assert delta == pytest.approx(sum(terms.values()), abs=1e-8)
        assert row["janaf_over_nasa"] == pytest.approx(math.exp(-delta/(study.R*t)), rel=2e-12)
        assert row["H298_aligned_janaf_over_nasa"] == pytest.approx(
            row["janaf_over_nasa"]*math.exp(terms["H298_anchor"]/(study.R*t)), rel=2e-12)
        if t < 1500:
            assert row["ln_Kp_curated"] is None
        else:
            assert 0.97 < row["curated_over_janaf"] < 1.03
    # Three H/G values rounded to 0.001 kJ/mol and one entropy to 0.001 J/mol/K.
    for row in report["source_samples"]:
        t = row["temperature_K"]
        rounding_bound = (4+1)*(1.5+0.0005*t)/(study.R*t)
        assert abs(row["janaf_raw_minus_hs_ln_K"]) < rounding_bound
    for t in (1499, 4001):
        with pytest.raises(ValueError, match="curated range"):
            study.curated_ln_k(t)


def test_unused_free_energy_column_remains_separate(report):
    audit = {r["temperature_K"]: r for r in report["janaf_free_energy_function_audit"]}
    assert audit[298.15]["difference_J_mol_K"] == pytest.approx(77.695, abs=1e-10)
    for t, row in audit.items():
        if t != 298.15:
            assert abs(row["difference_J_mol_K"]) < 0.001+0.5/t
    assert 0 not in audit  # The free-energy function is undefined at zero temperature.


def test_archived_footprint_counts_every_sulfur_atom(report):
    footprint = report["footprint"]
    with np.load(study.GRID) as grid:
        col = grid["species"].tolist().index("S4")
        elem = grid["elements"].tolist().index("S")
        # Derive independently from mole fraction and total amount, not ln_n.
        share = 4*np.exp(grid["ln_x"][:, col])*np.exp(grid["ln_n"]).sum(axis=1)/grid["budget"][:, elem]
        assert footprint["sulfur_fraction"] == pytest.approx(float(max(share)), rel=2e-12, abs=0)
    assert footprint["cases"] == 75 and footprint["species_count"] == 537
    assert footprint["model"] == "curated-cos-v1"
    assert footprint["all_cases_converged"]
    assert footprint["maximum_abs_target_h_error"] < 1e-11
    assert footprint["maximum_sulfur_conservation_residual"] < 1e-10
    assert report["curated_change"] == "none"
    assert report["new_equilibrium_substitution"] == "not_run"
    assert report["physical_accuracy"] == "unknown"


def test_electronic_weight_is_not_a_full_partition_correction(report):
    diagnostic = report["modern_diagnostic"]
    assert diagnostic["triplet_gap_J_mol"] == pytest.approx(45187.2)
    weights = diagnostic["triplet_over_singlet_weights"]
    assert weights["1500"] == pytest.approx(0.08009287721473486)
    assert weights["4000"] == pytest.approx(0.7709895558273729)
    assert "electronic factor only" in diagnostic["approximation"]
