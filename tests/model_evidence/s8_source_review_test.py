"""Check S8 source references, balanced thermochemistry and archive scope offline."""

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import s8_source_review as study
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
    assert header["H298_J_mol"] == pytest.approx(101277.122, abs=0.001)
    assert header["S298_J_mol_K"] == pytest.approx(432.545318, abs=0.000002)
    assert header["Cp298_J_mol_K"] == pytest.approx(156.502715, abs=0.000002)


def test_balanced_thermochemistry_and_energy_alignment(report):
    for row in report["source_samples"]:
        t = row["temperature_K"]
        terms = row["reaction_delta_G_components_J_mol"]
        delta = row["reaction_delta_G_J_mol"]
        molecule, atom = row["properties"]["S8"], row["properties"]["S"]
        assert delta == pytest.approx(molecule["delta_G_J_mol"]-8*atom["delta_G_J_mol"], abs=1e-8)
        assert row["atomic_delta_G_J_mol"] == pytest.approx(-8*atom["delta_G_J_mol"], abs=1e-8)
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
        rounding_bound = (8+1)*(1.5+0.0005*t)/(study.R*t)
        assert abs(row["janaf_raw_minus_hs_ln_K"]) < rounding_bound
    for t in (1499, 4001):
        with pytest.raises(ValueError, match="curated range"):
            study.curated_ln_k(t)


def test_unused_free_energy_column_remains_separate(report):
    audit = {r["temperature_K"]: r for r in report["janaf_free_energy_function_audit"]}
    assert audit[298.15]["difference_J_mol_K"] == pytest.approx(0, abs=1e-10)
    for t, row in audit.items():
        if t != 298.15:
            assert abs(row["difference_J_mol_K"]) < 0.001+0.5/t
    assert 0 not in audit  # The free-energy function is undefined at zero temperature.


def test_archived_footprint_counts_every_sulfur_atom(report):
    footprint = report["footprint"]
    with np.load(study.GRID) as grid:
        col = grid["species"].tolist().index("S8")
        elem = grid["elements"].tolist().index("S")
        # Derive independently from mole fraction and total amount, not ln_n.
        share = 8*np.exp(grid["ln_x"][:, col])*np.exp(grid["ln_n"]).sum(axis=1)/grid["budget"][:, elem]
        assert footprint["sulfur_fraction"] == pytest.approx(float(max(share)), rel=2e-12, abs=0)
    assert footprint["cases"] == 75 and footprint["species_count"] == 537
    assert footprint["model"] == "curated-cos-v1"
    assert footprint["all_cases_converged"]
    assert footprint["maximum_abs_target_h_error"] < 1e-11
    assert footprint["maximum_sulfur_conservation_residual"] < 1e-10
    assert report["curated_change"] == "none"
    assert report["new_equilibrium_substitution"] == "not_run"
    assert report["physical_accuracy"] == "unknown"


def test_direct_atomization_energy_retains_the_atomic_reference_update(report):
    d = report["modern_diagnostic"]
    assert d["W4L_direct_TAE0_J_mol"] == pytest.approx(2115723.28)
    assert d["JANAF_balanced_D0_J_mol"] == pytest.approx(2093492.0)
    assert d["delta_D0_J_mol"] == pytest.approx(22231.28)
    assert d["molecular_Hf0_change_J_mol"] == pytest.approx(-2131.04)
    assert d["eight_atomic_Hf0_changes_J_mol"] == pytest.approx(19974.72)
    assert d["rounded_minus_direct_TAE0_J_mol"] == pytest.approx(-125.52)
    assert d["rounded_Hf_balance_J_mol"] == pytest.approx(
        d["JANAF_balanced_D0_J_mol"]+d["eight_atomic_Hf0_changes_J_mol"]-d["molecular_Hf0_change_J_mol"])
    assert d["energy_only_K_over_JANAF"]["1500"] == pytest.approx(5.944931062180259)
    assert d["energy_only_K_over_JANAF"]["4000"] == pytest.approx(1.9512146972429958)
