"""Protect C4 energy/thermal and state conventions without runtime imports."""

import json
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import c4_source_review as study
from gas_source_comparison import check_equal, verify_hashes


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_c4_official_coefficients_and_frozen_nodes_replay(report):
    check_equal(report, json.loads((study.DATA / "comparison.json").read_text()))
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    verify_hashes(receipt["input_and_artifact_sha256"])
    assert report["official_nasa_matching_coefficients"] == 27
    assert report["fresh_janaf_frozen_matching_nodes"] == 4
    assert report["nasa_header_check"]["H298_J_mol"] == pytest.approx(1033903.742, abs=0.001)
    closures = [row["baseline_closure_ln_K"] for row in report["source_samples"]
                if row["baseline_closure_ln_K"] is not None]
    assert len(closures) == 4
    assert max(abs(value) for value in closures) < 2e-11


def test_c4_reversal_is_energy_entropy_cancellation_not_a_fit_artifact(report):
    rows = {row["temperature_K"]: row for row in report["source_samples"]}
    for t, row in rows.items():
        terms = row["reaction_delta_G_components_J_mol"]
        delta = row["reaction_delta_G_J_mol"]
        assert delta == pytest.approx(sum(terms.values()), abs=1e-8)
        assert delta == pytest.approx(row["properties"]["C4"]["delta_G_J_mol"]
                                     -4*row["properties"]["C"]["delta_G_J_mol"], abs=1e-8)
        assert terms["H298_anchor"] == pytest.approx(-63186.18793265, abs=1e-6)
        assert row["janaf_over_nasa"] == pytest.approx(
            math.exp(-delta/(report["gas_constant_J_mol_K"]*t)), rel=2e-12)
    assert rows[1500]["reaction_delta_G_J_mol"] < 0 < rows[3000]["reaction_delta_G_J_mol"]
    assert rows[1500]["curated_over_nasa"] == pytest.approx(6.997446101919495)
    assert rows[3000]["curated_over_nasa"] == pytest.approx(0.46554572333879574)
    for t in (1500, 3000):
        assert 0.03 < rows[t]["H298_aligned_janaf_over_nasa"] < 0.05
        assert 0.97 < rows[t]["curated_over_janaf"] < 1.03
    assert max(abs(row["janaf_raw_minus_hs_ln_K"]) for row in rows.values()) < 0.00025


def test_c4_zero_K_anchors_and_separate_isomer_energies(report):
    energies = report["energy_zero_K"]
    assert energies["janaf_D0_kJ_mol"] == pytest.approx(1880.639)
    assert energies["nasa_D0_kJ_mol"] == pytest.approx(1819.79042)
    assert energies["w4_linear_D0_kJ_mol"] == pytest.approx(1795.60544)
    assert energies["w4_cyclic_D0_kJ_mol"] == pytest.approx(1799.49656)
    assert energies["w4_cyclic_minus_linear_energy_kJ_mol"] == pytest.approx(-3.89112)
    weights = {row["temperature_K"]: row["added_electronic_weight_relative_to_triplet"]
               for row in report["lowest_excited_state"]}
    assert weights[1500] == pytest.approx(0.05110199273928642)
    assert weights[3000] == pytest.approx(0.1845751748191622)
    assert weights[1500] < weights[3000] < weights[4000]


def test_c4_footprint_is_reused_and_counts_all_four_carbons(report):
    footprint = report["footprint"]
    assert footprint["cases"] == 75
    assert footprint["species_count"] == 537
    assert footprint["all_cases_converged"]
    assert footprint["carbon_fraction"] == pytest.approx(1.0143889109069146e-6, rel=1e-12, abs=0)
    assert footprint["mole_fraction"] == pytest.approx(2.0528156099113751e-10, rel=1e-12, abs=0)
    assert footprint["temperature_K"] == 2000
    assert footprint["pressure_bar"] == 1e-6
    assert footprint["C_over_O"] == 1.5
    assert footprint["new_equilibrium_substitution"] == "not_run"
    assert report["physical_accuracy"] == "unknown"
    assert report["adoption"] == "none; retain current fit provisionally"
