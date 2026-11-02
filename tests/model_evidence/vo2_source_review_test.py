"""Offline checks for balanced VO2 source attribution and evidence boundaries."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
REVIEWS = ROOT / "mceg/reviews"
sys.path.insert(0, str(REVIEWS))
SPEC = importlib.util.spec_from_file_location("vo2_source_review", REVIEWS / "vo2_source_review.py")
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


@pytest.fixture(scope="module")
def result():
    return review.review()


def test_frozen_result_and_printed_nodes(result):
    expected = json.loads((review.DATA / "comparison.json").read_text())
    from gas_source_comparison import check_equal
    check_equal(result, expected)
    assert result["printed_node_max_abs_difference"] == 0
    assert result["current_fit_max_abs_ln_K_error_1500_4000"] < 0.0063


def test_balanced_atoms_are_material_and_energy_alignment_is_not_a_cure(result):
    row = next(row for row in result["samples"] if row["temperature_K"] == 1500)
    assert row["atomic_delta_G_J_mol"] == pytest.approx(1793.9379823, abs=1e-6)
    assert row["reaction_delta_G_components_J_mol"]["H298_anchor"] == pytest.approx(1862.61030234)
    assert row["janaf_over_nasa"] == pytest.approx(1.48304006968336)
    assert row["energy_aligned_janaf_over_nasa"] == pytest.approx(1.72192198572984)
    assert row["curated_over_nasa"] == pytest.approx(1.49239877571157)
    assert result["janaf_total_atomization_D0_kJ_mol"] == pytest.approx(1236.342)
    for row in result["samples"]:
        if row["baseline_closure_ln_K"] is not None:
            assert abs(row["baseline_closure_ln_K"]) < 2e-11
        if row["temperature_K"] < 1500:
            assert row["curated_over_nasa"] is None


def test_janaf_ground_only_reading_matches_caloric_table(result):
    data = json.loads((review.DATA / "literature_inputs.json").read_text())
    assert data["janaf_evaluation"]["electronic_levels_cm-1"] == [0]
    assert data["janaf_evaluation"]["electronic_weights"] == [2]
    assert data["janaf_evaluation"]["vibrations_cm-1"] == [970, 300, 930]
    assert data["janaf_evaluation"]["hf_stated_uncertainty_kJ_mol"] == 41.8
    rows = result["electronic_only_diagnostic"]
    assert rows[0]["kim_over_janaf_Q_el"] == pytest.approx(1.24179796869323)
    assert rows[1]["kim_over_janaf_Q_el"] == pytest.approx(1.53291380460997)
    assert all(row["janaf_list"]["Q_el"] == 2 for row in rows)
    assert all(abs(row["delta_Cp_J_mol_K"]) < 0.001
               for row in result["janaf_ground_only_rrho_closure"])
    # A two-level analytic limit checks normalization and caloric moments.
    q = review.electronic_partition([0, 1000], [2, 2], 1000, 8.31446261815324)
    p = 1 / (1 + np.e)
    assert q["Q_el"] == pytest.approx(2 * (1 + np.exp(-1)))
    assert q["Cp_el_J_mol_K"] == pytest.approx(8.31446261815324 * p * (1-p))


def test_saved_footprint_is_not_a_new_substitution(result):
    row = result["footprint"]
    assert row["all_cases_converged"] and row["cases"] == 75
    assert row["max_V_relative_residual"] < 2e-8
    assert (row["case_index"], row["temperature_K"], row["pressure_bar"]) == (4, 1500, 100)
    assert row["VO2_V_fraction"] == pytest.approx(0.6278414221003628)
    assert row["VO2_mole_fraction"] == pytest.approx(8.569022930925194e-9, abs=1e-20)
    assert sum(item["V_fraction"] for item in row["vanadium_allocation"]) == pytest.approx(1, abs=2e-8)
    assert result["equilibrium_source_substitution"] == "not_run"


def test_fulltext_energy_routes_share_the_auxiliary_reference(result):
    data = json.loads((review.DATA / "literature_inputs.json").read_text())
    source = data["balducci1983"]
    audit = result["balducci1983_selected_energy_audit"]
    assert audit["VO_D0_kJ_mol"] == 625.5
    assert audit["reaction_cycle_residual_kJ_mol"] == 0
    shifted = review.historical_energy_audit(
        source, review.read_table(review.DATA / "O-076.txt"), euo_shift=1)
    for key in ["VO2_D0_from_reaction2_kJ_mol", "VO2_D0_from_reaction3_kJ_mol"]:
        assert audit[key] == 1177
        # Independent reaction routes must retain their common auxiliary anchor.
        assert shifted[key] - audit[key] == 2
    assert source["VO2_selected_stated_uncertainty_kJ_mol"] == 11
    assert source["VO2_abstract_stated_uncertainty_kJ_mol"] == 18
    assert [row["derived_stated_uncertainty_kJ_mol"]
            for row in source["reactions"][1:]] == [18, 12]
    row = next(row for row in audit["thermal_nodes"] if row["temperature_K"] == 1500)
    assert row["H_minus_H298_kJ_mol"] == pytest.approx(63.590)
    assert row["delta_H_increment_vs_janaf_kJ_mol"] == pytest.approx(-1.088)


def test_altered_official_source_is_rejected(tmp_path, monkeypatch):
    import shutil
    shutil.copytree(review.DATA, tmp_path / "data")
    monkeypatch.setattr(review, "DATA", tmp_path / "data")
    path = review.DATA / "O-076.txt"
    path.write_text(path.read_text().replace("-232.630", "-132.630"))
    with pytest.raises(ValueError, match="Source hash mismatch"):
        review.review()
