"""Offline checks for the raw-equilibrium and limited thermal VO2 re-reduction."""

import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
REVIEWS = ROOT / "mceg/reviews"
sys.path.insert(0, str(REVIEWS))
SPEC = importlib.util.spec_from_file_location(
    "vo2_balducci_reduction", REVIEWS / "vo2_balducci_reduction.py")
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


@pytest.fixture(scope="module")
def result():
    return review.review()


@pytest.fixture(scope="module")
def inputs():
    return json.loads((review.DATA / "inputs.json").read_text())


def test_frozen_result_and_published_slopes_preserve_unresolved_mismatch(result):
    from gas_source_comparison import check_equal
    check_equal(result, json.loads((review.DATA / "comparison.json").read_text()))
    rows = result["second_law"]
    assert [row["count"] for row in rows] == [11, 21, 16]
    assert rows[0]["OLS_H_T_kJ_mol"] == pytest.approx(169.5, abs=0.05)
    assert rows[0]["OLS_slope_standard_error_kJ_mol"] == pytest.approx(4.2, abs=0.05)
    assert rows[1]["OLS_H_T_kJ_mol"] == pytest.approx(-87.0, abs=0.05)
    assert rows[1]["OLS_slope_standard_error_kJ_mol"] == pytest.approx(3.1, abs=0.05)
    assert rows[1]["printed_temperature_K"] == 1046
    assert rows[1]["harmonic_mean_temperature_K"] == pytest.approx(2046.16493754)
    assert rows[1]["H0_reduced_at_harmonic_temperature_kJ_mol"] == pytest.approx(-73.89955)
    assert rows[2]["OLS_H_T_kJ_mol"] == pytest.approx(81.48172455)
    assert rows[2]["printed_H_T_kJ_mol"] == 82.0
    assert rows[2]["OLS_slope_standard_error_kJ_mol"] > 4.1
    assert rows[2]["printed_H_T_uncertainty_kJ_mol"] == 2.9
    # The interrelated measurements cannot be pooled as independent energies.
    assert len(result["reaction_cycle"]["overlapping_rows"]) == 11
    assert result["reaction_cycle"]["max_abs_lnK_closure"] <= 0.00500000001
    assert result["adoption"].startswith("none;")


def test_third_law_retains_original_cell_and_pressure_alternatives(result, inputs):
    rows = result["third_law_reaction2"]["nodes"]
    disputed = next(row for row in rows if row["temperature_K"] == 2087)
    assert disputed["lnK2"] == 5.51
    assert disputed["printed_H0_kJ_mol"] == -74.8
    assert disputed["residual_assume_1atm_kJ_mol"] == pytest.approx(1.036527935)
    assert max(abs(row["residual_assume_1atm_kJ_mol"]) for row in rows
               if row["temperature_K"] != 2087) < 0.056
    for row in rows:
        pressure_shift = -inputs["gas_constant_J_mol_K"]*row["temperature_K"]*math.log(1.01325)/1000
        assert (row["H0_assume_tableIV_1atm_kJ_mol"]
                - row["H0_assume_tableIV_1bar_kJ_mol"]) == pytest.approx(pressure_shift)


def test_janaf_conversion_and_hermite_derivative_use_ground_reference():
    table = review.read_table(review.DATA / "O-026.txt")
    phi = review.janaf_phi(review.DATA / "O-026.txt")
    for t in (298.15, 1900., 2000., 2100.):
        h = (table[t][3]-table[0][3])*1000
        assert float(phi(t)) == pytest.approx(table[t][1]-h/t, abs=0.002)
        assert t*t*float(phi.derivative()(t)) == pytest.approx(h, abs=1e-8)
    assert np.isnan(phi(100.))


@pytest.mark.parametrize("mode", ["vibrations", "electronic", "combined"])
def test_caloric_derivatives_and_zero_point_reference(inputs, mode):
    for t in (1500., 2000., 3000., 4000.):
        node = review.thermal_change(t, inputs, mode)
        above = review.thermal_change(t+0.01, inputs, mode)
        below = review.thermal_change(t-0.01, inputs, mode)
        assert -(above["delta_G_J_mol"]-below["delta_G_J_mol"])/0.02 == pytest.approx(
            node["delta_S_J_mol_K"], abs=2e-8)
        assert (above["delta_H_J_mol"]-below["delta_H_J_mol"])/0.02 == pytest.approx(
            node["delta_Cp_J_mol_K"], abs=2e-8)
        assert node["delta_G_J_mol"] == pytest.approx(node["delta_H_J_mol"]-t*node["delta_S_J_mol_K"])
        assert node["delta_G_J_mol"] < 0
        assert node["delta_third_law_reaction2_or3_H0_kJ_mol"] == node["delta_G_J_mol"]/1000
    # No ZPE shift or second ground-doublet entropy is added at low temperature.
    cold = review.thermal_change(1., inputs, mode)
    assert abs(cold["delta_H_J_mol"]) < 1e-100
    assert abs(cold["delta_G_J_mol"]) < 1e-100


def test_euo_independent_reanchor_and_partial_changes_remain_separate(result):
    row = result["third_law_reaction2"]
    assert row["modern_VO_reanchor_keep_selected_minus74_kJ_mol"] == pytest.approx(1188.992997494128)
    assert row["mean_H0_assume_1atm_kJ_mol"] == pytest.approx(-73.7199197400534)
    variants = row["thermal_variants"]
    assert variants["combined"]["mean_delta_reaction2_H0_kJ_mol"] == pytest.approx(
        variants["vibrations"]["mean_delta_reaction2_H0_kJ_mol"]
        + variants["electronic"]["mean_delta_reaction2_H0_kJ_mol"])
    assert variants["combined"]["modern_VO_reanchored_D0_assume_1atm_kJ_mol"] == pytest.approx(
        1174.658321642735)
    # A lower third-law energy alone is not validation: the same caloric model
    # moves the second-law reduction oppositely and exposes a material mismatch.
    combined = variants["combined"]
    assert combined["second_law_delta_H0_kJ_mol"] == pytest.approx(6.607318809)
    assert combined["second_law_H0_kJ_mol"] == pytest.approx(-67.292231180)
    assert combined["third_law_mean_H0_assume_1atm_kJ_mol"] == pytest.approx(-88.334675851)
    assert combined["second_minus_third_law_H0_assume_1atm_kJ_mol"] == pytest.approx(21.042444671)
    components = result["vibration_components_2000K"]
    assert components["bend"]["delta_G_J_mol"]/1000 == pytest.approx(-8.814978, abs=1e-6)
    assert components["symmetric_stretch"]["delta_G_J_mol"]/1000 == pytest.approx(-0.272029, abs=1e-6)


def test_changed_raw_observation_is_rejected(tmp_path, monkeypatch):
    sources = json.loads((review.DATA / "sources.json").read_text())
    for row in sources["fixed_inputs"]:
        target = tmp_path / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / row["path"], target)
    copied_data = tmp_path / review.DATA.relative_to(ROOT)
    shutil.copyfile(review.DATA / "sources.json", copied_data / "sources.json")
    path = copied_data / "table_I.csv"
    path.write_text(path.read_text().replace("5.51,-74.8", "5.57,-74.8"))
    monkeypatch.setattr(review, "ROOT", tmp_path)
    monkeypatch.setattr(review, "DATA", copied_data)
    with pytest.raises(ValueError, match="Source hash mismatch: .*table_I.csv"):
        review.review()
