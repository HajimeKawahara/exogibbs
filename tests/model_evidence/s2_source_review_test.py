"""Check S2 energy balance, permutation counting and finite UV-list scope."""

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import s2_source_review as study
from gas_source_comparison import check_equal


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_current_balanced_anchor_and_historical_comparison(report):
    expected = json.loads((study.DATA / "comparison.json").read_text())
    actual = deepcopy(report)
    for left, right in zip(actual["source_samples"], expected["source_samples"]):
        for key in ("H_increment_J_mol", "G_minus_H0_J_mol"):
            a = left["delta_Gomez_minus_nasa"].pop(key)
            b = right["delta_Gomez_minus_nasa"].pop(key)
            assert a == pytest.approx(b, abs=1e-7, rel=0)
    for key in ("maximum_abs_h_error", "maximum_relative_sulfur_residual"):
        assert actual["footprint"].pop(key) == pytest.approx(
            expected["footprint"].pop(key), abs=1e-12, rel=0)
    check_equal(actual, expected)
    assert report["balanced_D0_J_mol"] == pytest.approx(
        {"atct": (2*276.830-127.370)*1000, "nasa9": 421557.26, "janaf": 421170})
    assert report["adopted_minus_nasa_D0_over_k_K"] == pytest.approx(569.5067189807596)
    old = json.loads((ROOT / "tests/data/validation/gas_source_comparison/baseline-001/comparison.json").read_text())
    for row in report["source_samples"]:
        t = row["temperature_K"]
        assert t*(row["reconstructed_hybrid_ln_K"]-row["nasa_ln_K"]) == pytest.approx(
            report["adopted_minus_nasa_D0_over_k_K"], abs=2e-10)
        if t >= 1500:
            assert row["curated_ln_K"] == pytest.approx(row["reconstructed_hybrid_ln_K"], abs=2e-12)
        original = next((x for x in old["samples"]
                         if x["species_id"] == "S2" and x["temperature_K"] == t), None)
        if original:
            assert row["nasa_ln_K"] == pytest.approx(original["ln_Kp"]["nasa"], abs=3e-11)
            assert row["janaf"]["native_ln_K"] == pytest.approx(original["ln_Kp"]["janaf"], abs=3e-11)
    with pytest.raises(ValueError, match="outside"):
        study.curated_ln_k(1000)


def test_same_version_update_cancels_and_ground_atom_threshold_agrees(report):
    assert report["latest_ATcT_balanced_D0_J_mol"] == pytest.approx((2*276.85-127.42)*1000)
    assert report["latest_minus_current_D0_J_mol"] == pytest.approx(-10)
    literature = json.loads((study.DATA / "literature.json").read_text())
    threshold = literature["Frederix2009"]
    assert abs(report["ATcT_D0_cm-1"]-threshold["D0_cm-1"]) < threshold["reported_uncertainty_cm-1"]
    assert literature["ATcT_1.228"]["S_S2_correlation_percent"] == 99.2
    for row in report["source_samples"]:
        assert row["latest_energy_only_over_current"] == pytest.approx(
            math.exp(-10/(study.major.R*row["temperature_K"])))


def test_permutation_weights_and_raw_label_conflict_are_preserved():
    inventory = json.loads((study.DATA / "state_inventory.json").read_text())
    with np.load(study.DATA / "states.npz") as archive:
        states = archive["states"]
        assert states.shape == (17766, 6)
        assert states[0].tolist() == [0, 1, 0, 0, 1, 0]
        np.testing.assert_array_equal(states[:, 1], 2*states[:, 2]+1)
        assert np.all(states[states[:, 5] == 0, 4] % 2 == 1)
        assert np.all(states[states[:, 5] == 1, 4] % 2 == 0)
        assert states[states[:, 5] == 0, 3].max() == 10
        assert states[:, 2].max() == 150
        assert states[:, 0].max() == 45188.777334
        assert study.moments(states, .01)["Q"] == pytest.approx(1)
        assert archive["published_pf"][[0, -1], 0].tolist() == [1, 5000]
    assert inventory["raw_labels"] == list(study.RAW_LABELS)
    assert inventory["published_pf"]["Gomez_equals_HITRAN_every_row"]
    assert "disagree" in inventory["label_note"]


def test_caloric_and_zero_shift_identities(report):
    for row in report["source_samples"]:
        t = row["temperature_K"]
        delta = row["delta_Gomez_minus_nasa"]
        assert delta["G_minus_H0_J_mol"] == pytest.approx(
            delta["H_increment_J_mol"]-t*delta["S_J_mol_K"], abs=2e-9)
        assert -delta["G_minus_H0_J_mol"]/(study.major.R*t) == pytest.approx(
            math.log(row["same_D0_atoms_natural_mass_K_ratio"]), abs=3e-14)
        assert abs(row["states_over_published_Q"]-1) < 4.5e-6
    with np.load(study.DATA / "states.npz") as archive:
        states = archive["states"].copy()
    t, epsilon = 2000, .01
    base = study.properties(study.moments(states, t), t, 64.13)
    upper = study.properties(study.moments(states, t+epsilon), t+epsilon, 64.13)
    lower = study.properties(study.moments(states, t-epsilon), t-epsilon, 64.13)
    assert (upper["H_increment_J_mol"]-lower["H_increment_J_mol"])/(2*epsilon) == pytest.approx(base["Cp_J_mol_K"], rel=1e-9)
    q = study.moments(states, t)["Q"]
    states[:, 0] += 15.120664
    assert study.moments(states, t)["Q"]/q == pytest.approx(math.exp(-study.C2*15.120664/t))
    # An extra symmetry division would halve Q and change entropy by -R ln 2.
    moment = study.moments(states, t)
    normal = study.properties(moment, t, 64.13)
    moment["Q"] /= 2
    wrong = study.properties(moment, t, 64.13)
    assert wrong["S_J_mol_K"]-normal["S_J_mol_K"] == pytest.approx(-study.major.R*math.log(2))


def test_uv_coverage_diagnostics_do_not_become_a_production_correction(report):
    rows = {row["temperature_K"]: row for row in report["source_samples"]}
    high = rows[4000]
    assert rows[1500]["same_D0_atoms_natural_mass_K_ratio"] == pytest.approx(.98343680421634)
    assert high["same_D0_atoms_natural_mass_K_ratio"] == pytest.approx(.71765670936989)
    assert high["Gomez"]["Cp_J_mol_K"] < rows[1000]["Gomez"]["Cp_J_mol_K"]
    assert high["Gomez"]["J_ge_145_Q_fraction"] > .0212
    assert high["Gomez"]["X_v10_Q_fraction"] > .0201
    assert high["Gomez"]["B_and_Bpp_Q_fraction"] < .000108
    assert high["above_adopted_D0_Q_fraction"] < .00004
    assert high["BC2016_over_nasa_Q"] == pytest.approx(.866306517313)
    assert high["janaf"]["same_D0_atoms_K_ratio"] == pytest.approx(.986131104816)
    assert report["production_change"] == "none"
    assert report["equilibrium_substitution"] == "not_run"


def test_archived_sulfur_footprint_is_current_and_balanced(report):
    f = report["footprint"]
    assert (f["model"], f["cases"], f["species"], f["all_converged"]) == ("curated-h2o-v1", 75, 537, True)
    assert f["maximum_abs_h_error"] < 1e-12
    assert f["maximum_relative_sulfur_residual"] < 3e-12
    assert f["maximum_sulfur_fraction"] == pytest.approx(.018816214961451493)
    assert f["maximum_mole_fraction"] == pytest.approx(2.1115951559232377e-7)
    assert (f["temperature_K"], f["pressure_bar"]) == (1500, 1e-6)


def test_receipt_pins_inputs_and_outputs():
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    for group in ("selected_input_sha256", "output_sha256"):
        for name, expected in receipt[group].items():
            assert sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
