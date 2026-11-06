"""Check H2S energy, spin normalization, source limits and archived sulfur scope."""

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
import h2s_source_review as study
from gas_source_comparison import check_equal


def check_replay(actual, expected):
    """Allow cancellation roundoff without hiding physical source changes."""
    actual, expected = deepcopy(actual), deepcopy(expected)
    for measured, archived in zip(actual["source_samples"], expected["source_samples"]):
        for key in ("H_increment_J_mol", "G_minus_H0_J_mol"):
            a = measured["delta_AYT2_minus_nasa"].pop(key)
            b = archived["delta_AYT2_minus_nasa"].pop(key)
            assert a == pytest.approx(b, abs=1e-7, rel=0), key
    for key in ("maximum_abs_h_error", "maximum_relative_sulfur_residual"):
        a, b = actual["footprint"].pop(key), expected["footprint"].pop(key)
        assert a == pytest.approx(b, abs=1e-12, rel=0), key
    check_equal(actual, expected)


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_same_version_energy_and_production_column_replay(report):
    expected = json.loads((study.DATA / "comparison.json").read_text())
    check_replay(report, expected)
    assert report["balanced_D0_J_mol"] == pytest.approx(
        {"atct": (2*216.034+276.830+17.370)*1000,
         "nasa9": 724673.796, "janaf": 724391}, abs=1e-6)
    assert report["first_dissociation_H_plus_SH_J_mol"] == pytest.approx(376214)
    old = json.loads((ROOT / "tests/data/validation/gas_source_comparison/baseline-001/comparison.json").read_text())
    for row in report["source_samples"]:
        t = row["temperature_K"]
        assert t*(row["reconstructed_hybrid_ln_K"]-row["nasa_ln_K"]) == pytest.approx(
            report["adopted_minus_nasa_D0_over_k_K"], abs=1e-9)
        if t >= 1500:
            assert row["curated_ln_K"] == pytest.approx(row["reconstructed_hybrid_ln_K"], abs=2e-12)
        original = next((x for x in old["samples"]
                         if x["species_id"] == "H2S1" and x["temperature_K"] == t), None)
        if original:
            assert row["nasa_ln_K"] == pytest.approx(original["ln_Kp"]["nasa"], abs=3e-11)
            assert row["janaf"]["native_ln_K"] == pytest.approx(original["ln_Kp"]["janaf"], abs=3e-11)
    with pytest.raises(ValueError, match="outside"):
        study.curated_ln_k(1000)


def test_raw_spin_weights_common_zero_and_assignment_anomaly_are_preserved():
    inventory = json.loads((study.DATA / "state_inventory.json").read_text())
    with np.load(study.DATA / "states.npz") as data:
        states = data["states"]
        assert len(states) == inventory["states"] == 220630
        assert states[0].tolist() == [0, 1, 0, 0]
        assert states[:, 0].max() == 34636.207111
        assert states[:, 2].max() == 40
        ratios = states[:, 1]/(2*states[:, 2]+1)
        wrong = np.flatnonzero(~np.isin(ratios, [1, 3]))
        assert wrong.tolist() == [76643]
        assert states[76643].tolist() == [11091.055557, 99, 13, 11091.05555]
        assert np.min(states[ratios == 3, 0]) == inventory["lowest_ortho_energy_cm-1"] == 13.74631
        assert inventory["source_type_counts"] == {"Ca": 213042, "Ma": 7588}
        assert inventory["states_with_changed_energy"] == 7424
        assert study.moments(states, 1)["Q_spin_included"] == pytest.approx(1, abs=3e-8)
        for year in (2017, 2021, 2025):
            assert data[f"tips{year}"][[0, -1], 0].tolist() == [1, 4000]
    metadata = json.loads((study.DATA / "AYT2.def.json").read_text())
    assert metadata["irreducible_representations"] == {"A1": 1, "A2": 1, "B1": 3, "B2": 3}
    assert metadata["dataset"]["states"]["max_energy"] == 10000
    assert metadata["dataset"]["max_temperature"] == 1500
    assert "76644 11091.055557" in (study.DATA / "state_assignment_excerpt.txt").read_text()


def test_caloric_identity_spin_and_published_source_ancestry(report):
    for row in report["source_samples"]:
        t = row["temperature_K"]
        p = row["AYT2"]
        assert p["Q_spin_free"] == pytest.approx(p["Q_spin_included"]/4)
        delta = row["delta_AYT2_minus_nasa"]
        assert delta["G_minus_H0_J_mol"] == pytest.approx(
            delta["H_increment_J_mol"]-t*delta["S_J_mol_K"], abs=2e-9)
        assert -delta["G_minus_H0_J_mol"]/(study.major.R*t) == pytest.approx(
            math.log(row["same_D0_atoms_natural_mass_K_ratio"]), abs=1e-13)
        assert math.exp(row["AYT2_same_anchor_ln_K"]-row["reconstructed_hybrid_ln_K"]) == pytest.approx(
            row["same_D0_atoms_natural_mass_K_ratio"], abs=1e-13)
        assert row["same_D0_atoms_isotope_mass_K_proxy"] == pytest.approx(
            row["same_D0_atoms_natural_mass_K_ratio"]*(33.98772106/34.08088)**1.5)
        for name, table in row["tables"].items():
            if name == "published_cp":
                assert abs(table["state_over_table"]-1) < 6.1e-5
            else:
                assert abs(table["state_over_table"]-1) < 6e-6
        assert row["tables"]["tips2017"]["value"] == row["tables"]["tips2021"]["value"]
        assert abs(row["current_over_original_calculated_Q"]-1) < 4.6e-6
        assert abs(row["displayed_J_weight_sensitivity_relative_Q"]) < 1.71e-6
        assert 2 < row["nasa_atomic_Q"]["H"] < 2.00004
        assert abs(row["S_five_lowest_levels_Q"]/row["nasa_atomic_Q"]["S"]-1) < 1.1e-4


def test_high_temperature_cutoff_is_not_mistaken_for_a_production_correction(report):
    rows = {row["temperature_K"]: row for row in report["source_samples"]}
    high = rows[4000]
    assert rows[1500]["same_D0_atoms_natural_mass_K_ratio"] == pytest.approx(.9977485063)
    assert high["same_D0_atoms_natural_mass_K_ratio"] == pytest.approx(.8919674384)
    assert high["AYT2"]["Cp_J_mol_K"] < rows[2000]["AYT2"]["Cp_J_mol_K"]
    assert high["AYT2"]["J_eq_40_fraction"] > .0085
    assert high["AYT2"]["J_ge_35_fraction"] > .07
    assert high["above_first_dissociation_fraction"] < 4.4e-8
    assert high["janaf"]["same_D0_atoms_K_ratio"] == pytest.approx(.9426885338)
    assert report["production_change"] == "none"
    assert report["equilibrium_substitution"] == "not_run"


def test_full_sulfur_budget_and_current_saved_potentials(report):
    f = report["footprint"]
    assert f["model"] == "curated-h2o-v1"
    assert (f["cases"], f["species"], f["all_converged"]) == (75, 537, True)
    assert f["maximum_relative_sulfur_residual"] < 3e-12
    assert f["maximum_abs_h_error"] < 2e-12
    h2s = f["sulfur_reservoirs"]["H2S1"]
    assert h2s["maximum_sulfur_fraction"] == pytest.approx(.9140081302163)
    assert (h2s["temperature_K"], h2s["pressure_bar"]) == (1500, 100)
    assert f["largest_sulfur_reservoirs"]["S1Si1"]["maximum_sulfur_fraction"] > .9999


def test_replay_rejects_energy_spin_and_thermal_perturbations(report):
    changed = deepcopy(report)
    changed["source_samples"][3]["delta_AYT2_minus_nasa"]["H_increment_J_mol"] += 1e-5
    with pytest.raises(AssertionError, match="H_increment"):
        check_replay(changed, report)
    changed = deepcopy(report)
    changed["source_samples"][3]["same_D0_atoms_natural_mass_K_ratio"] += 1e-8
    with pytest.raises(ValueError, match="same_D0"):
        check_replay(changed, report)
    with np.load(study.DATA / "states.npz") as data:
        states = data["states"].copy()
    # A wrong common zero or a duplicated spin factor must be detectable.
    original = study.moments(states, 1500)
    states[:, 0] += 1
    shifted = study.moments(states, 1500)
    assert shifted["Q_spin_included"]/original["Q_spin_included"] == pytest.approx(math.exp(-study.C2/1500))
    with pytest.raises(ValueError, match="Q_spin_included"):
        check_equal(shifted, original)
    duplicated = dict(original)
    duplicated["Q_spin_included"] *= 4
    normal = study.properties(original, 1500, 34.08088)
    wrong = study.properties(duplicated, 1500, 34.08088)
    assert wrong["S_J_mol_K"]-normal["S_J_mol_K"] == pytest.approx(study.major.R*math.log(4))
    with pytest.raises(ValueError, match="Q_spin_free"):
        check_equal(wrong, normal)


def test_receipt_pins_every_selected_input_and_output():
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    for group in ("selected_input_sha256", "output_sha256"):
        for name, expected in receipt[group].items():
            assert sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    sources = json.loads((study.DATA / "sources.json").read_text())
    for item in sources["downloads"]:
        if item["archived"]:
            assert sha256((study.DATA/item["file"]).read_bytes()).hexdigest() == item["sha256"]
