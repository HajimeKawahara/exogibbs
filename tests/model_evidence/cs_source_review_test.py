"""Check CS energy balance, state conventions and source-review boundaries."""

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
import cs_source_review as study
from gas_source_comparison import check_equal


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_balanced_energy_reproduces_curated_and_distinguishes_new_network(report):
    assert report["balanced_D0_J_mol"] == pytest.approx({
        "atct": (711.381+276.830-278.900)*1000,
        "nasa9": 710814.980, "janaf": 708837.0}, abs=1e-7)
    assert report["atct_1.228_balanced_D0_J_mol"] == pytest.approx(
        (711.370+276.85-278.53)*1000, abs=1e-7)
    assert report["atct_1.228_minus_adopted_D0_J_mol"] == pytest.approx(379.0)
    for row in report["source_samples"]:
        t = row["temperature_K"]
        assert row["atct_1.228_energy_only_K_ratio"] == pytest.approx(
            math.exp(379/(study.major.R*t)), rel=2e-14)
        if t >= 1500:
            assert row["curated_ln_K"] == pytest.approx(row["reconstructed_hybrid_ln_K"], abs=2e-12)
            assert t*(row["curated_ln_K"]-row["nasa_ln_K"]) == pytest.approx(
                report["adopted_minus_nasa_D0_over_k_K"], abs=1e-9)
    assert report["adopted_minus_nasa_D0_over_k_K"] == pytest.approx(-180.400027271855)
    with pytest.raises(ValueError, match="outside"):
        study.curated_ln_k(1000)


def test_actual_levels_define_spin_zero_and_differ_from_metadata(report):
    inventory = json.loads((study.DATA / "state_inventory.json").read_text())
    with np.load(study.DATA / "states.npz") as archive:
        states = archive["states"]
    assert states.shape == (11497, 4)
    assert states[0].tolist() == [0, 1, 0, 0]
    assert np.all(states[:, 1] == 2*states[:, 2]+1)
    assert states[:, 0].max() == inventory["maximum_energy_cm-1"] == 66286.878662
    assert inventory["metadata_maximum_energy_cm-1"] == 30000
    assert (states[:, 2].max(), states[:, 3].max()) == (258, 49)
    assert study.moments(states, 1)["Q"] == pytest.approx(1.2900, abs=5e-5)
    assert report["above_adopted_D0_states"] == 441


def test_moment_calorics_and_rounded_published_pf_have_distinct_scopes(report):
    for row in report["source_samples"]:
        t, modern = row["temperature_K"], row["JnK"]
        if t <= 3000:
            assert abs(modern["states_over_published_Q"]-1) < 6e-7
        else:
            assert modern["published_Q"] is None
            assert modern["states_over_published_Q"] is None
        # Gibbs identity independently links energy/entropy moments to Q.
        reconstructed = -(modern["delta_H_J_mol"]-t*modern["delta_S_J_mol_K"])/(study.major.R*t)
        assert reconstructed == pytest.approx(math.log(modern["Q_over_nasa_internal"]), abs=2e-14)
        assert modern["isotope_mass_K_proxy"] == pytest.approx(
            modern["Q_over_nasa_internal"]*(study.ISOTOPE_MASS/44.0757)**1.5)
        assert abs(modern["Q_over_nasa_internal"]-1) < .001
    top = report["source_samples"][-1]["JnK"]
    assert top["delta_Cp_J_mol_K"] == pytest.approx(-.3057011361, abs=1e-8)
    assert top["cut_diagnostics"]["below_adopted_D0"]["removed_Q_fraction"] < 5.7e-9
    assert top["cut_diagnostics"]["J_below_240"]["removed_Q_fraction"] < 3.3e-7
    assert top["cut_diagnostics"]["v_below_45"]["removed_Q_fraction"] < 2e-7
    assert top["cut_diagnostics"]["below_adopted_D0"]["full_minus_cut_Cp_J_mol_K"] < 1.9e-5


def test_electronic_scale_is_a_caloric_diagnostic_not_an_adopted_correction(report):
    literature = json.loads((study.DATA / "literature.json").read_text())
    constants = literature["electronic_scale_constants"]
    def logq(t):
        return math.log(study.electronic_scale(t, constants)["electronic_factor"])
    t, dt = 4000.0, 1.0
    derivative_cp = study.major.R*(2*t*(logq(t+dt)-logq(t-dt))/(2*dt)
        + t*t*(logq(t+dt)-2*logq(t)+logq(t-dt))/dt**2)
    scale = study.electronic_scale(t, constants)
    assert derivative_cp == pytest.approx(scale["Cp_increment_J_mol_K"], abs=2e-7)
    assert 1.00034 < scale["electronic_factor"] < 1.00035
    assert report["production_change"] == "none"
    assert report["equilibrium_substitution"] == "not_run"


def test_saved_grid_footprint_preserves_sulfur_and_replays_cs_potentials(report):
    footprint = report["footprint"]
    assert (footprint["cases"], footprint["species"], footprint["all_converged"]) == (75, 537, True)
    assert footprint["maximum_abs_h_error"] < 2e-12
    assert footprint["maximum_relative_sulfur_conservation_residual"] < 1e-10
    assert footprint["maximum_sulfur_fraction"] == pytest.approx(.9347359988213835)
    assert (footprint["case_index"], footprint["temperature_K"], footprint["pressure_bar"], footprint["C_over_O"]) == (55, 2000, 1e-6, 1.5)
    with np.load(study.GRID) as grid:
        n = np.exp(grid["ln_n"])
        sulfur = grid["elements"].tolist().index("S")
        assert n @ grid["formula_matrix"][sulfur] == pytest.approx(grid["budget"][:, sulfur], rel=1e-10, abs=0)
        column = grid["species"].tolist().index("C1S1")
        assert n[55, column]/n[55].sum() == pytest.approx(footprint["mole_fraction_at_maximum_sulfur_share"])


def test_archived_replay_allows_only_small_cancellation_in_caloric_differences(report):
    actual = deepcopy(report)
    expected = json.loads((study.DATA / "comparison.json").read_text())
    for measured, saved in zip(actual["source_samples"], expected["source_samples"]):
        m, s = measured["JnK"], saved["JnK"]
        assert m.pop("delta_H_J_mol") == pytest.approx(s.pop("delta_H_J_mol"), abs=1e-8, rel=0)
        for name in m["cut_diagnostics"]:
            assert m["cut_diagnostics"][name].pop("full_minus_cut_Cp_J_mol_K") == pytest.approx(
                s["cut_diagnostics"][name].pop("full_minus_cut_Cp_J_mol_K"), abs=1e-10, rel=0)
    check_equal(actual, expected)


def test_receipt_and_extraction_guard_pin_inputs(tmp_path):
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    for group in ("selected_input_sha256", "output_sha256"):
        for path, digest in receipt[group].items():
            assert sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    (tmp_path / "JnK.states.bz2").write_bytes(b"unreviewed upstream revision")
    with pytest.raises(ValueError, match="Unreviewed original"):
        study.extract(tmp_path, tmp_path / "out")
