"""Audit SO2 normalization, incomplete high-T states and archived sulphur scope."""

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
import so2_source_review as study
from gas_source_comparison import check_equal


def check_replay(actual: dict, expected: dict) -> None:
    """Bound cancellation/rounded-table derivative roundoff locally."""
    actual, expected = deepcopy(actual), deepcopy(expected)
    for row, saved in zip(actual["source_samples"], expected["source_samples"]):
        for label in ("ExoAmes", "TIPS2021", "TIPS2025"):
            current, reference = row["modern"][label], saved["modern"][label]
            # Caloric differences subtract large values; TIPS additionally
            # differentiates rounded tables through a small least-squares fit.
            for key in ("properties", "delta_from_NASA"):
                a, b = current.pop(key), reference.pop(key)
                for quantity in a:
                    assert a[quantity] == pytest.approx(b[quantity], abs=1e-7, rel=0), quantity
            if label.startswith("TIPS"):
                key = "Cp_50K_minus_100K_window_J_mol_K"
                assert current.pop(key) == pytest.approx(reference.pop(key), abs=1e-7, rel=0), key
    check_equal(actual, expected)


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_balanced_energy_and_current_polynomial(report):
    check_replay(report, json.loads((study.DATA / "comparison.json").read_text()))
    assert report["balanced_D0_J_mol"] == pytest.approx({
        "atct": (276.83+2*246.844+294.2)*1000,
        "nasa9": 1062769.902, "janaf": 1062613.0}, abs=1e-6)
    assert report["adopted_minus_NASA_D0_over_k_K"] == pytest.approx(235.030751838, abs=1e-8)
    for row in report["source_samples"]:
        if row["temperature_K"] >= 1500:
            assert row["curated_ln_K"] == pytest.approx(row["reconstructed_hybrid_ln_K"], abs=2e-12)
            assert math.exp(row["curated_ln_K"]-row["native_NASA_ln_K"]) == pytest.approx(
                row["curated_over_native_NASA_K"], abs=1e-12)
        assert abs(row["S_atom_diagnostic"]["NIST_five_level_over_NASA_Q"]-1) < 8e-5
    with pytest.raises(ValueError, match="outside"):
        study.curated_ln_k(1000)


def test_replay_does_not_hide_energy_spin_or_caloric_changes(report):
    saved = json.loads((study.DATA / "comparison.json").read_text())
    for key, increment in (("Q_over_NASA", 1e-7), ("Q", 1.0)):
        changed = deepcopy(report)
        changed["source_samples"][3]["modern"]["ExoAmes"][key] += increment
        with pytest.raises(ValueError, match=key):
            check_replay(changed, saved)
    changed = deepcopy(report)
    changed["source_samples"][3]["modern"]["ExoAmes"]["delta_from_NASA"]["Cp_J_mol_K"] += 1e-5
    with pytest.raises(AssertionError, match="Cp_J_mol_K"):
        check_replay(changed, saved)
    changed = deepcopy(report)
    changed["balanced_D0_J_mol"]["atct"] += 1.0
    with pytest.raises(ValueError, match="balanced_D0"):
        check_replay(changed, saved)


def test_state_sum_thermodynamic_identity_and_spin_normalization():
    energy = np.array([0., 100., 400., 1500.])
    weights = np.array([1., 3., 5., 7.])
    t, step = 1700., .05
    center = study.molecular_properties(t, study.moments(energy, weights, t), study.NATURAL_MASS_U)
    low = study.molecular_properties(t-step, study.moments(energy, weights, t-step), study.NATURAL_MASS_U)
    high = study.molecular_properties(t+step, study.moments(energy, weights, t+step), study.NATURAL_MASS_U)
    assert (high["Hrel_J_mol"]-low["Hrel_J_mol"])/(2*step) == pytest.approx(center["Cp_J_mol_K"], rel=1e-8)
    assert -(high["Grel_J_mol"]-low["Grel_J_mol"])/(2*step) == pytest.approx(center["S_J_mol_K"], rel=1e-8)
    assert study.moments(energy, weights, .01)["Q"] == 1
    # An erroneous second symmetry division leaves Cp unchanged but changes S/K.
    wrong = study.molecular_properties(t, study.moments(energy, weights/2, t), study.NATURAL_MASS_U)
    assert wrong["Cp_J_mol_K"] == center["Cp_J_mol_K"]
    assert wrong["S_J_mol_K"]-center["S_J_mol_K"] == pytest.approx(-study.R*math.log(2))


def test_distribution_limits_and_completeness_are_not_silently_repaired(report):
    states = json.loads((study.DATA / "state_samples.json").read_text())
    meta = json.loads((study.DATA / "exoames.def.json").read_text())
    assert states["states"] == 3270270
    assert meta["dataset"]["states"]["number_of_states"] == 3270271
    assert states["maximum_energy_cm-1"] == 14999.996460
    assert meta["dataset"]["states"]["max_energy"] == 8000
    assert states["maximum_J"] == 165 and states["all_weights_equal_2J_plus_1"]
    assert states["minimum_energy_cm-1"] == 0 and states["ground_degeneracy"] == 1
    assert meta["irreducible_representations"] == {"A1": 1, "A2": 1, "B1": 0, "B2": 0}
    for row in report["source_samples"]:
        if row["temperature_K"] <= 2000:
            assert abs(row["ExoAmes_direct_over_published_Q"]-1) < 1.13e-6
        else:
            assert row["ExoAmes_direct_over_published_Q"] is None
            assert not row["ExoAmes_within_distributed_PF_range"]
        modern = row["modern"]["ExoAmes"]
        delta = modern["delta_from_NASA"]
        assert -(delta["Hrel_J_mol"]-row["temperature_K"]*delta["S_J_mol_K"])/(study.R*row["temperature_K"]) == pytest.approx(
            math.log(modern["Q_over_NASA"]), abs=2e-14)
    hot = report["source_samples"][-1]
    assert hot["modern"]["ExoAmes"]["Q_over_NASA"] < .76
    assert hot["modern"]["ExoAmes"]["properties"]["Cp_J_mol_K"] < 34
    assert 1-hot["cutoff_diagnostics"]["E_below_14000"]["Q_fraction"] > .058
    assert 1-hot["cutoff_diagnostics"]["J_at_most_160"]["Q_fraction"] > .005


def test_tips_update_is_small_and_derivative_scope_is_explicit(report):
    tables = json.loads((study.DATA / "tips_tables.json").read_text())
    assert tables["2021"]["temperature_K"] == tables["2025"]["temperature_K"]
    assert tables["2025"]["temperature_K"][-1] == 5000
    assert max(abs(a/b-1) for a, b in zip(tables["2025"]["Q"], tables["2021"]["Q"])) < 2.02e-6
    for row in report["source_samples"]:
        tips = row["modern"]["TIPS2025"]
        assert abs(tips["Cp_50K_minus_100K_window_J_mol_K"]) < .047
        assert tips["same_D0_atoms_natural_mass_K_ratio"] == tips["Q_over_NASA"]
        assert tips["same_D0_atoms_isotope_mass_K_proxy"] == pytest.approx(
            tips["Q_over_NASA"]*(study.ISOTOPE_MASS_U/study.NATURAL_MASS_U)**1.5)
    assert report["source_samples"][-1]["modern"]["TIPS2025"]["Q_over_NASA"] == pytest.approx(.9614664981887063)


def test_current_archived_sulfur_budget_and_no_substitution(report):
    footprint = report["footprint"]
    assert footprint["model"] == "curated-h2o-v1"
    assert (footprint["cases"], footprint["species"], footprint["all_converged"]) == (75, 537, True)
    assert footprint["maximum_abs_SO2_h_error"] < 2e-12
    with np.load(study.GRID) as grid:
        sulfur = grid["elements"].tolist().index("S")
        n = np.exp(grid["ln_n"])
        assert n @ grid["formula_matrix"][sulfur] == pytest.approx(grid["budget"][:, sulfur], rel=1e-10, abs=0)
    point = footprint["sulfur_carriers"]["O2S1"]
    assert point["maximum_sulfur_fraction"] == pytest.approx(4.867625278377836e-6)
    assert point["maximum_mole_fraction"] == pytest.approx(1.0925102609349908e-10, rel=1e-10, abs=0)
    assert report["production_change"] == "none" and report["equilibrium_substitution"] == "not_run"


def test_pins_and_hash_guard_reject_changed_originals(tmp_path):
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    for group in ("selected_input_sha256", "output_sha256"):
        for path, digest in receipt[group].items():
            assert sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    sources = json.loads((study.DATA / "sources.json").read_text())
    for source in sources["downloads"]:
        if source["archived"]:
            assert sha256((study.DATA / source["file"]).read_bytes()).hexdigest() == source["sha256"]
    (tmp_path / "exoames.states.bz2").write_bytes(b"changed state file")
    with pytest.raises(ValueError, match="Changed original"):
        study.extract_sources(tmp_path)
    (tmp_path / "hapi.py").write_bytes(b"changed source")
    with pytest.raises(ValueError, match="Changed original"):
        study.extract_tips(tmp_path / "hapi.py")
