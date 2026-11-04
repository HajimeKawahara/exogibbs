"""Check OH source identity, balanced anchors, spin conventions and calorics."""

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
import oh_source_review as study
from gas_source_comparison import check_equal


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_balanced_energy_and_decomposition_reproduce_frozen_screen(report):
    actual = deepcopy(report)
    expected = json.loads((study.DATA / "comparison.json").read_text())
    # Cancellation in differences of O(1e5 J/mol) enthalpies is platform-dependent.
    for a, b in zip(actual["source_samples"], expected["source_samples"]):
        for name in study.DATASETS:
            key = "delta_H_from_nasa_common_R_J_mol"
            assert a["modern"][name].pop(key) == pytest.approx(
                b["modern"][name].pop(key), abs=1e-8, rel=0)
    check_equal(actual, expected)
    assert report["balanced_D0_J_mol"] == pytest.approx(
        {"atct_1_172": (216.034+246.844-37.274)*1000,
         "nasa": 425785.9, "janaf": 424435.0}, abs=1e-7)
    assert report["ATcT_minus_nasa_D0_over_k_K"] == pytest.approx(
        425604/study.major.R-425785.9/8.31451, abs=1e-10)
    baseline = json.loads((study.BASELINE / "comparison.json").read_text())
    for row in report["source_samples"]:
        assert sum(row["janaf_minus_nasa_ln_K_components"].values()) == pytest.approx(
            row["janaf_ln_K"]-row["nasa_ln_K"], abs=5e-14)
        old = next((r for r in baseline["samples"] if r["species_id"] == "H1O1"
                    and r["temperature_K"] == row["temperature_K"]), None)
        if old:
            for source in ("nasa", "janaf", "curated"):
                assert row[source+"_ln_K"] == pytest.approx(old["ln_Kp"][source], abs=2e-11)
    with pytest.raises(ValueError, match="outside"):
        study.curated_ln_k(1000)


def test_native_levels_have_half_integer_J_spin_two_and_actual_ground_zero():
    inventory = json.loads((study.DATA / "state_inventory.json").read_text())["datasets"]
    with np.load(study.DATA / "states.npz") as archive:
        for name, count in (("MYTHOS", 10606), ("MoLLIST-OH", 1878)):
            states = archive[name]
            assert len(states) == count == inventory[name]["state_count"]
            assert np.all(states[:, 1] == 2*(2*states[:, 2]+1))
            assert np.all(states[:, 2] % 1 == .5)
            ground = states[np.argmin(states[:, 0])]
            assert ground[:3].tolist() == [0, 8, 1.5]
            result = study.moments(states, .001, int(ground[3]), 425604/study.major.R)
            assert result["Q"] == pytest.approx(4, abs=1e-12)
        metadata = json.loads((study.DATA / "MoLLIST-OH.def.json").read_text())
        assert metadata["dataset"]["states"]["number_of_states"] == 2640
        assert len(archive["MoLLIST-OH"]) != 2640  # Distribution/metadata inconsistency retained.


def test_boltzmann_moments_obey_independent_caloric_identity(report):
    with np.load(study.DATA / "states.npz") as archive:
        states = archive["MYTHOS"]
        for t in (1500., 4000.):
            dt = .1
            lo = study.moments(states, t-dt, 3, 425604/study.major.R)
            hi = study.moments(states, t+dt, 3, 425604/study.major.R)
            mid = study.moments(states, t, 3, 425604/study.major.R)
            assert t*t*(math.log(hi["Q"])-math.log(lo["Q"]))/2/dt == pytest.approx(
                mid["mean_energy_K"], rel=2e-9)
            assert (hi["mean_energy_K"]-lo["mean_energy_K"])/2/dt == pytest.approx(
                mid["variance_energy_K2"]/t**2, rel=2e-9)
    for row in report["source_samples"]:
        for modern in row["modern"].values():
            dh = modern["delta_H_from_nasa_common_R_J_mol"]
            ds = modern["delta_S_from_nasa_common_R_J_mol_K"]
            t = row["temperature_K"]
            assert -(dh-t*ds)/(study.major.R*t) == pytest.approx(
                math.log(modern["Q_over_nasa_internal"]), abs=2e-14)


def test_primary_caloric_nodes_confirm_janaf_issue_without_claiming_independence(report):
    literature = json.loads((study.DATA / "literature.json").read_text())
    rows = {r["temperature_K"]: r for r in report["source_samples"]}
    for t, cp, entropy, increment in literature["iupac2005"]["nodes"]:
        modern = rows[t]["modern"]["MYTHOS"]
        assert abs(modern["Cp_J_mol_K"]-cp) < .04
        assert abs(modern["S_natural_mass_J_mol_K"]-entropy) < .016
        assert abs(modern["H_increment_J_mol"]/1000-increment) < .035
    low = rows[298.15]
    assert low["janaf_OH_H_increment_J_mol"] == 9172
    assert low["modern"]["MYTHOS"]["H_increment_J_mol"] == pytest.approx(8813.5584, abs=.001)
    assert rows[4000]["janaf_OH_Cp_J_mol_K"] == 37.885
    assert "not independent" in literature["iupac2005"]["ancestry"]


def test_distribution_and_high_temperature_sensitivities_are_scoped(report):
    for row in report["source_samples"]:
        modern = row["modern"]["MYTHOS"]
        if modern["published_Q_with_spin"]:
            assert abs(2*modern["Q"]-modern["published_Q_with_spin"]) <= 5.1e-5
        assert abs(modern["Q_over_nasa_internal"]-1) < .001
        assert modern["above_D0_Q_fraction"] < 3.36e-5
        assert modern["highest_J_shell_Q_fraction"] < 1.97e-6
        assert abs(modern["Cp_change_on_D0_cut_J_mol_K"]) < .0382
    high = report["source_samples"][-1]
    mollist = high["modern"]["MoLLIST-OH"]
    assert abs(2*mollist["Q"]-mollist["published_Q_with_spin"]) > .00005
    assert 1.0534 < high["modern"]["MYTHOS"]["ATcT_same_atoms_over_curated"] < 1.0535
    first = next(r for r in report["source_samples"] if r["temperature_K"] == 1500)
    assert 1.1292 < first["modern"]["MYTHOS"]["ATcT_same_atoms_over_curated"] < 1.1293


def test_current_footprint_closes_oxygen_and_is_not_a_substitution(report):
    f = report["footprint"]
    assert (f["model"], f["cases"], f["species"], f["all_converged"]) == (
        "curated-h2o-v1", 75, 537, True)
    assert f["maximum_abs_h_error"] < 2e-12
    assert f["maximum_relative_oxygen_residual"] < 1e-10
    assert f["case_index"] == 23
    assert f["maximum_oxygen_fraction"] == pytest.approx(.11239064812028421)
    assert report["production_change"] == "none"
    assert report["equilibrium_substitution"] == "not_run"


def test_offline_evidence_inputs_are_pinned():
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    for group in ("selected_input_sha256", "output_sha256"):
        for path, digest in receipt[group].items():
            assert sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    sources = json.loads((study.DATA / "sources.json").read_text())
    for source in sources["downloads"]:
        if source["archived"]:
            assert sha256((study.DATA / source["file"]).read_bytes()).hexdigest() == source["archive_sha256"]
