"""Check the recovered S4 numerical subset without author code or network."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import sulfur_state_followup as study
from gas_source_comparison import check_equal

DATA = ROOT / "tests/data/validation/s4_state_followup/20261105"


@pytest.fixture(scope="module")
def inputs():
    return json.loads((DATA / "inputs.json").read_text())


def test_frozen_reconstruction_and_receipt():
    check_equal(study.review("s4"), json.loads((DATA / "diagnostic.json").read_text()))
    receipt = json.loads((DATA / "receipt.json").read_text())
    for path, expected in receipt["selected_input_sha256"].items():
        assert sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    for path, expected in receipt["output_sha256"].items():
        assert sha256((DATA / path).read_bytes()).hexdigest() == expected, path


@pytest.mark.parametrize("temperature", [500., 1000., 1500., 3999.])
def test_thermodynamic_derivatives_include_changing_isomer_weights(inputs, temperature):
    step = 0.05
    before, at, after = [study.properties(inputs, t) for t in
                         (temperature-step, temperature, temperature+step)]
    h_key = "H_above_lowest_archived_E0_J_mol"
    cp_fd = (after[h_key]-before[h_key])/(2*step)
    g_before = before[h_key]-(temperature-step)*before["S_J_mol_K"]
    g_after = after[h_key]-(temperature+step)*after["S_J_mol_K"]
    assert cp_fd == pytest.approx(at["Cp_J_mol_K"], abs=3e-6)
    assert -(g_after-g_before)/(2*step) == pytest.approx(at["S_J_mol_K"], abs=3e-6)


def test_coordinates_are_translation_rotation_and_permutation_invariant(inputs):
    changed = deepcopy(inputs)
    for record in changed["states"].values():
        positions = np.array(record["positions_angstrom"])
        record["positions_angstrom"] = (positions[::-1, [1, 2, 0]]+[10., -20., 30.]).tolist()
    before, after = study.properties(inputs,1500.), study.properties(changed,1500.)
    for key in ("H_above_lowest_archived_E0_J_mol", "S_J_mol_K", "Cp_J_mol_K"):
        assert after[key] == pytest.approx(before[key], rel=2e-13)


def test_invalid_modes_and_domain_are_rejected(inputs):
    changed = deepcopy(inputs)
    next(iter(changed["states"].values()))["all_frequencies_cm-1"][0] = -100.
    with pytest.raises(AssertionError):
        study.properties(changed,1000.)
    for temperature in (249.,4001.):
        with pytest.raises(ValueError, match="250-4000"):
            study.properties(inputs,temperature)


def test_physical_w4_energy_and_explicitly_hybrid_temperature_bridge():
    report = study.review("s4")
    energy = report["energy_diagnostic"]
    assert energy["W4_physical_TAE0_J_mol"] == pytest.approx(954579.6)
    assert energy["janaf_balanced_D298_J_mol"] == 962149.
    assert energy["janaf_intrinsic_D0_J_mol"] is None
    assert energy["hybrid_JANAF_Jackson_D0_J_mol"] == pytest.approx(952625.916935,abs=1e-6)
    assert energy["W4_minus_hybrid_D0_J_mol"] == pytest.approx(1953.683065,abs=1e-6)
    assert report["state_count"] == 2 and report["new_equilibrium_solves"] == 0
    assert report["rows"][-1]["listed_over_lowest_Q"] < 1.008
