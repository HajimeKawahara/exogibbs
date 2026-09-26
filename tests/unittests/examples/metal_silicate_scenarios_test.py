"""Declared standard offsets preserve extensive energy and its derivative."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest


DIRECTORY = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate"
sys.path.insert(0, str(DIRECTORY))
try:
    SCENARIOS = importlib.import_module("m2_scenarios")
    FULL = importlib.import_module("full_potential")
finally:
    sys.path.pop(0)


def test_standard_offsets_shift_energy_gradient_and_scale_extensively():
    record = {"phases": {"silicate": ["sio2_melts", "H2_dissolved"],
                         "metal": ["Fe_metal", "H_metal"]}}
    callback = FULL.ideal_phase(lambda t, p: np.zeros(2))
    callback.energy_value_and_grad_rt = lambda t, p, n: (callback(t, p, n).gibbs_rt,
                                                         callback(t, p, n).mu_rt)
    callbacks = {name: callback for name in record["phases"]}
    scenario = {"standard_offsets_rt": {"H2_dissolved": 1.2, "H_metal": -.7}}
    shifted = SCENARIOS.apply_standard_offsets(record, callbacks, scenario)
    n = np.array([.7, .3])
    for phase, shift in (("silicate", 1.2), ("metal", -.7)):
        state = shifted[phase](2173.15, 80., n)
        assert state.gibbs_rt == pytest.approx(callback(2173.15, 80., n).gibbs_rt + n[1] * shift)
        energy, gradient = shifted[phase].energy_value_and_grad_rt(2173.15, 80., n)
        assert energy == pytest.approx(state.gibbs_rt)
        np.testing.assert_allclose(gradient, state.mu_rt)
        for index in range(2):
            step = np.eye(2)[index] * 1e-6
            derivative = (shifted[phase](2173.15, 80., n + step).gibbs_rt
                          - shifted[phase](2173.15, 80., n - step).gibbs_rt) / 2e-6
            assert derivative == pytest.approx(state.mu_rt[index], abs=1e-9)
        assert shifted[phase](2173.15, 80., 8 * n).gibbs_rt == pytest.approx(8 * state.gibbs_rt)
    assert SCENARIOS.apply_standard_offsets(record, callbacks, None)["metal"] is callback


def test_domain_override_preserves_defaults_and_returns_independent_arrays():
    lower, upper = SCENARIOS.metal_selection_domain()
    np.testing.assert_array_equal(lower, [.86, 0, 0, 0])
    np.testing.assert_array_equal(upper, [1., .08, .02, .04])
    _, changed = SCENARIOS.metal_selection_domain({"metal_bounds": {"upper": [1., .08, .04, .04]}})
    assert changed[2] == .04
    changed[2] = .8
    assert SCENARIOS.metal_selection_domain()[1][2] == .02


@pytest.mark.parametrize("scenario", [False, {"unknown": 0}, {"standard_offsets_rt": {"H_metal": True}},
    {"standard_offsets_rt": {"H_metal": float("nan")}},
    {"standard_offsets_rt": {"H_metal": 10 ** 1000}},
    {"standard_offsets_rt": {"H2": 1}}, {"metal_bounds": {"lower": [1, 1, 0, 0]}},
    {"metal_bounds": {"upper": [1, .08, False, .04]}}, {"metal_bounds": {"lower": [0, 0, 0, 0]}}])
def test_invalid_scenarios_are_rejected(scenario):
    with pytest.raises(ValueError):
        SCENARIOS.normalize_scenario(scenario)
