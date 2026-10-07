"""Declared standard offsets preserve extensive energy and its derivative."""

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


DIRECTORY = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate"
sys.path.insert(0, str(DIRECTORY))
try:
    SCENARIOS = importlib.import_module("m2_scenarios")
    FULL = importlib.import_module("full_potential")
    COUPLED = importlib.import_module("melts_coupled")
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


def test_measured_oxygen_standard_offset_preserves_energy_derivative_and_curvature():
    record = {"phases": {"metal": ["Fe_metal", "Si_metal", "O_metal", "H_metal"]}}
    original = FULL.ideal_phase(lambda t, p: np.array([-2., -3., -5., -7.]))
    original.energy_value_and_grad_rt = lambda t, p, n: (original(t, p, n).gibbs_rt,
                                                        original(t, p, n).mu_rt)
    # Example input from the ExoEOS Sakao standard reconstruction at 2173.15 K.
    # This is a conditional standard scenario, not a calibrated BSE uncertainty.
    offset = 1.6318386164676255
    changed = SCENARIOS.apply_standard_offsets(record, {"metal": original},
        {"standard_offsets_rt": {"O_metal": offset}})["metal"]
    n = np.array([.96, .005, .015, .02])
    old, new = original(2173.15, 267.2, n), changed(2173.15, 267.2, n)
    np.testing.assert_allclose(new.mu_rt-old.mu_rt, [0., 0., offset, 0.], atol=2e-15)
    assert new.gibbs_rt-old.gibbs_rt == pytest.approx(n[2]*offset)
    assert new.gibbs_rt == pytest.approx(n @ new.mu_rt)
    energy, gradient = changed.energy_value_and_grad_rt(2173.15, 267.2, n)
    assert energy == pytest.approx(new.gibbs_rt)
    np.testing.assert_allclose(gradient, new.mu_rt)
    for i in range(4):
        step = np.eye(4)[i]*1e-6
        derivative = (changed(2173.15, 267.2, n+step).gibbs_rt
                      -changed(2173.15, 267.2, n-step).gibbs_rt)/2e-6
        assert derivative == pytest.approx(new.mu_rt[i], abs=2e-8)
        curvature_old = (original(2173.15, 267.2, n+step).mu_rt
                         -original(2173.15, 267.2, n-step).mu_rt)/2e-6
        curvature_new = (changed(2173.15, 267.2, n+step).mu_rt
                         -changed(2173.15, 267.2, n-step).mu_rt)/2e-6
        np.testing.assert_allclose(curvature_new, curvature_old, atol=2e-9)
    assert changed(2173.15, 267.2, 3*n).gibbs_rt == pytest.approx(3*new.gibbs_rt)
    assert SCENARIOS.normalize_scenario()["standard_offsets_rt"]["O_metal"] == 0.


def test_reconstructed_water_capacity_offset_leaves_molecular_h2_unchanged():
    record = {"phases": {"silicate": ["sio2_melts", "h2o_melts", "H2_dissolved"]}}
    original = FULL.ideal_phase(lambda t, p: np.array([-2., -5., -7.]))
    original.energy_value_and_grad_rt = lambda t, p, n: (original(t, p, n).gibbs_rt, original(t, p, n).mu_rt)
    offset = float(2*np.log(2.265))
    changed = SCENARIOS.apply_standard_offsets(record, {"silicate": original},
        {"standard_offsets_rt": {"h2o_melts": offset}})["silicate"]
    n = np.array([.96, .03, .01])
    old, new = original(2173.15, 270., n), changed(2173.15, 270., n)
    np.testing.assert_allclose(new.mu_rt-old.mu_rt, [0., offset, 0.], atol=2e-15)
    assert new.gibbs_rt-old.gibbs_rt == pytest.approx(n[1]*offset)
    energy, gradient = changed.energy_value_and_grad_rt(2173.15, 270., n)
    assert energy == pytest.approx(new.gibbs_rt)
    np.testing.assert_allclose(gradient, new.mu_rt)
    for i in range(3):
        step = np.eye(3)[i]*1e-6
        derivative = (changed(2173.15, 270., n+step).gibbs_rt-changed(2173.15, 270., n-step).gibbs_rt)/2e-6
        assert derivative == pytest.approx(new.mu_rt[i], abs=2e-8)
        difference = ((changed(2173.15, 270., n+step).mu_rt-changed(2173.15, 270., n-step).mu_rt)
                      -(original(2173.15, 270., n+step).mu_rt-original(2173.15, 270., n-step).mu_rt))/2e-6
        np.testing.assert_allclose(difference, 0., atol=2e-9)
    assert changed(2173.15, 270., 3*n).gibbs_rt == pytest.approx(3*new.gibbs_rt)
    assert SCENARIOS.normalize_scenario()["standard_offsets_rt"]["h2o_melts"] == 0.


def test_saved_water_host_offset_matches_phase_and_preserves_gas_receipt():
    original = FULL.ideal_phase(lambda t, p: np.array([-2., -5.]))
    receipts = [{"H2O_gas_standard_RT": -3.}]
    def evaluate(t, p, n):
        state=original(t,p,n);rt=COUPLED.COMMON_R*t
        return {"gibbs_RT":state.gibbs_rt,"gibbs_J":state.gibbs_rt*rt,
                "mu_RT":list(state.mu_rt),"mu_J_mol":list(state.mu_rt*rt),
                "basis":{"common_R_J_mol_K":COUPLED.COMMON_R},
                "water_reconstruction":{"gas_H2O_standard_RT":-3.}}
    provider=SimpleNamespace(MODEL_ID=COUPLED.WATER_MODEL_ID,COMPONENTS=["sio2","h2o"],
        evaluate_liquid=evaluate,water_standard_receipts=receipts,
        energy_value_and_grad_rt=lambda t,p,n:(original(t,p,n).gibbs_rt,original(t,p,n).mu_rt))
    offset=1.6351495178699949
    saved=COUPLED.with_saved_water_standard_offset(provider,offset)
    phase=SCENARIOS.apply_standard_offsets({"phases":{"silicate":["sio2_melts","h2o_melts"]}},
            {"silicate":original},{"standard_offsets_rt":{"h2o_melts":offset}})["silicate"]
    n=np.array([.97,.03]);state=saved.evaluate_liquid(2173.15,270.,n)
    expected=phase(2173.15,270.,n)
    assert state["gibbs_RT"] == pytest.approx(expected.gibbs_rt)
    np.testing.assert_allclose(state["mu_RT"],expected.mu_rt)
    energy,gradient=saved.energy_value_and_grad_rt(2173.15,270.,n)
    assert energy == pytest.approx(expected.gibbs_rt)
    np.testing.assert_allclose(gradient,expected.mu_rt)
    assert state["water_reconstruction"]["gas_H2O_standard_RT"] == -3.
    assert state["water_reconstruction"]["standard_offset_rt"] == offset
    assert saved.water_standard_receipts is receipts
    assert COUPLED.with_saved_water_standard_offset(provider,0.) is provider
    with pytest.raises(ValueError,match="already applied"):
        COUPLED.with_saved_water_standard_offset(saved,offset)
    assert COUPLED.saved_liquid_model({"source_metadata":{"liquid_model":"published_water",
        "host_ledger":{"model_id":COUPLED.WATER_MODEL_ID}}}) == "published_water"
