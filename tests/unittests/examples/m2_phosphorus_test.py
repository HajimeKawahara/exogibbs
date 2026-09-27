"""Finite P enters the primitive source ledger and the exact alloy scalar."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest


DIRECTORY = Path(__file__).resolve().parents[3] / "examples/metal_silicate"
sys.path.insert(0, str(DIRECTORY))
try:
    SOURCE = importlib.import_module("m2_expanded_source")
    PHOSPHORUS = importlib.import_module("m2_phosphorus")
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def eos_checkout():
    eos = pytest.importorskip("exoeos")
    checkout = Path(eos.__file__).resolve().parents[2]
    if not (checkout / "examples/m2_material/phosphorus_reference.py").is_file():
        pytest.skip("Requires the phosphorus ExoEOS provider checkout.")
    return checkout


def build(checkout, tmp_path, **options):
    return SOURCE.build_expanded_bse_problem(
        checkout / "examples/m2_material/bse_inventory.json", checkout,
        tmp_path, sys.executable, gas_model="janaf_condensed", metal_model="phosphorus",
        initialization="canonical", **options)


def test_finite_p_builder_preserves_initial_atoms_and_admits_the_actual_scalar(eos_checkout, tmp_path):
    record, budget, callbacks, initial, metadata = build(eos_checkout, tmp_path)
    names = [n for phase in record["phases"].values() for n in phase]
    formula = np.array([[record["component_formulas"][n].get(e, 0) for n in names]
                        for e in record["elements"]])
    np.testing.assert_allclose(formula @ initial, budget, rtol=1e-12, atol=0)
    seed = np.asarray(metadata["numerical_initialization"]["initial_component_amounts_mol"])
    np.testing.assert_allclose(formula @ seed, budget, rtol=1e-12, atol=0)
    assert initial[names.index("P_metal")] == seed[names.index("P_metal")] == 0
    assert record["phases"]["metal"] == ["Fe_metal", "Si_metal", "O_metal", "H_metal", "P_metal"]
    lower, upper, curvature = SOURCE.phosphorus_metal_domain(metadata)
    assert lower.shape == upper.shape == (5,)
    assert lower[4] == 0 and upper[4] == .02 and curvature > 0
    n = np.array([.96, .003, .007, .025, .005])
    state = callbacks["metal"](2173.15, 1., n)
    energy, derivative = callbacks["metal"].energy_value_and_grad_rt(2173.15, 1., n)
    assert energy == pytest.approx(state.gibbs_rt, abs=1e-12)
    np.testing.assert_allclose(derivative, state.mu_rt, atol=1e-12, rtol=0)
    assert n @ state.mu_rt == pytest.approx(energy, abs=1e-12)
    assert metadata["numerical_execution"]["native_melt_calls"] == 0
    assert metadata["phosphorus_metal"]["standard"]["temperature_continued"]
    assert "Mg/Al/Ca/K/Ti/Cr alloy components" in metadata["atmosphere"]["missing_paths"]


def test_o_h_offsets_and_p_reference_are_separate(eos_checkout, tmp_path):
    base = build(eos_checkout, tmp_path)
    offset = build(eos_checkout, tmp_path, scenario={"standard_offsets_rt": {"O_metal": 1.6, "H_metal": -.4}})
    n = np.array([.96, .003, .007, .025, .005])
    original = base[2]["metal"](2173.15, 1., n)
    shifted = offset[2]["metal"](2173.15, 1., n)
    np.testing.assert_allclose(shifted.mu_rt - original.mu_rt, [0, 0, 1.6, -.4, 0], atol=1e-13)
    assert shifted.gibbs_rt - original.gibbs_rt == pytest.approx(1.6 * n[2] - .4 * n[3], abs=1e-13)
    assert base[-1]["phosphorus_metal"] == offset[-1]["phosphorus_metal"]
    alternative = build(eos_checkout, tmp_path, phosphorus_options={"gas_reference": "P2"})
    assert alternative[-1]["phosphorus_metal"]["standard"]["gas_species"] == "P2"
    assert alternative[-1]["phosphorus_metal"]["standard"]["standard_rt"] != base[-1]["phosphorus_metal"]["standard"]["standard_rt"]


def test_unsupported_models_options_and_four_component_bounds_fail(eos_checkout, tmp_path):
    with pytest.raises(ValueError, match="Four-component"):
        build(eos_checkout, tmp_path, scenario={"metal_bounds": {}})
    with pytest.raises(ValueError, match="Unknown phosphorus"):
        build(eos_checkout, tmp_path, phosphorus_options={"assume_error_bound": True})
    with pytest.raises(ValueError, match="finite number"):
        build(eos_checkout, tmp_path, phosphorus_options={"standard_shift_kcal_mol": True})


def test_zero_phosphorus_retains_independent_scalar_audit_on_present_support(eos_checkout, tmp_path):
    extended = build(eos_checkout, tmp_path)
    base = SOURCE.build_expanded_bse_problem(
        eos_checkout / "examples/m2_material/bse_inventory.json", eos_checkout,
        tmp_path, sys.executable, gas_model="janaf_condensed", initialization="canonical")
    n = np.array([.95, .003, .007, .04, 0.])
    state = extended[2]["metal"](2173.15, 1., n)
    old = base[2]["metal"](2173.15, 1., n[:4])
    energy, gradient = extended[2]["metal"].energy_value_and_grad_rt(2173.15, 1., n)
    assert energy == pytest.approx(state.gibbs_rt, abs=1e-12)
    assert energy == pytest.approx(old.gibbs_rt, abs=1e-12)
    np.testing.assert_allclose(gradient[:4], state.mu_rt[:4], atol=1e-12, rtol=0)
    np.testing.assert_allclose(gradient[:4], old.mu_rt, atol=1e-12, rtol=0)
    assert np.isneginf(gradient[4])
    assert n[:4] @ gradient[:4] == pytest.approx(energy, abs=1e-12)
