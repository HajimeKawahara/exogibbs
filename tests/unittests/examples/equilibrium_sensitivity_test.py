"""Numerical contracts for the gas-only equilibrium response example."""

from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def example():
    path = Path(__file__).resolve().parents[3] / "examples/equilibrium_sensitivity.py"
    spec = importlib.util.spec_from_file_location("equilibrium_sensitivity_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def model(example):
    setup = example.build_setup()
    state = jax.jit(example.make_state_function(setup))
    response = jax.jit(jax.jacrev(lambda theta, t: state(theta, t)[0].ln_n))
    return SimpleNamespace(setup=setup, state=state, response=response)


def test_reaction_controls_conserve_atoms_and_independently_shift_logk(example, model):
    a = np.asarray(model.setup.formula_matrix)
    assert model.setup.species == ("H2", "H2O", "CO", "CO2", "CH4")
    assert model.setup.elements == ("H", "C", "O")
    assert np.linalg.matrix_rank(a) == 3
    np.testing.assert_allclose(a @ example.REACTIONS, 0., atol=1.e-14)
    np.testing.assert_allclose(-example.REACTIONS.T @ example.LOGK_LIFT,
                               np.eye(2), atol=1.e-14)


@pytest.mark.parametrize("temperature", [800., 1000., 1200.])
def test_response_matches_finite_differences_and_equilibrium_tangent(example, model, temperature):
    zero = jnp.zeros(5)
    response = np.asarray(model.response(zero, temperature))
    checks = example.validate_response(model.setup, model.state, temperature, response)
    assert checks["max_fd_absolute_error"] < 2.e-7
    assert checks["max_tangent_absolute_error"] < 2.e-9
    result, _ = model.state(zero, temperature)
    differentiated_budget = np.asarray(model.setup.formula_matrix) @ (
        np.asarray(result.n)[:, None] * response)
    np.testing.assert_allclose(differentiated_budget,
                               np.column_stack((np.diag(example.BUDGET), np.zeros((3, 2)))),
                               atol=2.e-10, rtol=2.e-9)


def test_dilute_comparison_resolves_small_element_budgets(example, model):
    budget = np.array([1., 2.7e-4, 4.9e-4])
    state = jax.jit(example.make_state_function(model.setup, budget))
    response = jax.jacrev(lambda theta: state(theta, 1200.)[0].ln_n)(jnp.zeros(5))
    checks = example.validate_response(model.setup, state, 1200., np.asarray(response), budget)
    assert checks["max_relative_budget_error"] < 2.e-10


def test_temperature_and_pressure_follow_reaction_response_chain_rule(example, model):
    temperature = 1000.
    zero = jnp.zeros(5)
    response = np.asarray(model.response(zero, temperature))
    thermal, pressure = jax.jacrev(
        lambda logt, logp: model.state(zero, jnp.exp(logt), jnp.exp(logp))[0].ln_n,
        argnums=(0, 1),
    )(jnp.log(temperature), jnp.log(example.PRESSURE_BAR))
    logk_thermal = jax.jacrev(
        lambda logt: -jnp.asarray(example.REACTIONS).T @ model.setup.hvector_func(jnp.exp(logt))
    )(jnp.log(temperature))
    np.testing.assert_allclose(thermal, response[:, 3:] @ logk_thermal,
                               atol=2.e-9, rtol=2.e-9)
    np.testing.assert_allclose(pressure, 2 * response[:, 3], atol=2.e-9, rtol=2.e-9)


def test_element_energy_gauge_preserves_equilibrium_and_responses(example, model):
    offset = model.setup.formula_matrix.T @ jnp.array([1.2, -0.7, 2.3])
    shifted = replace(model.setup,
                      hvector_func=lambda t: model.setup.hvector_func(t) + offset)
    shifted_state = jax.jit(example.make_state_function(shifted))
    zero = jnp.zeros(5)
    reference, _ = model.state(zero, 1000.)
    result, diagnostics = shifted_state(zero, 1000.)
    example.require_converged(result, diagnostics)
    np.testing.assert_allclose(result.ln_n, reference.ln_n, atol=2.e-10, rtol=0)
    shifted_response = jax.jacrev(lambda theta: shifted_state(theta, 1000.)[0].ln_n)(zero)
    np.testing.assert_allclose(shifted_response, model.response(zero, 1000.),
                               atol=2.e-9, rtol=2.e-9)


def test_common_budget_scaling_changes_amounts_but_not_mole_fractions(example, model):
    zero = jnp.zeros(5)
    base, _ = model.state(zero, 1000.)
    scaled, diagnostics = model.state(jnp.array([jnp.log(3.), jnp.log(3.), jnp.log(3.), 0., 0.]),
                                      1000.)
    example.require_converged(scaled, diagnostics)
    np.testing.assert_allclose(scaled.n, 3 * base.n, atol=2.e-10, rtol=2.e-9)
    np.testing.assert_allclose(scaled.x, base.x, atol=2.e-10, rtol=2.e-9)
    response = np.asarray(model.response(zero, 1000.))
    fraction_response = response - np.asarray(base.x) @ response
    np.testing.assert_allclose(response[:, :3].sum(axis=1), 1., atol=2.e-9)
    np.testing.assert_allclose(fraction_response[:, :3].sum(axis=1), 0., atol=2.e-9)


@pytest.mark.parametrize("converged,residual,log_amount", [
    (False, 1.e-3, 0.), (True, np.nan, 0.), (True, 1.e-12, np.nan),
])
def test_nonconverged_or_nonfinite_states_are_rejected(example, converged, residual, log_amount):
    result = SimpleNamespace(ln_n=np.array([log_amount]))
    diagnostics = {"converged": converged, "final_residual": residual}
    with pytest.raises(RuntimeError, match="finite converged equilibrium"):
        example.require_converged(result, diagnostics)
