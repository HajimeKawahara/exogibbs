"""Finite-inventory controls for full-source nested magma--gas exchange."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.applications.magma_gas import MagmaGasInit, MagmaGasOptions
from exogibbs.equilibrium.gas.types import EquilibriumOptions


EXAMPLE_DIRECTORY = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate"
SPEC = importlib.util.spec_from_file_location(
    "metal_silicate_gas_exchange", EXAMPLE_DIRECTORY / "gas_exchange.py",
)
EXCHANGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXCHANGE
sys.path.insert(0, str(EXAMPLE_DIRECTORY))
try:
    SPEC.loader.exec_module(EXCHANGE)
finally:
    sys.path.pop(0)
RECORD = json.loads((EXAMPLE_DIRECTORY / "reference.json").read_text())
REFERENCE = json.loads((EXAMPLE_DIRECTORY / "equilibrium_reference.json").read_text())
SOURCE_CASES = [case for case in REFERENCE["cases"] if case["id"].startswith("source_full_")]
OPTIONS = MagmaGasOptions(root_tolerance=1e-10, max_iter=30)


def _problem(index=0):
    case = SOURCE_CASES[index]
    thermo = next(case0 for case0 in RECORD["cases"] if case0["T_K"] == case["T_K"])
    return EXCHANGE.build_problem(RECORD, thermo, case), case


@pytest.mark.parametrize("index", [0, 1])
def test_nested_gas_reproduces_independent_full_source_reference(index):
    problem, case = _problem(index)
    initial = MagmaGasInit(problem.model.reference_root + jnp.linspace(-0.15, 0.15, 22))
    result = EXCHANGE.solve(
        problem, case["T_K"], case["P_bar"], case["element_amounts_mol"], options=OPTIONS,
        init=initial,
    )
    state = result.model_state

    assert bool(result.diagnostics.converged)
    assert bool(result.diagnostics.outer_converged)
    assert bool(result.diagnostics.inner_converged)
    assert result.diagnostics.inner_iterations > 0
    assert result.root_variables.shape == (22,)
    assert state.reaction_residual.shape == (18,)
    np.testing.assert_allclose(state.component_amounts_mol, case["component_amounts_mol"], rtol=1e-8, atol=0)
    np.testing.assert_allclose(state.phase_amounts_mol, case["phase_amounts_mol"], rtol=1e-9)
    np.testing.assert_allclose(state.reaction_residual, 0, atol=1e-8)
    np.testing.assert_allclose(state.element_residual, 0, atol=1e-9)
    np.testing.assert_allclose(
        state.phase_element_amounts_mol.sum(axis=0), case["element_amounts_mol"], rtol=1e-9,
    )
    for phase_index, section in enumerate((slice(0, 11), slice(11, 15), slice(15, 25))):
        np.testing.assert_allclose(state.mole_fractions[section].sum(), 1, atol=1e-13)
        np.testing.assert_allclose(
            state.phase_element_amounts_mol[phase_index],
            problem.model.formula_matrix[:, section] @ state.component_amounts_mol[section],
            rtol=1e-12,
        )
    # The inner abundance normalization does not define the physical gas mass.
    assert result.element_abundances[2] == 1
    np.testing.assert_allclose(state.component_amounts_mol[15:].sum(), state.phase_amounts_mol[2])
    assert not np.isclose(float(result.gas.equilibrium.ntot), float(state.phase_amounts_mol[2]))


def test_finite_hydrogen_exchange_and_uniform_budget_scaling_are_differentiable():
    problem, case = _problem()
    budget = jnp.asarray(case["element_amounts_mol"])

    def calculate(temperature, amounts):
        return EXCHANGE.solve(problem, temperature, case["P_bar"], amounts, options=OPTIONS)

    compiled = jax.jit(calculate)
    base = compiled(case["T_K"], budget)
    scaled = compiled(case["T_K"], 1e4 * budget)
    np.testing.assert_allclose(scaled.model_state.mole_fractions, base.model_state.mole_fractions, rtol=1e-9)
    np.testing.assert_allclose(
        scaled.model_state.component_amounts_mol,
        1e4 * base.model_state.component_amounts_mol, rtol=1e-9,
    )

    enriched = compiled(case["T_K"], budget.at[4].multiply(1.1))
    assert bool(enriched.diagnostics.converged)
    np.testing.assert_allclose(enriched.model_state.element_residual, 0, atol=1e-9)
    np.testing.assert_allclose(enriched.model_state.reaction_residual, 0, atol=1e-8)
    np.testing.assert_allclose(
        enriched.model_state.phase_element_amounts_mol[:, 4].sum(), 1.1 * budget[4], rtol=1e-9,
    )
    # Each phase holds finite H, and the gas changes as that common budget changes.
    assert np.all(np.asarray(enriched.model_state.phase_element_amounts_mol[:, 4]) > 0)
    gas_hydrogen_change = (enriched.model_state.phase_element_amounts_mol[2, 4]
                           - base.model_state.phase_element_amounts_mol[2, 4])
    assert abs(float(gas_hydrogen_change)) > 1

    def metal_hydrogen(parameters):
        temperature, hydrogen = parameters
        return calculate(temperature, budget.at[4].set(hydrogen)).model_state.component_amounts_mol[14]

    parameters = jnp.asarray([case["T_K"], budget[4]])
    derivative = jax.jit(jax.grad(metal_hydrogen))(parameters)
    finite_difference = []
    for index, step in enumerate([0.01, 0.001]):
        offset = jnp.zeros(2).at[index].set(step)
        plus, minus = parameters + offset, parameters - offset
        upper = compiled(plus[0], budget.at[4].set(plus[1]))
        lower = compiled(minus[0], budget.at[4].set(minus[1]))
        assert bool(upper.diagnostics.converged & lower.diagnostics.converged)
        finite_difference.append(
            (upper.model_state.component_amounts_mol[14]
             - lower.model_state.component_amounts_mol[14]) / (2 * step)
        )
    assert np.all(np.isfinite(derivative))
    assert np.all(np.abs(derivative) > 1e-6)
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-5, atol=1e-8)


@pytest.mark.parametrize("failed_inner", [False, True])
def test_failed_nested_solution_reports_failure_and_nan_gradient(failed_inner):
    problem, case = _problem()
    budget = jnp.asarray(case["element_amounts_mol"])
    options = MagmaGasOptions(
        root_tolerance=1e-10, max_iter=0,
        equilibrium_options=EquilibriumOptions(max_iter=0 if failed_inner else 1000),
    )
    initial = MagmaGasInit(problem.model.reference_root + jnp.linspace(-0.03, 0.03, 22))

    def calculate(hydrogen):
        return EXCHANGE.solve(
            problem, case["T_K"], case["P_bar"], budget.at[4].set(hydrogen),
            init=initial, options=options,
        )

    result = calculate(budget[4])
    assert not bool(result.diagnostics.converged)
    assert not bool(result.diagnostics.outer_converged)
    assert bool(result.diagnostics.inner_converged) != failed_inner
    derivative = jax.grad(
        lambda hydrogen: calculate(hydrogen).model_state.component_amounts_mol[14]
    )(budget[4])
    assert np.isnan(derivative)


@pytest.mark.parametrize(
    "budget", [[1.0] * 6, [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
               [1.0] * 6 + [-1.0], [1.0] * 6 + [np.nan]],
)
def test_rejects_changed_or_invalid_full_source_support(budget):
    problem, case = _problem()
    with pytest.raises(ValueError, match="shape|positive"):
        EXCHANGE.solve(problem, case["T_K"], case["P_bar"], budget)


@pytest.mark.parametrize("temperature, pressure", [(0.0, 1.0), (2350.0, 0.0), (np.inf, 1.0)])
def test_rejects_nonpositive_or_nonfinite_local_conditions(temperature, pressure):
    problem, case = _problem()
    with pytest.raises(ValueError, match="finite and positive"):
        EXCHANGE.solve(problem, temperature, pressure, case["element_amounts_mol"])


@pytest.mark.parametrize("reordered", ["metal", "gas", "reactions", "initial"])
def test_rejects_reordered_source_or_initial_component_basis(reordered):
    record, case = deepcopy(RECORD), deepcopy(SOURCE_CASES[0])
    if reordered in ("metal", "gas"):
        record["phases"][reordered].reverse()
    elif reordered == "reactions":
        record["reactions"].reverse()
    else:
        case["active_components"].reverse()
    with pytest.raises(ValueError, match="order"):
        EXCHANGE.build_problem(record, record["cases"][0], case)


def test_rejects_changed_gas_activity_assumption():
    problem, case = _problem()
    problem = replace(problem, lnphi_func=lambda temperature, pressure, composition: jnp.zeros(10))
    with pytest.raises(ValueError, match="assumes ideal gas"):
        EXCHANGE.solve(problem, case["T_K"], case["P_bar"], case["element_amounts_mol"])
