"""Numerical controls for the example's fixed-phase local root."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.applications.magma_gas.types import MagmaGasOptions


EXAMPLE_DIRECTORY = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate"
SPEC = importlib.util.spec_from_file_location("metal_silicate_local", EXAMPLE_DIRECTORY / "local.py")
LOCAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LOCAL
SPEC.loader.exec_module(LOCAL)
RECORD = json.loads((EXAMPLE_DIRECTORY / "reference.json").read_text())
SPECIES = tuple(name for names in RECORD["phases"].values() for name in names)
FORMULA = np.asarray([
    [RECORD["component_formulas"][name].get(element, 0) for name in SPECIES]
    for element in RECORD["elements"]
], dtype=np.float64)
DRY_INDICES = np.asarray([SPECIES.index(name) for name in (
    "MgO_silicate", "SiO2_silicate", "MgSiO3_silicate", "FeO_silicate", "FeSiO3_silicate",
    "Fe_metal", "Si_metal", "O_metal",
)])


def _regular_activity(temperature, pressure, composition):
    del pressure
    interaction = 600.0 / temperature
    return interaction * (0.5 * (1 + jnp.sum(composition**2)) - composition)


def _control(*, nonideal=False, elemental_shift=False):
    fractions = np.asarray([0.2, 0.1, 0.5, 0.15, 0.05, 0.8, 0.15, 0.05])
    amounts = np.zeros(25)
    amounts[DRY_INDICES] = fractions * np.asarray([2] * 5 + [1] * 3)
    budgets = FORMULA @ amounts
    standard = np.zeros(25)
    standard[DRY_INDICES] = -np.log(fractions)
    if nonideal:
        standard[DRY_INDICES[5:]] -= np.asarray(_regular_activity(1200.0, 1.0, jnp.asarray(fractions[5:])))
    if elemental_shift:
        standard += FORMULA.T @ np.asarray([3., -2., 4., 1., 7., -5., 8.])

    def standards(temperature, pressure):
        del pressure
        return jnp.asarray(standard).at[DRY_INDICES[0]].add((temperature - 1200.0) / 1200.0)

    problem = LOCAL.build_problem(
        RECORD, budgets, standards,
        activity_functions={"metal": _regular_activity} if nonideal else None,
    )
    return problem, budgets, amounts


def _initial(amounts):
    return amounts * np.linspace(0.7, 1.3, len(amounts))


@pytest.mark.parametrize("nonideal", [False, True])
def test_known_solution_conserves_elements_and_phase_amounts(nonideal):
    problem, budgets, expected = _control(nonideal=nonideal)
    result = LOCAL.solve(problem, 1200.0, 1.0, budgets, initial_component_amounts_mol=_initial(expected))
    assert bool(result.root_solution.converged)
    assert result.root_solution.inner_iterations == 0
    assert problem.reaction_ids == ("R1", "R2", "R3", "R5")
    assert problem.formula_matrix.shape == (4, 8)
    assert problem.reaction_matrix.shape == (4, 8)
    np.testing.assert_allclose(result.component_amounts_mol, expected, atol=1e-11, rtol=1e-9)
    np.testing.assert_allclose(result.phase_amounts_mol, [2, 1], atol=1e-10)
    np.testing.assert_allclose(result.element_residual, 0, atol=1e-9)
    np.testing.assert_allclose(result.reaction_residual, 0, atol=1e-8)
    np.testing.assert_allclose(result.phase_element_amounts_mol.sum(axis=0), budgets, atol=1e-10)
    np.testing.assert_array_equal(result.component_amounts_mol[np.setdiff1d(np.arange(25), DRY_INDICES)], 0)
    for phase, indices in enumerate((DRY_INDICES[:5], DRY_INDICES[5:])):
        np.testing.assert_allclose(result.phase_element_amounts_mol[phase], FORMULA[:, indices] @ expected[indices], atol=1e-10)
        np.testing.assert_allclose(result.mole_fractions[indices].sum(), 1, atol=1e-14)


def test_uniform_budget_scaling_and_common_elemental_reference_are_invariant():
    problem, budgets, expected = _control(nonideal=True)
    base = LOCAL.solve(problem, 1200.0, 1.0, budgets, initial_component_amounts_mol=_initial(expected))
    for scale in (1e-8, 1e8):
        result = LOCAL.solve(problem, 1200.0, 1.0, scale * budgets, initial_component_amounts_mol=scale * _initial(expected))
        assert bool(result.root_solution.converged)
        np.testing.assert_allclose(result.mole_fractions, base.mole_fractions, atol=1e-11)
        np.testing.assert_allclose(result.phase_amounts_mol, scale * base.phase_amounts_mol, rtol=1e-10)
        np.testing.assert_allclose(result.root_solution.root_variables, base.root_solution.root_variables, atol=1e-11)
    shifted, _, _ = _control(nonideal=True, elemental_shift=True)
    result = LOCAL.solve(shifted, 1200.0, 1.0, budgets, initial_component_amounts_mol=_initial(expected))
    np.testing.assert_allclose(result.component_amounts_mol, base.component_amounts_mol, atol=1e-10)


def test_zero_interaction_recovers_explicit_ideal_solution():
    problem, budgets, expected = _control()
    ideal = LOCAL.solve(problem, 1200.0, 1.0, budgets, initial_component_amounts_mol=_initial(expected))
    zero = LOCAL.build_problem(
        RECORD, budgets, problem.standard_potentials_rt,
        activity_functions={"metal": lambda t, p, x: 0 * _regular_activity(t, p, x)},
    )
    result = LOCAL.solve(zero, 1200.0, 1.0, budgets, initial_component_amounts_mol=_initial(expected))
    np.testing.assert_allclose(result.component_amounts_mol, ideal.component_amounts_mol, atol=1e-11)


def test_jit_and_implicit_temperature_gradient_include_nonideal_composition_feedback():
    problem, budgets, expected = _control(nonideal=True)

    def amount(temperature):
        return LOCAL.solve(problem, temperature, 1.0, jnp.asarray(budgets), initial_component_amounts_mol=_initial(expected)).component_amounts_mol[12]

    compiled = jax.jit(amount)
    derivative = jax.jit(jax.grad(amount))(1200.0)
    finite_difference = (compiled(1200.1) - compiled(1199.9)) / 0.2
    assert np.isfinite(derivative)
    assert abs(derivative) > 1e-6
    np.testing.assert_allclose(derivative, finite_difference, atol=1e-9, rtol=2e-5)
    np.testing.assert_allclose(compiled(1200.0), 0.15, atol=1e-10)


def test_failed_solve_retains_residuals_and_nan_reverse_derivative():
    problem, budgets, _ = _control(nonideal=True)
    options = MagmaGasOptions(max_iter=0)

    def run(temperature):
        return LOCAL.solve(problem, temperature, 1.0, budgets, options=options)

    result = run(1200.0)
    assert not bool(result.root_solution.converged)
    assert result.root_solution.residual_norm > 1e-3
    assert np.all(np.isfinite(result.root_solution.residual))
    derivative = jax.grad(lambda t: run(t).component_amounts_mol[12])(1200.0)
    assert np.isnan(derivative)


def test_log_balance_remains_finite_for_a_very_small_positive_initial_amount():
    problem, budgets, _ = _control()
    conditions = LOCAL.LocalConditions(jnp.asarray(1200.0), jnp.asarray(1.0), jnp.asarray(budgets))
    residual, _ = LOCAL.evaluate(problem, conditions, jnp.full((8,), -100.0))
    assert np.all(np.isfinite(residual))
    assert np.all(np.asarray(residual[:4]) < -90)


def test_full_component_gas_control_retains_trace_carbon_and_pressure_term():
    amounts = np.asarray([
        .3, .2, .5, .1, .1, .01, .01, .02, .01, 1e-12, 2e-12,
        .8, .1, .05, .05, .4, 3e-12, 2e-12, 1e-12, .1, .2, .01, .02, .01, .005,
    ])
    budgets = FORMULA @ amounts
    fractions = np.concatenate([values / values.sum() for values in (amounts[:11], amounts[11:15], amounts[15:])])
    standard = -np.log(fractions)
    standard[15:] -= np.log(10.0)
    problem = LOCAL.build_problem(RECORD, budgets, lambda t, p: jnp.asarray(standard), phases=("silicate", "metal", "gas"))
    result = LOCAL.solve(problem, 1200.0, 10.0, budgets)
    assert bool(result.root_solution.converged)
    assert problem.reaction_ids == tuple(f"R{i}" for i in range(18))
    np.testing.assert_allclose(result.component_amounts_mol, amounts, atol=0, rtol=1e-8)
    np.testing.assert_allclose(result.element_residual, 0, atol=1e-9)
    np.testing.assert_allclose(result.reaction_residual, 0, atol=1e-8)
    np.testing.assert_allclose(result.phase_element_amounts_mol[:, 6].sum(), budgets[6], rtol=1e-9, atol=0)


def test_removed_component_preserves_net_equilibrium_constraints():
    record = {
        "elements": ["A"], "phases": {"silicate": ["a", "b", "c"]},
        "component_formulas": {"a": {"A": 1}, "b": {"A": 1}, "c": {"A": 1}},
        "reactions": [
            {"id": "ab", "stoichiometry": {"a": -1, "b": 1}},
            {"id": "bc", "stoichiometry": {"b": -1, "c": 1}},
        ],
    }
    problem = LOCAL.build_problem(
        record, [1.0], lambda t, p: jnp.zeros(3), phases=("silicate",),
        phase_components={"silicate": ("a", "c")},
    )
    result = LOCAL.solve(problem, 1200.0, 1.0, [1.0])
    assert problem.reaction_matrix.shape == (1, 2)
    np.testing.assert_allclose(result.component_amounts_mol, [.5, 0, .5], atol=1e-10)


def test_explicit_empirical_reaction_offsets_have_the_declared_sign():
    problem, budgets, expected = _control()
    offset = jnp.asarray([.2, -.3, .1, .4])
    # A standard shift of opposite affinity gives the same known root.
    correction = -np.linalg.pinv(np.asarray(problem.reaction_matrix)) @ np.asarray(offset)
    standard = problem.standard_potentials_rt(1200.0, 1.0).at[problem.species_indices].add(correction)
    empirical = LOCAL.build_problem(
        RECORD, budgets, lambda t, p: standard,
        reaction_offset=lambda t, p: offset,
    )
    result = LOCAL.solve(empirical, 1200.0, 1.0, budgets, initial_component_amounts_mol=_initial(expected))
    assert bool(result.root_solution.converged)
    np.testing.assert_allclose(result.component_amounts_mol, expected, atol=1e-10)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1.0, 0.0])
def test_invalid_temperature_and_pressure_are_rejected(invalid):
    problem, budgets, _ = _control()
    with pytest.raises(ValueError, match="temperature_k"):
        LOCAL.solve(problem, invalid, 1.0, budgets)
    with pytest.raises(ValueError, match="pressure_bar"):
        LOCAL.solve(problem, 1200.0, invalid, budgets)


def test_support_changes_missing_elements_and_absent_requested_phase_are_rejected():
    problem, budgets, _ = _control()
    changed = budgets.copy()
    changed[4] = 1.0
    with pytest.raises(ValueError, match="support changed"):
        LOCAL.solve(problem, 1200.0, 1.0, changed)
    with pytest.raises(ValueError, match="no components"):
        LOCAL.build_problem(RECORD, [0, 1, 0, 0, 0, 0, 0], problem.standard_potentials_rt)
    with pytest.raises(ValueError, match="span all positive"):
        LOCAL.build_problem(RECORD, budgets, problem.standard_potentials_rt, phases=("metal",))
    with pytest.raises(ValueError, match="positive"):
        LOCAL.build_problem(RECORD, np.zeros(7), problem.standard_potentials_rt)
    with pytest.raises(ValueError, match="nonnegative"):
        LOCAL.build_problem(RECORD, -budgets, problem.standard_potentials_rt)


def test_initial_amount_and_callback_shapes_are_checked():
    problem, budgets, amounts = _control()
    amounts[7] = 1e-30
    with pytest.raises(ValueError, match="exactly zero"):
        LOCAL.solve(problem, 1200.0, 1.0, budgets, initial_component_amounts_mol=amounts)
    with pytest.raises(ValueError, match="original component shape"):
        LOCAL.solve(problem, 1200.0, 1.0, budgets, initial_component_amounts_mol=np.ones(8))
    invalid = LOCAL.build_problem(RECORD, budgets, lambda t, p: jnp.zeros(8))
    with pytest.raises(ValueError, match="mu0"):
        LOCAL.solve(invalid, 1200.0, 1.0, budgets)
    invalid = LOCAL.build_problem(
        RECORD, budgets, problem.standard_potentials_rt,
        activity_functions={"metal": lambda t, p, x: jnp.zeros(2)},
    )
    with pytest.raises(ValueError, match="ln_gamma"):
        LOCAL.solve(invalid, 1200.0, 1.0, budgets)
