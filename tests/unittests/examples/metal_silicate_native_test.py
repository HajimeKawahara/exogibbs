"""Independent full-source and optional native-metal equilibrium regressions."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import importlib
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest


EXAMPLE_DIRECTORY = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate"


def _load_examples():
    names = ("reference", "local", "source", "generate_equilibrium_reference", "native")
    previous = {name: sys.modules.get(name) for name in names}
    previous_path = sys.path[:]
    try:
        sys.path.insert(0, str(EXAMPLE_DIRECTORY))
        for name in names:
            sys.modules.pop(name, None)
        return tuple(importlib.import_module(name) for name in names)
    finally:
        sys.path[:] = previous_path
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


REFERENCE, LOCAL, SOURCE, GENERATOR, NATIVE = _load_examples()
RECORD = REFERENCE.load_reference()
FIXTURE = NATIVE.load_equilibrium_reference()
CASES = {case["id"]: case for case in FIXTURE["cases"]}
SOURCE_CASES = tuple(name for name in CASES if name.startswith("source_full_"))
NATIVE_CASES = tuple(name for name in CASES if name.startswith("completed_dry_"))


@pytest.fixture(scope="module")
def exoeos():
    provider = pytest.importorskip("exoeos")
    if not hasattr(provider, "MaFeSiOLiquid"):
        pytest.skip("The optional ExoEOS installation predates MaFeSiOLiquid.")
    return provider


@lru_cache(maxsize=None)
def _compiled_case(case_id):
    case = CASES[case_id]
    problem = NATIVE.make_reference_problem(RECORD, case)
    initial = jnp.asarray(NATIVE.reference_initial_amounts(RECORD, case))

    @jax.jit
    def calculate(temperature, budget, initial_scale):
        return LOCAL.solve(
            problem, temperature, case["P_bar"], budget,
            initial_component_amounts_mol=initial_scale * initial,
        )

    return problem, calculate


def _solve_case(case_id):
    case = CASES[case_id]
    problem, calculate = _compiled_case(case_id)
    result = calculate(case["T_K"], jnp.asarray(case["element_amounts_mol"]), 1.0)
    return problem, result


def _independent_reactions(case, amounts):
    species, _, reactions = REFERENCE.component_matrices(RECORD)
    thermo = next(item for item in RECORD["cases"] if item["T_K"] == case["T_K"])
    rt = RECORD["source"]["gas_constant_J_mol_K"] * case["T_K"]
    chemical = REFERENCE.source_standard_potentials(RECORD, thermo) / rt
    dry = case["id"].startswith("completed_dry_")
    for phase, names in RECORD["phases"].items():
        indices = [species.index(name) for name in names if name in case["active_components"]]
        if not indices:
            continue
        fractions = amounts[indices] / amounts[indices].sum()
        correction = np.log(fractions)
        if phase == "metal":
            if dry:
                correction += (
                    GENERATOR.completed_metal_ln_gamma(case["T_K"], fractions)
                    + REFERENCE.metal_standard_shift(case["T_K"])
                )
            else:
                correction += GENERATOR.source_metal_ln_gamma(case["T_K"], fractions)
        elif phase == "gas":
            correction += np.log(case["P_bar"])
        chemical[indices] += correction
    indices = [int(name[1:]) for name in case["reaction_ids"]]
    residual = reactions[indices] @ chemical
    if not dry:
        residual[14] += np.log(case["P_bar"] / 1e4)
    return residual


def _assert_reference_case(case_id, problem, result):
    case = CASES[case_id]
    amounts = np.asarray(result.component_amounts_mol)
    budget = np.asarray(case["element_amounts_mol"])
    species, formula, _ = REFERENCE.component_matrices(RECORD)
    assert bool(result.root_solution.converged)
    assert tuple(problem.reaction_ids) == tuple(case["reaction_ids"])
    assert tuple(problem.species) == tuple(case["active_components"])
    np.testing.assert_allclose(amounts, case["component_amounts_mol"], rtol=1e-8, atol=0)
    np.testing.assert_allclose(result.phase_amounts_mol, case["phase_amounts_mol"], rtol=1e-8, atol=0)
    assert np.all(np.asarray(result.phase_amounts_mol) > 0)
    total = formula @ amounts
    positive = budget > 0
    np.testing.assert_allclose(total[positive] / budget[positive], 1.0, atol=1e-9, rtol=0)
    np.testing.assert_array_equal(total[~positive], np.zeros(np.count_nonzero(~positive)))

    for index, phase in enumerate(problem.phases):
        expected = np.zeros(len(RECORD["elements"]))
        for name in RECORD["phases"][phase]:
            for element, count in RECORD["component_formulas"][name].items():
                expected[RECORD["elements"].index(element)] += count * amounts[species.index(name)]
        np.testing.assert_allclose(result.phase_element_amounts_mol[index], expected, rtol=1e-13, atol=0)
        indices = [species.index(name) for name in RECORD["phases"][phase]]
        np.testing.assert_allclose(np.asarray(result.mole_fractions)[indices].sum(), 1.0, atol=1e-14)
    independent = _independent_reactions(case, amounts)
    np.testing.assert_allclose(independent, 0.0, atol=1e-8, rtol=0)
    np.testing.assert_allclose(result.reaction_residual, independent, atol=1e-12, rtol=0)


@pytest.mark.parametrize("case_id", SOURCE_CASES)
def test_full_source_network_matches_independent_frozen_equilibrium(case_id) -> None:
    problem, result = _solve_case(case_id)
    assert len(problem.species) == 25
    assert problem.phases == ("silicate", "metal", "gas")
    _assert_reference_case(case_id, problem, result)


@pytest.mark.parametrize("case_id", NATIVE_CASES)
def test_native_metal_and_ideal_silicate_match_independent_dry_equilibrium(case_id, exoeos) -> None:
    problem, result = _solve_case(case_id)
    assert len(problem.species) == 8
    assert problem.phases == ("silicate", "metal")
    _assert_reference_case(case_id, problem, result)


def test_native_activity_and_standard_conversion_match_independent_equations(exoeos) -> None:
    case = CASES[NATIVE_CASES[0]]
    problem, result = _solve_case(case["id"])
    model = exoeos.MaFeSiOLiquid()
    metal_indices = [problem.full_species.index(name + "_metal") for name in ("Fe", "Si", "O")]
    fractions = np.asarray(result.mole_fractions)[metal_indices]
    temperature = case["T_K"]
    native = np.asarray(problem.activity_functions[1](temperature, case["P_bar"], fractions))
    expected = GENERATOR.completed_metal_ln_gamma(temperature, fractions)
    np.testing.assert_allclose(native, expected, atol=5e-12, rtol=5e-12)

    thermo = next(item for item in RECORD["cases"] if item["T_K"] == temperature)
    rt = RECORD["source"]["gas_constant_J_mol_K"] * temperature
    source_standards = REFERENCE.source_standard_potentials(RECORD, thermo)[metal_indices] / rt
    shift = np.asarray(model.standard_state_shift_RT(temperature))
    np.testing.assert_allclose(shift, REFERENCE.metal_standard_shift(temperature), atol=5e-12)
    formal_standards = np.asarray(problem.standard_potentials_rt(temperature, case["P_bar"]))[metal_indices]
    formal_mu = formal_standards + np.log(fractions) + native
    completed_source_mu = source_standards + np.log(fractions) + expected + shift
    np.testing.assert_allclose(formal_mu, completed_source_mu, atol=5e-12, rtol=5e-12)


def test_native_jit_reverse_derivatives_and_amount_scaling(exoeos) -> None:
    case_id = NATIVE_CASES[0]
    case = CASES[case_id]
    problem, calculate = _compiled_case(case_id)
    temperature = case["T_K"]
    budget = jnp.asarray(case["element_amounts_mol"])
    result = calculate(temperature, budget, 1.0)
    silicon = problem.full_species.index("Si_metal")

    def output(t, b):
        return calculate(t, b, 1.0).mole_fractions[silicon]

    derivative_t, derivative_b = jax.jit(jax.grad(output, argnums=(0, 1)))(temperature, budget)
    dt, db = 0.02, 0.002
    expected_t = (output(temperature + dt, budget) - output(temperature - dt, budget)) / (2 * dt)
    silicon_element = RECORD["elements"].index("Si")
    plus = budget.at[silicon_element].add(db)
    minus = budget.at[silicon_element].add(-db)
    expected_b = (output(temperature, plus) - output(temperature, minus)) / (2 * db)
    np.testing.assert_allclose(derivative_t, expected_t, rtol=2e-5, atol=1e-10)
    np.testing.assert_allclose(derivative_b[silicon_element], expected_b, rtol=2e-5, atol=1e-10)
    assert abs(float(derivative_t)) > 1e-7
    assert abs(float(derivative_b[silicon_element])) > 1e-7

    for scale in (0.01, 1e4):
        scaled = calculate(temperature, scale * budget, scale)
        assert bool(scaled.root_solution.converged)
        np.testing.assert_allclose(scaled.mole_fractions, result.mole_fractions, rtol=1e-9, atol=0)
        np.testing.assert_allclose(scaled.component_amounts_mol, scale * result.component_amounts_mol, rtol=1e-9, atol=0)
        np.testing.assert_allclose(scaled.phase_element_amounts_mol, scale * result.phase_element_amounts_mol, rtol=1e-9, atol=0)


def test_common_elemental_standard_shift_preserves_native_equilibrium(exoeos) -> None:
    case = CASES[NATIVE_CASES[0]]
    problem, expected = _solve_case(case["id"])
    _, formula, _ = REFERENCE.component_matrices(RECORD)
    elemental_offset = jnp.asarray([2.5, -4., 7., -3., 1.5, 9., -8.])
    shift = jnp.asarray(formula.T) @ elemental_offset

    def shifted_standards(t, p):
        return problem.standard_potentials_rt(t, p) + shift

    shifted = replace(problem, standard_potentials_rt=shifted_standards)
    actual = LOCAL.solve(
        shifted, case["T_K"], case["P_bar"], case["element_amounts_mol"],
        initial_component_amounts_mol=NATIVE.reference_initial_amounts(RECORD, case),
    )
    assert bool(actual.root_solution.converged)
    np.testing.assert_allclose(actual.component_amounts_mol, expected.component_amounts_mol, atol=0, rtol=1e-9)


def test_zero_native_excess_recovers_ideal_mixing_with_the_same_standards(exoeos) -> None:
    case = CASES[NATIVE_CASES[0]]
    model = exoeos.MaFeSiOLiquid(interaction_K=jnp.zeros(3))
    problem = NATIVE.make_reference_problem(RECORD, case, model=model)
    ideal = replace(problem, activity_functions=(None, None))
    arguments = (case["T_K"], case["P_bar"], case["element_amounts_mol"])
    initial = NATIVE.reference_initial_amounts(RECORD, case)
    actual = LOCAL.solve(problem, *arguments, initial_component_amounts_mol=initial)
    expected = LOCAL.solve(ideal, *arguments, initial_component_amounts_mol=initial)
    assert bool(actual.root_solution.converged)
    assert bool(expected.root_solution.converged)
    np.testing.assert_allclose(actual.component_amounts_mol, expected.component_amounts_mol, atol=0, rtol=1e-9)
