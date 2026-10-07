"""Fixed-phase local mass-action calculation
=========================================

This example solves only the declared, positive-amount phase assemblage.
It does not select phases or establish liquid stability. Thermochemistry
and phase activities are supplied explicitly by the calling example.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from exogibbs.applications.magma_gas._root import (
    InnerRootDiagnostics,
    RootSolution,
    make_implicit_root_solver,
)
from exogibbs.applications.magma_gas.solve import _validate_options
from exogibbs.applications.magma_gas.types import MagmaGasOptions


StandardPotentialFunction = Callable[[jax.Array, jax.Array], jax.Array]
ActivityFunction = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


@dataclass(frozen=True)
class LocalProblem:
    """Static component support and callbacks for one local phase branch.

    Standards return ``mu0/(RT)`` in ``full_species`` order. Activities
    return ``ln_gamma`` in each active phase's component order; omitted
    callbacks explicitly select ideal mixing. Both callbacks use K/bar.
    """

    full_species: tuple[str, ...]
    elements: tuple[str, ...]
    phases: tuple[str, ...]
    species: tuple[str, ...]
    species_indices: np.ndarray
    element_indices: np.ndarray
    phase_slices: tuple[slice, ...]
    formula_matrix: jax.Array
    full_formula_matrix: jax.Array
    reaction_matrix: jax.Array
    reaction_ids: tuple[str, ...]
    standard_potentials_rt: StandardPotentialFunction
    activity_functions: tuple[Optional[ActivityFunction], ...]
    reaction_offset: Optional[StandardPotentialFunction]
    standard_pressure_bar: float


class LocalConditions(NamedTuple):
    temperature_k: jax.Array
    pressure_bar: jax.Array
    element_amounts_mol: jax.Array


class LocalResult(NamedTuple):
    """Amounts and independently evaluated residuals in the original basis.

    Component amounts and mole fractions use ``full_species`` order;
    excluded components are exactly zero. Phase arrays use ``phases``
    order, and element arrays retain the original ``elements`` order.
    """

    component_amounts_mol: jax.Array
    mole_fractions: jax.Array
    phase_amounts_mol: jax.Array
    phase_element_amounts_mol: jax.Array
    element_residual: jax.Array
    reaction_residual: jax.Array
    root_solution: RootSolution


def _budgets(value: Any, size: int) -> np.ndarray:
    amounts = np.asarray(value, dtype=np.float64)
    if amounts.shape != (size,):
        raise ValueError(f"element_amounts_mol must have shape {(size,)}.")
    if not np.all(np.isfinite(amounts)) or np.any(amounts < 0):
        raise ValueError("element_amounts_mol must be finite and nonnegative.")
    if not np.any(amounts > 0):
        raise ValueError("At least one element amount must be positive.")
    return amounts


def _reaction_basis(
    formula: np.ndarray, candidates: np.ndarray, candidate_ids: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Retain source reactions, supplementing the reduced null space if needed."""
    size = formula.shape[1] - formula.shape[0]
    rows = []
    ids = []
    for row, name in zip(candidates, candidate_ids):
        if np.any(row) and np.linalg.matrix_rank(np.asarray(rows + [row])) > len(rows):
            rows.append(row)
            ids.append(name)
    # Removing a component can leave a valid net reaction that was expressed
    # by several source reactions. Do not silently discard that equilibrium.
    _, _, vh = np.linalg.svd(formula, full_matrices=True)
    for row in vh[formula.shape[0]:]:
        if len(rows) == size:
            break
        if np.linalg.matrix_rank(np.asarray(rows + [row])) > len(rows):
            rows.append(row)
            ids.append(f"nullspace_{len(ids)}")
    return np.asarray(rows, dtype=np.float64).reshape(size, formula.shape[1]), tuple(ids)


def build_problem(
    record: Mapping[str, Any],
    element_amounts_mol: Any,
    standard_potentials_rt: StandardPotentialFunction,
    *,
    phases: Sequence[str] = ("silicate", "metal"),
    phase_components: Optional[Mapping[str, Sequence[str]]] = None,
    activity_functions: Optional[Mapping[str, ActivityFunction]] = None,
    reaction_offset: Optional[StandardPotentialFunction] = None,
) -> LocalProblem:
    """Freeze support outside JIT, removing components containing zero elements.

    ``phase_components`` explicitly selects a smaller model, when required.
    No trace material is inserted. Every requested phase must retain at
    least one component, and active element rows must be independent.
    ``reaction_offset(T, P)`` supplies explicit empirical corrections in
    ``reaction_ids`` order; such a model need not have a common Gibbs energy.
    """
    full_species = tuple(name for names in record["phases"].values() for name in names)
    elements = tuple(record["elements"])
    amounts = _budgets(element_amounts_mol, len(elements))
    phase_names = tuple(phases)
    if not phase_names or len(set(phase_names)) != len(phase_names):
        raise ValueError("phases must be nonempty and unique.")
    if any(phase not in record["phases"] for phase in phase_names):
        raise ValueError("Every phase must be declared in the reference record.")
    selected = dict(phase_components or {})
    activities = dict(activity_functions or {})
    if (set(selected) | set(activities)) - set(phase_names):
        raise ValueError("Component and activity mappings must name selected phases.")
    formula = np.asarray([
        [record["component_formulas"][name].get(element, 0) for name in full_species]
        for element in elements
    ], dtype=np.float64)
    active_elements = np.flatnonzero(amounts > 0)
    supported = np.all(formula[amounts == 0] == 0, axis=0)
    indices = []
    phase_slices = []
    for phase in phase_names:
        names = tuple(selected.get(phase, record["phases"][phase]))
        if len(set(names)) != len(names) or not set(names) <= set(record["phases"][phase]):
            raise ValueError("Phase components must be unique members of their declared phase.")
        active = [full_species.index(name) for name in names if supported[full_species.index(name)]]
        if not active:
            raise ValueError(f"Phase {phase!r} has no components supported by the supplied budget.")
        phase_slices.append(slice(len(indices), len(indices) + len(active)))
        indices.extend(active)
    indices = np.asarray(indices, dtype=np.int32)
    reduced_formula = formula[np.ix_(active_elements, indices)]
    if np.linalg.matrix_rank(reduced_formula) != len(active_elements):
        raise ValueError("Selected components must span all positive element budgets independently.")
    source_reactions = np.asarray([
        [reaction["stoichiometry"].get(name, 0.0) for name in full_species]
        for reaction in record["reactions"]
    ], dtype=np.float64).reshape(-1, len(full_species))
    if not np.allclose(formula @ source_reactions.T, 0.0, atol=1e-12, rtol=0):
        raise ValueError("Reference reactions must conserve every element.")
    excluded = np.ones(len(full_species), dtype=bool)
    excluded[indices] = False
    survives = np.all(source_reactions[:, excluded] == 0, axis=1)
    surviving = source_reactions[survives][:, indices]
    source_ids = [reaction.get("id", f"R{i}") for i, reaction in enumerate(record["reactions"])]
    reactions, reaction_ids = _reaction_basis(
        reduced_formula, surviving, [name for name, keep in zip(source_ids, survives) if keep],
    )
    standard_pressure = float(record.get("local_contract", {}).get("standard_pressure_bar", 1.0))
    if not np.isfinite(standard_pressure) or standard_pressure <= 0:
        raise ValueError("standard_pressure_bar must be finite and positive.")
    return LocalProblem(
        full_species, elements, phase_names, tuple(full_species[i] for i in indices),
        indices, active_elements, tuple(phase_slices), jnp.asarray(reduced_formula),
        jnp.asarray(formula), jnp.asarray(reactions), reaction_ids, standard_potentials_rt,
        tuple(activities.get(phase) for phase in phase_names), reaction_offset, standard_pressure,
    )


def _validate_conditions(problem: LocalProblem, conditions: LocalConditions) -> None:
    for name, value in (
        ("temperature_k", conditions.temperature_k),
        ("pressure_bar", conditions.pressure_bar),
    ):
        if value.ndim != 0:
            raise ValueError(f"{name} must be scalar.")
        if not isinstance(value, jax.core.Tracer) and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive.")
    budgets = conditions.element_amounts_mol
    if budgets.shape != (len(problem.elements),):
        raise ValueError("element_amounts_mol has the wrong shape.")
    if not isinstance(budgets, jax.core.Tracer):
        values = _budgets(budgets, len(problem.elements))
        if not np.array_equal(np.flatnonzero(values > 0), problem.element_indices):
            raise ValueError("Element support changed; build a new problem outside JIT.")


def evaluate(
    problem: LocalProblem, conditions: LocalConditions, root_variables: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    """Evaluate log balance and reaction residuals for log component amounts."""
    scale = jnp.sum(conditions.element_amounts_mol)
    amounts = scale * jnp.exp(root_variables)
    standard = jnp.asarray(problem.standard_potentials_rt(
        conditions.temperature_k, conditions.pressure_bar,
    ))
    if standard.shape != (len(problem.full_species),):
        raise ValueError("Standards must return one mu0/(RT) per original component.")
    potentials = standard[problem.species_indices]
    fractions = []
    phase_amounts = []
    contributions = []
    for phase, section, activity in zip(problem.phases, problem.phase_slices, problem.activity_functions):
        logx = root_variables[section] - logsumexp(root_variables[section])
        x = jnp.exp(logx)
        lngamma = jnp.zeros_like(x) if activity is None else jnp.asarray(activity(
            conditions.temperature_k, conditions.pressure_bar, x,
        ))
        if lngamma.shape != x.shape:
            raise ValueError("Activity callback must return one ln_gamma per active phase component.")
        correction = logx + lngamma
        if phase == "gas":
            correction = correction + jnp.log(conditions.pressure_bar / problem.standard_pressure_bar)
        potentials = potentials.at[section].add(correction)
        fractions.append(x)
        phase_amounts.append(jnp.sum(amounts[section]))
        contributions.append(problem.full_formula_matrix[:, problem.species_indices[section]] @ amounts[section])
    total_elements = jnp.sum(jnp.stack(contributions), axis=0)
    positive = conditions.element_amounts_mol[problem.element_indices]
    relative_balance = total_elements[problem.element_indices] / positive - 1
    reaction_residual = problem.reaction_matrix @ potentials
    if problem.reaction_offset is not None:
        offset = jnp.asarray(problem.reaction_offset(conditions.temperature_k, conditions.pressure_bar))
        if offset.shape != reaction_residual.shape:
            raise ValueError("Reaction offsets must follow the independent reaction_ids shape.")
        reaction_residual = reaction_residual + offset
    residual = jnp.concatenate((
        jnp.log(total_elements[problem.element_indices] / positive), reaction_residual,
    ))
    return residual, (
        amounts, jnp.concatenate(fractions), jnp.stack(phase_amounts),
        jnp.stack(contributions), relative_balance, reaction_residual,
    )


def solve(
    problem: LocalProblem,
    temperature_k: Any,
    pressure_bar: Any,
    element_amounts_mol: Any,
    *,
    initial_component_amounts_mol: Optional[Any] = None,
    options: Optional[MagmaGasOptions] = None,
) -> LocalResult:
    """Solve a fixed support; validate finite positive inputs before tracing.

    JIT evaluation requires the same positive element support used at build
    time. Reverse derivatives describe a smooth converged branch; failed
    solves retain raw diagnostics and give NaN implicit derivatives.
    The default budget-based initial guess does not identify a phase branch.
    Supply positive initial component amounts near the intended branch when
    that guess fails; no global search or phase removal is performed.
    """
    active_options = options or MagmaGasOptions(root_tolerance=1e-10, max_iter=80)
    _validate_options(active_options)
    dtype = jnp.result_type(
        jnp.asarray(temperature_k), jnp.asarray(pressure_bar),
        jnp.asarray(element_amounts_mol), problem.formula_matrix,
    )
    conditions = LocalConditions(*(
        jnp.asarray(value, dtype=dtype)
        for value in (temperature_k, pressure_bar, element_amounts_mol)
    ))
    _validate_conditions(problem, conditions)
    scale = jnp.sum(conditions.element_amounts_mol)
    if initial_component_amounts_mol is None:
        formula = problem.formula_matrix
        limits = jnp.where(
            formula > 0, conditions.element_amounts_mol[problem.element_indices, None]
            / jnp.where(formula > 0, formula, 1), jnp.inf,
        )
        initial_amounts = jnp.min(limits, axis=0) / len(problem.species)
    else:
        initial = jnp.asarray(initial_component_amounts_mol, dtype=dtype)
        if initial.shape != (len(problem.full_species),):
            raise ValueError("Initial component amounts must use the original component shape.")
        if not isinstance(initial, jax.core.Tracer):
            active = np.zeros(len(problem.full_species), dtype=bool)
            active[problem.species_indices] = True
            values = np.asarray(initial)
            if (not np.all(np.isfinite(values)) or np.any(values[active] <= 0)
                    or np.any(values[~active] != 0)):
                raise ValueError("Initial amounts must be positive on active components and exactly zero elsewhere.")
        initial_amounts = initial[problem.species_indices]
    initial_root = jax.lax.stop_gradient(jnp.log(initial_amounts / scale))

    def residual(dynamic_conditions, root):
        return evaluate(problem, dynamic_conditions, root)[0]

    def inner_diagnostics(dynamic_conditions, root):
        del dynamic_conditions
        # There is no nested gas solve in this direct mass-action example.
        return InnerRootDiagnostics(jnp.asarray(True), jnp.asarray(0), jnp.zeros((), root.dtype), jnp.zeros((), root.dtype))

    implicit_root = make_implicit_root_solver(residual, inner_diagnostics, active_options)
    root_solution = implicit_root(conditions, initial_root)
    _, values = evaluate(problem, conditions, root_solution.root_variables)
    amounts, fractions, phase_amounts, contributions, balance, reactions = values
    full_amounts = jnp.zeros(len(problem.full_species), dtype=dtype).at[problem.species_indices].set(amounts)
    full_fractions = jnp.zeros_like(full_amounts).at[problem.species_indices].set(fractions)
    full_balance = jnp.zeros(len(problem.elements), dtype=dtype).at[problem.element_indices].set(balance)
    return LocalResult(full_amounts, full_fractions, phase_amounts, contributions, full_balance, reactions, root_solution)
