"""Bounded host support search; all Gibbs minimizations use the JAX kernel."""

from collections import deque
from dataclasses import dataclass, replace
from functools import lru_cache, partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linprog

from .solve import _geometry, _primal
from .types import Options, Phase, Problem, R_SCALE, Result, Support


@lru_cache(maxsize=64)
def _solver(problem, support, options):
    return jax.jit(partial(_primal, problem, support=support, options=options))


def _seed(problem, support, budget, initial):
    """Find a face interior by linear feasibility, never by minimizing G."""
    indices, active, scale, matrix, inverse, _, _ = _geometry(problem, support)
    target = np.r_[budget / budget.sum(), np.zeros(len(active))] / scale
    domain = problem.domain_matrix
    inactive = [i for i in range(len(domain)) if i not in active]
    rows = np.concatenate((np.eye(len(indices)), domain[inactive][:, indices]))
    rows = rows[np.any(rows != 0, axis=1)]
    rows /= np.linalg.norm(rows, axis=1)[:, None]
    q = initial[indices] / budget.sum()
    if np.all(q > 0):
        row = np.linalg.svd(matrix, full_matrices=False)[2][:np.linalg.matrix_rank(matrix)]
        weighted = row*q
        q += weighted.T @ np.linalg.solve(weighted @ row.T, row @ inverse @ (target-matrix @ q))
    else:
        q += inverse @ (target - matrix @ q)
    if np.max(np.abs(matrix @ q - target)) < 1e-12 and np.all(rows @ q > 0):
        full = np.zeros_like(initial, dtype=float)
        full[indices] = q * budget.sum()
        return full
    # Maximize a common interior margin. A zero optimum is a boundary
    # candidate, not evidence that a physically small amount is absent.
    lp = linprog(np.r_[np.zeros(len(indices)), -1.],
                 A_ub=np.column_stack((-rows, np.ones(len(rows)))), b_ub=np.zeros(len(rows)),
                 A_eq=np.column_stack((matrix, np.zeros(len(matrix)))), b_eq=target,
                 bounds=[(None, None)] * len(indices) + [(0., 1.)], method="highs")
    if not lp.success:
        return None
    interior = lp.x[:-1]
    # Keep a boundary warm start close to its original composition. A small
    # feasible convex blend initializes a new face; it is not an amount floor.
    slack, margin = rows @ q, rows @ interior
    if (np.all(np.isfinite(q)) and np.max(np.abs(matrix @ q-target)) < 1e-12
            and np.all(slack >= -1e-12) and np.all(margin > 0)):
        required = np.max(np.where(slack < 0, -slack/np.maximum(margin-slack, 1e-300), 0.))
        weight = min(1., max(1e-4, 1.01*required))
        interior = (1-weight)*q+weight*interior
    full = np.zeros_like(initial, dtype=float)
    full[indices] = interior * budget.sum()
    return full


def _neighbors(problem, support, amounts, slack, multipliers, options):
    """Suggest faces; only a fresh constrained solve can accept a deletion."""
    indices = np.asarray(support.components)
    # Prefer whole vanishing phases before component faces. Dilute species
    # inside a finite phase can have arbitrarily small equilibrium amounts.
    vanished = {i for section in problem.component_slices
                if np.sum(amounts[section]) <= options.support_tolerance
                for i in indices if section.start <= i < section.stop}
    tiny = (np.asarray([i in vanished for i in indices]) if vanished
            else amounts[indices] <= options.support_tolerance)
    for i, row in enumerate(problem.domain_matrix):
        if i in support.active_constraints:
            if multipliers[i] < -options.kkt_tolerance:
                yield Support(support.components, tuple(j for j in support.active_constraints if j != i))
        elif np.any(row[indices]) and slack[i] <= options.support_tolerance:
            candidate = Support(support.components, tuple(sorted((*support.active_constraints, i))))
            try:
                _geometry(problem, candidate)
            except ValueError:
                continue
            yield candidate

    if np.any(tiny) and not np.all(tiny):
        kept = tuple(indices[~tiny])
        active = tuple(i for i in support.active_constraints
                       if np.any(problem.domain_matrix[i, list(kept)]))
        yield Support(kept, active)


def _search(problem, budget, initial, reachable, temperature, pressure, parameters, options):
    full_support = Support(tuple(np.flatnonzero(reachable)))
    positive = tuple(np.flatnonzero(reachable & (initial > 0)))
    queue = deque([Support(positive)] if positive else [])
    queue.append(full_support)
    seen, results = set(), []
    while queue and len(seen) < options.max_supports:
        support = queue.popleft()
        if support in seen:
            continue
        seen.add(support)
        try:
            seed = _seed(problem, support, budget, initial)
        except ValueError:
            continue
        if seed is None:
            continue
        result = _solver(problem, support, options)(
            temperature_k=temperature, pressure_pa=pressure,
            element_amounts_mol=budget, init=seed, parameters=parameters,
        )
        results.append(result)
        # A positive, converged phase is never deleted by an amount cutoff.
        if not (result.converged and result.local_kkt_valid and result.minimum_valid):
            queue.extend(_neighbors(problem, support, np.asarray(result.amounts_mol) / budget.sum(),
                                    np.asarray(result.domain_slack),
                                    np.asarray(result.constraint_multipliers_j_mol) / (R_SCALE * temperature),
                                    options))
    return results


@lru_cache(maxsize=64)
def _insertion_problem(phase):
    def energy(t, p, x, data):
        parameters, potentials = data
        return phase.gibbs_j(t, p, x, parameters) - potentials @ (jnp.asarray(phase.atom_matrix) @ x)

    return Problem(("components",), (Phase(phase.name, phase.components,
                   [np.ones(len(phase.components))], energy, phase.domain_matrix),))


def _insertion(phase, temperature, pressure, potentials, parameters, allowed, current, options):
    count = len(phase.components)
    if not np.any(allowed):
        return np.inf, np.inf, np.full(count, np.nan)
    problem = _insertion_problem(phase)
    support = Support(tuple(np.flatnonzero(allowed)))
    center = _seed(problem, support, np.ones(1), allowed.astype(float))
    if center is None:
        return np.nan, np.nan, np.full(count, np.nan)
    candidates = [center]
    if current.sum() > 0:
        candidates.append(current / current.sum())
    atoms = np.asarray(phase.atom_matrix)
    domain = np.asarray(phase.domain_matrix).reshape((-1, count))
    if phase.insertion_lower_bound_j_mol is not None:
        lower = float(phase.insertion_lower_bound_j_mol(
            temperature, pressure, jnp.asarray(potentials), parameters, tuple(np.flatnonzero(allowed)),
        )) / (R_SCALE * temperature)
        if np.isfinite(lower) and lower >= -options.kkt_tolerance:
            values = [(float((phase.gibbs_j(temperature, pressure, jnp.asarray(x), parameters)
                              - potentials @ (atoms @ x)) / (R_SCALE * temperature)), x)
                      for x in candidates if np.all(domain @ x >= -1e-12)]
            values = [(v, x) for v, x in values if np.isfinite(v)]
            if values:
                upper, composition = min(values, key=lambda item: item[0])
                if lower > upper + options.kkt_tolerance:
                    raise ValueError(f"{phase.name}: insertion lower bound contradicts an evaluated trial.")
                return lower, upper, composition
    # Feasible vertices and interior starts expose negative trials without
    # interpreting positive local minima as global lower bounds.
    for index in np.flatnonzero(allowed) if count > 1 else ():
        vertex = linprog(-np.eye(count)[index], A_ub=-domain if len(domain) else None,
                         b_ub=np.zeros(len(domain)) if len(domain) else None,
                         A_eq=np.ones((1, count)), b_eq=[1.],
                         bounds=[(0., None) if ok else (0., 0.) for ok in allowed], method="highs")
        if vertex.success:
            candidates.extend([vertex.x, .5 * center + .5 * vertex.x])
    data = (parameters, jnp.asarray(potentials))
    trials = list(candidates)
    if count > 1:
        for initial in candidates:
            if np.any(initial[allowed] <= 0):
                continue
            trials.extend(np.asarray(r.amounts_mol) for r in _search(
                problem, np.ones(1), initial, allowed, temperature, pressure, data, options,
            ))
    values = []
    for x in trials:
        if (np.all(np.isfinite(x)) and np.all(x >= 0) and not np.any(x[~allowed])
                and abs(x.sum() - 1.) <= 1e-10 and np.all(domain @ x >= -1e-12)):
            value = float((phase.gibbs_j(temperature, pressure, jnp.asarray(x), parameters)
                           - potentials @ (atoms @ x)) / (R_SCALE * temperature))
            if np.isfinite(value):
                values.append((value, x))
    if not values:
        return np.nan, np.nan, np.full(count, np.nan)
    upper, composition = min(values, key=lambda item: item[0])
    lower = upper if count == 1 else np.nan
    if phase.insertion_lower_bound_j_mol is not None:
        lower = float(phase.insertion_lower_bound_j_mol(
            temperature, pressure, jnp.asarray(potentials), parameters, tuple(np.flatnonzero(allowed)),
        )) / (R_SCALE * temperature)
        if np.isinf(lower) or lower > upper + options.kkt_tolerance:
            raise ValueError(f"{phase.name}: insertion lower bound contradicts an evaluated trial.")
    return lower, upper, composition


def audit_stability(problem, result, *, temperature_k, pressure_pa, element_amounts_mol,
                    parameters=(), options=Options()):
    """Repolish a supplied face and reevaluate its insertion trials and bounds.

    Bounds use the current T, P, elemental dual and parameters. A callback
    supplies J/mol components over the declared simplex/domain and allowed
    component indices. NaN means unavailable. The caller owns its proof;
    no cached certificate or positive local trial is promoted to a bound.
    """
    budget = np.asarray(element_amounts_mol, dtype=float)
    result = _solver(problem, result.support, options)(
        temperature_k=temperature_k, pressure_pa=pressure_pa,
        element_amounts_mol=budget, init=result.amounts_mol, parameters=parameters,
    )
    if not (result.converged and result.local_kkt_valid and result.minimum_valid):
        return replace(result, stability_status="unresolved")
    reachable = ~np.any(problem.atom_matrix[budget == 0] > 0, axis=0)
    potentials = np.nan_to_num(np.asarray(result.element_potentials_j_mol), nan=0.)
    bounds = [_insertion(p, float(temperature_k), float(pressure_pa), potentials, parameters,
                         reachable[s], np.asarray(result.amounts_mol)[s], options)
              for p, s in zip(problem.phases, problem.component_slices)]
    lower, upper, compositions = zip(*bounds)
    lower, upper = np.asarray(lower), np.asarray(upper)
    if np.any(upper < -options.kkt_tolerance):
        status = "unstable"
    elif np.all(lower >= -options.kkt_tolerance):
        status = "certified"
    else:
        status = "unresolved"
    return replace(result, insertion_lower=jnp.asarray(lower), insertion_upper=jnp.asarray(upper),
                   incipient_compositions=tuple(jnp.asarray(x) for x in compositions),
                   stability_status=status)


def select(problem, *, temperature_k, pressure_pa, element_amounts_mol, parameters,
           init, reachable, options):
    temperature, pressure = float(temperature_k), float(pressure_pa)
    initial = np.asarray(init, dtype=float)
    if (not np.isfinite(temperature) or temperature <= 0 or not np.isfinite(pressure) or pressure <= 0
            or initial.shape != (problem.atom_matrix.shape[1],)
            or not np.all(np.isfinite(initial)) or np.any(initial < 0)):
        raise ValueError("Supply finite positive T/P and a finite nonnegative full init vector.")
    if not np.any(reachable):
        raise ValueError("No components are reachable from the element budget.")
    results = _search(problem, element_amounts_mol, initial, reachable,
                      temperature, pressure, parameters, options)
    if not results:
        raise ValueError("No feasible component/domain support was found.")
    accepted = [r for r in results if r.converged and r.local_kkt_valid and r.minimum_valid]
    if not accepted:
        return replace(results[-1], stability_status="unresolved")
    best = min(accepted, key=lambda r: float(r.gibbs_j))
    return audit_stability(problem, best, temperature_k=temperature, pressure_pa=pressure,
                           element_amounts_mol=element_amounts_mol, parameters=parameters,
                           options=options)


@dataclass(frozen=True)
class PurePhaseUpdateResult:
    """Host search trace; ``result`` is the lowest accepted state visited.

    Attempts retain failed/non-descending proposals. ``stop_reason`` describes
    only the bounded pure-phase update, not global phase stability. Inspect
    ``result.stability_status`` and all original result diagnostics separately.
    """

    result: Result
    attempts: Tuple[Result, ...]
    inserted_phases: Tuple[str, ...]
    stop_reason: str


def solve_with_pure_phase_updates(
    problem: Problem, *, temperature_k: Any, pressure_pa: Any,
    element_amounts_mol: Any, init: Any, parameters: Any = (),
    options: Options = Options(), max_updates: int = 4,
) -> PurePhaseUpdateResult:
    """Solve, then reinsert negative omitted one-component phases, boundedly.

    Ordinary ``solve`` remains unchanged. This opt-in host workflow uses the
    exact live insertion cost of an absent pure phase, obtains a conserving
    interior seed through the existing feasibility routine, and repeats the
    existing support search. Only a fresh valid lower-G result is accepted.
    No component is removed by a small-amount cutoff, no amount floor is
    imposed, and positive local solution-phase minima never become bounds.

    A negative insertion in a present phase may require phase splitting;
    that remains unresolved. The returned trace also distinguishes exhausted
    update budgets, infeasible seeds and failed proposals from stability.
    This Python search entry point is not a transformed derivative API.
    """
    from .solve import solve

    if not isinstance(max_updates, int) or isinstance(max_updates, bool) or max_updates < 0:
        raise ValueError("max_updates must be a nonnegative integer.")
    arguments = dict(temperature_k=temperature_k, pressure_pa=pressure_pa,
                     element_amounts_mol=element_amounts_mol, parameters=parameters,
                     options=options)
    best = solve(problem, init=init, **arguments)
    attempts, inserted = [best], []
    budget = np.asarray(element_amounts_mol, dtype=float)
    valid = lambda r: bool(r.converged and r.local_kkt_valid and r.minimum_valid)
    reason = "invalid_initial" if not valid(best) else "no_negative_absent_pure_phase"
    if valid(best):
        for update in range(max_updates+1):
            candidates = []
            for i, (phase, section) in enumerate(zip(problem.phases, problem.component_slices)):
                if len(phase.components) != 1 or section.start in best.support.components:
                    continue
                atoms = np.asarray(phase.atom_matrix)[:, 0]
                if np.any(atoms[budget == 0] > 0):
                    continue
                # Recompute from the live standards and dual, not saved bounds.
                cost = float((phase.gibbs_j(temperature_k, pressure_pa, jnp.ones(1), parameters)
                              - best.element_potentials_j_mol @ atoms)/(R_SCALE*temperature_k))
                if np.isfinite(cost) and cost < -options.kkt_tolerance:
                    candidates.append((cost, i, section.start))
            if not candidates:
                reason = "no_negative_absent_pure_phase"
                break
            if update == max_updates:
                reason = "update_limit"
                break
            _, phase_index, component = min(candidates)
            support = Support(tuple(sorted((*best.support.components, component))),
                              best.support.active_constraints)
            seed = _seed(problem, support, budget, np.asarray(best.amounts_mol))
            if seed is None:
                reason = "infeasible_insertion"
                break
            trial = solve(problem, init=seed, **arguments)
            attempts.append(trial)
            inserted.append(problem.phases[phase_index].name)
            if not valid(trial):
                reason = "invalid_trial"
                break
            if float(trial.gibbs_j) >= float(best.gibbs_j):
                reason = "no_descent"
                break
            best = trial
    return PurePhaseUpdateResult(best, tuple(attempts), tuple(inserted), reason)
