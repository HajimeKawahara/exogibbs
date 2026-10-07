"""Constrained scalar Gibbs minimization on an explicitly declared phase branch.
=============================================================================

This example consumes the same absolute full-potential callbacks as the legacy
local root. Numerical acceptance establishes local first-order conditions,
not global phase stability or material calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares, linprog, lsq_linear, minimize, minimize_scalar

from full_potential import PhaseCallback
from local import LocalProblem, _budgets


class PhaseEvaluationError(ValueError):
    """A provider cannot evaluate a trial; this is not a thermodynamic value."""


@dataclass(frozen=True)
class GibbsMinimumResult:
    """Fresh local audits; arrays retain the original component/element basis."""

    component_amounts_mol: np.ndarray
    phase_amounts_mol: np.ndarray
    phase_element_amounts_mol: np.ndarray
    element_residual: np.ndarray
    reaction_residual: np.ndarray
    elemental_potentials_rt: np.ndarray
    gibbs_rt: float
    accepted: bool
    solver_message: str
    reduced_potentials_rt: np.ndarray
    audit_reasons: tuple[str, ...]
    derivative_error_rt: float
    extensivity_error: float
    initial_gibbs_rt: float
    constrained_kkt_residual_rt: np.ndarray
    composition_constraints: tuple[dict, ...]


def _feasible_start(formula: np.ndarray, budget: np.ndarray,
                    domain: Optional[np.ndarray] = None) -> np.ndarray:
    """Find a relative-interior inventory without adding any atoms or floors."""
    # Row scaling retains trace budgets, while component limits make the
    # max-min interior search insensitive to their absolute scales.
    limits = np.min(np.where(formula > 0, budget[:, None] / np.where(formula > 0, formula, 1), np.inf), axis=0)
    if not np.all(np.isfinite(limits)):
        raise ValueError("Every component must contain a budgeted element.")
    matrix = formula * limits / budget[:, None]
    size = formula.shape[1]
    inequalities = np.empty((0, size)) if domain is None else -domain * limits
    interior = linprog(
        np.r_[np.zeros(size), -1.],
        A_ub=np.vstack((np.c_[-np.eye(size), np.ones(size)],
                        np.c_[inequalities, np.zeros(len(inequalities))])),
        b_ub=np.zeros(size + len(inequalities)), A_eq=np.c_[matrix, np.zeros(len(budget))],
        b_eq=np.ones(len(budget)), bounds=[(0, None)] * (size + 1), method="highs",
    )
    if not interior.success:
        raise ValueError("The declared component support cannot realize the element inventory.")
    if interior.x[-1] > 0:
        return limits * interior.x[:-1]
    # An equality may force a supported component to zero. Average feasible
    # component maxima to find the relative interior of that smaller face.
    points = []
    for index in range(size):
        objective = np.zeros(size)
        objective[index] = -1.
        result = linprog(objective, A_eq=matrix, b_eq=np.ones(len(budget)),
                         A_ub=inequalities if len(inequalities) else None,
                         b_ub=np.zeros(len(inequalities)) if len(inequalities) else None,
                         bounds=(0, None), method="highs")
        if not result.success:
            raise ValueError("Could not resolve the feasible component support.")
        points.append(result.x)
    return limits * np.mean(points, axis=0)


def _composition_constraints(problem, bounds):
    """Build C n >= 0 in the active component order, with no phase floor."""
    bounds = dict(bounds or {})
    if set(bounds) - set(problem.phases):
        raise ValueError("Composition bounds must name active phases.")
    rows, labels, sections = [], [], []
    for phase, section in zip(problem.phases, problem.phase_slices):
        if phase not in bounds:
            continue
        values = tuple(map(np.asarray, bounds[phase]))
        if len(values) != 2 or not all(np.isrealobj(value) for value in values):
            raise ValueError("Supply real lower and upper phase composition bounds.")
        lo, hi = (value.astype(float) for value in values)
        size = section.stop - section.start
        if (lo.shape != (size,) or hi.shape != lo.shape
                or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi))
                or np.any(lo < 0) or np.any(hi > 1) or np.any(lo > hi)
                or lo.sum() > 1 or hi.sum() < 1):
            raise ValueError("Supply feasible composition bounds in active phase component order.")
        for offset, name in enumerate(problem.species[section]):
            for kind, value in (("lower", lo[offset]), ("upper", hi[offset])):
                if (kind == "lower" and value == 0) or (kind == "upper" and value == 1):
                    continue
                row = np.zeros(len(problem.species))
                row[section] = -value if kind == "lower" else value
                row[section.start + offset] += 1 if kind == "lower" else -1
                rows.append(row)
                labels.append({"phase": phase, "component": name, "bound": kind, "value": float(value)})
                sections.append(section)
    return np.asarray(rows).reshape((-1, len(problem.species))), labels, sections


def minimize_gibbs(
    problem: LocalProblem, temperature_k: float, pressure_bar: float,
    element_amounts_mol: np.ndarray, phase_callbacks: Mapping[str, PhaseCallback],
    *, initial_component_amounts_mol: Optional[np.ndarray] = None,
    phase_composition_bounds: Optional[Mapping[str, tuple[np.ndarray, np.ndarray]]] = None,
    maxiter: int = 1000, polish: bool = True,
) -> GibbsMinimumResult:
    """Minimize extensive G/(RT) with nonnegative amounts and exact atom budgets.

    Callbacks use K/bar/mol and return extensive ``gibbs_rt`` and its amount
    derivative ``mu_rt``. Zero components require continuous finite energy;
    singular endpoint chemical potentials are allowed during minimization.
    No ideal term, gas pressure term, or reaction offset is added here.
    Complete phase disappearance belongs to a separate phase candidate.

    Optional composition bounds follow each active phase's component order.
    They impose homogeneous inequalities ``C n >= 0`` and allow exact phase
    disappearance. Boundary multipliers enter the constrained KKT audit;
    raw reaction and reduced-potential diagnostics remain unmodified.

    SLSQP minimizes the scalar energy in inventory-normalized amount variables.
    An optional stationarity refinement may improve an interior minimum only
    if its energy does not increase. Final finite differences, extensivity,
    conservation and KKT conditions are evaluated anew. Acceptance tolerances
    are 1e-9 (relative atoms), 1e-8 (dimensionless KKT), 5e-6 (numerical energy
    derivative) and 5e-9 (extensivity). These are numerical, not material errors.
    """
    if problem.reaction_offset is not None:
        raise ValueError("Common Gibbs energy cannot include reaction-specific offsets.")
    if set(phase_callbacks) != set(problem.phases):
        raise ValueError("Supply exactly one full-potential callback per active phase.")
    if not all(np.isfinite(v) and v > 0 for v in (temperature_k, pressure_bar)):
        raise ValueError("Temperature and pressure must be positive and finite.")
    if not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer.")
    budgets = _budgets(element_amounts_mol, len(problem.elements))
    positive = budgets > 0
    if not np.array_equal(np.flatnonzero(positive), problem.element_indices):
        raise ValueError("Element support changed; rebuild the branch.")
    indices = problem.species_indices
    formula = np.asarray(problem.formula_matrix)
    scale = budgets.sum()
    normalized_budget = budgets[positive] / scale
    domain, domain_labels, domain_sections = _composition_constraints(problem, phase_composition_bounds)
    feasible = _feasible_start(formula, normalized_budget, domain)
    if any(feasible[section].sum() == 0 for section in problem.phase_slices):
        raise ValueError("A declared phase is unreachable; evaluate the separate absent branch.")
    if initial_component_amounts_mol is None:
        initial = feasible
    else:
        supplied = np.asarray(initial_component_amounts_mol, dtype=float)
        excluded = np.ones(len(problem.full_species), dtype=bool)
        excluded[indices] = False
        if (supplied.shape != excluded.shape or not np.all(np.isfinite(supplied))
                or np.any(supplied < 0) or np.any(supplied[excluded] != 0)):
            raise ValueError("Initial amounts must be nonnegative on active support and zero elsewhere.")
        initial = supplied[indices] / scale
        if not np.allclose(formula @ initial, normalized_budget, rtol=1e-9, atol=0):
            raise ValueError("Initial amounts must obey the declared absolute element inventory.")
        if np.any(domain @ initial < -1e-13 * initial.sum()):
            raise ValueError("Initial amounts must obey the declared phase composition bounds.")
    reachable = feasible > 0

    def evaluate(amounts):
        mu = np.empty(len(indices))
        gibbs = 0.0
        for phase, section in zip(problem.phases, problem.phase_slices):
            state = phase_callbacks[phase](temperature_k, pressure_bar, amounts[section])
            phase_mu = np.asarray(state.mu_rt, dtype=float)
            if phase_mu.shape != amounts[section].shape or not np.isfinite(state.gibbs_rt):
                raise ValueError("A phase returned unavailable energy or incorrectly shaped potentials.")
            present = amounts[section] > 0
            if not np.all(np.isfinite(phase_mu[present])):
                raise ValueError("A phase omitted a present component potential.")
            euler = float(amounts[section][present] @ phase_mu[present])
            if not np.isclose(state.gibbs_rt, euler, rtol=5e-9, atol=1e-10 * scale):
                raise ValueError("A phase's energy and potentials violate Euler's identity.")
            mu[section] = phase_mu
            gibbs += state.gibbs_rt
        return float(gibbs), mu

    def objective(x):
        try:
            return evaluate(scale * x)[0] / scale
        except PhaseEvaluationError:
            # A provider may reject a trial near its represented domain edge.
            # Only the line search can reject it: initial/final states and
            # independent audits must still have actual property evaluations.
            return np.inf

    def gradient(x):
        energy, mu = evaluate(scale * x)
        for phase, section in zip(problem.phases, problem.phase_slices):
            if x[section].sum() == 0 and feasible[section].sum() > 0:
                # A vanished phase has no unique composition derivative. Use
                # a feasible interior direction only for the scalar search;
                # final acceptance still requires a separate absent branch.
                state = phase_callbacks[phase](temperature_k, pressure_bar, scale * feasible[section])
                mu[section] = np.asarray(state.mu_rt)
        # Only the search direction uses finite one-sided secants where the
        # true derivative is singular. The scalar energy never uses a floor.
        mu[~reachable] = 0.
        for index in np.flatnonzero(reachable & ~np.isfinite(mu)):
            displaced = x.copy()
            step = 1e-8 * max(1., x.sum())
            displaced[index] += step
            mu[index] = (objective(displaced) - energy / scale) / step
        return mu

    initial_energy = evaluate(scale * initial)[0]
    row_matrix = formula / normalized_budget[:, None]
    constraints = [{"type": "eq", "fun": lambda x: row_matrix @ x - 1.,
                    "jac": lambda x: row_matrix}]
    if len(domain):
        domain_scale = np.array([max(initial[section].sum(), feasible[section].sum())
                                 for section in domain_sections])
        search_domain = domain / domain_scale[:, None]
        constraints.append({"type": "ineq", "fun": lambda x: search_domain @ x,
                            "jac": lambda x: search_domain})
    def scalar_solve(start):
        return minimize(objective, start, jac=gradient, method="SLSQP",
                        bounds=[(0., None) if exists else (0., 0.) for exists in reachable],
                        constraints=constraints, options={"ftol": 1e-14, "maxiter": maxiter})

    optimization = scalar_solve(initial)
    if len(domain) and any(optimization.x[section].sum() < 1e-12 * initial[section].sum()
                           for section in domain_sections):
        # A linearized step can collapse a complete ideal solution and lose
        # its composition. Re-enter on the existing atom-conserving segment
        # by scalar energy minimization; no positive amount bound is added.
        direction = optimization.x - initial
        line = minimize_scalar(lambda fraction: objective(initial + fraction * direction),
                               bounds=(0., 1.), method="bounded", options={"xatol": 1e-12})
        seed = initial + line.x * direction
        if line.success and objective(seed) < objective(optimization.x):
            optimization = scalar_solve(seed)
    final = optimization.x
    # Scalar minimization comes first. Refinement is not used to replace a
    # failed/incomplete minimization and cannot increase the returned energy.
    if polish and optimization.success and np.all(final[reachable] > 0):
        active_formula = formula[:, reachable]
        relative_slack = np.array([value / final[section].sum() if final[section].sum() else np.inf
                                  for value, section in zip(domain @ final, domain_sections)])
        active_domain = domain[relative_slack < 1e-8]
        tangent = np.vstack((active_formula, active_domain[:, reachable]))
        reaction = null_space(tangent).T
        def residual(log_amounts):
            candidate = np.zeros_like(final)
            candidate[reachable] = np.exp(log_amounts)
            _, mu = evaluate(scale * candidate)
            return np.r_[row_matrix @ candidate - 1., active_domain @ candidate,
                         reaction @ mu[reachable]]
        try:
            refined = least_squares(residual, np.log(final[reachable]),
                                    xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=maxiter)
            candidate = np.zeros_like(final)
            candidate[reachable] = np.exp(refined.x)
            if (np.max(np.abs(residual(refined.x)), initial=0.) < 1e-8
                    and all(value >= -1e-12 * candidate[section].sum()
                            for value, section in zip(domain @ candidate, domain_sections))
                    and objective(candidate) <= objective(final) + 1e-12):
                final = candidate
        except PhaseEvaluationError:
            pass  # Keep the actual scalar minimum; its final audits still run.

    amounts = scale * final
    energy, mu = evaluate(amounts)
    present = amounts > 0
    reasons = []
    if not optimization.success:
        reasons.append("scalar minimizer did not converge")
    if np.linalg.matrix_rank(formula[:, present]) < len(normalized_budget):
        reasons.append("present components do not identify all elemental potentials")
    elemental = np.zeros(len(budgets))
    relative_slack = np.array([value / amounts[section].sum() if amounts[section].sum() else np.inf
                              for value, section in zip(domain @ amounts, domain_sections)])
    active_domain = relative_slack < 1e-8
    multipliers = np.zeros(len(domain))
    if np.any(active_domain):
        system = np.c_[formula[:, present].T, domain[active_domain][:, present].T]
        dual = lsq_linear(system, mu[present],
                          bounds=(np.r_[np.full(len(normalized_budget), -np.inf),
                                         np.zeros(active_domain.sum())], np.inf),
                          tol=1e-14, max_iter=maxiter)
        elemental[positive] = dual.x[:len(normalized_budget)]
        multipliers[active_domain] = dual.x[len(normalized_budget):]
    else:
        elemental[positive] = np.linalg.lstsq(formula[:, present].T, mu[present], rcond=None)[0]
    reduced = mu - formula.T @ elemental[positive]
    constrained = reduced - domain.T @ multipliers
    reaction = null_space(formula[:, present]).T @ mu[present]
    if np.max(np.abs(constrained[present]), initial=0.) >= 1e-8:
        reasons.append("present-component KKT residual exceeds tolerance")
    if np.any(relative_slack < -1e-10):
        reasons.append("phase composition lies outside the declared domain")
    if np.max(np.abs(multipliers * relative_slack), initial=0.) >= 1e-8:
        reasons.append("composition-bound complementarity exceeds tolerance")
    absent_reachable = reachable & ~present
    if np.any(~np.isfinite(constrained[absent_reachable])) or np.any(constrained[absent_reachable] < -1e-8):
        reasons.append("zero-component insertion is unresolved or favorable")
    if any(amounts[section].sum() == 0 for section in problem.phase_slices):
        reasons.append("a declared phase disappeared; evaluate a separate phase candidate")

    # Differentiate an independent scalar where the callback supplies one.
    # Otherwise use each phase's own energy, avoiding cancellation against
    # unrelated phases, and adapt the stencil before Richardson extrapolation.
    derivative_error = 0.0
    for phase, section in zip(problem.phases, problem.phase_slices):
        callback = phase_callbacks[phase]
        phase_amounts = amounts[section]
        phase_present = phase_amounts > 0
        phase_energy = float(callback(temperature_k, pressure_bar, phase_amounts).gibbs_rt)
        differentiator = getattr(callback, "energy_value_and_grad_rt", None)
        if differentiator is not None:
            independent_energy, independent_mu = differentiator(temperature_k, pressure_bar, phase_amounts)
            independent_mu = np.asarray(independent_mu, dtype=float)
            if (independent_mu.shape != phase_amounts.shape
                    or not np.all(np.isfinite(independent_mu[phase_present]))
                    or not np.isfinite(independent_energy)
                    or not np.isclose(independent_energy, phase_energy, rtol=5e-9, atol=1e-10 * scale)):
                reasons.append("independent phase scalar is inconsistent or unavailable")
                derivative_error = np.inf
            else:
                derivative_error = max(derivative_error, np.max(np.abs(
                    independent_mu[phase_present] - mu[section][phase_present]), initial=0.))
            continue
        for index in np.flatnonzero(phase_present):
            magnitude = max(abs(phase_energy), phase_amounts.sum())
            fraction = np.clip(20 * np.finfo(float).eps * magnitude
                               / (phase_amounts[index] * 5e-7), 1e-4, .02)
            estimates = []
            for relative_step in (2 * fraction, fraction):
                step = phase_amounts[index] * relative_step
                delta = np.zeros_like(phase_amounts)
                delta[index] = step
                plus = callback(temperature_k, pressure_bar, phase_amounts + delta).gibbs_rt
                minus = callback(temperature_k, pressure_bar, phase_amounts - delta).gibbs_rt
                if not np.isfinite(plus) or not np.isfinite(minus):
                    estimates.append(np.nan)
                    break
                estimates.append((plus - minus) / (2 * step))
            if len(estimates) != 2 or not np.all(np.isfinite(estimates)):
                derivative_error = np.inf
                reasons.append("scalar derivative is unavailable at a finite-difference point")
                continue
            extrapolated = (4 * estimates[1] - estimates[0]) / 3
            if not np.isfinite(extrapolated):
                derivative_error = np.inf
                reasons.append("scalar derivative extrapolation is nonfinite")
                continue
            cancellation = 20 * np.finfo(float).eps * magnitude / (phase_amounts[index] * fraction)
            if cancellation > 5e-6:
                reasons.append("scalar derivative of a trace component is numerically unresolved")
            derivative_error = max(derivative_error, abs(extrapolated - mu[section][index]))
    if derivative_error >= 5e-6:
        reasons.append("scalar energy derivative disagrees with the supplied potentials")
    scaled_energy, scaled_mu = evaluate(1.7 * amounts)
    extensivity_error = max(abs(scaled_energy / 1.7 - energy) / scale,
                            np.max(np.abs(scaled_mu[present] - mu[present]), initial=0.))
    if extensivity_error >= 5e-9:
        reasons.append("phase energy is not extensive or its potentials are not intensive")
    # Last provider evaluation is always at the returned state.
    energy, _ = evaluate(amounts)
    full_amounts = np.zeros(len(problem.full_species))
    full_amounts[indices] = amounts
    full_formula = np.asarray(problem.full_formula_matrix)
    totals = full_formula @ full_amounts
    element_residual = np.zeros_like(budgets)
    element_residual[positive] = totals[positive] / budgets[positive] - 1.
    if np.max(np.abs(element_residual), initial=0.) >= 1e-9 or np.any(totals[~positive] != 0):
        reasons.append("element inventory audit failed")
    phase_amounts = np.array([amounts[s].sum() for s in problem.phase_slices])
    phase_elements = np.array([full_formula[:, indices[s]] @ amounts[s] for s in problem.phase_slices])
    full_reduced = np.full(len(problem.full_species), np.nan)
    full_reduced[indices] = reduced
    full_constrained = np.full(len(problem.full_species), np.nan)
    full_constrained[indices] = constrained
    constraint_report = tuple({**label, "fraction_slack": float(slack), "multiplier_rt": float(multiplier)}
                              for label, slack, multiplier in zip(domain_labels, relative_slack, multipliers))
    return GibbsMinimumResult(
        full_amounts, phase_amounts, phase_elements, element_residual, reaction,
        elemental, energy, not reasons, str(optimization.message), full_reduced,
        tuple(dict.fromkeys(reasons)), derivative_error, extensivity_error, initial_energy,
        full_constrained, constraint_report,
    )
