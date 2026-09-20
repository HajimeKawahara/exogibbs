"""Constrained scalar Gibbs minimization on an explicitly declared phase branch.

This example consumes the same absolute full-potential callbacks as the legacy
local root. Numerical acceptance establishes local first-order conditions,
not global phase stability or material calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares, linprog, minimize

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


def _feasible_start(formula: np.ndarray, budget: np.ndarray) -> np.ndarray:
    """Find a relative-interior inventory without adding any atoms or floors."""
    # Row scaling retains trace budgets, while component limits make the
    # max-min interior search insensitive to their absolute scales.
    limits = np.min(np.where(formula > 0, budget[:, None] / np.where(formula > 0, formula, 1), np.inf), axis=0)
    if not np.all(np.isfinite(limits)):
        raise ValueError("Every component must contain a budgeted element.")
    matrix = formula * limits / budget[:, None]
    size = formula.shape[1]
    interior = linprog(
        np.r_[np.zeros(size), -1.], A_ub=np.c_[-np.eye(size), np.ones(size)],
        b_ub=np.zeros(size), A_eq=np.c_[matrix, np.zeros(len(budget))],
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
                         bounds=(0, None), method="highs")
        if not result.success:
            raise ValueError("Could not resolve the feasible component support.")
        points.append(result.x)
    return limits * np.mean(points, axis=0)


def minimize_gibbs(
    problem: LocalProblem, temperature_k: float, pressure_bar: float,
    element_amounts_mol: np.ndarray, phase_callbacks: Mapping[str, PhaseCallback],
    *, initial_component_amounts_mol: Optional[np.ndarray] = None,
    maxiter: int = 1000, polish: bool = True,
) -> GibbsMinimumResult:
    """Minimize extensive G/(RT) with nonnegative amounts and exact atom budgets.

    Callbacks use K/bar/mol and return extensive ``gibbs_rt`` and its amount
    derivative ``mu_rt``. Zero components require continuous finite energy;
    singular endpoint chemical potentials are allowed during minimization.
    No ideal term, gas pressure term, or reaction offset is added here.
    Complete phase disappearance belongs to a separate phase candidate.

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
    feasible = _feasible_start(formula, normalized_budget)
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
        # Only the search direction uses finite one-sided secants where the
        # true derivative is singular. The scalar energy never uses a floor.
        for index in np.flatnonzero(~np.isfinite(mu)):
            displaced = x.copy()
            step = 1e-8 * max(1., x.sum())
            displaced[index] += step
            mu[index] = (objective(displaced) - energy / scale) / step
        return mu

    initial_energy = evaluate(scale * initial)[0]
    row_matrix = formula / normalized_budget[:, None]
    optimization = minimize(
        objective, initial, jac=gradient, method="SLSQP",
        bounds=[(0., None) if exists else (0., 0.) for exists in reachable],
        constraints={"type": "eq", "fun": lambda x: row_matrix @ x - 1.,
                     "jac": lambda x: row_matrix},
        options={"ftol": 1e-14, "maxiter": maxiter},
    )
    final = optimization.x
    # Scalar minimization comes first. Refinement is not used to replace a
    # failed/incomplete minimization and cannot increase the returned energy.
    if polish and optimization.success and np.all(final[reachable] > 0):
        active_formula = formula[:, reachable]
        reaction = null_space(active_formula).T
        def residual(log_amounts):
            candidate = np.zeros_like(final)
            candidate[reachable] = np.exp(log_amounts)
            _, mu = evaluate(scale * candidate)
            return np.r_[row_matrix @ candidate - 1., reaction @ mu[reachable]]
        try:
            refined = least_squares(residual, np.log(final[reachable]),
                                    xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=maxiter)
            candidate = np.zeros_like(final)
            candidate[reachable] = np.exp(refined.x)
            if (np.max(np.abs(residual(refined.x)), initial=0.) < 1e-8
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
    elemental[positive] = np.linalg.lstsq(formula[:, present].T, mu[present], rcond=None)[0]
    reduced = mu - formula.T @ elemental[positive]
    reaction = null_space(formula[:, present]).T @ mu[present]
    if np.max(np.abs(reduced[present]), initial=0.) >= 1e-8:
        reasons.append("present-component KKT residual exceeds tolerance")
    absent_reachable = reachable & ~present
    if np.any(~np.isfinite(reduced[absent_reachable])) or np.any(reduced[absent_reachable] < -1e-8):
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
                estimates.append((plus - minus) / (2 * step))
            extrapolated = (4 * estimates[1] - estimates[0]) / 3
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
    return GibbsMinimumResult(
        full_amounts, phase_amounts, phase_elements, element_residual, reaction,
        elemental, energy, not reasons, str(optimization.message), full_reduced,
        tuple(dict.fromkeys(reasons)), derivative_error, extensivity_error, initial_energy,
    )
