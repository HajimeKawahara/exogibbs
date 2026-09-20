"""Local roots using complete phase chemical potentials
=========================================================

Callbacks evaluate the current amounts and include standards and all mixing.
This optional example path has no JAX derivative or global stability claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy as jax_xlogy
import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp, xlogy

from local import LocalProblem, _budgets


@dataclass(frozen=True)
class PhaseState:
    """Full chemical potentials mu/(RT) and extensive Gibbs energy G/(RT)."""

    mu_rt: np.ndarray
    gibbs_rt: float


PhaseCallback = Callable[[float, float, np.ndarray], PhaseState]


@dataclass(frozen=True)
class FullPotentialResult:
    """Absolute local amounts, with a fresh independent acceptance evaluation."""

    component_amounts_mol: np.ndarray
    phase_amounts_mol: np.ndarray
    phase_element_amounts_mol: np.ndarray
    element_residual: np.ndarray
    reaction_residual: np.ndarray
    elemental_potentials_rt: np.ndarray
    gibbs_rt: float
    accepted: bool
    solver_message: str


def ideal_phase(
    standard_potentials_rt: Callable[[float, float], np.ndarray],
    *, gas: bool = False, standard_pressure_bar: float = 1.0,
) -> PhaseCallback:
    """Build an ideal phase in its active component order, using K and bar."""
    if not np.isfinite(standard_pressure_bar) or standard_pressure_bar <= 0:
        raise ValueError("The standard pressure must be positive and finite.")

    def evaluate(temperature: float, pressure: float, amounts: np.ndarray) -> PhaseState:
        n = np.asarray(amounts, dtype=float)
        standard = np.asarray(standard_potentials_rt(temperature, pressure), dtype=float)
        if (n.ndim != 1 or not n.size or standard.shape != n.shape
                or np.any(n < 0) or not np.all(np.isfinite(n))
                or not np.all(np.isfinite(standard))):
            raise ValueError("Ideal phase amounts and standards must be finite matching vectors.")
        if not all(np.isfinite(v) and v > 0 for v in (temperature, pressure)):
            raise ValueError("Temperature and pressure must be positive and finite.")
        if gas:
            standard = standard + np.log(pressure / standard_pressure_bar)
        total = n.sum()
        if total == 0:
            return PhaseState(np.full_like(n, np.nan), 0.)
        fractions = n / total
        with np.errstate(divide="ignore"):
            mu = standard + np.log(fractions)
        return PhaseState(mu, float(n @ standard + np.sum(xlogy(n, fractions))))

    def scalar(n, standard):
        total = jnp.sum(n)
        return jnp.dot(n, standard) + jnp.sum(jax_xlogy(n, n)) - jax_xlogy(total, total)

    differentiate = jax.jit(jax.value_and_grad(scalar, argnums=0))

    def energy_value_and_grad_rt(temperature, pressure, amounts):
        standard = jnp.asarray(standard_potentials_rt(temperature, pressure))
        if gas:
            standard = standard + jnp.log(pressure / standard_pressure_bar)
        return differentiate(jnp.asarray(amounts), standard)

    # This independent scalar derivative remains resolvable for trace ideal
    # components whose changes in the total energy are below float64 precision.
    evaluate.energy_value_and_grad_rt = energy_value_and_grad_rt
    return evaluate


def solve_full_potentials(
    problem: LocalProblem,
    temperature_k: float,
    pressure_bar: float,
    element_amounts_mol: np.ndarray,
    phase_callbacks: Mapping[str, PhaseCallback],
    *, initial_component_amounts_mol: Optional[np.ndarray] = None,
    max_nfev: int = 200,
) -> FullPotentialResult:
    """Solve one declared phase branch with a numerical host Jacobian.

Use ``build_problem`` to remove absent phases and exact-zero element support.
Full callbacks must follow each active phase's component order. No ideal term,
pressure term, or empirical reaction offset is added to their potentials.
Only residuals below 1e-9 (elements) / 1e-8 (reactions) are accepted. Acceptance
is local equation closure; candidate phase stability remains a separate gate.
"""
    if problem.reaction_offset is not None:
        raise ValueError("Full-potential roots require common standards without reaction offsets.")
    if set(phase_callbacks) != set(problem.phases):
        raise ValueError("Supply exactly one full-potential callback per active phase.")
    if not all(np.isfinite(value) and value > 0 for value in (temperature_k, pressure_bar)):
        raise ValueError("Temperature and pressure must be positive and finite.")
    budgets = _budgets(element_amounts_mol, len(problem.elements))
    positive = budgets > 0
    if not np.array_equal(np.flatnonzero(positive), problem.element_indices):
        raise ValueError("Element support changed; rebuild the branch.")
    indices = problem.species_indices
    formula = np.asarray(problem.formula_matrix)
    reactions = np.asarray(problem.reaction_matrix)
    scale = budgets.sum()
    if initial_component_amounts_mol is None:
        limits = np.where(formula > 0, budgets[positive, None] / np.where(formula > 0, formula, 1), np.inf)
        initial = np.min(limits, axis=0) / len(indices)
    else:
        full_initial = np.asarray(initial_component_amounts_mol, dtype=float)
        excluded = np.ones(len(problem.full_species), dtype=bool)
        excluded[indices] = False
        if (full_initial.shape != excluded.shape or not np.all(np.isfinite(full_initial))
                or np.any(full_initial[indices] <= 0) or np.any(full_initial[excluded] != 0)):
            raise ValueError("Initial amounts must be positive on the active support and zero elsewhere.")
        initial = full_initial[indices]

    def evaluate(log_amounts):
        amounts = scale * np.exp(log_amounts)
        mu = np.empty(len(indices))
        gibbs = 0.0
        for phase, section in zip(problem.phases, problem.phase_slices):
            state = phase_callbacks[phase](temperature_k, pressure_bar, amounts[section])
            phase_mu = np.asarray(state.mu_rt, dtype=float)
            if phase_mu.shape != amounts[section].shape or not np.all(np.isfinite(phase_mu)):
                raise ValueError("A phase returned unavailable or incorrectly shaped full potentials.")
            euler = float(amounts[section] @ phase_mu)
            if not np.isfinite(state.gibbs_rt) or not np.isclose(state.gibbs_rt, euler, rtol=5e-9, atol=1e-10 * scale):
                raise ValueError("A phase's full potentials and extensive Gibbs energy violate Euler's identity.")
            mu[section] = phase_mu
            gibbs += state.gibbs_rt
        totals = formula @ amounts
        residual = np.concatenate((np.log(totals / budgets[positive]), reactions @ mu))
        return residual, amounts, mu, gibbs

    root = least_squares(
        lambda z: evaluate(z)[0], np.log(initial / scale),
        xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=max_nfev,
    )
    # This calls the provider again at the returned composition, even when an
    # optimizer reports success; a frozen or stale potential is not acceptance.
    _, amounts, mu, gibbs = evaluate(root.x)
    full_amounts = np.zeros(len(problem.full_species))
    full_amounts[indices] = amounts
    full_formula = np.asarray(problem.full_formula_matrix)
    element_residual = np.zeros(len(budgets))
    element_residual[positive] = (full_formula @ full_amounts)[positive] / budgets[positive] - 1
    reaction_residual = reactions @ mu
    elemental = np.zeros(len(budgets))
    elemental[positive] = np.linalg.lstsq(formula.T, mu, rcond=None)[0]
    phase_amounts = np.array([amounts[section].sum() for section in problem.phase_slices])
    phase_elements = np.array([
        full_formula[:, indices[section]] @ amounts[section] for section in problem.phase_slices
    ])
    accepted = bool(
        np.max(np.abs(element_residual)) < 1e-9
        and np.max(np.abs(reaction_residual), initial=0.0) < 1e-8
    )
    return FullPotentialResult(
        full_amounts, phase_amounts, phase_elements, element_residual,
        reaction_residual, elemental, gibbs, accepted, str(root.message),
    )


def pure_phase_insertion_rt(
    standard_potential_rt: float, formula: np.ndarray, elemental_potentials_rt: np.ndarray,
) -> float:
    """Return D/(RT); a missing pure phase is admissible only for D >= 0."""
    return float(standard_potential_rt - np.asarray(formula) @ elemental_potentials_rt)


def ideal_solution_insertion_rt(
    standard_potentials_rt: np.ndarray, formula: np.ndarray, elemental_potentials_rt: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Exactly minimize insertion energy over an unconstrained ideal solution.

    This closed form searches its entire simplex, including limiting states.
    It must not be used for a nonideal alloy or a restricted Fe-rich domain.
    ``formula`` has element rows and component columns. Remove components
    containing exact-zero-budget elements before calling; their elemental
    potentials are undefined and cannot be inferred from a zero placeholder.
    """
    costs = np.asarray(standard_potentials_rt) - np.asarray(formula).T @ elemental_potentials_rt
    log_partition = logsumexp(-costs)
    return float(-log_partition), np.exp(-costs - log_partition)
