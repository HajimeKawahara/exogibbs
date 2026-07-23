"""Pure mathematical contracts for the fixed-support v2 solver."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from exogibbs.optimize.fixed_support_v2.types import (
    FixedSupportProblem,
    KKTComponentNorms,
    OriginalDirection,
    OriginalState,
    PhysicalAmounts,
    ResidualComponents,
)


def canonical_gas_source(
    gas_standard_source: Any,
    pressure: Any,
    reference_pressure: Any = 1.0,
) -> Any:
    """Return the iterate-independent source ``hgas + log(P / Pref)``."""

    hgas = jnp.asarray(gas_standard_source)
    pressure_value = jnp.asarray(pressure, dtype=hgas.dtype)
    reference_value = jnp.asarray(reference_pressure, dtype=hgas.dtype)
    return hgas + jnp.log(pressure_value / reference_value)


def canonical_gas_source_from_legacy(
    legacy_stationarity_source: Any,
    initial_qtot: Any,
) -> Any:
    """Convert legacy ``g_init = gamma - qtot_init`` to ``gamma`` once."""

    legacy_source = jnp.asarray(legacy_stationarity_source)
    return legacy_source + jnp.asarray(initial_qtot, dtype=legacy_source.dtype)


def physical_amounts(state: OriginalState) -> PhysicalAmounts:
    """Map the original log primal coordinates to physical amounts."""

    return PhysicalAmounts(
        gas=jnp.exp(jnp.asarray(state.q)),
        condensate=jnp.exp(jnp.asarray(state.r)),
        total_gas=jnp.exp(jnp.asarray(state.qtot)),
    )


def log_primal_coordinates(amounts: PhysicalAmounts) -> tuple[Any, Any, Any]:
    """Map strictly positive physical amounts to ``(q, r, qtot)``."""

    return (
        jnp.log(jnp.asarray(amounts.gas)),
        jnp.log(jnp.asarray(amounts.condensate)),
        jnp.log(jnp.asarray(amounts.total_gas)),
    )


def residual_components(
    problem: FixedSupportProblem,
    state: OriginalState,
) -> ResidualComponents:
    """Evaluate the canonical original log-KKT residual.

    The gas block is always ``q + gamma - qtot - Ag.T @ lambda``.  No
    reference iterate is captured, so evaluation at a trial state necessarily
    uses that trial's current ``qtot``.
    """

    q = jnp.asarray(state.q)
    dtype = q.dtype
    r = jnp.asarray(state.r, dtype=dtype)
    lambda_ = jnp.asarray(state.lambda_, dtype=dtype)
    rho = jnp.asarray(state.rho, dtype=dtype)
    qtot = jnp.asarray(state.qtot, dtype=dtype)
    epsilon = jnp.asarray(state.epsilon, dtype=dtype)
    ag = jnp.asarray(problem.gas_formula_matrix, dtype=dtype)
    ac = jnp.asarray(problem.condensate_formula_matrix, dtype=dtype)
    target = jnp.asarray(problem.target_inventory, dtype=dtype)
    gamma = jnp.asarray(problem.gamma, dtype=dtype)
    hcond = jnp.asarray(problem.condensate_standard_source, dtype=dtype)
    amounts = physical_amounts(state)
    eta = jnp.exp(rho)

    return ResidualComponents(
        gas_stationarity=q + gamma - qtot - ag.T @ lambda_,
        condensate_stationarity=hcond - ac.T @ lambda_ - eta,
        budget=ag @ amounts.gas + ac @ amounts.condensate - target,
        complementarity=r + rho - epsilon,
        total_density=jnp.asarray(
            [jnp.sum(amounts.gas) - amounts.total_gas], dtype=dtype
        ),
    )


def residual_vector(components: ResidualComponents) -> Any:
    """Concatenate residual blocks in the canonical equation order."""

    return jnp.concatenate(
        [
            jnp.ravel(components.gas_stationarity),
            jnp.ravel(components.condensate_stationarity),
            jnp.ravel(components.budget),
            jnp.ravel(components.complementarity),
            jnp.ravel(components.total_density),
        ]
    )


def kkt_component_norms(
    problem: FixedSupportProblem,
    state: OriginalState,
) -> KKTComponentNorms:
    """Return independently scaled convergence norms for all KKT blocks."""

    residual = residual_components(problem, state)
    dtype = jnp.asarray(state.q).dtype
    budget_scale = jnp.asarray(problem.budget_row_scale, dtype=dtype)
    total_scale = jnp.asarray(problem.total_density_row_scale, dtype=dtype)

    def max_abs(value):
        return jnp.max(jnp.abs(jnp.asarray(value)), initial=0.0)

    return KKTComponentNorms(
        gas_stationarity=max_abs(residual.gas_stationarity),
        condensate_stationarity=max_abs(residual.condensate_stationarity),
        budget_scaled=max_abs(budget_scale * residual.budget),
        complementarity=max_abs(residual.complementarity),
        total_density_scaled=max_abs(total_scale * residual.total_density),
    )


def scaled_equality_residual(
    problem: FixedSupportProblem,
    components: ResidualComponents,
) -> Any:
    """Return ``[Wb Cb, wt Ct]`` using fixed problem row scales."""

    dtype = jnp.asarray(components.budget).dtype
    budget_scale = jnp.asarray(problem.budget_row_scale, dtype=dtype)
    total_scale = jnp.asarray(problem.total_density_row_scale, dtype=dtype)
    return jnp.concatenate(
        [
            jnp.ravel(budget_scale * components.budget),
            jnp.ravel(total_scale * components.total_density),
        ]
    )


def filter_violation(
    problem: FixedSupportProblem,
    state: OriginalState,
) -> Any:
    """Return the original filter violation ``theta = norm_1([Wb Cb, wt Ct])``."""

    scaled = scaled_equality_residual(problem, residual_components(problem, state))
    return jnp.linalg.norm(scaled, ord=1)


def barrier_objective(
    problem: FixedSupportProblem,
    state: OriginalState,
) -> Any:
    """Return the canonical dimensionless Gibbs log-barrier objective."""

    q = jnp.asarray(state.q)
    dtype = q.dtype
    r = jnp.asarray(state.r, dtype=dtype)
    qtot = jnp.asarray(state.qtot, dtype=dtype)
    gamma = jnp.asarray(problem.gamma, dtype=dtype)
    hcond = jnp.asarray(problem.condensate_standard_source, dtype=dtype)
    amounts = physical_amounts(state)
    mu = jnp.exp(jnp.asarray(state.epsilon, dtype=dtype))
    return (
        jnp.dot(amounts.gas, gamma + q - qtot)
        + jnp.dot(amounts.condensate, hcond)
        - mu * jnp.sum(r)
    )


def barrier_objective_directional_derivative(
    problem: FixedSupportProblem,
    state: OriginalState,
    direction: OriginalDirection,
) -> Any:
    """Return ``grad(phi_mu).T direction`` at ``state``."""

    q = jnp.asarray(state.q)
    dtype = q.dtype
    r = jnp.asarray(state.r, dtype=dtype)
    qtot = jnp.asarray(state.qtot, dtype=dtype)
    dq = jnp.asarray(direction.q, dtype=dtype)
    dr = jnp.asarray(direction.r, dtype=dtype)
    dqtot = jnp.asarray(direction.qtot, dtype=dtype)
    gamma = jnp.asarray(problem.gamma, dtype=dtype)
    hcond = jnp.asarray(problem.condensate_standard_source, dtype=dtype)
    amounts = physical_amounts(state)
    mu = jnp.exp(jnp.asarray(state.epsilon, dtype=dtype))
    return (
        jnp.dot(amounts.gas * (gamma + q - qtot), dq)
        + jnp.dot(amounts.gas, dq - dqtot)
        + jnp.dot(amounts.condensate * hcond, dr)
        - mu * jnp.sum(dr)
    )


def linearized_residual_components(
    problem: FixedSupportProblem,
    state: OriginalState,
    direction: OriginalDirection,
) -> ResidualComponents:
    """Return ``F(state) + J(state) direction`` block by block."""

    current = residual_components(problem, state)
    q = jnp.asarray(state.q)
    dtype = q.dtype
    r = jnp.asarray(state.r, dtype=dtype)
    rho = jnp.asarray(state.rho, dtype=dtype)
    dq = jnp.asarray(direction.q, dtype=dtype)
    dr = jnp.asarray(direction.r, dtype=dtype)
    dlambda = jnp.asarray(direction.lambda_, dtype=dtype)
    drho = jnp.asarray(direction.rho, dtype=dtype)
    dqtot = jnp.asarray(direction.qtot, dtype=dtype)
    ag = jnp.asarray(problem.gas_formula_matrix, dtype=dtype)
    ac = jnp.asarray(problem.condensate_formula_matrix, dtype=dtype)
    n = jnp.exp(q)
    m = jnp.exp(r)
    eta = jnp.exp(rho)
    ntot = jnp.exp(jnp.asarray(state.qtot, dtype=dtype))

    return ResidualComponents(
        gas_stationarity=(
            current.gas_stationarity + dq - ag.T @ dlambda - dqtot
        ),
        condensate_stationarity=(
            current.condensate_stationarity - ac.T @ dlambda - eta * drho
        ),
        budget=current.budget + ag @ (n * dq) + ac @ (m * dr),
        complementarity=current.complementarity + dr + drho,
        total_density=current.total_density
        + jnp.asarray([jnp.dot(n, dq) - ntot * dqtot], dtype=dtype),
    )


def residual_jacobian(
    problem: FixedSupportProblem,
    state: OriginalState,
) -> Any:
    """Return the dense Jacobian in canonical residual and variable order.

    Residual rows are ``(Fg, Fc, Cb, T, Ct)`` and columns are
    ``(q, r, lambda, rho, qtot)``.  This dense form is a mathematical audit
    object; future production directions are obtained from the reduced system.
    """

    q = jnp.asarray(state.q)
    dtype = q.dtype
    r = jnp.asarray(state.r, dtype=dtype)
    rho = jnp.asarray(state.rho, dtype=dtype)
    ag = jnp.asarray(problem.gas_formula_matrix, dtype=dtype)
    ac = jnp.asarray(problem.condensate_formula_matrix, dtype=dtype)
    n_gas = q.shape[0]
    n_condensate = r.shape[0]
    n_elements = ag.shape[0]
    n = jnp.exp(q)
    m = jnp.exp(r)
    eta = jnp.exp(rho)
    ntot = jnp.exp(jnp.asarray(state.qtot, dtype=dtype))

    zeros = lambda rows, columns: jnp.zeros((rows, columns), dtype=dtype)
    gas_rows = jnp.concatenate(
        [
            jnp.eye(n_gas, dtype=dtype),
            zeros(n_gas, n_condensate),
            -ag.T,
            zeros(n_gas, n_condensate),
            -jnp.ones((n_gas, 1), dtype=dtype),
        ],
        axis=1,
    )
    condensate_rows = jnp.concatenate(
        [
            zeros(n_condensate, n_gas),
            zeros(n_condensate, n_condensate),
            -ac.T,
            -jnp.diag(eta),
            zeros(n_condensate, 1),
        ],
        axis=1,
    )
    budget_rows = jnp.concatenate(
        [
            ag * n[None, :],
            ac * m[None, :],
            zeros(n_elements, n_elements),
            zeros(n_elements, n_condensate),
            zeros(n_elements, 1),
        ],
        axis=1,
    )
    complementarity_rows = jnp.concatenate(
        [
            zeros(n_condensate, n_gas),
            jnp.eye(n_condensate, dtype=dtype),
            zeros(n_condensate, n_elements),
            jnp.eye(n_condensate, dtype=dtype),
            zeros(n_condensate, 1),
        ],
        axis=1,
    )
    total_density_row = jnp.concatenate(
        [
            n[None, :],
            zeros(1, n_condensate),
            zeros(1, n_elements),
            zeros(1, n_condensate),
            -jnp.asarray(ntot).reshape((1, 1)),
        ],
        axis=1,
    )
    return jnp.concatenate(
        [
            gas_rows,
            condensate_rows,
            budget_rows,
            complementarity_rows,
            total_density_row,
        ],
        axis=0,
    )


def amount_space_equality_jacobian(problem: FixedSupportProblem) -> Any:
    """Return the constant Jacobian of ``[Cb, Ct]`` with respect to ``(n,m,ntot)``."""

    ag = jnp.asarray(problem.gas_formula_matrix)
    dtype = ag.dtype
    ac = jnp.asarray(problem.condensate_formula_matrix, dtype=dtype)
    top = jnp.concatenate(
        [ag, ac, jnp.zeros((ag.shape[0], 1), dtype=dtype)], axis=1
    )
    bottom = jnp.concatenate(
        [
            jnp.ones((1, ag.shape[1]), dtype=dtype),
            jnp.zeros((1, ac.shape[1]), dtype=dtype),
            -jnp.ones((1, 1), dtype=dtype),
        ],
        axis=1,
    )
    return jnp.concatenate([top, bottom], axis=0)


__all__ = [
    "amount_space_equality_jacobian",
    "barrier_objective",
    "barrier_objective_directional_derivative",
    "canonical_gas_source",
    "canonical_gas_source_from_legacy",
    "filter_violation",
    "linearized_residual_components",
    "log_primal_coordinates",
    "physical_amounts",
    "kkt_component_norms",
    "residual_components",
    "residual_jacobian",
    "residual_vector",
    "scaled_equality_residual",
]
