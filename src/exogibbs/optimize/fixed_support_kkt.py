"""KKT and filter diagnostics for fixed-support PD-IPM solves."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def fixed_support_filter_theta(
    *,
    formula_matrix: Any,
    formula_matrix_cond_active: Any,
    element_inventory_target: Any,
    q: Any,
    r: Any,
    qtot: Any,
    relative_floor: Any,
    use_l1_norm: Any = False,
) -> Any:
    """Return scaled primal infeasibility for an Ipopt-style filter.

    Only equality constraints belong in ``theta``.  Stationarity and
    complementarity are deliberately excluded.  Element rows are scaled by
    their target inventory and the gas-total row by the current total amount.
    """

    q_array = jnp.asarray(q)
    dtype = q_array.dtype
    r_array = jnp.asarray(r, dtype=dtype)
    ag = jnp.asarray(formula_matrix, dtype=dtype)
    ac = jnp.asarray(formula_matrix_cond_active, dtype=dtype)
    target = jnp.asarray(element_inventory_target, dtype=dtype)
    floor = jnp.maximum(
        jnp.asarray(relative_floor, dtype=dtype),
        jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype),
    )
    n = jnp.exp(q_array)
    m = jnp.exp(r_array)
    ntot = jnp.exp(jnp.asarray(qtot, dtype=dtype))
    budget = ag @ n + ac @ m - target
    budget_scale = jnp.where(
        target > 0.0,
        jnp.maximum(jnp.abs(target), floor),
        jnp.asarray(1.0, dtype=dtype),
    )
    total_scale = jnp.maximum(jnp.abs(ntot), floor)
    scaled_constraints = jnp.concatenate(
        [budget / budget_scale, jnp.asarray([(jnp.sum(n) - ntot) / total_scale])]
    )
    max_norm = jnp.max(
        jnp.abs(scaled_constraints),
        initial=jnp.asarray(0.0, dtype=dtype),
    )
    l1_norm = jnp.sum(jnp.abs(scaled_constraints))
    return jnp.where(jnp.asarray(use_l1_norm, dtype=bool), l1_norm, max_norm)


def fixed_support_barrier_objective(
    *,
    q: Any,
    r: Any,
    qtot: Any,
    gas_stationarity_source: Any,
    condensate_standard_source: Any,
    qtot_reference: Any,
    epsilon: Any,
) -> Any:
    """Return the dimensionless Gibbs log-barrier objective ``phi_mu``."""

    q_array = jnp.asarray(q)
    dtype = q_array.dtype
    r_array = jnp.asarray(r, dtype=dtype)
    qtot_value = jnp.asarray(qtot, dtype=dtype)
    gas_source = jnp.asarray(gas_stationarity_source, dtype=dtype)
    cond_source = jnp.asarray(condensate_standard_source, dtype=dtype)
    mu = jnp.exp(jnp.asarray(epsilon, dtype=dtype))
    n = jnp.exp(q_array)
    m = jnp.exp(r_array)
    gas_mu_over_rt = (
        gas_source
        + jnp.asarray(qtot_reference, dtype=dtype)
        + q_array
        - qtot_value
    )
    return (
        jnp.dot(n, gas_mu_over_rt)
        + jnp.dot(m, cond_source)
        - mu * jnp.sum(r_array)
    )


def fixed_support_barrier_objective_linearized_change(
    *,
    q: Any,
    r: Any,
    qtot: Any,
    gas_stationarity_source: Any,
    condensate_standard_source: Any,
    qtot_reference: Any,
    epsilon: Any,
    delta_q: Any,
    delta_r: Any,
    delta_qtot: Any,
) -> Any:
    """Return ``grad(phi_mu)^T delta`` at the supplied current point."""

    q_array = jnp.asarray(q)
    dtype = q_array.dtype
    r_array = jnp.asarray(r, dtype=dtype)
    qtot_value = jnp.asarray(qtot, dtype=dtype)
    dq = jnp.asarray(delta_q, dtype=dtype)
    dr = jnp.asarray(delta_r, dtype=dtype)
    dqtot = jnp.asarray(delta_qtot, dtype=dtype)
    gas_source = jnp.asarray(gas_stationarity_source, dtype=dtype)
    cond_source = jnp.asarray(condensate_standard_source, dtype=dtype)
    mu = jnp.exp(jnp.asarray(epsilon, dtype=dtype))
    n = jnp.exp(q_array)
    m = jnp.exp(r_array)
    dn = n * dq
    dm = m * dr
    gas_mu_over_rt = (
        gas_source
        + jnp.asarray(qtot_reference, dtype=dtype)
        + q_array
        - qtot_value
    )
    return (
        jnp.dot(gas_mu_over_rt, dn)
        + jnp.dot(n, dq - dqtot)
        + jnp.dot(cond_source, dm)
        - mu * jnp.sum(dr)
    )


def fixed_support_full_newton_linearized_residual(
    *,
    formula_matrix: Any,
    formula_matrix_cond_active: Any,
    q: Any,
    r: Any,
    rho: Any,
    qtot: Any,
    gas_residual: Any,
    condensate_stationarity_residual: Any,
    budget_residual: Any,
    complementarity_residual: Any,
    total_density_residual: Any,
    delta_q: Any,
    delta_r: Any,
    delta_lambda: Any,
    delta_rho: Any,
    delta_qtot: Any,
) -> Any:
    """Return ``F + J_GIE delta`` in full KKT equation order."""

    q_array = jnp.asarray(q)
    dtype = q_array.dtype
    r_array = jnp.asarray(r, dtype=dtype)
    rho_array = jnp.asarray(rho, dtype=dtype)
    qtot_value = jnp.asarray(qtot, dtype=dtype)
    ag = jnp.asarray(formula_matrix, dtype=dtype)
    ac = jnp.asarray(formula_matrix_cond_active, dtype=dtype)
    dq = jnp.asarray(delta_q, dtype=dtype)
    dr = jnp.asarray(delta_r, dtype=dtype)
    dlam = jnp.asarray(delta_lambda, dtype=dtype)
    drho = jnp.asarray(delta_rho, dtype=dtype)
    dqtot = jnp.asarray(delta_qtot, dtype=dtype)
    n = jnp.exp(q_array)
    m = jnp.exp(r_array)
    eta = jnp.exp(rho_array)
    ntot = jnp.exp(qtot_value)
    gas_linearized = (
        jnp.asarray(gas_residual, dtype=dtype) + dq - ag.T @ dlam - dqtot
    )
    cond_linearized = (
        jnp.asarray(condensate_stationarity_residual, dtype=dtype)
        - ac.T @ dlam
        - eta * drho
    )
    budget_linearized = (
        jnp.asarray(budget_residual, dtype=dtype)
        + ag @ (n * dq)
        + ac @ (m * dr)
    )
    complementarity_linearized = (
        jnp.asarray(complementarity_residual, dtype=dtype) + dr + drho
    )
    total_linearized = (
        jnp.asarray(total_density_residual, dtype=dtype)
        + jnp.asarray([jnp.dot(n, dq) - ntot * dqtot], dtype=dtype)
    )
    return jnp.concatenate(
        [
            gas_linearized,
            cond_linearized,
            budget_linearized,
            complementarity_linearized,
            total_linearized,
        ]
    )


__all__ = [
    "fixed_support_barrier_objective",
    "fixed_support_barrier_objective_linearized_change",
    "fixed_support_filter_theta",
    "fixed_support_full_newton_linearized_residual",
]
