"""Convergence helpers for experimental fixed-support condensate solves."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def fixed_support_budget_relative_max(
    *,
    budget_residual: Any,
    target: Any,
    relative_floor: Any,
) -> Any:
    """Return max element-budget residual relative to a floored target scale."""

    budget = jnp.asarray(budget_residual)
    target_array = jnp.asarray(target, dtype=budget.dtype)
    floor = jnp.asarray(relative_floor, dtype=budget.dtype)
    scale = jnp.where(
        target_array > 0.0,
        jnp.maximum(jnp.abs(target_array), floor),
        jnp.asarray(1.0, dtype=budget.dtype),
    )
    relative = jnp.abs(budget) / scale
    return jnp.max(relative, initial=jnp.asarray(0.0, dtype=budget.dtype))


def fixed_support_batch_converged(
    *,
    gas_norm: Any,
    condensate_stationarity_norm: Any,
    complementarity_norm: Any,
    total_density_norm: Any,
    budget_relative_max: Any,
    log_tolerance: Any,
    budget_relative_tolerance: Any,
    total_density_tolerance: Any,
) -> Any:
    """Return a FastChem-style component-wise convergence mask.

    FastChem4 primarily uses log-space update tolerances for chemistry and
    condensate iterations, and a separate relative element-conservation
    tolerance.  The fixed-support diagnostic residuals are not updates, but the
    stationarity and log-complementarity blocks live in the same log-scale
    units.  Keep those tolerances independent of the barrier parameter.
    """

    log_tol = jnp.asarray(log_tolerance)
    budget_tol = jnp.asarray(budget_relative_tolerance)
    total_tol = jnp.asarray(total_density_tolerance)
    return (
        (jnp.asarray(gas_norm) <= log_tol)
        & (jnp.asarray(condensate_stationarity_norm) <= log_tol)
        & (jnp.asarray(complementarity_norm) <= log_tol)
        & (jnp.asarray(total_density_norm) <= total_tol)
        & (jnp.asarray(budget_relative_max) <= budget_tol)
    )


__all__ = [
    "fixed_support_batch_converged",
    "fixed_support_budget_relative_max",
]
