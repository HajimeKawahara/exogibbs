"""Dual-step rules for fixed-support primal-dual trial points."""

from __future__ import annotations

import jax.numpy as jnp


def fixed_support_min_dual_infeasibility_step(
    *,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond_active: jnp.ndarray,
    q_trial: jnp.ndarray,
    r_trial: jnp.ndarray,
    lambda_current: jnp.ndarray,
    rho_current: jnp.ndarray,
    delta_lambda: jnp.ndarray,
    delta_rho: jnp.ndarray,
    gas_stationarity_source: jnp.ndarray,
    condensate_standard_source: jnp.ndarray,
    epsilon: jnp.ndarray,
) -> jnp.ndarray:
    """Minimize the linearized dual KKT residual along one dual direction."""

    eta = jnp.exp(rho_current)
    residual = jnp.concatenate(
        [
            q_trial + gas_stationarity_source - formula_matrix.T @ lambda_current,
            condensate_standard_source
            - formula_matrix_cond_active.T @ lambda_current
            - eta,
            r_trial + rho_current - epsilon,
        ]
    )
    derivative = jnp.concatenate(
        [
            -(formula_matrix.T @ delta_lambda),
            -(formula_matrix_cond_active.T @ delta_lambda) - eta * delta_rho,
            delta_rho,
        ]
    )
    denominator = jnp.dot(derivative, derivative)
    alpha = -jnp.dot(residual, derivative) / jnp.maximum(
        denominator, jnp.asarray(1.0e-300, dtype=residual.dtype)
    )
    alpha = jnp.where(denominator > 0.0, alpha, 1.0)
    return jnp.clip(alpha, 0.0, 1.0)


def fixed_support_min_equality_dual_infeasibility_step(
    *,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond_active: jnp.ndarray,
    q_trial: jnp.ndarray,
    rho_trial: jnp.ndarray,
    lambda_current: jnp.ndarray,
    delta_lambda: jnp.ndarray,
    gas_stationarity_source: jnp.ndarray,
    condensate_standard_source: jnp.ndarray,
) -> jnp.ndarray:
    """Choose alpha_y after the bound-dual trial has been fixed."""

    residual = jnp.concatenate(
        [
            q_trial + gas_stationarity_source - formula_matrix.T @ lambda_current,
            condensate_standard_source
            - formula_matrix_cond_active.T @ lambda_current
            - jnp.exp(rho_trial),
        ]
    )
    derivative = jnp.concatenate(
        [
            -(formula_matrix.T @ delta_lambda),
            -(formula_matrix_cond_active.T @ delta_lambda),
        ]
    )
    denominator = jnp.dot(derivative, derivative)
    alpha = -jnp.dot(residual, derivative) / jnp.maximum(
        denominator, jnp.asarray(1.0e-300, dtype=residual.dtype)
    )
    alpha = jnp.where(denominator > 0.0, alpha, 1.0)
    return jnp.clip(alpha, 0.0, 1.0)


__all__ = [
    "fixed_support_min_dual_infeasibility_step",
    "fixed_support_min_equality_dual_infeasibility_step",
]
