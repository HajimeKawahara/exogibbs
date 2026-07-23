"""Reduced KKT second-order correction for fixed-support PD-IPM."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def fixed_support_soc_correction_direction(
    *,
    formula_matrix: Any,
    formula_matrix_cond_active: Any,
    gas_amounts: Any,
    condensate_amounts: Any,
    condensate_duals: Any,
    gas_inventory: Any,
    total_density_residual: Any,
    budget_defect: Any,
    total_density_defect: Any,
    max_abs_primal_step: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Solve the reduced KKT system for a pure primal nonlinear defect."""

    n = jnp.asarray(gas_amounts)
    dtype = n.dtype
    m = jnp.asarray(condensate_amounts, dtype=dtype)
    eta = jnp.asarray(condensate_duals, dtype=dtype)
    ag = jnp.asarray(formula_matrix, dtype=dtype)
    ac = jnp.asarray(formula_matrix_cond_active, dtype=dtype)
    gas_inventory_array = jnp.asarray(gas_inventory, dtype=dtype)
    j_vec = m / jnp.maximum(eta, jnp.asarray(1.0e-300, dtype=dtype))
    qhat = ag @ (n[:, None] * ag.T) + ac @ (j_vec[:, None] * ac.T)
    scale = jnp.maximum(
        jnp.mean(jnp.diag(qhat)),
        jnp.asarray(1.0, dtype=dtype),
    )
    qhat_regularized = qhat + jnp.asarray(1.0e-14, dtype=dtype) * scale * jnp.eye(
        qhat.shape[0], dtype=dtype
    )
    reduced_matrix = jnp.block(
        [
            [qhat_regularized, gas_inventory_array[:, None]],
            [
                gas_inventory_array[None, :],
                jnp.reshape(
                    jnp.asarray(total_density_residual, dtype=dtype),
                    (1, 1),
                ),
            ],
        ]
    )
    rhs = -jnp.concatenate(
        [
            jnp.asarray(budget_defect, dtype=dtype),
            jnp.reshape(jnp.asarray(total_density_defect, dtype=dtype), (1,)),
        ]
    )
    solution = jnp.linalg.lstsq(reduced_matrix, rhs, rcond=None)[0]
    solution = jnp.nan_to_num(solution, nan=0.0, posinf=0.0, neginf=0.0)
    delta_lambda = solution[:-1]
    delta_qtot = solution[-1]
    delta_q = ag.T @ delta_lambda + delta_qtot
    delta_rho = -(ac.T @ delta_lambda) / jnp.maximum(
        eta, jnp.asarray(1.0e-300, dtype=dtype)
    )
    delta_r = -delta_rho
    primal_max = jnp.max(
        jnp.abs(jnp.concatenate([delta_q, delta_r, jnp.reshape(delta_qtot, (1,))])),
        initial=jnp.asarray(0.0, dtype=dtype),
    )
    limit = jnp.maximum(jnp.asarray(max_abs_primal_step, dtype=dtype), 0.0)
    step_scale = jnp.minimum(
        1.0,
        limit / jnp.maximum(primal_max, jnp.asarray(1.0e-300, dtype=dtype)),
    )
    return (
        step_scale * delta_q,
        step_scale * delta_r,
        step_scale * delta_lambda,
        step_scale * delta_rho,
        step_scale * delta_qtot,
    )


__all__ = ["fixed_support_soc_correction_direction"]
