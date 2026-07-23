"""Charge-neutral coordinates for the fixed-support R-GIE solver."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax.scipy.special import logsumexp


def retract_fixed_support_charge_neutrality(
    *,
    log_gas_amounts: Any,
    element_potential: Any,
    charge_coefficients: Any,
    iterations: int = 16,
) -> tuple[Any, Any, Any, Any]:
    """Retract ``(q, lambda_e)`` onto neutrality at fixed ``q-z*lambda_e``."""

    q = jnp.asarray(log_gas_amounts)
    dtype = q.dtype
    lam = jnp.asarray(element_potential, dtype=dtype)
    z = jnp.asarray(charge_coefficients, dtype=dtype)
    invariant_log_amount = q - z * lam[-1]
    has_positive = jnp.any(z > 0.0)
    has_negative = jnp.any(z < 0.0)
    solvable = has_positive & has_negative
    positive_log_weight = logsumexp(
        jnp.where(z > 0.0, invariant_log_amount + jnp.log(jnp.abs(z)), -jnp.inf)
    )
    negative_log_weight = logsumexp(
        jnp.where(z < 0.0, invariant_log_amount + jnp.log(jnp.abs(z)), -jnp.inf)
    )
    positive_mean_charge = jnp.sum(jnp.where(z > 0.0, z, 0.0)) / jnp.maximum(
        jnp.sum(z > 0.0), 1
    )
    negative_mean_charge = jnp.sum(jnp.where(z < 0.0, z, 0.0)) / jnp.maximum(
        jnp.sum(z < 0.0), 1
    )
    electron_potential = (negative_log_weight - positive_log_weight) / jnp.maximum(
        positive_mean_charge - negative_mean_charge,
        jnp.asarray(1.0, dtype=dtype),
    )
    electron_potential = jnp.where(solvable, electron_potential, lam[-1])

    for _ in range(iterations):
        trial_log_amount = invariant_log_amount + z * electron_potential
        charged = z != 0.0
        log_scale = jnp.max(
            jnp.where(charged, trial_log_amount, -jnp.inf),
            initial=jnp.asarray(-jnp.inf, dtype=dtype),
        )
        normalized_amount = jnp.where(
            charged,
            jnp.exp(trial_log_amount - log_scale),
            0.0,
        )
        normalized_charge = jnp.sum(z * normalized_amount)
        normalized_susceptibility = jnp.sum(z * z * normalized_amount)
        update = normalized_charge / jnp.maximum(
            normalized_susceptibility,
            jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype),
        )
        electron_potential = jnp.where(
            solvable,
            electron_potential - update,
            electron_potential,
        )

    solved_q = invariant_log_amount + z * electron_potential
    solved_lam = lam.at[-1].set(electron_potential)
    retracted_q = jnp.where(solvable, solved_q, q)
    retracted_lam = jnp.where(solvable, solved_lam, lam)
    final_charged_log_amount = jnp.where(z != 0.0, retracted_q, -jnp.inf)
    final_log_scale = jnp.max(
        final_charged_log_amount,
        initial=jnp.asarray(-jnp.inf, dtype=dtype),
    )
    final_normalized_amount = jnp.where(
        z != 0.0,
        jnp.exp(retracted_q - final_log_scale),
        0.0,
    )
    normalized_charge_residual = jnp.sum(z * final_normalized_amount)
    normalized_susceptibility = jnp.sum(z * z * final_normalized_amount)
    return (
        retracted_q,
        retracted_lam,
        normalized_charge_residual,
        normalized_susceptibility,
    )


__all__ = ["retract_fixed_support_charge_neutrality"]
