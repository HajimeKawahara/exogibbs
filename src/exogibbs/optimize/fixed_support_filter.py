"""GPU-friendly persistent filter operations for fixed-support PD-IPM."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def fixed_support_filter_acceptance(
    *,
    trial_phi: Any,
    trial_theta: Any,
    trial_alpha: Any,
    trial_linearized_change: Any,
    finite: Any,
    current_phi: Any,
    current_theta: Any,
    initial_theta: Any,
    filter_phi: Any,
    filter_theta: Any,
    filter_valid: Any,
    gamma_phi: float = 1.0e-8,
    gamma_theta: float = 1.0e-5,
    theta_max_factor: float = 1.0e4,
    theta_min_factor: float = 1.0e-4,
    eta_phi: float = 1.0e-8,
    switching_delta: float = 1.0,
    switching_s_phi: float = 2.3,
    switching_s_theta: float = 1.1,
) -> tuple[Any, Any, Any, Any]:
    """Return Ipopt-style acceptance, f-type, Armijo, and history masks."""

    phi = jnp.asarray(trial_phi)
    dtype = phi.dtype
    theta = jnp.asarray(trial_theta, dtype=dtype)
    alpha = jnp.asarray(trial_alpha, dtype=dtype)
    linearized_change = jnp.asarray(trial_linearized_change, dtype=dtype)
    current_phi_value = jnp.asarray(current_phi, dtype=dtype)
    current_theta_value = jnp.asarray(current_theta, dtype=dtype)
    initial_theta_value = jnp.asarray(initial_theta, dtype=dtype)
    valid = jnp.asarray(filter_valid, dtype=bool)
    entry_phi = jnp.asarray(filter_phi, dtype=dtype)
    entry_theta = jnp.asarray(filter_theta, dtype=dtype)

    theta_max = jnp.asarray(theta_max_factor, dtype=dtype) * jnp.maximum(
        1.0, initial_theta_value
    )
    theta_min = jnp.asarray(theta_min_factor, dtype=dtype) * jnp.maximum(
        1.0, initial_theta_value
    )
    sufficient_current = (
        theta
        <= (1.0 - jnp.asarray(gamma_theta, dtype=dtype)) * current_theta_value
    ) | (
        phi - current_phi_value
        <= -jnp.asarray(gamma_phi, dtype=dtype) * current_theta_value
    )
    slope = linearized_change / jnp.maximum(
        alpha,
        jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype),
    )
    switching_rule = (slope < 0.0) & (
        alpha * jnp.power(-slope, jnp.asarray(switching_s_phi, dtype=dtype))
        > jnp.asarray(switching_delta, dtype=dtype)
        * jnp.power(current_theta_value, jnp.asarray(switching_s_theta, dtype=dtype))
    )
    f_type = switching_rule & (current_theta_value <= theta_min)
    armijo = (
        phi - current_phi_value
        <= jnp.asarray(eta_phi, dtype=dtype) * linearized_change
    )
    current_acceptable = jnp.where(f_type, armijo, sufficient_current)

    acceptable_per_entry = (phi[:, None] <= entry_phi[None, :]) | (
        theta[:, None] <= entry_theta[None, :]
    )
    history_acceptable = jnp.all(
        acceptable_per_entry | (~valid[None, :]),
        axis=1,
    )
    accepted = (
        jnp.asarray(finite, dtype=bool)
        & (theta <= theta_max)
        & current_acceptable
        & history_acceptable
    )
    return accepted, f_type, armijo, history_acceptable


def update_fixed_support_filter(
    *,
    filter_phi: Any,
    filter_theta: Any,
    filter_valid: Any,
    current_phi: Any,
    current_theta: Any,
    add_entry: Any,
    gamma_phi: float = 1.0e-8,
    gamma_theta: float = 1.0e-5,
) -> tuple[Any, Any, Any]:
    """Add one margin-adjusted entry and remove entries it dominates."""

    phi_entries = jnp.asarray(filter_phi)
    dtype = phi_entries.dtype
    theta_entries = jnp.asarray(filter_theta, dtype=dtype)
    valid = jnp.asarray(filter_valid, dtype=bool)
    phi_new = jnp.asarray(current_phi, dtype=dtype) - jnp.asarray(
        gamma_phi, dtype=dtype
    ) * jnp.asarray(current_theta, dtype=dtype)
    theta_new = (1.0 - jnp.asarray(gamma_theta, dtype=dtype)) * jnp.asarray(
        current_theta, dtype=dtype
    )
    dominated = valid & (phi_new <= phi_entries) & (theta_new <= theta_entries)
    retained_valid = valid & (~dominated)
    insert_index = jnp.argmax(~retained_valid)
    next_phi = phi_entries.at[insert_index].set(phi_new)
    next_theta = theta_entries.at[insert_index].set(theta_new)
    next_valid = retained_valid.at[insert_index].set(True)
    enabled = jnp.asarray(add_entry, dtype=bool)
    return (
        jnp.where(enabled, next_phi, phi_entries),
        jnp.where(enabled, next_theta, theta_entries),
        jnp.where(enabled, next_valid, valid),
    )


def prepare_fixed_support_restoration_filter(
    *,
    filter_phi: Any,
    filter_theta: Any,
    filter_valid: Any,
    current_phi: Any,
    current_theta: Any,
    phase_entered: Any,
    gamma_phi: float = 1.0e-8,
    gamma_theta: float = 1.0e-5,
) -> tuple[Any, Any, Any]:
    """Augment the outer filter with the pre-restoration current iterate."""

    return update_fixed_support_filter(
        filter_phi=filter_phi,
        filter_theta=filter_theta,
        filter_valid=filter_valid,
        current_phi=current_phi,
        current_theta=current_theta,
        add_entry=phase_entered,
        gamma_phi=gamma_phi,
        gamma_theta=gamma_theta,
    )


__all__ = [
    "fixed_support_filter_acceptance",
    "prepare_fixed_support_restoration_filter",
    "update_fixed_support_filter",
]
