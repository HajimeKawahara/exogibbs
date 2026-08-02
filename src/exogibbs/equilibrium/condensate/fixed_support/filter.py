"""Persistent original-filter operations with sequential reset semantics."""

from __future__ import annotations

import jax.numpy as jnp

from exogibbs.equilibrium.condensate.fixed_support.types import (
    FilterConfig,
    FilterState,
    FilterUpdateResult,
)


def empty_filter(capacity: int, dtype=jnp.float64) -> FilterState:
    """Create an empty fixed-capacity filter."""

    return FilterState(
        phi_entries=jnp.zeros((capacity,), dtype=dtype),
        theta_entries=jnp.zeros((capacity,), dtype=dtype),
        valid_entries=jnp.zeros((capacity,), dtype=bool),
        successive_filter_rejections=jnp.asarray(0, dtype=jnp.int32),
        reset_count=jnp.asarray(0, dtype=jnp.int32),
    )


def accept_to_current(
    *,
    trial_phi,
    trial_theta,
    alphas,
    linearized_objective_change,
    current_phi,
    current_theta,
    initial_theta,
    config: FilterConfig = FilterConfig(),
):
    """Return current-point acceptance and its component masks."""

    phi = jnp.asarray(trial_phi)
    dtype = phi.dtype
    theta = jnp.asarray(trial_theta, dtype=dtype)
    alpha = jnp.asarray(alphas, dtype=dtype)
    change = jnp.asarray(linearized_objective_change, dtype=dtype)
    current_phi = jnp.asarray(current_phi, dtype=dtype)
    current_theta = jnp.asarray(current_theta, dtype=dtype)
    initial_theta = jnp.asarray(initial_theta, dtype=dtype)
    theta_max = config.theta_max_factor * jnp.maximum(1.0, initial_theta)
    theta_min = config.theta_min_factor * jnp.maximum(1.0, initial_theta)
    sufficient_current = (
        theta <= (1.0 - config.gamma_theta) * current_theta
    ) | (phi - current_phi <= -config.gamma_phi * current_theta)
    slope = change / jnp.maximum(
        alpha, jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    )
    switching = (slope < 0.0) & (
        alpha * (-slope) ** config.switching_s_phi
        > config.switching_delta * current_theta**config.switching_s_theta
    )
    f_type = switching & (current_theta <= theta_min)
    roundoff_tolerance = (
        jnp.asarray(config.roundoff_tolerance_factor, dtype=dtype)
        * jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
        * jnp.maximum(1.0, jnp.maximum(jnp.abs(phi), jnp.abs(current_phi)))
    )
    armijo = (
        phi - current_phi
        <= config.eta_phi * change + roundoff_tolerance
    )
    current_acceptable = jnp.where(f_type, armijo, sufficient_current)
    return current_acceptable, f_type, armijo, theta <= theta_max


def accept_to_history(
    trial_phi,
    trial_theta,
    filter_state: FilterState,
):
    """Return componentwise acceptability against every valid filter entry."""

    phi = jnp.asarray(trial_phi)
    dtype = phi.dtype
    theta = jnp.asarray(trial_theta, dtype=dtype)
    acceptable = (phi[:, None] <= filter_state.phi_entries[None, :]) | (
        theta[:, None] <= filter_state.theta_entries[None, :]
    )
    return jnp.all(acceptable | (~filter_state.valid_entries[None, :]), axis=1)


def add_margin_adjusted_entry(
    filter_state: FilterState,
    *,
    phi,
    theta,
    add_entry=True,
    config: FilterConfig = FilterConfig(),
) -> FilterUpdateResult:
    """Add one entry, reporting rather than overwriting a full filter."""

    entries_phi = jnp.asarray(filter_state.phi_entries)
    dtype = entries_phi.dtype
    entries_theta = jnp.asarray(filter_state.theta_entries, dtype=dtype)
    valid = jnp.asarray(filter_state.valid_entries, dtype=bool)
    new_phi = jnp.asarray(phi, dtype=dtype) - config.gamma_phi * jnp.asarray(
        theta, dtype=dtype
    )
    new_theta = (1.0 - config.gamma_theta) * jnp.asarray(theta, dtype=dtype)
    dominated = valid & (new_phi <= entries_phi) & (new_theta <= entries_theta)
    retained = valid & (~dominated)
    has_slot = jnp.any(~retained)
    insert_index = jnp.argmax(~retained)
    enabled = jnp.asarray(add_entry, dtype=bool)
    do_insert = enabled & has_slot
    candidate_phi = entries_phi.at[insert_index].set(new_phi)
    candidate_theta = entries_theta.at[insert_index].set(new_theta)
    candidate_valid = retained.at[insert_index].set(True)
    next_state = FilterState(
        phi_entries=jnp.where(do_insert, candidate_phi, entries_phi),
        theta_entries=jnp.where(do_insert, candidate_theta, entries_theta),
        valid_entries=jnp.where(do_insert, candidate_valid, valid),
        successive_filter_rejections=filter_state.successive_filter_rejections,
        reset_count=filter_state.reset_count,
    )
    return FilterUpdateResult(
        state=next_state,
        capacity_exhausted=enabled & (~has_slot),
    )


def reset_from_sequential_rejection_history(
    filter_state: FilterState,
    *,
    step_accepted,
    last_rejection_was_history,
    config: FilterConfig = FilterConfig(),
) -> FilterState:
    """Update reset counters from only the last sequential rejection."""

    increment = jnp.asarray(step_accepted, dtype=bool) & jnp.asarray(
        last_rejection_was_history, dtype=bool
    )
    successive = jnp.where(
        increment,
        filter_state.successive_filter_rejections + 1,
        jnp.asarray(0, dtype=jnp.int32),
    )
    reset = (
        jnp.asarray(step_accepted, dtype=bool)
        & (successive >= config.reset_trigger)
        & (filter_state.reset_count < config.max_resets)
    )
    return FilterState(
        phi_entries=jnp.where(
            reset, jnp.zeros_like(filter_state.phi_entries), filter_state.phi_entries
        ),
        theta_entries=jnp.where(
            reset,
            jnp.zeros_like(filter_state.theta_entries),
            filter_state.theta_entries,
        ),
        valid_entries=jnp.where(
            reset,
            jnp.zeros_like(filter_state.valid_entries),
            filter_state.valid_entries,
        ),
        successive_filter_rejections=jnp.where(
            reset, jnp.asarray(0, dtype=jnp.int32), successive
        ),
        reset_count=filter_state.reset_count + reset.astype(jnp.int32),
    )


__all__ = [
    "accept_to_current",
    "accept_to_history",
    "add_margin_adjusted_entry",
    "empty_filter",
    "reset_from_sequential_rejection_history",
]
