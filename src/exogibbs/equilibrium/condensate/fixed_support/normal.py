"""One normal PD-IPM direction and ordered trial evaluation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from exogibbs.equilibrium.condensate.fixed_support.filter import (
    accept_to_current,
    accept_to_history,
)
from exogibbs.equilibrium.condensate.fixed_support.linear_solver import (
    normal_reduced_direction,
)
from exogibbs.equilibrium.condensate.fixed_support.problem import (
    barrier_objective,
    barrier_objective_directional_derivative,
    filter_violation,
    residual_components,
)
from exogibbs.equilibrium.condensate.fixed_support.types import (
    FilterState,
    FixedSupportProblem,
    FixedSupportV2Config,
    NormalConfig,
    NormalStepResult,
    NormalTrialBatch,
    NormalTrialSelection,
    OriginalState,
    TerminalStatus,
    TrialRejectionReason,
)


def ordered_alpha_ladder(
    config: NormalConfig = NormalConfig(),
    *,
    max_line_search_trials: int = 20,
    dtype=jnp.float64,
):
    """Return ``beta**i`` in sequential backtracking order."""

    if not 0.0 < config.backtracking_factor < 1.0:
        raise ValueError("backtracking_factor must be strictly between zero and one.")
    if max_line_search_trials < 1:
        raise ValueError("max_line_search_trials must be positive.")
    indices = jnp.arange(max_line_search_trials, dtype=dtype)
    return jnp.asarray(config.backtracking_factor, dtype=dtype) ** indices


def _trial_states(state, direction, alphas):
    def advance(value, delta):
        return value[None, ...] + alphas.reshape(
            (alphas.shape[0],) + (1,) * jnp.ndim(value)
        ) * delta[None, ...]

    q = jnp.asarray(state.q)
    dtype = q.dtype
    r = jnp.asarray(state.r, dtype=dtype)
    lambda_ = jnp.asarray(state.lambda_, dtype=dtype)
    rho = jnp.asarray(state.rho, dtype=dtype)
    qtot = jnp.asarray(state.qtot, dtype=dtype)
    return OriginalState(
        q=advance(q, jnp.asarray(direction.q, dtype=dtype)),
        r=advance(r, jnp.asarray(direction.r, dtype=dtype)),
        lambda_=advance(
            lambda_, jnp.asarray(direction.lambda_, dtype=dtype)
        ),
        rho=advance(rho, jnp.asarray(direction.rho, dtype=dtype)),
        qtot=qtot + alphas * jnp.asarray(direction.qtot, dtype=dtype),
        epsilon=jnp.broadcast_to(
            jnp.asarray(state.epsilon, dtype=dtype), (alphas.shape[0],)
        ),
        iteration=jnp.broadcast_to(
            jnp.asarray(state.iteration) + 1, (alphas.shape[0],)
        ),
    )


def select_ordered_trial(
    accepted,
    rejection_reasons,
    alphas,
    *,
    direction_status=TerminalStatus.NOT_TERMINATED,
) -> NormalTrialSelection:
    """Interpret parallel masks exactly as sequential backtracking."""

    accepted = jnp.asarray(accepted, dtype=bool)
    reasons = jnp.asarray(rejection_reasons, dtype=jnp.int32)
    alphas = jnp.asarray(alphas)
    any_accepted = jnp.any(accepted)
    first_index = jnp.argmax(accepted).astype(jnp.int32)
    selected_index = jnp.where(any_accepted, first_index, -1)
    rejected_prefix = jnp.arange(accepted.shape[0]) < jnp.where(
        any_accepted, first_index, accepted.shape[0]
    )
    last_index = jnp.where(any_accepted, first_index - 1, accepted.shape[0] - 1)
    has_last = last_index >= 0
    safe_last = jnp.maximum(last_index, 0)
    last_reason = jnp.where(
        has_last,
        reasons[safe_last],
        jnp.asarray(TrialRejectionReason.NONE, dtype=jnp.int32),
    )
    direction_failed = jnp.asarray(direction_status) != int(
        TerminalStatus.NOT_TERMINATED
    )
    status = jnp.where(
        direction_failed,
        jnp.asarray(direction_status, dtype=jnp.int32),
        jnp.where(
            any_accepted,
            jnp.asarray(TerminalStatus.NOT_TERMINATED, dtype=jnp.int32),
            jnp.asarray(
                TerminalStatus.NORMAL_LINE_SEARCH_FAILED, dtype=jnp.int32
            ),
        ),
    )
    return NormalTrialSelection(
        accepted=any_accepted & (~direction_failed),
        selected_index=jnp.where(direction_failed, -1, selected_index),
        selected_alpha=jnp.where(
            any_accepted & (~direction_failed),
            alphas[first_index],
            jnp.asarray(jnp.nan, dtype=alphas.dtype),
        ),
        rejected_prefix=jnp.where(
            direction_failed, jnp.ones_like(rejected_prefix), rejected_prefix
        ),
        last_rejection_reason=last_reason,
        status=status,
    )


def normal_step(
    problem: FixedSupportProblem,
    state: OriginalState,
    filter_state: FilterState,
    *,
    initial_theta,
    config: FixedSupportV2Config = FixedSupportV2Config(),
) -> NormalStepResult:
    """Evaluate one normal Newton step without mutating solver state."""

    current_residual = residual_components(problem, state)
    direction_result = normal_reduced_direction(
        problem,
        state,
        current_residual,
        config.linear_solver,
    )
    alphas = ordered_alpha_ladder(
        config.normal,
        max_line_search_trials=config.limits.max_line_search_trials,
        dtype=jnp.asarray(state.q).dtype,
    )
    trial_states = _trial_states(state, direction_result.direction, alphas)
    phi = jax.vmap(lambda trial: barrier_objective(problem, trial))(trial_states)
    theta = jax.vmap(lambda trial: filter_violation(problem, trial))(trial_states)
    current_phi = barrier_objective(problem, state)
    current_theta = filter_violation(problem, state)
    derivative = barrier_objective_directional_derivative(
        problem, state, direction_result.direction
    )
    linearized_change = alphas * derivative
    current_acceptable, f_type, armijo, within_theta_max = accept_to_current(
        trial_phi=phi,
        trial_theta=theta,
        alphas=alphas,
        linearized_objective_change=linearized_change,
        current_phi=current_phi,
        current_theta=current_theta,
        initial_theta=initial_theta,
        config=config.filter,
    )
    history_acceptable = accept_to_history(phi, theta, filter_state)
    finite = (
        jnp.isfinite(phi)
        & jnp.isfinite(theta)
        & jnp.all(jnp.isfinite(trial_states.q), axis=1)
        & jnp.all(jnp.isfinite(trial_states.r), axis=1)
        & jnp.all(jnp.isfinite(trial_states.lambda_), axis=1)
        & jnp.all(jnp.isfinite(trial_states.rho), axis=1)
        & jnp.all(jnp.isfinite(jnp.exp(trial_states.rho)), axis=1)
        & jnp.isfinite(trial_states.qtot)
    )
    direction_ok = direction_result.status == int(TerminalStatus.NOT_TERMINATED)
    accepted = (
        finite
        & within_theta_max
        & current_acceptable
        & history_acceptable
        & direction_ok
    )
    rejection_reasons = (
        (~finite).astype(jnp.int32) * int(TrialRejectionReason.NONFINITE)
        + (~within_theta_max).astype(jnp.int32)
        * int(TrialRejectionReason.THETA_MAX)
        + (~current_acceptable).astype(jnp.int32)
        * int(TrialRejectionReason.CURRENT_POINT)
        + (~history_acceptable).astype(jnp.int32)
        * int(TrialRejectionReason.FILTER_HISTORY)
    )
    trials = NormalTrialBatch(
        states=trial_states,
        alphas=alphas,
        phi=phi,
        theta=theta,
        linearized_objective_change=linearized_change,
        finite=finite,
        within_theta_max=within_theta_max,
        current_acceptable=current_acceptable,
        history_acceptable=history_acceptable,
        f_type=f_type,
        armijo=armijo,
        accepted=accepted,
        rejection_reasons=rejection_reasons,
    )
    selection = select_ordered_trial(
        accepted,
        rejection_reasons,
        alphas,
        direction_status=direction_result.status,
    )
    return NormalStepResult(
        direction_result=direction_result,
        trials=trials,
        selection=selection,
    )


__all__ = ["normal_step", "ordered_alpha_ladder", "select_ordered_trial"]
