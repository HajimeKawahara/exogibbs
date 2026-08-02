"""Exact Ipopt-control-flow SOC for the fixed-support log-KKT system."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from exogibbs.equilibrium.condensate.fixed_support.filter import (
    accept_to_current,
    accept_to_history,
)
from exogibbs.equilibrium.condensate.fixed_support.linear_solver import (
    reduced_direction_from_rhs,
)
from exogibbs.equilibrium.condensate.fixed_support.problem import (
    barrier_objective,
    barrier_objective_directional_derivative,
    filter_violation,
    physical_amounts,
    residual_components,
)
from exogibbs.equilibrium.condensate.fixed_support.types import (
    FixedSupportProblem,
    FixedSupportV2Config,
    NormalStepResult,
    OriginalState,
    SOCLinearizedResidualNorms,
    SOCStepResult,
    SOCTrialBatch,
    TerminalStatus,
    TrialRejectionReason,
)


class _SOCCarry(NamedTuple):
    trials: SOCTrialBatch
    previous_budget_rhs: object
    previous_total_rhs: object
    trial_state: OriginalState
    trial_budget_residual: object
    trial_total_residual: object
    previous_alpha_soc: object
    previous_theta: object
    active: object


def _validate_config(config: FixedSupportV2Config) -> None:
    if config.soc.max_corrections < 1:
        raise ValueError("max_corrections must be positive.")
    if not 0.0 < config.soc.kappa_soc < 1.0:
        raise ValueError("kappa_soc must be strictly between zero and one.")
    if not 0.0 < config.soc.fraction_to_boundary <= 1.0:
        raise ValueError("fraction_to_boundary must be in (0, 1].")


def _fraction_to_boundary(direction, fraction, dtype):
    bound = jnp.min(
        jnp.where(direction < 0.0, -1.0 / direction, jnp.inf),
        initial=jnp.asarray(jnp.inf, dtype=dtype),
    )
    return jnp.minimum(
        1.0, jnp.asarray(fraction, dtype=dtype) * bound
    )


def _linearized_residual_norms(problem, state, rhs, direction):
    amounts = physical_amounts(state)
    dtype = jnp.asarray(state.q).dtype
    ag = jnp.asarray(problem.gas_formula_matrix, dtype=dtype)
    ac = jnp.asarray(problem.condensate_formula_matrix, dtype=dtype)
    eta = jnp.exp(jnp.asarray(state.rho, dtype=dtype))

    def max_abs(value):
        return jnp.max(jnp.abs(value), initial=0.0)

    return SOCLinearizedResidualNorms(
        gas_stationarity=max_abs(
            rhs.gas_stationarity
            + direction.q
            - ag.T @ direction.lambda_
            - direction.qtot
        ),
        condensate_stationarity=max_abs(
            rhs.condensate_stationarity
            - ac.T @ direction.lambda_
            - eta * direction.rho
        ),
        budget=max_abs(
            rhs.budget
            + ag @ (amounts.gas * direction.q)
            + ac @ (amounts.condensate * direction.r)
        ),
        complementarity=max_abs(
            rhs.complementarity + direction.r + direction.rho
        ),
        total_density=max_abs(
            rhs.total_density
            + jnp.dot(amounts.gas, direction.q)
            - amounts.total_gas * direction.qtot
        ),
    )


def _empty_trials(state, config):
    count = config.soc.max_corrections
    dtype = jnp.asarray(state.q).dtype
    zero = jnp.zeros((count,), dtype=dtype)
    false = jnp.zeros((count,), dtype=bool)
    states = jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(value, (count,) + value.shape), state
    )
    ne = state.lambda_.shape[0]
    norms = SOCLinearizedResidualNorms(zero, zero, zero, zero, zero)
    return SOCTrialBatch(
        states=states,
        attempted=false,
        alpha_test=zero,
        alpha_soc=zero,
        alpha_y=zero,
        alpha_dual=zero,
        budget_rhs=jnp.zeros((count, ne), dtype=dtype),
        total_density_rhs=zero,
        phi=zero,
        theta=zero,
        finite=false,
        current_acceptable=false,
        history_acceptable=false,
        f_type=false,
        armijo=false,
        accepted=false,
        rejection_reasons=jnp.zeros((count,), dtype=jnp.int32),
        kappa_continue=false,
        solve_statuses=jnp.full(
            (count,), int(TerminalStatus.NOT_TERMINATED), dtype=jnp.int32
        ),
        linearized_residual_norms=norms,
    )


def _replace_trial(trials, index, **updates):
    return trials._replace(
        **{
            name: value.at[index].set(updates[name])
            for name, value in trials._asdict().items()
            if name in updates and name not in {"states", "linearized_residual_norms"}
        },
        states=jax.tree_util.tree_map(
            lambda values, replacement: values.at[index].set(replacement),
            trials.states,
            updates["states"],
        ),
        linearized_residual_norms=jax.tree_util.tree_map(
            lambda values, replacement: values.at[index].set(replacement),
            trials.linearized_residual_norms,
            updates["linearized_residual_norms"],
        ),
    )


def exact_soc_step(
    problem: FixedSupportProblem,
    state: OriginalState,
    filter_state,
    normal_step_result: NormalStepResult,
    *,
    initial_theta,
    config: FixedSupportV2Config = FixedSupportV2Config(),
) -> SOCStepResult:
    """Try the ordered method-0 SOC recurrence after a normal line failure."""

    _validate_config(config)
    current_residual = residual_components(problem, state)
    current_theta = filter_violation(problem, state)
    current_phi = barrier_objective(problem, state)
    derivative = barrier_objective_directional_derivative(
        problem, state, normal_step_result.direction_result.direction
    )
    normal_trials = normal_step_result.trials
    eligible_mask = (
        (~normal_trials.accepted)
        & normal_trials.finite
        & (normal_trials.theta >= current_theta)
        & (
            normal_step_result.direction_result.status
            == int(TerminalStatus.NOT_TERMINATED)
        )
    )
    eligible = jnp.asarray(config.soc.enabled) & jnp.any(eligible_mask)
    base_index = jnp.argmax(eligible_mask).astype(jnp.int32)
    base_state = jax.tree_util.tree_map(
        lambda values: values[base_index], normal_trials.states
    )
    base_residual = residual_components(problem, base_state)
    alpha_test = normal_trials.alphas[base_index]
    initial = _SOCCarry(
        trials=_empty_trials(state, config),
        previous_budget_rhs=current_residual.budget,
        previous_total_rhs=jnp.asarray(
            current_residual.total_density
        ).reshape(()),
        trial_state=base_state,
        trial_budget_residual=base_residual.budget,
        trial_total_residual=jnp.asarray(
            base_residual.total_density
        ).reshape(()),
        previous_alpha_soc=alpha_test,
        previous_theta=normal_trials.theta[base_index],
        active=eligible,
    )

    def body(index, carry):
        def take_soc(active_carry):
            dtype = jnp.asarray(state.q).dtype
            budget_rhs = (
                active_carry.trial_budget_residual
                + active_carry.previous_alpha_soc
                * active_carry.previous_budget_rhs
            )
            total_rhs = (
                active_carry.trial_total_residual
                + active_carry.previous_alpha_soc
                * active_carry.previous_total_rhs
            ).reshape(())
            rhs = current_residual._replace(
                budget=budget_rhs, total_density=total_rhs
            )
            direction_result = reduced_direction_from_rhs(
                problem,
                state,
                gas_rhs=rhs.gas_stationarity,
                condensate_rhs=rhs.condensate_stationarity,
                budget_rhs=rhs.budget,
                complementarity_rhs=rhs.complementarity,
                total_density_rhs=rhs.total_density,
                config=config.linear_solver,
                failure_status=TerminalStatus.SOC_LINEAR_SOLVE_FAILED,
            )
            direction = direction_result.direction
            alpha_soc = _fraction_to_boundary(
                direction.r, config.soc.fraction_to_boundary, dtype
            )
            alpha_dual = _fraction_to_boundary(
                direction.rho, config.soc.fraction_to_boundary, dtype
            )
            alpha_y = alpha_soc
            trial = OriginalState(
                q=state.q + alpha_soc * direction.q,
                r=state.r + alpha_soc * direction.r,
                lambda_=state.lambda_ + alpha_y * direction.lambda_,
                rho=state.rho + alpha_dual * direction.rho,
                qtot=state.qtot + alpha_soc * direction.qtot,
                epsilon=state.epsilon,
                iteration=state.iteration + 1,
            )
            phi = barrier_objective(problem, trial)
            theta = filter_violation(problem, trial)
            current_ok, f_type, armijo, within_theta = accept_to_current(
                trial_phi=phi[None],
                trial_theta=theta[None],
                alphas=alpha_test[None],
                linearized_objective_change=(alpha_test * derivative)[None],
                current_phi=current_phi,
                current_theta=current_theta,
                initial_theta=initial_theta,
                config=config.filter,
            )
            history_ok = accept_to_history(
                phi[None], theta[None], filter_state
            )[0]
            solve_ok = direction_result.status == int(
                TerminalStatus.NOT_TERMINATED
            )
            finite = (
                solve_ok
                & jnp.isfinite(phi)
                & jnp.isfinite(theta)
                & jnp.all(jnp.isfinite(trial.q))
                & jnp.all(jnp.isfinite(trial.r))
                & jnp.all(jnp.isfinite(trial.lambda_))
                & jnp.all(jnp.isfinite(trial.rho))
                & jnp.all(jnp.isfinite(jnp.exp(trial.rho)))
                & jnp.isfinite(trial.qtot)
            )
            accepted = finite & within_theta[0] & current_ok[0] & history_ok
            reasons = (
                (~finite).astype(jnp.int32)
                * int(TrialRejectionReason.NONFINITE)
                + (~within_theta[0]).astype(jnp.int32)
                * int(TrialRejectionReason.THETA_MAX)
                + (~current_ok[0]).astype(jnp.int32)
                * int(TrialRejectionReason.CURRENT_POINT)
                + (~history_ok).astype(jnp.int32)
                * int(TrialRejectionReason.FILTER_HISTORY)
            )
            continue_soc = (
                (~accepted)
                & finite
                & (theta <= config.soc.kappa_soc * active_carry.previous_theta)
            )
            linearized = _linearized_residual_norms(
                problem, state, rhs, direction
            )
            next_trials = _replace_trial(
                active_carry.trials,
                index,
                states=trial,
                attempted=jnp.asarray(True),
                alpha_test=alpha_test,
                alpha_soc=alpha_soc,
                alpha_y=alpha_y,
                alpha_dual=alpha_dual,
                budget_rhs=budget_rhs,
                total_density_rhs=total_rhs,
                phi=phi,
                theta=theta,
                finite=finite,
                current_acceptable=current_ok[0],
                history_acceptable=history_ok,
                f_type=f_type[0],
                armijo=armijo[0],
                accepted=accepted,
                rejection_reasons=reasons,
                kappa_continue=continue_soc,
                solve_statuses=direction_result.status,
                linearized_residual_norms=linearized,
            )
            trial_residual = residual_components(problem, trial)
            return _SOCCarry(
                trials=next_trials,
                previous_budget_rhs=budget_rhs,
                previous_total_rhs=total_rhs,
                trial_state=trial,
                trial_budget_residual=trial_residual.budget,
                trial_total_residual=jnp.asarray(
                    trial_residual.total_density
                ).reshape(()),
                previous_alpha_soc=alpha_soc,
                previous_theta=theta,
                active=continue_soc,
            )

        return jax.lax.cond(
            carry.active, take_soc, lambda inactive: inactive, carry
        )

    final = jax.lax.fori_loop(0, config.soc.max_corrections, body, initial)
    accepted = jnp.any(final.trials.accepted)
    selected_index = jnp.argmax(final.trials.accepted).astype(jnp.int32)
    candidate = jax.tree_util.tree_map(
        lambda values: values[selected_index], final.trials.states
    )
    selected_state = jax.tree_util.tree_map(
        lambda selected, current: jnp.where(accepted, selected, current),
        candidate,
        state,
    )
    return SOCStepResult(
        eligible=eligible,
        accepted=accepted,
        base_trial_index=base_index,
        selected_index=jnp.where(accepted, selected_index, -1),
        selected_state=selected_state,
        correction_count=jnp.sum(final.trials.attempted, dtype=jnp.int32),
        trials=final.trials,
    )


__all__ = ["exact_soc_step"]
