"""Outer barrier continuation for the fixed-support v2 controller."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from exogibbs.optimize.fixed_support_v2.controller import (
    controller_step,
    initialize_controller,
)
from exogibbs.optimize.fixed_support_v2.types import (
    ContinuationState,
    FixedSupportProblem,
    FixedSupportV2Config,
    OriginalState,
    SolverMode,
    TerminalStatus,
)


def _validate_schedule(config: FixedSupportV2Config) -> None:
    schedule = config.continuation.epsilon_schedule
    if not schedule:
        raise ValueError("epsilon_schedule must contain at least one stage.")
    if not all(math.isfinite(value) for value in schedule):
        raise ValueError("epsilon_schedule values must be finite.")
    if any(next_value >= value for value, next_value in zip(schedule, schedule[1:])):
        raise ValueError("epsilon_schedule must be strictly decreasing.")
    if config.continuation.initial_state_policy not in {"center", "provided"}:
        raise ValueError(
            "initial_state_policy must be either 'center' or 'provided'."
        )


def recenter_for_epsilon(state: OriginalState, epsilon) -> OriginalState:
    """Transfer a warm start and center bound multipliers at a new barrier."""

    dtype = jnp.asarray(state.q).dtype
    next_epsilon = jnp.asarray(epsilon, dtype=dtype)
    return state._replace(
        rho=next_epsilon - jnp.asarray(state.r, dtype=dtype),
        epsilon=next_epsilon,
        iteration=jnp.asarray(0, dtype=jnp.int32),
    )


def initialize_continuation(
    problem: FixedSupportProblem,
    original_state: OriginalState,
    config: FixedSupportV2Config = FixedSupportV2Config(),
) -> ContinuationState:
    """Create the first continuation stage under the configured init policy.

    ``center`` is the normal solver policy and enforces complementarity at the
    first schedule value. ``provided`` is an audit policy: it preserves the
    supplied ``q, r, lambda, rho, qtot, epsilon`` tuple exactly.
    """

    _validate_schedule(config)
    schedule = config.continuation.epsilon_schedule
    if config.continuation.initial_state_policy == "center":
        initial_state = recenter_for_epsilon(original_state, schedule[0])
    else:
        initial_state = original_state._replace(
            iteration=jnp.asarray(0, dtype=jnp.int32)
        )
    controller = initialize_controller(problem, initial_state, config)
    stage_count = len(schedule)
    stage_last_return_diagnostics = jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(
            value, (stage_count,) + jnp.asarray(value).shape
        ),
        controller.last_return_diagnostics,
    )
    return ContinuationState(
        controller=controller,
        stage_index=jnp.asarray(0, dtype=jnp.int32),
        completed_stage_count=jnp.asarray(0, dtype=jnp.int32),
        stage_statuses=jnp.full(
            (stage_count,),
            int(TerminalStatus.NOT_TERMINATED),
            dtype=jnp.int32,
        ),
        stage_normal_iteration_counts=jnp.zeros(
            (stage_count,), dtype=jnp.int32
        ),
        stage_restoration_call_counts=jnp.zeros(
            (stage_count,), dtype=jnp.int32
        ),
        stage_restoration_accepted_iteration_counts=jnp.zeros(
            (stage_count,), dtype=jnp.int32
        ),
        stage_last_return_diagnostics=stage_last_return_diagnostics,
        stage_soc_attempt_counts=jnp.zeros((stage_count,), dtype=jnp.int32),
        stage_soc_accepted_counts=jnp.zeros((stage_count,), dtype=jnp.int32),
        terminal_status=jnp.asarray(
            TerminalStatus.NOT_TERMINATED, dtype=jnp.int32
        ),
    )


def _record_current_stage(state: ContinuationState) -> ContinuationState:
    index = state.stage_index
    controller = state.controller
    return state._replace(
        stage_statuses=state.stage_statuses.at[index].set(
            controller.terminal_status
        ),
        stage_normal_iteration_counts=(
            state.stage_normal_iteration_counts.at[index].set(
                controller.normal_iteration_count
            )
        ),
        stage_restoration_call_counts=(
            state.stage_restoration_call_counts.at[index].set(
                controller.restoration_call_count
            )
        ),
        stage_restoration_accepted_iteration_counts=(
            state.stage_restoration_accepted_iteration_counts.at[index].set(
                controller.restoration_accepted_iteration_count
            )
        ),
        stage_last_return_diagnostics=jax.tree_util.tree_map(
            lambda stages, value: stages.at[index].set(value),
            state.stage_last_return_diagnostics,
            controller.last_return_diagnostics,
        ),
        stage_soc_attempt_counts=state.stage_soc_attempt_counts.at[index].set(
            controller.soc_attempt_count
        ),
        stage_soc_accepted_counts=(
            state.stage_soc_accepted_counts.at[index].set(
                controller.soc_accepted_count
            )
        ),
    )


def _advance_converged_stage(problem, state, config):
    recorded = _record_current_stage(state)
    next_completed = recorded.completed_stage_count + 1
    last_stage = recorded.stage_index == len(
        config.continuation.epsilon_schedule
    ) - 1

    def finish(_operand):
        return recorded._replace(
            completed_stage_count=next_completed,
            terminal_status=jnp.asarray(
                TerminalStatus.CONVERGED, dtype=jnp.int32
            ),
        )

    def start_next(_operand):
        next_index = recorded.stage_index + 1
        schedule = jnp.asarray(
            config.continuation.epsilon_schedule,
            dtype=recorded.controller.original_state.q.dtype,
        )
        warm_start = recenter_for_epsilon(
            recorded.controller.original_state, schedule[next_index]
        )
        return recorded._replace(
            controller=initialize_controller(problem, warm_start, config),
            stage_index=next_index,
            completed_stage_count=next_completed,
            terminal_status=jnp.asarray(
                TerminalStatus.NOT_TERMINATED, dtype=jnp.int32
            ),
        )

    return jax.lax.cond(last_stage, finish, start_next, operand=None)


def _finish_failed_stage(state: ContinuationState) -> ContinuationState:
    recorded = _record_current_stage(state)
    return recorded._replace(
        terminal_status=recorded.controller.terminal_status
    )


def continuation_step(
    problem: FixedSupportProblem,
    state: ContinuationState,
    config: FixedSupportV2Config = FixedSupportV2Config(),
) -> ContinuationState:
    """Advance one fixed-epsilon or outer-stage super-iteration."""

    _validate_schedule(config)

    def active(current):
        return current._replace(
            controller=controller_step(problem, current.controller, config)
        )

    branches = (
        active,
        active,
        lambda current: _advance_converged_stage(problem, current, config),
        _finish_failed_stage,
    )
    return jax.lax.cond(
        state.terminal_status == int(TerminalStatus.NOT_TERMINATED),
        lambda current: jax.lax.switch(
            current.controller.mode, branches, current
        ),
        lambda current: current,
        state,
    )


def solve_continuation(
    problem: FixedSupportProblem,
    initial_state: ContinuationState,
    config: FixedSupportV2Config = FixedSupportV2Config(),
) -> ContinuationState:
    """Solve every epsilon stage, stopping at the first typed failure."""

    _validate_schedule(config)
    return jax.lax.while_loop(
        lambda state: state.terminal_status
        == int(TerminalStatus.NOT_TERMINATED),
        lambda state: continuation_step(problem, state, config),
        initial_state,
    )


__all__ = [
    "continuation_step",
    "initialize_continuation",
    "recenter_for_epsilon",
    "solve_continuation",
]
