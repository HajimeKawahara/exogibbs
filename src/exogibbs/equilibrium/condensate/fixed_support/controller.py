"""Fixed-epsilon controller for normal and persistent restoration phases."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from exogibbs.equilibrium.condensate.fixed_support.filter import (
    add_margin_adjusted_entry,
    empty_filter,
    reset_from_sequential_rejection_history,
)
from exogibbs.equilibrium.condensate.fixed_support.normal import normal_step
from exogibbs.equilibrium.condensate.fixed_support.problem import (
    barrier_objective,
    filter_violation,
    kkt_component_norms,
)
from exogibbs.equilibrium.condensate.fixed_support.restoration import (
    initialize_restoration,
    restoration_advance,
)
from exogibbs.equilibrium.condensate.fixed_support.return_map import (
    apply_restoration_return,
)
from exogibbs.equilibrium.condensate.fixed_support.soc import exact_soc_step
from exogibbs.equilibrium.condensate.fixed_support.types import (
    ControllerState,
    FilterState,
    FixedSupportProblem,
    FixedSupportV2Config,
    KKTComponentNorms,
    OriginalState,
    RestorationReturnDiagnostics,
    RestorationState,
    SolverMode,
    TerminalStatus,
    TrialRejectionReason,
)


def _zero_kkt_norms(dtype) -> KKTComponentNorms:
    zero = jnp.asarray(0.0, dtype=dtype)
    return KKTComponentNorms(zero, zero, zero, zero, zero)


def _zero_return_diagnostics(dtype) -> RestorationReturnDiagnostics:
    zero = jnp.asarray(0.0, dtype=dtype)
    false = jnp.asarray(False)
    norms = _zero_kkt_norms(dtype)
    return RestorationReturnDiagnostics(
        alpha_dual=zero,
        bound_multiplier_reset=false,
        equality_multiplier_reset=false,
        representation_floor_applied=false,
        scaled_budget_injection_max=zero,
        scaled_total_density_injection=zero,
        pre_return_norms=norms,
        post_return_norms=norms,
    )


def _empty_restoration_state(
    problem: FixedSupportProblem,
    original_state: OriginalState,
) -> RestorationState:
    """Allocate an inactive fixed-shape placeholder without initializing an NLP."""

    dtype = jnp.asarray(original_state.q).dtype
    ng = original_state.q.shape[0]
    nc = original_state.r.shape[0]
    ne = problem.target_inventory.shape[0]
    x = jnp.zeros((ng + nc + 1,), dtype=dtype)
    equality = jnp.zeros((ne + 1,), dtype=dtype)
    return RestorationState(
        x=x,
        positive_slack=equality,
        negative_slack=equality,
        equality_dual=equality,
        lower_bound_dual_x=x,
        lower_bound_dual_positive=equality,
        lower_bound_dual_negative=equality,
        restoration_mu=jnp.asarray(0.0, dtype=dtype),
        entry_x=x,
        entry_original_state=original_state,
        entry_phi=jnp.asarray(0.0, dtype=dtype),
        entry_theta=jnp.asarray(0.0, dtype=dtype),
        variable_scales=jnp.ones_like(x),
        proximity_mask=jnp.ones_like(x, dtype=jnp.bool_),
        row_scales=jnp.ones_like(equality),
        iteration=jnp.asarray(0, dtype=jnp.int32),
        accepted_iteration_count=jnp.asarray(0, dtype=jnp.int32),
    )


def initialize_controller(
    problem: FixedSupportProblem,
    original_state: OriginalState,
    config: FixedSupportV2Config = FixedSupportV2Config(),
    *,
    filter_state: Optional[FilterState] = None,
) -> ControllerState:
    """Create one fixed-epsilon controller in NORMAL mode."""

    dtype = jnp.asarray(original_state.q).dtype
    active_filter = (
        empty_filter(config.limits.filter_capacity, dtype=dtype)
        if filter_state is None
        else filter_state
    )
    return ControllerState(
        mode=jnp.asarray(SolverMode.NORMAL, dtype=jnp.int32),
        original_state=original_state,
        filter_state=active_filter,
        restoration_state=_empty_restoration_state(problem, original_state),
        initial_theta=filter_violation(problem, original_state),
        normal_iteration_count=jnp.asarray(0, dtype=jnp.int32),
        restoration_call_count=jnp.asarray(0, dtype=jnp.int32),
        restoration_accepted_iteration_count=jnp.asarray(
            0, dtype=jnp.int32
        ),
        soc_attempt_count=jnp.asarray(0, dtype=jnp.int32),
        soc_accepted_count=jnp.asarray(0, dtype=jnp.int32),
        terminal_status=jnp.asarray(
            TerminalStatus.NOT_TERMINATED, dtype=jnp.int32
        ),
        last_return_diagnostics=_zero_return_diagnostics(dtype),
    )


def original_converged(
    problem: FixedSupportProblem,
    state: OriginalState,
    config: FixedSupportV2Config = FixedSupportV2Config(),
):
    """Check every original KKT component independently."""

    norms = kkt_component_norms(problem, state)
    return (
        (norms.gas_stationarity <= config.normal.stationarity_tolerance)
        & (
            norms.condensate_stationarity
            <= config.normal.stationarity_tolerance
        )
        & (norms.budget_scaled <= config.normal.budget_tolerance)
        & (
            norms.complementarity
            <= config.normal.complementarity_tolerance
        )
        & (
            norms.total_density_scaled
            <= config.normal.total_density_tolerance
        )
    )


def _replace_controller(state: ControllerState, **updates) -> ControllerState:
    return state._replace(**updates)


def _normal_controller_step(problem, controller, config):
    converged = original_converged(problem, controller.original_state, config)
    maxed = (
        controller.normal_iteration_count >= config.limits.max_normal_iterations
    )

    def finish_converged(_operand):
        return _replace_controller(
            controller,
            mode=jnp.asarray(SolverMode.CONVERGED, dtype=jnp.int32),
            terminal_status=jnp.asarray(
                TerminalStatus.CONVERGED, dtype=jnp.int32
            ),
        )

    def finish_maxed(_operand):
        return _replace_controller(
            controller,
            mode=jnp.asarray(SolverMode.FAILED, dtype=jnp.int32),
            terminal_status=jnp.asarray(
                TerminalStatus.NORMAL_MAX_ITER, dtype=jnp.int32
            ),
        )

    def attempt_normal(_operand):
        step = normal_step(
            problem,
            controller.original_state,
            controller.filter_state,
            initial_theta=controller.initial_theta,
            config=config,
        )
        next_normal_count = controller.normal_iteration_count + 1

        def accept_normal(_accepted_operand):
            index = step.selection.selected_index
            selected_state = jax.tree_util.tree_map(
                lambda values: values[index], step.trials.states
            )
            current_phi = barrier_objective(problem, controller.original_state)
            current_theta = filter_violation(problem, controller.original_state)
            selected_h_type = ~step.trials.f_type[index]
            added = add_margin_adjusted_entry(
                controller.filter_state,
                phi=current_phi,
                theta=current_theta,
                add_entry=selected_h_type,
                config=config.filter,
            )
            last_history_rejection = (
                step.selection.last_rejection_reason
                & int(TrialRejectionReason.FILTER_HISTORY)
            ) != 0
            next_filter = reset_from_sequential_rejection_history(
                added.state,
                step_accepted=True,
                last_rejection_was_history=last_history_rejection,
                config=config.filter,
            )
            return _replace_controller(
                controller,
                mode=jnp.where(
                    added.capacity_exhausted,
                    int(SolverMode.FAILED),
                    int(SolverMode.NORMAL),
                ).astype(jnp.int32),
                original_state=jax.tree_util.tree_map(
                    lambda selected, current: jnp.where(
                        added.capacity_exhausted, current, selected
                    ),
                    selected_state,
                    controller.original_state,
                ),
                filter_state=next_filter,
                normal_iteration_count=next_normal_count,
                terminal_status=jnp.where(
                    added.capacity_exhausted,
                    int(TerminalStatus.INTERNAL_CONTRACT_ERROR),
                    int(TerminalStatus.NOT_TERMINATED),
                ).astype(jnp.int32),
            )

        def reject_normal(_rejected_operand):
            line_search_failed = step.selection.status == int(
                TerminalStatus.NORMAL_LINE_SEARCH_FAILED
            )
            current_theta = filter_violation(problem, controller.original_state)
            soc = exact_soc_step(
                problem,
                controller.original_state,
                controller.filter_state,
                step,
                initial_theta=controller.initial_theta,
                config=config,
            )

            def accept_soc(_soc_accepted_operand):
                index = soc.selected_index
                current_phi = barrier_objective(
                    problem, controller.original_state
                )
                selected_h_type = ~soc.trials.f_type[index]
                added = add_margin_adjusted_entry(
                    controller.filter_state,
                    phi=current_phi,
                    theta=current_theta,
                    add_entry=selected_h_type,
                    config=config.filter,
                )
                previous_soc_reason = soc.trials.rejection_reasons[
                    jnp.maximum(index - 1, 0)
                ]
                base_reason = step.trials.rejection_reasons[
                    soc.base_trial_index
                ]
                last_reason = jnp.where(
                    index > 0, previous_soc_reason, base_reason
                )
                next_filter = reset_from_sequential_rejection_history(
                    added.state,
                    step_accepted=True,
                    last_rejection_was_history=(
                        last_reason & int(TrialRejectionReason.FILTER_HISTORY)
                    )
                    != 0,
                    config=config.filter,
                )
                return _replace_controller(
                    controller,
                    mode=jnp.where(
                        added.capacity_exhausted,
                        int(SolverMode.FAILED),
                        int(SolverMode.NORMAL),
                    ).astype(jnp.int32),
                    original_state=jax.tree_util.tree_map(
                        lambda selected, current: jnp.where(
                            added.capacity_exhausted, current, selected
                        ),
                        soc.selected_state,
                        controller.original_state,
                    ),
                    filter_state=next_filter,
                    normal_iteration_count=next_normal_count,
                    soc_attempt_count=controller.soc_attempt_count + 1,
                    soc_accepted_count=controller.soc_accepted_count + 1,
                    terminal_status=jnp.where(
                        added.capacity_exhausted,
                        int(TerminalStatus.INTERNAL_CONTRACT_ERROR),
                        int(TerminalStatus.NOT_TERMINATED),
                    ).astype(jnp.int32),
                )

            def reject_soc(_soc_rejected_operand):
                material = (
                    current_theta
                    > config.normal.restoration_entry_theta_threshold
                )
                calls_available = (
                    controller.restoration_call_count
                    < config.limits.max_restoration_calls
                )
                enter = line_search_failed & material & calls_available
                current_phi = barrier_objective(
                    problem, controller.original_state
                )
                prepared = add_margin_adjusted_entry(
                    controller.filter_state,
                    phi=current_phi,
                    theta=current_theta,
                    add_entry=enter,
                    config=config.filter,
                )
                failed_status = jnp.where(
                    ~line_search_failed,
                    step.selection.status,
                    jnp.where(
                        ~material,
                        int(TerminalStatus.NORMAL_DUAL_STEP_FAILED),
                        int(TerminalStatus.RESTORATION_MAX_CALLS),
                    ),
                ).astype(jnp.int32)
                final_enter = enter & (~prepared.capacity_exhausted)
                initialized_restoration = jax.lax.cond(
                    final_enter,
                    lambda _operand: initialize_restoration(
                        problem, controller.original_state, config
                    ),
                    lambda _operand: controller.restoration_state,
                    operand=None,
                )
                status = jnp.where(
                    prepared.capacity_exhausted,
                    int(TerminalStatus.INTERNAL_CONTRACT_ERROR),
                    jnp.where(
                        final_enter,
                        int(TerminalStatus.NOT_TERMINATED),
                        failed_status,
                    ),
                ).astype(jnp.int32)
                return _replace_controller(
                    controller,
                    mode=jnp.where(
                        final_enter,
                        int(SolverMode.RESTORATION),
                        int(SolverMode.FAILED),
                    ).astype(jnp.int32),
                    filter_state=prepared.state,
                    restoration_state=initialized_restoration,
                    normal_iteration_count=next_normal_count,
                    restoration_call_count=(
                        controller.restoration_call_count
                        + final_enter.astype(jnp.int32)
                    ),
                    soc_attempt_count=(
                        controller.soc_attempt_count
                        + soc.eligible.astype(jnp.int32)
                    ),
                    terminal_status=status,
                )

            return jax.lax.cond(
                soc.accepted, accept_soc, reject_soc, operand=None
            )

        return jax.lax.cond(
            step.selection.accepted,
            accept_normal,
            reject_normal,
            operand=None,
        )

    return jax.lax.cond(
        converged,
        finish_converged,
        lambda _operand: jax.lax.cond(
            maxed, finish_maxed, attempt_normal, operand=None
        ),
        operand=None,
    )


def _restoration_controller_step(problem, controller, config):
    advanced = restoration_advance(
        problem,
        controller.restoration_state,
        controller.filter_state,
        config,
    )
    returned = advanced.status == int(TerminalStatus.RESTORATION_RETURN_ACCEPTED)
    accepted_iteration_delta = jnp.maximum(
        advanced.state.accepted_iteration_count
        - controller.restoration_state.accepted_iteration_count,
        jnp.asarray(0, dtype=jnp.int32),
    )
    next_accepted_iteration_count = (
        controller.restoration_accepted_iteration_count
        + accepted_iteration_delta
    )

    def apply_return(_operand):
        mapped = apply_restoration_return(problem, advanced.state, config)
        return _replace_controller(
            controller,
            mode=jnp.where(
                mapped.accepted,
                int(SolverMode.NORMAL),
                int(SolverMode.FAILED),
            ).astype(jnp.int32),
            original_state=jax.tree_util.tree_map(
                lambda restored, current: jnp.where(
                    mapped.accepted, restored, current
                ),
                mapped.original_state,
                controller.original_state,
            ),
            restoration_state=advanced.state,
            restoration_accepted_iteration_count=(
                next_accepted_iteration_count
            ),
            terminal_status=jnp.where(
                mapped.accepted,
                int(TerminalStatus.NOT_TERMINATED),
                mapped.status,
            ).astype(jnp.int32),
            last_return_diagnostics=mapped.diagnostics,
        )

    def persist_or_fail(_operand):
        continuing = advanced.status == int(TerminalStatus.NOT_TERMINATED)
        return _replace_controller(
            controller,
            mode=jnp.where(
                continuing,
                int(SolverMode.RESTORATION),
                int(SolverMode.FAILED),
            ).astype(jnp.int32),
            restoration_state=advanced.state,
            restoration_accepted_iteration_count=(
                next_accepted_iteration_count
            ),
            terminal_status=jnp.where(
                continuing,
                int(TerminalStatus.NOT_TERMINATED),
                advanced.status,
            ).astype(jnp.int32),
        )

    return jax.lax.cond(returned, apply_return, persist_or_fail, operand=None)


def controller_step(
    problem: FixedSupportProblem,
    controller: ControllerState,
    config: FixedSupportV2Config = FixedSupportV2Config(),
) -> ControllerState:
    """Advance exactly one phase-specific fixed-epsilon super-iteration."""

    branches = (
        lambda operand: _normal_controller_step(problem, operand, config),
        lambda operand: _restoration_controller_step(problem, operand, config),
        lambda operand: operand,
        lambda operand: operand,
    )
    return jax.lax.switch(controller.mode, branches, controller)


def solve_fixed_epsilon(
    problem: FixedSupportProblem,
    initial_controller: ControllerState,
    config: FixedSupportV2Config = FixedSupportV2Config(),
) -> ControllerState:
    """Run the complete fixed-epsilon NORMAL/RESTORATION controller."""

    def active(state):
        return (state.mode == int(SolverMode.NORMAL)) | (
            state.mode == int(SolverMode.RESTORATION)
        )

    return jax.lax.while_loop(
        active,
        lambda state: controller_step(problem, state, config),
        initial_controller,
    )


__all__ = [
    "controller_step",
    "initialize_controller",
    "original_converged",
    "solve_fixed_epsilon",
]
