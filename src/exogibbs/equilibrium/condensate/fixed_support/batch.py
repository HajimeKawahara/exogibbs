"""Prepared-profile adapter for the fixed-support v2 solver."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import time
from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp

from exogibbs.equilibrium.condensate.fixed_support.continuation import (
    initialize_continuation,
    solve_continuation,
)
from exogibbs.equilibrium.condensate.fixed_support.normal import normal_step
from exogibbs.equilibrium.condensate.fixed_support.problem import (
    barrier_objective,
    filter_violation,
    kkt_component_norms,
    physical_amounts,
)
from exogibbs.equilibrium.condensate.fixed_support.restoration import (
    initialize_restoration,
    restoration_barrier_objective,
    restoration_elastic_objective,
    restoration_equalities,
    restoration_iteration,
    restoration_residuals,
)
from exogibbs.equilibrium.condensate.fixed_support.types import (
    ContinuationState,
    FixedSupportProblem,
    FixedSupportV2Config,
    KKTComponentNorms,
    OriginalState,
    TerminalStatus,
)


class FixedSupportV2BucketResult(NamedTuple):
    """Batched numerical result for one common-support bucket."""

    continuation: ContinuationState
    final_kkt_norms: KKTComponentNorms


class FixedSupportV2ProductionBucketResult(NamedTuple):
    """Minimal fixed-support output required by the production lifecycle."""

    final_state: OriginalState
    terminal_status: Any
    completed_stage_count: Any
    final_kkt_norms: KKTComponentNorms


@dataclass(frozen=True)
class FixedSupportV2BucketExecution:
    """One compiled bucket result with separately measured timings."""

    result: FixedSupportV2BucketResult | FixedSupportV2ProductionBucketResult
    compilation_seconds: float
    execution_seconds: float
    backend: str


@dataclass(frozen=True)
class PreparedFixedSupportV2Bucket:
    """Backend-neutral prepared inputs for one common-support v2 bucket."""

    support_indices: tuple[int, ...]
    layer_indices: tuple[int, ...]
    formula_matrix_cond_active: Any
    ln_nk_init: Any
    ln_mk_init: Any
    ln_ntot_init: Any
    element_potential_init: Any | None
    rho_init: Any | None
    barrier_epsilon_init: Any | None
    element_inventory_target: Any
    hvector: Any
    hvector_cond_active: Any
    ln_normalized_pressure: Any


@dataclass(frozen=True)
class PreparedFixedSupportV2LayerState:
    """Validated per-layer state used only to construct v2 profile buckets."""

    ln_nk: Any
    ln_mk: Any
    ln_ntot: Any
    element_potential: Any | None = None
    rho: Any | None = None
    barrier_epsilon: Any | None = None


def prepare_fixed_support_v2_buckets(
    *,
    init_states: Sequence[PreparedFixedSupportV2LayerState],
    support_indices_by_layer: Sequence[Sequence[int]],
    formula_matrix_cond: Any,
    element_inventory_target_by_layer: Any,
    hvector_by_layer: Any,
    hvector_cond_by_layer: Any,
    ln_normalized_pressure_by_layer: Any,
) -> tuple[PreparedFixedSupportV2Bucket, ...]:
    """Group validated layer states into exact-support v2 buckets."""

    n_layers = len(init_states)
    if len(support_indices_by_layer) != n_layers:
        raise ValueError(
            "init_states and support_indices_by_layer must have matching lengths."
        )

    formula_matrix_cond = jnp.asarray(
        formula_matrix_cond,
        dtype=jnp.float64,
    )
    if formula_matrix_cond.ndim != 2:
        raise ValueError("formula_matrix_cond must be two-dimensional.")
    targets = jnp.asarray(
        element_inventory_target_by_layer,
        dtype=jnp.float64,
    )
    hvectors = jnp.asarray(hvector_by_layer, dtype=jnp.float64)
    hcond = jnp.asarray(hvector_cond_by_layer, dtype=jnp.float64)
    ln_pressures = jnp.asarray(
        ln_normalized_pressure_by_layer,
        dtype=jnp.float64,
    )
    for name, values in (
        ("element_inventory_target_by_layer", targets),
        ("hvector_by_layer", hvectors),
        ("hvector_cond_by_layer", hcond),
        ("ln_normalized_pressure_by_layer", ln_pressures),
    ):
        if values.shape[0] != n_layers:
            raise ValueError(f"{name} must have one row per layer.")
    if targets.ndim != 2:
        raise ValueError(
            "element_inventory_target_by_layer must be two-dimensional."
        )
    if hvectors.ndim != 2:
        raise ValueError("hvector_by_layer must be two-dimensional.")
    if hcond.ndim != 2:
        raise ValueError("hvector_cond_by_layer must be two-dimensional.")
    if ln_pressures.ndim != 1:
        raise ValueError(
            "ln_normalized_pressure_by_layer must be one-dimensional."
        )
    if targets.shape[1] != formula_matrix_cond.shape[0]:
        raise ValueError(
            "element inventory and formula_matrix_cond must share the "
            "element dimension."
        )
    if hcond.shape[1] != formula_matrix_cond.shape[1]:
        raise ValueError(
            "hvector_cond_by_layer must have one value per condensate."
        )

    groups: dict[tuple[int, ...], list[int]] = {}
    for layer_index, indices in enumerate(support_indices_by_layer):
        support = tuple(int(index) for index in indices)
        if not support:
            raise ValueError("fixed-support v2 buckets require non-empty support.")
        if len(set(support)) != len(support):
            raise ValueError("fixed-support v2 support indices must be unique.")
        if any(
            index < 0 or index >= formula_matrix_cond.shape[1]
            for index in support
        ):
            raise ValueError(
                "fixed-support v2 support contains an out-of-range index."
            )
        groups.setdefault(support, []).append(layer_index)

    buckets = []
    for support, layer_indices in groups.items():
        support_array = jnp.asarray(support, dtype=jnp.int32)
        ln_nk_init = []
        ln_mk_init = []
        ln_ntot_init = []
        element_potential_init = []
        rho_init = []
        barrier_epsilon_init = []
        have_element_potential = True
        have_rho = True
        have_barrier_epsilon = True
        for layer_index in layer_indices:
            state = init_states[layer_index]
            ln_nk = jnp.asarray(state.ln_nk, dtype=jnp.float64)
            ln_mk = jnp.asarray(state.ln_mk, dtype=jnp.float64)
            ln_ntot = jnp.asarray(state.ln_ntot, dtype=jnp.float64)
            if ln_nk.ndim != 1:
                raise ValueError("ln_nk must be one-dimensional.")
            if ln_ntot.ndim != 0:
                raise ValueError("ln_ntot must be scalar.")
            if ln_mk.ndim != 1:
                raise ValueError("ln_mk must be one-dimensional.")
            if ln_mk.shape[0] == formula_matrix_cond.shape[1]:
                ln_mk = ln_mk[support_array]
            elif ln_mk.shape[0] != support_array.shape[0]:
                raise ValueError(
                    "ln_mk must have full condensate length or support length."
                )
            ln_nk_init.append(ln_nk)
            ln_mk_init.append(ln_mk)
            ln_ntot_init.append(ln_ntot)

            if state.element_potential is None:
                have_element_potential = False
            else:
                element_potential = jnp.asarray(
                    state.element_potential,
                    dtype=jnp.float64,
                )
                if element_potential.shape != (
                    formula_matrix_cond.shape[0],
                ):
                    raise ValueError(
                        "element_potential must have one value per element."
                    )
                element_potential_init.append(element_potential)

            if state.rho is None:
                have_rho = False
            else:
                rho = jnp.asarray(state.rho, dtype=jnp.float64)
                if rho.ndim != 1:
                    raise ValueError("rho must be one-dimensional.")
                if rho.shape[0] == formula_matrix_cond.shape[1]:
                    rho = rho[support_array]
                elif rho.shape[0] != support_array.shape[0]:
                    raise ValueError(
                        "rho must have full condensate length or support length."
                    )
                rho_init.append(rho)

            if state.barrier_epsilon is None:
                have_barrier_epsilon = False
            else:
                barrier_epsilon = jnp.asarray(
                    state.barrier_epsilon,
                    dtype=jnp.float64,
                )
                if barrier_epsilon.ndim != 0:
                    raise ValueError("barrier_epsilon must be scalar.")
                barrier_epsilon_init.append(barrier_epsilon)

        layer_array = jnp.asarray(layer_indices, dtype=jnp.int32)
        buckets.append(
            PreparedFixedSupportV2Bucket(
                support_indices=support,
                layer_indices=tuple(layer_indices),
                formula_matrix_cond_active=formula_matrix_cond[
                    :, support_array
                ],
                ln_nk_init=jnp.stack(ln_nk_init),
                ln_mk_init=jnp.stack(ln_mk_init),
                ln_ntot_init=jnp.stack(ln_ntot_init),
                element_potential_init=(
                    jnp.stack(element_potential_init)
                    if have_element_potential
                    else None
                ),
                rho_init=jnp.stack(rho_init) if have_rho else None,
                barrier_epsilon_init=(
                    jnp.stack(barrier_epsilon_init)
                    if have_barrier_epsilon
                    else None
                ),
                element_inventory_target=targets[layer_array],
                hvector=hvectors[layer_array],
                hvector_cond_active=hcond[layer_array][:, support_array],
                ln_normalized_pressure=ln_pressures[layer_array],
            )
        )
    return tuple(buckets)


@lru_cache(maxsize=32)
def _compiled_solver_factory(
    config: FixedSupportV2Config,
    include_solver_diagnostics: bool = True,
):
    """Reuse one JIT identity so repeated prepared calls hit JAX's cache."""

    def solve_one(problem, state):
        initial = initialize_continuation(problem, state, config)
        continuation = solve_continuation(problem, initial, config)
        norms = kkt_component_norms(
            problem, continuation.controller.original_state
        )
        if include_solver_diagnostics:
            return FixedSupportV2BucketResult(continuation, norms)
        return FixedSupportV2ProductionBucketResult(
            final_state=continuation.controller.original_state,
            terminal_status=continuation.terminal_status,
            completed_stage_count=continuation.completed_stage_count,
            final_kkt_norms=norms,
        )

    return jax.jit(jax.vmap(solve_one))


def _block_until_ready(tree) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        leaf.block_until_ready()


def _validate_profile_structure(
    buckets: Sequence[Any],
    *,
    layer_count: int,
    condensate_count: int,
) -> None:
    if layer_count < 0 or condensate_count < 0:
        raise ValueError("layer_count and condensate_count must be non-negative.")
    layer_indices = []
    for bucket in buckets:
        layers = tuple(int(value) for value in bucket.layer_indices)
        support = tuple(int(value) for value in bucket.support_indices)
        if len(set(layers)) != len(layers):
            raise ValueError("A prepared bucket contains duplicate layer indices.")
        if any(index < 0 or index >= layer_count for index in layers):
            raise ValueError("A prepared bucket contains an invalid layer index.")
        if len(set(support)) != len(support):
            raise ValueError("A prepared bucket contains duplicate support indices.")
        if any(index < 0 or index >= condensate_count for index in support):
            raise ValueError("A prepared bucket contains an invalid support index.")
        layer_indices.extend(layers)
    if sorted(layer_indices) != list(range(layer_count)):
        raise ValueError(
            "Prepared buckets must cover every profile layer exactly once."
        )


def _prepared_problem_batch(
    bucket,
    formula_matrix,
    *,
    budget_relative_floor,
):
    q = jnp.asarray(bucket.ln_nk_init)
    dtype = q.dtype
    batch_size = q.shape[0]
    ag = jnp.asarray(formula_matrix, dtype=dtype)
    ac = jnp.asarray(bucket.formula_matrix_cond_active, dtype=dtype)
    target = jnp.asarray(bucket.element_inventory_target, dtype=dtype)
    qtot = jnp.asarray(bucket.ln_ntot_init, dtype=dtype)
    gamma_from_thermo = jnp.asarray(bucket.hvector, dtype=dtype) + jnp.asarray(
        bucket.ln_normalized_pressure, dtype=dtype
    )[:, None]
    gamma = gamma_from_thermo
    floor = jnp.asarray(budget_relative_floor, dtype=dtype)
    budget_scale = 1.0 / jnp.maximum(jnp.abs(target), floor)
    total_scale = 1.0 / jnp.maximum(jnp.exp(qtot), floor)
    support = jnp.asarray(bucket.support_indices, dtype=jnp.int32)
    return FixedSupportProblem(
        gas_formula_matrix=jnp.broadcast_to(
            ag, (batch_size,) + ag.shape
        ),
        condensate_formula_matrix=jnp.broadcast_to(
            ac, (batch_size,) + ac.shape
        ),
        target_inventory=target,
        gamma=gamma,
        condensate_standard_source=jnp.asarray(
            bucket.hvector_cond_active, dtype=dtype
        ),
        support_indices=jnp.broadcast_to(
            support, (batch_size,) + support.shape
        ),
        budget_row_scale=budget_scale,
        total_density_row_scale=total_scale,
    )


def _prepared_original_state_batch(bucket, problems, config):
    q = jnp.asarray(bucket.ln_nk_init)
    dtype = q.dtype
    r = jnp.asarray(bucket.ln_mk_init, dtype=dtype)
    qtot = jnp.asarray(bucket.ln_ntot_init, dtype=dtype)
    if not config.continuation.epsilon_schedule:
        raise ValueError("epsilon_schedule must contain at least one stage.")
    first_epsilon = jnp.asarray(
        config.continuation.epsilon_schedule[0], dtype=dtype
    )
    if bucket.element_potential_init is None:
        ag = problems.gas_formula_matrix[0]
        rhs = q + problems.gamma - qtot[:, None]
        lambda_ = jax.vmap(
            lambda row: jnp.linalg.lstsq(ag.T, row, rcond=None)[0]
        )(rhs)
    else:
        lambda_ = jnp.asarray(bucket.element_potential_init, dtype=dtype)
    batch_size = q.shape[0]
    if config.continuation.initial_state_policy == "provided":
        if bucket.rho_init is None or bucket.barrier_epsilon_init is None:
            raise ValueError(
                "initial_state_policy='provided' requires prepared rho and "
                "barrier_epsilon for every layer."
            )
        rho = jnp.asarray(bucket.rho_init, dtype=dtype)
        epsilon = jnp.asarray(bucket.barrier_epsilon_init, dtype=dtype)
        if rho.shape != r.shape:
            raise ValueError("Prepared rho must have the same shape as ln_mk_init.")
        if epsilon.shape != (batch_size,):
            raise ValueError(
                "Prepared barrier_epsilon must have one scalar per layer."
            )
        if not bool(jnp.all(epsilon == first_epsilon)):
            raise ValueError(
                "Provided barrier_epsilon must equal the first epsilon_schedule "
                "stage for an exact-state comparison."
            )
    else:
        rho = first_epsilon - r
        epsilon = jnp.full((batch_size,), first_epsilon, dtype=dtype)
    return OriginalState(
        q=q,
        r=r,
        lambda_=lambda_,
        rho=rho,
        qtot=qtot,
        epsilon=epsilon,
        iteration=jnp.zeros((batch_size,), dtype=jnp.int32),
    )


def solve_prepared_bucket_v2(
    bucket,
    formula_matrix,
    config: FixedSupportV2Config = FixedSupportV2Config(),
    *,
    budget_relative_floor: float = 1.0e-6,
    include_solver_diagnostics: bool = True,
) -> FixedSupportV2BucketExecution:
    """Compile and run one same-support prepared bucket with v2."""

    problems = _prepared_problem_batch(
        bucket,
        formula_matrix,
        budget_relative_floor=budget_relative_floor,
    )
    states = _prepared_original_state_batch(bucket, problems, config)

    batched = _compiled_solver_factory(config, include_solver_diagnostics)
    compile_start = time.perf_counter()
    compiled = batched.lower(problems, states).compile()
    compilation_seconds = time.perf_counter() - compile_start
    execution_start = time.perf_counter()
    result = compiled(problems, states)
    _block_until_ready(result)
    execution_seconds = time.perf_counter() - execution_start
    return FixedSupportV2BucketExecution(
        result=result,
        compilation_seconds=compilation_seconds,
        execution_seconds=execution_seconds,
        backend=jax.default_backend(),
    )


@lru_cache(maxsize=32)
def _compiled_terminal_restoration_diagnostics_factory(config):
    """Reuse the compact terminal-restoration diagnostic JIT."""

    def diagnose(problem, controller):
        available = controller.restoration_call_count > 0
        initialized = initialize_restoration(
            problem, controller.original_state, config
        )
        state = jax.tree_util.tree_map(
            lambda stored, fallback: jnp.where(available, stored, fallback),
            controller.restoration_state,
            initialized,
        )
        ng = state.entry_original_state.q.shape[0]
        nc = state.entry_original_state.r.shape[0]
        current_original = state.entry_original_state._replace(
            q=jnp.log(state.x[:ng]),
            r=jnp.log(state.x[ng : ng + nc]),
            qtot=jnp.log(state.x[-1]),
        )
        entry_amounts = physical_amounts(state.entry_original_state)
        raw_entry_x = jnp.concatenate(
            [
                entry_amounts.gas,
                entry_amounts.condensate,
                entry_amounts.total_gas.reshape((1,)),
            ]
        )
        entry_injection = state.entry_x - raw_entry_x
        ag = jnp.asarray(problem.gas_formula_matrix, dtype=state.x.dtype)
        ac = jnp.asarray(
            problem.condensate_formula_matrix, dtype=state.x.dtype
        )
        scaled_budget_injection = state.row_scales[:-1] * (
            ag @ entry_injection[:ng]
            + ac @ entry_injection[ng : ng + nc]
        )
        scaled_total_injection = state.row_scales[-1] * (
            jnp.sum(entry_injection[:ng]) - entry_injection[-1]
        )

        def max_abs(value):
            return jnp.max(jnp.abs(value), initial=0.0)

        state_values = jnp.concatenate(
            [
                state.x,
                state.positive_slack,
                state.negative_slack,
                state.equality_dual,
                state.lower_bound_dual_x,
                state.lower_bound_dual_positive,
                state.lower_bound_dual_negative,
            ]
        )
        positive_values = jnp.concatenate(
            [
                state.x,
                state.positive_slack,
                state.negative_slack,
                state.lower_bound_dual_x,
                state.lower_bound_dual_positive,
                state.lower_bound_dual_negative,
            ]
        )
        residual = restoration_residuals(problem, state, config.restoration)
        replay_state_usable = (
            jnp.all(jnp.isfinite(state_values))
            & jnp.all(positive_values > 0.0)
        )
        replay_state = jax.tree_util.tree_map(
            lambda current, fallback: jnp.where(
                replay_state_usable, current, fallback
            ),
            state,
            initialized,
        )
        replay = restoration_iteration(problem, replay_state, config)
        return {
            "available": available,
            "controller_mode": controller.mode,
            "terminal_status": controller.terminal_status,
            "restoration_call_count": controller.restoration_call_count,
            "iteration": state.iteration,
            "accepted_iteration_count": state.accepted_iteration_count,
            "state_values_finite": jnp.all(jnp.isfinite(state_values)),
            "state_strictly_positive": jnp.all(positive_values > 0.0),
            "minimum_positive_state_value": jnp.min(positive_values),
            "maximum_abs_state_value": max_abs(state_values),
            "entry_phi": state.entry_phi,
            "entry_theta": state.entry_theta,
            "current_phi": barrier_objective(problem, current_original),
            "current_theta": filter_violation(problem, current_original),
            "elastic_objective": restoration_elastic_objective(
                state, config.restoration
            ),
            "barrier_objective": restoration_barrier_objective(
                state, config.restoration
            ),
            "equality_violation_l1": jnp.linalg.norm(
                restoration_equalities(problem, state), ord=1
            ),
            "residual_max_norms": jax.tree_util.tree_map(max_abs, residual),
            "entry_floor_applied": jnp.any(entry_injection > 0.0),
            "entry_floor_scaled_budget_injection_max": max_abs(
                scaled_budget_injection
            ),
            "entry_floor_scaled_total_density_injection": jnp.abs(
                scaled_total_injection
            ),
            "next_iteration_replay": {
                "state_usable": replay_state_usable,
                "status": replay.status,
                "accepted": replay.accepted,
                "selected_index": replay.selected_index,
                "selected_alpha": replay.selected_alpha,
                "direction_status": replay.direction_result.status,
                "direction_diagnostics": replay.direction_result.diagnostics,
                "alphas": replay.trials.alphas,
                "finite_and_positive": replay.trials.finite_and_positive,
                "objective_acceptable": replay.trials.objective_acceptable,
                "constraint_acceptable": replay.trials.constraint_acceptable,
                "accepted_trials": replay.trials.accepted,
                "elastic_objective": replay.trials.elastic_objective,
                "barrier_objective": replay.trials.barrier_objective,
                "equality_violation": replay.trials.equality_violation,
            },
        }

    return jax.jit(jax.vmap(diagnose))


def _terminal_restoration_diagnostics(problems, continuation, config):
    """Replay one terminal restoration step for compact failure diagnosis."""

    compiled = _compiled_terminal_restoration_diagnostics_factory(config)
    return compiled(problems, continuation.controller)


@lru_cache(maxsize=32)
def _compiled_terminal_normal_diagnostics_factory(config):
    """Reuse the compact terminal-normal diagnostic JIT."""

    def diagnose(problem, controller):
        status = controller.terminal_status
        available = (
            (status == int(TerminalStatus.NORMAL_LINE_SEARCH_FAILED))
            | (status == int(TerminalStatus.NORMAL_DUAL_STEP_FAILED))
            | (status == int(TerminalStatus.NORMAL_LINEAR_SOLVE_FAILED))
            | (status == int(TerminalStatus.NORMAL_MAX_ITER))
            | (status == int(TerminalStatus.RESTORATION_MAX_CALLS))
        )
        replay = normal_step(
            problem,
            controller.original_state,
            controller.filter_state,
            initial_theta=controller.initial_theta,
            config=config,
        )
        return {
            "available": available,
            "controller_mode": controller.mode,
            "terminal_status": status,
            "current_phi": barrier_objective(
                problem, controller.original_state
            ),
            "current_theta": filter_violation(
                problem, controller.original_state
            ),
            "current_kkt_norms": kkt_component_norms(
                problem, controller.original_state
            ),
            "direction_status": replay.direction_result.status,
            "direction_diagnostics": replay.direction_result.diagnostics,
            "selection_status": replay.selection.status,
            "accepted": replay.selection.accepted,
            "selected_index": replay.selection.selected_index,
            "selected_alpha": replay.selection.selected_alpha,
            "alphas": replay.trials.alphas,
            "phi": replay.trials.phi,
            "theta": replay.trials.theta,
            "finite": replay.trials.finite,
            "within_theta_max": replay.trials.within_theta_max,
            "current_acceptable": replay.trials.current_acceptable,
            "history_acceptable": replay.trials.history_acceptable,
            "f_type": replay.trials.f_type,
            "armijo": replay.trials.armijo,
            "accepted_trials": replay.trials.accepted,
            "rejection_reasons": replay.trials.rejection_reasons,
        }

    return jax.jit(jax.vmap(diagnose))


def _terminal_normal_diagnostics(problems, continuation, config):
    """Replay one terminal normal step for compact failure diagnosis."""

    compiled = _compiled_terminal_normal_diagnostics_factory(config)
    return compiled(problems, continuation.controller)


def run_fixed_support_profile(
    *,
    buckets: Sequence[Any],
    formula_matrix,
    layer_count: int,
    condensate_count: int,
    config: FixedSupportV2Config = FixedSupportV2Config(),
    budget_relative_floor: float = 1.0e-6,
    include_terminal_diagnostics: bool = True,
) -> dict[str, Any]:
    """Run prepared fixed-support buckets without full-catalog decisions."""

    _validate_profile_structure(
        buckets,
        layer_count=layer_count,
        condensate_count=condensate_count,
    )
    ag = jnp.asarray(formula_matrix)
    gas_log_amounts = jnp.zeros((layer_count, ag.shape[1]), dtype=ag.dtype)
    condensate_amounts = jnp.zeros(
        (layer_count, condensate_count), dtype=ag.dtype
    )
    total_gas_log_amount = jnp.zeros((layer_count,), dtype=ag.dtype)
    element_potential = jnp.zeros((layer_count, ag.shape[0]), dtype=ag.dtype)
    terminal_status = jnp.full(
        (layer_count,), int(TerminalStatus.INTERNAL_CONTRACT_ERROR), dtype=jnp.int32
    )
    completed_stage_count = jnp.zeros((layer_count,), dtype=jnp.int32)
    final_kkt_norms = KKTComponentNorms(
        *(
            jnp.zeros((layer_count,), dtype=ag.dtype)
            for _ in KKTComponentNorms._fields
        )
    )
    target_by_layer = jnp.zeros((layer_count, ag.shape[0]), dtype=ag.dtype)
    support_mask = jnp.zeros((layer_count, condensate_count), dtype=bool)
    final_state_values_finite = jnp.zeros((layer_count,), dtype=bool)
    bucket_reports = []
    compilation_seconds = 0.0
    execution_seconds = 0.0
    diagnostic_seconds = 0.0
    backend = jax.default_backend()

    for bucket in buckets:
        execution = solve_prepared_bucket_v2(
            bucket,
            ag,
            config,
            budget_relative_floor=budget_relative_floor,
            include_solver_diagnostics=include_terminal_diagnostics,
        )
        result = execution.result
        problems = _prepared_problem_batch(
            bucket,
            ag,
            budget_relative_floor=budget_relative_floor,
        )
        restoration_diagnostics = None
        normal_diagnostics = None
        if include_terminal_diagnostics:
            continuation = result.continuation
            diagnostic_start = time.perf_counter()
            restoration_diagnostics = _terminal_restoration_diagnostics(
                problems, continuation, config
            )
            normal_diagnostics = _terminal_normal_diagnostics(
                problems, continuation, config
            )
            _block_until_ready((restoration_diagnostics, normal_diagnostics))
            diagnostic_seconds += time.perf_counter() - diagnostic_start
            terminal = continuation.terminal_status
            completed_stages = continuation.completed_stage_count
            stage_statuses = continuation.stage_statuses
            stage_normal_iterations = (
                continuation.stage_normal_iteration_counts
            )
            stage_restoration_calls = (
                continuation.stage_restoration_call_counts
            )
            stage_restoration_accepted = (
                continuation.stage_restoration_accepted_iteration_counts
            )
            stage_last_return = continuation.stage_last_return_diagnostics
            stage_soc_attempts = continuation.stage_soc_attempt_counts
            stage_soc_accepted = continuation.stage_soc_accepted_counts
            last_return = continuation.controller.last_return_diagnostics
            final = continuation.controller.original_state
        else:
            terminal = result.terminal_status
            completed_stages = result.completed_stage_count
            stage_statuses = None
            stage_normal_iterations = None
            stage_restoration_calls = None
            stage_restoration_accepted = None
            stage_last_return = None
            stage_soc_attempts = None
            stage_soc_accepted = None
            last_return = None
            final = result.final_state
        layers = jnp.asarray(bucket.layer_indices, dtype=jnp.int32)
        support = jnp.asarray(bucket.support_indices, dtype=jnp.int32)
        gas_log_amounts = gas_log_amounts.at[layers].set(final.q)
        condensate_amounts = condensate_amounts.at[
            layers[:, None], support[None, :]
        ].set(jnp.exp(final.r))
        total_gas_log_amount = total_gas_log_amount.at[layers].set(final.qtot)
        element_potential = element_potential.at[layers].set(final.lambda_)
        terminal_status = terminal_status.at[layers].set(
            terminal
        )
        completed_stage_count = completed_stage_count.at[layers].set(
            completed_stages
        )
        final_kkt_norms = jax.tree_util.tree_map(
            lambda current, bucket_value: current.at[layers].set(bucket_value),
            final_kkt_norms,
            result.final_kkt_norms,
        )
        target_by_layer = target_by_layer.at[layers].set(
            bucket.element_inventory_target
        )
        support_mask = support_mask.at[layers[:, None], support[None, :]].set(True)
        bucket_state_finite = (
            jnp.all(jnp.isfinite(final.q), axis=1)
            & jnp.all(jnp.isfinite(final.r), axis=1)
            & jnp.all(jnp.isfinite(final.lambda_), axis=1)
            & jnp.all(jnp.isfinite(final.rho), axis=1)
            & jnp.isfinite(final.qtot)
            & jnp.isfinite(final.epsilon)
        )
        final_state_values_finite = final_state_values_finite.at[layers].set(
            bucket_state_finite
        )
        compilation_seconds += execution.compilation_seconds
        execution_seconds += execution.execution_seconds
        backend = execution.backend
        bucket_reports.append(
            {
                "support_indices": tuple(
                    int(value) for value in bucket.support_indices
                ),
                "layer_indices": tuple(int(value) for value in bucket.layer_indices),
                "compilation_seconds": execution.compilation_seconds,
                "execution_seconds": execution.execution_seconds,
                "terminal_status": terminal,
                "completed_stage_count": completed_stages,
                "stage_statuses": stage_statuses,
                "stage_normal_iteration_counts": stage_normal_iterations,
                "stage_restoration_call_counts": stage_restoration_calls,
                "stage_restoration_accepted_iteration_counts": (
                    stage_restoration_accepted
                ),
                "stage_last_return_diagnostics": stage_last_return,
                "stage_soc_attempt_counts": stage_soc_attempts,
                "stage_soc_accepted_counts": stage_soc_accepted,
                "final_kkt_norms": result.final_kkt_norms,
                "terminal_restoration_diagnostics": restoration_diagnostics,
                "terminal_normal_diagnostics": normal_diagnostics,
                "last_return_diagnostics": last_return,
            }
        )

    fixed_support_converged = terminal_status == int(TerminalStatus.CONVERGED)
    return {
        "schema": "exogibbs_fixed_support_v2_prepared_profile_v1",
        "experimental": True,
        "production_preset_promoted": False,
        "backend": backend,
        "compilation_seconds": compilation_seconds,
        "execution_seconds": execution_seconds,
        "diagnostic_seconds": diagnostic_seconds,
        "gas_log_amounts": gas_log_amounts,
        "condensate_amounts": condensate_amounts,
        "total_gas_log_amount": total_gas_log_amount,
        "element_potential": element_potential,
        "terminal_status": terminal_status,
        "completed_stage_count": completed_stage_count,
        "final_kkt_norms": final_kkt_norms,
        "final_state_values_finite": final_state_values_finite,
        "fixed_support_converged": fixed_support_converged,
        "element_inventory_target": target_by_layer,
        "support_mask": support_mask,
        "bucket_reports": tuple(bucket_reports),
    }


__all__ = [
    "FixedSupportV2BucketExecution",
    "FixedSupportV2ProductionBucketResult",
    "FixedSupportV2BucketResult",
    "PreparedFixedSupportV2Bucket",
    "PreparedFixedSupportV2LayerState",
    "prepare_fixed_support_v2_buckets",
    "run_fixed_support_profile",
    "solve_prepared_bucket_v2",
]
