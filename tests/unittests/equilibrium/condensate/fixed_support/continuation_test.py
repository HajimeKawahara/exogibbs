import math

import jax
import jax.numpy as jnp
import pytest

from exogibbs.equilibrium.condensate.fixed_support.continuation import (
    continuation_step,
    initialize_continuation,
    recenter_for_epsilon,
    solve_continuation,
)
from exogibbs.equilibrium.condensate.fixed_support.problem import (
    residual_components,
)
from exogibbs.equilibrium.condensate.fixed_support.types import (
    ContinuationConfig,
    FixedSupportProblem,
    FixedSupportV2Config,
    OriginalState,
    SOCConfig,
    SolverLimitConfig,
    SolverMode,
    TerminalStatus,
)

jax.config.update("jax_enable_x64", True)


def _equilibrium_fixture():
    problem = FixedSupportProblem(
        gas_formula_matrix=jnp.asarray([[1.0]]),
        condensate_formula_matrix=jnp.asarray([[1.0]]),
        target_inventory=jnp.asarray([1.0]),
        gamma=jnp.asarray([0.0]),
        condensate_standard_source=jnp.asarray([0.5]),
        support_indices=jnp.asarray([0], dtype=jnp.int32),
        budget_row_scale=jnp.asarray([1.0]),
        total_density_row_scale=jnp.asarray(1.0 / 0.8),
        condensate_slot_mask=jnp.asarray([True]),
    )
    state = OriginalState(
        q=jnp.log(jnp.asarray([0.8])),
        r=jnp.log(jnp.asarray([0.2])),
        lambda_=jnp.asarray([0.0]),
        rho=jnp.log(jnp.asarray([0.5])),
        qtot=jnp.log(0.8),
        epsilon=jnp.log(0.1),
        iteration=jnp.asarray(7, dtype=jnp.int32),
    )
    return problem, state


def _config(schedule, *, max_normal_iterations=20):
    return FixedSupportV2Config(
        continuation=ContinuationConfig(epsilon_schedule=tuple(schedule)),
        soc=SOCConfig(enabled=False),
        limits=SolverLimitConfig(
            max_normal_iterations=max_normal_iterations,
            max_line_search_trials=8,
            max_restoration_calls=1,
            max_restoration_iterations=20,
            max_restoration_line_search_trials=8,
        ),
    )


def test_recenter_for_epsilon_preserves_warm_primal_and_centers_complementarity():
    problem, state = _equilibrium_fixture()
    epsilon = -7.0
    centered = recenter_for_epsilon(state, epsilon)
    residual = residual_components(problem, centered)

    assert jnp.array_equal(centered.q, state.q)
    assert jnp.array_equal(centered.r, state.r)
    assert jnp.array_equal(centered.lambda_, state.lambda_)
    assert jnp.array_equal(centered.qtot, state.qtot)
    assert float(centered.epsilon) == epsilon
    assert int(centered.iteration) == 0
    assert residual.complementarity == pytest.approx(jnp.zeros((1,)))


def test_recenter_for_epsilon_anchors_dummy_slots_at_unit_source_solution():
    _problem, state = _equilibrium_fixture()
    padded = state._replace(
        r=jnp.asarray([state.r[0], 3.0]),
        rho=jnp.asarray([state.rho[0], -4.0]),
    )
    epsilon = -7.0

    centered = jax.jit(recenter_for_epsilon)(
        padded,
        epsilon,
        jnp.asarray([True, False]),
    )

    assert centered.r[0] == state.r[0]
    assert centered.rho[0] == pytest.approx(epsilon - state.r[0])
    assert centered.r[1] == epsilon
    assert centered.rho[1] == 0.0
    assert centered.r + centered.rho == pytest.approx(
        jnp.full((2,), epsilon)
    )


def test_provided_initial_state_policy_preserves_exact_solver_state():
    problem, state = _equilibrium_fixture()
    supplied = state._replace(
        rho=jnp.asarray([-3.25]),
        iteration=jnp.asarray(9, dtype=jnp.int32),
    )
    config = FixedSupportV2Config(
        continuation=ContinuationConfig(
            epsilon_schedule=(float(state.epsilon),),
            initial_state_policy="provided",
        ),
        soc=SOCConfig(enabled=False),
        limits=SolverLimitConfig(max_normal_iterations=0),
    )

    initial = initialize_continuation(problem, supplied, config)

    assert jnp.array_equal(initial.controller.original_state.q, supplied.q)
    assert jnp.array_equal(initial.controller.original_state.r, supplied.r)
    assert jnp.array_equal(
        initial.controller.original_state.lambda_, supplied.lambda_
    )
    assert jnp.array_equal(initial.controller.original_state.rho, supplied.rho)
    assert jnp.array_equal(initial.controller.original_state.qtot, supplied.qtot)
    assert jnp.array_equal(
        initial.controller.original_state.epsilon, supplied.epsilon
    )
    assert int(initial.controller.original_state.iteration) == 0


def test_provided_initial_state_policy_preserves_real_slot_dtypes():
    problem, state = _equilibrium_fixture()
    supplied = state._replace(
        r=jnp.asarray(state.r, dtype=jnp.float32),
        rho=jnp.asarray(state.rho, dtype=jnp.float32),
        epsilon=jnp.asarray(state.epsilon, dtype=jnp.float32),
    )
    config = FixedSupportV2Config(
        continuation=ContinuationConfig(
            epsilon_schedule=(float(state.epsilon),),
            initial_state_policy="provided",
        ),
        soc=SOCConfig(enabled=False),
        limits=SolverLimitConfig(max_normal_iterations=0),
    )

    initial = initialize_continuation(problem, supplied, config)
    preserved = initial.controller.original_state

    assert preserved.r.dtype == supplied.r.dtype
    assert preserved.rho.dtype == supplied.rho.dtype
    assert preserved.epsilon.dtype == supplied.epsilon.dtype
    assert jnp.array_equal(preserved.r, supplied.r)
    assert jnp.array_equal(preserved.rho, supplied.rho)
    assert jnp.array_equal(preserved.epsilon, supplied.epsilon)


def test_continuation_rejects_unknown_initial_state_policy():
    problem, state = _equilibrium_fixture()
    config = FixedSupportV2Config(
        continuation=ContinuationConfig(
            epsilon_schedule=(float(state.epsilon),),
            initial_state_policy="unknown",  # type: ignore[arg-type]
        )
    )

    with pytest.raises(ValueError, match="initial_state_policy"):
        initialize_continuation(problem, state, config)


@pytest.mark.parametrize(
    "schedule",
    [(), (-11.0, -11.0), (-13.0, -11.0), (-11.0, math.inf)],
)
def test_continuation_rejects_invalid_schedules(schedule):
    problem, state = _equilibrium_fixture()
    with pytest.raises(ValueError):
        initialize_continuation(problem, state, _config(schedule))


def test_stage_transition_is_separate_and_strict_stage_must_converge_itself():
    problem, state = _equilibrium_fixture()
    schedule = (math.log(0.1), math.log(0.01))
    config = _config(schedule, max_normal_iterations=0)
    initial = initialize_continuation(problem, state, config)

    fixed_epsilon_done = continuation_step(problem, initial, config)
    assert int(fixed_epsilon_done.controller.mode) == SolverMode.CONVERGED
    assert int(fixed_epsilon_done.stage_index) == 0
    assert int(fixed_epsilon_done.completed_stage_count) == 0

    advanced = continuation_step(problem, fixed_epsilon_done, config)
    assert int(advanced.controller.mode) == SolverMode.NORMAL
    assert int(advanced.stage_index) == 1
    assert int(advanced.completed_stage_count) == 1
    assert int(advanced.stage_statuses[0]) == TerminalStatus.CONVERGED
    assert int(advanced.stage_statuses[1]) == TerminalStatus.NOT_TERMINATED
    assert float(advanced.controller.original_state.epsilon) == pytest.approx(
        schedule[1]
    )
    assert advanced.controller.original_state.rho == pytest.approx(
        schedule[1] - advanced.controller.original_state.r
    )
    assert int(advanced.controller.original_state.iteration) == 0
    assert not bool(jnp.any(advanced.controller.filter_state.valid_entries))

    strict_stage_attempted = continuation_step(problem, advanced, config)
    assert int(strict_stage_attempted.controller.mode) == SolverMode.FAILED
    assert int(strict_stage_attempted.terminal_status) == (
        TerminalStatus.NOT_TERMINATED
    )

    failed = continuation_step(problem, strict_stage_attempted, config)
    assert int(failed.terminal_status) == TerminalStatus.NORMAL_MAX_ITER
    assert int(failed.completed_stage_count) == 1
    assert int(failed.stage_statuses[1]) == TerminalStatus.NORMAL_MAX_ITER


def test_single_stage_continuation_is_jittable_and_reports_convergence():
    problem, state = _equilibrium_fixture()
    config = _config((math.log(0.1),))
    initial = initialize_continuation(problem, state, config)
    result = jax.jit(lambda carry: solve_continuation(problem, carry, config))(
        initial
    )

    assert int(result.terminal_status) == TerminalStatus.CONVERGED
    assert int(result.completed_stage_count) == 1
    assert int(result.stage_statuses[0]) == TerminalStatus.CONVERGED


def test_stage_record_preserves_restoration_return_diagnostics():
    problem, state = _equilibrium_fixture()
    config = _config((math.log(0.1),))
    initial = initialize_continuation(problem, state, config)
    return_diagnostics = initial.controller.last_return_diagnostics._replace(
        alpha_dual=jnp.asarray(0.25),
        bound_multiplier_reset=jnp.asarray(True),
    )
    completed_controller = initial.controller._replace(
        mode=jnp.asarray(SolverMode.CONVERGED, dtype=jnp.int32),
        terminal_status=jnp.asarray(TerminalStatus.CONVERGED, dtype=jnp.int32),
        last_return_diagnostics=return_diagnostics,
    )

    recorded = continuation_step(
        problem,
        initial._replace(controller=completed_controller),
        config,
    )

    assert recorded.stage_last_return_diagnostics.alpha_dual == pytest.approx(
        jnp.asarray([0.25])
    )
    assert jnp.array_equal(
        recorded.stage_last_return_diagnostics.bound_multiplier_reset,
        jnp.asarray([True]),
    )


def test_nontrivial_four_stage_schedule_converges_each_stage_independently():
    problem, state = _equilibrium_fixture()
    schedule = tuple(math.log(value) for value in (0.1, 0.01, 0.001, 0.0001))
    config = _config(schedule)
    result = solve_continuation(
        problem, initialize_continuation(problem, state, config), config
    )

    assert int(result.terminal_status) == TerminalStatus.CONVERGED
    assert int(result.completed_stage_count) == len(schedule)
    assert jnp.array_equal(
        result.stage_statuses,
        jnp.full((len(schedule),), TerminalStatus.CONVERGED, dtype=jnp.int32),
    )
    assert int(result.stage_normal_iteration_counts[0]) == 0
    assert bool(jnp.all(result.stage_normal_iteration_counts[1:] > 0))
    assert jnp.array_equal(
        result.stage_restoration_accepted_iteration_counts,
        jnp.zeros((len(schedule),), dtype=jnp.int32),
    )
    assert result.stage_last_return_diagnostics.alpha_dual.shape == (
        len(schedule),
    )


def test_batched_layers_advance_stages_independently_and_match_single_solves():
    problem, exact = _equilibrium_fixture()
    schedule = (math.log(0.1), math.log(0.01))
    config = _config(schedule, max_normal_iterations=0)
    perturbed = exact._replace(q=exact.q + 0.1)
    initial_states = (
        initialize_continuation(problem, exact, config),
        initialize_continuation(problem, perturbed, config),
    )
    independent = tuple(
        solve_continuation(problem, initial, config) for initial in initial_states
    )
    batched_input = jax.tree_util.tree_map(
        lambda first, second: jnp.stack([first, second]), *initial_states
    )
    batched = jax.jit(
        jax.vmap(lambda carry: solve_continuation(problem, carry, config))
    )(batched_input)
    expected = jax.tree_util.tree_map(
        lambda first, second: jnp.stack([first, second]), *independent
    )

    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, batched, expected)
    )
    assert jnp.array_equal(
        batched.completed_stage_count, jnp.asarray([1, 0], dtype=jnp.int32)
    )
    assert jnp.array_equal(
        batched.stage_index, jnp.asarray([1, 0], dtype=jnp.int32)
    )
    assert jnp.array_equal(
        batched.terminal_status,
        jnp.asarray(
            [TerminalStatus.NORMAL_MAX_ITER, TerminalStatus.NORMAL_MAX_ITER],
            dtype=jnp.int32,
        ),
    )


def test_terminal_continuation_state_is_frozen():
    problem, state = _equilibrium_fixture()
    config = _config((math.log(0.1),))
    result = solve_continuation(
        problem, initialize_continuation(problem, state, config), config
    )
    repeated = continuation_step(problem, result, config)

    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, repeated, result)
    )
