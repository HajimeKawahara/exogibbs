import jax
import jax.numpy as jnp
import pytest

from exogibbs.equilibrium.condensate.fixed_support.controller import (
    controller_step,
    initialize_controller,
    solve_fixed_epsilon,
)
from exogibbs.equilibrium.condensate.fixed_support.filter import empty_filter
from exogibbs.equilibrium.condensate.fixed_support.restoration import (
    initialize_restoration,
    solve_restoration,
)
from exogibbs.equilibrium.condensate.fixed_support.return_map import (
    apply_restoration_return,
)
from exogibbs.equilibrium.condensate.fixed_support.types import (
    FilterConfig,
    FixedSupportProblem,
    FixedSupportV2Config,
    OriginalState,
    RestorationConfig,
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
    )
    state = OriginalState(
        q=jnp.log(jnp.asarray([0.8])),
        r=jnp.log(jnp.asarray([0.2])),
        lambda_=jnp.asarray([0.0]),
        rho=jnp.log(jnp.asarray([0.5])),
        qtot=jnp.log(0.8),
        epsilon=jnp.log(0.1),
        iteration=jnp.asarray(0, dtype=jnp.int32),
    )
    return problem, state


def _infeasible_fixture():
    problem = FixedSupportProblem(
        gas_formula_matrix=jnp.asarray([[1.0, 1.0]]),
        condensate_formula_matrix=jnp.asarray([[1.0]]),
        target_inventory=jnp.asarray([1.0]),
        gamma=jnp.zeros((2,)),
        condensate_standard_source=jnp.zeros((1,)),
        support_indices=jnp.asarray([0], dtype=jnp.int32),
        budget_row_scale=jnp.asarray([1.0]),
        total_density_row_scale=jnp.asarray(1.0 / 0.8),
    )
    state = OriginalState(
        q=jnp.log(jnp.asarray([0.05, 0.05])),
        r=jnp.log(jnp.asarray([0.05])),
        lambda_=jnp.zeros((1,)),
        rho=jnp.log(jnp.asarray([1.0e-6])),
        qtot=jnp.log(0.8),
        epsilon=jnp.log(1.0e-7),
        iteration=jnp.asarray(0, dtype=jnp.int32),
    )
    return problem, state


def _controller_config(**kwargs):
    kwargs.setdefault("soc", SOCConfig(enabled=False))
    return FixedSupportV2Config(
        limits=SolverLimitConfig(
            max_normal_iterations=20,
            max_line_search_trials=8,
            max_restoration_calls=2,
            max_restoration_iterations=50,
            max_restoration_line_search_trials=20,
        ),
        **kwargs,
    )


def test_return_map_uses_linearized_bound_update_and_zero_lambda():
    problem, entry = _equilibrium_fixture()
    config = _controller_config()
    restoration = initialize_restoration(problem, entry, config)._replace(
        x=jnp.asarray([0.8, 0.1, 0.8])
    )
    result = apply_restoration_return(problem, restoration, config)

    assert bool(result.accepted)
    assert result.original_state.q == pytest.approx(jnp.log(jnp.asarray([0.8])))
    assert result.original_state.r == pytest.approx(jnp.log(jnp.asarray([0.1])))
    assert jnp.exp(result.original_state.rho) == pytest.approx(jnp.asarray([0.75]))
    assert jnp.array_equal(result.original_state.lambda_, jnp.zeros((1,)))
    assert float(result.diagnostics.alpha_dual) == pytest.approx(1.0)
    assert bool(result.diagnostics.equality_multiplier_reset)
    assert not bool(result.diagnostics.bound_multiplier_reset)


def test_return_map_resets_all_bound_multipliers_above_global_threshold():
    problem, entry = _equilibrium_fixture()
    config = _controller_config(
        restoration=RestorationConfig(bound_multiplier_reset_threshold=0.5)
    )
    restoration = initialize_restoration(problem, entry, config)._replace(
        x=jnp.asarray([0.8, 0.1, 0.8])
    )
    result = apply_restoration_return(problem, restoration, config)

    assert bool(result.accepted)
    assert bool(result.diagnostics.bound_multiplier_reset)
    assert jnp.array_equal(jnp.exp(result.original_state.rho), jnp.ones((1,)))


def test_return_map_rejects_material_representational_floor_injection():
    problem, entry = _equilibrium_fixture()
    config = _controller_config(
        restoration=RestorationConfig(
            representation_floor=0.1,
            representation_floor_injection_tolerance=0.0,
        )
    )
    restoration = initialize_restoration(problem, entry, config)._replace(
        x=jnp.asarray([1.0e-4, 1.0e-4, 0.8])
    )
    result = apply_restoration_return(problem, restoration, config)

    assert not bool(result.accepted)
    assert bool(result.diagnostics.representation_floor_applied)
    assert int(result.status) == TerminalStatus.RETURN_REPRESENTATION_FLOOR_FAILED


def test_converged_controller_freezes_original_state():
    problem, state = _equilibrium_fixture()
    config = _controller_config()
    controller = initialize_controller(problem, state, config)
    result = controller_step(problem, controller, config)

    assert int(result.mode) == SolverMode.CONVERGED
    assert int(result.terminal_status) == TerminalStatus.CONVERGED
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, result.original_state, state)
    )
    assert int(result.normal_iteration_count) == 0


def test_feasible_normal_failure_is_typed_dual_failure():
    problem, state = _equilibrium_fixture()
    state = state._replace(lambda_=jnp.asarray([0.2]))
    config = _controller_config(
        filter=FilterConfig(roundoff_tolerance_factor=0.0)
    )
    controller = initialize_controller(problem, state, config)
    result = controller_step(problem, controller, config)

    assert int(result.mode) == SolverMode.FAILED
    assert int(result.terminal_status) == TerminalStatus.NORMAL_DUAL_STEP_FAILED
    assert int(result.restoration_call_count) == 0


def test_dual_only_normal_step_is_not_rejected_by_roundoff():
    problem, state = _equilibrium_fixture()
    state = state._replace(lambda_=jnp.asarray([0.01]))
    config = _controller_config()
    controller = initialize_controller(problem, state, config)
    result = controller_step(problem, controller, config)

    assert int(result.mode) == SolverMode.NORMAL
    assert int(result.terminal_status) == TerminalStatus.NOT_TERMINATED
    assert int(result.normal_iteration_count) == 1
    assert result.original_state.lambda_ == pytest.approx(jnp.zeros((1,)), abs=1.0e-14)


def test_restoration_entry_and_next_step_do_not_mix_or_reinitialize():
    problem, state = _infeasible_fixture()
    config = _controller_config(
        filter=FilterConfig(theta_max_factor=0.0),
    )
    controller = initialize_controller(problem, state, config)
    entered = controller_step(problem, controller, config)

    assert int(entered.mode) == SolverMode.RESTORATION
    assert int(entered.restoration_call_count) == 1
    assert int(entered.restoration_accepted_iteration_count) == 0
    assert int(entered.restoration_state.iteration) == 0
    assert int(entered.restoration_state.accepted_iteration_count) == 0
    assert int(jnp.sum(entered.filter_state.valid_entries)) == 1
    entry_x = entered.restoration_state.entry_x
    entry_slack = entered.restoration_state.positive_slack
    entry_filter = entered.filter_state

    advanced = controller_step(problem, entered, config)
    assert int(advanced.mode) == SolverMode.RESTORATION
    assert int(advanced.restoration_state.iteration) == 1
    assert int(advanced.restoration_state.accepted_iteration_count) == 1
    assert int(advanced.restoration_accepted_iteration_count) == 1
    assert jnp.array_equal(advanced.restoration_state.entry_x, entry_x)
    assert not jnp.array_equal(
        advanced.restoration_state.positive_slack, entry_slack
    )
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            jnp.array_equal, advanced.filter_state, entry_filter
        )
    )


def test_accepted_restoration_exit_applies_return_once_and_resumes_normal():
    problem, state = _infeasible_fixture()
    config = _controller_config()
    restoration = initialize_restoration(problem, state, config)
    filter_state = empty_filter(config.limits.filter_capacity, dtype=state.q.dtype)
    solved = solve_restoration(
        problem, restoration, filter_state, config
    )
    controller = initialize_controller(problem, state, config)._replace(
        mode=jnp.asarray(SolverMode.RESTORATION, dtype=jnp.int32),
        restoration_state=solved.state,
        restoration_call_count=jnp.asarray(1, dtype=jnp.int32),
    )
    returned = controller_step(problem, controller, config)

    assert int(returned.mode) == SolverMode.NORMAL
    assert int(returned.terminal_status) == TerminalStatus.NOT_TERMINATED
    assert int(returned.restoration_call_count) == 1
    assert jnp.array_equal(returned.original_state.lambda_, jnp.zeros((1,)))
    assert bool(returned.last_return_diagnostics.equality_multiplier_reset)
    assert int(returned.restoration_state.iteration) == int(solved.state.iteration)


def test_restoration_failure_propagates_one_primary_status():
    problem, state = _infeasible_fixture()
    config = _controller_config()
    controller = initialize_controller(problem, state, config)._replace(
        mode=jnp.asarray(SolverMode.RESTORATION, dtype=jnp.int32),
    )
    result = controller_step(problem, controller, config)

    assert int(result.mode) == SolverMode.FAILED
    assert int(result.terminal_status) == TerminalStatus.RESTORATION_NONFINITE


def test_material_failure_with_no_restoration_calls_has_typed_status():
    problem, state = _infeasible_fixture()
    config = FixedSupportV2Config(
        filter=FilterConfig(theta_max_factor=0.0),
        limits=SolverLimitConfig(
            max_normal_iterations=5,
            max_line_search_trials=4,
            max_restoration_calls=0,
        ),
    )
    result = controller_step(
        problem, initialize_controller(problem, state, config), config
    )

    assert int(result.mode) == SolverMode.FAILED
    assert int(result.terminal_status) == TerminalStatus.RESTORATION_MAX_CALLS


def test_fixed_epsilon_controller_is_jittable_and_converges_exact_fixture():
    problem, state = _equilibrium_fixture()
    config = _controller_config()
    controller = initialize_controller(problem, state, config)
    result = jax.jit(
        lambda p, c: solve_fixed_epsilon(p, c, config)
    )(problem, controller)

    assert int(result.mode) == SolverMode.CONVERGED
    assert int(result.terminal_status) == TerminalStatus.CONVERGED


def test_fixed_epsilon_minus_eleven_contract_fixture_converges():
    problem, state = _equilibrium_fixture()
    mu = jnp.exp(-11.0)
    state = state._replace(
        rho=jnp.log(jnp.asarray([mu / 0.2])),
        epsilon=jnp.asarray(-11.0),
    )
    problem = problem._replace(
        condensate_standard_source=jnp.asarray([mu / 0.2])
    )
    config = _controller_config()
    result = solve_fixed_epsilon(
        problem, initialize_controller(problem, state, config), config
    )

    assert int(result.mode) == SolverMode.CONVERGED
    assert int(result.terminal_status) == TerminalStatus.CONVERGED


def test_batched_fixed_epsilon_matches_independent_controller_solves():
    problem, state = _equilibrium_fixture()
    config = _controller_config()
    controllers = (
        initialize_controller(problem, state, config),
        initialize_controller(
            problem, state._replace(lambda_=jnp.asarray([0.01])), config
        ),
    )
    independent = tuple(
        solve_fixed_epsilon(problem, controller, config)
        for controller in controllers
    )
    batched_input = jax.tree_util.tree_map(
        lambda first, second: jnp.stack([first, second]), *controllers
    )
    batched = jax.jit(
        jax.vmap(lambda controller: solve_fixed_epsilon(problem, controller, config))
    )(batched_input)

    expected = jax.tree_util.tree_map(
        lambda first, second: jnp.stack([first, second]), *independent
    )
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, batched, expected)
    )

    reversed_input = jax.tree_util.tree_map(
        lambda values: values[::-1], batched_input
    )
    reversed_result = jax.jit(
        jax.vmap(lambda controller: solve_fixed_epsilon(problem, controller, config))
    )(reversed_input)
    restored_order = jax.tree_util.tree_map(
        lambda values: values[::-1], reversed_result
    )
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, restored_order, batched)
    )
