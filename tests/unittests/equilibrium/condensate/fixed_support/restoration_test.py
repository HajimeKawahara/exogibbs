import jax
import jax.numpy as jnp
import pytest

from exogibbs.equilibrium.condensate.fixed_support.filter import empty_filter
from exogibbs.equilibrium.condensate.fixed_support.problem import (
    residual_components,
)
from exogibbs.equilibrium.condensate.fixed_support.restoration import (
    initialize_restoration,
    restoration_barrier_directional_derivative,
    restoration_barrier_objective,
    restoration_direction,
    restoration_elastic_objective,
    restoration_iteration,
    restoration_kkt_jacobian,
    restoration_residual_vector,
    restoration_residuals,
    restoration_step_limits,
    solve_restoration,
)
from exogibbs.equilibrium.condensate.fixed_support.types import (
    FixedSupportProblem,
    FixedSupportV2Config,
    OriginalState,
    RestorationConfig,
    SolverLimitConfig,
    TerminalStatus,
)

jax.config.update("jax_enable_x64", True)


def _fixture(*, epsilon=1.0e-7):
    ag = jnp.asarray([[1.0, 1.0]])
    ac = jnp.asarray([[1.0]])
    target = jnp.asarray([1.0])
    q = jnp.log(jnp.asarray([0.05, 0.05]))
    r = jnp.log(jnp.asarray([0.05]))
    qtot = jnp.log(0.8)
    problem = FixedSupportProblem(
        gas_formula_matrix=ag,
        condensate_formula_matrix=ac,
        target_inventory=target,
        gamma=jnp.zeros((2,)),
        condensate_standard_source=jnp.zeros((1,)),
        support_indices=jnp.asarray([0], dtype=jnp.int32),
        budget_row_scale=1.0 / target,
        total_density_row_scale=jnp.asarray(1.0 / 0.8),
        condensate_slot_mask=jnp.asarray([True]),
    )
    original = OriginalState(
        q=q,
        r=r,
        lambda_=jnp.zeros((1,)),
        rho=jnp.log(jnp.asarray([1.0e-6])),
        qtot=qtot,
        epsilon=jnp.log(epsilon),
        iteration=jnp.asarray(4, dtype=jnp.int32),
    )
    return problem, original


def _flat_direction(direction):
    return jnp.concatenate([jnp.ravel(block) for block in direction])


def _padded_fixture(*, epsilon=1.0e-7):
    problem, original = _fixture(epsilon=epsilon)
    padded_problem = problem._replace(
        condensate_formula_matrix=jnp.asarray([[1.0, 0.0]]),
        condensate_standard_source=jnp.asarray([0.0, 1.0]),
        support_indices=jnp.asarray([0, 0], dtype=jnp.int32),
        condensate_slot_mask=jnp.asarray([True, False]),
    )
    padded_original = original._replace(
        r=jnp.asarray([original.r[0], 3.0]),
        rho=jnp.asarray([original.rho[0], -4.0]),
    )
    return padded_problem, padded_original


def test_default_restoration_limit_covers_large_support_recovery():
    assert SolverLimitConfig().max_restoration_iterations == 100


def test_legacy_restoration_state_without_slot_mask_remains_jittable():
    problem, original = _fixture()
    config = FixedSupportV2Config()
    state = initialize_restoration(problem, original, config)._replace(
        proximity_mask=None
    )

    result = jax.jit(
        lambda current: restoration_iteration(problem, current, config)
    )(state)

    assert result.state.proximity_mask is None


def test_dummy_restoration_block_is_exact_and_independent_of_real_direction():
    problem, original = _fixture(epsilon=1.0e-2)
    padded_problem, padded_original = _padded_fixture(epsilon=1.0e-2)
    config = FixedSupportV2Config(
        restoration=RestorationConfig(elastic_penalty=2.0)
    )
    state = initialize_restoration(problem, original, config)
    padded_state = initialize_restoration(
        padded_problem, padded_original, config
    )
    direction = restoration_direction(problem, state, config.restoration)
    padded_direction = restoration_direction(
        padded_problem, padded_state, config.restoration
    )
    residual = restoration_residuals(
        padded_problem, padded_state, config.restoration
    )
    mu = jnp.exp(padded_original.epsilon)
    real_x_indices = jnp.asarray([0, 1, 2, 4])

    assert padded_state.x[3] == pytest.approx(mu)
    assert padded_state.lower_bound_dual_x[3] == pytest.approx(1.0)
    assert jnp.array_equal(
        padded_state.proximity_mask,
        jnp.asarray([True, True, True, False, True]),
    )
    assert residual.dual_x[3] == pytest.approx(0.0)
    assert residual.complementarity_x[3] == pytest.approx(0.0)
    assert padded_direction.direction.x[3] == pytest.approx(0.0)
    assert padded_direction.direction.lower_bound_dual_x[3] == pytest.approx(
        0.0
    )
    assert padded_direction.direction.x[real_x_indices] == pytest.approx(
        direction.direction.x
    )

    def dummy_barrier_objective(amount):
        trial_x = padded_state.x.at[3].set(amount)
        return restoration_barrier_objective(
            padded_state._replace(x=trial_x), config.restoration
        )

    assert jax.grad(dummy_barrier_objective)(mu) == pytest.approx(0.0)


def test_restoration_schur_direction_matches_dense_kkt():
    problem, original = _fixture(epsilon=1.0e-2)
    config = FixedSupportV2Config(
        restoration=RestorationConfig(elastic_penalty=2.0)
    )
    initialized = initialize_restoration(problem, original, config)
    state = initialized._replace(
        x=initialized.x * jnp.asarray([1.1, 0.9, 1.2, 0.95]),
        positive_slack=initialized.positive_slack * jnp.asarray([1.1, 0.85]),
        negative_slack=initialized.negative_slack * jnp.asarray([0.9, 1.2]),
        equality_dual=jnp.asarray([0.2, -0.1]),
        lower_bound_dual_x=initialized.lower_bound_dual_x
        * jnp.asarray([1.3, 0.8, 1.1, 0.9]),
        lower_bound_dual_positive=initialized.lower_bound_dual_positive
        * jnp.asarray([0.75, 1.25]),
        lower_bound_dual_negative=initialized.lower_bound_dual_negative
        * jnp.asarray([1.2, 0.7]),
    )
    result = restoration_direction(problem, state, config.restoration)
    residual = restoration_residual_vector(
        restoration_residuals(problem, state, config.restoration)
    )
    dense = jnp.linalg.solve(
        restoration_kkt_jacobian(problem, state, config.restoration),
        -residual,
    )

    assert int(result.status) == TerminalStatus.NOT_TERMINATED
    assert _flat_direction(result.direction) == pytest.approx(dense, abs=2.0e-11)
    assert float(result.diagnostics.relative_schur_residual) < 1.0e-12
    assert float(result.diagnostics.full_kkt_residual_norm) < 1.0e-10


def test_restoration_merit_derivative_and_fraction_to_boundary_contracts():
    problem, original = _fixture(epsilon=1.0e-2)
    config = FixedSupportV2Config(
        restoration=RestorationConfig(elastic_penalty=2.0)
    )
    state = initialize_restoration(problem, original, config)
    direction = restoration_direction(
        problem, state, config.restoration
    ).direction
    predicted = restoration_barrier_directional_derivative(
        state, direction, config.restoration
    )

    def objective(x, p, v):
        return restoration_barrier_objective(
            state._replace(x=x, positive_slack=p, negative_slack=v),
            config.restoration,
        )

    actual = jax.jvp(
        objective,
        (state.x, state.positive_slack, state.negative_slack),
        (direction.x, direction.positive_slack, direction.negative_slack),
    )[1]
    alpha_primal, alpha_dual = restoration_step_limits(
        state, direction, config.restoration.fraction_to_boundary
    )
    alpha = jnp.minimum(alpha_primal, alpha_dual)

    assert predicted == pytest.approx(actual, abs=1.0e-12)
    assert jnp.all(state.x + alpha * direction.x > 0.0)
    assert jnp.all(
        state.positive_slack + alpha * direction.positive_slack > 0.0
    )
    assert jnp.all(
        state.negative_slack + alpha * direction.negative_slack > 0.0
    )
    assert jnp.all(
        state.lower_bound_dual_x + alpha * direction.lower_bound_dual_x > 0.0
    )


def test_second_iteration_consumes_persistent_first_iteration_state():
    problem, original = _fixture()
    config = FixedSupportV2Config(
        limits=SolverLimitConfig(max_restoration_line_search_trials=20)
    )
    initialized = initialize_restoration(problem, original, config)
    first = restoration_iteration(problem, initialized, config)
    expected_second_direction = restoration_direction(
        problem, first.state, config.restoration
    ).direction
    second = restoration_iteration(problem, first.state, config)

    assert bool(first.accepted)
    assert bool(second.accepted)
    assert int(first.state.accepted_iteration_count) == 1
    assert int(second.state.accepted_iteration_count) == 2
    assert jnp.array_equal(second.state.entry_x, initialized.entry_x)
    assert jnp.array_equal(
        second.state.variable_scales, initialized.variable_scales
    )
    assert first.state.positive_slack != pytest.approx(
        initialized.positive_slack
    )
    assert second.direction_result.direction.positive_slack == pytest.approx(
        expected_second_direction.positive_slack
    )


def test_standalone_restoration_reaches_typed_return_with_feasibility():
    problem, original = _fixture()
    config = FixedSupportV2Config(
        limits=SolverLimitConfig(
            max_restoration_iterations=50,
            max_restoration_line_search_trials=20,
        )
    )
    state = initialize_restoration(problem, original, config)
    result = solve_restoration(
        problem,
        state,
        empty_filter(config.limits.filter_capacity, dtype=state.x.dtype),
        config,
    )
    ng = original.q.shape[0]
    nc = original.r.shape[0]
    restored = original._replace(
        q=jnp.log(result.state.x[:ng]),
        r=jnp.log(result.state.x[ng : ng + nc]),
        qtot=jnp.log(result.state.x[-1]),
    )
    residual = residual_components(problem, restored)

    assert bool(result.return_accepted)
    assert int(result.status) == TerminalStatus.RESTORATION_RETURN_ACCEPTED
    assert int(result.state.accepted_iteration_count) > 1
    assert float(result.original_theta) < 0.1 * float(state.entry_theta)
    assert jnp.max(jnp.abs(residual.budget)) < 1.0e-8
    assert jnp.abs(residual.total_density[0] / 0.8) < 1.0e-8

    compiled = jax.jit(
        lambda p, s, f: solve_restoration(p, s, f, config)
    )(
        problem,
        state,
        empty_filter(config.limits.filter_capacity, dtype=state.x.dtype),
    )
    assert int(compiled.status) == TerminalStatus.RESTORATION_RETURN_ACCEPTED
    assert compiled.state.x == pytest.approx(result.state.x)

    resumed = solve_restoration(
        problem,
        result.state,
        empty_filter(config.limits.filter_capacity, dtype=state.x.dtype),
        config,
    )
    assert bool(resumed.return_accepted)
    assert int(resumed.status) == TerminalStatus.RESTORATION_RETURN_ACCEPTED
    assert int(resumed.state.iteration) == int(result.state.iteration)


def test_restoration_max_iteration_and_nonfinite_failures_are_typed():
    problem, original = _fixture()
    max_config = FixedSupportV2Config(
        restoration=RestorationConfig(
            required_reduction=1.0e-20,
            kkt_tolerance=0.0,
            budget_tolerance=0.0,
            total_density_tolerance=0.0,
        ),
        limits=SolverLimitConfig(
            max_restoration_iterations=1,
            max_restoration_line_search_trials=20,
        ),
    )
    state = initialize_restoration(problem, original, max_config)
    maxed = solve_restoration(
        problem,
        state,
        empty_filter(max_config.limits.filter_capacity, dtype=state.x.dtype),
        max_config,
    )

    nonfinite_original = original._replace(q=jnp.asarray([jnp.nan, -1000.0]))
    nonfinite_state = initialize_restoration(
        problem, nonfinite_original, FixedSupportV2Config()
    )
    nonfinite = solve_restoration(
        problem,
        nonfinite_state,
        empty_filter(2, dtype=state.x.dtype),
    )

    assert int(maxed.status) == TerminalStatus.RESTORATION_MAX_ITER
    assert int(nonfinite.status) == TerminalStatus.RESTORATION_NONFINITE


def test_restoration_trace_amounts_do_not_overflow_schur_elimination():
    problem, original = _fixture(epsilon=1.0e-5)
    config = FixedSupportV2Config()
    trace_original = original._replace(
        q=jnp.asarray([-1000.0, jnp.log(0.1)]),
        r=jnp.asarray([jnp.log(1.0e-300)]),
    )
    state = initialize_restoration(problem, trace_original, config)
    result = restoration_direction(problem, state, config.restoration)

    expected_push = (
        config.restoration.interior_push_fraction * state.variable_scales[0]
    )
    assert state.x[0] == pytest.approx(
        expected_push, rel=1.0e-15
    )
    assert state.x[0] > config.restoration.representation_floor
    assert jnp.all(jnp.isfinite(state.lower_bound_dual_x))
    assert float(jnp.max(state.lower_bound_dual_x)) < 1.0e50
    assert bool(result.diagnostics.raw_direction_finite)
    assert jnp.isfinite(result.diagnostics.relative_schur_residual)
    assert jnp.isfinite(result.diagnostics.relative_full_kkt_residual)
    assert int(result.status) == TerminalStatus.NOT_TERMINATED


def test_restoration_linear_failure_is_typed_without_sanitized_success():
    problem, original = _fixture()
    config = FixedSupportV2Config()
    state = initialize_restoration(problem, original, config)._replace(
        lower_bound_dual_positive=jnp.zeros((2,))
    )
    result = restoration_direction(problem, state, config.restoration)

    assert int(result.status) == TerminalStatus.RESTORATION_LINEAR_SOLVE_FAILED
    assert not bool(result.diagnostics.raw_direction_finite)


def test_restoration_line_search_failure_is_typed_without_best_trial_rescue():
    problem, original = _fixture()
    config = FixedSupportV2Config(
        restoration=RestorationConfig(armijo_fraction=1.0),
        limits=SolverLimitConfig(max_restoration_line_search_trials=1),
    )
    state = initialize_restoration(problem, original, config)
    result = restoration_iteration(problem, state, config)

    assert not bool(result.accepted)
    assert int(result.selected_index) == -1
    assert int(result.status) == TerminalStatus.RESTORATION_LINE_SEARCH_FAILED
    assert jnp.array_equal(result.state.x, state.x)
    assert jnp.array_equal(result.state.positive_slack, state.positive_slack)


def test_restoration_line_search_uses_barrier_merit_not_elastic_monotonicity():
    problem, original = _fixture(epsilon=1.0e-2)
    config = FixedSupportV2Config(
        restoration=RestorationConfig(elastic_penalty=2.0),
        limits=SolverLimitConfig(max_restoration_line_search_trials=20),
    )
    state = initialize_restoration(problem, original, config)
    for _ in range(4):
        iteration = restoration_iteration(problem, state, config)
        assert bool(iteration.accepted)
        state = iteration.state

    current_elastic = restoration_elastic_objective(
        state, config.restoration
    )
    iteration = restoration_iteration(problem, state, config)
    selected = int(iteration.selected_index)

    assert bool(iteration.accepted)
    assert bool(iteration.trials.objective_acceptable[selected])
    assert bool(iteration.trials.constraint_acceptable[selected])
    assert (
        iteration.trials.elastic_objective[selected] > current_elastic
    )
