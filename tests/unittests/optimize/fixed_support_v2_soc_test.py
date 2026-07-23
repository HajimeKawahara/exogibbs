import jax
import jax.numpy as jnp
import pytest

from exogibbs.optimize.fixed_support_v2.controller import (
    controller_step,
    initialize_controller,
)
from exogibbs.optimize.fixed_support_v2.filter import empty_filter
from exogibbs.optimize.fixed_support_v2.linear_solver import (
    reduced_direction_from_rhs,
)
from exogibbs.optimize.fixed_support_v2.normal import normal_step
from exogibbs.optimize.fixed_support_v2.problem import (
    canonical_gas_source,
    filter_violation,
    residual_components,
    residual_jacobian,
)
from exogibbs.optimize.fixed_support_v2.soc import exact_soc_step
from exogibbs.optimize.fixed_support_v2.types import (
    FilterConfig,
    FixedSupportProblem,
    FixedSupportV2Config,
    OriginalState,
    SOCConfig,
    SolverLimitConfig,
    SolverMode,
    TerminalStatus,
    TrialRejectionReason,
)

jax.config.update("jax_enable_x64", True)


def _fixture():
    ag = jnp.asarray([[1.0, 0.0, 1.0], [0.0, 2.0, 1.0]])
    ac = jnp.asarray([[1.0, 2.0], [1.0, 0.0]])
    target = jnp.asarray([0.42, 0.55])
    qtot = jnp.log(0.45)
    problem = FixedSupportProblem(
        gas_formula_matrix=ag,
        condensate_formula_matrix=ac,
        target_inventory=target,
        gamma=canonical_gas_source(jnp.asarray([0.12, -0.07, 0.21]), 0.3),
        condensate_standard_source=jnp.asarray([0.31, -0.22]),
        support_indices=jnp.asarray([2, 5], dtype=jnp.int32),
        budget_row_scale=1.0 / target,
        total_density_row_scale=1.0 / jnp.exp(qtot),
    )
    state = OriginalState(
        q=jnp.log(jnp.asarray([0.21, 0.18, 0.04])),
        r=jnp.log(jnp.asarray([0.025, 0.015])),
        lambda_=jnp.asarray([0.08, -0.11]),
        rho=jnp.log(jnp.asarray([0.004, 0.006])),
        qtot=qtot,
        epsilon=jnp.log(1.0e-7),
        iteration=jnp.asarray(3, dtype=jnp.int32),
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


def _finite_soc_fixture():
    problem = FixedSupportProblem(
        gas_formula_matrix=jnp.asarray(
            [[0.49302152332664795, 0.432691248310353]]
        ),
        condensate_formula_matrix=jnp.asarray([[0.49881787952147566]]),
        target_inventory=jnp.asarray([0.7417846284038427]),
        gamma=jnp.asarray([0.17984160211230824, 0.04976841615513102]),
        condensate_standard_source=jnp.asarray([2.026164128810794]),
        support_indices=jnp.asarray([0], dtype=jnp.int32),
        budget_row_scale=jnp.asarray([1.0 / 0.7417846284038427]),
        total_density_row_scale=jnp.asarray(1.0 / 0.8277107516608474),
    )
    state = OriginalState(
        q=jnp.log(jnp.asarray([0.43966721008714027, 0.14666067254281664])),
        r=jnp.log(jnp.asarray([0.17289349451898284])),
        lambda_=jnp.asarray([0.3874949048849519]),
        rho=jnp.asarray([0.41901971154528717]),
        qtot=jnp.log(0.8277107516608474),
        epsilon=jnp.asarray(-7.0),
        iteration=jnp.asarray(0, dtype=jnp.int32),
    )
    return problem, state


def _config(*, enabled=True, max_corrections=4, theta_max_factor=1.0e4):
    return FixedSupportV2Config(
        filter=FilterConfig(theta_max_factor=theta_max_factor),
        soc=SOCConfig(
            enabled=enabled,
            max_corrections=max_corrections,
            kappa_soc=0.99,
            fraction_to_boundary=0.995,
        ),
        limits=SolverLimitConfig(
            max_normal_iterations=20,
            max_line_search_trials=1,
            max_restoration_calls=2,
            max_restoration_iterations=50,
            max_restoration_line_search_trials=20,
        ),
    )


def _flatten(direction):
    return jnp.concatenate(
        [
            direction.q,
            direction.r,
            direction.lambda_,
            direction.rho,
            direction.qtot.reshape((1,)),
        ]
    )


def test_generic_rhs_reduced_direction_matches_dense_full_kkt():
    problem, state = _fixture()
    rhs_blocks = (
        jnp.asarray([0.3, -0.2, 0.1]),
        jnp.asarray([0.05, -0.08]),
        jnp.asarray([0.003, -0.002]),
        jnp.asarray([0.02, -0.03]),
        jnp.asarray([0.001]),
    )
    result = reduced_direction_from_rhs(
        problem,
        state,
        gas_rhs=rhs_blocks[0],
        condensate_rhs=rhs_blocks[1],
        budget_rhs=rhs_blocks[2],
        complementarity_rhs=rhs_blocks[3],
        total_density_rhs=rhs_blocks[4],
    )
    dense = jnp.linalg.solve(
        residual_jacobian(problem, state), -jnp.concatenate(rhs_blocks)
    )

    assert int(result.status) == TerminalStatus.NOT_TERMINATED
    assert _flatten(result.direction) == pytest.approx(dense, abs=2.0e-11)


def test_soc_uses_current_origin_and_separates_alpha_roles():
    problem, state = _finite_soc_fixture()
    config = _config()
    filter_state = empty_filter(config.limits.filter_capacity, state.q.dtype)
    normal = normal_step(
        problem,
        state,
        filter_state,
        initial_theta=filter_violation(problem, state),
        config=config,
    )
    result = exact_soc_step(
        problem,
        state,
        filter_state,
        normal,
        initial_theta=filter_violation(problem, state),
        config=config,
    )
    index = 0
    current = residual_components(problem, state)
    direction = reduced_direction_from_rhs(
        problem,
        state,
        gas_rhs=current.gas_stationarity,
        condensate_rhs=current.condensate_stationarity,
        budget_rhs=result.trials.budget_rhs[index],
        complementarity_rhs=current.complementarity,
        total_density_rhs=result.trials.total_density_rhs[index],
        failure_status=TerminalStatus.SOC_LINEAR_SOLVE_FAILED,
    ).direction
    trial = jax.tree_util.tree_map(lambda values: values[index], result.trials.states)

    assert bool(result.eligible)
    assert bool(result.accepted)
    assert int(result.base_trial_index) == 0
    assert float(result.trials.alpha_test[index]) == pytest.approx(1.0)
    assert float(result.trials.alpha_soc[index]) < 1.0
    assert result.trials.alpha_y[index] == pytest.approx(
        result.trials.alpha_soc[index]
    )
    assert trial.q == pytest.approx(
        state.q + result.trials.alpha_soc[index] * direction.q
    )
    assert trial.r == pytest.approx(
        state.r + result.trials.alpha_soc[index] * direction.r
    )
    assert trial.lambda_ == pytest.approx(
        state.lambda_ + result.trials.alpha_y[index] * direction.lambda_
    )
    assert trial.rho == pytest.approx(
        state.rho + result.trials.alpha_dual[index] * direction.rho
    )
    assert trial.qtot == pytest.approx(
        state.qtot + result.trials.alpha_soc[index] * direction.qtot
    )


def test_repeated_soc_uses_rhs_recurrence_and_kappa_stop():
    problem, state = _finite_soc_fixture()
    config = _config(max_corrections=4)
    capacity = config.limits.filter_capacity
    impossible_filter = empty_filter(capacity, state.q.dtype)._replace(
        phi_entries=jnp.full((capacity,), -1.0e100),
        theta_entries=jnp.full((capacity,), -1.0e100),
        valid_entries=jnp.ones((capacity,), dtype=bool),
    )
    initial_theta = filter_violation(problem, state)
    normal = normal_step(
        problem,
        state,
        impossible_filter,
        initial_theta=initial_theta,
        config=config,
    )
    result = exact_soc_step(
        problem,
        state,
        impossible_filter,
        normal,
        initial_theta=initial_theta,
        config=config,
    )
    first_residual = residual_components(
        problem,
        jax.tree_util.tree_map(lambda values: values[0], result.trials.states),
    )

    assert not bool(result.accepted)
    assert int(result.correction_count) == 4
    assert result.trials.kappa_continue.tolist() == [True, True, True, False]
    assert result.trials.budget_rhs[1] == pytest.approx(
        first_residual.budget
        + result.trials.alpha_soc[0] * result.trials.budget_rhs[0]
    )
    assert result.trials.total_density_rhs[1] == pytest.approx(
        first_residual.total_density[0]
        + result.trials.alpha_soc[0]
        * result.trials.total_density_rhs[0]
    )


def test_soc_blockwise_linearized_diagnostics_match_exact_rhs_equations():
    problem, state = _finite_soc_fixture()
    config = _config()
    filter_state = empty_filter(config.limits.filter_capacity, state.q.dtype)
    normal = normal_step(
        problem,
        state,
        filter_state,
        initial_theta=filter_violation(problem, state),
        config=config,
    )
    result = exact_soc_step(
        problem,
        state,
        filter_state,
        normal,
        initial_theta=filter_violation(problem, state),
        config=config,
    )
    current = residual_components(problem, state)
    direction = reduced_direction_from_rhs(
        problem,
        state,
        gas_rhs=current.gas_stationarity,
        condensate_rhs=current.condensate_stationarity,
        budget_rhs=result.trials.budget_rhs[0],
        complementarity_rhs=current.complementarity,
        total_density_rhs=result.trials.total_density_rhs[0],
        failure_status=TerminalStatus.SOC_LINEAR_SOLVE_FAILED,
    ).direction
    amounts = jax.tree_util.tree_map(
        jnp.exp, (state.q, state.r, state.rho, state.qtot)
    )
    scales = (
        jnp.max(jnp.abs(current.gas_stationarity))
        + jnp.max(jnp.abs(direction.q))
        + jnp.max(jnp.abs(problem.gas_formula_matrix.T @ direction.lambda_))
        + jnp.abs(direction.qtot),
        jnp.max(jnp.abs(current.condensate_stationarity))
        + jnp.max(
            jnp.abs(problem.condensate_formula_matrix.T @ direction.lambda_)
        )
        + jnp.max(jnp.abs(amounts[2] * direction.rho)),
        jnp.max(jnp.abs(result.trials.budget_rhs[0]))
        + jnp.max(
            jnp.abs(
                problem.gas_formula_matrix @ (amounts[0] * direction.q)
            )
        )
        + jnp.max(
            jnp.abs(
                problem.condensate_formula_matrix
                @ (amounts[1] * direction.r)
            )
        ),
        jnp.max(jnp.abs(current.complementarity))
        + jnp.max(jnp.abs(direction.r))
        + jnp.max(jnp.abs(direction.rho)),
        jnp.abs(result.trials.total_density_rhs[0])
        + jnp.abs(jnp.dot(amounts[0], direction.q))
        + jnp.abs(amounts[3] * direction.qtot),
    )
    recorded = [float(values[0]) for values in result.trials.linearized_residual_norms]

    for residual, scale in zip(recorded, scales):
        assert residual / max(float(scale), 1.0) < 2.0e-12


def test_controller_soc_rescues_rejected_normal_sequence_before_restoration():
    problem, state = _finite_soc_fixture()
    enabled = _config(enabled=True)
    disabled = _config(enabled=False)
    with_soc = jax.jit(lambda carry: controller_step(problem, carry, enabled))(
        initialize_controller(problem, state, enabled)
    )
    without_soc = controller_step(
        problem, initialize_controller(problem, state, disabled), disabled
    )

    assert int(with_soc.mode) == SolverMode.NORMAL
    assert int(with_soc.terminal_status) == TerminalStatus.NOT_TERMINATED
    assert int(with_soc.soc_attempt_count) == 1
    assert int(with_soc.soc_accepted_count) == 1
    assert int(with_soc.restoration_call_count) == 0
    assert int(without_soc.mode) == SolverMode.RESTORATION
    assert int(without_soc.terminal_status) == TerminalStatus.NOT_TERMINATED
    assert int(without_soc.restoration_call_count) == 1


def test_failed_soc_enters_unchanged_restoration_on_next_super_iteration():
    problem, state = _infeasible_fixture()
    config = _config(enabled=True, theta_max_factor=0.0)
    result = controller_step(
        problem, initialize_controller(problem, state, config), config
    )

    assert int(result.mode) == SolverMode.RESTORATION
    assert int(result.terminal_status) == TerminalStatus.NOT_TERMINATED
    assert int(result.soc_attempt_count) == 1
    assert int(result.soc_accepted_count) == 0
    assert int(result.restoration_call_count) == 1
    assert int(result.restoration_state.iteration) == 0


def test_soc_rejects_finite_log_dual_when_physical_multiplier_overflows():
    problem, state = _fixture()
    config = _config(enabled=True)
    filter_state = empty_filter(config.limits.filter_capacity, state.q.dtype)
    normal = normal_step(
        problem,
        state,
        filter_state,
        initial_theta=filter_violation(problem, state),
        config=config,
    )
    result = exact_soc_step(
        problem,
        state,
        filter_state,
        normal,
        initial_theta=filter_violation(problem, state),
        config=config,
    )

    assert bool(result.eligible)
    assert not bool(result.accepted)
    assert not bool(result.trials.finite[0])
    assert (
        int(result.trials.rejection_reasons[0])
        & int(TrialRejectionReason.NONFINITE)
    )


@pytest.mark.parametrize(
    "soc",
    [
        SOCConfig(max_corrections=0),
        SOCConfig(kappa_soc=0.0),
        SOCConfig(kappa_soc=1.0),
        SOCConfig(fraction_to_boundary=0.0),
        SOCConfig(fraction_to_boundary=1.1),
    ],
)
def test_soc_rejects_invalid_configuration(soc):
    problem, state = _fixture()
    config = FixedSupportV2Config(soc=soc)
    filter_state = empty_filter(config.limits.filter_capacity, state.q.dtype)
    normal = normal_step(
        problem,
        state,
        filter_state,
        initial_theta=filter_violation(problem, state),
        config=config,
    )
    with pytest.raises(ValueError):
        exact_soc_step(
            problem,
            state,
            filter_state,
            normal,
            initial_theta=filter_violation(problem, state),
            config=config,
        )
