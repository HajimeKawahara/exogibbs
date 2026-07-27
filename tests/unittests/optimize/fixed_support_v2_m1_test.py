import jax
import jax.numpy as jnp
import pytest

from exogibbs.optimize.fixed_support_v2.filter import (
    add_margin_adjusted_entry,
    empty_filter,
    reset_from_sequential_rejection_history,
)
from exogibbs.optimize.fixed_support_v2.linear_solver import (
    normal_reduced_direction,
)
from exogibbs.optimize.fixed_support_v2.normal import (
    normal_step,
    ordered_alpha_ladder,
    select_ordered_trial,
)
from exogibbs.optimize.fixed_support_v2.problem import (
    canonical_gas_source,
    filter_violation,
    residual_components,
    residual_jacobian,
    residual_vector,
)
from exogibbs.optimize.fixed_support_v2.types import (
    FilterConfig,
    FixedSupportProblem,
    FixedSupportV2Config,
    NormalConfig,
    OriginalState,
    SolverLimitConfig,
    TerminalStatus,
    TrialRejectionReason,
)

jax.config.update("jax_enable_x64", True)


def _fixture():
    ag = jnp.asarray([[1.0, 0.0, 1.0], [0.0, 2.0, 1.0]])
    ac = jnp.asarray([[1.0, 2.0], [1.0, 0.0]])
    target = jnp.asarray([0.42, 0.55])
    q = jnp.log(jnp.asarray([0.21, 0.18, 0.04]))
    r = jnp.log(jnp.asarray([0.025, 0.015]))
    qtot = jnp.log(0.45)
    problem = FixedSupportProblem(
        gas_formula_matrix=ag,
        condensate_formula_matrix=ac,
        target_inventory=target,
        gamma=canonical_gas_source(
            jnp.asarray([0.12, -0.07, 0.21]), 0.3
        ),
        condensate_standard_source=jnp.asarray([0.31, -0.22]),
        support_indices=jnp.asarray([2, 5], dtype=jnp.int32),
        budget_row_scale=1.0 / target,
        total_density_row_scale=1.0 / jnp.exp(qtot),
    )
    state = OriginalState(
        q=q,
        r=r,
        lambda_=jnp.asarray([0.08, -0.11]),
        rho=jnp.log(jnp.asarray([0.004, 0.006])),
        qtot=qtot,
        epsilon=jnp.log(1.0e-7),
        iteration=jnp.asarray(3, dtype=jnp.int32),
    )
    return problem, state


def _flatten_direction(direction):
    return jnp.concatenate(
        [
            direction.q,
            direction.r,
            direction.lambda_,
            direction.rho,
            direction.qtot.reshape((1,)),
        ]
    )


def test_normal_reduced_direction_matches_dense_kkt():
    problem, state = _fixture()
    residual = residual_components(problem, state)
    result = normal_reduced_direction(problem, state, residual)
    dense = jnp.linalg.solve(
        residual_jacobian(problem, state), -residual_vector(residual)
    )
    assert int(result.status) == TerminalStatus.NOT_TERMINATED
    assert _flatten_direction(result.direction) == pytest.approx(
        dense, abs=2.0e-11
    )
    assert float(result.diagnostics.relative_residual) < 1.0e-12


def test_singular_reduced_system_returns_typed_linear_failure():
    problem = FixedSupportProblem(
        gas_formula_matrix=jnp.zeros((1, 2)),
        condensate_formula_matrix=jnp.zeros((1, 1)),
        target_inventory=jnp.zeros((1,)),
        gamma=jnp.zeros((2,)),
        condensate_standard_source=jnp.zeros((1,)),
        support_indices=jnp.asarray([0], dtype=jnp.int32),
        budget_row_scale=jnp.ones((1,)),
        total_density_row_scale=jnp.asarray(1.0),
    )
    state = OriginalState(
        q=jnp.log(jnp.asarray([0.4, 0.6])),
        r=jnp.log(jnp.asarray([0.1])),
        lambda_=jnp.zeros((1,)),
        rho=jnp.log(jnp.asarray([0.1])),
        qtot=jnp.log(1.0),
        epsilon=jnp.log(1.0e-7),
        iteration=jnp.asarray(0, dtype=jnp.int32),
    )

    result = normal_reduced_direction(
        problem, state, residual_components(problem, state)
    )

    assert int(result.status) == TerminalStatus.NORMAL_LINEAR_SOLVE_FAILED
    assert not bool(result.diagnostics.raw_solution_finite)


def test_ordered_parallel_selection_matches_sequential_reference():
    alphas = ordered_alpha_ladder(
        NormalConfig(backtracking_factor=0.25), max_line_search_trials=5
    )
    accepted = jnp.asarray([False, False, True, True, False])
    reasons = jnp.asarray(
        [
            TrialRejectionReason.FILTER_HISTORY,
            TrialRejectionReason.CURRENT_POINT,
            TrialRejectionReason.NONE,
            TrialRejectionReason.NONE,
            TrialRejectionReason.FILTER_HISTORY,
        ],
        dtype=jnp.int32,
    )
    selection = select_ordered_trial(accepted, reasons, alphas)

    sequential_index = next(
        index for index, acceptable in enumerate(accepted.tolist()) if acceptable
    )
    assert alphas == pytest.approx(jnp.asarray([1.0, 0.25, 0.0625, 0.015625, 0.00390625]))
    assert int(selection.selected_index) == sequential_index
    assert float(selection.selected_alpha) == pytest.approx(float(alphas[2]))
    assert selection.rejected_prefix.tolist() == [True, True, False, False, False]
    assert int(selection.last_rejection_reason) == TrialRejectionReason.CURRENT_POINT


def test_no_acceptable_trial_has_typed_failure_and_no_rescue():
    accepted = jnp.zeros((4,), dtype=bool)
    reasons = jnp.full(
        (4,), int(TrialRejectionReason.CURRENT_POINT), dtype=jnp.int32
    )
    selection = select_ordered_trial(
        accepted, reasons, jnp.asarray([1.0, 0.5, 0.25, 0.125])
    )

    assert not bool(selection.accepted)
    assert int(selection.selected_index) == -1
    assert jnp.isnan(selection.selected_alpha)
    assert selection.rejected_prefix.tolist() == [True, True, True, True]
    assert int(selection.status) == TerminalStatus.NORMAL_LINE_SEARCH_FAILED


def test_filter_capacity_and_sequential_reset_contracts():
    config = FilterConfig(reset_trigger=2, max_resets=1)
    state = empty_filter(2)
    first = add_margin_adjusted_entry(
        state, phi=10.0, theta=2.0, config=config
    )
    second = add_margin_adjusted_entry(
        first.state, phi=9.0, theta=3.0, config=config
    )
    full = add_margin_adjusted_entry(
        second.state, phi=11.0, theta=4.0, config=config
    )

    assert not bool(first.capacity_exhausted)
    assert not bool(second.capacity_exhausted)
    assert bool(full.capacity_exhausted)
    assert jnp.array_equal(full.state.valid_entries, second.state.valid_entries)

    once = reset_from_sequential_rejection_history(
        second.state,
        step_accepted=True,
        last_rejection_was_history=True,
        config=config,
    )
    reset = reset_from_sequential_rejection_history(
        once,
        step_accepted=True,
        last_rejection_was_history=True,
        config=config,
    )
    assert int(once.successive_filter_rejections) == 1
    assert int(reset.successive_filter_rejections) == 0
    assert int(reset.reset_count) == 1
    assert not bool(jnp.any(reset.valid_entries))


def test_normal_step_selects_first_acceptable_current_origin_trial():
    problem, state = _fixture()
    config = FixedSupportV2Config(
        normal=NormalConfig(backtracking_factor=0.5),
        limits=SolverLimitConfig(max_line_search_trials=8),
    )
    result = normal_step(
        problem,
        state,
        empty_filter(8, dtype=state.q.dtype),
        initial_theta=filter_violation(problem, state),
        config=config,
    )
    first = next(
        index
        for index, acceptable in enumerate(result.trials.accepted.tolist())
        if acceptable
    )
    alpha = result.trials.alphas[first]
    direction = result.direction_result.direction

    assert bool(result.selection.accepted)
    assert int(result.selection.selected_index) == first
    assert result.trials.states.q[first] == pytest.approx(
        state.q + alpha * direction.q
    )
    assert result.trials.states.r[first] == pytest.approx(
        state.r + alpha * direction.r
    )
    assert result.trials.states.lambda_[first] == pytest.approx(
        state.lambda_ + alpha * direction.lambda_
    )
    assert result.trials.states.rho[first] == pytest.approx(
        state.rho + alpha * direction.rho
    )
    assert result.trials.states.qtot[first] == pytest.approx(
        state.qtot + alpha * direction.qtot
    )
    assert not hasattr(result.trials, "soc")
    assert not hasattr(result.trials, "restoration")

    compiled = jax.jit(
        lambda p, s, f, initial: normal_step(
            p, s, f, initial_theta=initial, config=config
        )
    )(
        problem,
        state,
        empty_filter(8, dtype=state.q.dtype),
        filter_violation(problem, state),
    )
    assert int(compiled.selection.selected_index) == first
    assert float(compiled.selection.selected_alpha) == pytest.approx(float(alpha))
