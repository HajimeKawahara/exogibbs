"""Tests for condensate initializer support geometry utilities."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

from exogibbs.equilibrium.condensate.support_geometry import (
    finite_barrier_trace_capacity_report,
    monotone_formula_row_mask,
    reduce_initial_condensate_support_to_basic,
)


_DIAGNOSTIC_FIELDS = {
    "role",
    "attempted",
    "applied",
    "input_support_rank",
    "output_support_rank",
    "output_dropped_support_indices",
    "output_scaled_inventory_residual_max_abs",
    "fallback_reason",
}


def test_monotone_formula_rows_use_the_joint_species_catalog() -> None:
    gas_formula = np.asarray([[1.0, 0.0], [1.0, -1.0]])
    condensate_formula = np.asarray([[2.0], [0.0]])

    mask = monotone_formula_row_mask(gas_formula, condensate_formula)
    sign_flipped = monotone_formula_row_mask(
        gas_formula * np.asarray([[1.0], [-1.0]]),
        condensate_formula * np.asarray([[1.0], [-1.0]]),
    )

    np.testing.assert_array_equal(mask, [True, False])
    np.testing.assert_array_equal(sign_flipped, [True, False])


def test_finite_barrier_report_detects_trace_phase_capacity() -> None:
    report = finite_barrier_trace_capacity_report(
        condensate_formula_matrix_full=np.asarray(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64
        ),
        target_inventory=np.asarray([1.0, 1.0e-12], dtype=np.float64),
        support_indices=(2, 0),
        monotone_constraint_row_mask=(True, True),
        log_barrier=np.log(1.0e-5),
    )

    assert report["schema"] == "exogibbs_finite_barrier_trace_capacity_v1"
    assert report["support_indices"] == (0, 2)
    assert report["support_count"] == 2
    assert report["finite_barrier_amount"] == pytest.approx(1.0e-5)
    assert report["support_phase_capacity_bounded"] == (True, True)
    assert report["support_phase_maximum_amounts"] == pytest.approx(
        (1.0, 1.0e-12)
    )
    assert report["capacity_to_barrier_ratios"] == pytest.approx(
        (1.0e5, 1.0e-7)
    )
    assert report["minimum_capacity_to_barrier_ratio"] == pytest.approx(1.0e-7)
    assert report["trace_capacity_support_indices"] == (2,)
    assert report["trace_capacity_count"] == 1
    assert report["trace_capacity_detected"]


def test_finite_barrier_report_is_support_order_deterministic() -> None:
    arguments = {
        "condensate_formula_matrix_full": np.asarray(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64
        ),
        "target_inventory": np.asarray([1.0, 0.5], dtype=np.float64),
        "monotone_constraint_row_mask": (True, True),
        "log_barrier": np.log(1.0e-5),
    }

    forward = finite_barrier_trace_capacity_report(
        **arguments, support_indices=(2, 0)
    )
    reverse = finite_barrier_trace_capacity_report(
        **arguments, support_indices=(0, 2)
    )
    empty = finite_barrier_trace_capacity_report(
        **arguments, support_indices=()
    )

    assert forward == reverse
    assert not forward["trace_capacity_detected"]
    assert forward["trace_capacity_support_indices"] == ()
    assert empty["support_indices"] == ()
    assert empty["support_phase_capacity_bounded"] == ()
    assert empty["support_phase_maximum_amounts"] == ()
    assert empty["capacity_to_barrier_ratios"] == ()
    assert empty["minimum_capacity_to_barrier_ratio"] is None
    assert not empty["trace_capacity_detected"]


def test_finite_barrier_report_ignores_signed_nonconsuming_rows() -> None:
    report = finite_barrier_trace_capacity_report(
        condensate_formula_matrix_full=np.asarray(
            [[1.0], [-1.0]], dtype=np.float64
        ),
        target_inventory=np.asarray([1.0e-12, -1.0], dtype=np.float64),
        support_indices=(0,),
        monotone_constraint_row_mask=(True, False),
        log_barrier=np.log(1.0e-5),
    )

    assert report["capacity_geometry_valid"]
    assert report["support_phase_maximum_amounts"] == pytest.approx((1.0e-12,))
    assert report["trace_capacity_detected"]


def test_finite_barrier_report_excludes_nonmonotone_positive_rows() -> None:
    report = finite_barrier_trace_capacity_report(
        condensate_formula_matrix_full=np.asarray(
            [[1.0], [1.0]], dtype=np.float64
        ),
        target_inventory=np.asarray([1.0, 0.0], dtype=np.float64),
        support_indices=(0,),
        monotone_constraint_row_mask=(True, False),
        log_barrier=np.log(1.0e-5),
    )

    assert report["monotone_constraint_row_mask"] == (True, False)
    assert report["support_phase_maximum_amounts"] == pytest.approx((1.0,))
    assert not report["trace_capacity_detected"]


def test_finite_barrier_report_leaves_unconsumed_phase_unbounded() -> None:
    report = finite_barrier_trace_capacity_report(
        condensate_formula_matrix_full=np.asarray([[1.0]], dtype=np.float64),
        target_inventory=np.asarray([0.0], dtype=np.float64),
        support_indices=(0,),
        monotone_constraint_row_mask=(False,),
        log_barrier=np.log(10.0),
    )

    assert report["capacity_geometry_valid"]
    assert report["support_phase_capacity_bounded"] == (False,)
    assert report["support_phase_maximum_amounts"] == (None,)
    assert report["capacity_to_barrier_ratios"] == (None,)
    assert report["minimum_capacity_to_barrier_ratio"] is None
    assert not report["trace_capacity_detected"]


def test_finite_barrier_report_fails_closed_for_consumed_negative_target() -> None:
    report = finite_barrier_trace_capacity_report(
        condensate_formula_matrix_full=np.asarray(
            [[1.0], [1.0]], dtype=np.float64
        ),
        target_inventory=np.asarray([1.0, -1.0], dtype=np.float64),
        support_indices=(0,),
        monotone_constraint_row_mask=(True, True),
        log_barrier=np.log(1.0e-5),
    )

    assert not report["capacity_geometry_valid"]
    assert not report["trace_capacity_detected"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        (
            {"condensate_formula_matrix_full": np.ones(2, dtype=np.float64)},
            "two-dimensional",
        ),
        (
            {"target_inventory": np.ones(3, dtype=np.float64)},
            "one value per element",
        ),
        (
            {"monotone_constraint_row_mask": (True,)},
            "one value per element",
        ),
        ({"support_indices": (0, 0)}, "unique catalog indices"),
        ({"support_indices": (2,)}, "unique catalog indices"),
        ({"log_barrier": np.inf}, "must be finite"),
        ({"log_barrier": -1000.0}, "positive finite amount"),
    ),
)
def test_finite_barrier_report_rejects_invalid_inputs(
    overrides: dict[str, Any], message: str
) -> None:
    arguments = {
        "condensate_formula_matrix_full": np.eye(2, dtype=np.float64),
        "target_inventory": np.ones(2, dtype=np.float64),
        "support_indices": (0,),
        "monotone_constraint_row_mask": (True, True),
        "log_barrier": -11.0,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        finite_barrier_trace_capacity_report(**arguments)


def _assert_uniform_diagnostics(
    report: dict[str, Any],
    *,
    role: str,
    attempted: bool,
    applied: bool,
    input_rank: int,
    output_rank: int,
    dropped: tuple[int, ...],
    fallback_reason: str | None,
) -> None:
    assert _DIAGNOSTIC_FIELDS <= report.keys()
    assert report["role"] == role
    assert report["attempted"] is attempted
    assert report["applied"] is applied
    assert report["input_support_rank"] == input_rank
    assert report["output_support_rank"] == output_rank
    assert report["output_dropped_support_indices"] == dropped
    assert np.isfinite(report["output_scaled_inventory_residual_max_abs"])
    assert report["fallback_reason"] == fallback_reason


def _rank_deficient_problem() -> dict[str, Any]:
    return {
        "condensate_formula_matrix_full": np.asarray(
            [[1.0, 1.0], [0.0, 0.0]], dtype=np.float64
        ),
        "condensate_standard_source_full": np.asarray(
            [0.0, 1.0], dtype=np.float64
        ),
        "target_inventory": np.asarray([1.0, 0.0], dtype=np.float64),
        "condensate_amounts": np.asarray([0.4, 0.6], dtype=np.float64),
        "support_indices": (0, 1),
        "budget_scale": np.asarray([1.0, 1.0], dtype=np.float64),
        "budget_tolerance": 1.0e-8,
        "enabled": True,
        "diagnostic_role": "finite_barrier_initializer",
    }


def test_reduction_preserves_scaled_inventory_and_returns_full_rank_support(
) -> None:
    formula = np.asarray(
        [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    amounts = np.asarray([0.2, 0.3, 0.4], dtype=np.float64)
    budget_scale = np.asarray([2.0, 2.5], dtype=np.float64)
    budget_tolerance = 1.0e-8

    support, reduced_amounts, report = (
        reduce_initial_condensate_support_to_basic(
            condensate_formula_matrix_full=formula,
            condensate_standard_source_full=np.asarray(
                [2.0, 1.0, 0.5], dtype=np.float64
            ),
            target_inventory=np.asarray([1.0, 1.0], dtype=np.float64),
            condensate_amounts=amounts,
            support_indices=(2, 0, 1),
            budget_scale=budget_scale,
            budget_tolerance=budget_tolerance,
            enabled=True,
            diagnostic_role="finite_barrier_initializer",
        )
    )

    scaled_residual = budget_scale * (
        formula @ reduced_amounts - formula @ amounts
    )
    positive_support = tuple(np.flatnonzero(reduced_amounts > 0.0))

    assert support == (1, 2)
    assert positive_support == (1, 2)
    assert np.max(np.abs(scaled_residual)) <= budget_tolerance
    assert np.linalg.matrix_rank(formula[:, positive_support]) == len(
        positive_support
    )
    _assert_uniform_diagnostics(
        report,
        role="finite_barrier_initializer",
        attempted=True,
        applied=True,
        input_rank=2,
        output_rank=2,
        dropped=(0,),
        fallback_reason=None,
    )
    assert report["output_scaled_inventory_residual_max_abs"] == pytest.approx(
        np.max(np.abs(scaled_residual))
    )


@pytest.mark.parametrize(
    ("formula", "target", "amounts", "support", "input_rank"),
    (
        (
            np.eye(2, dtype=np.float64),
            np.ones(2, dtype=np.float64),
            np.asarray([0.4, 0.6], dtype=np.float64),
            (1, 0),
            2,
        ),
        (
            np.asarray([[1.0, 1.0], [-1.0, -1.0]], dtype=np.float64),
            np.asarray([1.0, 0.0], dtype=np.float64),
            np.asarray([0.4, 0.6], dtype=np.float64),
            (0, 1),
            1,
        ),
    ),
    ids=("full-rank", "signed-stoichiometry"),
)
def test_noneligible_input_is_a_noop(
    formula: np.ndarray,
    target: np.ndarray,
    amounts: np.ndarray,
    support: tuple[int, ...],
    input_rank: int,
) -> None:
    def unexpected_solver(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("LP solver must not run for noneligible input")

    returned_support, returned_amounts, report = (
        reduce_initial_condensate_support_to_basic(
            condensate_formula_matrix_full=formula,
            condensate_standard_source_full=np.zeros(2, dtype=np.float64),
            target_inventory=target,
            condensate_amounts=amounts,
            support_indices=support,
            budget_scale=np.ones(formula.shape[0], dtype=np.float64),
            budget_tolerance=1.0e-8,
            enabled=True,
            diagnostic_role="zero_barrier_exact_solve",
            linear_program_solver=unexpected_solver,
        )
    )

    assert returned_support == support
    np.testing.assert_array_equal(returned_amounts, amounts)
    _assert_uniform_diagnostics(
        report,
        role="zero_barrier_exact_solve",
        attempted=False,
        applied=False,
        input_rank=input_rank,
        output_rank=input_rank,
        dropped=(),
        fallback_reason="not_rank_reduction_eligible",
    )


def test_empty_support_is_a_noop() -> None:
    support, amounts, report = reduce_initial_condensate_support_to_basic(
        condensate_formula_matrix_full=np.zeros((2, 0), dtype=np.float64),
        condensate_standard_source_full=np.zeros(0, dtype=np.float64),
        target_inventory=np.ones(2, dtype=np.float64),
        condensate_amounts=np.zeros(0, dtype=np.float64),
        support_indices=(),
        budget_scale=np.ones(2, dtype=np.float64),
        budget_tolerance=1.0e-8,
        enabled=True,
        diagnostic_role="finite_barrier_initializer",
    )

    assert support == ()
    np.testing.assert_array_equal(amounts, np.zeros(0, dtype=np.float64))
    _assert_uniform_diagnostics(
        report,
        role="finite_barrier_initializer",
        attempted=False,
        applied=False,
        input_rank=0,
        output_rank=0,
        dropped=(),
        fallback_reason="not_rank_reduction_eligible",
    )


def _raising_solver(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise RuntimeError("unit-test LP exception")


def _unsuccessful_solver(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return SimpleNamespace(
        success=False,
        status=2,
        message="unit-test infeasible LP",
        nit=3,
        x=np.zeros(2, dtype=np.float64),
    )


@pytest.mark.parametrize(
    ("solver", "fallback_reason"),
    (
        (_raising_solver, "solver_exception"),
        (_unsuccessful_solver, "linear_program_failed"),
    ),
    ids=("exception", "unsuccessful-result"),
)
def test_lp_failure_returns_the_original_support(
    solver: Callable[..., Any], fallback_reason: str
) -> None:
    problem = _rank_deficient_problem()
    original_amounts = problem["condensate_amounts"].copy()

    support, returned_amounts, report = (
        reduce_initial_condensate_support_to_basic(
            **problem, linear_program_solver=solver
        )
    )

    assert support == (0, 1)
    np.testing.assert_array_equal(returned_amounts, original_amounts)
    _assert_uniform_diagnostics(
        report,
        role="finite_barrier_initializer",
        attempted=True,
        applied=False,
        input_rank=1,
        output_rank=1,
        dropped=(),
        fallback_reason=fallback_reason,
    )


def test_postsolve_validation_failure_returns_the_original_support() -> None:
    def inventory_violating_solver(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return SimpleNamespace(
            success=True,
            status=0,
            message="unit-test inventory mismatch",
            nit=1,
            x=np.asarray([0.25, 0.0], dtype=np.float64),
        )

    problem = _rank_deficient_problem()
    original_amounts = problem["condensate_amounts"].copy()
    support, returned_amounts, report = (
        reduce_initial_condensate_support_to_basic(
            **problem, linear_program_solver=inventory_violating_solver
        )
    )

    assert support == (0, 1)
    np.testing.assert_array_equal(returned_amounts, original_amounts)
    assert report["scaled_inventory_residual_max_abs"] > report[
        "scaled_inventory_residual_tolerance"
    ]
    assert report["candidate_dropped_support_indices"] == (1,)
    _assert_uniform_diagnostics(
        report,
        role="finite_barrier_initializer",
        attempted=True,
        applied=False,
        input_rank=1,
        output_rank=1,
        dropped=(),
        fallback_reason="postsolve_validation_failed",
    )


def test_reduction_is_deterministic_under_reversed_input_support() -> None:
    common = {
        "condensate_formula_matrix_full": np.asarray(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        ),
        "condensate_standard_source_full": np.asarray(
            [2.0, 1.0, 0.5], dtype=np.float64
        ),
        "target_inventory": np.asarray([1.0, 1.0], dtype=np.float64),
        "condensate_amounts": np.asarray([0.2, 0.3, 0.4], dtype=np.float64),
        "budget_scale": np.asarray([1.0, 1.0], dtype=np.float64),
        "budget_tolerance": 1.0e-8,
        "enabled": True,
        "diagnostic_role": "finite_barrier_initializer",
    }

    forward = reduce_initial_condensate_support_to_basic(
        **common, support_indices=(2, 0, 1)
    )
    reverse = reduce_initial_condensate_support_to_basic(
        **common, support_indices=(1, 0, 2)
    )

    assert forward[0] == (1, 2)
    assert forward[0] == reverse[0]
    np.testing.assert_array_equal(forward[1], reverse[1])
    assert forward[2]["canonical_support_indices"] == (0, 1, 2)
    assert reverse[2]["canonical_support_indices"] == (0, 1, 2)
    assert forward[2]["output_dropped_support_indices"] == (0,)
    assert forward[2]["output_dropped_support_indices"] == reverse[2][
        "output_dropped_support_indices"
    ]
