"""Tests for condensate initializer support geometry utilities."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

from exogibbs.equilibrium.condensate.support_geometry import (
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
