"""Contracts separating one-support numerical attempts from phase selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from exogibbs.equilibrium.condensate.fixed_support import zero_barrier


_FORMULATIONS = ("normalized", "log", "dense")


def _attempt_arguments(formulation: str) -> dict[str, Any]:
    arguments = {
        "gas_formula_matrix": np.eye(2),
        "condensate_formula_matrix_full": np.asarray(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
        ),
        "target_inventory": np.ones(2),
        "gas_standard_source": np.zeros(2),
        "condensate_standard_source_full": np.zeros(3),
        "gas_log_amounts_init": np.log(np.asarray([0.5, 0.5])),
        "condensate_amounts_init": np.asarray([0.5, 0.5, 0.0]),
        "total_gas_log_amount_init": 0.0,
        "element_potential_init": np.zeros(2),
        "support_indices": (1, 0),
        "budget_scale": np.ones(2),
        "max_function_evaluations": 3,
    }
    if formulation == "log":
        arguments["budget_tolerance"] = 1.0e-8
    return arguments


@pytest.mark.parametrize("formulation", _FORMULATIONS)
def test_numerical_attempt_preserves_support_and_leaves_audit_to_controller(
    monkeypatch: pytest.MonkeyPatch, formulation: str
) -> None:
    calls = []

    def solve(residual, initial, **kwargs):
        del residual
        calls.append(initial.copy())
        values = initial.copy()
        phase_offset = 2 if formulation == "dense" else 3
        values[phase_offset] = (
            kwargs["bounds"][0][phase_offset]
            if formulation == "log"
            else -0.25
        )
        return OptimizeResult(
            x=values,
            success=True,
            status=1,
            message="unit-test numerical termination",
            nfev=2,
            cost=0.0,
            optimality=0.0,
        )

    def forbidden_audit(**kwargs):
        del kwargs
        raise AssertionError("The numerical kernel must not certify a state.")

    monkeypatch.setattr(zero_barrier, "_least_squares_with_scipy_overflow_guard", solve)
    monkeypatch.setattr(zero_barrier, "_audit_zero_barrier_state", forbidden_audit)
    arguments = _attempt_arguments(formulation)
    initial_amounts = arguments["condensate_amounts_init"].copy()
    budget = zero_barrier._FunctionEvaluationBudget(4)
    attempt = getattr(zero_barrier, f"_solve_{formulation}_support_once")(
        **arguments, function_evaluation_budget=budget
    )

    assert len(calls) == 1
    assert attempt.support_indices == (1, 0)
    assert attempt.state is not None
    assert attempt.function_evaluations == budget.used == 2
    assert attempt.optimizer_success
    assert "physical_root_certified" not in attempt.report
    assert "drop_authorized_by_root" not in attempt.report
    assert attempt.state.condensate_amounts[2] == 0.0
    np.testing.assert_array_equal(arguments["condensate_amounts_init"], initial_amounts)
    if formulation == "log":
        assert attempt.lower_bound_support_indices == (1,)
        assert attempt.state.condensate_amounts[1] > 0.0
    else:
        assert attempt.state.condensate_amounts[1] == -0.25


@pytest.mark.parametrize("formulation", _FORMULATIONS)
def test_numerical_attempt_charges_exception_without_changing_support(
    monkeypatch: pytest.MonkeyPatch, formulation: str
) -> None:
    def failed_solve(*args, **kwargs):
        del args, kwargs
        raise ValueError("unit-test optimizer exception")

    monkeypatch.setattr(
        zero_barrier, "_least_squares_with_scipy_overflow_guard", failed_solve
    )
    budget = zero_barrier._FunctionEvaluationBudget(4)
    attempt = getattr(zero_barrier, f"_solve_{formulation}_support_once")(
        **_attempt_arguments(formulation), function_evaluation_budget=budget
    )

    assert attempt.state is None
    assert attempt.support_indices == (1, 0)
    assert attempt.function_evaluations == budget.used == 3
    assert attempt.report["function_evaluations_conservative"]
    assert not attempt.optimizer_success
    assert attempt.optimizer_status is None


@pytest.mark.parametrize("formulation", _FORMULATIONS)
def test_numerical_attempt_skips_optimizer_when_budget_is_exhausted(
    monkeypatch: pytest.MonkeyPatch, formulation: str
) -> None:
    def forbidden_solve(*args, **kwargs):
        del args, kwargs
        raise AssertionError("An exhausted attempt must not invoke the optimizer.")

    monkeypatch.setattr(
        zero_barrier, "_least_squares_with_scipy_overflow_guard", forbidden_solve
    )
    budget = zero_barrier._FunctionEvaluationBudget(3, used=3)
    attempt = getattr(zero_barrier, f"_solve_{formulation}_support_once")(
        **_attempt_arguments(formulation), function_evaluation_budget=budget
    )

    assert attempt.state is None
    assert attempt.support_indices == (1, 0)
    assert attempt.function_evaluations == 0
    assert budget.used == 3
    assert attempt.report["failure_reason"] == "function_evaluation_limit_reached"


def test_variable_scaling_retry_preserves_the_requested_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scales = []

    def solve(residual, initial, **kwargs):
        del residual
        scales.append(kwargs["x_scale"].copy())
        values = initial.copy()
        values[-1] = -0.25
        return OptimizeResult(
            x=values, success=True, status=1, message="done", nfev=1,
            cost=0.0, optimality=0.0,
        )

    monkeypatch.setattr(zero_barrier, "_least_squares_with_scipy_overflow_guard", solve)
    arguments = _attempt_arguments("normalized")
    arguments["element_potential_init"] = np.asarray([4.0, 9.0])
    budget = zero_barrier._FunctionEvaluationBudget(2)
    attempts = [
        zero_barrier._solve_normalized_support_once(
            **arguments,
            variable_scaling=scaling,
            function_evaluation_budget=budget,
        )
        for scaling in ("initializer_relative", "dimensionless_unit")
    ]

    assert [attempt.support_indices for attempt in attempts] == [(1, 0), (1, 0)]
    assert all(attempt.state.condensate_amounts[0] < 0.0 for attempt in attempts)
    assert budget.used == 2
    np.testing.assert_array_equal(scales[0][:2], np.asarray([4.0, 9.0]))
    np.testing.assert_array_equal(scales[1], np.ones_like(scales[1]))
