"""Tests for the physical zero-barrier active-support refinement."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from exogibbs.equilibrium.condensate.fixed_support import zero_barrier
from exogibbs.equilibrium.condensate.fixed_support.zero_barrier import (
    ZeroBarrierPolishResult,
    _least_squares_with_scipy_overflow_guard,
    _physical_zero_barrier_audit,
    _solve_normalized_gas_reduced_linear_support,
    _solve_reduced_log_domain_active_support,
    _solve_reduced_log_domain_support_branches,
    polish_zero_barrier_active_support,
)


def _disable_dual_support_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = zero_barrier._select_support_with_zero_barrier_dual

    def disabled(**kwargs):
        return original(**(kwargs | {"enabled": False}))

    monkeypatch.setattr(
        zero_barrier,
        "_select_support_with_zero_barrier_dual",
        disabled,
    )


def test_least_squares_guard_is_local_to_scipy_scalar_divide_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()

    def warning_solver(*args, **kwargs):
        del args, kwargs
        warnings.warn_explicit(
            "overflow encountered in scalar divide",
            RuntimeWarning,
            filename="scipy/optimize/_lsq/common.py",
            lineno=1,
            module="scipy.optimize._lsq.common",
        )
        return sentinel

    monkeypatch.setattr(
        "exogibbs.equilibrium.condensate.fixed_support.zero_barrier."
        "least_squares",
        warning_solver,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = _least_squares_with_scipy_overflow_guard(
            lambda values: values,
            np.ones(1, dtype=np.float64),
        )
        assert result is sentinel
        with pytest.raises(RuntimeWarning, match="scalar divide"):
            warnings.warn_explicit(
                "overflow encountered in scalar divide",
                RuntimeWarning,
                filename="other_solver.py",
                lineno=1,
                module="other_solver",
            )


def _one_active_phase_problem():
    gas_formula = np.asarray(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    condensate_formula = np.eye(2, dtype=np.float64)
    expected_gas = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
    expected_condensates = np.asarray([0.7, 0.0], dtype=np.float64)
    expected_potential = np.asarray([0.4, -0.2], dtype=np.float64)
    gamma = (
        gas_formula.T @ expected_potential - np.log(expected_gas)
    )
    hcond = np.asarray([0.4, -0.1], dtype=np.float64)
    target = (
        gas_formula @ expected_gas
        + condensate_formula @ expected_condensates
    )
    return (
        gas_formula,
        condensate_formula,
        expected_gas,
        expected_condensates,
        expected_potential,
        gamma,
        hcond,
        target,
    )


def test_zero_barrier_polish_restores_all_physical_kkt_blocks() -> None:
    (
        gas_formula,
        condensate_formula,
        expected_gas,
        expected_condensates,
        expected_potential,
        gamma,
        hcond,
        target,
    ) = _one_active_phase_problem()

    result = polish_zero_barrier_active_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond,
        gas_log_amounts_init=np.log(expected_gas) + 2.0e-2,
        condensate_amounts_init=np.asarray([0.65, 0.0]),
        total_gas_log_amount_init=1.0e-2,
        element_potential_init=expected_potential + 1.0e-2,
        support_indices=(0,),
        budget_relative_floor=1.0e-12,
    )

    assert result.accepted
    assert result.support_indices == (0,)
    assert np.exp(result.gas_log_amounts) == pytest.approx(expected_gas)
    assert result.condensate_amounts == pytest.approx(expected_condensates)
    assert result.element_potential == pytest.approx(expected_potential)
    assert result.report["active_condensate_driving_max_abs"] < 1.0e-10
    assert result.report["inactive_condensate_violation_max_abs"] == 0.0
    assert result.report["budget_scaled_max_abs"] < 1.0e-10
    assert result.report["total_density_scaled_abs"] < 1.0e-10
    assert result.report["zero_barrier_dual_support_oracle"]["applied"]
    assert not result.report["finite_barrier_homotopy_initializer"][
        "enabled"
    ]
    assert not result.report["alternative_basic_support_portfolio"][
        "enabled"
    ]
    assert not result.report["alternative_basic_support_portfolio"][
        "attempted"
    ]


@pytest.mark.parametrize("amount_scale", (1.0e-12, 1.0, 1.0e8))
def test_zero_barrier_dual_support_oracle_is_amount_gauge_covariant(
    amount_scale: float,
) -> None:
    evaluation_budget = zero_barrier._FunctionEvaluationBudget(limit=20)
    result = zero_barrier._select_support_with_zero_barrier_dual(
        gas_formula_matrix=np.eye(2, dtype=np.float64),
        condensate_formula_matrix_full=np.eye(2, dtype=np.float64),
        target_inventory=amount_scale * np.ones(2, dtype=np.float64),
        gas_standard_source=np.full(2, np.log(2.0), dtype=np.float64),
        condensate_standard_source_full=np.asarray([0.0, 1.0]),
        gas_log_amounts_init=np.log(
            amount_scale * np.asarray([0.5, 0.5])
        ),
        condensate_amounts_init=amount_scale
        * np.asarray([0.5, 0.0]),
        total_gas_log_amount_init=float(np.log(amount_scale)),
        element_potential_init=np.zeros(2, dtype=np.float64),
        condensate_valid_mask=np.ones(2, dtype=bool),
        stationarity_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=20,
        enabled=True,
        function_evaluation_budget=evaluation_budget,
    )

    report = result["report"]
    assert result["applied"]
    assert result["support_indices"] == (0,)
    np.testing.assert_allclose(
        result["element_potential"], np.zeros(2), atol=1.0e-12
    )
    np.testing.assert_allclose(
        np.exp(result["gas_log_amounts"]) / amount_scale,
        np.asarray([0.5, 0.5]),
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        result["condensate_amounts"] / amount_scale,
        np.asarray([0.5, 0.0]),
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    assert result["total_gas_log_amount"] - np.log(amount_scale) == (
        pytest.approx(0.0, abs=1.0e-12)
    )
    assert report["dual_feasibility_passed"]
    assert report["support_structure_passed"]
    assert report["smallest_inactive_driving"] == pytest.approx(1.0)
    assert evaluation_budget.used == report["function_evaluations"]


def test_zero_barrier_dual_support_oracle_excludes_structural_zero_phases(
) -> None:
    result = zero_barrier._select_support_with_zero_barrier_dual(
        gas_formula_matrix=np.asarray([[1.0], [0.0]]),
        condensate_formula_matrix_full=np.eye(2, dtype=np.float64),
        target_inventory=np.asarray([1.0, 0.0]),
        gas_standard_source=np.zeros(1, dtype=np.float64),
        condensate_standard_source_full=np.asarray([0.0, -100.0]),
        gas_log_amounts_init=np.zeros(1, dtype=np.float64),
        condensate_amounts_init=np.asarray([0.5, 0.5]),
        total_gas_log_amount_init=0.0,
        element_potential_init=np.zeros(2, dtype=np.float64),
        condensate_valid_mask=np.ones(2, dtype=bool),
        stationarity_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=20,
        enabled=True,
    )

    report = result["report"]
    assert result["applied"]
    assert result["support_indices"] == (0,)
    assert report["zero_target_rows"] == (1,)
    assert report["selectable_condensate_count"] == 1
    assert result["condensate_amounts"][1] == 0.0


def test_zero_barrier_dual_support_oracle_charges_optimizer_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raising_optimizer(*args, **kwargs):
        del args, kwargs
        raise ValueError("unit-test optimizer failure")

    monkeypatch.setattr(zero_barrier, "minimize", raising_optimizer)
    evaluation_budget = zero_barrier._FunctionEvaluationBudget(limit=5)
    result = zero_barrier._select_support_with_zero_barrier_dual(
        gas_formula_matrix=np.eye(2, dtype=np.float64),
        condensate_formula_matrix_full=np.eye(2, dtype=np.float64),
        target_inventory=np.ones(2, dtype=np.float64),
        gas_standard_source=np.full(2, np.log(2.0), dtype=np.float64),
        condensate_standard_source_full=np.asarray([0.0, 1.0]),
        gas_log_amounts_init=np.log(np.asarray([0.5, 0.5])),
        condensate_amounts_init=np.asarray([0.5, 0.0]),
        total_gas_log_amount_init=0.0,
        element_potential_init=np.zeros(2, dtype=np.float64),
        condensate_valid_mask=np.ones(2, dtype=bool),
        stationarity_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=5,
        enabled=True,
        function_evaluation_budget=evaluation_budget,
    )

    report = result["report"]
    assert not result["applied"]
    assert report["failure_reason"] == "optimizer_exception"
    assert report["function_evaluations_conservative"]
    assert report["function_evaluations"] == 5
    assert evaluation_budget.used == evaluation_budget.limit


@pytest.mark.parametrize("amount_scale", (1.0e-12, 1.0, 1.0e8))
def test_zero_barrier_dual_adds_phase_absent_from_initial_support(
    amount_scale: float,
) -> None:
    gas_amounts = amount_scale * np.asarray([0.4, 0.1])
    result = polish_zero_barrier_active_support(
        gas_formula_matrix=np.eye(2, dtype=np.float64),
        condensate_formula_matrix_full=np.eye(2, dtype=np.float64),
        target_inventory=amount_scale * np.asarray([1.0, 0.1]),
        gas_standard_source=np.log(np.asarray([1.25, 5.0])),
        condensate_standard_source_full=np.asarray([0.0, 1.0]),
        gas_log_amounts_init=np.log(gas_amounts),
        condensate_amounts_init=amount_scale
        * np.asarray([0.0, 0.6]),
        total_gas_log_amount_init=float(np.log(np.sum(gas_amounts))),
        element_potential_init=np.zeros(2, dtype=np.float64),
        support_indices=(1,),
    )

    assert result.accepted
    assert result.support_indices == (0,)
    assert result.report["zero_barrier_dual_support_oracle"]["applied"]
    np.testing.assert_allclose(
        np.exp(result.gas_log_amounts) / amount_scale,
        np.asarray([0.4, 0.1]),
        rtol=1.0e-11,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        result.condensate_amounts / amount_scale,
        np.asarray([0.6, 0.0]),
        rtol=1.0e-11,
        atol=1.0e-13,
    )
    assert result.report["inactive_condensate_violation_max_abs"] == 0.0


def test_zero_barrier_dual_constraint_jacobians_match_finite_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_minimize = zero_barrier.minimize
    checked = False

    def checking_minimize(
        objective, values, *, jac, constraints, method, options
    ):
        nonlocal checked
        probe = np.asarray([0.1, -0.2], dtype=np.float64)
        step = 1.0e-6
        for constraint in constraints:
            analytic = np.asarray(constraint["jac"](probe))
            columns = []
            for index in range(probe.size):
                plus = probe.copy()
                minus = probe.copy()
                plus[index] += step
                minus[index] -= step
                columns.append(
                    (
                        np.asarray(constraint["fun"](plus))
                        - np.asarray(constraint["fun"](minus))
                    )
                    / (2.0 * step)
                )
            numerical = np.stack(columns, axis=-1)
            np.testing.assert_allclose(
                analytic, numerical, rtol=2.0e-6, atol=2.0e-8
            )
        checked = True
        return original_minimize(
            objective,
            values,
            jac=jac,
            constraints=constraints,
            method=method,
            options=options,
        )

    monkeypatch.setattr(zero_barrier, "minimize", checking_minimize)
    result = zero_barrier._select_support_with_zero_barrier_dual(
        gas_formula_matrix=np.eye(2, dtype=np.float64),
        condensate_formula_matrix_full=np.eye(2, dtype=np.float64),
        target_inventory=np.asarray([1.0, 0.1]),
        gas_standard_source=np.log(np.asarray([1.25, 5.0])),
        condensate_standard_source_full=np.asarray([0.0, 1.0]),
        gas_log_amounts_init=np.log(np.asarray([0.4, 0.1])),
        condensate_amounts_init=np.asarray([0.6, 0.0]),
        total_gas_log_amount_init=float(np.log(0.5)),
        element_potential_init=np.zeros(2, dtype=np.float64),
        condensate_valid_mask=np.ones(2, dtype=bool),
        stationarity_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=20,
        enabled=True,
    )

    assert checked
    assert result["applied"]
    assert result["support_indices"] == (0,)


def test_zero_barrier_dual_rejects_infeasible_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InfeasibleOptimization:
        x = np.asarray([10.0, 10.0])
        success = True
        status = 0
        message = "synthetic infeasible success"
        nit = 1

    monkeypatch.setattr(
        zero_barrier,
        "minimize",
        lambda *args, **kwargs: InfeasibleOptimization(),
    )
    result = zero_barrier._select_support_with_zero_barrier_dual(
        gas_formula_matrix=np.eye(2, dtype=np.float64),
        condensate_formula_matrix_full=np.eye(2, dtype=np.float64),
        target_inventory=np.asarray([1.0, 0.1]),
        gas_standard_source=np.log(np.asarray([1.25, 5.0])),
        condensate_standard_source_full=np.asarray([0.0, 1.0]),
        gas_log_amounts_init=np.log(np.asarray([0.4, 0.1])),
        condensate_amounts_init=np.asarray([0.6, 0.0]),
        total_gas_log_amount_init=float(np.log(0.5)),
        element_potential_init=np.zeros(2, dtype=np.float64),
        condensate_valid_mask=np.ones(2, dtype=bool),
        stationarity_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=20,
        enabled=True,
    )

    assert not result["applied"]
    assert result["report"]["optimizer_success"]
    assert not result["report"]["dual_feasibility_passed"]
    assert result["report"]["failure_reason"] == "dual_feasibility_failed"


def test_zero_barrier_dual_rejects_rank_deficient_tight_support() -> None:
    result = zero_barrier._select_support_with_zero_barrier_dual(
        gas_formula_matrix=np.ones((1, 1)),
        condensate_formula_matrix_full=np.ones((1, 2)),
        target_inventory=np.ones(1),
        gas_standard_source=np.zeros(1),
        condensate_standard_source_full=np.zeros(2),
        gas_log_amounts_init=np.zeros(1),
        condensate_amounts_init=np.asarray([0.5, 0.5]),
        total_gas_log_amount_init=0.0,
        element_potential_init=np.zeros(1),
        condensate_valid_mask=np.ones(2, dtype=bool),
        stationarity_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=20,
        enabled=True,
    )

    assert not result["applied"]
    assert result["report"]["selected_support_indices"] == (0, 1)
    assert not result["report"]["support_structure_passed"]
    assert result["report"]["failure_reason"] == (
        "selected_support_not_full_column_rank"
    )


def test_normalized_gas_reduced_jacobian_and_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        gas_formula,
        condensate_formula,
        expected_gas,
        expected_condensates,
        expected_potential,
        gamma,
        hcond,
        target,
    ) = _one_active_phase_problem()
    original_solver = zero_barrier._least_squares_with_scipy_overflow_guard
    checked = False

    def checking_solver(residual, values, *, jac, **kwargs):
        nonlocal checked
        analytic = jac(values)
        numerical = np.empty_like(analytic)
        for column in range(values.size):
            step = 1.0e-6 * max(1.0, abs(float(values[column])))
            forward = values.copy()
            backward = values.copy()
            forward[column] += step
            backward[column] -= step
            numerical[:, column] = (
                residual(forward) - residual(backward)
            ) / (2.0 * step)
        np.testing.assert_allclose(analytic, numerical, rtol=2.0e-6, atol=2.0e-8)
        assert values.size == gas_formula.shape[0] + 1 + 1
        checked = True
        return original_solver(residual, values, jac=jac, **kwargs)

    monkeypatch.setattr(
        zero_barrier,
        "_least_squares_with_scipy_overflow_guard",
        checking_solver,
    )
    reduced = _solve_normalized_gas_reduced_linear_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond,
        gas_log_amounts_init=np.log(expected_gas) + 1.0e-3,
        condensate_amounts_init=expected_condensates,
        total_gas_log_amount_init=1.0e-3,
        element_potential_init=expected_potential + 1.0e-3,
        support_indices=(0,),
        condensate_valid_mask=np.ones(2, dtype=bool),
        budget_scale=np.reciprocal(target),
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=400,
    )

    attempt = reduced["report"]["attempts"][0]
    assert checked
    assert reduced["accepted"]
    assert attempt["reduced_variable_count"] == 4
    assert attempt["eliminated_gas_variable_count"] == 3


def test_normalized_linear_domain_accepts_signed_exact_zero_row() -> None:
    result = zero_barrier._normalized_linear_domain_eligibility(
        gas_formula_matrix=np.asarray([[1.0, 1.0], [1.0, -1.0]]),
        target_inventory=np.asarray([1.0, 0.0]),
    )

    assert result == (True, "eligible")


def test_gas_capacities_ignore_signed_constraint_rows() -> None:
    target = np.asarray([1.0, 0.0])
    formula = np.asarray([[1.0, 1.0], [1.0, -1.0]])

    capacities = zero_barrier._gas_elemental_capacities(
        formula,
        target,
        np.asarray([True, False]),
    )
    sign_flipped = zero_barrier._gas_elemental_capacities(
        formula * np.asarray([[1.0], [-1.0]]),
        target,
        np.asarray([True, False]),
    )

    np.testing.assert_array_equal(capacities, [1.0, 1.0])
    np.testing.assert_array_equal(sign_flipped, capacities)


@pytest.mark.parametrize(
    ("formula", "target", "expected_reason"),
    (
        (
            np.eye(2),
            np.asarray([1.0, -1.0]),
            "invalid_normalized_linear_structure",
        ),
        (
            np.eye(2),
            np.asarray([1.0]),
            "invalid_normalized_linear_structure",
        ),
        (
            np.asarray([[1.0, np.nan]]),
            np.asarray([1.0]),
            "invalid_normalized_linear_structure",
        ),
        (np.eye(2), np.zeros(2), "no_positive_target_row"),
    ),
)
def test_normalized_linear_domain_rejects_invalid_inputs(
    formula: np.ndarray,
    target: np.ndarray,
    expected_reason: str,
) -> None:
    result = zero_barrier._normalized_linear_domain_eligibility(
        gas_formula_matrix=formula,
        target_inventory=target,
    )

    assert result == (False, expected_reason)


def test_self_reopening_drop_uses_strict_tolerance_and_valid_mask() -> None:
    solve_report = {"dropped_support_indices": (0, 1)}
    candidate = {
        "audit": {"full_driving": np.asarray([-1.1e-8, -1.0e-8])}
    }

    reopened = zero_barrier._self_reopening_dropped_support_indices(
        solve_report=solve_report,
        candidate=candidate,
        condensate_valid_mask=np.asarray([True, True]),
        support_closure_tolerance=1.0e-8,
    )
    invalid_phase_excluded = (
        zero_barrier._self_reopening_dropped_support_indices(
            solve_report=solve_report,
            candidate=candidate,
            condensate_valid_mask=np.asarray([False, True]),
            support_closure_tolerance=1.0e-8,
        )
    )

    assert reopened == (0,)
    assert invalid_phase_excluded == ()


@pytest.mark.parametrize(
    ("solve_report", "candidate", "valid_mask"),
    (
        ({"dropped_support_indices": (0,)}, None, np.ones(1, dtype=bool)),
        ({"dropped_support_indices": (0,)}, {}, np.ones(1, dtype=bool)),
        (
            {"dropped_support_indices": (0,)},
            {"audit": {"full_driving": np.asarray([np.nan])}},
            np.ones(1, dtype=bool),
        ),
        (
            {"dropped_support_indices": (1,)},
            {"audit": {"full_driving": np.zeros(1)}},
            np.ones(1, dtype=bool),
        ),
    ),
)
def test_self_reopening_drop_rejects_invalid_candidates(
    solve_report: dict,
    candidate: dict | None,
    valid_mask: np.ndarray,
) -> None:
    assert (
        zero_barrier._self_reopening_dropped_support_indices(
            solve_report=solve_report,
            candidate=candidate,
            condensate_valid_mask=valid_mask,
            support_closure_tolerance=1.0e-8,
        )
        == ()
    )


@pytest.mark.parametrize(
    ("candidate", "local_kkt_passed", "expected"),
    (
        (
            None,
            False,
            (False, "candidate_unavailable"),
        ),
        (
            {"optimizer_success": True, "optimizer_status": 1, "audit": {}},
            False,
            (False, "optimizer_succeeded"),
        ),
        (
            {
                "optimizer_success": False,
                "optimizer_status": -1,
                "audit": {"finite": True},
            },
            False,
            (False, "not_function_evaluation_limit"),
        ),
        (
            {
                "optimizer_success": False,
                "optimizer_status": 0,
                "audit": {"finite": False},
            },
            False,
            (False, "terminal_candidate_not_finite"),
        ),
        (
            {
                "optimizer_success": False,
                "optimizer_status": 0,
                "audit": {"finite": True},
            },
            True,
            (False, "local_kkt_already_satisfied"),
        ),
        (
            {
                "optimizer_success": False,
                "optimizer_status": 0,
                "audit": {
                    "finite": True,
                    "positive_active_amounts": False,
                },
            },
            False,
            (False, "terminal_active_amounts_not_positive"),
        ),
        (
            {
                "optimizer_success": False,
                "optimizer_status": 0,
                "audit": {
                    "finite": True,
                    "positive_active_amounts": True,
                },
            },
            False,
            (True, "finite_function_evaluation_limit"),
        ),
    ),
)
def test_normalized_linear_unit_restart_eligibility(
    candidate: dict | None,
    local_kkt_passed: bool,
    expected: tuple[bool, str],
) -> None:
    assert zero_barrier._normalized_linear_unit_restart_eligibility(
        candidate=candidate,
        local_kkt_passed=local_kkt_passed,
    ) == expected


def test_normalized_linear_variable_scaling_is_explicit() -> None:
    values = np.asarray([-1.0e8, 0.25], dtype=np.float64)

    np.testing.assert_array_equal(
        zero_barrier._normalized_linear_variable_scale(
            values, "initializer_relative"
        ),
        np.asarray([1.0e8, 1.0]),
    )
    np.testing.assert_array_equal(
        zero_barrier._normalized_linear_variable_scale(
            values, "dimensionless_unit"
        ),
        np.ones(2),
    )
    with pytest.raises(
        ValueError, match="Unknown normalized variable scaling"
    ):
        zero_barrier._normalized_linear_variable_scale(values, "unknown")


def _run_mock_self_reopening_initializer_portfolio(
    monkeypatch: pytest.MonkeyPatch,
    *,
    raw_succeeds: bool,
    regularized_stalls: bool = False,
    unit_restart_succeeds: bool = True,
    observed_budgets: list[tuple[int, int]] | None = None,
    observed_restart_seeds: list[tuple[np.ndarray, ...]] | None = None,
):
    gas_formula = np.asarray(
        [[1.0, 1.0], [1.0, -1.0]], dtype=np.float64
    )
    condensate_formula = np.eye(2, dtype=np.float64)
    target = np.asarray([1.0, 1.0e-20], dtype=np.float64)
    gas_standard = -np.log(np.asarray([0.25, 0.75]))
    condensate_standard = np.zeros(2, dtype=np.float64)
    face_total_gas = 2.0e-20
    face_gas = face_total_gas * np.asarray([0.75, 0.25])
    face_amounts = np.asarray([1.0, 0.0])
    face_potential = np.asarray([0.0, np.log(3.0)])
    edge_total_gas = 1.0e-20
    edge_gas = edge_total_gas * np.asarray([0.25, 0.75])
    edge_amounts = np.asarray([1.0, 1.5e-20])
    edge_potential = np.zeros(2, dtype=np.float64)
    budget_scale = np.reciprocal(target)

    def fake_regularized_initializer(**kwargs):
        del kwargs
        return (
            np.log(face_gas),
            float(np.log(face_total_gas)),
            face_potential.copy(),
            {
                "schema": "unit_test_capacity_regularized_initializer",
                "applied": True,
                "element_potential_recomputed": True,
                "element_potential_fit_rank": 2,
            },
        )

    calls: list[tuple[str, str]] = []

    def candidate(
        *,
        gas: np.ndarray,
        amounts: np.ndarray,
        total_gas: float,
        potential: np.ndarray,
        support: tuple[int, ...],
        function_evaluations: int,
    ) -> dict:
        audit = _physical_zero_barrier_audit(
            gas_formula_matrix=gas_formula,
            condensate_formula_matrix_full=condensate_formula,
            target_inventory=target,
            gas_standard_source=gas_standard,
            condensate_standard_source_full=condensate_standard,
            gas_log_amounts=np.log(gas),
            condensate_amounts=amounts,
            total_gas_log_amount=float(np.log(total_gas)),
            element_potential=potential,
            support_indices=support,
            condensate_valid_mask=np.ones(2, dtype=bool),
            budget_scale=budget_scale,
            optimizer_success=True,
            optimizer_status=1,
            stationarity_tolerance=1.0e-8,
            budget_tolerance=1.0e-8,
            total_density_tolerance=1.0e-8,
            support_closure_tolerance=1.0e-8,
        )
        return {
            "accepted": audit["accepted"],
            "gas_log_amounts": np.log(gas),
            "condensate_amounts": amounts.copy(),
            "total_gas_log_amount": float(np.log(total_gas)),
            "element_potential": potential.copy(),
            "support_indices": support,
            "optimizer_success": True,
            "optimizer_status": 1,
            "optimizer_message": "unit-test exact root",
            "function_evaluations": function_evaluations,
            "active_phase_at_lower_bound": False,
            "audit": audit,
        }

    def fake_normalized_solve(*, function_evaluation_budget, **kwargs):
        if observed_budgets is not None:
            observed_budgets.append(
                (
                    function_evaluation_budget.limit,
                    function_evaluation_budget.remaining,
                )
            )
        variable_scaling = kwargs["variable_scaling"]
        initializer = (
            "capacity_regularized"
            if not calls or variable_scaling == "dimensionless_unit"
            else "unregularized"
        )
        calls.append((initializer, variable_scaling))
        function_evaluations = 2 if initializer != "unregularized" else 3
        function_evaluation_budget.consume(function_evaluations)
        report = {
            "schema": "unit_test_normalized_solve",
            "attempted": True,
            "accepted": False,
            "dropped_support_indices": (
                (1,) if len(calls) == 1 and not regularized_stalls else ()
            ),
            "attempts": (
                {"function_evaluations": function_evaluations},
            ),
        }
        if regularized_stalls and len(calls) == 1:
            solved_candidate = candidate(
                gas=edge_gas,
                amounts=edge_amounts,
                total_gas=edge_total_gas,
                potential=edge_potential,
                support=(0, 1),
                function_evaluations=function_evaluations,
            )
            solved_candidate["accepted"] = False
            solved_candidate["optimizer_success"] = False
            solved_candidate["optimizer_status"] = 0
            solved_candidate["audit"] = dict(solved_candidate["audit"])
            solved_candidate["audit"]["accepted"] = False
            solved_candidate["audit"]["budget_scaled_max_abs"] = 1.0
        elif regularized_stalls and variable_scaling == "dimensionless_unit":
            if observed_restart_seeds is not None:
                observed_restart_seeds.append(
                    (
                        np.asarray(kwargs["gas_log_amounts_init"]).copy(),
                        np.asarray(kwargs["condensate_amounts_init"]).copy(),
                        np.asarray(kwargs["element_potential_init"]).copy(),
                        np.asarray(kwargs["support_indices"]).copy(),
                    )
                )
            solved_candidate = (
                candidate(
                    gas=edge_gas,
                    amounts=edge_amounts,
                    total_gas=edge_total_gas,
                    potential=edge_potential,
                    support=(0, 1),
                    function_evaluations=function_evaluations,
                )
                if unit_restart_succeeds
                else None
            )
            report["accepted"] = bool(unit_restart_succeeds)
        elif len(calls) == 1:
            solved_candidate = candidate(
                gas=face_gas,
                amounts=face_amounts,
                total_gas=face_total_gas,
                potential=face_potential,
                support=(0,),
                function_evaluations=function_evaluations,
            )
        elif raw_succeeds:
            solved_candidate = candidate(
                gas=edge_gas,
                amounts=edge_amounts,
                total_gas=edge_total_gas,
                potential=edge_potential,
                support=(0, 1),
                function_evaluations=function_evaluations,
            )
            report["accepted"] = True
        else:
            solved_candidate = None
        return {
            "accepted": bool(
                solved_candidate is not None
                and solved_candidate["accepted"]
            ),
            "candidate": solved_candidate,
            "report": report,
        }

    monkeypatch.setattr(
        zero_barrier,
        "_capacity_regularized_initializer",
        fake_regularized_initializer,
    )
    monkeypatch.setattr(
        zero_barrier,
        "_solve_normalized_gas_reduced_linear_support",
        fake_normalized_solve,
    )
    evaluation_budget = zero_barrier._FunctionEvaluationBudget(8)
    result = zero_barrier._polish_zero_barrier_support_once(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gas_standard,
        condensate_standard_source_full=condensate_standard,
        gas_log_amounts_init=np.log(edge_gas),
        condensate_amounts_init=edge_amounts,
        total_gas_log_amount_init=float(np.log(edge_total_gas)),
        element_potential_init=edge_potential,
        support_indices=(0, 1),
        max_function_evaluations=4,
        function_evaluation_budget=evaluation_budget,
        reduce_initial_support=False,
        use_zero_barrier_dual=False,
        use_finite_barrier_homotopy=False,
    )
    return result, calls, evaluation_budget


def test_self_reopening_regularized_root_retries_and_selects_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls, budget = _run_mock_self_reopening_initializer_portfolio(
        monkeypatch,
        raw_succeeds=True,
    )

    portfolio = result.report[
        "normalized_gas_reduced_initializer_portfolio"
    ]
    assert result.accepted
    assert result.support_indices == (0, 1)
    assert calls == [
        ("capacity_regularized", "initializer_relative"),
        ("unregularized", "initializer_relative"),
    ]
    assert portfolio["raw_retry_attempted"]
    assert portfolio["raw_retry_reason"] == "self_reopening_support_drop"
    assert portfolio["deferred_initializer"] == "capacity_regularized"
    assert portfolio["selected_initializer"] == "unregularized"
    assert portfolio["attempts"][0]["selection_deferred"]
    assert portfolio["attempts"][0][
        "self_reopening_dropped_support_indices"
    ] == (1,)
    assert portfolio["attempts"][1]["selected"]
    assert len(portfolio["discarded_solve_reports"]) == 1
    assert portfolio["discarded_solve_reports"][0]["initializer"] == (
        "capacity_regularized"
    )
    assert budget.used == 5
    linear_evaluations, reduced_evaluations = (
        zero_barrier._zero_barrier_report_function_evaluations(result.report)
    )
    assert linear_evaluations == 0
    assert reduced_evaluations == budget.used


def test_self_reopening_regularized_root_is_restored_when_raw_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls, budget = _run_mock_self_reopening_initializer_portfolio(
        monkeypatch,
        raw_succeeds=False,
    )

    portfolio = result.report[
        "normalized_gas_reduced_initializer_portfolio"
    ]
    assert not result.accepted
    assert result.support_indices == (0,)
    assert calls == [
        ("capacity_regularized", "initializer_relative"),
        ("unregularized", "initializer_relative"),
    ]
    assert portfolio["raw_retry_attempted"]
    assert portfolio["selected_initializer"] == "capacity_regularized"
    assert portfolio["attempts"][0]["selected"]
    assert portfolio["attempts"][0][
        "selected_after_raw_retry_failure"
    ]
    assert not portfolio["attempts"][1]["selected"]
    assert len(portfolio["discarded_solve_reports"]) == 1
    assert portfolio["discarded_solve_reports"][0]["initializer"] == (
        "unregularized"
    )
    assert portfolio["discarded_solve_reports"][0]["discard_reason"] == (
        "raw_retry_failed"
    )
    assert budget.used == 5
    linear_evaluations, reduced_evaluations = (
        zero_barrier._zero_barrier_report_function_evaluations(result.report)
    )
    assert linear_evaluations == 0
    assert reduced_evaluations == budget.used


def test_stalled_regularized_solve_restarts_from_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_budgets: list[tuple[int, int]] = []
    observed_restart_seeds: list[tuple[np.ndarray, ...]] = []
    result, calls, budget = _run_mock_self_reopening_initializer_portfolio(
        monkeypatch,
        raw_succeeds=True,
        regularized_stalls=True,
        observed_budgets=observed_budgets,
        observed_restart_seeds=observed_restart_seeds,
    )

    portfolio = result.report[
        "normalized_gas_reduced_initializer_portfolio"
    ]
    assert result.accepted
    assert calls == [
        ("capacity_regularized", "initializer_relative"),
        ("capacity_regularized", "dimensionless_unit"),
    ]
    assert portfolio["dimensionless_unit_restart_eligible"]
    assert portfolio["dimensionless_unit_restart_reason"] == (
        "finite_function_evaluation_limit"
    )
    assert portfolio["dimensionless_unit_restart_attempted"]
    assert portfolio["regularized_function_evaluation_limit"] == 4
    assert portfolio["regularized_function_evaluations"] == 4
    assert portfolio["raw_function_evaluation_reserve"] == 4
    assert observed_budgets[:2] == [(4, 4), (2, 2)]
    assert budget.used == 4
    linear_evaluations, reduced_evaluations = (
        zero_barrier._zero_barrier_report_function_evaluations(result.report)
    )
    assert linear_evaluations == 0
    assert reduced_evaluations == budget.used
    assert not portfolio["unregularized_attempted"]
    assert not portfolio["raw_retry_attempted"]
    assert portfolio["selected_initializer"] == "capacity_regularized"
    assert portfolio["selected_variable_scaling"] == "dimensionless_unit"
    assert portfolio["attempts"][1]["restart_from_terminal_state"]
    assert portfolio["attempts"][1]["local_kkt_passed"]
    assert portfolio["attempts"][1]["selected"]
    assert len(observed_restart_seeds) == 1
    restart_q, restart_m, restart_lambda, restart_support = (
        observed_restart_seeds[0]
    )
    np.testing.assert_allclose(
        restart_q, np.log(1.0e-20 * np.asarray([0.25, 0.75]))
    )
    np.testing.assert_allclose(restart_m, np.asarray([1.0, 1.5e-20]))
    np.testing.assert_allclose(restart_lambda, np.zeros(2))
    np.testing.assert_array_equal(restart_support, np.asarray([0, 1]))


def test_failed_unit_restart_preserves_raw_initializer_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_budgets: list[tuple[int, int]] = []
    result, calls, budget = _run_mock_self_reopening_initializer_portfolio(
        monkeypatch,
        raw_succeeds=True,
        regularized_stalls=True,
        unit_restart_succeeds=False,
        observed_budgets=observed_budgets,
    )

    portfolio = result.report[
        "normalized_gas_reduced_initializer_portfolio"
    ]
    assert result.accepted
    assert calls == [
        ("capacity_regularized", "initializer_relative"),
        ("capacity_regularized", "dimensionless_unit"),
        ("unregularized", "initializer_relative"),
    ]
    assert observed_budgets == [(4, 4), (2, 2), (8, 4)]
    assert budget.used == 7
    assert portfolio["dimensionless_unit_restart_attempted"]
    assert portfolio["raw_retry_attempted"]
    assert portfolio["selected_initializer"] == "unregularized"
    assert portfolio["selected_variable_scaling"] == "initializer_relative"
    assert len(portfolio["attempts"]) == 3
    assert portfolio["regularized_function_evaluations"] == 4
    assert portfolio["raw_function_evaluation_reserve"] == 4


def test_normalized_gas_reduced_primary_handles_signed_zero_budget() -> None:
    gas_formula = np.asarray(
        [[1.0, 1.0], [1.0, -1.0]], dtype=np.float64
    )
    condensate_formula = np.asarray([[1.0], [0.0]], dtype=np.float64)
    expected_gas = np.asarray([0.25, 0.25], dtype=np.float64)
    expected_potential = np.asarray([0.2, 0.1], dtype=np.float64)
    qtot = float(np.log(np.sum(expected_gas)))
    gamma = (
        gas_formula.T @ expected_potential
        - np.log(expected_gas)
        + qtot
    )

    result = polish_zero_barrier_active_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=np.asarray([1.0, 0.0], dtype=np.float64),
        gas_standard_source=gamma,
        condensate_standard_source_full=np.asarray([0.2]),
        gas_log_amounts_init=np.log(expected_gas) + 1.0e-3,
        condensate_amounts_init=np.asarray([0.49]),
        total_gas_log_amount_init=qtot + 1.0e-3,
        element_potential_init=expected_potential + 1.0e-3,
        support_indices=(0,),
        budget_relative_floor=1.0e-6,
    )

    reconstructed = (
        gas_formula @ np.exp(result.gas_log_amounts)
        + condensate_formula @ result.condensate_amounts
    )
    assert result.accepted
    assert result.report["selected_numerical_formulation"] == (
        "normalized_gas_reduced_linear_amounts"
    )
    assert not result.report["linear_amount_physical_audit"]["attempted"]
    np.testing.assert_allclose(reconstructed, [1.0, 0.0], atol=1.0e-12)


def test_zero_barrier_audit_rejects_negative_inactive_driving() -> None:
    (
        gas_formula,
        condensate_formula,
        expected_gas,
        expected_condensates,
        expected_potential,
        gamma,
        hcond,
        target,
    ) = _one_active_phase_problem()
    hcond[1] = -0.3

    audit = _physical_zero_barrier_audit(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond,
        gas_log_amounts=np.log(expected_gas),
        condensate_amounts=expected_condensates,
        total_gas_log_amount=0.0,
        element_potential=expected_potential,
        support_indices=(0,),
        condensate_valid_mask=np.ones(2, dtype=bool),
        budget_scale=np.reciprocal(target),
        optimizer_success=True,
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
    )

    assert not audit["accepted"]
    assert audit["active_condensate_driving_max_abs"] < 1.0e-10
    assert audit["inactive_condensate_violation_max_abs"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("optimizer_success", "optimizer_status", "accepted", "source"),
    (
        (True, 1, True, "optimizer_success"),
        (False, 0, True, "physical_kkt_after_optimizer_limit"),
        (False, -1, False, None),
        (False, None, False, None),
        (True, -1, False, None),
    ),
)
def test_zero_barrier_audit_separates_physics_from_optimizer_termination(
    optimizer_success: bool,
    optimizer_status: int | None,
    accepted: bool,
    source: str | None,
) -> None:
    (
        gas_formula,
        condensate_formula,
        expected_gas,
        expected_condensates,
        expected_potential,
        gamma,
        hcond,
        target,
    ) = _one_active_phase_problem()

    audit = _physical_zero_barrier_audit(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond,
        gas_log_amounts=np.log(expected_gas),
        condensate_amounts=expected_condensates,
        total_gas_log_amount=0.0,
        element_potential=expected_potential,
        support_indices=(0,),
        condensate_valid_mask=np.ones(2, dtype=bool),
        budget_scale=np.reciprocal(target),
        optimizer_success=optimizer_success,
        optimizer_status=optimizer_status,
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
    )

    assert audit["physical_root_certified"]
    assert audit["optimizer_termination_eligible"] is (source is not None)
    assert audit["accepted"] is accepted
    assert audit["acceptance_source"] == source


@pytest.mark.parametrize("failure", ("nonfinite", "stationarity", "budget"))
def test_optimizer_limit_acceptance_requires_full_physical_certificate(
    failure: str,
) -> None:
    (
        gas_formula,
        condensate_formula,
        expected_gas,
        expected_condensates,
        expected_potential,
        gamma,
        hcond,
        target,
    ) = _one_active_phase_problem()
    gas_log_amounts = np.log(expected_gas)
    condensate_amounts = expected_condensates.copy()
    if failure == "nonfinite":
        gas_log_amounts[0] = np.nan
    elif failure == "stationarity":
        gas_log_amounts[0] += 1.0e-4
    else:
        condensate_amounts[0] += 1.0e-4

    audit = _physical_zero_barrier_audit(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond,
        gas_log_amounts=gas_log_amounts,
        condensate_amounts=condensate_amounts,
        total_gas_log_amount=0.0,
        element_potential=expected_potential,
        support_indices=(0,),
        condensate_valid_mask=np.ones(2, dtype=bool),
        budget_scale=np.reciprocal(target),
        optimizer_success=False,
        optimizer_status=0,
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
    )

    assert not audit["physical_root_certified"]
    assert audit["optimizer_termination_eligible"]
    assert not audit["accepted"]
    assert audit["acceptance_source"] is None


def test_optimizer_limit_local_root_cannot_seed_support_transition() -> None:
    """Require optimizer convergence before changing condensate support."""

    (
        gas_formula,
        condensate_formula,
        expected_gas,
        expected_condensates,
        expected_potential,
        gamma,
        hcond,
        target,
    ) = _one_active_phase_problem()
    hcond[1] = -0.3
    audit = _physical_zero_barrier_audit(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond,
        gas_log_amounts=np.log(expected_gas),
        condensate_amounts=expected_condensates,
        total_gas_log_amount=0.0,
        element_potential=expected_potential,
        support_indices=(0,),
        condensate_valid_mask=np.ones(2, dtype=bool),
        budget_scale=np.reciprocal(target),
        optimizer_success=False,
        optimizer_status=0,
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
    )

    assert audit["optimizer_termination_eligible"]
    assert not audit["physical_root_certified"]
    assert not audit["accepted"]
    assert not zero_barrier._physical_audit_local_kkt_passed(
        audit,
        optimizer_success=False,
        optimizer_status=0,
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
    )
    failure_reasons = zero_barrier._local_zero_barrier_kkt_failure_reasons(
        audit | {"optimizer_success": False, "optimizer_status": 0},
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
    )
    assert failure_reasons == ("optimizer_failed",)


def test_zero_barrier_closure_only_failure_skips_deletion_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        gas_formula,
        condensate_formula,
        expected_gas,
        expected_condensates,
        expected_potential,
        gamma,
        hcond,
        target,
    ) = _one_active_phase_problem()
    hcond[1] = -0.3

    def unexpected_fallback(**kwargs):
        del kwargs
        raise AssertionError("closure-only state must reach the add-back loop")

    monkeypatch.setattr(
        zero_barrier,
        "_solve_reduced_log_domain_support_branches",
        unexpected_fallback,
    )
    result = zero_barrier._polish_zero_barrier_support_once(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond,
        gas_log_amounts_init=np.log(expected_gas),
        condensate_amounts_init=expected_condensates,
        total_gas_log_amount_init=0.0,
        element_potential_init=expected_potential,
        support_indices=(0,),
        budget_relative_floor=1.0e-12,
    )

    fallback = result.report["reduced_log_domain_fallback"]
    assert not result.accepted
    assert result.report["linear_amount_physical_audit"]["local_kkt_passed"]
    assert not fallback["attempted"]
    assert fallback["skip_reason"] == (
        "reduced_primary_inactive_support_closure_only"
    )
    assert result.report["selected_numerical_formulation"] == (
        "normalized_gas_reduced_linear_amounts"
    )


def _mock_polish_result(
    *,
    accepted: bool,
    support_indices: tuple[int, ...],
    full_driving: tuple[float, ...],
    budget_scaled_max_abs: float = 0.0,
    inactive_violation_max_abs: float | None = None,
    function_evaluations: int = 0,
) -> ZeroBarrierPolishResult:
    condensate_count = len(full_driving)
    inactive_violation = (
        max(
            (
                -driving
                for index, driving in enumerate(full_driving)
                if index not in set(support_indices) and driving < 0.0
            ),
            default=0.0,
        )
        if inactive_violation_max_abs is None
        else inactive_violation_max_abs
    )
    amounts = np.zeros(condensate_count, dtype=np.float64)
    if support_indices:
        amounts[np.asarray(support_indices, dtype=np.int64)] = 0.1
    return ZeroBarrierPolishResult(
        accepted=accepted,
        gas_log_amounts=np.asarray([-0.4, -1.1], dtype=np.float64),
        condensate_amounts=amounts,
        total_gas_log_amount=-0.1,
        element_potential=np.asarray([0.2, -0.3], dtype=np.float64),
        support_indices=support_indices,
        report={
            "polish_schema": "unit_test",
            "accepted": accepted,
            "initial_support_indices": support_indices,
            "final_support_indices": support_indices,
            "dropped_support_indices": (),
            "optimizer_success": True,
            "finite": True,
            "positive_active_amounts": True,
            "gas_stationarity_max_abs": 0.0,
            "active_condensate_driving_max_abs": 0.0,
            "inactive_condensate_violation_max_abs": inactive_violation,
            "budget_scaled_max_abs": budget_scaled_max_abs,
            "total_density_scaled_abs": 0.0,
            "full_condensate_driving": full_driving,
            "selected_numerical_formulation": "unit_test",
            "attempts": (
                {"function_evaluations": function_evaluations},
            ),
            "reduced_log_domain_fallback": {"nodes": ()},
        },
    )


def _mock_polish_arguments(
    *,
    condensate_valid_mask: np.ndarray | None = None,
    support_indices: tuple[int, ...] = (0,),
    target_inventory: np.ndarray | None = None,
) -> dict:
    return {
        "gas_formula_matrix": np.eye(2, dtype=np.float64),
        "condensate_formula_matrix_full": np.asarray(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64
        ),
        "target_inventory": (
            np.ones(2, dtype=np.float64)
            if target_inventory is None
            else target_inventory
        ),
        "gas_standard_source": np.zeros(2, dtype=np.float64),
        "condensate_standard_source_full": np.zeros(3, dtype=np.float64),
        "gas_log_amounts_init": np.log(
            np.asarray([0.5, 0.5], dtype=np.float64)
        ),
        "condensate_amounts_init": np.zeros(3, dtype=np.float64),
        "total_gas_log_amount_init": 0.0,
        "element_potential_init": np.zeros(2, dtype=np.float64),
        "support_indices": support_indices,
        "condensate_valid_mask": condensate_valid_mask,
    }


@pytest.mark.parametrize("amount_scale", (1.0e-12, 1.0, 1.0e8))
def test_basic_support_reduction_preserves_inventory_and_amount_gauge(
    amount_scale: float,
) -> None:
    condensate_formula = np.asarray(
        [
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    hcond = np.asarray([2.0, 1.0, 0.5], dtype=np.float64)
    target = amount_scale * np.asarray([1.0, 1.0, 0.0])
    amounts = amount_scale * np.asarray([0.2, 0.3, 0.4])
    support = (2, 0, 1)

    reduced_support, reduced_amounts, report = (
        zero_barrier._reduce_initial_condensate_support_to_basic(
            condensate_formula_matrix_full=condensate_formula,
            condensate_standard_source_full=hcond,
            target_inventory=target,
            condensate_amounts=amounts,
            support_indices=support,
            budget_scale=np.asarray(
                [1.0 / amount_scale, 1.0 / amount_scale, 1.0]
            ),
            budget_tolerance=1.0e-8,
            enabled=True,
        )
    )

    assert reduced_support == (1, 2)
    np.testing.assert_allclose(
        reduced_amounts / amount_scale,
        np.asarray([0.0, 0.5, 0.4]),
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        condensate_formula @ reduced_amounts,
        condensate_formula @ amounts,
        rtol=1.0e-12,
        atol=1.0e-14 * amount_scale,
    )
    assert report["applied"]
    assert report["role"] == "zero_barrier_exact_solve_initializer"
    assert report["initial_support_rank"] == 2
    assert report["final_support_rank"] == 2
    assert report["dropped_support_indices"] == (0,)
    assert report["objective_after"] < report["objective_before"]


def test_basic_support_reduction_falls_back_on_solver_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    condensate_formula = np.asarray([[1.0, 1.0]], dtype=np.float64)
    amounts = np.asarray([0.4, 0.6], dtype=np.float64)

    def raise_solver_error(*args, **kwargs):
        del args, kwargs
        raise ValueError("unit-test LP failure")

    monkeypatch.setattr(zero_barrier, "linprog", raise_solver_error)
    support, returned_amounts, report = (
        zero_barrier._reduce_initial_condensate_support_to_basic(
            condensate_formula_matrix_full=condensate_formula,
            condensate_standard_source_full=np.asarray([0.0, 1.0]),
            target_inventory=np.asarray([1.0]),
            condensate_amounts=amounts,
            support_indices=(0, 1),
            budget_scale=np.asarray([1.0]),
            budget_tolerance=1.0e-8,
            enabled=True,
        )
    )

    assert support == (0, 1)
    np.testing.assert_array_equal(returned_amounts, amounts)
    assert report["attempted"]
    assert not report["applied"]
    assert report["failure_reason"] == "solver_exception"


def test_basic_support_reduction_falls_back_on_empty_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    amounts = np.zeros(2, dtype=np.float64)

    class EmptySolution:
        success = True
        status = 0
        message = "unit-test empty basic solution"
        nit = 1
        x = np.zeros(2, dtype=np.float64)

    monkeypatch.setattr(
        zero_barrier,
        "linprog",
        lambda *args, **kwargs: EmptySolution(),
    )
    support, returned_amounts, report = (
        zero_barrier._reduce_initial_condensate_support_to_basic(
            condensate_formula_matrix_full=np.asarray([[1.0, 1.0]]),
            condensate_standard_source_full=np.asarray([0.0, 1.0]),
            target_inventory=np.asarray([1.0]),
            condensate_amounts=amounts,
            support_indices=(0, 1),
            budget_scale=np.asarray([1.0]),
            budget_tolerance=1.0e-8,
            enabled=True,
        )
    )

    assert support == (0, 1)
    np.testing.assert_array_equal(returned_amounts, amounts)
    assert not report["applied"]
    assert report["candidate_support_indices"] == ()
    assert report["candidate_dropped_support_indices"] == (0, 1)
    assert "final_support_indices" not in report
    assert "dropped_support_indices" not in report
    assert report["failure_reason"] == "postsolve_validation_failed"


def test_basic_support_reduction_is_stable_across_input_order() -> None:
    common = {
        "condensate_formula_matrix_full": np.asarray([[1.0, 1.0]]),
        "condensate_standard_source_full": np.asarray([1.0, 1.0]),
        "target_inventory": np.asarray([1.0]),
        "condensate_amounts": np.asarray([0.5, 0.5]),
        "budget_scale": np.asarray([1.0]),
        "budget_tolerance": 1.0e-8,
        "enabled": True,
    }

    forward = zero_barrier._reduce_initial_condensate_support_to_basic(
        **common, support_indices=(0, 1)
    )
    reverse = zero_barrier._reduce_initial_condensate_support_to_basic(
        **common, support_indices=(1, 0)
    )

    assert forward[0] == reverse[0]
    np.testing.assert_array_equal(forward[1], reverse[1])
    assert forward[2]["canonical_support_indices"] == (0, 1)
    assert reverse[2]["canonical_support_indices"] == (0, 1)


def test_basic_support_reduction_skips_signed_stoichiometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_solver(*args, **kwargs):
        del args, kwargs
        raise AssertionError("linprog must not run for signed stoichiometry")

    monkeypatch.setattr(zero_barrier, "linprog", unexpected_solver)
    amounts = np.asarray([0.4, 0.6])
    support, returned_amounts, report = (
        zero_barrier._reduce_initial_condensate_support_to_basic(
            condensate_formula_matrix_full=np.asarray(
                [[1.0, 1.0], [-1.0, -1.0]]
            ),
            condensate_standard_source_full=np.asarray([0.0, 1.0]),
            target_inventory=np.asarray([1.0, 0.0]),
            condensate_amounts=amounts,
            support_indices=(0, 1),
            budget_scale=np.asarray([1.0, 1.0]),
            budget_tolerance=1.0e-8,
            enabled=True,
        )
    )

    assert support == (0, 1)
    np.testing.assert_array_equal(returned_amounts, amounts)
    assert not report["attempted"]
    assert not report["applied"]
    assert report["skip_reason"] == "not_rank_reduction_eligible"


@pytest.mark.parametrize("amount_scale", (1.0e-12, 1.0, 1.0e8))
def test_alternative_basic_support_candidates_are_deterministic_and_covariant(
    amount_scale: float,
) -> None:
    condensate_formula = np.asarray(
        [[1.0, 1.0, 0.0, 2.0], [0.0, 1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    target = amount_scale * np.asarray([2.0, 2.0])
    amounts = amount_scale * np.asarray([0.5, 1.0, 0.0, 0.0])
    common = {
        "condensate_formula_matrix_full": condensate_formula,
        "target_inventory": target,
        "condensate_amounts": amounts,
        "budget_scale": np.full(2, 1.0 / amount_scale),
        "budget_tolerance": 1.0e-8,
    }

    forward, forward_report = (
        zero_barrier._build_alternative_basic_support_candidates(
            **common,
            support_indices=(0, 1, 2, 3),
        )
    )
    reverse, reverse_report = (
        zero_barrier._build_alternative_basic_support_candidates(
            **common,
            support_indices=(3, 2, 1, 0),
        )
    )
    expected_order = ((0, 1), (0, 2), (1, 3), (2, 3))
    assert (
        tuple(item["support_indices"] for item in forward)
        == expected_order
    )
    assert (
        tuple(item["support_indices"] for item in reverse)
        == expected_order
    )
    assert forward_report["visited_basis_indices"] == (
        reverse_report["visited_basis_indices"]
    )
    for candidate in forward:
        np.testing.assert_allclose(
            condensate_formula @ candidate["condensate_amounts"],
            condensate_formula @ amounts,
            rtol=1.0e-12,
            atol=1.0e-14 * amount_scale,
        )
    assert forward_report["initial_support_rank"] == 2
    assert forward_report["initial_support_nullity"] == 2
    assert not forward_report["node_limit_reached"]


def test_alternative_basic_support_candidate_search_is_bounded() -> None:
    phase_count = zero_barrier._REDUCED_SUPPORT_NODE_LIMIT + 8
    candidates, report = (
        zero_barrier._build_alternative_basic_support_candidates(
            condensate_formula_matrix_full=np.ones((1, phase_count)),
            target_inventory=np.ones(1),
            condensate_amounts=np.full(phase_count, 1.0 / phase_count),
            support_indices=tuple(range(phase_count)),
            budget_scale=np.ones(1),
            budget_tolerance=1.0e-8,
        )
    )

    assert report["attempted"]
    assert report["node_limit_reached"]
    assert report["visited_basis_count"] == (
        zero_barrier._REDUCED_SUPPORT_NODE_LIMIT
    )
    assert len(candidates) == zero_barrier._REDUCED_SUPPORT_NODE_LIMIT


@pytest.mark.parametrize("amount_scale", (1.0e-12, 1.0, 1.0e8))
def test_support_release_candidates_order_and_transform_amounts(
    amount_scale: float,
) -> None:
    condensate_formula = np.column_stack(
        [np.eye(3, dtype=np.float64), np.ones(3, dtype=np.float64)]
    )
    target = amount_scale * np.asarray([1.0, 1.0e-9, 1.0e-3])
    amounts = amount_scale * np.asarray([0.4, 2.0e-10, 3.0e-4, 0.2])

    candidates, report = zero_barrier._build_support_release_candidates(
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        condensate_amounts=amounts,
        support_indices=(2, 0, 1),
        max_support_nodes=8,
    )

    assert report["eligible"]
    assert report["attempted"]
    assert not report["condensate_inventory_preserved"]
    assert report["source_support_indices"] == (0, 1, 2)
    assert report["candidate_ordering"] == (
        "removed_maximum_amount_scale_sum_then_support_indices"
    )
    assert not report["node_limit_reached"]
    assert tuple(
        candidate["support_indices"] for candidate in candidates
    ) == ((0, 2), (0, 1), (0,), (1, 2), (2,), (1,), ())
    assert tuple(
        record["support_indices"]
        for record in report["candidate_records"]
    ) == tuple(
        candidate["support_indices"] for candidate in candidates
    )
    for candidate in candidates:
        support = candidate["support_indices"]
        expected = np.zeros_like(amounts)
        expected[np.asarray(support, dtype=np.int64)] = amounts[
            np.asarray(support, dtype=np.int64)
        ]
        np.testing.assert_array_equal(
            candidate["condensate_amounts"], expected
        )
    assert not np.allclose(
        condensate_formula @ candidates[0]["condensate_amounts"],
        condensate_formula @ np.asarray([*amounts[:3], 0.0]),
        rtol=0.0,
        atol=0.0,
    )


def test_support_release_candidate_order_preserves_trace_burdens() -> None:
    target = np.asarray([1.0, 1.0e-80, 1.0e-90])
    candidates, _report = zero_barrier._build_support_release_candidates(
        condensate_formula_matrix_full=np.eye(3),
        target_inventory=target,
        condensate_amounts=0.5 * target,
        support_indices=(0, 1, 2),
        max_support_nodes=8,
    )

    assert tuple(
        candidate["support_indices"] for candidate in candidates
    ) == ((0, 1), (0, 2), (0,), (1, 2), (1,), (2,), ())


def test_support_release_can_select_the_gas_only_face() -> None:
    result = zero_barrier._solve_support_release_portfolio(
        gas_formula_matrix=np.ones((1, 1), dtype=np.float64),
        condensate_formula_matrix_full=np.ones((1, 1), dtype=np.float64),
        target_inventory=np.ones(1, dtype=np.float64),
        gas_standard_source=np.zeros(1, dtype=np.float64),
        condensate_standard_source_full=-np.ones(1, dtype=np.float64),
        gas_log_amounts_init=np.log(np.asarray([0.5])),
        condensate_amounts_init=np.asarray([0.5]),
        total_gas_log_amount_init=float(np.log(0.5)),
        element_potential_init=np.zeros(1, dtype=np.float64),
        support_indices=(0,),
        condensate_valid_mask=np.ones(1, dtype=bool),
        budget_scale=np.ones(1, dtype=np.float64),
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=100,
        enabled=True,
    )

    report = result["report"]
    assert result["selected"]
    assert not result["accepted"]
    assert result["candidate"]["support_indices"] == ()
    assert report["attempted"]
    assert report["local_kkt_selected"]
    assert report["selected_support_indices"] == ()
    assert report["role"] == "initializer_only"
    assert not report["condensate_inventory_preserved"]
    assert report["final_physical_audit_authoritative"]


def test_support_release_keeps_log_domain_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        zero_barrier,
        "_solve_normalized_gas_reduced_linear_support",
        lambda **kwargs: {
            "accepted": False,
            "candidate": None,
            "report": {"support_indices": tuple(kwargs["support_indices"])},
        },
    )

    def unexpected_log_fallback(**kwargs):
        del kwargs
        raise AssertionError("default support release must remain normalized-linear")

    monkeypatch.setattr(
        zero_barrier,
        "_solve_reduced_log_domain_active_support",
        unexpected_log_fallback,
    )
    result = zero_barrier._solve_support_release_portfolio(
        gas_formula_matrix=np.eye(2, dtype=np.float64),
        condensate_formula_matrix_full=np.eye(2, dtype=np.float64),
        target_inventory=np.ones(2, dtype=np.float64),
        gas_standard_source=np.zeros(2, dtype=np.float64),
        condensate_standard_source_full=np.zeros(2, dtype=np.float64),
        gas_log_amounts_init=np.log(np.asarray([0.5, 0.5])),
        condensate_amounts_init=np.asarray([0.5, 0.5]),
        total_gas_log_amount_init=0.0,
        element_potential_init=np.zeros(2, dtype=np.float64),
        support_indices=(0, 1),
        condensate_valid_mask=np.ones(2, dtype=bool),
        budget_scale=np.ones(2, dtype=np.float64),
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=20,
        enabled=True,
    )

    assert not result["selected"]
    assert result["report"]["log_domain_fallback_reason"] == "disabled"


def test_support_release_can_prefer_log_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_log_portfolio(**kwargs):
        calls.append(tuple(kwargs["candidates"][0]["support_indices"]))
        return {
            "accepted": False,
            "selected": True,
            "local_kkt_selected": True,
            "candidate": {"support_indices": ()},
            "initializer_regularization": None,
            "solve_attempts": (
                {
                    "support_indices": (),
                    "formulation": "reduced_log_domain",
                },
            ),
            "stop_reason": "local_kkt_selected_for_active_set_closure",
            "fallback_reason": "attempted",
        }

    monkeypatch.setattr(
        zero_barrier,
        "_solve_log_domain_support_candidate_portfolio",
        fake_log_portfolio,
    )

    def unexpected_normalized_solve(**kwargs):
        del kwargs
        raise AssertionError("log-domain support release must run first")

    monkeypatch.setattr(
        zero_barrier,
        "_solve_normalized_gas_reduced_linear_support",
        unexpected_normalized_solve,
    )
    result = zero_barrier._solve_support_release_portfolio(
        gas_formula_matrix=np.ones((1, 1), dtype=np.float64),
        condensate_formula_matrix_full=np.ones((1, 1), dtype=np.float64),
        target_inventory=np.ones(1, dtype=np.float64),
        gas_standard_source=np.zeros(1, dtype=np.float64),
        condensate_standard_source_full=-np.ones(1, dtype=np.float64),
        gas_log_amounts_init=np.log(np.asarray([0.5])),
        condensate_amounts_init=np.asarray([0.5]),
        total_gas_log_amount_init=float(np.log(0.5)),
        element_potential_init=np.zeros(1, dtype=np.float64),
        support_indices=(0,),
        condensate_valid_mask=np.ones(1, dtype=bool),
        budget_scale=np.ones(1, dtype=np.float64),
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=20,
        enabled=True,
        enable_log_domain_fallback=True,
        prefer_log_domain=True,
    )

    assert calls == [()]
    assert result["selected"]
    assert not result["accepted"]
    assert result["report"]["prefer_log_domain"]
    assert result["report"]["selected_formulation"] == "reduced_log_domain"


@pytest.mark.parametrize(
    ("formula", "amounts", "expected_reason"),
    (
        (
            np.ones((1, 2), dtype=np.float64),
            np.asarray([0.5, 0.5]),
            "source_support_not_releasable",
        ),
        (
            np.eye(2, dtype=np.float64),
            np.asarray([0.5, -0.5]),
            "invalid_numerical_initializer",
        ),
    ),
)
def test_support_release_candidates_fail_closed(
    formula: np.ndarray,
    amounts: np.ndarray,
    expected_reason: str,
) -> None:
    candidates, report = zero_barrier._build_support_release_candidates(
        condensate_formula_matrix_full=formula,
        target_inventory=np.ones(formula.shape[0]),
        condensate_amounts=amounts,
        support_indices=tuple(range(formula.shape[1])),
    )

    assert candidates == ()
    assert not report["eligible"]
    assert report["skip_reason"] == expected_reason


def test_alternative_basic_support_precedes_a_redundant_exact_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DegenerateSolution:
        success = True
        status = 0
        message = "unit-test nonbasic LP vertex"
        nit = 1
        x = np.asarray([1.0 / 3.0, 1.0 / 3.0])

    monkeypatch.setattr(
        zero_barrier,
        "linprog",
        lambda *args, **kwargs: DegenerateSolution(),
    )
    result = zero_barrier._polish_zero_barrier_support_once(
        gas_formula_matrix=np.ones((1, 1)),
        condensate_formula_matrix_full=np.ones((1, 2)),
        target_inventory=np.asarray([1.5]),
        gas_standard_source=np.zeros(1),
        condensate_standard_source_full=np.zeros(2),
        gas_log_amounts_init=np.log(np.asarray([0.5])),
        condensate_amounts_init=np.asarray([0.5, 0.5]),
        total_gas_log_amount_init=float(np.log(0.5)),
        element_potential_init=np.zeros(1),
        support_indices=(0, 1),
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=400,
        use_zero_barrier_dual=False,
        use_finite_barrier_homotopy=False,
    )

    assert result.accepted
    assert len(result.support_indices) == 1
    assert result.report["basic_support_reduction"]["attempted"]
    assert not result.report["basic_support_reduction"]["applied"]
    portfolio = result.report["alternative_basic_support_portfolio"]
    assert portfolio["attempted"]
    assert portfolio["accepted"]
    assert portfolio["selected_support_indices"] == result.support_indices
    assert not result.report["normalized_gas_reduced_primary"]["attempted"]


def test_structural_terminal_root_precedes_alternative_basic_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DegenerateSolution:
        success = True
        status = 0
        message = "unit-test nonbasic LP vertex"
        nit = 1
        x = np.asarray([1.0 / 3.0, 1.0 / 3.0])

    monkeypatch.setattr(
        zero_barrier,
        "linprog",
        lambda *args, **kwargs: DegenerateSolution(),
    )
    result = zero_barrier._polish_zero_barrier_support_once(
        gas_formula_matrix=np.asarray([[1.0], [0.0]]),
        condensate_formula_matrix_full=np.asarray(
            [[1.0, 1.0], [0.0, 0.0]]
        ),
        target_inventory=np.asarray([1.5, 0.0]),
        gas_standard_source=np.zeros(1),
        condensate_standard_source_full=np.zeros(2),
        gas_log_amounts_init=np.log(np.asarray([0.5])),
        condensate_amounts_init=np.asarray([0.5, 0.5]),
        total_gas_log_amount_init=float(np.log(0.5)),
        element_potential_init=np.zeros(2),
        support_indices=(0, 1),
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=400,
        use_zero_barrier_dual=False,
        use_finite_barrier_homotopy=False,
    )

    assert result.accepted
    assert result.support_indices == (0, 1)
    assert result.report["selected_numerical_formulation"] == (
        "structural_zero_normalized_gas_reduced_linear_amounts"
    )
    assert result.report["structural_zero_reduced_log_rescue"]["accepted"]
    portfolio = result.report["alternative_basic_support_portfolio"]
    assert not portfolio["attempted"]
    assert portfolio["skip_reason"] == (
        "structural_zero_certified_support_selected"
    )


def _mock_alternative_portfolio_candidates():
    candidates = tuple(
        {
            "support_indices": (index,),
            "condensate_amounts": np.asarray([1.0, 0.0])
            if index == 0
            else np.asarray([0.0, 1.0]),
        }
        for index in range(2)
    )
    report = {
        "schema": "unit_test_basic_support_candidates",
        "eligible": True,
        "attempted": True,
    }
    return candidates, report


def test_optimizer_directed_release_keeps_the_feasible_basis_initializer(
) -> None:
    candidates, _ = _mock_alternative_portfolio_candidates()
    solve_attempts = (
        {
            "support_indices": (0,),
            "formulation": "normalized_gas_reduced_linear_amounts",
            "accepted": False,
            "local_kkt_passed": False,
            "solve": {
                "attempts": (
                    {
                        "optimizer_success": False,
                        "active_condensate_amounts": (-1.0,),
                    },
                )
            },
        },
        {
            "support_indices": (1,),
            "formulation": "normalized_gas_reduced_linear_amounts",
            "accepted": False,
            "local_kkt_passed": False,
            "solve": {
                "attempts": (
                    {
                        "optimizer_success": True,
                        "active_condensate_amounts": (-0.25,),
                    },
                )
            },
        },
    )

    source, report = (
        zero_barrier._select_optimizer_directed_support_release_source(
            candidates=candidates,
            solve_attempts=solve_attempts,
        )
    )

    assert source is None
    assert report["attempted"]
    assert not report["selected"]

    mixed_sign_candidates = (
        {
            "support_indices": (0, 1),
            "condensate_amounts": np.asarray([0.4, 0.6]),
        },
    )
    source, report = (
        zero_barrier._select_optimizer_directed_support_release_source(
            candidates=mixed_sign_candidates,
            solve_attempts=(
                {
                    "support_indices": (0, 1),
                    "formulation": (
                        "normalized_gas_reduced_linear_amounts"
                    ),
                    "accepted": False,
                    "local_kkt_passed": False,
                    "solve": {
                        "attempts": (
                            {
                                "optimizer_success": True,
                                "active_condensate_amounts": (-0.2, 0.3),
                            },
                        )
                    },
                },
            ),
        )
    )

    assert source["support_indices"] == (0, 1)
    np.testing.assert_array_equal(source["condensate_amounts"], [0.4, 0.6])
    assert report["selected"]
    assert report["selected_attempt_index"] == 0
    assert report["nonpositive_support_indices"] == (0,)


@pytest.mark.parametrize(
    ("directed_support", "nonpositive", "expected_source"),
    (
        ((0, 1, 8), (0,), "optimizer_terminated_nonpositive_alternative_basis"),
        ((1, 5, 7), (7,), "selected_basic_support"),
    ),
)
def test_support_release_source_does_not_repeat_an_already_tried_face(
    directed_support: tuple[int, ...],
    nonpositive: tuple[int, ...],
    expected_source: str,
) -> None:
    directed_amounts = np.zeros(9, dtype=np.float64)
    directed_amounts[np.asarray(directed_support, dtype=np.int64)] = 0.25

    source, report = zero_barrier._choose_support_release_source(
        default_support_indices=(1, 5),
        default_condensate_amounts=np.ones(9, dtype=np.float64),
        optimizer_directed_source={
            "support_indices": directed_support,
            "condensate_amounts": directed_amounts,
        },
        optimizer_directed_report={
            "nonpositive_support_indices": nonpositive,
        },
        already_tried_supports=((1, 5),),
    )

    assert report["selected_source"] == expected_source
    if expected_source == "selected_basic_support":
        assert source["support_indices"] == (1, 5)
        assert report["fallback_reason"] == "suggested_face_already_tried"
    else:
        assert source["support_indices"] == directed_support
        assert report["optimizer_directed_source_used"]


def _run_mock_alternative_portfolio(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: tuple[str, ...],
    evaluation_limit: int,
    excluded_supports: tuple[tuple[int, ...], ...] = (),
    downstream_function_evaluation_reserve: int = 0,
):
    monkeypatch.setattr(
        zero_barrier,
        "_build_alternative_basic_support_candidates",
        lambda **kwargs: _mock_alternative_portfolio_candidates(),
    )
    calls = []

    def fake_solve(**kwargs):
        outcome = outcomes[len(calls)]
        calls.append(tuple(kwargs["support_indices"]))
        budget = kwargs["function_evaluation_budget"]
        budget.consume(1)
        local_kkt = outcome in ("local", "accepted")
        accepted = outcome == "accepted"
        candidate = {
            "accepted": accepted,
            "gas_log_amounts": np.zeros(1),
            "condensate_amounts": kwargs["condensate_amounts_init"],
            "total_gas_log_amount": 0.0,
            "element_potential": np.zeros(1),
            "support_indices": tuple(kwargs["support_indices"]),
            "optimizer_success": True,
            "optimizer_status": 1,
            "optimizer_message": "unit test",
            "function_evaluations": 1,
            "audit": {
                "accepted": accepted,
                "finite": True,
                "positive_active_amounts": local_kkt,
                "gas_stationarity_max_abs": 0.0,
                "active_condensate_driving_max_abs": 0.0,
                "budget_scaled_max_abs": 0.0,
                "total_density_scaled_abs": 0.0,
            },
        }
        return {
            "accepted": accepted,
            "candidate": candidate,
            "report": {
                "attempts": ({"function_evaluations": 1},),
            },
        }

    monkeypatch.setattr(
        zero_barrier,
        "_solve_normalized_gas_reduced_linear_support",
        fake_solve,
    )
    budget = zero_barrier._FunctionEvaluationBudget(evaluation_limit)
    result = zero_barrier._solve_alternative_basic_support_portfolio(
        gas_formula_matrix=np.ones((1, 1)),
        condensate_formula_matrix_full=np.ones((1, 2)),
        target_inventory=np.ones(1),
        gas_standard_source=np.zeros(1),
        condensate_standard_source_full=np.zeros(2),
        gas_log_amounts_init=np.zeros(1),
        condensate_amounts_init=np.asarray([0.5, 0.5]),
        total_gas_log_amount_init=0.0,
        element_potential_init=np.zeros(1),
        support_indices=(0, 1),
        condensate_valid_mask=np.ones(2, dtype=bool),
        budget_scale=np.ones(1),
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=10,
        enabled=True,
        excluded_supports=excluded_supports,
        downstream_function_evaluation_reserve=(
            downstream_function_evaluation_reserve
        ),
        function_evaluation_budget=budget,
    )
    return result, calls, budget


def test_alternative_basic_support_continues_after_rejected_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls, budget = _run_mock_alternative_portfolio(
        monkeypatch,
        outcomes=("rejected", "accepted"),
        evaluation_limit=2,
    )

    assert result["accepted"]
    assert result["selected"]
    assert result["candidate"]["support_indices"] == (1,)
    assert calls == [(0,), (1,)]
    assert budget.used == 2


def test_alternative_basic_support_skips_an_already_tried_basis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls, budget = _run_mock_alternative_portfolio(
        monkeypatch,
        outcomes=("accepted",),
        evaluation_limit=1,
        excluded_supports=((0,),),
    )

    assert result["accepted"]
    assert result["candidate"]["support_indices"] == (1,)
    assert result["report"]["excluded_support_indices"] == ((0,),)
    assert calls == [(1,)]
    assert budget.used == 1


def test_alternative_basic_support_retries_after_selected_basis_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BasicSolution:
        success = True
        status = 0
        message = "unit-test basic LP vertex"
        nit = 1
        x = np.asarray([2.0 / 3.0, 0.0])

    monkeypatch.setattr(
        zero_barrier,
        "linprog",
        lambda *args, **kwargs: BasicSolution(),
    )
    calls = []

    def fake_normalized_solve(**kwargs):
        support = tuple(kwargs["support_indices"])
        calls.append(support)
        kwargs["function_evaluation_budget"].consume(1)
        amounts = np.zeros(2, dtype=np.float64)
        amounts[support[0]] = 1.0
        audit = zero_barrier._physical_zero_barrier_audit(
            gas_formula_matrix=np.ones((1, 1)),
            condensate_formula_matrix_full=np.ones((1, 2)),
            target_inventory=np.asarray([1.5]),
            gas_standard_source=np.zeros(1),
            condensate_standard_source_full=np.zeros(2),
            gas_log_amounts=np.log(np.asarray([0.5])),
            condensate_amounts=amounts,
            total_gas_log_amount=float(np.log(0.5)),
            element_potential=np.zeros(1),
            support_indices=support,
            condensate_valid_mask=np.ones(2, dtype=bool),
            budget_scale=np.ones(1),
            optimizer_success=True,
            optimizer_status=1,
            stationarity_tolerance=1.0e-8,
            budget_tolerance=1.0e-8,
            total_density_tolerance=1.0e-8,
            support_closure_tolerance=1.0e-8,
        )
        accepted = len(calls) > 1
        if not accepted:
            audit = dict(audit)
            audit["accepted"] = False
            audit["budget_scaled_max_abs"] = 1.0
        candidate = {
            "accepted": accepted,
            "gas_log_amounts": np.log(np.asarray([0.5])),
            "condensate_amounts": amounts,
            "total_gas_log_amount": float(np.log(0.5)),
            "element_potential": np.zeros(1),
            "support_indices": support,
            "optimizer_success": True,
            "optimizer_status": 1,
            "optimizer_message": "unit test",
            "function_evaluations": 1,
            "audit": audit,
        }
        return {
            "accepted": accepted,
            "candidate": candidate,
            "report": {
                "schema": "unit_test_normalized_solve",
                "attempts": (
                    {
                        "support_indices": support,
                        "function_evaluations": 1,
                    },
                ),
            },
        }

    monkeypatch.setattr(
        zero_barrier,
        "_solve_normalized_gas_reduced_linear_support",
        fake_normalized_solve,
    )
    budget = zero_barrier._FunctionEvaluationBudget(31)
    result = zero_barrier._polish_zero_barrier_support_once(
        gas_formula_matrix=np.ones((1, 1)),
        condensate_formula_matrix_full=np.ones((1, 2)),
        target_inventory=np.asarray([1.5]),
        gas_standard_source=np.zeros(1),
        condensate_standard_source_full=np.zeros(2),
        gas_log_amounts_init=np.log(np.asarray([0.5])),
        condensate_amounts_init=np.asarray([0.5, 0.5]),
        total_gas_log_amount_init=float(np.log(0.5)),
        element_potential_init=np.zeros(1),
        support_indices=(0, 1),
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=10,
        function_evaluation_budget=budget,
        use_zero_barrier_dual=False,
        use_finite_barrier_homotopy=False,
    )

    assert result.accepted
    assert calls == [(0,), (1,)]
    portfolio = result.report["alternative_basic_support_portfolio"]
    assert portfolio["trigger"] == (
        "selected_basic_support_local_root_failed"
    )
    assert portfolio["excluded_support_indices"] == ((0,),)
    assert portfolio["selected_support_indices"] == (1,)
    assert budget.used == 2
    linear_evaluations, reduced_evaluations = (
        zero_barrier._zero_barrier_report_function_evaluations(result.report)
    )
    assert linear_evaluations == 0
    assert reduced_evaluations == budget.used


def test_alternative_basic_support_returns_local_root_for_outer_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls, budget = _run_mock_alternative_portfolio(
        monkeypatch,
        outcomes=("local",),
        evaluation_limit=2,
    )

    assert not result["accepted"]
    assert result["selected"]
    assert result["candidate"]["support_indices"] == (0,)
    assert result["report"]["local_kkt_selected"]
    assert calls == [(0,)]
    assert budget.used == 1


def test_alternative_basic_support_fails_closed_after_all_rejections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls, budget = _run_mock_alternative_portfolio(
        monkeypatch,
        outcomes=("rejected", "rejected"),
        evaluation_limit=3,
    )

    assert not result["accepted"]
    assert not result["selected"]
    assert result["candidate"] is None
    assert result["report"]["stop_reason"] == "all_candidates_rejected"
    assert calls == [(0,), (1,)]
    assert budget.used == 2
    assert budget.remaining == 1


def test_alternative_basic_support_fails_closed_at_shared_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls, budget = _run_mock_alternative_portfolio(
        monkeypatch,
        outcomes=("rejected",),
        evaluation_limit=1,
    )

    assert not result["accepted"]
    assert not result["selected"]
    assert result["candidate"] is None
    assert result["report"]["stop_reason"] == (
        "function_evaluation_limit_reached"
    )
    assert calls == [(0,)]
    assert budget.used == budget.limit


@pytest.mark.parametrize(
    (
        "evaluation_limit",
        "outcomes",
        "expected_calls",
        "expected_used",
        "expected_portfolio_limit",
    ),
    (
        (20, (), [], 0, 0),
        (25, ("rejected", "rejected"), [(0,), (1,)], 2, 5),
        (30, ("rejected", "rejected"), [(0,), (1,)], 2, 10),
    ),
)
def test_alternative_basic_support_preserves_downstream_budget(
    monkeypatch: pytest.MonkeyPatch,
    evaluation_limit: int,
    outcomes: tuple[str, ...],
    expected_calls: list[tuple[int, ...]],
    expected_used: int,
    expected_portfolio_limit: int,
) -> None:
    result, calls, budget = _run_mock_alternative_portfolio(
        monkeypatch,
        outcomes=outcomes,
        evaluation_limit=evaluation_limit,
        downstream_function_evaluation_reserve=20,
    )

    assert not result["selected"]
    assert calls == expected_calls
    assert budget.used == expected_used
    assert budget.remaining >= 20
    report = result["report"]
    assert report["attempted"] is bool(expected_calls)
    assert report["downstream_function_evaluation_reserve_requested"] == 20
    assert report["downstream_function_evaluation_reserve"] == 20
    assert report["portfolio_function_evaluation_limit"] == (
        expected_portfolio_limit
    )
    assert report["support_release_source_indices"] == (0,)
    assert result["support_release_source"]["support_indices"] == (0,)
    np.testing.assert_array_equal(
        result["support_release_source"]["condensate_amounts"],
        [1.0, 0.0],
    )


def test_budget_partition_preserves_all_remaining_work_when_reserve_is_larger(
) -> None:
    parent = zero_barrier._FunctionEvaluationBudget(7)

    solve_budget, child, reserved = (
        zero_barrier._partition_function_evaluation_budget(parent, 20)
    )

    assert solve_budget is child
    assert child.limit == 0
    assert reserved == 7
    assert parent.remaining == 7


def test_zero_barrier_active_set_adds_tied_phase_with_zero_target_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    first = _mock_polish_result(
        accepted=False,
        support_indices=(0,),
        full_driving=(0.0, -0.5, -0.5),
    )
    second = _mock_polish_result(
        accepted=True,
        support_indices=(0, 1),
        full_driving=(0.0, 0.0, 1.0),
    )
    first.report["basic_support_reduction"] = {
        "schema": "unit_test_basic_support_reduction",
        "applied": True,
    }
    first.report["alternative_basic_support_portfolio"] = {
        "schema": "unit_test_alternative_basic_support",
        "enabled": True,
        "attempted": True,
        "accepted": False,
        "local_kkt_selected": True,
        "selected_support_indices": (0,),
        "solve_attempts": (),
    }

    def fake_polish_once(**kwargs):
        calls.append(kwargs)
        return (first, second)[len(calls) - 1]

    monkeypatch.setattr(
        zero_barrier,
        "_polish_zero_barrier_support_once",
        fake_polish_once,
    )

    result = polish_zero_barrier_active_support(
        **_mock_polish_arguments(
            support_indices=(0, 1),
            target_inventory=np.asarray([1.0, 0.0], dtype=np.float64),
        )
    )

    closure = result.report["exact_active_set_closure"]
    assert result.accepted
    assert result.support_indices == (0, 1)
    assert tuple(call["support_indices"] for call in calls) == (
        (0, 1),
        (0, 1),
    )
    assert tuple(call["reduce_initial_support"] for call in calls) == (
        True,
        False,
    )
    np.testing.assert_array_equal(
        calls[1]["gas_log_amounts_init"], first.gas_log_amounts
    )
    np.testing.assert_array_equal(
        calls[1]["condensate_amounts_init"], first.condensate_amounts
    )
    assert calls[1]["total_gas_log_amount_init"] == first.total_gas_log_amount
    np.testing.assert_array_equal(
        calls[1]["element_potential_init"], first.element_potential
    )
    assert closure["termination_reason"] == "accepted"
    assert closure["added_support_indices"] == (1,)
    assert closure["visited_supports"] == ((0, 1), (0, 1))
    assert closure["visited_output_supports"] == ((0,), (0, 1))
    assert result.report["basic_support_reduction"] == {
        "schema": "unit_test_basic_support_reduction",
        "applied": True,
    }
    assert result.report["alternative_basic_support_portfolio"] == (
        first.report["alternative_basic_support_portfolio"]
    )
    assert closure["rounds"][0][
        "alternative_basic_support_attempted"
    ]
    assert closure["rounds"][0][
        "alternative_basic_support_indices"
    ] == (0,)
    assert not closure["rounds"][1][
        "alternative_basic_support_attempted"
    ]
    np.testing.assert_array_equal(
        calls[1]["target_inventory"], np.asarray([1.0, 0.0])
    )


def test_zero_barrier_active_set_excludes_temperature_invalid_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    results = (
        _mock_polish_result(
            accepted=False,
            support_indices=(0,),
            full_driving=(0.0, -2.0, -1.0),
            inactive_violation_max_abs=1.0,
        ),
        _mock_polish_result(
            accepted=True,
            support_indices=(0, 2),
            full_driving=(0.0, -2.0, 0.0),
            inactive_violation_max_abs=0.0,
        ),
    )

    def fake_polish_once(**kwargs):
        calls.append(kwargs)
        return results[len(calls) - 1]

    monkeypatch.setattr(
        zero_barrier,
        "_polish_zero_barrier_support_once",
        fake_polish_once,
    )
    arguments = _mock_polish_arguments(
        condensate_valid_mask=np.asarray([True, False, True])
    )

    result = polish_zero_barrier_active_support(**arguments)

    assert result.accepted
    assert tuple(call["support_indices"] for call in calls) == ((0,), (0, 2))
    assert result.report["exact_active_set_closure"][
        "added_support_indices"
    ] == (2,)


def test_zero_barrier_active_set_stops_on_repeated_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    rejected = _mock_polish_result(
        accepted=False,
        support_indices=(0,),
        full_driving=(0.0, -1.0, 1.0),
    )

    def fake_polish_once(**kwargs):
        calls.append(kwargs)
        return rejected

    monkeypatch.setattr(
        zero_barrier,
        "_polish_zero_barrier_support_once",
        fake_polish_once,
    )

    result = polish_zero_barrier_active_support(**_mock_polish_arguments())

    closure = result.report["exact_active_set_closure"]
    assert not result.accepted
    assert len(calls) == 2
    assert closure["termination_reason"] == "support_cycle_detected"
    assert closure["round_count"] == 2
    assert closure["added_support_indices"] == (1,)
    assert closure["blacklisted_addition_edges"] == (((0,), 1),)


def test_zero_barrier_active_set_tries_next_phase_after_rejected_addition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    results = (
        _mock_polish_result(
            accepted=False,
            support_indices=(0,),
            full_driving=(0.0, -2.0, -1.0),
        ),
        _mock_polish_result(
            accepted=False,
            support_indices=(0,),
            full_driving=(0.0, -2.0, -1.0),
        ),
        _mock_polish_result(
            accepted=True,
            support_indices=(0, 2),
            full_driving=(0.0, 1.0, 0.0),
        ),
    )

    def fake_polish_once(**kwargs):
        calls.append(kwargs)
        return results[len(calls) - 1]

    monkeypatch.setattr(
        zero_barrier,
        "_polish_zero_barrier_support_once",
        fake_polish_once,
    )

    result = polish_zero_barrier_active_support(**_mock_polish_arguments())

    closure = result.report["exact_active_set_closure"]
    assert result.accepted
    assert tuple(call["support_indices"] for call in calls) == (
        (0,),
        (0, 1),
        (0, 2),
    )
    assert closure["termination_reason"] == "accepted"
    assert closure["added_support_indices"] == (1, 2)
    assert closure["blacklisted_addition_edges"] == (((0,), 1),)
    assert closure["rounds"][1]["rejected_added_support_index"] == 1


def test_zero_barrier_active_set_backtracks_to_root_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    results = (
        _mock_polish_result(
            accepted=False,
            support_indices=(),
            full_driving=(-1.0, -2.0, 1.0),
        ),
        _mock_polish_result(
            accepted=False,
            support_indices=(1,),
            full_driving=(-1.0, 0.0, 1.0),
        ),
        _mock_polish_result(
            accepted=False,
            support_indices=(),
            full_driving=(-1.0, -2.0, 1.0),
        ),
        _mock_polish_result(
            accepted=True,
            support_indices=(0,),
            full_driving=(0.0, 1.0, 1.0),
        ),
    )
    results[2].gas_log_amounts[:] = -9.0

    def fake_polish_once(**kwargs):
        calls.append(kwargs)
        return results[len(calls) - 1]

    monkeypatch.setattr(
        zero_barrier,
        "_polish_zero_barrier_support_once",
        fake_polish_once,
    )

    result = polish_zero_barrier_active_support(
        **_mock_polish_arguments(support_indices=(0, 1))
    )

    closure = result.report["exact_active_set_closure"]
    assert result.accepted
    assert result.support_indices == (0,)
    assert tuple(call["support_indices"] for call in calls) == (
        (0, 1),
        (1,),
        (1, 0),
        (0,),
    )
    np.testing.assert_array_equal(
        calls[3]["gas_log_amounts_init"], results[0].gas_log_amounts
    )
    assert closure["search_strategy"] == "bounded_depth_first"
    assert closure["termination_reason"] == "accepted"
    assert closure["added_support_indices"] == (1, 0, 0)
    assert closure["blacklisted_addition_edges"] == (
        ((1,), 0),
        ((), 1),
    )
    assert closure["visited_output_supports"] == ((), (1,), (0,))
    assert closure["rounds"][2]["rejected_added_support_index"] == 0
    assert closure["rounds"][2]["rejected_addition_edge"] == ((1,), 0)
    assert closure["rounds"][2]["backtracked_support_indices"] == ((1,),)
    assert closure["rounds"][2]["backtracked_addition_edges"] == (
        ((1,), 0),
        ((), 1),
    )
    assert closure["rounds"][2]["addition_base_support_indices"] == ()


def test_zero_barrier_active_set_does_not_retry_local_budget_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    rejected = _mock_polish_result(
        accepted=False,
        support_indices=(0,),
        full_driving=(0.0, -1.0, 1.0),
        budget_scaled_max_abs=1.0e-3,
    )

    def fake_polish_once(**kwargs):
        calls.append(kwargs)
        return rejected

    monkeypatch.setattr(
        zero_barrier,
        "_polish_zero_barrier_support_once",
        fake_polish_once,
    )

    result = polish_zero_barrier_active_support(**_mock_polish_arguments())

    closure = result.report["exact_active_set_closure"]
    assert not result.accepted
    assert len(calls) == 1
    assert closure["termination_reason"] == "local_kkt_failed"
    assert closure["rounds"][0]["local_kkt_failure_reasons"] == (
        "element_budget",
    )


def test_zero_barrier_active_set_enforces_cumulative_evaluation_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    outputs = (
        ((0,), (0.0, -1.0, 1.0)),
        ((1,), (1.0, 0.0, -1.0)),
        ((2,), (-1.0, 1.0, 0.0)),
        ((), (-1.0, 1.0, 1.0)),
    )

    def fake_polish_once(**kwargs):
        budget = kwargs["function_evaluation_budget"]
        call_evaluations = min(
            2 * kwargs["max_function_evaluations"], budget.remaining
        )
        budget.consume(call_evaluations)
        calls.append(kwargs)
        support, driving = outputs[len(calls) - 1]
        return _mock_polish_result(
            accepted=False,
            support_indices=support,
            full_driving=driving,
            function_evaluations=call_evaluations,
        )

    monkeypatch.setattr(
        zero_barrier,
        "_polish_zero_barrier_support_once",
        fake_polish_once,
    )

    result = polish_zero_barrier_active_support(
        **_mock_polish_arguments(),
        max_function_evaluations=2,
    )

    closure = result.report["exact_active_set_closure"]
    assert not result.accepted
    assert len(calls) == 4
    assert closure["termination_reason"] == "function_evaluation_limit_reached"
    assert closure["cumulative_function_evaluations"] == 16
    assert closure["function_evaluation_limit"] == 16
    assert all(
        round_report["function_evaluations"] <= 4
        for round_report in closure["rounds"]
    )


def test_zero_barrier_active_set_charges_solver_exceptions_to_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_dual_support_oracle(monkeypatch)
    residual_evaluations = 0

    def raising_solver(residual, values, **kwargs):
        nonlocal residual_evaluations
        for _ in range(kwargs["max_nfev"]):
            residual(values)
            residual_evaluations += 1
        raise ValueError("unit-test solver failure")

    monkeypatch.setattr(
        zero_barrier,
        "_least_squares_with_scipy_overflow_guard",
        raising_solver,
    )
    condensate_formula = np.eye(5, dtype=np.float64)

    result = polish_zero_barrier_active_support(
        gas_formula_matrix=np.eye(5, dtype=np.float64),
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=np.ones(5, dtype=np.float64),
        gas_standard_source=np.zeros(5, dtype=np.float64),
        condensate_standard_source_full=np.zeros(5, dtype=np.float64),
        gas_log_amounts_init=np.log(
            np.full(5, 0.5, dtype=np.float64)
        ),
        condensate_amounts_init=np.full(5, 0.5, dtype=np.float64),
        total_gas_log_amount_init=float(np.log(2.5)),
        element_potential_init=np.zeros(5, dtype=np.float64),
        support_indices=(0, 1, 2, 3, 4),
        max_function_evaluations=2,
    )

    closure = result.report["exact_active_set_closure"]
    assert not result.accepted
    assert closure["termination_reason"] == "function_evaluation_limit_reached"
    assert closure["cumulative_function_evaluations"] == 16
    assert closure["function_evaluation_limit"] == 16
    assert residual_evaluations == 16


def test_zero_barrier_polish_drops_a_negative_phase_and_updates_support() -> None:
    gas_formula = np.eye(2, dtype=np.float64)
    condensate_formula = np.asarray([[0.0], [1.0]], dtype=np.float64)
    target = np.asarray([1.0, 0.01], dtype=np.float64)
    active_gas = np.asarray([1.0, 0.25], dtype=np.float64)

    result = polish_zero_barrier_active_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=np.zeros(2),
        condensate_standard_source_full=np.asarray([np.log(0.2)]),
        gas_log_amounts_init=np.log(active_gas),
        condensate_amounts_init=np.asarray([1.0e-3]),
        total_gas_log_amount_init=np.log(np.sum(active_gas)),
        element_potential_init=np.log(np.asarray([0.8, 0.2])),
        support_indices=(0,),
        budget_relative_floor=1.0e-12,
    )

    assert result.accepted
    assert result.support_indices == ()
    assert result.report["dropped_support_indices"] == (0,)
    assert result.condensate_amounts.tolist() == [0.0]
    assert np.exp(result.gas_log_amounts) == pytest.approx(target)
    assert result.report["full_condensate_driving"][0] > 0.0
    assert result.report["selected_numerical_formulation"] == (
        "normalized_gas_reduced_linear_amounts"
    )
    assert not result.report["linear_amount_physical_audit"]["attempted"]


def test_zero_barrier_polish_rejects_incompatible_active_stationarity() -> None:
    (
        gas_formula,
        condensate_formula,
        expected_gas,
        expected_condensates,
        expected_potential,
        gamma,
        hcond,
        target,
    ) = _one_active_phase_problem()
    hcond[1] = -0.3

    result = polish_zero_barrier_active_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond,
        gas_log_amounts_init=np.log(expected_gas),
        condensate_amounts_init=np.asarray([0.65, 0.03]),
        total_gas_log_amount_init=0.0,
        element_potential_init=expected_potential,
        support_indices=(0, 1),
        budget_relative_floor=1.0e-12,
    )

    assert not result.accepted
    assert (
        result.report["active_condensate_driving_max_abs"] > 1.0e-8
        or result.report["total_density_scaled_abs"] > 1.0e-8
    )


def test_zero_barrier_polish_uses_relative_scaling_for_positive_trace() -> None:
    gas_formula = np.eye(2, dtype=np.float64)
    condensate_formula = np.asarray([[1.0], [0.0]], dtype=np.float64)
    expected_gas = np.asarray([0.5, 1.0e-20], dtype=np.float64)
    expected_condensate = np.asarray([0.5], dtype=np.float64)
    target = np.asarray([1.0, 1.0e-20], dtype=np.float64)
    qtot = float(np.log(np.sum(expected_gas)))
    potential = np.asarray([0.2, -0.4], dtype=np.float64)
    gamma = potential - np.log(expected_gas) + qtot

    result = polish_zero_barrier_active_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=np.asarray([potential[0]]),
        gas_log_amounts_init=np.log(np.asarray([0.5, 1.0e-18])),
        condensate_amounts_init=expected_condensate,
        total_gas_log_amount_init=qtot,
        element_potential_init=potential,
        support_indices=(0,),
        budget_relative_floor=1.0e-6,
    )

    reconstructed = (
        gas_formula @ np.exp(result.gas_log_amounts)
        + condensate_formula @ result.condensate_amounts
    )
    assert result.accepted
    assert reconstructed[1] == pytest.approx(target[1], rel=1.0e-8)
    assert result.report["budget_scaled_max_abs"] < 1.0e-8
    assert result.report["budget_scaling"].startswith("relative_for_nonzero")
    assert result.report["selected_numerical_formulation"] == (
        "normalized_gas_reduced_linear_amounts"
    )
    assert not result.report["linear_amount_physical_audit"]["attempted"]


def test_zero_barrier_polish_regularizes_trace_barrier_initializer() -> None:
    gas_formula = np.asarray(
        [
            [2.0, 2.0, 0.0, 0.0, 4.0],
            [0.0, 1.0, 2.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    condensate_formula = np.asarray(
        [[0.0, 0.0], [1.0, 2.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    target = np.asarray(
        [
            2.8959369615671957e8,
            7.7510262401527523e-22,
            1.0406303843280405e7,
        ],
        dtype=np.float64,
    )
    gamma = np.asarray(
        [
            -78.68871595,
            -171.79173500,
            -89.90713332,
            -154.77418944,
            -222.42561704,
        ],
        dtype=np.float64,
    )

    result = polish_zero_barrier_active_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=np.asarray(
            [-227.97601192, -346.40090319],
            dtype=np.float64,
        ),
        gas_log_amounts_init=np.asarray(
            [
                18.63566502187771,
                -2.15476346e16,
                -4.30952692e16,
                -2.15476346e16,
                16.157922319260653,
            ],
            dtype=np.float64,
        ),
        condensate_amounts_init=np.asarray(
            [target[1], 0.0],
            dtype=np.float64,
        ),
        total_gas_log_amount_init=18.71626062818451,
        element_potential_init=np.asarray(
            [-39.3846558, -2.15476346e16, -67.4453322],
            dtype=np.float64,
        ),
        support_indices=(0,),
        budget_relative_floor=1.0e-6,
    )

    regularization = result.report["initializer_regularization"]
    reconstructed = (
        gas_formula @ np.exp(result.gas_log_amounts)
        + condensate_formula @ result.condensate_amounts
    )
    assert result.accepted
    assert result.support_indices == (0,)
    assert regularization["applied"]
    assert regularization["capacity_fraction"] == pytest.approx(
        np.sqrt(np.finfo(np.float64).eps)
    )
    assert regularization["regularized_gas_count"] == 3
    assert regularization["regularized_gas_mask"] == (
        False,
        True,
        True,
        True,
        False,
    )
    assert regularization["element_potential_recomputed"]
    assert regularization["element_potential_fit_rank"] == 3
    assert reconstructed == pytest.approx(target, rel=1.0e-12)
    assert result.condensate_amounts[0] == pytest.approx(
        1.27414995e-22,
        rel=1.0e-6,
    )
    assert result.report["active_condensate_driving_max_abs"] < 1.0e-8


def test_reduced_log_fallback_branches_layer_839_trace_support() -> None:
    gas_formula = np.asarray(
        [
            [2.0, 2.0, 0.0, 0.0, 4.0],
            [0.0, 1.0, 2.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    condensate_formula = np.asarray(
        [[0.0, 0.0], [1.0, 2.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    target = (
        np.asarray(
            [0.9653123147, 5.43352573e-38, 0.0346876853],
            dtype=np.float64,
        )
        * 3.0e8
    )
    gamma = np.asarray(
        [
            -104.36002887563116,
            -225.45008100585233,
            -119.1085856180165,
            -201.02350479285428,
            -296.7652055580194,
        ],
        dtype=np.float64,
    )
    hcond = np.asarray(
        [-296.684067078042, -451.80912549650316],
        dtype=np.float64,
    )
    gas_init = np.asarray(
        [
            (target[0] - 4.0 * target[2]) / 2.0,
            target[1] * 1.0e-3,
            target[1] * 1.0e-3,
            target[1] * 1.0e-3,
            target[2],
        ],
        dtype=np.float64,
    )

    result = polish_zero_barrier_active_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond,
        gas_log_amounts_init=np.log(gas_init),
        condensate_amounts_init=np.asarray(
            [0.5 * target[1], 0.25 * target[1]],
            dtype=np.float64,
        ),
        total_gas_log_amount_init=float(np.log(np.sum(gas_init))),
        element_potential_init=np.zeros(3, dtype=np.float64),
        support_indices=(0, 1),
        budget_relative_floor=1.0e-6,
    )

    gas_inventory = gas_formula @ np.exp(result.gas_log_amounts)
    reconstructed = (
        gas_inventory + condensate_formula @ result.condensate_amounts
    )
    linear_supports = tuple(
        attempt["support_indices"] for attempt in result.report["attempts"]
    )
    fallback = result.report["reduced_log_domain_fallback"]

    assert result.accepted
    assert result.support_indices == (0,)
    selected = result.report["selected_numerical_formulation"]
    # BLAS-level differences can reverse the order of two negative amounts
    # near 1e-27.  Either route is valid if its own audit and closure pass.
    assert selected in {
        "capacity_scaled_linear_amounts",
        "normalized_gas_reduced_linear_amounts",
        "reduced_log_domain_support_search",
        "structural_zero_reduced_log_domain",
        "structural_zero_normalized_gas_reduced_linear_amounts",
    }
    basic_reduction = result.report["basic_support_reduction"]
    if basic_reduction["applied"] and linear_supports:
        assert basic_reduction["initial_support_indices"] == (0, 1)
        assert basic_reduction["final_support_indices"] == (0,)
        assert linear_supports[0] == (0,)
    elif linear_supports:
        assert linear_supports[0] == (0, 1)
    if selected == "reduced_log_domain_support_search":
        assert linear_supports == ((0, 1), (1,), ())
        assert all(
            amount < 0.0
            for amount in result.report["attempts"][0][
                "active_condensate_amounts"
            ]
        )
        assert fallback["accepted"]
        assert fallback["visited_supports"] == ((0, 1), (1,), (0,))
        assert tuple(node["accepted"] for node in fallback["nodes"]) == (
            False,
            False,
            True,
        )
        assert all(
            not node["solve"]["greedy_drop_enabled"]
            for node in fallback["nodes"]
        )
    elif selected == "capacity_scaled_linear_amounts":
        assert linear_supports[-1] == (0,)
        assert result.report["linear_amount_physical_audit"]["accepted"]
        assert not fallback["attempted"]
        assert not fallback["accepted"]
        assert fallback["skip_reason"] == (
            "linear_amount_physical_audit_accepted"
        )
    else:
        assert not linear_supports
        assert not result.report["linear_amount_physical_audit"]["attempted"]
        assert fallback["skip_reason"] == (
            "reduced_primary_physical_audit_accepted"
        )
        if selected.startswith("structural_zero_"):
            rescue = result.report[
                "structural_zero_reduced_log_rescue"
            ]
            assert rescue["attempted"]
            assert rescue["accepted"]
    assert reconstructed == pytest.approx(target, rel=1.0e-12)
    assert gas_inventory[1] / target[1] == pytest.approx(
        0.7947931234846488,
        rel=1.0e-10,
    )
    assert result.condensate_amounts[0] > 0.0
    assert result.condensate_amounts[1] == 0.0
    assert result.report["full_condensate_driving"][1] > 0.0

    # Exercise the breadth-first fallback independently of the linear route.
    branched = _solve_reduced_log_domain_support_branches(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond,
        gas_log_amounts_init=np.log(gas_init),
        condensate_amounts_init=np.asarray(
            [0.5 * target[1], 0.25 * target[1]],
            dtype=np.float64,
        ),
        total_gas_log_amount_init=float(np.log(np.sum(gas_init))),
        element_potential_init=np.zeros(3, dtype=np.float64),
        support_indices=(0, 1),
        condensate_valid_mask=np.ones(2, dtype=bool),
        budget_scale=np.reciprocal(target),
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=400,
    )
    branch_report = branched["report"]

    assert branched["accepted"]
    assert branched["candidate"]["support_indices"] == (0,)
    assert branch_report["visited_supports"] == ((0, 1), (1,), (0,))
    assert tuple(
        node["accepted"] for node in branch_report["nodes"]
    ) == (False, False, True)
    assert all(
        not node["solve"]["greedy_drop_enabled"]
        for node in branch_report["nodes"]
    )


def test_reduced_log_support_converges_at_capacity_boundary() -> None:
    delta = 1.0e-14
    target = np.ones(2, dtype=np.float64)
    qtot = float(np.log1p(delta))
    result = _solve_reduced_log_domain_active_support(
        gas_formula_matrix=np.eye(2, dtype=np.float64),
        condensate_formula_matrix_full=np.asarray(
            [[0.0], [1.0]], dtype=np.float64
        ),
        target_inventory=target,
        gas_standard_source=np.asarray(
            [qtot, -np.log(delta) + qtot], dtype=np.float64
        ),
        condensate_standard_source_full=np.zeros(1, dtype=np.float64),
        gas_log_amounts_init=np.log(
            np.asarray([0.5, 0.5], dtype=np.float64)
        ),
        condensate_amounts_init=np.asarray([0.5], dtype=np.float64),
        total_gas_log_amount_init=0.0,
        element_potential_init=np.zeros(2, dtype=np.float64),
        support_indices=(0,),
        condensate_valid_mask=np.asarray([True]),
        budget_scale=np.ones(2, dtype=np.float64),
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=400,
        allow_greedy_drop=False,
    )

    candidate = result["candidate"]
    attempt = result["report"]["attempts"][0]
    audit = candidate["audit"]
    assert result["accepted"]
    assert candidate["support_indices"] == (0,)
    assert candidate["condensate_amounts"][0] / target[1] > 0.999999999
    assert attempt["optimizer_success"]
    assert not attempt["active_phase_at_lower_bound"]
    assert audit["budget_scaled_max_abs"] < 1.0e-8
    assert audit["gas_stationarity_max_abs"] < 1.0e-8
    assert audit["active_condensate_driving_max_abs"] < 1.0e-8
    assert audit["inactive_condensate_violation_max_abs"] == 0.0
    assert audit["total_density_scaled_abs"] < 1.0e-8


def test_reduced_log_support_keeps_signed_charge_budget_linear() -> None:
    gas_formula = np.asarray(
        [[1.0, 1.0, 0.0], [0.0, 1.0, -1.0]],
        dtype=np.float64,
    )
    condensate_formula = np.empty((2, 0), dtype=np.float64)
    target = np.asarray([1.0, 0.0], dtype=np.float64)
    result = _solve_reduced_log_domain_active_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=np.zeros(3, dtype=np.float64),
        condensate_standard_source_full=np.empty(0, dtype=np.float64),
        gas_log_amounts_init=np.log(
            np.asarray([0.2, 0.2, 0.2], dtype=np.float64)
        ),
        condensate_amounts_init=np.empty(0, dtype=np.float64),
        total_gas_log_amount_init=float(np.log(0.6)),
        element_potential_init=np.zeros(2, dtype=np.float64),
        support_indices=(),
        condensate_valid_mask=np.empty(0, dtype=bool),
        budget_scale=np.ones(2, dtype=np.float64),
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=400,
        allow_greedy_drop=False,
    )

    candidate = result["candidate"]
    reconstructed = (
        gas_formula @ np.exp(candidate["gas_log_amounts"])
        + condensate_formula @ candidate["condensate_amounts"]
    )
    assert result["accepted"]
    report = result["report"]
    assert report["schema"] == (
        "exogibbs_zero_barrier_reduced_log_domain_v2"
    )
    assert report["log_budget_rows"] == (0,)
    assert report["linear_budget_rows"] == (1,)
    attempt = report["attempts"][0]
    assert attempt["log_budget_residual_max_abs"] < 1.0e-8
    assert attempt["linear_budget_scaled_residual_max_abs"] < 1.0e-8
    np.testing.assert_allclose(reconstructed, target, atol=1.0e-12)
    assert candidate["audit"]["accepted"]


def test_mixed_budget_jacobian_includes_active_charged_condensate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked = False

    def checking_solver(fun, x0, *, jac, **kwargs):
        nonlocal checked
        del kwargs
        analytic = jac(x0)
        step = 1.0e-6
        numeric = np.column_stack(
            [
                (
                    fun(x0 + step * np.eye(x0.size)[column])
                    - fun(x0 - step * np.eye(x0.size)[column])
                )
                / (2.0 * step)
                for column in range(x0.size)
            ]
        )
        np.testing.assert_allclose(analytic, numeric, rtol=1.0e-6, atol=1.0e-8)
        assert analytic[-1, -1] != 0.0
        checked = True
        residual = fun(x0)
        return SimpleNamespace(
            x=x0,
            success=False,
            status=0,
            message="Jacobian inspection",
            nfev=1,
            cost=0.5 * float(residual @ residual),
            optimality=float(np.max(np.abs(analytic.T @ residual))),
        )

    monkeypatch.setattr(
        zero_barrier,
        "_least_squares_with_scipy_overflow_guard",
        checking_solver,
    )
    result = _solve_reduced_log_domain_active_support(
        gas_formula_matrix=np.asarray(
            [[1.0, 0.0], [0.0, -1.0]], dtype=np.float64
        ),
        condensate_formula_matrix_full=np.asarray(
            [[1.0], [1.0]], dtype=np.float64
        ),
        target_inventory=np.asarray([1.0, 0.0], dtype=np.float64),
        gas_standard_source=np.zeros(2, dtype=np.float64),
        condensate_standard_source_full=np.zeros(1, dtype=np.float64),
        gas_log_amounts_init=np.log(
            np.asarray([0.5, 0.1], dtype=np.float64)
        ),
        condensate_amounts_init=np.asarray([0.1], dtype=np.float64),
        total_gas_log_amount_init=float(np.log(0.6)),
        element_potential_init=np.zeros(2, dtype=np.float64),
        support_indices=(0,),
        condensate_valid_mask=np.ones(1, dtype=bool),
        budget_scale=np.ones(2, dtype=np.float64),
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=20,
        allow_greedy_drop=False,
    )

    assert checked
    assert result["report"]["budget_residual_formulation"] == (
        "mixed_log_linear"
    )
    assert result["report"]["log_budget_rows"] == (0,)
    assert result["report"]["linear_budget_rows"] == (1,)


def test_reduced_log_support_rejects_active_phase_at_amount_floor() -> None:
    true_relative_amount = 1.0e-12
    target = np.ones(2, dtype=np.float64)
    gas_fractions = np.asarray(
        [
            (1.0 - true_relative_amount) / (2.0 - true_relative_amount),
            1.0 / (2.0 - true_relative_amount),
        ],
        dtype=np.float64,
    )
    common_arguments = {
        "gas_formula_matrix": np.eye(2, dtype=np.float64),
        "condensate_formula_matrix_full": np.asarray(
            [[1.0], [0.0]], dtype=np.float64
        ),
        "target_inventory": target,
        "gas_standard_source": np.zeros(2, dtype=np.float64),
        "condensate_standard_source_full": np.asarray(
            [np.log(gas_fractions[0])], dtype=np.float64
        ),
        "gas_log_amounts_init": np.log(target),
        "condensate_amounts_init": np.asarray(
            [true_relative_amount], dtype=np.float64
        ),
        "total_gas_log_amount_init": float(np.log(np.sum(target))),
        "element_potential_init": np.log(
            np.asarray([0.5, 0.5], dtype=np.float64)
        ),
        "support_indices": (0,),
        "condensate_valid_mask": np.asarray([True]),
        "budget_scale": np.reciprocal(target),
        "stationarity_tolerance": 1.0e-8,
        "budget_tolerance": 1.0e-8,
        "total_density_tolerance": 1.0e-8,
        "support_closure_tolerance": 1.0e-8,
        "max_function_evaluations": 400,
    }

    branched = _solve_reduced_log_domain_support_branches(
        **common_arguments
    )
    first_node = branched["report"]["nodes"][0]
    first_attempt = first_node["solve"]["attempts"][0]

    assert first_attempt["physical_audit_accepted"]
    assert first_attempt["active_phase_at_lower_bound"]
    assert first_attempt["lower_bound_support_indices"] == (0,)
    assert not first_node["accepted"]
    assert branched["accepted"]
    assert branched["candidate"]["support_indices"] == ()
    assert branched["report"]["visited_supports"] == ((0,), ())

    greedy = _solve_reduced_log_domain_active_support(
        **common_arguments,
        allow_greedy_drop=True,
    )

    assert greedy["accepted"]
    assert greedy["candidate"]["support_indices"] == ()
    assert greedy["report"]["dropped_support_indices"] == (0,)

    exhausted_budget = zero_barrier._FunctionEvaluationBudget(
        limit=first_attempt["function_evaluations"]
    )
    exhausted = _solve_reduced_log_domain_active_support(
        **common_arguments,
        allow_greedy_drop=True,
        function_evaluation_budget=exhausted_budget,
    )

    assert not exhausted["accepted"]
    assert exhausted["candidate"]["active_phase_at_lower_bound"]
    assert exhausted["candidate"]["lower_bound_support_indices"] == (0,)
    assert exhausted["report"]["attempts"][-1]["failure_reason"] == (
        "function_evaluation_limit_reached"
    )


def _zero_row_inactive_phase_problem():
    return {
        "gas_formula_matrix": np.asarray([[1.0], [0.0]]),
        "condensate_formula_matrix_full": np.eye(2, dtype=np.float64),
        "target_inventory": np.asarray([1.0, 0.0]),
        "gas_standard_source": np.asarray([0.0]),
        "condensate_standard_source_full": np.asarray([0.0, -1.0]),
        "gas_log_amounts_init": np.asarray([np.log(0.5)]),
        "condensate_amounts_init": np.asarray([0.5, 0.0]),
        "total_gas_log_amount_init": float(np.log(0.5)),
        "element_potential_init": np.zeros(2, dtype=np.float64),
        "condensate_valid_mask": np.ones(2, dtype=bool),
        "budget_relative_floor": 1.0e-6,
    }


def test_structural_zero_reconstructs_inactive_phase_potential() -> None:
    result = polish_zero_barrier_active_support(
        **_zero_row_inactive_phase_problem(),
        support_indices=(0,),
    )

    rescue = result.report["structural_zero_reduced_log_rescue"]
    closure = result.report["exact_active_set_closure"]
    assert result.accepted
    assert result.support_indices == (0,)
    assert result.report["selected_numerical_formulation"] == (
        "structural_zero_normalized_gas_reduced_linear_amounts"
    )
    assert rescue["inner_formulation"] == (
        "normalized_gas_reduced_linear_amounts"
    )
    assert rescue["suppressed_gas_indices"] == ()
    assert rescue["inactive_zero_row_phase_potential_limits"][0][0] == 1
    assert result.element_potential[1] < -1.0
    assert result.report["inactive_condensate_violation_max_abs"] == 0.0
    assert not result.report["linear_amount_physical_audit"]["attempted"]
    assert closure["round_count"] == 1
    assert closure["added_support_indices"] == ()


def test_structural_zero_drops_active_impossible_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_dual_support_oracle(monkeypatch)
    result = polish_zero_barrier_active_support(
        **_zero_row_inactive_phase_problem(),
        support_indices=(0, 1),
    )

    rescue = result.report["structural_zero_reduced_log_rescue"]
    closure = result.report["exact_active_set_closure"]
    assert result.accepted
    assert result.support_indices == (0,)
    assert rescue["structural_zero_dropped_support_indices"] == (1,)
    assert rescue["accepted"]
    assert result.condensate_amounts[1] == 0.0
    assert not result.report["linear_amount_physical_audit"]["attempted"]
    assert closure["round_count"] == 1
    assert closure["added_support_indices"] == ()


def test_structural_zero_stabilizes_favorable_phases_across_zero_rows() -> None:
    gas_formula = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    condensate_formula = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )
    result = polish_zero_barrier_active_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=np.asarray([1.0, 0.0, 0.0]),
        gas_standard_source=np.zeros(3, dtype=np.float64),
        condensate_standard_source_full=np.asarray(
            [0.0, -100.0, -300.0], dtype=np.float64
        ),
        gas_log_amounts_init=np.log(
            np.asarray([0.5, 1.0e-300, 1.0e-300])
        ),
        condensate_amounts_init=np.asarray([0.5, 0.0, 0.0]),
        total_gas_log_amount_init=float(np.log(0.5)),
        element_potential_init=np.zeros(3, dtype=np.float64),
        support_indices=(0,),
        condensate_valid_mask=np.ones(3, dtype=bool),
        budget_relative_floor=1.0e-6,
    )

    rescue = result.report["structural_zero_reduced_log_rescue"]
    closure = result.report["exact_active_set_closure"]
    assert result.accepted
    assert result.support_indices == (0,)
    assert rescue["zero_target_rows"] == (1, 2)
    assert rescue["suppressed_gas_indices"] == (1, 2)
    assert tuple(
        index
        for index, _limit in rescue[
            "inactive_zero_row_phase_potential_limits"
        ]
    ) == (1, 2)
    assert result.report["inactive_condensate_violation_max_abs"] == 0.0
    assert result.report["full_condensate_driving"][1] >= 0.0
    assert result.report["full_condensate_driving"][2] >= 0.0
    assert not result.report["linear_amount_physical_audit"]["attempted"]
    assert closure["round_count"] == 1
    assert closure["added_support_indices"] == ()


def test_reduced_log_fallback_skips_zero_monotone_budget_row() -> None:
    gas_formula = np.eye(2, dtype=np.float64)
    target = np.asarray([1.0, 0.0], dtype=np.float64)
    result = polish_zero_barrier_active_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=np.asarray(
            [[1.0], [0.0]], dtype=np.float64
        ),
        target_inventory=target,
        gas_standard_source=np.zeros(2, dtype=np.float64),
        condensate_standard_source_full=np.zeros(1, dtype=np.float64),
        gas_log_amounts_init=np.log(
            np.asarray([0.5, 1.0e-300], dtype=np.float64)
        ),
        condensate_amounts_init=np.asarray([0.5], dtype=np.float64),
        total_gas_log_amount_init=float(np.log(0.5 + 1.0e-300)),
        element_potential_init=np.zeros(2, dtype=np.float64),
        support_indices=(0,),
        budget_relative_floor=1.0e-6,
        max_function_evaluations=10,
    )

    fallback = result.report["reduced_log_domain_fallback"]
    assert not fallback["eligible"]
    assert not fallback["attempted"]
    assert not fallback["accepted"]
    assert result.accepted
    assert result.report["selected_numerical_formulation"] == (
        "structural_zero_normalized_gas_reduced_linear_amounts"
    )
    rescue = result.report["structural_zero_reduced_log_rescue"]
    assert rescue["attempted"]
    assert rescue["accepted"]
    assert rescue["zero_target_rows"] == (1,)
    assert fallback["skip_reason"] == (
        "reduced_primary_physical_audit_accepted"
    )


def _finite_barrier_homotopy_problem(amount_scale: float) -> dict:
    gas_amount = amount_scale
    condensate_amounts = amount_scale * np.asarray([0.1, 0.001])
    target = np.asarray(
        [gas_amount + np.sum(condensate_amounts)], dtype=np.float64
    )
    return {
        "gas_formula_matrix": np.ones((1, 1), dtype=np.float64),
        "condensate_formula_matrix_full": np.ones(
            (1, 2), dtype=np.float64
        ),
        "target_inventory": target,
        "gas_standard_source": np.zeros(1, dtype=np.float64),
        "condensate_standard_source_full": np.asarray(
            [0.1, 10.0], dtype=np.float64
        ),
        "gas_log_amounts_init": np.asarray(
            [np.log(gas_amount)], dtype=np.float64
        ),
        "condensate_amounts_init": condensate_amounts,
        "total_gas_log_amount_init": float(np.log(gas_amount)),
        "element_potential_init": np.zeros(1, dtype=np.float64),
        "support_indices": (0, 1),
        "budget_scale": np.reciprocal(target),
        "max_function_evaluations": 400,
        "enabled": True,
    }


@pytest.mark.parametrize("amount_scale", (1.0e-12, 1.0, 1.0e8))
def test_finite_barrier_homotopy_is_guarded_and_amount_gauge_covariant(
    amount_scale: float,
) -> None:
    evaluation_budget = zero_barrier._FunctionEvaluationBudget(limit=400)
    result = zero_barrier._select_support_with_finite_barrier_homotopy(
        **_finite_barrier_homotopy_problem(amount_scale),
        function_evaluation_budget=evaluation_budget,
    )

    report = result["report"]
    assert result["applied"]
    assert result["support_indices"] == (0,)
    assert report["maximum_step_count"] == 12
    assert report["attempted_step_count"] == 12
    assert report["certified_step_count"] == 12
    assert report["selected_step_index"] == 11
    assert report["continuation_termination_reason"] == (
        "maximum_step_count_reached"
    )
    assert report["minimum_capacity_relative_gap_ratio"] == 4.0
    assert report["final_largest_capacity_relative_gap_ratio"] == (
        pytest.approx(100.0)
    )
    assert len(report["rounds"]) == 12
    assert all(
        item["optimizer_success"]
        and item["continuation_residual_max_abs"] <= 1.0e-8
        for item in report["rounds"]
    )
    assert evaluation_budget.used == sum(
        item["function_evaluations"] for item in report["rounds"]
    )
    assert result["condensate_amounts"][0] / amount_scale == (
        pytest.approx(1.0e-7, rel=1.0e-8)
    )
    assert result["condensate_amounts"][1] == 0.0
    assert result["total_gas_log_amount"] - np.log(amount_scale) == (
        pytest.approx(np.log(1.1009999), abs=1.0e-8)
    )


def test_finite_barrier_homotopy_keeps_deepest_certified_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_solver = zero_barrier._least_squares_with_scipy_overflow_guard
    call_count = 0

    def solver_with_failed_last_step(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        result = original_solver(*args, **kwargs)
        if call_count == 7:
            result.success = False
            result.status = 0
            result.message = "synthetic continuation loss"
        return result

    monkeypatch.setattr(
        zero_barrier,
        "_FINITE_BARRIER_HOMOTOPY_MAXIMUM_STEP_COUNT",
        7,
    )
    monkeypatch.setattr(
        zero_barrier,
        "_least_squares_with_scipy_overflow_guard",
        solver_with_failed_last_step,
    )
    result = zero_barrier._select_support_with_finite_barrier_homotopy(
        **_finite_barrier_homotopy_problem(1.0)
    )

    report = result["report"]
    assert result["applied"]
    assert result["support_indices"] == (0,)
    assert report["attempted_step_count"] == 7
    assert report["certified_step_count"] == 6
    assert report["selected_step_index"] == 5
    assert report["continuation_termination_reason"] == (
        "continuation_certificate_lost"
    )
    assert not report["rounds"][-1]["optimizer_success"]


def test_finite_barrier_homotopy_falls_back_when_input_is_not_central(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _finite_barrier_homotopy_problem(1.0)
    arguments["condensate_amounts_init"] = np.asarray([0.1, 0.002])

    def unexpected_solver(*args, **kwargs):
        del args, kwargs
        raise AssertionError("a noncentral input must fail before continuation")

    monkeypatch.setattr(
        zero_barrier,
        "_least_squares_with_scipy_overflow_guard",
        unexpected_solver,
    )
    result = zero_barrier._select_support_with_finite_barrier_homotopy(
        **arguments
    )

    assert not result["applied"]
    assert result["support_indices"] == (0, 1)
    assert result["report"]["failure_reason"] == "centrality_guard_failed"
    assert result["report"]["skip_reason"] == (
        "finite_barrier_state_not_central"
    )


@pytest.mark.parametrize("amount_scale", (1.0e-12, 1.0, 1.0e8))
def test_rank_one_support_pivot_is_amount_gauge_covariant(
    amount_scale: float,
) -> None:
    condensate_formula = np.asarray(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64
    )
    amounts = amount_scale * np.asarray([0.2, 0.5, 0.0])
    result = zero_barrier._pivot_rank_one_support_addition(
        condensate_formula_matrix_full=condensate_formula,
        condensate_standard_source_full=np.asarray([0.0, 0.0, -1.0]),
        target_inventory=amount_scale * np.ones(2, dtype=np.float64),
        condensate_amounts=amounts,
        support_indices=(0, 1),
        added_support_index=2,
        budget_scale=np.full(2, 1.0 / amount_scale),
    )

    assert result["applied"]
    assert result["support_indices"] == (1, 2)
    assert result["report"]["leaving_support_index"] == 0
    np.testing.assert_allclose(
        result["condensate_amounts"] / amount_scale,
        np.asarray([0.0, 0.3, 0.2]),
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        condensate_formula @ result["condensate_amounts"],
        condensate_formula @ amounts,
        rtol=1.0e-12,
        atol=1.0e-14 * amount_scale,
    )
    assert (
        result["report"]["scaled_inventory_residual_max_abs"]
        <= 1.0e-10
    )


def test_rank_one_support_pivot_rejects_a_tied_limiter() -> None:
    condensate_formula = np.asarray(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64
    )
    amounts = np.asarray([0.2, 0.2, 0.0])
    result = zero_barrier._pivot_rank_one_support_addition(
        condensate_formula_matrix_full=condensate_formula,
        condensate_standard_source_full=np.asarray([0.0, 0.0, -1.0]),
        target_inventory=np.ones(2, dtype=np.float64),
        condensate_amounts=amounts,
        support_indices=(0, 1),
        added_support_index=2,
        budget_scale=np.ones(2, dtype=np.float64),
    )

    assert not result["applied"]
    assert result["report"]["failure_reason"] == (
        "nonunique_limiting_phase"
    )
    np.testing.assert_array_equal(result["condensate_amounts"], amounts)


def test_homotopy_selected_support_root_failure_retries_original_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized_supports = []

    def fake_homotopy(*, enabled, support_indices, **kwargs):
        del kwargs
        support = tuple(support_indices)
        applied = bool(enabled)
        selected = (support[0],) if applied else support
        return {
            "applied": applied,
            "support_indices": selected,
            "gas_log_amounts": np.log(np.asarray([0.5, 0.5])),
            "condensate_amounts": np.asarray([0.5, 0.0]),
            "total_gas_log_amount": 0.0,
            "element_potential": np.zeros(2),
            "report": {
                "schema": "unit_test_homotopy",
                "enabled": enabled,
                "applied": applied,
                "rounds": (),
                "selected_support_indices": selected,
            },
        }

    def failed_normalized(*, support_indices, **kwargs):
        del kwargs
        normalized_supports.append(tuple(support_indices))
        return {
            "accepted": False,
            "candidate": None,
            "report": {
                "schema": "unit_test_normalized_failure",
                "attempted": True,
                "accepted": False,
                "attempts": (),
            },
        }

    def raising_dense_solver(*args, **kwargs):
        del args, kwargs
        raise ValueError("unit-test dense failure")

    monkeypatch.setattr(
        zero_barrier,
        "_select_support_with_finite_barrier_homotopy",
        fake_homotopy,
    )
    monkeypatch.setattr(
        zero_barrier,
        "_solve_normalized_gas_reduced_linear_support",
        failed_normalized,
    )
    monkeypatch.setattr(
        zero_barrier,
        "_least_squares_with_scipy_overflow_guard",
        raising_dense_solver,
    )
    result = zero_barrier._polish_zero_barrier_support_once(
        gas_formula_matrix=np.eye(2, dtype=np.float64),
        condensate_formula_matrix_full=np.eye(2, dtype=np.float64),
        target_inventory=np.ones(2, dtype=np.float64),
        gas_standard_source=np.zeros(2, dtype=np.float64),
        condensate_standard_source_full=np.zeros(2, dtype=np.float64),
        gas_log_amounts_init=np.log(np.asarray([0.5, 0.5])),
        condensate_amounts_init=np.asarray([0.5, 0.5]),
        total_gas_log_amount_init=0.0,
        element_potential_init=np.zeros(2, dtype=np.float64),
        support_indices=(0, 1),
        function_evaluation_budget=zero_barrier._FunctionEvaluationBudget(1),
        max_function_evaluations=1,
    )

    fallback = result.report["support_initializer_postselection_fallback"]
    assert not result.accepted
    assert normalized_supports[:2] == [(0,), (0, 1)]
    assert fallback["attempted"]
    assert fallback["reason"] == "selected_support_local_root_failed"
    assert fallback["selected_support_indices"] == (0,)


def test_initializer_fallback_chain_preserves_diagnostics_and_evaluations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_dual(*, enabled, function_evaluation_budget, **kwargs):
        del kwargs
        applied = bool(enabled)
        evaluations = 2 if applied else 0
        if applied:
            function_evaluation_budget.consume(evaluations)
        return {
            "applied": applied,
            "support_indices": (1,) if applied else (),
            "gas_log_amounts": np.asarray([np.log(0.5)]),
            "condensate_amounts": np.asarray([0.0, 0.5, 0.0]),
            "total_gas_log_amount": float(np.log(0.5)),
            "element_potential": np.zeros(1),
            "report": {
                "schema": "unit_test_dual",
                "enabled": bool(enabled),
                "attempted": bool(enabled),
                "applied": applied,
                "function_evaluations": evaluations,
            },
        }

    def fake_homotopy(
        *, enabled, function_evaluation_budget, support_indices, **kwargs
    ):
        del kwargs
        applied = bool(enabled)
        evaluations = 3 if applied else 0
        if applied:
            function_evaluation_budget.consume(evaluations)
        return {
            "applied": applied,
            "support_indices": (2,) if applied else tuple(support_indices),
            "gas_log_amounts": np.asarray([np.log(0.5)]),
            "condensate_amounts": np.asarray([0.0, 0.0, 0.5]),
            "total_gas_log_amount": float(np.log(0.5)),
            "element_potential": np.zeros(1),
            "report": {
                "schema": "unit_test_homotopy",
                "enabled": bool(enabled),
                "applied": applied,
                "rounds": (
                    ({"function_evaluations": evaluations},)
                    if applied
                    else ()
                ),
            },
        }

    def identity_basic_reduction(
        *, condensate_amounts, support_indices, enabled, **kwargs
    ):
        del kwargs
        return (
            tuple(support_indices),
            np.asarray(condensate_amounts, dtype=np.float64).copy(),
            {
                "schema": "unit_test_basic_support",
                "enabled": bool(enabled),
                "attempted": False,
                "applied": False,
            },
        )

    def fake_normalized(
        *, support_indices, function_evaluation_budget, **kwargs
    ):
        del kwargs
        support = tuple(support_indices)
        function_evaluation_budget.consume(1)
        report = {
            "schema": "unit_test_normalized",
            "attempted": True,
            "accepted": False,
            "attempts": ({"function_evaluations": 1},),
        }
        if support != (0, 1, 2):
            return {"accepted": False, "candidate": None, "report": report}

        gas_log_amounts = np.asarray([np.log(0.5)])
        condensate_amounts = np.asarray([0.5, 0.0, 0.0])
        total_gas_log_amount = float(np.log(0.5))
        element_potential = np.zeros(1)
        audit = _physical_zero_barrier_audit(
            gas_formula_matrix=np.ones((1, 1)),
            condensate_formula_matrix_full=np.ones((1, 3)),
            target_inventory=np.ones(1),
            gas_standard_source=np.zeros(1),
            condensate_standard_source_full=np.asarray([0.0, 1.0, 2.0]),
            gas_log_amounts=gas_log_amounts,
            condensate_amounts=condensate_amounts,
            total_gas_log_amount=total_gas_log_amount,
            element_potential=element_potential,
            support_indices=(0,),
            condensate_valid_mask=np.ones(3, dtype=bool),
            budget_scale=np.ones(1),
            optimizer_success=True,
            stationarity_tolerance=1.0e-8,
            budget_tolerance=1.0e-8,
            total_density_tolerance=1.0e-8,
            support_closure_tolerance=1.0e-8,
        )
        candidate = {
            "gas_log_amounts": gas_log_amounts,
            "condensate_amounts": condensate_amounts,
            "total_gas_log_amount": total_gas_log_amount,
            "element_potential": element_potential,
            "support_indices": (0,),
            "optimizer_success": True,
            "optimizer_status": 1,
            "optimizer_message": "unit-test accepted",
            "function_evaluations": 1,
            "active_phase_at_lower_bound": False,
            "audit": audit,
        }
        report["accepted"] = True
        return {"accepted": True, "candidate": candidate, "report": report}

    monkeypatch.setattr(
        zero_barrier, "_select_support_with_zero_barrier_dual", fake_dual
    )
    monkeypatch.setattr(
        zero_barrier,
        "_select_support_with_finite_barrier_homotopy",
        fake_homotopy,
    )
    monkeypatch.setattr(
        zero_barrier,
        "_reduce_initial_condensate_support_to_basic",
        identity_basic_reduction,
    )
    monkeypatch.setattr(
        zero_barrier,
        "_solve_normalized_gas_reduced_linear_support",
        fake_normalized,
    )

    result = polish_zero_barrier_active_support(
        gas_formula_matrix=np.ones((1, 1)),
        condensate_formula_matrix_full=np.ones((1, 3)),
        target_inventory=np.ones(1),
        gas_standard_source=np.zeros(1),
        condensate_standard_source_full=np.asarray([0.0, 1.0, 2.0]),
        gas_log_amounts_init=np.asarray([np.log(0.5)]),
        condensate_amounts_init=np.asarray([0.5, 0.25, 0.25]),
        total_gas_log_amount_init=float(np.log(0.5)),
        element_potential_init=np.zeros(1),
        support_indices=(0, 1, 2),
        max_function_evaluations=20,
    )

    assert result.accepted
    fallback = result.report["support_initializer_postselection_fallback"]
    retry_diagnostics = fallback["retry_initializer_diagnostics"]
    assert fallback["selected_support_source"] == (
        "zero_barrier_dual_support"
    )
    assert retry_diagnostics["finite_barrier_homotopy_initializer"][
        "applied"
    ]
    nested_fallback = retry_diagnostics[
        "support_initializer_postselection_fallback"
    ]
    assert nested_fallback["selected_support_source"] == (
        "finite_barrier_homotopy"
    )
    closure = result.report["exact_active_set_closure"]
    assert closure["round_count"] == 1
    assert closure["cumulative_linear_function_evaluations"] == 0
    assert closure["cumulative_reduced_function_evaluations"] == 8
    assert closure["cumulative_function_evaluations"] == 8
    assert closure["rounds"][0]["function_evaluations"] == 8


class _FailedOptimization:
    def __init__(self, values: np.ndarray):
        self.x = np.asarray(values, dtype=np.float64)
        self.success = False
        self.status = 0
        self.message = "unit-test nonconvergence"
        self.nfev = 1
        self.cost = 1.0
        self.optimality = 1.0


def test_exact_physical_root_survives_optimizer_evaluation_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        gas_formula,
        condensate_formula,
        expected_gas,
        expected_condensates,
        expected_potential,
        gamma,
        hcond,
        target,
    ) = _one_active_phase_problem()
    original_solver = zero_barrier._least_squares_with_scipy_overflow_guard

    def solver_at_limit(*args, **kwargs):
        optimization = original_solver(*args, **kwargs)
        optimization.success = False
        optimization.status = 0
        optimization.message = (
            "The maximum number of function evaluations is exceeded."
        )
        return optimization

    monkeypatch.setattr(
        zero_barrier,
        "_least_squares_with_scipy_overflow_guard",
        solver_at_limit,
    )
    result = zero_barrier._polish_zero_barrier_support_once(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond,
        gas_log_amounts_init=np.log(expected_gas),
        condensate_amounts_init=expected_condensates,
        total_gas_log_amount_init=0.0,
        element_potential_init=expected_potential,
        support_indices=(0,),
        condensate_valid_mask=np.ones(2, dtype=bool),
        max_function_evaluations=400,
        reduce_initial_support=False,
        use_zero_barrier_dual=False,
        use_finite_barrier_homotopy=False,
    )

    assert result.accepted
    assert result.support_indices == (0,)
    assert not result.report["optimizer_success"]
    assert result.report["optimizer_status"] == 0
    assert result.report["optimizer_termination_eligible"]
    assert result.report["physical_root_certified"]
    assert result.report["acceptance_source"] == (
        "physical_kkt_after_optimizer_limit"
    )
    attempt = result.report["normalized_gas_reduced_primary"][
        "attempts"
    ][0]
    assert not attempt["drop_authorized_by_root"]
    assert attempt["acceptance_source"] == (
        "physical_kkt_after_optimizer_limit"
    )


def test_normalized_solver_does_not_drop_from_a_nonconverged_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        gas_formula,
        condensate_formula,
        expected_gas,
        expected_condensates,
        expected_potential,
        gamma,
        hcond,
        target,
    ) = _one_active_phase_problem()

    def failed_solver(residual, values, **kwargs):
        del residual, kwargs
        failed_values = np.asarray(values, dtype=np.float64).copy()
        failed_values[-1] = -0.25
        return _FailedOptimization(failed_values)

    monkeypatch.setattr(
        zero_barrier,
        "_least_squares_with_scipy_overflow_guard",
        failed_solver,
    )
    result = _solve_normalized_gas_reduced_linear_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond,
        gas_log_amounts_init=np.log(expected_gas),
        condensate_amounts_init=expected_condensates,
        total_gas_log_amount_init=0.0,
        element_potential_init=expected_potential,
        support_indices=(0,),
        condensate_valid_mask=np.ones(2, dtype=bool),
        budget_scale=np.reciprocal(target),
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=400,
    )

    assert result["candidate"]["support_indices"] == (0,)
    assert result["candidate"]["condensate_amounts"][0] < 0.0
    assert result["report"]["dropped_support_indices"] == ()
    assert len(result["report"]["attempts"]) == 1
    assert not result["report"]["attempts"][0][
        "drop_authorized_by_root"
    ]


def test_reduced_log_solver_does_not_drop_from_a_nonconverged_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_solver(residual, values, **kwargs):
        del residual
        failed_values = np.asarray(values, dtype=np.float64).copy()
        failed_values[-1] = kwargs["bounds"][0][-1]
        return _FailedOptimization(failed_values)

    monkeypatch.setattr(
        zero_barrier,
        "_least_squares_with_scipy_overflow_guard",
        failed_solver,
    )
    result = _solve_reduced_log_domain_active_support(
        gas_formula_matrix=np.eye(2, dtype=np.float64),
        condensate_formula_matrix_full=np.asarray(
            [[1.0], [0.0]], dtype=np.float64
        ),
        target_inventory=np.ones(2, dtype=np.float64),
        gas_standard_source=np.zeros(2, dtype=np.float64),
        condensate_standard_source_full=np.zeros(1, dtype=np.float64),
        gas_log_amounts_init=np.log(np.asarray([0.5, 0.5])),
        condensate_amounts_init=np.asarray([0.5]),
        total_gas_log_amount_init=0.0,
        element_potential_init=np.zeros(2, dtype=np.float64),
        support_indices=(0,),
        condensate_valid_mask=np.ones(1, dtype=bool),
        budget_scale=np.ones(2, dtype=np.float64),
        stationarity_tolerance=1.0e-8,
        budget_tolerance=1.0e-8,
        total_density_tolerance=1.0e-8,
        support_closure_tolerance=1.0e-8,
        max_function_evaluations=400,
        allow_greedy_drop=True,
    )

    assert result["candidate"]["support_indices"] == (0,)
    assert result["candidate"]["active_phase_at_lower_bound"]
    assert result["report"]["dropped_support_indices"] == ()
    assert len(result["report"]["attempts"]) == 1
    assert not result["report"]["attempts"][0][
        "drop_authorized_by_root"
    ]


def test_dense_solver_does_not_drop_from_a_nonconverged_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_normalized(**kwargs):
        del kwargs
        return {
            "accepted": False,
            "candidate": None,
            "report": {
                "schema": "unit_test_normalized_failure",
                "attempted": True,
                "accepted": False,
                "attempts": (),
            },
        }

    def failed_dense_solver(residual, values, **kwargs):
        del residual, kwargs
        failed_values = np.asarray(values, dtype=np.float64).copy()
        failed_values[2] = -0.25
        return _FailedOptimization(failed_values)

    monkeypatch.setattr(
        zero_barrier,
        "_solve_normalized_gas_reduced_linear_support",
        failed_normalized,
    )
    monkeypatch.setattr(
        zero_barrier,
        "_least_squares_with_scipy_overflow_guard",
        failed_dense_solver,
    )
    result = zero_barrier._polish_zero_barrier_support_once(
        gas_formula_matrix=np.eye(2, dtype=np.float64),
        condensate_formula_matrix_full=np.asarray(
            [[1.0], [0.0]], dtype=np.float64
        ),
        target_inventory=np.ones(2, dtype=np.float64),
        gas_standard_source=np.zeros(2, dtype=np.float64),
        condensate_standard_source_full=np.zeros(1, dtype=np.float64),
        gas_log_amounts_init=np.log(np.asarray([0.5, 0.5])),
        condensate_amounts_init=np.asarray([0.5]),
        total_gas_log_amount_init=0.0,
        element_potential_init=np.zeros(2, dtype=np.float64),
        support_indices=(0,),
        max_function_evaluations=1,
        function_evaluation_budget=zero_barrier._FunctionEvaluationBudget(1),
        use_zero_barrier_dual=False,
    )

    assert result.support_indices == (0,)
    assert result.condensate_amounts[0] < 0.0
    assert result.report["dropped_support_indices"] == ()
    assert len(result.report["attempts"]) == 1
    assert not result.report["attempts"][0]["drop_authorized_by_root"]
