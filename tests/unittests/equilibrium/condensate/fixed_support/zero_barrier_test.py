"""Tests for the physical zero-barrier active-support refinement."""

from __future__ import annotations

import warnings

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

    assert reduced_support == (2, 1)
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


@pytest.mark.parametrize(
    ("gas_formula", "target", "skip_reason"),
    (
        (
            np.eye(2, dtype=np.float64),
            np.asarray([1.0, 0.0], dtype=np.float64),
            "nonpositive_target_row",
        ),
        (
            np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64),
            np.asarray([1.0, 1.0], dtype=np.float64),
            "signed_stoichiometry_row",
        ),
    ),
)
def test_reduced_log_fallback_skips_nonlogarithmic_budget_rows(
    gas_formula,
    target,
    skip_reason,
) -> None:
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
    if skip_reason == "nonpositive_target_row":
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
    else:
        assert fallback["skip_reason"] == skip_reason


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
