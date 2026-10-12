"""A near-boundary initializer still has to pass every physical gate."""

import json
from pathlib import Path

import numpy as np
import pytest

from exogibbs.equilibrium.condensate.fixed_support import zero_barrier


def test_high_h_hold_initializer_reaches_an_independently_audited_parcel():
    path = Path(__file__).with_name("data") / "m2_high_h_hold_warm.json"
    result = zero_barrier.polish_zero_barrier_active_support(**json.loads(path.read_text()))
    dual = result.report["zero_barrier_dual_support_oracle"]
    assert dual["applied"]
    assert dual["feasibility_tolerance"] == 1e-10
    assert abs(dual["gas_normalization_log_residual"]) <= 1e-10
    assert dual["minimum_condensate_driving"] >= -1e-10
    assert result.accepted and result.audit.accepted
    assert not result.audit.local_kkt_failure_reasons
    assert np.all(result.condensate_amounts >= 0)


@pytest.mark.parametrize("remaining, violation, applied", [
    (1, 1.4e-10, True),
    (0, 1.4e-10, False),
    (1, 5e-10, False),
    (1, 2e-8, False),
])
def test_repair_respects_both_feasibility_gates_and_shared_budget(
    monkeypatch, remaining, violation, applied,
):
    def stopped(objective, values, **kwargs):
        objective(values)
        return zero_barrier.OptimizeResult(
            x=np.array([violation, -violation]), success=False, status=8,
            message="Stopped near the gas boundary", nit=1,
        )
    monkeypatch.setattr(zero_barrier, "minimize", stopped)
    budget = zero_barrier._FunctionEvaluationBudget(1 + remaining)
    result = zero_barrier._select_support_with_zero_barrier_dual(
        gas_formula_matrix=np.eye(2), condensate_formula_matrix_full=np.eye(2),
        target_inventory=np.ones(2), gas_standard_source=np.full(2, np.log(2.)),
        condensate_standard_source_full=np.array([0., 1.]),
        gas_log_amounts_init=np.log(np.full(2, .5)), condensate_amounts_init=np.array([.5, 0.]),
        total_gas_log_amount_init=0., element_potential_init=np.zeros(2),
        condensate_valid_mask=np.ones(2, dtype=bool), stationarity_tolerance=1e-8,
        support_closure_tolerance=1e-8, max_function_evaluations=20, enabled=True,
        function_evaluation_budget=budget,
    )
    report = result["report"]
    assert result["applied"] == applied
    assert not report["optimizer_success"]
    assert report["function_evaluations"] == budget.used <= budget.limit
    assert report["feasibility_tolerance"] == 1e-10
    if applied:
        assert report["condensate_feasibility_evaluations"] == 1
        assert abs(report["gas_normalization_log_residual"]) <= 1e-10
        assert report["minimum_condensate_driving"] >= -1e-10
    else:
        assert report["failure_reason"] == "dual_feasibility_failed"
