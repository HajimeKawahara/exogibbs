"""Support-initializer fallback preserves inputs, order, and shared work."""

import numpy as np
import pytest

from exogibbs.equilibrium.condensate.fixed_support import zero_barrier
from exogibbs.equilibrium.condensate.fixed_support.zero_barrier_contracts import (
    SupportInitializationStage as Stage,
)


def _terminal_result():
    return zero_barrier.ZeroBarrierPolishResult(
        accepted=False,
        gas_log_amounts=np.zeros(1),
        condensate_amounts=np.ones(1),
        total_gas_log_amount=0.0,
        element_potential=np.zeros(1),
        support_indices=(0,),
        report={
            "zero_barrier_dual_support_oracle": {"enabled": False},
            "finite_barrier_homotopy_initializer": {"enabled": False},
            "selected_numerical_formulation": "normalized_gas_reduced_linear_amounts",
        },
    )


@pytest.mark.parametrize(
    "dual,homotopy,first_stage",
    [
        (True, True, Stage.DUAL_THEN_HOMOTOPY),
        (True, False, Stage.DUAL_ONLY),
        (False, True, Stage.HOMOTOPY),
        (False, False, Stage.ORIGINAL),
    ],
)
def test_legacy_flags_only_select_the_first_stage(
    monkeypatch, dual, homotopy, first_stage,
):
    calls = []
    terminal = _terminal_result()

    def run_stage(*, initialization_stage, **kwargs):
        calls.append(initialization_stage)
        return terminal

    monkeypatch.setattr(zero_barrier, "_run_zero_barrier_support_stage", run_stage)
    result = zero_barrier._polish_zero_barrier_support_once(
        use_zero_barrier_dual=dual, use_finite_barrier_homotopy=homotopy,
    )
    assert result is terminal
    assert calls == [first_stage]


@pytest.mark.parametrize("initial_homotopy", [False, True])
def test_failed_proposals_advance_without_changing_original_state_or_budget(
    monkeypatch, initial_homotopy,
):
    initial_amounts = np.asarray([0.2, 0.8])
    budget = zero_barrier._FunctionEvaluationBudget(10)
    calls = []
    terminal = _terminal_result()

    def run_stage(*, initialization_stage, **kwargs):
        assert kwargs["condensate_amounts_init"] is initial_amounts
        assert kwargs["support_indices"] == (1, 0)
        assert kwargs["function_evaluation_budget"] is budget
        calls.append((initialization_stage, budget.remaining))
        budget.consume((2, 3, 5)[len(calls) - 1])
        if initialization_stage == Stage.ORIGINAL:
            return terminal
        return zero_barrier._SupportInitializationRetry(
            next_stage=(Stage.ORIGINAL if initialization_stage == Stage.HOMOTOPY
                        else Stage.HOMOTOPY),
            dual_report={"stage": initialization_stage.value},
            homotopy_report={"stage": initialization_stage.value},
            fallback_report={
                "selected_support_indices": (1,),
                "remaining_function_evaluations_before_retry": budget.remaining,
            },
        )

    monkeypatch.setattr(zero_barrier, "_run_zero_barrier_support_stage", run_stage)
    result = zero_barrier._polish_zero_barrier_support_once(
        condensate_amounts_init=initial_amounts,
        support_indices=(1, 0),
        function_evaluation_budget=budget,
        use_finite_barrier_homotopy=initial_homotopy,
    )
    first = Stage.DUAL_THEN_HOMOTOPY if initial_homotopy else Stage.DUAL_ONLY
    assert calls == [(first, 10), (Stage.HOMOTOPY, 8), (Stage.ORIGINAL, 5)]
    assert budget.used == 10
    assert result.support_indices == terminal.support_indices
    assert result.gas_log_amounts is terminal.gas_log_amounts
    fallback = result.report["support_initializer_postselection_fallback"]
    assert fallback["remaining_function_evaluations_before_retry"] == 8
    nested = fallback["retry_initializer_diagnostics"]
    assert nested["zero_barrier_dual_support_oracle"]["stage"] == "homotopy"
    nested_fallback = nested["support_initializer_postselection_fallback"]
    assert nested_fallback["remaining_function_evaluations_before_retry"] == 5
    assert not nested_fallback["retry_accepted"]
    assert nested_fallback["retry_support_indices"] == (0,)
    assert "support_initializer_postselection_fallback" not in terminal.report
