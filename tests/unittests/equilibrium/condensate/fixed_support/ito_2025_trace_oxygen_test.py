"""Regression tests for the trace-oxygen Ito rainout initializer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from exogibbs.equilibrium.condensate.fixed_support import zero_barrier
from exogibbs.equilibrium.condensate.fixed_support.zero_barrier import (
    polish_zero_barrier_active_support,
)


FIXTURE = (
    Path(__file__).with_name("data")
    / "ito_2025_layer_652_polish.npz"
)


def _fixture_arguments() -> dict[str, object]:
    with np.load(FIXTURE) as stored:
        return {
            name: stored[name].copy()
            for name in stored.files
        } | {
            "total_gas_log_amount_init": float(
                stored["total_gas_log_amount_init"]
            ),
            "support_indices": tuple(
                int(index) for index in stored["support_indices"]
            ),
        }


def _assert_physical_root(
    result,
    arguments: dict[str, object],
) -> None:
    gas_formula = np.asarray(arguments["gas_formula_matrix"])
    condensate_formula = np.asarray(
        arguments["condensate_formula_matrix_full"]
    )
    target = np.asarray(arguments["target_inventory"])
    gas_standard = np.asarray(arguments["gas_standard_source"])
    condensate_standard = np.asarray(
        arguments["condensate_standard_source_full"]
    )
    gas = np.exp(result.gas_log_amounts)
    reconstructed = (
        gas_formula @ gas
        + condensate_formula @ result.condensate_amounts
    )
    gas_stationarity = (
        result.gas_log_amounts
        + gas_standard
        - result.total_gas_log_amount
        - gas_formula.T @ result.element_potential
    )
    driving = condensate_standard - condensate_formula.T @ (
        result.element_potential
    )
    support_mask = np.zeros(condensate_formula.shape[1], dtype=bool)
    support_mask[np.asarray(result.support_indices, dtype=np.int64)] = True

    assert result.accepted
    assert np.max(np.abs(gas_stationarity)) <= 1.0e-8
    assert np.max(np.abs(driving[support_mask]), initial=0.0) <= 1.0e-8
    assert np.max(
        np.maximum(-driving[~support_mask], 0.0), initial=0.0
    ) <= 1.0e-8
    np.testing.assert_allclose(reconstructed, target, rtol=1.0e-8)
    assert abs(
        np.sum(np.exp(result.gas_log_amounts - result.total_gas_log_amount))
        - 1.0
    ) <= 1.0e-8


def test_regularized_primary_polishes_ito_layer_652_trace_oxygen() -> None:
    """Use a physical initializer for an actual propagated-rainout state.

    The provider-independent binary64 inputs were captured at ExoGibbs
    revision 01b21d4 from Ito Layer 652 after the public finite-barrier solve.
    The layer has T=725.1856 K, total P=72.150681 bar, and reactive
    P=61.40642645277065 bar.  Its positive oxygen inventory is approximately
    1.82e-23, while the finite-barrier oxygen potential is approximately
    -1.83e18.
    """

    arguments = _fixture_arguments()
    gas_formula = np.asarray(arguments["gas_formula_matrix"])
    condensate_formula = np.asarray(
        arguments["condensate_formula_matrix_full"]
    )
    target = np.asarray(arguments["target_inventory"])
    potential = np.asarray(arguments["element_potential_init"])

    assert gas_formula.shape == (3, 5)
    assert condensate_formula.shape == (3, 2)
    assert np.linalg.matrix_rank(gas_formula) == 3
    assert np.all(gas_formula >= 0.0)
    assert np.all(condensate_formula >= 0.0)
    assert np.all(target > 0.0)
    assert target[1] < 1.0e-20
    assert abs(potential[1]) > 1.0e18
    assert arguments["support_indices"] == (0, 1)
    assert np.all(arguments["condensate_valid_mask"])

    result = polish_zero_barrier_active_support(
        **arguments,
        max_function_evaluations=100,
    )

    _assert_physical_root(result, arguments)
    assert result.support_indices == (0,)
    regularization = result.report["initializer_regularization"]
    portfolio = result.report[
        "normalized_gas_reduced_initializer_portfolio"
    ]
    normalized = result.report["normalized_gas_reduced_primary"]
    closure = result.report["exact_active_set_closure"]
    assert regularization["applied"]
    assert regularization["element_potential_recomputed"]
    assert regularization["element_potential_fit_rank"] == 3
    assert regularization[
        "reduced_primary_minimum_inventory_fraction"
    ] < np.finfo(np.float64).eps
    assert regularization[
        "reduced_primary_relative_precision_threshold"
    ] == np.finfo(np.float64).eps
    assert regularization[
        "reduced_primary_minimum_inventory_at_or_below_binary64_epsilon"
    ]
    assert regularization["applied_to_reduced_primary"]
    assert portfolio["regularized_attempted"]
    selected_initializer = portfolio["selected_initializer"]
    assert selected_initializer in {"capacity_regularized", "unregularized"}
    assert normalized["initializer"] == selected_initializer
    assert portfolio["unregularized_attempted"] is (
        selected_initializer == "unregularized"
    )
    assert portfolio["raw_retry_attempted"] is (
        selected_initializer == "unregularized"
    )
    assert sum(
        attempt["function_evaluations"]
        for attempt in normalized["attempts"]
    ) <= 100
    cumulative_evaluations = closure["cumulative_function_evaluations"]
    assert cumulative_evaluations == sum(
        round_report["function_evaluations"]
        for round_report in closure["rounds"]
    )
    assert cumulative_evaluations <= closure[
        "function_evaluation_limit"
    ]
    assert not result.report["linear_amount_physical_audit"]["attempted"]


@pytest.mark.parametrize(
    ("inventory_fraction_multiplier", "regularized_expected"),
    ((1.0, True), (4.0, False)),
)
def test_regularized_primary_inventory_fraction_gate(
    monkeypatch: pytest.MonkeyPatch,
    inventory_fraction_multiplier: float,
    regularized_expected: bool,
) -> None:
    """Route the epsilon boundary and a resolved trace deterministically."""

    arguments = _fixture_arguments()
    target = np.asarray(arguments["target_inventory"]).copy()
    inventory_scale = float(np.max(target))
    target[1] = (
        inventory_fraction_multiplier
        * np.finfo(np.float64).eps
        * inventory_scale
    )
    arguments["target_inventory"] = target
    initial_potential_scales = []

    def reject_normalized_primary(**kwargs):
        initial_potential_scales.append(
            float(np.max(np.abs(kwargs["element_potential_init"])))
        )
        return {
            "accepted": False,
            "candidate": None,
            "report": {
                "schema": "unit_test_normalized_rejection",
                "attempted": True,
                "accepted": False,
                "attempts": (),
            },
        }

    monkeypatch.setattr(
        zero_barrier,
        "_solve_normalized_gas_reduced_linear_support",
        reject_normalized_primary,
    )
    result = zero_barrier._polish_zero_barrier_support_once(
        **arguments,
        max_function_evaluations=1,
        use_zero_barrier_dual=False,
        use_finite_barrier_homotopy=False,
    )

    regularization = result.report["initializer_regularization"]
    portfolio = result.report[
        "normalized_gas_reduced_initializer_portfolio"
    ]
    assert regularization["applied"]
    assert regularization["element_potential_recomputed"]
    assert regularization["element_potential_fit_rank"] == 3
    assert regularization[
        "reduced_primary_minimum_inventory_fraction"
    ] == inventory_fraction_multiplier * np.finfo(np.float64).eps
    assert regularization[
        "reduced_primary_minimum_inventory_at_or_below_binary64_epsilon"
    ] is regularized_expected
    assert regularization[
        "eligible_for_reduced_primary"
    ] is regularized_expected
    if regularized_expected:
        assert regularization["reduced_primary_skip_reason"] is None
        assert portfolio["regularized_attempted"]
        assert portfolio["raw_retry_attempted"]
        assert initial_potential_scales[0] < 1.0e3
        assert initial_potential_scales[1] > 1.0e18
    else:
        assert regularization["reduced_primary_skip_reason"] == (
            "minimum_inventory_fraction_above_binary64_epsilon"
        )
        assert not portfolio["regularized_attempted"]
        assert initial_potential_scales[0] > 1.0e18
    assert portfolio["unregularized_attempted"]


def test_regularized_primary_failure_retries_raw_initializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the uncapped raw route when the fast initializer is rejected."""

    arguments = _fixture_arguments()
    original_solver = (
        zero_barrier._solve_normalized_gas_reduced_linear_support
    )
    initial_potential_scales = []

    def fail_regularized_once(**kwargs):
        initial_potential_scales.append(
            float(np.max(np.abs(kwargs["element_potential_init"])))
        )
        if len(initial_potential_scales) == 1:
            budget = kwargs["function_evaluation_budget"]
            assert budget is not None
            budget.consume(1)
            return {
                "accepted": False,
                "candidate": None,
                "report": {
                    "schema": "unit_test_regularized_failure",
                    "attempted": True,
                    "accepted": False,
                    "attempts": (
                        {
                            "function_evaluations": 1,
                            "failure_reason": "unit_test_rejection",
                        },
                    ),
                },
            }
        return original_solver(**kwargs)

    monkeypatch.setattr(
        zero_barrier,
        "_solve_normalized_gas_reduced_linear_support",
        fail_regularized_once,
    )
    result = polish_zero_barrier_active_support(**arguments)

    _assert_physical_root(result, arguments)
    portfolio = result.report[
        "normalized_gas_reduced_initializer_portfolio"
    ]
    assert initial_potential_scales[0] < 1.0e3
    assert initial_potential_scales[1] > 1.0e18
    assert portfolio["regularized_attempted"]
    assert portfolio["raw_retry_attempted"]
    assert portfolio["selected_initializer"] == "unregularized"
    assert len(portfolio["discarded_solve_reports"]) == 1
    assert result.report["normalized_gas_reduced_primary"][
        "initializer"
    ] == "unregularized"


def test_regularized_primary_reserves_unregularized_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cap the fast attempt without consuming the protected raw allowance."""

    arguments = _fixture_arguments()
    budget = zero_barrier._FunctionEvaluationBudget(3)
    calls = []

    def exhaust_supplied_budget(**kwargs):
        supplied_budget = kwargs["function_evaluation_budget"]
        assert supplied_budget is not None
        evaluations = supplied_budget.remaining
        calls.append(
            (
                float(
                    np.max(np.abs(kwargs["element_potential_init"]))
                ),
                evaluations,
            )
        )
        supplied_budget.consume(evaluations)
        return {
            "accepted": False,
            "candidate": None,
            "report": {
                "schema": "unit_test_budget_exhaustion",
                "attempted": True,
                "accepted": False,
                "attempts": (
                    {"function_evaluations": evaluations},
                ),
            },
        }

    monkeypatch.setattr(
        zero_barrier,
        "_solve_normalized_gas_reduced_linear_support",
        exhaust_supplied_budget,
    )
    result = zero_barrier._polish_zero_barrier_support_once(
        **arguments,
        max_function_evaluations=2,
        function_evaluation_budget=budget,
        use_zero_barrier_dual=False,
        use_finite_barrier_homotopy=False,
    )

    portfolio = result.report[
        "normalized_gas_reduced_initializer_portfolio"
    ]
    assert calls[0][0] < 1.0e3
    assert calls[1][0] > 1.0e18
    assert [evaluations for _scale, evaluations in calls] == [1, 2]
    assert budget.used == 3
    assert portfolio["regularized_function_evaluation_limit"] == 1
    assert portfolio["raw_function_evaluation_reserve"] == 2
    assert portfolio["regularized_function_evaluations"] == 1
    assert portfolio["regularized_attempted"]
    assert portfolio["raw_retry_attempted"]
    assert zero_barrier._zero_barrier_report_function_evaluations(
        result.report
    ) == (0, 3)


def test_initializer_summary_traverses_nested_support_retries() -> None:
    """Include discarded portfolios from every initializer retry depth."""

    def portfolio(*attempts, raw_retry=False):
        return {
            "regularized_attempted": any(
                initializer == "capacity_regularized"
                for initializer, _evaluations in attempts
            ),
            "unregularized_attempted": any(
                initializer == "unregularized"
                for initializer, _evaluations in attempts
            ),
            "raw_retry_attempted": raw_retry,
            "attempts": tuple(
                {
                    "initializer": initializer,
                    "function_evaluations": evaluations,
                }
                for initializer, evaluations in attempts
            ),
        }

    report = {
        "normalized_gas_reduced_initializer_portfolio": portfolio(
            ("capacity_regularized", 1)
        ),
        "support_initializer_postselection_fallback": {
            "selected_support_normalized_initializer_portfolio": portfolio(
                ("capacity_regularized", 2),
                ("unregularized", 3),
                raw_retry=True,
            ),
            "retry_initializer_diagnostics": {
                "support_initializer_postselection_fallback": {
                    "selected_support_normalized_initializer_portfolio": (
                        portfolio(("unregularized", 4))
                    ),
                    "retry_initializer_diagnostics": {},
                }
            },
        },
    }

    assert zero_barrier._zero_barrier_initializer_portfolio_summary(
        report
    ) == {
        "regularized_attempt_count": 2,
        "regularized_function_evaluations": 3,
        "unregularized_attempt_count": 2,
        "unregularized_function_evaluations": 7,
        "raw_retry_count": 1,
    }
