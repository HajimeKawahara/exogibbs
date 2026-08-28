"""Regression for a positive trace element near condensate capacity."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from exogibbs.equilibrium.condensate.fixed_support.zero_barrier import (
    polish_zero_barrier_active_support,
)


FIXTURE = (
    Path(__file__).with_name("data")
    / "rocky_raccoon_trace_mg_polish.npz"
)


def test_zero_barrier_polishes_positive_trace_mg_with_signed_charge() -> None:
    """Resolve a host-side state captured after the finite-barrier solve.

    The fixture was captured at ExoGibbs revision 01b21d4 from
    ``tests/rocky_raccoon/test_real_column.py`` in ExoExamples at
    T=1433.764595 K and P=8796.093022 bar.  It contains the
    provider-independent binary64 inputs passed to zero-barrier polishing for
    H, Mg, Si, O, C, and charge.  The initial support is SiO2, MgSiO3, MgO,
    and Mg2SiO4 in the captured catalog.
    """

    with np.load(FIXTURE) as stored:
        gas_formula = stored["gas_formula_matrix"]
        condensate_formula = stored[
            "condensate_formula_matrix_full"
        ]
        target = stored["target_inventory"]
        gas_standard = stored["gas_standard_source"]
        condensate_standard = stored[
            "condensate_standard_source_full"
        ]
        gas_log_amounts = stored["gas_log_amounts_init"]
        condensate_amounts = stored["condensate_amounts_init"]
        total_gas_log_amount = float(
            stored["total_gas_log_amount_init"]
        )
        element_potential = stored["element_potential_init"]
        support = tuple(
            int(index) for index in stored["support_indices"]
        )

    assert gas_formula.shape == (6, 70)
    assert condensate_formula.shape == (6, 14)
    assert np.linalg.matrix_rank(gas_formula) == 6
    assert 0.0 < target[1] < 1.0e-11
    assert target[-1] == 0.0
    assert np.any(gas_formula[-1] > 0.0)
    assert np.any(gas_formula[-1] < 0.0)
    assert support == (1, 8, 7, 9)

    tolerance = 1.0e-8
    result = polish_zero_barrier_active_support(
        gas_formula_matrix=gas_formula,
        condensate_formula_matrix_full=condensate_formula,
        target_inventory=target,
        gas_standard_source=gas_standard,
        condensate_standard_source_full=condensate_standard,
        gas_log_amounts_init=gas_log_amounts,
        condensate_amounts_init=condensate_amounts,
        total_gas_log_amount_init=total_gas_log_amount,
        element_potential_init=element_potential,
        support_indices=support,
        condensate_valid_mask=np.ones(
            condensate_formula.shape[1], dtype=bool
        ),
        stationarity_tolerance=tolerance,
        budget_tolerance=tolerance,
        total_density_tolerance=tolerance,
        support_closure_tolerance=tolerance,
        budget_relative_floor=1.0e-6,
        max_function_evaluations=400,
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
    driving = (
        condensate_standard
        - condensate_formula.T @ result.element_potential
    )
    support_mask = np.zeros(condensate_formula.shape[1], dtype=bool)
    support_mask[np.asarray(result.support_indices, dtype=np.int64)] = True
    inventory_scale = float(np.max(target[target != 0.0]))
    zero_target_scale = max(
        1.0e-6,
        np.finfo(np.float64).eps * inventory_scale,
        1.0e-300,
    )
    budget_scale = np.reciprocal(
        np.where(target != 0.0, np.abs(target), zero_target_scale)
    )
    scaled_budget_residual = budget_scale * (reconstructed - target)
    total_density_residual = (
        np.sum(np.exp(result.gas_log_amounts - result.total_gas_log_amount))
        - 1.0
    )

    assert result.accepted
    assert result.report["optimizer_success"]
    assert result.report["exact_active_set_closure"][
        "termination_reason"
    ] == "accepted"
    closure = result.report["exact_active_set_closure"]
    assert tuple(
        round_report["selected_normalized_initializer"]
        for round_report in closure["rounds"]
    ) == ("unregularized",)
    assert sum(
        round_report[
            "regularized_normalized_initializer_attempt_count"
        ]
        for round_report in closure["rounds"]
    ) == 0
    assert sum(
        round_report[
            "unregularized_normalized_initializer_attempt_count"
        ]
        for round_report in closure["rounds"]
    ) == 1
    assert closure["cumulative_function_evaluations"] < 100
    assert np.all(result.condensate_amounts[support_mask] > 0.0)
    assert np.max(np.abs(gas_stationarity)) <= tolerance
    assert np.max(np.abs(driving[support_mask])) <= tolerance
    assert np.max(
        np.maximum(-driving[~support_mask], 0.0), initial=0.0
    ) <= tolerance
    assert np.max(np.abs(scaled_budget_residual)) <= tolerance
    assert abs(total_density_residual) <= tolerance
    assert abs(reconstructed[-1]) <= 1.0e-14

    condensed_mg = float(
        condensate_formula[1] @ result.condensate_amounts
    )
    gas_mg = float(gas_formula[1] @ gas)
    assert 0.0 < condensed_mg <= target[1]
    assert 0.0 < gas_mg < target[1]
