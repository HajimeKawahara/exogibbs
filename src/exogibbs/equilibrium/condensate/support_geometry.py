"""Host-side geometry utilities for condensate support initializers."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import linprog


BASIC_SUPPORT_LP_ITERATION_LIMIT = 1000
BASIC_SUPPORT_RELATIVE_AMOUNT_FLOOR = 64.0 * np.finfo(np.float64).eps


def maximum_condensate_amount_scales(
    condensate_formula_matrix: np.ndarray,
    target_inventory: np.ndarray,
) -> np.ndarray:
    """Return conservative per-phase amount scales from elemental capacity."""

    target_scale = max(
        float(np.max(np.abs(target_inventory), initial=0.0)),
        1.0,
    )
    scales = []
    for column in condensate_formula_matrix.T:
        consuming = column > 0.0
        if not np.any(consuming):
            scales.append(target_scale)
            continue
        capacities = target_inventory[consuming] / column[consuming]
        nonnegative = capacities[capacities >= 0.0]
        if nonnegative.size == 0:
            scales.append(target_scale)
            continue
        scales.append(max(float(np.min(nonnegative)), 1.0e-300 * target_scale))
    return np.asarray(scales, dtype=np.float64)


def finite_barrier_trace_capacity_report(
    *,
    condensate_formula_matrix_full: np.ndarray,
    target_inventory: np.ndarray,
    support_indices: Sequence[int],
    monotone_constraint_row_mask: Sequence[bool],
    log_barrier: float,
) -> dict[str, Any]:
    """Report active-phase capacities below one finite barrier amount.

    The comparison is diagnostic only.  It identifies supports containing a
    phase whose conservative elemental capacity is no larger than the first
    finite-barrier amount. Only caller-identified monotone constraint rows
    bound a phase; signed rows such as charge balance must be excluded because
    negative carriers can offset a positive phase coefficient. Callers remain
    responsible for solver routing and physical acceptance.
    """

    formula = np.asarray(condensate_formula_matrix_full, dtype=np.float64)
    target = np.asarray(target_inventory, dtype=np.float64)
    support = tuple(int(index) for index in support_indices)
    if formula.ndim != 2:
        raise ValueError("condensate_formula_matrix_full must be two-dimensional.")
    element_count, condensate_count = formula.shape
    if target.shape != (element_count,):
        raise ValueError("target_inventory must have one value per element.")
    monotone_rows = np.asarray(monotone_constraint_row_mask, dtype=bool)
    if monotone_rows.shape != (element_count,):
        raise ValueError(
            "monotone_constraint_row_mask must have one value per element."
        )
    if len(set(support)) != len(support) or any(
        index < 0 or index >= condensate_count for index in support
    ):
        raise ValueError("support_indices must contain unique catalog indices.")
    if not np.all(np.isfinite(formula)):
        raise ValueError("condensate_formula_matrix_full must be finite.")
    if not np.all(np.isfinite(target)):
        raise ValueError("target_inventory must be finite.")
    log_barrier_value = float(log_barrier)
    if not np.isfinite(log_barrier_value):
        raise ValueError("log_barrier must be finite.")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        barrier_amount = float(np.exp(log_barrier_value))
    if not np.isfinite(barrier_amount) or barrier_amount <= 0.0:
        raise ValueError("log_barrier must produce a positive finite amount.")

    canonical_support = tuple(sorted(support))
    active = np.asarray(canonical_support, dtype=np.int64)
    capacity_formula = formula[monotone_rows]
    capacity_target = target[monotone_rows]
    active_formula = capacity_formula[:, active]
    negative_target_consumed = (capacity_target[:, None] < 0.0) & (
        active_formula > 0.0
    )
    capacity_geometry_valid = bool(
        np.all(capacity_formula >= 0.0)
        and not np.any(negative_target_consumed)
    )
    capacities = maximum_condensate_amount_scales(
        active_formula,
        capacity_target,
    )
    ratios = capacities / barrier_amount
    capacity_bounded = np.any(active_formula > 0.0, axis=0)
    trace_mask = (
        capacity_geometry_valid
        & capacity_bounded
        & (capacities <= barrier_amount)
    )
    trace_support = tuple(
        index
        for index, is_trace in zip(canonical_support, trace_mask.tolist())
        if is_trace
    )
    bounded_ratios = ratios[capacity_bounded]
    minimum_ratio = (
        float(np.min(bounded_ratios)) if bounded_ratios.size else None
    )
    return {
        "schema": "exogibbs_finite_barrier_trace_capacity_v1",
        "support_indices": canonical_support,
        "support_count": len(canonical_support),
        "monotone_constraint_row_mask": tuple(
            bool(value) for value in monotone_rows.tolist()
        ),
        "monotone_constraint_row_count": int(np.count_nonzero(monotone_rows)),
        "capacity_geometry_valid": capacity_geometry_valid,
        "finite_barrier_log_barrier": log_barrier_value,
        "finite_barrier_amount": barrier_amount,
        "support_phase_capacity_bounded": tuple(
            bool(value) for value in capacity_bounded.tolist()
        ),
        "support_phase_maximum_amounts": tuple(
            float(value) if bounded else None
            for value, bounded in zip(
                capacities.tolist(), capacity_bounded.tolist()
            )
        ),
        "capacity_to_barrier_ratios": tuple(
            float(value) if bounded else None
            for value, bounded in zip(ratios.tolist(), capacity_bounded.tolist())
        ),
        "minimum_capacity_to_barrier_ratio": minimum_ratio,
        "trace_capacity_support_indices": trace_support,
        "trace_capacity_count": len(trace_support),
        "trace_capacity_detected": bool(trace_support),
    }


def reduce_initial_condensate_support_to_basic(
    *,
    condensate_formula_matrix_full: np.ndarray,
    condensate_standard_source_full: np.ndarray,
    target_inventory: np.ndarray,
    condensate_amounts: np.ndarray,
    support_indices: Sequence[int],
    budget_scale: np.ndarray,
    budget_tolerance: float,
    enabled: bool,
    diagnostic_role: str,
    linear_program_solver: Callable[..., Any] | None = None,
    disabled_reason: str = "disabled",
) -> tuple[tuple[int, ...], np.ndarray, dict[str, Any]]:
    """Return a nonnegative basic representation of an initial inventory.

    The linear program preserves the input condensate inventory and minimizes
    the supplied linear standard-source objective.  The returned support is an
    initializer only; downstream solve and physical acceptance logic remain
    authoritative.  Every ineligible or failed reduction returns the original
    support and a copy of the original full-catalog amounts.
    """

    formula = np.asarray(condensate_formula_matrix_full, dtype=np.float64)
    standard_source = np.asarray(
        condensate_standard_source_full, dtype=np.float64
    )
    target = np.asarray(target_inventory, dtype=np.float64)
    full_amounts = np.asarray(condensate_amounts, dtype=np.float64)
    row_scale = np.asarray(budget_scale, dtype=np.float64)
    support = tuple(int(index) for index in support_indices)
    if formula.ndim != 2:
        raise ValueError("condensate_formula_matrix_full must be two-dimensional.")
    element_count, condensate_count = formula.shape
    if standard_source.shape != (condensate_count,):
        raise ValueError(
            "condensate_standard_source_full must have one value per condensate."
        )
    if target.shape != (element_count,):
        raise ValueError("target_inventory must have one value per element.")
    if full_amounts.shape != (condensate_count,):
        raise ValueError("condensate_amounts must have one value per condensate.")
    if row_scale.shape != (element_count,):
        raise ValueError("budget_scale must have one value per element.")
    if len(set(support)) != len(support) or any(
        index < 0 or index >= condensate_count for index in support
    ):
        raise ValueError("support_indices must contain unique catalog indices.")
    if not np.isfinite(budget_tolerance) or budget_tolerance < 0.0:
        raise ValueError("budget_tolerance must be finite and non-negative.")
    role = str(diagnostic_role)
    if not role:
        raise ValueError("diagnostic_role must be non-empty.")

    canonical_support = tuple(sorted(support))
    active = np.asarray(canonical_support, dtype=np.int64)
    ac = formula[:, active]
    hcond = standard_source[active]
    amounts_before = full_amounts[active]
    amount_scales = maximum_condensate_amount_scales(ac, target)
    scaled_matrix = row_scale[:, None] * ac * amount_scales[None, :]
    finite_geometry = bool(
        np.all(np.isfinite(scaled_matrix))
        and np.all(np.isfinite(row_scale))
        and np.all(row_scale > 0.0)
    )
    input_rank = (
        int(np.linalg.matrix_rank(scaled_matrix))
        if finite_geometry and active.size
        else 0
    )
    residual_tolerance = min(0.1 * float(budget_tolerance), 1.0e-9)
    base_report: dict[str, Any] = {
        "schema": "exogibbs_condensate_basic_support_reduction_v1",
        "role": role,
        "enabled": bool(enabled),
        "eligible": False,
        "attempted": False,
        "applied": False,
        "method": "scipy_linprog_highs_dual_simplex",
        "initial_support_indices": support,
        "canonical_support_indices": canonical_support,
        "initial_support_count": len(support),
        "initial_support_rank": input_rank,
        "initial_support_nullity": len(support) - input_rank,
        "input_support_indices": support,
        "input_support_count": len(support),
        "input_support_rank": input_rank,
        "input_support_nullity": len(support) - input_rank,
        "output_support_indices": support,
        "output_support_count": len(support),
        "output_support_rank": input_rank,
        "output_support_nullity": len(support) - input_rank,
        "output_dropped_support_indices": (),
        "output_scaled_inventory_residual_max_abs": 0.0,
        "scaled_inventory_residual_tolerance": residual_tolerance,
        "fallback_reason": None,
        "relative_amount_floor": BASIC_SUPPORT_RELATIVE_AMOUNT_FLOOR,
        "iteration_limit": BASIC_SUPPORT_LP_ITERATION_LIMIT,
    }

    def fallback(reason: str, *, skipped: bool = False) -> tuple[
        tuple[int, ...], np.ndarray, dict[str, Any]
    ]:
        base_report["fallback_reason"] = reason
        base_report["skip_reason" if skipped else "failure_reason"] = reason
        return support, full_amounts.copy(), base_report

    if not enabled:
        return fallback(str(disabled_reason), skipped=True)
    eligible = bool(
        support
        and len(support) > input_rank
        and finite_geometry
        and np.all(np.isfinite(ac))
        and np.all(ac >= 0.0)
        and np.all(np.any(ac > 0.0, axis=0))
        and np.all(np.isfinite(hcond))
        and np.all(np.isfinite(amounts_before))
        and np.all(amounts_before >= 0.0)
        and np.all(np.isfinite(amount_scales))
        and np.all(amount_scales > 0.0)
    )
    base_report["eligible"] = eligible
    if not eligible:
        return fallback("not_rank_reduction_eligible", skipped=True)

    burden = ac @ amounts_before
    scaled_burden = row_scale * burden
    objective = hcond * amount_scales
    objective_scale = max(
        float(np.max(np.abs(objective), initial=0.0)),
        1.0e-300,
    )
    base_report["attempted"] = True
    solve_lp = linprog if linear_program_solver is None else linear_program_solver
    try:
        solution = solve_lp(
            objective / objective_scale,
            A_eq=scaled_matrix,
            b_eq=scaled_burden,
            bounds=(0.0, None),
            method="highs-ds",
            options={
                "dual_feasibility_tolerance": 1.0e-10,
                "maxiter": BASIC_SUPPORT_LP_ITERATION_LIMIT,
                "presolve": True,
                "primal_feasibility_tolerance": 1.0e-10,
            },
        )
        solver_success = bool(solution.success)
        solver_status = int(solution.status)
        solver_message = str(solution.message)
        solver_iterations = int(solution.nit)
    except (
        FloatingPointError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        base_report.update(
            {
                "solver_success": False,
                "solver_status": -1,
                "solver_message": str(error),
                "solver_iterations": 0,
            }
        )
        return fallback("solver_exception")

    base_report.update(
        {
            "solver_success": solver_success,
            "solver_status": solver_status,
            "solver_message": solver_message,
            "solver_iterations": solver_iterations,
        }
    )
    if not solver_success:
        return fallback("linear_program_failed")

    relative_amounts = np.asarray(solution.x, dtype=np.float64)
    if relative_amounts.shape == (len(active),):
        retained_mask = relative_amounts > BASIC_SUPPORT_RELATIVE_AMOUNT_FLOOR
        reduced_amounts = np.where(
            retained_mask,
            amount_scales * relative_amounts,
            0.0,
        )
    else:
        reduced_amounts = np.zeros(len(active), dtype=np.float64)
    amount_by_index = {
        int(index): float(amount)
        for index, amount in zip(active.tolist(), reduced_amounts.tolist())
        if amount > 0.0
    }
    amount_scale_by_index = {
        int(index): float(scale)
        for index, scale in zip(active.tolist(), amount_scales.tolist())
    }
    reduced_support = tuple(
        index for index in canonical_support if index in amount_by_index
    )
    reduced_full_amounts = np.zeros_like(full_amounts)
    if reduced_support:
        reduced_active = np.asarray(reduced_support, dtype=np.int64)
        reduced_full_amounts[reduced_active] = np.asarray(
            [amount_by_index[index] for index in reduced_support],
            dtype=np.float64,
        )
        reconstructed = formula[:, reduced_active] @ reduced_full_amounts[
            reduced_active
        ]
        reduced_scaled_matrix = (
            row_scale[:, None]
            * formula[:, reduced_active]
            * np.asarray(
                [amount_scale_by_index[index] for index in reduced_support],
                dtype=np.float64,
            )[None, :]
        )
        reduced_rank = int(np.linalg.matrix_rank(reduced_scaled_matrix))
    else:
        reconstructed = np.zeros_like(burden)
        reduced_rank = 0
    scaled_inventory_residual = row_scale * (reconstructed - burden)
    residual_norm = float(
        np.max(np.abs(scaled_inventory_residual), initial=0.0)
    )
    reduced_set = set(reduced_support)
    candidate_dropped = tuple(
        index for index in canonical_support if index not in reduced_set
    )
    valid = bool(
        relative_amounts.shape == (len(active),)
        and reduced_support
        and len(reduced_support) < len(support)
        and len(reduced_support) <= input_rank
        and reduced_rank == len(reduced_support)
        and np.all(np.isfinite(reduced_full_amounts))
        and residual_norm <= residual_tolerance
    )
    base_report.update(
        {
            "applied": valid,
            "candidate_support_indices": reduced_support,
            "candidate_support_count": len(reduced_support),
            "candidate_support_rank": reduced_rank,
            "candidate_dropped_support_indices": candidate_dropped,
            "objective_before": float(hcond @ amounts_before),
            "objective_after": float(hcond @ reduced_full_amounts[active]),
            "scaled_inventory_residual_max_abs": residual_norm,
        }
    )
    if not valid:
        return fallback("postsolve_validation_failed")

    base_report.update(
        {
            "final_support_indices": reduced_support,
            "final_support_count": len(reduced_support),
            "final_support_rank": reduced_rank,
            "dropped_support_indices": candidate_dropped,
            "output_support_indices": reduced_support,
            "output_support_count": len(reduced_support),
            "output_support_rank": reduced_rank,
            "output_support_nullity": len(reduced_support) - reduced_rank,
            "output_dropped_support_indices": candidate_dropped,
            "output_scaled_inventory_residual_max_abs": residual_norm,
        }
    )
    return reduced_support, reduced_full_amounts, base_report


__all__ = (
    "BASIC_SUPPORT_LP_ITERATION_LIMIT",
    "BASIC_SUPPORT_RELATIVE_AMOUNT_FLOOR",
    "finite_barrier_trace_capacity_report",
    "maximum_condensate_amount_scales",
    "reduce_initial_condensate_support_to_basic",
)
