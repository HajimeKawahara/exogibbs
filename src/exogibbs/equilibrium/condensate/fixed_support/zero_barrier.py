"""Host-side zero-barrier refinement for a converged active support."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares, linprog, minimize
from scipy.special import logsumexp


_INITIALIZER_CAPACITY_FRACTION = float(np.sqrt(np.finfo(np.float64).eps))
_REDUCED_SUPPORT_NODE_LIMIT = 32
_ACTIVE_SET_CLOSURE_ROUND_LIMIT = 8
_BASIC_SUPPORT_LP_ITERATION_LIMIT = 1000
_BASIC_SUPPORT_RELATIVE_AMOUNT_FLOOR = 64.0 * np.finfo(np.float64).eps
_FINITE_BARRIER_HOMOTOPY_CENTRALITY_TOLERANCE = 1.0e-4
_FINITE_BARRIER_HOMOTOPY_RESIDUAL_TOLERANCE = 1.0e-8
_FINITE_BARRIER_HOMOTOPY_MINIMUM_GAP_RATIO = 4.0
_FINITE_BARRIER_HOMOTOPY_MAXIMUM_STEP_COUNT = 12
_FINITE_BARRIER_HOMOTOPY_MAX_NFEV_PER_STEP = 100
_DUAL_SUPPORT_ORACLE_ITERATION_LIMIT = 200
_DUAL_SUPPORT_ORACLE_FEASIBILITY_TOLERANCE = 1.0e-10
_SIMPLEX_PIVOT_INVENTORY_RESIDUAL_TOLERANCE = 1.0e-10


def _least_squares_with_scipy_overflow_guard(*args: Any, **kwargs: Any) -> Any:
    """Run SciPy least squares while hiding one benign internal overflow warning."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"overflow encountered in scalar divide",
            category=RuntimeWarning,
            module=r"scipy\.optimize\._lsq\.common$",
        )
        return least_squares(*args, **kwargs)


@dataclass(frozen=True)
class ZeroBarrierPolishResult:
    """Physical state and audit report returned by zero-barrier refinement."""

    accepted: bool
    gas_log_amounts: np.ndarray
    condensate_amounts: np.ndarray
    total_gas_log_amount: float
    element_potential: np.ndarray
    support_indices: tuple[int, ...]
    report: dict[str, Any]


@dataclass
class _FunctionEvaluationBudget:
    """Shared hard limit for nonlinear evaluations in one active-set search."""

    limit: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(int(self.limit) - int(self.used), 0)

    def consume(self, count: int) -> None:
        evaluations = int(count)
        if evaluations < 0 or evaluations > self.remaining:
            raise RuntimeError("Invalid zero-barrier function-evaluation count.")
        self.used += evaluations


def _function_evaluation_call_limit(
    max_function_evaluations: int,
    budget: _FunctionEvaluationBudget | None,
) -> int:
    """Return the remaining per-call evaluation allowance."""

    call_limit = int(max_function_evaluations)
    if budget is not None:
        call_limit = min(call_limit, budget.remaining)
    return max(call_limit, 0)


def _maximum_condensate_amount_scales(
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


class _DualSupportOracleEvaluationLimit(RuntimeError):
    """Internal stop raised when the shared nonlinear budget is exhausted."""


def _select_support_with_zero_barrier_dual(
    *,
    gas_formula_matrix: np.ndarray,
    condensate_formula_matrix_full: np.ndarray,
    target_inventory: np.ndarray,
    gas_standard_source: np.ndarray,
    condensate_standard_source_full: np.ndarray,
    gas_log_amounts_init: np.ndarray,
    condensate_amounts_init: np.ndarray,
    total_gas_log_amount_init: float,
    element_potential_init: np.ndarray,
    condensate_valid_mask: np.ndarray,
    stationarity_tolerance: float,
    support_closure_tolerance: float,
    max_function_evaluations: int,
    enabled: bool,
    function_evaluation_budget: _FunctionEvaluationBudget | None = None,
) -> dict[str, Any]:
    """Select a zero-barrier support from the convex thermodynamic dual.

    Structural zero rows and the species that consume them are removed.  On
    the remaining physical gas branch, the dual maximizes the target-weighted
    element potential subject to gas normalization and nonnegative driving
    for every temperature-valid condensate.  The selected tight constraints
    are only an initializer: an independent exact solve and full physical
    audit remain authoritative.
    """

    ag_full = np.asarray(gas_formula_matrix, dtype=np.float64)
    ac_full = np.asarray(
        condensate_formula_matrix_full, dtype=np.float64
    )
    target_full = np.asarray(target_inventory, dtype=np.float64)
    gamma_full = np.asarray(gas_standard_source, dtype=np.float64)
    hcond_full = np.asarray(
        condensate_standard_source_full, dtype=np.float64
    )
    q_initial_full = np.asarray(
        gas_log_amounts_init, dtype=np.float64
    )
    amounts_initial_full = np.asarray(
        condensate_amounts_init, dtype=np.float64
    )
    lambda_initial_full = np.asarray(
        element_potential_init, dtype=np.float64
    )
    valid_mask = np.asarray(condensate_valid_mask, dtype=bool)
    base_report: dict[str, Any] = {
        "schema": "exogibbs_zero_barrier_dual_support_oracle_v1",
        "role": "initializer_support_selection_only",
        "enabled": bool(enabled),
        "eligible": False,
        "attempted": False,
        "applied": False,
        "method": "scipy_slsqp_convex_dual_gas_boundary",
        "iteration_limit": _DUAL_SUPPORT_ORACLE_ITERATION_LIMIT,
        "feasibility_tolerance": (
            _DUAL_SUPPORT_ORACLE_FEASIBILITY_TOLERANCE
        ),
        "support_selection_tolerance": max(
            float(stationarity_tolerance),
            float(support_closure_tolerance),
        ),
    }

    def failed(reason: str) -> dict[str, Any]:
        base_report["failure_reason"] = reason
        return {
            "applied": False,
            "support_indices": (),
            "gas_log_amounts": q_initial_full.copy(),
            "condensate_amounts": amounts_initial_full.copy(),
            "total_gas_log_amount": float(total_gas_log_amount_init),
            "element_potential": lambda_initial_full.copy(),
            "report": base_report,
        }

    if not enabled:
        base_report["skip_reason"] = "disabled_after_first_active_set_pass"
        return failed("disabled")
    if (
        np.any(target_full < 0.0)
        or np.any(ag_full < 0.0)
        or np.any(ac_full < 0.0)
    ):
        base_report["skip_reason"] = "nonnegative_structure_required"
        return failed("ineligible_structure")

    positive_rows = target_full > 0.0
    zero_rows = target_full == 0.0
    if not np.any(positive_rows):
        base_report["skip_reason"] = "no_positive_target_row"
        return failed("ineligible_structure")
    suppressed_gases = (
        np.any(ag_full[zero_rows] > 0.0, axis=0)
        if np.any(zero_rows)
        else np.zeros(ag_full.shape[1], dtype=bool)
    )
    retained_gases = ~suppressed_gases
    selectable_condensates = valid_mask & (
        ~np.any(ac_full[zero_rows] > 0.0, axis=0)
        if np.any(zero_rows)
        else np.ones(ac_full.shape[1], dtype=bool)
    )
    selectable_indices = np.flatnonzero(selectable_condensates)
    if not np.any(retained_gases):
        base_report["skip_reason"] = "no_structurally_possible_gas"
        return failed("ineligible_structure")
    if not selectable_indices.size:
        base_report["skip_reason"] = "no_selectable_condensate"
        return failed("ineligible_structure")

    ag = ag_full[positive_rows][:, retained_gases]
    ac = ac_full[positive_rows][:, selectable_indices]
    target = target_full[positive_rows]
    gamma = gamma_full[retained_gases]
    hcond = hcond_full[selectable_indices]
    lambda_reference = lambda_initial_full[positive_rows]
    if (
        np.any(~np.any(ag > 0.0, axis=1))
        or np.any(~np.isfinite(lambda_reference))
    ):
        base_report["skip_reason"] = "unrepresented_positive_target_row"
        return failed("ineligible_structure")

    target_max = float(np.max(target))
    target_ratio = target / target_max
    coordinate_scale = np.sqrt(target_max / target)
    selection_tolerance = max(
        float(stationarity_tolerance),
        float(support_closure_tolerance),
    )
    feasibility_tolerance = min(
        _DUAL_SUPPORT_ORACLE_FEASIBILITY_TOLERANCE,
        max(selection_tolerance, 64.0 * np.finfo(np.float64).eps),
    )
    base_report.update(
        {
            "eligible": True,
            "attempted": True,
            "positive_target_rows": tuple(
                int(index)
                for index in np.flatnonzero(positive_rows).tolist()
            ),
            "zero_target_rows": tuple(
                int(index)
                for index in np.flatnonzero(zero_rows).tolist()
            ),
            "suppressed_gas_indices": tuple(
                int(index)
                for index in np.flatnonzero(suppressed_gases).tolist()
            ),
            "selectable_condensate_count": int(selectable_indices.size),
            "feasibility_tolerance": feasibility_tolerance,
        }
    )

    cached_values: np.ndarray | None = None
    cached_state: tuple[
        np.ndarray, float, np.ndarray, np.ndarray
    ] | None = None

    def state(values: np.ndarray):
        nonlocal cached_values, cached_state
        current = np.asarray(values, dtype=np.float64)
        if cached_values is not None and np.array_equal(current, cached_values):
            assert cached_state is not None
            return cached_state
        potential = lambda_reference + coordinate_scale * current
        logits = ag.T @ potential - gamma
        normalization = float(logsumexp(logits))
        fractions = np.exp(logits - normalization)
        mean_formula = ag @ fractions
        driving = hcond - ac.T @ potential
        cached_values = current.copy()
        cached_state = (potential, normalization, mean_formula, driving)
        return cached_state

    evaluation_limit = min(
        _DUAL_SUPPORT_ORACLE_ITERATION_LIMIT,
        _function_evaluation_call_limit(
            max_function_evaluations,
            function_evaluation_budget,
        ),
    )
    if evaluation_limit <= 0:
        base_report["skip_reason"] = "function_evaluation_limit_reached"
        return failed("function_evaluation_limit_reached")
    objective_evaluations = 0

    def objective(values: np.ndarray) -> float:
        nonlocal objective_evaluations
        if objective_evaluations >= evaluation_limit:
            raise _DualSupportOracleEvaluationLimit
        objective_evaluations += 1
        return -float(target_ratio @ (coordinate_scale * values))

    def objective_jacobian(values: np.ndarray) -> np.ndarray:
        del values
        return -(target_ratio * coordinate_scale)

    def normalization_constraint(values: np.ndarray) -> float:
        return -state(values)[1]

    def normalization_jacobian(values: np.ndarray) -> np.ndarray:
        return -(state(values)[2] * coordinate_scale)

    def driving_constraint(values: np.ndarray) -> np.ndarray:
        return state(values)[3]

    def driving_jacobian(values: np.ndarray) -> np.ndarray:
        del values
        return -ac.T * coordinate_scale[None, :]

    try:
        optimization = minimize(
            objective,
            np.zeros_like(lambda_reference),
            jac=objective_jacobian,
            constraints=(
                {
                    "type": "eq",
                    "fun": normalization_constraint,
                    "jac": normalization_jacobian,
                },
                {
                    "type": "ineq",
                    "fun": driving_constraint,
                    "jac": driving_jacobian,
                },
            ),
            method="SLSQP",
            options={
                "disp": False,
                "ftol": 1.0e-12,
                "maxiter": _DUAL_SUPPORT_ORACLE_ITERATION_LIMIT,
            },
        )
    except _DualSupportOracleEvaluationLimit:
        if function_evaluation_budget is not None:
            function_evaluation_budget.consume(evaluation_limit)
        base_report.update(
            {
                "function_evaluations": evaluation_limit,
                "function_evaluations_conservative": True,
                "skip_reason": "function_evaluation_limit_reached",
            }
        )
        return failed("function_evaluation_limit_reached")
    except (FloatingPointError, OverflowError, ValueError) as error:
        if function_evaluation_budget is not None:
            function_evaluation_budget.consume(evaluation_limit)
        base_report.update(
            {
                "function_evaluations": evaluation_limit,
                "function_evaluations_conservative": bool(
                    function_evaluation_budget is not None
                ),
                "optimizer_success": False,
                "optimizer_status": -1,
                "optimizer_message": f"{type(error).__name__}: {error}",
            }
        )
        return failed("optimizer_exception")

    if function_evaluation_budget is not None:
        function_evaluation_budget.consume(objective_evaluations)
    potential, normalization, _mean_formula, driving = state(optimization.x)
    finite_solution = bool(
        np.all(np.isfinite(optimization.x))
        and np.all(np.isfinite(potential))
        and np.isfinite(normalization)
        and np.all(np.isfinite(driving))
    )
    feasible = bool(
        finite_solution
        and abs(normalization) <= feasibility_tolerance
        and float(np.min(driving)) >= -feasibility_tolerance
    )
    tight_mask = driving <= selection_tolerance
    selected_support = tuple(
        int(index) for index in selectable_indices[tight_mask].tolist()
    )
    selected_matrix = ac_full[positive_rows][:, selected_support]
    selected_rank = (
        int(np.linalg.matrix_rank(selected_matrix))
        if selected_support
        else 0
    )
    support_valid = bool(
        selected_support
        and len(selected_support) <= int(np.count_nonzero(positive_rows))
        and selected_rank == len(selected_support)
    )
    inactive_driving = driving[~tight_mask]
    base_report.update(
        {
            "optimizer_success": bool(optimization.success),
            "optimizer_status": int(optimization.status),
            "optimizer_message": str(optimization.message),
            "optimizer_iterations": int(optimization.nit),
            "function_evaluations": int(objective_evaluations),
            "gas_normalization_log_residual": normalization,
            "minimum_condensate_driving": float(np.min(driving)),
            "selected_support_indices": selected_support,
            "selected_support_count": len(selected_support),
            "selected_support_rank": selected_rank,
            "smallest_inactive_driving": (
                float(np.min(inactive_driving))
                if inactive_driving.size
                else np.inf
            ),
            "dual_feasibility_passed": feasible,
            "support_structure_passed": support_valid,
        }
    )
    if not optimization.success:
        return failed("optimizer_failed")
    if not feasible:
        return failed("dual_feasibility_failed")
    if not support_valid:
        return failed("selected_support_not_full_column_rank")

    q_selected = q_initial_full.copy()
    logits = ag.T @ potential - gamma
    normalization = float(logsumexp(logits))
    q_selected[retained_gases] = (
        float(total_gas_log_amount_init) + logits - normalization
    )
    amounts_selected = np.zeros_like(amounts_initial_full)
    selected_array = np.asarray(selected_support, dtype=np.int64)
    amounts_selected[selected_array] = np.maximum(
        amounts_initial_full[selected_array], 0.0
    )
    lambda_selected = lambda_initial_full.copy()
    lambda_selected[positive_rows] = potential
    base_report["applied"] = True
    return {
        "applied": True,
        "support_indices": selected_support,
        "gas_log_amounts": q_selected,
        "condensate_amounts": amounts_selected,
        "total_gas_log_amount": float(total_gas_log_amount_init),
        "element_potential": lambda_selected,
        "report": base_report,
    }


def _select_support_with_finite_barrier_homotopy(
    *,
    gas_formula_matrix: np.ndarray,
    condensate_formula_matrix_full: np.ndarray,
    target_inventory: np.ndarray,
    gas_standard_source: np.ndarray,
    condensate_standard_source_full: np.ndarray,
    gas_log_amounts_init: np.ndarray,
    condensate_amounts_init: np.ndarray,
    total_gas_log_amount_init: float,
    element_potential_init: np.ndarray,
    support_indices: Sequence[int],
    budget_scale: np.ndarray,
    max_function_evaluations: int,
    enabled: bool,
    function_evaluation_budget: _FunctionEvaluationBudget | None = None,
) -> dict[str, Any]:
    """Select a zero-barrier support by following the finite central path.

    This is an initializer only.  It removes structurally impossible species,
    analytically eliminates gas log amounts, and takes bounded half-decade
    barrier steps.  It retains the deepest continuation state whose max-norm
    residual is at most ``1e-8`` and stops when the next step loses that
    certificate.  A support is selected only when the final adjacent
    capacity-relative amount gap is at least four.  The subsequent exact
    solve and physical audit remain authoritative.
    """

    ag_full = np.asarray(gas_formula_matrix, dtype=np.float64)
    ac_full = np.asarray(
        condensate_formula_matrix_full, dtype=np.float64
    )
    target_full = np.asarray(target_inventory, dtype=np.float64)
    gamma_full = np.asarray(gas_standard_source, dtype=np.float64)
    hcond_full = np.asarray(
        condensate_standard_source_full, dtype=np.float64
    )
    q_initial_full = np.asarray(
        gas_log_amounts_init, dtype=np.float64
    )
    amounts_initial_full = np.asarray(
        condensate_amounts_init, dtype=np.float64
    )
    lambda_initial_full = np.asarray(
        element_potential_init, dtype=np.float64
    )
    initial_support = tuple(int(index) for index in support_indices)
    base_report: dict[str, Any] = {
        "schema": "exogibbs_zero_barrier_finite_homotopy_initializer_v1",
        "role": "initializer_support_selection_only",
        "enabled": bool(enabled),
        "eligible": False,
        "attempted": False,
        "applied": False,
        "maximum_step_count": (
            _FINITE_BARRIER_HOMOTOPY_MAXIMUM_STEP_COUNT
        ),
        "barrier_log10_step": -0.5,
        "centrality_relative_spread_tolerance": (
            _FINITE_BARRIER_HOMOTOPY_CENTRALITY_TOLERANCE
        ),
        "continuation_residual_tolerance": (
            _FINITE_BARRIER_HOMOTOPY_RESIDUAL_TOLERANCE
        ),
        "minimum_capacity_relative_gap_ratio": (
            _FINITE_BARRIER_HOMOTOPY_MINIMUM_GAP_RATIO
        ),
        "maximum_function_evaluations_per_step": (
            _FINITE_BARRIER_HOMOTOPY_MAX_NFEV_PER_STEP
        ),
        "initial_support_indices": initial_support,
    }

    def failed(reason: str) -> dict[str, Any]:
        base_report["failure_reason"] = reason
        return {
            "applied": False,
            "support_indices": initial_support,
            "gas_log_amounts": q_initial_full.copy(),
            "condensate_amounts": amounts_initial_full.copy(),
            "total_gas_log_amount": float(total_gas_log_amount_init),
            "element_potential": lambda_initial_full.copy(),
            "report": base_report,
        }

    if not enabled:
        base_report["skip_reason"] = "disabled_after_first_active_set_pass"
        return failed("disabled")
    if (
        np.any(target_full < 0.0)
        or np.any(ag_full < 0.0)
        or np.any(ac_full < 0.0)
    ):
        base_report["skip_reason"] = "nonnegative_structure_required"
        return failed("ineligible_structure")

    positive_rows = target_full > 0.0
    zero_rows = target_full == 0.0
    if not np.any(positive_rows):
        base_report["skip_reason"] = "no_positive_target_row"
        return failed("ineligible_structure")
    suppressed_gases = (
        np.any(ag_full[zero_rows] > 0.0, axis=0)
        if np.any(zero_rows)
        else np.zeros(ag_full.shape[1], dtype=bool)
    )
    retained_gases = ~suppressed_gases
    if not np.any(retained_gases):
        base_report["skip_reason"] = "no_structurally_possible_gas"
        return failed("ineligible_structure")
    forced_dropped = tuple(
        index
        for index in initial_support
        if np.any(ac_full[zero_rows, index] > 0.0)
    )
    forced_set = set(forced_dropped)
    continuation_support = tuple(
        index for index in initial_support if index not in forced_set
    )
    base_report.update(
        {
            "positive_target_rows": tuple(
                int(index)
                for index in np.flatnonzero(positive_rows).tolist()
            ),
            "zero_target_rows": tuple(
                int(index)
                for index in np.flatnonzero(zero_rows).tolist()
            ),
            "suppressed_gas_indices": tuple(
                int(index)
                for index in np.flatnonzero(suppressed_gases).tolist()
            ),
            "structural_zero_dropped_support_indices": forced_dropped,
            "continuation_support_indices": continuation_support,
        }
    )
    if len(continuation_support) < 2:
        base_report["skip_reason"] = "fewer_than_two_selectable_phases"
        return failed("no_support_gap_possible")

    active = np.asarray(continuation_support, dtype=np.int64)
    ag = ag_full[positive_rows][:, retained_gases]
    ac = ac_full[positive_rows][:, active]
    target = target_full[positive_rows]
    gamma = gamma_full[retained_gases]
    hcond = hcond_full[active]
    if np.any(~np.any(ac > 0.0, axis=0)):
        base_report["skip_reason"] = "zero_stoichiometry_active_phase"
        return failed("ineligible_structure")
    amount_scales = _maximum_condensate_amount_scales(ac, target)
    active_amounts_initial = amounts_initial_full[active]
    lambda_initial = lambda_initial_full[positive_rows]
    initial_driving = hcond - ac.T @ lambda_initial
    central_products = active_amounts_initial * initial_driving
    central_products_valid = bool(
        np.all(np.isfinite(central_products))
        and np.all(central_products > 0.0)
        and np.all(np.isfinite(amount_scales))
        and np.all(amount_scales > 0.0)
        and np.all(np.isfinite(active_amounts_initial))
        and np.all(active_amounts_initial > 0.0)
    )
    if not central_products_valid:
        base_report["skip_reason"] = "invalid_finite_barrier_central_products"
        return failed("centrality_guard_failed")
    mu0 = float(np.median(central_products))
    centrality_spread = float(
        np.max(np.abs(central_products / mu0 - 1.0), initial=0.0)
    )
    base_report.update(
        {
            "initial_barrier_scale": mu0,
            "initial_centrality_relative_spread_max_abs": (
                centrality_spread
            ),
        }
    )
    if (
        not np.isfinite(mu0)
        or mu0 <= 0.0
        or centrality_spread
        > _FINITE_BARRIER_HOMOTOPY_CENTRALITY_TOLERANCE
    ):
        base_report["skip_reason"] = "finite_barrier_state_not_central"
        return failed("centrality_guard_failed")

    log_relative_initial = (
        np.log(active_amounts_initial) - np.log(amount_scales)
    )
    if not np.all(np.isfinite(log_relative_initial)):
        base_report["skip_reason"] = "nonfinite_capacity_relative_amount"
        return failed("centrality_guard_failed")
    element_count = target.size
    x = np.concatenate(
        [
            lambda_initial,
            [float(total_gas_log_amount_init)],
            log_relative_initial,
        ]
    )
    rounds: list[dict[str, Any]] = []
    base_report["eligible"] = True
    base_report["attempted"] = True

    def state(values: np.ndarray):
        lambda_ = values[:element_count]
        qtot = float(values[element_count])
        log_relative = values[element_count + 1 :]
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            amounts = amount_scales * np.exp(log_relative)
        logits = ag.T @ lambda_ - gamma
        normalization = float(logsumexp(logits))
        log_fractions = logits - normalization
        fractions = np.exp(log_fractions)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            gas = np.exp(qtot + log_fractions)
        gas_inventory = ag @ gas
        driving = hcond - ac.T @ lambda_
        return (
            lambda_,
            qtot,
            log_relative,
            amounts,
            normalization,
            log_fractions,
            fractions,
            gas,
            gas_inventory,
            driving,
        )

    last_certified_x: np.ndarray | None = None
    continuation_termination_reason = "maximum_step_count_reached"
    for step_index in range(
        _FINITE_BARRIER_HOMOTOPY_MAXIMUM_STEP_COUNT
    ):
        mu = float(mu0 * 10.0 ** (-0.5 * (step_index + 1)))

        def residual(values: np.ndarray) -> np.ndarray:
            (
                _lambda,
                _qtot,
                _log_relative,
                amounts,
                normalization,
                _log_fractions,
                _fractions,
                _gas,
                gas_inventory,
                driving,
            ) = state(values)
            return np.concatenate(
                [
                    amounts * driving / mu - 1.0,
                    np.asarray([normalization], dtype=np.float64),
                    budget_scale[positive_rows]
                    * (gas_inventory + ac @ amounts - target),
                ]
            )

        def jacobian(values: np.ndarray) -> np.ndarray:
            (
                _lambda,
                _qtot,
                _log_relative,
                amounts,
                _normalization,
                _log_fractions,
                fractions,
                gas,
                gas_inventory,
                driving,
            ) = state(values)
            support_count = active.size
            variable_count = element_count + 1 + support_count
            matrix = np.zeros(
                (variable_count, variable_count), dtype=np.float64
            )
            matrix[:support_count, :element_count] = (
                -(amounts / mu)[:, None] * ac.T
            )
            matrix[
                :support_count, element_count + 1 :
            ] = np.diag(amounts * driving / mu)
            mean_formula = ag @ fractions
            matrix[support_count, :element_count] = mean_formula
            budget_row_start = support_count + 1
            centered_formula = ag.T - mean_formula
            gas_covariance = ag @ (gas[:, None] * centered_formula)
            positive_budget_scale = budget_scale[positive_rows]
            matrix[budget_row_start:, :element_count] = (
                positive_budget_scale[:, None] * gas_covariance
            )
            matrix[budget_row_start:, element_count] = (
                positive_budget_scale * gas_inventory
            )
            matrix[budget_row_start:, element_count + 1 :] = (
                positive_budget_scale[:, None]
                * ac
                * amounts[None, :]
            )
            return matrix

        call_evaluation_limit = min(
            _FINITE_BARRIER_HOMOTOPY_MAX_NFEV_PER_STEP,
            _function_evaluation_call_limit(
                max_function_evaluations,
                function_evaluation_budget,
            ),
        )
        if call_evaluation_limit <= 0:
            rounds.append(
                {
                    "step_index": step_index,
                    "barrier_scale": mu,
                    "optimizer_success": False,
                    "function_evaluations": 0,
                    "failure_reason": "function_evaluation_limit_reached",
                }
            )
            continuation_termination_reason = (
                "function_evaluation_limit_reached"
            )
            if last_certified_x is None:
                base_report["rounds"] = tuple(rounds)
                return failed("continuation_failed")
            break
        try:
            optimization = _least_squares_with_scipy_overflow_guard(
                residual,
                x,
                jac=jacobian,
                method="trf",
                x_scale="jac",
                ftol=1.0e-13,
                xtol=1.0e-13,
                gtol=1.0e-13,
                max_nfev=call_evaluation_limit,
            )
        except (FloatingPointError, OverflowError, ValueError) as error:
            conservative_evaluations = 0
            if function_evaluation_budget is not None:
                function_evaluation_budget.consume(call_evaluation_limit)
                conservative_evaluations = call_evaluation_limit
            rounds.append(
                {
                    "step_index": step_index,
                    "barrier_scale": mu,
                    "optimizer_success": False,
                    "function_evaluations": conservative_evaluations,
                    "function_evaluations_conservative": bool(
                        function_evaluation_budget is not None
                    ),
                    "failure_reason": f"{type(error).__name__}: {error}",
                }
            )
            continuation_termination_reason = "optimizer_exception"
            if last_certified_x is None:
                base_report["rounds"] = tuple(rounds)
                return failed("continuation_failed")
            break
        if function_evaluation_budget is not None:
            function_evaluation_budget.consume(int(optimization.nfev))
        x = np.asarray(optimization.x, dtype=np.float64)
        final_residual = residual(x)
        residual_norm = float(
            np.max(np.abs(final_residual), initial=0.0)
        )
        log_relative = x[element_count + 1 :]
        sorted_log_relative = np.sort(log_relative)
        log_gaps = np.diff(sorted_log_relative)
        gap_index = int(np.argmax(log_gaps))
        split_count = int(log_relative.size - gap_index - 1)
        round_report = {
            "step_index": step_index,
            "barrier_scale": mu,
            "optimizer_success": bool(optimization.success),
            "optimizer_status": int(optimization.status),
            "optimizer_message": str(optimization.message),
            "function_evaluations": int(optimization.nfev),
            "continuation_residual_max_abs": residual_norm,
            "largest_log_capacity_relative_gap": float(
                log_gaps[gap_index]
            ),
            "support_count_above_largest_gap": split_count,
        }
        rounds.append(round_report)
        if (
            not optimization.success
            or not np.all(np.isfinite(final_residual))
            or residual_norm
            > _FINITE_BARRIER_HOMOTOPY_RESIDUAL_TOLERANCE
        ):
            continuation_termination_reason = (
                "continuation_certificate_lost"
            )
            if last_certified_x is None:
                base_report["rounds"] = tuple(rounds)
                return failed("continuation_failed")
            break
        last_certified_x = x.copy()

    if last_certified_x is None:
        base_report["rounds"] = tuple(rounds)
        return failed("continuation_failed")
    x = last_certified_x
    certified_rounds = tuple(
        item
        for item in rounds
        if item.get("optimizer_success", False)
        and item.get("continuation_residual_max_abs", np.inf)
        <= _FINITE_BARRIER_HOMOTOPY_RESIDUAL_TOLERANCE
    )
    base_report.update(
        {
            "rounds": tuple(rounds),
            "attempted_step_count": len(rounds),
            "certified_step_count": len(certified_rounds),
            "selected_step_index": int(certified_rounds[-1]["step_index"]),
            "continuation_termination_reason": (
                continuation_termination_reason
            ),
        }
    )

    final_state = state(x)
    (
        lambda_final,
        qtot_final,
        log_relative_final,
        amounts_final,
        _normalization_final,
        log_fractions_final,
        _fractions_final,
        _gas_final,
        _gas_inventory_final,
        _driving_final,
    ) = final_state
    sorted_log_relative = np.sort(log_relative_final)
    log_gaps = np.diff(sorted_log_relative)
    gap_index = int(np.argmax(log_gaps))
    largest_log_gap = float(log_gaps[gap_index])
    gap_lower = float(sorted_log_relative[gap_index])
    gap_upper = float(sorted_log_relative[gap_index + 1])
    gap_guard_passed = bool(
        largest_log_gap
        >= np.log(_FINITE_BARRIER_HOMOTOPY_MINIMUM_GAP_RATIO)
    )
    base_report.update(
        {
            "final_largest_log_capacity_relative_gap": largest_log_gap,
            "final_largest_capacity_relative_gap_ratio": float(
                np.exp(largest_log_gap)
            ),
            "final_gap_lower_log_capacity_relative_amount": gap_lower,
            "final_gap_upper_log_capacity_relative_amount": gap_upper,
            "gap_guard_passed": gap_guard_passed,
        }
    )
    if not gap_guard_passed:
        base_report["skip_reason"] = "capacity_relative_gap_not_clear"
        return failed("support_gap_guard_failed")

    gap_midpoint = 0.5 * (gap_lower + gap_upper)
    retained_active_mask = log_relative_final > gap_midpoint
    selected_support = tuple(
        index
        for index, retained in zip(
            continuation_support, retained_active_mask.tolist()
        )
        if retained
    )
    if not selected_support or len(selected_support) >= len(initial_support):
        base_report["skip_reason"] = "support_gap_did_not_reduce_support"
        return failed("support_gap_guard_failed")

    full_q = q_initial_full.copy()
    full_q[retained_gases] = qtot_final + log_fractions_final
    full_amounts = np.zeros_like(amounts_initial_full)
    selected_set = set(selected_support)
    for index, amount in zip(
        continuation_support, amounts_final.tolist()
    ):
        if index in selected_set:
            full_amounts[index] = float(amount)
    full_lambda = lambda_initial_full.copy()
    full_lambda[positive_rows] = lambda_final
    dropped_support = tuple(
        index for index in initial_support if index not in selected_set
    )
    base_report.update(
        {
            "applied": True,
            "selected_support_indices": selected_support,
            "selected_support_count": len(selected_support),
            "dropped_support_indices": dropped_support,
        }
    )
    return {
        "applied": True,
        "support_indices": selected_support,
        "gas_log_amounts": full_q,
        "condensate_amounts": full_amounts,
        "total_gas_log_amount": qtot_final,
        "element_potential": full_lambda,
        "report": base_report,
    }


def _reduce_initial_condensate_support_to_basic(
    *,
    condensate_formula_matrix_full: np.ndarray,
    condensate_standard_source_full: np.ndarray,
    target_inventory: np.ndarray,
    condensate_amounts: np.ndarray,
    support_indices: Sequence[int],
    budget_scale: np.ndarray,
    budget_tolerance: float,
    enabled: bool,
) -> tuple[tuple[int, ...], np.ndarray, dict[str, Any]]:
    """Return a nonnegative basic representation of initial condensates.

    The linear program preserves the finite-barrier condensate inventory and
    minimizes its zero-barrier Gibbs term.  It is an initializer reduction;
    the joint exact solve and independent physical audit remain authoritative.
    """

    support = tuple(int(index) for index in support_indices)
    full_amounts = np.asarray(condensate_amounts, dtype=np.float64)
    canonical_support = tuple(sorted(support))
    active = np.asarray(canonical_support, dtype=np.int64)
    ac = condensate_formula_matrix_full[:, active]
    hcond = condensate_standard_source_full[active]
    amounts_before = full_amounts[active]
    amount_scales = _maximum_condensate_amount_scales(
        ac, target_inventory
    )
    scaled_matrix = (
        budget_scale[:, None] * ac * amount_scales[None, :]
    )
    input_rank = int(np.linalg.matrix_rank(scaled_matrix))
    base_report: dict[str, Any] = {
        "schema": "exogibbs_zero_barrier_basic_support_reduction_v1",
        "role": "initial_exact_solve_only",
        "enabled": bool(enabled),
        "attempted": False,
        "applied": False,
        "method": "scipy_linprog_highs_dual_simplex",
        "initial_support_indices": support,
        "canonical_support_indices": canonical_support,
        "initial_support_count": len(support),
        "initial_support_rank": input_rank,
        "initial_support_nullity": len(support) - input_rank,
        "relative_amount_floor": (
            _BASIC_SUPPORT_RELATIVE_AMOUNT_FLOOR
        ),
        "iteration_limit": _BASIC_SUPPORT_LP_ITERATION_LIMIT,
    }
    if not enabled:
        base_report["skip_reason"] = "disabled_for_active_set_retry"
        return support, full_amounts.copy(), base_report
    eligible = bool(
        support
        and len(support) > input_rank
        and np.all(ac >= 0.0)
        and np.all(np.any(ac > 0.0, axis=0))
        and np.all(np.isfinite(amounts_before))
        and np.all(amounts_before >= 0.0)
        and np.all(np.isfinite(amount_scales))
        and np.all(amount_scales > 0.0)
    )
    if not eligible:
        base_report["skip_reason"] = "not_rank_reduction_eligible"
        return support, full_amounts.copy(), base_report

    burden = ac @ amounts_before
    scaled_burden = budget_scale * burden
    objective = hcond * amount_scales
    objective_scale = max(
        float(np.max(np.abs(objective), initial=0.0)),
        1.0e-300,
    )
    base_report["attempted"] = True
    try:
        solution = linprog(
            objective / objective_scale,
            A_eq=scaled_matrix,
            b_eq=scaled_burden,
            bounds=(0.0, None),
            method="highs-ds",
            options={
                "dual_feasibility_tolerance": 1.0e-10,
                "maxiter": _BASIC_SUPPORT_LP_ITERATION_LIMIT,
                "presolve": True,
                "primal_feasibility_tolerance": 1.0e-10,
            },
        )
    except (FloatingPointError, OverflowError, ValueError) as error:
        base_report.update(
            {
                "solver_success": False,
                "solver_status": -1,
                "solver_message": str(error),
                "solver_iterations": 0,
                "failure_reason": "solver_exception",
            }
        )
        return support, full_amounts.copy(), base_report

    base_report.update(
        {
            "solver_success": bool(solution.success),
            "solver_status": int(solution.status),
            "solver_message": str(solution.message),
            "solver_iterations": int(solution.nit),
        }
    )
    if not solution.success:
        base_report["failure_reason"] = "linear_program_failed"
        return support, full_amounts.copy(), base_report

    relative_amounts = np.asarray(solution.x, dtype=np.float64)
    retained_mask = (
        relative_amounts > _BASIC_SUPPORT_RELATIVE_AMOUNT_FLOOR
    )
    reduced_amounts = np.where(
        retained_mask,
        amount_scales * relative_amounts,
        0.0,
    )
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
        index for index in support if index in amount_by_index
    )
    reduced_full_amounts = np.zeros_like(full_amounts)
    if reduced_support:
        reduced_full_amounts[
            np.asarray(reduced_support, dtype=np.int64)
        ] = np.asarray(
            [amount_by_index[index] for index in reduced_support],
            dtype=np.float64,
        )
    reconstructed = (
        condensate_formula_matrix_full[:, reduced_support]
        @ reduced_full_amounts[np.asarray(reduced_support, dtype=np.int64)]
        if reduced_support
        else np.zeros_like(burden)
    )
    scaled_inventory_residual = budget_scale * (reconstructed - burden)
    residual_norm = float(
        np.max(np.abs(scaled_inventory_residual), initial=0.0)
    )
    residual_tolerance = min(0.1 * float(budget_tolerance), 1.0e-9)
    reduced_scaled_matrix = (
        budget_scale[:, None]
        * condensate_formula_matrix_full[:, reduced_support]
        * np.asarray(
            [amount_scale_by_index[index] for index in reduced_support],
            dtype=np.float64,
        )[None, :]
        if reduced_support
        else np.zeros((target_inventory.size, 0), dtype=np.float64)
    )
    reduced_rank = (
        int(np.linalg.matrix_rank(reduced_scaled_matrix))
        if reduced_support
        else 0
    )
    candidate_dropped = tuple(
        index for index in support if index not in set(reduced_support)
    )
    valid = bool(
        reduced_support
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
            "objective_after": float(
                hcond @ reduced_full_amounts[active]
            ),
            "scaled_inventory_residual_max_abs": residual_norm,
            "scaled_inventory_residual_tolerance": residual_tolerance,
        }
    )
    if not valid:
        base_report["failure_reason"] = "postsolve_validation_failed"
        return support, full_amounts.copy(), base_report
    base_report.update(
        {
            "final_support_indices": reduced_support,
            "final_support_count": len(reduced_support),
            "final_support_rank": reduced_rank,
            "dropped_support_indices": candidate_dropped,
        }
    )
    return reduced_support, reduced_full_amounts, base_report


def _build_alternative_basic_support_candidates(
    *,
    condensate_formula_matrix_full: np.ndarray,
    target_inventory: np.ndarray,
    condensate_amounts: np.ndarray,
    support_indices: Sequence[int],
    budget_scale: np.ndarray,
    budget_tolerance: float,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Build a bounded portfolio of feasible bases for one inventory burden.

    A rank-deficient condensate support can admit several nonnegative basic
    representations.  The LP reduction above provides a useful starting
    point, but one LP vertex is not sufficient to select the zero-barrier
    active set.  This routine traverses the deterministic one-column-exchange
    graph of full-rank bases and leaves physical selection to the exact solve.
    """

    support = tuple(int(index) for index in support_indices)
    canonical_support = tuple(sorted(support))
    active = np.asarray(canonical_support, dtype=np.int64)
    ac = condensate_formula_matrix_full[:, active]
    amounts = np.asarray(condensate_amounts, dtype=np.float64)
    amount_scales = _maximum_condensate_amount_scales(ac, target_inventory)
    scaled_matrix = budget_scale[:, None] * ac * amount_scales[None, :]
    support_rank = int(np.linalg.matrix_rank(scaled_matrix))
    report: dict[str, Any] = {
        "schema": "exogibbs_zero_barrier_basic_support_candidates_v1",
        "eligible": False,
        "attempted": False,
        "initial_support_indices": support,
        "canonical_support_indices": canonical_support,
        "initial_support_count": len(support),
        "initial_support_rank": support_rank,
        "initial_support_nullity": len(support) - support_rank,
        "relative_amount_floor": _BASIC_SUPPORT_RELATIVE_AMOUNT_FLOOR,
        "node_limit": _REDUCED_SUPPORT_NODE_LIMIT,
        "visited_basis_indices": (),
        "candidate_records": (),
        "feasible_support_indices": (),
        "candidate_count": 0,
    }
    amounts_on_support = amounts[active]
    eligible = bool(
        support_rank > 0
        and len(support) > support_rank
        and np.all(ac >= 0.0)
        and np.all(np.any(ac > 0.0, axis=0))
        and np.all(np.isfinite(amounts_on_support))
        and np.all(amounts_on_support >= 0.0)
        and np.all(np.isfinite(amount_scales))
        and np.all(amount_scales > 0.0)
    )
    report["eligible"] = eligible
    if not eligible:
        report["skip_reason"] = "not_rank_deficient_support_eligible"
        return (), report

    burden = ac @ amounts_on_support
    scaled_burden = budget_scale * burden
    residual_tolerance = min(0.1 * float(budget_tolerance), 1.0e-9)
    position_by_index = {
        int(index): position
        for position, index in enumerate(canonical_support)
    }

    def basis_rank(candidate_support: tuple[int, ...]) -> int:
        positions = np.asarray(
            [position_by_index[index] for index in candidate_support],
            dtype=np.int64,
        )
        return int(np.linalg.matrix_rank(scaled_matrix[:, positions]))

    seed: list[int] = []
    seed_rank = 0
    for index in canonical_support:
        proposed = tuple(sorted((*seed, index)))
        proposed_rank = basis_rank(proposed)
        if proposed_rank > seed_rank:
            seed = list(proposed)
            seed_rank = proposed_rank
        if seed_rank == support_rank:
            break
    seed_support = tuple(seed)
    report["seed_support_indices"] = seed_support
    if len(seed_support) != support_rank:
        report["skip_reason"] = "full_rank_seed_not_found"
        return (), report

    report["attempted"] = True
    queue = [seed_support]
    discovered = {seed_support}
    visited: list[tuple[int, ...]] = []
    truncated = False
    while queue and len(visited) < _REDUCED_SUPPORT_NODE_LIMIT:
        basis = queue.pop(0)
        visited.append(basis)
        child_bases = set()
        basis_set = set(basis)
        for dropped_index in basis:
            retained = basis_set - {dropped_index}
            for added_index in canonical_support:
                if added_index in basis_set:
                    continue
                child = tuple(sorted((*retained, added_index)))
                if child in discovered or basis_rank(child) != support_rank:
                    continue
                child_bases.add(child)
        for child in sorted(child_bases):
            if len(discovered) >= _REDUCED_SUPPORT_NODE_LIMIT:
                truncated = True
                break
            discovered.add(child)
            queue.append(child)

    candidates: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    for candidate_support in visited:
        indices = np.asarray(candidate_support, dtype=np.int64)
        positions = np.asarray(
            [position_by_index[index] for index in candidate_support],
            dtype=np.int64,
        )
        candidate_scales = amount_scales[positions]
        candidate_matrix = scaled_matrix[:, positions]
        try:
            relative_amounts, _, candidate_rank, _ = np.linalg.lstsq(
                candidate_matrix,
                scaled_burden,
                rcond=None,
            )
        except np.linalg.LinAlgError:
            candidate_records.append(
                {
                    "support_indices": candidate_support,
                    "eligible": False,
                    "rejection_reason": "least_squares_failed",
                }
            )
            continue
        residual = candidate_matrix @ relative_amounts - scaled_burden
        residual_norm = float(np.max(np.abs(residual), initial=0.0))
        finite = bool(np.all(np.isfinite(relative_amounts)))
        positive = bool(
            finite
            and np.all(
                relative_amounts
                > _BASIC_SUPPORT_RELATIVE_AMOUNT_FLOOR
            )
        )
        feasible = bool(
            int(candidate_rank) == support_rank
            and positive
            and residual_norm <= residual_tolerance
        )
        if int(candidate_rank) != support_rank:
            rejection_reason = "rank_deficient_basis"
        elif not finite:
            rejection_reason = "nonfinite_amounts"
        elif not positive:
            rejection_reason = "nonpositive_amounts"
        elif residual_norm > residual_tolerance:
            rejection_reason = "inventory_residual"
        else:
            rejection_reason = None
        candidate_amounts = np.zeros_like(amounts)
        if finite:
            candidate_amounts[indices] = candidate_scales * relative_amounts
        candidate_records.append(
            {
                "support_indices": candidate_support,
                "support_rank": int(candidate_rank),
                "eligible": feasible,
                "rejection_reason": rejection_reason,
                "scaled_inventory_residual_max_abs": residual_norm,
            }
        )
        if feasible:
            candidates.append(
                {
                    "support_indices": candidate_support,
                    "condensate_amounts": candidate_amounts,
                }
            )

    candidates.sort(key=lambda candidate: tuple(candidate["support_indices"]))
    report.update(
        {
            "visited_basis_indices": tuple(visited),
            "visited_basis_count": len(visited),
            "node_limit_reached": truncated,
            "candidate_records": tuple(candidate_records),
            "feasible_support_indices": tuple(
                tuple(candidate["support_indices"])
                for candidate in candidates
            ),
            "candidate_count": len(candidates),
            "candidate_ordering": "canonical_support_indices",
        }
    )
    if not candidates:
        report["failure_reason"] = "no_feasible_basic_support"
    return tuple(candidates), report


def _physical_zero_barrier_audit(
    *,
    gas_formula_matrix: np.ndarray,
    condensate_formula_matrix_full: np.ndarray,
    target_inventory: np.ndarray,
    gas_standard_source: np.ndarray,
    condensate_standard_source_full: np.ndarray,
    gas_log_amounts: np.ndarray,
    condensate_amounts: np.ndarray,
    total_gas_log_amount: float,
    element_potential: np.ndarray,
    support_indices: Sequence[int],
    condensate_valid_mask: np.ndarray,
    budget_scale: np.ndarray,
    optimizer_success: bool,
    optimizer_status: int | None = None,
    stationarity_tolerance: float,
    budget_tolerance: float,
    total_density_tolerance: float,
    support_closure_tolerance: float,
    budget_residual_amount_scale: float = 1.0,
) -> dict[str, Any]:
    """Audit one candidate independently of its numerical formulation."""

    budget_amount_scale = float(budget_residual_amount_scale)
    if not np.isfinite(budget_amount_scale) or budget_amount_scale <= 0.0:
        raise ValueError(
            "budget_residual_amount_scale must be finite and positive."
        )
    q = np.asarray(gas_log_amounts, dtype=np.float64)
    amounts = np.asarray(condensate_amounts, dtype=np.float64)
    lambda_ = np.asarray(element_potential, dtype=np.float64)
    qtot = float(total_gas_log_amount)
    support = tuple(int(index) for index in support_indices)
    with np.errstate(over="ignore", invalid="ignore"):
        gas = np.exp(q)
        total_gas = np.exp(qtot)
        total_density_residual_scaled = np.sum(np.exp(q - qtot)) - 1.0
    full_driving = (
        condensate_standard_source_full
        - condensate_formula_matrix_full.T @ lambda_
    )
    support_mask = np.zeros(amounts.shape[0], dtype=bool)
    if support:
        support_mask[np.asarray(support, dtype=np.int64)] = True
    inactive_mask = (~support_mask) & condensate_valid_mask
    inactive_violation = np.where(
        inactive_mask,
        np.maximum(-full_driving, 0.0),
        0.0,
    )
    gas_stationarity = (
        q
        + gas_standard_source
        - qtot
        - gas_formula_matrix.T @ lambda_
    )
    budget_residual = (
        gas_formula_matrix @ gas
        + condensate_formula_matrix_full @ amounts
        - target_inventory
    )
    budget_residual_scaled = budget_scale * np.divide(
        budget_residual,
        budget_amount_scale,
    )
    active_driving = full_driving[support_mask]
    active_amounts = amounts[support_mask]
    finite = bool(
        all(
            np.all(np.isfinite(value))
            for value in (
                q,
                gas,
                amounts,
                lambda_,
                full_driving,
                budget_residual_scaled,
                gas_stationarity,
                np.asarray(
                    [qtot, total_gas, total_density_residual_scaled],
                    dtype=np.float64,
                ),
            )
        )
    )

    def max_abs(values: np.ndarray) -> float:
        return float(np.max(np.abs(values), initial=0.0))

    gas_stationarity_norm = max_abs(gas_stationarity)
    active_driving_norm = max_abs(active_driving)
    inactive_violation_norm = max_abs(inactive_violation)
    budget_norm = max_abs(budget_residual_scaled)
    total_norm = abs(float(total_density_residual_scaled))
    positive_active_amounts = bool(
        not support or np.all(active_amounts > 0.0)
    )
    physical_root_certified = bool(
        finite
        and positive_active_amounts
        and gas_stationarity_norm <= stationarity_tolerance
        and active_driving_norm <= stationarity_tolerance
        and inactive_violation_norm <= support_closure_tolerance
        and budget_norm <= budget_tolerance
        and total_norm <= total_density_tolerance
    )
    eligible_acceptance_source = _zero_barrier_acceptance_source(
        optimizer_success=optimizer_success,
        optimizer_status=optimizer_status,
    )
    accepted = bool(
        physical_root_certified and eligible_acceptance_source is not None
    )
    return {
        "accepted": accepted,
        "acceptance_source": (
            eligible_acceptance_source if accepted else None
        ),
        "optimizer_termination_eligible": (
            eligible_acceptance_source is not None
        ),
        "physical_root_certified": physical_root_certified,
        "finite": finite,
        "positive_active_amounts": positive_active_amounts,
        "gas": gas,
        "full_driving": full_driving,
        "gas_stationarity_max_abs": gas_stationarity_norm,
        "active_condensate_driving_max_abs": active_driving_norm,
        "inactive_condensate_violation_max_abs": inactive_violation_norm,
        "budget_scaled_max_abs": budget_norm,
        "total_density_scaled_abs": total_norm,
    }


def _zero_barrier_acceptance_source(
    *,
    optimizer_success: bool,
    optimizer_status: int | None,
) -> str | None:
    """Classify optimizer termination that may carry a certified root.

    SciPy status zero means that the function-evaluation limit was reached.
    Such a candidate may still be accepted, but only after the independent
    physical audit certifies every final KKT, positivity, and closure block.
    Negative, missing, and otherwise inconsistent failure statuses fail closed.
    """

    if optimizer_status is not None and int(optimizer_status) < 0:
        return None
    if optimizer_success:
        return "optimizer_success"
    if optimizer_status == 0:
        return "physical_kkt_after_optimizer_limit"
    return None


def _zero_barrier_local_root_eligible(
    *,
    optimizer_success: bool,
    optimizer_status: int | None,
    terminal_root_accepted: bool,
) -> bool:
    """Return whether a root can terminate or advance support search.

    Optimizer-converged local roots may seed a support transition.  A status
    zero root is only eligible after the full physical audit accepts it as a
    terminal root, so a function-limit candidate cannot add or delete phases.
    """

    source = _zero_barrier_acceptance_source(
        optimizer_success=optimizer_success,
        optimizer_status=optimizer_status,
    )
    return bool(
        source == "optimizer_success"
        or (
            source == "physical_kkt_after_optimizer_limit"
            and terminal_root_accepted
        )
    )


def _physical_audit_local_kkt_passed(
    audit: dict[str, Any],
    *,
    optimizer_success: bool,
    optimizer_status: int | None = None,
    stationarity_tolerance: float,
    budget_tolerance: float,
    total_density_tolerance: float,
) -> bool:
    """Return whether a root can terminate or advance fixed-support search."""

    return bool(
        audit["finite"]
        and _zero_barrier_local_root_eligible(
            optimizer_success=optimizer_success,
            optimizer_status=optimizer_status,
            terminal_root_accepted=bool(audit["accepted"]),
        )
        and audit["positive_active_amounts"]
        and audit["gas_stationarity_max_abs"] <= stationarity_tolerance
        and audit["active_condensate_driving_max_abs"]
        <= stationarity_tolerance
        and audit["budget_scaled_max_abs"] <= budget_tolerance
        and audit["total_density_scaled_abs"] <= total_density_tolerance
    )


def _physical_audit_root_blocks_passed(
    audit: dict[str, Any],
    *,
    optimizer_success: bool,
    stationarity_tolerance: float,
    budget_tolerance: float,
    total_density_tolerance: float,
) -> bool:
    """Return whether equality/root blocks authorize a support deletion.

    Positivity and inactive closure are intentionally excluded: a converged
    exact support root may expose a non-positive active amount that must be
    removed.  Optimizer failure or any unresolved equality block, however,
    must never be interpreted as evidence for deleting a phase.
    """

    return bool(
        audit["finite"]
        and optimizer_success
        and audit["gas_stationarity_max_abs"] <= stationarity_tolerance
        and audit["active_condensate_driving_max_abs"]
        <= stationarity_tolerance
        and audit["budget_scaled_max_abs"] <= budget_tolerance
        and audit["total_density_scaled_abs"] <= total_density_tolerance
    )


def _solve_normalized_gas_reduced_linear_support(
    *,
    gas_formula_matrix: np.ndarray,
    condensate_formula_matrix_full: np.ndarray,
    target_inventory: np.ndarray,
    gas_standard_source: np.ndarray,
    condensate_standard_source_full: np.ndarray,
    gas_log_amounts_init: np.ndarray,
    condensate_amounts_init: np.ndarray,
    total_gas_log_amount_init: float,
    element_potential_init: np.ndarray,
    support_indices: Sequence[int],
    condensate_valid_mask: np.ndarray,
    budget_scale: np.ndarray,
    stationarity_tolerance: float,
    budget_tolerance: float,
    total_density_tolerance: float,
    support_closure_tolerance: float,
    max_function_evaluations: int,
    function_evaluation_budget: _FunctionEvaluationBudget | None = None,
) -> dict[str, Any]:
    """Solve a support after analytically eliminating gas log amounts.

    Gas stationarity gives gas fractions from element potentials.  The
    remaining unknowns are element potentials, total gas, and capacity-scaled
    linear condensate amounts.  Keeping amounts linear preserves the existing
    deterministic negative-amount drop rule, while the system dimension no
    longer grows with the gas catalog.
    """

    ag = gas_formula_matrix
    ac_full = condensate_formula_matrix_full
    target = target_inventory
    gamma = gas_standard_source
    hcond_full = condensate_standard_source_full
    element_count, gas_count = ag.shape
    condensate_count = ac_full.shape[1]
    current_support = tuple(int(index) for index in support_indices)
    current_qtot = float(total_gas_log_amount_init)
    current_lambda = np.asarray(
        element_potential_init, dtype=np.float64
    ).copy()
    current_full_m = np.asarray(
        condensate_amounts_init, dtype=np.float64
    ).copy()
    dropped: list[int] = []
    attempts: list[dict[str, Any]] = []
    last_candidate: dict[str, Any] | None = None

    for _drop_round in range(len(current_support) + 1):
        active = np.asarray(current_support, dtype=np.int64)
        ac = ac_full[:, active]
        hcond = hcond_full[active]
        amount_scales = _maximum_condensate_amount_scales(ac, target)
        active_initial = np.maximum(
            current_full_m[active], 1.0e-300 * amount_scales
        )
        relative_initial = active_initial / amount_scales
        x0 = np.concatenate(
            [current_lambda, [current_qtot], relative_initial]
        )

        def unpack(values: np.ndarray):
            lambda_ = values[:element_count]
            qtot = values[element_count]
            relative_amounts = values[element_count + 1 :]
            return lambda_, qtot, amount_scales * relative_amounts

        def gas_state(values: np.ndarray):
            lambda_, qtot, amounts = unpack(values)
            logits = ag.T @ lambda_ - gamma
            normalization = float(logsumexp(logits))
            log_fractions = logits - normalization
            fractions = np.exp(log_fractions)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                gas = np.exp(qtot + log_fractions)
            return (
                lambda_,
                qtot,
                amounts,
                logits,
                normalization,
                log_fractions,
                fractions,
                gas,
            )

        def residual(values: np.ndarray) -> np.ndarray:
            (
                lambda_,
                _qtot,
                amounts,
                _logits,
                normalization,
                _log_fractions,
                _fractions,
                gas,
            ) = gas_state(values)
            # Trust-region trial points may overflow before being rejected.
            # Preserve the non-finite residual for the optimizer and final
            # physical audit, but do not leak a benign NumPy warning.
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                budget_residual = budget_scale * (
                    ag @ gas + ac @ amounts - target
                )
            return np.concatenate(
                [
                    hcond - ac.T @ lambda_,
                    np.asarray([normalization], dtype=np.float64),
                    budget_residual,
                ]
            )

        def jacobian(values: np.ndarray) -> np.ndarray:
            (
                _lambda,
                _qtot,
                _amounts,
                _logits,
                _normalization,
                _log_fractions,
                fractions,
                gas,
            ) = gas_state(values)
            support_count = len(current_support)
            variable_count = element_count + 1 + support_count
            matrix = np.zeros(
                (variable_count, variable_count), dtype=np.float64
            )
            if support_count:
                matrix[:support_count, :element_count] = -ac.T
            normalization_row = support_count
            mean_formula = ag @ fractions
            matrix[normalization_row, :element_count] = mean_formula
            budget_row_start = support_count + 1
            gas_inventory = ag @ gas
            gas_covariance = (
                ag @ (gas[:, None] * ag.T)
                - gas_inventory[:, None] * mean_formula[None, :]
            )
            matrix[budget_row_start:, :element_count] = (
                budget_scale[:, None] * gas_covariance
            )
            matrix[budget_row_start:, element_count] = (
                budget_scale * gas_inventory
            )
            if support_count:
                matrix[budget_row_start:, element_count + 1 :] = (
                    budget_scale[:, None]
                    * ac
                    * amount_scales[None, :]
                )
            return matrix

        call_evaluation_limit = _function_evaluation_call_limit(
            max_function_evaluations,
            function_evaluation_budget,
        )
        if call_evaluation_limit <= 0:
            attempts.append(
                {
                    "support_indices": current_support,
                    "optimizer_success": False,
                    "function_evaluations": 0,
                    "failure_reason": "function_evaluation_limit_reached",
                }
            )
            break
        # Finite-barrier potentials can dwarf the other variables for trace
        # elements.  Preserve their initializer-relative scale: capping it
        # inflates the scaled trust radius and can send total gas into an
        # underflow plateau.
        variable_scale = np.maximum(np.abs(x0), 1.0)
        try:
            optimization = _least_squares_with_scipy_overflow_guard(
                residual,
                x0,
                jac=jacobian,
                method="trf",
                x_scale=variable_scale,
                ftol=1.0e-13,
                xtol=1.0e-13,
                gtol=1.0e-13,
                max_nfev=call_evaluation_limit,
            )
        except (FloatingPointError, OverflowError, ValueError) as error:
            conservative_evaluations = 0
            if function_evaluation_budget is not None:
                function_evaluation_budget.consume(call_evaluation_limit)
                conservative_evaluations = call_evaluation_limit
            attempts.append(
                {
                    "support_indices": current_support,
                    "optimizer_success": False,
                    "function_evaluations": conservative_evaluations,
                    "function_evaluations_conservative": bool(
                        function_evaluation_budget is not None
                    ),
                    "failure_reason": f"{type(error).__name__}: {error}",
                }
            )
            break
        if function_evaluation_budget is not None:
            function_evaluation_budget.consume(int(optimization.nfev))
        (
            lambda_,
            qtot,
            active_amounts,
            _logits,
            normalization,
            log_fractions,
            _fractions,
            _gas,
        ) = gas_state(optimization.x)
        q = qtot + log_fractions
        full_m = np.zeros(condensate_count, dtype=np.float64)
        if active.size:
            full_m[active] = active_amounts
        audit = _physical_zero_barrier_audit(
            gas_formula_matrix=ag,
            condensate_formula_matrix_full=ac_full,
            target_inventory=target,
            gas_standard_source=gamma,
            condensate_standard_source_full=hcond_full,
            gas_log_amounts=q,
            condensate_amounts=full_m,
            total_gas_log_amount=qtot,
            element_potential=lambda_,
            support_indices=current_support,
            condensate_valid_mask=condensate_valid_mask,
            budget_scale=budget_scale,
            optimizer_success=bool(optimization.success),
            optimizer_status=int(optimization.status),
            stationarity_tolerance=stationarity_tolerance,
            budget_tolerance=budget_tolerance,
            total_density_tolerance=total_density_tolerance,
            support_closure_tolerance=support_closure_tolerance,
        )
        drop_authorized_by_root = _physical_audit_root_blocks_passed(
            audit,
            optimizer_success=bool(optimization.success),
            stationarity_tolerance=stationarity_tolerance,
            budget_tolerance=budget_tolerance,
            total_density_tolerance=total_density_tolerance,
        )
        attempts.append(
            {
                "support_indices": current_support,
                "optimizer_success": bool(optimization.success),
                "optimizer_status": int(optimization.status),
                "optimizer_message": str(optimization.message),
                "function_evaluations": int(optimization.nfev),
                "cost": float(optimization.cost),
                "optimality": float(optimization.optimality),
                "reduced_variable_count": int(optimization.x.size),
                "eliminated_gas_variable_count": gas_count,
                "normalization_log_residual": normalization,
                "physical_root_certified": audit[
                    "physical_root_certified"
                ],
                "acceptance_source": audit["acceptance_source"],
                "drop_authorized_by_root": drop_authorized_by_root,
                "active_condensate_amounts": tuple(
                    float(value) for value in active_amounts.tolist()
                ),
            }
        )
        last_candidate = {
            "accepted": bool(audit["accepted"]),
            "gas_log_amounts": q,
            "condensate_amounts": full_m,
            "total_gas_log_amount": float(qtot),
            "element_potential": lambda_,
            "support_indices": current_support,
            "optimizer_success": bool(optimization.success),
            "optimizer_status": int(optimization.status),
            "optimizer_message": str(optimization.message),
            "function_evaluations": int(optimization.nfev),
            "audit": audit,
        }
        nonpositive = np.flatnonzero(active_amounts <= 0.0)
        if not nonpositive.size:
            break
        if not drop_authorized_by_root:
            break
        relative_amounts = active_amounts / amount_scales
        local_drop = int(
            nonpositive[np.argmin(relative_amounts[nonpositive])]
        )
        dropped_index = current_support[local_drop]
        dropped.append(dropped_index)
        current_support = tuple(
            index for index in current_support if index != dropped_index
        )
        current_qtot = float(qtot)
        current_lambda = lambda_
        current_full_m = np.zeros(condensate_count, dtype=np.float64)
        for index, amount in zip(active.tolist(), active_amounts.tolist()):
            if index != dropped_index:
                current_full_m[index] = max(float(amount), 1.0e-300)

    accepted = bool(last_candidate and last_candidate["accepted"])
    return {
        "accepted": accepted,
        "candidate": last_candidate,
        "report": {
            "schema": (
                "exogibbs_zero_barrier_normalized_gas_reduced_linear_v1"
            ),
            "attempted": True,
            "accepted": accepted,
            "gas_variable_count": gas_count,
            "maximum_reduced_variable_count": (
                element_count + 1 + len(tuple(support_indices))
            ),
            "dropped_support_indices": tuple(dropped),
            "attempts": tuple(attempts),
        },
    }


def _solve_alternative_basic_support_portfolio(
    *,
    gas_formula_matrix: np.ndarray,
    condensate_formula_matrix_full: np.ndarray,
    target_inventory: np.ndarray,
    gas_standard_source: np.ndarray,
    condensate_standard_source_full: np.ndarray,
    gas_log_amounts_init: np.ndarray,
    condensate_amounts_init: np.ndarray,
    total_gas_log_amount_init: float,
    element_potential_init: np.ndarray,
    support_indices: Sequence[int],
    condensate_valid_mask: np.ndarray,
    budget_scale: np.ndarray,
    stationarity_tolerance: float,
    budget_tolerance: float,
    total_density_tolerance: float,
    support_closure_tolerance: float,
    max_function_evaluations: int,
    enabled: bool,
    function_evaluation_budget: _FunctionEvaluationBudget | None = None,
) -> dict[str, Any]:
    """Try rank-deficient support bases under one hard work budget."""

    report: dict[str, Any] = {
        "schema": (
            "exogibbs_zero_barrier_alternative_basic_support_portfolio_v1"
        ),
        "enabled": bool(enabled),
        "eligible": False,
        "attempted": False,
        "accepted": False,
        "local_kkt_selected": False,
        "candidate_generation": None,
        "solve_attempts": (),
        "selected_support_indices": None,
    }
    if not enabled:
        report["skip_reason"] = "disabled"
        return {
            "accepted": False,
            "selected": False,
            "candidate": None,
            "report": report,
        }

    candidates, generation_report = (
        _build_alternative_basic_support_candidates(
            condensate_formula_matrix_full=(
                condensate_formula_matrix_full
            ),
            target_inventory=target_inventory,
            condensate_amounts=condensate_amounts_init,
            support_indices=support_indices,
            budget_scale=budget_scale,
            budget_tolerance=budget_tolerance,
        )
    )
    report["eligible"] = bool(generation_report["eligible"])
    report["candidate_generation"] = generation_report
    if not candidates:
        report["skip_reason"] = generation_report.get(
            "skip_reason",
            generation_report.get(
                "failure_reason", "no_feasible_basic_support"
            ),
        )
        return {
            "accepted": False,
            "selected": False,
            "candidate": None,
            "report": report,
        }

    solve_attempts: list[dict[str, Any]] = []
    selected_candidate: dict[str, Any] | None = None
    report["attempted"] = True
    for candidate in candidates:
        if (
            function_evaluation_budget is not None
            and function_evaluation_budget.remaining <= 0
        ):
            report["stop_reason"] = "function_evaluation_limit_reached"
            break
        solve = _solve_normalized_gas_reduced_linear_support(
            gas_formula_matrix=gas_formula_matrix,
            condensate_formula_matrix_full=(
                condensate_formula_matrix_full
            ),
            target_inventory=target_inventory,
            gas_standard_source=gas_standard_source,
            condensate_standard_source_full=(
                condensate_standard_source_full
            ),
            gas_log_amounts_init=gas_log_amounts_init,
            condensate_amounts_init=candidate["condensate_amounts"],
            total_gas_log_amount_init=total_gas_log_amount_init,
            element_potential_init=element_potential_init,
            support_indices=candidate["support_indices"],
            condensate_valid_mask=condensate_valid_mask,
            budget_scale=budget_scale,
            stationarity_tolerance=stationarity_tolerance,
            budget_tolerance=budget_tolerance,
            total_density_tolerance=total_density_tolerance,
            support_closure_tolerance=support_closure_tolerance,
            max_function_evaluations=max_function_evaluations,
            function_evaluation_budget=function_evaluation_budget,
        )
        solved_candidate = solve["candidate"]
        local_kkt_passed = bool(
            solved_candidate is not None
            and _physical_audit_local_kkt_passed(
                solved_candidate["audit"],
                optimizer_success=bool(
                    solved_candidate["optimizer_success"]
                ),
                optimizer_status=int(solved_candidate["optimizer_status"]),
                stationarity_tolerance=stationarity_tolerance,
                budget_tolerance=budget_tolerance,
                total_density_tolerance=total_density_tolerance,
            )
        )
        solve_attempts.append(
            {
                "support_indices": tuple(candidate["support_indices"]),
                "accepted": bool(solve["accepted"]),
                "local_kkt_passed": local_kkt_passed,
                "solve": solve["report"],
            }
        )
        if solve["accepted"]:
            selected_candidate = solved_candidate
            report["accepted"] = True
            report["selected_support_indices"] = tuple(
                selected_candidate["support_indices"]
            )
            report["stop_reason"] = "physical_audit_accepted"
            break
        if local_kkt_passed:
            selected_candidate = solved_candidate
            report["local_kkt_selected"] = True
            report["selected_support_indices"] = tuple(
                selected_candidate["support_indices"]
            )
            report["stop_reason"] = (
                "local_kkt_selected_for_active_set_closure"
            )
            break
    else:
        report["stop_reason"] = "all_candidates_rejected"

    report["solve_attempts"] = tuple(solve_attempts)
    report["solve_attempt_count"] = len(solve_attempts)
    return {
        "accepted": bool(report["accepted"]),
        "selected": bool(selected_candidate is not None),
        "candidate": selected_candidate,
        "report": report,
    }


def _reduced_log_domain_eligibility(
    *,
    gas_formula_matrix: np.ndarray,
    condensate_formula_matrix_full: np.ndarray,
    target_inventory: np.ndarray,
    support_indices: Sequence[int],
) -> tuple[bool, str]:
    """Return whether all budget rows admit a positive log formulation."""

    if np.any(target_inventory <= 0.0):
        return False, "nonpositive_target_row"
    if np.any(gas_formula_matrix < 0.0) or np.any(
        condensate_formula_matrix_full < 0.0
    ):
        return False, "signed_stoichiometry_row"
    active = np.asarray(tuple(support_indices), dtype=np.int64)
    active_condensates = condensate_formula_matrix_full[:, active]
    available = np.any(gas_formula_matrix > 0.0, axis=1)
    if active.size:
        available = available | np.any(active_condensates > 0.0, axis=1)
        if np.any(~np.any(active_condensates > 0.0, axis=0)):
            return False, "active_phase_without_positive_stoichiometry"
    if np.any(~available):
        return False, "positive_target_without_active_source"
    return True, "eligible"


def _solve_reduced_log_domain_active_support(
    *,
    gas_formula_matrix: np.ndarray,
    condensate_formula_matrix_full: np.ndarray,
    target_inventory: np.ndarray,
    gas_standard_source: np.ndarray,
    condensate_standard_source_full: np.ndarray,
    gas_log_amounts_init: np.ndarray,
    condensate_amounts_init: np.ndarray,
    total_gas_log_amount_init: float,
    element_potential_init: np.ndarray,
    support_indices: Sequence[int],
    condensate_valid_mask: np.ndarray,
    budget_scale: np.ndarray,
    stationarity_tolerance: float,
    budget_tolerance: float,
    total_density_tolerance: float,
    support_closure_tolerance: float,
    max_function_evaluations: int,
    allow_greedy_drop: bool = True,
    function_evaluation_budget: _FunctionEvaluationBudget | None = None,
) -> dict[str, Any]:
    """Solve positive nonnegative-stoichiometry budgets in log space."""

    ag = gas_formula_matrix
    ac_full = condensate_formula_matrix_full
    target = target_inventory
    gamma = gas_standard_source
    hcond_full = condensate_standard_source_full
    element_count = ag.shape[0]
    condensate_count = ac_full.shape[1]
    inventory_total = float(np.sum(target))
    log_inventory_total = float(np.log(inventory_total))
    log_beta = np.log(target) - log_inventory_total
    relative_amount_floor = max(
        64.0 * np.finfo(np.float64).eps,
        min(1.0e-2, 0.1 * max(float(budget_tolerance), 0.0)),
    )
    log_relative_amount_floor = float(np.log(relative_amount_floor))
    lower_bound_tolerance = max(
        1.0e-8,
        1.0e-6 * abs(log_relative_amount_floor),
    )
    current_support = tuple(int(index) for index in support_indices)
    current_qtot = float(total_gas_log_amount_init)
    current_lambda = np.asarray(
        element_potential_init, dtype=np.float64
    ).copy()
    current_full_m = np.asarray(
        condensate_amounts_init, dtype=np.float64
    ).copy()
    dropped: list[int] = []
    attempts: list[dict[str, Any]] = []
    last_candidate: dict[str, Any] | None = None

    for _drop_round in range(len(current_support) + 1):
        eligible, reason = _reduced_log_domain_eligibility(
            gas_formula_matrix=ag,
            condensate_formula_matrix_full=ac_full,
            target_inventory=target,
            support_indices=current_support,
        )
        if not eligible:
            attempts.append(
                {
                    "support_indices": current_support,
                    "optimizer_success": False,
                    "failure_reason": reason,
                }
            )
            break
        active = np.asarray(current_support, dtype=np.int64)
        ac = ac_full[:, active]
        hcond = hcond_full[active]
        log_kappa_values = []
        for column in ac.T:
            consuming = column > 0.0
            log_kappa_values.append(
                float(
                    np.min(
                        log_beta[consuming] - np.log(column[consuming])
                    )
                )
            )
        log_kappa = np.asarray(log_kappa_values, dtype=np.float64)
        if active.size:
            active_initial = current_full_m[active]
            log_relative_initial = np.full(
                active.shape,
                log_relative_amount_floor,
                dtype=np.float64,
            )
            positive_initial = active_initial > 0.0
            log_relative_initial[positive_initial] = (
                np.log(active_initial[positive_initial])
                - log_inventory_total
                - log_kappa[positive_initial]
            )
            log_relative_initial = np.clip(
                log_relative_initial,
                log_relative_amount_floor,
                0.0,
            )
        else:
            log_relative_initial = np.empty((0,), dtype=np.float64)
        y_initial = current_qtot - log_inventory_total
        x0 = np.concatenate(
            [current_lambda, [y_initial], log_relative_initial]
        )

        def unpack(values: np.ndarray):
            lambda_ = values[:element_count]
            y = values[element_count]
            v = values[element_count + 1 :]
            return lambda_, y, v

        log_gas_coefficients = np.full(ag.shape, -np.inf, dtype=np.float64)
        positive_gas_coefficients = ag > 0.0
        log_gas_coefficients[positive_gas_coefficients] = np.log(
            ag[positive_gas_coefficients]
        )
        log_condensate_coefficients = np.full(
            ac.shape, -np.inf, dtype=np.float64
        )
        positive_condensate_coefficients = ac > 0.0
        if active.size:
            log_condensate_coefficients[
                positive_condensate_coefficients
            ] = (
                np.log(ac[positive_condensate_coefficients])
                + np.broadcast_to(log_kappa, ac.shape)[
                    positive_condensate_coefficients
                ]
            )

        def log_budget_state(values: np.ndarray):
            lambda_, y, v = unpack(values)
            logits = ag.T @ lambda_ - gamma
            gas_terms = log_gas_coefficients + logits[None, :] + y
            condensate_terms = log_condensate_coefficients + v[None, :]
            all_terms = np.concatenate(
                [gas_terms, condensate_terms], axis=1
            )
            log_budgets = logsumexp(all_terms, axis=1)
            return (
                lambda_,
                y,
                v,
                logits,
                gas_terms,
                condensate_terms,
                log_budgets,
            )

        def residual(values: np.ndarray) -> np.ndarray:
            (
                lambda_,
                _y,
                _v,
                logits,
                _gas_terms,
                _condensate_terms,
                log_budgets,
            ) = log_budget_state(values)
            return np.concatenate(
                [
                    hcond - ac.T @ lambda_,
                    np.asarray([logsumexp(logits)], dtype=np.float64),
                    log_budgets - log_beta,
                ]
            )

        def jacobian(values: np.ndarray) -> np.ndarray:
            (
                _lambda,
                _y,
                _v,
                logits,
                gas_terms,
                condensate_terms,
                log_budgets,
            ) = log_budget_state(values)
            support_count = len(current_support)
            variable_count = element_count + 1 + support_count
            matrix = np.zeros(
                (variable_count, variable_count), dtype=np.float64
            )
            if support_count:
                matrix[:support_count, :element_count] = -ac.T
            normalization_row = support_count
            normalized_gas = np.exp(logits - logsumexp(logits))
            matrix[normalization_row, :element_count] = (
                ag @ normalized_gas
            )
            budget_row_start = support_count + 1
            y_column = element_count
            v_column_start = element_count + 1
            gas_weights = np.exp(gas_terms - log_budgets[:, None])
            condensate_weights = np.exp(
                condensate_terms - log_budgets[:, None]
            )
            matrix[budget_row_start:, :element_count] = (
                gas_weights @ ag.T
            )
            matrix[budget_row_start:, y_column] = np.sum(
                gas_weights, axis=1
            )
            if support_count:
                matrix[
                    budget_row_start:, v_column_start:
                ] = condensate_weights
            return matrix

        lower = np.concatenate(
            [
                np.full(element_count + 1, -np.inf, dtype=np.float64),
                np.full(
                    len(current_support),
                    log_relative_amount_floor,
                    dtype=np.float64,
                ),
            ]
        )
        upper = np.concatenate(
            [
                np.full(element_count + 1, np.inf, dtype=np.float64),
                # Capacity normalizes v but is not an optimizer constraint.
                # A hard upper bound creates projected-gradient false
                # convergence when a phase consumes nearly all its capacity;
                # the independent physical budget audit remains the gate.
                np.full(len(current_support), np.inf, dtype=np.float64),
            ]
        )
        variable_scale = np.clip(np.maximum(np.abs(x0), 1.0), 1.0, 100.0)
        call_evaluation_limit = _function_evaluation_call_limit(
            max_function_evaluations,
            function_evaluation_budget,
        )
        if call_evaluation_limit <= 0:
            attempts.append(
                {
                    "support_indices": current_support,
                    "optimizer_success": False,
                    "function_evaluations": 0,
                    "failure_reason": "function_evaluation_limit_reached",
                }
            )
            break
        try:
            optimization = _least_squares_with_scipy_overflow_guard(
                residual,
                x0,
                jac=jacobian,
                bounds=(lower, upper),
                method="trf",
                x_scale=variable_scale,
                ftol=1.0e-13,
                xtol=1.0e-13,
                gtol=1.0e-13,
                max_nfev=call_evaluation_limit,
            )
        except (FloatingPointError, OverflowError, ValueError) as error:
            conservative_evaluations = 0
            if function_evaluation_budget is not None:
                function_evaluation_budget.consume(call_evaluation_limit)
                conservative_evaluations = call_evaluation_limit
            attempts.append(
                {
                    "support_indices": current_support,
                    "optimizer_success": False,
                    "function_evaluations": conservative_evaluations,
                    "function_evaluations_conservative": bool(
                        function_evaluation_budget is not None
                    ),
                    "failure_reason": f"{type(error).__name__}: {error}",
                }
            )
            break
        if function_evaluation_budget is not None:
            function_evaluation_budget.consume(int(optimization.nfev))
        lambda_, y, v = unpack(optimization.x)
        logits = ag.T @ lambda_ - gamma
        qtot = log_inventory_total + y
        q = qtot + logits
        full_m = np.zeros(condensate_count, dtype=np.float64)
        if active.size:
            log_active_amounts = (
                log_inventory_total + log_kappa + v
            )
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                full_m[active] = np.exp(log_active_amounts)
        at_lower_bound = np.flatnonzero(
            v <= log_relative_amount_floor + lower_bound_tolerance
        )
        lower_bound_support_indices = tuple(
            current_support[int(local_index)]
            for local_index in at_lower_bound
        )
        audit = _physical_zero_barrier_audit(
            gas_formula_matrix=ag,
            condensate_formula_matrix_full=ac_full,
            target_inventory=target,
            gas_standard_source=gamma,
            condensate_standard_source_full=hcond_full,
            gas_log_amounts=q,
            condensate_amounts=full_m,
            total_gas_log_amount=qtot,
            element_potential=lambda_,
            support_indices=current_support,
            condensate_valid_mask=condensate_valid_mask,
            budget_scale=budget_scale,
            optimizer_success=bool(optimization.success),
            optimizer_status=int(optimization.status),
            stationarity_tolerance=stationarity_tolerance,
            budget_tolerance=budget_tolerance,
            total_density_tolerance=total_density_tolerance,
            support_closure_tolerance=support_closure_tolerance,
        )
        drop_authorized_by_root = _physical_audit_root_blocks_passed(
            audit,
            optimizer_success=bool(optimization.success),
            stationarity_tolerance=stationarity_tolerance,
            budget_tolerance=budget_tolerance,
            total_density_tolerance=total_density_tolerance,
        )
        accepted = bool(audit["accepted"] and not at_lower_bound.size)
        log_residual = residual(optimization.x)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            relative_phase_amounts = np.exp(v)
        attempt = {
            "support_indices": current_support,
            "optimizer_success": bool(optimization.success),
            "optimizer_status": int(optimization.status),
            "optimizer_message": str(optimization.message),
            "function_evaluations": int(optimization.nfev),
            "cost": float(optimization.cost),
            "optimality": float(optimization.optimality),
            "log_domain_residual_max_abs": float(
                np.max(np.abs(log_residual), initial=0.0)
            ),
            "relative_phase_amounts": tuple(
                float(value) for value in relative_phase_amounts.tolist()
            ),
            "active_phase_at_lower_bound": bool(at_lower_bound.size),
            "lower_bound_support_indices": lower_bound_support_indices,
            "physical_audit_accepted": bool(audit["accepted"]),
            "physical_root_certified": audit["physical_root_certified"],
            "acceptance_source": audit["acceptance_source"],
            "drop_authorized_by_root": drop_authorized_by_root,
            "physical_budget_scaled_max_abs": audit[
                "budget_scaled_max_abs"
            ],
        }
        attempts.append(attempt)
        last_candidate = {
            "accepted": accepted,
            "gas_log_amounts": q,
            "condensate_amounts": full_m,
            "total_gas_log_amount": float(qtot),
            "element_potential": lambda_,
            "support_indices": current_support,
            "optimizer_success": bool(optimization.success),
            "optimizer_status": int(optimization.status),
            "optimizer_message": str(optimization.message),
            "function_evaluations": int(optimization.nfev),
            "active_phase_at_lower_bound": bool(at_lower_bound.size),
            "lower_bound_support_indices": lower_bound_support_indices,
            "audit": audit,
        }
        if accepted:
            break
        if not allow_greedy_drop or not current_support:
            break
        if not at_lower_bound.size:
            break
        if not drop_authorized_by_root:
            break
        local_drop = int(at_lower_bound[np.argmin(v[at_lower_bound])])
        dropped_index = current_support[local_drop]
        dropped.append(dropped_index)
        current_support = tuple(
            index for index in current_support if index != dropped_index
        )
        current_qtot = float(qtot)
        current_lambda = lambda_
        current_full_m = full_m

    accepted = bool(last_candidate and last_candidate["accepted"])
    return {
        "accepted": accepted,
        "candidate": last_candidate,
        "report": {
            "schema": "exogibbs_zero_barrier_reduced_log_domain_v1",
            "eligible": True,
            "attempted": True,
            "accepted": accepted,
            "inventory_normalization": inventory_total,
            "relative_amount_floor": relative_amount_floor,
            "greedy_drop_enabled": bool(allow_greedy_drop),
            "dropped_support_indices": tuple(dropped),
            "attempts": tuple(attempts),
        },
    }


def _solve_structural_zero_reduced_log_domain_active_support(
    *,
    gas_formula_matrix: np.ndarray,
    condensate_formula_matrix_full: np.ndarray,
    target_inventory: np.ndarray,
    gas_standard_source: np.ndarray,
    condensate_standard_source_full: np.ndarray,
    gas_log_amounts_init: np.ndarray,
    condensate_amounts_init: np.ndarray,
    total_gas_log_amount_init: float,
    element_potential_init: np.ndarray,
    support_indices: Sequence[int],
    condensate_valid_mask: np.ndarray,
    budget_scale: np.ndarray,
    stationarity_tolerance: float,
    budget_tolerance: float,
    total_density_tolerance: float,
    support_closure_tolerance: float,
    max_function_evaluations: int,
    function_evaluation_budget: _FunctionEvaluationBudget | None = None,
) -> dict[str, Any]:
    """Run the normalized reduced solve after removing zero rows.

    With nonnegative stoichiometry, an exactly zero inventory row forces every
    species using that row to zero.  Those gases and rows can be omitted from
    the positive log solve.  Their potentials are then reconstructed so their
    total contribution is safely below the unchanged full physical audit.
    """

    ag = gas_formula_matrix
    ac_full = condensate_formula_matrix_full
    target = target_inventory
    gamma = gas_standard_source
    initial_support = tuple(int(index) for index in support_indices)
    positive_rows = target > 0.0
    zero_rows = target == 0.0
    forced_dropped = tuple(
        index
        for index in initial_support
        if np.any(ac_full[zero_rows, index] > 0.0)
    )
    support = tuple(
        index for index in initial_support if index not in set(forced_dropped)
    )
    amounts_for_solve = np.asarray(
        condensate_amounts_init, dtype=np.float64
    ).copy()
    if forced_dropped:
        amounts_for_solve[
            np.asarray(forced_dropped, dtype=np.int64)
        ] = 0.0
    base_report: dict[str, Any] = {
        "schema": (
            "exogibbs_zero_barrier_structural_zero_log_rescue_v1"
        ),
        "eligible": False,
        "attempted": False,
        "accepted": False,
        "zero_target_rows": tuple(
            int(index) for index in np.flatnonzero(zero_rows).tolist()
        ),
        "initial_support_indices": initial_support,
        "structural_zero_dropped_support_indices": forced_dropped,
    }
    if np.any(target < 0.0):
        base_report["skip_reason"] = "negative_target_row"
        return {"accepted": False, "candidate": None, "report": base_report}
    if np.any(ag < 0.0) or np.any(ac_full < 0.0):
        base_report["skip_reason"] = "signed_stoichiometry_row"
        return {"accepted": False, "candidate": None, "report": base_report}
    if not np.any(positive_rows):
        base_report["skip_reason"] = "no_positive_target_row"
        return {"accepted": False, "candidate": None, "report": base_report}
    suppressed_gases = (
        np.any(ag[zero_rows] > 0.0, axis=0)
        if np.any(zero_rows)
        else np.zeros(ag.shape[1], dtype=bool)
    )
    retained_gases = ~suppressed_gases
    if not np.any(retained_gases):
        base_report["skip_reason"] = "no_gas_without_zero_target_element"
        return {"accepted": False, "candidate": None, "report": base_report}
    eligible, reason = _reduced_log_domain_eligibility(
        gas_formula_matrix=ag[positive_rows][:, retained_gases],
        condensate_formula_matrix_full=ac_full[positive_rows],
        target_inventory=target[positive_rows],
        support_indices=support,
    )
    base_report.update(
        {
            "eligible": eligible,
            "retained_gas_count": int(np.count_nonzero(retained_gases)),
            "suppressed_gas_indices": tuple(
                int(index)
                for index in np.flatnonzero(suppressed_gases).tolist()
            ),
        }
    )
    if not eligible:
        base_report["skip_reason"] = reason
        return {"accepted": False, "candidate": None, "report": base_report}

    base_report["attempted"] = True
    reduced = _solve_normalized_gas_reduced_linear_support(
        gas_formula_matrix=ag[positive_rows][:, retained_gases],
        condensate_formula_matrix_full=ac_full[positive_rows],
        target_inventory=target[positive_rows],
        gas_standard_source=gamma[retained_gases],
        condensate_standard_source_full=condensate_standard_source_full,
        gas_log_amounts_init=np.asarray(gas_log_amounts_init)[retained_gases],
        condensate_amounts_init=amounts_for_solve,
        total_gas_log_amount_init=total_gas_log_amount_init,
        element_potential_init=np.asarray(element_potential_init)[positive_rows],
        support_indices=support,
        condensate_valid_mask=condensate_valid_mask,
        budget_scale=budget_scale[positive_rows],
        stationarity_tolerance=stationarity_tolerance,
        budget_tolerance=budget_tolerance,
        total_density_tolerance=total_density_tolerance,
        support_closure_tolerance=support_closure_tolerance,
        max_function_evaluations=max_function_evaluations,
        function_evaluation_budget=function_evaluation_budget,
    )
    candidate = reduced["candidate"]
    normalized_local_kkt_passed = bool(
        candidate is not None
        and _physical_audit_local_kkt_passed(
            candidate["audit"],
            optimizer_success=bool(candidate["optimizer_success"]),
            optimizer_status=int(candidate["optimizer_status"]),
            stationarity_tolerance=stationarity_tolerance,
            budget_tolerance=budget_tolerance,
            total_density_tolerance=total_density_tolerance,
        )
    )
    base_report["normalized_linear_local_kkt_passed"] = (
        normalized_local_kkt_passed
    )
    if normalized_local_kkt_passed:
        base_report["inner_formulation"] = (
            "normalized_gas_reduced_linear_amounts"
        )
    else:
        base_report["normalized_linear_solve"] = reduced["report"]
        reduced = _solve_reduced_log_domain_active_support(
            gas_formula_matrix=ag[positive_rows][:, retained_gases],
            condensate_formula_matrix_full=ac_full[positive_rows],
            target_inventory=target[positive_rows],
            gas_standard_source=gamma[retained_gases],
            condensate_standard_source_full=(
                condensate_standard_source_full
            ),
            gas_log_amounts_init=np.asarray(gas_log_amounts_init)[
                retained_gases
            ],
            condensate_amounts_init=amounts_for_solve,
            total_gas_log_amount_init=total_gas_log_amount_init,
            element_potential_init=np.asarray(element_potential_init)[
                positive_rows
            ],
            support_indices=support,
            condensate_valid_mask=condensate_valid_mask,
            budget_scale=budget_scale[positive_rows],
            stationarity_tolerance=stationarity_tolerance,
            budget_tolerance=budget_tolerance,
            total_density_tolerance=total_density_tolerance,
            support_closure_tolerance=support_closure_tolerance,
            max_function_evaluations=max_function_evaluations,
            allow_greedy_drop=True,
            function_evaluation_budget=function_evaluation_budget,
        )
        base_report["inner_formulation"] = "reduced_log_domain"
    base_report["solve"] = reduced["report"]
    candidate = reduced["candidate"]
    if candidate is None:
        base_report["failure_reason"] = "reduced_solver_no_candidate"
        return {"accepted": False, "candidate": None, "report": base_report}

    full_lambda = np.zeros_like(target, dtype=np.float64)
    full_lambda[positive_rows] = np.asarray(
        candidate["element_potential"], dtype=np.float64
    )
    qtot = float(candidate["total_gas_log_amount"])
    full_amounts = np.asarray(
        candidate["condensate_amounts"], dtype=np.float64
    )
    full_support = tuple(candidate["support_indices"])
    log_fraction_cap: float | None = None
    zero_potential: float | None = None
    inactive_phase_limits: tuple[tuple[int, float], ...] = ()
    zero_potential_limits = [0.0]
    if np.any(zero_rows) and np.any(suppressed_gases):
        if budget_tolerance <= 0.0 or total_density_tolerance <= 0.0:
            base_report["failure_reason"] = "zero_tolerance_reconstruction"
            return {"accepted": False, "candidate": None, "report": base_report}
        suppressed_count = int(np.count_nonzero(suppressed_gases))
        coefficient_sums = np.sum(ag[:, suppressed_gases], axis=1)
        log_fraction_limits = [
            float(np.log(0.01 * total_density_tolerance / suppressed_count))
        ]
        for row in np.flatnonzero(coefficient_sums > 0.0):
            amount_limit = 0.01 * budget_tolerance / budget_scale[row]
            if not np.isfinite(amount_limit) or amount_limit <= 0.0:
                base_report["failure_reason"] = (
                    "invalid_structural_zero_amount_limit"
                )
                return {
                    "accepted": False,
                    "candidate": None,
                    "report": base_report,
                }
            log_fraction_limits.append(
                float(
                    np.log(amount_limit)
                    - qtot
                    - np.log(coefficient_sums[row])
                )
            )
        log_fraction_cap = min(-50.0, *log_fraction_limits)
        base_logits = (
            ag[positive_rows][:, suppressed_gases].T
            @ full_lambda[positive_rows]
            - gamma[suppressed_gases]
        )
        zero_coefficients = np.sum(
            ag[zero_rows][:, suppressed_gases], axis=0
        )
        zero_potential_limits.extend(
            float(value)
            for value in (
                (log_fraction_cap - base_logits) / zero_coefficients
            ).tolist()
        )

    if np.any(zero_rows):
        support_mask = np.zeros(ac_full.shape[1], dtype=bool)
        if full_support:
            support_mask[np.asarray(full_support, dtype=np.int64)] = True
        zero_condensate_coefficients = np.sum(ac_full[zero_rows], axis=0)
        inactive_zero_phases = np.flatnonzero(
            condensate_valid_mask
            & ~support_mask
            & (zero_condensate_coefficients > 0.0)
        )
        base_driving = (
            condensate_standard_source_full
            - ac_full[positive_rows].T @ full_lambda[positive_rows]
        )
        inactive_phase_limits = tuple(
            (
                int(index),
                float(
                    (
                        base_driving[index]
                        - support_closure_tolerance
                    )
                    / zero_condensate_coefficients[index]
                ),
            )
            for index in inactive_zero_phases.tolist()
        )
        zero_potential_limits.extend(
            limit for _index, limit in inactive_phase_limits
        )
        zero_potential = float(min(zero_potential_limits))
        full_lambda[zero_rows] = zero_potential

    full_q = qtot + ag.T @ full_lambda - gamma
    audit = _physical_zero_barrier_audit(
        gas_formula_matrix=ag,
        condensate_formula_matrix_full=ac_full,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=condensate_standard_source_full,
        gas_log_amounts=full_q,
        condensate_amounts=full_amounts,
        total_gas_log_amount=qtot,
        element_potential=full_lambda,
        support_indices=full_support,
        condensate_valid_mask=condensate_valid_mask,
        budget_scale=budget_scale,
        optimizer_success=bool(candidate["optimizer_success"]),
        optimizer_status=int(candidate["optimizer_status"]),
        stationarity_tolerance=stationarity_tolerance,
        budget_tolerance=budget_tolerance,
        total_density_tolerance=total_density_tolerance,
        support_closure_tolerance=support_closure_tolerance,
    )
    active_phase_at_lower_bound = bool(
        candidate.get("active_phase_at_lower_bound", False)
    )
    accepted = bool(
        audit["accepted"] and not active_phase_at_lower_bound
    )
    base_report.update(
        {
            "accepted": accepted,
            "active_phase_at_lower_bound": active_phase_at_lower_bound,
            "lower_bound_support_indices": tuple(
                candidate.get("lower_bound_support_indices", ())
            ),
            "reconstruction_log_gas_fraction_cap": log_fraction_cap,
            "reconstructed_zero_row_potential": zero_potential,
            "inactive_zero_row_phase_potential_limits": (
                inactive_phase_limits
            ),
            "full_physical_audit": {
                key: audit[key]
                for key in (
                    "accepted",
                    "finite",
                    "positive_active_amounts",
                    "gas_stationarity_max_abs",
                    "active_condensate_driving_max_abs",
                    "inactive_condensate_violation_max_abs",
                    "budget_scaled_max_abs",
                    "total_density_scaled_abs",
                )
            },
        }
    )
    full_candidate = dict(candidate)
    full_candidate.update(
        {
            "accepted": accepted,
            "gas_log_amounts": full_q,
            "condensate_amounts": full_amounts,
            "total_gas_log_amount": qtot,
            "element_potential": full_lambda,
            "support_indices": full_support,
            "audit": audit,
        }
    )
    return {
        "accepted": accepted,
        "candidate": full_candidate,
        "report": base_report,
    }


def _solve_reduced_log_domain_support_branches(
    *,
    gas_formula_matrix: np.ndarray,
    condensate_formula_matrix_full: np.ndarray,
    target_inventory: np.ndarray,
    gas_standard_source: np.ndarray,
    condensate_standard_source_full: np.ndarray,
    gas_log_amounts_init: np.ndarray,
    condensate_amounts_init: np.ndarray,
    total_gas_log_amount_init: float,
    element_potential_init: np.ndarray,
    support_indices: Sequence[int],
    condensate_valid_mask: np.ndarray,
    budget_scale: np.ndarray,
    stationarity_tolerance: float,
    budget_tolerance: float,
    total_density_tolerance: float,
    support_closure_tolerance: float,
    max_function_evaluations: int,
    max_support_nodes: int = _REDUCED_SUPPORT_NODE_LIMIT,
    function_evaluation_budget: _FunctionEvaluationBudget | None = None,
) -> dict[str, Any]:
    """Explore bounded leave-one-out supports with exact physical closure.

    A simultaneous solve with two incompatible phases can return negative
    linear amounts for both.  Greedily dropping the most negative phase can
    remove the physically required phase.  This deterministic breadth-first
    search retains the original support as an initializer hypothesis, then
    explores every one-phase removal up to a fixed node budget.  Removed
    phases are inactive and therefore never receive stationarity equations.
    """

    initial_support = tuple(int(index) for index in support_indices)
    queue: list[tuple[int, ...]] = [initial_support]
    queued = {initial_support}
    visited: set[tuple[int, ...]] = set()
    node_reports: list[dict[str, Any]] = []
    accepted_result: dict[str, Any] | None = None

    evaluation_limit_reached = False
    while queue and len(visited) < int(max_support_nodes):
        if (
            function_evaluation_budget is not None
            and function_evaluation_budget.remaining <= 0
        ):
            evaluation_limit_reached = True
            break
        support = queue.pop(0)
        if support in visited:
            continue
        visited.add(support)
        result = _solve_reduced_log_domain_active_support(
            gas_formula_matrix=gas_formula_matrix,
            condensate_formula_matrix_full=(
                condensate_formula_matrix_full
            ),
            target_inventory=target_inventory,
            gas_standard_source=gas_standard_source,
            condensate_standard_source_full=(
                condensate_standard_source_full
            ),
            gas_log_amounts_init=gas_log_amounts_init,
            condensate_amounts_init=condensate_amounts_init,
            total_gas_log_amount_init=total_gas_log_amount_init,
            element_potential_init=element_potential_init,
            support_indices=support,
            condensate_valid_mask=condensate_valid_mask,
            budget_scale=budget_scale,
            stationarity_tolerance=stationarity_tolerance,
            budget_tolerance=budget_tolerance,
            total_density_tolerance=total_density_tolerance,
            support_closure_tolerance=support_closure_tolerance,
            max_function_evaluations=max_function_evaluations,
            allow_greedy_drop=False,
            function_evaluation_budget=function_evaluation_budget,
        )
        node_reports.append(
            {
                "support_indices": support,
                "accepted": bool(result["accepted"]),
                "solve": result["report"],
            }
        )
        if result["accepted"]:
            accepted_result = result
            break
        for local_index in range(len(support)):
            child = support[:local_index] + support[local_index + 1 :]
            if child not in visited and child not in queued:
                queue.append(child)
                queued.add(child)

    accepted = bool(accepted_result and accepted_result["accepted"])
    return {
        "accepted": accepted,
        "candidate": (
            None if accepted_result is None else accepted_result["candidate"]
        ),
        "report": {
            "schema": (
                "exogibbs_zero_barrier_reduced_support_search_v1"
            ),
            "eligible": True,
            "attempted": True,
            "accepted": accepted,
            "initial_support_indices": initial_support,
            "accepted_support_indices": (
                None
                if accepted_result is None
                else accepted_result["candidate"]["support_indices"]
            ),
            "node_limit": int(max_support_nodes),
            "visited_node_count": len(visited),
            "node_limit_reached": bool(
                queue and len(visited) >= int(max_support_nodes)
            ),
            "function_evaluation_limit_reached": (
                evaluation_limit_reached
            ),
            "visited_supports": tuple(
                node["support_indices"] for node in node_reports
            ),
            "nodes": tuple(node_reports),
        },
    }


def _gas_elemental_capacities(
    gas_formula_matrix: np.ndarray,
    target_inventory: np.ndarray,
) -> np.ndarray:
    """Return maximum gas amounts allowed by every consumed element."""

    capacities = np.zeros(gas_formula_matrix.shape[1], dtype=np.float64)
    for index, column in enumerate(gas_formula_matrix.T):
        consuming = column > 0.0
        if not np.any(consuming):
            continue
        inventories = target_inventory[consuming]
        if np.any(inventories <= 0.0):
            continue
        candidates = inventories / column[consuming]
        positive = candidates[
            np.isfinite(candidates) & (candidates > 0.0)
        ]
        if positive.size == candidates.size:
            capacities[index] = float(np.min(positive))
    return capacities


def _pivot_rank_one_support_addition(
    *,
    condensate_formula_matrix_full: np.ndarray,
    condensate_standard_source_full: np.ndarray,
    target_inventory: np.ndarray,
    condensate_amounts: np.ndarray,
    support_indices: Sequence[int],
    added_support_index: int,
    budget_scale: np.ndarray,
) -> dict[str, Any]:
    """Pivot one favorable phase into a rank-one dependent support.

    The null direction is formed in capacity-relative amount coordinates, so
    a uniform change of the inventory gauge leaves the pivot unchanged.  This
    is only an initializer transformation: rank, descent, nonnegativity,
    unique-limiter, and scaled inventory-preservation guards must all pass.
    """

    ac_full = np.asarray(
        condensate_formula_matrix_full, dtype=np.float64
    )
    hcond_full = np.asarray(
        condensate_standard_source_full, dtype=np.float64
    )
    target = np.asarray(target_inventory, dtype=np.float64)
    full_amounts = np.asarray(condensate_amounts, dtype=np.float64)
    base_support = tuple(int(index) for index in support_indices)
    added_index = int(added_support_index)
    extended_support = base_support + (added_index,)
    report: dict[str, Any] = {
        "schema": "exogibbs_zero_barrier_rank_one_simplex_pivot_v1",
        "attempted": False,
        "applied": False,
        "coordinate_system": "capacity_relative_condensate_amount",
        "base_support_indices": base_support,
        "added_support_index": added_index,
        "extended_support_indices": extended_support,
        "inventory_residual_tolerance": (
            _SIMPLEX_PIVOT_INVENTORY_RESIDUAL_TOLERANCE
        ),
    }

    def failed(reason: str) -> dict[str, Any]:
        report["failure_reason"] = reason
        return {
            "applied": False,
            "support_indices": extended_support,
            "condensate_amounts": full_amounts.copy(),
            "report": report,
        }

    if (
        not base_support
        or added_index in set(base_support)
        or np.any(target < 0.0)
        or np.any(ac_full < 0.0)
    ):
        return failed("ineligible_structure")
    positive_rows = target > 0.0
    zero_rows = target == 0.0
    if (
        not np.any(positive_rows)
        or (
            np.any(zero_rows)
            and np.any(ac_full[zero_rows, added_index] > 0.0)
        )
    ):
        return failed("structurally_impossible_added_phase")

    extended = np.asarray(extended_support, dtype=np.int64)
    ac = ac_full[positive_rows][:, extended]
    target_positive = target[positive_rows]
    amount_scales = _maximum_condensate_amount_scales(
        ac, target_positive
    )
    if (
        np.any(~np.any(ac > 0.0, axis=0))
        or not np.all(np.isfinite(amount_scales))
        or np.any(amount_scales <= 0.0)
    ):
        return failed("invalid_capacity_scale")
    base_amounts = full_amounts[np.asarray(base_support, dtype=np.int64)]
    if not np.all(np.isfinite(base_amounts)) or np.any(base_amounts <= 0.0):
        return failed("nonpositive_base_amount")

    scaled_matrix = (
        np.asarray(budget_scale, dtype=np.float64)[positive_rows, None]
        * ac
        * amount_scales[None, :]
    )
    base_rank = int(np.linalg.matrix_rank(scaled_matrix[:, :-1]))
    extended_rank = int(np.linalg.matrix_rank(scaled_matrix))
    report.update(
        {
            "attempted": True,
            "base_rank": base_rank,
            "extended_rank": extended_rank,
            "extended_nullity": len(extended_support) - extended_rank,
        }
    )
    if (
        base_rank != len(base_support)
        or extended_rank != len(base_support)
    ):
        return failed("not_one_new_rank_deficiency")

    try:
        _left, _singular_values, right_vectors = np.linalg.svd(
            scaled_matrix, full_matrices=True
        )
    except np.linalg.LinAlgError:
        return failed("nullspace_factorization_failed")
    relative_direction = np.asarray(
        right_vectors[-1], dtype=np.float64
    )
    direction_scale = float(
        np.max(np.abs(relative_direction), initial=0.0)
    )
    direction_tolerance = 64.0 * np.finfo(np.float64).eps * max(
        direction_scale, 1.0
    )
    if abs(float(relative_direction[-1])) <= direction_tolerance:
        return failed("added_phase_has_no_null_direction")
    relative_direction *= np.sign(relative_direction[-1])
    decreasing = np.flatnonzero(
        relative_direction[:-1] < -direction_tolerance
    )
    if not decreasing.size:
        return failed("no_leaving_phase")

    amount_direction = amount_scales * relative_direction
    objective_terms = hcond_full[extended] * amount_direction
    objective_direction = float(np.sum(objective_terms))
    objective_scale = max(float(np.sum(np.abs(objective_terms))), 1.0e-300)
    objective_tolerance = 64.0 * np.finfo(np.float64).eps * objective_scale
    report["objective_direction"] = objective_direction
    report["objective_direction_tolerance"] = objective_tolerance
    if objective_direction >= -objective_tolerance:
        return failed("nondecreasing_pivot_direction")

    relative_before = np.concatenate(
        [base_amounts / amount_scales[:-1], np.asarray([0.0])]
    )
    limiting_steps = (
        relative_before[decreasing]
        / -relative_direction[decreasing]
    )
    if (
        not np.all(np.isfinite(limiting_steps))
        or np.any(limiting_steps <= 0.0)
    ):
        return failed("invalid_limiting_step")
    step = float(np.min(limiting_steps))
    tie_tolerance = 64.0 * np.finfo(np.float64).eps * max(step, 1.0)
    limiting = decreasing[
        np.abs(limiting_steps - step) <= tie_tolerance
    ]
    if limiting.size != 1:
        report["limiting_local_indices"] = tuple(
            int(index) for index in limiting.tolist()
        )
        return failed("nonunique_limiting_phase")
    leaving_local = int(limiting[0])
    leaving_index = base_support[leaving_local]
    relative_after = relative_before + step * relative_direction
    relative_tolerance = 256.0 * np.finfo(np.float64).eps * max(
        float(np.max(np.abs(relative_after), initial=0.0)), 1.0
    )
    relative_after[leaving_local] = 0.0
    if (
        np.any(relative_after < -relative_tolerance)
        or np.any(
            np.delete(relative_after, leaving_local)
            <= relative_tolerance
        )
    ):
        return failed("pivot_amount_nonpositivity")
    relative_after = np.maximum(relative_after, 0.0)
    extended_amounts_after = amount_scales * relative_after
    candidate_amounts = full_amounts.copy()
    candidate_amounts[extended] = extended_amounts_after
    candidate_amounts[leaving_index] = 0.0
    candidate_support = tuple(
        index for index in extended_support if index != leaving_index
    )
    inventory_residual = (
        ac_full
        @ (candidate_amounts - full_amounts)
    )
    scaled_inventory_residual = (
        np.asarray(budget_scale, dtype=np.float64) * inventory_residual
    )
    residual_norm = float(
        np.max(np.abs(scaled_inventory_residual), initial=0.0)
    )
    report.update(
        {
            "leaving_support_index": leaving_index,
            "pivot_step": step,
            "scaled_inventory_residual_max_abs": residual_norm,
            "candidate_support_indices": candidate_support,
        }
    )
    if (
        not np.all(np.isfinite(candidate_amounts))
        or residual_norm
        > _SIMPLEX_PIVOT_INVENTORY_RESIDUAL_TOLERANCE
    ):
        return failed("inventory_preservation_failed")
    report["applied"] = True
    return {
        "applied": True,
        "support_indices": candidate_support,
        "condensate_amounts": candidate_amounts,
        "report": report,
    }


def _capacity_regularized_initializer(
    *,
    gas_formula_matrix: np.ndarray,
    target_inventory: np.ndarray,
    gas_standard_source: np.ndarray,
    gas_log_amounts: np.ndarray,
    total_gas_log_amount: float,
    element_potential: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, dict[str, Any]]:
    """Move numerically absent gases into a scale-aware local basin.

    A finite-barrier endpoint can contain log amounts with magnitude far
    beyond the exponential underflow threshold when an active phase consumes
    a trace element.  Such a state is a valid source of support information,
    but it is not a useful local initializer for the physical zero-barrier
    equations.  The floor used here is a small fraction of the maximum amount
    permitted by every element in a gas species.  It changes only the host
    optimizer's initial point; the final acceptance audit remains unchanged.
    """

    q_before = np.asarray(gas_log_amounts, dtype=np.float64)
    lambda_before = np.asarray(element_potential, dtype=np.float64)
    capacities = _gas_elemental_capacities(
        gas_formula_matrix,
        target_inventory,
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        amount_before = np.exp(q_before)
        amount_floors = capacities * _INITIALIZER_CAPACITY_FRACTION
    usable_floor = (
        np.isfinite(amount_floors)
        & (amount_floors > 0.0)
    )
    regularized_mask = usable_floor & (amount_before < amount_floors)
    q_after = q_before.copy()
    if np.any(regularized_mask):
        q_after[regularized_mask] = np.log(
            amount_floors[regularized_mask]
        )
        qtot_after = float(np.logaddexp.reduce(q_after))
        fit_mask = capacities > 0.0
        fit_matrix = gas_formula_matrix[:, fit_mask].T
        fit_target = (
            q_after[fit_mask]
            + gas_standard_source[fit_mask]
            - qtot_after
        )
        try:
            fitted_potential, _residuals, fit_rank, _singular_values = (
                np.linalg.lstsq(
                    fit_matrix,
                    fit_target,
                    rcond=None,
                )
            )
            if fit_rank == gas_formula_matrix.shape[0]:
                lambda_candidate = fitted_potential
                fit_strategy = "full_rank_direct_least_squares"
            else:
                # Retain the supplied potential only in directions that the
                # positive-capacity gases cannot identify.
                correction_target = (
                    fit_target - fit_matrix @ lambda_before
                )
                correction = np.linalg.lstsq(
                    fit_matrix,
                    correction_target,
                    rcond=None,
                )[0]
                lambda_candidate = lambda_before + correction
                fit_strategy = "rank_deficient_anchored_least_squares"
            potential_recomputed = bool(
                np.all(np.isfinite(lambda_candidate))
            )
        except np.linalg.LinAlgError:
            fit_rank = 0
            lambda_candidate = lambda_before
            potential_recomputed = False
            fit_strategy = "least_squares_failed_preserved_input"
        lambda_after = (
            lambda_candidate
            if potential_recomputed
            else lambda_before.copy()
        )
        fit_residual = (
            fit_matrix @ lambda_after - fit_target
        )
        fit_residual_max_abs = float(
            np.max(np.abs(fit_residual), initial=0.0)
        )
        fit_row_count = int(fit_matrix.shape[0])
    else:
        qtot_after = float(total_gas_log_amount)
        lambda_after = lambda_before.copy()
        fit_rank = 0
        fit_row_count = 0
        fit_residual_max_abs = 0.0
        potential_recomputed = False
        fit_strategy = "not_needed"

    report = {
        "schema": (
            "exogibbs_zero_barrier_capacity_regularized_initializer_v1"
        ),
        "applied": bool(np.any(regularized_mask)),
        "capacity_fraction": _INITIALIZER_CAPACITY_FRACTION,
        "regularized_gas_count": int(np.count_nonzero(regularized_mask)),
        "regularized_gas_mask": tuple(
            bool(value) for value in regularized_mask.tolist()
        ),
        "gas_elemental_capacities": tuple(
            float(value) for value in capacities.tolist()
        ),
        "gas_amount_floors": tuple(
            float(value) for value in amount_floors.tolist()
        ),
        "gas_log_amounts_before": tuple(
            float(value) for value in q_before.tolist()
        ),
        "gas_log_amounts_after": tuple(
            float(value) for value in q_after.tolist()
        ),
        "total_gas_log_amount_before": float(total_gas_log_amount),
        "total_gas_log_amount_after": qtot_after,
        "element_potential_recomputed": potential_recomputed,
        "element_potential_fit_row_count": fit_row_count,
        "element_potential_fit_rank": int(fit_rank),
        "element_potential_fit_strategy": fit_strategy,
        "element_potential_fit_residual_max_abs": (
            fit_residual_max_abs
        ),
        "element_potential_before": tuple(
            float(value) for value in lambda_before.tolist()
        ),
        "element_potential_after": tuple(
            float(value) for value in lambda_after.tolist()
        ),
    }
    return q_after, qtot_after, lambda_after, report


def _polish_zero_barrier_support_once(
    *,
    gas_formula_matrix: Any,
    condensate_formula_matrix_full: Any,
    target_inventory: Any,
    gas_standard_source: Any,
    condensate_standard_source_full: Any,
    gas_log_amounts_init: Any,
    condensate_amounts_init: Any,
    total_gas_log_amount_init: Any,
    element_potential_init: Any,
    support_indices: Sequence[int],
    condensate_valid_mask: Any | None = None,
    stationarity_tolerance: float = 1.0e-8,
    budget_tolerance: float = 1.0e-8,
    total_density_tolerance: float = 1.0e-8,
    support_closure_tolerance: float = 1.0e-8,
    budget_relative_floor: float = 1.0e-6,
    max_function_evaluations: int = 400,
    function_evaluation_budget: _FunctionEvaluationBudget | None = None,
    reduce_initial_support: bool = True,
    use_zero_barrier_dual: bool = True,
    use_finite_barrier_homotopy: bool = True,
) -> ZeroBarrierPolishResult:
    """Run one exact zero-barrier solve/drop and reduced-support search.

    The primary solve analytically eliminates gas log amounts and solves for
    element potentials, total gas, and capacity-scaled linear condensate
    amounts.  Structural zero rows are removed before that reduced solve.  A
    dense joint solve remains as a compatibility fallback.  Every returned
    state must pass the unchanged physical stationarity, inventory,
    total-density, positivity, and inactive-support closure checks.
    """

    ag = np.asarray(gas_formula_matrix, dtype=np.float64)
    ac_full = np.asarray(condensate_formula_matrix_full, dtype=np.float64)
    target = np.asarray(target_inventory, dtype=np.float64)
    gamma = np.asarray(gas_standard_source, dtype=np.float64)
    hcond_full = np.asarray(
        condensate_standard_source_full,
        dtype=np.float64,
    )
    q_initial = np.asarray(gas_log_amounts_init, dtype=np.float64)
    full_m_initial = np.asarray(condensate_amounts_init, dtype=np.float64)
    lambda_initial = np.asarray(element_potential_init, dtype=np.float64)
    qtot_initial = float(np.asarray(total_gas_log_amount_init, dtype=np.float64))
    support_initial = tuple(int(index) for index in support_indices)

    element_count, gas_count = ag.shape
    condensate_count = ac_full.shape[1] if ac_full.ndim == 2 else -1
    expected_shapes = bool(
        ac_full.shape == (element_count, condensate_count)
        and target.shape == (element_count,)
        and gamma.shape == (gas_count,)
        and hcond_full.shape == (condensate_count,)
        and q_initial.shape == (gas_count,)
        and full_m_initial.shape == (condensate_count,)
        and lambda_initial.shape == (element_count,)
    )
    support_valid = bool(
        len(set(support_initial)) == len(support_initial)
        and all(0 <= index < condensate_count for index in support_initial)
    )
    numerical_inputs = (
        ag,
        ac_full,
        target,
        gamma,
        hcond_full,
        q_initial,
        full_m_initial,
        lambda_initial,
        np.asarray([qtot_initial]),
    )
    finite_inputs = all(np.all(np.isfinite(value)) for value in numerical_inputs)
    if not expected_shapes or not support_valid or not finite_inputs:
        raise ValueError("Invalid zero-barrier active-support polish inputs.")
    if not support_initial:
        raise ValueError("Zero-barrier polish requires a non-empty support.")
    if any(
        tolerance < 0.0
        for tolerance in (
            stationarity_tolerance,
            budget_tolerance,
            total_density_tolerance,
            support_closure_tolerance,
            budget_relative_floor,
        )
    ):
        raise ValueError("Zero-barrier polish tolerances must be non-negative.")
    if int(max_function_evaluations) <= 0:
        raise ValueError("max_function_evaluations must be positive.")

    if condensate_valid_mask is None:
        valid_mask = np.ones(condensate_count, dtype=bool)
    else:
        valid_mask = np.asarray(condensate_valid_mask, dtype=bool)
        if valid_mask.shape != (condensate_count,):
            raise ValueError(
                "condensate_valid_mask must have one value per condensate."
            )
    if any(not valid_mask[index] for index in support_initial):
        raise ValueError("The active support contains a temperature-invalid phase.")

    positive_target = np.abs(target[target != 0.0])
    inventory_scale = (
        float(np.max(positive_target)) if positive_target.size else 1.0
    )
    zero_target_absolute_scale = max(
        float(budget_relative_floor),
        np.finfo(np.float64).eps * inventory_scale,
        1.0e-300,
    )
    budget_denominator = np.where(
        target != 0.0,
        np.maximum(np.abs(target), 1.0e-300),
        zero_target_absolute_scale,
    )
    budget_scale = 1.0 / budget_denominator
    dual_support = _select_support_with_zero_barrier_dual(
        gas_formula_matrix=ag,
        condensate_formula_matrix_full=ac_full,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond_full,
        gas_log_amounts_init=q_initial,
        condensate_amounts_init=full_m_initial,
        total_gas_log_amount_init=qtot_initial,
        element_potential_init=lambda_initial,
        condensate_valid_mask=valid_mask,
        stationarity_tolerance=stationarity_tolerance,
        support_closure_tolerance=support_closure_tolerance,
        max_function_evaluations=max_function_evaluations,
        enabled=bool(reduce_initial_support and use_zero_barrier_dual),
        function_evaluation_budget=function_evaluation_budget,
    )
    dual_support_report = dict(dual_support["report"])
    finite_homotopy = _select_support_with_finite_barrier_homotopy(
        gas_formula_matrix=ag,
        condensate_formula_matrix_full=ac_full,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond_full,
        gas_log_amounts_init=q_initial,
        condensate_amounts_init=full_m_initial,
        total_gas_log_amount_init=qtot_initial,
        element_potential_init=lambda_initial,
        support_indices=support_initial,
        budget_scale=budget_scale,
        max_function_evaluations=max_function_evaluations,
        enabled=bool(
            reduce_initial_support
            and use_finite_barrier_homotopy
            and not dual_support["applied"]
        ),
        function_evaluation_budget=function_evaluation_budget,
    )
    finite_homotopy_report = dict(finite_homotopy["report"])
    if dual_support["applied"]:
        reduced_support = tuple(dual_support["support_indices"])
        reduced_full_m = np.asarray(
            dual_support["condensate_amounts"], dtype=np.float64
        )
        reduced_q_initial = np.asarray(
            dual_support["gas_log_amounts"], dtype=np.float64
        )
        reduced_qtot_initial = float(
            dual_support["total_gas_log_amount"]
        )
        reduced_lambda_initial = np.asarray(
            dual_support["element_potential"], dtype=np.float64
        )
        _, _, basic_support_reduction = (
            _reduce_initial_condensate_support_to_basic(
                condensate_formula_matrix_full=ac_full,
                condensate_standard_source_full=hcond_full,
                target_inventory=target,
                condensate_amounts=reduced_full_m,
                support_indices=reduced_support,
                budget_scale=budget_scale,
                budget_tolerance=budget_tolerance,
                enabled=False,
            )
        )
        basic_support_reduction["skip_reason"] = (
            "zero_barrier_dual_support_selected"
        )
        basic_support_reduction["input_initializer"] = (
            "zero_barrier_dual_support"
        )
    elif finite_homotopy["applied"]:
        homotopy_support = tuple(finite_homotopy["support_indices"])
        homotopy_full_m = np.asarray(
            finite_homotopy["condensate_amounts"], dtype=np.float64
        )
        (
            reduced_support,
            reduced_full_m,
            basic_support_reduction,
        ) = _reduce_initial_condensate_support_to_basic(
            condensate_formula_matrix_full=ac_full,
            condensate_standard_source_full=hcond_full,
            target_inventory=target,
            condensate_amounts=homotopy_full_m,
            support_indices=homotopy_support,
            budget_scale=budget_scale,
            budget_tolerance=budget_tolerance,
            enabled=bool(reduce_initial_support),
        )
        basic_support_reduction["input_initializer"] = (
            "finite_barrier_homotopy_selected_support"
        )
        reduced_q_initial = np.asarray(
            finite_homotopy["gas_log_amounts"], dtype=np.float64
        )
        reduced_qtot_initial = float(
            finite_homotopy["total_gas_log_amount"]
        )
        reduced_lambda_initial = np.asarray(
            finite_homotopy["element_potential"], dtype=np.float64
        )
    else:
        (
            reduced_support,
            reduced_full_m,
            basic_support_reduction,
        ) = _reduce_initial_condensate_support_to_basic(
            condensate_formula_matrix_full=ac_full,
            condensate_standard_source_full=hcond_full,
            target_inventory=target,
            condensate_amounts=full_m_initial,
            support_indices=support_initial,
            budget_scale=budget_scale,
            budget_tolerance=budget_tolerance,
            enabled=bool(reduce_initial_support),
        )
        reduced_q_initial = q_initial.copy()
        reduced_qtot_initial = qtot_initial
        reduced_lambda_initial = lambda_initial.copy()
    (
        q_initial,
        qtot_initial,
        lambda_initial,
        initializer_regularization,
    ) = _capacity_regularized_initializer(
        gas_formula_matrix=ag,
        target_inventory=target,
        gas_standard_source=gamma,
        gas_log_amounts=reduced_q_initial,
        total_gas_log_amount=reduced_qtot_initial,
        element_potential=reduced_lambda_initial,
    )
    (
        regularized_primary_structure_eligible,
        regularized_primary_structure_reason,
    ) = _reduced_log_domain_eligibility(
        gas_formula_matrix=ag,
        condensate_formula_matrix_full=ac_full,
        target_inventory=target,
        support_indices=reduced_support,
    )
    regularized_primary_full_rank = bool(
        initializer_regularization["element_potential_fit_rank"]
        == element_count
    )
    minimum_inventory_fraction = (
        float(np.min(positive_target) / inventory_scale)
        if positive_target.size
        else 1.0
    )
    binary64_relative_precision = float(np.finfo(np.float64).eps)
    minimum_inventory_at_or_below_binary64_epsilon = bool(
        minimum_inventory_fraction <= binary64_relative_precision
    )
    use_regularized_reduced_primary = bool(
        regularized_primary_structure_eligible
        and initializer_regularization["applied"]
        and initializer_regularization["element_potential_recomputed"]
        and regularized_primary_full_rank
        and minimum_inventory_at_or_below_binary64_epsilon
    )
    if not regularized_primary_structure_eligible:
        regularized_primary_skip_reason = (
            f"ineligible_structure:{regularized_primary_structure_reason}"
        )
    elif not initializer_regularization["applied"]:
        regularized_primary_skip_reason = "regularization_not_applied"
    elif not initializer_regularization["element_potential_recomputed"]:
        regularized_primary_skip_reason = "potential_not_recomputed"
    elif not regularized_primary_full_rank:
        regularized_primary_skip_reason = "potential_fit_not_full_rank"
    elif not minimum_inventory_at_or_below_binary64_epsilon:
        regularized_primary_skip_reason = (
            "minimum_inventory_fraction_above_binary64_epsilon"
        )
    else:
        regularized_primary_skip_reason = None
    initializer_regularization["eligible_for_reduced_primary"] = (
        use_regularized_reduced_primary
    )
    initializer_regularization["applied_to_reduced_primary"] = False
    initializer_regularization["reduced_primary_structure_eligible"] = (
        regularized_primary_structure_eligible
    )
    initializer_regularization["reduced_primary_structure_reason"] = (
        regularized_primary_structure_reason
    )
    initializer_regularization["reduced_primary_full_rank_fit"] = (
        regularized_primary_full_rank
    )
    initializer_regularization[
        "reduced_primary_minimum_inventory_fraction"
    ] = minimum_inventory_fraction
    initializer_regularization[
        "reduced_primary_relative_precision_threshold"
    ] = binary64_relative_precision
    initializer_regularization[
        "reduced_primary_minimum_inventory_at_or_below_binary64_epsilon"
    ] = minimum_inventory_at_or_below_binary64_epsilon
    initializer_regularization["reduced_primary_skip_reason"] = (
        regularized_primary_skip_reason
    )
    initializer_regularization[
        "applied_to_dense_compatibility_fallback"
    ] = True
    current_support = reduced_support
    current_q = q_initial.copy()
    current_qtot = qtot_initial
    current_lambda = lambda_initial.copy()
    current_full_m = reduced_full_m
    dropped = list(
        index for index in support_initial if index not in set(reduced_support)
    )
    attempts: list[dict[str, Any]] = []
    last_optimizer_success = False
    last_optimizer_status: int | None = None
    last_optimizer_message = "not run"
    last_nfev = 0
    selected_formulation = "capacity_scaled_linear_amounts"
    reduced_primary_selected = False
    dense_solver_attempted = False
    primary_candidate: dict[str, Any] | None = None
    normalized_primary_report: dict[str, Any] = {
        "schema": (
            "exogibbs_zero_barrier_normalized_gas_reduced_linear_v1"
        ),
        "attempted": False,
        "accepted": False,
        "skip_reason": "structural_zero_reduced_solver_selected",
    }
    normalized_initializer_portfolio_report: dict[str, Any] = {
        "schema": (
            "exogibbs_zero_barrier_normalized_initializer_portfolio_v1"
        ),
        "regularized_initializer_eligible": (
            use_regularized_reduced_primary
        ),
        "regularized_initializer_skip_reason": (
            regularized_primary_skip_reason
        ),
        "attempted": False,
        "regularized_attempted": False,
        "regularized_budget_skipped": False,
        "regularized_function_evaluation_limit": None,
        "regularized_function_evaluations": 0,
        "raw_function_evaluation_reserve": None,
        "unregularized_attempted": False,
        "raw_retry_attempted": False,
        "selected_initializer": None,
        "attempts": (),
        "discarded_solve_reports": (),
    }
    alternative_basic_support_enabled = bool(
        reduce_initial_support
        and basic_support_reduction.get("attempted", False)
        and not basic_support_reduction.get("applied", False)
        and basic_support_reduction.get("initial_support_nullity", 0) > 0
    )
    structural_log_rescue_report: dict[str, Any] = {
        "schema": (
            "exogibbs_zero_barrier_structural_zero_log_rescue_v1"
        ),
        "eligible": False,
        "attempted": False,
        "accepted": False,
        "skip_reason": "normalized_gas_reduced_local_kkt_passed",
    }
    structural_zero_preferred = bool(
        np.any(target == 0.0)
        and np.all(target >= 0.0)
        and np.all(ag >= 0.0)
        and np.all(ac_full >= 0.0)
    )
    if not structural_zero_preferred:
        structural_log_rescue_report["skip_reason"] = (
            "no_eligible_structural_zero_rows"
        )

    def run_structural_log_rescue() -> dict[str, Any]:
        return _solve_structural_zero_reduced_log_domain_active_support(
            gas_formula_matrix=ag,
            condensate_formula_matrix_full=ac_full,
            target_inventory=target,
            gas_standard_source=gamma,
            condensate_standard_source_full=hcond_full,
            gas_log_amounts_init=reduced_q_initial,
            condensate_amounts_init=reduced_full_m,
            total_gas_log_amount_init=reduced_qtot_initial,
            element_potential_init=reduced_lambda_initial,
            support_indices=reduced_support,
            condensate_valid_mask=valid_mask,
            budget_scale=budget_scale,
            stationarity_tolerance=stationarity_tolerance,
            budget_tolerance=budget_tolerance,
            total_density_tolerance=total_density_tolerance,
            support_closure_tolerance=support_closure_tolerance,
            max_function_evaluations=max_function_evaluations,
            function_evaluation_budget=function_evaluation_budget,
        )

    def candidate_has_local_kkt(candidate: dict[str, Any] | None) -> bool:
        return bool(
            candidate is not None
            and not candidate.get("active_phase_at_lower_bound", False)
            and _physical_audit_local_kkt_passed(
                candidate["audit"],
                optimizer_success=bool(candidate["optimizer_success"]),
                optimizer_status=int(candidate["optimizer_status"]),
                stationarity_tolerance=stationarity_tolerance,
                budget_tolerance=budget_tolerance,
                total_density_tolerance=total_density_tolerance,
            )
        )

    def structural_selected_formulation() -> str:
        if structural_log_rescue_report.get("inner_formulation") == (
            "reduced_log_domain"
        ):
            return "structural_zero_reduced_log_domain"
        return "structural_zero_normalized_gas_reduced_linear_amounts"

    if structural_zero_preferred:
        structural_log_rescue = run_structural_log_rescue()
        structural_log_rescue_report = dict(
            structural_log_rescue["report"]
        )
        primary_candidate = structural_log_rescue["candidate"]
        if candidate_has_local_kkt(primary_candidate):
            reduced_primary_selected = True
            selected_formulation = structural_selected_formulation()

    structural_support_full_rank = bool(
        reduced_primary_selected
        and primary_candidate is not None
        and len(tuple(primary_candidate["support_indices"]))
        == np.linalg.matrix_rank(
            ac_full[:, tuple(primary_candidate["support_indices"])]
        )
    )
    structural_terminal_accepted = bool(
        reduced_primary_selected
        and primary_candidate is not None
        and primary_candidate["audit"]["accepted"]
    )
    alternative_basic_support_solve_enabled = bool(
        alternative_basic_support_enabled
        and not structural_support_full_rank
        and not structural_terminal_accepted
    )
    alternative_basic_support = (
        _solve_alternative_basic_support_portfolio(
            gas_formula_matrix=ag,
            condensate_formula_matrix_full=ac_full,
            target_inventory=target,
            gas_standard_source=gamma,
            condensate_standard_source_full=hcond_full,
            gas_log_amounts_init=reduced_q_initial,
            condensate_amounts_init=reduced_full_m,
            total_gas_log_amount_init=reduced_qtot_initial,
            element_potential_init=reduced_lambda_initial,
            support_indices=reduced_support,
            condensate_valid_mask=valid_mask,
            budget_scale=budget_scale,
            stationarity_tolerance=stationarity_tolerance,
            budget_tolerance=budget_tolerance,
            total_density_tolerance=total_density_tolerance,
            support_closure_tolerance=support_closure_tolerance,
            max_function_evaluations=max_function_evaluations,
            enabled=alternative_basic_support_solve_enabled,
            function_evaluation_budget=function_evaluation_budget,
        )
    )
    alternative_basic_support_report = dict(
        alternative_basic_support["report"]
    )
    if alternative_basic_support_enabled and (
        structural_support_full_rank or structural_terminal_accepted
    ):
        alternative_basic_support_report["skip_reason"] = (
            "structural_zero_certified_support_selected"
        )
    if alternative_basic_support["selected"]:
        primary_candidate = alternative_basic_support["candidate"]
        reduced_primary_selected = True
        selected_formulation = (
            "alternative_basic_support_normalized_gas_reduced_linear_amounts"
        )
        normalized_primary_report["skip_reason"] = (
            "alternative_basic_support_selected"
        )

    if not reduced_primary_selected:
        normalized_initializers = []
        if use_regularized_reduced_primary:
            normalized_initializers.append(
                (
                    "capacity_regularized",
                    q_initial,
                    qtot_initial,
                    lambda_initial,
                )
            )
        normalized_initializers.append(
            (
                "unregularized",
                reduced_q_initial,
                reduced_qtot_initial,
                reduced_lambda_initial,
            )
        )
        discarded_solve_reports = []
        initializer_attempts = []
        for (
            initializer_name,
            normalized_q_initial,
            normalized_qtot_initial,
            normalized_lambda_initial,
        ) in normalized_initializers:
            attempt_evaluation_budget = function_evaluation_budget
            regularized_child_budget = None
            if initializer_name == "capacity_regularized":
                if function_evaluation_budget is None:
                    raw_reserve = None
                    regularized_limit = int(max_function_evaluations)
                else:
                    raw_reserve = min(
                        int(max_function_evaluations),
                        function_evaluation_budget.remaining,
                    )
                    regularized_limit = min(
                        int(max_function_evaluations),
                        max(
                            function_evaluation_budget.remaining
                            - raw_reserve,
                            0,
                        ),
                    )
                normalized_initializer_portfolio_report[
                    "regularized_function_evaluation_limit"
                ] = regularized_limit
                normalized_initializer_portfolio_report[
                    "raw_function_evaluation_reserve"
                ] = raw_reserve
                if regularized_limit <= 0:
                    normalized_initializer_portfolio_report[
                        "regularized_budget_skipped"
                    ] = True
                    continue
                regularized_child_budget = _FunctionEvaluationBudget(
                    regularized_limit
                )
                attempt_evaluation_budget = regularized_child_budget
            try:
                normalized_primary = (
                    _solve_normalized_gas_reduced_linear_support(
                        gas_formula_matrix=ag,
                        condensate_formula_matrix_full=ac_full,
                        target_inventory=target,
                        gas_standard_source=gamma,
                        condensate_standard_source_full=hcond_full,
                        gas_log_amounts_init=normalized_q_initial,
                        condensate_amounts_init=reduced_full_m,
                        total_gas_log_amount_init=normalized_qtot_initial,
                        element_potential_init=normalized_lambda_initial,
                        support_indices=reduced_support,
                        condensate_valid_mask=valid_mask,
                        budget_scale=budget_scale,
                        stationarity_tolerance=stationarity_tolerance,
                        budget_tolerance=budget_tolerance,
                        total_density_tolerance=total_density_tolerance,
                        support_closure_tolerance=(
                            support_closure_tolerance
                        ),
                        max_function_evaluations=max_function_evaluations,
                        function_evaluation_budget=(
                            attempt_evaluation_budget
                        ),
                    )
                )
            finally:
                if regularized_child_budget is not None:
                    regularized_evaluations = regularized_child_budget.used
                    normalized_initializer_portfolio_report[
                        "regularized_function_evaluations"
                    ] = regularized_evaluations
                    if function_evaluation_budget is not None:
                        function_evaluation_budget.consume(
                            regularized_evaluations
                        )
            attempt_report = dict(normalized_primary["report"])
            attempt_report["initializer"] = initializer_name
            attempt_candidate = normalized_primary["candidate"]
            local_kkt_passed = candidate_has_local_kkt(attempt_candidate)
            attempt_audit = (
                attempt_candidate.get("audit", {})
                if attempt_candidate is not None
                else {}
            )
            initializer_attempt = {
                "initializer": initializer_name,
                "function_evaluations": sum(
                    int(attempt.get("function_evaluations", 0))
                    for attempt in attempt_report.get("attempts", ())
                ),
                "local_kkt_passed": local_kkt_passed,
                "optimizer_success": (
                    None
                    if attempt_candidate is None
                    else bool(attempt_candidate["optimizer_success"])
                ),
                "positive_active_amounts": attempt_audit.get(
                    "positive_active_amounts"
                ),
                "gas_stationarity_max_abs": attempt_audit.get(
                    "gas_stationarity_max_abs"
                ),
                "active_condensate_driving_max_abs": attempt_audit.get(
                    "active_condensate_driving_max_abs"
                ),
                "budget_scaled_max_abs": attempt_audit.get(
                    "budget_scaled_max_abs"
                ),
                "total_density_scaled_abs": attempt_audit.get(
                    "total_density_scaled_abs"
                ),
                "selected": False,
            }
            initializer_attempts.append(initializer_attempt)
            normalized_initializer_portfolio_report["attempted"] = True
            if initializer_name == "capacity_regularized":
                initializer_regularization[
                    "applied_to_reduced_primary"
                ] = True
                normalized_initializer_portfolio_report[
                    "regularized_attempted"
                ] = True
            else:
                normalized_initializer_portfolio_report[
                    "unregularized_attempted"
                ] = True
                normalized_initializer_portfolio_report[
                    "raw_retry_attempted"
                ] = bool(discarded_solve_reports)
            normalized_primary_report = attempt_report
            primary_candidate = attempt_candidate
            if local_kkt_passed:
                reduced_primary_selected = True
                selected_formulation = (
                    "normalized_gas_reduced_linear_amounts"
                )
                normalized_initializer_portfolio_report[
                    "selected_initializer"
                ] = initializer_name
                initializer_attempt["selected"] = True
                break
            if initializer_name != normalized_initializers[-1][0]:
                discarded_solve_reports.append(
                    {
                        "initializer": initializer_name,
                        "solve": attempt_report,
                    }
                )
        normalized_initializer_portfolio_report[
            "attempts"
        ] = tuple(initializer_attempts)
        normalized_initializer_portfolio_report[
            "discarded_solve_reports"
        ] = tuple(discarded_solve_reports)

    support_initializer_applied = bool(
        dual_support["applied"] or finite_homotopy["applied"]
    )
    if support_initializer_applied and not reduced_primary_selected:
        remaining_before_retry = (
            None
            if function_evaluation_budget is None
            else function_evaluation_budget.remaining
        )
        retry_result = _polish_zero_barrier_support_once(
            gas_formula_matrix=gas_formula_matrix,
            condensate_formula_matrix_full=(
                condensate_formula_matrix_full
            ),
            target_inventory=target_inventory,
            gas_standard_source=gas_standard_source,
            condensate_standard_source_full=(
                condensate_standard_source_full
            ),
            gas_log_amounts_init=gas_log_amounts_init,
            condensate_amounts_init=condensate_amounts_init,
            total_gas_log_amount_init=total_gas_log_amount_init,
            element_potential_init=element_potential_init,
            support_indices=support_indices,
            condensate_valid_mask=condensate_valid_mask,
            stationarity_tolerance=stationarity_tolerance,
            budget_tolerance=budget_tolerance,
            total_density_tolerance=total_density_tolerance,
            support_closure_tolerance=support_closure_tolerance,
            budget_relative_floor=budget_relative_floor,
            max_function_evaluations=max_function_evaluations,
            function_evaluation_budget=function_evaluation_budget,
            reduce_initial_support=reduce_initial_support,
            use_zero_barrier_dual=False,
            use_finite_barrier_homotopy=not finite_homotopy["applied"],
        )
        retry_report = dict(retry_result.report)
        retry_initializer_diagnostics = {
            "schema": (
                "exogibbs_zero_barrier_retry_initializer_diagnostics_v1"
            ),
            "zero_barrier_dual_support_oracle": retry_report.get(
                "zero_barrier_dual_support_oracle"
            ),
            "finite_barrier_homotopy_initializer": retry_report.get(
                "finite_barrier_homotopy_initializer"
            ),
            "support_initializer_postselection_fallback": retry_report.get(
                "support_initializer_postselection_fallback"
            ),
        }
        retry_report["zero_barrier_dual_support_oracle"] = (
            dual_support_report
        )
        retry_report["finite_barrier_homotopy_initializer"] = (
            finite_homotopy_report
        )
        retry_report["support_initializer_postselection_fallback"] = {
            "schema": (
                "exogibbs_zero_barrier_support_initializer_fallback_v1"
            ),
            "attempted": True,
            "reason": "selected_support_local_root_failed",
            "selected_support_indices": reduced_support,
            "selected_support_source": (
                "zero_barrier_dual_support"
                if dual_support["applied"]
                else "finite_barrier_homotopy"
            ),
            "remaining_function_evaluations_before_retry": (
                remaining_before_retry
            ),
            "selected_support_normalized_solve": (
                normalized_primary_report
            ),
            "selected_support_normalized_initializer_portfolio": (
                normalized_initializer_portfolio_report
            ),
            "selected_support_alternative_basic_support_portfolio": (
                alternative_basic_support_report
            ),
            "selected_support_structural_zero_solve": (
                structural_log_rescue_report
            ),
            "retry_accepted": bool(retry_result.accepted),
            "retry_support_indices": tuple(retry_result.support_indices),
            "retry_selected_numerical_formulation": retry_report.get(
                "selected_numerical_formulation"
            ),
            "retry_initializer_diagnostics": retry_initializer_diagnostics,
        }
        return ZeroBarrierPolishResult(
            accepted=retry_result.accepted,
            gas_log_amounts=retry_result.gas_log_amounts,
            condensate_amounts=retry_result.condensate_amounts,
            total_gas_log_amount=retry_result.total_gas_log_amount,
            element_potential=retry_result.element_potential,
            support_indices=retry_result.support_indices,
            report=retry_report,
        )

    if reduced_primary_selected and primary_candidate is not None:
        current_q = np.asarray(
            primary_candidate["gas_log_amounts"], dtype=np.float64
        )
        current_full_m = np.asarray(
            primary_candidate["condensate_amounts"], dtype=np.float64
        )
        current_qtot = float(primary_candidate["total_gas_log_amount"])
        current_lambda = np.asarray(
            primary_candidate["element_potential"], dtype=np.float64
        )
        current_support = tuple(primary_candidate["support_indices"])
        last_optimizer_success = bool(
            primary_candidate["optimizer_success"]
        )
        last_optimizer_status = int(primary_candidate["optimizer_status"])
        last_optimizer_message = str(
            primary_candidate["optimizer_message"]
        )
        last_nfev = int(primary_candidate["function_evaluations"])
        dropped = [
            index
            for index in support_initial
            if index not in set(current_support)
        ]

    dense_drop_rounds = (
        0 if reduced_primary_selected else len(current_support) + 1
    )
    for _drop_round in range(dense_drop_rounds):
        dense_solver_attempted = True
        active = np.asarray(current_support, dtype=np.int64)
        ac = ac_full[:, active]
        hcond = hcond_full[active]
        amount_scales = _maximum_condensate_amount_scales(ac, target)
        m_initial = np.maximum(current_full_m[active], 1.0e-300 * amount_scales)
        u_initial = m_initial / amount_scales
        x0 = np.concatenate(
            [current_q, u_initial, [current_qtot], current_lambda]
        )

        def unpack(values: np.ndarray):
            q = values[:gas_count]
            u_start = gas_count
            u = values[u_start : u_start + len(current_support)]
            qtot = values[u_start + len(current_support)]
            lambda_ = values[u_start + len(current_support) + 1 :]
            return q, amount_scales * u, qtot, lambda_

        def residual(values: np.ndarray) -> np.ndarray:
            q, amounts, qtot, lambda_ = unpack(values)
            with np.errstate(
                over="ignore",
                under="ignore",
                invalid="ignore",
            ):
                gas = np.exp(q)
                gas_fractions = np.exp(q - qtot)
                result = np.concatenate(
                    [
                        q + gamma - qtot - ag.T @ lambda_,
                        hcond - ac.T @ lambda_,
                        budget_scale * (
                            ag @ gas + ac @ amounts - target
                        ),
                        np.asarray(
                            [np.sum(gas_fractions) - 1.0],
                            dtype=np.float64,
                        ),
                    ]
                )
            return result

        def jacobian(values: np.ndarray) -> np.ndarray:
            q, _amounts, qtot, _lambda = unpack(values)
            with np.errstate(
                over="ignore",
                under="ignore",
                invalid="ignore",
            ):
                gas = np.exp(q)
                gas_fractions = np.exp(q - qtot)
            support_count = len(current_support)
            variable_count = gas_count + support_count + 1 + element_count
            matrix = np.zeros((variable_count, variable_count), dtype=np.float64)
            gas_rows = slice(0, gas_count)
            cond_rows = slice(gas_count, gas_count + support_count)
            budget_rows = slice(
                gas_count + support_count,
                gas_count + support_count + element_count,
            )
            total_row = variable_count - 1
            u_columns = slice(gas_count, gas_count + support_count)
            qtot_column = gas_count + support_count
            lambda_columns = slice(qtot_column + 1, variable_count)
            matrix[gas_rows, :gas_count] = np.eye(gas_count)
            matrix[gas_rows, qtot_column] = -1.0
            matrix[gas_rows, lambda_columns] = -ag.T
            matrix[cond_rows, lambda_columns] = -ac.T
            with np.errstate(
                over="ignore",
                under="ignore",
                invalid="ignore",
            ):
                matrix[budget_rows, :gas_count] = (
                    budget_scale[:, None] * ag * gas[None, :]
                )
                matrix[budget_rows, u_columns] = (
                    budget_scale[:, None]
                    * ac
                    * amount_scales[None, :]
                )
                matrix[total_row, :gas_count] = gas_fractions
                matrix[total_row, qtot_column] = -np.sum(gas_fractions)
            return matrix

        call_evaluation_limit = _function_evaluation_call_limit(
            max_function_evaluations,
            function_evaluation_budget,
        )
        if call_evaluation_limit <= 0:
            last_optimizer_success = False
            last_optimizer_status = None
            last_optimizer_message = "function evaluation limit reached"
            last_nfev = 0
            attempts.append(
                {
                    "support_indices": current_support,
                    "optimizer_success": False,
                    "optimizer_status": last_optimizer_status,
                    "optimizer_message": last_optimizer_message,
                    "function_evaluations": 0,
                    "failure_reason": "function_evaluation_limit_reached",
                }
            )
            break
        try:
            optimization = _least_squares_with_scipy_overflow_guard(
                residual,
                x0,
                jac=jacobian,
                method="trf",
                x_scale="jac",
                ftol=1.0e-13,
                xtol=1.0e-13,
                gtol=1.0e-13,
                max_nfev=call_evaluation_limit,
            )
        except (FloatingPointError, OverflowError, ValueError) as error:
            conservative_evaluations = 0
            if function_evaluation_budget is not None:
                function_evaluation_budget.consume(call_evaluation_limit)
                conservative_evaluations = call_evaluation_limit
            last_optimizer_success = False
            last_optimizer_status = None
            last_optimizer_message = f"{type(error).__name__}: {error}"
            last_nfev = conservative_evaluations
            attempts.append(
                {
                    "support_indices": current_support,
                    "optimizer_success": False,
                    "optimizer_status": last_optimizer_status,
                    "optimizer_message": last_optimizer_message,
                    "function_evaluations": last_nfev,
                    "function_evaluations_conservative": bool(
                        function_evaluation_budget is not None
                    ),
                    "failure_reason": "linear_amount_solver_exception",
                }
            )
            break
        if function_evaluation_budget is not None:
            function_evaluation_budget.consume(int(optimization.nfev))
        q, active_amounts, qtot, lambda_ = unpack(optimization.x)
        last_optimizer_success = bool(optimization.success)
        last_optimizer_status = int(optimization.status)
        last_optimizer_message = str(optimization.message)
        last_nfev = int(optimization.nfev)
        candidate_full_m = np.zeros(condensate_count, dtype=np.float64)
        candidate_full_m[active] = active_amounts
        candidate_audit = _physical_zero_barrier_audit(
            gas_formula_matrix=ag,
            condensate_formula_matrix_full=ac_full,
            target_inventory=target,
            gas_standard_source=gamma,
            condensate_standard_source_full=hcond_full,
            gas_log_amounts=q,
            condensate_amounts=candidate_full_m,
            total_gas_log_amount=qtot,
            element_potential=lambda_,
            support_indices=current_support,
            condensate_valid_mask=valid_mask,
            budget_scale=budget_scale,
            optimizer_success=last_optimizer_success,
            optimizer_status=last_optimizer_status,
            stationarity_tolerance=stationarity_tolerance,
            budget_tolerance=budget_tolerance,
            total_density_tolerance=total_density_tolerance,
            support_closure_tolerance=support_closure_tolerance,
        )
        drop_authorized_by_root = _physical_audit_root_blocks_passed(
            candidate_audit,
            optimizer_success=last_optimizer_success,
            stationarity_tolerance=stationarity_tolerance,
            budget_tolerance=budget_tolerance,
            total_density_tolerance=total_density_tolerance,
        )
        attempts.append(
            {
                "support_indices": current_support,
                "optimizer_success": last_optimizer_success,
                "optimizer_status": last_optimizer_status,
                "optimizer_message": last_optimizer_message,
                "function_evaluations": last_nfev,
                "cost": float(optimization.cost),
                "optimality": float(optimization.optimality),
                "physical_root_certified": candidate_audit[
                    "physical_root_certified"
                ],
                "acceptance_source": candidate_audit[
                    "acceptance_source"
                ],
                "drop_authorized_by_root": drop_authorized_by_root,
                "active_condensate_amounts": tuple(
                    float(value) for value in active_amounts.tolist()
                ),
            }
        )
        nonpositive = np.flatnonzero(active_amounts <= 0.0)
        if nonpositive.size:
            current_q = q
            current_qtot = float(qtot)
            current_lambda = lambda_
            current_full_m = candidate_full_m
            if not drop_authorized_by_root:
                break
            relative_amounts = active_amounts / amount_scales
            local_drop = int(
                nonpositive[
                    np.argmin(relative_amounts[nonpositive])
                ]
            )
            dropped_index = current_support[local_drop]
            dropped.append(dropped_index)
            current_support = tuple(
                index for index in current_support if index != dropped_index
            )
            current_full_m = np.zeros(condensate_count, dtype=np.float64)
            for index, amount in zip(active.tolist(), active_amounts.tolist()):
                if index != dropped_index:
                    current_full_m[index] = max(float(amount), 1.0e-300)
            continue
        current_q = q
        current_qtot = float(qtot)
        current_lambda = lambda_
        current_full_m = candidate_full_m
        break

    linear_audit = _physical_zero_barrier_audit(
        gas_formula_matrix=ag,
        condensate_formula_matrix_full=ac_full,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond_full,
        gas_log_amounts=current_q,
        condensate_amounts=current_full_m,
        total_gas_log_amount=current_qtot,
        element_potential=current_lambda,
        support_indices=current_support,
        condensate_valid_mask=valid_mask,
        budget_scale=budget_scale,
        optimizer_success=last_optimizer_success,
        optimizer_status=last_optimizer_status,
        stationarity_tolerance=stationarity_tolerance,
        budget_tolerance=budget_tolerance,
        total_density_tolerance=total_density_tolerance,
        support_closure_tolerance=support_closure_tolerance,
    )
    linear_local_kkt_passed = _physical_audit_local_kkt_passed(
        linear_audit,
        optimizer_success=last_optimizer_success,
        optimizer_status=last_optimizer_status,
        stationarity_tolerance=stationarity_tolerance,
        budget_tolerance=budget_tolerance,
        total_density_tolerance=total_density_tolerance,
    )
    fallback_eligible, fallback_reason = _reduced_log_domain_eligibility(
        gas_formula_matrix=ag,
        condensate_formula_matrix_full=ac_full,
        target_inventory=target,
        support_indices=reduced_support,
    )
    fallback_report: dict[str, Any] = {
        "schema": "exogibbs_zero_barrier_reduced_support_search_v1",
        "eligible": fallback_eligible,
        "attempted": False,
        "accepted": False,
        "skip_reason": (
            (
                "reduced_primary_physical_audit_accepted"
                if linear_audit["accepted"]
                else "reduced_primary_inactive_support_closure_only"
            )
            if reduced_primary_selected
            else (
                "linear_amount_physical_audit_accepted"
                if linear_audit["accepted"]
                else (
                    "linear_amount_inactive_support_closure_only"
                    if linear_local_kkt_passed
                    else fallback_reason
                )
            )
        ),
    }
    if (
        not reduced_primary_selected
        and not linear_audit["accepted"]
        and not linear_local_kkt_passed
        and fallback_eligible
    ):
        reduced = _solve_reduced_log_domain_support_branches(
            gas_formula_matrix=ag,
            condensate_formula_matrix_full=ac_full,
            target_inventory=target,
            gas_standard_source=gamma,
            condensate_standard_source_full=hcond_full,
            gas_log_amounts_init=q_initial,
            condensate_amounts_init=reduced_full_m,
            total_gas_log_amount_init=qtot_initial,
            element_potential_init=lambda_initial,
            support_indices=reduced_support,
            condensate_valid_mask=valid_mask,
            budget_scale=budget_scale,
            stationarity_tolerance=stationarity_tolerance,
            budget_tolerance=budget_tolerance,
            total_density_tolerance=total_density_tolerance,
            support_closure_tolerance=support_closure_tolerance,
            max_function_evaluations=max_function_evaluations,
            function_evaluation_budget=function_evaluation_budget,
        )
        fallback_report = dict(reduced["report"])
        if reduced["accepted"]:
            candidate = reduced["candidate"]
            current_q = np.asarray(
                candidate["gas_log_amounts"], dtype=np.float64
            )
            current_full_m = np.asarray(
                candidate["condensate_amounts"], dtype=np.float64
            )
            current_qtot = float(candidate["total_gas_log_amount"])
            current_lambda = np.asarray(
                candidate["element_potential"], dtype=np.float64
            )
            current_support = tuple(candidate["support_indices"])
            last_optimizer_success = bool(candidate["optimizer_success"])
            last_optimizer_status = int(candidate["optimizer_status"])
            last_optimizer_message = str(candidate["optimizer_message"])
            last_nfev = int(candidate["function_evaluations"])
            dropped = [
                index
                for index in support_initial
                if index not in set(current_support)
            ]
            selected_formulation = "reduced_log_domain_support_search"

    audit = _physical_zero_barrier_audit(
        gas_formula_matrix=ag,
        condensate_formula_matrix_full=ac_full,
        target_inventory=target,
        gas_standard_source=gamma,
        condensate_standard_source_full=hcond_full,
        gas_log_amounts=current_q,
        condensate_amounts=current_full_m,
        total_gas_log_amount=current_qtot,
        element_potential=current_lambda,
        support_indices=current_support,
        condensate_valid_mask=valid_mask,
        budget_scale=budget_scale,
        optimizer_success=last_optimizer_success,
        optimizer_status=last_optimizer_status,
        stationarity_tolerance=stationarity_tolerance,
        budget_tolerance=budget_tolerance,
        total_density_tolerance=total_density_tolerance,
        support_closure_tolerance=support_closure_tolerance,
    )
    full_driving = audit["full_driving"]
    finite = bool(audit["finite"])
    positive_active_amounts = bool(audit["positive_active_amounts"])
    gas_stationarity_norm = float(audit["gas_stationarity_max_abs"])
    active_driving_norm = float(
        audit["active_condensate_driving_max_abs"]
    )
    inactive_violation_norm = float(
        audit["inactive_condensate_violation_max_abs"]
    )
    budget_norm = float(audit["budget_scaled_max_abs"])
    total_norm = float(audit["total_density_scaled_abs"])
    accepted = bool(audit["accepted"])
    report = {
        "polish_schema": "exogibbs_zero_barrier_active_support_polish_v2",
        "attempted": True,
        "accepted": accepted,
        "initial_support_indices": support_initial,
        "final_support_indices": current_support,
        "dropped_support_indices": tuple(dropped),
        "optimizer_success": last_optimizer_success,
        "optimizer_status": last_optimizer_status,
        "optimizer_message": last_optimizer_message,
        "optimizer_termination_eligible": audit[
            "optimizer_termination_eligible"
        ],
        "physical_root_certified": audit["physical_root_certified"],
        "acceptance_source": audit["acceptance_source"],
        "function_evaluations": last_nfev,
        "finite": finite,
        "positive_active_amounts": positive_active_amounts,
        "gas_stationarity_max_abs": gas_stationarity_norm,
        "active_condensate_driving_max_abs": active_driving_norm,
        "inactive_condensate_violation_max_abs": inactive_violation_norm,
        "budget_scaled_max_abs": budget_norm,
        "total_density_scaled_abs": total_norm,
        "stationarity_tolerance": float(stationarity_tolerance),
        "budget_tolerance": float(budget_tolerance),
        "total_density_tolerance": float(total_density_tolerance),
        "support_closure_tolerance": float(support_closure_tolerance),
        "budget_scaling": (
            "relative_for_nonzero_targets_absolute_for_exact_zero_targets"
        ),
        "zero_target_absolute_scale": zero_target_absolute_scale,
        "initializer_regularization": initializer_regularization,
        "zero_barrier_dual_support_oracle": dual_support_report,
        "finite_barrier_homotopy_initializer": finite_homotopy_report,
        "basic_support_reduction": basic_support_reduction,
        "selected_numerical_formulation": selected_formulation,
        "normalized_gas_reduced_primary": normalized_primary_report,
        "normalized_gas_reduced_initializer_portfolio": (
            normalized_initializer_portfolio_report
        ),
        "alternative_basic_support_portfolio": (
            alternative_basic_support_report
        ),
        "structural_zero_reduced_log_rescue": (
            structural_log_rescue_report
        ),
        "linear_amount_physical_audit": {
            key: linear_audit[key]
            for key in (
                "accepted",
                "acceptance_source",
                "optimizer_termination_eligible",
                "physical_root_certified",
                "finite",
                "positive_active_amounts",
                "gas_stationarity_max_abs",
                "active_condensate_driving_max_abs",
                "inactive_condensate_violation_max_abs",
                "budget_scaled_max_abs",
                "total_density_scaled_abs",
            )
        }
        | {
            "attempted": dense_solver_attempted,
            "role": "compatibility_fallback",
            "audit_source": (
                "capacity_scaled_linear_amounts"
                if dense_solver_attempted
                else selected_formulation
            ),
            "local_kkt_passed": linear_local_kkt_passed,
        },
        "selected_physical_audit": {
            key: audit[key]
            for key in (
                "accepted",
                "acceptance_source",
                "optimizer_termination_eligible",
                "physical_root_certified",
                "finite",
                "positive_active_amounts",
                "gas_stationarity_max_abs",
                "active_condensate_driving_max_abs",
                "inactive_condensate_violation_max_abs",
                "budget_scaled_max_abs",
                "total_density_scaled_abs",
            )
        }
        | {"formulation": selected_formulation},
        "reduced_log_domain_fallback": fallback_report,
        "full_condensate_driving": tuple(
            float(value) for value in full_driving.tolist()
        ),
        "element_potential": tuple(
            float(value) for value in current_lambda.tolist()
        ),
        "attempts": tuple(attempts),
    }
    return ZeroBarrierPolishResult(
        accepted=accepted,
        gas_log_amounts=np.asarray(current_q, dtype=np.float64),
        condensate_amounts=np.asarray(current_full_m, dtype=np.float64),
        total_gas_log_amount=float(current_qtot),
        element_potential=np.asarray(current_lambda, dtype=np.float64),
        support_indices=current_support,
        report=report,
    )


def _local_zero_barrier_kkt_failure_reasons(
    report: dict[str, Any],
    *,
    stationarity_tolerance: float,
    budget_tolerance: float,
    total_density_tolerance: float,
) -> tuple[str, ...]:
    """Return failed physical blocks other than inactive-support closure."""

    reasons = []
    if not report["finite"]:
        reasons.append("nonfinite_state")
    if not _zero_barrier_local_root_eligible(
        optimizer_success=bool(report["optimizer_success"]),
        optimizer_status=report.get("optimizer_status"),
        terminal_root_accepted=bool(report["accepted"]),
    ):
        reasons.append("optimizer_failed")
    if not report["positive_active_amounts"]:
        reasons.append("nonpositive_active_amount")
    if report["gas_stationarity_max_abs"] > stationarity_tolerance:
        reasons.append("gas_stationarity")
    if (
        report["active_condensate_driving_max_abs"]
        > stationarity_tolerance
    ):
        reasons.append("active_condensate_stationarity")
    if report["budget_scaled_max_abs"] > budget_tolerance:
        reasons.append("element_budget")
    if report["total_density_scaled_abs"] > total_density_tolerance:
        reasons.append("total_density")
    return tuple(reasons)


def _zero_barrier_initializer_portfolio_summary(
    report: dict[str, Any],
) -> dict[str, int]:
    """Summarize initializer attempts across nested support retries."""

    summary = {
        "regularized_attempt_count": 0,
        "regularized_function_evaluations": 0,
        "unregularized_attempt_count": 0,
        "unregularized_function_evaluations": 0,
        "raw_retry_count": 0,
    }

    def add_portfolio(portfolio: dict[str, Any]) -> None:
        summary["regularized_attempt_count"] += int(
            bool(portfolio.get("regularized_attempted", False))
        )
        summary["unregularized_attempt_count"] += int(
            bool(portfolio.get("unregularized_attempted", False))
        )
        summary["raw_retry_count"] += int(
            bool(portfolio.get("raw_retry_attempted", False))
        )
        for attempt in portfolio.get("attempts", ()):
            initializer = attempt.get("initializer")
            evaluations = int(attempt.get("function_evaluations", 0))
            if initializer == "capacity_regularized":
                summary["regularized_function_evaluations"] += evaluations
            elif initializer == "unregularized":
                summary["unregularized_function_evaluations"] += evaluations

    def add_fallback(fallback: dict[str, Any]) -> None:
        add_portfolio(
            fallback.get(
                "selected_support_normalized_initializer_portfolio", {}
            )
        )
        nested = fallback.get("retry_initializer_diagnostics", {}).get(
            "support_initializer_postselection_fallback"
        )
        if nested:
            add_fallback(nested)

    add_portfolio(
        report.get("normalized_gas_reduced_initializer_portfolio", {})
    )
    fallback = report.get("support_initializer_postselection_fallback")
    if fallback:
        add_fallback(fallback)
    return summary


def _zero_barrier_report_function_evaluations(
    report: dict[str, Any],
) -> tuple[int, int]:
    """Return dense and gas-eliminated evaluation counts for one solve pass."""

    def discarded_initializer_evaluations(
        portfolio_report: dict[str, Any],
    ) -> int:
        return sum(
            int(attempt.get("function_evaluations", 0))
            for discarded in portfolio_report.get(
                "discarded_solve_reports", ()
            )
            for attempt in discarded.get("solve", {}).get("attempts", ())
        )

    def alternative_support_evaluations(
        portfolio_report: dict[str, Any],
    ) -> int:
        return sum(
            int(attempt.get("function_evaluations", 0))
            for candidate in portfolio_report.get("solve_attempts", ())
            for attempt in candidate.get("solve", {}).get("attempts", ())
        )

    linear = sum(
        int(attempt.get("function_evaluations", 0))
        for attempt in report.get("attempts", ())
    )
    reduced = sum(
        int(attempt.get("function_evaluations", 0))
        for attempt in report.get(
            "normalized_gas_reduced_primary", {}
        ).get("attempts", ())
    )
    reduced += discarded_initializer_evaluations(
        report.get(
            "normalized_gas_reduced_initializer_portfolio", {}
        )
    )
    reduced += alternative_support_evaluations(
        report.get("alternative_basic_support_portfolio", {})
    )
    reduced += sum(
        int(round_report.get("function_evaluations", 0))
        for round_report in report.get(
            "finite_barrier_homotopy_initializer", {}
        ).get("rounds", ())
    )
    reduced += int(
        report.get("zero_barrier_dual_support_oracle", {}).get(
            "function_evaluations", 0
        )
    )
    structural_report = report.get(
        "structural_zero_reduced_log_rescue", {}
    )
    for solve_key in ("normalized_linear_solve", "solve"):
        structural_solve = structural_report.get(solve_key, {})
        reduced += sum(
            int(attempt.get("function_evaluations", 0))
            for attempt in structural_solve.get("attempts", ())
        )

    def retry_initializer_evaluations(
        fallback_report: dict[str, Any],
    ) -> int:
        retry_reduced = sum(
            int(attempt.get("function_evaluations", 0))
            for attempt in fallback_report.get(
                "selected_support_normalized_solve", {}
            ).get("attempts", ())
        )
        retry_reduced += discarded_initializer_evaluations(
            fallback_report.get(
                "selected_support_normalized_initializer_portfolio", {}
            )
        )
        retry_reduced += alternative_support_evaluations(
            fallback_report.get(
                "selected_support_alternative_basic_support_portfolio", {}
            )
        )
        discarded_structural = fallback_report.get(
            "selected_support_structural_zero_solve", {}
        )
        for solve_key in ("normalized_linear_solve", "solve"):
            structural_solve = discarded_structural.get(solve_key, {})
            retry_reduced += sum(
                int(attempt.get("function_evaluations", 0))
                for attempt in structural_solve.get("attempts", ())
            )
        retry_diagnostics = fallback_report.get(
            "retry_initializer_diagnostics", {}
        )
        retry_reduced += int(
            (
                retry_diagnostics.get(
                    "zero_barrier_dual_support_oracle"
                )
                or {}
            ).get("function_evaluations", 0)
        )
        retry_reduced += sum(
            int(round_report.get("function_evaluations", 0))
            for round_report in (
                retry_diagnostics.get(
                    "finite_barrier_homotopy_initializer"
                )
                or {}
            ).get("rounds", ())
        )
        nested_fallback = retry_diagnostics.get(
            "support_initializer_postselection_fallback"
        )
        if nested_fallback:
            retry_reduced += retry_initializer_evaluations(nested_fallback)
        return retry_reduced

    initializer_retry = report.get(
        "support_initializer_postselection_fallback", {}
    )
    reduced += retry_initializer_evaluations(initializer_retry)
    fallback = report.get("reduced_log_domain_fallback", {})
    for node in fallback.get("nodes", ()):
        solve_report = node.get("solve", {})
        reduced += sum(
            int(attempt.get("function_evaluations", 0))
            for attempt in solve_report.get("attempts", ())
        )
    return linear, reduced


def polish_zero_barrier_active_support(
    *,
    gas_formula_matrix: Any,
    condensate_formula_matrix_full: Any,
    target_inventory: Any,
    gas_standard_source: Any,
    condensate_standard_source_full: Any,
    gas_log_amounts_init: Any,
    condensate_amounts_init: Any,
    total_gas_log_amount_init: Any,
    element_potential_init: Any,
    support_indices: Sequence[int],
    condensate_valid_mask: Any | None = None,
    stationarity_tolerance: float = 1.0e-8,
    budget_tolerance: float = 1.0e-8,
    total_density_tolerance: float = 1.0e-8,
    support_closure_tolerance: float = 1.0e-8,
    budget_relative_floor: float = 1.0e-6,
    max_function_evaluations: int = 400,
) -> ZeroBarrierPolishResult:
    """Refine and close an exact zero-barrier condensate support.

    Each solve pass may remove phases with non-positive exact amounts.  When
    every local physical block passes but an inactive phase remains favorable,
    the most negative temperature-valid phase is added and the exact solve is
    repeated from the refined state.  The deterministic search caches locally
    valid states, rejects addition edges that return to a visited state, and
    backtracks to untried candidates.  It fails closed on local KKT failure or
    exhausted search and work limits.
    """

    original_support = tuple(int(index) for index in support_indices)
    current_arguments = {
        "gas_formula_matrix": gas_formula_matrix,
        "condensate_formula_matrix_full": condensate_formula_matrix_full,
        "target_inventory": target_inventory,
        "gas_standard_source": gas_standard_source,
        "condensate_standard_source_full": (
            condensate_standard_source_full
        ),
        "gas_log_amounts_init": gas_log_amounts_init,
        "condensate_amounts_init": condensate_amounts_init,
        "total_gas_log_amount_init": total_gas_log_amount_init,
        "element_potential_init": element_potential_init,
        "support_indices": original_support,
        "condensate_valid_mask": condensate_valid_mask,
        "stationarity_tolerance": stationarity_tolerance,
        "budget_tolerance": budget_tolerance,
        "total_density_tolerance": total_density_tolerance,
        "support_closure_tolerance": support_closure_tolerance,
        "budget_relative_floor": budget_relative_floor,
        "max_function_evaluations": max_function_evaluations,
        "reduce_initial_support": True,
    }
    condensate_count = int(
        np.asarray(condensate_standard_source_full).size
    )
    valid_mask = (
        np.ones(condensate_count, dtype=bool)
        if condensate_valid_mask is None
        else np.asarray(condensate_valid_mask, dtype=bool)
    )
    closure_target = np.asarray(target_inventory, dtype=np.float64)
    closure_ac_full = np.asarray(
        condensate_formula_matrix_full, dtype=np.float64
    )
    closure_hcond_full = np.asarray(
        condensate_standard_source_full, dtype=np.float64
    )
    closure_positive_target = np.abs(
        closure_target[closure_target != 0.0]
    )
    closure_inventory_scale = (
        float(np.max(closure_positive_target))
        if closure_positive_target.size
        else 1.0
    )
    closure_zero_target_scale = max(
        float(budget_relative_floor),
        np.finfo(np.float64).eps * closure_inventory_scale,
        1.0e-300,
    )
    closure_budget_scale = np.reciprocal(
        np.where(
            closure_target != 0.0,
            np.maximum(np.abs(closure_target), 1.0e-300),
            closure_zero_target_scale,
        )
    )
    round_limit = _ACTIVE_SET_CLOSURE_ROUND_LIMIT
    evaluation_limit = (
        max(1, int(max_function_evaluations))
        * _ACTIVE_SET_CLOSURE_ROUND_LIMIT
    )
    evaluation_budget = _FunctionEvaluationBudget(evaluation_limit)
    current_arguments["function_evaluation_budget"] = evaluation_budget
    visited_inputs: list[tuple[int, ...]] = []
    visited_output_keys: set[tuple[int, ...]] = set()
    visited_outputs: list[tuple[int, ...]] = []
    node_results: dict[tuple[int, ...], ZeroBarrierPolishResult] = {}
    node_incoming_edges: dict[
        tuple[int, ...], tuple[tuple[int, ...], int] | None
    ] = {}
    node_stack: list[tuple[int, ...]] = []
    added_support_indices: list[int] = []
    blacklisted_addition_edges: list[tuple[tuple[int, ...], int]] = []
    blacklisted_addition_edge_set: set[
        tuple[tuple[int, ...], int]
    ] = set()
    pending_addition_edge: tuple[tuple[int, ...], int] | None = None
    round_reports: list[dict[str, Any]] = []
    cumulative_linear_evaluations = 0
    cumulative_reduced_evaluations = 0
    termination_reason = "round_limit_reached"
    final_result: ZeroBarrierPolishResult | None = None
    initial_basic_support_reduction: dict[str, Any] | None = None
    initial_alternative_basic_support: dict[str, Any] | None = None
    initial_dual_support_oracle: dict[str, Any] | None = None
    initial_finite_homotopy: dict[str, Any] | None = None

    for round_index in range(round_limit):
        input_support = tuple(current_arguments["support_indices"])
        attempted_addition_edge = pending_addition_edge
        pending_addition_edge = None
        visited_inputs.append(input_support)
        evaluations_before_round = evaluation_budget.used
        result = _polish_zero_barrier_support_once(**current_arguments)
        evaluations_in_round = (
            evaluation_budget.used - evaluations_before_round
        )
        current_arguments["reduce_initial_support"] = False
        final_result = result
        output_support = tuple(result.support_indices)
        report = result.report
        if (
            initial_basic_support_reduction is None
            and "basic_support_reduction" in report
        ):
            initial_basic_support_reduction = dict(
                report["basic_support_reduction"]
            )
        if (
            initial_alternative_basic_support is None
            and report.get(
                "alternative_basic_support_portfolio", {}
            ).get("enabled", False)
        ):
            initial_alternative_basic_support = dict(
                report["alternative_basic_support_portfolio"]
            )
        if (
            initial_dual_support_oracle is None
            and report.get("zero_barrier_dual_support_oracle", {}).get(
                "enabled", False
            )
        ):
            initial_dual_support_oracle = dict(
                report["zero_barrier_dual_support_oracle"]
            )
        if (
            initial_finite_homotopy is None
            and report.get("finite_barrier_homotopy_initializer", {}).get(
                "enabled", False
            )
        ):
            initial_finite_homotopy = dict(
                report["finite_barrier_homotopy_initializer"]
            )
        linear_evaluations, reduced_evaluations = (
            _zero_barrier_report_function_evaluations(report)
        )
        initializer_portfolio_summary = (
            _zero_barrier_initializer_portfolio_summary(report)
        )
        if linear_evaluations + reduced_evaluations != evaluations_in_round:
            raise RuntimeError(
                "Zero-barrier diagnostics do not match the shared "
                "function-evaluation budget."
            )
        cumulative_linear_evaluations += linear_evaluations
        cumulative_reduced_evaluations += reduced_evaluations
        failure_reasons = _local_zero_barrier_kkt_failure_reasons(
            report,
            stationarity_tolerance=stationarity_tolerance,
            budget_tolerance=budget_tolerance,
            total_density_tolerance=total_density_tolerance,
        )
        output_key = tuple(sorted(output_support))
        added_index: int | None = None
        added_driving: float | None = None
        rejected_added_index: int | None = None
        rejected_addition_edge: tuple[tuple[int, ...], int] | None = None
        addition_base_support: tuple[int, ...] | None = None
        addition_pivot_report: dict[str, Any] | None = None
        backtracked_supports: list[tuple[int, ...]] = []
        backtracked_edges: list[tuple[tuple[int, ...], int]] = []
        action = "stop"
        stop = True

        if result.accepted:
            visited_output_keys.add(output_key)
            visited_outputs.append(output_support)
            termination_reason = "accepted"
            action = "accepted"
        elif evaluation_budget.remaining <= 0:
            visited_output_keys.add(output_key)
            visited_outputs.append(output_support)
            termination_reason = "function_evaluation_limit_reached"
            action = "stop_evaluation_limit"
        elif failure_reasons:
            visited_output_keys.add(output_key)
            visited_outputs.append(output_support)
            termination_reason = "local_kkt_failed"
            action = "stop_local_kkt_failure"
        else:
            if output_key in visited_output_keys:
                if attempted_addition_edge is None:
                    termination_reason = "support_cycle_detected"
                    action = "stop_cycle"
                else:
                    rejected_added_index = attempted_addition_edge[1]
                    rejected_addition_edge = attempted_addition_edge
                    if (
                        attempted_addition_edge
                        not in blacklisted_addition_edge_set
                    ):
                        blacklisted_addition_edge_set.add(
                            attempted_addition_edge
                        )
                        blacklisted_addition_edges.append(
                            attempted_addition_edge
                        )
                        backtracked_edges.append(attempted_addition_edge)
            else:
                visited_output_keys.add(output_key)
                visited_outputs.append(output_support)
                node_results[output_key] = result
                node_incoming_edges[output_key] = attempted_addition_edge
                node_stack.append(output_key)

            if action != "stop_cycle":
                selected_base_result: ZeroBarrierPolishResult | None = None
                selected_base_key: tuple[int, ...] | None = None
                while node_stack:
                    base_key = node_stack[-1]
                    base_result = node_results[base_key]
                    base_support = tuple(base_result.support_indices)
                    base_driving = np.asarray(
                        base_result.report["full_condensate_driving"],
                        dtype=np.float64,
                    )
                    active_mask = np.zeros(condensate_count, dtype=bool)
                    if base_support:
                        active_mask[
                            np.asarray(base_support, dtype=np.int64)
                        ] = True
                    all_candidates = tuple(
                        sorted(
                            (
                                int(index)
                                for index in np.flatnonzero(
                                    valid_mask
                                    & ~active_mask
                                    & (
                                        base_driving
                                        < -float(support_closure_tolerance)
                                    )
                                ).tolist()
                            ),
                            key=lambda index: (
                                float(base_driving[index]),
                                index,
                            ),
                        )
                    )
                    selected_index: int | None = None
                    for candidate_index in all_candidates:
                        candidate_edge = (base_key, candidate_index)
                        if (
                            candidate_edge
                            in blacklisted_addition_edge_set
                        ):
                            continue
                        candidate_support = base_support + (candidate_index,)
                        candidate_key = tuple(sorted(candidate_support))
                        if candidate_key in visited_output_keys:
                            blacklisted_addition_edge_set.add(candidate_edge)
                            blacklisted_addition_edges.append(candidate_edge)
                            backtracked_edges.append(candidate_edge)
                            continue
                        selected_index = candidate_index
                        break

                    if selected_index is not None:
                        added_index = selected_index
                        added_driving = float(base_driving[selected_index])
                        addition_base_support = base_support
                        selected_base_result = base_result
                        selected_base_key = base_key
                        break
                    if not all_candidates:
                        termination_reason = (
                            "inactive_violation_without_addable_phase"
                        )
                        action = "stop_no_addable_phase"
                        break

                    exhausted_key = node_stack.pop()
                    exhausted_result = node_results[exhausted_key]
                    backtracked_supports.append(
                        tuple(exhausted_result.support_indices)
                    )
                    incoming_edge = node_incoming_edges[exhausted_key]
                    if incoming_edge is None:
                        termination_reason = "support_cycle_detected"
                        action = "stop_cycle"
                        break
                    if incoming_edge not in blacklisted_addition_edge_set:
                        blacklisted_addition_edge_set.add(incoming_edge)
                        blacklisted_addition_edges.append(incoming_edge)
                        backtracked_edges.append(incoming_edge)

                if selected_base_result is not None:
                    unpivoted_next_support = (
                        tuple(selected_base_result.support_indices)
                        + (int(added_index),)
                    )
                    addition_pivot = _pivot_rank_one_support_addition(
                        condensate_formula_matrix_full=closure_ac_full,
                        condensate_standard_source_full=(
                            closure_hcond_full
                        ),
                        target_inventory=closure_target,
                        condensate_amounts=(
                            selected_base_result.condensate_amounts
                        ),
                        support_indices=(
                            selected_base_result.support_indices
                        ),
                        added_support_index=int(added_index),
                        budget_scale=closure_budget_scale,
                    )
                    addition_pivot_report = dict(
                        addition_pivot["report"]
                    )
                    next_support = tuple(
                        addition_pivot["support_indices"]
                    )
                    next_condensate_amounts = (
                        np.asarray(
                            addition_pivot["condensate_amounts"],
                            dtype=np.float64,
                        )
                        if addition_pivot["applied"]
                        else selected_base_result.condensate_amounts
                    )
                    if not addition_pivot["applied"]:
                        next_support = unpivoted_next_support
                    if evaluation_budget.remaining <= 0:
                        termination_reason = (
                            "function_evaluation_limit_reached"
                        )
                        action = "stop_evaluation_limit"
                    elif round_index + 1 >= round_limit:
                        termination_reason = "round_limit_reached"
                        action = "stop_round_limit"
                    else:
                        action = "add_inactive_phase"
                        stop = False
                        added_support_indices.append(int(added_index))
                        pending_addition_edge = (
                            tuple(selected_base_key),
                            int(added_index),
                        )
                        current_arguments.update(
                            gas_log_amounts_init=(
                                selected_base_result.gas_log_amounts
                            ),
                            condensate_amounts_init=(
                                next_condensate_amounts
                            ),
                            total_gas_log_amount_init=(
                                selected_base_result.total_gas_log_amount
                            ),
                            element_potential_init=(
                                selected_base_result.element_potential
                            ),
                            support_indices=next_support,
                        )

        round_reports.append(
            {
                "round_index": round_index,
                "input_support_indices": input_support,
                "output_support_indices": output_support,
                "dropped_support_indices": tuple(
                    index
                    for index in input_support
                    if index not in set(output_support)
                ),
                "accepted": bool(result.accepted),
                "local_kkt_passed": not failure_reasons,
                "local_kkt_failure_reasons": failure_reasons,
                "inactive_condensate_violation_max_abs": float(
                    report["inactive_condensate_violation_max_abs"]
                ),
                "action": action,
                "added_support_index": added_index,
                "added_support_driving": added_driving,
                "addition_base_support_indices": addition_base_support,
                "rank_one_simplex_pivot": addition_pivot_report,
                "rejected_added_support_index": rejected_added_index,
                "rejected_addition_edge": rejected_addition_edge,
                "backtracked_support_indices": tuple(backtracked_supports),
                "backtracked_addition_edges": tuple(backtracked_edges),
                "selected_numerical_formulation": report[
                    "selected_numerical_formulation"
                ],
                "selected_normalized_initializer": report.get(
                    "normalized_gas_reduced_initializer_portfolio", {}
                ).get("selected_initializer"),
                "alternative_basic_support_attempted": bool(
                    report.get(
                        "alternative_basic_support_portfolio", {}
                    ).get("attempted", False)
                ),
                "alternative_basic_support_selected": bool(
                    report.get(
                        "alternative_basic_support_portfolio", {}
                    ).get("selected_support_indices")
                ),
                "alternative_basic_support_indices": report.get(
                    "alternative_basic_support_portfolio", {}
                ).get("selected_support_indices"),
                "regularized_normalized_initializer_attempted": bool(
                    initializer_portfolio_summary[
                        "regularized_attempt_count"
                    ]
                ),
                "raw_normalized_initializer_retry_attempted": bool(
                    initializer_portfolio_summary["raw_retry_count"]
                ),
                "regularized_normalized_initializer_attempt_count": (
                    initializer_portfolio_summary[
                        "regularized_attempt_count"
                    ]
                ),
                "regularized_normalized_initializer_function_evaluations": (
                    initializer_portfolio_summary[
                        "regularized_function_evaluations"
                    ]
                ),
                "unregularized_normalized_initializer_attempt_count": (
                    initializer_portfolio_summary[
                        "unregularized_attempt_count"
                    ]
                ),
                "unregularized_normalized_initializer_function_evaluations": (
                    initializer_portfolio_summary[
                        "unregularized_function_evaluations"
                    ]
                ),
                "raw_normalized_initializer_retry_count": (
                    initializer_portfolio_summary["raw_retry_count"]
                ),
                "linear_function_evaluations": linear_evaluations,
                "reduced_function_evaluations": reduced_evaluations,
                "function_evaluations": (
                    linear_evaluations + reduced_evaluations
                ),
            }
        )
        if stop:
            break

    if final_result is None:
        raise RuntimeError("Zero-barrier active-set closure did not run.")
    final_report = dict(final_result.report)
    final_support = tuple(final_result.support_indices)
    final_report["initial_support_indices"] = original_support
    final_report["final_support_indices"] = final_support
    final_report["dropped_support_indices"] = tuple(
        index for index in original_support if index not in set(final_support)
    )
    if initial_basic_support_reduction is not None:
        final_report["basic_support_reduction"] = (
            initial_basic_support_reduction
        )
    if initial_alternative_basic_support is not None:
        final_report["alternative_basic_support_portfolio"] = (
            initial_alternative_basic_support
        )
    if initial_dual_support_oracle is not None:
        final_report["zero_barrier_dual_support_oracle"] = (
            initial_dual_support_oracle
        )
    if initial_finite_homotopy is not None:
        final_report["finite_barrier_homotopy_initializer"] = (
            initial_finite_homotopy
        )
    final_report["exact_active_set_closure"] = {
        "schema": "exogibbs_zero_barrier_exact_active_set_closure_v2",
        "attempted": True,
        "search_strategy": "bounded_depth_first",
        "addition_attempted": bool(added_support_indices),
        "accepted": bool(final_result.accepted),
        "termination_reason": termination_reason,
        "round_limit": round_limit,
        "function_evaluation_limit": evaluation_limit,
        "round_count": len(round_reports),
        "initial_support_indices": original_support,
        "final_support_indices": final_support,
        "visited_supports": tuple(visited_inputs),
        "visited_output_supports": tuple(visited_outputs),
        "added_support_indices": tuple(added_support_indices),
        "blacklisted_addition_edges": tuple(blacklisted_addition_edges),
        "cumulative_linear_function_evaluations": (
            cumulative_linear_evaluations
        ),
        "cumulative_reduced_function_evaluations": (
            cumulative_reduced_evaluations
        ),
        "cumulative_function_evaluations": (
            cumulative_linear_evaluations
            + cumulative_reduced_evaluations
        ),
        "rounds": tuple(round_reports),
    }
    return ZeroBarrierPolishResult(
        accepted=final_result.accepted,
        gas_log_amounts=final_result.gas_log_amounts,
        condensate_amounts=final_result.condensate_amounts,
        total_gas_log_amount=final_result.total_gas_log_amount,
        element_potential=final_result.element_potential,
        support_indices=final_support,
        report=final_report,
    )


__all__ = (
    "ZeroBarrierPolishResult",
    "polish_zero_barrier_active_support",
)
