"""Host-side zero-barrier refinement for a converged active support."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp


_INITIALIZER_CAPACITY_FRACTION = float(np.sqrt(np.finfo(np.float64).eps))
_REDUCED_SUPPORT_NODE_LIMIT = 32


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
    stationarity_tolerance: float,
    budget_tolerance: float,
    total_density_tolerance: float,
    support_closure_tolerance: float,
) -> dict[str, Any]:
    """Audit one candidate independently of its numerical formulation."""

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
    budget_residual_scaled = budget_scale * (
        gas_formula_matrix @ gas
        + condensate_formula_matrix_full @ amounts
        - target_inventory
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
    accepted = bool(
        finite
        and optimizer_success
        and positive_active_amounts
        and gas_stationarity_norm <= stationarity_tolerance
        and active_driving_norm <= stationarity_tolerance
        and inactive_violation_norm <= support_closure_tolerance
        and budget_norm <= budget_tolerance
        and total_norm <= total_density_tolerance
    )
    return {
        "accepted": accepted,
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
    current_q = np.asarray(gas_log_amounts_init, dtype=np.float64).copy()
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

        def budget_log_terms(
            element_index: int,
            *,
            logits: np.ndarray,
            y: float,
            v: np.ndarray,
        ) -> tuple[np.ndarray, tuple[tuple[str, int], ...]]:
            terms: list[float] = []
            owners: list[tuple[str, int]] = []
            for gas_index in np.flatnonzero(ag[element_index] > 0.0):
                terms.append(
                    float(
                        y
                        + logits[gas_index]
                        + np.log(ag[element_index, gas_index])
                    )
                )
                owners.append(("gas", int(gas_index)))
            for local_index in np.flatnonzero(ac[element_index] > 0.0):
                terms.append(
                    float(
                        np.log(ac[element_index, local_index])
                        + log_kappa[local_index]
                        + v[local_index]
                    )
                )
                owners.append(("condensate", int(local_index)))
            return np.asarray(terms, dtype=np.float64), tuple(owners)

        def residual(values: np.ndarray) -> np.ndarray:
            lambda_, y, v = unpack(values)
            logits = ag.T @ lambda_ - gamma
            budget_residuals = np.empty(element_count, dtype=np.float64)
            for element_index in range(element_count):
                terms, _owners = budget_log_terms(
                    element_index,
                    logits=logits,
                    y=y,
                    v=v,
                )
                budget_residuals[element_index] = (
                    logsumexp(terms) - log_beta[element_index]
                )
            return np.concatenate(
                [
                    hcond - ac.T @ lambda_,
                    np.asarray([logsumexp(logits)], dtype=np.float64),
                    budget_residuals,
                ]
            )

        def jacobian(values: np.ndarray) -> np.ndarray:
            lambda_, y, v = unpack(values)
            logits = ag.T @ lambda_ - gamma
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
            for element_index in range(element_count):
                terms, owners = budget_log_terms(
                    element_index,
                    logits=logits,
                    y=y,
                    v=v,
                )
                weights = np.exp(terms - logsumexp(terms))
                row = budget_row_start + element_index
                for weight, (kind, local_index) in zip(weights, owners):
                    if kind == "gas":
                        matrix[row, :element_count] += (
                            weight * ag[:, local_index]
                        )
                        matrix[row, y_column] += weight
                    else:
                        matrix[row, v_column_start + local_index] += weight
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
                np.zeros(len(current_support), dtype=np.float64),
            ]
        )
        variable_scale = np.clip(np.maximum(np.abs(x0), 1.0), 1.0, 100.0)
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
                max_nfev=int(max_function_evaluations),
            )
        except (FloatingPointError, OverflowError, ValueError) as error:
            attempts.append(
                {
                    "support_indices": current_support,
                    "optimizer_success": False,
                    "failure_reason": f"{type(error).__name__}: {error}",
                }
            )
            break
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
            stationarity_tolerance=stationarity_tolerance,
            budget_tolerance=budget_tolerance,
            total_density_tolerance=total_density_tolerance,
            support_closure_tolerance=support_closure_tolerance,
        )
        accepted = bool(audit["accepted"] and not at_lower_bound.size)
        log_residual = residual(optimization.x)
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
                float(value) for value in np.exp(v).tolist()
            ),
            "active_phase_at_lower_bound": bool(at_lower_bound.size),
            "lower_bound_support_indices": lower_bound_support_indices,
            "physical_audit_accepted": bool(audit["accepted"]),
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
            "audit": audit,
        }
        if accepted:
            break
        if not allow_greedy_drop or not current_support:
            break
        if not at_lower_bound.size:
            break
        local_drop = int(at_lower_bound[np.argmin(v[at_lower_bound])])
        dropped_index = current_support[local_drop]
        dropped.append(dropped_index)
        current_support = tuple(
            index for index in current_support if index != dropped_index
        )
        current_q = q
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

    while queue and len(visited) < int(max_support_nodes):
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
    """Refine one fixed support against the physical zero-barrier KKT system.

    Gas log amounts, active condensate amounts, total gas, and element
    potentials are solved together.  Condensate amounts use capacity-scaled
    linear coordinates so that a phase with a negative exact amount can be
    removed explicitly.  A returned state is accepted only after all physical
    stationarity, inventory, total-density, positivity, and inactive-support
    closure checks pass.
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
    (
        q_initial,
        qtot_initial,
        lambda_initial,
        initializer_regularization,
    ) = _capacity_regularized_initializer(
        gas_formula_matrix=ag,
        target_inventory=target,
        gas_standard_source=gamma,
        gas_log_amounts=q_initial,
        total_gas_log_amount=qtot_initial,
        element_potential=lambda_initial,
    )
    current_support = support_initial
    current_q = q_initial.copy()
    current_qtot = qtot_initial
    current_lambda = lambda_initial.copy()
    current_full_m = full_m_initial.copy()
    dropped: list[int] = []
    attempts: list[dict[str, Any]] = []
    last_optimizer_success = False
    last_optimizer_status = 0
    last_optimizer_message = "not run"
    last_nfev = 0

    for _drop_round in range(len(support_initial) + 1):
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
                max_nfev=int(max_function_evaluations),
            )
        except (FloatingPointError, OverflowError, ValueError) as error:
            last_optimizer_success = False
            last_optimizer_status = 0
            last_optimizer_message = f"{type(error).__name__}: {error}"
            last_nfev = 0
            attempts.append(
                {
                    "support_indices": current_support,
                    "optimizer_success": False,
                    "optimizer_status": last_optimizer_status,
                    "optimizer_message": last_optimizer_message,
                    "function_evaluations": 0,
                    "failure_reason": "linear_amount_solver_exception",
                }
            )
            break
        q, active_amounts, qtot, lambda_ = unpack(optimization.x)
        last_optimizer_success = bool(optimization.success)
        last_optimizer_status = int(optimization.status)
        last_optimizer_message = str(optimization.message)
        last_nfev = int(optimization.nfev)
        attempts.append(
            {
                "support_indices": current_support,
                "optimizer_success": last_optimizer_success,
                "optimizer_status": last_optimizer_status,
                "optimizer_message": last_optimizer_message,
                "function_evaluations": last_nfev,
                "cost": float(optimization.cost),
                "optimality": float(optimization.optimality),
                "active_condensate_amounts": tuple(
                    float(value) for value in active_amounts.tolist()
                ),
            }
        )
        nonpositive = np.flatnonzero(active_amounts <= 0.0)
        if nonpositive.size:
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
            current_q = q
            current_qtot = float(qtot)
            current_lambda = lambda_
            current_full_m = np.zeros(condensate_count, dtype=np.float64)
            for index, amount in zip(active.tolist(), active_amounts.tolist()):
                if index != dropped_index:
                    current_full_m[index] = max(float(amount), 1.0e-300)
            continue
        current_q = q
        current_qtot = float(qtot)
        current_lambda = lambda_
        current_full_m = np.zeros(condensate_count, dtype=np.float64)
        current_full_m[active] = active_amounts
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
        stationarity_tolerance=stationarity_tolerance,
        budget_tolerance=budget_tolerance,
        total_density_tolerance=total_density_tolerance,
        support_closure_tolerance=support_closure_tolerance,
    )
    fallback_eligible, fallback_reason = _reduced_log_domain_eligibility(
        gas_formula_matrix=ag,
        condensate_formula_matrix_full=ac_full,
        target_inventory=target,
        support_indices=support_initial,
    )
    fallback_report: dict[str, Any] = {
        "schema": "exogibbs_zero_barrier_reduced_support_search_v1",
        "eligible": fallback_eligible,
        "attempted": False,
        "accepted": False,
        "skip_reason": (
            "linear_amount_physical_audit_accepted"
            if linear_audit["accepted"]
            else fallback_reason
        ),
    }
    selected_formulation = "capacity_scaled_linear_amounts"
    if not linear_audit["accepted"] and fallback_eligible:
        reduced = _solve_reduced_log_domain_support_branches(
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
            condensate_valid_mask=valid_mask,
            budget_scale=budget_scale,
            stationarity_tolerance=stationarity_tolerance,
            budget_tolerance=budget_tolerance,
            total_density_tolerance=total_density_tolerance,
            support_closure_tolerance=support_closure_tolerance,
            max_function_evaluations=max_function_evaluations,
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
        "polish_schema": "exogibbs_zero_barrier_active_support_polish_v1",
        "attempted": True,
        "accepted": accepted,
        "initial_support_indices": support_initial,
        "final_support_indices": current_support,
        "dropped_support_indices": tuple(dropped),
        "optimizer_success": last_optimizer_success,
        "optimizer_status": last_optimizer_status,
        "optimizer_message": last_optimizer_message,
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
        "selected_numerical_formulation": selected_formulation,
        "linear_amount_physical_audit": {
            key: linear_audit[key]
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


__all__ = (
    "ZeroBarrierPolishResult",
    "polish_zero_barrier_active_support",
)
