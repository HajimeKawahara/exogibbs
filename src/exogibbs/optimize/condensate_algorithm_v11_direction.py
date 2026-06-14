"""Diagnostic direction helpers for algorithm-v1.1 condensate PD-IPM probes.

These helpers are explicit import only. They do not call production solvers,
FastChem4, pyfastchem, or preset/default wiring.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class AlgorithmV11Direction:
    """Primal-dual diagnostic direction for algorithm-v1.1 state variables."""

    delta_q: np.ndarray
    delta_r: np.ndarray
    delta_lambda: np.ndarray
    delta_rho: np.ndarray
    delta_qtot: float
    direction_kind: str
    diagnostic_only: bool = True
    production_behavior_change: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["delta_q"] = self.delta_q.tolist()
        payload["delta_r"] = self.delta_r.tolist()
        payload["delta_lambda"] = self.delta_lambda.tolist()
        payload["delta_rho"] = self.delta_rho.tolist()
        return payload


def _as_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def _as_matrix(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def build_linear_budget_total_density_restoration_direction(
    *,
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond_active: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    external_condensate_budget: Sequence[float] | None = None,
    q: Sequence[float],
    r: Sequence[float],
    lambda_size: int,
    rho_size: int,
    qtot: float,
) -> AlgorithmV11Direction:
    """Build a minimum-norm linearized budget and total-density restoration direction."""

    ag = _as_matrix(formula_matrix, "formula_matrix")
    ac = _as_matrix(formula_matrix_cond_active, "formula_matrix_cond_active")
    target = _as_vector(element_inventory_target, "element_inventory_target")
    external_budget = (
        np.zeros_like(target, dtype=np.float64)
        if external_condensate_budget is None
        else _as_vector(external_condensate_budget, "external_condensate_budget")
    )
    q_array = _as_vector(q, "q")
    r_array = _as_vector(r, "r")
    qtot_value = float(qtot)
    if not np.isfinite(qtot_value):
        raise ValueError("qtot must be finite.")
    if ag.shape[0] != ac.shape[0] or ag.shape[0] != target.shape[0]:
        raise ValueError("formula matrices and element_inventory_target row counts must match.")
    if external_budget.shape[0] != target.shape[0]:
        raise ValueError("external_condensate_budget length must match element rows.")
    if ag.shape[1] != q_array.shape[0]:
        raise ValueError("formula_matrix columns must match q.")
    if ac.shape[1] != r_array.shape[0]:
        raise ValueError("formula_matrix_cond_active columns must match r.")
    if lambda_size < 0 or rho_size < 0:
        raise ValueError("lambda_size and rho_size must be non-negative.")

    n = np.exp(q_array)
    m = np.exp(r_array)
    ntot = float(np.exp(qtot_value))
    budget = ag @ n + ac @ m + external_budget - target
    total_density = float(np.sum(n) - ntot)
    jac_budget = np.concatenate(
        [ag * n[None, :], ac * m[None, :], np.zeros((ag.shape[0], 1))],
        axis=1,
    )
    jac_total = np.concatenate([n, np.zeros_like(m), np.array([-ntot])])[None, :]
    jacobian = np.concatenate([jac_budget, jac_total], axis=0)
    rhs = -np.concatenate([budget, np.array([total_density])])
    direction, *_ = np.linalg.lstsq(jacobian, rhs, rcond=None)
    delta_q = direction[: q_array.shape[0]]
    delta_r = direction[q_array.shape[0] : q_array.shape[0] + r_array.shape[0]]
    delta_qtot = float(direction[-1])
    return AlgorithmV11Direction(
        delta_q=delta_q,
        delta_r=delta_r,
        delta_lambda=np.zeros(int(lambda_size), dtype=np.float64),
        delta_rho=np.zeros(int(rho_size), dtype=np.float64),
        delta_qtot=delta_qtot,
        direction_kind="linear_budget_total_density_restoration_direction",
    )


def build_linear_budget_total_density_amount_gas_direction(
    *,
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond_active: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    external_condensate_budget: Sequence[float] | None = None,
    gas_stationarity_source: Sequence[float],
    q: Sequence[float],
    r: Sequence[float],
    lam: Sequence[float],
    rho: Sequence[float],
    qtot: float,
    target_direction: AlgorithmV11Direction | None = None,
    budget_weight: float = 10.0,
    total_density_weight: float = 10.0,
    amount_gas_weight: float = 1.0,
    target_direction_weight: float = 1.0e-2,
    target_direction_component_mode: str = "all",
    budget_row_scaling_policy: str = "absolute",
    relative_budget_floor_factor: float = 1.0e-300,
    max_abs_delta_q: float = 2.0,
    max_abs_delta_r: float = 2.0,
    max_abs_delta_lambda: float = 100.0,
) -> AlgorithmV11Direction:
    """Build a linearized joint budget, total-density, and amount-gas direction."""

    ag = _as_matrix(formula_matrix, "formula_matrix")
    ac = _as_matrix(formula_matrix_cond_active, "formula_matrix_cond_active")
    target = _as_vector(element_inventory_target, "element_inventory_target")
    external_budget = (
        np.zeros_like(target, dtype=np.float64)
        if external_condensate_budget is None
        else _as_vector(external_condensate_budget, "external_condensate_budget")
    )
    gas_source = _as_vector(gas_stationarity_source, "gas_stationarity_source")
    q_array = _as_vector(q, "q")
    r_array = _as_vector(r, "r")
    lam_array = _as_vector(lam, "lam")
    rho_array = _as_vector(rho, "rho")
    qtot_value = float(qtot)
    if not np.isfinite(qtot_value):
        raise ValueError("qtot must be finite.")
    if ag.shape[0] != ac.shape[0] or ag.shape[0] != target.shape[0]:
        raise ValueError("formula matrices and element_inventory_target row counts must match.")
    if external_budget.shape[0] != target.shape[0]:
        raise ValueError("external_condensate_budget length must match element rows.")
    if ag.shape[1] != q_array.shape[0] or gas_source.shape[0] != q_array.shape[0]:
        raise ValueError("gas arrays must match q.")
    if ac.shape[1] != r_array.shape[0]:
        raise ValueError("formula_matrix_cond_active columns must match r.")
    if ag.shape[0] != lam_array.shape[0]:
        raise ValueError("lambda length must match formula_matrix rows.")

    weights = (
        float(budget_weight),
        float(total_density_weight),
        float(amount_gas_weight),
        float(target_direction_weight),
    )
    if any((not np.isfinite(weight) or weight < 0.0) for weight in weights):
        raise ValueError("direction weights must be finite and non-negative.")
    if budget_row_scaling_policy not in {"absolute", "relative_target"}:
        raise ValueError("budget_row_scaling_policy must be absolute or relative_target.")
    floor_factor = float(relative_budget_floor_factor)
    if not np.isfinite(floor_factor) or floor_factor <= 0.0:
        raise ValueError("relative_budget_floor_factor must be finite and positive.")

    n = np.exp(q_array)
    m = np.exp(r_array)
    ntot = float(np.exp(qtot_value))
    gas_residual = q_array + gas_source - ag.T @ lam_array

    q_size = q_array.shape[0]
    r_size = r_array.shape[0]
    lam_size = lam_array.shape[0]
    rho_size = rho_array.shape[0]
    variable_size = q_size + r_size + lam_size + rho_size + 1

    def block(
        q_block: np.ndarray | None = None,
        r_block: np.ndarray | None = None,
        lam_block: np.ndarray | None = None,
        rho_block: np.ndarray | None = None,
        qtot_block: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.concatenate(
            [
                np.zeros(q_size) if q_block is None else q_block,
                np.zeros(r_size) if r_block is None else r_block,
                np.zeros(lam_size) if lam_block is None else lam_block,
                np.zeros(rho_size) if rho_block is None else rho_block,
                np.zeros(1) if qtot_block is None else qtot_block,
            ]
        )

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    budget_scale = np.sqrt(float(budget_weight))
    if budget_scale > 0.0:
        budget = ag @ n + ac @ m + external_budget - target
        if budget_row_scaling_policy == "relative_target":
            positive_target = target[target > 0.0]
            target_scale = (
                float(np.max(positive_target)) if positive_target.size else 1.0
            )
            floor = max(float(np.finfo(np.float64).tiny), floor_factor * target_scale)
            row_weights = 1.0 / np.maximum(np.abs(target), floor)
            row_weights = np.where(target > 0.0, row_weights, 0.0)
            row_weights = np.where(np.isfinite(row_weights), row_weights, 0.0)
        else:
            row_weights = np.ones(ag.shape[0], dtype=np.float64)
        for row_index in range(ag.shape[0]):
            rows.append(
                budget_scale
                * row_weights[row_index]
                * block(
                    q_block=ag[row_index, :] * n,
                    r_block=ac[row_index, :] * m,
                )
            )
            rhs.append(float(-budget_scale * row_weights[row_index] * budget[row_index]))

    total_scale = np.sqrt(float(total_density_weight))
    if total_scale > 0.0:
        rows.append(total_scale * block(q_block=n, qtot_block=np.array([-ntot])))
        rhs.append(float(-total_scale * (np.sum(n) - ntot)))

    gas_scale = np.sqrt(float(amount_gas_weight))
    if gas_scale > 0.0:
        for species_index in range(q_size):
            q_part = np.zeros(q_size)
            q_part[species_index] = n[species_index] * (gas_residual[species_index] + 1.0)
            lam_part = -n[species_index] * ag[:, species_index]
            rows.append(gas_scale * block(q_block=q_part, lam_block=lam_part))
            rhs.append(float(-gas_scale * n[species_index] * gas_residual[species_index]))

    target_scale = np.sqrt(float(target_direction_weight))
    if target_direction is not None and target_scale > 0.0:
        target_vector = np.concatenate(
            [
                np.asarray(target_direction.delta_q, dtype=np.float64),
                np.asarray(target_direction.delta_r, dtype=np.float64),
                np.asarray(target_direction.delta_lambda, dtype=np.float64),
                np.asarray(target_direction.delta_rho, dtype=np.float64),
                np.array([float(target_direction.delta_qtot)]),
            ]
        )
        if target_vector.shape[0] != variable_size:
            raise ValueError("target_direction shape does not match the joint direction.")
        if target_direction_component_mode == "all":
            rows.append(target_scale * np.eye(variable_size))
            rhs.extend((target_scale * target_vector).tolist())
        elif target_direction_component_mode == "condensate_dual_only":
            selector = np.zeros((r_size + rho_size, variable_size), dtype=np.float64)
            r_start = q_size
            rho_start = q_size + r_size + lam_size
            selector[:r_size, r_start : r_start + r_size] = np.eye(r_size)
            selector[r_size:, rho_start : rho_start + rho_size] = np.eye(rho_size)
            rows.append(target_scale * selector)
            rhs.extend((target_scale * (selector @ target_vector)).tolist())
        else:
            raise ValueError(
                "target_direction_component_mode must be all or condensate_dual_only."
            )

    if not rows:
        raise ValueError("at least one positive direction weight is required.")
    matrix = np.vstack(rows)
    vector = np.asarray(rhs, dtype=np.float64)
    solution, *_ = np.linalg.lstsq(matrix, vector, rcond=None)

    delta_q = solution[:q_size]
    delta_r = solution[q_size : q_size + r_size]
    offset = q_size + r_size
    delta_lambda = solution[offset : offset + lam_size]
    offset += lam_size
    delta_rho = solution[offset : offset + rho_size]
    delta_qtot = float(solution[-1])

    def clip_delta(values: np.ndarray, limit: float) -> np.ndarray:
        limit_value = float(limit)
        if not np.isfinite(limit_value) or limit_value <= 0.0:
            raise ValueError("delta limits must be finite and positive.")
        norm_inf = float(np.max(np.abs(values))) if values.size else 0.0
        if norm_inf <= limit_value or norm_inf == 0.0:
            return values
        return values * (limit_value / norm_inf)

    delta_q = clip_delta(delta_q, max_abs_delta_q)
    delta_r = clip_delta(delta_r, max_abs_delta_r)
    delta_lambda = clip_delta(delta_lambda, max_abs_delta_lambda)

    return AlgorithmV11Direction(
        delta_q=delta_q,
        delta_r=delta_r,
        delta_lambda=delta_lambda,
        delta_rho=delta_rho,
        delta_qtot=delta_qtot,
        direction_kind="linear_budget_total_density_amount_gas_direction",
    )


def build_active_condensate_budget_correction_direction(
    *,
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond_active: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    external_condensate_budget: Sequence[float] | None = None,
    q: Sequence[float],
    r: Sequence[float],
    lambda_size: int,
    rho_size: int,
    max_abs_delta_r: float = 2.0,
    damping: float = 1.0e-20,
    relative_budget_weighting: bool = False,
    relative_budget_floor_factor: float = 1.0e-300,
    enforce_condensate_capacity: bool = False,
) -> AlgorithmV11Direction:
    """Build a least-squares active-condensate amount correction direction."""

    ag = _as_matrix(formula_matrix, "formula_matrix")
    ac = _as_matrix(formula_matrix_cond_active, "formula_matrix_cond_active")
    target = _as_vector(element_inventory_target, "element_inventory_target")
    external_budget = (
        np.zeros_like(target, dtype=np.float64)
        if external_condensate_budget is None
        else _as_vector(external_condensate_budget, "external_condensate_budget")
    )
    q_array = _as_vector(q, "q")
    r_array = _as_vector(r, "r")
    if ag.shape[0] != ac.shape[0] or ag.shape[0] != target.shape[0]:
        raise ValueError("formula matrices and element_inventory_target row counts must match.")
    if external_budget.shape[0] != target.shape[0]:
        raise ValueError("external_condensate_budget length must match element rows.")
    if ag.shape[1] != q_array.shape[0]:
        raise ValueError("formula_matrix columns must match q.")
    if ac.shape[1] != r_array.shape[0]:
        raise ValueError("formula_matrix_cond_active columns must match r.")
    if lambda_size < 0 or rho_size < 0:
        raise ValueError("lambda_size and rho_size must be non-negative.")
    limit = float(max_abs_delta_r)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("max_abs_delta_r must be finite and positive.")
    damping_value = float(damping)
    if not np.isfinite(damping_value) or damping_value < 0.0:
        raise ValueError("damping must be finite and non-negative.")
    floor_factor = float(relative_budget_floor_factor)
    if not np.isfinite(floor_factor) or floor_factor <= 0.0:
        raise ValueError("relative_budget_floor_factor must be finite and positive.")

    n = np.exp(q_array)
    m = np.exp(r_array)
    budget = ag @ n + ac @ m + external_budget - target
    jac_cond = ac * m[None, :]
    if jac_cond.size == 0 or jac_cond.shape[1] == 0:
        delta_r = np.zeros_like(r_array)
    else:
        if damping_value > 0.0:
            matrix = np.vstack(
                [
                    jac_cond,
                    np.sqrt(damping_value) * np.eye(jac_cond.shape[1], dtype=np.float64),
                ]
            )
            rhs = np.concatenate([-budget, np.zeros(jac_cond.shape[1], dtype=np.float64)])
        else:
            matrix = jac_cond
            rhs = -budget
        if relative_budget_weighting:
            positive_target = target[target > 0.0]
            target_scale = (
                float(np.max(positive_target)) if positive_target.size else 1.0
            )
            floor = max(float(np.finfo(np.float64).tiny), floor_factor * target_scale)
            row_weights = 1.0 / np.maximum(np.abs(target), floor)
            row_weights = np.where(target > 0.0, row_weights, 0.0)
            row_weights = np.where(np.isfinite(row_weights), row_weights, 0.0)
            matrix[: jac_cond.shape[0], :] *= row_weights[:, None]
            rhs[: jac_cond.shape[0]] *= row_weights
        delta_r, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
        norm_inf = float(np.max(np.abs(delta_r))) if delta_r.size else 0.0
        if norm_inf > limit and norm_inf > 0.0:
            delta_r = delta_r * (limit / norm_inf)
        if enforce_condensate_capacity:
            with np.errstate(divide="ignore", invalid="ignore"):
                per_element_limits = np.where(ac > 0.0, target[:, None] / ac, np.inf)
            cond_capacity = np.min(per_element_limits, axis=0)
            finite_positive_capacity = np.isfinite(cond_capacity) & (cond_capacity > 0.0)
            if np.any(finite_positive_capacity):
                log_capacity = np.full_like(r_array, np.inf)
                log_capacity[finite_positive_capacity] = np.log(
                    cond_capacity[finite_positive_capacity]
                )
                delta_r = np.minimum(delta_r, log_capacity - r_array)

    return AlgorithmV11Direction(
        delta_q=np.zeros_like(q_array),
        delta_r=np.asarray(delta_r, dtype=np.float64),
        delta_lambda=np.zeros(int(lambda_size), dtype=np.float64),
        delta_rho=np.zeros(int(rho_size), dtype=np.float64),
        delta_qtot=0.0,
        direction_kind="active_condensate_budget_correction_direction",
    )


def blend_algorithm_v11_directions(
    *,
    algorithm_direction: AlgorithmV11Direction,
    restoration_direction: AlgorithmV11Direction,
    algorithm_fraction: float,
) -> AlgorithmV11Direction:
    """Blend an algorithm direction with a restoration direction."""

    beta = float(algorithm_fraction)
    if not np.isfinite(beta) or beta < 0.0 or beta > 1.0:
        raise ValueError("algorithm_fraction must be finite and in [0, 1].")
    if algorithm_direction.delta_q.shape != restoration_direction.delta_q.shape:
        raise ValueError("delta_q shapes must match.")
    if algorithm_direction.delta_r.shape != restoration_direction.delta_r.shape:
        raise ValueError("delta_r shapes must match.")
    if algorithm_direction.delta_lambda.shape != restoration_direction.delta_lambda.shape:
        raise ValueError("delta_lambda shapes must match.")
    if algorithm_direction.delta_rho.shape != restoration_direction.delta_rho.shape:
        raise ValueError("delta_rho shapes must match.")
    return AlgorithmV11Direction(
        delta_q=beta * algorithm_direction.delta_q + (1.0 - beta) * restoration_direction.delta_q,
        delta_r=beta * algorithm_direction.delta_r + (1.0 - beta) * restoration_direction.delta_r,
        delta_lambda=beta * algorithm_direction.delta_lambda
        + (1.0 - beta) * restoration_direction.delta_lambda,
        delta_rho=beta * algorithm_direction.delta_rho + (1.0 - beta) * restoration_direction.delta_rho,
        delta_qtot=beta * algorithm_direction.delta_qtot
        + (1.0 - beta) * restoration_direction.delta_qtot,
        direction_kind="algorithm_v11_residual_norm_blend_direction",
    )
