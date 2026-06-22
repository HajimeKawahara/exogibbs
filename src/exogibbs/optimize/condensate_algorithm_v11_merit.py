"""Diagnostic algorithm-v1.1 barrier-penalty merit helpers.

The helpers in this module reconstruct a scalar merit from the primal
variables used by the explicit PD-IPM R-GIE diagnostics. They are explicit
import only and do not call solvers, FastChem4, or pyfastchem.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AlgorithmV11MeritBreakdown:
    """Dimensionless algorithm-v1.1 barrier-penalty merit breakdown."""

    merit_schema: str
    total_merit: float
    gibbs_objective: float
    condensate_log_barrier: float
    budget_penalty: float
    total_density_penalty: float
    charge_penalty: float | None
    budget_l1: float
    total_density_abs: float
    charge_abs: float | None
    barrier_parameter: float
    equality_penalty_weight: float
    total_density_penalty_weight: float
    charge_penalty_weight: float | None
    finite: bool
    diagnostic_only: bool
    production_behavior_change: bool

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AlgorithmV11PArmijoSelection:
    """P-based Armijo line-search selection for algorithm-v1.1 diagnostics."""

    selection_schema: str
    selected: bool
    selected_index: int | None
    selected_alpha: float | None
    selected_merit: float | None
    selected_armijo_rhs: float | None
    current_merit: float
    directional_derivative: float | None
    c1: float
    finite_trial_count: int
    armijo_trial_count: int
    rejected_trial_count: int
    selection_policy: str
    diagnostic_only: bool
    production_behavior_change: bool

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AlgorithmV11LinearizedMeritDecrease:
    """Linearized barrier-penalty merit decrease for one direction."""

    decrease_schema: str
    delta_p_linearized: float
    gibbs_linearized_delta: float
    condensate_barrier_linearized_delta: float
    budget_penalty_linearized_delta: float
    total_density_penalty_linearized_delta: float
    charge_penalty_linearized_delta: float | None
    current_budget_l1: float
    linearized_budget_l1: float
    current_total_density_abs: float
    linearized_total_density_abs: float
    finite: bool
    diagnostic_only: bool
    production_behavior_change: bool

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


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


def _finite_float(value: Any, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite.")
    return converted


def compute_algorithm_v11_barrier_penalty_merit(
    *,
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond_active: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    external_condensate_budget: Sequence[float] | None = None,
    gas_stationarity_source: Sequence[float],
    condensate_standard_source: Sequence[float],
    q: Sequence[float],
    r: Sequence[float],
    qtot: float,
    epsilon: float,
    qtot_reference: float,
    equality_penalty_weight: float = 1.0,
    total_density_penalty_weight: float | None = None,
    charge_row_index: int | None = None,
    charge_penalty_weight: float | None = None,
) -> AlgorithmV11MeritBreakdown:
    """Compute the dimensionless v1.1 primal barrier-penalty merit.

    The gas source used by the reduced equations contains the current
    ``-qtot`` contribution. This helper reconstructs a qtot-independent source
    by adding ``qtot_reference`` and evaluates candidate gas chemical potentials
    as ``q + gas_source + qtot_reference - qtot``.
    """

    ag = _as_matrix(formula_matrix, "formula_matrix")
    ac = _as_matrix(formula_matrix_cond_active, "formula_matrix_cond_active")
    target = _as_vector(element_inventory_target, "element_inventory_target")
    external_budget = (
        np.zeros_like(target, dtype=np.float64)
        if external_condensate_budget is None
        else _as_vector(external_condensate_budget, "external_condensate_budget")
    )
    gas_source = _as_vector(gas_stationarity_source, "gas_stationarity_source")
    cond_source = _as_vector(condensate_standard_source, "condensate_standard_source")
    q_array = _as_vector(q, "q")
    r_array = _as_vector(r, "r")
    qtot_value = _finite_float(qtot, "qtot")
    qtot_ref = _finite_float(qtot_reference, "qtot_reference")
    eps = _finite_float(epsilon, "epsilon")
    penalty = _finite_float(equality_penalty_weight, "equality_penalty_weight")
    if penalty < 0.0:
        raise ValueError("equality_penalty_weight must be non-negative.")
    total_penalty = penalty if total_density_penalty_weight is None else _finite_float(
        total_density_penalty_weight,
        "total_density_penalty_weight",
    )
    if total_penalty < 0.0:
        raise ValueError("total_density_penalty_weight must be non-negative.")
    charge_penalty = None
    if charge_penalty_weight is not None:
        charge_penalty = _finite_float(charge_penalty_weight, "charge_penalty_weight")
        if charge_penalty < 0.0:
            raise ValueError("charge_penalty_weight must be non-negative.")

    if ag.shape[0] != ac.shape[0] or ag.shape[0] != target.shape[0]:
        raise ValueError("formula matrices and element_inventory_target row counts must match.")
    if external_budget.shape[0] != target.shape[0]:
        raise ValueError("external_condensate_budget length must match element rows.")
    if ag.shape[1] != q_array.shape[0] or gas_source.shape[0] != q_array.shape[0]:
        raise ValueError("gas vectors must match formula_matrix columns.")
    if ac.shape[1] != r_array.shape[0] or cond_source.shape[0] != r_array.shape[0]:
        raise ValueError("condensate vectors must match formula_matrix_cond_active columns.")
    if charge_row_index is not None and not (0 <= int(charge_row_index) < target.shape[0]):
        raise ValueError("charge_row_index must identify an element row.")

    n = np.exp(q_array)
    m = np.exp(r_array)
    barrier_parameter = float(np.exp(eps))
    gas_standard_source = gas_source + qtot_ref
    gas_mu_over_rt = gas_standard_source + q_array - qtot_value
    gibbs = float(np.dot(n, gas_mu_over_rt) + np.dot(m, cond_source))
    barrier = float(-barrier_parameter * np.sum(r_array))
    budget = ag @ n + ac @ m + external_budget - target
    budget_l1 = float(np.sum(np.abs(budget)))
    total_density_abs = float(abs(np.sum(n) - np.exp(qtot_value)))
    budget_penalty = float(penalty * budget_l1)
    total_density_penalty = float(total_penalty * total_density_abs)
    charge_abs = None
    charge_term = None
    if charge_row_index is not None:
        charge_abs = float(abs(budget[int(charge_row_index)]))
        charge_weight = penalty if charge_penalty is None else charge_penalty
        charge_term = float(charge_weight * charge_abs)
    total = gibbs + barrier + budget_penalty + total_density_penalty
    if charge_term is not None:
        total += charge_term
    finite = bool(
        np.all(np.isfinite(n))
        and np.all(np.isfinite(m))
        and all(
            math.isfinite(value)
            for value in (
                total,
                gibbs,
                barrier,
                budget_penalty,
                total_density_penalty,
                budget_l1,
                total_density_abs,
            )
        )
        and (charge_term is None or math.isfinite(charge_term))
    )
    return AlgorithmV11MeritBreakdown(
        merit_schema="exogibbs_algorithm_v11_barrier_penalty_merit_v1",
        total_merit=float(total),
        gibbs_objective=gibbs,
        condensate_log_barrier=barrier,
        budget_penalty=budget_penalty,
        total_density_penalty=total_density_penalty,
        charge_penalty=charge_term,
        budget_l1=budget_l1,
        total_density_abs=total_density_abs,
        charge_abs=charge_abs,
        barrier_parameter=barrier_parameter,
        equality_penalty_weight=penalty,
        total_density_penalty_weight=total_penalty,
        charge_penalty_weight=charge_penalty,
        finite=finite,
        diagnostic_only=True,
        production_behavior_change=False,
    )


def compute_algorithm_v11_linearized_merit_decrease(
    *,
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond_active: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    external_condensate_budget: Sequence[float] | None = None,
    gas_stationarity_source: Sequence[float],
    condensate_standard_source: Sequence[float],
    q: Sequence[float],
    r: Sequence[float],
    qtot: float,
    epsilon: float,
    delta_q: Sequence[float],
    delta_r: Sequence[float],
    delta_qtot: float,
    qtot_reference: float,
    equality_penalty_weight: float = 1.0,
    total_density_penalty_weight: float | None = None,
    charge_row_index: int | None = None,
    charge_penalty_weight: float | None = None,
) -> AlgorithmV11LinearizedMeritDecrease:
    """Compute ``P_l(x, dx) - P(x)`` for the v1.1 merit.

    This is the Armijo directional-decrease quantity from a linearized
    barrier-penalty merit. Negative values indicate a descent direction.
    """

    ag = _as_matrix(formula_matrix, "formula_matrix")
    ac = _as_matrix(formula_matrix_cond_active, "formula_matrix_cond_active")
    target = _as_vector(element_inventory_target, "element_inventory_target")
    external_budget = (
        np.zeros_like(target, dtype=np.float64)
        if external_condensate_budget is None
        else _as_vector(external_condensate_budget, "external_condensate_budget")
    )
    gas_source = _as_vector(gas_stationarity_source, "gas_stationarity_source")
    cond_source = _as_vector(condensate_standard_source, "condensate_standard_source")
    q_array = _as_vector(q, "q")
    r_array = _as_vector(r, "r")
    dq = _as_vector(delta_q, "delta_q")
    dr = _as_vector(delta_r, "delta_r")
    qtot_value = _finite_float(qtot, "qtot")
    dqtot = _finite_float(delta_qtot, "delta_qtot")
    qtot_ref = _finite_float(qtot_reference, "qtot_reference")
    eps = _finite_float(epsilon, "epsilon")
    penalty = _finite_float(equality_penalty_weight, "equality_penalty_weight")
    if penalty < 0.0:
        raise ValueError("equality_penalty_weight must be non-negative.")
    total_penalty = penalty if total_density_penalty_weight is None else _finite_float(
        total_density_penalty_weight,
        "total_density_penalty_weight",
    )
    if total_penalty < 0.0:
        raise ValueError("total_density_penalty_weight must be non-negative.")
    charge_penalty = None
    if charge_penalty_weight is not None:
        charge_penalty = _finite_float(charge_penalty_weight, "charge_penalty_weight")
        if charge_penalty < 0.0:
            raise ValueError("charge_penalty_weight must be non-negative.")

    if ag.shape[0] != ac.shape[0] or ag.shape[0] != target.shape[0]:
        raise ValueError("formula matrices and element_inventory_target row counts must match.")
    if external_budget.shape[0] != target.shape[0]:
        raise ValueError("external_condensate_budget length must match element rows.")
    if ag.shape[1] != q_array.shape[0] or gas_source.shape[0] != q_array.shape[0]:
        raise ValueError("gas vectors must match formula_matrix columns.")
    if ac.shape[1] != r_array.shape[0] or cond_source.shape[0] != r_array.shape[0]:
        raise ValueError("condensate vectors must match formula_matrix_cond_active columns.")
    if dq.shape[0] != q_array.shape[0] or dr.shape[0] != r_array.shape[0]:
        raise ValueError("direction vectors must match q and r.")
    if charge_row_index is not None and not (0 <= int(charge_row_index) < target.shape[0]):
        raise ValueError("charge_row_index must identify an element row.")

    n = np.exp(q_array)
    m = np.exp(r_array)
    ntot = float(np.exp(qtot_value))
    dn = n * dq
    dm = m * dr
    dntot = ntot * dqtot
    barrier_parameter = float(np.exp(eps))
    gas_standard_source = gas_source + qtot_ref
    gas_mu_over_rt = gas_standard_source + q_array - qtot_value
    gibbs_delta = float(
        np.dot(gas_mu_over_rt, dn)
        + np.dot(n, dq - dqtot)
        + np.dot(cond_source, dm)
    )
    barrier_delta = float(-barrier_parameter * np.sum(dr))
    budget = ag @ n + ac @ m + external_budget - target
    linearized_budget = budget + ag @ dn + ac @ dm
    current_budget_l1 = float(np.sum(np.abs(budget)))
    linearized_budget_l1 = float(np.sum(np.abs(linearized_budget)))
    budget_penalty_delta = float(penalty * (linearized_budget_l1 - current_budget_l1))
    total_density = float(np.sum(n) - ntot)
    linearized_total_density = float(total_density + np.sum(dn) - dntot)
    total_density_delta = float(
        total_penalty * (abs(linearized_total_density) - abs(total_density))
    )
    charge_delta = None
    if charge_row_index is not None:
        charge_weight = penalty if charge_penalty is None else charge_penalty
        charge_delta = float(
            charge_weight
            * (
                abs(linearized_budget[int(charge_row_index)])
                - abs(budget[int(charge_row_index)])
            )
        )
    delta_p = gibbs_delta + barrier_delta + budget_penalty_delta + total_density_delta
    if charge_delta is not None:
        delta_p += charge_delta
    finite = bool(
        np.all(np.isfinite(n))
        and np.all(np.isfinite(m))
        and all(
            math.isfinite(value)
            for value in (
                delta_p,
                gibbs_delta,
                barrier_delta,
                budget_penalty_delta,
                total_density_delta,
                current_budget_l1,
                linearized_budget_l1,
                abs(total_density),
                abs(linearized_total_density),
            )
        )
        and (charge_delta is None or math.isfinite(charge_delta))
    )
    return AlgorithmV11LinearizedMeritDecrease(
        decrease_schema="exogibbs_algorithm_v11_linearized_merit_decrease_v1",
        delta_p_linearized=float(delta_p),
        gibbs_linearized_delta=gibbs_delta,
        condensate_barrier_linearized_delta=barrier_delta,
        budget_penalty_linearized_delta=budget_penalty_delta,
        total_density_penalty_linearized_delta=total_density_delta,
        charge_penalty_linearized_delta=charge_delta,
        current_budget_l1=current_budget_l1,
        linearized_budget_l1=linearized_budget_l1,
        current_total_density_abs=float(abs(total_density)),
        linearized_total_density_abs=float(abs(linearized_total_density)),
        finite=finite,
        diagnostic_only=True,
        production_behavior_change=False,
    )


def estimate_directional_derivative_from_trials(
    *,
    current_merit: float,
    trials: Sequence[Mapping[str, Any]],
    relative_difference_floor: float = 1.0e-12,
) -> float | None:
    """Estimate the merit directional derivative from finite trial merits.

    Very small trial alphas can have a merit difference that is numerically
    indistinguishable from zero. Those trials are skipped before selecting the
    smallest informative alpha.
    """

    current = _finite_float(current_merit, "current_merit")
    floor = _finite_float(relative_difference_floor, "relative_difference_floor")
    if floor < 0.0:
        raise ValueError("relative_difference_floor must be non-negative.")
    finite_trials: list[tuple[float, float]] = []
    for trial in trials:
        try:
            alpha = float(trial.get("alpha"))
            merit = float(trial.get("p_merit", trial.get("merit")))
        except (TypeError, ValueError):
            continue
        if alpha > 0.0 and math.isfinite(alpha) and math.isfinite(merit):
            finite_trials.append((alpha, merit))
    if not finite_trials:
        return None
    difference_floor = floor * max(abs(current), 1.0)
    informative_trials = [
        (alpha, merit)
        for alpha, merit in sorted(finite_trials, key=lambda item: item[0])
        if abs(merit - current) > difference_floor
    ]
    if not informative_trials:
        return 0.0
    alpha, merit = informative_trials[0]
    slope = (merit - current) / alpha
    return float(slope) if math.isfinite(slope) else None


def select_p_based_armijo_trial(
    trials: Sequence[Mapping[str, Any]],
    *,
    current_merit: float,
    directional_derivative: float | None = None,
    c1: float = 1.0e-4,
    choose_largest_alpha: bool = True,
) -> AlgorithmV11PArmijoSelection:
    """Select a finite trial satisfying P-based Armijo sufficient decrease."""

    current = _finite_float(current_merit, "current_merit")
    c1_value = _finite_float(c1, "c1")
    if c1_value <= 0.0 or c1_value >= 1.0:
        raise ValueError("c1 must be in the interval (0, 1).")
    slope = (
        estimate_directional_derivative_from_trials(
            current_merit=current,
            trials=trials,
        )
        if directional_derivative is None
        else float(directional_derivative)
    )
    if slope is None or not math.isfinite(slope) or slope >= 0.0:
        return AlgorithmV11PArmijoSelection(
            selection_schema="exogibbs_algorithm_v11_p_armijo_selection_v1",
            selected=False,
            selected_index=None,
            selected_alpha=None,
            selected_merit=None,
            selected_armijo_rhs=None,
            current_merit=current,
            directional_derivative=None if slope is None else float(slope),
            c1=c1_value,
            finite_trial_count=sum(
                1
                for trial in trials
                if bool(trial.get("all_finite", False))
                and math.isfinite(float(trial.get("p_merit", trial.get("merit", math.nan))))
            ),
            armijo_trial_count=0,
            rejected_trial_count=0,
            selection_policy="p_based_armijo_sufficient_decrease",
            diagnostic_only=True,
            production_behavior_change=False,
        )

    finite_trials: list[dict[str, float | int]] = []
    armijo_trials: list[dict[str, float | int]] = []
    rejected = 0
    for index, trial in enumerate(trials):
        try:
            alpha = float(trial.get("alpha"))
            merit = float(trial.get("p_merit", trial.get("merit")))
        except (TypeError, ValueError):
            rejected += 1
            continue
        all_finite = bool(trial.get("all_finite", math.isfinite(alpha) and math.isfinite(merit)))
        if alpha <= 0.0 or alpha > 1.0 or not math.isfinite(alpha) or not math.isfinite(merit) or not all_finite:
            rejected += 1
            continue
        rhs = current + c1_value * alpha * slope
        record = {"index": index, "alpha": alpha, "merit": merit, "armijo_rhs": rhs}
        finite_trials.append(record)
        if merit <= rhs:
            armijo_trials.append(record)

    if not armijo_trials:
        return AlgorithmV11PArmijoSelection(
            selection_schema="exogibbs_algorithm_v11_p_armijo_selection_v1",
            selected=False,
            selected_index=None,
            selected_alpha=None,
            selected_merit=None,
            selected_armijo_rhs=None,
            current_merit=current,
            directional_derivative=float(slope),
            c1=c1_value,
            finite_trial_count=len(finite_trials),
            armijo_trial_count=0,
            rejected_trial_count=rejected,
            selection_policy="p_based_armijo_sufficient_decrease",
            diagnostic_only=True,
            production_behavior_change=False,
        )

    selected = (
        max(armijo_trials, key=lambda row: (row["alpha"], -row["merit"]))
        if choose_largest_alpha
        else min(armijo_trials, key=lambda row: row["merit"])
    )
    return AlgorithmV11PArmijoSelection(
        selection_schema="exogibbs_algorithm_v11_p_armijo_selection_v1",
        selected=True,
        selected_index=int(selected["index"]),
        selected_alpha=float(selected["alpha"]),
        selected_merit=float(selected["merit"]),
        selected_armijo_rhs=float(selected["armijo_rhs"]),
        current_merit=current,
        directional_derivative=float(slope),
        c1=c1_value,
        finite_trial_count=len(finite_trials),
        armijo_trial_count=len(armijo_trials),
        rejected_trial_count=rejected,
        selection_policy="p_based_armijo_sufficient_decrease",
        diagnostic_only=True,
        production_behavior_change=False,
    )


__all__ = (
    "AlgorithmV11MeritBreakdown",
    "AlgorithmV11LinearizedMeritDecrease",
    "AlgorithmV11PArmijoSelection",
    "compute_algorithm_v11_barrier_penalty_merit",
    "compute_algorithm_v11_linearized_merit_decrease",
    "estimate_directional_derivative_from_trials",
    "select_p_based_armijo_trial",
)
