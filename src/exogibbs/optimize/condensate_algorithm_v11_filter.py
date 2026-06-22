"""Filter-style trial acceptance for algorithm-v1.1 diagnostics.

This module is explicit-import only. It implements diagnostic globalization
helpers for corrected reduced R-GIE experiments and does not call production
solvers, FastChem4, pyfastchem, or preset/default wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AlgorithmV11FilterSelection:
    """Filter-style diagnostic trial selection report."""

    selection_schema: str
    selected: bool
    selected_index: int | None
    selected_alpha: float | None
    selected_merit: float | None
    selected_theta: float | None
    current_merit: float
    current_theta: float
    finite_trial_count: int
    filter_trial_count: int
    rejected_trial_count: int
    theta_reduction_fraction: float
    merit_reduction_tolerance: float
    selection_policy: str
    selected_reason: str | None
    diagnostic_only: bool
    production_behavior_change: bool

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AlgorithmV11FilterVariantChoice:
    """Diagnostic selection among completed filter/globalization variants."""

    choice_schema: str
    selected: bool
    selected_name: str | None
    selected_index: int | None
    selected_residual_l2: float | None
    baseline_name: str
    baseline_residual_l2: float | None
    candidate_count: int
    finite_candidate_count: int
    require_final_barrier_progress_if_baseline_reached: bool
    selection_policy: str
    selected_reason: str | None
    diagnostic_only: bool
    production_behavior_change: bool

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class GasStationarityFrameDecomposition:
    """Diagnostic split of gas stationarity by log and amount frames."""

    decomposition_schema: str
    raw_norm: float
    amount_weighted_norm: float
    sqrt_amount_weighted_norm: float
    active_raw_norm: float
    active_amount_weighted_norm: float
    trace_raw_norm: float
    trace_amount_weighted_norm: float
    active_species_count: int
    trace_species_count: int
    total_species_count: int
    active_amount_floor: float
    trace_amount_ceiling: float
    amount_to_raw_ratio: float
    trace_raw_fraction: float
    raw_frame_trace_dominated: bool
    amount_frame_nearly_closed: bool
    diagnostic_only: bool
    production_behavior_change: bool

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AlgorithmV11HTypeSelection:
    """Ipopt-style diagnostic h-type selection report."""

    selection_schema: str
    selected: bool
    selected_index: int | None
    selected_alpha: float | None
    selected_theta: float | None
    current_theta: float
    finite_trial_count: int
    h_type_trial_count: int
    rejected_trial_count: int
    theta_reduction_fraction: float
    selection_policy: str
    selected_reason: str | None
    diagnostic_only: bool
    production_behavior_change: bool

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AlgorithmV11NoArmijoRecoveryClassification:
    """Diagnostic classification for no-Armijo and final-centering exits."""

    classification_schema: str
    recovery_class: str
    restoration_candidate: bool
    h_type_candidate: bool
    raw_frame_only_failure: bool
    dominant_component: str | None
    dominant_component_value: float | None
    amount_weighted_gas_to_raw_gas_ratio: float | None
    reason: str
    diagnostic_only: bool
    production_behavior_change: bool

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AlgorithmV11SoftRestorationSelection:
    """Diagnostic soft-restoration selection with proximity control."""

    selection_schema: str
    selected: bool
    selected_index: int | None
    selected_alpha: float | None
    selected_theta: float | None
    selected_proximity: float | None
    selected_score: float | None
    current_theta: float
    finite_trial_count: int
    restoration_trial_count: int
    rejected_trial_count: int
    theta_reduction_fraction: float
    proximity_weight: float
    max_proximity: float | None
    selection_policy: str
    selected_reason: str | None
    diagnostic_only: bool
    production_behavior_change: bool

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AlgorithmV11FinalCenteringClassification:
    """Diagnostic final-centering classification across raw and amount frames."""

    classification_schema: str
    final_centering_class: str
    amount_frame_centered: bool
    raw_frame_remaining: bool
    physical_restoration_needed: bool
    raw_frame_diagnostic_needed: bool
    reason: str
    diagnostic_only: bool
    production_behavior_change: bool

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AlgorithmV11PhysicalConvergenceGate:
    """Diagnostic physical convergence gate using amount-frame components."""

    gate_schema: str
    physical_converged: bool
    raw_frame_centered: bool
    raw_frame_diagnostic_only: bool
    needs_restoration: bool
    needs_algorithmic_update: bool
    budget_ok: bool
    total_density_ok: bool
    amount_weighted_gas_ok: bool
    amount_weighted_condensate_ok: bool
    finite_components: bool
    budget: float
    total_density: float
    amount_weighted_gas: float
    amount_weighted_condensate: float
    raw_residual_l2: float
    physical_threshold: float
    raw_threshold: float
    reason: str
    diagnostic_only: bool
    production_behavior_change: bool

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


DEFAULT_FILTER_COMPONENT_WEIGHTS = {
    "budget": 1.0,
    "total_density": 1.0,
}


def constraint_violation_from_components(
    components: Mapping[str, Any],
    component_weights: Mapping[str, float] | None = None,
    component_scales: Mapping[str, float] | None = None,
    minimum_component_scale: float = 1.0e-300,
) -> float:
    """Return a weighted scalar constraint violation from residual components."""

    weights = DEFAULT_FILTER_COMPONENT_WEIGHTS if component_weights is None else component_weights
    scale_floor = float(minimum_component_scale)
    if not math.isfinite(scale_floor) or scale_floor <= 0.0:
        raise ValueError("minimum_component_scale must be finite and positive.")
    total = 0.0
    for name, weight in weights.items():
        weight_value = float(weight)
        value = float(components.get(name, math.nan))
        if not math.isfinite(weight_value) or weight_value < 0.0:
            raise ValueError("filter component weights must be finite and non-negative.")
        if not math.isfinite(value):
            return math.inf
        scale = 1.0
        if component_scales is not None:
            scale = float(component_scales.get(name, 1.0))
            if not math.isfinite(scale) or scale < 0.0:
                raise ValueError("filter component scales must be finite and non-negative.")
            scale = max(scale, scale_floor)
        total += weight_value * value / scale
    return float(total)


def select_filter_restoration_trial(
    trials: Sequence[Mapping[str, Any]],
    *,
    current_merit: float,
    current_components: Mapping[str, Any],
    theta_reduction_fraction: float = 1.0e-4,
    merit_reduction_tolerance: float = 0.0,
    component_weights: Mapping[str, float] | None = None,
    component_scales: Mapping[str, float] | None = None,
    minimum_component_scale: float = 1.0e-300,
    choose_largest_alpha: bool = True,
) -> AlgorithmV11FilterSelection:
    """Select a trial by filter-style merit-or-feasibility progress."""

    current_p = float(current_merit)
    if not math.isfinite(current_p):
        raise ValueError("current_merit must be finite.")
    theta = constraint_violation_from_components(
        current_components,
        component_weights=component_weights,
        component_scales=component_scales,
        minimum_component_scale=minimum_component_scale,
    )
    if not math.isfinite(theta):
        raise ValueError("current_components must define a finite constraint violation.")
    theta_fraction = float(theta_reduction_fraction)
    if not math.isfinite(theta_fraction) or theta_fraction < 0.0 or theta_fraction >= 1.0:
        raise ValueError("theta_reduction_fraction must be finite and in [0, 1).")
    merit_tolerance = float(merit_reduction_tolerance)
    if not math.isfinite(merit_tolerance) or merit_tolerance < 0.0:
        raise ValueError("merit_reduction_tolerance must be finite and non-negative.")

    finite_trials: list[dict[str, float | int | str]] = []
    accepted: list[dict[str, float | int | str]] = []
    rejected = 0
    for index, trial in enumerate(trials):
        try:
            alpha = float(trial.get("alpha"))
            merit = float(trial.get("p_merit", trial.get("merit")))
            trial_theta = constraint_violation_from_components(
                trial.get("residual_components", {}),
                component_weights=component_weights,
                component_scales=component_scales,
                minimum_component_scale=minimum_component_scale,
            )
        except (TypeError, ValueError):
            rejected += 1
            continue
        all_finite = bool(
            trial.get(
                "all_finite",
                math.isfinite(alpha) and math.isfinite(merit) and math.isfinite(trial_theta),
            )
        )
        if (
            alpha <= 0.0
            or alpha > 1.0
            or not math.isfinite(alpha)
            or not math.isfinite(merit)
            or not math.isfinite(trial_theta)
            or not all_finite
        ):
            rejected += 1
            continue
        theta_accepts = trial_theta <= (1.0 - theta_fraction) * theta
        merit_accepts = merit <= current_p - merit_tolerance
        reason = None
        if theta_accepts and merit_accepts:
            reason = "theta_and_merit_progress"
        elif theta_accepts:
            reason = "theta_progress"
        elif merit_accepts and trial_theta <= theta:
            reason = "merit_progress_with_theta_nonworsening"
        record = {
            "index": index,
            "alpha": alpha,
            "merit": merit,
            "theta": trial_theta,
            "reason": reason or "rejected",
        }
        finite_trials.append(record)
        if reason is not None:
            accepted.append(record)

    if not accepted:
        return AlgorithmV11FilterSelection(
            selection_schema="exogibbs_algorithm_v11_filter_selection_v1",
            selected=False,
            selected_index=None,
            selected_alpha=None,
            selected_merit=None,
            selected_theta=None,
            current_merit=current_p,
            current_theta=theta,
            finite_trial_count=len(finite_trials),
            filter_trial_count=0,
            rejected_trial_count=rejected,
            theta_reduction_fraction=theta_fraction,
            merit_reduction_tolerance=merit_tolerance,
            selection_policy="filter_restoration_merit_or_feasibility_progress",
            selected_reason=None,
            diagnostic_only=True,
            production_behavior_change=False,
        )

    selected = (
        max(accepted, key=lambda row: (float(row["alpha"]), -float(row["theta"]), -float(row["merit"])))
        if choose_largest_alpha
        else min(accepted, key=lambda row: (float(row["theta"]), float(row["merit"])))
    )
    return AlgorithmV11FilterSelection(
        selection_schema="exogibbs_algorithm_v11_filter_selection_v1",
        selected=True,
        selected_index=int(selected["index"]),
        selected_alpha=float(selected["alpha"]),
        selected_merit=float(selected["merit"]),
        selected_theta=float(selected["theta"]),
        current_merit=current_p,
        current_theta=theta,
        finite_trial_count=len(finite_trials),
        filter_trial_count=len(accepted),
        rejected_trial_count=rejected,
        theta_reduction_fraction=theta_fraction,
        merit_reduction_tolerance=merit_tolerance,
        selection_policy="filter_restoration_merit_or_feasibility_progress",
        selected_reason=str(selected["reason"]),
        diagnostic_only=True,
        production_behavior_change=False,
    )


def decompose_gas_stationarity_frame(
    *,
    gas_residual: Sequence[float],
    ln_nk: Sequence[float],
    active_amount_floor: float = 1.0e-30,
    trace_amount_ceiling: float = 1.0e-40,
    amount_nearly_closed_threshold: float = 1.0e-6,
    trace_raw_fraction_threshold: float = 0.5,
) -> GasStationarityFrameDecomposition:
    """Split gas stationarity into raw, amount-weighted, active, and trace frames."""

    residual = [float(value) for value in gas_residual]
    q = [float(value) for value in ln_nk]
    if len(residual) != len(q):
        raise ValueError("gas_residual and ln_nk must have the same length.")
    if not residual:
        raise ValueError("gas_residual must not be empty.")
    active_floor = float(active_amount_floor)
    trace_ceiling = float(trace_amount_ceiling)
    amount_threshold = float(amount_nearly_closed_threshold)
    trace_fraction_threshold = float(trace_raw_fraction_threshold)
    if (
        not math.isfinite(active_floor)
        or not math.isfinite(trace_ceiling)
        or not math.isfinite(amount_threshold)
        or not math.isfinite(trace_fraction_threshold)
        or active_floor < 0.0
        or trace_ceiling < 0.0
        or amount_threshold < 0.0
        or trace_fraction_threshold < 0.0
        or trace_fraction_threshold > 1.0
    ):
        raise ValueError("frame thresholds must be finite and non-negative.")

    amounts: list[float] = []
    for value in q:
        if not math.isfinite(value):
            raise ValueError("ln_nk must contain finite values.")
        amounts.append(math.exp(value))
    for value in residual:
        if not math.isfinite(value):
            raise ValueError("gas_residual must contain finite values.")

    def norm(values: Sequence[float]) -> float:
        return math.sqrt(sum(float(value) * float(value) for value in values))

    raw_norm = norm(residual)
    amount_weighted = [amount * value for amount, value in zip(amounts, residual)]
    sqrt_amount_weighted = [math.sqrt(amount) * value for amount, value in zip(amounts, residual)]
    active_raw = [
        value for amount, value in zip(amounts, residual) if amount >= active_floor
    ]
    active_amount = [
        amount * value
        for amount, value in zip(amounts, residual)
        if amount >= active_floor
    ]
    trace_raw = [
        value for amount, value in zip(amounts, residual) if amount <= trace_ceiling
    ]
    trace_amount = [
        amount * value
        for amount, value in zip(amounts, residual)
        if amount <= trace_ceiling
    ]
    trace_raw_norm = norm(trace_raw)
    amount_norm = norm(amount_weighted)
    raw_ratio_denominator = max(raw_norm, 1.0e-300)
    trace_fraction = trace_raw_norm / raw_ratio_denominator
    return GasStationarityFrameDecomposition(
        decomposition_schema="exogibbs_algorithm_v11_gas_stationarity_frame_decomposition_v1",
        raw_norm=raw_norm,
        amount_weighted_norm=amount_norm,
        sqrt_amount_weighted_norm=norm(sqrt_amount_weighted),
        active_raw_norm=norm(active_raw),
        active_amount_weighted_norm=norm(active_amount),
        trace_raw_norm=trace_raw_norm,
        trace_amount_weighted_norm=norm(trace_amount),
        active_species_count=len(active_raw),
        trace_species_count=len(trace_raw),
        total_species_count=len(residual),
        active_amount_floor=active_floor,
        trace_amount_ceiling=trace_ceiling,
        amount_to_raw_ratio=amount_norm / raw_ratio_denominator,
        trace_raw_fraction=trace_fraction,
        raw_frame_trace_dominated=bool(trace_fraction >= trace_fraction_threshold),
        amount_frame_nearly_closed=bool(amount_norm <= amount_threshold),
        diagnostic_only=True,
        production_behavior_change=False,
    )


def select_h_type_constraint_trial(
    trials: Sequence[Mapping[str, Any]],
    *,
    current_components: Mapping[str, Any],
    theta_reduction_fraction: float = 1.0e-4,
    component_weights: Mapping[str, float] | None = None,
    component_scales: Mapping[str, float] | None = None,
    minimum_component_scale: float = 1.0e-300,
    choose_largest_alpha: bool = True,
) -> AlgorithmV11HTypeSelection:
    """Select a constraint-improving h-type trial independent of objective progress."""

    theta = constraint_violation_from_components(
        current_components,
        component_weights=component_weights,
        component_scales=component_scales,
        minimum_component_scale=minimum_component_scale,
    )
    if not math.isfinite(theta):
        raise ValueError("current_components must define a finite constraint violation.")
    theta_fraction = float(theta_reduction_fraction)
    if not math.isfinite(theta_fraction) or theta_fraction < 0.0 or theta_fraction >= 1.0:
        raise ValueError("theta_reduction_fraction must be finite and in [0, 1).")

    finite_trials: list[dict[str, float | int]] = []
    accepted: list[dict[str, float | int]] = []
    rejected = 0
    for index, trial in enumerate(trials):
        try:
            alpha = float(trial.get("alpha"))
            trial_theta = constraint_violation_from_components(
                trial.get("residual_components", {}),
                component_weights=component_weights,
                component_scales=component_scales,
                minimum_component_scale=minimum_component_scale,
            )
        except (TypeError, ValueError):
            rejected += 1
            continue
        all_finite = bool(
            trial.get("all_finite", math.isfinite(alpha) and math.isfinite(trial_theta))
        )
        if (
            alpha <= 0.0
            or alpha > 1.0
            or not math.isfinite(alpha)
            or not math.isfinite(trial_theta)
            or not all_finite
        ):
            rejected += 1
            continue
        record = {"index": index, "alpha": alpha, "theta": trial_theta}
        finite_trials.append(record)
        if trial_theta <= (1.0 - theta_fraction) * theta:
            accepted.append(record)

    if not accepted:
        return AlgorithmV11HTypeSelection(
            selection_schema="exogibbs_algorithm_v11_h_type_selection_v1",
            selected=False,
            selected_index=None,
            selected_alpha=None,
            selected_theta=None,
            current_theta=theta,
            finite_trial_count=len(finite_trials),
            h_type_trial_count=0,
            rejected_trial_count=rejected,
            theta_reduction_fraction=theta_fraction,
            selection_policy="h_type_constraint_reduction_only",
            selected_reason=None,
            diagnostic_only=True,
            production_behavior_change=False,
        )

    selected = (
        max(accepted, key=lambda row: (float(row["alpha"]), -float(row["theta"])))
        if choose_largest_alpha
        else min(accepted, key=lambda row: (float(row["theta"]), -float(row["alpha"])))
    )
    return AlgorithmV11HTypeSelection(
        selection_schema="exogibbs_algorithm_v11_h_type_selection_v1",
        selected=True,
        selected_index=int(selected["index"]),
        selected_alpha=float(selected["alpha"]),
        selected_theta=float(selected["theta"]),
        current_theta=theta,
        finite_trial_count=len(finite_trials),
        h_type_trial_count=len(accepted),
        rejected_trial_count=rejected,
        theta_reduction_fraction=theta_fraction,
        selection_policy="h_type_constraint_reduction_only",
        selected_reason="theta_progress",
        diagnostic_only=True,
        production_behavior_change=False,
    )


def classify_no_armijo_recovery(
    components: Mapping[str, Any],
    *,
    status: str,
    amount_weighted_gas_ratio_threshold: float = 1.0e-6,
    budget_failure_threshold: float = 1.0e-6,
) -> AlgorithmV11NoArmijoRecoveryClassification:
    """Classify no-Armijo or final-centering records without declaring failure."""

    finite_components: dict[str, float] = {}
    for name, value in components.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite_components[str(name)] = number
    if not finite_components:
        return AlgorithmV11NoArmijoRecoveryClassification(
            classification_schema="exogibbs_algorithm_v11_no_armijo_recovery_classification_v1",
            recovery_class="nonfinite_or_missing_components",
            restoration_candidate=False,
            h_type_candidate=False,
            raw_frame_only_failure=False,
            dominant_component=None,
            dominant_component_value=None,
            amount_weighted_gas_to_raw_gas_ratio=None,
            reason="No finite residual components were available for recovery classification.",
            diagnostic_only=True,
            production_behavior_change=False,
        )

    dominant_name, dominant_value = max(finite_components.items(), key=lambda item: item[1])
    raw_gas = finite_components.get("gas")
    amount_gas = finite_components.get("amount_weighted_gas")
    ratio = None
    if raw_gas is not None and raw_gas > 0.0 and amount_gas is not None:
        ratio = amount_gas / raw_gas
    raw_frame_only = bool(
        dominant_name == "gas"
        and ratio is not None
        and ratio <= float(amount_weighted_gas_ratio_threshold)
    )
    budget_value = finite_components.get("budget", 0.0)
    if raw_frame_only:
        recovery_class = "raw_frame_only_failure"
        reason = (
            "Raw gas stationarity dominates, but the amount-weighted gas residual "
            "is nearly closed; treat this as a frame-sensitive centering issue."
        )
        restoration_candidate = False
        h_type_candidate = True
    elif budget_value > float(budget_failure_threshold) and dominant_name == "budget":
        recovery_class = "budget_failure"
        reason = "The budget component dominates and should be handled by constraint restoration."
        restoration_candidate = True
        h_type_candidate = True
    elif dominant_name in {"complementarity", "condensate"}:
        recovery_class = "barrier_centering_failure"
        reason = "The dominant component is an interior-path centering component."
        restoration_candidate = True
        h_type_candidate = False
    elif str(status) == "no_p_armijo_trial":
        recovery_class = "armijo_acceptance_gap"
        reason = "No Armijo trial was accepted, but finite components remain classifiable."
        restoration_candidate = True
        h_type_candidate = True
    else:
        recovery_class = "classified_nonterminal_centering_gap"
        reason = "The record is finite and should be routed through filter/restoration lifecycle."
        restoration_candidate = True
        h_type_candidate = True
    return AlgorithmV11NoArmijoRecoveryClassification(
        classification_schema="exogibbs_algorithm_v11_no_armijo_recovery_classification_v1",
        recovery_class=recovery_class,
        restoration_candidate=restoration_candidate,
        h_type_candidate=h_type_candidate,
        raw_frame_only_failure=raw_frame_only,
        dominant_component=dominant_name,
        dominant_component_value=dominant_value,
        amount_weighted_gas_to_raw_gas_ratio=ratio,
        reason=reason,
        diagnostic_only=True,
        production_behavior_change=False,
    )


def select_soft_restoration_trial(
    trials: Sequence[Mapping[str, Any]],
    *,
    current_components: Mapping[str, Any],
    theta_reduction_fraction: float = 1.0e-4,
    component_weights: Mapping[str, float] | None = None,
    component_scales: Mapping[str, float] | None = None,
    minimum_component_scale: float = 1.0e-300,
    proximity_weight: float = 1.0e-3,
    max_proximity: float | None = None,
) -> AlgorithmV11SoftRestorationSelection:
    """Select a constraint-improving restoration trial with proximity penalty.

    Trial proximity is read from ``proximity`` when present. If absent, ``alpha``
    is used as a conservative recorded-trial proxy: smaller accepted steps stay
    closer to the current iterate.
    """

    theta = constraint_violation_from_components(
        current_components,
        component_weights=component_weights,
        component_scales=component_scales,
        minimum_component_scale=minimum_component_scale,
    )
    if not math.isfinite(theta):
        raise ValueError("current_components must define a finite constraint violation.")
    theta_fraction = float(theta_reduction_fraction)
    penalty = float(proximity_weight)
    if not math.isfinite(theta_fraction) or theta_fraction < 0.0 or theta_fraction >= 1.0:
        raise ValueError("theta_reduction_fraction must be finite and in [0, 1).")
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ValueError("proximity_weight must be finite and non-negative.")
    proximity_limit = None if max_proximity is None else float(max_proximity)
    if proximity_limit is not None and (
        not math.isfinite(proximity_limit) or proximity_limit < 0.0
    ):
        raise ValueError("max_proximity must be finite and non-negative when provided.")

    finite_trials: list[dict[str, float | int]] = []
    accepted: list[dict[str, float | int]] = []
    rejected = 0
    theta_floor = max(theta, minimum_component_scale)
    for index, trial in enumerate(trials):
        try:
            alpha = float(trial.get("alpha"))
            trial_theta = constraint_violation_from_components(
                trial.get("residual_components", {}),
                component_weights=component_weights,
                component_scales=component_scales,
                minimum_component_scale=minimum_component_scale,
            )
            proximity = float(trial.get("proximity", alpha))
        except (TypeError, ValueError):
            rejected += 1
            continue
        all_finite = bool(
            trial.get(
                "all_finite",
                math.isfinite(alpha) and math.isfinite(trial_theta) and math.isfinite(proximity),
            )
        )
        if (
            alpha <= 0.0
            or alpha > 1.0
            or proximity < 0.0
            or not math.isfinite(alpha)
            or not math.isfinite(trial_theta)
            or not math.isfinite(proximity)
            or not all_finite
        ):
            rejected += 1
            continue
        record = {
            "index": index,
            "alpha": alpha,
            "theta": trial_theta,
            "proximity": proximity,
            "score": trial_theta / theta_floor + penalty * proximity,
        }
        finite_trials.append(record)
        proximity_accepts = proximity_limit is None or proximity <= proximity_limit
        theta_accepts = trial_theta <= (1.0 - theta_fraction) * theta
        if theta_accepts and proximity_accepts:
            accepted.append(record)

    if not accepted:
        return AlgorithmV11SoftRestorationSelection(
            selection_schema="exogibbs_algorithm_v11_soft_restoration_selection_v1",
            selected=False,
            selected_index=None,
            selected_alpha=None,
            selected_theta=None,
            selected_proximity=None,
            selected_score=None,
            current_theta=theta,
            finite_trial_count=len(finite_trials),
            restoration_trial_count=0,
            rejected_trial_count=rejected,
            theta_reduction_fraction=theta_fraction,
            proximity_weight=penalty,
            max_proximity=proximity_limit,
            selection_policy="soft_restoration_theta_progress_with_proximity_penalty",
            selected_reason=None,
            diagnostic_only=True,
            production_behavior_change=False,
        )

    selected = min(
        accepted,
        key=lambda row: (float(row["score"]), float(row["theta"]), float(row["proximity"])),
    )
    return AlgorithmV11SoftRestorationSelection(
        selection_schema="exogibbs_algorithm_v11_soft_restoration_selection_v1",
        selected=True,
        selected_index=int(selected["index"]),
        selected_alpha=float(selected["alpha"]),
        selected_theta=float(selected["theta"]),
        selected_proximity=float(selected["proximity"]),
        selected_score=float(selected["score"]),
        current_theta=theta,
        finite_trial_count=len(finite_trials),
        restoration_trial_count=len(accepted),
        rejected_trial_count=rejected,
        theta_reduction_fraction=theta_fraction,
        proximity_weight=penalty,
        max_proximity=proximity_limit,
        selection_policy="soft_restoration_theta_progress_with_proximity_penalty",
        selected_reason="theta_progress_with_proximity_control",
        diagnostic_only=True,
        production_behavior_change=False,
    )


def classify_final_centering_frame(
    components: Mapping[str, Any],
    *,
    raw_residual_l2: float,
    amount_threshold: float = 1.0e-6,
    raw_threshold: float = 1.0e-6,
) -> AlgorithmV11FinalCenteringClassification:
    """Classify final centering without collapsing raw and amount frames."""

    amount_limit = float(amount_threshold)
    raw_limit = float(raw_threshold)
    raw_value = float(raw_residual_l2)
    if (
        not math.isfinite(amount_limit)
        or not math.isfinite(raw_limit)
        or not math.isfinite(raw_value)
        or amount_limit < 0.0
        or raw_limit < 0.0
    ):
        raise ValueError("centering thresholds and raw_residual_l2 must be finite.")

    def component(name: str) -> float:
        try:
            value = float(components.get(name, 0.0))
        except (TypeError, ValueError):
            return math.inf
        return value if math.isfinite(value) else math.inf

    amount_components = {
        "budget": component("budget"),
        "total_density": component("total_density"),
        "amount_weighted_gas": component("amount_weighted_gas"),
        "amount_weighted_condensate": component("amount_weighted_condensate"),
    }
    amount_frame_centered = all(value <= amount_limit for value in amount_components.values())
    raw_frame_remaining = raw_value > raw_limit
    if amount_frame_centered and raw_frame_remaining:
        final_class = "amount_frame_centered_raw_frame_remaining"
        physical_restoration_needed = False
        raw_frame_diagnostic_needed = True
        reason = (
            "Budget, total-density, and amount-weighted stationarity are centered "
            "while raw log-space residual remains; keep raw residual as diagnostic."
        )
    elif amount_frame_centered:
        final_class = "amount_and_raw_frames_centered"
        physical_restoration_needed = False
        raw_frame_diagnostic_needed = False
        reason = "Both amount-frame and raw-frame residuals satisfy the supplied thresholds."
    else:
        final_class = "amount_frame_not_centered"
        physical_restoration_needed = True
        raw_frame_diagnostic_needed = raw_frame_remaining
        reason = "At least one physical amount-frame component exceeds the supplied threshold."
    return AlgorithmV11FinalCenteringClassification(
        classification_schema="exogibbs_algorithm_v11_final_centering_classification_v1",
        final_centering_class=final_class,
        amount_frame_centered=amount_frame_centered,
        raw_frame_remaining=raw_frame_remaining,
        physical_restoration_needed=physical_restoration_needed,
        raw_frame_diagnostic_needed=raw_frame_diagnostic_needed,
        reason=reason,
        diagnostic_only=True,
        production_behavior_change=False,
    )


def evaluate_amount_frame_physical_convergence(
    components: Mapping[str, Any],
    *,
    raw_residual_l2: float,
    physical_threshold: float = 1.0e-6,
    raw_threshold: float = 1.0e-6,
) -> AlgorithmV11PhysicalConvergenceGate:
    """Evaluate physical convergence without requiring raw-frame centering."""

    physical_limit = float(physical_threshold)
    raw_limit = float(raw_threshold)
    raw_value = float(raw_residual_l2)
    if (
        not math.isfinite(physical_limit)
        or not math.isfinite(raw_limit)
        or not math.isfinite(raw_value)
        or physical_limit < 0.0
        or raw_limit < 0.0
    ):
        raise ValueError("convergence thresholds and raw_residual_l2 must be finite.")

    def component(name: str) -> float:
        try:
            return float(components.get(name, math.inf))
        except (TypeError, ValueError):
            return math.inf

    budget = component("budget")
    total_density = component("total_density")
    amount_gas = component("amount_weighted_gas")
    amount_cond = component("amount_weighted_condensate")
    values = (budget, total_density, amount_gas, amount_cond, raw_value)
    finite = all(math.isfinite(value) for value in values)
    budget_ok = finite and budget <= physical_limit
    total_density_ok = finite and total_density <= physical_limit
    amount_gas_ok = finite and amount_gas <= physical_limit
    amount_cond_ok = finite and amount_cond <= physical_limit
    physical_converged = bool(
        finite and budget_ok and total_density_ok and amount_gas_ok and amount_cond_ok
    )
    raw_centered = bool(finite and raw_value <= raw_limit)
    raw_diagnostic_only = bool(physical_converged and not raw_centered)
    needs_restoration = bool(finite and not physical_converged)
    needs_algorithmic_update = bool(not finite)
    if not finite:
        reason = "At least one convergence component is nonfinite."
    elif physical_converged and raw_centered:
        reason = "Physical amount-frame and raw-frame convergence gates are both satisfied."
    elif physical_converged:
        reason = (
            "Physical amount-frame convergence is satisfied; raw-frame residual "
            "remains diagnostic-only."
        )
    else:
        reason = "At least one physical amount-frame convergence component exceeds threshold."
    return AlgorithmV11PhysicalConvergenceGate(
        gate_schema="exogibbs_algorithm_v11_amount_frame_physical_convergence_gate_v1",
        physical_converged=physical_converged,
        raw_frame_centered=raw_centered,
        raw_frame_diagnostic_only=raw_diagnostic_only,
        needs_restoration=needs_restoration,
        needs_algorithmic_update=needs_algorithmic_update,
        budget_ok=budget_ok,
        total_density_ok=total_density_ok,
        amount_weighted_gas_ok=amount_gas_ok,
        amount_weighted_condensate_ok=amount_cond_ok,
        finite_components=finite,
        budget=budget,
        total_density=total_density,
        amount_weighted_gas=amount_gas,
        amount_weighted_condensate=amount_cond,
        raw_residual_l2=raw_value,
        physical_threshold=physical_limit,
        raw_threshold=raw_limit,
        reason=reason,
        diagnostic_only=True,
        production_behavior_change=False,
    )


def select_filter_variant_by_residual_and_progress(
    variants: Sequence[Mapping[str, Any]],
    *,
    baseline_name: str = "baseline_p_armijo_or_best_residual",
    require_final_barrier_progress_if_baseline_reached: bool = True,
) -> AlgorithmV11FilterVariantChoice:
    """Select a completed diagnostic variant without weakening baseline progress.

    This helper is for post-run policy selection across independently evaluated
    globalization variants. If the baseline reaches the final barrier, a filter
    variant must also reach it before it can replace the baseline.
    """

    finite: list[dict[str, Any]] = []
    baseline: dict[str, Any] | None = None
    for index, variant in enumerate(variants):
        name = str(variant.get("config_name", variant.get("name", "")))
        try:
            residual_l2 = float(variant["final_residual_l2"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(residual_l2):
            continue
        row = {
            "index": index,
            "name": name,
            "residual_l2": residual_l2,
            "reached_final_barrier": bool(variant.get("reached_final_barrier", False)),
            "converged_at_final_barrier": bool(variant.get("converged_at_final_barrier", False)),
        }
        finite.append(row)
        if name == baseline_name:
            baseline = row

    if baseline is None:
        raise ValueError("variants must include the named baseline.")

    candidates = finite
    if (
        require_final_barrier_progress_if_baseline_reached
        and baseline["reached_final_barrier"]
    ):
        candidates = [row for row in finite if row["reached_final_barrier"]]
    if not candidates:
        candidates = [baseline]

    selected = min(
        candidates,
        key=lambda row: (
            float(row["residual_l2"]),
            0 if row["converged_at_final_barrier"] else 1,
            0 if row["reached_final_barrier"] else 1,
            int(row["index"]),
        ),
    )
    selected_reason = (
        "baseline_retained"
        if selected["name"] == baseline_name
        else "lower_residual_with_required_progress"
    )
    return AlgorithmV11FilterVariantChoice(
        choice_schema="exogibbs_algorithm_v11_filter_variant_choice_v1",
        selected=True,
        selected_name=str(selected["name"]),
        selected_index=int(selected["index"]),
        selected_residual_l2=float(selected["residual_l2"]),
        baseline_name=baseline_name,
        baseline_residual_l2=float(baseline["residual_l2"]),
        candidate_count=len(variants),
        finite_candidate_count=len(finite),
        require_final_barrier_progress_if_baseline_reached=bool(
            require_final_barrier_progress_if_baseline_reached
        ),
        selection_policy="residual_minimization_with_baseline_barrier_progress_guard",
        selected_reason=selected_reason,
        diagnostic_only=True,
        production_behavior_change=False,
    )


__all__ = (
    "AlgorithmV11FilterSelection",
    "AlgorithmV11FilterVariantChoice",
    "AlgorithmV11HTypeSelection",
    "AlgorithmV11NoArmijoRecoveryClassification",
    "AlgorithmV11SoftRestorationSelection",
    "AlgorithmV11FinalCenteringClassification",
    "AlgorithmV11PhysicalConvergenceGate",
    "DEFAULT_FILTER_COMPONENT_WEIGHTS",
    "GasStationarityFrameDecomposition",
    "classify_no_armijo_recovery",
    "classify_final_centering_frame",
    "constraint_violation_from_components",
    "decompose_gas_stationarity_frame",
    "evaluate_amount_frame_physical_convergence",
    "select_h_type_constraint_trial",
    "select_soft_restoration_trial",
    "select_filter_variant_by_residual_and_progress",
    "select_filter_restoration_trial",
)
