"""Ipopt-informed diagnostic filter helpers for condensate PD-IPM trials.

This module is explicit-import only. It does not call production solvers,
FastChem4, pyfastchem, presets, or default wiring. The helpers are intended to
separate objective descent failures from residual-progress candidates during
condensate PD-IPM diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class IpoptFilterScaleReport:
    """Component scale report for Ipopt-style filter diagnostics."""

    report_schema: str
    component_scales: dict[str, float]
    component_weights: dict[str, float]
    global_scale_floor: float
    component_scale_floors: dict[str, float]
    ipopt_unit_floor_enabled: bool
    diagnostic_only: bool
    production_behavior_change: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IpoptHTypeFilterSelection:
    """Ipopt-style h-type filter selection report."""

    selection_schema: str
    selected: bool
    selected_index: int | None
    selected_alpha: float | None
    selected_theta: float | None
    selected_p_merit: float | None
    current_theta: float
    current_p_merit: float | None
    finite_trial_count: int
    accepted_trial_count: int
    rejected_trial_count: int
    theta_reduction_fraction: float
    component_scales: dict[str, float]
    component_weights: dict[str, float]
    protected_component_max_normalized_increase: float | None
    protected_components: tuple[str, ...]
    selected_reason: str | None
    selection_policy: str
    diagnostic_only: bool
    production_behavior_change: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IpoptFilterEntry:
    """Persistent filter entry for diagnostic objective and violation pairs."""

    p_merit: float
    theta: float
    iteration: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IpoptPersistentFilterReport:
    """Persistent filter acceptability report."""

    report_schema: str
    acceptable: bool
    p_merit: float
    theta: float
    entry_count: int
    blocking_entries: tuple[dict[str, Any], ...]
    gamma_p: float
    gamma_theta: float
    theta_max: float
    selected_reason: str
    diagnostic_only: bool
    production_behavior_change: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_nonnegative(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return number


def _finite_number(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _validate_weights(component_weights: Mapping[str, float]) -> dict[str, float]:
    weights: dict[str, float] = {}
    if not component_weights:
        raise ValueError("component_weights must not be empty.")
    for name, value in component_weights.items():
        weights[str(name)] = _finite_nonnegative(value, "component weight")
    return weights


def build_ipopt_component_scales(
    current_components: Mapping[str, Any],
    *,
    component_weights: Mapping[str, float],
    global_scale_floor: float = 1.0,
    component_scale_floors: Mapping[str, float] | None = None,
    ipopt_unit_floor_enabled: bool = True,
) -> IpoptFilterScaleReport:
    """Build component scales with a floor for zero current residuals.

    Ipopt derives filter thresholds from constraint violation scales that do
    not collapse when the current violation is zero. This helper applies the
    same diagnostic idea component-by-component.
    """

    weights = _validate_weights(component_weights)
    global_floor = _finite_nonnegative(global_scale_floor, "global_scale_floor")
    if global_floor == 0.0:
        raise ValueError("global_scale_floor must be positive.")
    floors = {
        str(name): _finite_nonnegative(value, "component scale floor")
        for name, value in (component_scale_floors or {}).items()
    }
    scales: dict[str, float] = {}
    for name in weights:
        value = float(current_components.get(name, math.nan))
        if not math.isfinite(value):
            raise ValueError("current_components must contain finite weighted components.")
        candidates = [abs(value), global_floor, floors.get(name, 0.0)]
        if ipopt_unit_floor_enabled:
            candidates.append(1.0)
        scale = max(candidates)
        if scale <= 0.0 or not math.isfinite(scale):
            raise ValueError("computed component scales must be finite and positive.")
        scales[name] = float(scale)
    return IpoptFilterScaleReport(
        report_schema="exogibbs_ipopt_filter_scale_report_v1",
        component_scales=scales,
        component_weights=weights,
        global_scale_floor=global_floor,
        component_scale_floors=floors,
        ipopt_unit_floor_enabled=bool(ipopt_unit_floor_enabled),
        diagnostic_only=True,
        production_behavior_change=False,
    )


def ipopt_scaled_theta(
    components: Mapping[str, Any],
    *,
    component_weights: Mapping[str, float],
    component_scales: Mapping[str, float],
) -> float:
    """Return a component-scaled filter violation."""

    weights = _validate_weights(component_weights)
    if set(weights) - set(component_scales):
        raise ValueError("component_scales must define every weighted component.")
    total = 0.0
    for name, weight in weights.items():
        value = float(components.get(name, math.nan))
        scale = float(component_scales[name])
        if not math.isfinite(value):
            return math.inf
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("component_scales must be finite and positive.")
        total += weight * value / scale
    return float(total)


def _protected_components_pass(
    *,
    current_components: Mapping[str, Any],
    trial_components: Mapping[str, Any],
    component_scales: Mapping[str, float],
    protected_components: Sequence[str],
    max_normalized_increase: float | None,
) -> bool:
    if max_normalized_increase is None:
        return True
    limit = _finite_nonnegative(max_normalized_increase, "max_normalized_increase")
    for name in protected_components:
        current = float(current_components.get(name, math.nan))
        trial = float(trial_components.get(name, math.nan))
        scale = float(component_scales.get(name, math.nan))
        if not (math.isfinite(current) and math.isfinite(trial) and math.isfinite(scale)):
            return False
        if scale <= 0.0:
            return False
        if (trial - current) / scale > limit:
            return False
    return True


def select_ipopt_h_type_filter_trial(
    trials: Sequence[Mapping[str, Any]],
    *,
    current_components: Mapping[str, Any],
    current_p_merit: float | None = None,
    component_weights: Mapping[str, float],
    component_scale_floors: Mapping[str, float] | None = None,
    global_scale_floor: float = 1.0,
    theta_reduction_fraction: float = 1.0e-4,
    protected_components: Sequence[str] = (),
    protected_component_max_normalized_increase: float | None = None,
    choose_largest_alpha: bool = True,
) -> IpoptHTypeFilterSelection:
    """Select an h-type trial using scale floors and optional guard components."""

    scale_report = build_ipopt_component_scales(
        current_components,
        component_weights=component_weights,
        global_scale_floor=global_scale_floor,
        component_scale_floors=component_scale_floors,
        ipopt_unit_floor_enabled=True,
    )
    theta_fraction = float(theta_reduction_fraction)
    if not math.isfinite(theta_fraction) or theta_fraction < 0.0 or theta_fraction >= 1.0:
        raise ValueError("theta_reduction_fraction must be finite and in [0, 1).")
    current_theta = ipopt_scaled_theta(
        current_components,
        component_weights=scale_report.component_weights,
        component_scales=scale_report.component_scales,
    )
    if not math.isfinite(current_theta):
        raise ValueError("current_components must define a finite scaled theta.")
    current_merit_value = None if current_p_merit is None else float(current_p_merit)
    if current_merit_value is not None and not math.isfinite(current_merit_value):
        raise ValueError("current_p_merit must be finite when provided.")

    finite_trials: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    rejected = 0
    for index, trial in enumerate(trials):
        components = trial.get("residual_components", {})
        try:
            alpha = float(trial.get("alpha"))
            p_merit = float(trial.get("p_merit", math.nan))
            trial_theta = ipopt_scaled_theta(
                components,
                component_weights=scale_report.component_weights,
                component_scales=scale_report.component_scales,
            )
        except (TypeError, ValueError):
            rejected += 1
            continue
        all_finite = bool(
            trial.get(
                "all_finite",
                math.isfinite(alpha) and math.isfinite(trial_theta),
            )
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
        record = {
            "index": int(index),
            "alpha": float(alpha),
            "theta": float(trial_theta),
            "p_merit": None if not math.isfinite(p_merit) else float(p_merit),
        }
        finite_trials.append(record)
        theta_accepts = trial_theta <= (1.0 - theta_fraction) * current_theta
        protected_accepts = _protected_components_pass(
            current_components=current_components,
            trial_components=components,
            component_scales=scale_report.component_scales,
            protected_components=protected_components,
            max_normalized_increase=protected_component_max_normalized_increase,
        )
        if theta_accepts and protected_accepts:
            accepted.append(record)

    if not accepted:
        return IpoptHTypeFilterSelection(
            selection_schema="exogibbs_ipopt_h_type_filter_selection_v1",
            selected=False,
            selected_index=None,
            selected_alpha=None,
            selected_theta=None,
            selected_p_merit=None,
            current_theta=float(current_theta),
            current_p_merit=current_merit_value,
            finite_trial_count=len(finite_trials),
            accepted_trial_count=0,
            rejected_trial_count=rejected,
            theta_reduction_fraction=theta_fraction,
            component_scales=scale_report.component_scales,
            component_weights=scale_report.component_weights,
            protected_component_max_normalized_increase=(
                None
                if protected_component_max_normalized_increase is None
                else float(protected_component_max_normalized_increase)
            ),
            protected_components=tuple(str(name) for name in protected_components),
            selected_reason=None,
            selection_policy="ipopt_informed_h_type_scaled_theta",
            diagnostic_only=True,
            production_behavior_change=False,
        )

    selected = (
        max(accepted, key=lambda row: (float(row["alpha"]), -float(row["theta"])))
        if choose_largest_alpha
        else min(accepted, key=lambda row: (float(row["theta"]), -float(row["alpha"])))
    )
    return IpoptHTypeFilterSelection(
        selection_schema="exogibbs_ipopt_h_type_filter_selection_v1",
        selected=True,
        selected_index=int(selected["index"]),
        selected_alpha=float(selected["alpha"]),
        selected_theta=float(selected["theta"]),
        selected_p_merit=selected["p_merit"],
        current_theta=float(current_theta),
        current_p_merit=current_merit_value,
        finite_trial_count=len(finite_trials),
        accepted_trial_count=len(accepted),
        rejected_trial_count=rejected,
        theta_reduction_fraction=theta_fraction,
        component_scales=scale_report.component_scales,
        component_weights=scale_report.component_weights,
        protected_component_max_normalized_increase=(
            None
            if protected_component_max_normalized_increase is None
            else float(protected_component_max_normalized_increase)
        ),
        protected_components=tuple(str(name) for name in protected_components),
        selected_reason="scaled_theta_progress",
        selection_policy="ipopt_informed_h_type_scaled_theta",
        diagnostic_only=True,
        production_behavior_change=False,
    )


def is_acceptable_to_persistent_filter(
    *,
    p_merit: float,
    theta: float,
    entries: Sequence[IpoptFilterEntry | Mapping[str, Any]],
    gamma_p: float = 1.0e-8,
    gamma_theta: float = 1.0e-5,
    theta_max: float = math.inf,
) -> IpoptPersistentFilterReport:
    """Check Ipopt-style persistent filter acceptability.

    A trial pair is acceptable to every stored entry if it improves either the
    objective-like component or the violation-like component relative to that
    entry. This mirrors the diagnostic role of Ipopt's filter memory without
    wiring it into production solvers.
    """

    p_value = _finite_number(p_merit, "p_merit")
    theta_value = _finite_nonnegative(theta, "theta")
    gamma_p_value = _finite_nonnegative(gamma_p, "gamma_p")
    gamma_theta_value = _finite_nonnegative(gamma_theta, "gamma_theta")
    theta_max_value = float(theta_max)
    if not math.isfinite(theta_max_value):
        theta_max_value = math.inf
    elif theta_max_value < 0.0:
        raise ValueError("theta_max must be finite and non-negative or infinity.")
    if theta_value > theta_max_value:
        return IpoptPersistentFilterReport(
            report_schema="exogibbs_ipopt_persistent_filter_report_v1",
            acceptable=False,
            p_merit=p_value,
            theta=theta_value,
            entry_count=len(entries),
            blocking_entries=(),
            gamma_p=gamma_p_value,
            gamma_theta=gamma_theta_value,
            theta_max=theta_max_value,
            selected_reason="theta_exceeds_theta_max",
            diagnostic_only=True,
            production_behavior_change=False,
        )

    blocking: list[dict[str, Any]] = []
    normalized_entries: list[IpoptFilterEntry] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, IpoptFilterEntry):
            normalized = entry
        else:
            normalized = IpoptFilterEntry(
                p_merit=_finite_number(entry["p_merit"], "entry p_merit"),
                theta=_finite_nonnegative(entry["theta"], "entry theta"),
                iteration=int(entry.get("iteration", index)),
            )
        normalized_entries.append(normalized)
        p_accepts = p_value <= normalized.p_merit - gamma_p_value * theta_value
        theta_accepts = theta_value <= (1.0 - gamma_theta_value) * normalized.theta
        if not (p_accepts or theta_accepts):
            blocking.append(normalized.as_dict())

    return IpoptPersistentFilterReport(
        report_schema="exogibbs_ipopt_persistent_filter_report_v1",
        acceptable=not blocking,
        p_merit=p_value,
        theta=theta_value,
        entry_count=len(normalized_entries),
        blocking_entries=tuple(blocking),
        gamma_p=gamma_p_value,
        gamma_theta=gamma_theta_value,
        theta_max=theta_max_value,
        selected_reason="acceptable_to_filter" if not blocking else "blocked_by_filter_memory",
        diagnostic_only=True,
        production_behavior_change=False,
    )


def augment_persistent_filter(
    entries: Sequence[IpoptFilterEntry | Mapping[str, Any]],
    *,
    p_merit: float,
    theta: float,
    iteration: int,
    gamma_p: float = 1.0e-8,
    gamma_theta: float = 1.0e-5,
) -> tuple[IpoptFilterEntry, ...]:
    """Return a filter with dominated entries removed and a new entry added."""

    p_value = _finite_number(p_merit, "p_merit")
    theta_value = _finite_nonnegative(theta, "theta")
    gamma_p_value = _finite_nonnegative(gamma_p, "gamma_p")
    gamma_theta_value = _finite_nonnegative(gamma_theta, "gamma_theta")
    new_entry = IpoptFilterEntry(p_merit=p_value, theta=theta_value, iteration=int(iteration))
    kept: list[IpoptFilterEntry] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, IpoptFilterEntry):
            normalized = entry
        else:
            normalized = IpoptFilterEntry(
                p_merit=_finite_number(entry["p_merit"], "entry p_merit"),
                theta=_finite_nonnegative(entry["theta"], "entry theta"),
                iteration=int(entry.get("iteration", index)),
            )
        dominated_by_new = (
            new_entry.p_merit <= normalized.p_merit - gamma_p_value * new_entry.theta
            and new_entry.theta <= (1.0 - gamma_theta_value) * normalized.theta
        )
        if not dominated_by_new:
            kept.append(normalized)
    kept.append(new_entry)
    return tuple(kept)


__all__ = (
    "IpoptFilterEntry",
    "IpoptFilterScaleReport",
    "IpoptHTypeFilterSelection",
    "IpoptPersistentFilterReport",
    "augment_persistent_filter",
    "build_ipopt_component_scales",
    "is_acceptable_to_persistent_filter",
    "ipopt_scaled_theta",
    "select_ipopt_h_type_filter_trial",
)
