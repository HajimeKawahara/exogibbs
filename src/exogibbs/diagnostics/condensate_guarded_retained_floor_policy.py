"""Policy gate helpers for guarded retained-floor diagnostics.

This module contains explicit default-off policy configuration and gate
validation for guarded retained-floor real-callsite experiments. It does not
call production solvers and does not import FastChem4.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class GuardedRetainedFloorPolicyConfig:
    """Explicit opt-in guarded retained-floor policy configuration."""

    explicit_opt_in: bool
    direct_solve_threshold: float
    retained_amount_update_factor: float
    retained_amount_floor: float
    budget_tolerance: float
    require_kkt_improvement: bool
    diagnostic_only: bool = True
    default_off: bool = True
    production_behavior_change: bool = False
    production_return_signature_change: bool = False
    preset_default_wiring_change: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GuardedRetainedFloorPolicyGateReport:
    """Policy gate validation report."""

    gate_schema: str
    gate_passed: bool
    config: GuardedRetainedFloorPolicyConfig
    case_count: int
    budget_safe_count: int
    kkt_improved_count: int
    finite_candidate_pair_count: int
    floor_selected_count: int
    failed_reasons: tuple[str, ...]
    diagnostic_only: bool
    default_off: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config"] = self.config.as_dict()
        return payload


def validate_guarded_retained_floor_policy_config(
    config: GuardedRetainedFloorPolicyConfig,
) -> None:
    """Validate that a guarded retained-floor policy is explicit and diagnostic."""

    if not config.explicit_opt_in:
        raise ValueError("guarded retained-floor policy requires explicit_opt_in.")
    if not config.diagnostic_only or not config.default_off:
        raise ValueError("guarded retained-floor policy must be diagnostic-only and default-off.")
    if config.production_behavior_change:
        raise ValueError("guarded retained-floor policy must not change production behavior.")
    if config.production_return_signature_change:
        raise ValueError("guarded retained-floor policy must not change return signatures.")
    if config.preset_default_wiring_change:
        raise ValueError("guarded retained-floor policy must not wire presets or defaults.")
    positive_fields = {
        "direct_solve_threshold": config.direct_solve_threshold,
        "retained_amount_update_factor": config.retained_amount_update_factor,
        "retained_amount_floor": config.retained_amount_floor,
        "budget_tolerance": config.budget_tolerance,
    }
    for name, value in positive_fields.items():
        if value <= 0.0 or not np.isfinite(value):
            raise ValueError(f"{name} must be positive and finite.")


def build_guarded_retained_floor_policy_gate_report(
    *,
    config: GuardedRetainedFloorPolicyConfig,
    selected_budget_nonworse: Sequence[bool],
    selected_kkt_improved: Sequence[bool],
    finite_candidate_pair: Sequence[bool],
    selected_candidate_labels: Sequence[str],
) -> GuardedRetainedFloorPolicyGateReport:
    """Build a production-adjacent gate report from selected case metrics."""

    validate_guarded_retained_floor_policy_config(config)
    lengths = {
        len(selected_budget_nonworse),
        len(selected_kkt_improved),
        len(finite_candidate_pair),
        len(selected_candidate_labels),
    }
    if len(lengths) != 1:
        raise ValueError("selected metric sequences must have the same length.")
    case_count = len(selected_budget_nonworse)
    if case_count == 0:
        raise ValueError("at least one case is required.")
    budget_safe_count = sum(bool(value) for value in selected_budget_nonworse)
    kkt_improved_count = sum(bool(value) for value in selected_kkt_improved)
    finite_count = sum(bool(value) for value in finite_candidate_pair)
    floor_selected_count = sum(label != "no_floor" for label in selected_candidate_labels)
    failed: list[str] = []
    if budget_safe_count != case_count:
        failed.append("not_all_selected_cases_are_budget_safe")
    if config.require_kkt_improvement and kkt_improved_count != case_count:
        failed.append("not_all_selected_cases_improve_kkt_diagnostic")
    if finite_count != case_count:
        failed.append("not_all_candidate_pairs_are_finite")
    return GuardedRetainedFloorPolicyGateReport(
        gate_schema="exogibbs_guarded_retained_floor_policy_gate_report_v1",
        gate_passed=not failed,
        config=config,
        case_count=case_count,
        budget_safe_count=budget_safe_count,
        kkt_improved_count=kkt_improved_count,
        finite_candidate_pair_count=finite_count,
        floor_selected_count=floor_selected_count,
        failed_reasons=tuple(failed),
        diagnostic_only=True,
        default_off=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
    )


__all__ = (
    "GuardedRetainedFloorPolicyConfig",
    "GuardedRetainedFloorPolicyGateReport",
    "build_guarded_retained_floor_policy_gate_report",
    "validate_guarded_retained_floor_policy_config",
)
