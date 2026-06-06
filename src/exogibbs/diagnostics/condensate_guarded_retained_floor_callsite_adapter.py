"""Dry-run callsite adapter for guarded retained-floor diagnostics.

This helper packages the selected guarded retained-floor candidate into an
explicit opt-in callsite payload. It does not call production solvers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from exogibbs.diagnostics.condensate_retained_floor_selector import (
    RetainedFloorCandidate,
    select_guarded_retained_floor_candidate,
)


@dataclass(frozen=True)
class GuardedRetainedFloorCallsiteCandidatePayload:
    """Candidate payload for a guarded retained-floor callsite dry run."""

    label: str
    support_indices: tuple[int, ...]
    support_amounts_init: tuple[float, ...]
    budget_residual: float
    kkt_residual: float
    budget_nonworse: bool
    kkt_improved: bool
    retained_amount_floor: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GuardedRetainedFloorCallsitePayload:
    """Selected explicit opt-in callsite payload."""

    payload_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    selected_candidate_label: str
    selection_reason: str
    support_indices: tuple[int, ...]
    support_amounts_init: tuple[float, ...]
    retained_amount_floor: float | None
    selected_budget_residual: float
    selected_kkt_residual: float
    fastchem4_trace_public_runtime_constructor_inputs_used: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_payload(candidate: GuardedRetainedFloorCallsiteCandidatePayload, name: str) -> None:
    if not candidate.label:
        raise ValueError(f"{name}.label must be non-empty.")
    if len(candidate.support_indices) != len(candidate.support_amounts_init):
        raise ValueError(f"{name} support indices and amounts must have the same length.")
    if any(int(index) < 0 for index in candidate.support_indices):
        raise ValueError(f"{name}.support_indices must be non-negative.")
    amounts = np.asarray(candidate.support_amounts_init, dtype=np.float64)
    if amounts.ndim != 1 or not np.all(np.isfinite(amounts)) or np.any(amounts < 0.0):
        raise ValueError(f"{name}.support_amounts_init must be finite and non-negative.")
    if candidate.budget_residual < 0.0 or not np.isfinite(candidate.budget_residual):
        raise ValueError(f"{name}.budget_residual must be finite and non-negative.")
    if candidate.kkt_residual < 0.0 or not np.isfinite(candidate.kkt_residual):
        raise ValueError(f"{name}.kkt_residual must be finite and non-negative.")
    if candidate.retained_amount_floor is not None and (
        candidate.retained_amount_floor <= 0.0
        or not np.isfinite(candidate.retained_amount_floor)
    ):
        raise ValueError(f"{name}.retained_amount_floor must be positive when provided.")


def build_guarded_retained_floor_callsite_payload(
    *,
    explicit_opt_in: bool,
    no_floor_payload: GuardedRetainedFloorCallsiteCandidatePayload,
    floor_payload: GuardedRetainedFloorCallsiteCandidatePayload,
) -> GuardedRetainedFloorCallsitePayload:
    """Select and package a guarded retained-floor callsite dry-run payload."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for guarded retained-floor callsite payloads.")
    _validate_payload(no_floor_payload, "no_floor_payload")
    _validate_payload(floor_payload, "floor_payload")
    selection = select_guarded_retained_floor_candidate(
        no_floor_candidate=RetainedFloorCandidate(
            label=no_floor_payload.label,
            budget_residual=no_floor_payload.budget_residual,
            kkt_residual=no_floor_payload.kkt_residual,
            budget_nonworse=no_floor_payload.budget_nonworse,
            kkt_improved=no_floor_payload.kkt_improved,
            retained_amount_floor=no_floor_payload.retained_amount_floor,
        ),
        floor_candidate=RetainedFloorCandidate(
            label=floor_payload.label,
            budget_residual=floor_payload.budget_residual,
            kkt_residual=floor_payload.kkt_residual,
            budget_nonworse=floor_payload.budget_nonworse,
            kkt_improved=floor_payload.kkt_improved,
            retained_amount_floor=floor_payload.retained_amount_floor,
        ),
    )
    selected_payload = (
        floor_payload
        if selection.selected_candidate.label == floor_payload.label
        else no_floor_payload
    )
    return GuardedRetainedFloorCallsitePayload(
        payload_schema="exogibbs_guarded_retained_floor_callsite_payload_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        selected_candidate_label=selection.selected_candidate.label,
        selection_reason=selection.selection_reason,
        support_indices=selected_payload.support_indices,
        support_amounts_init=selected_payload.support_amounts_init,
        retained_amount_floor=selected_payload.retained_amount_floor,
        selected_budget_residual=selection.selected_candidate.budget_residual,
        selected_kkt_residual=selection.selected_candidate.kkt_residual,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
    )


__all__ = (
    "GuardedRetainedFloorCallsiteCandidatePayload",
    "GuardedRetainedFloorCallsitePayload",
    "build_guarded_retained_floor_callsite_payload",
)
