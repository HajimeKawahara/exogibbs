"""Explicit diagnostic adapter for component-safe payload callsite inputs.

This module converts validated component-safe policy payloads into restricted
support callsite input records. It does not call production solvers, import
FastChem4, import pyfastchem, or wire any default behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from exogibbs.condensates.native_bundle import validate_native_bundle_provenance
from exogibbs.diagnostics.condensate_residual_balanced_direction import (
    ComponentSafePolicyPayload,
)


FORBIDDEN_PROVENANCE = {
    "fastchem4_trace",
    "fastchem4_public",
    "fastchem4_runtime",
    "branch_replay",
    "reference_fit",
    "unknown_reference",
}


@dataclass(frozen=True)
class ComponentSafeCallsiteInputs:
    """Restricted-support callsite input record derived from a component-safe payload."""

    adapter_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    case_id: str
    payload_kind: str
    selected_policy: str
    solver_call_policy: str
    support_indices: tuple[int, ...]
    support_amounts_init: tuple[float, ...]
    support_indices_shape_matches: bool
    support_amounts_init_shape_matches: bool
    finite_solver_inputs: bool
    normal_default_path_unchanged: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_provenance(field_provenance: Mapping[str, str] | None) -> None:
    provenance = validate_native_bundle_provenance(field_provenance or {})
    for value in provenance.values():
        if value in FORBIDDEN_PROVENANCE:
            raise ValueError("Forbidden reference provenance cannot enter callsite adapter inputs.")


def _as_index_vector(values: Sequence[int], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    if np.any(array < 0):
        raise ValueError(f"{name} must contain nonnegative indices.")
    if len(set(int(value) for value in array.tolist())) != array.shape[0]:
        raise ValueError(f"{name} must not contain duplicate indices.")
    return array


def _as_amount_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must contain positive values.")
    return array


def build_component_safe_callsite_inputs(
    *,
    payload: ComponentSafePolicyPayload | Mapping[str, Any],
    support_indices: Sequence[int] = (),
    support_amounts_init: Sequence[float] = (),
    explicit_opt_in: bool,
    max_abs_log_update: float = 50.0,
    field_provenance: Mapping[str, str] | None = None,
) -> ComponentSafeCallsiteInputs:
    """Convert a component-safe payload into restricted-support callsite inputs."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for component-safe callsite inputs.")
    _validate_provenance(field_provenance)
    data = payload.as_dict() if isinstance(payload, ComponentSafePolicyPayload) else dict(payload)
    payload_kind = str(data["payload_kind"])
    if payload_kind in {"stagnant_noop", "classified_skip"}:
        return ComponentSafeCallsiteInputs(
            adapter_schema="exogibbs_component_safe_callsite_inputs_v1",
            diagnostic_only=True,
            default_off=True,
            explicit_opt_in=True,
            production_behavior_change=False,
            production_return_signature_change=False,
            preset_default_wiring_change=False,
            fastchem4_trace_public_runtime_constructor_inputs_used=False,
            case_id=str(data["case_id"]),
            payload_kind=payload_kind,
            selected_policy=str(data["selected_policy"]),
            solver_call_policy=f"skip_solver_{payload_kind}",
            support_indices=(),
            support_amounts_init=(),
            support_indices_shape_matches=True,
            support_amounts_init_shape_matches=True,
            finite_solver_inputs=True,
            normal_default_path_unchanged=True,
        )
    if payload_kind != "update":
        raise ValueError("payload_kind must be update, stagnant_noop, or classified_skip.")
    indices = _as_index_vector(support_indices, "support_indices")
    amounts = _as_amount_vector(support_amounts_init, "support_amounts_init")
    if indices.shape[0] != amounts.shape[0]:
        raise ValueError("support_indices and support_amounts_init must have matching length.")
    if indices.shape[0] != int(data["support_size"]):
        raise ValueError("support input length must match payload support_size.")
    delta_m = np.asarray(data["delta_ln_mk"], dtype=np.float64)
    if delta_m.ndim != 1:
        raise ValueError("payload delta_ln_mk must be a one-dimensional vector.")
    if delta_m.shape[0] != amounts.shape[0]:
        raise ValueError("payload delta_ln_mk length must match support_amounts_init.")
    if not np.all(np.isfinite(delta_m)):
        raise ValueError("payload delta_ln_mk must contain finite values.")
    scaled_update = float(data["lambda_trial"]) * delta_m
    if scaled_update.size and np.max(np.abs(scaled_update)) > float(max_abs_log_update):
        raise ValueError("payload condensate log update exceeds max_abs_log_update.")
    updated_amounts = amounts * np.exp(scaled_update)
    finite = bool(np.all(np.isfinite(updated_amounts)) and np.all(updated_amounts > 0.0))
    return ComponentSafeCallsiteInputs(
        adapter_schema="exogibbs_component_safe_callsite_inputs_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        case_id=str(data["case_id"]),
        payload_kind=payload_kind,
        selected_policy=str(data["selected_policy"]),
        solver_call_policy="call_restricted_solver_explicit_opt_in",
        support_indices=tuple(int(value) for value in indices.tolist()),
        support_amounts_init=tuple(float(value) for value in updated_amounts.tolist()),
        support_indices_shape_matches=indices.shape[0] == updated_amounts.shape[0],
        support_amounts_init_shape_matches=indices.shape[0] == updated_amounts.shape[0],
        finite_solver_inputs=finite,
        normal_default_path_unchanged=True,
    )
