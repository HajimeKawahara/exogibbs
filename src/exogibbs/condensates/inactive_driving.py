"""Inactive condensate driving diagnostics.

These helpers evaluate condensate activity driving for an already-computed gas
and condensate state.  They are diagnostic-only and do not change solver state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class InactiveDrivingSummary:
    """Positive inactive-driving summary for one condensate subset."""

    max_positive_inactive_driving: float
    positive_inactive_count: int
    top_positive_inactive: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class InactiveDrivingReport:
    """Diagnostic report for inactive condensate driving."""

    report_schema: str
    diagnostic_only: bool
    default_off: bool
    production_behavior_change: bool
    temperature: Optional[float]
    activity_threshold: float
    active_floor: float
    all_condensates: InactiveDrivingSummary
    temperature_valid_condensates: InactiveDrivingSummary
    temperature_invalid_positive_inactive_count: int
    temperature_invalid_max_positive_inactive_driving: float
    candidate_temperature_valid: Mapping[str, bool]
    fastchem4_trace_public_runtime_constructor_inputs_used: bool

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["all_condensates"] = asdict(self.all_condensates)
        payload["temperature_valid_condensates"] = asdict(
            self.temperature_valid_condensates
        )
        return payload


def _as_vector(
    values: Sequence[float],
    name: str,
    expected: Optional[int] = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    if expected is not None and array.shape[0] != expected:
        raise ValueError(
            f"{name} length mismatch: got {array.shape[0]}, expected {expected}."
        )
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


def _positive_inactive_summary(
    *,
    indices: np.ndarray,
    condensate_species_order: tuple[str, ...],
    driving: np.ndarray,
    amounts: np.ndarray,
) -> InactiveDrivingSummary:
    top = sorted(
        (
            {
                "index": int(index),
                "species": condensate_species_order[int(index)],
                "driving": float(driving[int(index)]),
                "amount": float(amounts[int(index)]),
            }
            for index in indices
        ),
        key=lambda row: float(row["driving"]),
        reverse=True,
    )
    return InactiveDrivingSummary(
        max_positive_inactive_driving=float(top[0]["driving"]) if top else 0.0,
        positive_inactive_count=len(top),
        top_positive_inactive=tuple(top[:20]),
    )


def evaluate_inactive_condensate_driving(
    *,
    formula_matrix_cond: Sequence[Sequence[float]],
    condensate_species_order: Sequence[str],
    condensate_amounts: Sequence[float],
    hvector_cond: Sequence[float],
    element_potential: Sequence[float],
    temperature: Optional[float] = None,
    condensate_temperature_validity_upper: Optional[Sequence[float]] = None,
    active_floor: float = 1.0e-50,
    activity_threshold: float = 0.0,
) -> InactiveDrivingReport:
    """Evaluate inactive positive condensate driving with validity bookkeeping."""

    ac = _as_matrix(formula_matrix_cond, "formula_matrix_cond")
    ncond = ac.shape[1]
    species = tuple(str(item) for item in condensate_species_order)
    if len(species) != ncond:
        raise ValueError(
            "condensate_species_order length must match formula_matrix_cond columns."
        )
    amounts = _as_vector(condensate_amounts, "condensate_amounts", ncond)
    hcond = _as_vector(hvector_cond, "hvector_cond", ncond)
    potential = _as_vector(element_potential, "element_potential", ac.shape[0])
    if condensate_temperature_validity_upper is None:
        validity_upper = np.full((ncond,), np.inf, dtype=np.float64)
    else:
        validity_upper = _as_vector(
            condensate_temperature_validity_upper,
            "condensate_temperature_validity_upper",
            ncond,
        )
    temp_value = None if temperature is None else float(temperature)
    temperature_valid_mask = (
        np.ones((ncond,), dtype=bool)
        if temp_value is None
        else temp_value <= validity_upper
    )
    driving = ac.T @ potential - hcond
    active = amounts > float(active_floor)
    inactive_positive = (~active) & (driving > float(activity_threshold))
    all_indices = np.where(inactive_positive)[0]
    valid_indices = np.where(inactive_positive & temperature_valid_mask)[0]
    invalid_indices = np.where(inactive_positive & (~temperature_valid_mask))[0]
    invalid_summary = _positive_inactive_summary(
        indices=invalid_indices,
        condensate_species_order=species,
        driving=driving,
        amounts=amounts,
    )
    return InactiveDrivingReport(
        report_schema="exogibbs_inactive_condensate_driving_report_v1",
        diagnostic_only=True,
        default_off=True,
        production_behavior_change=False,
        temperature=temp_value,
        activity_threshold=float(activity_threshold),
        active_floor=float(active_floor),
        all_condensates=_positive_inactive_summary(
            indices=all_indices,
            condensate_species_order=species,
            driving=driving,
            amounts=amounts,
        ),
        temperature_valid_condensates=_positive_inactive_summary(
            indices=valid_indices,
            condensate_species_order=species,
            driving=driving,
            amounts=amounts,
        ),
        temperature_invalid_positive_inactive_count=(
            invalid_summary.positive_inactive_count
        ),
        temperature_invalid_max_positive_inactive_driving=(
            invalid_summary.max_positive_inactive_driving
        ),
        candidate_temperature_valid={
            species[index]: bool(temperature_valid_mask[index])
            for index in range(ncond)
        },
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
    )
