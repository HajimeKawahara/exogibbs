"""Default-off positive-support selection diagnostics for condensates.

This module selects positive condensate support candidates from explicit
ExoGibbs-native arrays. It does not import FastChem4, call pyfastchem, call
production solvers, or connect to presets/defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from exogibbs.condensates.native_bundle import validate_native_bundle_provenance


@dataclass(frozen=True)
class PositiveSupportSelectionReport:
    """Diagnostic report for positive condensate support selection."""

    diagnostic_only: bool
    default_off: bool
    production_behavior_change: bool
    policy_schema: str
    policy_name: str
    positive_support_indices: tuple[int, ...]
    positive_support_names: tuple[str, ...]
    zero_bound_candidate_indices: tuple[int, ...]
    zero_bound_candidate_names: tuple[str, ...]
    candidate_scores: Mapping[str, float]
    max_positive_support_count: int
    require_thermochemical_favorable: bool
    field_provenance: Mapping[str, str]
    fastchem4_trace_values_used: bool
    fastchem4_public_values_used_as_constructor_inputs: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActivityDrivenSupportSelectionReport:
    """Native activity-driven condensate support selection report."""

    diagnostic_only: bool
    default_off: bool
    production_behavior_change: bool
    policy_schema: str
    policy_name: str
    positive_support_indices: tuple[int, ...]
    positive_support_names: tuple[str, ...]
    inactive_positive_indices: tuple[int, ...]
    inactive_positive_names: tuple[str, ...]
    candidate_driving: Mapping[str, float]
    candidate_capacity: Mapping[str, float]
    candidate_temperature_valid: Mapping[str, bool]
    temperature: float | None
    activity_threshold: float
    max_positive_support_count: int | None
    field_provenance: Mapping[str, str]
    fastchem4_trace_values_used: bool
    fastchem4_public_values_used_as_constructor_inputs: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_vector(values: Sequence[float], name: str, expected: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    if expected is not None and array.shape[0] != expected:
        raise ValueError(f"{name} length mismatch: got {array.shape[0]}, expected {expected}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_matrix(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _capacity(column: np.ndarray, target: np.ndarray) -> float:
    positive = column > 0.0
    if not np.any(positive):
        return float("inf")
    budgets = target[positive]
    if np.any(budgets <= 0.0):
        return 0.0
    return float(np.min(budgets / column[positive]))


def select_positive_support_candidates(
    *,
    formula_matrix_cond: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    condensate_species_order: Sequence[str],
    hvector_cond: Sequence[float],
    max_positive_support_count: int = 1,
    require_thermochemical_favorable: bool = True,
    field_provenance: Mapping[str, str] | None = None,
) -> PositiveSupportSelectionReport:
    """Select positive support candidates from native condensate arrays."""

    provenance = validate_native_bundle_provenance(field_provenance)
    ac = _as_matrix(formula_matrix_cond, "formula_matrix_cond")
    ncond = ac.shape[1]
    target = _as_vector(element_inventory_target, "element_inventory_target", ac.shape[0])
    hcond = _as_vector(hvector_cond, "hvector_cond", ncond)
    species = tuple(str(item) for item in condensate_species_order)
    if len(species) != ncond:
        raise ValueError("condensate_species_order length must match formula_matrix_cond columns.")
    if int(max_positive_support_count) <= 0:
        raise ValueError("max_positive_support_count must be positive.")

    candidates = []
    scores: dict[str, float] = {}
    for index, label in enumerate(species):
        cap = _capacity(ac[:, index], target)
        if cap <= 0.0:
            continue
        if require_thermochemical_favorable and hcond[index] >= 0.0:
            scores[label] = float("inf")
            continue
        score = float(hcond[index])
        scores[label] = score
        candidates.append((score, index))
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected = tuple(index for _, index in candidates[: int(max_positive_support_count)])
    zero_bound = tuple(index for index in range(ncond) if index not in set(selected))
    return PositiveSupportSelectionReport(
        diagnostic_only=True,
        default_off=True,
        production_behavior_change=False,
        policy_schema="exogibbs_condensate_positive_support_selection_policy_v1",
        policy_name="thermochemical_topk_budget_capacity",
        positive_support_indices=selected,
        positive_support_names=tuple(species[index] for index in selected),
        zero_bound_candidate_indices=zero_bound,
        zero_bound_candidate_names=tuple(species[index] for index in zero_bound),
        candidate_scores=scores,
        max_positive_support_count=int(max_positive_support_count),
        require_thermochemical_favorable=bool(require_thermochemical_favorable),
        field_provenance={
            "formula_matrix_cond": provenance.get("formula_matrix_cond", "exogibbs_native"),
            "element_inventory_target": provenance.get(
                "element_inventory_target", "exogibbs_native"
            ),
            "hvector_cond": provenance.get("hvector_cond", "exogibbs_native_thermochemistry"),
            "positive_support": "derived_from_native_thermochemistry_and_budget_capacity",
        },
        fastchem4_trace_values_used=False,
        fastchem4_public_values_used_as_constructor_inputs=False,
    )


def select_activity_driven_support_candidates(
    *,
    formula_matrix_cond: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    condensate_species_order: Sequence[str],
    hvector_cond: Sequence[float],
    element_potential: Sequence[float],
    max_positive_support_count: int | None = 12,
    activity_threshold: float = 0.0,
    existing_support_indices: Sequence[int] = (),
    temperature: float | None = None,
    condensate_temperature_validity_upper: Sequence[float] | None = None,
    field_provenance: Mapping[str, str] | None = None,
) -> ActivityDrivenSupportSelectionReport:
    """Select condensate support candidates from native activity driving."""

    provenance = validate_native_bundle_provenance(field_provenance)
    ac = _as_matrix(formula_matrix_cond, "formula_matrix_cond")
    ncond = ac.shape[1]
    target = _as_vector(element_inventory_target, "element_inventory_target", ac.shape[0])
    hcond = _as_vector(hvector_cond, "hvector_cond", ncond)
    potential = _as_vector(element_potential, "element_potential", ac.shape[0])
    species = tuple(str(item) for item in condensate_species_order)
    if len(species) != ncond:
        raise ValueError("condensate_species_order length must match formula_matrix_cond columns.")
    if max_positive_support_count is not None and int(max_positive_support_count) <= 0:
        raise ValueError("max_positive_support_count must be positive.")
    support_existing = tuple(int(index) for index in existing_support_indices)
    if len(set(support_existing)) != len(support_existing):
        raise ValueError("existing_support_indices must not contain duplicates.")
    if any(index < 0 or index >= ncond for index in support_existing):
        raise ValueError("existing_support_indices contain an out-of-range condensate index.")
    if condensate_temperature_validity_upper is None:
        validity_upper = np.full((ncond,), np.inf, dtype=np.float64)
    else:
        validity_upper = _as_vector(
            condensate_temperature_validity_upper,
            "condensate_temperature_validity_upper",
            ncond,
        )

    driving = ac.T @ potential - hcond
    candidates: list[tuple[float, float, int]] = []
    scores: dict[str, float] = {}
    capacities: dict[str, float] = {}
    temperature_valid: dict[str, bool] = {}
    threshold = float(activity_threshold)
    existing_set = set(support_existing)
    temp_value = None if temperature is None else float(temperature)
    for index, label in enumerate(species):
        cap = _capacity(ac[:, index], target)
        capacities[label] = float(cap)
        score = float(driving[index])
        scores[label] = score
        is_temperature_valid = temp_value is None or temp_value <= float(validity_upper[index])
        temperature_valid[label] = bool(is_temperature_valid)
        if not is_temperature_valid:
            continue
        if cap <= 0.0 or not np.isfinite(cap):
            continue
        if score <= threshold:
            continue
        candidates.append((cap, score, index))
    ordered_indices = [
        int(index)
        for _, _, index in sorted(candidates, key=lambda item: (-item[0], -item[1], item[2]))
    ]
    selected = (
        tuple(ordered_indices)
        if max_positive_support_count is None
        else tuple(ordered_indices[: int(max_positive_support_count)])
    )
    inactive_positive = tuple(index for index in ordered_indices if index not in existing_set)
    return ActivityDrivenSupportSelectionReport(
        diagnostic_only=True,
        default_off=True,
        production_behavior_change=False,
        policy_schema="exogibbs_condensate_activity_driven_support_selection_policy_v1",
        policy_name=(
            "native_activity_threshold_capacity_priority_all_positive"
            if max_positive_support_count is None
            else "native_activity_threshold_capacity_priority_bounded_topk"
        ),
        positive_support_indices=selected,
        positive_support_names=tuple(species[index] for index in selected),
        inactive_positive_indices=inactive_positive,
        inactive_positive_names=tuple(species[index] for index in inactive_positive),
        candidate_driving=scores,
        candidate_capacity=capacities,
        candidate_temperature_valid=temperature_valid,
        temperature=temp_value,
        activity_threshold=threshold,
        max_positive_support_count=(
            None if max_positive_support_count is None else int(max_positive_support_count)
        ),
        field_provenance={
            "formula_matrix_cond": provenance.get("formula_matrix_cond", "exogibbs_native"),
            "element_inventory_target": provenance.get(
                "element_inventory_target", "exogibbs_native"
            ),
            "hvector_cond": provenance.get("hvector_cond", "exogibbs_native_thermochemistry"),
            "element_potential": provenance.get("element_potential", "exogibbs_native_gas_state"),
            "condensate_temperature_validity_upper": provenance.get(
                "condensate_temperature_validity_upper",
                "not_provided",
            ),
            "positive_support": "derived_from_native_activity_driving",
        },
        fastchem4_trace_values_used=False,
        fastchem4_public_values_used_as_constructor_inputs=False,
    )
