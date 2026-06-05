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
