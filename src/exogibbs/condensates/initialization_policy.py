"""Default-off condensate initialization policy diagnostics.

This module recommends conservative native condensate seed amounts from
explicit ExoGibbs-native stoichiometry and elemental budgets. It does not
import FastChem4, call pyfastchem, call production solvers, or connect to
presets/defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from exogibbs.condensates.native_bundle import validate_native_bundle_provenance


@dataclass(frozen=True)
class CondensateSeedPolicyReport:
    """Diagnostic seed policy report for native condensate initialization."""

    diagnostic_only: bool
    default_off: bool
    production_behavior_change: bool
    policy_schema: str
    support_indices: tuple[int, ...]
    support_names: tuple[str, ...]
    recommended_amounts: tuple[float, ...]
    recommended_ln_amounts: tuple[float, ...]
    capacity_limited_amounts: tuple[float, ...]
    amount_gauge: str
    fastchem4_first_step_equivalent_gauge: str
    seed_fraction: float
    max_seed_amount: float
    min_seed_amount: float
    preserve_budget_fraction: bool
    field_provenance: Mapping[str, str]
    fastchem4_trace_values_used: bool
    fastchem4_public_values_used_as_constructor_inputs: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
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


def _support_tuple(support_indices: Sequence[int], ncond: int) -> tuple[int, ...]:
    support = tuple(int(index) for index in support_indices)
    if not support:
        raise ValueError("support_indices must be non-empty.")
    if len(set(support)) != len(support):
        raise ValueError("support_indices must not contain duplicates.")
    if any(index < 0 or index >= ncond for index in support):
        raise ValueError("support_indices contains an out-of-range index.")
    return support


def _column_capacity(column: np.ndarray, budget: np.ndarray) -> float:
    positive = column > 0.0
    if not np.any(positive):
        return float("inf")
    positive_budget = budget[positive]
    if np.any(positive_budget <= 0.0):
        return 0.0
    return float(np.min(positive_budget / column[positive]))


def recommend_budget_preserving_seed_amounts(
    *,
    formula_matrix_cond: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    condensate_species_order: Sequence[str],
    support_indices: Sequence[int],
    seed_fraction: float = 1.0e-6,
    max_seed_amount: float = 1.0e-6,
    min_seed_amount: float = 1.0e-300,
    preserve_budget_fraction: bool = True,
    field_provenance: Mapping[str, str] | None = None,
) -> CondensateSeedPolicyReport:
    """Recommend seed amounts from native budget capacity.

    By default, the seeds are globally rescaled so their combined elemental
    burden consumes no more than ``seed_fraction`` of any available budget.
    The shared-budget limit takes precedence over ``min_seed_amount`` when the
    two constraints are incompatible.
    Setting ``preserve_budget_fraction=False`` keeps the per-species
    ``seed_fraction * capacity`` amounts without this shared-budget rescale.
    """

    provenance = validate_native_bundle_provenance(field_provenance)
    ac = _as_matrix(formula_matrix_cond, "formula_matrix_cond")
    target = _as_vector(element_inventory_target, "element_inventory_target")
    species = tuple(str(item) for item in condensate_species_order)
    seed_fraction_value = float(seed_fraction)
    max_seed_amount_value = float(max_seed_amount)
    min_seed_amount_value = float(min_seed_amount)
    if ac.shape[0] != target.shape[0]:
        raise ValueError("formula_matrix_cond rows must match element_inventory_target length.")
    if ac.shape[1] != len(species):
        raise ValueError("formula_matrix_cond columns must match condensate_species_order length.")
    if not np.isfinite(seed_fraction_value) or seed_fraction_value <= 0.0:
        raise ValueError("seed_fraction must be finite and positive.")
    if not np.isfinite(max_seed_amount_value) or max_seed_amount_value <= 0.0:
        raise ValueError("max_seed_amount must be finite and positive.")
    if not np.isfinite(min_seed_amount_value) or min_seed_amount_value <= 0.0:
        raise ValueError("min_seed_amount must be finite and positive.")
    if min_seed_amount_value > max_seed_amount_value:
        raise ValueError("min_seed_amount must not exceed max_seed_amount.")
    support = _support_tuple(support_indices, ac.shape[1])

    capacity_limited = []
    recommended = []
    for index in support:
        capacity = _column_capacity(ac[:, index], target)
        if capacity <= 0.0:
            raise ValueError(
                "support_indices contains a condensate that cannot receive "
                "a positive seed from the element inventory target."
            )
        raw = seed_fraction_value * capacity if np.isfinite(capacity) else max_seed_amount_value
        bounded = min(max_seed_amount_value, max(min_seed_amount_value, raw))
        capacity_limited.append(float(capacity))
        recommended.append(float(bounded))
    recommended_array = np.asarray(recommended, dtype=np.float64)
    if bool(preserve_budget_fraction):
        positive_target = target > 0.0
        relative_burden = (
            ac[positive_target][:, np.asarray(support, dtype=np.int64)]
            / target[positive_target, None]
        ) @ recommended_array
        if not np.all(np.isfinite(relative_burden)):
            raise ValueError("Seed burden calculation produced a non-finite value.")
        if np.any(relative_burden > 0.0):
            fraction = float(np.max(relative_burden))
            scale = min(1.0, seed_fraction_value / fraction)
            recommended_array = recommended_array * scale
            recommended = tuple(float(value) for value in recommended_array.tolist())
    if np.any(~np.isfinite(recommended_array)) or np.any(recommended_array <= 0.0):
        raise ValueError("The requested support cannot receive finite positive seed amounts.")

    return CondensateSeedPolicyReport(
        diagnostic_only=True,
        default_off=True,
        production_behavior_change=False,
        policy_schema="exogibbs_condensate_initialization_policy_v1",
        support_indices=support,
        support_names=tuple(species[index] for index in support),
        recommended_amounts=tuple(recommended),
        recommended_ln_amounts=tuple(float(np.log(value)) for value in recommended),
        capacity_limited_amounts=tuple(capacity_limited),
        amount_gauge="element_inventory_target_fraction",
        fastchem4_first_step_equivalent_gauge=(
            "number_density_divided_by_initial_gas_phase_total_element_density"
        ),
        seed_fraction=seed_fraction_value,
        max_seed_amount=max_seed_amount_value,
        min_seed_amount=min_seed_amount_value,
        preserve_budget_fraction=bool(preserve_budget_fraction),
        field_provenance={
            "formula_matrix_cond": provenance.get("formula_matrix_cond", "exogibbs_native"),
            "element_inventory_target": provenance.get("element_inventory_target", "exogibbs_native"),
            "recommended_amounts": "derived_from_native_budget_capacity",
        },
        fastchem4_trace_values_used=False,
        fastchem4_public_values_used_as_constructor_inputs=False,
    )


def compute_seed_budget_fraction(
    *,
    formula_matrix_cond: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    support_indices: Sequence[int],
    seed_amounts: Sequence[float],
) -> float:
    """Return the maximum elemental fraction consumed by proposed seeds."""

    ac = _as_matrix(formula_matrix_cond, "formula_matrix_cond")
    target = _as_vector(element_inventory_target, "element_inventory_target")
    support = _support_tuple(support_indices, ac.shape[1])
    amounts = _as_vector(seed_amounts, "seed_amounts")
    if amounts.shape[0] != len(support):
        raise ValueError("seed_amounts length must match support_indices length.")
    if np.any(amounts < 0.0):
        raise ValueError("seed_amounts must be non-negative.")
    positive = np.abs(target) > 0.0
    support_array = np.asarray(support, dtype=np.int64)
    zero_budget_burden = ac[~positive][:, support_array] @ amounts
    if np.any(np.abs(zero_budget_burden) > 0.0):
        return float("inf")
    if not np.any(positive):
        raise ValueError("element_inventory_target must contain a nonzero budget.")
    relative_burden = (
        ac[positive][:, support_array] / np.abs(target[positive, None])
    ) @ amounts
    return float(np.max(np.abs(relative_burden)))
