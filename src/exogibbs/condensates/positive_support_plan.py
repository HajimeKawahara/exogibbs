"""Default-off positive support initialization plan diagnostics.

This module combines native positive-support selection with conservative
budget-preserving seed amounts. It does not import FastChem4, call
pyfastchem, call production solvers, or connect to presets/defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from exogibbs.condensates.initialization_policy import (
    recommend_budget_preserving_seed_amounts,
)
from exogibbs.condensates.support_selection_policy import (
    select_positive_support_candidates,
)


@dataclass(frozen=True)
class PositiveSupportInitializationPlan:
    """Default-off diagnostic positive-support initialization plan."""

    diagnostic_only: bool
    default_off: bool
    production_behavior_change: bool
    plan_schema: str
    positive_support_indices: tuple[int, ...]
    positive_support_names: tuple[str, ...]
    zero_bound_candidate_indices: tuple[int, ...]
    zero_bound_candidate_names: tuple[str, ...]
    empty_positive_support_allowed: bool
    recommended_amounts: tuple[float, ...]
    recommended_ln_amounts: tuple[float, ...]
    seed_fraction: float
    max_seed_amount: float
    support_policy: str
    field_provenance: Mapping[str, str]
    fastchem4_trace_values_used: bool
    fastchem4_public_values_used_as_constructor_inputs: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_positive_support_initialization_plan(
    *,
    formula_matrix_cond: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    condensate_species_order: Sequence[str],
    hvector_cond: Sequence[float],
    max_positive_support_count: int = 1,
    seed_fraction: float = 1.0e-3,
    max_seed_amount: float = 1.0e-3,
    min_seed_amount: float = 1.0e-300,
    allow_empty_positive_support: bool = True,
    field_provenance: Mapping[str, str] | None = None,
) -> PositiveSupportInitializationPlan:
    """Build a native positive-support initialization plan."""

    selection = select_positive_support_candidates(
        formula_matrix_cond=formula_matrix_cond,
        element_inventory_target=element_inventory_target,
        condensate_species_order=condensate_species_order,
        hvector_cond=hvector_cond,
        max_positive_support_count=max_positive_support_count,
        require_thermochemical_favorable=True,
        field_provenance=field_provenance,
    )
    if not selection.positive_support_indices:
        if not allow_empty_positive_support:
            raise ValueError("empty positive support is not allowed by this plan.")
        return PositiveSupportInitializationPlan(
            diagnostic_only=True,
            default_off=True,
            production_behavior_change=False,
            plan_schema="exogibbs_condensate_positive_support_initialization_plan_v1",
            positive_support_indices=(),
            positive_support_names=(),
            zero_bound_candidate_indices=selection.zero_bound_candidate_indices,
            zero_bound_candidate_names=selection.zero_bound_candidate_names,
            empty_positive_support_allowed=True,
            recommended_amounts=(),
            recommended_ln_amounts=(),
            seed_fraction=float(seed_fraction),
            max_seed_amount=float(max_seed_amount),
            support_policy=selection.policy_name,
            field_provenance={
                **dict(selection.field_provenance),
                "recommended_amounts": "empty_positive_support_has_no_seed_amounts",
            },
            fastchem4_trace_values_used=False,
            fastchem4_public_values_used_as_constructor_inputs=False,
        )

    seed = recommend_budget_preserving_seed_amounts(
        formula_matrix_cond=formula_matrix_cond,
        element_inventory_target=element_inventory_target,
        condensate_species_order=condensate_species_order,
        support_indices=selection.positive_support_indices,
        seed_fraction=seed_fraction,
        max_seed_amount=max_seed_amount,
        min_seed_amount=min_seed_amount,
        field_provenance=field_provenance,
    )
    return PositiveSupportInitializationPlan(
        diagnostic_only=True,
        default_off=True,
        production_behavior_change=False,
        plan_schema="exogibbs_condensate_positive_support_initialization_plan_v1",
        positive_support_indices=selection.positive_support_indices,
        positive_support_names=selection.positive_support_names,
        zero_bound_candidate_indices=selection.zero_bound_candidate_indices,
        zero_bound_candidate_names=selection.zero_bound_candidate_names,
        empty_positive_support_allowed=False,
        recommended_amounts=seed.recommended_amounts,
        recommended_ln_amounts=seed.recommended_ln_amounts,
        seed_fraction=float(seed_fraction),
        max_seed_amount=float(max_seed_amount),
        support_policy=selection.policy_name,
        field_provenance={
            **dict(selection.field_provenance),
            "recommended_amounts": "derived_from_native_budget_capacity",
        },
        fastchem4_trace_values_used=False,
        fastchem4_public_values_used_as_constructor_inputs=False,
    )
