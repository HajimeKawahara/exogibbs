"""Support seeding for the production fixed-support v2 lifecycle."""

from __future__ import annotations

from typing import Any, Sequence

from exogibbs.condensates.initialization_policy import (
    recommend_budget_preserving_seed_amounts,
)


AMOUNT_FLOOR = 1.0e-300


def seed_fixed_support_payload(
    *,
    setup: Any,
    element_inventory_target: Sequence[float],
    support_indices: Sequence[int],
    seed_fraction: float = 1.0e-3,
    max_seed_amount: float = 1.0e-3,
    min_seed_amount: float = AMOUNT_FLOOR,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Return deduplicated support indices and budget-preserving seeds."""

    support = tuple(dict.fromkeys(int(index) for index in support_indices))
    if not support:
        return (), ()
    seed = recommend_budget_preserving_seed_amounts(
        formula_matrix_cond=setup.formula_matrix_cond,
        element_inventory_target=element_inventory_target,
        condensate_species_order=setup.condensate_species,
        support_indices=support,
        seed_fraction=float(seed_fraction),
        max_seed_amount=float(max_seed_amount),
        min_seed_amount=float(min_seed_amount),
        field_provenance={
            "formula_matrix_cond": "exogibbs_condensate_chemical_setup",
            "element_inventory_target": "exogibbs_fixed_support_payload_budget",
        },
    )
    return support, tuple(float(value) for value in seed.recommended_amounts)


__all__ = ("AMOUNT_FLOOR", "seed_fixed_support_payload")
