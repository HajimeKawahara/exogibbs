"""Tests for native condensate initialization policies."""

from __future__ import annotations

import pytest

from exogibbs.condensates.initialization_policy import (
    recommend_budget_preserving_seed_amounts,
)


def test_capacity_fraction_policy_keeps_per_species_capacity_seed() -> None:
    preserved = recommend_budget_preserving_seed_amounts(
        formula_matrix_cond=[
            [2.0, 1.0],
            [1.0, 1.0],
        ],
        element_inventory_target=[1.0, 1.0],
        condensate_species_order=("H2O_s", "HO_s"),
        support_indices=(0, 1),
        seed_fraction=1.0e-3,
        max_seed_amount=1.0,
        preserve_budget_fraction=True,
    )
    capacity = recommend_budget_preserving_seed_amounts(
        formula_matrix_cond=[
            [2.0, 1.0],
            [1.0, 1.0],
        ],
        element_inventory_target=[1.0, 1.0],
        condensate_species_order=("H2O_s", "HO_s"),
        support_indices=(0, 1),
        seed_fraction=1.0e-3,
        max_seed_amount=1.0,
        preserve_budget_fraction=False,
    )

    assert preserved.recommended_amounts == pytest.approx((2.5e-4, 5.0e-4))
    assert capacity.recommended_amounts == pytest.approx((5.0e-4, 1.0e-3))
    assert preserved.preserve_budget_fraction is True
    assert capacity.preserve_budget_fraction is False
    assert capacity.amount_gauge == "element_inventory_target_fraction"
    assert (
        capacity.fastchem4_first_step_equivalent_gauge
        == "number_density_divided_by_initial_gas_phase_total_element_density"
    )
    assert capacity.fastchem4_trace_values_used is False
    assert capacity.fastchem4_public_values_used_as_constructor_inputs is False
