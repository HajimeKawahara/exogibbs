"""Tests for native condensate initialization policies."""

from __future__ import annotations

import pytest

from exogibbs.condensates.initialization_policy import (
    compute_seed_budget_fraction,
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


def test_shared_budget_limit_takes_precedence_over_minimum_seed() -> None:
    report = recommend_budget_preserving_seed_amounts(
        formula_matrix_cond=[[1.0, 1.0]],
        element_inventory_target=[1.0],
        condensate_species_order=("A(s)", "B(s)"),
        support_indices=(0, 1),
        seed_fraction=0.1,
        max_seed_amount=1.0,
        min_seed_amount=0.1,
        preserve_budget_fraction=True,
    )

    consumed_fraction = compute_seed_budget_fraction(
        formula_matrix_cond=[[1.0, 1.0]],
        element_inventory_target=[1.0],
        support_indices=(0, 1),
        seed_amounts=report.recommended_amounts,
    )

    assert report.recommended_amounts == pytest.approx((0.05, 0.05))
    assert consumed_fraction == pytest.approx(0.1)


def test_shared_budget_scaling_avoids_absolute_burden_overflow() -> None:
    report = recommend_budget_preserving_seed_amounts(
        formula_matrix_cond=[[1.0e308, 1.0e308]],
        element_inventory_target=[1.0e308],
        condensate_species_order=("A(s)", "B(s)"),
        support_indices=(0, 1),
        seed_fraction=1.0,
        max_seed_amount=1.0,
        min_seed_amount=0.1,
        preserve_budget_fraction=True,
    )

    assert report.recommended_amounts == pytest.approx((0.5, 0.5))
    assert compute_seed_budget_fraction(
        formula_matrix_cond=[[1.0e308, 1.0e308]],
        element_inventory_target=[1.0e308],
        support_indices=(0, 1),
        seed_amounts=report.recommended_amounts,
    ) == pytest.approx(1.0)


def test_zero_capacity_support_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot receive a positive seed"):
        recommend_budget_preserving_seed_amounts(
            formula_matrix_cond=[[1.0], [1.0]],
            element_inventory_target=[1.0, 0.0],
            condensate_species_order=("AB(s)",),
            support_indices=(0,),
            seed_fraction=0.1,
            max_seed_amount=1.0,
            min_seed_amount=1.0e-3,
            preserve_budget_fraction=True,
        )

    assert compute_seed_budget_fraction(
        formula_matrix_cond=[[1.0], [1.0]],
        element_inventory_target=[1.0, 0.0],
        support_indices=(0,),
        seed_amounts=(1.0e-3,),
    ) == float("inf")


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("seed_fraction", float("nan")),
        ("seed_fraction", float("inf")),
        ("max_seed_amount", float("nan")),
        ("max_seed_amount", float("inf")),
        ("min_seed_amount", float("nan")),
        ("min_seed_amount", float("inf")),
    ),
)
def test_rejects_nonfinite_seed_parameters(name: str, value: float) -> None:
    arguments = {
        "formula_matrix_cond": [[1.0]],
        "element_inventory_target": [1.0],
        "condensate_species_order": ("A(s)",),
        "support_indices": (0,),
    }
    arguments[name] = value

    with pytest.raises(ValueError, match=name):
        recommend_budget_preserving_seed_amounts(**arguments)
