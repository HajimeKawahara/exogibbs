"""Tests for native condensate initialization policies."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
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


@pytest.mark.parametrize("inventory", (1.0e-310, 9.108388204e-314))
def test_seed_budget_preserves_positive_subnormal_inventory(inventory) -> None:
    report = recommend_budget_preserving_seed_amounts(
        formula_matrix_cond=[[1.0, 1.0]],
        element_inventory_target=[inventory],
        condensate_species_order=("A(s)", "B(s)"),
        support_indices=(0, 1),
        seed_fraction=1.0e-3,
    )
    np.testing.assert_allclose(
        report.recommended_amounts,
        [0.5e-3 * inventory] * 2,
        rtol=1.0e-6,
        atol=0.0,
    )
    assert all(amount > 0.0 for amount in report.recommended_amounts)
    assert compute_seed_budget_fraction(
        formula_matrix_cond=[[1.0, 1.0]],
        element_inventory_target=[inventory],
        support_indices=(0, 1),
        seed_amounts=report.recommended_amounts,
    ) == pytest.approx(1.0e-3, rel=1.0e-6)


@pytest.mark.parametrize("seed_fraction", (1.0e-3, 0.8))
def test_unrepresentable_positive_seed_is_rejected_without_inventory_floor(seed_fraction):
    with pytest.raises(ValueError, match="finite positive seed amounts"):
        recommend_budget_preserving_seed_amounts(
            formula_matrix_cond=[[1.0]],
            element_inventory_target=[np.nextafter(0.0, 1.0)],
            condensate_species_order=("A(s)",),
            support_indices=(0,),
            seed_fraction=seed_fraction,
        )


@pytest.mark.parametrize(
    ("inventory", "seed_fraction", "minimum_seed"),
    (
        (np.nextafter(1.0, 0.0), np.nextafter(1.0e-3, np.inf), 1.0e-300),
        (np.nextafter(1.0e-310, np.inf), 0.8, 1.0e-300),
        (np.nextafter(1.0e308, 0.0), 0.8, 1.0e-300),
        (1.0e-310, 0.8, 1.0e300),
    ),
)
def test_seed_rounding_never_exceeds_exact_shared_budget(
    inventory, seed_fraction, minimum_seed
):
    report = recommend_budget_preserving_seed_amounts(
        formula_matrix_cond=[[1.0, 1.0, 1.0]],
        element_inventory_target=[inventory],
        condensate_species_order=("A(s)", "B(s)", "C(s)"),
        support_indices=(0, 1, 2),
        seed_fraction=seed_fraction,
        max_seed_amount=1.0e308,
        min_seed_amount=minimum_seed,
    )
    actual = sum(Fraction.from_float(value) for value in report.recommended_amounts)
    quota = Fraction.from_float(float(inventory)) * Fraction.from_float(seed_fraction)
    assert actual <= quota
    assert all(value > 0.0 for value in report.recommended_amounts)


@pytest.mark.parametrize(
    ("formula", "target", "amounts", "expected"),
    (
        ([[1.0, 1.0], [1.0, -1.0]], [1.0e-310, 0.0],
         [1.0e-310, 1.0e-310], 2.0),
        ([[1.0, 1.0], [1.0, -1.0]], [1.0e-310, 0.0],
         [1.0e-310, np.nextafter(1.0e-310, 0.0)], float("inf")),
        ([[1.0, 1.0], [1.0e308, -1.0e308]], [10.0, 0.0],
         [2.0, np.nextafter(2.0, np.inf)], float("inf")),
        ([[1.0, 1.0], [1.0e308, -1.0e308]], [10.0, 0.0],
         [2.0, 2.0], 0.4),
        ([[1.0, 1.0], [1.0e-300, -1.0e-300]], [1.0, 0.0],
         [1.0e-300, np.nextafter(1.0e-300, np.inf)], float("inf")),
        ([[1.0, 0.0], [0.0, -1.0]], [1.0e-310, -1.0e-310],
         [1.0e-313, 2.0e-313], 2.0e-3),
        ([[1.0, 0.0], [0.0, 1.0]], [1.0, 0.0], [0.5, 0.0], 0.5),
        ([[1.0, 0.0], [0.0, 1.0e-300]], [1.0, 0.0],
         [0.5, 1.0e-300], float("inf")),
    ),
)
def test_seed_budget_fraction_respects_signed_and_zero_rows(
    formula, target, amounts, expected
) -> None:
    assert compute_seed_budget_fraction(
        formula_matrix_cond=formula,
        element_inventory_target=target,
        support_indices=(0, 1),
        seed_amounts=amounts,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("signed_row", "amounts"),
    (
        ([3.0, -1.0, -1.0], [1.2, 3.5999999999999996, 2.220446049250313e-16]),
        ([3.0, -1.0, 1.0], [0.1, 0.30000000000000004, 2.7755575615628914e-17]),
    ),
)
def test_seed_budget_fraction_certifies_cancellation_before_product_rounding(
    signed_row, amounts
):
    exact_burden = sum(
        Fraction.from_float(coefficient) * Fraction.from_float(amount)
        for coefficient, amount in zip(signed_row, amounts)
    )
    assert exact_burden == 0
    assert compute_seed_budget_fraction(
        formula_matrix_cond=[[1.0, 1.0, 1.0], signed_row],
        element_inventory_target=[10.0, 0.0],
        support_indices=(0, 1, 2),
        seed_amounts=amounts,
    ) == pytest.approx(sum(amounts) / 10.0)


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
