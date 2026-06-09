"""Tests for native condensate support selection policies."""

from __future__ import annotations

import pytest

from exogibbs.condensates.support_selection_policy import (
    select_activity_driven_support_candidates,
)


def test_activity_driven_support_selects_positive_native_driving() -> None:
    report = select_activity_driven_support_candidates(
        formula_matrix_cond=[
            [2.0, 20.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        element_inventory_target=[1.0, 1.0],
        condensate_species_order=("H2O(s)", "H(s)", "O(s)"),
        hvector_cond=[-2.0, 0.0, 4.0],
        element_potential=[0.1, 0.2],
        max_positive_support_count=2,
        activity_threshold=0.0,
        field_provenance={
            "formula_matrix_cond": "unit_test_native",
            "element_inventory_target": "unit_test_native",
            "hvector_cond": "unit_test_native",
            "element_potential": "unit_test_native",
        },
    )

    assert report.positive_support_indices == (0, 1)
    assert report.positive_support_names == ("H2O(s)", "H(s)")
    assert report.inactive_positive_indices == (0, 1)
    assert report.inactive_positive_names == ("H2O(s)", "H(s)")
    assert report.candidate_driving["H2O(s)"] == pytest.approx(2.4)
    assert report.candidate_driving["O(s)"] == pytest.approx(-3.8)
    assert report.fastchem4_trace_values_used is False
    assert report.fastchem4_public_values_used_as_constructor_inputs is False


def test_activity_driven_support_respects_existing_support_for_inactive_list() -> None:
    report = select_activity_driven_support_candidates(
        formula_matrix_cond=[
            [2.0, 20.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        element_inventory_target=[1.0, 1.0],
        condensate_species_order=("H2O(s)", "H(s)", "O(s)"),
        hvector_cond=[-2.0, 0.0, 4.0],
        element_potential=[0.1, 0.2],
        max_positive_support_count=2,
        existing_support_indices=(0,),
    )

    assert report.positive_support_indices == (0, 1)
    assert report.inactive_positive_indices == (1,)
    assert report.inactive_positive_names == ("H(s)",)


def test_activity_driven_support_can_return_all_positive_candidates() -> None:
    report = select_activity_driven_support_candidates(
        formula_matrix_cond=[
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        element_inventory_target=[1.0, 1.0],
        condensate_species_order=("A(s)", "B(s)", "C(s)"),
        hvector_cond=[-1.0, -1.0, -1.0],
        element_potential=[1.0, 1.0],
        max_positive_support_count=None,
    )

    assert report.policy_name == "native_activity_threshold_capacity_priority_all_positive"
    assert report.max_positive_support_count is None
    assert set(report.positive_support_names) == {"A(s)", "B(s)", "C(s)"}


def test_activity_driven_support_rejects_temperature_invalid_candidates() -> None:
    report = select_activity_driven_support_candidates(
        formula_matrix_cond=[
            [2.0, 20.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        element_inventory_target=[1.0, 1.0],
        condensate_species_order=("H2O(s)", "H(s)", "O(s)"),
        hvector_cond=[-2.0, 0.0, 4.0],
        element_potential=[0.1, 0.2],
        max_positive_support_count=2,
        temperature=500.0,
        condensate_temperature_validity_upper=[400.0, 1000.0, 1000.0],
    )

    assert report.positive_support_indices == (1,)
    assert report.positive_support_names == ("H(s)",)
    assert report.candidate_temperature_valid["H2O(s)"] is False
    assert report.candidate_temperature_valid["H(s)"] is True
