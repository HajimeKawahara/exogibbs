"""Tests for inactive condensate driving diagnostics."""

from __future__ import annotations

import pytest

from exogibbs.condensates.inactive_driving import (
    evaluate_inactive_condensate_driving,
)


def test_inactive_driving_reports_all_and_temperature_valid_subsets() -> None:
    report = evaluate_inactive_condensate_driving(
        formula_matrix_cond=[[1.0, 1.0, 1.0]],
        condensate_species_order=("cold_s", "hot_s", "active_s"),
        condensate_amounts=[0.0, 0.0, 1.0e-10],
        hvector_cond=[-10.0, -2.0, -100.0],
        element_potential=[0.0],
        temperature=500.0,
        condensate_temperature_validity_upper=[300.0, 1000.0, 1000.0],
    )

    payload = report.as_dict()
    assert payload["all_condensates"]["positive_inactive_count"] == 2
    assert payload["all_condensates"]["max_positive_inactive_driving"] == pytest.approx(10.0)
    assert payload["all_condensates"]["top_positive_inactive"][0]["species"] == "cold_s"
    assert payload["temperature_valid_condensates"]["positive_inactive_count"] == 1
    assert payload["temperature_valid_condensates"]["max_positive_inactive_driving"] == pytest.approx(2.0)
    assert payload["temperature_valid_condensates"]["top_positive_inactive"][0]["species"] == "hot_s"
    assert payload["temperature_invalid_positive_inactive_count"] == 1
    assert payload["temperature_invalid_max_positive_inactive_driving"] == pytest.approx(10.0)
    assert payload["candidate_temperature_valid"] == {
        "cold_s": False,
        "hot_s": True,
        "active_s": True,
    }
    assert payload["fastchem4_trace_public_runtime_constructor_inputs_used"] is False


def test_inactive_driving_without_validity_metadata_treats_all_species_as_valid() -> None:
    report = evaluate_inactive_condensate_driving(
        formula_matrix_cond=[[1.0, 1.0]],
        condensate_species_order=("a_s", "b_s"),
        condensate_amounts=[0.0, 0.0],
        hvector_cond=[-3.0, 1.0],
        element_potential=[0.0],
        temperature=2000.0,
    )

    payload = report.as_dict()
    assert payload["all_condensates"]["positive_inactive_count"] == 1
    assert payload["temperature_valid_condensates"]["positive_inactive_count"] == 1
    assert payload["temperature_invalid_positive_inactive_count"] == 0
