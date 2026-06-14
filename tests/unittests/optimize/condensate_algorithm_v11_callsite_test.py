"""Tests for the algorithm-v1.1 thermo-valid diagnostic callsite."""

from __future__ import annotations

import math

import pytest

from exogibbs.optimize.condensate_algorithm_v11_callsite import (
    algorithm_v11_experimental_high_start_callsite_policy,
    run_algorithm_v11_thermo_valid_continuation_callsite,
    run_algorithm_v11_thermo_valid_reduced_callsite,
)
from exogibbs.optimize.pdipm_rgie_cond import build_pdipm_rgie_condensate_state


def _state():
    return build_pdipm_rgie_condensate_state(
        ln_nk=[math.log(0.8), math.log(0.2)],
        ln_mk=[math.log(1.0e-8), math.log(1.0e-6)],
        element_potential=[0.0, 0.0],
        rho=[math.log(1.0e-5), math.log(1.0e-4)],
        field_provenance={
            "ln_nk": "synthetic_control",
            "ln_mk": "synthetic_control",
            "element_potential": "synthetic_control",
            "rho": "synthetic_control",
            "eta": "synthetic_control",
        },
    )


def test_algorithm_v11_thermo_valid_callsite_filters_and_runs() -> None:
    report = run_algorithm_v11_thermo_valid_reduced_callsite(
        explicit_opt_in=True,
        state=_state(),
        support_indices=[0, 1],
        formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
        formula_matrix_cond_active=[[1.0, 0.0], [0.0, 1.0]],
        element_inventory_target=[0.80000001, 0.200001],
        gas_stationarity_source=[0.1, -0.05],
        condensate_standard_source=[1.0e20, 0.2],
        epsilon=math.log(1.0e-11),
        species_names=["bad", "good"],
        alpha_candidates=[1.0, 0.1, 0.01],
        max_abs_delta_q=1.0e300,
        max_abs_delta_r=1.0e300,
        max_abs_delta_rho=1.0e300,
        max_abs_delta_lambda=1.0e300,
        field_provenance={
            "ln_mk": "synthetic_control",
            "rho": "synthetic_control",
            "eta": "synthetic_control",
        },
    )

    assert report.default_off is True
    assert report.diagnostic_only is True
    assert report.production_behavior_change is False
    assert report.production_return_signature_change is False
    assert report.preset_default_wiring_change is False
    assert report.fastchem4_trace_public_runtime_constructor_inputs_used is False
    assert report.original_support_count == 2
    assert report.filtered_support_count == 1
    assert report.removed_support_count == 1
    assert report.filter_report.removed_species_names == ("bad",)
    assert report.reduced_step_report.finite_trial_step is True
    assert report.reduced_step_report.initial_budget_l2 == pytest.approx(0.0, abs=1.0e-14)
    assert len(report.reduced_step_report.delta_r) == 1


def test_algorithm_v11_thermo_valid_callsite_requires_opt_in() -> None:
    with pytest.raises(ValueError, match="explicit_opt_in"):
        run_algorithm_v11_thermo_valid_reduced_callsite(
            explicit_opt_in=False,
            state=_state(),
            support_indices=[0],
            formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
            formula_matrix_cond_active=[[1.0], [0.0]],
            element_inventory_target=[0.8, 0.2],
            gas_stationarity_source=[0.1, -0.05],
            condensate_standard_source=[0.2],
            epsilon=math.log(1.0e-11),
        )


def test_algorithm_v11_thermo_valid_continuation_callsite_runs_ipopt_like_policy() -> None:
    report = run_algorithm_v11_thermo_valid_continuation_callsite(
        explicit_opt_in=True,
        state=_state(),
        support_indices=[0, 1],
        formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
        formula_matrix_cond_active=[[1.0, 0.0], [0.0, 1.0]],
        element_inventory_target=[0.80000001, 0.200001],
        gas_stationarity_source=[0.1, -0.05],
        condensate_standard_source=[1.0e20, 0.2],
        initial_epsilon=math.log(1.0e-8),
        final_epsilon=math.log(1.0e-10),
        species_names=["bad", "good"],
        sentinel_abs_threshold=1.0e10,
        barrier_schedule_policy="ipopt_like_monotone",
        ipopt_enable_superlinear_decrease=False,
        max_outer_iterations=4,
        max_inner_iterations=4,
        center_tolerance_multiplier=1.0e8,
        center_metric_policy="amount_weighted_kkt_max",
        alpha_grid=[1.0, 0.5, 0.25, 0.1],
        equality_penalty_weight=1000.0,
        total_density_penalty_weight=1000.0,
        max_abs_delta_q=1.5,
        max_abs_delta_r=1.25,
        max_abs_delta_rho=1.1,
        max_abs_delta_lambda=50.0,
        field_provenance={
            "ln_mk": "synthetic_control",
            "rho": "synthetic_control",
            "eta": "synthetic_control",
        },
    )

    payload = report.as_dict()
    continuation = payload["continuation_report"]
    assert payload["report_schema"] == (
        "exogibbs_algorithm_v11_thermo_valid_continuation_callsite_report_v1"
    )
    assert payload["diagnostic_only"] is True
    assert payload["production_behavior_change"] is False
    assert payload["production_return_signature_change"] is False
    assert payload["preset_default_wiring_change"] is False
    assert payload["fastchem4_trace_public_runtime_constructor_inputs_used"] is False
    assert payload["original_support_count"] == 2
    assert payload["filtered_support_count"] == 1
    assert continuation["barrier_schedule_policy"] == "ipopt_like_monotone"
    assert continuation["ipopt_enable_superlinear_decrease"] is False
    assert continuation["outer_records"]


def test_algorithm_v11_high_start_policy_runs_through_continuation_callsite() -> None:
    policy = algorithm_v11_experimental_high_start_callsite_policy()
    report = run_algorithm_v11_thermo_valid_continuation_callsite(
        explicit_opt_in=True,
        state=_state(),
        support_indices=[0, 1],
        formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
        formula_matrix_cond_active=[[1.0, 0.0], [0.0, 1.0]],
        element_inventory_target=[0.80000001, 0.200001],
        gas_stationarity_source=[0.1, -0.05],
        condensate_standard_source=[1.0e20, 0.2],
        species_names=["bad", "good"],
        sentinel_abs_threshold=1.0e10,
        field_provenance={
            "ln_mk": "synthetic_control",
            "rho": "synthetic_control",
            "eta": "synthetic_control",
        },
        **policy,
    )

    payload = report.as_dict()
    continuation = payload["continuation_report"]
    assert policy["barrier_schedule_policy"] == "ipopt_like_monotone"
    assert policy["ipopt_enable_superlinear_decrease"] is False
    assert policy["max_outer_iterations"] == 14
    assert policy["max_inner_iterations"] == 120
    assert continuation["barrier_schedule_policy"] == "ipopt_like_monotone"
    assert continuation["ipopt_enable_superlinear_decrease"] is False
    assert payload["filtered_support_count"] == 1


def test_algorithm_v11_continuation_callsite_forwards_generic_filter_policy() -> None:
    report = run_algorithm_v11_thermo_valid_continuation_callsite(
        explicit_opt_in=True,
        state=_state(),
        support_indices=[0, 1],
        formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
        formula_matrix_cond_active=[[1.0, 0.0], [0.0, 1.0]],
        element_inventory_target=[0.80000001, 0.200001],
        gas_stationarity_source=[0.1, -0.05],
        condensate_standard_source=[1.0e20, 0.2],
        initial_epsilon=math.log(1.0e-8),
        final_epsilon=math.log(1.0e-10),
        species_names=["bad", "good"],
        sentinel_abs_threshold=1.0e10,
        max_outer_iterations=1,
        max_inner_iterations=1,
        center_tolerance_multiplier=1.0e-16,
        alpha_grid=[1.0e-3],
        trial_acceptance_policy="ipopt_persistent_h_type",
        filter_component_scale_policy="current",
        ipopt_h_type_component_weights={
            "budget": 1.0,
            "total_density": 1.0,
            "amount_weighted_gas": 1.0,
        },
        ipopt_h_type_theta_reduction_fraction=1.0e-5,
        ipopt_h_type_protected_components=["budget"],
        ipopt_h_type_protected_component_max_normalized_increase=1.0,
        require_residual_nonworsening=False,
        residual_worsening_tolerance=0.1,
        field_provenance={
            "ln_mk": "synthetic_control",
            "rho": "synthetic_control",
            "eta": "synthetic_control",
        },
    )

    continuation = report.as_dict()["continuation_report"]
    inner = continuation["outer_records"][0]["inner_records"][0]
    assert inner["trial_acceptance_policy"] == "ipopt_persistent_h_type"
    assert inner["filter_component_scale_policy"] == "current"
    assert inner["ipopt_h_type_theta_reduction_fraction"] == 1.0e-5
    assert inner["ipopt_h_type_protected_components"] == ("budget",)
    assert inner["require_residual_nonworsening"] is False
    assert inner["residual_worsening_tolerance"] == 0.1


def test_algorithm_v11_thermo_valid_continuation_callsite_requires_opt_in() -> None:
    with pytest.raises(ValueError, match="explicit_opt_in"):
        run_algorithm_v11_thermo_valid_continuation_callsite(
            explicit_opt_in=False,
            state=_state(),
            support_indices=[0],
            formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
            formula_matrix_cond_active=[[1.0], [0.0]],
            element_inventory_target=[0.8, 0.2],
            gas_stationarity_source=[0.1, -0.05],
            condensate_standard_source=[0.2],
            initial_epsilon=math.log(1.0e-8),
            final_epsilon=math.log(1.0e-10),
        )


def test_algorithm_v11_thermo_valid_callsite_rejects_forbidden_provenance() -> None:
    with pytest.raises(ValueError):
        run_algorithm_v11_thermo_valid_reduced_callsite(
            explicit_opt_in=True,
            state=_state(),
            support_indices=[0],
            formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
            formula_matrix_cond_active=[[1.0], [0.0]],
            element_inventory_target=[0.8, 0.2],
            gas_stationarity_source=[0.1, -0.05],
            condensate_standard_source=[0.2],
            epsilon=math.log(1.0e-11),
            field_provenance={"ln_mk": "fastchem4_public"},
        )
