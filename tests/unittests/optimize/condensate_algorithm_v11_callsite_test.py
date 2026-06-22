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
    assert report.reduced_step_report.step_control_policy == "component_clip"
    assert report.reduced_step_report.fraction_to_boundary_alpha == pytest.approx(1.0)
    assert report.reduced_step_report.initial_budget_l2 == pytest.approx(0.0, abs=1.0e-14)
    assert len(report.reduced_step_report.delta_r) == 1


def test_algorithm_v11_thermo_valid_callsite_forwards_scalar_step_policy() -> None:
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
        alpha_candidates=[1.0],
        step_control_policy="scalar_fraction_to_boundary",
        fraction_to_boundary_safety=0.5,
        field_provenance={
            "ln_mk": "synthetic_control",
            "rho": "synthetic_control",
            "eta": "synthetic_control",
        },
    )

    assert report.reduced_step_report.step_control_policy == "scalar_fraction_to_boundary"
    assert report.reduced_step_report.fraction_to_boundary_safety == pytest.approx(0.5)
    assert 0.0 < report.reduced_step_report.fraction_to_boundary_alpha <= 1.0
    blocker = report.reduced_step_report.fraction_to_boundary_blocker_report
    assert blocker is not None
    assert blocker["report_schema"] == "exogibbs_fraction_to_boundary_blocker_report_v1"
    assert blocker["limiting_variable_group"] in {"r", "rho", None}
    assert "top_blockers" in blocker


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
    assert inner["status"] == "ipopt_h_filter_selected"
    assert inner["selected_acceptance_source"] == "ipopt_h_filter"
    assert inner["line_search_failure_summary"] is None
    assert inner["require_residual_nonworsening"] is False
    assert inner["residual_worsening_tolerance"] == 0.1
    direction = inner["direction_records"][0]
    assert "persistent_filter_f_type_report" in direction
    assert "persistent_filter_f_type_protected_report" in direction


def test_algorithm_v11_continuation_splits_ipopt_filter_rejection_status() -> None:
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
        ipopt_h_type_theta_reduction_fraction=0.5,
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
    summary = inner["line_search_failure_summary"]
    assert inner["status"] == "no_acceptable_ipopt_filter_trial"
    assert continuation["stopped_reason"] == "no_acceptable_ipopt_filter_trial"
    assert inner["selected_acceptance_source"] is None
    assert summary["status"] == "no_acceptable_ipopt_filter_trial"
    assert summary["trial_acceptance_policy"] == "ipopt_persistent_h_type"
    assert summary["finite_trial_count"] > 0


def test_algorithm_v11_continuation_callsite_forwards_scalar_step_policy() -> None:
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
        step_control_policy="scalar_fraction_to_boundary",
        fraction_to_boundary_safety=0.5,
        field_provenance={
            "ln_mk": "synthetic_control",
            "rho": "synthetic_control",
            "eta": "synthetic_control",
        },
    )

    continuation = report.as_dict()["continuation_report"]
    outer = continuation["outer_records"][0]
    inner = outer["inner_records"][0]
    assert inner["step_control_policy"] == "scalar_fraction_to_boundary"
    assert inner["step_fraction_to_boundary_safety"] == pytest.approx(0.5)
    assert 0.0 < inner["step_fraction_to_boundary_alpha"] <= 1.0
    blocker = inner["step_fraction_to_boundary_blocker_report"]
    assert blocker["report_schema"] == "exogibbs_fraction_to_boundary_blocker_report_v1"
    assert blocker["limiting_variable_group"] in {"r", "rho", None}
    if blocker["top_blockers"]:
        assert blocker["top_blockers"][0]["species_name"] == "good"
    assert outer["center_metric_ratio_after_outer"] == pytest.approx(
        outer["center_metric_after_outer"] / outer["center_threshold"]
    )
    assert outer["center_metric_excess_after_outer"] == pytest.approx(
        max(0.0, outer["center_metric_after_outer"] - outer["center_threshold"])
    )


def test_algorithm_v11_core_mode_forces_pdipm_mainline_policy() -> None:
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
        continuation_mode="pdipm_core",
        step_control_policy="component_clip",
        direction_policy="algorithm_v11_with_condensate_budget_correction",
        trial_acceptance_policy="p_armijo_or_best_residual",
        field_provenance={
            "ln_mk": "synthetic_control",
            "rho": "synthetic_control",
            "eta": "synthetic_control",
        },
    )

    continuation = report.as_dict()["continuation_report"]
    inner = continuation["outer_records"][0]["inner_records"][0]
    direction = inner["direction_records"][0]
    assert continuation["continuation_mode"] == "pdipm_core"
    assert continuation["filter_accept_count"] >= 0
    assert continuation["restoration_count"] >= 0
    assert continuation["barrier_update_count"] >= 0
    assert continuation["tiny_step_count"] == 0
    assert inner["continuation_mode"] == "pdipm_core"
    assert inner["step_control_policy"] == "scalar_fraction_to_boundary"
    assert inner["direction_policy"] == "algorithm_v11_reduced"
    assert inner["trial_acceptance_policy"] == "ipopt_persistent_h_type"
    assert inner["enable_native_soft_restoration_fallback"] is True
    assert inner["enable_dedicated_restoration_filter_phase"] is True
    assert inner["ipopt_tiny_step_alpha_threshold"] == pytest.approx(1.0e-8)
    assert inner["ipopt_tiny_step_consecutive_limit"] == 1
    assert inner["ipopt_tiny_step_switch_to_restoration"] is True
    assert inner["tiny_step_detected"] is False
    assert "fraction_to_boundary_alpha_primal" in direction
    assert "fraction_to_boundary_alpha_dual" in direction
    assert "fraction_to_boundary_alpha_combined" in direction


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
