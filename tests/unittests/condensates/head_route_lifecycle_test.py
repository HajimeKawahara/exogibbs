"""Tests for condensate HEAD route lifecycle wiring."""

from __future__ import annotations

import math

import pytest

from exogibbs.condensates.head_route_standard_gate import (
    CONVERGED,
    CONVERGED_WITH_CAVEAT,
)
from exogibbs.condensates.center_primary_fallback import (
    build_center_primary_fallback_candidate,
    select_center_primary_fallback,
)
from exogibbs.condensates.electron_refresh import (
    check_source_convention_safe_electron_refresh,
)
from exogibbs.condensates.frontier_refresh import (
    select_frontier_refresh_from_metrics,
)
from exogibbs.condensates.head_route_lifecycle import (
    run_condensate_head_route_lifecycle,
)
from exogibbs.condensates.route_result import (
    build_head_route_lifecycle_result,
    infer_metric_status_from_selected_route,
)


def _base_kwargs(case_id: str = "solar_silicate_first_condensation__T1400_P1"):
    ln_nk = [math.log(0.8), math.log(0.2)]
    gas_source = [0.1, -0.05]
    element_potential = [ln_nk[0] + gas_source[0], ln_nk[1] + gas_source[1]]
    return {
        "explicit_opt_in": True,
        "case_id": case_id,
        "ln_nk": ln_nk,
        "support_indices": [1],
        "support_amounts": [1.0e-6],
        "formula_matrix": [[1.0, 0.0], [0.0, 1.0]],
        "formula_matrix_cond": [[2.0, 1.0], [0.0, 3.0]],
        "element_inventory_target": [0.800001, 0.200003],
        "element_potential": element_potential,
        "gas_stationarity_source": gas_source,
        "condensate_standard_source": [0.2],
        "field_provenance": {
            "ln_nk": "exogibbs_native",
            "support_indices": "exogibbs_native",
            "support_amounts": "exogibbs_native",
            "element_potential": "exogibbs_native_derived",
        },
    }


def test_center_primary_fallback_selects_first_guarded_candidate() -> None:
    rejected = build_center_primary_fallback_candidate(
        candidate_name="too_large_budget",
        converged_at_final_barrier=True,
        final_center_ratio=0.5,
        budget_ratio=2.0,
    )
    accepted = build_center_primary_fallback_candidate(
        candidate_name="center_primary_budget_guard",
        converged_at_final_barrier=True,
        final_center_ratio=0.9,
        budget_ratio=1.02,
    )

    report = select_center_primary_fallback(
        explicit_opt_in=True,
        primary_summary={"row_status": "not_centered"},
        candidates=[rejected, accepted],
    )

    assert report.accepted is True
    assert report.selected_candidate_name == "center_primary_budget_guard"
    assert report.classification.startswith("center_primary_budget_guard_accepts")
    assert report.production_behavior_change is False


def test_electron_refresh_checks_source_convention_gauge() -> None:
    report = check_source_convention_safe_electron_refresh(
        explicit_opt_in=True,
        ln_nk=[math.log(0.8), math.log(0.2)],
        element_potential=[math.log(0.8) + 0.1, math.log(0.2) - 0.05],
        formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
        gas_stationarity_source=[0.1, -0.05],
    )

    assert report.accepted is True
    assert report.classification == "electron_refresh_source_convention_compatible"
    assert report.sentinel_count == 0
    assert report.gas_lambda_gauge_residual_max_abs == pytest.approx(0.0, abs=1.0e-14)


def test_frontier_refresh_selects_first_accepted_candidate() -> None:
    report = select_frontier_refresh_from_metrics(
        explicit_opt_in=True,
        case_id="carbon_rich_CaS_MgS_AlN_window__T700_P1_corrected",
        max_accepted_budget=1.0e-8,
        max_accepted_amount_weighted_gas=1.0,
        candidate_metrics=[
            {
                "policy_name": "adaptive_floor_frontier_repair",
                "floor_value": 1.0e-300,
                "budget": 1.0e-6,
                "amount_weighted_gas": 0.1,
            },
            {
                "policy_name": "adaptive_floor_frontier_repair",
                "floor_value": 1.0e-200,
                "budget": 1.0e-10,
                "amount_weighted_gas": 0.2,
            },
        ],
    )

    assert report.accepted is True
    assert report.selected_policy == "adaptive_floor_frontier_repair"
    assert report.selected_floor_value == pytest.approx(1.0e-200)


def test_route_result_maps_primary_to_tight_convergence() -> None:
    assert infer_metric_status_from_selected_route(
        "m4310_full_promoted_policy_route",
        "accepted",
    ) == "tight_residual_components"

    result = build_head_route_lifecycle_result(
        explicit_opt_in=True,
        case_id="solar_silicate_first_condensation__T1400_P1",
        family="solar_silicate_first_condensation",
        selected_route="m4310_full_promoted_policy_route",
        integrated_status="accepted",
    )

    assert result.standard_path_status == CONVERGED
    assert result.converged is True


def test_lifecycle_connects_support_input_and_primary_route() -> None:
    report = run_condensate_head_route_lifecycle(
        **_base_kwargs(),
        primary_summary={
            "row_status": "centered",
            "converged_at_final_barrier": True,
        },
    )

    payload = report.as_dict()
    assert report.support_boundary.support_indices == (1,)
    assert report.continuation_input.support_indices == (1,)
    assert report.route_result.standard_path_status == CONVERGED
    assert report.route_selection_report["selected_route"] == "m4310_full_promoted_policy_route"
    assert payload["support_boundary"]["boundary_schema"] == "exogibbs_condensate_support_boundary_v1"
    assert payload["continuation_input"]["input_schema"] == "exogibbs_condensate_continuation_input_v1"


def test_lifecycle_centers_initial_dual_at_primary_policy_epsilon() -> None:
    initial_epsilon = math.log(1.0e-8)
    report = run_condensate_head_route_lifecycle(
        **_base_kwargs(),
        primary_continuation_policy={
            "initial_epsilon": initial_epsilon,
            "final_epsilon": initial_epsilon,
        },
        primary_summary={
            "row_status": "centered",
            "converged_at_final_barrier": True,
        },
    )

    state = report.continuation_input.state
    assert report.continuation_input.barrier_epsilon == pytest.approx(initial_epsilon)
    assert state.rho is not None
    assert state.ln_mk[0] + state.rho[0] == pytest.approx(initial_epsilon)


def test_lifecycle_runs_primary_continuation_when_summary_is_absent() -> None:
    report = run_condensate_head_route_lifecycle(
        **_base_kwargs(),
        primary_continuation_policy={
            "initial_epsilon": math.log(1.0e-8),
            "final_epsilon": math.log(1.0e-8),
            "barrier_schedule_policy": "fixed_tau",
            "max_outer_iterations": 1,
            "max_inner_iterations": 1,
            "center_tolerance_multiplier": 1.0e12,
            "center_metric_policy": "amount_weighted_kkt_max",
            "alpha_grid": [1.0e-3],
            "equality_penalty_weight": 10.0,
            "total_density_penalty_weight": 10.0,
            "max_abs_delta_q": 1.0,
            "max_abs_delta_r": 1.0,
            "max_abs_delta_rho": 1.0,
            "max_abs_delta_lambda": 10.0,
            "require_residual_nonworsening": False,
        },
    )

    assert report.primary_execution_report is not None
    assert report.primary_execution_report["continuation_report"]["outer_records"]
    assert report.primary_summary["stopped_reason"]
    assert report.route_selection_report["primary_summary"]["continuation_report"]


def test_lifecycle_connects_center_primary_fallback() -> None:
    fallback = build_center_primary_fallback_candidate(
        candidate_name="accepted_fallback",
        converged_at_final_barrier=True,
        final_center_ratio=0.8,
        budget_ratio=1.01,
    )

    report = run_condensate_head_route_lifecycle(
        **_base_kwargs("lowT_strong_condensation_budget_stress__T500_P1"),
        primary_summary={"row_status": "not_centered", "converged_at_final_barrier": False},
        center_fallback_candidates=[fallback],
    )

    assert report.center_fallback_report is not None
    assert report.center_fallback_report.accepted is True
    assert report.route_selection_report["selected_route"] == "m4326_center_primary_budget_guard_fallback"
    assert report.route_result.standard_path_status == CONVERGED_WITH_CAVEAT


def test_lifecycle_runs_center_fallback_continuation_policy() -> None:
    report = run_condensate_head_route_lifecycle(
        **_base_kwargs("lowT_strong_condensation_budget_stress__T500_P1"),
        primary_summary={"row_status": "not_centered", "converged_at_final_barrier": False},
        center_fallback_continuation_policies=[
            {
                "candidate_name": "executed_center_fallback",
                "initial_epsilon": math.log(1.0e-8),
                "final_epsilon": math.log(1.0e-8),
                "barrier_schedule_policy": "fixed_tau",
                "max_outer_iterations": 1,
                "max_inner_iterations": 1,
                "center_tolerance_multiplier": 1.0e12,
                "center_metric_policy": "amount_weighted_kkt_max",
                "alpha_grid": [1.0e-3],
                "equality_penalty_weight": 10.0,
                "total_density_penalty_weight": 10.0,
                "max_abs_delta_q": 1.0,
                "max_abs_delta_r": 1.0,
                "max_abs_delta_rho": 1.0,
                "max_abs_delta_lambda": 10.0,
                "require_residual_nonworsening": False,
            }
        ],
    )

    assert report.center_fallback_report is not None
    assert report.center_fallback_report.candidates[0].candidate_name == "executed_center_fallback"
    assert report.center_fallback_report.candidates[0].metadata["continuation_report"]


def test_lifecycle_connects_electron_refresh_as_refresh_route() -> None:
    report = run_condensate_head_route_lifecycle(
        **_base_kwargs("solar_metal_sulfide_or_Fe_Ni_S_region__T700_P1"),
        primary_summary={"row_status": "not_centered", "converged_at_final_barrier": False},
        electron_refresh_enabled=True,
    )

    assert report.electron_refresh_report is not None
    assert report.electron_refresh_report.accepted is True
    assert report.route_selection_report["selected_route"] == "fastchem4_style_electron_refresh_route"
    assert report.route_result.standard_path_status == CONVERGED_WITH_CAVEAT


def test_lifecycle_connects_frontier_refresh_policy() -> None:
    report = run_condensate_head_route_lifecycle(
        **_base_kwargs("carbon_rich_CaS_MgS_AlN_window__T700_P1_corrected"),
        primary_summary={"row_status": "not_centered", "converged_at_final_barrier": False},
        frontier_refresh_candidate_metrics=[
            {
                "policy_name": "adaptive_floor_frontier_repair",
                "floor_value": 1.0e-300,
                "budget": 1.0e-6,
                "amount_weighted_gas": 0.1,
            },
            {
                "policy_name": "adaptive_floor_frontier_repair",
                "floor_value": 1.0e-200,
                "budget": 1.0e-10,
                "amount_weighted_gas": 0.2,
            },
        ],
    )

    assert report.frontier_refresh_report is not None
    assert report.frontier_refresh_report["accepted"] is True
    assert report.route_selection_report["selected_route"] == "adaptive_floor_frontier_repair"
    assert report.route_result.standard_path_status == CONVERGED


def test_lifecycle_runs_frontier_refresh_continuation_policy() -> None:
    report = run_condensate_head_route_lifecycle(
        **_base_kwargs("carbon_rich_CaS_MgS_AlN_window__T700_P1_corrected"),
        primary_summary={"row_status": "not_centered", "converged_at_final_barrier": False},
        max_frontier_refresh_budget=1.0,
        max_frontier_refresh_amount_weighted_gas=1.0,
        frontier_refresh_continuation_policies=[
            {
                "policy_name": "adaptive_floor_frontier_repair",
                "candidate_kind": "executed_frontier_continuation",
                "floor_value": 1.0e-200,
                "initial_epsilon": math.log(1.0e-8),
                "final_epsilon": math.log(1.0e-8),
                "barrier_schedule_policy": "fixed_tau",
                "max_outer_iterations": 1,
                "max_inner_iterations": 1,
                "center_tolerance_multiplier": 1.0e12,
                "center_metric_policy": "amount_weighted_kkt_max",
                "alpha_grid": [1.0e-3],
                "equality_penalty_weight": 10.0,
                "total_density_penalty_weight": 10.0,
                "max_abs_delta_q": 1.0,
                "max_abs_delta_r": 1.0,
                "max_abs_delta_rho": 1.0,
                "max_abs_delta_lambda": 10.0,
                "require_residual_nonworsening": False,
            }
        ],
    )

    assert report.frontier_refresh_report is not None
    assert report.frontier_refresh_report["candidate_count"] == 1
    candidate = report.frontier_refresh_report["candidates"][0]
    assert candidate["candidate_kind"] == "executed_frontier_continuation"
    assert candidate["metadata"]["continuation_report"]
