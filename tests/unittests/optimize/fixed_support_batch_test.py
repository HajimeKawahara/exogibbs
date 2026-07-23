from exogibbs.optimize.fixed_support_batch import (
    FIXED_SUPPORT_BATCH_RESIDUAL_COMPONENT_LABELS,
    FIXED_SUPPORT_BATCH_STOP_REASON_LABELS,
    build_ipopt_current_iterate_filter_mask,
    build_fixed_support_batch_metadata,
)


def test_ipopt_current_iterate_filter_accepts_complementarity_progress():
    mask = build_ipopt_current_iterate_filter_mask(
        finite=[True, True, True, False],
        protected_theta=[1.0005e-8, 1.2e-8, 0.9e-8, 1.0e-8],
        complementarity_merit=[4.0e-5, 4.0e-5, 9.9e-5, 4.0e-5],
        initial_protected_theta=1.0e-8,
        initial_complementarity_merit=1.0e-4,
        required_complementarity_factor=0.5,
        relaxed_fallback_enabled=False,
        relaxed_fallback_factor=0.999,
    )

    assert mask.tolist() == [True, False, True, False]


def test_build_fixed_support_batch_metadata_preserves_export_schema():
    metadata = build_fixed_support_batch_metadata(
        accepted_count=3,
        normal_accepted_count=1,
        fallback_accepted_count=1,
        restoration_accepted_count=1,
        soc_accepted_count=0,
        adaptive_regularization_selected_count=2,
        rejected_trial_count=5,
        tiny_step_consecutive_count=7,
        final_step_size=0.25,
        stop_reason_code=1,
        dominant_residual_component_index=2,
        final_log_activity_correction=[0.0],
        final_element_potential=[1.0],
        initial_residual=10.0,
        lambda_selection_index=0,
        line_search_alpha_boundary=0.5,
        line_search_alpha_r=0.75,
        line_search_alpha_rho=0.5,
        line_search_selected_trial_index=2,
        line_search_selected_trial_alpha=0.25,
        line_search_selected_trial_residual=8.0,
        line_search_accepted_candidate_count=1,
        line_search_fallback_candidate_count=2,
        line_search_best_trial_index=3,
        line_search_best_trial_alpha=0.125,
        line_search_best_trial_residual=7.5,
        line_search_best_trial_gas_residual=1.25,
        line_search_best_trial_condensate_stationarity_residual=2.25,
        line_search_best_trial_budget_residual=3.25,
        line_search_best_trial_budget_relative_residual_max=0.2,
        line_search_best_trial_complementarity_residual=4.25,
        line_search_best_trial_total_density_residual=5.25,
        line_search_finite_candidate_count=13,
        line_search_combined_improved_candidate_count=8,
        line_search_budget_relative_not_worse_candidate_count=4,
        line_search_filter_candidate_count=5,
        line_search_budget_not_broken_candidate_count=6,
        line_search_budget_relative_not_broken_candidate_count=4,
        line_search_combined_not_worse_candidate_count=9,
        line_search_best_trial_finite=True,
        line_search_best_trial_combined_improved=True,
        line_search_best_trial_budget_relative_not_worse=False,
        line_search_best_trial_filter_accepted=True,
        line_search_best_trial_budget_not_broken=True,
        line_search_best_trial_budget_relative_not_broken=False,
        line_search_best_trial_combined_not_worse=True,
        line_search_best_trial_accepted=False,
        line_search_best_trial_fallback_accepted=False,
        line_search_soc_candidate_count=26,
        line_search_soc_accepted_candidate_count=0,
        line_search_soc_fallback_candidate_count=1,
        line_search_soc_budget_relative_not_worse_candidate_count=2,
        line_search_soc_filter_candidate_count=3,
        line_search_soc_best_trial_present=True,
        line_search_soc_best_trial_index=7,
        line_search_soc_best_trial_alpha=0.0625,
        line_search_soc_best_trial_residual=6.5,
        line_search_soc_best_trial_gas_residual=1.125,
        line_search_soc_best_trial_condensate_stationarity_residual=2.125,
        line_search_soc_best_trial_budget_residual=3.125,
        line_search_soc_best_trial_budget_relative_residual_max=0.15,
        line_search_soc_best_trial_complementarity_residual=4.125,
        line_search_soc_best_trial_total_density_residual=5.125,
        line_search_soc_best_trial_combined_improved=True,
        line_search_soc_best_trial_budget_relative_not_worse=False,
        line_search_soc_best_trial_filter_accepted=True,
        line_search_soc_best_trial_accepted=False,
        line_search_soc_best_trial_fallback_accepted=False,
        line_search_selected_trial_gas_residual=1.5,
        line_search_selected_trial_condensate_stationarity_residual=2.5,
        line_search_selected_trial_budget_residual=3.5,
        line_search_selected_trial_budget_relative_residual_max=0.25,
        line_search_selected_trial_complementarity_residual=4.5,
        line_search_selected_trial_total_density_residual=5.5,
        line_search_candidate_diagnostics=tuple([float(i)] for i in range(28)),
        gas_residual_norm=1.0,
        condensate_stationarity_residual_norm=2.0,
        budget_residual_norm=3.0,
        budget_relative_residual_max=0.125,
        complementarity_residual_norm=4.0,
        total_density_residual_norm=5.0,
        rho_initialization="unit_activity",
        lambda_initialization="best_residual",
        effective_epsilon=-10.0,
        budget_relative_acceptance_floor=1.0e-3,
        budget_direction_projection_strength=0.0,
        convergence_log_tolerance=1.0e-5,
        convergence_budget_relative_tolerance=1.0e-4,
        convergence_budget_relative_floor=1.0e-8,
        convergence_total_density_tolerance=1.0e-5,
        tiny_step_consecutive_limit=50,
        relaxed_stationarity_fallback_enabled=False,
        relaxed_stationarity_fallback_factor=0.999,
        adaptive_regularization_enabled=True,
        adaptive_regularization_base=1.0e-10,
        second_order_correction_enabled=True,
        second_order_correction_max_abs_step=1.0,
        second_order_correction_trial_order="interleave",
        second_order_correction_budget_passes=4,
        second_order_correction_dual_repair=True,
        second_order_correction_policy="legacy_budget_projection",
        second_order_correction_kappa_soc=0.99,
        second_order_correction_alpha_y_policy="full",
        ipopt_filter_acceptance_enabled=True,
        ipopt_filter_policy="current_iterate",
        ipopt_filter_theta_norm="max_scaled",
        ipopt_filter_budget_relative_max=0.25,
        line_search_candidate_selection_policy="ipopt_vectorized_max_alpha",
        use_legacy_capacity_epsilon=False,
        use_log_amount_boundary=False,
        use_log_activity_boundary=False,
        step_control_policy="scalar_fraction_to_boundary",
    )

    payload = metadata["pdipm_rgie_v11_activity_correction_fixed_support_batch"]
    assert payload["schema"] == (
        "exogibbs_pdipm_rgie_v11_activity_correction_fixed_support_batch_v1"
    )
    assert payload["stop_reason_labels"] == FIXED_SUPPORT_BATCH_STOP_REASON_LABELS
    assert payload["residual_component_labels"] == (
        FIXED_SUPPORT_BATCH_RESIDUAL_COMPONENT_LABELS
    )
    assert payload["accepted_iteration_count"] == 3
    assert payload["tiny_step_consecutive_count"] == 7
    assert payload["tiny_step_consecutive_limit"] == 50
    assert payload["use_log_amount_boundary"] is False
    assert payload["use_log_activity_boundary"] is False
    assert payload["line_search_alpha_boundary"] == 0.5
    assert payload["line_search_alpha_r"] == 0.75
    assert payload["line_search_alpha_rho"] == 0.5
    assert payload["line_search_selected_trial_index"] == 2
    assert payload["line_search_selected_trial_alpha"] == 0.25
    assert payload["line_search_selected_trial_residual"] == 8.0
    assert payload["line_search_accepted_candidate_count"] == 1
    assert payload["line_search_fallback_candidate_count"] == 2
    assert payload["line_search_best_trial_index"] == 3
    assert payload["line_search_best_trial_alpha"] == 0.125
    assert payload["line_search_best_trial_residual"] == 7.5
    assert payload["line_search_best_trial_budget_relative_residual_max"] == 0.2
    assert payload["line_search_finite_candidate_count"] == 13
    assert payload["line_search_combined_improved_candidate_count"] == 8
    assert payload["line_search_budget_relative_not_worse_candidate_count"] == 4
    assert payload["line_search_best_trial_finite"] is True
    assert payload["line_search_best_trial_budget_relative_not_worse"] is False
    assert payload["line_search_best_trial_accepted"] is False
    assert payload["line_search_soc_candidate_count"] == 26
    assert payload["line_search_soc_best_trial_present"] is True
    assert payload["line_search_soc_best_trial_index"] == 7
    assert payload["line_search_soc_best_trial_alpha"] == 0.0625
    assert payload["line_search_soc_best_trial_budget_relative_residual_max"] == 0.15
    assert payload["line_search_soc_best_trial_filter_accepted"] is True
    assert payload["line_search_candidate_diagnostics"]["alpha"][0] == 0.0
    assert (
        payload["line_search_candidate_diagnostics"]["soc_trial"][0] == 16.0
    )
    assert payload["line_search_candidate_diagnostics"]["filter_theta"][0] == 19.0
    assert payload["line_search_candidate_diagnostics"]["barrier_objective"][0] == 20.0
    assert (
        payload["line_search_candidate_diagnostics"][
            "barrier_objective_linearized_change"
        ][0]
        == 21.0
    )
    assert (
        payload["line_search_candidate_diagnostics"][
            "full_newton_linearized_residual"
        ][0]
        == 22.0
    )
    assert payload["line_search_candidate_diagnostics"]["filter_f_type"][0] == 23.0
    assert payload["line_search_candidate_diagnostics"]["filter_armijo"][0] == 24.0
    assert (
        payload["line_search_candidate_diagnostics"][
            "filter_history_accepted"
        ][0]
        == 25.0
    )
    assert (
        payload["line_search_candidate_diagnostics"][
            "filter_entry_count_before"
        ][0]
        == 26.0
    )
    assert (
        payload["line_search_candidate_diagnostics"][
            "soft_restoration_accepted"
        ][0]
        == 27.0
    )
    assert payload["second_order_correction_trial_order"] == "interleave"
    assert payload["second_order_correction_policy"] == "legacy_budget_projection"
    assert payload["second_order_correction_kappa_soc"] == 0.99
    assert payload["second_order_correction_alpha_y_policy"] == "full"
    assert payload["second_order_correction_budget_passes"] == 4
    assert payload["second_order_correction_max_soc"] == 4
    assert payload["second_order_correction_dual_repair"] is True
    assert payload["ipopt_filter_acceptance_enabled"] is True
    assert payload["ipopt_filter_policy"] == "current_iterate"
    assert (
        payload["budget_restoration_policy"]
        == "positive_negative_slack_proximity_v1"
    )
    assert payload["ipopt_filter_budget_relative_max"] == 0.25
    assert (
        payload["line_search_candidate_selection_policy"]
        == "ipopt_vectorized_max_alpha"
    )
    assert payload["budget_residual_norm"] == 3.0
    assert payload["budget_relative_residual_max"] == 0.125
    assert payload["convergence_policy"] == "fastchem_style_componentwise_v1"
    assert payload["convergence_log_tolerance"] == 1.0e-5
    assert payload["convergence_budget_relative_floor"] == 1.0e-8
