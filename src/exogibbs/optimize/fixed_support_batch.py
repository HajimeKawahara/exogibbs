"""Shared helpers for experimental fixed-support batch condensate solves."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

FIXED_SUPPORT_BATCH_LAMBDA_CANDIDATE_LABELS = (
    "provided",
    "gas_lstsq",
    "gas_cond_lstsq",
    "damped_gas_lstsq",
    "damped_gas_cond_lstsq",
)
FIXED_SUPPORT_BATCH_STOP_REASON_LABELS = (
    "converged",
    "max_iter",
    "max_iter_tiny_step",
    "no_accepted_trial",
    "nonfinite_residual",
    "unknown_not_converged",
    "tiny_step_stalled",
)
FIXED_SUPPORT_BATCH_RESIDUAL_COMPONENT_LABELS = (
    "gas",
    "condensate_stationarity",
    "budget",
    "complementarity",
    "total_density",
)
FIXED_SUPPORT_RESTORATION_DIAGNOSTIC_LABELS = (
    "full",
    "gas",
    "condensate_stationarity",
    "budget_relative_max",
    "complementarity",
    "total_density",
)
FIXED_SUPPORT_RESTORATION_FIRST_NORMAL_TYPE_LABELS = (
    "normal",
    "stationarity_restoration",
    "soc",
    "not_recorded",
)
FIXED_SUPPORT_BATCH_DEFAULT_EPSILON_SCHEDULE = (
    0.0,
    -1.0,
    -2.0,
    -4.0,
    -6.0,
    -8.0,
    -10.0,
)
FIXED_SUPPORT_BATCH_LAMBDA_INITIALIZATIONS = (
    "provided",
    "gas_lstsq",
    "gas_cond_lstsq",
    "best_residual",
)


def build_ipopt_current_iterate_filter_mask(
    *,
    finite: Any,
    protected_theta: Any,
    complementarity_merit: Any,
    initial_protected_theta: Any,
    initial_complementarity_merit: Any,
    required_complementarity_factor: Any,
    relaxed_fallback_enabled: Any,
    relaxed_fallback_factor: Any,
    theta_reduction_fraction: float = 1.0e-4,
    theta_absolute_slack: float = 1.0e-10,
    theta_relative_growth_limit: float = 1.001,
) -> Any:
    """Return an Ipopt-style current-iterate filter mask for trial points.

    ``protected_theta`` is the constraint-violation side of the filter.  For
    the fixed-support batch diagnostic it covers gas stationarity, condensate
    stationarity, budget, and total-density residuals.  ``complementarity_merit``
    is the objective-like side that may improve while theta is only protected.
    """

    theta = jnp.asarray(protected_theta)
    phi = jnp.asarray(complementarity_merit)
    initial_theta = jnp.asarray(initial_protected_theta, dtype=theta.dtype)
    initial_phi = jnp.asarray(initial_complementarity_merit, dtype=phi.dtype)
    required_factor = jnp.asarray(required_complementarity_factor, dtype=phi.dtype)
    relaxed_factor = jnp.asarray(relaxed_fallback_factor, dtype=phi.dtype)
    finite_mask = jnp.asarray(finite, dtype=bool)
    strict_phi_improved = phi <= required_factor * initial_phi
    relaxed_phi_improved = phi <= relaxed_factor * initial_phi
    phi_improved = jnp.where(
        jnp.asarray(relaxed_fallback_enabled, dtype=bool),
        relaxed_phi_improved,
        strict_phi_improved,
    )
    theta_fraction = jnp.asarray(theta_reduction_fraction, dtype=theta.dtype)
    theta_reduced = theta <= (1.0 - theta_fraction) * initial_theta
    theta_limit = jnp.maximum(
        initial_theta + jnp.asarray(theta_absolute_slack, dtype=theta.dtype),
        jnp.asarray(theta_relative_growth_limit, dtype=theta.dtype) * initial_theta,
    )
    theta_not_broken = theta <= theta_limit
    return finite_mask & (phi_improved | theta_reduced) & theta_not_broken


def build_fixed_support_batch_metadata(
    *,
    accepted_count: Any,
    normal_accepted_count: Any,
    fallback_accepted_count: Any,
    restoration_accepted_count: Any,
    soc_accepted_count: Any,
    adaptive_regularization_selected_count: Any,
    rejected_trial_count: Any,
    tiny_step_consecutive_count: Any,
    final_step_size: Any,
    stop_reason_code: Any,
    dominant_residual_component_index: Any,
    final_log_activity_correction: Any,
    final_element_potential: Any,
    initial_residual: Any,
    lambda_selection_index: Any,
    line_search_alpha_boundary: Any,
    line_search_alpha_r: Any,
    line_search_alpha_rho: Any,
    line_search_selected_trial_index: Any,
    line_search_selected_trial_alpha: Any,
    line_search_selected_trial_residual: Any,
    line_search_accepted_candidate_count: Any,
    line_search_fallback_candidate_count: Any,
    line_search_best_trial_index: Any,
    line_search_best_trial_alpha: Any,
    line_search_best_trial_residual: Any,
    line_search_best_trial_gas_residual: Any,
    line_search_best_trial_condensate_stationarity_residual: Any,
    line_search_best_trial_budget_residual: Any,
    line_search_best_trial_budget_relative_residual_max: Any,
    line_search_best_trial_complementarity_residual: Any,
    line_search_best_trial_total_density_residual: Any,
    line_search_finite_candidate_count: Any,
    line_search_combined_improved_candidate_count: Any,
    line_search_budget_relative_not_worse_candidate_count: Any,
    line_search_filter_candidate_count: Any,
    line_search_budget_not_broken_candidate_count: Any,
    line_search_budget_relative_not_broken_candidate_count: Any,
    line_search_combined_not_worse_candidate_count: Any,
    line_search_best_trial_finite: Any,
    line_search_best_trial_combined_improved: Any,
    line_search_best_trial_budget_relative_not_worse: Any,
    line_search_best_trial_filter_accepted: Any,
    line_search_best_trial_budget_not_broken: Any,
    line_search_best_trial_budget_relative_not_broken: Any,
    line_search_best_trial_combined_not_worse: Any,
    line_search_best_trial_accepted: Any,
    line_search_best_trial_fallback_accepted: Any,
    line_search_soc_candidate_count: Any,
    line_search_soc_accepted_candidate_count: Any,
    line_search_soc_fallback_candidate_count: Any,
    line_search_soc_budget_relative_not_worse_candidate_count: Any,
    line_search_soc_filter_candidate_count: Any,
    line_search_soc_best_trial_present: Any,
    line_search_soc_best_trial_index: Any,
    line_search_soc_best_trial_alpha: Any,
    line_search_soc_best_trial_residual: Any,
    line_search_soc_best_trial_gas_residual: Any,
    line_search_soc_best_trial_condensate_stationarity_residual: Any,
    line_search_soc_best_trial_budget_residual: Any,
    line_search_soc_best_trial_budget_relative_residual_max: Any,
    line_search_soc_best_trial_complementarity_residual: Any,
    line_search_soc_best_trial_total_density_residual: Any,
    line_search_soc_best_trial_combined_improved: Any,
    line_search_soc_best_trial_budget_relative_not_worse: Any,
    line_search_soc_best_trial_filter_accepted: Any,
    line_search_soc_best_trial_accepted: Any,
    line_search_soc_best_trial_fallback_accepted: Any,
    line_search_selected_trial_gas_residual: Any,
    line_search_selected_trial_condensate_stationarity_residual: Any,
    line_search_selected_trial_budget_residual: Any,
    line_search_selected_trial_budget_relative_residual_max: Any,
    line_search_selected_trial_complementarity_residual: Any,
    line_search_selected_trial_total_density_residual: Any,
    line_search_candidate_diagnostics: Any,
    gas_residual_norm: Any,
    condensate_stationarity_residual_norm: Any,
    budget_residual_norm: Any,
    budget_relative_residual_max: Any,
    complementarity_residual_norm: Any,
    total_density_residual_norm: Any,
    rho_initialization: str,
    lambda_initialization: str,
    effective_epsilon: float,
    budget_relative_acceptance_floor: float,
    budget_direction_projection_strength: float,
    convergence_log_tolerance: float,
    convergence_budget_relative_tolerance: float,
    convergence_budget_relative_floor: float,
    convergence_total_density_tolerance: float,
    tiny_step_consecutive_limit: int,
    relaxed_stationarity_fallback_enabled: bool,
    relaxed_stationarity_fallback_factor: float,
    adaptive_regularization_enabled: bool,
    adaptive_regularization_base: float,
    second_order_correction_enabled: bool,
    second_order_correction_max_abs_step: float,
    second_order_correction_trial_order: str,
    second_order_correction_budget_passes: int,
    second_order_correction_dual_repair: bool,
    second_order_correction_policy: str,
    second_order_correction_kappa_soc: float,
    second_order_correction_alpha_y_policy: str,
    second_order_correction_charge_solve_policy: str = "coupled",
    second_order_correction_reduced_mode_policy: str = "full",
    second_order_correction_diagnostic_mode_vector_policy: str = (
        "smallest_right_singular"
    ),
    budget_restoration_enabled: bool = False,
    budget_restoration_coordinate_policy: str = "log",
    budget_restoration_dual_recenter: bool = False,
    budget_restoration_dual_recenter_policy: str = "hard",
    budget_restoration_proximity_weight: float = 1.0e-4,
    budget_restoration_max_abs_step: float = 2.0,
    budget_restoration_passes: int = 1,
    budget_restoration_phase_enabled: bool = False,
    budget_restoration_phase_theta_reduction: float = 0.9,
    budget_restoration_phase_cooldown_iterations: int = 1,
    restoration_phase_entry_theta_at_stop: Any = 0.0,
    restoration_phase_active_at_stop: Any = False,
    restoration_phase_cooldown_at_stop: Any = 0,
    amount_restoration_accepted_count: Any = 0,
    restoration_phase_entry_count: Any = 0,
    restoration_phase_exit_count: Any = 0,
    restoration_bound_multiplier_reset_count: Any = 0,
    restoration_equality_multiplier_reset_count: Any = 0,
    restoration_last_exit_theta: Any = 0.0,
    restoration_last_dual_alpha: Any = 1.0,
    restoration_entry_residual_vector: Any = 0.0,
    restoration_best_residual_vector: Any = 0.0,
    restoration_best_theta: Any = 0.0,
    restoration_last_exit_predual_residual_vector: Any = 0.0,
    restoration_last_exit_postdual_residual_vector: Any = 0.0,
    restoration_first_normal_residual_vector: Any = 0.0,
    restoration_first_normal_attempted: Any = False,
    restoration_first_normal_accepted: Any = False,
    restoration_first_normal_selected_type: Any = 3,
    restoration_return_probe_pending: Any = False,
    restoration_active_accepted_count: Any = 0,
    restoration_last_active_accepted_count: Any = 0,
    ipopt_filter_acceptance_enabled: bool,
    ipopt_filter_policy: str,
    ipopt_filter_theta_norm: str,
    ipopt_filter_budget_relative_max: float,
    line_search_candidate_selection_policy: str,
    use_legacy_capacity_epsilon: bool,
    use_log_amount_boundary: bool,
    use_log_activity_boundary: bool,
    step_control_policy: str,
) -> dict[str, Any]:
    """Build the exported metadata for one fixed-support batch solve."""

    return {
        "pdipm_rgie_v11_activity_correction_fixed_support_batch": {
            "schema": "exogibbs_pdipm_rgie_v11_activity_correction_fixed_support_batch_v1",
            "experimental": True,
            "production_route_wiring": False,
            "accepted_iteration_count": accepted_count,
            "normal_accepted_iteration_count": normal_accepted_count,
            "fallback_accepted_iteration_count": fallback_accepted_count,
            "stationarity_restoration_accepted_iteration_count": (
                restoration_accepted_count
            ),
            "second_order_correction_accepted_iteration_count": soc_accepted_count,
            "adaptive_regularization_selected_iteration_count": (
                adaptive_regularization_selected_count
            ),
            "rejected_trial_count": rejected_trial_count,
            "tiny_step_consecutive_count": tiny_step_consecutive_count,
            "final_step_size": final_step_size,
            "stop_reason_code": stop_reason_code,
            "stop_reason_labels": FIXED_SUPPORT_BATCH_STOP_REASON_LABELS,
            "dominant_residual_component_index": dominant_residual_component_index,
            "residual_component_labels": FIXED_SUPPORT_BATCH_RESIDUAL_COMPONENT_LABELS,
            "final_log_activity_correction": final_log_activity_correction,
            "final_element_potential": final_element_potential,
            "initial_residual": initial_residual,
            "lambda_selection_index": lambda_selection_index,
            "lambda_candidate_labels": FIXED_SUPPORT_BATCH_LAMBDA_CANDIDATE_LABELS,
            "line_search_alpha_boundary": line_search_alpha_boundary,
            "line_search_alpha_r": line_search_alpha_r,
            "line_search_alpha_rho": line_search_alpha_rho,
            "line_search_selected_trial_index": line_search_selected_trial_index,
            "line_search_selected_trial_alpha": line_search_selected_trial_alpha,
            "line_search_selected_trial_residual": (
                line_search_selected_trial_residual
            ),
            "line_search_accepted_candidate_count": (
                line_search_accepted_candidate_count
            ),
            "line_search_fallback_candidate_count": (
                line_search_fallback_candidate_count
            ),
            "line_search_best_trial_index": line_search_best_trial_index,
            "line_search_best_trial_alpha": line_search_best_trial_alpha,
            "line_search_best_trial_residual": line_search_best_trial_residual,
            "line_search_best_trial_gas_residual": (
                line_search_best_trial_gas_residual
            ),
            "line_search_best_trial_condensate_stationarity_residual": (
                line_search_best_trial_condensate_stationarity_residual
            ),
            "line_search_best_trial_budget_residual": (
                line_search_best_trial_budget_residual
            ),
            "line_search_best_trial_budget_relative_residual_max": (
                line_search_best_trial_budget_relative_residual_max
            ),
            "line_search_best_trial_complementarity_residual": (
                line_search_best_trial_complementarity_residual
            ),
            "line_search_best_trial_total_density_residual": (
                line_search_best_trial_total_density_residual
            ),
            "line_search_finite_candidate_count": line_search_finite_candidate_count,
            "line_search_combined_improved_candidate_count": (
                line_search_combined_improved_candidate_count
            ),
            "line_search_budget_relative_not_worse_candidate_count": (
                line_search_budget_relative_not_worse_candidate_count
            ),
            "line_search_filter_candidate_count": line_search_filter_candidate_count,
            "line_search_budget_not_broken_candidate_count": (
                line_search_budget_not_broken_candidate_count
            ),
            "line_search_budget_relative_not_broken_candidate_count": (
                line_search_budget_relative_not_broken_candidate_count
            ),
            "line_search_combined_not_worse_candidate_count": (
                line_search_combined_not_worse_candidate_count
            ),
            "line_search_best_trial_finite": line_search_best_trial_finite,
            "line_search_best_trial_combined_improved": (
                line_search_best_trial_combined_improved
            ),
            "line_search_best_trial_budget_relative_not_worse": (
                line_search_best_trial_budget_relative_not_worse
            ),
            "line_search_best_trial_filter_accepted": (
                line_search_best_trial_filter_accepted
            ),
            "line_search_best_trial_budget_not_broken": (
                line_search_best_trial_budget_not_broken
            ),
            "line_search_best_trial_budget_relative_not_broken": (
                line_search_best_trial_budget_relative_not_broken
            ),
            "line_search_best_trial_combined_not_worse": (
                line_search_best_trial_combined_not_worse
            ),
            "line_search_best_trial_accepted": line_search_best_trial_accepted,
            "line_search_best_trial_fallback_accepted": (
                line_search_best_trial_fallback_accepted
            ),
            "line_search_soc_candidate_count": line_search_soc_candidate_count,
            "line_search_soc_accepted_candidate_count": (
                line_search_soc_accepted_candidate_count
            ),
            "line_search_soc_fallback_candidate_count": (
                line_search_soc_fallback_candidate_count
            ),
            "line_search_soc_budget_relative_not_worse_candidate_count": (
                line_search_soc_budget_relative_not_worse_candidate_count
            ),
            "line_search_soc_filter_candidate_count": (
                line_search_soc_filter_candidate_count
            ),
            "line_search_soc_best_trial_present": line_search_soc_best_trial_present,
            "line_search_soc_best_trial_index": line_search_soc_best_trial_index,
            "line_search_soc_best_trial_alpha": line_search_soc_best_trial_alpha,
            "line_search_soc_best_trial_residual": line_search_soc_best_trial_residual,
            "line_search_soc_best_trial_gas_residual": (
                line_search_soc_best_trial_gas_residual
            ),
            "line_search_soc_best_trial_condensate_stationarity_residual": (
                line_search_soc_best_trial_condensate_stationarity_residual
            ),
            "line_search_soc_best_trial_budget_residual": (
                line_search_soc_best_trial_budget_residual
            ),
            "line_search_soc_best_trial_budget_relative_residual_max": (
                line_search_soc_best_trial_budget_relative_residual_max
            ),
            "line_search_soc_best_trial_complementarity_residual": (
                line_search_soc_best_trial_complementarity_residual
            ),
            "line_search_soc_best_trial_total_density_residual": (
                line_search_soc_best_trial_total_density_residual
            ),
            "line_search_soc_best_trial_combined_improved": (
                line_search_soc_best_trial_combined_improved
            ),
            "line_search_soc_best_trial_budget_relative_not_worse": (
                line_search_soc_best_trial_budget_relative_not_worse
            ),
            "line_search_soc_best_trial_filter_accepted": (
                line_search_soc_best_trial_filter_accepted
            ),
            "line_search_soc_best_trial_accepted": (
                line_search_soc_best_trial_accepted
            ),
            "line_search_soc_best_trial_fallback_accepted": (
                line_search_soc_best_trial_fallback_accepted
            ),
            "line_search_selected_trial_gas_residual": (
                line_search_selected_trial_gas_residual
            ),
            "line_search_selected_trial_condensate_stationarity_residual": (
                line_search_selected_trial_condensate_stationarity_residual
            ),
            "line_search_selected_trial_budget_residual": (
                line_search_selected_trial_budget_residual
            ),
            "line_search_selected_trial_budget_relative_residual_max": (
                line_search_selected_trial_budget_relative_residual_max
            ),
            "line_search_selected_trial_complementarity_residual": (
                line_search_selected_trial_complementarity_residual
            ),
            "line_search_selected_trial_total_density_residual": (
                line_search_selected_trial_total_density_residual
            ),
            "line_search_candidate_diagnostics": {
                "alpha": line_search_candidate_diagnostics[0],
                "residual": line_search_candidate_diagnostics[1],
                "gas_residual": line_search_candidate_diagnostics[2],
                "condensate_stationarity_residual": (
                    line_search_candidate_diagnostics[3]
                ),
                "budget_residual": line_search_candidate_diagnostics[4],
                "budget_relative_residual_max": line_search_candidate_diagnostics[5],
                "complementarity_residual": line_search_candidate_diagnostics[6],
                "total_density_residual": line_search_candidate_diagnostics[7],
                "finite": line_search_candidate_diagnostics[8],
                "accepted": line_search_candidate_diagnostics[9],
                "fallback_accepted": line_search_candidate_diagnostics[10],
                "filter_accepted": line_search_candidate_diagnostics[11],
                "budget_relative_not_worse": line_search_candidate_diagnostics[12],
                "budget_not_broken": line_search_candidate_diagnostics[13],
                "budget_relative_not_broken": line_search_candidate_diagnostics[14],
                "combined_not_worse": line_search_candidate_diagnostics[15],
                "soc_trial": line_search_candidate_diagnostics[16],
                "restoration_trial": line_search_candidate_diagnostics[17],
                **(
                    {
                        "budget_restoration_trial": (
                            line_search_candidate_diagnostics[18]
                        )
                    }
                    if len(line_search_candidate_diagnostics) > 18
                    else {}
                ),
                **(
                    {
                        "filter_theta": line_search_candidate_diagnostics[19],
                        "barrier_objective": line_search_candidate_diagnostics[20],
                        "barrier_objective_linearized_change": (
                            line_search_candidate_diagnostics[21]
                        ),
                    }
                    if len(line_search_candidate_diagnostics) > 21
                    else {}
                ),
                **(
                    {
                        "full_newton_linearized_residual": (
                            line_search_candidate_diagnostics[22]
                        )
                    }
                    if len(line_search_candidate_diagnostics) > 22
                    else {}
                ),
                **(
                    {
                        "filter_f_type": line_search_candidate_diagnostics[23],
                        "filter_armijo": line_search_candidate_diagnostics[24],
                        "filter_history_accepted": (
                            line_search_candidate_diagnostics[25]
                        ),
                        "filter_entry_count_before": (
                            line_search_candidate_diagnostics[26]
                        ),
                    }
                    if len(line_search_candidate_diagnostics) > 26
                    else {}
                ),
                **(
                    {
                        "soft_restoration_accepted": (
                            line_search_candidate_diagnostics[27]
                        )
                    }
                    if len(line_search_candidate_diagnostics) > 27
                    else {}
                ),
                **(
                    {
                        "ipopt_soc_eligible_iteration_count": (
                            line_search_candidate_diagnostics[28]
                        ),
                        "ipopt_soc_correction_count": (
                            line_search_candidate_diagnostics[29]
                        ),
                        "ipopt_soc_finite_direction_iteration_count": (
                            line_search_candidate_diagnostics[30]
                        ),
                        "ipopt_soc_filter_accepted_iteration_count": (
                            line_search_candidate_diagnostics[31]
                        ),
                        "ipopt_soc_kappa_stopped_iteration_count": (
                            line_search_candidate_diagnostics[32]
                        ),
                        "ipopt_soc_last_normal_theta": (
                            line_search_candidate_diagnostics[33]
                        ),
                        "ipopt_soc_last_final_theta": (
                            line_search_candidate_diagnostics[34]
                        ),
                        "ipopt_soc_last_normal_phi": (
                            line_search_candidate_diagnostics[35]
                        ),
                        "ipopt_soc_last_final_phi": (
                            line_search_candidate_diagnostics[36]
                        ),
                        "ipopt_soc_last_alpha": (
                            line_search_candidate_diagnostics[37]
                        ),
                    }
                    if len(line_search_candidate_diagnostics) > 37
                    else {}
                ),
                **(
                    {
                        "ipopt_soc_max_solve_linear_residual": (
                            line_search_candidate_diagnostics[38]
                        ),
                        "ipopt_soc_max_solve_solution_norm": (
                            line_search_candidate_diagnostics[39]
                        ),
                        "ipopt_soc_min_solve_singular_value": (
                            line_search_candidate_diagnostics[40]
                        ),
                        "ipopt_soc_max_solve_condition_estimate": (
                            line_search_candidate_diagnostics[41]
                        ),
                        "ipopt_soc_max_scaled_solve_condition_estimate": (
                            line_search_candidate_diagnostics[42]
                        ),
                        "ipopt_soc_max_relative_solve_linear_residual": (
                            line_search_candidate_diagnostics[43]
                        ),
                    }
                    if len(line_search_candidate_diagnostics) > 43
                    else {}
                ),
                **(
                    {
                        "ipopt_filter_consecutive_rejection_count": (
                            line_search_candidate_diagnostics[44]
                        ),
                        "ipopt_filter_reset_count": (
                            line_search_candidate_diagnostics[45]
                        ),
                    }
                    if len(line_search_candidate_diagnostics) > 45
                    else {}
                ),
                **(
                    {
                        "ipopt_soc_first_selected_recorded": (
                            line_search_candidate_diagnostics[46]
                        ),
                        "ipopt_soc_first_selected_solution_norm": (
                            line_search_candidate_diagnostics[47]
                        ),
                        "ipopt_soc_first_selected_relative_linear_residual": (
                            line_search_candidate_diagnostics[48]
                        ),
                        "ipopt_soc_first_selected_condition_estimate": (
                            line_search_candidate_diagnostics[49]
                        ),
                        "ipopt_soc_first_selected_scaled_condition_estimate": (
                            line_search_candidate_diagnostics[50]
                        ),
                        "ipopt_soc_first_selected_smallest_singular_value": (
                            line_search_candidate_diagnostics[51]
                        ),
                        "ipopt_soc_first_selected_null_lambda_norm": (
                            line_search_candidate_diagnostics[52]
                        ),
                        "ipopt_soc_first_selected_null_qtot_abs": (
                            line_search_candidate_diagnostics[53]
                        ),
                        "ipopt_soc_first_selected_null_dominant_lambda_index": (
                            line_search_candidate_diagnostics[54]
                        ),
                        "ipopt_soc_first_selected_null_dominant_lambda_abs": (
                            line_search_candidate_diagnostics[55]
                        ),
                    }
                    if len(line_search_candidate_diagnostics) > 55
                    else {}
                ),
                **(
                    {
                        "ipopt_soc_first_selected_diagnostic_mode_vector": (
                            line_search_candidate_diagnostics[56]
                        ),
                    }
                    if len(line_search_candidate_diagnostics) > 56
                    else {}
                ),
            },
            "gas_residual_norm": gas_residual_norm,
            "condensate_stationarity_residual_norm": (
                condensate_stationarity_residual_norm
            ),
            "budget_residual_norm": budget_residual_norm,
            "budget_relative_residual_max": budget_relative_residual_max,
            "complementarity_residual_norm": complementarity_residual_norm,
            "total_density_residual_norm": total_density_residual_norm,
            "rho_initialization": str(rho_initialization),
            "lambda_initialization": str(lambda_initialization),
            "effective_epsilon": effective_epsilon,
            "budget_relative_acceptance_floor": budget_relative_acceptance_floor,
            "budget_direction_projection_strength": budget_direction_projection_strength,
            "convergence_policy": "fastchem_style_componentwise_v1",
            "convergence_log_tolerance": convergence_log_tolerance,
            "convergence_budget_relative_tolerance": (
                convergence_budget_relative_tolerance
            ),
            "convergence_budget_relative_floor": convergence_budget_relative_floor,
            "convergence_total_density_tolerance": (
                convergence_total_density_tolerance
            ),
            "tiny_step_consecutive_limit": int(tiny_step_consecutive_limit),
            "relaxed_stationarity_fallback_enabled": (
                relaxed_stationarity_fallback_enabled
            ),
            "relaxed_stationarity_fallback_factor": (
                relaxed_stationarity_fallback_factor
            ),
            "adaptive_regularization_enabled": adaptive_regularization_enabled,
            "adaptive_regularization_base": adaptive_regularization_base,
            "second_order_correction_enabled": second_order_correction_enabled,
            "second_order_correction_policy": (
                second_order_correction_policy
            ),
            "second_order_correction_kappa_soc": second_order_correction_kappa_soc,
            "second_order_correction_alpha_y_policy": (
                second_order_correction_alpha_y_policy
            ),
            "second_order_correction_charge_solve_policy": (
                second_order_correction_charge_solve_policy
            ),
            "second_order_correction_reduced_mode_policy": (
                second_order_correction_reduced_mode_policy
            ),
            "second_order_correction_diagnostic_mode_vector_policy": (
                second_order_correction_diagnostic_mode_vector_policy
            ),
            "second_order_correction_max_abs_step": (
                second_order_correction_max_abs_step
            ),
            "second_order_correction_trial_order": (
                second_order_correction_trial_order
            ),
            "second_order_correction_budget_passes": (
                second_order_correction_budget_passes
            ),
            "second_order_correction_max_soc": (
                second_order_correction_budget_passes
            ),
            "second_order_correction_dual_repair": (
                second_order_correction_dual_repair
            ),
            "budget_restoration_enabled": budget_restoration_enabled,
            "budget_restoration_coordinate_policy": (
                budget_restoration_coordinate_policy
            ),
            "budget_restoration_dual_recenter": budget_restoration_dual_recenter,
            "budget_restoration_dual_recenter_policy": (
                budget_restoration_dual_recenter_policy
            ),
            "budget_restoration_policy": "positive_negative_slack_proximity_v1",
            "budget_restoration_proximity_weight": (
                budget_restoration_proximity_weight
            ),
            "budget_restoration_max_abs_step": budget_restoration_max_abs_step,
            "budget_restoration_passes": int(budget_restoration_passes),
            "budget_restoration_phase_enabled": budget_restoration_phase_enabled,
            "budget_restoration_phase_theta_reduction": (
                budget_restoration_phase_theta_reduction
            ),
            "budget_restoration_phase_cooldown_iterations": int(
                budget_restoration_phase_cooldown_iterations
            ),
            "restoration_phase_entry_theta_at_stop": (
                restoration_phase_entry_theta_at_stop
            ),
            "restoration_phase_active_at_stop": restoration_phase_active_at_stop,
            "restoration_phase_cooldown_at_stop": restoration_phase_cooldown_at_stop,
            "amount_restoration_accepted_iteration_count": (
                amount_restoration_accepted_count
            ),
            "restoration_phase_entry_count": restoration_phase_entry_count,
            "restoration_phase_exit_count": restoration_phase_exit_count,
            "restoration_bound_multiplier_reset_count": (
                restoration_bound_multiplier_reset_count
            ),
            "restoration_equality_multiplier_reset_count": (
                restoration_equality_multiplier_reset_count
            ),
            "restoration_last_exit_theta": restoration_last_exit_theta,
            "restoration_last_dual_alpha": restoration_last_dual_alpha,
            "restoration_residual_diagnostic_labels": (
                FIXED_SUPPORT_RESTORATION_DIAGNOSTIC_LABELS
            ),
            "restoration_entry_residual_vector": (
                restoration_entry_residual_vector
            ),
            "restoration_best_residual_vector": (
                restoration_best_residual_vector
            ),
            "restoration_best_theta": restoration_best_theta,
            "restoration_last_exit_predual_residual_vector": (
                restoration_last_exit_predual_residual_vector
            ),
            "restoration_last_exit_postdual_residual_vector": (
                restoration_last_exit_postdual_residual_vector
            ),
            "restoration_first_normal_residual_vector": (
                restoration_first_normal_residual_vector
            ),
            "restoration_first_normal_attempted": (
                restoration_first_normal_attempted
            ),
            "restoration_first_normal_accepted": (
                restoration_first_normal_accepted
            ),
            "restoration_first_normal_selected_type": (
                restoration_first_normal_selected_type
            ),
            "restoration_first_normal_selected_type_labels": (
                FIXED_SUPPORT_RESTORATION_FIRST_NORMAL_TYPE_LABELS
            ),
            "restoration_return_probe_pending": restoration_return_probe_pending,
            "restoration_active_accepted_iteration_count_at_stop": (
                restoration_active_accepted_count
            ),
            "restoration_last_active_accepted_iteration_count": (
                restoration_last_active_accepted_count
            ),
            "ipopt_filter_acceptance_enabled": ipopt_filter_acceptance_enabled,
            "ipopt_filter_policy": ipopt_filter_policy,
            "ipopt_filter_theta_norm": ipopt_filter_theta_norm,
            "ipopt_filter_budget_relative_max": ipopt_filter_budget_relative_max,
            "line_search_candidate_selection_policy": (
                line_search_candidate_selection_policy
            ),
            "use_legacy_capacity_epsilon": use_legacy_capacity_epsilon,
            "use_log_amount_boundary": use_log_amount_boundary,
            "use_log_activity_boundary": use_log_activity_boundary,
            "step_control_policy": step_control_policy,
        }
    }


__all__ = [
    "FIXED_SUPPORT_BATCH_DEFAULT_EPSILON_SCHEDULE",
    "FIXED_SUPPORT_BATCH_LAMBDA_CANDIDATE_LABELS",
    "FIXED_SUPPORT_BATCH_LAMBDA_INITIALIZATIONS",
    "FIXED_SUPPORT_BATCH_RESIDUAL_COMPONENT_LABELS",
    "FIXED_SUPPORT_BATCH_STOP_REASON_LABELS",
    "FIXED_SUPPORT_RESTORATION_DIAGNOSTIC_LABELS",
    "FIXED_SUPPORT_RESTORATION_FIRST_NORMAL_TYPE_LABELS",
    "build_ipopt_current_iterate_filter_mask",
    "build_fixed_support_batch_metadata",
]
