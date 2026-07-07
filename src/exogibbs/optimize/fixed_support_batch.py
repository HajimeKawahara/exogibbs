"""Shared helpers for experimental fixed-support batch condensate solves."""

from __future__ import annotations

from typing import Any

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
)
FIXED_SUPPORT_BATCH_RESIDUAL_COMPONENT_LABELS = (
    "gas",
    "condensate_stationarity",
    "budget",
    "complementarity",
    "total_density",
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


def build_fixed_support_batch_metadata(
    *,
    accepted_count: Any,
    normal_accepted_count: Any,
    fallback_accepted_count: Any,
    restoration_accepted_count: Any,
    soc_accepted_count: Any,
    adaptive_regularization_selected_count: Any,
    rejected_trial_count: Any,
    final_step_size: Any,
    stop_reason_code: Any,
    dominant_residual_component_index: Any,
    final_log_activity_correction: Any,
    final_element_potential: Any,
    initial_residual: Any,
    lambda_selection_index: Any,
    gas_residual_norm: Any,
    condensate_stationarity_residual_norm: Any,
    budget_residual_norm: Any,
    complementarity_residual_norm: Any,
    total_density_residual_norm: Any,
    rho_initialization: str,
    lambda_initialization: str,
    effective_epsilon: float,
    budget_relative_acceptance_floor: float,
    budget_direction_projection_strength: float,
    relaxed_stationarity_fallback_enabled: bool,
    relaxed_stationarity_fallback_factor: float,
    adaptive_regularization_enabled: bool,
    adaptive_regularization_base: float,
    second_order_correction_enabled: bool,
    second_order_correction_max_abs_step: float,
    use_legacy_capacity_epsilon: bool,
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
            "gas_residual_norm": gas_residual_norm,
            "condensate_stationarity_residual_norm": (
                condensate_stationarity_residual_norm
            ),
            "budget_residual_norm": budget_residual_norm,
            "complementarity_residual_norm": complementarity_residual_norm,
            "total_density_residual_norm": total_density_residual_norm,
            "rho_initialization": str(rho_initialization),
            "lambda_initialization": str(lambda_initialization),
            "effective_epsilon": effective_epsilon,
            "budget_relative_acceptance_floor": budget_relative_acceptance_floor,
            "budget_direction_projection_strength": budget_direction_projection_strength,
            "relaxed_stationarity_fallback_enabled": (
                relaxed_stationarity_fallback_enabled
            ),
            "relaxed_stationarity_fallback_factor": (
                relaxed_stationarity_fallback_factor
            ),
            "adaptive_regularization_enabled": adaptive_regularization_enabled,
            "adaptive_regularization_base": adaptive_regularization_base,
            "second_order_correction_enabled": second_order_correction_enabled,
            "second_order_correction_max_abs_step": (
                second_order_correction_max_abs_step
            ),
            "use_legacy_capacity_epsilon": use_legacy_capacity_epsilon,
            "step_control_policy": step_control_policy,
        }
    }


__all__ = [
    "FIXED_SUPPORT_BATCH_DEFAULT_EPSILON_SCHEDULE",
    "FIXED_SUPPORT_BATCH_LAMBDA_CANDIDATE_LABELS",
    "FIXED_SUPPORT_BATCH_LAMBDA_INITIALIZATIONS",
    "FIXED_SUPPORT_BATCH_RESIDUAL_COMPONENT_LABELS",
    "FIXED_SUPPORT_BATCH_STOP_REASON_LABELS",
    "build_fixed_support_batch_metadata",
]
