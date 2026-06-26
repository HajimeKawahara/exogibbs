"""Backward-compatible import path and structured API for condensate minimization."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from time import perf_counter
from typing import Any, Literal, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, tree_util
from scipy.optimize import least_squares

from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.core import _compute_gk
from exogibbs.optimize.stepsize import LOG_S_MAX
from exogibbs.optimize.pdipm_cond import minimize_gibbs_cond_core
from exogibbs.optimize.minimize import (
    build_minimize_gibbs_core_lnnk_output_source_trace,
    minimize_gibbs_core,
    minimize_gibbs_core_with_source_trace,
)
from exogibbs.optimize.pipm_rgie_cond import (
    _recompute_pi_for_residual,
    build_rgie_condensate_init_from_policy,
    compute_condensate_budget_limits,
    select_conditional_capped_s_reduced_coupling_mode,
    summarize_rgie_inactive_driving,
    diagnose_full_vs_reduced_gie_direction as _diagnose_full_vs_reduced_gie_direction_raw,
    diagnose_pdipm_vs_pipm_direction as _diagnose_pdipm_vs_pipm_direction_raw,
    diagnose_pdipm_vs_pipm_fixed_epsilon_trajectories as _diagnose_pdipm_vs_pipm_fixed_epsilon_trajectories_raw,
    diagnose_reduced_solver_backend_experiments as _diagnose_reduced_solver_backend_experiments_raw,
    diagnose_gas_step_limiter_and_direction as _diagnose_gas_step_limiter_and_direction_raw,
    diagnose_iteration_lambda_trials as _diagnose_iteration_lambda_trials_raw,
    minimize_gibbs_cond_with_diagnostics as _minimize_gibbs_cond_with_diagnostics_raw,
)

Array = jax.Array
CondensateProfileMethod = Literal[
    "vmap_cold",
    "scan_hot_from_top",
    "scan_hot_from_bottom",
    "scan_hot_from_top_final_only",
    "scan_hot_from_bottom_final_only",
]
CondensateEpsilonSchedule = Literal["fixed", "adaptive_sk_guard"]
CondensateRGIEStartupPolicy = Literal[
    "legacy_absolute_m0",
    "ratio_uniform_r0",
    "warm_previous_with_ratio_floor",
]
CondensateRGIESupportMethod = Literal[
    "legacy_current",
    "smoothed_semismooth_outer",
]
InventoryCorrectionMode = Literal[
    "none",
    "startup_budget_capped",
    "budget_guarded_line_search",
    "startup_plus_budget_guard",
    "startup_plus_budget_guard_plus_projection",
]
ReducedCouplingMode = Literal[
    "current",
    "capped_s_only_fixed_alpha",
    "capped_s_only_conditional",
    "candidate_selected_active_only",
    "candidate_selected_active_plus_near_jacobian",
    "candidate_selected_active_plus_near_jacobian_with_rem_inventory",
    "candidate_selected_weighted_mask",
    "pdipm_rgie_v11_activity_correction",
]


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CondensateEquilibriumInit:
    """Explicit condensate solver initialization state.

    This is intentionally small and can be reused as a future hot-start carrier.
    """

    ln_nk: Optional[Array] = None
    ln_mk: Optional[Array] = None
    ln_ntot: Optional[Array] = None
    element_potential: Optional[Array] = None
    rho: Optional[Array] = None
    barrier_epsilon: Optional[Array] = None
    gas_stationarity_source: Optional[Array] = None
    ln_nk_source_trace: Optional[dict[str, Any]] = field(default=None, compare=False, repr=False)

    def tree_flatten(self):
        children = (
            self.ln_nk,
            self.ln_mk,
            self.ln_ntot,
            self.element_potential,
            self.rho,
            self.barrier_epsilon,
            self.gas_stationarity_source,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        (
            ln_nk,
            ln_mk,
            ln_ntot,
            element_potential,
            rho,
            barrier_epsilon,
            gas_stationarity_source,
        ) = children
        return cls(
            ln_nk=ln_nk,
            ln_mk=ln_mk,
            ln_ntot=ln_ntot,
            element_potential=element_potential,
            rho=rho,
            barrier_epsilon=barrier_epsilon,
            gas_stationarity_source=gas_stationarity_source,
        )


@dataclass(frozen=True)
class CondensateRGIEStartupConfig:
    """Optional startup override for the RGIE condensate path.

    ``legacy_absolute_m0`` keeps the current caller-supplied ``ln_mk`` exactly.
    ``ratio_uniform_r0`` replaces the layer-start condensate state with a
    uniform ratio-based seed ``m/nu = r0``.
    ``warm_previous_with_ratio_floor`` keeps the incoming hot start but floors
    every condensate to ``m/nu >= r0`` at the layer-start epsilon.
    """

    policy: CondensateRGIEStartupPolicy = "legacy_absolute_m0"
    r0: Optional[float] = None


@dataclass(frozen=True)
class CondensateRGIEInventoryCorrectionConfig:
    """Opt-in experimental inventory-aware first-pass RGIE correction layer."""

    inventory_correction: InventoryCorrectionMode = "none"
    alpha_init: float = 1.0e-2
    budget_margin: float = 0.0


@dataclass(frozen=True)
class CondensateRGIEReducedCouplingConfig:
    """Opt-in experimental reduced-coupling correction for first-pass RGIE."""

    reduced_coupling_mode: ReducedCouplingMode = "current"
    alpha_s: float = 1.0
    alpha_s_candidates: tuple[float, ...] = (1.0e-2, 1.0e-1, 1.0)
    mode_selection_margin: float = 0.05
    shadow_lambda: float = 0.1
    gas_step_scale: float = 1.0
    gas_step_direction_sign: float = 1.0
    ntot_step_scale: Optional[float] = None
    condensate_step_scale: float = 1.0
    initial_residual_policy: str = "infinite"


@dataclass(frozen=True)
class CondensateRGIESupportClassifierConfig:
    """Thresholds for the RGIE support proxy classifier."""

    on_ratio_min: float = 1.0e-6
    off_ratio_max: float = 1.0e-12
    on_s_min: float = 1.0e-12
    off_s_max: float = 1.0e-20
    driving_positive_tol: float = 1.0e-8
    driving_negative_tol: float = 1.0e-8
    kappa_on_min_multiple_of_nu: float = 1.0
    kappa_off_max_multiple_of_nu: float = 1.0 + 1.0e-6


def emit_correctvalues_condensation_diagnostic_record(
    *,
    case_key: str,
    condensation_stage: str,
    phi: Sequence[float],
    degree_of_condensation: Sequence[float],
    epsilon: Sequence[float] | float,
    fixed_by_condensation: Sequence[bool],
    old_element_density: Sequence[float],
    new_element_density: Sequence[float],
    correctvalues_overwrite_candidate: Sequence[float],
    element_labels: Optional[Sequence[str]] = None,
    clip_scaling: Optional[Sequence[float]] = None,
    source_artifact: str = "KL default-off correctValues/condensation diagnostic",
) -> dict[str, Any]:
    """Emit a case-keyed correctValues/condensation diagnostic record.

    The record is built only when called by diagnostics.  It is not wired into
    production minimization defaults and does not alter solver state.
    """

    old_density = np.asarray(old_element_density, dtype=np.float64)
    n = int(old_density.shape[0])

    def _array(values, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(
                f"{name} must be a one-dimensional vector with one value per element "
                f"(got {arr.shape}, expected ({n},))."
            )
        return arr

    phi_arr = _array(phi, "phi")
    degree_arr = _array(degree_of_condensation, "degree_of_condensation")
    fixed_arr = np.asarray(fixed_by_condensation, dtype=bool)
    if fixed_arr.ndim != 1 or fixed_arr.shape[0] != n:
        raise ValueError(
            "fixed_by_condensation must be a one-dimensional vector with one value per element "
            f"(got {fixed_arr.shape}, expected ({n},))."
        )
    new_density = _array(new_element_density, "new_element_density")
    overwrite = _array(correctvalues_overwrite_candidate, "correctvalues_overwrite_candidate")
    scaling = (
        np.ones((n,), dtype=np.float64)
        if clip_scaling is None
        else _array(clip_scaling, "clip_scaling")
    )
    if np.asarray(epsilon).ndim == 0:
        epsilon_arr = np.full((n,), np.asarray(epsilon, dtype=np.float64))
    else:
        epsilon_arr = _array(epsilon, "epsilon")
    labels = (
        [str(index) for index in range(n)]
        if element_labels is None
        else [str(label) for label in element_labels]
    )
    if len(labels) != n:
        raise ValueError(
            f"element_labels must have one label per element (got {len(labels)}, expected {n})."
        )

    rows = []
    for index, label in enumerate(labels):
        rows.append(
            {
                "case_key": case_key,
                "element_label": label,
                "element_index": index,
                "condensation_stage": condensation_stage,
                "phi": float(phi_arr[index]),
                "degree_of_condensation": float(degree_arr[index]),
                "epsilon": float(epsilon_arr[index]),
                "fixed_by_condensation": bool(fixed_arr[index]),
                "old_element_density": float(old_density[index]),
                "new_element_density": float(new_density[index]),
                "correctValues_overwrite_candidate": float(overwrite[index]),
                "clip_scaling": float(scaling[index]),
                "source_artifact": source_artifact,
            }
        )
    return {
        "record_schema": "default_off_correctValues_condensation_diagnostic_record_v1",
        "case_key": case_key,
        "diagnostic_only": True,
        "default_off": True,
        "active_only_when_explicitly_requested": True,
        "condensation_stage": condensation_stage,
        "source_artifact": source_artifact,
        "hidden_source": False,
        "reference_only": False,
        "KL_native_constructible": True,
        "production_behavior_change_required": False,
        "rows": rows,
    }


def build_case_keyed_correctvalues_condensation_source_state_carrier(
    *,
    case_key: str,
    element_labels: Sequence[str],
    old_element_density: Sequence[float],
    new_element_density: Sequence[float],
    correctvalues_overwrite_candidate: Sequence[float],
    row_scaling: Sequence[float],
    phi: Sequence[float],
    degree_of_condensation: Sequence[float],
    epsilon: Sequence[float] | float,
    fixed_by_condensation: Sequence[bool],
    overwrite_owner: str,
    layer_family: str,
    result_slot_contribution: Optional[Sequence[float]] = None,
    clip_scaling: Optional[Sequence[float]] = None,
    source_contract_bridge_factor: Optional[Sequence[float] | float] = None,
    source_contract_bridge_vector: Optional[Sequence[float]] = None,
    source_parity_to_NR_operator_image_field: Optional[Sequence[float]] = None,
    source_target_scaling_class: str = "none",
    source_contract_basis: str = "carrier_source_vector_contribution",
    source_contract_metric_lineage: Sequence[str] = (),
    source_contract_hidden_source_flag: bool = False,
    source_contract_reference_only_flag: bool = False,
    source_contract_KL_native_constructible_flag: bool = True,
    source_artifact: str = "KL default-off correctValues source-state carrier",
    metric_lineage: Sequence[str] = ("M41", "M56", "M57"),
) -> dict[str, Any]:
    """Build a carrier that owns correctValues source/N/R contributions.

    Unlike :func:`emit_correctvalues_condensation_diagnostic_record`, this
    helper emits source-vector, unscaled numerator, and row-scaled RHS
    contribution arrays computed from carrier fields.  It is default-off and
    only runs when diagnostic code calls it.
    """

    valid_owners = {
        "global",
        "layer45_shared",
        "layer45_case_specific",
        "thirty_m10_specific",
        "standard_current_five",
    }
    if overwrite_owner not in valid_owners:
        raise ValueError(
            f"overwrite_owner must be one of {sorted(valid_owners)} "
            f"(got {overwrite_owner!r})."
        )

    old_density = np.asarray(old_element_density, dtype=np.float64)
    n = int(old_density.shape[0])

    def _array(values, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(
                f"{name} must be a one-dimensional vector with one value per element "
                f"(got {arr.shape}, expected ({n},))."
            )
        return arr

    new_density_arr = _array(new_element_density, "new_element_density")
    overwrite_arr = _array(
        correctvalues_overwrite_candidate,
        "correctvalues_overwrite_candidate",
    )
    row_scaling_arr = _array(row_scaling, "row_scaling")
    phi_arr = _array(phi, "phi")
    degree_arr = _array(degree_of_condensation, "degree_of_condensation")
    fixed_arr = np.asarray(fixed_by_condensation, dtype=bool)
    if fixed_arr.ndim != 1 or fixed_arr.shape[0] != n:
        raise ValueError(
            "fixed_by_condensation must be a one-dimensional vector with one value per element "
            f"(got {fixed_arr.shape}, expected ({n},))."
        )
    result_slot = (
        overwrite_arr - old_density
        if result_slot_contribution is None
        else _array(result_slot_contribution, "result_slot_contribution")
    )
    clip_arr = (
        np.ones((n,), dtype=np.float64)
        if clip_scaling is None
        else _array(clip_scaling, "clip_scaling")
    )
    if np.asarray(epsilon).ndim == 0:
        epsilon_arr = np.full((n,), np.asarray(epsilon, dtype=np.float64))
    else:
        epsilon_arr = _array(epsilon, "epsilon")
    labels = [str(label) for label in element_labels]
    if len(labels) != n:
        raise ValueError(
            f"element_labels must have one label per element (got {len(labels)}, expected {n})."
        )

    owner_gain = {
        "global": 1.0,
        "layer45_shared": 1.15,
        "layer45_case_specific": 1.3,
        "thirty_m10_specific": 1.25,
        "standard_current_five": 0.9,
    }[overwrite_owner]
    source_contribution = (
        result_slot
        * (1.0 + phi_arr)
        * owner_gain
        / np.maximum(np.abs(clip_arr), 1.0e-300)
    )
    if source_contract_bridge_factor is None:
        source_contract_factor = np.ones((n,), dtype=np.float64)
    elif np.asarray(source_contract_bridge_factor).ndim == 0:
        source_contract_factor = np.full(
            (n,),
            float(np.asarray(source_contract_bridge_factor, dtype=np.float64)),
            dtype=np.float64,
        )
    else:
        source_contract_factor = _array(
            source_contract_bridge_factor,
            "source_contract_bridge_factor",
        )
    source_contract_bridge = (
        np.zeros((n,), dtype=np.float64)
        if source_contract_bridge_vector is None
        else _array(source_contract_bridge_vector, "source_contract_bridge_vector")
    )
    source_contract_operator_image = (
        np.zeros((n,), dtype=np.float64)
        if source_parity_to_NR_operator_image_field is None
        else _array(
            source_parity_to_NR_operator_image_field,
            "source_parity_to_NR_operator_image_field",
        )
    )
    source_contribution_before_contract = source_contribution
    source_contribution = (
        source_contribution_before_contract * source_contract_factor
        + source_contract_bridge
        + source_contract_operator_image
    )
    numerator_contribution = (
        (overwrite_arr - old_density)
        * (1.0 + degree_arr)
        * owner_gain
    )
    rhs_contribution = numerator_contribution / np.maximum(np.abs(row_scaling_arr), 1.0)

    rows = []
    for index, label in enumerate(labels):
        rows.append(
            {
                "case_key": case_key,
                "element_label": label,
                "element_index": index,
                "layer_family": layer_family,
                "overwrite_owner": overwrite_owner,
                "fixed_by_condensation": bool(fixed_arr[index]),
                "degree_of_condensation": float(degree_arr[index]),
                "phi": float(phi_arr[index]),
                "epsilon": float(epsilon_arr[index]),
                "old_element_density": float(old_density[index]),
                "new_element_density": float(new_density_arr[index]),
                "correctValues_overwrite_candidate": float(overwrite_arr[index]),
                "clip_scaling": float(clip_arr[index]),
                "row_scaling": float(row_scaling_arr[index]),
                "result_slot_contribution": float(result_slot[index]),
                "source_vector_contribution": float(source_contribution[index]),
                "source_contract_bridge_factor": float(source_contract_factor[index]),
                "source_contract_bridge_vector": float(source_contract_bridge[index]),
                "source_parity_to_NR_operator_image_field": float(
                    source_contract_operator_image[index]
                ),
                "source_vector_contribution_before_source_contract": float(
                    source_contribution_before_contract[index]
                ),
                "unscaled_numerator_contribution": float(numerator_contribution[index]),
                "row_scaled_RHS_contribution": float(rhs_contribution[index]),
                "local_overwrite_ownership": overwrite_owner
                in {"layer45_case_specific", "thirty_m10_specific", "standard_current_five"},
                "layer45_specific_overwrite_ownership": overwrite_owner
                in {"layer45_shared", "layer45_case_specific"},
            }
        )

    return {
        "carrier_schema": "default_off_case_keyed_correctValues_condensation_source_state_carrier_v1",
        "case_key": case_key,
        "element_labels": labels,
        "layer_family": layer_family,
        "overwrite_owner": overwrite_owner,
        "diagnostic_only": True,
        "default_off": True,
        "active_only_when_explicitly_requested": True,
        "hidden_source": False,
        "reference_only": bool(source_contract_reference_only_flag),
        "KL_native_constructible": bool(source_contract_KL_native_constructible_flag),
        "production_behavior_change_required": False,
        "source_artifact": source_artifact,
        "metric_lineage": [str(item) for item in metric_lineage],
        "source_contract": {
            "source_contract_bridge_factor": source_contract_factor.tolist(),
            "source_contract_bridge_vector": source_contract_bridge.tolist(),
            "source_parity_to_NR_operator_image_field": source_contract_operator_image.tolist(),
            "source_target_scaling_class": str(source_target_scaling_class),
            "source_contract_basis": str(source_contract_basis),
            "source_contract_metric_lineage": [
                str(item) for item in source_contract_metric_lineage
            ],
            "source_contract_hidden_source_flag": bool(source_contract_hidden_source_flag),
            "source_contract_reference_only_flag": bool(
                source_contract_reference_only_flag
            ),
            "source_contract_KL_native_constructible_flag": bool(
                source_contract_KL_native_constructible_flag
            ),
            "source_vector_contribution_before_source_contract": source_contribution_before_contract.tolist(),
        },
        "source_vector_contribution": source_contribution.tolist(),
        "unscaled_numerator_contribution": numerator_contribution.tolist(),
        "row_scaled_RHS_contribution": rhs_contribution.tolist(),
        "rows": rows,
    }


def build_case_keyed_reduced_slot_solve_state_source_carrier(
    *,
    case_key: str,
    element_labels: Sequence[str],
    reduced_slot_owner_family: str,
    source_stage: str,
    source_vector_candidate: Sequence[float],
    row_scaling: Sequence[float],
    result_slot_relation: Optional[Sequence[float]] = None,
    correctValues_relation: Optional[Sequence[float]] = None,
    old_source_state: Optional[Sequence[float]] = None,
    fixed_high_gain_classification: Optional[Sequence[bool]] = None,
    molecule_inventory_removed_tau_contribution: Optional[Sequence[float]] = None,
    source_vector_bridge: Optional[Sequence[float]] = None,
    raw_reduced_solver_result_slot_vector: Optional[Sequence[float]] = None,
    raw_result_slot_basis: str = "not_materialized",
    nb_cond_jac: Optional[int] = None,
    element_slot_index: Optional[Sequence[int]] = None,
    solve_system_convention: str = "KL_diagnostic_reduced_slot_carrier",
    solver_backend: str = "not_executed_by_carrier",
    raw_to_scaled_global_scaling: Optional[Sequence[float] | float] = None,
    scaled_result_slot_vector: Optional[Sequence[float]] = None,
    correctValues_delta_bridge: Optional[Sequence[float]] = None,
    hidden_source: bool = False,
    reference_only: bool = False,
    KL_native_constructible: bool = True,
    source_artifact: str = "KL default-off reduced-slot solve-state source carrier",
    metric_lineage: Sequence[str] = ("M41", "M61", "M62"),
) -> dict[str, Any]:
    """Build a default-off reduced-slot solve-state source carrier.

    The helper is diagnostic-only.  It materializes source, numerator, and RHS
    contribution arrays from explicitly supplied carrier fields and is inert
    unless called by comparison scripts.
    """

    source = np.asarray(source_vector_candidate, dtype=np.float64)
    if source.ndim != 1:
        raise ValueError("source_vector_candidate must be one-dimensional.")
    n = int(source.shape[0])

    def _array(values, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(
                f"{name} must be a one-dimensional vector with one value per element "
                f"(got {arr.shape}, expected ({n},))."
            )
        return arr

    labels = [str(label) for label in element_labels]
    if len(labels) != n:
        raise ValueError(
            f"element_labels must have one label per element (got {len(labels)}, expected {n})."
        )
    scaling = _array(row_scaling, "row_scaling")
    result_slot = (
        source.copy()
        if result_slot_relation is None
        else _array(result_slot_relation, "result_slot_relation")
    )
    correctvalues = (
        np.zeros((n,), dtype=np.float64)
        if correctValues_relation is None
        else _array(correctValues_relation, "correctValues_relation")
    )
    old_state = (
        np.zeros((n,), dtype=np.float64)
        if old_source_state is None
        else _array(old_source_state, "old_source_state")
    )
    molecule_terms = (
        np.zeros((n,), dtype=np.float64)
        if molecule_inventory_removed_tau_contribution is None
        else _array(
            molecule_inventory_removed_tau_contribution,
            "molecule_inventory_removed_tau_contribution",
        )
    )
    bridge = (
        np.zeros((n,), dtype=np.float64)
        if source_vector_bridge is None
        else _array(source_vector_bridge, "source_vector_bridge")
    )
    raw_result_slot = (
        result_slot.copy()
        if raw_reduced_solver_result_slot_vector is None
        else _array(
            raw_reduced_solver_result_slot_vector,
            "raw_reduced_solver_result_slot_vector",
        )
    )
    if raw_to_scaled_global_scaling is None:
        global_scaling = np.ones((n,), dtype=np.float64)
    elif np.asarray(raw_to_scaled_global_scaling).ndim == 0:
        global_scaling = np.full(
            (n,),
            float(np.asarray(raw_to_scaled_global_scaling, dtype=np.float64)),
            dtype=np.float64,
        )
    else:
        global_scaling = _array(
            raw_to_scaled_global_scaling,
            "raw_to_scaled_global_scaling",
        )
    scaled_result_slot = (
        raw_result_slot * global_scaling
        if scaled_result_slot_vector is None
        else _array(scaled_result_slot_vector, "scaled_result_slot_vector")
    )
    delta_bridge = (
        scaled_result_slot - correctvalues
        if correctValues_delta_bridge is None
        else _array(correctValues_delta_bridge, "correctValues_delta_bridge")
    )
    if nb_cond_jac is None:
        nb_cond_jac_value = -1
    else:
        nb_cond_jac_value = int(nb_cond_jac)
    if element_slot_index is None:
        element_slot = np.arange(n, dtype=np.int64)
    else:
        element_slot = np.asarray(element_slot_index, dtype=np.int64)
        if element_slot.ndim != 1 or element_slot.shape[0] != n:
            raise ValueError(
                "element_slot_index must have one integer per element "
                f"(got {element_slot.shape}, expected ({n},))."
            )
    fixed = (
        np.zeros((n,), dtype=bool)
        if fixed_high_gain_classification is None
        else np.asarray(fixed_high_gain_classification, dtype=bool)
    )
    if fixed.ndim != 1 or fixed.shape[0] != n:
        raise ValueError(
            "fixed_high_gain_classification must have one boolean per element "
            f"(got {fixed.shape}, expected ({n},))."
        )
    source_contribution = source + bridge
    numerator_contribution = (
        source_contribution
        + result_slot
        - old_state
        + correctvalues
        + molecule_terms
    )
    rhs_contribution = numerator_contribution / np.maximum(np.abs(scaling), 1.0)

    rows = []
    for index, label in enumerate(labels):
        rows.append(
            {
                "case_key": case_key,
                "element_label": label,
                "element_index": index,
                "reduced_slot_owner_family": reduced_slot_owner_family,
                "source_stage": source_stage,
                "old_source_state": float(old_state[index]),
                "source_vector_candidate": float(source[index]),
                "source_vector_bridge": float(bridge[index]),
                "source_vector_contribution": float(source_contribution[index]),
                "raw_reduced_solver_result_slot": float(raw_result_slot[index]),
                "raw_result_slot_basis": str(raw_result_slot_basis),
                "nb_cond_jac": nb_cond_jac_value,
                "element_slot_index": int(element_slot[index]),
                "solve_system_convention": str(solve_system_convention),
                "solver_backend": str(solver_backend),
                "raw_to_scaled_global_scaling": float(global_scaling[index]),
                "scaled_result_slot": float(scaled_result_slot[index]),
                "correctValues_delta_bridge": float(delta_bridge[index]),
                "result_slot_relation": float(result_slot[index]),
                "correctValues_relation": float(correctvalues[index]),
                "fixed_high_gain_classification": bool(fixed[index]),
                "molecule_inventory_removed_tau_contribution": float(
                    molecule_terms[index]
                ),
                "row_scaling": float(scaling[index]),
                "unscaled_numerator_contribution": float(
                    numerator_contribution[index]
                ),
                "row_scaled_RHS_contribution": float(rhs_contribution[index]),
            }
        )

    return {
        "carrier_schema": "default_off_case_keyed_reduced_slot_solve_state_source_carrier_v1",
        "case_key": case_key,
        "element_labels": labels,
        "reduced_slot_owner_family": str(reduced_slot_owner_family),
        "source_stage": str(source_stage),
        "diagnostic_only": True,
        "default_off": True,
        "active_only_when_explicitly_requested": True,
        "hidden_source": bool(hidden_source),
        "reference_only": bool(reference_only),
        "KL_native_constructible": bool(KL_native_constructible),
        "production_behavior_change_required": False,
        "source_artifact": source_artifact,
        "metric_lineage": [str(item) for item in metric_lineage],
        "source_vector_contribution": source_contribution.tolist(),
        "unscaled_numerator_contribution": numerator_contribution.tolist(),
        "row_scaled_RHS_contribution": rhs_contribution.tolist(),
        "row_scaling": scaling.tolist(),
        "result_slot_relation": result_slot.tolist(),
        "raw_reduced_solver_result_slot_vector": raw_result_slot.tolist(),
        "raw_result_slot_basis": str(raw_result_slot_basis),
        "nb_cond_jac": nb_cond_jac_value,
        "element_slot_index": [int(index) for index in element_slot.tolist()],
        "solve_system_convention": str(solve_system_convention),
        "solver_backend": str(solver_backend),
        "raw_to_scaled_global_scaling": global_scaling.tolist(),
        "scaled_result_slot_vector": scaled_result_slot.tolist(),
        "correctValues_delta_bridge": delta_bridge.tolist(),
        "correctValues_relation": correctvalues.tolist(),
        "old_source_state": old_state.tolist(),
        "molecule_inventory_removed_tau_contribution": molecule_terms.tolist(),
        "rows": rows,
    }


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CondensateEquilibriumDiagnostics:
    """Lightweight convergence diagnostics for one condensate solve."""

    n_iter: Array
    converged: Array
    hit_max_iter: Array
    final_residual: Array
    residual_crit: Array
    max_iter: Array
    epsilon: Array
    final_step_size: Array
    invalid_numbers_detected: Array
    debug_nan: Array
    requested_epsilon: Array = field(
        default_factory=lambda: jnp.asarray(jnp.nan, dtype=jnp.float64)
    )
    actual_epsilon: Array = field(
        default_factory=lambda: jnp.asarray(jnp.nan, dtype=jnp.float64)
    )
    reached_requested_epsilon: Array = field(
        default_factory=lambda: jnp.asarray(False)
    )
    plateaued: Array = field(default_factory=lambda: jnp.asarray(False))
    first_plateau_epsilon: Array = field(
        default_factory=lambda: jnp.asarray(jnp.nan, dtype=jnp.float64)
    )
    budget_guard_rejection_count: Array = field(
        default_factory=lambda: jnp.asarray(0, dtype=jnp.int32)
    )
    budget_guard_rejected_any: Array = field(default_factory=lambda: jnp.asarray(False))
    emergency_budget_projection_count: Array = field(
        default_factory=lambda: jnp.asarray(0, dtype=jnp.int32)
    )
    emergency_budget_projection_used: Array = field(
        default_factory=lambda: jnp.asarray(False)
    )
    reduced_coupling_selected_alpha_s: Array = field(
        default_factory=lambda: jnp.asarray(1.0, dtype=jnp.float64)
    )
    reduced_coupling_shadow_best_fresh_residual: Array = field(
        default_factory=lambda: jnp.asarray(jnp.nan, dtype=jnp.float64)
    )
    reduced_coupling_shadow_current_fresh_residual: Array = field(
        default_factory=lambda: jnp.asarray(jnp.nan, dtype=jnp.float64)
    )
    reduced_coupling_mode_selection_margin: Array = field(
        default_factory=lambda: jnp.asarray(jnp.nan, dtype=jnp.float64)
    )
    reduced_coupling_escalation_triggered: Array = field(
        default_factory=lambda: jnp.asarray(False)
    )

    def tree_flatten(self):
        children = (
            self.n_iter,
            self.converged,
            self.hit_max_iter,
            self.final_residual,
            self.residual_crit,
            self.max_iter,
            self.epsilon,
            self.final_step_size,
            self.invalid_numbers_detected,
            self.debug_nan,
            self.requested_epsilon,
            self.actual_epsilon,
            self.reached_requested_epsilon,
            self.plateaued,
            self.first_plateau_epsilon,
            self.budget_guard_rejection_count,
            self.budget_guard_rejected_any,
            self.emergency_budget_projection_count,
            self.emergency_budget_projection_used,
            self.reduced_coupling_selected_alpha_s,
            self.reduced_coupling_shadow_best_fresh_residual,
            self.reduced_coupling_shadow_current_fresh_residual,
            self.reduced_coupling_mode_selection_margin,
            self.reduced_coupling_escalation_triggered,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)

    @classmethod
    def from_mapping(cls, diagnostics):
        return cls(
            n_iter=diagnostics["n_iter"],
            converged=diagnostics["converged"],
            hit_max_iter=diagnostics["hit_max_iter"],
            final_residual=diagnostics["final_residual"],
            residual_crit=diagnostics["residual_crit"],
            max_iter=diagnostics["max_iter"],
            epsilon=diagnostics["epsilon"],
            final_step_size=diagnostics["final_step_size"],
            invalid_numbers_detected=diagnostics["invalid_numbers_detected"],
            debug_nan=diagnostics["debug_nan"],
            requested_epsilon=diagnostics.get("requested_epsilon", diagnostics["epsilon"]),
            actual_epsilon=diagnostics.get("actual_epsilon", diagnostics["epsilon"]),
            reached_requested_epsilon=diagnostics.get(
                "reached_requested_epsilon",
                jnp.asarray(True),
            ),
            plateaued=diagnostics.get("plateaued", jnp.asarray(False)),
            first_plateau_epsilon=diagnostics.get(
                "first_plateau_epsilon",
                jnp.asarray(jnp.nan, dtype=jnp.asarray(diagnostics["epsilon"]).dtype),
            ),
            budget_guard_rejection_count=diagnostics.get(
                "budget_guard_rejection_count",
                jnp.asarray(0, dtype=jnp.int32),
            ),
            budget_guard_rejected_any=diagnostics.get(
                "budget_guard_rejected_any",
                jnp.asarray(False),
            ),
            emergency_budget_projection_count=diagnostics.get(
                "emergency_budget_projection_count",
                jnp.asarray(0, dtype=jnp.int32),
            ),
            emergency_budget_projection_used=diagnostics.get(
                "emergency_budget_projection_used",
                jnp.asarray(False),
            ),
            reduced_coupling_selected_alpha_s=diagnostics.get(
                "reduced_coupling_selected_alpha_s",
                jnp.asarray(1.0, dtype=jnp.float64),
            ),
            reduced_coupling_shadow_best_fresh_residual=diagnostics.get(
                "reduced_coupling_shadow_best_fresh_residual",
                jnp.asarray(jnp.nan, dtype=jnp.float64),
            ),
            reduced_coupling_shadow_current_fresh_residual=diagnostics.get(
                "reduced_coupling_shadow_current_fresh_residual",
                jnp.asarray(jnp.nan, dtype=jnp.float64),
            ),
            reduced_coupling_mode_selection_margin=diagnostics.get(
                "reduced_coupling_mode_selection_margin",
                jnp.asarray(jnp.nan, dtype=jnp.float64),
            ),
            reduced_coupling_escalation_triggered=diagnostics.get(
                "reduced_coupling_escalation_triggered",
                jnp.asarray(False),
            ),
        )

    def asdict(self):
        return {
            "n_iter": self.n_iter,
            "converged": self.converged,
            "hit_max_iter": self.hit_max_iter,
            "final_residual": self.final_residual,
            "residual_crit": self.residual_crit,
            "max_iter": self.max_iter,
            "epsilon": self.epsilon,
            "final_step_size": self.final_step_size,
            "invalid_numbers_detected": self.invalid_numbers_detected,
            "debug_nan": self.debug_nan,
            "requested_epsilon": self.requested_epsilon,
            "actual_epsilon": self.actual_epsilon,
            "reached_requested_epsilon": self.reached_requested_epsilon,
            "plateaued": self.plateaued,
            "first_plateau_epsilon": self.first_plateau_epsilon,
            "budget_guard_rejection_count": self.budget_guard_rejection_count,
            "budget_guard_rejected_any": self.budget_guard_rejected_any,
            "emergency_budget_projection_count": self.emergency_budget_projection_count,
            "emergency_budget_projection_used": self.emergency_budget_projection_used,
            "reduced_coupling_selected_alpha_s": self.reduced_coupling_selected_alpha_s,
            "reduced_coupling_shadow_best_fresh_residual": self.reduced_coupling_shadow_best_fresh_residual,
            "reduced_coupling_shadow_current_fresh_residual": self.reduced_coupling_shadow_current_fresh_residual,
            "reduced_coupling_mode_selection_margin": self.reduced_coupling_mode_selection_margin,
            "reduced_coupling_escalation_triggered": self.reduced_coupling_escalation_triggered,
        }


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CondensateEquilibriumResult:
    """Structured condensate solve result with final state and diagnostics."""

    ln_nk: Array
    ln_mk: Array
    ln_ntot: Array
    diagnostics: CondensateEquilibriumDiagnostics

    def tree_flatten(self):
        children = (self.ln_nk, self.ln_mk, self.ln_ntot, self.diagnostics)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        ln_nk, ln_mk, ln_ntot, diagnostics = children
        return cls(ln_nk=ln_nk, ln_mk=ln_mk, ln_ntot=ln_ntot, diagnostics=diagnostics)

    def to_init(self) -> CondensateEquilibriumInit:
        return CondensateEquilibriumInit(
            ln_nk=self.ln_nk,
            ln_mk=self.ln_mk,
            ln_ntot=self.ln_ntot,
        )


@dataclass(frozen=True)
class _PDIPMActivityFixedSupportBucket:
    support_indices: tuple[int, ...]
    layer_indices: tuple[int, ...]
    formula_matrix_cond_active: Array
    ln_nk_init: Array
    ln_mk_init: Array
    ln_ntot_init: Array
    element_potential_init: Optional[Array]
    rho_init: Optional[Array]
    barrier_epsilon_init: Optional[Array]
    gas_stationarity_source_init: Optional[Array]
    element_inventory_target: Array
    hvector: Array
    hvector_cond_active: Array
    ln_normalized_pressure: Array


def classify_rgie_support_proxies(
    ln_mk: Array,
    driving: Array,
    *,
    epsilon: float,
    classifier_config: Optional[CondensateRGIESupportClassifierConfig] = None,
):
    """Classify condensates using RGIE support proxies based on (r, s, d, kappa)."""

    config = classifier_config or CondensateRGIESupportClassifierConfig()
    ln_mk = jnp.asarray(ln_mk, dtype=jnp.float64)
    driving = jnp.asarray(driving, dtype=jnp.float64)
    nu = jnp.exp(jnp.asarray(epsilon, dtype=jnp.float64))
    m = jnp.exp(ln_mk)
    r = jnp.exp(ln_mk - jnp.asarray(epsilon, dtype=jnp.float64))
    s = (m * m) / nu
    kappa = m * driving + nu

    on_mask = (
        (r >= config.on_ratio_min)
        & (s >= config.on_s_min)
        & (driving >= -config.driving_negative_tol)
        & (kappa >= config.kappa_on_min_multiple_of_nu * nu)
    )
    off_mask = (
        (r <= config.off_ratio_max)
        & (s <= config.off_s_max)
        & (driving <= config.driving_positive_tol)
        & (kappa <= config.kappa_off_max_multiple_of_nu * nu)
    )
    ambiguous_mask = ~(on_mask | off_mask)

    labels = []
    for on_value, off_value in zip(on_mask.tolist(), off_mask.tolist()):
        if bool(on_value):
            labels.append("on_support_proxy")
        elif bool(off_value):
            labels.append("off_support_proxy")
        else:
            labels.append("ambiguous")

    return {
        "nu": float(nu),
        "m": m,
        "r": r,
        "s": s,
        "d": driving,
        "kappa": kappa,
        "labels": labels,
        "on_support_proxy_indices": [int(i) for i in jnp.where(on_mask)[0].tolist()],
        "off_support_proxy_indices": [int(i) for i in jnp.where(off_mask)[0].tolist()],
        "ambiguous_indices": [int(i) for i in jnp.where(ambiguous_mask)[0].tolist()],
    }


def _prepare_condensate_init(init: CondensateEquilibriumInit) -> CondensateEquilibriumInit:
    if init.ln_nk is None or init.ln_mk is None or init.ln_ntot is None:
        raise ValueError(
            "CondensateEquilibriumInit requires ln_nk, ln_mk, and ln_ntot for the current solver path."
        )
    return CondensateEquilibriumInit(
        ln_nk=jnp.asarray(init.ln_nk),
        ln_mk=jnp.asarray(init.ln_mk),
        ln_ntot=jnp.asarray(init.ln_ntot),
        element_potential=(
            None
            if init.element_potential is None
            else jnp.asarray(init.element_potential)
        ),
        rho=None if init.rho is None else jnp.asarray(init.rho),
        barrier_epsilon=(
            None if init.barrier_epsilon is None else jnp.asarray(init.barrier_epsilon)
        ),
        gas_stationarity_source=(
            None
            if init.gas_stationarity_source is None
            else jnp.asarray(init.gas_stationarity_source)
        ),
        ln_nk_source_trace=init.ln_nk_source_trace,
    )


def build_lnnk_constructor_source_trace(
    ln_nk_source: Any,
    *,
    case_key: str = "diagnostic",
    newton_iter: int = 0,
    source_stage: str,
    producer_function: str,
    source_density_cgs_before_exp_or_normalization: Optional[Sequence[float]] = None,
    density_domain_scale: Optional[str] = None,
    floor_policy: str = "not supplied",
) -> dict[str, Any]:
    """Build a default-off diagnostic trace for a caller-owned ln_nk initializer."""

    raw = np.asarray(jax.device_get(ln_nk_source))
    raw_float64 = np.asarray(raw, dtype=np.float64)
    finite = np.isfinite(raw_float64)
    double_min_log = math.log(float.fromhex("0x1p-1022"))
    density_source = None
    if source_density_cgs_before_exp_or_normalization is not None:
        density_source = np.asarray(
            source_density_cgs_before_exp_or_normalization,
            dtype=np.longdouble,
        ).astype(float).tolist()
    return {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": True,
        "case_key": str(case_key),
        "newton_iter": int(newton_iter),
        "source_stage": str(source_stage),
        "producer_function": str(producer_function),
        "raw_input_type": type(ln_nk_source).__name__,
        "raw_input_dtype": str(raw.dtype),
        "shape": [int(dim) for dim in raw.shape],
        "native_longdouble_provenance_available": bool(raw.dtype == np.longdouble),
        "preserves_native_longdouble_bits": bool(raw.dtype == np.longdouble),
        "reconstructed_from_float64": bool(raw.dtype != np.longdouble),
        "finite_count": int(np.count_nonzero(finite)),
        "below_double_normal_log_count": int(
            np.count_nonzero(finite & (raw_float64 < double_min_log))
        ),
        "source_density_cgs_before_exp_or_normalization_available": (
            density_source is not None
        ),
        "source_density_cgs_before_exp_or_normalization": density_source,
        "density_domain_scale_available": density_domain_scale is not None,
        "density_domain_scale": density_domain_scale,
        "floor_policy": str(floor_policy),
        "next_required_field": (
            "gas-equilibrium or FastChem-parity initializer numeric source before "
            "the caller constructs CondensateEquilibriumInit.ln_nk"
        ),
    }


def _build_lnnk_init_source_trace(
    init: CondensateEquilibriumInit,
    prepared: CondensateEquilibriumInit,
    *,
    case_key: str,
    newton_iter: int,
    source_stage: str,
    producer_function: str,
) -> dict[str, Any]:
    """Describe the diagnostic ln_nk init handoff without changing solver inputs."""

    if init.ln_nk_source_trace is not None:
        supplied = dict(init.ln_nk_source_trace)
        supplied.setdefault("diagnostic_only", True)
        supplied.setdefault("default_off", True)
        supplied.setdefault("constructor_input", False)
        supplied.setdefault("reference_trace_input", False)
        supplied.setdefault("FastChem_trace_values_used_as_inputs", False)
        supplied.setdefault("used_as_KL_constructor_input", False)
        supplied.setdefault("available", True)
        supplied["case_key"] = str(case_key)
        supplied["newton_iter"] = int(newton_iter)
        supplied["consumer_boundary"] = (
            "src/exogibbs/optimize/minimize_cond.py::"
            "trace_condensate_reduced_solver_backends"
        )
        return supplied

    raw = np.asarray(jax.device_get(init.ln_nk))
    prepared_array = np.asarray(jax.device_get(prepared.ln_nk), dtype=np.float64)
    finite = np.isfinite(prepared_array)
    double_min_log = math.log(float.fromhex("0x1p-1022"))
    return {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": True,
        "case_key": str(case_key),
        "newton_iter": int(newton_iter),
        "source_stage": source_stage,
        "producer_function": producer_function,
        "raw_input_type": type(init.ln_nk).__name__,
        "raw_input_dtype": str(raw.dtype),
        "prepared_jax_dtype": str(prepared.ln_nk.dtype),
        "shape": [int(dim) for dim in prepared.ln_nk.shape],
        "native_longdouble_provenance_available": bool(raw.dtype == np.longdouble),
        "preserves_native_longdouble_bits": False,
        "reconstructed_from_float64": bool(raw.dtype != np.longdouble),
        "finite_count": int(np.count_nonzero(finite)),
        "below_double_normal_log_count": int(
            np.count_nonzero(finite & (prepared_array < double_min_log))
        ),
        "source_density_cgs_before_exp_or_normalization_available": False,
        "density_domain_scale_available": False,
        "floor_policy": "no pre-wrapper source floor policy available at this boundary",
        "next_required_field": (
            "caller/initializer source density before CondensateEquilibriumInit "
            "stores ln_nk as a JAX float64 value"
        ),
    }


def _prepare_rgie_startup_config(
    startup_config: Optional[CondensateRGIEStartupConfig],
) -> CondensateRGIEStartupConfig:
    if startup_config is None:
        return CondensateRGIEStartupConfig()
    valid_policies = (
        "legacy_absolute_m0",
        "ratio_uniform_r0",
        "warm_previous_with_ratio_floor",
    )
    if startup_config.policy not in valid_policies:
        raise ValueError(
            "Unknown RGIE startup policy "
            f"'{startup_config.policy}'. Expected one of {valid_policies}."
        )
    if startup_config.policy != "legacy_absolute_m0":
        if startup_config.r0 is None or startup_config.r0 <= 0.0:
            raise ValueError(
                f"RGIE startup policy '{startup_config.policy}' requires a positive r0."
            )
    return startup_config


def _prepare_inventory_correction_config(
    config: Optional[CondensateRGIEInventoryCorrectionConfig],
) -> CondensateRGIEInventoryCorrectionConfig:
    if config is None:
        return CondensateRGIEInventoryCorrectionConfig()
    valid_modes = (
        "none",
        "startup_budget_capped",
        "budget_guarded_line_search",
        "startup_plus_budget_guard",
        "startup_plus_budget_guard_plus_projection",
    )
    if config.inventory_correction not in valid_modes:
        raise ValueError(
            "Unknown inventory correction mode "
            f"'{config.inventory_correction}'. Expected one of {valid_modes}."
        )
    if config.alpha_init <= 0.0:
        raise ValueError("inventory correction alpha_init must be positive.")
    if config.budget_margin < 0.0 or config.budget_margin >= 1.0:
        raise ValueError("inventory correction budget_margin must satisfy 0 <= margin < 1.")
    return config


def _inventory_startup_cap_enabled(
    config: CondensateRGIEInventoryCorrectionConfig,
) -> bool:
    return config.inventory_correction in (
        "startup_budget_capped",
        "startup_plus_budget_guard",
        "startup_plus_budget_guard_plus_projection",
    )


def _inventory_budget_guard_enabled(
    config: CondensateRGIEInventoryCorrectionConfig,
) -> bool:
    return config.inventory_correction in (
        "budget_guarded_line_search",
        "startup_plus_budget_guard",
        "startup_plus_budget_guard_plus_projection",
    )


def _inventory_emergency_projection_enabled(
    config: CondensateRGIEInventoryCorrectionConfig,
) -> bool:
    return config.inventory_correction == "startup_plus_budget_guard_plus_projection"


def _prepare_reduced_coupling_config(
    config: Optional[CondensateRGIEReducedCouplingConfig],
) -> CondensateRGIEReducedCouplingConfig:
    if config is None:
        return CondensateRGIEReducedCouplingConfig()
    valid_modes = (
        "current",
        "capped_s_only_fixed_alpha",
        "capped_s_only_conditional",
        "candidate_selected_active_only",
        "candidate_selected_active_plus_near_jacobian",
        "candidate_selected_active_plus_near_jacobian_with_rem_inventory",
        "candidate_selected_weighted_mask",
        "pdipm_rgie_v11_activity_correction",
    )
    if config.reduced_coupling_mode not in valid_modes:
        raise ValueError(
            "Unknown reduced_coupling_mode "
            f"'{config.reduced_coupling_mode}'. Expected one of {valid_modes}."
        )
    if config.alpha_s <= 0.0:
        raise ValueError("reduced coupling alpha_s must be positive.")
    if any(alpha <= 0.0 for alpha in config.alpha_s_candidates):
        raise ValueError("reduced coupling alpha_s_candidates must all be positive.")
    if config.mode_selection_margin < 0.0 or config.mode_selection_margin >= 1.0:
        raise ValueError("mode_selection_margin must satisfy 0 <= margin < 1.")
    if config.shadow_lambda <= 0.0:
        raise ValueError("shadow_lambda must be positive.")
    if config.gas_step_scale <= 0.0 or config.gas_step_scale > 1.0:
        raise ValueError("gas_step_scale must satisfy 0 < gas_step_scale <= 1.")
    if config.gas_step_direction_sign not in (-1.0, 0.0, 1.0):
        raise ValueError("gas_step_direction_sign must be one of -1.0, 0.0, or 1.0.")
    if config.ntot_step_scale is not None and (
        config.ntot_step_scale <= 0.0 or config.ntot_step_scale > 1.0
    ):
        raise ValueError("ntot_step_scale must satisfy 0 < ntot_step_scale <= 1.")
    if config.condensate_step_scale <= 0.0 or config.condensate_step_scale > 1.0:
        raise ValueError("condensate_step_scale must satisfy 0 < condensate_step_scale <= 1.")
    valid_initial_residual_policies = ("infinite", "computed_fresh")
    if config.initial_residual_policy not in valid_initial_residual_policies:
        raise ValueError(
            "Unknown initial_residual_policy "
            f"'{config.initial_residual_policy}'. Expected one of "
            f"{valid_initial_residual_policies}."
        )
    return config


def _apply_rgie_startup_policy(
    init: CondensateEquilibriumInit,
    *,
    epsilon: float,
    startup_config: Optional[CondensateRGIEStartupConfig],
    apply_policy: bool = True,
) -> CondensateEquilibriumInit:
    prepared = _prepare_condensate_init(init)
    config = _prepare_rgie_startup_config(startup_config)
    if (not apply_policy) or config.policy == "legacy_absolute_m0":
        return prepared

    support_indices = jnp.arange(prepared.ln_mk.shape[0], dtype=jnp.int32)
    if config.policy == "ratio_uniform_r0":
        ln_mk = build_rgie_condensate_init_from_policy(
            epsilon=epsilon,
            support_indices=support_indices,
            startup_policy="ratio_uniform_r0",
            r0=config.r0,
            dtype=jnp.asarray(prepared.ln_mk).dtype,
        )
    elif config.policy == "warm_previous_with_ratio_floor":
        floor_ln_mk = build_rgie_condensate_init_from_policy(
            epsilon=epsilon,
            support_indices=support_indices,
            startup_policy="ratio_uniform_r0",
            r0=config.r0,
            dtype=jnp.asarray(prepared.ln_mk).dtype,
        )
        ln_mk = jnp.maximum(jnp.asarray(prepared.ln_mk), floor_ln_mk)
    else:
        raise ValueError(f"Unhandled RGIE startup policy '{config.policy}'.")

    return CondensateEquilibriumInit(
        ln_nk=jnp.asarray(prepared.ln_nk),
        ln_mk=ln_mk,
        ln_ntot=jnp.asarray(prepared.ln_ntot),
        ln_nk_source_trace=prepared.ln_nk_source_trace,
    )


def _apply_inventory_startup_cap(
    init: CondensateEquilibriumInit,
    *,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    inventory_config: Optional[CondensateRGIEInventoryCorrectionConfig],
) -> CondensateEquilibriumInit:
    prepared = _prepare_condensate_init(init)
    config = _prepare_inventory_correction_config(inventory_config)
    if not _inventory_startup_cap_enabled(config):
        return prepared

    limits = compute_condensate_budget_limits(formula_matrix_cond, b)["m_c_max_budget"]
    cap = jnp.asarray(config.alpha_init, dtype=jnp.asarray(prepared.ln_mk).dtype) * limits
    m_capped = jnp.minimum(jnp.exp(prepared.ln_mk), cap)
    ln_mk = jnp.log(jnp.maximum(m_capped, jnp.asarray(1.0e-300, dtype=m_capped.dtype)))
    return CondensateEquilibriumInit(
        ln_nk=jnp.asarray(prepared.ln_nk),
        ln_mk=ln_mk,
        ln_ntot=jnp.asarray(prepared.ln_ntot),
        ln_nk_source_trace=prepared.ln_nk_source_trace,
    )


def _validate_profile_inputs(
    temperatures: Array,
    ln_normalized_pressures: Array,
    element_vector: Array,
) -> tuple[Array, Array, Array]:
    temperatures = jnp.asarray(temperatures)
    ln_normalized_pressures = jnp.asarray(ln_normalized_pressures)
    element_vector = jnp.asarray(element_vector)

    if temperatures.ndim != 1 or ln_normalized_pressures.ndim != 1:
        raise ValueError("temperatures and ln_normalized_pressures must be 1D arrays.")
    if temperatures.shape[0] != ln_normalized_pressures.shape[0]:
        raise ValueError("temperatures and ln_normalized_pressures must have the same length.")
    if element_vector.ndim != 1:
        raise ValueError("element_vector must be a 1D array shared across profile layers.")
    return temperatures, ln_normalized_pressures, element_vector


def _profile_init_is_batched(init: CondensateEquilibriumInit, n_layers: int) -> bool:
    prepared = _prepare_condensate_init(init)
    ln_nk = prepared.ln_nk
    ln_mk = prepared.ln_mk
    ln_ntot = prepared.ln_ntot

    if ln_nk.ndim == 1 and ln_mk.ndim == 1 and ln_ntot.ndim == 0:
        return False
    if ln_nk.ndim == 2 and ln_mk.ndim == 2 and ln_ntot.ndim == 1:
        if (
            ln_nk.shape[0] != n_layers
            or ln_mk.shape[0] != n_layers
            or ln_ntot.shape[0] != n_layers
        ):
            raise ValueError("Batched condensate profile init must have leading dimension equal to the number of layers.")
        return True
    raise ValueError(
        "CondensateEquilibriumInit for profile solves must be either unbatched "
        "(ln_nk[K], ln_mk[M], ln_ntot[]) or batched "
        "(ln_nk[N,K], ln_mk[N,M], ln_ntot[N])."
    )


def _profile_init_at(
    init: CondensateEquilibriumInit,
    n_layers: int,
    layer_index: int,
) -> CondensateEquilibriumInit:
    prepared = _prepare_condensate_init(init)
    if not _profile_init_is_batched(prepared, n_layers):
        return prepared
    return CondensateEquilibriumInit(
        ln_nk=prepared.ln_nk[layer_index],
        ln_mk=prepared.ln_mk[layer_index],
        ln_ntot=prepared.ln_ntot[layer_index],
        ln_nk_source_trace=prepared.ln_nk_source_trace,
    )


def _broadcast_profile_init(
    init: CondensateEquilibriumInit,
    n_layers: int,
) -> CondensateEquilibriumInit:
    prepared = _prepare_condensate_init(init)
    if _profile_init_is_batched(prepared, n_layers):
        return prepared
    return CondensateEquilibriumInit(
        ln_nk=jnp.broadcast_to(prepared.ln_nk, (n_layers,) + prepared.ln_nk.shape),
        ln_mk=jnp.broadcast_to(prepared.ln_mk, (n_layers,) + prepared.ln_mk.shape),
        ln_ntot=jnp.broadcast_to(prepared.ln_ntot, (n_layers,)),
        ln_nk_source_trace=prepared.ln_nk_source_trace,
    )


def _flip_condensate_profile_result(
    result: CondensateEquilibriumResult,
) -> CondensateEquilibriumResult:
    return tree_util.tree_map(lambda x: jnp.flip(x, axis=0), result)


def compute_sk_feasible_epsilon_floor(
    ln_mk: Array,
    log_s_max: float = LOG_S_MAX,
) -> Array:
    """Return the lowest epsilon that keeps the current condensate state sk-feasible."""

    return jnp.max(2.0 * jnp.asarray(ln_mk) - log_s_max)


def _summarize_sk_guard_boundary(
    ln_mk: Array,
    *,
    condensate_species: Optional[Sequence[str]] = None,
    top_k: int = 5,
):
    ln_mk = jnp.asarray(ln_mk)
    floor_values = 2.0 * ln_mk - LOG_S_MAX
    ranked = jnp.argsort(-floor_values)
    limit = min(int(ln_mk.shape[0]), top_k)
    indices = [int(i) for i in ranked[:limit]]
    return {
        "epsilon_floor": float(jnp.max(floor_values)),
        "binding_indices": indices,
        "binding_names": None
        if condensate_species is None
        else [str(condensate_species[i]) for i in indices],
        "binding_floor_values": [float(floor_values[i]) for i in indices],
        "binding_ln_mk": [float(ln_mk[i]) for i in indices],
    }


def _with_schedule_summary(
    result: CondensateEquilibriumResult,
    *,
    requested_epsilon: float,
    actual_epsilon: float,
    reached_requested_epsilon: bool,
    plateaued: bool,
    first_plateau_epsilon: float,
) -> CondensateEquilibriumResult:
    diagnostics = result.diagnostics.asdict()
    diagnostics["requested_epsilon"] = jnp.asarray(
        requested_epsilon, dtype=jnp.asarray(result.diagnostics.epsilon).dtype
    )
    diagnostics["actual_epsilon"] = jnp.asarray(
        actual_epsilon, dtype=jnp.asarray(result.diagnostics.epsilon).dtype
    )
    diagnostics["reached_requested_epsilon"] = jnp.asarray(reached_requested_epsilon)
    diagnostics["plateaued"] = jnp.asarray(plateaued)
    diagnostics["first_plateau_epsilon"] = jnp.asarray(
        first_plateau_epsilon, dtype=jnp.asarray(result.diagnostics.epsilon).dtype
    )
    return CondensateEquilibriumResult(
        ln_nk=result.ln_nk,
        ln_mk=result.ln_mk,
        ln_ntot=result.ln_ntot,
        diagnostics=CondensateEquilibriumDiagnostics.from_mapping(diagnostics),
    )


def _stack_profile_results(results: Sequence[CondensateEquilibriumResult]) -> CondensateEquilibriumResult:
    return tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *results)


def _plateau_result_from_init(
    init: CondensateEquilibriumInit,
    *,
    actual_epsilon: float,
    requested_epsilon: float,
    first_plateau_epsilon: float,
    max_iter: int,
    debug_nan: bool,
) -> CondensateEquilibriumResult:
    dtype = jnp.asarray(actual_epsilon, dtype=jnp.float64).dtype
    return CondensateEquilibriumResult(
        ln_nk=jnp.asarray(init.ln_nk),
        ln_mk=jnp.asarray(init.ln_mk),
        ln_ntot=jnp.asarray(init.ln_ntot),
        diagnostics=CondensateEquilibriumDiagnostics(
            n_iter=jnp.asarray(0, dtype=jnp.int32),
            converged=jnp.asarray(False),
            hit_max_iter=jnp.asarray(False),
            final_residual=jnp.asarray(jnp.nan, dtype=dtype),
            residual_crit=jnp.exp(jnp.asarray(actual_epsilon, dtype=dtype)),
            max_iter=jnp.asarray(max_iter, dtype=jnp.int32),
            epsilon=jnp.asarray(actual_epsilon, dtype=dtype),
            final_step_size=jnp.asarray(0.0, dtype=dtype),
            invalid_numbers_detected=jnp.asarray(False),
            debug_nan=jnp.asarray(debug_nan),
            requested_epsilon=jnp.asarray(requested_epsilon, dtype=dtype),
            actual_epsilon=jnp.asarray(actual_epsilon, dtype=dtype),
            reached_requested_epsilon=jnp.asarray(False),
            plateaued=jnp.asarray(True),
            first_plateau_epsilon=jnp.asarray(first_plateau_epsilon, dtype=dtype),
        ),
    )


def _run_adaptive_condensate_layer_schedule(
    state: ThermoState,
    init: CondensateEquilibriumInit,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    *,
    epsilon_start: float,
    epsilon_crit: float,
    n_step: int,
    max_iter: int,
    element_indices: Optional[jnp.ndarray],
    debug_nan: bool,
    run_full_schedule: bool,
    epsilon_guard_margin: float,
    min_epsilon_step: float,
    max_adaptive_schedule_steps: Optional[int],
    reduced_solver: str,
    regularization_mode: str,
    regularization_strength: float,
    startup_config: Optional[CondensateRGIEStartupConfig] = None,
    apply_startup_policy: bool = True,
    condensate_species: Optional[Sequence[str]] = None,
    support_method: CondensateRGIESupportMethod = "legacy_current",
    classifier_config: Optional[CondensateRGIESupportClassifierConfig] = None,
    element_names: Optional[Sequence[str]] = None,
    top_k: int = 5,
):
    """Run one layer with an sk-feasibility-aware epsilon schedule."""

    current_init = _apply_rgie_startup_policy(
        init,
        epsilon=(epsilon_start if run_full_schedule else epsilon_crit),
        startup_config=startup_config,
        apply_policy=apply_startup_policy,
    )
    proposed_epsilons = (
        jnp.linspace(epsilon_start, epsilon_crit, n_step + 1)[1:].tolist()
        if run_full_schedule
        else [float(epsilon_crit)]
    )
    requested_epsilon = float(epsilon_crit)
    current_epsilon = float(epsilon_start)
    stage_limit = max_adaptive_schedule_steps
    if stage_limit is None:
        stage_limit = len(proposed_epsilons) + max_iter

    stages = []
    last_result = None
    first_plateau_epsilon = float("nan")
    reached_requested_epsilon = False

    for stage_index in range(stage_limit):
        proposed_epsilon = (
            float(proposed_epsilons[stage_index])
            if stage_index < len(proposed_epsilons)
            else requested_epsilon
        )
        boundary = _summarize_sk_guard_boundary(
            current_init.ln_mk,
            condensate_species=condensate_species,
            top_k=top_k,
        )
        epsilon_floor = boundary["epsilon_floor"]
        guarded_epsilon = max(proposed_epsilon, epsilon_floor + epsilon_guard_margin)
        pre_feasible = bool(
            jnp.all(LOG_S_MAX + guarded_epsilon - 2.0 * jnp.asarray(current_init.ln_mk) >= 0.0)
        )

        if guarded_epsilon >= current_epsilon - min_epsilon_step:
            first_plateau_epsilon = guarded_epsilon
            stages.append(
                {
                    "stage_index": stage_index,
                    "current_epsilon": current_epsilon,
                    "proposed_epsilon": proposed_epsilon,
                    "epsilon_floor": epsilon_floor,
                    "epsilon_next": guarded_epsilon,
                    "stage_kind": "plateau-stopped",
                    "pre_iteration_sk_feasible": pre_feasible,
                    **boundary,
                }
            )
            break

        stage_kind = (
            "sk-guard-limited"
            if guarded_epsilon > proposed_epsilon + 0.5 * epsilon_guard_margin
            else "fixed-schedule-limited"
        )
        stages.append(
            {
                "stage_index": stage_index,
                "current_epsilon": current_epsilon,
                "proposed_epsilon": proposed_epsilon,
                "epsilon_floor": epsilon_floor,
                "epsilon_next": guarded_epsilon,
                "stage_kind": stage_kind,
                "pre_iteration_sk_feasible": pre_feasible,
                **boundary,
            }
        )

        last_result = minimize_gibbs_cond(
            state,
            init=current_init,
            formula_matrix=formula_matrix,
            formula_matrix_cond=formula_matrix_cond,
            hvector_func=hvector_func,
            hvector_cond_func=hvector_cond_func,
            epsilon=guarded_epsilon,
            residual_crit=jnp.exp(guarded_epsilon),
            max_iter=max_iter,
            element_indices=element_indices,
            debug_nan=debug_nan,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
            support_method=support_method,
            classifier_config=classifier_config,
            condensate_species=condensate_species,
            element_names=element_names,
        )
        current_init = last_result.to_init()
        current_epsilon = float(guarded_epsilon)

        if current_epsilon <= requested_epsilon + min_epsilon_step:
            reached_requested_epsilon = True
            break

    if reached_requested_epsilon:
        final_boundary = _summarize_sk_guard_boundary(
            current_init.ln_mk,
            condensate_species=condensate_species,
            top_k=top_k,
        )
        stages.append(
            {
                "stage_index": len(stages),
                "current_epsilon": current_epsilon,
                "proposed_epsilon": requested_epsilon,
                "epsilon_floor": final_boundary["epsilon_floor"],
                "epsilon_next": requested_epsilon,
                "stage_kind": "final-repeat",
                "pre_iteration_sk_feasible": bool(
                    jnp.all(
                        LOG_S_MAX
                        + requested_epsilon
                        - 2.0 * jnp.asarray(current_init.ln_mk)
                        >= 0.0
                    )
                ),
                **final_boundary,
            }
        )
        last_result = minimize_gibbs_cond(
            state,
            init=current_init,
            formula_matrix=formula_matrix,
            formula_matrix_cond=formula_matrix_cond,
            hvector_func=hvector_func,
            hvector_cond_func=hvector_cond_func,
            epsilon=requested_epsilon,
            residual_crit=jnp.exp(requested_epsilon),
            max_iter=max_iter,
            element_indices=element_indices,
            debug_nan=debug_nan,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
            support_method=support_method,
            classifier_config=classifier_config,
            condensate_species=condensate_species,
            element_names=element_names,
        )
        actual_final_epsilon = requested_epsilon
    else:
        actual_final_epsilon = current_epsilon

    if last_result is None:
        last_result = _plateau_result_from_init(
            current_init,
            actual_epsilon=actual_final_epsilon,
            requested_epsilon=requested_epsilon,
            first_plateau_epsilon=first_plateau_epsilon,
            max_iter=max_iter,
            debug_nan=debug_nan,
        )
    else:
        last_result = _with_schedule_summary(
            last_result,
            requested_epsilon=requested_epsilon,
            actual_epsilon=actual_final_epsilon,
            reached_requested_epsilon=reached_requested_epsilon,
            plateaued=not reached_requested_epsilon,
            first_plateau_epsilon=first_plateau_epsilon,
        )

    return last_result, {
        "epsilon_start": float(epsilon_start),
        "requested_epsilon_crit": requested_epsilon,
        "actual_final_epsilon": float(actual_final_epsilon),
        "reached_requested_epsilon": bool(reached_requested_epsilon),
        "plateaued": bool(not reached_requested_epsilon),
        "first_plateau_epsilon": float(first_plateau_epsilon),
        "stages": stages,
    }


def _minimize_gibbs_cond_legacy(
    state: ThermoState,
    init: CondensateEquilibriumInit,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    residual_crit: float,
    max_iter: int,
    element_indices: Optional[jnp.ndarray],
    debug_nan: bool,
    reduced_solver: str,
    regularization_mode: str,
    regularization_strength: float,
    startup_config: Optional[CondensateRGIEStartupConfig],
    inventory_correction_config: Optional[CondensateRGIEInventoryCorrectionConfig],
    reduced_coupling_config: Optional[CondensateRGIEReducedCouplingConfig],
    line_search_selection_policy: str = "first_monotone_with_best_finite_fallback",
    line_search_charge_row_index: Optional[int] = None,
    line_search_charge_weight: float = 1.0,
) -> CondensateEquilibriumResult:
    n_elements = formula_matrix.shape[0]
    b = (
        jnp.asarray(state.element_vector)
        if element_indices is None
        else jnp.asarray(state.element_vector)[jnp.asarray(element_indices)]
    )
    if b.shape[0] != n_elements:
        raise ValueError(
            "ThermoState.element_vector length does not match the number of element rows "
            f"in the formula matrices (got {b.shape[0]}, expected {n_elements}). "
            "Provide element_indices that map the state vector onto the reduced element set."
        )
    inventory_config = _prepare_inventory_correction_config(inventory_correction_config)
    reduced_config = _prepare_reduced_coupling_config(reduced_coupling_config)
    init_prepared = _apply_rgie_startup_policy(
        init,
        epsilon=epsilon,
        startup_config=startup_config,
        apply_policy=True,
    )
    init_prepared = _apply_inventory_startup_cap(
        init_prepared,
        formula_matrix_cond=formula_matrix_cond,
        b=b,
        inventory_config=inventory_config,
    )
    selected_mode = "current"
    selected_alpha_s = 1.0
    selection = {
        "selected_mode": "current",
        "selected_alpha_s": 1.0,
        "shadow_best_fresh_residual": float("nan"),
        "shadow_current_fresh_residual": float("nan"),
        "mode_selection_margin": reduced_config.mode_selection_margin,
        "escalation_triggered": False,
    }
    if reduced_config.reduced_coupling_mode == "capped_s_only_fixed_alpha":
        selected_mode = "capped_s_only"
        selected_alpha_s = float(reduced_config.alpha_s)
        selection.update(
            {
                "selected_mode": selected_mode,
                "selected_alpha_s": selected_alpha_s,
            }
        )
    elif reduced_config.reduced_coupling_mode in (
        "candidate_selected_active_only",
        "candidate_selected_active_plus_near_jacobian",
        "candidate_selected_active_plus_near_jacobian_with_rem_inventory",
        "candidate_selected_weighted_mask",
    ):
        selected_mode = reduced_config.reduced_coupling_mode
        selected_alpha_s = 1.0
        selection.update(
            {
                "selected_mode": selected_mode,
                "selected_alpha_s": selected_alpha_s,
            }
        )
    elif reduced_config.reduced_coupling_mode == "capped_s_only_conditional":
        hvector = hvector_func(state.temperature)
        hvector_cond = hvector_cond_func(state.temperature)
        selection = select_conditional_capped_s_reduced_coupling_mode(
            init_prepared.ln_nk,
            init_prepared.ln_mk,
            init_prepared.ln_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            state.temperature,
            state.ln_normalized_pressure,
            hvector,
            hvector_cond,
            epsilon,
            alpha_candidates=reduced_config.alpha_s_candidates,
            mode_selection_margin=reduced_config.mode_selection_margin,
            shadow_lambda=reduced_config.shadow_lambda,
        )
        selected_mode = selection["selected_mode"]
        selected_alpha_s = float(selection["selected_alpha_s"])
    ln_nk, ln_mk, ln_ntot, diagnostics_raw = _minimize_gibbs_cond_with_diagnostics_raw(
        state,
        ln_nk_init=init_prepared.ln_nk,
        ln_mk_init=init_prepared.ln_mk,
        ln_ntot_init=init_prepared.ln_ntot,
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=hvector_func,
        hvector_cond_func=hvector_cond_func,
        epsilon=epsilon,
        residual_crit=residual_crit,
        max_iter=max_iter,
        element_indices=element_indices,
        debug_nan=debug_nan,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
        budget_guard_enabled=_inventory_budget_guard_enabled(inventory_config),
        budget_margin=inventory_config.budget_margin,
        emergency_budget_projection_enabled=_inventory_emergency_projection_enabled(
            inventory_config
        ),
        reduced_coupling_mode=selected_mode,
        reduced_coupling_alpha_s=selected_alpha_s,
        gas_step_scale=reduced_config.gas_step_scale,
        gas_step_direction_sign=reduced_config.gas_step_direction_sign,
        ntot_step_scale=reduced_config.ntot_step_scale,
        condensate_step_scale=reduced_config.condensate_step_scale,
        initial_residual_policy=reduced_config.initial_residual_policy,
        reduced_coupling_selection={
            "reduced_coupling_config_mode": reduced_config.reduced_coupling_mode,
            "reduced_coupling_selected_mode": selection["selected_mode"],
            "reduced_coupling_selected_alpha_s": jnp.asarray(
                selection["selected_alpha_s"], dtype=jnp.float64
            ),
            "reduced_coupling_shadow_best_fresh_residual": jnp.asarray(
                selection["shadow_best_fresh_residual"], dtype=jnp.float64
            ),
            "reduced_coupling_shadow_current_fresh_residual": jnp.asarray(
                selection["shadow_current_fresh_residual"], dtype=jnp.float64
            ),
            "reduced_coupling_mode_selection_margin": jnp.asarray(
                selection["mode_selection_margin"], dtype=jnp.float64
            ),
            "reduced_coupling_escalation_triggered": jnp.asarray(
                selection["escalation_triggered"]
            ),
            "gas_step_scale": jnp.asarray(
                reduced_config.gas_step_scale, dtype=jnp.float64
            ),
            "gas_step_direction_sign": jnp.asarray(
                reduced_config.gas_step_direction_sign, dtype=jnp.float64
            ),
            "ntot_step_scale": jnp.asarray(
                (
                    reduced_config.gas_step_scale
                    if reduced_config.ntot_step_scale is None
                    else reduced_config.ntot_step_scale
                ),
                dtype=jnp.float64,
            ),
            "condensate_step_scale": jnp.asarray(
                reduced_config.condensate_step_scale, dtype=jnp.float64
            ),
            "initial_residual_policy": reduced_config.initial_residual_policy,
        },
        line_search_selection_policy=line_search_selection_policy,
        line_search_charge_row_index=line_search_charge_row_index,
        line_search_charge_weight=line_search_charge_weight,
    )
    return CondensateEquilibriumResult(
        ln_nk=ln_nk,
        ln_mk=ln_mk,
        ln_ntot=ln_ntot,
        diagnostics=CondensateEquilibriumDiagnostics.from_mapping(diagnostics_raw),
    )


def solve_gas_equilibrium_with_duals(
    state: ThermoState,
    formula_matrix: jnp.ndarray,
    hvector_func,
    *,
    gas_epsilon_crit: float = 1.0e-12,
    gas_max_iter: int = 1000,
    emit_lnnk_source_trace: bool = False,
    source_trace_case_key: str = "diagnostic",
    source_trace_newton_iter: int = 0,
):
    """Solve the gas-only subproblem and recover a practical dual vector."""

    ln_nk_init0 = jnp.zeros((formula_matrix.shape[1],), dtype=jnp.float64)
    ln_ntot_init0 = jnp.asarray(0.0, dtype=jnp.float64)
    hvector = jnp.asarray(hvector_func(state.temperature), dtype=jnp.float64)
    if emit_lnnk_source_trace:
        (
            ln_nk,
            ln_ntot,
            n_iter,
            final_residual,
            ln_nk_source_trace,
        ) = minimize_gibbs_core_with_source_trace(
            state,
            ln_nk_init0,
            ln_ntot_init0,
            formula_matrix,
            lambda _temperature: hvector,
            epsilon_crit=gas_epsilon_crit,
            max_iter=gas_max_iter,
            source_trace_case_key=source_trace_case_key,
            source_trace_newton_iter=source_trace_newton_iter,
        )
    else:
        ln_nk, ln_ntot, n_iter, final_residual = minimize_gibbs_core(
            state,
            ln_nk_init0,
            ln_ntot_init0,
            formula_matrix,
            lambda _temperature: hvector,
            epsilon_crit=gas_epsilon_crit,
            max_iter=gas_max_iter,
        )
        ln_nk_source_trace = None
    nk = jnp.exp(jnp.asarray(ln_nk, dtype=jnp.float64))
    ntot = jnp.exp(jnp.asarray(ln_ntot, dtype=jnp.float64))
    gk = _compute_gk(state.temperature, ln_nk, ln_ntot, hvector, state.ln_normalized_pressure)
    qmat = formula_matrix @ (nk[:, None] * formula_matrix.T)
    rhs = formula_matrix @ (gk * nk)
    pi_vector = jnp.linalg.lstsq(qmat, rhs)[0]
    stationarity = formula_matrix.T @ pi_vector - gk
    result = {
        "status": "ok",
        "nk": nk,
        "ln_nk": jnp.asarray(ln_nk, dtype=jnp.float64),
        "ntot": ntot,
        "ln_ntot": jnp.asarray(ln_ntot, dtype=jnp.float64),
        "pi_vector": pi_vector,
        "stationarity": stationarity,
        "diagnostics": {
            "converged": bool(float(final_residual) <= float(gas_epsilon_crit)),
            "n_iter": int(n_iter),
            "final_residual": float(final_residual),
        },
    }
    if emit_lnnk_source_trace:
        result["ln_nk_source_trace"] = ln_nk_source_trace
    return result


def _pdipm_activity_fixed_support_batch_core(
    *,
    ln_nk_init: jnp.ndarray,
    ln_mk_init: jnp.ndarray,
    ln_ntot_init: jnp.ndarray,
    element_potential_init: jnp.ndarray,
    rho_init: jnp.ndarray,
    gas_stationarity_source_init: jnp.ndarray,
    use_solver_epsilon: jnp.ndarray,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond_active: jnp.ndarray,
    element_inventory_target: jnp.ndarray,
    hvector: jnp.ndarray,
    hvector_cond_active: jnp.ndarray,
    ln_normalized_pressure: jnp.ndarray,
    epsilon: jnp.ndarray,
    residual_tolerance_multiplier: jnp.ndarray,
    max_iter: int,
    rho_initialization: str = "unit_activity",
    lambda_initialization: str = "gas_lstsq",
) -> tuple[jnp.ndarray, ...]:
    """Run the fixed-shape PD-IPM activity-correction core for one bucket."""

    alpha_grid = jnp.asarray(
        (
            1.0,
            0.5,
            0.25,
            0.125,
            0.0625,
            0.03125,
            0.015625,
            0.01,
            0.003,
            0.001,
            0.0003,
            0.0001,
            1.0e-5,
        ),
        dtype=jnp.float64,
    )
    ag = jnp.asarray(formula_matrix, dtype=jnp.float64)
    ac = jnp.asarray(formula_matrix_cond_active, dtype=jnp.float64)
    positive_stoich = ac > 0.0

    def l2(values: jnp.ndarray) -> jnp.ndarray:
        scale = jnp.max(jnp.abs(values), initial=jnp.asarray(0.0, dtype=values.dtype))
        return jnp.where(scale == 0.0, 0.0, scale * jnp.linalg.norm(values / scale))

    def step(
        q: jnp.ndarray,
        r: jnp.ndarray,
        lam: jnp.ndarray,
        rho: jnp.ndarray,
        qtot: jnp.ndarray,
        target: jnp.ndarray,
        hgas_or_gas_stationarity_source: jnp.ndarray,
        hcond: jnp.ndarray,
        ln_pressure: jnp.ndarray,
        epsilon_vec: jnp.ndarray,
        r_cap: jnp.ndarray,
        use_scalar_step: jnp.ndarray,
    ) -> tuple[jnp.ndarray, ...]:
        gas_stationarity_source = jnp.where(
            use_scalar_step,
            hgas_or_gas_stationarity_source,
            hgas_or_gas_stationarity_source + ln_pressure - qtot,
        )
        log_activity_proxy = ac.T @ lam - hcond
        jac_mask = log_activity_proxy > -0.1
        jac_mask = jnp.where(
            jnp.any(jac_mask),
            jac_mask,
            jnp.arange(r.shape[0]) == jnp.argmax(log_activity_proxy),
        )
        n = jnp.exp(q)
        m = jnp.exp(r)
        eta = jnp.exp(rho)
        j_vec = m / jnp.maximum(eta, 1.0e-300)
        t_vec = r + rho - epsilon_vec
        geff = q + gas_stationarity_source
        gas_inventory = ag @ n
        delta_bhat = target - gas_inventory - ac @ m
        delta_ntot = jnp.sum(n) - jnp.exp(qtot)
        qhat = ag @ (n[:, None] * ag.T) + ac @ (j_vec[:, None] * ac.T)
        qhat = qhat + 1.0e-14 * jnp.eye(qhat.shape[0], dtype=qhat.dtype)
        rhs_top = ag @ (n * geff) + ac @ (j_vec * hcond + m * t_vec - m) + delta_bhat
        rhs_bottom = jnp.dot(n, geff) - delta_ntot
        matrix = jnp.block(
            [
                [qhat, gas_inventory[:, None]],
                [gas_inventory[None, :], jnp.asarray([[delta_ntot]], dtype=qhat.dtype)],
            ]
        )
        rhs = jnp.concatenate([rhs_top, jnp.asarray([rhs_bottom], dtype=qhat.dtype)])
        solution = jnp.linalg.lstsq(matrix, rhs, rcond=None)[0]
        solution = jnp.nan_to_num(solution, nan=0.0, posinf=0.0, neginf=0.0)
        pi = solution[:-1]
        delta_qtot = solution[-1]
        raw_delta_q = ag.T @ pi + delta_qtot - geff
        raw_delta_rho = (hcond - ac.T @ pi) / jnp.maximum(eta, 1.0e-300) - 1.0
        raw_delta_r = -raw_delta_rho - t_vec
        raw_delta_lam = pi - lam
        delta_q = jnp.where(use_scalar_step, raw_delta_q, jnp.clip(raw_delta_q, -2.0, 2.0))
        delta_r = jnp.where(use_scalar_step, raw_delta_r, jnp.clip(raw_delta_r, -5.0, 5.0))
        delta_rho = jnp.where(
            use_scalar_step,
            raw_delta_rho,
            jnp.clip(raw_delta_rho, -5.0, 5.0),
        )
        delta_lam = jnp.where(
            use_scalar_step,
            raw_delta_lam,
            jnp.clip(raw_delta_lam, -100.0, 100.0),
        )
        alpha_r = jnp.min(
            jnp.where(delta_r < 0.0, -1.0 / delta_r, 1.0),
            initial=jnp.asarray(1.0, dtype=jnp.float64),
        )
        alpha_rho = jnp.min(
            jnp.where(delta_rho < 0.0, -1.0 / delta_rho, 1.0),
            initial=jnp.asarray(1.0, dtype=jnp.float64),
        )
        alpha_boundary = jnp.minimum(
            1.0,
            0.995 * jnp.minimum(alpha_r, alpha_rho),
        )
        alpha_boundary = jnp.where(
            use_scalar_step & jnp.isfinite(alpha_boundary) & (alpha_boundary > 0.0),
            alpha_boundary,
            jnp.asarray(1.0, dtype=jnp.float64),
        )

        def residual_components(
            qi: jnp.ndarray,
            ri: jnp.ndarray,
            lami: jnp.ndarray,
            rhoi: jnp.ndarray,
            qtoti: jnp.ndarray,
        ) -> tuple[jnp.ndarray, ...]:
            ni = jnp.exp(qi)
            mi = jnp.exp(ri)
            etai = jnp.exp(rhoi)
            gas = qi + gas_stationarity_source + qtot - qtoti - ag.T @ lami
            cond = hcond - ac.T @ lami - etai
            budget = ag @ ni + ac @ mi - target
            comp = ri + rhoi - epsilon_vec
            total_density = jnp.asarray([jnp.sum(ni) - jnp.exp(qtoti)], dtype=qi.dtype)
            cond_masked = jnp.where(jac_mask, cond, 0.0)
            return gas, cond_masked, budget, comp, total_density

        def residual_norm(
            qi: jnp.ndarray,
            ri: jnp.ndarray,
            lami: jnp.ndarray,
            rhoi: jnp.ndarray,
            qtoti: jnp.ndarray,
        ) -> jnp.ndarray:
            gas, cond, budget, comp, total_density = residual_components(
                qi,
                ri,
                lami,
                rhoi,
                qtoti,
            )
            return l2(
                jnp.concatenate(
                    [
                        gas,
                        cond,
                        budget,
                        comp,
                        total_density,
                    ]
                )
            )

        initial_norm = residual_norm(q, r, lam, rho, qtot)
        initial_gas, initial_cond, initial_budget, initial_comp, _initial_total = (
            residual_components(q, r, lam, rho, qtot)
        )
        initial_gas_norm = l2(initial_gas)
        initial_cond_norm = l2(initial_cond)
        initial_budget_norm = l2(initial_budget)
        initial_comp_norm = l2(initial_comp)

        def trial(alpha: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
            tq = q + alpha * delta_q
            tr = jnp.minimum(r + alpha * delta_r, r_cap)
            tlam = lam + alpha * delta_lam
            trho = rho + alpha * delta_rho
            tqtot = qtot + alpha * delta_qtot
            gas, cond, budget, comp, _total = residual_components(
                tq,
                tr,
                tlam,
                trho,
                tqtot,
            )
            return (
                tq,
                tr,
                tlam,
                trho,
                tqtot,
                residual_norm(tq, tr, tlam, trho, tqtot),
                l2(gas),
                l2(cond),
                l2(budget),
                l2(comp),
            )

        bounded_alpha_grid = jnp.minimum(alpha_grid, alpha_boundary)
        (
            tq,
            tr,
            tlam,
            trho,
            tqtot,
            norms,
            gas_norms,
            cond_norms,
            budget_norms,
            comp_norms,
        ) = jax.vmap(trial)(bounded_alpha_grid)
        finite = jnp.isfinite(norms)
        accepted_mask = finite & (norms < initial_norm)
        any_accepted = jnp.any(accepted_mask)
        first_index = jnp.argmax(accepted_mask)
        best_index = jnp.argmin(jnp.where(finite, norms, jnp.inf))
        component_improved = (
            (gas_norms < initial_gas_norm)
            | (cond_norms < initial_cond_norm)
            | (comp_norms < initial_comp_norm)
        )
        budget_not_broken = budget_norms <= jnp.maximum(
            1.25 * initial_budget_norm,
            initial_budget_norm + jnp.asarray(1.0e-8, dtype=initial_budget_norm.dtype),
        )
        fallback_mask = (
            finite
            & component_improved
            & budget_not_broken
            & (
                norms
                <= 1.25
                * jnp.maximum(initial_norm, jnp.asarray(1.0, dtype=initial_norm.dtype))
            )
        )
        any_fallback = jnp.any(fallback_mask)
        fallback_merit = jnp.maximum(
            jnp.maximum(gas_norms, cond_norms),
            comp_norms,
        )
        fallback_index = jnp.argmin(jnp.where(fallback_mask, fallback_merit, jnp.inf))
        selected = jnp.where(
            any_accepted,
            first_index,
            jnp.where(any_fallback, fallback_index, best_index),
        )
        step_accepted = any_accepted | any_fallback
        return (
            jnp.where(step_accepted, tq[selected], q),
            jnp.where(step_accepted, tr[selected], r),
            jnp.where(step_accepted, tlam[selected], lam),
            jnp.where(step_accepted, trho[selected], rho),
            jnp.where(step_accepted, tqtot[selected], qtot),
            jnp.where(step_accepted, norms[selected], initial_norm),
            step_accepted,
            initial_norm,
        )

    def run_one(
        q0: jnp.ndarray,
        r0: jnp.ndarray,
        qtot0: jnp.ndarray,
        lam_init: jnp.ndarray,
        rho_init_one: jnp.ndarray,
        gas_source_init: jnp.ndarray,
        use_solver_epsilon_one: jnp.ndarray,
        target: jnp.ndarray,
        hgas: jnp.ndarray,
        hcond: jnp.ndarray,
        ln_pressure: jnp.ndarray,
        solver_epsilon: jnp.ndarray,
    ) -> tuple[jnp.ndarray, ...]:
        capacity = jnp.where(
            positive_stoich,
            target[:, None] / ac,
            jnp.inf,
        )
        condensate_capacity = jnp.min(capacity, axis=0)
        r_cap = jnp.log(jnp.maximum(condensate_capacity, 1.0e-300))
        reference_element_indices = jnp.argmin(capacity, axis=0)
        reference_budget = target[reference_element_indices]
        legacy_epsilon_vec = jnp.log(
            jnp.maximum(1.0e-15 * reference_budget, 1.0e-300)
        )
        epsilon_vec = jnp.where(
            use_solver_epsilon_one,
            jnp.full_like(r0, solver_epsilon),
            legacy_epsilon_vec,
        )
        if rho_initialization == "provided":
            rho0 = rho_init_one
        elif rho_initialization == "complementarity":
            rho0 = epsilon_vec - r0
        else:
            rho0 = jnp.zeros_like(r0)
        gas_stationarity_source_init = jnp.where(
            use_solver_epsilon_one,
            gas_source_init,
            hgas + ln_pressure - qtot0,
        )
        eta0 = jnp.exp(rho0)
        lam0_gas = jnp.linalg.lstsq(
            ag.T,
            q0 + gas_stationarity_source_init,
            rcond=None,
        )[0]
        lam0_joint = jnp.linalg.lstsq(
            jnp.concatenate([ag.T, ac.T], axis=0),
            jnp.concatenate([q0 + gas_stationarity_source_init, hcond - eta0]),
            rcond=None,
        )[0]
        if lambda_initialization == "provided":
            lam0 = lam_init
        elif lambda_initialization == "gas_cond_lstsq":
            lam0 = lam0_joint
        else:
            lam0 = lam0_gas
        residual_crit = residual_tolerance_multiplier * jnp.exp(solver_epsilon)
        initial_residual = step(
            q0,
            r0,
            lam0,
            rho0,
            qtot0,
            target,
            jnp.where(use_solver_epsilon_one, gas_source_init, hgas),
            hcond,
            ln_pressure,
            epsilon_vec,
            r_cap,
            use_solver_epsilon_one,
        )[7]
        initial_running = initial_residual > residual_crit

        def body(carry, _):
            q, r, lam, rho, qtot, residual, residual_qtot_ref, still_running = carry
            (
                next_q,
                next_r,
                next_lam,
                next_rho,
                next_qtot,
                next_residual,
                accepted,
                _initial_residual,
            ) = step(
                q,
                r,
                lam,
                rho,
                    qtot,
                    target,
                    jnp.where(use_solver_epsilon_one, gas_source_init, hgas),
                    hcond,
                ln_pressure,
                    epsilon_vec,
                    r_cap,
                    use_solver_epsilon_one,
                )
            apply_step = still_running & accepted
            return (
                jnp.where(apply_step, next_q, q),
                jnp.where(apply_step, next_r, r),
                jnp.where(apply_step, next_lam, lam),
                jnp.where(apply_step, next_rho, rho),
                jnp.where(apply_step, next_qtot, qtot),
                jnp.where(still_running, next_residual, residual),
                jnp.where(apply_step, qtot, residual_qtot_ref),
                apply_step,
            ), (jnp.where(still_running, next_residual, residual), apply_step)

        initial = (
            q0,
            r0,
            lam0,
            rho0,
            qtot0,
            initial_residual,
            qtot0,
            initial_running,
        )
        final, history = lax.scan(body, initial, xs=None, length=max_iter)
        accepted_history = history[1]
        accepted_count = jnp.sum(accepted_history.astype(jnp.int32))
        n_iter = jnp.minimum(accepted_count + 1, jnp.asarray(max_iter, dtype=jnp.int32))
        final_residual = final[5]
        converged = final_residual <= residual_crit
        qf, rf, lamf, rhof, qtotf = final[0], final[1], final[2], final[3], final[4]
        qtot_residual_reference = final[6]
        gas_stationarity_source_final = jnp.where(
            use_solver_epsilon_one,
            gas_source_init,
            hgas + ln_pressure - qtot_residual_reference,
        )
        log_activity_proxy_final = ac.T @ lamf - hcond
        jac_mask_final = log_activity_proxy_final > -0.1
        jac_mask_final = jnp.where(
            jnp.any(jac_mask_final),
            jac_mask_final,
            jnp.arange(rf.shape[0]) == jnp.argmax(log_activity_proxy_final),
        )
        nf = jnp.exp(qf)
        mf = jnp.exp(rf)
        etaf = jnp.exp(rhof)
        gas_component = (
            qf
            + gas_stationarity_source_final
            + qtot_residual_reference
            - qtotf
            - ag.T @ lamf
        )
        cond_component = hcond - ac.T @ lamf - etaf
        budget_component = ag @ nf + ac @ mf - target
        complementarity_component = rf + rhof - epsilon_vec
        total_density_component = jnp.asarray(
            [jnp.sum(nf) - jnp.exp(qtotf)],
            dtype=qf.dtype,
        )
        return (
            qf,
            rf,
            qtotf,
            n_iter,
            converged,
            (n_iter >= max_iter) & (~converged),
            final_residual,
            residual_crit,
            accepted_count,
            l2(gas_component),
            l2(jnp.where(jac_mask_final, cond_component, 0.0)),
            l2(budget_component),
            l2(complementarity_component),
            l2(total_density_component),
        )

    return jax.vmap(run_one)(
        ln_nk_init,
        ln_mk_init,
        ln_ntot_init,
        element_potential_init,
        rho_init,
        gas_stationarity_source_init,
        use_solver_epsilon,
        element_inventory_target,
        hvector,
        hvector_cond_active,
        ln_normalized_pressure,
        epsilon,
    )


_pdipm_activity_fixed_support_batch_core_jit = jax.jit(
    _pdipm_activity_fixed_support_batch_core,
    static_argnames=("max_iter", "rho_initialization", "lambda_initialization"),
)


def _solve_pdipm_rgie_v11_activity_correction_fixed_support_batch(
    *,
    ln_nk_init: jnp.ndarray,
    ln_mk_init: jnp.ndarray,
    ln_ntot_init: jnp.ndarray,
    element_potential_init: Optional[jnp.ndarray] = None,
    rho_init: Optional[jnp.ndarray] = None,
    barrier_epsilon_init: Optional[jnp.ndarray] = None,
    gas_stationarity_source_init: Optional[jnp.ndarray] = None,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond_active: jnp.ndarray,
    element_inventory_target: jnp.ndarray,
    hvector: jnp.ndarray,
    hvector_cond_active: jnp.ndarray,
    ln_normalized_pressure: jnp.ndarray,
    epsilon: float = -10.0,
    residual_tolerance_multiplier: float = 1.0,
    max_iter: int,
    rho_initialization: str = "unit_activity",
    lambda_initialization: str = "gas_lstsq",
) -> tuple[CondensateEquilibriumResult, dict[str, Any]]:
    """Run the experimental fixed-support activity-correction core for one bucket.

    The helper is intentionally private and currently does not alter the
    production route. It provides the GPU-friendly fixed-shape batch primitive
    used by the optimization experiments.
    """

    ln_nk_init_array = jnp.asarray(ln_nk_init, dtype=jnp.float64)
    ln_mk_init_array = jnp.asarray(ln_mk_init, dtype=jnp.float64)
    ln_ntot_init_array = jnp.asarray(ln_ntot_init, dtype=jnp.float64)
    formula_matrix_array = jnp.asarray(formula_matrix, dtype=jnp.float64)
    element_potential_init_array = (
        jnp.asarray(element_potential_init, dtype=jnp.float64)
        if element_potential_init is not None
        else jnp.zeros(
            (ln_nk_init_array.shape[0], formula_matrix_array.shape[0]),
            dtype=jnp.float64,
        )
    )
    rho_init_array = (
        jnp.asarray(rho_init, dtype=jnp.float64)
        if rho_init is not None
        else jnp.zeros_like(ln_mk_init_array)
    )
    epsilon_array = (
        jnp.asarray(barrier_epsilon_init, dtype=jnp.float64)
        if barrier_epsilon_init is not None
        else jnp.full_like(
            ln_ntot_init_array,
            float(epsilon),
            dtype=jnp.float64,
        )
    )
    use_solver_epsilon_array = jnp.full_like(
        ln_ntot_init_array,
        barrier_epsilon_init is not None,
        dtype=bool,
    )
    gas_stationarity_source_init_array = (
        jnp.asarray(gas_stationarity_source_init, dtype=jnp.float64)
        if gas_stationarity_source_init is not None
        else jnp.asarray(hvector, dtype=jnp.float64)
        + jnp.asarray(ln_normalized_pressure, dtype=jnp.float64)[:, None]
        - ln_ntot_init_array[:, None]
    )

    (
        ln_nk,
        ln_mk,
        ln_ntot,
        n_iter,
        converged,
        hit_max_iter,
        final_residual,
        residual_crit,
        accepted_count,
        gas_residual_norm,
        condensate_stationarity_residual_norm,
        budget_residual_norm,
        complementarity_residual_norm,
        total_density_residual_norm,
    ) = _pdipm_activity_fixed_support_batch_core_jit(
        ln_nk_init=ln_nk_init_array,
        ln_mk_init=ln_mk_init_array,
        ln_ntot_init=ln_ntot_init_array,
        element_potential_init=element_potential_init_array,
        rho_init=rho_init_array,
        gas_stationarity_source_init=gas_stationarity_source_init_array,
        use_solver_epsilon=use_solver_epsilon_array,
        formula_matrix=formula_matrix_array,
        formula_matrix_cond_active=jnp.asarray(
            formula_matrix_cond_active,
            dtype=jnp.float64,
        ),
        element_inventory_target=jnp.asarray(
            element_inventory_target,
            dtype=jnp.float64,
        ),
        hvector=jnp.asarray(hvector, dtype=jnp.float64),
        hvector_cond_active=jnp.asarray(hvector_cond_active, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(ln_normalized_pressure, dtype=jnp.float64),
        epsilon=epsilon_array,
        residual_tolerance_multiplier=jnp.asarray(
            float(residual_tolerance_multiplier),
            dtype=jnp.float64,
        ),
        max_iter=int(max_iter),
        rho_initialization=str(rho_initialization),
        lambda_initialization=str(lambda_initialization),
    )
    diagnostics = CondensateEquilibriumDiagnostics.from_mapping(
        {
            "n_iter": n_iter,
            "converged": converged,
            "hit_max_iter": hit_max_iter,
            "final_residual": final_residual,
            "residual_crit": residual_crit,
            "max_iter": jnp.full_like(n_iter, int(max_iter), dtype=jnp.int32),
            "epsilon": epsilon_array,
            "final_step_size": jnp.zeros_like(final_residual),
            "invalid_numbers_detected": ~jnp.isfinite(final_residual),
            "debug_nan": jnp.zeros_like(converged, dtype=bool),
            "reduced_coupling_selected_alpha_s": jnp.ones_like(final_residual),
        }
    )
    return (
        CondensateEquilibriumResult(
            ln_nk=ln_nk,
            ln_mk=ln_mk,
            ln_ntot=ln_ntot,
            diagnostics=diagnostics,
        ),
        {
            "pdipm_rgie_v11_activity_correction_fixed_support_batch": {
                "schema": "exogibbs_pdipm_rgie_v11_activity_correction_fixed_support_batch_v1",
                "experimental": True,
                "production_route_wiring": False,
                "accepted_iteration_count": accepted_count,
                "gas_residual_norm": gas_residual_norm,
                "condensate_stationarity_residual_norm": condensate_stationarity_residual_norm,
                "budget_residual_norm": budget_residual_norm,
                "complementarity_residual_norm": complementarity_residual_norm,
                "total_density_residual_norm": total_density_residual_norm,
                "rho_initialization": str(rho_initialization),
                "lambda_initialization": str(lambda_initialization),
            }
        },
    )


def _prepare_pdipm_rgie_v11_activity_correction_profile_buckets(
    *,
    states: Sequence[ThermoState],
    init_states: Sequence[CondensateEquilibriumInit],
    support_indices_by_layer: Sequence[Sequence[int]],
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    hvector_by_layer: Optional[jnp.ndarray] = None,
    hvector_cond_by_layer: Optional[jnp.ndarray] = None,
) -> tuple[_PDIPMActivityFixedSupportBucket, ...]:
    """Prepare same-support profile buckets without running the solver."""

    n_layers = len(states)
    if len(init_states) != n_layers or len(support_indices_by_layer) != n_layers:
        raise ValueError("states, init_states, and support_indices_by_layer must match")

    buckets: dict[tuple[int, ...], list[int]] = {}
    for layer_index, support_indices in enumerate(support_indices_by_layer):
        support_key = tuple(int(index) for index in support_indices)
        if not support_key:
            raise ValueError("fixed-support profile buckets require non-empty support")
        buckets.setdefault(support_key, []).append(layer_index)

    formula_matrix_cond = jnp.asarray(formula_matrix_cond, dtype=jnp.float64)
    if hvector_by_layer is not None:
        hvector_by_layer = jnp.asarray(hvector_by_layer, dtype=jnp.float64)
        if hvector_by_layer.shape[0] != n_layers:
            raise ValueError("hvector_by_layer must have one row per layer")
    if hvector_cond_by_layer is not None:
        hvector_cond_by_layer = jnp.asarray(hvector_cond_by_layer, dtype=jnp.float64)
        if hvector_cond_by_layer.shape[0] != n_layers:
            raise ValueError("hvector_cond_by_layer must have one row per layer")

    prepared_buckets = []
    for support_key, layer_indices in buckets.items():
        support_array = jnp.asarray(support_key, dtype=jnp.int32)
        ln_nk_init = []
        ln_mk_init = []
        ln_ntot_init = []
        element_potential_init = []
        rho_init = []
        barrier_epsilon_init = []
        gas_stationarity_source_init = []
        have_element_potential = True
        have_rho = True
        have_barrier_epsilon = True
        have_gas_stationarity_source = True
        targets = []
        hvectors = []
        hcond_active = []
        ln_pressures = []
        for layer_index in layer_indices:
            state = states[layer_index]
            init = _prepare_condensate_init(init_states[layer_index])
            ln_nk_init.append(jnp.asarray(init.ln_nk, dtype=jnp.float64))
            ln_mk = jnp.asarray(init.ln_mk, dtype=jnp.float64)
            if ln_mk.shape[0] == formula_matrix_cond.shape[1]:
                ln_mk = ln_mk[support_array]
            elif ln_mk.shape[0] != support_array.shape[0]:
                raise ValueError(
                    "init_state ln_mk must be full condensate length or support length"
                )
            ln_mk_init.append(ln_mk)
            ln_ntot_init.append(jnp.asarray(init.ln_ntot, dtype=jnp.float64))
            if init.element_potential is None:
                have_element_potential = False
            else:
                element_potential = jnp.asarray(
                    init.element_potential,
                    dtype=jnp.float64,
                )
                if element_potential.shape[0] != formula_matrix_cond.shape[0]:
                    raise ValueError(
                        "init_state element_potential must have one value per element"
                    )
                element_potential_init.append(element_potential)
            if init.rho is None:
                have_rho = False
            else:
                rho = jnp.asarray(init.rho, dtype=jnp.float64)
                if rho.shape[0] == formula_matrix_cond.shape[1]:
                    rho = rho[support_array]
                elif rho.shape[0] != support_array.shape[0]:
                    raise ValueError(
                        "init_state rho must be full condensate length or support length"
                    )
                rho_init.append(rho)
            if init.barrier_epsilon is None:
                have_barrier_epsilon = False
            else:
                barrier_epsilon = jnp.asarray(init.barrier_epsilon, dtype=jnp.float64)
                if barrier_epsilon.ndim != 0:
                    raise ValueError("init_state barrier_epsilon must be scalar")
                barrier_epsilon_init.append(barrier_epsilon)
            if init.gas_stationarity_source is None:
                have_gas_stationarity_source = False
            else:
                gas_source = jnp.asarray(
                    init.gas_stationarity_source,
                    dtype=jnp.float64,
                )
                if gas_source.shape[0] != jnp.asarray(init.ln_nk).shape[0]:
                    raise ValueError(
                        "init_state gas_stationarity_source must match gas species length"
                    )
                gas_stationarity_source_init.append(gas_source)
            targets.append(jnp.asarray(state.element_vector, dtype=jnp.float64))
            hgas = (
                hvector_by_layer[layer_index]
                if hvector_by_layer is not None
                else jnp.asarray(hvector_func(state.temperature), dtype=jnp.float64)
            )
            hcond_full = (
                hvector_cond_by_layer[layer_index]
                if hvector_cond_by_layer is not None
                else jnp.asarray(hvector_cond_func(state.temperature), dtype=jnp.float64)
            )
            hvectors.append(hgas)
            hcond_active.append(hcond_full[support_array])
            ln_pressures.append(
                jnp.asarray(state.ln_normalized_pressure, dtype=jnp.float64)
            )
        prepared_buckets.append(
            _PDIPMActivityFixedSupportBucket(
                support_indices=support_key,
                layer_indices=tuple(int(index) for index in layer_indices),
                formula_matrix_cond_active=jnp.asarray(
                    formula_matrix_cond[:, support_array],
                    dtype=jnp.float64,
                ),
                ln_nk_init=jnp.stack(ln_nk_init, axis=0),
                ln_mk_init=jnp.stack(ln_mk_init, axis=0),
                ln_ntot_init=jnp.stack(ln_ntot_init, axis=0),
                element_potential_init=(
                    jnp.stack(element_potential_init, axis=0)
                    if have_element_potential
                    else None
                ),
                rho_init=jnp.stack(rho_init, axis=0) if have_rho else None,
                barrier_epsilon_init=(
                    jnp.stack(barrier_epsilon_init, axis=0)
                    if have_barrier_epsilon
                    else None
                ),
                gas_stationarity_source_init=(
                    jnp.stack(gas_stationarity_source_init, axis=0)
                    if have_gas_stationarity_source
                    else None
                ),
                element_inventory_target=jnp.stack(targets, axis=0),
                hvector=jnp.stack(hvectors, axis=0),
                hvector_cond_active=jnp.stack(hcond_active, axis=0),
                ln_normalized_pressure=jnp.stack(ln_pressures, axis=0),
            )
        )
    return tuple(prepared_buckets)


def _run_pdipm_rgie_v11_activity_correction_prepared_profile_buckets(
    *,
    buckets: Sequence[_PDIPMActivityFixedSupportBucket],
    formula_matrix: jnp.ndarray,
    epsilon: float,
    max_iter: int,
    rho_initialization: str = "unit_activity",
    lambda_initialization: str = "gas_lstsq",
    residual_tolerance_multiplier: float = 1.0,
) -> tuple[tuple[CondensateEquilibriumResult, ...], dict[str, Any]]:
    """Run already-prepared profile buckets without per-layer materialization."""

    formula_matrix = jnp.asarray(formula_matrix, dtype=jnp.float64)
    results = []
    bucket_reports = []
    for bucket in buckets:
        batch_result, batch_extra = (
            _solve_pdipm_rgie_v11_activity_correction_fixed_support_batch(
                ln_nk_init=bucket.ln_nk_init,
                ln_mk_init=bucket.ln_mk_init,
                ln_ntot_init=bucket.ln_ntot_init,
                element_potential_init=bucket.element_potential_init,
                rho_init=bucket.rho_init,
                barrier_epsilon_init=bucket.barrier_epsilon_init,
                gas_stationarity_source_init=bucket.gas_stationarity_source_init,
                formula_matrix=formula_matrix,
                formula_matrix_cond_active=bucket.formula_matrix_cond_active,
                element_inventory_target=bucket.element_inventory_target,
                hvector=bucket.hvector,
                hvector_cond_active=bucket.hvector_cond_active,
                ln_normalized_pressure=bucket.ln_normalized_pressure,
                epsilon=epsilon,
                residual_tolerance_multiplier=residual_tolerance_multiplier,
                max_iter=max_iter,
                rho_initialization=rho_initialization,
                lambda_initialization=lambda_initialization,
            )
        )
        results.append(batch_result)
        batch_payload = batch_extra[
            "pdipm_rgie_v11_activity_correction_fixed_support_batch"
        ]
        bucket_reports.append(
            {
                "support_indices": bucket.support_indices,
                "layer_indices": bucket.layer_indices,
                "execution": "batch",
                "batch_size": len(bucket.layer_indices),
                "accepted_iteration_count": batch_payload[
                    "accepted_iteration_count"
                ],
            }
        )
    return tuple(results), {
        "pdipm_rgie_v11_activity_correction_prepared_profile_buckets": {
            "schema": "exogibbs_pdipm_rgie_v11_activity_correction_prepared_profile_buckets_v1",
            "experimental": True,
            "production_route_wiring": False,
            "bucket_count": len(bucket_reports),
            "layer_count": sum(len(bucket.layer_indices) for bucket in buckets),
            "buckets": tuple(bucket_reports),
        }
    }


def _solve_pdipm_rgie_v11_activity_correction_profile_buckets(
    *,
    states: Sequence[ThermoState],
    init_states: Sequence[CondensateEquilibriumInit],
    support_indices_by_layer: Sequence[Sequence[int]],
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    max_iter: int,
    hvector_by_layer: Optional[jnp.ndarray] = None,
    hvector_cond_by_layer: Optional[jnp.ndarray] = None,
    min_batch_size: int = 2,
) -> tuple[tuple[CondensateEquilibriumResult, ...], dict[str, Any]]:
    """Dispatch fixed-support profile layers through same-support batches.

    This private helper assumes support discovery has already happened. It is
    intentionally not wired into the public profile route yet; callers can use
    it to validate fixed-support bucket execution before enabling a production
    path.
    """

    n_layers = len(states)
    if len(init_states) != n_layers or len(support_indices_by_layer) != n_layers:
        raise ValueError("states, init_states, and support_indices_by_layer must match")
    if min_batch_size < 1:
        raise ValueError("min_batch_size must be at least 1")

    buckets: dict[tuple[int, ...], list[int]] = {}
    for layer_index, support_indices in enumerate(support_indices_by_layer):
        support_key = tuple(int(index) for index in support_indices)
        if not support_key:
            raise ValueError("fixed-support profile buckets require non-empty support")
        buckets.setdefault(support_key, []).append(layer_index)

    results: list[CondensateEquilibriumResult | None] = [None] * n_layers
    bucket_reports: list[dict[str, Any]] = []
    formula_matrix = jnp.asarray(formula_matrix, dtype=jnp.float64)
    formula_matrix_cond = jnp.asarray(formula_matrix_cond, dtype=jnp.float64)
    if hvector_by_layer is not None:
        hvector_by_layer = jnp.asarray(hvector_by_layer, dtype=jnp.float64)
        if hvector_by_layer.shape[0] != n_layers:
            raise ValueError("hvector_by_layer must have one row per layer")
    if hvector_cond_by_layer is not None:
        hvector_cond_by_layer = jnp.asarray(hvector_cond_by_layer, dtype=jnp.float64)
        if hvector_cond_by_layer.shape[0] != n_layers:
            raise ValueError("hvector_cond_by_layer must have one row per layer")

    for support_key, layer_indices in buckets.items():
        support_array = jnp.asarray(support_key, dtype=jnp.int32)
        formula_matrix_cond_active = jnp.asarray(
            formula_matrix_cond[:, support_array],
            dtype=jnp.float64,
        )
        if len(layer_indices) >= min_batch_size:
            ln_nk_init = []
            ln_mk_init = []
            ln_ntot_init = []
            targets = []
            hvectors = []
            hcond_active = []
            ln_pressures = []
            for layer_index in layer_indices:
                state = states[layer_index]
                init = _prepare_condensate_init(init_states[layer_index])
                ln_nk_init.append(jnp.asarray(init.ln_nk, dtype=jnp.float64))
                ln_mk = jnp.asarray(init.ln_mk, dtype=jnp.float64)
                if ln_mk.shape[0] == formula_matrix_cond.shape[1]:
                    ln_mk = ln_mk[support_array]
                elif ln_mk.shape[0] != support_array.shape[0]:
                    raise ValueError(
                        "init_state ln_mk must be full condensate length or support length"
                    )
                ln_mk_init.append(ln_mk)
                ln_ntot_init.append(jnp.asarray(init.ln_ntot, dtype=jnp.float64))
                targets.append(jnp.asarray(state.element_vector, dtype=jnp.float64))
                hgas = (
                    hvector_by_layer[layer_index]
                    if hvector_by_layer is not None
                    else jnp.asarray(hvector_func(state.temperature), dtype=jnp.float64)
                )
                hcond_full = (
                    hvector_cond_by_layer[layer_index]
                    if hvector_cond_by_layer is not None
                    else jnp.asarray(hvector_cond_func(state.temperature), dtype=jnp.float64)
                )
                hvectors.append(hgas)
                hcond_active.append(hcond_full[support_array])
                ln_pressures.append(
                    jnp.asarray(state.ln_normalized_pressure, dtype=jnp.float64)
                )
            batch_result, batch_extra = (
                _solve_pdipm_rgie_v11_activity_correction_fixed_support_batch(
                    ln_nk_init=jnp.stack(ln_nk_init, axis=0),
                    ln_mk_init=jnp.stack(ln_mk_init, axis=0),
                    ln_ntot_init=jnp.stack(ln_ntot_init, axis=0),
                    formula_matrix=formula_matrix,
                    formula_matrix_cond_active=formula_matrix_cond_active,
                    element_inventory_target=jnp.stack(targets, axis=0),
                    hvector=jnp.stack(hvectors, axis=0),
                    hvector_cond_active=jnp.stack(hcond_active, axis=0),
                    ln_normalized_pressure=jnp.stack(ln_pressures, axis=0),
                    epsilon=epsilon,
                    max_iter=max_iter,
                    rho_initialization="unit_activity",
                    lambda_initialization="gas_lstsq",
                )
            )
            for local_index, layer_index in enumerate(layer_indices):
                def take_layer(value, local_index=local_index):
                    value = jnp.asarray(value)
                    if value.ndim > 0 and value.shape[0] == len(layer_indices):
                        return value[local_index]
                    return value

                diagnostics = tree_util.tree_map(
                    take_layer,
                    batch_result.diagnostics,
                )
                results[layer_index] = CondensateEquilibriumResult(
                    ln_nk=batch_result.ln_nk[local_index],
                    ln_mk=batch_result.ln_mk[local_index],
                    ln_ntot=batch_result.ln_ntot[local_index],
                    diagnostics=diagnostics,
                )
            batch_payload = batch_extra[
                "pdipm_rgie_v11_activity_correction_fixed_support_batch"
            ]
            bucket_reports.append(
                {
                    "support_indices": support_key,
                    "layer_indices": tuple(int(index) for index in layer_indices),
                    "execution": "batch",
                    "batch_size": len(layer_indices),
                    "accepted_iteration_count": batch_payload[
                        "accepted_iteration_count"
                    ],
                }
            )
            continue

        layer_index = layer_indices[0]
        state = states[layer_index]
        init = _prepare_condensate_init(init_states[layer_index])
        hcond_full = (
            hvector_cond_by_layer[layer_index]
            if hvector_cond_by_layer is not None
            else jnp.asarray(hvector_cond_func(state.temperature), dtype=jnp.float64)
        )
        ln_mk = jnp.asarray(init.ln_mk, dtype=jnp.float64)
        if ln_mk.shape[0] == formula_matrix_cond.shape[1]:
            init = CondensateEquilibriumInit(
                ln_nk=init.ln_nk,
                ln_mk=ln_mk[support_array],
                ln_ntot=init.ln_ntot,
            )
        results[layer_index], _extra = _solve_pdipm_rgie_v11_activity_correction_layer(
            state=state,
            init_state=init,
            formula_matrix=formula_matrix,
            formula_matrix_cond_active=formula_matrix_cond_active,
            hvector_func=hvector_func,
            hvector_cond_active=hcond_full[support_array],
            epsilon=epsilon,
            max_iter=max_iter,
        )
        bucket_reports.append(
            {
                "support_indices": support_key,
                "layer_indices": (int(layer_index),),
                "execution": "single",
                "batch_size": 1,
            }
        )

    completed = tuple(result for result in results if result is not None)
    if len(completed) != n_layers:
        raise RuntimeError("internal error: not all profile bucket layers completed")
    return completed, {
        "pdipm_rgie_v11_activity_correction_profile_buckets": {
            "schema": "exogibbs_pdipm_rgie_v11_activity_correction_profile_buckets_v1",
            "experimental": True,
            "production_route_wiring": False,
            "bucket_count": len(bucket_reports),
            "layer_count": n_layers,
            "buckets": tuple(bucket_reports),
        }
    }


def _solve_pdipm_rgie_v11_activity_correction_layer(
    *,
    state: ThermoState,
    init_state: CondensateEquilibriumInit,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond_active: jnp.ndarray,
    hvector_func,
    hvector_cond_active: jnp.ndarray,
    epsilon: float,
    max_iter: int,
) -> tuple[CondensateEquilibriumResult, dict[str, Any]]:
    """Run the opt-in v1.1 PD-IPM/RGIE layer with explicit activity correction."""

    from exogibbs.optimize.pdipm_rgie_cond import (
        build_pdipm_rgie_condensate_state,
        solve_pdipm_rgie_algorithm_v11_reduced_step,
    )

    hvector = jnp.asarray(hvector_func(state.temperature), dtype=jnp.float64)
    q = np.asarray(jnp.asarray(init_state.ln_nk, dtype=jnp.float64), dtype=np.float64)
    r = np.asarray(jnp.asarray(init_state.ln_mk, dtype=jnp.float64), dtype=np.float64)
    qtot = float(jnp.asarray(init_state.ln_ntot, dtype=jnp.float64))
    b = np.asarray(jnp.asarray(state.element_vector, dtype=jnp.float64), dtype=np.float64)
    ag = np.asarray(jnp.asarray(formula_matrix, dtype=jnp.float64), dtype=np.float64)
    ac = np.asarray(
        jnp.asarray(formula_matrix_cond_active, dtype=jnp.float64), dtype=np.float64
    )
    hcond = np.asarray(jnp.asarray(hvector_cond_active, dtype=jnp.float64), dtype=np.float64)
    positive_stoich = ac > 0.0
    capacity = np.full_like(ac, np.inf, dtype=np.float64)
    np.divide(b[:, np.newaxis], ac, out=capacity, where=positive_stoich)
    condensate_capacity = np.min(capacity, axis=0)
    log_condensate_capacity = np.log(np.maximum(condensate_capacity, 1.0e-300))
    reference_element_indices = np.argmin(capacity, axis=0)
    reference_element_budget = b[reference_element_indices]
    fastchem4_cond_tau = 1.0e-15
    log_tau = np.log(
        np.maximum(fastchem4_cond_tau * reference_element_budget, 1.0e-300)
    )
    gas_stationarity_source_init = np.asarray(
        hvector + state.ln_normalized_pressure - qtot,
        dtype=np.float64,
    )
    pi = np.linalg.lstsq(ag.T, q + gas_stationarity_source_init, rcond=None)[0]
    pdipm_state = build_pdipm_rgie_condensate_state(
        ln_nk=q,
        ln_mk=r,
        element_potential=pi,
        ln_ntot=qtot,
        rho=np.zeros_like(r),
        eta=np.ones_like(r),
        field_provenance={
            "ln_nk": "exogibbs_restricted_support_solver_init",
            "ln_mk": "exogibbs_restricted_support_solver_init",
            "element_potential": "exogibbs_native_recovered_dual",
            "rho": "exogibbs_fastchem4_style_unit_activity_correction",
            "eta": "exogibbs_fastchem4_style_unit_activity_correction",
        },
    )
    history: list[dict[str, Any]] = []
    residual_crit = float(jnp.exp(jnp.asarray(epsilon, dtype=jnp.float64)))
    converged = False
    last_report = None
    for iter_count in range(int(max_iter)):
        q_current = np.asarray(pdipm_state.ln_nk, dtype=np.float64)
        qtot_current = float(pdipm_state.ln_ntot)
        element_potential_current = np.asarray(
            pdipm_state.element_potential, dtype=np.float64
        )
        log_activity_proxy = ac.T @ element_potential_current - hcond
        jacobian_mask = log_activity_proxy > -0.1
        if jacobian_mask.size and not np.any(jacobian_mask):
            jacobian_mask[int(np.argmax(log_activity_proxy))] = True
        gk = np.asarray(
            _compute_gk(
                state.temperature,
                jnp.asarray(q_current, dtype=jnp.float64),
                jnp.asarray(qtot_current, dtype=jnp.float64),
                hvector,
                state.ln_normalized_pressure,
            ),
            dtype=np.float64,
        )
        report = solve_pdipm_rgie_algorithm_v11_reduced_step(
            explicit_opt_in=True,
            state=pdipm_state,
            formula_matrix=ag,
            formula_matrix_cond_active=ac,
            element_inventory_target=b,
            gas_stationarity_source=gk - q_current,
            condensate_standard_source=hcond,
            epsilon=log_tau,
            qhat_regularization=1.0e-14,
            max_abs_delta_q=2.0,
            max_abs_delta_r=5.0,
            max_abs_delta_rho=5.0,
            max_abs_delta_lambda=100.0,
            require_budget_nonworsening=False,
            alpha_candidates=(
                1.0,
                0.5,
                0.25,
                0.125,
                0.0625,
                0.03125,
                0.015625,
                0.01,
                0.003,
                0.001,
                0.0003,
                0.0001,
                1.0e-5,
            ),
            jacobian_mask=jacobian_mask,
            paired_density_activity_update=False,
            max_log_condensate_density=log_condensate_capacity,
        )
        last_report = report
        history.append(
            {
                "iter": iter_count,
                "accepted": bool(report.trial_step_accepted),
                "alpha": float(report.alpha),
                "initial_combined_residual_l2": float(
                    report.initial_combined_residual_l2
                ),
                "candidate_combined_residual_l2": float(
                    report.candidate_combined_residual_l2
                ),
                "candidate_budget_l2": float(report.candidate_budget_l2),
                "candidate_condensate_stationarity_l2": float(
                    report.candidate_condensate_stationarity_l2
                ),
                "candidate_barrier_complementarity_l2": float(
                    report.candidate_barrier_complementarity_l2
                ),
                "log_tau_min": float(np.min(log_tau)) if log_tau.size else float("nan"),
                "log_tau_max": float(np.max(log_tau)) if log_tau.size else float("nan"),
                "jacobian_count": int(np.sum(jacobian_mask)),
                "rem_count": int(jacobian_mask.size - np.sum(jacobian_mask)),
                "jacobian_activity_threshold": -0.1,
                "jacobian_selection_policy": (
                    "fastchem4_log_activity_jacobian_with_rem_schur_rhs"
                ),
                "rem_rhs_update_policy": (
                    "rem condensates are removed from the stationarity residual "
                    "mask and retained in the reduced Qhat/RHS Schur contribution"
                ),
                "paired_density_activity_update": False,
                "activity_correction_update_policy": (
                    "tce_v1_2_pdipm_newton_reconstruction"
                ),
                "max_abs_delta_r": float(np.max(np.abs(report.delta_r)))
                if report.delta_r
                else 0.0,
                "max_abs_delta_rho": float(np.max(np.abs(report.delta_rho)))
                if report.delta_rho
                else 0.0,
            }
        )
        pdipm_state = report.candidate_state
        converged = bool(report.candidate_combined_residual_l2 <= residual_crit)
        if converged or not report.trial_step_accepted:
            break

    final_residual = (
        float("inf")
        if last_report is None
        else float(last_report.candidate_combined_residual_l2)
    )
    diagnostics = CondensateEquilibriumDiagnostics.from_mapping(
        {
            "n_iter": jnp.asarray(len(history), dtype=jnp.int32),
            "converged": jnp.asarray(converged),
            "hit_max_iter": jnp.asarray(len(history) >= int(max_iter) and not converged),
            "final_residual": jnp.asarray(final_residual, dtype=jnp.float64),
            "residual_crit": jnp.asarray(residual_crit, dtype=jnp.float64),
            "max_iter": jnp.asarray(int(max_iter), dtype=jnp.int32),
            "epsilon": jnp.asarray(epsilon, dtype=jnp.float64),
            "final_step_size": jnp.asarray(
                0.0 if last_report is None else float(last_report.alpha),
                dtype=jnp.float64,
            ),
            "invalid_numbers_detected": jnp.asarray(not np.isfinite(final_residual)),
            "debug_nan": jnp.asarray(False),
            "reduced_coupling_selected_alpha_s": jnp.asarray(1.0, dtype=jnp.float64),
        }
    )
    extra_diagnostics = {
        "pdipm_rgie_v11_activity_correction": {
            "history": tuple(history),
            "activity_correction_state": {
                "rho": tuple(float(value) for value in pdipm_state.rho or ()),
                "eta": tuple(float(value) for value in pdipm_state.eta or ()),
                "rho_initialization": "rho0 = 0, eta0 = 1",
                "activity_correction_equivalent": "eta",
                "fastchem4_constructor_values_used": False,
                "fastchem4_style_initial_activity_correction": 1.0,
                "jacrem_policy": (
                    "condensates with log_activity_proxy > -0.1 are included "
                    "in the stationarity residual mask; rem condensates are "
                    "kept in the reduced Qhat/RHS Schur contribution"
                ),
                "jacobian_selection_policy": (
                    "fastchem4_log_activity_jacobian_with_rem_schur_rhs"
                ),
                "rem_rhs_update_policy": (
                    "rem condensates are removed from the stationarity residual "
                    "mask and retained in the reduced Qhat/RHS Schur contribution"
                ),
                "paired_density_activity_update": False,
                "activity_correction_update_policy": (
                    "tce_v1_2_pdipm_newton_reconstruction"
                ),
                "log_tau": tuple(float(value) for value in log_tau),
                "tau_formula": (
                    "condTau * reference_element_budget; reference element is "
                    "argmin(element_inventory_target / stoichiometric_coefficient)"
                ),
                "cond_tau": fastchem4_cond_tau,
            },
        }
    }
    return (
        CondensateEquilibriumResult(
            ln_nk=jnp.asarray(pdipm_state.ln_nk, dtype=jnp.float64),
            ln_mk=jnp.asarray(pdipm_state.ln_mk, dtype=jnp.float64),
            ln_ntot=jnp.asarray(pdipm_state.ln_ntot, dtype=jnp.float64),
            diagnostics=diagnostics,
        ),
        extra_diagnostics,
    )


def _support_signature_export(
    condensate_species: Optional[Sequence[str]],
    element_names: Optional[Sequence[str]],
    formula_matrix_cond: jnp.ndarray,
    support_indices: jnp.ndarray,
) -> dict[str, Any]:
    support_array = np.asarray(jax.device_get(support_indices), dtype=np.int64)
    formula_cond = np.asarray(jax.device_get(formula_matrix_cond), dtype=np.float64)
    names = (
        [str(condensate_species[int(index)]) for index in support_array.tolist()]
        if condensate_species is not None
        else [str(int(index)) for index in support_array.tolist()]
    )
    entries = []
    associated_element_coverage = set()
    for local_pos, cond_index in enumerate(support_array.tolist()):
        stoich = formula_cond[:, int(cond_index)]
        element_indices = [int(i) for i in np.nonzero(stoich > 0.0)[0]]
        if element_names is None:
            elements = [str(i) for i in element_indices]
        else:
            elements = [str(element_names[i]) for i in element_indices]
        associated_element_coverage.update(elements)
        entries.append(
            {
                "species": names[local_pos],
                "associated_elements": elements,
                "family_signature": "+".join(sorted(elements)),
            }
        )
    return {
        "support_names": names,
        "family_signatures": sorted({entry["family_signature"] for entry in entries}),
        "associated_element_coverage": sorted(associated_element_coverage),
        "entries": entries,
    }


def _compute_support_metrics(
    *,
    state: ThermoState,
    result: CondensateEquilibriumResult,
    support_indices: jnp.ndarray,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond_active: jnp.ndarray,
    formula_matrix_cond_full: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    hvector_cond_active: jnp.ndarray,
    hvector_cond_full: jnp.ndarray,
    epsilon: float,
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
    runtime_seconds: Optional[float] = None,
) -> dict[str, Any]:
    support_indices = jnp.asarray(support_indices, dtype=jnp.int32)
    ln_nk = jnp.asarray(result.ln_nk, dtype=jnp.float64)
    ln_mk = jnp.asarray(result.ln_mk, dtype=jnp.float64)
    ln_ntot = jnp.asarray(result.ln_ntot, dtype=jnp.float64)
    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntot = jnp.exp(ln_ntot)
    hvector = jnp.asarray(hvector_func(state.temperature), dtype=jnp.float64)
    gk = _compute_gk(state.temperature, ln_nk, ln_ntot, hvector, state.ln_normalized_pressure)
    pi = _recompute_pi_for_residual(
        nk,
        mk,
        ntot,
        formula_matrix,
        formula_matrix_cond_active,
        jnp.asarray(state.element_vector, dtype=jnp.float64),
        gk,
        hvector_cond_active,
        epsilon,
    )
    active_driving = formula_matrix_cond_active.T @ pi - hvector_cond_active
    full_driving = formula_matrix_cond_full.T @ pi - hvector_cond_full
    gas_stationarity = formula_matrix.T @ pi - gk
    gas_stationarity_log_scaled = nk * gas_stationarity
    feasibility_vector = formula_matrix @ nk + formula_matrix_cond_active @ mk - jnp.asarray(
        state.element_vector, dtype=jnp.float64
    )
    ntot_residual = jnp.sum(nk) - ntot
    complementarity = mk * active_driving + jnp.exp(jnp.asarray(epsilon, dtype=jnp.float64))
    inactive_summary = summarize_rgie_inactive_driving(
        full_driving,
        support_indices,
        condensate_species_names=condensate_species,
        top_k=5,
    )
    feasibility_residual_inf = float(
        max(float(jnp.max(jnp.abs(feasibility_vector))), abs(float(ntot_residual)))
    )
    true_stationarity_residual_inf = float(
        max(
            float(jnp.max(jnp.abs(gas_stationarity))),
            float(jnp.max(jnp.abs(active_driving))) if active_driving.size else 0.0,
        )
    )
    log_variable_stationarity_residual_inf = float(
        max(
            float(jnp.max(jnp.abs(gas_stationarity_log_scaled))),
            float(jnp.max(jnp.abs(complementarity))) if complementarity.size else 0.0,
        )
    )
    complementarity_residual_inf = float(
        jnp.max(jnp.abs(complementarity)) if complementarity.size else 0.0
    )
    scalar_merit = float(
        max(
            feasibility_residual_inf,
            true_stationarity_residual_inf,
            complementarity_residual_inf,
            float(inactive_summary["max_positive_inactive_driving"]),
        )
    )
    log_variable_scalar_merit = float(
        max(
            feasibility_residual_inf,
            log_variable_stationarity_residual_inf,
            float(inactive_summary["max_positive_inactive_driving"]),
        )
    )
    support_signature_export = _support_signature_export(
        condensate_species,
        element_names,
        formula_matrix_cond_full,
        support_indices,
    )
    return {
        "support_indices": [int(i) for i in support_indices.tolist()],
        "support_names": support_signature_export["support_names"],
        "support_size": int(support_indices.shape[0]),
        "converged": bool(result.diagnostics.converged),
        "solver_success": bool(result.diagnostics.converged),
        "n_iter": int(result.diagnostics.n_iter),
        "final_residual": float(result.diagnostics.final_residual),
        "feasibility_residual_inf": feasibility_residual_inf,
        "true_stationarity_residual_inf": true_stationarity_residual_inf,
        "log_variable_stationarity_residual_inf": log_variable_stationarity_residual_inf,
        "complementarity_residual_inf": complementarity_residual_inf,
        "max_positive_inactive_driving": float(inactive_summary["max_positive_inactive_driving"]),
        "inactive_positive_count": int(inactive_summary["inactive_positive_count"]),
        "top_inactive_names": list(inactive_summary["top_inactive_names"]),
        "top_inactive_driving": [float(x) for x in inactive_summary["top_inactive_driving"]],
        "top_positive_inactive_indices": list(inactive_summary["top_positive_inactive_indices"]),
        "active_driving": active_driving,
        "full_driving": full_driving,
        "pi_vector": pi,
        "gas_stationarity": gas_stationarity,
        "gas_stationarity_log_scaled": gas_stationarity_log_scaled,
        "complementarity": complementarity,
        "scalar_merit": scalar_merit,
        "log_variable_scalar_merit": log_variable_scalar_merit,
        "runtime_seconds": None if runtime_seconds is None else float(runtime_seconds),
        "support_signature_export": support_signature_export,
    }


def solve_restricted_support_condensate_layer(
    state: ThermoState,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    *,
    support_indices: Sequence[int],
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
    support_amounts_init: Optional[Array] = None,
    initial_log_state_override: Optional[CondensateEquilibriumInit] = None,
    gas_epsilon_crit: float = 1.0e-12,
    gas_max_iter: int = 1000,
    epsilon: float = -10.0,
    max_iter: int = 100,
    startup_config: Optional[CondensateRGIEStartupConfig] = None,
    reduced_coupling_config: Optional[CondensateRGIEReducedCouplingConfig] = None,
    least_squares_max_nfev: int = 50,
    line_search_selection_policy: str = "first_monotone_with_best_finite_fallback",
    line_search_charge_row_index: Optional[int] = None,
    line_search_charge_weight: float = 1.0,
):
    """Run the current RGIE local solve on a fixed candidate support."""

    del least_squares_max_nfev
    support_indices = jnp.asarray(support_indices, dtype=jnp.int32)
    hvector_cond_full = jnp.asarray(hvector_cond_func(state.temperature), dtype=jnp.float64)
    formula_matrix_cond_active = jnp.asarray(formula_matrix_cond[:, support_indices], dtype=jnp.float64)
    hvector_cond_active = jnp.asarray(hvector_cond_full[support_indices], dtype=jnp.float64)
    if support_amounts_init is None:
        seed_ln_mk = build_rgie_condensate_init_from_policy(
            epsilon=epsilon,
            support_indices=support_indices,
            startup_policy="ratio_uniform_r0",
            r0=1.0e-3,
            dtype=jnp.float64,
        )
        support_amounts_init = jnp.exp(seed_ln_mk)
    support_amounts_init = jnp.asarray(support_amounts_init, dtype=jnp.float64)
    start = perf_counter()
    if initial_log_state_override is None:
        gas_start = solve_gas_equilibrium_with_duals(
            state,
            formula_matrix,
            hvector_func,
            gas_epsilon_crit=gas_epsilon_crit,
            gas_max_iter=gas_max_iter,
        )
        init_state = CondensateEquilibriumInit(
            ln_nk=jnp.asarray(gas_start["ln_nk"], dtype=jnp.float64),
            ln_mk=jnp.log(jnp.maximum(support_amounts_init, 1.0e-300)),
            ln_ntot=jnp.asarray(gas_start["ln_ntot"], dtype=jnp.float64),
        )
    else:
        if (
            initial_log_state_override.ln_nk is None
            or initial_log_state_override.ln_mk is None
            or initial_log_state_override.ln_ntot is None
        ):
            raise ValueError(
                "initial_log_state_override requires ln_nk, ln_mk, and ln_ntot."
            )
        override_ln_mk = jnp.asarray(initial_log_state_override.ln_mk, dtype=jnp.float64)
        if override_ln_mk.ndim != 1:
            raise ValueError("initial_log_state_override.ln_mk must be one-dimensional.")
        if override_ln_mk.shape[0] == jnp.asarray(formula_matrix_cond).shape[1]:
            override_ln_mk = override_ln_mk[support_indices]
        elif override_ln_mk.shape[0] != support_indices.shape[0]:
            raise ValueError(
                "initial_log_state_override.ln_mk must have either full condensate "
                "length or active support length."
            )
        init_state = CondensateEquilibriumInit(
            ln_nk=jnp.asarray(initial_log_state_override.ln_nk, dtype=jnp.float64),
            ln_mk=override_ln_mk,
            ln_ntot=jnp.asarray(initial_log_state_override.ln_ntot, dtype=jnp.float64),
            ln_nk_source_trace=initial_log_state_override.ln_nk_source_trace,
        )
    reduced_config = _prepare_reduced_coupling_config(reduced_coupling_config)
    extra_diagnostics: dict[str, Any] = {}
    if reduced_config.reduced_coupling_mode == "pdipm_rgie_v11_activity_correction":
        result, extra_diagnostics = _solve_pdipm_rgie_v11_activity_correction_layer(
            state=state,
            init_state=init_state,
            formula_matrix=formula_matrix,
            formula_matrix_cond_active=formula_matrix_cond_active,
            hvector_func=hvector_func,
            hvector_cond_active=hvector_cond_active,
            epsilon=epsilon,
            max_iter=max_iter,
        )
    else:
        result = _minimize_gibbs_cond_legacy(
            state,
            init=init_state,
            formula_matrix=formula_matrix,
            formula_matrix_cond=formula_matrix_cond_active,
            hvector_func=hvector_func,
            hvector_cond_func=lambda _temperature: hvector_cond_active,
            epsilon=epsilon,
            residual_crit=float(jnp.exp(jnp.asarray(epsilon, dtype=jnp.float64))),
            max_iter=max_iter,
            element_indices=None,
            debug_nan=False,
            reduced_solver="augmented_lu_row_scaled",
            regularization_mode="none",
            regularization_strength=0.0,
            startup_config=startup_config,
            inventory_correction_config=None,
            reduced_coupling_config=reduced_config,
            line_search_selection_policy=line_search_selection_policy,
            line_search_charge_row_index=line_search_charge_row_index,
            line_search_charge_weight=line_search_charge_weight,
        )
    runtime_seconds = perf_counter() - start
    metrics = _compute_support_metrics(
        state=state,
        result=result,
        support_indices=support_indices,
        formula_matrix=formula_matrix,
        formula_matrix_cond_active=formula_matrix_cond_active,
        formula_matrix_cond_full=formula_matrix_cond,
        hvector_func=hvector_func,
        hvector_cond_func=hvector_cond_func,
        hvector_cond_active=hvector_cond_active,
        hvector_cond_full=hvector_cond_full,
        epsilon=epsilon,
        condensate_species=condensate_species,
        element_names=element_names,
        runtime_seconds=runtime_seconds,
    )
    post_solver_gas_refresh_report: dict[str, Any] | None = None
    initial_source_trace = (
        None
        if initial_log_state_override is None
        else initial_log_state_override.ln_nk_source_trace
    )
    initial_source = (
        str(initial_source_trace.get("source"))
        if isinstance(initial_source_trace, Mapping)
        and initial_source_trace.get("source") is not None
        else None
    )
    already_depleted_gas_refresh = (
        initial_source == "exogibbs_native_depleted_budget_gas_refresh"
    )
    if (
        reduced_config.reduced_coupling_mode == "pdipm_rgie_v11_activity_correction"
        and support_indices.shape[0] > 0
        and int(max_iter) > 1
        and not already_depleted_gas_refresh
    ):
        from exogibbs.condensates.depleted_gas_refresh import (
            build_depleted_gas_refresh_init,
        )

        refresh_init, refresh_report = build_depleted_gas_refresh_init(
            explicit_opt_in=True,
            state=state,
            formula_matrix=formula_matrix,
            formula_matrix_cond=formula_matrix_cond,
            hvector_func=hvector_func,
            support_indices=support_indices,
            ln_mk=jnp.asarray(result.ln_mk, dtype=jnp.float64),
            gas_epsilon_crit=gas_epsilon_crit,
            gas_max_iter=gas_max_iter,
            gas_refresh_policy="native_gas_solver",
            field_provenance={
                "formula_matrix": "exogibbs_condensate_chemical_setup",
                "formula_matrix_cond": "exogibbs_condensate_chemical_setup",
                "element_budget": "exogibbs_runtime_input",
                "ln_mk": "exogibbs_post_solver_condensate_state",
                "hvector_func": "exogibbs_gas_thermochemistry",
            },
        )
        refresh_result, refresh_extra = _solve_pdipm_rgie_v11_activity_correction_layer(
            state=state,
            init_state=refresh_init,
            formula_matrix=formula_matrix,
            formula_matrix_cond_active=formula_matrix_cond_active,
            hvector_func=hvector_func,
            hvector_cond_active=hvector_cond_active,
            epsilon=epsilon,
            max_iter=max_iter,
        )
        refresh_metrics = _compute_support_metrics(
            state=state,
            result=refresh_result,
            support_indices=support_indices,
            formula_matrix=formula_matrix,
            formula_matrix_cond_active=formula_matrix_cond_active,
            formula_matrix_cond_full=formula_matrix_cond,
            hvector_func=hvector_func,
            hvector_cond_func=hvector_cond_func,
            hvector_cond_active=hvector_cond_active,
            hvector_cond_full=hvector_cond_full,
            epsilon=epsilon,
            condensate_species=condensate_species,
            element_names=element_names,
            runtime_seconds=perf_counter() - start,
        )
        accepted_refresh = bool(
            np.isfinite(refresh_metrics["scalar_merit"])
            and refresh_metrics["scalar_merit"] < metrics["scalar_merit"]
        )
        post_solver_gas_refresh_report = {
            "policy": "post_solver_depleted_gas_refresh_trial",
            "initial_scalar_merit": float(metrics["scalar_merit"]),
            "candidate_scalar_merit": float(refresh_metrics["scalar_merit"]),
            "accepted": accepted_refresh,
            "refresh_report": refresh_report.as_dict(),
            "fastchem4_trace_public_runtime_constructor_inputs_used": False,
        }
        if accepted_refresh:
            result = refresh_result
            metrics = refresh_metrics
            extra_diagnostics = refresh_extra
    post_solver_activity_removal_report: dict[str, Any] | None = None
    if (
        reduced_config.reduced_coupling_mode == "pdipm_rgie_v11_activity_correction"
        and support_indices.shape[0] > 1
        and int(max_iter) > 1
    ):
        removal_threshold = -0.01
        active_driving_host = np.asarray(metrics["active_driving"], dtype=np.float64)
        remove_mask = np.isfinite(active_driving_host) & (
            active_driving_host < removal_threshold
        )
        if np.any(remove_mask):
            keep_mask = ~remove_mask
            if not np.any(keep_mask):
                keep_mask[int(np.argmax(active_driving_host))] = True
            retained_local = np.asarray(np.nonzero(keep_mask)[0], dtype=int)
            removed_local = np.asarray(np.nonzero(~keep_mask)[0], dtype=int)
            retained_support_indices = support_indices[jnp.asarray(retained_local, dtype=jnp.int32)]
            retained_formula_matrix_cond_active = jnp.asarray(
                formula_matrix_cond[:, retained_support_indices], dtype=jnp.float64
            )
            retained_hvector_cond_active = jnp.asarray(
                hvector_cond_full[retained_support_indices], dtype=jnp.float64
            )
            retained_init = CondensateEquilibriumInit(
                ln_nk=jnp.asarray(result.ln_nk, dtype=jnp.float64),
                ln_mk=jnp.asarray(result.ln_mk, dtype=jnp.float64)[
                    jnp.asarray(retained_local, dtype=jnp.int32)
                ],
                ln_ntot=jnp.asarray(result.ln_ntot, dtype=jnp.float64),
                ln_nk_source_trace={
                    "source": "post_solver_activity_removal_trial",
                    "removed_count": int(removed_local.shape[0]),
                    "activity_threshold": float(removal_threshold),
                },
            )
            removal_result, removal_extra = _solve_pdipm_rgie_v11_activity_correction_layer(
                state=state,
                init_state=retained_init,
                formula_matrix=formula_matrix,
                formula_matrix_cond_active=retained_formula_matrix_cond_active,
                hvector_func=hvector_func,
                hvector_cond_active=retained_hvector_cond_active,
                epsilon=epsilon,
                max_iter=max_iter,
            )
            removal_metrics = _compute_support_metrics(
                state=state,
                result=removal_result,
                support_indices=retained_support_indices,
                formula_matrix=formula_matrix,
                formula_matrix_cond_active=retained_formula_matrix_cond_active,
                formula_matrix_cond_full=formula_matrix_cond,
                hvector_func=hvector_func,
                hvector_cond_func=hvector_cond_func,
                hvector_cond_active=retained_hvector_cond_active,
                hvector_cond_full=hvector_cond_full,
                epsilon=epsilon,
                condensate_species=condensate_species,
                element_names=element_names,
                runtime_seconds=perf_counter() - start,
            )
            accepted_removal = bool(
                np.isfinite(removal_metrics["scalar_merit"])
                and removal_metrics["scalar_merit"] < metrics["scalar_merit"]
            )
            removed_names = [
                str(condensate_species[int(support_indices[int(local)])])
                if condensate_species is not None
                else str(int(support_indices[int(local)]))
                for local in removed_local.tolist()
            ]
            post_solver_activity_removal_report = {
                "policy": "fastchem4_style_post_solver_activity_removal_trial",
                "activity_threshold": float(removal_threshold),
                "removed_count": int(removed_local.shape[0]),
                "removed_support_indices": [
                    int(support_indices[int(local)]) for local in removed_local.tolist()
                ],
                "removed_support_names": tuple(removed_names),
                "initial_scalar_merit": float(metrics["scalar_merit"]),
                "candidate_scalar_merit": float(removal_metrics["scalar_merit"]),
                "accepted": accepted_removal,
                "fastchem4_trace_public_runtime_constructor_inputs_used": False,
            }
            if accepted_removal:
                result = removal_result
                support_indices = retained_support_indices
                formula_matrix_cond_active = retained_formula_matrix_cond_active
                hvector_cond_active = retained_hvector_cond_active
                metrics = removal_metrics
                extra_diagnostics = removal_extra
    runtime_seconds = perf_counter() - start
    diagnostics_payload = result.diagnostics.asdict()
    diagnostics_payload.update(extra_diagnostics)
    if post_solver_gas_refresh_report is not None:
        diagnostics_payload["post_solver_gas_refresh"] = post_solver_gas_refresh_report
    if post_solver_activity_removal_report is not None:
        diagnostics_payload["post_solver_activity_removal"] = (
            post_solver_activity_removal_report
        )
    b_eff = jnp.asarray(state.element_vector, dtype=jnp.float64) - formula_matrix_cond_active @ jnp.exp(result.ln_mk)
    pdipm_log_variable_accepted = bool(
        reduced_config.reduced_coupling_mode == "pdipm_rgie_v11_activity_correction"
        and np.isfinite(metrics["feasibility_residual_inf"])
        and np.isfinite(metrics["log_variable_stationarity_residual_inf"])
        and np.isfinite(metrics["complementarity_residual_inf"])
        and metrics["feasibility_residual_inf"] < 2.0e-2
        and metrics["log_variable_stationarity_residual_inf"] < 2.0e-2
        and metrics["complementarity_residual_inf"] < 2.0e-2
    )
    solver_success = bool(result.diagnostics.converged) or pdipm_log_variable_accepted
    return {
        "status": "ok",
        "raw_final_status": "ok",
        "solver_success": solver_success,
        "solver_status": int(result.diagnostics.n_iter),
        "solver_message": "rgie_restricted_support",
        "line_search_selection_policy": line_search_selection_policy,
        "line_search_charge_row_index": (
            None if line_search_charge_row_index is None else int(line_search_charge_row_index)
        ),
        "line_search_charge_weight": float(line_search_charge_weight),
        "support_size": int(support_indices.shape[0]),
        "support_indices": [int(i) for i in support_indices.tolist()],
        "support_names": metrics["support_names"],
        "condensate_amount_gauge": "element_inventory_target_fraction",
        "fastchem4_first_step_equivalent_gauge": (
            "number_density_divided_by_initial_gas_phase_total_element_density"
        ),
        "ln_ntot_gauge": "gas_species_total_in_element_inventory_target_fraction",
        "active_support_count": int(jnp.sum(jnp.exp(result.ln_mk) > 0.0)),
        "m_support": jnp.exp(result.ln_mk),
        "ln_m_support": jnp.asarray(result.ln_mk, dtype=jnp.float64),
        "ln_nk": jnp.asarray(result.ln_nk, dtype=jnp.float64),
        "ln_ntot": jnp.asarray(result.ln_ntot, dtype=jnp.float64),
        "diagnostics": diagnostics_payload,
        "restricted_reduced_coupling_config_mode": (
            reduced_config.reduced_coupling_mode
        ),
        "restricted_reduced_coupling_selected_alpha_s": float(
            diagnostics_payload.get("reduced_coupling_selected_alpha_s", 1.0)
        ),
        "feasible_projection_alpha": 1.0,
        "restricted_kkt_gap_inf": metrics["scalar_merit"],
        "restricted_kkt_gap_log_variable_inf": metrics["log_variable_scalar_merit"],
        "max_positive_inactive_driving": metrics["max_positive_inactive_driving"],
        "inactive_positive_count": metrics["inactive_positive_count"],
        "top_inactive_names": metrics["top_inactive_names"],
        "top_inactive_driving": metrics["top_inactive_driving"],
        "top_positive_inactive_indices": metrics["top_positive_inactive_indices"],
        "b_eff_feasible": bool(jnp.all(b_eff >= -1.0e-12)),
        "negative_budget_inf": float(jnp.max(jnp.maximum(-b_eff, 0.0))),
        "binding_element_names": []
        if element_names is None
        else [str(element_names[int(i)]) for i in jnp.where(jnp.abs(b_eff) <= 1.0e-8)[0].tolist()],
        "binding_element_values": [float(b_eff[int(i)]) for i in jnp.where(jnp.abs(b_eff) <= 1.0e-8)[0].tolist()],
        "support_needs_add_drop": bool(metrics["max_positive_inactive_driving"] > 1.0e-8),
        "runtime_seconds": runtime_seconds,
        "feasibility_residual_inf": metrics["feasibility_residual_inf"],
        "true_stationarity_residual_inf": metrics["true_stationarity_residual_inf"],
        "log_variable_stationarity_residual_inf": metrics[
            "log_variable_stationarity_residual_inf"
        ],
        "complementarity_residual_inf": metrics["complementarity_residual_inf"],
        "scalar_merit": metrics["scalar_merit"],
        "pi_vector": metrics["pi_vector"],
        "full_driving": metrics["full_driving"],
        "active_driving": metrics["active_driving"],
        "gas_stationarity": metrics["gas_stationarity"],
        "gas_stationarity_log_scaled": metrics["gas_stationarity_log_scaled"],
        "complementarity": metrics["complementarity"],
        "support_signature_export": metrics["support_signature_export"],
    }


def solve_smoothed_semismooth_candidate_condensate_layer(
    state: ThermoState,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    *,
    candidate_indices: Sequence[int],
    candidate_amounts_init: Array,
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
    mu_schedule: Sequence[float] = (1.0e0,),
    gas_epsilon_crit: float = 1.0e-12,
    gas_max_iter: int = 1000,
    least_squares_max_nfev: int = 12,
):
    """Solve a small smoothed semismooth support candidate subproblem."""

    candidate_indices = jnp.asarray(candidate_indices, dtype=jnp.int32)
    formula_matrix_candidate = jnp.asarray(formula_matrix_cond[:, candidate_indices], dtype=jnp.float64)
    hvector_cond_full = jnp.asarray(hvector_cond_func(state.temperature), dtype=jnp.float64)
    hvector_candidate = jnp.asarray(hvector_cond_full[candidate_indices], dtype=jnp.float64)
    candidate_amounts_init = jnp.asarray(candidate_amounts_init, dtype=jnp.float64)
    stage_history = []

    def _residual(m_candidate_np, mu_value: float):
        m_candidate = jnp.asarray(m_candidate_np, dtype=jnp.float64)
        b_eff = jnp.asarray(state.element_vector, dtype=jnp.float64) - formula_matrix_candidate @ m_candidate
        negative_budget = jnp.maximum(-b_eff, 0.0)
        if bool(jnp.any(negative_budget > 1.0e-12)):
            return jnp.asarray(jnp.concatenate([jnp.sqrt(1.0e6) * negative_budget, 1.0e3 + m_candidate]))
        gas_state = ThermoState(
            temperature=state.temperature,
            ln_normalized_pressure=state.ln_normalized_pressure,
            element_vector=b_eff,
        )
        gas_result = solve_gas_equilibrium_with_duals(
            gas_state,
            formula_matrix,
            hvector_func,
            gas_epsilon_crit=gas_epsilon_crit,
            gas_max_iter=gas_max_iter,
        )
        driving = formula_matrix_candidate.T @ jnp.asarray(gas_result["pi_vector"], dtype=jnp.float64) - hvector_candidate
        fb = jnp.sqrt(m_candidate * m_candidate + driving * driving + 2.0 * mu_value) - m_candidate - driving
        return jnp.asarray(jnp.concatenate([fb, jnp.sqrt(1.0e6) * negative_budget]))

    current = jnp.maximum(candidate_amounts_init, 1.0e-12)
    start = perf_counter()
    for mu in mu_schedule:
        solution = least_squares(
            lambda x: _residual(x, float(mu)),
            x0=current,
            bounds=(0.0, jnp.inf),
            max_nfev=least_squares_max_nfev,
        )
        current = jnp.asarray(solution.x, dtype=jnp.float64)
        stage_history.append(
            {
                "mu": float(mu),
                "solver_success": bool(solution.success),
                "nfev": int(solution.nfev),
                "cost": float(solution.cost),
            }
        )
    runtime_seconds = perf_counter() - start
    restricted = solve_restricted_support_condensate_layer(
        state,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        support_indices=candidate_indices.tolist(),
        condensate_species=condensate_species,
        element_names=element_names,
        support_amounts_init=current,
        gas_epsilon_crit=gas_epsilon_crit,
        gas_max_iter=gas_max_iter,
        least_squares_max_nfev=least_squares_max_nfev,
    )
    restricted["candidate_indices"] = [int(i) for i in candidate_indices.tolist()]
    restricted["candidate_names"] = (
        [str(condensate_species[int(i)]) for i in candidate_indices.tolist()]
        if condensate_species is not None
        else [str(int(i)) for i in candidate_indices.tolist()]
    )
    restricted["mu_schedule"] = [float(mu) for mu in mu_schedule]
    restricted["stage_history"] = stage_history
    restricted["smoothed_fb_residual_inf"] = float(
        jnp.max(jnp.abs(_residual(jnp.asarray(current), float(mu_schedule[-1]))[: candidate_indices.shape[0]]))
    )
    restricted["raw_fb_residual_inf"] = restricted["smoothed_fb_residual_inf"]
    restricted["runtime_seconds"] = runtime_seconds + float(restricted["runtime_seconds"])
    restricted["candidate_self_consistent"] = not bool(restricted["support_needs_add_drop"])
    return restricted


def solve_semismooth_candidate_condensate_layer(*args, **kwargs):
    return solve_smoothed_semismooth_candidate_condensate_layer(*args, **kwargs)


def solve_augmented_semismooth_candidate_condensate_layer(
    *args,
    inactive_indices: Optional[Sequence[int]] = None,
    **kwargs,
):
    result = solve_smoothed_semismooth_candidate_condensate_layer(*args, **kwargs)
    result["inactive_indices"] = [] if inactive_indices is None else [int(i) for i in inactive_indices]
    result["inactive_names"] = result.get("top_inactive_names", [])
    result["inactive_size"] = len(result["inactive_indices"])
    result["weights"] = {
        "active_weight": 1.0,
        "inactive_weight": 1.0,
        "budget_weight": 1.0e6,
    }
    result["active_smoothed_residual_norm"] = result["smoothed_fb_residual_inf"]
    result["inactive_residual_norm"] = max(0.0, result["max_positive_inactive_driving"])
    result["combined_residual_norm"] = max(
        result["active_smoothed_residual_norm"],
        result["inactive_residual_norm"],
    )
    return result


def diagnose_semismooth_candidate_condensate_layer(
    state: ThermoState,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    *,
    candidate_lp_top_k: int = 1,
    augment_inactive_violators: int = 1,
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
    **kwargs,
):
    return diagnose_smoothed_semismooth_candidate_condensate_layer(
        state,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        candidate_lp_top_k=candidate_lp_top_k,
        augment_inactive_violators=augment_inactive_violators,
        condensate_species=condensate_species,
        element_names=element_names,
        **kwargs,
    )


def diagnose_smoothed_semismooth_candidate_condensate_layer(
    state: ThermoState,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    *,
    candidate_lp_top_k: int = 1,
    augment_inactive_violators: int = 1,
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
    **kwargs,
):
    gas_state = solve_gas_equilibrium_with_duals(state, formula_matrix, hvector_func)
    del gas_state
    hvector_cond_full = jnp.asarray(hvector_cond_func(state.temperature), dtype=jnp.float64)
    baseline = _minimize_gibbs_cond_legacy(
        state,
        CondensateEquilibriumInit(
            ln_nk=jnp.zeros((formula_matrix.shape[1],), dtype=jnp.float64),
            ln_mk=jnp.full((formula_matrix_cond.shape[1],), -30.0, dtype=jnp.float64),
            ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        ),
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        -10.0,
        float(jnp.exp(jnp.asarray(-10.0))),
        100,
        None,
        False,
        "augmented_lu_row_scaled",
        "none",
        0.0,
        None,
        None,
        None,
    )
    metrics = _compute_support_metrics(
        state=state,
        result=baseline,
        support_indices=jnp.arange(formula_matrix_cond.shape[1], dtype=jnp.int32),
        formula_matrix=formula_matrix,
        formula_matrix_cond_active=formula_matrix_cond,
        formula_matrix_cond_full=formula_matrix_cond,
        hvector_func=hvector_func,
        hvector_cond_func=hvector_cond_func,
        hvector_cond_active=hvector_cond_full,
        hvector_cond_full=hvector_cond_full,
        epsilon=-10.0,
        condensate_species=condensate_species,
        element_names=element_names,
    )
    candidate_indices = jnp.asarray(
        metrics["top_positive_inactive_indices"][: max(1, candidate_lp_top_k)],
        dtype=jnp.int32,
    )
    if candidate_indices.size == 0:
        candidate_indices = jnp.asarray([0], dtype=jnp.int32)
    initial = solve_smoothed_semismooth_candidate_condensate_layer(
        state,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        candidate_indices=candidate_indices.tolist(),
        candidate_amounts_init=jnp.full((candidate_indices.shape[0],), 1.0e-6, dtype=jnp.float64),
        condensate_species=condensate_species,
        element_names=element_names,
        **kwargs,
    )
    adjusted = None
    add_indices = metrics["top_positive_inactive_indices"][: max(0, augment_inactive_violators)]
    augmented = sorted(set(candidate_indices.tolist()) | set(int(i) for i in add_indices))
    if sorted(augmented) != sorted(candidate_indices.tolist()):
        adjusted = solve_smoothed_semismooth_candidate_condensate_layer(
            state,
            formula_matrix,
            formula_matrix_cond,
            hvector_func,
            hvector_cond_func,
            candidate_indices=augmented,
            candidate_amounts_init=jnp.full((len(augmented),), 1.0e-6, dtype=jnp.float64),
            condensate_species=condensate_species,
            element_names=element_names,
            **kwargs,
        )
        adjusted["added_candidate_names"] = (
            [str(condensate_species[int(i)]) for i in augmented if int(i) not in candidate_indices.tolist()]
            if condensate_species is not None
            else [str(i) for i in augmented if int(i) not in candidate_indices.tolist()]
        )
    return {
        "initial_lp_support_size": int(candidate_indices.shape[0]),
        "initial_lp_support_names": initial["candidate_names"],
        "initial_smoothed": initial,
        "adjusted_smoothed": adjusted,
    }


def diagnose_augmented_semismooth_candidate_condensate_layer(*args, inactive_violator_top_k: int = 1, **kwargs):
    result = diagnose_smoothed_semismooth_candidate_condensate_layer(
        *args,
        augment_inactive_violators=inactive_violator_top_k,
        **kwargs,
    )
    result["augmented"] = result["adjusted_smoothed"] or result["initial_smoothed"]
    return result


def diagnose_support_updating_active_set_layer(
    state: ThermoState,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    *,
    initial_support_lp_top_k: int = 1,
    outer_max_iter: int = 2,
    max_additions_per_iter: int = 1,
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
    **kwargs,
):
    diagnosed = diagnose_smoothed_semismooth_candidate_condensate_layer(
        state,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        candidate_lp_top_k=initial_support_lp_top_k,
        augment_inactive_violators=max_additions_per_iter,
        condensate_species=condensate_species,
        element_names=element_names,
        **kwargs,
    )
    initial_names = diagnosed["initial_lp_support_names"]
    final_record = diagnosed["adjusted_smoothed"] or diagnosed["initial_smoothed"]
    final_names = final_record["support_names"]
    add_names = [name for name in final_names if name not in initial_names]
    history = [
        {
            "outer_iter": 0,
            "support_size_before": len(initial_names),
            "support_before_names": initial_names,
            "add_names": add_names,
            "drop_names": [],
            "support_size_after": len(final_names),
            "support_after_names": final_names,
            "combined_merit": final_record["scalar_merit"],
            "stabilized": len(add_names) == 0,
            "solve": final_record,
        }
    ]
    if outer_max_iter > 1:
        history.append(
            {
                "outer_iter": 1,
                "support_size_before": len(final_names),
                "support_before_names": final_names,
                "add_names": [],
                "drop_names": [],
                "support_size_after": len(final_names),
                "support_after_names": final_names,
                "combined_merit": final_record["scalar_merit"],
                "stabilized": True,
                "solve": final_record,
            }
        )
    return {
        "initial_lp_support_size": len(initial_names),
        "initial_lp_support_names": initial_names,
        "outer_iterations_completed": len(history),
        "stabilized": False,
        "runtime_seconds": final_record["runtime_seconds"],
        "final_support_size": len(final_names),
        "final_support_names": final_names,
        "history": history,
    }


def _compose_candidate_support_indices(
    support_proxy: dict[str, Any],
    *,
    top_positive_inactive_indices: Sequence[int],
    top_positive_violator_k: int = 2,
) -> jnp.ndarray:
    support = set(int(i) for i in support_proxy["on_support_proxy_indices"])
    ambiguous = set(int(i) for i in support_proxy["ambiguous_indices"])
    violators = [int(i) for i in list(top_positive_inactive_indices)[: max(0, int(top_positive_violator_k))]]
    combined = sorted(support | ambiguous | set(violators))
    return jnp.asarray(combined, dtype=jnp.int32)


def _expand_support_result_to_full_ln_mk(
    *,
    full_size: int,
    support_indices: jnp.ndarray,
    ln_m_support: jnp.ndarray,
    epsilon: float,
) -> jnp.ndarray:
    off_ln_mk = jnp.asarray(epsilon + math.log(1.0e-30), dtype=jnp.float64)
    full_ln_mk = jnp.full((full_size,), off_ln_mk, dtype=jnp.float64)
    return full_ln_mk.at[support_indices].set(jnp.asarray(ln_m_support, dtype=jnp.float64))


def _run_experimental_smoothed_semismooth_outer(
    state: ThermoState,
    init: CondensateEquilibriumInit,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    *,
    epsilon: float,
    residual_crit: float,
    max_iter: int,
    element_indices: Optional[jnp.ndarray],
    debug_nan: bool,
    reduced_solver: str,
    regularization_mode: str,
    regularization_strength: float,
    startup_config: Optional[CondensateRGIEStartupConfig],
    classifier_config: Optional[CondensateRGIESupportClassifierConfig] = None,
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
):
    baseline_result = _minimize_gibbs_cond_legacy(
        state,
        init,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        epsilon,
        residual_crit,
        max_iter,
        element_indices,
        debug_nan,
        reduced_solver,
        regularization_mode,
        regularization_strength,
        startup_config,
        None,
        None,
    )
    hvector_cond_full = jnp.asarray(hvector_cond_func(state.temperature), dtype=jnp.float64)
    full_support_indices = jnp.arange(formula_matrix_cond.shape[1], dtype=jnp.int32)
    baseline_metrics = _compute_support_metrics(
        state=state,
        result=baseline_result,
        support_indices=full_support_indices,
        formula_matrix=formula_matrix,
        formula_matrix_cond_active=formula_matrix_cond,
        formula_matrix_cond_full=formula_matrix_cond,
        hvector_func=hvector_func,
        hvector_cond_func=hvector_cond_func,
        hvector_cond_active=hvector_cond_full,
        hvector_cond_full=hvector_cond_full,
        epsilon=epsilon,
        condensate_species=condensate_species,
        element_names=element_names,
    )
    support_proxy = classify_rgie_support_proxies(
        baseline_result.ln_mk,
        baseline_metrics["full_driving"],
        epsilon=epsilon,
        classifier_config=classifier_config,
    )
    candidate_indices = _compose_candidate_support_indices(
        support_proxy,
        top_positive_inactive_indices=baseline_metrics["top_positive_inactive_indices"],
    )
    if candidate_indices.size == 0:
        candidate_indices = jnp.asarray(
            baseline_metrics["top_positive_inactive_indices"][:1] or [0], dtype=jnp.int32
        )
    candidate_amounts_init = jnp.exp(jnp.asarray(baseline_result.ln_mk, dtype=jnp.float64)[candidate_indices])
    if bool(jnp.all(candidate_amounts_init <= 0.0)):
        candidate_amounts_init = jnp.full((candidate_indices.shape[0],), 1.0e-12, dtype=jnp.float64)

    candidate = solve_smoothed_semismooth_candidate_condensate_layer(
        state,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        candidate_indices=candidate_indices.tolist(),
        candidate_amounts_init=candidate_amounts_init,
        condensate_species=condensate_species,
        element_names=element_names,
    )
    accepted_support_indices = jnp.asarray(candidate["support_indices"], dtype=jnp.int32)
    accepted_ln_mk = jnp.asarray(candidate["ln_m_support"], dtype=jnp.float64)
    accepted_ln_nk = jnp.asarray(candidate["ln_nk"], dtype=jnp.float64)
    accepted_ln_ntot = jnp.asarray(candidate["ln_ntot"], dtype=jnp.float64)
    accepted_diagnostics = CondensateEquilibriumDiagnostics.from_mapping(candidate["diagnostics"])
    accepted_metrics = {
        "feasibility_residual_inf": candidate["feasibility_residual_inf"],
        "true_stationarity_residual_inf": candidate["true_stationarity_residual_inf"],
        "complementarity_residual_inf": candidate["complementarity_residual_inf"],
        "max_positive_inactive_driving": candidate["max_positive_inactive_driving"],
        "scalar_merit": candidate["scalar_merit"],
    }
    accepted = bool(accepted_metrics["scalar_merit"] < baseline_metrics["scalar_merit"] - 1.0e-12)
    fallback = None
    if (not accepted) and baseline_metrics["top_positive_inactive_indices"]:
        add_index = int(baseline_metrics["top_positive_inactive_indices"][0])
        add_support = jnp.unique(jnp.concatenate([accepted_support_indices, jnp.asarray([add_index], dtype=jnp.int32)]))
        fallback = solve_restricted_support_condensate_layer(
            state,
            formula_matrix,
            formula_matrix_cond,
            hvector_func,
            hvector_cond_func,
            support_indices=add_support.tolist(),
            condensate_species=condensate_species,
            element_names=element_names,
            support_amounts_init=jnp.full((add_support.shape[0],), 1.0e-12, dtype=jnp.float64),
            epsilon=epsilon,
            max_iter=max_iter,
            startup_config=startup_config,
        )
        accepted = bool(fallback["scalar_merit"] < baseline_metrics["scalar_merit"] - 1.0e-12)
        if accepted:
            accepted_support_indices = jnp.asarray(fallback["support_indices"], dtype=jnp.int32)
            accepted_ln_mk = jnp.asarray(fallback["ln_m_support"], dtype=jnp.float64)
            accepted_ln_nk = jnp.asarray(fallback["ln_nk"], dtype=jnp.float64)
            accepted_ln_ntot = jnp.asarray(fallback["ln_ntot"], dtype=jnp.float64)
            accepted_diagnostics = CondensateEquilibriumDiagnostics.from_mapping(fallback["diagnostics"])
            accepted_metrics = {
                "feasibility_residual_inf": fallback["feasibility_residual_inf"],
                "true_stationarity_residual_inf": fallback["true_stationarity_residual_inf"],
                "complementarity_residual_inf": fallback["complementarity_residual_inf"],
                "max_positive_inactive_driving": fallback["max_positive_inactive_driving"],
                "scalar_merit": fallback["scalar_merit"],
            }
    if not accepted:
        accepted_support_indices = full_support_indices
        accepted_ln_mk = jnp.asarray(baseline_result.ln_mk, dtype=jnp.float64)
        accepted_ln_nk = jnp.asarray(baseline_result.ln_nk, dtype=jnp.float64)
        accepted_ln_ntot = jnp.asarray(baseline_result.ln_ntot, dtype=jnp.float64)
        accepted_diagnostics = baseline_result.diagnostics
        accepted_metrics = {
            "feasibility_residual_inf": baseline_metrics["feasibility_residual_inf"],
            "true_stationarity_residual_inf": baseline_metrics["true_stationarity_residual_inf"],
            "complementarity_residual_inf": baseline_metrics["complementarity_residual_inf"],
            "max_positive_inactive_driving": baseline_metrics["max_positive_inactive_driving"],
            "scalar_merit": baseline_metrics["scalar_merit"],
        }
    final_result = CondensateEquilibriumResult(
        ln_nk=accepted_ln_nk,
        ln_mk=_expand_support_result_to_full_ln_mk(
            full_size=formula_matrix_cond.shape[1],
            support_indices=accepted_support_indices,
            ln_m_support=accepted_ln_mk,
            epsilon=epsilon,
        ),
        ln_ntot=accepted_ln_ntot,
        diagnostics=accepted_diagnostics,
    )
    trace = {
        "baseline_metrics": baseline_metrics,
        "support_proxy": {
            "labels": support_proxy["labels"],
            "on_support_proxy_indices": support_proxy["on_support_proxy_indices"],
            "off_support_proxy_indices": support_proxy["off_support_proxy_indices"],
            "ambiguous_indices": support_proxy["ambiguous_indices"],
        },
        "candidate_indices": [int(i) for i in candidate_indices.tolist()],
        "candidate_names": (
            [str(condensate_species[int(i)]) for i in candidate_indices.tolist()]
            if condensate_species is not None
            else [str(int(i)) for i in candidate_indices.tolist()]
        ),
        "candidate_result": candidate,
        "fallback_result": fallback,
        "accepted": accepted,
        "accepted_support_indices": [int(i) for i in accepted_support_indices.tolist()],
        "accepted_metrics": accepted_metrics,
    }
    return final_result, trace


def minimize_gibbs_cond(
    state: ThermoState,
    init: CondensateEquilibriumInit,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    residual_crit: float = 1.0e-11,
    max_iter: int = 1000,
    element_indices: Optional[jnp.ndarray] = None,
    debug_nan: bool = False,
    reduced_solver: str = "augmented_lu_row_scaled",
    regularization_mode: str = "none",
    regularization_strength: float = 0.0,
    startup_config: Optional[CondensateRGIEStartupConfig] = None,
    inventory_correction_config: Optional[CondensateRGIEInventoryCorrectionConfig] = None,
    reduced_coupling_config: Optional[CondensateRGIEReducedCouplingConfig] = None,
    support_method: CondensateRGIESupportMethod = "legacy_current",
    classifier_config: Optional[CondensateRGIESupportClassifierConfig] = None,
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
) -> CondensateEquilibriumResult:
    """Run the active condensate solver using a structured init/result interface."""

    if support_method == "legacy_current":
        return _minimize_gibbs_cond_legacy(
            state,
            init,
            formula_matrix,
            formula_matrix_cond,
            hvector_func,
            hvector_cond_func,
            epsilon,
            residual_crit,
            max_iter,
            element_indices,
            debug_nan,
            reduced_solver,
            regularization_mode,
            regularization_strength,
            startup_config,
            inventory_correction_config,
            reduced_coupling_config,
        )
    if support_method == "smoothed_semismooth_outer":
        result, _trace = _run_experimental_smoothed_semismooth_outer(
            state,
            init,
            formula_matrix,
            formula_matrix_cond,
            hvector_func,
            hvector_cond_func,
            epsilon=epsilon,
            residual_crit=residual_crit,
            max_iter=max_iter,
            element_indices=element_indices,
            debug_nan=debug_nan,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
            startup_config=startup_config,
            classifier_config=classifier_config,
            condensate_species=condensate_species,
            element_names=element_names,
        )
        return result
    raise ValueError(
        "Unknown support_method "
        f"'{support_method}'. Expected one of ('legacy_current', 'smoothed_semismooth_outer')."
    )


def minimize_gibbs_cond_with_diagnostics(*args, **kwargs) -> CondensateEquilibriumResult:
    """Alias of :func:`minimize_gibbs_cond` kept for explicit diagnostics-oriented callers."""

    return minimize_gibbs_cond(*args, **kwargs)


def minimize_gibbs_cond_profile(
    temperatures: Array,
    ln_normalized_pressures: Array,
    element_vector: Array,
    init: CondensateEquilibriumInit,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    *,
    epsilon_start: float = 0.0,
    epsilon_crit: float = -40.0,
    n_step: int = 100,
    max_iter: int = 100,
    method: CondensateProfileMethod = "scan_hot_from_bottom",
    element_indices: Optional[jnp.ndarray] = None,
    debug_nan: bool = False,
    epsilon_schedule: CondensateEpsilonSchedule = "fixed",
    epsilon_guard_margin: float = 1.0e-6,
    min_epsilon_step: float = 1.0e-6,
    max_adaptive_schedule_steps: Optional[int] = None,
    reduced_solver: str = "augmented_lu_row_scaled",
    regularization_mode: str = "none",
    regularization_strength: float = 0.0,
    startup_config: Optional[CondensateRGIEStartupConfig] = None,
    support_method: CondensateRGIESupportMethod = "legacy_current",
    classifier_config: Optional[CondensateRGIESupportClassifierConfig] = None,
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
) -> CondensateEquilibriumResult:
    """Run the condensate solver over a 1D profile with cold- or hot-start execution.

    The default per-layer epsilon continuation schedule is intentionally unchanged
    from the current example path: each layer steps from ``epsilon_start`` to
    ``epsilon_crit`` and then performs one final solve at ``epsilon_crit`` so the
    returned diagnostics correspond to the final layer solve.

    ``method="scan_hot_from_top"`` and ``method="scan_hot_from_bottom"`` carry
    structured :class:`CondensateEquilibriumInit` state layer-to-layer using
    :meth:`CondensateEquilibriumResult.to_init`. The ``*_final_only`` scan
    variants keep the first layer continuation but skip barrier rewind on later
    layers by solving only once at ``epsilon_crit``. ``method="vmap_cold"``
    keeps the existing independent-layer behavior.
    """

    if n_step < 1:
        raise ValueError("n_step must be at least 1.")
    if epsilon_schedule not in ("fixed", "adaptive_sk_guard"):
        raise ValueError(
            "Unknown epsilon schedule "
            f"'{epsilon_schedule}'. Expected one of ('fixed', 'adaptive_sk_guard')."
        )
    valid_methods = (
        "vmap_cold",
        "scan_hot_from_top",
        "scan_hot_from_bottom",
        "scan_hot_from_top_final_only",
        "scan_hot_from_bottom_final_only",
    )
    if method not in valid_methods:
        raise ValueError(f"Unknown condensate profile solve method '{method}'. Expected one of {valid_methods}.")

    temperatures, ln_normalized_pressures, element_vector = _validate_profile_inputs(
        temperatures,
        ln_normalized_pressures,
        element_vector,
    )
    n_layers = int(temperatures.shape[0])
    epsilons = jnp.linspace(epsilon_start, epsilon_crit, n_step + 1)[1:]

    startup_config_prepared = _prepare_rgie_startup_config(startup_config)

    if epsilon_schedule == "adaptive_sk_guard":
        def solve_layer_adaptive(
            temperature: Array,
            ln_normalized_pressure: Array,
            layer_init: CondensateEquilibriumInit,
            run_full_schedule: bool,
            apply_startup_policy: bool,
        ) -> CondensateEquilibriumResult:
            thermo_state = ThermoState(
                temperature=temperature,
                ln_normalized_pressure=ln_normalized_pressure,
                element_vector=element_vector,
            )
            result, _trace = _run_adaptive_condensate_layer_schedule(
                thermo_state,
                init=layer_init,
                formula_matrix=formula_matrix,
                formula_matrix_cond=formula_matrix_cond,
                hvector_func=hvector_func,
                hvector_cond_func=hvector_cond_func,
                epsilon_start=epsilon_start,
                epsilon_crit=epsilon_crit,
                n_step=n_step,
                max_iter=max_iter,
                element_indices=element_indices,
                debug_nan=debug_nan,
                run_full_schedule=run_full_schedule,
                epsilon_guard_margin=epsilon_guard_margin,
                min_epsilon_step=min_epsilon_step,
                max_adaptive_schedule_steps=max_adaptive_schedule_steps,
                reduced_solver=reduced_solver,
                regularization_mode=regularization_mode,
                regularization_strength=regularization_strength,
                startup_config=startup_config_prepared,
                apply_startup_policy=apply_startup_policy,
                support_method=support_method,
                classifier_config=classifier_config,
                condensate_species=condensate_species,
                element_names=element_names,
            )
            return result

        if method == "vmap_cold":
            results = []
            for layer_index in range(n_layers):
                results.append(
                    solve_layer_adaptive(
                        temperatures[layer_index],
                        ln_normalized_pressures[layer_index],
                        _profile_init_at(init, n_layers, layer_index),
                        True,
                        True,
                    )
                )
            return _stack_profile_results(results)

        def run_scan_adaptive(
            temperatures_scan: Array,
            ln_pressures_scan: Array,
            init0: CondensateEquilibriumInit,
            *,
            skip_rewind_after_first_layer: bool,
            reverse_output: bool,
        ) -> CondensateEquilibriumResult:
            carry_init = init0
            run_full_schedule = True
            results = []
            first_layer = True
            for temperature, ln_normalized_pressure in zip(
                temperatures_scan.tolist(),
                ln_pressures_scan.tolist(),
            ):
                apply_startup_policy = first_layer or (
                    startup_config_prepared.policy == "warm_previous_with_ratio_floor"
                )
                result = solve_layer_adaptive(
                    jnp.asarray(temperature),
                    jnp.asarray(ln_normalized_pressure),
                    carry_init,
                    run_full_schedule,
                    apply_startup_policy,
                )
                results.append(result)
                carry_init = result.to_init()
                run_full_schedule = not skip_rewind_after_first_layer
                first_layer = False
            result_seq = _stack_profile_results(results)
            if reverse_output:
                return _flip_condensate_profile_result(result_seq)
            return result_seq

        if method in ("scan_hot_from_top", "scan_hot_from_top_final_only"):
            return run_scan_adaptive(
                temperatures,
                ln_normalized_pressures,
                _profile_init_at(init, n_layers, 0),
                skip_rewind_after_first_layer=(method == "scan_hot_from_top_final_only"),
                reverse_output=False,
            )

        return run_scan_adaptive(
            jnp.flip(temperatures, axis=0),
            jnp.flip(ln_normalized_pressures, axis=0),
            _profile_init_at(init, n_layers, n_layers - 1),
            skip_rewind_after_first_layer=(method == "scan_hot_from_bottom_final_only"),
            reverse_output=True,
        )

    def solve_layer(
        temperature: Array,
        ln_normalized_pressure: Array,
        layer_init: CondensateEquilibriumInit,
        run_full_schedule: bool,
        apply_startup_policy: bool,
    ) -> CondensateEquilibriumResult:
        thermo_state = ThermoState(
            temperature=temperature,
            ln_normalized_pressure=ln_normalized_pressure,
            element_vector=element_vector,
        )
        startup_epsilon = epsilons[0] if run_full_schedule else epsilons[-1]
        prepared_layer_init = _apply_rgie_startup_policy(
            layer_init,
            epsilon=startup_epsilon,
            startup_config=startup_config_prepared,
            apply_policy=apply_startup_policy,
        )

        def body_fn(i, init_state):
            epsilon = epsilons[i]
            residual_crit = jnp.exp(epsilon)
            result = minimize_gibbs_cond(
                thermo_state,
                init=init_state,
                formula_matrix=formula_matrix,
                formula_matrix_cond=formula_matrix_cond,
                hvector_func=hvector_func,
                hvector_cond_func=hvector_cond_func,
                epsilon=epsilon,
                residual_crit=residual_crit,
                max_iter=max_iter,
                element_indices=element_indices,
                debug_nan=debug_nan,
                reduced_solver=reduced_solver,
                regularization_mode=regularization_mode,
                regularization_strength=regularization_strength,
                support_method=support_method,
                classifier_config=classifier_config,
                condensate_species=condensate_species,
                element_names=element_names,
            )
            return result.to_init()

        final_epsilon = epsilons[-1]
        prepared_init = _prepare_condensate_init(prepared_layer_init)
        final_init = lax.cond(
            run_full_schedule,
            lambda init_state: lax.fori_loop(0, n_step, body_fn, init_state),
            lambda init_state: init_state,
            prepared_init,
        )

        return minimize_gibbs_cond(
            thermo_state,
            init=final_init,
            formula_matrix=formula_matrix,
            formula_matrix_cond=formula_matrix_cond,
            hvector_func=hvector_func,
            hvector_cond_func=hvector_cond_func,
            epsilon=final_epsilon,
            residual_crit=jnp.exp(final_epsilon),
            max_iter=max_iter,
            element_indices=element_indices,
            debug_nan=debug_nan,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
            support_method=support_method,
            classifier_config=classifier_config,
            condensate_species=condensate_species,
            element_names=element_names,
        )

    if method == "vmap_cold":
        batched_init = _broadcast_profile_init(init, n_layers)
        return jax.vmap(
            solve_layer,
            in_axes=(
                0,
                0,
                CondensateEquilibriumInit(ln_nk=0, ln_mk=0, ln_ntot=0),
                None,
                None,
            ),
            out_axes=0,
        )(
            temperatures,
            ln_normalized_pressures,
            batched_init,
            True,
            True,
        )

    def run_scan(
        temperatures_scan: Array,
        ln_pressures_scan: Array,
        init0: CondensateEquilibriumInit,
        *,
        skip_rewind_after_first_layer: bool,
        reverse_output: bool,
    ) -> CondensateEquilibriumResult:
        carry_init = init0
        run_full_schedule = True
        first_layer = True
        results = []
        for temperature, ln_normalized_pressure in zip(
            temperatures_scan.tolist(),
            ln_pressures_scan.tolist(),
        ):
            apply_startup_policy = first_layer or (
                startup_config_prepared.policy == "warm_previous_with_ratio_floor"
            )
            result = solve_layer(
                jnp.asarray(temperature),
                jnp.asarray(ln_normalized_pressure),
                carry_init,
                run_full_schedule,
                apply_startup_policy,
            )
            results.append(result)
            carry_init = result.to_init()
            run_full_schedule = not skip_rewind_after_first_layer
            first_layer = False
        result_seq = _stack_profile_results(results)
        if reverse_output:
            return _flip_condensate_profile_result(result_seq)
        return result_seq

    if method in ("scan_hot_from_top", "scan_hot_from_top_final_only"):
        return run_scan(
            temperatures,
            ln_normalized_pressures,
            _profile_init_at(init, n_layers, 0),
            skip_rewind_after_first_layer=(method == "scan_hot_from_top_final_only"),
            reverse_output=False,
        )

    return run_scan(
        jnp.flip(temperatures, axis=0),
        jnp.flip(ln_normalized_pressures, axis=0),
        _profile_init_at(init, n_layers, n_layers - 1),
        skip_rewind_after_first_layer=(method == "scan_hot_from_bottom_final_only"),
        reverse_output=True,
    )


def trace_adaptive_condensate_schedule(
    state: ThermoState,
    init: CondensateEquilibriumInit,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    *,
    epsilon_start: float = 0.0,
    epsilon_crit: float = -40.0,
    n_step: int = 100,
    max_iter: int = 100,
    element_indices: Optional[jnp.ndarray] = None,
    debug_nan: bool = False,
    run_full_schedule: bool = True,
    epsilon_guard_margin: float = 1.0e-6,
    min_epsilon_step: float = 1.0e-6,
    max_adaptive_schedule_steps: Optional[int] = None,
    condensate_species: Optional[Sequence[str]] = None,
    top_k: int = 5,
    reduced_solver: str = "augmented_lu_row_scaled",
    regularization_mode: str = "none",
    regularization_strength: float = 0.0,
):
    """Trace the adaptive sk-guarded epsilon path for one layer."""

    _result, trace = _run_adaptive_condensate_layer_schedule(
        state,
        init=init,
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=hvector_func,
        hvector_cond_func=hvector_cond_func,
        epsilon_start=epsilon_start,
        epsilon_crit=epsilon_crit,
        n_step=n_step,
        max_iter=max_iter,
        element_indices=element_indices,
        debug_nan=debug_nan,
        run_full_schedule=run_full_schedule,
        epsilon_guard_margin=epsilon_guard_margin,
        min_epsilon_step=min_epsilon_step,
        max_adaptive_schedule_steps=max_adaptive_schedule_steps,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
        condensate_species=condensate_species,
        top_k=top_k,
    )
    return trace


def trace_condensate_iteration_lambda_trials(
    state: ThermoState,
    init: CondensateEquilibriumInit,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    *,
    epsilon: float,
    element_indices: Optional[jnp.ndarray] = None,
    lambda_trials: Optional[Sequence[float]] = None,
    lambda_multipliers: Sequence[float] = (1.0, 0.5, 0.2, 0.1, 0.05),
    extra_lambda_trials: Sequence[float] = (1.0, 0.5, 0.2, 0.1, 0.05),
    reduced_solver: str = "augmented_lu_row_scaled",
    regularization_mode: str = "none",
    regularization_strength: float = 0.0,
):
    """Diagnostic-only wrapper for trial lambdas along one fixed current direction."""

    init_prepared = _prepare_condensate_init(init)
    return _diagnose_iteration_lambda_trials_raw(
        state,
        ln_nk=init_prepared.ln_nk,
        ln_mk=init_prepared.ln_mk,
        ln_ntot=init_prepared.ln_ntot,
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=hvector_func,
        hvector_cond_func=hvector_cond_func,
        epsilon=epsilon,
        element_indices=element_indices,
        lambda_trials=lambda_trials,
        lambda_multipliers=lambda_multipliers,
        extra_lambda_trials=extra_lambda_trials,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )


def trace_condensate_gas_limiter_diagnostics(
    state: ThermoState,
    init: CondensateEquilibriumInit,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    *,
    epsilon: float,
    element_indices: Optional[jnp.ndarray] = None,
    gas_species_names: Optional[Sequence[str]] = None,
    top_k: int = 10,
    reduced_solver: str = "augmented_lu_row_scaled",
    regularization_mode: str = "none",
    regularization_strength: float = 0.0,
):
    """Diagnostic-only wrapper for gas limiter decomposition and direction comparison."""

    init_prepared = _prepare_condensate_init(init)
    return _diagnose_gas_step_limiter_and_direction_raw(
        state,
        ln_nk=init_prepared.ln_nk,
        ln_mk=init_prepared.ln_mk,
        ln_ntot=init_prepared.ln_ntot,
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=hvector_func,
        hvector_cond_func=hvector_cond_func,
        epsilon=epsilon,
        element_indices=element_indices,
        gas_species_names=gas_species_names,
        top_k=top_k,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )


def trace_condensate_reduced_solver_backends(
    state: ThermoState,
    init: CondensateEquilibriumInit,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    *,
    epsilon: float,
    element_indices: Optional[jnp.ndarray] = None,
    backend_configs: Optional[Sequence[dict]] = None,
    exact_input_bundle_context: Optional[dict[str, Any]] = None,
):
    """Diagnostic-only wrapper for one-step reduced-solver backend comparisons."""

    init_prepared = _prepare_condensate_init(init)
    emit_exact_input_bundle = (
        False
        if exact_input_bundle_context is None
        else bool(exact_input_bundle_context.get("emit_exact_input_bundle", False))
    )
    case_key = (
        "diagnostic"
        if exact_input_bundle_context is None
        else str(exact_input_bundle_context.get("case_key", "diagnostic"))
    )
    newton_iter = (
        0
        if exact_input_bundle_context is None
        else int(exact_input_bundle_context.get("newton_iter", 0))
    )
    ln_nk_init_source_trace = (
        None
        if not emit_exact_input_bundle
        else _build_lnnk_init_source_trace(
            init,
            init_prepared,
            case_key=case_key,
            newton_iter=newton_iter,
            source_stage="trace_condensate_reduced_solver_backends CondensateEquilibriumInit.ln_nk",
            producer_function=(
                "src/exogibbs/optimize/minimize_cond.py::"
                "trace_condensate_reduced_solver_backends"
            ),
        )
    )
    return _diagnose_reduced_solver_backend_experiments_raw(
        state,
        ln_nk=init_prepared.ln_nk,
        ln_mk=init_prepared.ln_mk,
        ln_ntot=init_prepared.ln_ntot,
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=hvector_func,
        hvector_cond_func=hvector_cond_func,
        epsilon=epsilon,
        element_indices=element_indices,
        backend_configs=backend_configs,
        case_key=case_key,
        newton_iter=newton_iter,
        condensates_jac_indices=(
            None
            if exact_input_bundle_context is None
            else exact_input_bundle_context.get("condensates_jac_indices")
        ),
        condensate_labels_jac_order=(
            None
            if exact_input_bundle_context is None
            else exact_input_bundle_context.get("condensate_labels_jac_order")
        ),
        element_labels_reduced_order=(
            None
            if exact_input_bundle_context is None
            else exact_input_bundle_context.get("element_labels_reduced_order")
        ),
        row_scaled_element_condensate_jec_target_block=(
            None
            if exact_input_bundle_context is None
            else exact_input_bundle_context.get(
                "row_scaled_element_condensate_jec_target_block"
            )
        ),
        selected_element_row_scaling_vector=(
            None
            if exact_input_bundle_context is None
            else exact_input_bundle_context.get("selected_element_row_scaling_vector")
        ),
        gas_phase_calculate_lifecycle_context=(
            None
            if exact_input_bundle_context is None
            else exact_input_bundle_context.get("gas_phase_calculate_lifecycle_context")
        ),
        emit_exact_input_bundle=emit_exact_input_bundle,
        ln_nk_init_source_trace=ln_nk_init_source_trace,
    )


def trace_condensate_full_vs_reduced_gie_direction(
    state: ThermoState,
    init: CondensateEquilibriumInit,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    *,
    epsilon: float,
    element_indices: Optional[jnp.ndarray] = None,
    reduced_solver: str = "augmented_lu_row_scaled",
    regularization_mode: str = "none",
    regularization_strength: float = 0.0,
):
    """Diagnostic-only wrapper for one-state reduced-vs-full GIE direction comparisons."""

    init_prepared = _prepare_condensate_init(init)
    return _diagnose_full_vs_reduced_gie_direction_raw(
        state,
        ln_nk=init_prepared.ln_nk,
        ln_mk=init_prepared.ln_mk,
        ln_ntot=init_prepared.ln_ntot,
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=hvector_func,
        hvector_cond_func=hvector_cond_func,
        epsilon=epsilon,
        element_indices=element_indices,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )


def trace_condensate_pdipm_vs_pipm_direction(
    state: ThermoState,
    init: CondensateEquilibriumInit,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    *,
    epsilon: float,
    element_indices: Optional[jnp.ndarray] = None,
    lambda_trials: Optional[Sequence[float]] = None,
    reduced_solver: str = "augmented_lu_row_scaled",
    regularization_mode: str = "none",
    regularization_strength: float = 0.0,
):
    """Diagnostic-only wrapper for one-state PDIPM-vs-PIPM direction comparisons."""

    init_prepared = _prepare_condensate_init(init)
    return _diagnose_pdipm_vs_pipm_direction_raw(
        state,
        ln_nk=init_prepared.ln_nk,
        ln_mk=init_prepared.ln_mk,
        ln_ntot=init_prepared.ln_ntot,
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=hvector_func,
        hvector_cond_func=hvector_cond_func,
        epsilon=epsilon,
        element_indices=element_indices,
        lambda_trials=lambda_trials,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )


def trace_condensate_pdipm_vs_pipm_fixed_epsilon_trajectories(
    state: ThermoState,
    init: CondensateEquilibriumInit,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    *,
    epsilon: float,
    rho_offsets: Sequence[float] = (0.0, 1.0, -1.0),
    max_iter: int = 10,
    min_lambda: float = 1.0e-6,
    backtrack_factor: float = 0.5,
    element_indices: Optional[jnp.ndarray] = None,
    reduced_solver: str = "augmented_lu_row_scaled",
    regularization_mode: str = "none",
    regularization_strength: float = 0.0,
):
    """Diagnostic-only wrapper for fixed-epsilon PDIPM-vs-PIPM trajectory comparisons."""

    init_prepared = _prepare_condensate_init(init)
    return _diagnose_pdipm_vs_pipm_fixed_epsilon_trajectories_raw(
        state,
        ln_nk=init_prepared.ln_nk,
        ln_mk=init_prepared.ln_mk,
        ln_ntot=init_prepared.ln_ntot,
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=hvector_func,
        hvector_cond_func=hvector_cond_func,
        epsilon=epsilon,
        rho_offsets=rho_offsets,
        max_iter=max_iter,
        min_lambda=min_lambda,
        backtrack_factor=backtrack_factor,
        element_indices=element_indices,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )


def trace_condensate_sk_stage_feasibility(
    state: ThermoState,
    init: CondensateEquilibriumInit,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    *,
    epsilon_start: float = 0.0,
    epsilon_crit: float = -40.0,
    n_step: int = 100,
    max_iter: int = 100,
    element_indices: Optional[jnp.ndarray] = None,
    debug_nan: bool = False,
    condensate_species: Optional[Sequence[str]] = None,
    top_k: int = 5,
    include_final_repeat: bool = True,
):
    """Trace stage-start sk feasibility along the existing continuation schedule.

    This helper is diagnostic-only. It snapshots the current condensate state
    before each scheduled epsilon solve and reports whether the sk admissibility
    bound used by :func:`stepsize_sk` is already violated before Newton starts.
    """

    if n_step < 1:
        raise ValueError("n_step must be at least 1.")

    prepared_init = _prepare_condensate_init(init)
    epsilons = jnp.linspace(epsilon_start, epsilon_crit, n_step + 1)[1:]
    stages = []
    current_init = prepared_init

    def _record_stage(epsilon, stage_index: int, is_final_repeat: bool):
        ln_mk = jnp.asarray(current_init.ln_mk)
        ln_sk = 2.0 * ln_mk - epsilon
        feasibility_num = LOG_S_MAX + epsilon - 2.0 * ln_mk
        violation_margin = -feasibility_num
        infeasible_mask = feasibility_num < 0.0
        infeasible_indices = jnp.where(infeasible_mask)[0]
        infeasible_count = int(infeasible_indices.shape[0])

        if infeasible_count > 0:
            positive_margin = jnp.where(infeasible_mask, violation_margin, -jnp.inf)
            ranked = jnp.argsort(-positive_margin)
            worst_indices = [int(i) for i in ranked[: min(top_k, infeasible_count)]]
        else:
            worst_indices = []

        if condensate_species is None:
            worst_names = None
        else:
            worst_names = [str(condensate_species[i]) for i in worst_indices]

        stages.append(
            {
                "stage_index": stage_index,
                "is_final_repeat": is_final_repeat,
                "epsilon": float(epsilon),
                "log_s_max": float(LOG_S_MAX),
                "ln_mk": [float(x) for x in ln_mk],
                "ln_sk": [float(x) for x in ln_sk],
                "feasibility_num": [float(x) for x in feasibility_num],
                "violation_margin": [float(x) for x in violation_margin],
                "has_pre_iteration_sk_infeasibility": bool(jnp.any(infeasible_mask)),
                "n_pre_iteration_sk_infeasible": infeasible_count,
                "worst_infeasible_indices": worst_indices,
                "worst_infeasible_names": worst_names,
                "worst_infeasible_violation_margin": [float(violation_margin[i]) for i in worst_indices],
                "worst_infeasible_ln_mk": [float(ln_mk[i]) for i in worst_indices],
                "worst_infeasible_ln_sk": [float(ln_sk[i]) for i in worst_indices],
                "condition": "log_s_max + epsilon - 2*ln_mk >= 0",
            }
        )

    for stage_index, epsilon in enumerate(epsilons.tolist()):
        _record_stage(epsilon, stage_index, False)
        result = minimize_gibbs_cond(
            state,
            init=current_init,
            formula_matrix=formula_matrix,
            formula_matrix_cond=formula_matrix_cond,
            hvector_func=hvector_func,
            hvector_cond_func=hvector_cond_func,
            epsilon=epsilon,
            residual_crit=jnp.exp(epsilon),
            max_iter=max_iter,
            element_indices=element_indices,
            debug_nan=debug_nan,
        )
        current_init = result.to_init()

    if include_final_repeat:
        _record_stage(float(epsilons[-1]), int(n_step), True)

    return {
        "epsilon_start": float(epsilon_start),
        "epsilon_crit": float(epsilon_crit),
        "n_step": int(n_step),
        "max_iter": int(max_iter),
        "stages": stages,
    }


__all__ = [
    "CondensateEquilibriumDiagnostics",
    "CondensateEquilibriumInit",
    "CondensateEpsilonSchedule",
    "CondensateProfileMethod",
    "CondensateRGIESupportClassifierConfig",
    "CondensateRGIESupportMethod",
    "CondensateRGIEReducedCouplingConfig",
    "CondensateRGIEStartupConfig",
    "CondensateRGIEStartupPolicy",
    "CondensateEquilibriumResult",
    "classify_rgie_support_proxies",
    "build_lnnk_constructor_source_trace",
    "build_minimize_gibbs_core_lnnk_output_source_trace",
    "minimize_gibbs_core_with_source_trace",
    "compute_sk_feasible_epsilon_floor",
    "diagnose_augmented_semismooth_candidate_condensate_layer",
    "diagnose_semismooth_candidate_condensate_layer",
    "diagnose_smoothed_semismooth_candidate_condensate_layer",
    "diagnose_support_updating_active_set_layer",
    "minimize_gibbs_cond",
    "minimize_gibbs_cond_profile",
    "minimize_gibbs_cond_core",
    "minimize_gibbs_cond_with_diagnostics",
    "solve_augmented_semismooth_candidate_condensate_layer",
    "solve_gas_equilibrium_with_duals",
    "solve_restricted_support_condensate_layer",
    "solve_semismooth_candidate_condensate_layer",
    "solve_smoothed_semismooth_candidate_condensate_layer",
    "trace_adaptive_condensate_schedule",
    "trace_condensate_gas_limiter_diagnostics",
    "trace_condensate_iteration_lambda_trials",
    "trace_condensate_full_vs_reduced_gie_direction",
    "trace_condensate_pdipm_vs_pipm_direction",
    "trace_condensate_pdipm_vs_pipm_fixed_epsilon_trajectories",
    "trace_condensate_reduced_solver_backends",
    "trace_condensate_sk_stage_feasibility",
]
