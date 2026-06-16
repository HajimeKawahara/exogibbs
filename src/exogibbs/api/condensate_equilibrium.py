"""Production-facing condensate equilibrium API shell.

This module defines the first condensate-specific public API surface. It keeps
gas-only equilibrium behavior separate and routes condensate-enabled calls
through the current condensate HEAD route contract.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Literal, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from exogibbs.api.chemistry import ChemicalSetup, ThermoState
from exogibbs.condensates.head_route_standard_gate import (
    BUDGET_TRADEOFF_STATUS,
    CONVERGED,
    CONVERGED_WITH_CAVEAT,
    HEAD_ROUTE_STANDARD,
    NOT_CONVERGED,
    classify_head_route_standard_gate_row,
)


Array = jax.Array
CondensateRoute = Literal["head_v1"]
CondensateResidualPolicy = Literal["head_route_tiers_v1"]
CondensateWarmStartGasRefreshPolicy = Literal["native_gas_solver"]
CondensatePrimaryAcceptanceGuard = Literal["tight_weighted_components"]
CondensateSeedInitializationPolicy = Literal[
    "budget_preserving_fraction",
    "capacity_fraction",
    "max_density",
]
CONDENSATE_HEAD_ROUTE_VERSION = "v1.6"
CONDENSATE_HEAD_ROUTE_NAME = "head_route_v1_6_actual_support_outer_loop_growth"
HEAD_ROUTE_SOFT_RESTORATION_COMPONENT_WEIGHTS = {
    "budget": 1.0,
    "total_density": 1.0,
    "amount_weighted_gas": 1.0,
    "amount_weighted_condensate": 1.0,
}
HEAD_ROUTE_IPOPT_H_TYPE_COMPONENT_WEIGHTS = {
    "budget": 1.0,
    "total_density": 1.0,
    "amount_weighted_gas": 1.0,
    "amount_weighted_condensate": 1.0,
    "complementarity": 1.0,
}
HEAD_ROUTE_IPOPT_H_TYPE_PROTECTED_COMPONENTS = ("budget", "total_density")
HEAD_ROUTE_RELATIVE_BUDGET_CORRECTION_COMPONENT_WEIGHTS = {
    "relative_budget_max": 1.0,
    **HEAD_ROUTE_IPOPT_H_TYPE_COMPONENT_WEIGHTS,
}
HEAD_ROUTE_RELATIVE_BUDGET_CORRECTION_PROTECTED_COMPONENTS = (
    "relative_budget_max",
    *HEAD_ROUTE_IPOPT_H_TYPE_PROTECTED_COMPONENTS,
)


@dataclass(frozen=True)
class CondensateChemicalSetup:
    """Gas and condensate thermochemistry bundle for condensate equilibrium."""

    gas_setup: ChemicalSetup
    condensate_setup: ChemicalSetup
    formula_matrix: Array
    formula_matrix_cond: Array
    gas_species: tuple[str, ...]
    condensate_species: tuple[str, ...]
    elements: tuple[str, ...]


@dataclass(frozen=True)
class CondensateEquilibriumOptions:
    """Options for the condensate HEAD route standard path."""

    route: CondensateRoute = HEAD_ROUTE_STANDARD
    case_id: Optional[str] = None
    allow_caveat_tiers: bool = True
    return_diagnostics: bool = False
    max_outer_iterations: Optional[int] = None
    max_inner_iterations: Optional[int] = None
    residual_policy: CondensateResidualPolicy = "head_route_tiers_v1"
    metric_status: Optional[str] = None
    selected_route: str = "head_v1_restricted_support"
    max_positive_support_count: Optional[int] = None
    max_activity_support_count: Optional[int] = None
    seed_initialization_policy: CondensateSeedInitializationPolicy = "max_density"
    seed_fraction: float = 1.0e-3
    max_seed_amount: float = 1.0e-3
    min_seed_amount: float = 1.0e-300
    allow_empty_positive_support: bool = True
    enable_support_outer_loop: bool = True
    max_support_outer_iterations: int = 4
    max_support_add_per_round: Optional[int] = None
    support_activity_threshold: float = 0.0
    enable_head_route_warm_start: bool = True
    enable_depleted_gas_refresh: bool = True
    warm_start_gas_refresh_policy: CondensateWarmStartGasRefreshPolicy = "native_gas_solver"
    restricted_reduced_coupling_mode: str = "pdipm_rgie_v11_activity_correction"
    restricted_reduced_coupling_alpha_s: float = 1.0
    head_route_primary_center_tolerance_multiplier: Optional[float] = None
    head_route_primary_residual_worsening_tolerance: Optional[float] = None
    head_route_primary_require_residual_nonworsening: Optional[bool] = None
    head_route_primary_acceptance_guard: Optional[CondensatePrimaryAcceptanceGuard] = None
    head_route_primary_guard_max_budget: float = 1.0e-8
    head_route_primary_guard_max_amount_weighted_gas: float = 1.0e-8
    head_route_primary_guard_max_gas_stationarity: float = 1.0
    head_route_primary_guard_max_condensate_stationarity: float = 10.0
    head_route_primary_summary: Optional[Mapping[str, Any]] = None
    head_route_refresh_policy_summary: Optional[Mapping[str, Any]] = None
    enable_head_route_center_gate_retry: bool = False
    head_route_center_gate_retry_multiplier: float = 1.0e11
    enable_head_route_residual_worsening_retry: bool = False
    head_route_residual_worsening_retry_tolerance: float = 2.0e-2
    enable_head_route_soft_restoration_retry: bool = False
    head_route_soft_restoration_proximity_weight: float = 1.0e-2
    head_route_soft_restoration_max_proximity: Optional[float] = 10.0
    enable_head_route_ipopt_h_type_retry: bool = False
    head_route_ipopt_h_type_theta_reduction_fraction: float = 1.0e-4
    head_route_ipopt_h_type_protected_component_max_normalized_increase: float = 1.0
    enable_head_route_condensate_budget_correction_retry: bool = True
    enable_support_cap_retry: bool = True
    support_cap_retry_count: int = 34
    support_cap_retry_counts: Optional[Sequence[int]] = (34, 48, 80, 128)
    enable_support_growth_staging_retry: bool = True
    support_growth_staging_retry_add_per_rounds: Optional[Sequence[int]] = (64, 32, 16, 8)
    enable_support_closure_retry_gate: bool = True
    support_closure_max_positive_inactive_driving: float = 5.0e2
    enable_native_seed_fallback: bool = True
    enable_full_condensate_budget_residual_gate: bool = True
    full_condensate_budget_relative_tolerance: float = 1.0e-3


@dataclass(frozen=True)
class CondensateEquilibriumResult:
    """Result container for the condensate equilibrium standard path."""

    gas_ln_n: Array
    gas_n: Array
    gas_x: Array
    gas_ntot: Array
    condensate_amounts: Array
    condensate_support_indices: Array
    condensate_support_names: tuple[str, ...]
    acceptance_tier: str
    selected_route: str
    status: str
    converged: bool
    diagnostics: Optional[Mapping[str, Any]] = None
    head_route_version: str = CONDENSATE_HEAD_ROUTE_VERSION
    head_route_name: str = CONDENSATE_HEAD_ROUTE_NAME


def validate_condensate_chemical_setup(setup: CondensateChemicalSetup) -> None:
    """Validate gas-condensate setup compatibility for HEAD route calls."""

    if not isinstance(setup.gas_setup, ChemicalSetup):
        raise TypeError("gas_setup must be a ChemicalSetup.")
    if not isinstance(setup.condensate_setup, ChemicalSetup):
        raise TypeError("condensate_setup must be a ChemicalSetup.")
    if setup.gas_setup.elements is None:
        raise ValueError("gas_setup.elements is required for condensate equilibrium.")
    if setup.condensate_setup.elements is None:
        raise ValueError("condensate_setup.elements is required for condensate equilibrium.")
    if tuple(setup.gas_setup.elements) != tuple(setup.condensate_setup.elements):
        raise ValueError("gas and condensate element orders must match.")
    formula_matrix = jnp.asarray(setup.formula_matrix)
    formula_matrix_cond = jnp.asarray(setup.formula_matrix_cond)
    if formula_matrix.ndim != 2:
        raise ValueError("formula_matrix must be a two-dimensional array.")
    if formula_matrix_cond.ndim != 2:
        raise ValueError("formula_matrix_cond must be a two-dimensional array.")
    if formula_matrix.shape[0] != formula_matrix_cond.shape[0]:
        raise ValueError("gas and condensate formula matrices must have the same element count.")
    if formula_matrix.shape[0] != len(setup.elements):
        raise ValueError("elements length must match formula matrix rows.")
    if formula_matrix.shape[1] != len(setup.gas_species):
        raise ValueError("gas_species length must match formula_matrix columns.")
    if formula_matrix_cond.shape[1] != len(setup.condensate_species):
        raise ValueError("condensate_species length must match formula_matrix_cond columns.")


def build_condensate_chemical_setup(
    *,
    gas_setup: ChemicalSetup,
    condensate_setup: ChemicalSetup,
) -> CondensateChemicalSetup:
    """Build and validate a gas-condensate chemical setup bundle."""

    if gas_setup.elements is None:
        raise ValueError("gas_setup.elements is required for condensate equilibrium.")
    if gas_setup.species is None:
        raise ValueError("gas_setup.species is required for condensate equilibrium.")
    if condensate_setup.elements is None:
        raise ValueError("condensate_setup.elements is required for condensate equilibrium.")
    if condensate_setup.species is None:
        raise ValueError("condensate_setup.species is required for condensate equilibrium.")
    setup = CondensateChemicalSetup(
        gas_setup=gas_setup,
        condensate_setup=condensate_setup,
        formula_matrix=jnp.asarray(gas_setup.formula_matrix),
        formula_matrix_cond=jnp.asarray(condensate_setup.formula_matrix),
        gas_species=tuple(gas_setup.species),
        condensate_species=tuple(condensate_setup.species),
        elements=tuple(gas_setup.elements),
    )
    validate_condensate_chemical_setup(setup)
    return setup


def _ln_normalized_pressure(pressure: float, reference_pressure: float) -> Array:
    return jnp.log(jnp.asarray(pressure) / jnp.asarray(reference_pressure))


def _full_condensate_amounts(
    *,
    support_indices: Sequence[int],
    support_amounts: Array,
    condensate_count: int,
) -> Array:
    indices = jnp.asarray(support_indices, dtype=jnp.int32)
    amounts = jnp.asarray(support_amounts)
    if indices.ndim != 1:
        raise ValueError("support_indices must be one-dimensional.")
    if amounts.ndim != 1:
        raise ValueError("support_amounts must be one-dimensional.")
    if indices.shape[0] != amounts.shape[0]:
        raise ValueError("support_indices and support_amounts must have the same length.")
    if bool(jnp.any(indices < 0)) or bool(jnp.any(indices >= condensate_count)):
        raise ValueError("support_indices contain an out-of-range condensate index.")
    return jnp.zeros((condensate_count,), dtype=amounts.dtype).at[indices].set(amounts)


def _external_condensate_amounts_vector(
    *,
    support_indices: Sequence[int],
    support_amounts: Sequence[float],
    condensate_count: int,
) -> Array:
    """Return a full-length vector for condensates externalized from the solver."""

    indices = tuple(int(index) for index in support_indices)
    amounts = jnp.asarray(support_amounts, dtype=jnp.float64)
    if amounts.ndim != 1:
        raise ValueError("external support_amounts must be one-dimensional.")
    if len(indices) != amounts.shape[0]:
        raise ValueError("external support_indices and support_amounts must have the same length.")
    full = jnp.zeros((condensate_count,), dtype=amounts.dtype)
    if indices:
        full = full.at[jnp.asarray(indices, dtype=jnp.int32)].add(amounts)
    return full


def _merge_external_condensate_amounts(
    *,
    condensate_amounts: Array,
    external_condensate_amounts: Sequence[float] | Array | None,
) -> Array:
    """Add externally budgeted condensates back to the public full vector."""

    amounts = jnp.asarray(condensate_amounts, dtype=jnp.float64)
    if external_condensate_amounts is None:
        return amounts
    external = jnp.asarray(external_condensate_amounts, dtype=jnp.float64)
    if external.ndim != 1 or external.shape[0] != amounts.shape[0]:
        raise ValueError("external_condensate_amounts must match condensate_count.")
    return amounts + external


def _validate_options(options: CondensateEquilibriumOptions) -> None:
    if options.route != HEAD_ROUTE_STANDARD:
        raise ValueError(f"Unsupported condensate route '{options.route}'. Expected '{HEAD_ROUTE_STANDARD}'.")
    if options.residual_policy != "head_route_tiers_v1":
        raise ValueError("Only residual_policy='head_route_tiers_v1' is supported.")
    if options.max_positive_support_count is not None and options.max_positive_support_count <= 0:
        raise ValueError("max_positive_support_count must be positive.")
    if options.max_activity_support_count is not None and options.max_activity_support_count <= 0:
        raise ValueError("max_activity_support_count must be positive.")
    valid_seed_initialization_policies = {
        "budget_preserving_fraction",
        "capacity_fraction",
        "max_density",
    }
    if options.seed_initialization_policy not in valid_seed_initialization_policies:
        raise ValueError(
            "seed_initialization_policy must be one of "
            f"{sorted(valid_seed_initialization_policies)}."
        )
    if options.seed_fraction <= 0.0:
        raise ValueError("seed_fraction must be positive.")
    if options.max_seed_amount <= 0.0:
        raise ValueError("max_seed_amount must be positive.")
    if options.min_seed_amount <= 0.0:
        raise ValueError("min_seed_amount must be positive.")
    if options.max_support_outer_iterations <= 0:
        raise ValueError("max_support_outer_iterations must be positive.")
    if options.max_support_add_per_round is not None and options.max_support_add_per_round <= 0:
        raise ValueError("max_support_add_per_round must be positive.")
    if options.warm_start_gas_refresh_policy != "native_gas_solver":
        raise ValueError("Only warm_start_gas_refresh_policy='native_gas_solver' is supported.")
    valid_reduced_coupling_modes = {
        "current",
        "capped_s_only_fixed_alpha",
        "capped_s_only_conditional",
        "candidate_selected_active_only",
        "candidate_selected_active_plus_near_jacobian",
        "candidate_selected_active_plus_near_jacobian_with_rem_inventory",
        "candidate_selected_weighted_mask",
        "pdipm_rgie_v11_activity_correction",
    }
    if options.restricted_reduced_coupling_mode not in valid_reduced_coupling_modes:
        raise ValueError(
            "restricted_reduced_coupling_mode must be one of "
            f"{sorted(valid_reduced_coupling_modes)}."
        )
    if options.restricted_reduced_coupling_alpha_s <= 0.0:
        raise ValueError("restricted_reduced_coupling_alpha_s must be positive.")
    if (
        options.head_route_primary_center_tolerance_multiplier is not None
        and (
            not math.isfinite(float(options.head_route_primary_center_tolerance_multiplier))
            or options.head_route_primary_center_tolerance_multiplier <= 0.0
        )
    ):
        raise ValueError(
            "head_route_primary_center_tolerance_multiplier must be finite and positive."
        )
    if (
        options.head_route_primary_residual_worsening_tolerance is not None
        and (
            not math.isfinite(float(options.head_route_primary_residual_worsening_tolerance))
            or options.head_route_primary_residual_worsening_tolerance < 0.0
        )
    ):
        raise ValueError(
            "head_route_primary_residual_worsening_tolerance must be finite and non-negative."
        )
    if (
        options.head_route_primary_require_residual_nonworsening is not None
        and not isinstance(options.head_route_primary_require_residual_nonworsening, bool)
    ):
        raise TypeError("head_route_primary_require_residual_nonworsening must be a bool.")
    if (
        options.head_route_primary_acceptance_guard is not None
        and options.head_route_primary_acceptance_guard != "tight_weighted_components"
    ):
        raise ValueError(
            "head_route_primary_acceptance_guard must be None or 'tight_weighted_components'."
        )
    for name, value in (
        ("head_route_primary_guard_max_budget", options.head_route_primary_guard_max_budget),
        (
            "head_route_primary_guard_max_amount_weighted_gas",
            options.head_route_primary_guard_max_amount_weighted_gas,
        ),
        (
            "head_route_primary_guard_max_gas_stationarity",
            options.head_route_primary_guard_max_gas_stationarity,
        ),
        (
            "head_route_primary_guard_max_condensate_stationarity",
            options.head_route_primary_guard_max_condensate_stationarity,
        ),
    ):
        if not math.isfinite(float(value)) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if not isinstance(options.enable_head_route_center_gate_retry, bool):
        raise TypeError("enable_head_route_center_gate_retry must be a bool.")
    if (
        not math.isfinite(float(options.head_route_center_gate_retry_multiplier))
        or options.head_route_center_gate_retry_multiplier <= 0.0
    ):
        raise ValueError(
            "head_route_center_gate_retry_multiplier must be finite and positive."
        )
    if not isinstance(options.enable_head_route_residual_worsening_retry, bool):
        raise TypeError("enable_head_route_residual_worsening_retry must be a bool.")
    if (
        not math.isfinite(float(options.head_route_residual_worsening_retry_tolerance))
        or options.head_route_residual_worsening_retry_tolerance < 0.0
    ):
        raise ValueError(
            "head_route_residual_worsening_retry_tolerance must be finite and non-negative."
        )
    if not isinstance(options.enable_head_route_soft_restoration_retry, bool):
        raise TypeError("enable_head_route_soft_restoration_retry must be a bool.")
    if (
        not math.isfinite(float(options.head_route_soft_restoration_proximity_weight))
        or options.head_route_soft_restoration_proximity_weight < 0.0
    ):
        raise ValueError(
            "head_route_soft_restoration_proximity_weight must be finite and non-negative."
        )
    if (
        options.head_route_soft_restoration_max_proximity is not None
        and (
            not math.isfinite(float(options.head_route_soft_restoration_max_proximity))
            or options.head_route_soft_restoration_max_proximity < 0.0
        )
    ):
        raise ValueError(
            "head_route_soft_restoration_max_proximity must be finite and non-negative."
        )
    if not isinstance(options.enable_head_route_ipopt_h_type_retry, bool):
        raise TypeError("enable_head_route_ipopt_h_type_retry must be a bool.")
    if (
        not math.isfinite(float(options.head_route_ipopt_h_type_theta_reduction_fraction))
        or options.head_route_ipopt_h_type_theta_reduction_fraction < 0.0
        or options.head_route_ipopt_h_type_theta_reduction_fraction >= 1.0
    ):
        raise ValueError(
            "head_route_ipopt_h_type_theta_reduction_fraction must be finite and in [0, 1)."
        )
    if (
        not math.isfinite(
            float(
                options.head_route_ipopt_h_type_protected_component_max_normalized_increase
            )
        )
        or options.head_route_ipopt_h_type_protected_component_max_normalized_increase < 0.0
    ):
        raise ValueError(
            "head_route_ipopt_h_type_protected_component_max_normalized_increase "
            "must be finite and non-negative."
        )
    if not isinstance(options.enable_head_route_condensate_budget_correction_retry, bool):
        raise TypeError("enable_head_route_condensate_budget_correction_retry must be a bool.")
    if not isinstance(options.enable_support_cap_retry, bool):
        raise TypeError("enable_support_cap_retry must be a bool.")
    if options.support_cap_retry_count <= 0:
        raise ValueError("support_cap_retry_count must be positive.")
    if options.support_cap_retry_counts is not None:
        if len(tuple(options.support_cap_retry_counts)) == 0:
            raise ValueError("support_cap_retry_counts must not be empty.")
        for count in options.support_cap_retry_counts:
            if int(count) <= 0:
                raise ValueError("support_cap_retry_counts entries must be positive.")
    if not isinstance(options.enable_support_growth_staging_retry, bool):
        raise TypeError("enable_support_growth_staging_retry must be a bool.")
    if options.support_growth_staging_retry_add_per_rounds is not None:
        if len(tuple(options.support_growth_staging_retry_add_per_rounds)) == 0:
            raise ValueError("support_growth_staging_retry_add_per_rounds must not be empty.")
        for count in options.support_growth_staging_retry_add_per_rounds:
            if int(count) <= 0:
                raise ValueError(
                    "support_growth_staging_retry_add_per_rounds entries must be positive."
                )
    if not isinstance(options.enable_support_closure_retry_gate, bool):
        raise TypeError("enable_support_closure_retry_gate must be a bool.")
    if (
        not math.isfinite(float(options.support_closure_max_positive_inactive_driving))
        or options.support_closure_max_positive_inactive_driving < 0.0
    ):
        raise ValueError(
            "support_closure_max_positive_inactive_driving must be finite and non-negative."
        )
    if not isinstance(options.enable_full_condensate_budget_residual_gate, bool):
        raise TypeError("enable_full_condensate_budget_residual_gate must be a bool.")
    if (
        not math.isfinite(float(options.full_condensate_budget_relative_tolerance))
        or options.full_condensate_budget_relative_tolerance < 0.0
    ):
        raise ValueError(
            "full_condensate_budget_relative_tolerance must be finite and non-negative."
        )


def _full_condensate_element_budget_residual_report(
    *,
    setup: CondensateChemicalSetup,
    gas_n: Array,
    condensate_amounts: Array,
    element_inventory_target: Array,
    relative_tolerance: float,
) -> dict[str, Any]:
    target = jnp.asarray(element_inventory_target, dtype=jnp.float64)
    if target.ndim != 1 or target.shape[0] != len(setup.elements):
        raise ValueError("element_inventory_target must have one value per element.")
    gas_amounts = jnp.asarray(gas_n, dtype=jnp.float64)
    cond_amounts = jnp.asarray(condensate_amounts, dtype=jnp.float64)
    if gas_amounts.ndim != 1 or gas_amounts.shape[0] != len(setup.gas_species):
        raise ValueError("gas_n must have one value per gas species.")
    if cond_amounts.ndim != 1 or cond_amounts.shape[0] != len(setup.condensate_species):
        raise ValueError("condensate_amounts must have one value per condensate species.")
    gas_budget = jnp.asarray(setup.formula_matrix, dtype=jnp.float64) @ gas_amounts
    condensate_budget = (
        jnp.asarray(setup.formula_matrix_cond, dtype=jnp.float64) @ cond_amounts
    )
    reconstructed = gas_budget + condensate_budget
    residual = reconstructed - target
    denominator = jnp.maximum(jnp.abs(target), 1.0e-300)
    signed_relative = residual / denominator
    absolute_relative = jnp.abs(signed_relative)
    gate_mask = jnp.asarray(
        tuple(str(element) not in {"e-", "electron"} for element in setup.elements),
        dtype=bool,
    )
    gated_absolute_relative = jnp.where(gate_mask, absolute_relative, 0.0)
    finite = bool(jnp.all(jnp.isfinite(jnp.where(gate_mask, absolute_relative, 0.0))))
    sanitized = jnp.where(
        jnp.isfinite(gated_absolute_relative),
        gated_absolute_relative,
        jnp.inf,
    )
    max_index = int(jnp.argmax(sanitized))
    max_abs_relative = float(gated_absolute_relative[max_index])
    tolerance = float(relative_tolerance)
    accepted = finite and max_abs_relative <= tolerance
    return {
        "gate_schema": "exogibbs_full_condensate_element_budget_residual_gate_v1",
        "gate_name": "full_condensate_element_budget_residual",
        "accepted": bool(accepted),
        "relative_tolerance": tolerance,
        "max_abs_relative_residual": max_abs_relative,
        "max_abs_relative_residual_element": setup.elements[max_index],
        "max_abs_relative_residual_element_index": max_index,
        "element_names": tuple(str(element) for element in setup.elements),
        "ignored_element_names": tuple(
            str(element)
            for element in setup.elements
            if str(element) in {"e-", "electron"}
        ),
        "element_budget_target": tuple(float(value) for value in target.tolist()),
        "element_budget_reconstructed": tuple(float(value) for value in reconstructed.tolist()),
        "element_budget_residual": tuple(float(value) for value in residual.tolist()),
        "element_signed_relative_residual": tuple(
            float(value) for value in signed_relative.tolist()
        ),
        "element_abs_relative_residual": tuple(
            float(value) for value in absolute_relative.tolist()
        ),
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }


def _apply_full_condensate_budget_residual_gate(
    *,
    setup: CondensateChemicalSetup,
    gas_n: Array,
    condensate_amounts: Array,
    element_inventory_target: Array | None,
    status: str,
    acceptance_tier: str,
    warning_messages: tuple[str, ...],
    metadata: dict[str, Any],
    enabled: bool,
    relative_tolerance: float,
) -> tuple[str, str, tuple[str, ...], dict[str, Any]]:
    if element_inventory_target is None:
        return status, acceptance_tier, warning_messages, metadata
    report = _full_condensate_element_budget_residual_report(
        setup=setup,
        gas_n=gas_n,
        condensate_amounts=condensate_amounts,
        element_inventory_target=element_inventory_target,
        relative_tolerance=relative_tolerance,
    )
    metadata["full_condensate_budget_residual_gate"] = report
    if (
        not enabled
        or report["accepted"]
        or status not in {CONVERGED, CONVERGED_WITH_CAVEAT}
    ):
        return status, acceptance_tier, warning_messages, metadata
    metadata.setdefault("pre_full_condensate_budget_gate_status", status)
    metadata.setdefault(
        "pre_full_condensate_budget_gate_acceptance_tier",
        acceptance_tier,
    )
    warnings = tuple(warning_messages) + (
        "The full condensate vector element-wise relative budget residual exceeded the accepted threshold.",
    )
    return (
        NOT_CONVERGED,
        "full_condensate_element_budget_residual_failed",
        warnings,
        metadata,
    )


def _full_condensate_budget_gate_report_for_support_state(
    *,
    setup: CondensateChemicalSetup,
    gas_ln_n: Array,
    support_indices: Sequence[int],
    support_amounts: Array,
    external_condensate_amounts: Sequence[float] | Array | None = None,
    element_inventory_target: Array,
    relative_tolerance: float,
) -> dict[str, Any]:
    condensate_amounts = _full_condensate_amounts(
        support_indices=support_indices,
        support_amounts=support_amounts,
        condensate_count=len(setup.condensate_species),
    )
    condensate_amounts = _merge_external_condensate_amounts(
        condensate_amounts=condensate_amounts,
        external_condensate_amounts=external_condensate_amounts,
    )
    return _full_condensate_element_budget_residual_report(
        setup=setup,
        gas_n=jnp.exp(jnp.asarray(gas_ln_n)),
        condensate_amounts=condensate_amounts,
        element_inventory_target=element_inventory_target,
        relative_tolerance=relative_tolerance,
    )


def _final_state_support_indices_from_lifecycle_payload(
    lifecycle_payload: Mapping[str, Any],
    *,
    fallback_support_indices: Sequence[int],
) -> tuple[int, ...]:
    """Return support indices matching a lifecycle continuation final_state."""

    primary_execution = lifecycle_payload.get("primary_execution_report")
    if isinstance(primary_execution, Mapping):
        filter_report = primary_execution.get("filter_report")
        if isinstance(filter_report, Mapping):
            valid_support_indices = filter_report.get("valid_support_indices")
            if valid_support_indices is not None:
                try:
                    return tuple(int(index) for index in valid_support_indices)
                except (TypeError, ValueError):
                    pass
    continuation_input = lifecycle_payload.get("continuation_input", {})
    if isinstance(continuation_input, Mapping):
        support_indices = continuation_input.get("support_indices")
        if support_indices is not None:
            try:
                return tuple(int(index) for index in support_indices)
            except (TypeError, ValueError):
                pass
    return tuple(int(index) for index in fallback_support_indices)


def _lifecycle_final_state_payload(
    lifecycle_payload: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    primary_execution_payload = lifecycle_payload.get("primary_execution_report")
    continuation_payload = (
        primary_execution_payload.get("continuation_report", {})
        if isinstance(primary_execution_payload, Mapping)
        else {}
    )
    final_state_payload = (
        continuation_payload.get("final_state")
        if isinstance(continuation_payload, Mapping)
        else None
    )
    return final_state_payload if isinstance(final_state_payload, Mapping) else None


def _external_condensate_amounts_from_lifecycle_payload(
    lifecycle_payload: Mapping[str, Any],
    *,
    condensate_count: int,
) -> Array | None:
    """Return full-length amounts for condensates externalized by lifecycle filters."""

    carried = lifecycle_payload.get("input_external_condensate_amounts")
    carried_array = None
    if carried is not None:
        try:
            carried_array = jnp.asarray(carried, dtype=jnp.float64)
            if carried_array.ndim != 1 or carried_array.shape[0] != condensate_count:
                carried_array = None
        except (TypeError, ValueError):
            carried_array = None
    primary_execution = lifecycle_payload.get("primary_execution_report")
    if not isinstance(primary_execution, Mapping):
        return carried_array
    support_indices = primary_execution.get("external_condensate_support_indices", ())
    support_amounts = primary_execution.get("external_condensate_amounts", ())
    if not support_indices and not support_amounts:
        return carried_array
    try:
        externalized = _external_condensate_amounts_vector(
            support_indices=support_indices,
            support_amounts=support_amounts,
            condensate_count=condensate_count,
        )
        if carried_array is not None:
            externalized = externalized + carried_array
        return externalized
    except (TypeError, ValueError):
        return carried_array


def _polish_support_amounts_for_full_condensate_budget_gate(
    *,
    setup: CondensateChemicalSetup,
    gas_ln_n: Array,
    support_indices: Sequence[int],
    support_amounts: Array,
    external_condensate_amounts: Sequence[float] | Array | None = None,
    element_inventory_target: Array,
    relative_tolerance: float,
    max_iterations: int = 8,
    max_abs_delta_r: float = 2.0,
) -> tuple[jnp.ndarray, Mapping[str, Any] | None]:
    support = tuple(int(index) for index in support_indices)
    if len(support) == 0:
        return jnp.asarray(support_amounts, dtype=jnp.float64), None
    amounts = np.asarray(support_amounts, dtype=np.float64).copy()
    if amounts.ndim != 1 or amounts.shape[0] != len(support):
        return jnp.asarray(support_amounts, dtype=jnp.float64), None
    if not np.all(np.isfinite(amounts)) or np.any(amounts < 0.0):
        return jnp.asarray(support_amounts, dtype=jnp.float64), None

    gas_n = np.exp(np.asarray(gas_ln_n, dtype=np.float64))
    target = np.asarray(element_inventory_target, dtype=np.float64)
    ag = np.asarray(setup.formula_matrix, dtype=np.float64)
    ac_full = np.asarray(setup.formula_matrix_cond, dtype=np.float64)
    ac = ac_full[:, support]
    external = (
        np.zeros((ac_full.shape[1],), dtype=np.float64)
        if external_condensate_amounts is None
        else np.asarray(external_condensate_amounts, dtype=np.float64)
    )
    if external.ndim != 1 or external.shape[0] != ac_full.shape[1]:
        return jnp.asarray(support_amounts, dtype=jnp.float64), None
    external_budget = ac_full @ external
    gas_budget = ag @ gas_n
    positive_target = target[target > 0.0]
    target_scale = float(np.max(positive_target)) if positive_target.size else 1.0
    floor = max(float(np.finfo(np.float64).tiny), 1.0e-300 * target_scale)
    row_weights = 1.0 / np.maximum(np.abs(target), floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_element_limits = np.where(ac > 0.0, target[:, None] / ac, np.inf)
    capacity = np.min(per_element_limits, axis=0)
    finite_capacity = np.isfinite(capacity) & (capacity > 0.0)

    initial_report = _full_condensate_budget_gate_report_for_support_state(
        setup=setup,
        gas_ln_n=gas_ln_n,
        support_indices=support,
        support_amounts=jnp.asarray(amounts),
        external_condensate_amounts=external,
        element_inventory_target=element_inventory_target,
        relative_tolerance=relative_tolerance,
    )
    accepted = bool(initial_report["accepted"])
    iteration_count = 0
    cap_count_total = 0
    top_up_count = 0
    for iteration in range(int(max_iterations)):
        if accepted:
            break
        budget = gas_budget + ac @ amounts + external_budget - target
        jac = ac * amounts[None, :]
        if jac.size == 0 or jac.shape[1] == 0:
            break
        matrix = jac * row_weights[:, None]
        rhs = -budget * row_weights
        delta_r, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
        if not np.all(np.isfinite(delta_r)):
            break
        norm_inf = float(np.max(np.abs(delta_r))) if delta_r.size else 0.0
        if norm_inf > max_abs_delta_r and norm_inf > 0.0:
            delta_r = delta_r * (max_abs_delta_r / norm_inf)
        trial = amounts * np.exp(delta_r)
        if np.any(finite_capacity):
            before = trial.copy()
            trial[finite_capacity] = np.minimum(
                trial[finite_capacity],
                capacity[finite_capacity],
            )
            cap_count_total += int(np.count_nonzero(trial < before))
        if not np.all(np.isfinite(trial)) or np.any(trial < 0.0):
            break
        amounts = trial
        iteration_count = iteration + 1
        report = _full_condensate_budget_gate_report_for_support_state(
            setup=setup,
            gas_ln_n=gas_ln_n,
            support_indices=support,
            support_amounts=jnp.asarray(amounts),
            external_condensate_amounts=external,
            element_inventory_target=element_inventory_target,
            relative_tolerance=relative_tolerance,
        )
        accepted = bool(report["accepted"])

    for _ in range(8):
        report = _full_condensate_budget_gate_report_for_support_state(
            setup=setup,
            gas_ln_n=gas_ln_n,
            support_indices=support,
            support_amounts=jnp.asarray(amounts),
            external_condensate_amounts=external,
            element_inventory_target=element_inventory_target,
            relative_tolerance=relative_tolerance,
        )
        if bool(report["accepted"]):
            break
        signed_relative = np.asarray(
            report["element_signed_relative_residual"],
            dtype=np.float64,
        )
        for index, element in enumerate(setup.elements):
            if str(element) in {"e-", "electron"}:
                signed_relative[index] = 0.0
        element_index = int(np.argmax(np.abs(signed_relative)))
        if signed_relative[element_index] >= 0.0:
            break
        deficit = -float(signed_relative[element_index]) * max(
            abs(float(target[element_index])),
            1.0e-300,
        )
        stoich = ac[element_index, :]
        room = np.where(
            finite_capacity & (stoich > 0.0),
            np.maximum(capacity - amounts, 0.0),
            0.0,
        )
        if not np.any(room > 0.0):
            break
        candidate_scores = room * stoich
        condensate_index = int(np.argmax(candidate_scores))
        if candidate_scores[condensate_index] <= 0.0:
            break
        increase = min(room[condensate_index], deficit / stoich[condensate_index])
        if increase <= 0.0 or not np.isfinite(increase):
            break
        amounts[condensate_index] += increase
        top_up_count += 1

    final_report = _full_condensate_budget_gate_report_for_support_state(
        setup=setup,
        gas_ln_n=gas_ln_n,
        support_indices=support,
        support_amounts=jnp.asarray(amounts),
        external_condensate_amounts=external,
        element_inventory_target=element_inventory_target,
        relative_tolerance=relative_tolerance,
    )
    polish_report = {
        "polish_schema": "exogibbs_full_condensate_budget_amount_polish_v1",
        "triggered": not bool(initial_report["accepted"]),
        "accepted": bool(final_report["accepted"]),
        "iteration_count": iteration_count,
        "capacity_cap_count": cap_count_total,
        "capacity_top_up_count": top_up_count,
        "initial_full_condensate_budget_gate": initial_report,
        "final_full_condensate_budget_gate": final_report,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }
    if final_report["accepted"]:
        return jnp.asarray(amounts, dtype=jnp.float64), polish_report
    return jnp.asarray(support_amounts, dtype=jnp.float64), polish_report


def _least_squares_element_potential(
    *,
    formula_matrix: Array,
    gas_ln_n: Array,
    gas_stationarity_source: Array,
) -> Array:
    ag = jnp.asarray(formula_matrix)
    q = jnp.asarray(gas_ln_n)
    source = jnp.asarray(gas_stationarity_source)
    if ag.ndim != 2:
        raise ValueError("formula_matrix must be two-dimensional.")
    if q.ndim != 1 or source.ndim != 1 or q.shape != source.shape:
        raise ValueError("gas_ln_n and gas_stationarity_source must be same-length vectors.")
    if ag.shape[1] != q.shape[0]:
        raise ValueError("formula_matrix column count must match gas_ln_n length.")
    return jnp.linalg.lstsq(ag.T, q + source, rcond=None)[0]


def _head_lifecycle_primary_summary(*, solver_success: bool) -> Mapping[str, Any]:
    if solver_success:
        return {
            "row_status": "centered",
            "converged_at_final_barrier": True,
            "reason": "restricted_solver_success_used_as_head_lifecycle_primary_boundary",
        }
    return {
        "row_status": "not_centered",
        "converged_at_final_barrier": False,
        "reason": "restricted_solver_failed_before_head_lifecycle_primary_boundary",
    }


def _head_lifecycle_primary_policy(options: CondensateEquilibriumOptions) -> Mapping[str, Any]:
    policy: dict[str, Any] = {}
    if options.max_outer_iterations is not None:
        policy["max_outer_iterations"] = int(options.max_outer_iterations)
    if options.max_inner_iterations is not None:
        policy["max_inner_iterations"] = int(options.max_inner_iterations)
    if options.head_route_primary_center_tolerance_multiplier is not None:
        policy["center_tolerance_multiplier"] = float(
            options.head_route_primary_center_tolerance_multiplier
        )
    if options.head_route_primary_residual_worsening_tolerance is not None:
        policy["residual_worsening_tolerance"] = float(
            options.head_route_primary_residual_worsening_tolerance
        )
    if options.head_route_primary_require_residual_nonworsening is not None:
        policy["require_residual_nonworsening"] = bool(
            options.head_route_primary_require_residual_nonworsening
        )
    return policy


def _head_route_selected_route_override(options: CondensateEquilibriumOptions) -> str | None:
    if options.case_id is None:
        return None
    if options.head_route_primary_summary is None and options.head_route_refresh_policy_summary is None:
        return None
    return options.selected_route


def _run_lifecycle_from_warm_start_candidate(
    *,
    setup: CondensateChemicalSetup,
    T: float,
    P: float,
    Pref: float,
    b: Array,
    options: CondensateEquilibriumOptions,
    candidate: Any,
) -> Mapping[str, Any]:
    if candidate is None or candidate.initial_log_state_override is None:
        return {
            "report_schema": "exogibbs_condensate_head_route_lifecycle_report_v1",
            "explicit_opt_in": True,
            "production_behavior_change": False,
            "production_return_signature_change": False,
            "preset_default_wiring_change": False,
            "fastchem4_trace_public_runtime_constructor_inputs_used": False,
            "case_id": "runtime_layer",
            "family": "runtime_layer",
            "lifecycle_skipped_reason": "restricted_solver_failed_without_refresh_warm_start_state",
            "route_result": {
                "result_schema": "exogibbs_condensate_head_route_result_v1",
                "case_id": "runtime_layer",
                "family": "runtime_layer",
                "selected_route": options.selected_route,
                "integrated_status": "not_accepted",
                "metric_status": options.metric_status or "runtime_solver_failed",
                "acceptance_tier": "runtime_solver_failed",
                "standard_path_status": NOT_CONVERGED,
                "converged": False,
                "warning_messages": (
                    "The restricted support solver failed and no refresh warm-start state was available.",
                ),
                "diagnostics": {},
            },
        }
    from exogibbs.condensates.head_route_lifecycle import (
        run_condensate_head_route_lifecycle,
    )

    init_state = candidate.initial_log_state_override
    ln_nk = jnp.asarray(init_state.ln_nk)
    ln_mk = jnp.asarray(init_state.ln_mk)
    support_indices = tuple(int(index) for index in candidate.support_indices)
    if ln_mk.shape[0] == len(setup.condensate_species):
        support_amounts = jnp.exp(ln_mk[jnp.asarray(support_indices, dtype=jnp.int32)])
    else:
        support_amounts = jnp.exp(ln_mk)
    gas_stationarity_source = (
        jnp.asarray(setup.gas_setup.hvector_func(float(T)))
        + _ln_normalized_pressure(P, Pref)
    )
    element_potential = _least_squares_element_potential(
        formula_matrix=setup.formula_matrix,
        gas_ln_n=ln_nk,
        gas_stationarity_source=gas_stationarity_source,
    )
    condensate_hvector = jnp.asarray(setup.condensate_setup.hvector_func(float(T)))
    try:
        lifecycle_report = run_condensate_head_route_lifecycle(
            explicit_opt_in=True,
            case_id=options.case_id or "runtime_layer",
            ln_nk=ln_nk,
            support_indices=support_indices,
            support_amounts=support_amounts,
            formula_matrix=setup.formula_matrix,
            formula_matrix_cond=setup.formula_matrix_cond,
            element_inventory_target=jnp.asarray(b),
            element_potential=element_potential,
            gas_stationarity_source=gas_stationarity_source,
            condensate_standard_source=jnp.asarray(
                [condensate_hvector[index] for index in support_indices]
            ),
            primary_summary=options.head_route_primary_summary,
            primary_continuation_policy=_head_lifecycle_primary_policy(options),
            refresh_policy_summary=options.head_route_refresh_policy_summary,
            primary_acceptance_guard=options.head_route_primary_acceptance_guard,
            primary_guard_max_budget=options.head_route_primary_guard_max_budget,
            primary_guard_max_amount_weighted_gas=(
                options.head_route_primary_guard_max_amount_weighted_gas
            ),
            primary_guard_max_gas_stationarity=(
                options.head_route_primary_guard_max_gas_stationarity
            ),
            primary_guard_max_condensate_stationarity=(
                options.head_route_primary_guard_max_condensate_stationarity
            ),
            metric_status=options.metric_status,
            selected_route_override=_head_route_selected_route_override(options),
            field_provenance={
                "ln_nk": "exogibbs_head_route_refresh_warm_start",
                "support_indices": "exogibbs_head_route_warm_start_candidate",
                "support_amounts": "exogibbs_head_route_warm_start_candidate",
                "element_potential": "exogibbs_native_least_squares_gas_gauge",
            },
        )
        return lifecycle_report.as_dict()
    except Exception as exc:  # noqa: BLE001 - runtime diagnostics preserve the failure.
        return {
            "report_schema": "exogibbs_condensate_head_route_lifecycle_report_v1",
            "explicit_opt_in": True,
            "production_behavior_change": False,
            "production_return_signature_change": False,
            "preset_default_wiring_change": False,
            "fastchem4_trace_public_runtime_constructor_inputs_used": False,
            "case_id": "runtime_layer",
            "family": "runtime_layer",
            "lifecycle_failed_reason": f"{type(exc).__name__}: {exc}",
            "route_result": {
                "result_schema": "exogibbs_condensate_head_route_result_v1",
                "case_id": "runtime_layer",
                "family": "runtime_layer",
                "selected_route": options.selected_route,
                "integrated_status": "not_accepted",
                "metric_status": options.metric_status or "runtime_lifecycle_failed",
                "acceptance_tier": "runtime_lifecycle_failed",
                "standard_path_status": NOT_CONVERGED,
                "converged": False,
                "warning_messages": (
                    "The HEAD route lifecycle failed from the refresh warm-start state.",
                ),
                "diagnostics": {"exception_type": type(exc).__name__},
            },
        }


def _run_lifecycle_from_native_state(
    *,
    setup: CondensateChemicalSetup,
    T: float,
    P: float,
    Pref: float,
    b: Array,
    options: CondensateEquilibriumOptions,
    ln_nk: Array,
    support_indices: Sequence[int],
    support_amounts: Array,
    external_condensate_amounts: Sequence[float] | Array | None = None,
    element_potential: Array | None,
    element_potential_source: str,
    field_source: str,
    primary_continuation_policy: Mapping[str, Any],
):
    from exogibbs.condensates.head_route_lifecycle import (
        run_condensate_head_route_lifecycle,
    )

    gas_stationarity_source = (
        jnp.asarray(setup.gas_setup.hvector_func(float(T)))
        + _ln_normalized_pressure(P, Pref)
    )
    ln_nk_array = jnp.asarray(ln_nk, dtype=jnp.float64)
    if element_potential is None:
        element_potential = _least_squares_element_potential(
            formula_matrix=setup.formula_matrix,
            gas_ln_n=ln_nk_array,
            gas_stationarity_source=gas_stationarity_source,
        )
    element_potential_array = jnp.asarray(element_potential, dtype=jnp.float64)
    support = tuple(int(index) for index in support_indices)
    support_amount_array = jnp.maximum(
        jnp.asarray(support_amounts, dtype=jnp.float64),
        jnp.asarray(1.0e-300, dtype=jnp.float64),
    )
    external_budget = None
    if external_condensate_amounts is not None:
        external_amounts = jnp.asarray(external_condensate_amounts, dtype=jnp.float64)
        if external_amounts.ndim != 1 or external_amounts.shape[0] != len(setup.condensate_species):
            raise ValueError("external_condensate_amounts must have one value per condensate species.")
        external_budget = jnp.asarray(setup.formula_matrix_cond, dtype=jnp.float64) @ external_amounts
    condensate_hvector = jnp.asarray(setup.condensate_setup.hvector_func(float(T)))
    return run_condensate_head_route_lifecycle(
        explicit_opt_in=True,
        case_id=options.case_id or "runtime_layer",
        ln_nk=ln_nk_array,
        support_indices=support,
        support_amounts=support_amount_array,
        formula_matrix=setup.formula_matrix,
        formula_matrix_cond=setup.formula_matrix_cond,
        element_inventory_target=jnp.asarray(b),
        element_potential=element_potential_array,
        gas_stationarity_source=gas_stationarity_source,
        condensate_standard_source=jnp.asarray(
            [condensate_hvector[index] for index in support]
        ),
        external_condensate_budget=external_budget,
        primary_summary=options.head_route_primary_summary,
        primary_continuation_policy=primary_continuation_policy,
        refresh_policy_summary=options.head_route_refresh_policy_summary,
        primary_acceptance_guard=options.head_route_primary_acceptance_guard,
        primary_guard_max_budget=options.head_route_primary_guard_max_budget,
        primary_guard_max_amount_weighted_gas=(
            options.head_route_primary_guard_max_amount_weighted_gas
        ),
        primary_guard_max_gas_stationarity=(
            options.head_route_primary_guard_max_gas_stationarity
        ),
        primary_guard_max_condensate_stationarity=(
            options.head_route_primary_guard_max_condensate_stationarity
        ),
        metric_status=options.metric_status,
        selected_route_override=_head_route_selected_route_override(options),
        field_provenance={
            "ln_nk": field_source,
            "support_indices": field_source,
            "support_amounts": field_source,
            "element_potential": element_potential_source,
        },
    )


def _run_lifecycle_from_restricted_solver_state(
    *,
    setup: CondensateChemicalSetup,
    T: float,
    P: float,
    Pref: float,
    b: Array,
    options: CondensateEquilibriumOptions,
    solver: Mapping[str, Any],
    solver_ln_nk: Array,
    solver_support_indices: Sequence[int],
    solver_support_amounts: Array,
    primary_continuation_policy: Mapping[str, Any],
):
    if "pi_vector" in solver:
        element_potential = jnp.asarray(solver["pi_vector"], dtype=jnp.float64)
        element_potential_source = "exogibbs_restricted_solver_dual"
    else:
        element_potential = None
        element_potential_source = "exogibbs_native_least_squares_gas_gauge"
    return _run_lifecycle_from_native_state(
        setup=setup,
        T=T,
        P=P,
        Pref=Pref,
        b=b,
        options=options,
        ln_nk=solver_ln_nk,
        support_indices=solver_support_indices,
        support_amounts=solver_support_amounts,
        external_condensate_amounts=None,
        element_potential=element_potential,
        element_potential_source=element_potential_source,
        field_source="exogibbs_restricted_support_solver_output",
        primary_continuation_policy=primary_continuation_policy,
    )


def _is_current_barrier_center_gate_block(lifecycle_payload: Mapping[str, Any]) -> bool:
    primary_execution = lifecycle_payload.get("primary_execution_report")
    if not isinstance(primary_execution, Mapping):
        return False
    continuation = primary_execution.get("continuation_report")
    if not isinstance(continuation, Mapping):
        return False
    return str(continuation.get("stopped_reason")) == "current_barrier_not_centered"


def _is_residual_nonworsening_candidate_block(lifecycle_payload: Mapping[str, Any]) -> bool:
    primary_execution = lifecycle_payload.get("primary_execution_report")
    if not isinstance(primary_execution, Mapping):
        return False
    continuation = primary_execution.get("continuation_report")
    if not isinstance(continuation, Mapping):
        return False
    if str(continuation.get("stopped_reason")) != "no_p_armijo_trial":
        return False
    outer_records = continuation.get("outer_records", ())
    if not outer_records:
        return False
    final_outer = outer_records[-1]
    if not isinstance(final_outer, Mapping):
        return False
    inner_records = final_outer.get("inner_records", ())
    if not inner_records:
        return False
    final_inner = inner_records[-1]
    if not isinstance(final_inner, Mapping):
        return False
    if final_inner.get("selected_trial") is not None:
        return False
    for direction in final_inner.get("direction_records", ()):
        if not isinstance(direction, Mapping):
            continue
        p_selection = direction.get("p_armijo_selection")
        filter_selection = direction.get("filter_selection")
        if isinstance(p_selection, Mapping) and bool(p_selection.get("selected", False)):
            return True
        if isinstance(filter_selection, Mapping) and bool(filter_selection.get("selected", False)):
            return True
    return False


def _status_from_metric_status(
    *,
    metric_status: Optional[str],
    selected_route: str,
    solver_success: bool,
    allow_caveat_tiers: bool,
) -> tuple[str, str, tuple[str, ...]]:
    if metric_status is None:
        return ("runtime_unclassified", CONVERGED if solver_success else NOT_CONVERGED, ())
    if not solver_success:
        return (
            "runtime_solver_failed",
            NOT_CONVERGED,
            ("The restricted support solver did not report success.",),
        )
    gate = classify_head_route_standard_gate_row(
        condensate_enabled=True,
        case_id="runtime_layer",
        family="runtime_layer",
        selected_route=selected_route,
        metric_status=metric_status,
    )
    if gate.standard_path_status == CONVERGED_WITH_CAVEAT and not allow_caveat_tiers:
        return (gate.acceptance_tier, NOT_CONVERGED, gate.warning_messages)
    return (gate.acceptance_tier, gate.standard_path_status, gate.warning_messages)


def build_condensate_equilibrium_result_from_solver_payload(
    *,
    setup: CondensateChemicalSetup,
    gas_ln_n: Sequence[float],
    support_indices: Sequence[int],
    support_amounts: Sequence[float],
    external_condensate_amounts: Sequence[float] | Array | None = None,
    selected_route: str,
    metric_status: Optional[str],
    solver_success: bool,
    allow_caveat_tiers: bool = True,
    diagnostics: Optional[Mapping[str, Any]] = None,
    element_inventory_target: Array | None = None,
    enable_full_condensate_budget_residual_gate: bool = True,
    full_condensate_budget_relative_tolerance: float = 1.0e-3,
) -> CondensateEquilibriumResult:
    """Build a production-facing condensate result from explicit solver arrays."""

    validate_condensate_chemical_setup(setup)
    gas_ln_n_array = jnp.asarray(gas_ln_n)
    if gas_ln_n_array.ndim != 1 or gas_ln_n_array.shape[0] != len(setup.gas_species):
        raise ValueError("gas_ln_n must have one value per gas species.")
    gas_n = jnp.exp(gas_ln_n_array)
    gas_ntot = jnp.sum(gas_n)
    gas_x = gas_n / jnp.clip(gas_ntot, 1.0e-300)
    condensate_amounts = _full_condensate_amounts(
        support_indices=support_indices,
        support_amounts=jnp.asarray(support_amounts),
        condensate_count=len(setup.condensate_species),
    )
    condensate_amounts = _merge_external_condensate_amounts(
        condensate_amounts=condensate_amounts,
        external_condensate_amounts=external_condensate_amounts,
    )
    acceptance_tier, status, warnings = _status_from_metric_status(
        metric_status=metric_status,
        selected_route=selected_route,
        solver_success=solver_success,
        allow_caveat_tiers=allow_caveat_tiers,
    )
    support_index_array = jnp.asarray(support_indices, dtype=jnp.int32)
    support_names = tuple(setup.condensate_species[int(index)] for index in support_index_array.tolist())
    metadata: dict[str, Any] = dict(diagnostics or {})
    metadata.setdefault("route", HEAD_ROUTE_STANDARD)
    metadata.setdefault("head_route_version", CONDENSATE_HEAD_ROUTE_VERSION)
    metadata.setdefault("head_route_name", CONDENSATE_HEAD_ROUTE_NAME)
    metadata.setdefault("selected_route", selected_route)
    metadata.setdefault("acceptance_tier", acceptance_tier)
    metadata.setdefault("warning_messages", warnings)
    metadata.setdefault("fastchem4_trace_public_runtime_constructor_inputs_used", False)
    status, acceptance_tier, warnings, metadata = _apply_full_condensate_budget_residual_gate(
        setup=setup,
        gas_n=gas_n,
        condensate_amounts=condensate_amounts,
        element_inventory_target=element_inventory_target,
        status=status,
        acceptance_tier=acceptance_tier,
        warning_messages=warnings,
        metadata=metadata,
        enabled=enable_full_condensate_budget_residual_gate,
        relative_tolerance=full_condensate_budget_relative_tolerance,
    )
    metadata["acceptance_tier"] = acceptance_tier
    metadata["warning_messages"] = warnings
    return CondensateEquilibriumResult(
        gas_ln_n=gas_ln_n_array,
        gas_n=gas_n,
        gas_x=gas_x,
        gas_ntot=gas_ntot,
        condensate_amounts=condensate_amounts,
        condensate_support_indices=support_index_array,
        condensate_support_names=support_names,
        acceptance_tier=acceptance_tier,
        selected_route=selected_route,
        status=status,
        converged=status in {CONVERGED, CONVERGED_WITH_CAVEAT},
        diagnostics=metadata,
    )


def _build_empty_support_gas_result(
    *,
    setup: CondensateChemicalSetup,
    gas_ln_n: Sequence[float],
    diagnostics: Optional[Mapping[str, Any]],
    element_inventory_target: Array | None = None,
    enable_full_condensate_budget_residual_gate: bool = True,
    full_condensate_budget_relative_tolerance: float = 1.0e-3,
) -> CondensateEquilibriumResult:
    gas_ln_n_array = jnp.asarray(gas_ln_n)
    gas_n = jnp.exp(gas_ln_n_array)
    gas_ntot = jnp.sum(gas_n)
    gas_x = gas_n / jnp.clip(gas_ntot, 1.0e-300)
    metadata = dict(diagnostics or {})
    metadata.setdefault("route", HEAD_ROUTE_STANDARD)
    metadata.setdefault("head_route_version", CONDENSATE_HEAD_ROUTE_VERSION)
    metadata.setdefault("head_route_name", CONDENSATE_HEAD_ROUTE_NAME)
    metadata.setdefault("selected_route", "head_v1_empty_positive_support_gas_only")
    metadata.setdefault("acceptance_tier", "runtime_empty_positive_support")
    metadata.setdefault("warning_messages", ())
    metadata.setdefault("fastchem4_trace_public_runtime_constructor_inputs_used", False)
    condensate_amounts = jnp.zeros((len(setup.condensate_species),), dtype=gas_n.dtype)
    status, acceptance_tier, warnings, metadata = _apply_full_condensate_budget_residual_gate(
        setup=setup,
        gas_n=gas_n,
        condensate_amounts=condensate_amounts,
        element_inventory_target=element_inventory_target,
        status=CONVERGED,
        acceptance_tier="runtime_empty_positive_support",
        warning_messages=(),
        metadata=metadata,
        enabled=enable_full_condensate_budget_residual_gate,
        relative_tolerance=full_condensate_budget_relative_tolerance,
    )
    metadata["acceptance_tier"] = acceptance_tier
    metadata["warning_messages"] = warnings
    return CondensateEquilibriumResult(
        gas_ln_n=gas_ln_n_array,
        gas_n=gas_n,
        gas_x=gas_x,
        gas_ntot=gas_ntot,
        condensate_amounts=condensate_amounts,
        condensate_support_indices=jnp.asarray((), dtype=jnp.int32),
        condensate_support_names=(),
        acceptance_tier=acceptance_tier,
        selected_route="head_v1_empty_positive_support_gas_only",
        status=status,
        converged=status in {CONVERGED, CONVERGED_WITH_CAVEAT},
        diagnostics=metadata,
    )


def _build_native_seed_fallback_result(
    *,
    setup: CondensateChemicalSetup,
    T: float,
    P: float,
    b: Array,
    Pref: float,
    candidate: Any,
    support_selection_report: Mapping[str, Any] | None,
    warm_start_report: Any,
    solver_attempts: Sequence[Mapping[str, Any]],
    selected_warm_start_candidate: Mapping[str, Any] | None,
    lifecycle_payload: Mapping[str, Any],
    allow_caveat_tiers: bool,
    return_diagnostics: bool,
    enable_full_condensate_budget_residual_gate: bool = True,
    full_condensate_budget_relative_tolerance: float = 1.0e-3,
    restricted_solver_success: bool = False,
    restricted_solver_payload: Mapping[str, Any] | None = None,
) -> CondensateEquilibriumResult:
    from exogibbs.api.equilibrium import EquilibriumOptions, equilibrium

    if candidate.initial_log_state_override is not None:
        fallback_gas_ln_n = jnp.asarray(candidate.initial_log_state_override.ln_nk)
        fallback_gas_source = "selected_warm_start_candidate_gas_state"
    else:
        gas_result = equilibrium(
            setup.gas_setup,
            T,
            P,
            jnp.asarray(b),
            Pref=Pref,
            options=EquilibriumOptions(),
            return_diagnostics=False,
        )
        fallback_gas_ln_n = gas_result.ln_n
        fallback_gas_source = "native_gas_equilibrium"
    diagnostics_payload: Optional[Mapping[str, Any]]
    if return_diagnostics:
        diagnostics_payload = {
            "restricted_solver_success": bool(restricted_solver_success),
            "solver_success": True,
            "support_selection": support_selection_report,
            "head_route_warm_start": warm_start_report.as_dict(),
            "head_route_solver_attempts": tuple(solver_attempts),
            "selected_warm_start_candidate": selected_warm_start_candidate,
            "head_route_lifecycle": lifecycle_payload,
            "restricted_solver_payload_for_support_growth": None
            if restricted_solver_payload is None
            else {
                "ln_nk": restricted_solver_payload.get("ln_nk"),
                "support_indices": restricted_solver_payload.get("support_indices"),
                "m_support": restricted_solver_payload.get("m_support"),
                "pi_vector": restricted_solver_payload.get("pi_vector"),
                "max_positive_inactive_driving": restricted_solver_payload.get(
                    "max_positive_inactive_driving"
                ),
                "top_positive_inactive_indices": restricted_solver_payload.get(
                    "top_positive_inactive_indices"
                ),
                "restricted_kkt_gap_log_variable_inf": restricted_solver_payload.get(
                    "restricted_kkt_gap_log_variable_inf"
                ),
            },
            "native_seed_fallback": {
                "fallback_schema": "exogibbs_native_budget_seed_fallback_v1",
                "selected_policy": "native_budget_seed_fallback_budget_tradeoff",
                "accepted": True,
                "fallback_gas_source": fallback_gas_source,
                "reason": (
                    "The primary lifecycle did not converge or was not accepted; "
                    "the API returned the best available native gas boundary with the "
                    "budget-preserving condensate seed as a caveat-bearing HEAD route fallback."
                ),
                "fastchem4_trace_public_runtime_constructor_inputs_used": False,
            },
        }
        for lifecycle_key, diagnostic_key in (
            ("center_gate_retry_report", "head_route_center_gate_retry"),
            (
                "residual_worsening_retry_report",
                "head_route_residual_worsening_retry",
            ),
            (
                "soft_restoration_retry_report",
                "head_route_soft_restoration_retry",
            ),
            (
                "ipopt_h_type_retry_report",
                "head_route_ipopt_h_type_retry",
            ),
        ):
            retry_report = lifecycle_payload.get(lifecycle_key)
            if isinstance(retry_report, Mapping):
                diagnostics_payload[diagnostic_key] = retry_report
    else:
        diagnostics_payload = None
    return build_condensate_equilibrium_result_from_solver_payload(
        setup=setup,
        gas_ln_n=fallback_gas_ln_n,
        support_indices=tuple(int(index) for index in candidate.support_indices),
        support_amounts=tuple(float(value) for value in candidate.support_amounts_init),
        selected_route="native_budget_seed_fallback_budget_tradeoff",
        metric_status=BUDGET_TRADEOFF_STATUS,
        solver_success=True,
        allow_caveat_tiers=allow_caveat_tiers,
        diagnostics=diagnostics_payload,
        element_inventory_target=b,
        enable_full_condensate_budget_residual_gate=(
            enable_full_condensate_budget_residual_gate
        ),
        full_condensate_budget_relative_tolerance=(
            full_condensate_budget_relative_tolerance
        ),
    )


def _activity_driven_support_report(
    *,
    setup: CondensateChemicalSetup,
    T: float,
    P: float,
    b: Array,
    Pref: float,
    gas_ln_n: Array,
    options: CondensateEquilibriumOptions,
    existing_support_indices: Sequence[int] = (),
    max_positive_support_count: int | None = None,
    element_potential_override: Array | None = None,
) -> Mapping[str, Any]:
    from exogibbs.condensates.support_selection_policy import (
        select_activity_driven_support_candidates,
    )

    gas_stationarity_source = (
        jnp.asarray(setup.gas_setup.hvector_func(float(T)))
        + _ln_normalized_pressure(P, Pref)
    )
    element_potential = (
        jnp.asarray(element_potential_override, dtype=jnp.float64)
        if element_potential_override is not None
        else _least_squares_element_potential(
            formula_matrix=setup.formula_matrix,
            gas_ln_n=jnp.asarray(gas_ln_n),
            gas_stationarity_source=gas_stationarity_source,
        )
    )
    report = select_activity_driven_support_candidates(
        formula_matrix_cond=setup.formula_matrix_cond,
        element_inventory_target=jnp.asarray(b),
        condensate_species_order=setup.condensate_species,
        hvector_cond=setup.condensate_setup.hvector_func(float(T)),
        element_potential=element_potential,
        max_positive_support_count=(
            options.max_activity_support_count
            if max_positive_support_count is None
            else int(max_positive_support_count)
        ),
        activity_threshold=options.support_activity_threshold,
        existing_support_indices=existing_support_indices,
        temperature=float(T),
        condensate_temperature_validity_upper=setup.condensate_setup.metadata.get(
            "temperature_validity_upper"
        )
        if setup.condensate_setup.metadata is not None
        else None,
        field_provenance={
            "formula_matrix_cond": "exogibbs_condensate_chemical_setup",
            "element_inventory_target": "exogibbs_runtime_input",
            "hvector_cond": "exogibbs_condensate_thermochemistry",
            "element_potential": "exogibbs_restricted_solver_dual"
            if element_potential_override is not None
            else "exogibbs_native_least_squares_gas_gauge",
            "condensate_temperature_validity_upper": "exogibbs_condensate_chemical_setup_metadata",
        },
    )
    return report.as_dict()


def _support_count_cap(options: CondensateEquilibriumOptions) -> int | None:
    return None if options.max_positive_support_count is None else int(options.max_positive_support_count)


def _support_cap_retry_sequence(options: CondensateEquilibriumOptions) -> tuple[int, ...]:
    counts = (
        (int(options.support_cap_retry_count),)
        if options.support_cap_retry_counts is None
        else tuple(int(count) for count in options.support_cap_retry_counts)
    )
    return tuple(dict.fromkeys(sorted(counts)))


def _support_growth_staging_retry_sequence(
    options: CondensateEquilibriumOptions,
) -> tuple[int, ...]:
    counts = (
        ()
        if options.support_growth_staging_retry_add_per_rounds is None
        else tuple(int(count) for count in options.support_growth_staging_retry_add_per_rounds)
    )
    return tuple(dict.fromkeys(counts))


def _remaining_support_slots(
    support_count: int,
    options: CondensateEquilibriumOptions,
) -> int | None:
    cap = _support_count_cap(options)
    if cap is None:
        return None
    return max(0, cap - int(support_count))


def _support_add_count(
    *,
    inactive_count: int,
    support_count: int,
    options: CondensateEquilibriumOptions,
    allow_additions: bool = True,
) -> int:
    if not allow_additions:
        return 0
    remaining = _remaining_support_slots(support_count, options)
    if remaining == 0:
        return 0
    add_limit = (
        int(inactive_count)
        if options.max_support_add_per_round is None
        else int(options.max_support_add_per_round)
    )
    if remaining is not None:
        add_limit = min(add_limit, remaining)
    return min(add_limit, int(inactive_count))


def _support_closure_retry_gate_report(
    *,
    setup: CondensateChemicalSetup,
    T: float,
    P: float,
    b: Array,
    Pref: float,
    result: CondensateEquilibriumResult,
    options: CondensateEquilibriumOptions,
) -> Mapping[str, Any]:
    """Report whether a retry result has acceptable inactive support closure."""

    if not options.enable_support_closure_retry_gate:
        return {
            "gate_schema": "exogibbs_support_closure_retry_gate_v1",
            "enabled": False,
            "accepted": True,
            "max_positive_inactive_driving": 0.0,
            "positive_inactive_count": 0,
            "fastchem4_trace_public_runtime_constructor_inputs_used": False,
        }
    try:
        report = _activity_driven_support_report(
            setup=setup,
            T=T,
            P=P,
            b=b,
            Pref=Pref,
            gas_ln_n=result.gas_ln_n,
            options=options,
            existing_support_indices=tuple(
                int(index) for index in result.condensate_support_indices.tolist()
            ),
        )
    except (TypeError, ValueError, RuntimeError, KeyError) as exc:
        return {
            "gate_schema": "exogibbs_support_closure_retry_gate_v1",
            "enabled": True,
            "accepted": False,
            "error": f"{type(exc).__name__}: {exc}",
            "max_positive_inactive_driving": float("inf"),
            "positive_inactive_count": -1,
            "fastchem4_trace_public_runtime_constructor_inputs_used": False,
        }
    inactive = tuple(int(index) for index in report.get("inactive_positive_indices", ()))
    driving = report.get("candidate_driving", {})
    top_rows = sorted(
        (
            {
                "index": index,
                "species": str(setup.condensate_species[index]),
                "driving": float(driving.get(str(setup.condensate_species[index]), 0.0)),
            }
            for index in inactive
        ),
        key=lambda row: row["driving"],
        reverse=True,
    )
    max_driving = float(top_rows[0]["driving"]) if top_rows else 0.0
    tolerance = float(options.support_closure_max_positive_inactive_driving)
    return {
        "gate_schema": "exogibbs_support_closure_retry_gate_v1",
        "enabled": True,
        "accepted": bool(max_driving <= tolerance),
        "max_positive_inactive_driving": max_driving,
        "max_positive_inactive_driving_tolerance": tolerance,
        "positive_inactive_count": len(inactive),
        "top_positive_inactive": tuple(top_rows[:20]),
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }


def _budget_seed_for_support(
    *,
    setup: CondensateChemicalSetup,
    b: Array,
    support_indices: Sequence[int],
    options: CondensateEquilibriumOptions,
) -> tuple[float, ...]:
    from exogibbs.condensates.initialization_policy import (
        recommend_budget_preserving_seed_amounts,
    )

    seed = recommend_budget_preserving_seed_amounts(
        formula_matrix_cond=setup.formula_matrix_cond,
        element_inventory_target=jnp.asarray(b),
        condensate_species_order=setup.condensate_species,
        support_indices=support_indices,
        seed_fraction=1.0
        if options.seed_initialization_policy == "max_density"
        else options.seed_fraction,
        max_seed_amount=1.0e300
        if options.seed_initialization_policy == "max_density"
        else options.max_seed_amount,
        min_seed_amount=options.min_seed_amount,
        preserve_budget_fraction=(
            options.seed_initialization_policy == "budget_preserving_fraction"
        ),
        field_provenance={
            "formula_matrix_cond": "exogibbs_condensate_chemical_setup",
            "element_inventory_target": "exogibbs_runtime_input",
            "recommended_amounts": (
                "derived_from_native_budget_capacity_with_shared_budget_fraction"
                if options.seed_initialization_policy == "budget_preserving_fraction"
                else "derived_from_native_budget_capacity_without_shared_budget_rescale"
            ),
        },
    )
    return tuple(float(value) for value in seed.recommended_amounts)


def _positive_support_amounts_for_warm_start(
    amounts: Sequence[float],
    *,
    min_seed_amount: float,
) -> tuple[float, ...]:
    floor = float(min_seed_amount)
    return tuple(
        float(value) if math.isfinite(float(value)) and float(value) > 0.0 else floor
        for value in amounts
    )


def _seed_gauge_payload(options: CondensateEquilibriumOptions) -> Mapping[str, Any]:
    """Describe the native amount gauge used by API-generated condensate seeds."""

    return {
        "seed_initialization_policy": options.seed_initialization_policy,
        "amount_gauge": "element_inventory_target_fraction",
        "fastchem4_first_step_equivalent_gauge": (
            "number_density_divided_by_initial_gas_phase_total_element_density"
        ),
        "fastchem4_constructor_values_used": False,
        "uses_b_not_b_normalized_by_sum_b": True,
        "max_density_formula": (
            "min_positive_element(element_inventory_target[element] / "
            "stoichiometric_coefficient[element, condensate])"
        ),
    }


def _support_selection_payload_from_activity_report(
    *,
    report: Mapping[str, Any],
    support_indices: Sequence[int],
    support_names: Sequence[str],
    support_amounts_init: Sequence[float],
    seed_initialization_policy: str,
    terminated_reason: str,
    outer_iterations: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    support = tuple(int(index) for index in support_indices)
    names = tuple(str(name) for name in support_names)
    amounts = tuple(float(value) for value in support_amounts_init)
    if len(names) != len(support):
        raise ValueError("support_names length must match support_indices length.")
    return {
        "selection_schema": "exogibbs_condensate_activity_driven_support_outer_loop_v1",
        "selection_mode": "activity_driven_support_outer_loop",
        "solver_inputs": {
            "support_indices": support,
            "support_amounts_init": amounts,
            "support_names": names,
            "seed_initialization_policy": str(seed_initialization_policy),
            "amount_gauge": "element_inventory_target_fraction",
            "fastchem4_first_step_equivalent_gauge": (
                "number_density_divided_by_initial_gas_phase_total_element_density"
            ),
            "uses_b_not_b_normalized_by_sum_b": True,
            "empty_positive_support": len(support) == 0,
        },
        "activity_selection": dict(report),
        "outer_loop": {
            "loop_schema": "exogibbs_condensate_support_outer_loop_v1",
            "terminated_reason": terminated_reason,
            "iterations": tuple(outer_iterations),
            "fastchem4_trace_public_runtime_constructor_inputs_used": False,
        },
        "fastchem4_trace_values_used": False,
        "fastchem4_public_values_used_as_constructor_inputs": False,
        "fastchem4_runtime_values_used_as_constructor_inputs": False,
    }


def _with_support_outer_loop_diagnostics(
    *,
    result: CondensateEquilibriumResult,
    support_selection_report: Mapping[str, Any],
    return_diagnostics: bool,
) -> CondensateEquilibriumResult:
    if not return_diagnostics:
        return result
    diagnostics = dict(result.diagnostics or {})
    diagnostics["support_selection"] = support_selection_report
    diagnostics["support_outer_loop"] = support_selection_report.get("outer_loop")
    return replace(result, diagnostics=diagnostics)


def _with_support_cap_retry_diagnostics(
    *,
    result: CondensateEquilibriumResult,
    retry_report: Mapping[str, Any],
    return_diagnostics: bool,
) -> CondensateEquilibriumResult:
    if not return_diagnostics:
        return result
    diagnostics = dict(result.diagnostics or {})
    diagnostics["support_cap_retry"] = retry_report
    return replace(result, diagnostics=diagnostics)


def _with_support_growth_staging_retry_diagnostics(
    *,
    result: CondensateEquilibriumResult,
    retry_report: Mapping[str, Any],
    return_diagnostics: bool,
) -> CondensateEquilibriumResult:
    if not return_diagnostics:
        return result
    diagnostics = dict(result.diagnostics or {})
    diagnostics["support_growth_staging_retry"] = retry_report
    return replace(result, diagnostics=diagnostics)


def _with_support_budget_preserving_seed_retry_diagnostics(
    *,
    result: CondensateEquilibriumResult,
    retry_report: Mapping[str, Any],
    return_diagnostics: bool,
) -> CondensateEquilibriumResult:
    if not return_diagnostics:
        return result
    diagnostics = dict(result.diagnostics or {})
    diagnostics["support_budget_preserving_seed_retry"] = retry_report
    return replace(result, diagnostics=diagnostics)


def _run_activity_driven_support_outer_loop(
    *,
    setup: CondensateChemicalSetup,
    T: float,
    P: float,
    b: Array,
    Pref: float,
    options: CondensateEquilibriumOptions,
) -> CondensateEquilibriumResult:
    from exogibbs.api.equilibrium import EquilibriumOptions, equilibrium

    explicit_options = replace(
        options,
        enable_support_outer_loop=False,
        enable_head_route_center_gate_retry=True,
        enable_head_route_residual_worsening_retry=True,
        enable_head_route_soft_restoration_retry=True,
        enable_head_route_ipopt_h_type_retry=True,
    )
    gas_result = equilibrium(
        setup.gas_setup,
        T,
        P,
        jnp.asarray(b),
        Pref=Pref,
        options=EquilibriumOptions(),
        return_diagnostics=False,
    )
    current_report = _activity_driven_support_report(
        setup=setup,
        T=T,
        P=P,
        b=b,
        Pref=Pref,
        gas_ln_n=gas_result.ln_n,
        options=options,
    )
    initial_positive = tuple(int(index) for index in current_report["positive_support_indices"])
    initial_add_count = _support_add_count(
        inactive_count=len(initial_positive),
        support_count=0,
        options=options,
    )
    current_support = initial_positive[:initial_add_count]
    outer_iterations: list[Mapping[str, Any]] = [
        {
            "iteration": 0,
            "state_source": "native_gas_equilibrium",
            "positive_support_indices": initial_positive,
            "positive_support_names": tuple(current_report["positive_support_names"]),
            "added_support_indices": current_support,
            "added_support_names": tuple(
                setup.condensate_species[int(index)] for index in current_support
            ),
        }
    ]
    if not current_support:
        support_selection_report = _support_selection_payload_from_activity_report(
            report=current_report,
            support_indices=(),
            support_names=(),
            support_amounts_init=(),
            seed_initialization_policy=options.seed_initialization_policy,
            terminated_reason="empty_positive_support",
            outer_iterations=outer_iterations,
        )
        if not options.allow_empty_positive_support:
            raise ValueError("No positive condensate support candidates were selected.")
        diagnostics = (
            {"support_selection": support_selection_report}
            if options.return_diagnostics
            else None
        )
        empty_result = _build_empty_support_gas_result(
            setup=setup,
            gas_ln_n=gas_result.ln_n,
            diagnostics=diagnostics,
            element_inventory_target=b,
            enable_full_condensate_budget_residual_gate=(
                options.enable_full_condensate_budget_residual_gate
            ),
            full_condensate_budget_relative_tolerance=(
                options.full_condensate_budget_relative_tolerance
            ),
        )
        gate = (empty_result.diagnostics or {}).get(
            "full_condensate_budget_residual_gate",
            {},
        )
        if (
            options.enable_full_condensate_budget_residual_gate
            and not empty_result.converged
            and isinstance(gate, Mapping)
            and not bool(gate.get("accepted", True))
        ):
            strict_gas_result = equilibrium(
                setup.gas_setup,
                T,
                P,
                jnp.asarray(b),
                Pref=Pref,
                options=EquilibriumOptions(epsilon_crit=1.0e-12),
                return_diagnostics=False,
            )
            strict_diagnostics = dict(diagnostics or {})
            strict_diagnostics["empty_support_strict_gas_retry"] = {
                "retry_schema": "exogibbs_empty_support_strict_gas_retry_v1",
                "triggered": True,
                "epsilon_crit": 1.0e-12,
                "initial_full_condensate_budget_gate": gate,
                "fastchem4_trace_public_runtime_constructor_inputs_used": False,
            }
            strict_result = _build_empty_support_gas_result(
                setup=setup,
                gas_ln_n=strict_gas_result.ln_n,
                diagnostics=strict_diagnostics,
                element_inventory_target=b,
                enable_full_condensate_budget_residual_gate=(
                    options.enable_full_condensate_budget_residual_gate
                ),
                full_condensate_budget_relative_tolerance=(
                    options.full_condensate_budget_relative_tolerance
                ),
            )
            if options.return_diagnostics:
                strict_gate = (strict_result.diagnostics or {}).get(
                    "full_condensate_budget_residual_gate",
                    {},
                )
                retry_report = dict(
                    (strict_result.diagnostics or {}).get(
                        "empty_support_strict_gas_retry",
                        {},
                    )
                )
                retry_report["accepted"] = bool(strict_result.converged)
                retry_report["retry_full_condensate_budget_gate"] = strict_gate
                strict_result = replace(
                    strict_result,
                    diagnostics={
                        **dict(strict_result.diagnostics or {}),
                        "empty_support_strict_gas_retry": retry_report,
                    },
                )
            if strict_result.converged:
                return strict_result
        return empty_result

    support_amounts = _budget_seed_for_support(
        setup=setup,
        b=b,
        support_indices=current_support,
        options=options,
    )
    last_result: CondensateEquilibriumResult | None = None
    terminated_reason = "max_support_outer_iterations_reached"
    for outer_index in range(1, options.max_support_outer_iterations + 1):
        last_result = condensate_equilibrium(
            setup,
            T,
            P,
            b,
            Pref=Pref,
            support_indices=current_support,
            support_amounts_init=support_amounts,
            options=explicit_options,
        )
        fallback_solver_payload = None
        if last_result.selected_route == "native_budget_seed_fallback_budget_tradeoff":
            fallback_solver_payload = (last_result.diagnostics or {}).get(
                "restricted_solver_payload_for_support_growth"
            )
        if (
            last_result.selected_route == "native_budget_seed_fallback_budget_tradeoff"
            and not fallback_solver_payload
        ):
            terminated_reason = "support_growth_stopped_after_unaccepted_head_route_result"
            outer_iterations.append(
                {
                    "iteration": outer_index,
                    "state_source": "head_route_result",
                    "result_status": last_result.status,
                    "selected_route": last_result.selected_route,
                    "added_support_indices": (),
                    "added_support_names": (),
                    "reason": (
                        "Do not grow activity support from a caveat fallback gas "
                        "state; support additions require an accepted HEAD route "
                        "condensate state."
                    ),
                }
            )
            break
        support_growth_ln_nk = (
            fallback_solver_payload["ln_nk"]
            if fallback_solver_payload
            else last_result.gas_ln_n
        )
        support_growth_existing = (
            tuple(int(index) for index in fallback_solver_payload["support_indices"])
            if fallback_solver_payload
            else tuple(
                int(index) for index in last_result.condensate_support_indices.tolist()
            )
        )
        support_growth_pi = (
            fallback_solver_payload.get("pi_vector")
            if fallback_solver_payload
            else None
        )
        current_report = _activity_driven_support_report(
            setup=setup,
            T=T,
            P=P,
            b=b,
            Pref=Pref,
            gas_ln_n=support_growth_ln_nk,
            options=options,
            existing_support_indices=support_growth_existing,
            element_potential_override=support_growth_pi,
        )
        existing = set(support_growth_existing)
        inactive_positive = tuple(
            int(index)
            for index in current_report["inactive_positive_indices"]
            if int(index) not in existing
        )
        add_count = _support_add_count(
            inactive_count=len(inactive_positive),
            support_count=len(current_support),
            options=options,
            allow_additions=outer_index < int(options.max_support_outer_iterations),
        )
        added = inactive_positive[:add_count]
        outer_iterations.append(
            {
                "iteration": outer_index,
                "state_source": "restricted_solver_output"
                if fallback_solver_payload
                else "head_route_result",
                "selected_route": last_result.selected_route,
                "positive_support_indices": tuple(int(index) for index in current_report["positive_support_indices"]),
                "positive_support_names": tuple(str(name) for name in current_report["positive_support_names"]),
                "inactive_positive_indices": inactive_positive,
                "inactive_positive_names": tuple(str(name) for name in current_report["inactive_positive_names"]),
                "added_support_indices": added,
                "added_support_names": tuple(
                    setup.condensate_species[int(index)] for index in added
                ),
            }
        )
        if not added:
            support_cap = _support_count_cap(options)
            if inactive_positive and support_cap is not None and len(current_support) >= support_cap:
                terminated_reason = "max_positive_support_count_reached"
            elif inactive_positive:
                terminated_reason = "max_support_outer_iterations_reached"
            else:
                terminated_reason = "no_inactive_positive_support"
            break
        previous_support_amounts = (
            _positive_support_amounts_for_warm_start(
                fallback_solver_payload["m_support"],
                min_seed_amount=options.min_seed_amount,
            )
            if fallback_solver_payload
            else _positive_support_amounts_for_warm_start(
                (
                    float(last_result.condensate_amounts[int(index)])
                    for index in support_growth_existing
                ),
                min_seed_amount=options.min_seed_amount,
            )
        )
        added_support_amounts = _budget_seed_for_support(
            setup=setup,
            b=b,
            support_indices=added,
            options=options,
        )
        current_support = support_growth_existing + added
        support_amounts = previous_support_amounts + added_support_amounts

    if last_result is None:
        raise RuntimeError("Support outer loop did not produce a condensate result.")
    support_selection_report = _support_selection_payload_from_activity_report(
        report=current_report,
        support_indices=current_support,
        support_names=tuple(setup.condensate_species[int(index)] for index in current_support),
        support_amounts_init=support_amounts,
        seed_initialization_policy=options.seed_initialization_policy,
        terminated_reason=terminated_reason,
        outer_iterations=outer_iterations,
    )
    retry_caps = tuple(
        cap
        for cap in _support_cap_retry_sequence(options)
        if cap < len(tuple(current_support))
    )
    if (
        options.enable_support_cap_retry
        and options.max_positive_support_count is None
        and last_result.selected_route == "native_budget_seed_fallback_budget_tradeoff"
        and retry_caps
    ):
        retry_attempts = []
        for retry_cap in retry_caps:
            retry_options = replace(
                options,
                case_id=None
                if options.case_id is None
                else f"{options.case_id}__support_cap_retry_{retry_cap}",
                enable_support_cap_retry=False,
                max_positive_support_count=int(retry_cap),
            )
            try:
                retry_result = condensate_equilibrium(
                    setup,
                    T,
                    P,
                    b,
                    Pref=Pref,
                    options=retry_options,
                )
            except Exception as exc:  # noqa: BLE001 - retry candidates are optional.
                retry_attempts.append(
                    {
                        "support_cap": int(retry_cap),
                        "selected_route": "exception",
                        "status": "exception",
                        "support_count": 0,
                        "accepted": False,
                        "route_promoted": False,
                        "support_closure_accepted": False,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    }
                )
                continue
            retry_route_promoted = (
                retry_result.selected_route != "native_budget_seed_fallback_budget_tradeoff"
            )
            retry_accepted = bool(retry_result.converged)
            support_closure_gate = _support_closure_retry_gate_report(
                setup=setup,
                T=T,
                P=P,
                b=b,
                Pref=Pref,
                result=retry_result,
                options=options,
            )
            retry_support_closure_accepted = bool(
                support_closure_gate.get("accepted", False)
            )
            retry_attempt = {
                "support_cap": int(retry_cap),
                "selected_route": retry_result.selected_route,
                "status": retry_result.status,
                "support_count": len(tuple(retry_result.condensate_support_names)),
                "accepted": bool(retry_accepted),
                "route_promoted": bool(retry_route_promoted),
                "support_closure_gate": support_closure_gate,
                "support_closure_accepted": retry_support_closure_accepted,
            }
            retry_attempts.append(retry_attempt)
            if retry_route_promoted and retry_support_closure_accepted:
                retry_report = {
                    "retry_schema": "exogibbs_support_free_support_cap_retry_v1",
                    "triggered": True,
                    "accepted": bool(retry_accepted and retry_support_closure_accepted),
                    "route_promoted": True,
                    "support_closure_accepted": True,
                    "support_cap": int(retry_cap),
                    "support_cap_sequence": tuple(int(cap) for cap in retry_caps),
                    "attempts": tuple(retry_attempts),
                    "initial_selected_route": last_result.selected_route,
                    "initial_status": last_result.status,
                    "initial_support_count": len(tuple(current_support)),
                    "retry_selected_route": retry_result.selected_route,
                    "retry_status": retry_result.status,
                    "retry_support_count": len(
                        tuple(retry_result.condensate_support_names)
                    ),
                    "retry_support_closure_gate": support_closure_gate,
                    "fastchem4_trace_public_runtime_constructor_inputs_used": False,
                }
                return _with_support_cap_retry_diagnostics(
                    result=retry_result,
                    retry_report=retry_report,
                    return_diagnostics=options.return_diagnostics,
                )
    staged_retry_counts = _support_growth_staging_retry_sequence(options)
    if (
        options.enable_support_growth_staging_retry
        and options.max_positive_support_count is None
        and options.max_support_add_per_round is None
        and last_result.selected_route == "native_budget_seed_fallback_budget_tradeoff"
        and staged_retry_counts
    ):
        retry_attempts = []
        for add_per_round in staged_retry_counts:
            retry_options = replace(
                options,
                case_id=None
                if options.case_id is None
                else f"{options.case_id}__support_growth_staging_retry_{add_per_round}",
                enable_support_cap_retry=False,
                enable_support_growth_staging_retry=False,
                max_support_add_per_round=int(add_per_round),
            )
            try:
                retry_result = condensate_equilibrium(
                    setup,
                    T,
                    P,
                    b,
                    Pref=Pref,
                    options=retry_options,
                )
            except Exception as exc:  # noqa: BLE001 - retry candidates are optional.
                retry_attempts.append(
                    {
                        "max_support_add_per_round": int(add_per_round),
                        "selected_route": "exception",
                        "status": "exception",
                        "support_count": 0,
                        "support_outer_terminated_reason": None,
                        "accepted": False,
                        "route_promoted": False,
                        "support_closure_accepted": False,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    }
                )
                continue
            retry_route_promoted = (
                retry_result.selected_route != "native_budget_seed_fallback_budget_tradeoff"
            )
            retry_accepted = bool(retry_result.converged)
            support_closure_gate = _support_closure_retry_gate_report(
                setup=setup,
                T=T,
                P=P,
                b=b,
                Pref=Pref,
                result=retry_result,
                options=options,
            )
            retry_support_closure_accepted = bool(
                support_closure_gate.get("accepted", False)
            )
            retry_outer = (retry_result.diagnostics or {}).get("support_outer_loop", {})
            retry_attempt = {
                "max_support_add_per_round": int(add_per_round),
                "selected_route": retry_result.selected_route,
                "status": retry_result.status,
                "support_count": len(tuple(retry_result.condensate_support_names)),
                "support_outer_terminated_reason": retry_outer.get("terminated_reason")
                if isinstance(retry_outer, Mapping)
                else None,
                "accepted": bool(retry_accepted),
                "route_promoted": bool(retry_route_promoted),
                "support_closure_gate": support_closure_gate,
                "support_closure_accepted": retry_support_closure_accepted,
            }
            retry_attempts.append(retry_attempt)
            if retry_route_promoted and retry_support_closure_accepted:
                retry_report = {
                    "retry_schema": "exogibbs_support_free_support_growth_staging_retry_v1",
                    "triggered": True,
                    "accepted": bool(retry_accepted and retry_support_closure_accepted),
                    "route_promoted": True,
                    "support_closure_accepted": True,
                    "max_support_add_per_round": int(add_per_round),
                    "max_support_add_per_round_sequence": tuple(
                        int(count) for count in staged_retry_counts
                    ),
                    "attempts": tuple(retry_attempts),
                    "initial_selected_route": last_result.selected_route,
                    "initial_status": last_result.status,
                    "initial_support_count": len(tuple(current_support)),
                    "initial_support_outer_terminated_reason": terminated_reason,
                    "retry_selected_route": retry_result.selected_route,
                    "retry_status": retry_result.status,
                    "retry_support_count": len(
                        tuple(retry_result.condensate_support_names)
                    ),
                    "retry_support_closure_gate": support_closure_gate,
                    "fastchem4_trace_public_runtime_constructor_inputs_used": False,
                }
                return _with_support_growth_staging_retry_diagnostics(
                    result=retry_result,
                    retry_report=retry_report,
                    return_diagnostics=options.return_diagnostics,
                )
    if (
        options.seed_initialization_policy != "budget_preserving_fraction"
        and last_result.selected_route == "native_budget_seed_fallback_budget_tradeoff"
        and not last_result.converged
    ):
        retry_options = replace(
            options,
            case_id=None
            if options.case_id is None
            else f"{options.case_id}__budget_preserving_seed_retry",
            seed_initialization_policy="budget_preserving_fraction",
            enable_support_cap_retry=False,
            enable_support_growth_staging_retry=False,
        )
        retry_result = condensate_equilibrium(
            setup,
            T,
            P,
            b,
            Pref=Pref,
            options=retry_options,
        )
        retry_report = {
            "retry_schema": "exogibbs_support_free_budget_preserving_seed_retry_v1",
            "triggered": True,
            "accepted": bool(retry_result.converged),
            "route_promoted": bool(
                retry_result.selected_route
                != "native_budget_seed_fallback_budget_tradeoff"
            ),
            "initial_seed_initialization_policy": options.seed_initialization_policy,
            "retry_seed_initialization_policy": "budget_preserving_fraction",
            "initial_selected_route": last_result.selected_route,
            "initial_status": last_result.status,
            "initial_support_count": len(tuple(current_support)),
            "retry_selected_route": retry_result.selected_route,
            "retry_status": retry_result.status,
            "retry_support_count": len(tuple(retry_result.condensate_support_names)),
            "fastchem4_trace_public_runtime_constructor_inputs_used": False,
        }
        if retry_result.converged:
            return _with_support_budget_preserving_seed_retry_diagnostics(
                result=retry_result,
                retry_report=retry_report,
                return_diagnostics=options.return_diagnostics,
            )
        last_result = _with_support_budget_preserving_seed_retry_diagnostics(
            result=last_result,
            retry_report=retry_report,
            return_diagnostics=options.return_diagnostics,
        )
    return _with_support_outer_loop_diagnostics(
        result=last_result,
        support_selection_report=support_selection_report,
        return_diagnostics=options.return_diagnostics,
    )


def condensate_equilibrium(
    setup: CondensateChemicalSetup,
    T: float,
    P: float,
    b: Array,
    *,
    Pref: float = 1.0,
    support_indices: Optional[Sequence[int]] = None,
    support_amounts_init: Optional[Sequence[float]] = None,
    options: Optional[CondensateEquilibriumOptions] = None,
) -> CondensateEquilibriumResult:
    """Compute one condensate-enabled equilibrium layer through HEAD route v1.

    When no support is supplied, the HEAD route builds native activity-driven
    support from ExoGibbs thermochemistry and the caller's element budget.
    Explicit support payloads are still accepted for controlled experiments.
    """

    opts = options or CondensateEquilibriumOptions()
    validate_condensate_chemical_setup(setup)
    _validate_options(opts)
    if support_indices is None and opts.enable_support_outer_loop:
        return _run_activity_driven_support_outer_loop(
            setup=setup,
            T=T,
            P=P,
            b=b,
            Pref=Pref,
            options=opts,
        )
    support_selection_report: Optional[Mapping[str, Any]] = None
    if support_indices is None:
        from exogibbs.condensates.positive_support_initializer import (
            build_positive_support_initializer_report,
        )

        support_plan = build_positive_support_initializer_report(
            formula_matrix_cond=setup.formula_matrix_cond,
            element_inventory_target=jnp.asarray(b),
            condensate_species_order=setup.condensate_species,
            hvector_cond=setup.condensate_setup.hvector_func(float(T)),
            max_positive_support_count=(
                int(setup.formula_matrix_cond.shape[1])
                if opts.max_positive_support_count is None
                else int(opts.max_positive_support_count)
            ),
            seed_fraction=opts.seed_fraction,
            max_seed_amount=opts.max_seed_amount,
            min_seed_amount=opts.min_seed_amount,
            allow_empty_positive_support=opts.allow_empty_positive_support,
            field_provenance={
                "formula_matrix_cond": "exogibbs_condensate_chemical_setup",
                "element_inventory_target": "exogibbs_runtime_input",
                "hvector_cond": "exogibbs_condensate_thermochemistry",
            },
        )
        support_selection_report = support_plan.as_dict()
        support_indices = support_plan.solver_inputs.support_indices
        support_amounts_init = support_plan.solver_inputs.support_amounts_init
        support_selection_report = dict(support_selection_report)
        solver_inputs = dict(support_selection_report.get("solver_inputs", {}))
        solver_inputs.update(_seed_gauge_payload(opts))
        support_selection_report["solver_inputs"] = solver_inputs
        if opts.seed_initialization_policy != "budget_preserving_fraction":
            support_amounts_init = _budget_seed_for_support(
                setup=setup,
                b=b,
                support_indices=support_indices,
                options=opts,
            )
            solver_inputs["support_amounts_init"] = tuple(
                float(value) for value in support_amounts_init
            )
            solver_inputs.update(_seed_gauge_payload(opts))
            support_selection_report["solver_inputs"] = solver_inputs
    else:
        explicit_indices = tuple(int(index) for index in support_indices)
        explicit_amounts = (
            ()
            if support_amounts_init is None
            else tuple(float(value) for value in jnp.asarray(support_amounts_init).tolist())
        )
        support_selection_report = {
            "selection_schema": "exogibbs_explicit_condensate_support_payload_v1",
            "selection_mode": "explicit_support_payload",
            "solver_inputs": {
                "support_indices": explicit_indices,
                "support_amounts_init": explicit_amounts,
                "seed_initialization_policy": "explicit_support_payload",
                "amount_gauge": "caller_supplied_explicit_payload",
                "empty_positive_support": len(explicit_indices) == 0,
            },
            "fastchem4_trace_values_used": False,
            "fastchem4_public_values_used_as_constructor_inputs": False,
            "fastchem4_runtime_values_used_as_constructor_inputs": False,
        }
    from exogibbs.optimize.minimize_cond import (
        CondensateEquilibriumInit,
        CondensateRGIEReducedCouplingConfig,
        solve_restricted_support_condensate_layer,
    )
    from exogibbs.api.equilibrium import EquilibriumOptions, equilibrium

    state = ThermoState(
        temperature=float(T),
        ln_normalized_pressure=_ln_normalized_pressure(P, Pref),
        element_vector=jnp.asarray(b),
    )
    if len(tuple(support_indices)) == 0:
        gas_result = equilibrium(
            setup.gas_setup,
            T,
            P,
            jnp.asarray(b),
            Pref=Pref,
            options=EquilibriumOptions(),
            return_diagnostics=False,
        )
        diagnostics = {"support_selection": support_selection_report} if opts.return_diagnostics else None
        return _build_empty_support_gas_result(
            setup=setup,
            gas_ln_n=gas_result.ln_n,
            diagnostics=diagnostics,
            element_inventory_target=b,
            enable_full_condensate_budget_residual_gate=(
                opts.enable_full_condensate_budget_residual_gate
            ),
            full_condensate_budget_relative_tolerance=(
                opts.full_condensate_budget_relative_tolerance
            ),
        )
    solve_kwargs: dict[str, Any] = {}
    if opts.max_inner_iterations is not None:
        solve_kwargs["max_iter"] = int(opts.max_inner_iterations)
    solve_kwargs["reduced_coupling_config"] = CondensateRGIEReducedCouplingConfig(
        reduced_coupling_mode=opts.restricted_reduced_coupling_mode,
        alpha_s=float(opts.restricted_reduced_coupling_alpha_s),
    )
    from exogibbs.condensates.head_route_warm_start import (
        build_condensate_head_route_warm_start_report,
    )

    if support_amounts_init is None:
        raise ValueError("support_amounts_init is required for non-empty condensate support.")
    baseline_gas_result = equilibrium(
        setup.gas_setup,
        T,
        P,
        jnp.asarray(b),
        Pref=Pref,
        options=EquilibriumOptions(),
        return_diagnostics=False,
    )
    baseline_initial_log_state = CondensateEquilibriumInit(
        ln_nk=jnp.asarray(baseline_gas_result.ln_n, dtype=jnp.float64),
        ln_mk=jnp.log(jnp.maximum(jnp.asarray(support_amounts_init), 1.0e-300)),
        ln_ntot=jnp.log(jnp.asarray(baseline_gas_result.ntot, dtype=jnp.float64)),
        ln_nk_source_trace={
            "source": "exogibbs_api_fresh_gas_equilibrium",
            "reason": (
                "Keep support selection and restricted solver baseline "
                "initialization on the same native gas state."
            ),
        },
    )
    warm_start_report = build_condensate_head_route_warm_start_report(
        explicit_opt_in=True,
        state=state,
        formula_matrix=setup.formula_matrix,
        formula_matrix_cond=setup.formula_matrix_cond,
        hvector_func=setup.gas_setup.hvector_func,
        support_indices=support_indices,
        support_amounts_init=jnp.asarray(support_amounts_init),
        baseline_initial_log_state_override=baseline_initial_log_state,
        enable_depleted_gas_refresh=(
            opts.enable_head_route_warm_start and opts.enable_depleted_gas_refresh
        ),
        gas_refresh_policy=opts.warm_start_gas_refresh_policy,
        field_provenance={
            "formula_matrix": "exogibbs_condensate_chemical_setup",
            "formula_matrix_cond": "exogibbs_condensate_chemical_setup",
            "element_budget": "exogibbs_runtime_input",
            "ln_mk": "exogibbs_head_route_positive_support_seed",
            "hvector_func": "exogibbs_gas_thermochemistry",
        },
    )
    solver_attempts: list[dict[str, Any]] = []
    solver: Mapping[str, Any] | None = None
    selected_warm_start_candidate: Mapping[str, Any] | None = None
    selected_warm_start_candidate_object = None
    selected_solver_success = False
    for candidate_index, candidate in enumerate(warm_start_report.candidates):
        if not candidate.finite_solver_inputs:
            solver_attempts.append(
                {
                    "candidate_index": candidate_index,
                    "candidate_name": candidate.candidate_name,
                    "candidate_kind": candidate.candidate_kind,
                    "attempt_status": "skipped_nonfinite_solver_inputs",
                    "solver_success": False,
                }
            )
            continue
        attempt = solve_restricted_support_condensate_layer(
            state,
            setup.formula_matrix,
            setup.formula_matrix_cond,
            setup.gas_setup.hvector_func,
            setup.condensate_setup.hvector_func,
            support_indices=candidate.support_indices,
            condensate_species=setup.condensate_species,
            element_names=setup.elements,
            support_amounts_init=jnp.asarray(candidate.support_amounts_init),
            initial_log_state_override=candidate.initial_log_state_override,
            **solve_kwargs,
        )
        attempt_success = bool(attempt["solver_success"])
        attempt_diagnostics = attempt.get("diagnostics", {})
        solver_attempts.append(
            {
                "candidate_index": candidate_index,
                "candidate_name": candidate.candidate_name,
                "candidate_kind": candidate.candidate_kind,
                "attempt_status": "solver_success" if attempt_success else "solver_failed",
                "solver_success": attempt_success,
                "restricted_reduced_coupling_config_mode": attempt.get(
                    "restricted_reduced_coupling_config_mode"
                ),
                "final_residual": attempt_diagnostics.get("final_residual")
                if isinstance(attempt_diagnostics, Mapping)
                else None,
                "n_iter": attempt_diagnostics.get("n_iter")
                if isinstance(attempt_diagnostics, Mapping)
                else None,
                "hit_max_iter": attempt_diagnostics.get("hit_max_iter")
                if isinstance(attempt_diagnostics, Mapping)
                else None,
            }
        )
        if solver is None or attempt_success or not selected_solver_success:
            solver = attempt
            selected_warm_start_candidate_object = warm_start_report.candidates[candidate_index]
            selected_warm_start_candidate = selected_warm_start_candidate_object.as_dict()
            selected_solver_success = attempt_success
        if attempt_success:
            break
    if solver is None:
        raise RuntimeError("No finite condensate HEAD route warm-start candidate was available.")
    restricted_solver_success = bool(solver["solver_success"])
    solver_ln_nk = jnp.asarray(solver["ln_nk"])
    solver_support_indices = tuple(int(index) for index in solver["support_indices"])
    solver_support_amounts = jnp.asarray(solver["m_support"])
    lifecycle_payload: Mapping[str, Any]
    lifecycle_selected_route = opts.selected_route
    lifecycle_metric_status = opts.metric_status
    lifecycle_converged = False
    center_gate_retry_report: Mapping[str, Any] | None = None
    residual_worsening_retry_report: Mapping[str, Any] | None = None
    soft_restoration_retry_report: Mapping[str, Any] | None = None
    ipopt_h_type_retry_report: Mapping[str, Any] | None = None
    condensate_budget_correction_retry_report: Mapping[str, Any] | None = None
    full_budget_amount_polish_report: Mapping[str, Any] | None = None
    result_ln_nk = solver_ln_nk
    result_support_indices = solver_support_indices
    result_support_amounts = solver_support_amounts
    result_external_condensate_amounts: Array | None = None
    if restricted_solver_success:
        primary_policy = _head_lifecycle_primary_policy(opts)
        lifecycle_report = _run_lifecycle_from_restricted_solver_state(
            setup=setup,
            T=T,
            P=P,
            Pref=Pref,
            b=b,
            options=opts,
            solver=solver,
            solver_ln_nk=solver_ln_nk,
            solver_support_indices=solver_support_indices,
            solver_support_amounts=solver_support_amounts,
            primary_continuation_policy=primary_policy,
        )
        lifecycle_payload = lifecycle_report.as_dict()
        lifecycle_selected_route = lifecycle_report.route_result.selected_route
        lifecycle_metric_status = lifecycle_report.route_result.metric_status
        lifecycle_converged = bool(lifecycle_report.route_result.converged)
        if (
            not lifecycle_converged
            and opts.enable_head_route_center_gate_retry
            and opts.metric_status is None
            and opts.head_route_primary_summary is None
            and opts.head_route_refresh_policy_summary is None
            and _is_current_barrier_center_gate_block(lifecycle_payload)
        ):
            retry_policy = {
                **primary_policy,
                "center_tolerance_multiplier": float(
                    opts.head_route_center_gate_retry_multiplier
                ),
            }
            retry_lifecycle_report = _run_lifecycle_from_restricted_solver_state(
                setup=setup,
                T=T,
                P=P,
                Pref=Pref,
                b=b,
                options=opts,
                solver=solver,
                solver_ln_nk=solver_ln_nk,
                solver_support_indices=solver_support_indices,
                solver_support_amounts=solver_support_amounts,
                primary_continuation_policy=retry_policy,
            )
            retry_payload = retry_lifecycle_report.as_dict()
            retry_accepted = bool(retry_lifecycle_report.route_result.converged)
            center_gate_retry_report = {
                "retry_schema": "exogibbs_head_route_center_gate_retry_v1",
                "triggered": True,
                "accepted": retry_accepted,
                "center_tolerance_multiplier": float(
                    opts.head_route_center_gate_retry_multiplier
                ),
                "initial_stopped_reason": "current_barrier_not_centered",
                "retry_selected_route": retry_lifecycle_report.route_result.selected_route,
                "retry_metric_status": retry_lifecycle_report.route_result.metric_status,
            }
            if retry_accepted:
                lifecycle_report = retry_lifecycle_report
                lifecycle_payload = retry_payload
                lifecycle_selected_route = retry_lifecycle_report.route_result.selected_route
                lifecycle_metric_status = retry_lifecycle_report.route_result.metric_status
                lifecycle_converged = True
            else:
                lifecycle_payload = {
                    **dict(lifecycle_payload),
                    "center_gate_retry_report": center_gate_retry_report,
                }
        if (
            not lifecycle_converged
            and opts.enable_head_route_residual_worsening_retry
            and opts.metric_status is None
            and opts.head_route_primary_summary is None
            and opts.head_route_refresh_policy_summary is None
            and _is_residual_nonworsening_candidate_block(lifecycle_payload)
        ):
            residual_retry_policy = {
                **primary_policy,
                "residual_worsening_tolerance": float(
                    opts.head_route_residual_worsening_retry_tolerance
                ),
            }
            residual_retry_lifecycle_report = _run_lifecycle_from_restricted_solver_state(
                setup=setup,
                T=T,
                P=P,
                Pref=Pref,
                b=b,
                options=opts,
                solver=solver,
                solver_ln_nk=solver_ln_nk,
                solver_support_indices=solver_support_indices,
                solver_support_amounts=solver_support_amounts,
                primary_continuation_policy=residual_retry_policy,
            )
            residual_retry_payload = residual_retry_lifecycle_report.as_dict()
            residual_retry_accepted = bool(
                residual_retry_lifecycle_report.route_result.converged
            )
            residual_center_retry_report: Mapping[str, Any] | None = None
            if (
                not residual_retry_accepted
                and opts.enable_head_route_center_gate_retry
                and _is_current_barrier_center_gate_block(residual_retry_payload)
            ):
                residual_center_policy = {
                    **residual_retry_policy,
                    "center_tolerance_multiplier": float(
                        opts.head_route_center_gate_retry_multiplier
                    ),
                }
                residual_center_lifecycle_report = (
                    _run_lifecycle_from_restricted_solver_state(
                        setup=setup,
                        T=T,
                        P=P,
                        Pref=Pref,
                        b=b,
                        options=opts,
                        solver=solver,
                        solver_ln_nk=solver_ln_nk,
                        solver_support_indices=solver_support_indices,
                        solver_support_amounts=solver_support_amounts,
                        primary_continuation_policy=residual_center_policy,
                    )
                )
                residual_center_payload = residual_center_lifecycle_report.as_dict()
                residual_center_accepted = bool(
                    residual_center_lifecycle_report.route_result.converged
                )
                residual_center_retry_report = {
                    "retry_schema": "exogibbs_head_route_center_gate_retry_v1",
                    "triggered": True,
                    "accepted": residual_center_accepted,
                    "center_tolerance_multiplier": float(
                        opts.head_route_center_gate_retry_multiplier
                    ),
                    "initial_stopped_reason": "current_barrier_not_centered",
                    "retry_selected_route": (
                        residual_center_lifecycle_report.route_result.selected_route
                    ),
                    "retry_metric_status": (
                        residual_center_lifecycle_report.route_result.metric_status
                    ),
                }
                if residual_center_accepted:
                    residual_retry_lifecycle_report = residual_center_lifecycle_report
                    residual_retry_payload = residual_center_payload
                    residual_retry_accepted = True
            residual_worsening_retry_report = {
                "retry_schema": "exogibbs_head_route_residual_worsening_retry_v1",
                "triggered": True,
                "accepted": residual_retry_accepted,
                "residual_worsening_tolerance": float(
                    opts.head_route_residual_worsening_retry_tolerance
                ),
                "initial_stopped_reason": "no_p_armijo_trial",
                "retry_selected_route": (
                    residual_retry_lifecycle_report.route_result.selected_route
                ),
                "retry_metric_status": (
                    residual_retry_lifecycle_report.route_result.metric_status
                ),
                "center_gate_retry_report": residual_center_retry_report,
            }
            if residual_retry_accepted:
                lifecycle_report = residual_retry_lifecycle_report
                lifecycle_payload = residual_retry_payload
                lifecycle_selected_route = (
                    residual_retry_lifecycle_report.route_result.selected_route
                )
                lifecycle_metric_status = (
                    residual_retry_lifecycle_report.route_result.metric_status
                )
                lifecycle_converged = True
                if residual_center_retry_report is not None:
                    center_gate_retry_report = residual_center_retry_report
            else:
                lifecycle_payload = {
                    **dict(lifecycle_payload),
                    "residual_worsening_retry_report": residual_worsening_retry_report,
                }
        if (
            not lifecycle_converged
            and opts.enable_head_route_soft_restoration_retry
            and opts.metric_status is None
            and opts.head_route_primary_summary is None
            and opts.head_route_refresh_policy_summary is None
        ):
            soft_restoration_policy = {
                **primary_policy,
                "center_tolerance_multiplier": float(
                    opts.head_route_center_gate_retry_multiplier
                ),
                "enable_native_soft_restoration_fallback": True,
                "soft_restoration_component_weights": dict(
                    HEAD_ROUTE_SOFT_RESTORATION_COMPONENT_WEIGHTS
                ),
                "soft_restoration_proximity_weight": float(
                    opts.head_route_soft_restoration_proximity_weight
                ),
                "soft_restoration_max_proximity": (
                    None
                    if opts.head_route_soft_restoration_max_proximity is None
                    else float(opts.head_route_soft_restoration_max_proximity)
                ),
            }
            soft_restoration_lifecycle_report = _run_lifecycle_from_restricted_solver_state(
                setup=setup,
                T=T,
                P=P,
                Pref=Pref,
                b=b,
                options=opts,
                solver=solver,
                solver_ln_nk=solver_ln_nk,
                solver_support_indices=solver_support_indices,
                solver_support_amounts=solver_support_amounts,
                primary_continuation_policy=soft_restoration_policy,
            )
            soft_restoration_payload = soft_restoration_lifecycle_report.as_dict()
            soft_restoration_accepted = bool(
                soft_restoration_lifecycle_report.route_result.converged
            )
            soft_restoration_retry_report = {
                "retry_schema": "exogibbs_head_route_soft_restoration_retry_v1",
                "triggered": True,
                "accepted": soft_restoration_accepted,
                "component_weights": dict(HEAD_ROUTE_SOFT_RESTORATION_COMPONENT_WEIGHTS),
                "center_tolerance_multiplier": float(
                    opts.head_route_center_gate_retry_multiplier
                ),
                "soft_restoration_proximity_weight": float(
                    opts.head_route_soft_restoration_proximity_weight
                ),
                "soft_restoration_max_proximity": (
                    None
                    if opts.head_route_soft_restoration_max_proximity is None
                    else float(opts.head_route_soft_restoration_max_proximity)
                ),
                "initial_selected_route": lifecycle_selected_route,
                "retry_selected_route": (
                    soft_restoration_lifecycle_report.route_result.selected_route
                ),
                "retry_metric_status": (
                    soft_restoration_lifecycle_report.route_result.metric_status
                ),
            }
            if soft_restoration_accepted:
                lifecycle_report = soft_restoration_lifecycle_report
                lifecycle_payload = soft_restoration_payload
                lifecycle_selected_route = (
                    soft_restoration_lifecycle_report.route_result.selected_route
                )
                lifecycle_metric_status = (
                    soft_restoration_lifecycle_report.route_result.metric_status
                )
                lifecycle_converged = True
            else:
                lifecycle_payload = {
                    **dict(lifecycle_payload),
                    "soft_restoration_retry_report": soft_restoration_retry_report,
                }
        if (
            not lifecycle_converged
            and opts.enable_head_route_ipopt_h_type_retry
            and opts.metric_status is None
            and opts.head_route_primary_summary is None
            and opts.head_route_refresh_policy_summary is None
        ):
            ipopt_h_type_policy = {
                **primary_policy,
                "center_tolerance_multiplier": float(
                    opts.head_route_center_gate_retry_multiplier
                ),
                "trial_acceptance_policy": "ipopt_persistent_h_type",
                "filter_component_weights": dict(
                    HEAD_ROUTE_IPOPT_H_TYPE_COMPONENT_WEIGHTS
                ),
                "ipopt_h_type_component_weights": dict(
                    HEAD_ROUTE_IPOPT_H_TYPE_COMPONENT_WEIGHTS
                ),
                "ipopt_h_type_theta_reduction_fraction": float(
                    opts.head_route_ipopt_h_type_theta_reduction_fraction
                ),
                "ipopt_h_type_protected_components": tuple(
                    HEAD_ROUTE_IPOPT_H_TYPE_PROTECTED_COMPONENTS
                ),
                "ipopt_h_type_protected_component_max_normalized_increase": float(
                    opts.head_route_ipopt_h_type_protected_component_max_normalized_increase
                ),
                "persistent_filter_gamma_p": 1.0e-8,
                "persistent_filter_gamma_theta": 1.0e-5,
                "persistent_filter_theta_max_factor": 1.0e4,
                "require_residual_nonworsening": False,
            }
            ipopt_h_type_lifecycle_report = _run_lifecycle_from_restricted_solver_state(
                setup=setup,
                T=T,
                P=P,
                Pref=Pref,
                b=b,
                options=opts,
                solver=solver,
                solver_ln_nk=solver_ln_nk,
                solver_support_indices=solver_support_indices,
                solver_support_amounts=solver_support_amounts,
                primary_continuation_policy=ipopt_h_type_policy,
            )
            ipopt_h_type_payload = ipopt_h_type_lifecycle_report.as_dict()
            ipopt_h_type_accepted = bool(
                ipopt_h_type_lifecycle_report.route_result.converged
            )
            ipopt_h_type_retry_report = {
                "retry_schema": "exogibbs_head_route_ipopt_h_type_retry_v1",
                "triggered": True,
                "accepted": ipopt_h_type_accepted,
                "trial_acceptance_policy": "ipopt_persistent_h_type",
                "component_weights": dict(HEAD_ROUTE_IPOPT_H_TYPE_COMPONENT_WEIGHTS),
                "protected_components": tuple(
                    HEAD_ROUTE_IPOPT_H_TYPE_PROTECTED_COMPONENTS
                ),
                "protected_component_max_normalized_increase": float(
                    opts.head_route_ipopt_h_type_protected_component_max_normalized_increase
                ),
                "theta_reduction_fraction": float(
                    opts.head_route_ipopt_h_type_theta_reduction_fraction
                ),
                "center_tolerance_multiplier": float(
                    opts.head_route_center_gate_retry_multiplier
                ),
                "require_residual_nonworsening": False,
                "initial_selected_route": lifecycle_selected_route,
                "retry_selected_route": (
                    ipopt_h_type_lifecycle_report.route_result.selected_route
                ),
                "retry_metric_status": (
                    ipopt_h_type_lifecycle_report.route_result.metric_status
                ),
            }
            if ipopt_h_type_accepted:
                lifecycle_report = ipopt_h_type_lifecycle_report
                lifecycle_payload = ipopt_h_type_payload
                lifecycle_selected_route = (
                    ipopt_h_type_lifecycle_report.route_result.selected_route
                )
                lifecycle_metric_status = (
                    ipopt_h_type_lifecycle_report.route_result.metric_status
                )
                lifecycle_converged = True
            else:
                lifecycle_payload = {
                    **dict(lifecycle_payload),
                    "ipopt_h_type_retry_report": ipopt_h_type_retry_report,
                }
        if (
            lifecycle_converged
            and opts.enable_head_route_condensate_budget_correction_retry
            and opts.enable_full_condensate_budget_residual_gate
            and opts.metric_status is None
            and opts.head_route_primary_summary is None
            and opts.head_route_refresh_policy_summary is None
            and selected_warm_start_candidate_object is not None
        ):
            final_state_payload = _lifecycle_final_state_payload(lifecycle_payload)
            external_final_amounts = _external_condensate_amounts_from_lifecycle_payload(
                lifecycle_payload,
                condensate_count=len(setup.condensate_species),
            )
            final_state_support_indices = _final_state_support_indices_from_lifecycle_payload(
                lifecycle_payload,
                fallback_support_indices=solver_support_indices,
            )
            initial_gate_report = None
            if isinstance(final_state_payload, Mapping):
                try:
                    initial_gate_report = (
                        _full_condensate_budget_gate_report_for_support_state(
                            setup=setup,
                            gas_ln_n=jnp.asarray(final_state_payload["ln_nk"]),
                            support_indices=final_state_support_indices,
                            support_amounts=jnp.exp(
                                jnp.asarray(final_state_payload["ln_mk"])
                            ),
                            external_condensate_amounts=external_final_amounts,
                            element_inventory_target=b,
                            relative_tolerance=(
                                opts.full_condensate_budget_relative_tolerance
                            ),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    initial_gate_report = None
            if initial_gate_report is not None and not bool(initial_gate_report["accepted"]):
                budget_correction_policy = {
                    **primary_policy,
                    "direction_policy": "joint_budget_amount_gas_linearized_no_prior",
                    "budget_row_scaling_policy": "relative_target",
                    "trial_acceptance_policy": "ipopt_persistent_h_type",
                    "filter_component_weights": dict(
                        HEAD_ROUTE_RELATIVE_BUDGET_CORRECTION_COMPONENT_WEIGHTS
                    ),
                    "ipopt_h_type_component_weights": dict(
                        HEAD_ROUTE_RELATIVE_BUDGET_CORRECTION_COMPONENT_WEIGHTS
                    ),
                    "ipopt_h_type_theta_reduction_fraction": float(
                        opts.head_route_ipopt_h_type_theta_reduction_fraction
                    ),
                    "ipopt_h_type_protected_components": tuple(
                        HEAD_ROUTE_RELATIVE_BUDGET_CORRECTION_PROTECTED_COMPONENTS
                    ),
                    "ipopt_h_type_protected_component_max_normalized_increase": float(
                        opts.head_route_ipopt_h_type_protected_component_max_normalized_increase
                    ),
                    "center_tolerance_multiplier": float(
                        opts.head_route_center_gate_retry_multiplier
                    ),
                    "persistent_filter_gamma_p": 1.0e-8,
                    "persistent_filter_gamma_theta": 1.0e-5,
                    "persistent_filter_theta_max_factor": 1.0e4,
                    "require_residual_nonworsening": False,
                }
                budget_correction_lifecycle_report = _run_lifecycle_from_native_state(
                    setup=setup,
                    T=T,
                    P=P,
                    Pref=Pref,
                    b=b,
                    options=opts,
                    ln_nk=jnp.asarray(final_state_payload["ln_nk"]),
                    support_indices=final_state_support_indices,
                    support_amounts=jnp.exp(jnp.asarray(final_state_payload["ln_mk"])),
                    external_condensate_amounts=external_final_amounts,
                    element_potential=None,
                    element_potential_source=(
                        "exogibbs_lifecycle_final_state_least_squares_gas_gauge"
                    ),
                    field_source="exogibbs_lifecycle_final_state",
                    primary_continuation_policy=budget_correction_policy,
                )
                budget_correction_payload = budget_correction_lifecycle_report.as_dict()
                if external_final_amounts is not None:
                    budget_correction_payload = {
                        **dict(budget_correction_payload),
                        "input_external_condensate_amounts": tuple(
                            float(value)
                            for value in jnp.asarray(
                                external_final_amounts,
                                dtype=jnp.float64,
                            ).tolist()
                        ),
                    }
                retry_gate_report = None
                retry_primary_payload = budget_correction_payload.get(
                    "primary_execution_report"
                )
                retry_continuation_payload = (
                    retry_primary_payload.get("continuation_report", {})
                    if isinstance(retry_primary_payload, Mapping)
                    else {}
                )
                retry_final_state_payload = (
                    retry_continuation_payload.get("final_state")
                    if isinstance(retry_continuation_payload, Mapping)
                    else None
                )
                retry_final_state_support_indices = (
                    _final_state_support_indices_from_lifecycle_payload(
                        budget_correction_payload,
                        fallback_support_indices=solver_support_indices,
                    )
                )
                retry_external_amounts = _external_condensate_amounts_from_lifecycle_payload(
                    budget_correction_payload,
                    condensate_count=len(setup.condensate_species),
                )
                if isinstance(retry_final_state_payload, Mapping):
                    try:
                        retry_gate_report = (
                            _full_condensate_budget_gate_report_for_support_state(
                                setup=setup,
                                gas_ln_n=jnp.asarray(retry_final_state_payload["ln_nk"]),
                                support_indices=retry_final_state_support_indices,
                                support_amounts=jnp.exp(
                                    jnp.asarray(retry_final_state_payload["ln_mk"])
                                ),
                                external_condensate_amounts=retry_external_amounts,
                                element_inventory_target=b,
                                relative_tolerance=(
                                    opts.full_condensate_budget_relative_tolerance
                                ),
                            )
                        )
                    except (KeyError, TypeError, ValueError):
                        retry_gate_report = None
                budget_correction_accepted = bool(
                    retry_gate_report is not None and retry_gate_report["accepted"]
                )
                condensate_budget_correction_retry_report = {
                    "retry_schema": (
                        "exogibbs_head_route_condensate_budget_correction_retry_v1"
                    ),
                    "triggered": True,
                    "accepted": budget_correction_accepted,
                    "direction_policy": "joint_budget_amount_gas_linearized_no_prior",
                    "budget_row_scaling_policy": "relative_target",
                    "trial_acceptance_policy": "ipopt_persistent_h_type",
                    "initial_full_condensate_budget_gate": initial_gate_report,
                    "retry_full_condensate_budget_gate": retry_gate_report,
                    "initial_selected_route": lifecycle_selected_route,
                    "retry_start_state": "lifecycle_final_state",
                    "retry_selected_route": (
                        budget_correction_lifecycle_report.route_result.selected_route
                    ),
                    "retry_metric_status": (
                        budget_correction_lifecycle_report.route_result.metric_status
                    ),
                }
                if budget_correction_accepted:
                    lifecycle_report = budget_correction_lifecycle_report
                    lifecycle_payload = budget_correction_payload
                    lifecycle_selected_route = str(lifecycle_selected_route)
                    lifecycle_metric_status = str(lifecycle_metric_status)
                    lifecycle_converged = True
                else:
                    lifecycle_payload = {
                        **dict(lifecycle_payload),
                        "condensate_budget_correction_retry_report": (
                            condensate_budget_correction_retry_report
                        ),
                    }
    else:
        lifecycle_payload = _run_lifecycle_from_warm_start_candidate(
            setup=setup,
            T=T,
            P=P,
            Pref=Pref,
            b=b,
            options=opts,
            candidate=selected_warm_start_candidate_object,
        )
        route_result_payload = lifecycle_payload["route_result"]
        lifecycle_selected_route = str(route_result_payload["selected_route"])
        lifecycle_metric_status = str(route_result_payload["metric_status"])
        lifecycle_converged = bool(route_result_payload["converged"])
        final_state_payload = _lifecycle_final_state_payload(lifecycle_payload)
        if lifecycle_converged and isinstance(final_state_payload, Mapping):
            result_support_indices = _final_state_support_indices_from_lifecycle_payload(
                lifecycle_payload,
                fallback_support_indices=(
                    selected_warm_start_candidate_object.support_indices
                ),
            )
            final_ln_mk = jnp.asarray(final_state_payload["ln_mk"])
            if final_ln_mk.ndim == 1 and final_ln_mk.shape[0] == len(result_support_indices):
                result_ln_nk = jnp.asarray(final_state_payload["ln_nk"])
                result_support_amounts = jnp.exp(final_ln_mk)
        elif (
            opts.enable_native_seed_fallback
            and opts.head_route_primary_summary is None
            and opts.head_route_refresh_policy_summary is None
            and selected_warm_start_candidate_object is not None
            and selected_warm_start_candidate_object.finite_solver_inputs
        ):
            return _build_native_seed_fallback_result(
                setup=setup,
                T=T,
                P=P,
                b=b,
                Pref=Pref,
                candidate=selected_warm_start_candidate_object,
                support_selection_report=support_selection_report,
                warm_start_report=warm_start_report,
                solver_attempts=solver_attempts,
                selected_warm_start_candidate=selected_warm_start_candidate,
                lifecycle_payload=lifecycle_payload,
                allow_caveat_tiers=opts.allow_caveat_tiers,
                return_diagnostics=opts.return_diagnostics,
                enable_full_condensate_budget_residual_gate=(
                    opts.enable_full_condensate_budget_residual_gate
                ),
                full_condensate_budget_relative_tolerance=(
                    opts.full_condensate_budget_relative_tolerance
                ),
                restricted_solver_success=False,
            )
    if (
        lifecycle_converged
        and selected_warm_start_candidate_object is not None
    ):
        final_state_payload = _lifecycle_final_state_payload(lifecycle_payload)
        result_external_condensate_amounts = (
            _external_condensate_amounts_from_lifecycle_payload(
                lifecycle_payload,
                condensate_count=len(setup.condensate_species),
            )
        )
        final_support_indices = _final_state_support_indices_from_lifecycle_payload(
            lifecycle_payload,
            fallback_support_indices=solver_support_indices,
        )
        if isinstance(final_state_payload, Mapping):
            final_ln_mk = jnp.asarray(final_state_payload["ln_mk"])
            if final_ln_mk.ndim == 1 and final_ln_mk.shape[0] == len(final_support_indices):
                result_ln_nk = jnp.asarray(final_state_payload["ln_nk"])
                result_support_indices = final_support_indices
                result_support_amounts = jnp.exp(final_ln_mk)
    if (
        lifecycle_converged
        and opts.enable_full_condensate_budget_residual_gate
        and b is not None
        and len(result_support_indices) > 0
    ):
        polished_amounts, polish_report = (
            _polish_support_amounts_for_full_condensate_budget_gate(
                setup=setup,
                gas_ln_n=result_ln_nk,
                support_indices=result_support_indices,
                support_amounts=result_support_amounts,
                external_condensate_amounts=result_external_condensate_amounts,
                element_inventory_target=b,
                relative_tolerance=opts.full_condensate_budget_relative_tolerance,
            )
        )
        if polish_report is not None:
            full_budget_amount_polish_report = polish_report
            if bool(polish_report["accepted"]):
                result_support_amounts = polished_amounts
    if (
        not lifecycle_converged
        and opts.enable_native_seed_fallback
        and opts.metric_status is None
        and opts.head_route_primary_summary is None
        and opts.head_route_refresh_policy_summary is None
        and selected_warm_start_candidate_object is not None
        and selected_warm_start_candidate_object.finite_solver_inputs
    ):
        return _build_native_seed_fallback_result(
            setup=setup,
            T=T,
            P=P,
            b=b,
            Pref=Pref,
            candidate=selected_warm_start_candidate_object,
            support_selection_report=support_selection_report,
            warm_start_report=warm_start_report,
            solver_attempts=solver_attempts,
            selected_warm_start_candidate=selected_warm_start_candidate,
            lifecycle_payload=lifecycle_payload,
            allow_caveat_tiers=opts.allow_caveat_tiers,
            return_diagnostics=opts.return_diagnostics,
            enable_full_condensate_budget_residual_gate=(
                opts.enable_full_condensate_budget_residual_gate
            ),
            full_condensate_budget_relative_tolerance=(
                opts.full_condensate_budget_relative_tolerance
            ),
            restricted_solver_success=restricted_solver_success,
            restricted_solver_payload=solver if restricted_solver_success else None,
        )
    diagnostics_payload: Optional[Mapping[str, Any]]
    if opts.return_diagnostics:
        diagnostics_payload = {
            **solver,
            "restricted_solver_success": restricted_solver_success,
            "solver_success": bool(lifecycle_converged),
            "support_selection": support_selection_report,
            "head_route_warm_start": warm_start_report.as_dict(),
            "head_route_solver_attempts": tuple(solver_attempts),
            "selected_warm_start_candidate": selected_warm_start_candidate,
            "head_route_lifecycle": lifecycle_payload,
        }
        if center_gate_retry_report is not None:
            diagnostics_payload["head_route_center_gate_retry"] = center_gate_retry_report
        if residual_worsening_retry_report is not None:
            diagnostics_payload["head_route_residual_worsening_retry"] = (
                residual_worsening_retry_report
            )
        if soft_restoration_retry_report is not None:
            diagnostics_payload["head_route_soft_restoration_retry"] = (
                soft_restoration_retry_report
            )
        if ipopt_h_type_retry_report is not None:
            diagnostics_payload["head_route_ipopt_h_type_retry"] = (
                ipopt_h_type_retry_report
            )
        if condensate_budget_correction_retry_report is not None:
            diagnostics_payload["head_route_condensate_budget_correction_retry"] = (
                condensate_budget_correction_retry_report
            )
        if full_budget_amount_polish_report is not None:
            diagnostics_payload["full_condensate_budget_amount_polish"] = (
                full_budget_amount_polish_report
            )
    else:
        diagnostics_payload = None
    return build_condensate_equilibrium_result_from_solver_payload(
        setup=setup,
        gas_ln_n=result_ln_nk,
        support_indices=result_support_indices,
        support_amounts=result_support_amounts,
        external_condensate_amounts=result_external_condensate_amounts,
        selected_route=lifecycle_selected_route,
        metric_status=lifecycle_metric_status,
        solver_success=bool(lifecycle_converged),
        allow_caveat_tiers=opts.allow_caveat_tiers,
        diagnostics=diagnostics_payload,
        element_inventory_target=b,
        enable_full_condensate_budget_residual_gate=(
            opts.enable_full_condensate_budget_residual_gate
        ),
        full_condensate_budget_relative_tolerance=(
            opts.full_condensate_budget_relative_tolerance
        ),
    )


def condensate_equilibrium_profile(*args: Any, **kwargs: Any) -> Any:
    """Profile condensate equilibrium placeholder for the HEAD route API."""

    raise NotImplementedError("condensate_equilibrium_profile will be connected after one-layer HEAD route wiring.")


__all__ = (
    "CondensateChemicalSetup",
    "CondensateEquilibriumOptions",
    "CondensateEquilibriumResult",
    "build_condensate_chemical_setup",
    "build_condensate_equilibrium_result_from_solver_payload",
    "condensate_equilibrium",
    "condensate_equilibrium_profile",
    "validate_condensate_chemical_setup",
)
