"""Production-facing condensate equilibrium API shell.

This module defines the first condensate-specific public API surface. It keeps
gas-only equilibrium behavior separate and routes condensate-enabled calls
through the current condensate HEAD route contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Literal, Mapping, Optional, Protocol, Sequence, runtime_checkable
import weakref

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import lsq_linear

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
CondensatePrimaryStepControlPolicy = Literal[
    "component_clip",
    "scalar_fraction_to_boundary",
]
CondensatePrimaryContinuationMode = Literal[
    "legacy_policy",
    "pdipm_core",
    "pdipm_core_single_loop",
]
CondensatePrimaryDualInitializationPolicy = Literal[
    "centered_from_epsilon",
    "ipopt_push_floor",
]
CondensateProfileMethod = Literal[
    "auto",
    "vmap_cold",
    "scan_hot_from_top",
    "scan_hot_from_bottom",
]
CondensateProfileWarmStartSupportPolicy = Literal[
    "previous_solution",
    "explicit_payload",
]
CondensateSeedInitializationPolicy = Literal[
    "budget_preserving_fraction",
    "capacity_fraction",
    "max_density",
]
CondensateFixedSupportGasInitPolicy = Literal[
    "depleted_budget",
    "full_budget",
]
CONDENSATE_HEAD_ROUTE_VERSION = "v1.18"
CONDENSATE_HEAD_ROUTE_NAME = (
    "head_route_v1_18_pdipm_lifecycle_support_growth"
)
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
_SETUP_NUMPY_FORMULA_CACHE: dict[
    int,
    tuple[weakref.ReferenceType[Any], np.ndarray, np.ndarray],
] = {}


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
    profile_method: Optional[CondensateProfileMethod] = None
    profile_warm_start_support_policy: CondensateProfileWarmStartSupportPolicy = (
        "previous_solution"
    )
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
    fixed_support_gas_init_policy: CondensateFixedSupportGasInitPolicy = (
        "depleted_budget"
    )
    warm_start_gas_refresh_policy: CondensateWarmStartGasRefreshPolicy = "native_gas_solver"
    restricted_reduced_coupling_mode: str = "pdipm_rgie_v11_activity_correction"
    restricted_reduced_coupling_alpha_s: float = 1.0
    enable_experimental_profile_fixed_support_batch: bool = False
    enable_experimental_profile_fixed_support_fallback_rescue: bool = False
    experimental_profile_fixed_support_rescue_prune_relative_floors: Sequence[
        float
    ] = (1.0e-5, 1.0e-3)
    experimental_profile_fixed_support_rescue_rho_initialization: str = (
        "complementarity"
    )
    experimental_profile_fixed_support_rescue_lambda_initialization: str = (
        "best_residual"
    )
    experimental_profile_fixed_support_rescue_residual_tolerance_multiplier: float = (
        1.0e9
    )
    head_route_primary_center_tolerance_multiplier: Optional[float] = None
    head_route_primary_residual_worsening_tolerance: Optional[float] = None
    head_route_primary_require_residual_nonworsening: Optional[bool] = None
    head_route_primary_acceptance_guard: Optional[CondensatePrimaryAcceptanceGuard] = None
    head_route_primary_guard_max_budget: float = 1.0e-8
    head_route_primary_guard_max_amount_weighted_gas: float = 1.0e-8
    head_route_primary_guard_max_gas_stationarity: float = 1.0
    head_route_primary_guard_max_condensate_stationarity: float = 10.0
    head_route_primary_continuation_mode: CondensatePrimaryContinuationMode = "pdipm_core"
    head_route_primary_step_control_policy: CondensatePrimaryStepControlPolicy = (
        "scalar_fraction_to_boundary"
    )
    head_route_primary_fraction_to_boundary_safety: float = 0.995
    head_route_primary_tiny_step_alpha_threshold: float = 1.0e-8
    head_route_primary_tiny_step_consecutive_limit: int = 1
    head_route_primary_tiny_step_switch_to_restoration: bool = True
    head_route_primary_ipopt_allow_fast_monotone_decrease: bool = True
    head_route_primary_dual_initialization_policy: (
        CondensatePrimaryDualInitializationPolicy
    ) = "ipopt_push_floor"
    head_route_primary_dual_push_floor: Optional[float] = 1.0e-1
    head_route_primary_summary: Optional[Mapping[str, Any]] = None
    head_route_refresh_policy_summary: Optional[Mapping[str, Any]] = None
    enable_head_route_scalar_step_control_retry: bool = True
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
    support_closure_max_positive_inactive_count: Optional[int] = None
    enable_lifecycle_final_state_support_growth: bool = True
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


@dataclass(frozen=True)
class CondensateEquilibriumInit:
    """Optional condensate profile initial guess for one layer."""

    gas_ln_n: Optional[Array] = None
    gas_ntot: Optional[Array] = None
    condensate_amounts: Optional[Array] = None
    support_indices: Optional[Sequence[int]] = None
    support_amounts: Optional[Sequence[float]] = None
    element_potential: Optional[Array] = None
    rho: Optional[Array] = None
    barrier_epsilon: Optional[Array] = None
    gas_stationarity_source: Optional[Array] = None


@dataclass(frozen=True)
class CondensateEquilibriumInitRequest:
    """Inputs available to a condensate profile initializer for one layer."""

    setup: CondensateChemicalSetup
    T: float
    P: float
    b: Array
    Pref: float = 1.0
    layer_index: Optional[int] = None
    user_init: Optional[CondensateEquilibriumInit] = None
    previous_solution: Optional[CondensateEquilibriumInit] = None


@runtime_checkable
class CondensateEquilibriumInitializer(Protocol):
    """Produce an optional condensate initial guess for one profile layer."""

    def __call__(
        self,
        request: CondensateEquilibriumInitRequest,
    ) -> CondensateEquilibriumInit:
        ...


@dataclass(frozen=True)
class DefaultCondensateEquilibriumInitializer:
    """Use explicit per-layer init first, then the previous profile solution."""

    def __call__(
        self,
        request: CondensateEquilibriumInitRequest,
    ) -> CondensateEquilibriumInit:
        if request.user_init is not None:
            return request.user_init
        if request.previous_solution is not None:
            return request.previous_solution
        return CondensateEquilibriumInit()


@dataclass(frozen=True)
class CondensateEquilibriumProfileResult:
    """Result container for a Python-level condensate profile solve."""

    layers: tuple[CondensateEquilibriumResult, ...]
    method: CondensateProfileMethod
    diagnostics: Optional[Mapping[str, Any]] = None
    batched_arrays: Optional[Mapping[str, Array]] = None


@dataclass(frozen=True)
class ExperimentalCondensateProfileFixedSupportBatchPlan:
    """Reusable experimental fixed-support profile plan.

    This is an opt-in GPU-oriented surface for repeated profile evaluations
    with fixed condensate support. It intentionally does not change the default
    ``condensate_equilibrium_profile`` route.
    """

    setup: CondensateChemicalSetup
    buckets: Sequence[Any]
    formula_matrix: Array
    max_iter: int
    n_layers: int
    condensate_count: int
    bucket_layer_index_arrays: tuple[Array, ...] = ()


@dataclass(frozen=True)
class ExperimentalCondensateProfileFixedSupportPruneRescuePlan:
    """Prepared prune-rescue plan for fallback layers of a fixed-support plan."""

    rescue_plan: Optional[ExperimentalCondensateProfileFixedSupportBatchPlan]
    metadata: Mapping[str, Any]


@dataclass
class ExperimentalCondensateProfileFixedSupportPruneRescueCache:
    """Cache prune-rescue plans keyed by fallback layer set and prune floors."""

    plans: dict[tuple[Any, ...], ExperimentalCondensateProfileFixedSupportPruneRescuePlan] = field(
        default_factory=dict
    )
    prepare_count: int = 0
    hit_count: int = 0


_ExperimentalProfileFixedSupportBatchPlan = (
    ExperimentalCondensateProfileFixedSupportBatchPlan
)
_ExperimentalProfileFixedSupportPruneRescuePlan = (
    ExperimentalCondensateProfileFixedSupportPruneRescuePlan
)
_ExperimentalProfileFixedSupportPruneRescueCache = (
    ExperimentalCondensateProfileFixedSupportPruneRescueCache
)
_DEFAULT_CONDENSATE_INITIALIZER = DefaultCondensateEquilibriumInitializer()


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


def _formula_matrices_numpy(setup: CondensateChemicalSetup) -> tuple[np.ndarray, np.ndarray]:
    """Return cached NumPy formula matrices for Python-side reports/restoration."""

    key = id(setup)
    cached = _SETUP_NUMPY_FORMULA_CACHE.get(key)
    if cached is not None:
        setup_ref, formula_matrix, formula_matrix_cond = cached
        if setup_ref() is setup:
            return formula_matrix, formula_matrix_cond
    formula_matrix = np.asarray(setup.formula_matrix, dtype=np.float64)
    formula_matrix_cond = np.asarray(setup.formula_matrix_cond, dtype=np.float64)

    def _drop_cache(_ref: weakref.ReferenceType[Any], *, cache_key: int = key) -> None:
        _SETUP_NUMPY_FORMULA_CACHE.pop(cache_key, None)

    _SETUP_NUMPY_FORMULA_CACHE[key] = (
        weakref.ref(setup, _drop_cache),
        formula_matrix,
        formula_matrix_cond,
    )
    return formula_matrix, formula_matrix_cond


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
    indices = np.asarray(tuple(int(index) for index in support_indices), dtype=np.int64)
    amounts = np.asarray(support_amounts)
    if indices.ndim != 1:
        raise ValueError("support_indices must be one-dimensional.")
    if amounts.ndim != 1:
        raise ValueError("support_amounts must be one-dimensional.")
    if indices.shape[0] != amounts.shape[0]:
        raise ValueError("support_indices and support_amounts must have the same length.")
    if np.any(indices < 0) or np.any(indices >= condensate_count):
        raise ValueError("support_indices contain an out-of-range condensate index.")
    full = np.zeros((condensate_count,), dtype=amounts.dtype)
    if indices.size:
        full[indices] = amounts
    return jnp.asarray(full)


def _external_condensate_amounts_vector(
    *,
    support_indices: Sequence[int],
    support_amounts: Sequence[float],
    condensate_count: int,
) -> Array:
    """Return a full-length vector for condensates externalized from the solver."""

    indices = tuple(int(index) for index in support_indices)
    amounts = np.asarray(support_amounts, dtype=np.float64)
    if amounts.ndim != 1:
        raise ValueError("external support_amounts must be one-dimensional.")
    if len(indices) != amounts.shape[0]:
        raise ValueError("external support_indices and support_amounts must have the same length.")
    full = np.zeros((condensate_count,), dtype=amounts.dtype)
    if indices:
        full[np.asarray(indices, dtype=np.int64)] += amounts
    return jnp.asarray(full, dtype=jnp.float64)


def _merge_external_condensate_amounts(
    *,
    condensate_amounts: Array,
    external_condensate_amounts: Sequence[float] | Array | None,
) -> Array:
    """Add externally budgeted condensates back to the public full vector."""

    amounts = np.asarray(condensate_amounts, dtype=np.float64)
    if external_condensate_amounts is None:
        return jnp.asarray(amounts, dtype=jnp.float64)
    external = np.asarray(external_condensate_amounts, dtype=np.float64)
    if external.ndim != 1 or external.shape[0] != amounts.shape[0]:
        raise ValueError("external_condensate_amounts must match condensate_count.")
    return jnp.asarray(amounts + external, dtype=jnp.float64)


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
    if options.fixed_support_gas_init_policy not in {
        "depleted_budget",
        "full_budget",
    }:
        raise ValueError(
            "fixed_support_gas_init_policy must be 'depleted_budget' or "
            "'full_budget'."
        )
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
    if not isinstance(options.enable_experimental_profile_fixed_support_batch, bool):
        raise TypeError(
            "enable_experimental_profile_fixed_support_batch must be a bool."
        )
    if not isinstance(
        options.enable_experimental_profile_fixed_support_fallback_rescue,
        bool,
    ):
        raise TypeError(
            "enable_experimental_profile_fixed_support_fallback_rescue must be a bool."
        )
    rescue_floors = tuple(
        float(value)
        for value in options.experimental_profile_fixed_support_rescue_prune_relative_floors
    )
    if not rescue_floors or any(
        not math.isfinite(value) or value <= 0.0 for value in rescue_floors
    ):
        raise ValueError(
            "experimental_profile_fixed_support_rescue_prune_relative_floors "
            "must contain positive finite values."
        )
    if (
        not math.isfinite(
            float(
                options.experimental_profile_fixed_support_rescue_residual_tolerance_multiplier
            )
        )
        or options.experimental_profile_fixed_support_rescue_residual_tolerance_multiplier
        <= 0.0
    ):
        raise ValueError(
            "experimental_profile_fixed_support_rescue_residual_tolerance_multiplier "
            "must be positive and finite."
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
    if options.head_route_primary_step_control_policy not in (
        "component_clip",
        "scalar_fraction_to_boundary",
    ):
        raise ValueError(
            "head_route_primary_step_control_policy must be 'component_clip' "
            "or 'scalar_fraction_to_boundary'."
        )
    if options.head_route_primary_continuation_mode not in (
        "legacy_policy",
        "pdipm_core",
        "pdipm_core_single_loop",
    ):
        raise ValueError(
            "head_route_primary_continuation_mode must be 'legacy_policy', "
            "'pdipm_core', or 'pdipm_core_single_loop'."
        )
    if (
        not math.isfinite(float(options.head_route_primary_fraction_to_boundary_safety))
        or options.head_route_primary_fraction_to_boundary_safety <= 0.0
        or options.head_route_primary_fraction_to_boundary_safety >= 1.0
    ):
        raise ValueError(
            "head_route_primary_fraction_to_boundary_safety must be finite and in (0, 1)."
        )
    if (
        not math.isfinite(float(options.head_route_primary_tiny_step_alpha_threshold))
        or options.head_route_primary_tiny_step_alpha_threshold < 0.0
    ):
        raise ValueError(
            "head_route_primary_tiny_step_alpha_threshold must be finite and non-negative."
        )
    if options.head_route_primary_tiny_step_consecutive_limit < 1:
        raise ValueError("head_route_primary_tiny_step_consecutive_limit must be positive.")
    if not isinstance(
        options.head_route_primary_tiny_step_switch_to_restoration, bool
    ):
        raise TypeError(
            "head_route_primary_tiny_step_switch_to_restoration must be a bool."
        )
    if options.head_route_primary_dual_initialization_policy not in (
        "centered_from_epsilon",
        "ipopt_push_floor",
    ):
        raise ValueError(
            "head_route_primary_dual_initialization_policy must be "
            "'centered_from_epsilon' or 'ipopt_push_floor'."
        )
    if options.head_route_primary_dual_initialization_policy == "ipopt_push_floor":
        if options.head_route_primary_dual_push_floor is None:
            raise ValueError(
                "head_route_primary_dual_push_floor is required for ipopt_push_floor."
            )
        if (
            not math.isfinite(float(options.head_route_primary_dual_push_floor))
            or options.head_route_primary_dual_push_floor <= 0.0
        ):
            raise ValueError(
                "head_route_primary_dual_push_floor must be finite and positive."
            )
    if not isinstance(options.enable_head_route_scalar_step_control_retry, bool):
        raise TypeError("enable_head_route_scalar_step_control_retry must be a bool.")
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
    if (
        options.support_closure_max_positive_inactive_count is not None
        and int(options.support_closure_max_positive_inactive_count) < 0
    ):
        raise ValueError(
            "support_closure_max_positive_inactive_count must be non-negative or None."
        )
    if not isinstance(options.enable_lifecycle_final_state_support_growth, bool):
        raise TypeError("enable_lifecycle_final_state_support_growth must be a bool.")
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
    target = np.asarray(element_inventory_target, dtype=np.float64)
    if target.ndim != 1 or target.shape[0] != len(setup.elements):
        raise ValueError("element_inventory_target must have one value per element.")
    gas_amounts = np.asarray(gas_n, dtype=np.float64)
    cond_amounts = np.asarray(condensate_amounts, dtype=np.float64)
    if gas_amounts.ndim != 1 or gas_amounts.shape[0] != len(setup.gas_species):
        raise ValueError("gas_n must have one value per gas species.")
    if cond_amounts.ndim != 1 or cond_amounts.shape[0] != len(setup.condensate_species):
        raise ValueError("condensate_amounts must have one value per condensate species.")
    formula_matrix, formula_matrix_cond = _formula_matrices_numpy(setup)
    gas_budget = formula_matrix @ gas_amounts
    condensate_budget = formula_matrix_cond @ cond_amounts
    reconstructed = gas_budget + condensate_budget
    residual = reconstructed - target
    denominator = np.maximum(np.abs(target), 1.0e-300)
    signed_relative = residual / denominator
    absolute_relative = np.abs(signed_relative)
    gate_mask = np.asarray(
        tuple(str(element) not in {"e-", "electron"} for element in setup.elements),
        dtype=bool,
    )
    gated_absolute_relative = np.where(gate_mask, absolute_relative, 0.0)
    finite = bool(np.all(np.isfinite(np.where(gate_mask, absolute_relative, 0.0))))
    sanitized = np.where(
        np.isfinite(gated_absolute_relative),
        gated_absolute_relative,
        np.inf,
    )
    max_index = int(np.argmax(sanitized))
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
    support = np.asarray(tuple(int(index) for index in support_indices), dtype=np.int64)
    support_values = np.asarray(support_amounts, dtype=np.float64)
    if support.ndim != 1:
        raise ValueError("support_indices must be one-dimensional.")
    if support_values.ndim != 1:
        raise ValueError("support_amounts must be one-dimensional.")
    if support.shape[0] != support_values.shape[0]:
        raise ValueError("support_indices and support_amounts must have the same length.")
    condensate_count = len(setup.condensate_species)
    if np.any(support < 0) or np.any(support >= condensate_count):
        raise ValueError("support_indices contain an out-of-range condensate index.")
    condensate_amounts = np.zeros((condensate_count,), dtype=np.float64)
    if support.size:
        condensate_amounts[support] = support_values
    if external_condensate_amounts is not None:
        external = np.asarray(external_condensate_amounts, dtype=np.float64)
        if external.ndim != 1 or external.shape[0] != condensate_count:
            raise ValueError("external_condensate_amounts must match condensate_count.")
        condensate_amounts = condensate_amounts + external
    return _full_condensate_element_budget_residual_report(
        setup=setup,
        gas_n=np.exp(np.asarray(gas_ln_n, dtype=np.float64)),
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
    ag, ac_full = _formula_matrices_numpy(setup)
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
    bounded_lsq_report: dict[str, Any] | None = None
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

    report = _full_condensate_budget_gate_report_for_support_state(
        setup=setup,
        gas_ln_n=gas_ln_n,
        support_indices=support,
        support_amounts=jnp.asarray(amounts),
        external_condensate_amounts=external,
        element_inventory_target=element_inventory_target,
        relative_tolerance=relative_tolerance,
    )
    if not bool(report["accepted"]):
        active_rows = np.asarray(
            [str(element) not in {"e-", "electron"} for element in setup.elements],
            dtype=bool,
        )
        rhs = target - gas_budget - external_budget
        lower = np.zeros((len(support),), dtype=np.float64)
        upper = np.full((len(support),), np.inf, dtype=np.float64)
        upper[finite_capacity] = capacity[finite_capacity]
        bounded_lsq_attempts: list[dict[str, Any]] = []
        for regularization in (0.0, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2):
            matrix = ac[active_rows, :] * row_weights[active_rows, None]
            vector = rhs[active_rows] * row_weights[active_rows]
            if regularization > 0.0:
                sqrt_reg = math.sqrt(float(regularization))
                matrix = np.vstack(
                    [matrix, sqrt_reg * np.eye(len(support), dtype=np.float64)]
                )
                vector = np.concatenate([vector, sqrt_reg * amounts])
            try:
                solution = lsq_linear(
                    matrix,
                    vector,
                    bounds=(lower, upper),
                    method="bvls",
                    max_iter=500,
                    tol=1.0e-12,
                )
            except (ValueError, RuntimeError, FloatingPointError):
                continue
            trial = np.asarray(solution.x, dtype=np.float64)
            if (
                trial.ndim != 1
                or trial.shape[0] != len(support)
                or not np.all(np.isfinite(trial))
                or np.any(trial < 0.0)
            ):
                continue
            trial_report = _full_condensate_budget_gate_report_for_support_state(
                setup=setup,
                gas_ln_n=gas_ln_n,
                support_indices=support,
                support_amounts=jnp.asarray(trial),
                external_condensate_amounts=external,
                element_inventory_target=element_inventory_target,
                relative_tolerance=relative_tolerance,
            )
            attempt = {
                "regularization": float(regularization),
                "accepted": bool(trial_report["accepted"]),
                "solver_success": bool(solution.success),
                "solver_status": int(solution.status),
                "solver_cost": float(solution.cost),
                "solver_optimality": float(solution.optimality),
                "max_abs_relative_residual": float(
                    trial_report["max_abs_relative_residual"]
                ),
                "max_abs_relative_residual_element": trial_report[
                    "max_abs_relative_residual_element"
                ],
            }
            bounded_lsq_attempts.append(attempt)
            if bool(trial_report["accepted"]):
                amounts = trial
                report = trial_report
                accepted = True
                break
        bounded_lsq_report = {
            "restoration_schema": (
                "exogibbs_full_condensate_budget_bounded_lsq_amount_restoration_v1"
            ),
            "triggered": True,
            "accepted": bool(accepted),
            "attempts": tuple(bounded_lsq_attempts),
        }

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
        deficit_indices = np.flatnonzero(signed_relative < -relative_tolerance)
        if deficit_indices.size == 0:
            break
        element_index = int(
            deficit_indices[np.argmax(np.abs(signed_relative[deficit_indices]))]
        )
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
        "bounded_lsq_amount_restoration": bounded_lsq_report,
        "initial_full_condensate_budget_gate": initial_report,
        "final_full_condensate_budget_gate": final_report,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }
    if final_report["accepted"]:
        return jnp.asarray(amounts, dtype=jnp.float64), polish_report
    return jnp.asarray(support_amounts, dtype=jnp.float64), polish_report


def _polish_gas_log_amounts_for_full_condensate_budget_gate(
    *,
    setup: CondensateChemicalSetup,
    gas_ln_n: Array,
    condensate_amounts: Array,
    element_inventory_target: Array,
    relative_tolerance: float,
    max_iterations: int = 16,
    max_abs_delta_q: float = 2.0,
) -> tuple[jnp.ndarray, Mapping[str, Any] | None]:
    """Restore full element budget by minimally adjusting gas log amounts."""

    q = np.asarray(gas_ln_n, dtype=np.float64).copy()
    condensates = np.asarray(condensate_amounts, dtype=np.float64)
    target = np.asarray(element_inventory_target, dtype=np.float64)
    if (
        q.ndim != 1
        or q.shape[0] != len(setup.gas_species)
        or condensates.ndim != 1
        or condensates.shape[0] != len(setup.condensate_species)
        or target.ndim != 1
        or target.shape[0] != len(setup.elements)
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(condensates))
        or not np.all(np.isfinite(target))
    ):
        return jnp.asarray(gas_ln_n, dtype=jnp.float64), None

    ag, ac = _formula_matrices_numpy(setup)
    condensate_budget = ac @ condensates
    positive_target = target[target > 0.0]
    target_scale = float(np.max(positive_target)) if positive_target.size else 1.0
    floor = max(float(np.finfo(np.float64).tiny), 1.0e-300 * target_scale)
    row_weights = 1.0 / np.maximum(np.abs(target), floor)
    ignored = [str(element) in {"e-", "electron"} for element in setup.elements]
    active_rows = np.asarray([not value for value in ignored], dtype=bool)

    def gate_report(q_values: np.ndarray) -> dict[str, Any]:
        return _full_condensate_element_budget_residual_report(
            setup=setup,
            gas_n=np.exp(q_values),
            condensate_amounts=condensates,
            element_inventory_target=element_inventory_target,
            relative_tolerance=relative_tolerance,
        )

    initial_report = gate_report(q)
    if bool(initial_report["accepted"]):
        return jnp.asarray(q, dtype=jnp.float64), None

    accepted = False
    iteration_count = 0
    final_report = initial_report
    for iteration in range(int(max_iterations)):
        with np.errstate(over="ignore", invalid="ignore"):
            gas_n = np.exp(q)
        if not np.all(np.isfinite(gas_n)):
            break
        budget = ag @ gas_n + condensate_budget - target
        jac = ag * gas_n[None, :]
        matrix = jac[active_rows, :] * row_weights[active_rows, None]
        rhs = -budget[active_rows] * row_weights[active_rows]
        if matrix.size == 0:
            break
        delta_q, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
        if not np.all(np.isfinite(delta_q)):
            break
        norm_inf = float(np.max(np.abs(delta_q))) if delta_q.size else 0.0
        if norm_inf > max_abs_delta_q and norm_inf > 0.0:
            delta_q = delta_q * (max_abs_delta_q / norm_inf)
        trial_q = q + delta_q
        if not np.all(np.isfinite(trial_q)):
            break
        q = trial_q
        iteration_count = iteration + 1
        final_report = gate_report(q)
        accepted = bool(final_report["accepted"])
        if accepted:
            break

    polish_report = {
        "polish_schema": "exogibbs_full_condensate_budget_gas_log_amount_polish_v1",
        "triggered": True,
        "accepted": bool(accepted),
        "iteration_count": iteration_count,
        "initial_full_condensate_budget_gate": initial_report,
        "final_full_condensate_budget_gate": final_report,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }
    if accepted:
        return jnp.asarray(q, dtype=jnp.float64), polish_report
    return jnp.asarray(q, dtype=jnp.float64), polish_report


def _restore_full_budget_feasibility_for_active_support(
    *,
    setup: CondensateChemicalSetup,
    gas_ln_n: Array,
    support_indices: Sequence[int],
    support_amounts: Array,
    external_condensate_amounts: Sequence[float] | Array | None = None,
    element_inventory_target: Array,
    relative_tolerance: float,
    max_iterations: int = 64,
    max_abs_delta: float = 4.0,
) -> tuple[jnp.ndarray, jnp.ndarray, Mapping[str, Any] | None]:
    """Restore full-budget feasibility by jointly moving gas and active amounts."""

    support = tuple(int(index) for index in support_indices)
    if not support:
        return (
            jnp.asarray(gas_ln_n, dtype=jnp.float64),
            jnp.asarray(support_amounts, dtype=jnp.float64),
            None,
        )
    q = np.asarray(gas_ln_n, dtype=np.float64).copy()
    amounts = np.asarray(support_amounts, dtype=np.float64).copy()
    target = np.asarray(element_inventory_target, dtype=np.float64)
    if (
        q.ndim != 1
        or q.shape[0] != len(setup.gas_species)
        or amounts.ndim != 1
        or amounts.shape[0] != len(support)
        or target.ndim != 1
        or target.shape[0] != len(setup.elements)
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(amounts))
        or np.any(amounts < 0.0)
        or not np.all(np.isfinite(target))
    ):
        return (
            jnp.asarray(gas_ln_n, dtype=jnp.float64),
            jnp.asarray(support_amounts, dtype=jnp.float64),
            None,
        )

    ag, ac_full = _formula_matrices_numpy(setup)
    ac = ac_full[:, support]
    external = (
        np.zeros((ac_full.shape[1],), dtype=np.float64)
        if external_condensate_amounts is None
        else np.asarray(external_condensate_amounts, dtype=np.float64)
    )
    if external.ndim != 1 or external.shape[0] != ac_full.shape[1]:
        return (
            jnp.asarray(gas_ln_n, dtype=jnp.float64),
            jnp.asarray(support_amounts, dtype=jnp.float64),
            None,
        )
    external_budget = ac_full @ external
    positive_target = target[target > 0.0]
    target_scale = float(np.max(positive_target)) if positive_target.size else 1.0
    floor = max(float(np.finfo(np.float64).tiny), 1.0e-300 * target_scale)
    row_weights = 1.0 / np.maximum(np.abs(target), floor)
    active_rows = np.asarray(
        [str(element) not in {"e-", "electron"} for element in setup.elements],
        dtype=bool,
    )

    def gate_report(q_values: np.ndarray, amount_values: np.ndarray) -> dict[str, Any]:
        return _full_condensate_budget_gate_report_for_support_state(
            setup=setup,
            gas_ln_n=jnp.asarray(q_values, dtype=jnp.float64),
            support_indices=support,
            support_amounts=jnp.asarray(amount_values, dtype=jnp.float64),
            external_condensate_amounts=external,
            element_inventory_target=element_inventory_target,
            relative_tolerance=relative_tolerance,
        )

    initial_report = gate_report(q, amounts)
    if bool(initial_report["accepted"]):
        return jnp.asarray(q, dtype=jnp.float64), jnp.asarray(amounts, dtype=jnp.float64), None

    best_q = q.copy()
    best_amounts = amounts.copy()
    best_report = initial_report
    accepted = False
    iteration_count = 0
    accepted_step_count = 0
    rejected_step_count = 0
    for iteration in range(int(max_iterations)):
        with np.errstate(over="ignore", invalid="ignore"):
            gas_n = np.exp(q)
        if not np.all(np.isfinite(gas_n)):
            break
        safe_amounts = np.maximum(amounts, 1.0e-300)
        budget = ag @ gas_n + ac @ amounts + external_budget - target
        gas_jac = ag * gas_n[None, :]
        amount_jac = ac * safe_amounts[None, :]
        matrix = np.concatenate([gas_jac, amount_jac], axis=1)
        matrix = matrix[active_rows, :] * row_weights[active_rows, None]
        rhs = -budget[active_rows] * row_weights[active_rows]
        if matrix.size == 0:
            break
        try:
            delta, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(delta)):
            break
        norm_inf = float(np.max(np.abs(delta))) if delta.size else 0.0
        if norm_inf > max_abs_delta and norm_inf > 0.0:
            delta = delta * (max_abs_delta / norm_inf)
        delta_q = delta[: q.shape[0]]
        delta_r = delta[q.shape[0] :]
        current_score = float(best_report["max_abs_relative_residual"])
        accepted_trial = False
        for step_scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial_q = q + step_scale * delta_q
            trial_amounts = amounts * np.exp(step_scale * delta_r)
            if (
                not np.all(np.isfinite(trial_q))
                or not np.all(np.isfinite(trial_amounts))
                or np.any(trial_amounts < 0.0)
            ):
                continue
            trial_report = gate_report(trial_q, trial_amounts)
            trial_score = float(trial_report["max_abs_relative_residual"])
            if trial_score < current_score:
                q = trial_q
                amounts = trial_amounts
                best_q = trial_q
                best_amounts = trial_amounts
                best_report = trial_report
                accepted = bool(trial_report["accepted"])
                accepted_trial = True
                accepted_step_count += 1
                break
        iteration_count = iteration + 1
        if accepted:
            break
        if not accepted_trial:
            rejected_step_count += 1
            break

    restoration_report = {
        "restoration_schema": (
            "exogibbs_full_budget_joint_gas_active_amount_restoration_v1"
        ),
        "triggered": True,
        "accepted": bool(accepted),
        "iteration_count": iteration_count,
        "accepted_step_count": accepted_step_count,
        "rejected_step_count": rejected_step_count,
        "max_abs_delta": float(max_abs_delta),
        "initial_full_condensate_budget_gate": initial_report,
        "final_full_condensate_budget_gate": best_report,
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }
    return (
        jnp.asarray(best_q, dtype=jnp.float64),
        jnp.asarray(best_amounts, dtype=jnp.float64),
        restoration_report,
    )


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
    policy["continuation_mode"] = str(options.head_route_primary_continuation_mode)
    policy["step_control_policy"] = str(options.head_route_primary_step_control_policy)
    policy["fraction_to_boundary_safety"] = float(
        options.head_route_primary_fraction_to_boundary_safety
    )
    policy["ipopt_tiny_step_alpha_threshold"] = float(
        options.head_route_primary_tiny_step_alpha_threshold
    )
    policy["ipopt_tiny_step_consecutive_limit"] = int(
        options.head_route_primary_tiny_step_consecutive_limit
    )
    policy["ipopt_tiny_step_switch_to_restoration"] = bool(
        options.head_route_primary_tiny_step_switch_to_restoration
    )
    policy["ipopt_allow_fast_monotone_decrease"] = bool(
        options.head_route_primary_ipopt_allow_fast_monotone_decrease
    )
    return policy


def _continuation_stopped_reason_from_lifecycle_payload(
    lifecycle_payload: Mapping[str, Any],
) -> str | None:
    primary = lifecycle_payload.get("primary_execution_report")
    if not isinstance(primary, Mapping):
        return None
    continuation = primary.get("continuation_report")
    if not isinstance(continuation, Mapping):
        return None
    reason = continuation.get("stopped_reason")
    return None if reason is None else str(reason)


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
            condensate_species_names=setup.condensate_species,
            primary_summary=options.head_route_primary_summary,
            primary_continuation_policy=_head_lifecycle_primary_policy(options),
            dual_initialization_policy=(
                options.head_route_primary_dual_initialization_policy
            ),
            dual_push_floor=options.head_route_primary_dual_push_floor,
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
        condensate_species_names=setup.condensate_species,
        external_condensate_budget=external_budget,
        primary_summary=options.head_route_primary_summary,
        primary_continuation_policy=primary_continuation_policy,
        dual_initialization_policy=options.head_route_primary_dual_initialization_policy,
        dual_push_floor=options.head_route_primary_dual_push_floor,
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
    line_search_stopped_reasons = {
        "no_p_armijo_trial",
        "no_finite_trial",
        "no_accepted_trial",
        "no_acceptable_ipopt_filter_trial",
        "ipopt_h_filter_rejected",
        "ipopt_persistent_filter_rejected",
        "ipopt_persistent_filter_f_type_rejected",
        "filter_restoration_rejected",
        "tiny_step_no_restoration_trial",
        "tiny_step_requires_restoration",
    }
    if str(continuation.get("stopped_reason")) not in line_search_stopped_reasons:
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


def _mapping_at(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _build_caveat_route_breakdown(
    *,
    selected_route: str,
    status: str,
    acceptance_tier: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a compact explanation of caveat-bearing HEAD route results."""

    lifecycle = _mapping_at(metadata, "head_route_lifecycle")
    route_result = _mapping_at(lifecycle, "route_result")
    route_selection = _mapping_at(lifecycle, "route_selection_report")
    primary = _mapping_at(lifecycle, "primary_execution_report")
    continuation = _mapping_at(primary, "continuation_report")
    if not continuation:
        primary_summary = _mapping_at(lifecycle, "primary_summary")
        continuation = _mapping_at(primary_summary, "continuation_report")
    native_fallback = _mapping_at(metadata, "native_seed_fallback")
    full_budget_gate = _mapping_at(metadata, "full_condensate_budget_residual_gate")
    gas_polish = _mapping_at(metadata, "full_condensate_budget_gas_log_amount_polish")
    joint_restoration = _mapping_at(
        metadata,
        "full_condensate_budget_joint_restoration",
    )
    soft_retry = _mapping_at(metadata, "head_route_soft_restoration_retry")
    ipopt_h_retry = _mapping_at(metadata, "head_route_ipopt_h_type_retry")

    caveat_reasons: list[str] = []
    route = str(selected_route)
    if status == CONVERGED_WITH_CAVEAT:
        if "budget_tradeoff" in route:
            caveat_reasons.append("native_budget_tradeoff_route")
        elif "raw_gas" in route or "electron" in route:
            caveat_reasons.append("raw_gas_or_electron_caveat_route")
        else:
            caveat_reasons.append("accepted_non_tier1_route")
    elif status == NOT_CONVERGED:
        caveat_reasons.append("not_converged")
    else:
        caveat_reasons.append("none")
    if bool(native_fallback):
        caveat_reasons.append("primary_lifecycle_not_accepted_before_fallback")
    if bool(joint_restoration):
        caveat_reasons.append("final_full_budget_joint_restoration_used")
    elif bool(gas_polish):
        caveat_reasons.append("final_full_budget_gas_polish_used")
    if bool(soft_retry) and not bool(soft_retry.get("accepted", False)):
        caveat_reasons.append("soft_restoration_retry_rejected")
    if bool(ipopt_h_retry) and not bool(ipopt_h_retry.get("accepted", False)):
        caveat_reasons.append("ipopt_h_type_retry_rejected")
    if bool(full_budget_gate) and not bool(full_budget_gate.get("accepted", False)):
        caveat_reasons.append("final_full_budget_gate_rejected")

    return {
        "report_schema": "exogibbs_condensate_caveat_route_breakdown_v1",
        "selected_route": route,
        "status": str(status),
        "acceptance_tier": str(acceptance_tier),
        "is_caveat": bool(status == CONVERGED_WITH_CAVEAT),
        "caveat_reasons": tuple(caveat_reasons),
        "primary": {
            "status": primary.get("status"),
            "stopped_reason": continuation.get("stopped_reason"),
            "reached_final_barrier": continuation.get("reached_final_barrier"),
            "converged_at_final_barrier": continuation.get(
                "converged_at_final_barrier"
            ),
            "final_barrier": continuation.get("final_barrier"),
            "outer_iteration_count": continuation.get("outer_iteration_count"),
            "inner_iteration_count": continuation.get("inner_iteration_count"),
            "filter_accept_count": continuation.get("filter_accept_count"),
            "restoration_count": continuation.get("restoration_count"),
            "tiny_step_count": continuation.get("tiny_step_count"),
        },
        "selector": {
            "selected_route": route_selection.get("selected_route"),
            "integrated_status": route_selection.get("integrated_status"),
            "route_reason": route_selection.get("route_reason"),
            "primary_centered": route_selection.get("primary_centered"),
            "fallback_available": route_selection.get("fallback_available"),
            "fallback_accepted": route_selection.get("fallback_accepted"),
            "refresh_policy_available": route_selection.get(
                "refresh_policy_available"
            ),
            "refresh_policy_accepted": route_selection.get("refresh_policy_accepted"),
        },
        "route_result": {
            "selected_route": route_result.get("selected_route"),
            "metric_status": route_result.get("metric_status"),
            "standard_path_status": route_result.get("standard_path_status"),
            "acceptance_tier": route_result.get("acceptance_tier"),
            "converged": route_result.get("converged"),
        },
        "fallback": {
            "available": bool(native_fallback),
            "fallback_gas_source": native_fallback.get("fallback_gas_source"),
            "fallback_support_amount_source": native_fallback.get(
                "fallback_support_amount_source"
            ),
            "restricted_solver_success": metadata.get("restricted_solver_success"),
            "solver_success": metadata.get("solver_success"),
        },
        "final_budget_gate": {
            "available": bool(full_budget_gate),
            "accepted": full_budget_gate.get("accepted"),
            "relative_l2": full_budget_gate.get("relative_l2"),
            "relative_max_abs": full_budget_gate.get("relative_max_abs"),
            "relative_tolerance": full_budget_gate.get("relative_tolerance"),
        },
        "final_restoration": {
            "gas_log_amount_polish_used": bool(gas_polish),
            "gas_log_amount_polish_accepted": gas_polish.get("accepted"),
            "joint_restoration_used": bool(joint_restoration),
            "joint_restoration_accepted": joint_restoration.get("accepted"),
            "joint_restoration_initial_max_abs_relative_residual": _mapping_at(
                joint_restoration,
                "initial_full_condensate_budget_gate",
            ).get("max_abs_relative_residual"),
            "joint_restoration_final_max_abs_relative_residual": _mapping_at(
                joint_restoration,
                "final_full_condensate_budget_gate",
            ).get("max_abs_relative_residual"),
        },
        "pdipm_retry_attempts": {
            "soft_restoration_triggered": bool(soft_retry),
            "soft_restoration_accepted": soft_retry.get("accepted"),
            "soft_restoration_trigger_mode": soft_retry.get("trigger_mode"),
            "soft_restoration_initial_stopped_reason": soft_retry.get(
                "initial_stopped_reason"
            ),
            "soft_restoration_retry_selected_route": soft_retry.get(
                "retry_selected_route"
            ),
            "soft_restoration_retry_metric_status": soft_retry.get(
                "retry_metric_status"
            ),
            "ipopt_h_type_triggered": bool(ipopt_h_retry),
            "ipopt_h_type_accepted": ipopt_h_retry.get("accepted"),
            "ipopt_h_type_retry_selected_route": ipopt_h_retry.get(
                "retry_selected_route"
            ),
            "ipopt_h_type_retry_metric_status": ipopt_h_retry.get(
                "retry_metric_status"
            ),
        },
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }


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
    support_amounts_array = jnp.asarray(support_amounts, dtype=jnp.float64)
    condensate_amounts = _full_condensate_amounts(
        support_indices=support_indices,
        support_amounts=support_amounts_array,
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
    metadata: dict[str, Any] = dict(diagnostics or {})
    if (
        enable_full_condensate_budget_residual_gate
        and element_inventory_target is not None
        and status in {CONVERGED, CONVERGED_WITH_CAVEAT}
    ):
        polished_gas_ln_n, gas_polish_report = (
            _polish_gas_log_amounts_for_full_condensate_budget_gate(
                setup=setup,
                gas_ln_n=gas_ln_n_array,
                condensate_amounts=condensate_amounts,
                element_inventory_target=element_inventory_target,
                relative_tolerance=full_condensate_budget_relative_tolerance,
            )
        )
        if gas_polish_report is not None:
            metadata["full_condensate_budget_gas_log_amount_polish"] = (
                gas_polish_report
            )
            initial_gate = gas_polish_report["initial_full_condensate_budget_gate"]
            final_gate = gas_polish_report["final_full_condensate_budget_gate"]
            fallback_improved = (
                selected_route == "native_budget_seed_fallback_budget_tradeoff"
                and float(final_gate["max_abs_relative_residual"])
                < float(initial_gate["max_abs_relative_residual"])
            )
            if bool(gas_polish_report["accepted"]) or fallback_improved:
                gas_ln_n_array = polished_gas_ln_n
        if selected_route == "native_budget_seed_fallback_budget_tradeoff":
            restored_gas_ln_n, restored_support_amounts, restoration_report = (
                _restore_full_budget_feasibility_for_active_support(
                    setup=setup,
                    gas_ln_n=gas_ln_n_array,
                    support_indices=support_indices,
                    support_amounts=support_amounts_array,
                    external_condensate_amounts=external_condensate_amounts,
                    element_inventory_target=element_inventory_target,
                    relative_tolerance=full_condensate_budget_relative_tolerance,
                )
            )
        else:
            restored_gas_ln_n = gas_ln_n_array
            restored_support_amounts = support_amounts_array
            restoration_report = None
        if restoration_report is not None:
            metadata["full_condensate_budget_joint_restoration"] = (
                restoration_report
            )
            initial_gate = restoration_report["initial_full_condensate_budget_gate"]
            final_gate = restoration_report["final_full_condensate_budget_gate"]
            fallback_improved = (
                selected_route == "native_budget_seed_fallback_budget_tradeoff"
                and float(final_gate["max_abs_relative_residual"])
                < float(initial_gate["max_abs_relative_residual"])
            )
            if bool(restoration_report["accepted"]) or fallback_improved:
                gas_ln_n_array = restored_gas_ln_n
                support_amounts_array = restored_support_amounts
                condensate_amounts = _full_condensate_amounts(
                    support_indices=support_indices,
                    support_amounts=support_amounts_array,
                    condensate_count=len(setup.condensate_species),
                )
                condensate_amounts = _merge_external_condensate_amounts(
                    condensate_amounts=condensate_amounts,
                    external_condensate_amounts=external_condensate_amounts,
                )
    gas_n = jnp.exp(gas_ln_n_array)
    gas_ntot = jnp.sum(gas_n)
    gas_x = gas_n / jnp.clip(gas_ntot, 1.0e-300)
    support_index_array = jnp.asarray(support_indices, dtype=jnp.int32)
    support_names = tuple(setup.condensate_species[int(index)] for index in support_index_array.tolist())
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
    metadata["caveat_route_breakdown"] = _build_caveat_route_breakdown(
        selected_route=selected_route,
        status=status,
        acceptance_tier=acceptance_tier,
        metadata=metadata,
    )
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

    seed_support_indices = tuple(int(index) for index in candidate.support_indices)
    seed_support_amounts = tuple(
        float(value) for value in candidate.support_amounts_init
    )
    if candidate.initial_log_state_override is not None:
        seed_gas_ln_n = jnp.asarray(candidate.initial_log_state_override.ln_nk)
        seed_gas_source = "selected_warm_start_candidate_gas_state"
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
        seed_gas_ln_n = gas_result.ln_n
        seed_gas_source = "native_gas_equilibrium"
    fallback_gas_ln_n = seed_gas_ln_n
    fallback_gas_source = seed_gas_source
    fallback_support_indices = seed_support_indices
    fallback_support_amounts = seed_support_amounts
    external_condensate_amounts = None
    final_state_candidate_selected = False
    final_state_payload = _lifecycle_final_state_payload(lifecycle_payload)
    if isinstance(final_state_payload, Mapping):
        try:
            final_support_indices = _final_state_support_indices_from_lifecycle_payload(
                lifecycle_payload,
                fallback_support_indices=seed_support_indices,
            )
            final_ln_nk = jnp.asarray(final_state_payload["ln_nk"], dtype=jnp.float64)
            final_ln_mk = jnp.asarray(final_state_payload["ln_mk"], dtype=jnp.float64)
        except (KeyError, TypeError, ValueError):
            final_support_indices = ()
            final_ln_nk = jnp.asarray(())
            final_ln_mk = jnp.asarray(())
        if (
            final_ln_nk.ndim == 1
            and final_ln_mk.ndim == 1
            and final_ln_mk.shape[0] == len(final_support_indices)
            and bool(jnp.all(jnp.isfinite(final_ln_nk)))
            and bool(jnp.all(jnp.isfinite(final_ln_mk)))
        ):
            final_amounts = jnp.exp(final_ln_mk)
            if bool(jnp.all(jnp.isfinite(final_amounts))):
                final_gas_ln_n = final_ln_nk
                final_support_amounts = tuple(
                    float(value) for value in final_amounts.tolist()
                )
                final_external_condensate_amounts = (
                    _external_condensate_amounts_from_lifecycle_payload(
                        lifecycle_payload,
                        condensate_count=len(setup.condensate_species),
                    )
                )
                if enable_full_condensate_budget_residual_gate:
                    polished_amounts, amount_polish_report = (
                        _polish_support_amounts_for_full_condensate_budget_gate(
                            setup=setup,
                            gas_ln_n=final_gas_ln_n,
                            support_indices=final_support_indices,
                            support_amounts=final_amounts,
                            external_condensate_amounts=(
                                final_external_condensate_amounts
                            ),
                            element_inventory_target=b,
                            relative_tolerance=full_condensate_budget_relative_tolerance,
                        )
                    )
                    if amount_polish_report is not None and bool(
                        amount_polish_report["accepted"]
                    ):
                        final_support_amounts = tuple(
                            float(value) for value in polished_amounts.tolist()
                        )
                if enable_full_condensate_budget_residual_gate:
                    (
                        best_effort_gas_ln_n,
                        best_effort_gas_polish_report,
                    ) = _polish_gas_log_amounts_for_full_condensate_budget_gate(
                        setup=setup,
                        gas_ln_n=final_gas_ln_n,
                        condensate_amounts=_full_condensate_amounts(
                            support_indices=final_support_indices,
                            support_amounts=jnp.asarray(
                                final_support_amounts,
                                dtype=jnp.float64,
                            ),
                            condensate_count=len(setup.condensate_species),
                        ),
                        element_inventory_target=b,
                        relative_tolerance=full_condensate_budget_relative_tolerance,
                    )
                    if best_effort_gas_polish_report is not None:
                        initial_gate = best_effort_gas_polish_report[
                            "initial_full_condensate_budget_gate"
                        ]
                        final_gate = best_effort_gas_polish_report[
                            "final_full_condensate_budget_gate"
                        ]
                        if float(final_gate["max_abs_relative_residual"]) < float(
                            initial_gate["max_abs_relative_residual"]
                        ):
                            final_gas_ln_n = best_effort_gas_ln_n
                if enable_full_condensate_budget_residual_gate:
                    seed_gate = _full_condensate_budget_gate_report_for_support_state(
                        setup=setup,
                        gas_ln_n=seed_gas_ln_n,
                        support_indices=seed_support_indices,
                        support_amounts=jnp.asarray(
                            seed_support_amounts,
                            dtype=jnp.float64,
                        ),
                        element_inventory_target=b,
                        relative_tolerance=full_condensate_budget_relative_tolerance,
                    )
                    final_gate = _full_condensate_budget_gate_report_for_support_state(
                        setup=setup,
                        gas_ln_n=final_gas_ln_n,
                        support_indices=final_support_indices,
                        support_amounts=jnp.asarray(
                            final_support_amounts,
                            dtype=jnp.float64,
                        ),
                        external_condensate_amounts=(
                            final_external_condensate_amounts
                        ),
                        element_inventory_target=b,
                        relative_tolerance=full_condensate_budget_relative_tolerance,
                    )
                    use_final_state = (
                        bool(final_gate["accepted"])
                        or (
                            not bool(seed_gate["accepted"])
                            and float(final_gate["max_abs_relative_residual"])
                            < float(seed_gate["max_abs_relative_residual"])
                        )
                    )
                else:
                    use_final_state = True
                if use_final_state:
                    fallback_gas_ln_n = final_gas_ln_n
                    fallback_gas_source = "lifecycle_final_state"
                    fallback_support_indices = final_support_indices
                    fallback_support_amounts = final_support_amounts
                    external_condensate_amounts = final_external_condensate_amounts
                    final_state_candidate_selected = True
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
                "fallback_support_amount_source": (
                    "lifecycle_final_state"
                    if fallback_gas_source == "lifecycle_final_state"
                    else "selected_warm_start_candidate_seed"
                ),
                "reason": (
                    "The primary lifecycle did not converge or was not accepted; "
                    "the API returned the best available ExoGibbs-native boundary "
                    "with condensate amounts as a caveat-bearing HEAD route fallback."
                ),
                "fastchem4_trace_public_runtime_constructor_inputs_used": False,
            },
        }
        if (
            final_state_candidate_selected
            and "amount_polish_report" in locals()
            and amount_polish_report is not None
        ):
            diagnostics_payload["full_condensate_budget_amount_polish"] = (
                amount_polish_report
            )
        if (
            final_state_candidate_selected
            and "best_effort_gas_polish_report" in locals()
            and best_effort_gas_polish_report is not None
        ):
            diagnostics_payload[
                "native_seed_fallback_best_effort_gas_log_amount_polish"
            ] = best_effort_gas_polish_report
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
        support_indices=fallback_support_indices,
        support_amounts=fallback_support_amounts,
        external_condensate_amounts=external_condensate_amounts,
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
            "max_positive_inactive_driving_accepted": True,
            "positive_inactive_count": 0,
            "positive_inactive_count_tolerance": None,
            "positive_inactive_count_accepted": True,
            "fastchem4_trace_public_runtime_constructor_inputs_used": False,
        }
    diagnostics = result.diagnostics or {}
    inactive_report = diagnostics.get("inactive_condensate_driving")
    if isinstance(inactive_report, Mapping):
        valid_report = inactive_report.get("temperature_valid_condensates")
        if isinstance(valid_report, Mapping):
            top_rows = tuple(valid_report.get("top_positive_inactive", ()))
            max_driving = float(valid_report.get("max_positive_inactive_driving", 0.0))
            positive_count = int(valid_report.get("positive_inactive_count", 0))
            tolerance = float(options.support_closure_max_positive_inactive_driving)
            count_tolerance = (
                None
                if options.support_closure_max_positive_inactive_count is None
                else int(options.support_closure_max_positive_inactive_count)
            )
            driving_accepted = bool(max_driving <= tolerance)
            count_accepted = bool(
                count_tolerance is None or positive_count <= count_tolerance
            )
            return {
                "gate_schema": "exogibbs_support_closure_retry_gate_v1",
                "enabled": True,
                "closure_scope": "temperature_valid_condensates",
                "accepted": bool(driving_accepted and count_accepted),
                "max_positive_inactive_driving": max_driving,
                "max_positive_inactive_driving_tolerance": tolerance,
                "max_positive_inactive_driving_accepted": driving_accepted,
                "positive_inactive_count": positive_count,
                "positive_inactive_count_tolerance": count_tolerance,
                "positive_inactive_count_accepted": count_accepted,
                "top_positive_inactive": top_rows[:20],
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
            "max_positive_inactive_driving_accepted": False,
            "positive_inactive_count": -1,
            "positive_inactive_count_tolerance": (
                None
                if options.support_closure_max_positive_inactive_count is None
                else int(options.support_closure_max_positive_inactive_count)
            ),
            "positive_inactive_count_accepted": False,
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
    count_tolerance = (
        None
        if options.support_closure_max_positive_inactive_count is None
        else int(options.support_closure_max_positive_inactive_count)
    )
    driving_accepted = bool(max_driving <= tolerance)
    count_accepted = bool(
        count_tolerance is None or len(inactive) <= count_tolerance
    )
    return {
        "gate_schema": "exogibbs_support_closure_retry_gate_v1",
        "enabled": True,
        "accepted": bool(driving_accepted and count_accepted),
        "max_positive_inactive_driving": max_driving,
        "max_positive_inactive_driving_tolerance": tolerance,
        "max_positive_inactive_driving_accepted": driving_accepted,
        "positive_inactive_count": len(inactive),
        "positive_inactive_count_tolerance": count_tolerance,
        "positive_inactive_count_accepted": count_accepted,
        "top_positive_inactive": tuple(top_rows[:20]),
        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
    }


def _support_closure_retry_candidate_score(
    support_closure_gate: Mapping[str, Any],
    *,
    support_count: int,
) -> tuple[float, float, int]:
    """Return a sortable closure score for retry candidates.

    Lower is better.  The gate remains a feasibility filter; this score prevents
    the retry path from stopping at the first barely acceptable cap when a later
    cap closes more inactive support.
    """

    positive_count = support_closure_gate.get("positive_inactive_count", math.inf)
    max_driving = support_closure_gate.get("max_positive_inactive_driving", math.inf)
    try:
        count_value = float(positive_count)
    except (TypeError, ValueError):
        count_value = math.inf
    try:
        driving_value = float(max_driving)
    except (TypeError, ValueError):
        driving_value = math.inf
    return (count_value, driving_value, int(support_count))


@dataclass(frozen=True)
class _SupportClosureRetryCandidate:
    """Selectable fallback retry candidate with a common closure score."""

    retry_kind: str
    result: CondensateEquilibriumResult
    support_closure_gate: Mapping[str, Any]
    attempt: Mapping[str, Any]
    score: tuple[float, float, int]


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


def _condensate_init_from_result(
    result: CondensateEquilibriumResult,
    *,
    min_seed_amount: float,
) -> CondensateEquilibriumInit:
    """Build a reusable profile initial guess from an accepted layer result."""

    support_indices = tuple(
        int(index) for index in np.asarray(result.condensate_support_indices).tolist()
    )
    full_amounts = np.asarray(result.condensate_amounts, dtype=np.float64)
    support_amounts = tuple(
        float(max(full_amounts[index], min_seed_amount)) for index in support_indices
    )
    return CondensateEquilibriumInit(
        gas_ln_n=result.gas_ln_n,
        gas_ntot=jnp.asarray(result.gas_ntot, dtype=jnp.float64),
        condensate_amounts=result.condensate_amounts,
        support_indices=support_indices,
        support_amounts=support_amounts,
    )


def _support_payload_from_condensate_init(
    init: CondensateEquilibriumInit | None,
    *,
    setup: CondensateChemicalSetup,
    min_seed_amount: float,
) -> tuple[tuple[int, ...], tuple[float, ...]] | None:
    """Return finite support payload from an optional profile initializer."""

    if init is None:
        return None
    if init.support_indices is not None:
        support_indices = tuple(int(index) for index in init.support_indices)
        if init.support_amounts is not None:
            support_amounts = _positive_support_amounts_for_warm_start(
                init.support_amounts,
                min_seed_amount=min_seed_amount,
            )
        elif init.condensate_amounts is not None:
            amounts = np.asarray(init.condensate_amounts, dtype=np.float64)
            if amounts.ndim != 1 or amounts.shape[0] != len(setup.condensate_species):
                return None
            support_amounts = _positive_support_amounts_for_warm_start(
                (amounts[index] for index in support_indices),
                min_seed_amount=min_seed_amount,
            )
        else:
            return None
    elif init.condensate_amounts is not None:
        amounts = np.asarray(init.condensate_amounts, dtype=np.float64)
        if amounts.ndim != 1 or amounts.shape[0] != len(setup.condensate_species):
            return None
        active = np.flatnonzero(np.isfinite(amounts) & (amounts > 0.0))
        support_indices = tuple(int(index) for index in active.tolist())
        support_amounts = _positive_support_amounts_for_warm_start(
            (amounts[index] for index in support_indices),
            min_seed_amount=min_seed_amount,
        )
    else:
        return None
    if len(support_indices) != len(support_amounts):
        return None
    if len(set(support_indices)) != len(support_indices):
        return None
    if any(index < 0 or index >= len(setup.condensate_species) for index in support_indices):
        return None
    if not all(math.isfinite(value) and value > 0.0 for value in support_amounts):
        return None
    return support_indices, support_amounts


def _solver_log_state_from_condensate_init(
    init: CondensateEquilibriumInit | None,
    *,
    setup: CondensateChemicalSetup,
    support_amounts_init: Sequence[float],
    source: str,
) -> Any | None:
    """Build an internal restricted-solver init from a profile initializer."""

    if init is None or init.gas_ln_n is None:
        return None
    gas_ln_n = jnp.asarray(init.gas_ln_n, dtype=jnp.float64)
    if gas_ln_n.ndim != 1 or gas_ln_n.shape[0] != len(setup.gas_species):
        return None
    if not bool(jnp.all(jnp.isfinite(gas_ln_n))):
        return None
    if init.gas_ntot is None:
        gas_ntot = jnp.sum(jnp.exp(gas_ln_n))
    else:
        gas_ntot = jnp.asarray(init.gas_ntot, dtype=jnp.float64)
    if not bool(jnp.all(jnp.isfinite(gas_ntot))) or float(gas_ntot) <= 0.0:
        return None
    support_amounts = jnp.asarray(support_amounts_init, dtype=jnp.float64)
    if (
        support_amounts.ndim != 1
        or not bool(jnp.all(jnp.isfinite(support_amounts)))
        or not bool(jnp.all(support_amounts > 0.0))
    ):
        return None
    from exogibbs.optimize.minimize_cond import CondensateEquilibriumInit as SolverInit

    element_potential = None
    if init.element_potential is not None:
        element_potential = jnp.asarray(init.element_potential, dtype=jnp.float64)
        if (
            element_potential.ndim != 1
            or element_potential.shape[0] != len(setup.elements)
            or not bool(jnp.all(jnp.isfinite(element_potential)))
        ):
            element_potential = None
    rho = None
    if init.rho is not None:
        rho = jnp.asarray(init.rho, dtype=jnp.float64)
        if (
            rho.ndim != 1
            or rho.shape[0] not in {len(support_amounts_init), len(setup.condensate_species)}
            or not bool(jnp.all(jnp.isfinite(rho)))
        ):
            rho = None
    barrier_epsilon = None
    if init.barrier_epsilon is not None:
        barrier_epsilon = jnp.asarray(init.barrier_epsilon, dtype=jnp.float64)
        if barrier_epsilon.ndim != 0 or not bool(jnp.isfinite(barrier_epsilon)):
            barrier_epsilon = None
    gas_stationarity_source = None
    if init.gas_stationarity_source is not None:
        gas_stationarity_source = jnp.asarray(
            init.gas_stationarity_source,
            dtype=jnp.float64,
        )
        if (
            gas_stationarity_source.ndim != 1
            or gas_stationarity_source.shape[0] != len(setup.gas_species)
            or not bool(jnp.all(jnp.isfinite(gas_stationarity_source)))
        ):
            gas_stationarity_source = None

    return SolverInit(
        ln_nk=gas_ln_n,
        ln_mk=jnp.log(jnp.maximum(support_amounts, 1.0e-300)),
        ln_ntot=jnp.log(jnp.asarray(gas_ntot, dtype=jnp.float64)),
        element_potential=element_potential,
        rho=rho,
        barrier_epsilon=barrier_epsilon,
        gas_stationarity_source=gas_stationarity_source,
        ln_nk_source_trace={
            "source": source,
            "reason": "Profile or caller-provided condensate warm-start gas state.",
        },
    )


def _full_condensate_amounts_for_support(
    *,
    setup: CondensateChemicalSetup,
    support_indices: Sequence[int],
    support_amounts: Sequence[float],
) -> Array:
    support = tuple(int(index) for index in support_indices)
    amounts = jnp.asarray(support_amounts, dtype=jnp.float64)
    if amounts.ndim != 1 or amounts.shape[0] != len(support):
        raise ValueError("support_indices and support_amounts must have the same length.")
    full = jnp.zeros((len(setup.condensate_species),), dtype=jnp.float64)
    if support:
        full = full.at[jnp.asarray(support, dtype=jnp.int32)].set(amounts)
    return full


def _fixed_support_gas_budget_for_init(
    *,
    setup: CondensateChemicalSetup,
    b: Array,
    support_indices: Sequence[int],
    support_amounts: Sequence[float],
    policy: str,
) -> tuple[Array, Mapping[str, Any]]:
    target = jnp.asarray(b, dtype=jnp.float64)
    if policy == "full_budget":
        return target, {
            "policy": "full_budget",
            "clipped_negative_l1": 0.0,
            "max_condensed_fraction": 0.0,
        }
    if policy != "depleted_budget":
        raise ValueError(
            "fixed_support_gas_init_policy must be 'depleted_budget' or "
            "'full_budget'."
        )
    full_amounts = _full_condensate_amounts_for_support(
        setup=setup,
        support_indices=support_indices,
        support_amounts=support_amounts,
    )
    condensed_budget = (
        jnp.asarray(setup.formula_matrix_cond, dtype=jnp.float64) @ full_amounts
    )
    raw_remaining = target - condensed_budget
    remaining = jnp.maximum(raw_remaining, 0.0)
    positive_target = target > 0.0
    condensed_fraction = jnp.where(
        positive_target,
        condensed_budget / jnp.maximum(target, 1.0e-300),
        0.0,
    )
    diagnostics = {
        "policy": "depleted_budget",
        "clipped_negative_l1": float(jnp.sum(jnp.maximum(-raw_remaining, 0.0))),
        "max_condensed_fraction": float(
            jnp.max(condensed_fraction, initial=jnp.asarray(0.0, dtype=jnp.float64))
        ),
    }
    return remaining, diagnostics


def _default_fixed_support_solver_log_state(
    *,
    setup: CondensateChemicalSetup,
    T: float,
    P: float,
    b: Array,
    Pref: float,
    support_indices: Sequence[int],
    support_amounts_init: Sequence[float],
    options: CondensateEquilibriumOptions,
    source: str,
) -> Any:
    from exogibbs.api.equilibrium import EquilibriumOptions, equilibrium
    from exogibbs.optimize.minimize_cond import CondensateEquilibriumInit as SolverInit

    gas_budget, budget_trace = _fixed_support_gas_budget_for_init(
        setup=setup,
        b=b,
        support_indices=support_indices,
        support_amounts=support_amounts_init,
        policy=options.fixed_support_gas_init_policy,
    )
    gas_result = equilibrium(
        setup.gas_setup,
        T,
        P,
        gas_budget,
        Pref=Pref,
        options=EquilibriumOptions(),
        return_diagnostics=False,
    )
    return SolverInit(
        ln_nk=jnp.asarray(gas_result.ln_n, dtype=jnp.float64),
        ln_mk=jnp.log(jnp.maximum(jnp.asarray(support_amounts_init), 1.0e-300)),
        ln_ntot=jnp.log(jnp.asarray(gas_result.ntot, dtype=jnp.float64)),
        ln_nk_source_trace={
            "source": source,
            "fixed_support_gas_init_policy": options.fixed_support_gas_init_policy,
            "gas_budget": budget_trace,
            "reason": (
                "Initialize fixed-support gas on the policy-selected element "
                "budget before the restricted condensate solve."
            ),
        },
    )


def _lifecycle_final_state_support_growth_payload(
    result: CondensateEquilibriumResult,
    *,
    enabled: bool,
) -> Mapping[str, Any] | None:
    """Return a finite lifecycle-final-state payload for support growth."""

    if not enabled:
        return None
    diagnostics = result.diagnostics or {}
    lifecycle_payload = diagnostics.get("head_route_lifecycle")
    if not isinstance(lifecycle_payload, Mapping):
        return None
    final_state_payload = _lifecycle_final_state_payload(lifecycle_payload)
    if not isinstance(final_state_payload, Mapping):
        return None
    try:
        support_indices = _final_state_support_indices_from_lifecycle_payload(
            lifecycle_payload,
            fallback_support_indices=tuple(
                int(index) for index in result.condensate_support_indices.tolist()
            ),
        )
        ln_nk = jnp.asarray(final_state_payload["ln_nk"], dtype=jnp.float64)
        ln_mk = jnp.asarray(final_state_payload["ln_mk"], dtype=jnp.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if (
        ln_nk.ndim != 1
        or ln_mk.ndim != 1
        or ln_mk.shape[0] != len(support_indices)
        or not bool(jnp.all(jnp.isfinite(ln_nk)))
        or not bool(jnp.all(jnp.isfinite(ln_mk)))
    ):
        return None
    m_support = jnp.exp(ln_mk)
    if not bool(jnp.all(jnp.isfinite(m_support))):
        return None
    return {
        "ln_nk": ln_nk,
        "support_indices": support_indices,
        "m_support": m_support,
        "pi_vector": None,
        "state_source": "lifecycle_final_state",
    }


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


def _with_support_closure_retry_selection_diagnostics(
    *,
    result: CondensateEquilibriumResult,
    selected_retry_key: str,
    retry_report: Mapping[str, Any],
    selection_report: Mapping[str, Any],
    return_diagnostics: bool,
) -> CondensateEquilibriumResult:
    if not return_diagnostics:
        return result
    diagnostics = dict(result.diagnostics or {})
    diagnostics[selected_retry_key] = retry_report
    diagnostics["support_closure_retry_selection"] = selection_report
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


def _with_inactive_condensate_driving_diagnostics(
    *,
    result: CondensateEquilibriumResult,
    setup: CondensateChemicalSetup,
    T: float,
    P: float,
    Pref: float,
    options: CondensateEquilibriumOptions,
) -> CondensateEquilibriumResult:
    if not options.return_diagnostics:
        return result
    from exogibbs.condensates.inactive_driving import (
        evaluate_inactive_condensate_driving,
    )

    gas_stationarity_source = setup.gas_setup.hvector_func(float(T)) + (
        _ln_normalized_pressure(P, Pref)
    )
    element_potential = _least_squares_element_potential(
        formula_matrix=setup.formula_matrix,
        gas_ln_n=result.gas_ln_n,
        gas_stationarity_source=gas_stationarity_source,
    )
    report = evaluate_inactive_condensate_driving(
        formula_matrix_cond=setup.formula_matrix_cond,
        condensate_species_order=setup.condensate_species,
        condensate_amounts=result.condensate_amounts,
        hvector_cond=setup.condensate_setup.hvector_func(float(T)),
        element_potential=element_potential,
        temperature=float(T),
        condensate_temperature_validity_upper=setup.condensate_setup.metadata.get(
            "temperature_validity_upper"
        ),
        active_floor=1.0e-50,
        activity_threshold=options.support_activity_threshold,
    )
    diagnostics = dict(result.diagnostics or {})
    diagnostics["inactive_condensate_driving"] = report.as_dict()
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
        try:
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
        except RuntimeError as exc:
            if (
                last_result is not None
                and "No finite condensate HEAD route warm-start candidate" in str(exc)
            ):
                terminated_reason = "support_growth_stopped_after_nonfinite_warm_start"
                outer_iterations.append(
                    {
                        "iteration": outer_index,
                        "state_source": "head_route_result",
                        "selected_route": last_result.selected_route,
                        "result_status": last_result.status,
                        "added_support_indices": (),
                        "added_support_names": (),
                        "reason": (
                            "Stop support growth after the next candidate support "
                            "produces no finite warm-start; keep the last accepted "
                            "HEAD route result."
                        ),
                    }
                )
                break
            raise
        fallback_solver_payload = None
        if last_result.selected_route == "native_budget_seed_fallback_budget_tradeoff":
            fallback_solver_payload = (last_result.diagnostics or {}).get(
                "restricted_solver_payload_for_support_growth"
            )
            if not fallback_solver_payload:
                fallback_solver_payload = _lifecycle_final_state_support_growth_payload(
                    last_result,
                    enabled=options.enable_lifecycle_final_state_support_growth,
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
                "state_source": str(
                    fallback_solver_payload.get("state_source", "restricted_solver_output")
                )
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
    retry_candidates: list[_SupportClosureRetryCandidate] = []
    support_cap_retry_attempts: list[Mapping[str, Any]] = []
    support_growth_staging_retry_attempts: list[Mapping[str, Any]] = []
    scalar_step_control_retry_attempts: list[Mapping[str, Any]] = []
    lifecycle_support_closure_retry_attempts: list[Mapping[str, Any]] = []
    retry_caps = _support_cap_retry_sequence(options)
    initial_support_closure_gate = None
    retry_triggered = (
        last_result.selected_route == "native_budget_seed_fallback_budget_tradeoff"
        and options.max_positive_support_count is None
    )
    if (
        not retry_triggered
        and last_result.converged
        and options.max_positive_support_count is None
        and options.max_support_add_per_round is None
    ):
        initial_support_closure_gate = _support_closure_retry_gate_report(
            setup=setup,
            T=T,
            P=P,
            b=b,
            Pref=Pref,
            result=last_result,
            options=options,
        )
        retry_triggered = (
            not bool(initial_support_closure_gate.get("accepted", False))
            and float(
                initial_support_closure_gate.get(
                    "max_positive_inactive_driving",
                    0.0,
                )
            )
            >= 1.0e3
            and int(initial_support_closure_gate.get("positive_inactive_count", 0)) >= 50
        )
    if retry_triggered and options.enable_support_cap_retry and retry_caps:
        for retry_cap in retry_caps:
            retry_options = replace(
                options,
                case_id=None
                if options.case_id is None
                else f"{options.case_id}__support_cap_retry_{retry_cap}",
                enable_support_cap_retry=False,
                enable_head_route_scalar_step_control_retry=False,
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
                support_cap_retry_attempts.append(
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
            support_cap_retry_attempts.append(retry_attempt)
            if retry_route_promoted and retry_accepted and retry_support_closure_accepted:
                retry_candidates.append(
                    _SupportClosureRetryCandidate(
                        retry_kind="support_cap_retry",
                        result=retry_result,
                        support_closure_gate=support_closure_gate,
                        attempt=retry_attempt,
                        score=_support_closure_retry_candidate_score(
                            support_closure_gate,
                            support_count=len(
                                tuple(retry_result.condensate_support_names)
                            ),
                        ),
                    )
                )
    staged_retry_counts = _support_growth_staging_retry_sequence(options)
    if (
        retry_triggered
        and options.enable_support_growth_staging_retry
        and options.max_support_add_per_round is None
        and staged_retry_counts
    ):
        for add_per_round in staged_retry_counts:
            retry_options = replace(
                options,
                case_id=None
                if options.case_id is None
                else f"{options.case_id}__support_growth_staging_retry_{add_per_round}",
                enable_support_cap_retry=False,
                enable_support_growth_staging_retry=False,
                enable_head_route_scalar_step_control_retry=False,
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
                support_growth_staging_retry_attempts.append(
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
            support_growth_staging_retry_attempts.append(retry_attempt)
            if retry_route_promoted and retry_accepted and retry_support_closure_accepted:
                retry_candidates.append(
                    _SupportClosureRetryCandidate(
                        retry_kind="support_growth_staging_retry",
                        result=retry_result,
                        support_closure_gate=support_closure_gate,
                        attempt=retry_attempt,
                        score=_support_closure_retry_candidate_score(
                            support_closure_gate,
                            support_count=len(
                                tuple(retry_result.condensate_support_names)
                            ),
                        ),
                    )
                )
    if (
        retry_triggered
        and options.enable_head_route_scalar_step_control_retry
        and options.head_route_primary_step_control_policy != "scalar_fraction_to_boundary"
    ):
        retry_options = replace(
            options,
            case_id=None
            if options.case_id is None
            else f"{options.case_id}__scalar_step_control_retry",
            enable_support_cap_retry=False,
            enable_support_growth_staging_retry=False,
            enable_head_route_scalar_step_control_retry=False,
            head_route_primary_step_control_policy="scalar_fraction_to_boundary",
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
            scalar_step_control_retry_attempts.append(
                {
                    "step_control_policy": "scalar_fraction_to_boundary",
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
        else:
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
                "step_control_policy": "scalar_fraction_to_boundary",
                "fraction_to_boundary_safety": float(
                    options.head_route_primary_fraction_to_boundary_safety
                ),
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
            scalar_step_control_retry_attempts.append(retry_attempt)
            if retry_route_promoted and retry_accepted and retry_support_closure_accepted:
                retry_candidates.append(
                    _SupportClosureRetryCandidate(
                        retry_kind="scalar_step_control_retry",
                        result=retry_result,
                        support_closure_gate=support_closure_gate,
                        attempt=retry_attempt,
                        score=_support_closure_retry_candidate_score(
                            support_closure_gate,
                            support_count=len(
                                tuple(retry_result.condensate_support_names)
                            ),
                        ),
                    )
                )
    if (
        retry_triggered
        and options.enable_lifecycle_final_state_support_growth
        and terminated_reason == "max_support_outer_iterations_reached"
    ):
        lifecycle_payload = _lifecycle_final_state_support_growth_payload(
            last_result,
            enabled=options.enable_lifecycle_final_state_support_growth,
        )
        lifecycle_attempt: dict[str, Any] = {
            "state_source": "lifecycle_final_state",
            "selected_route": "not_attempted",
            "status": "not_attempted",
            "support_count": len(tuple(current_support)),
            "added_support_count": 0,
            "accepted": False,
            "route_promoted": False,
            "support_closure_accepted": False,
        }
        if lifecycle_payload is None:
            lifecycle_attempt["reason"] = "missing_lifecycle_final_state_payload"
            lifecycle_support_closure_retry_attempts.append(lifecycle_attempt)
        else:
            try:
                lifecycle_existing = tuple(
                    int(index) for index in lifecycle_payload["support_indices"]
                )
                lifecycle_amounts = _positive_support_amounts_for_warm_start(
                    tuple(float(value) for value in lifecycle_payload["m_support"]),
                    min_seed_amount=options.min_seed_amount,
                )
                lifecycle_report = _activity_driven_support_report(
                    setup=setup,
                    T=T,
                    P=P,
                    b=b,
                    Pref=Pref,
                    gas_ln_n=lifecycle_payload["ln_nk"],
                    options=options,
                    existing_support_indices=lifecycle_existing,
                    element_potential_override=lifecycle_payload.get("pi_vector"),
                )
                lifecycle_existing_set = set(lifecycle_existing)
                lifecycle_inactive_positive = tuple(
                    int(index)
                    for index in lifecycle_report["inactive_positive_indices"]
                    if int(index) not in lifecycle_existing_set
                )
                lifecycle_add_count = _support_add_count(
                    inactive_count=len(lifecycle_inactive_positive),
                    support_count=len(lifecycle_existing),
                    options=options,
                )
                lifecycle_added = lifecycle_inactive_positive[:lifecycle_add_count]
                lifecycle_attempt.update(
                    {
                        "support_count": len(lifecycle_existing),
                        "inactive_positive_count": len(lifecycle_inactive_positive),
                        "added_support_count": len(lifecycle_added),
                        "added_support_indices": lifecycle_added,
                        "added_support_names": tuple(
                            setup.condensate_species[int(index)]
                            for index in lifecycle_added
                        ),
                    }
                )
                if not lifecycle_added:
                    lifecycle_attempt["reason"] = "no_lifecycle_inactive_positive_support"
                    lifecycle_support_closure_retry_attempts.append(lifecycle_attempt)
                else:
                    lifecycle_added_amounts = _budget_seed_for_support(
                        setup=setup,
                        b=b,
                        support_indices=lifecycle_added,
                        options=options,
                    )
                    retry_options = replace(
                        options,
                        case_id=None
                        if options.case_id is None
                        else (
                            f"{options.case_id}"
                            "__lifecycle_final_state_support_closure_retry"
                        ),
                        enable_support_outer_loop=False,
                        enable_support_cap_retry=False,
                        enable_support_growth_staging_retry=False,
                        enable_head_route_scalar_step_control_retry=False,
                        enable_head_route_center_gate_retry=True,
                        enable_head_route_residual_worsening_retry=True,
                        enable_head_route_soft_restoration_retry=True,
                        enable_head_route_ipopt_h_type_retry=True,
                    )
                    retry_result = condensate_equilibrium(
                        setup,
                        T,
                        P,
                        b,
                        Pref=Pref,
                        support_indices=lifecycle_existing + lifecycle_added,
                        support_amounts_init=(
                            lifecycle_amounts + lifecycle_added_amounts
                        ),
                        options=retry_options,
                    )
                    retry_route_promoted = (
                        retry_result.selected_route
                        != "native_budget_seed_fallback_budget_tradeoff"
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
                    lifecycle_attempt.update(
                        {
                            "selected_route": retry_result.selected_route,
                            "status": retry_result.status,
                            "support_count": len(
                                tuple(retry_result.condensate_support_names)
                            ),
                            "accepted": bool(retry_accepted),
                            "route_promoted": bool(retry_route_promoted),
                            "support_closure_gate": support_closure_gate,
                            "support_closure_accepted": (
                                retry_support_closure_accepted
                            ),
                        }
                    )
                    lifecycle_support_closure_retry_attempts.append(
                        lifecycle_attempt
                    )
                    if (
                        retry_route_promoted
                        and retry_accepted
                        and retry_support_closure_accepted
                    ):
                        retry_candidates.append(
                            _SupportClosureRetryCandidate(
                                retry_kind=(
                                    "lifecycle_final_state_support_closure_retry"
                                ),
                                result=retry_result,
                                support_closure_gate=support_closure_gate,
                                attempt=lifecycle_attempt,
                                score=_support_closure_retry_candidate_score(
                                    support_closure_gate,
                                    support_count=len(
                                        tuple(retry_result.condensate_support_names)
                                    ),
                                ),
                            )
                        )
            except Exception as exc:  # noqa: BLE001 - retry candidates are optional.
                lifecycle_attempt.update(
                    {
                        "selected_route": "exception",
                        "status": "exception",
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    }
                )
                lifecycle_support_closure_retry_attempts.append(lifecycle_attempt)
    if retry_candidates:
        best_retry = min(retry_candidates, key=lambda candidate: candidate.score)
        selected_attempt = dict(best_retry.attempt)
        selection_report = {
            "selection_schema": "exogibbs_support_free_cross_retry_selection_v1",
            "triggered": True,
            "accepted": True,
            "selection_policy": "best_support_closure_score_across_retry_kinds",
            "selected_retry_kind": best_retry.retry_kind,
            "support_closure_score": best_retry.score,
            "candidate_count": len(retry_candidates),
            "support_cap_retry_attempts": tuple(support_cap_retry_attempts),
            "support_growth_staging_retry_attempts": tuple(
                support_growth_staging_retry_attempts
            ),
            "scalar_step_control_retry_attempts": tuple(
                scalar_step_control_retry_attempts
            ),
            "lifecycle_final_state_support_closure_retry_attempts": tuple(
                lifecycle_support_closure_retry_attempts
            ),
            "initial_selected_route": last_result.selected_route,
            "initial_status": last_result.status,
            "initial_support_count": len(tuple(current_support)),
            "initial_support_outer_terminated_reason": terminated_reason,
            "initial_support_closure_gate": initial_support_closure_gate,
            "retry_selected_route": best_retry.result.selected_route,
            "retry_status": best_retry.result.status,
            "retry_support_count": len(tuple(best_retry.result.condensate_support_names)),
            "retry_support_closure_gate": best_retry.support_closure_gate,
            "fastchem4_trace_public_runtime_constructor_inputs_used": False,
        }
        if best_retry.retry_kind == "support_cap_retry":
            retry_report = {
                "retry_schema": "exogibbs_support_free_support_cap_retry_v1",
                "triggered": True,
                "accepted": True,
                "route_promoted": True,
                "support_closure_accepted": True,
                "selection_policy": "best_support_closure_score_across_retry_kinds",
                "support_closure_score": best_retry.score,
                "support_cap": int(selected_attempt.get("support_cap", 0)),
                "support_cap_sequence": tuple(int(cap) for cap in retry_caps),
                "attempts": tuple(support_cap_retry_attempts),
                "initial_selected_route": last_result.selected_route,
                "initial_status": last_result.status,
                "initial_support_count": len(tuple(current_support)),
                "initial_support_closure_gate": initial_support_closure_gate,
                "retry_selected_route": best_retry.result.selected_route,
                "retry_status": best_retry.result.status,
                "retry_support_count": len(
                    tuple(best_retry.result.condensate_support_names)
                ),
                "retry_support_closure_gate": best_retry.support_closure_gate,
                "fastchem4_trace_public_runtime_constructor_inputs_used": False,
            }
            return _with_support_closure_retry_selection_diagnostics(
                result=best_retry.result,
                selected_retry_key="support_cap_retry",
                retry_report=retry_report,
                selection_report=selection_report,
                return_diagnostics=options.return_diagnostics,
            )
        if best_retry.retry_kind == "support_growth_staging_retry":
            retry_report = {
                "retry_schema": "exogibbs_support_free_support_growth_staging_retry_v1",
                "triggered": True,
                "accepted": True,
                "route_promoted": True,
                "support_closure_accepted": True,
                "selection_policy": "best_support_closure_score_across_retry_kinds",
                "support_closure_score": best_retry.score,
                "max_support_add_per_round": int(
                    selected_attempt.get("max_support_add_per_round", 0)
                ),
                "max_support_add_per_round_sequence": tuple(
                    int(count) for count in staged_retry_counts
                ),
                "attempts": tuple(support_growth_staging_retry_attempts),
                "initial_selected_route": last_result.selected_route,
                "initial_status": last_result.status,
                "initial_support_count": len(tuple(current_support)),
                "initial_support_outer_terminated_reason": terminated_reason,
                "initial_support_closure_gate": initial_support_closure_gate,
                "retry_selected_route": best_retry.result.selected_route,
                "retry_status": best_retry.result.status,
                "retry_support_count": len(tuple(best_retry.result.condensate_support_names)),
                "retry_support_closure_gate": best_retry.support_closure_gate,
                "fastchem4_trace_public_runtime_constructor_inputs_used": False,
            }
            return _with_support_closure_retry_selection_diagnostics(
                result=best_retry.result,
                selected_retry_key="support_growth_staging_retry",
                retry_report=retry_report,
                selection_report=selection_report,
                return_diagnostics=options.return_diagnostics,
            )
        if best_retry.retry_kind == "lifecycle_final_state_support_closure_retry":
            retry_report = {
                "retry_schema": (
                    "exogibbs_lifecycle_final_state_support_closure_retry_v1"
                ),
                "triggered": True,
                "accepted": True,
                "route_promoted": True,
                "support_closure_accepted": True,
                "selection_policy": "best_support_closure_score_across_retry_kinds",
                "support_closure_score": best_retry.score,
                "attempts": tuple(lifecycle_support_closure_retry_attempts),
                "initial_selected_route": last_result.selected_route,
                "initial_status": last_result.status,
                "initial_support_count": len(tuple(current_support)),
                "initial_support_outer_terminated_reason": terminated_reason,
                "initial_support_closure_gate": initial_support_closure_gate,
                "retry_selected_route": best_retry.result.selected_route,
                "retry_status": best_retry.result.status,
                "retry_support_count": len(
                    tuple(best_retry.result.condensate_support_names)
                ),
                "retry_support_closure_gate": best_retry.support_closure_gate,
                "fastchem4_trace_public_runtime_constructor_inputs_used": False,
            }
            return _with_support_closure_retry_selection_diagnostics(
                result=best_retry.result,
                selected_retry_key="lifecycle_final_state_support_closure_retry",
                retry_report=retry_report,
                selection_report=selection_report,
                return_diagnostics=options.return_diagnostics,
            )
        retry_report = {
            "retry_schema": "exogibbs_support_free_scalar_step_control_retry_v1",
            "triggered": True,
            "accepted": True,
            "route_promoted": True,
            "support_closure_accepted": True,
            "selection_policy": "best_support_closure_score_across_retry_kinds",
            "support_closure_score": best_retry.score,
            "step_control_policy": str(
                selected_attempt.get(
                    "step_control_policy", "scalar_fraction_to_boundary"
                )
            ),
            "fraction_to_boundary_safety": float(
                selected_attempt.get(
                    "fraction_to_boundary_safety",
                    options.head_route_primary_fraction_to_boundary_safety,
                )
            ),
            "attempts": tuple(scalar_step_control_retry_attempts),
            "initial_selected_route": last_result.selected_route,
            "initial_status": last_result.status,
            "initial_support_count": len(tuple(current_support)),
            "initial_support_outer_terminated_reason": terminated_reason,
            "initial_support_closure_gate": initial_support_closure_gate,
            "retry_selected_route": best_retry.result.selected_route,
            "retry_status": best_retry.result.status,
            "retry_support_count": len(tuple(best_retry.result.condensate_support_names)),
            "retry_support_closure_gate": best_retry.support_closure_gate,
            "fastchem4_trace_public_runtime_constructor_inputs_used": False,
        }
        return _with_support_closure_retry_selection_diagnostics(
            result=best_retry.result,
            selected_retry_key="scalar_step_control_retry",
            retry_report=retry_report,
            selection_report=selection_report,
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
            enable_head_route_scalar_step_control_retry=False,
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
    init: Optional[CondensateEquilibriumInit] = None,
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
        return _with_inactive_condensate_driving_diagnostics(
            result=_run_activity_driven_support_outer_loop(
                setup=setup,
                T=T,
                P=P,
                b=b,
                Pref=Pref,
                options=opts,
            ),
            setup=setup,
            T=T,
            P=P,
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
        empty_result = _build_empty_support_gas_result(
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
        gate = (empty_result.diagnostics or {}).get(
            "full_condensate_budget_residual_gate",
            {},
        )
        if (
            opts.enable_full_condensate_budget_residual_gate
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
                    opts.enable_full_condensate_budget_residual_gate
                ),
                full_condensate_budget_relative_tolerance=(
                    opts.full_condensate_budget_relative_tolerance
                ),
            )
            if opts.return_diagnostics:
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
                empty_result = strict_result
        return _with_inactive_condensate_driving_diagnostics(
            result=empty_result,
            setup=setup,
            T=T,
            P=P,
            Pref=Pref,
            options=opts,
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
    profile_initial_log_state = _solver_log_state_from_condensate_init(
        init,
        setup=setup,
        support_amounts_init=support_amounts_init,
        source="exogibbs_profile_or_caller_condensate_init",
    )
    if profile_initial_log_state is None:
        baseline_initial_log_state = _default_fixed_support_solver_log_state(
            setup=setup,
            T=T,
            P=P,
            b=b,
            Pref=Pref,
            support_indices=support_indices,
            support_amounts_init=support_amounts_init,
            options=opts,
            source="exogibbs_api_fixed_support_gas_init",
        )
    else:
        baseline_initial_log_state = profile_initial_log_state
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
            initial_soft_restoration_stopped_reason = (
                _continuation_stopped_reason_from_lifecycle_payload(
                    lifecycle_payload,
                )
            )
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
                "trigger_mode": "manual_option",
                "initial_stopped_reason": initial_soft_restoration_stopped_reason,
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
                budget_correction_direction_policy = (
                    "joint_budget_amount_gas_condensate_linearized_no_prior"
                )
                budget_correction_policy = {
                    **primary_policy,
                    "direction_policy": budget_correction_direction_policy,
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
                retry_gas_polish_report = None
                if (
                    retry_gate_report is not None
                    and not bool(retry_gate_report["accepted"])
                    and isinstance(retry_final_state_payload, Mapping)
                    and isinstance(retry_primary_payload, Mapping)
                    and isinstance(retry_continuation_payload, Mapping)
                ):
                    try:
                        retry_support_amounts = jnp.exp(
                            jnp.asarray(retry_final_state_payload["ln_mk"])
                        )
                        retry_full_amounts = _full_condensate_amounts(
                            support_indices=retry_final_state_support_indices,
                            support_amounts=retry_support_amounts,
                            condensate_count=len(setup.condensate_species),
                        )
                        retry_full_amounts = _merge_external_condensate_amounts(
                            condensate_amounts=retry_full_amounts,
                            external_condensate_amounts=retry_external_amounts,
                        )
                        polished_retry_ln_n, retry_gas_polish_report = (
                            _polish_gas_log_amounts_for_full_condensate_budget_gate(
                                setup=setup,
                                gas_ln_n=jnp.asarray(
                                    retry_final_state_payload["ln_nk"]
                                ),
                                condensate_amounts=retry_full_amounts,
                                element_inventory_target=b,
                                relative_tolerance=(
                                    opts.full_condensate_budget_relative_tolerance
                                ),
                            )
                        )
                    except (KeyError, TypeError, ValueError):
                        retry_gas_polish_report = None
                    if (
                        retry_gas_polish_report is not None
                        and retry_gas_polish_report["accepted"]
                    ):
                        retry_gate_report = retry_gas_polish_report[
                            "final_full_condensate_budget_gate"
                        ]
                        retry_final_state_payload = {
                            **dict(retry_final_state_payload),
                            "ln_nk": tuple(
                                float(value)
                                for value in jnp.asarray(
                                    polished_retry_ln_n,
                                    dtype=jnp.float64,
                                ).tolist()
                            ),
                        }
                        retry_continuation_payload = {
                            **dict(retry_continuation_payload),
                            "final_state": retry_final_state_payload,
                        }
                        retry_primary_payload = {
                            **dict(retry_primary_payload),
                            "continuation_report": retry_continuation_payload,
                        }
                        budget_correction_payload = {
                            **dict(budget_correction_payload),
                            "primary_execution_report": retry_primary_payload,
                        }
                budget_correction_accepted = bool(
                    retry_gate_report is not None and retry_gate_report["accepted"]
                )
                condensate_budget_correction_retry_report = {
                    "retry_schema": (
                        "exogibbs_head_route_condensate_budget_correction_retry_v1"
                    ),
                    "triggered": True,
                    "accepted": budget_correction_accepted,
                    "direction_policy": budget_correction_direction_policy,
                    "budget_row_scaling_policy": "relative_target",
                    "trial_acceptance_policy": "ipopt_persistent_h_type",
                    "initial_full_condensate_budget_gate": initial_gate_report,
                    "retry_full_condensate_budget_gate": retry_gate_report,
                    "retry_gas_log_amount_polish": retry_gas_polish_report,
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
                    lifecycle_selected_route = (
                        budget_correction_lifecycle_report.route_result.selected_route
                    )
                    lifecycle_metric_status = (
                        budget_correction_lifecycle_report.route_result.metric_status
                    )
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
            return _with_inactive_condensate_driving_diagnostics(
                result=_build_native_seed_fallback_result(
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
                ),
                setup=setup,
                T=T,
                P=P,
                Pref=Pref,
                options=opts,
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
        return _with_inactive_condensate_driving_diagnostics(
            result=_build_native_seed_fallback_result(
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
            ),
            setup=setup,
            T=T,
            P=P,
            Pref=Pref,
            options=opts,
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
    final_result = _with_inactive_condensate_driving_diagnostics(
        result=build_condensate_equilibrium_result_from_solver_payload(
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
        ),
        setup=setup,
        T=T,
        P=P,
        Pref=Pref,
        options=opts,
    )
    if (
        support_selection_report is not None
        and support_selection_report.get("selection_mode") == "explicit_support_payload"
        and opts.enable_lifecycle_final_state_support_growth
        and opts.metric_status is None
        and opts.head_route_primary_summary is None
        and opts.head_route_refresh_policy_summary is None
    ):
        support_closure_gate = _support_closure_retry_gate_report(
            setup=setup,
            T=T,
            P=P,
            b=b,
            Pref=Pref,
            result=final_result,
            options=opts,
        )
        full_budget_gate = (final_result.diagnostics or {}).get(
            "full_condensate_budget_residual_gate",
            {},
        )
        retry_triggered = (
            not bool(final_result.converged)
            or not bool(support_closure_gate.get("accepted", False))
            or (
                isinstance(full_budget_gate, Mapping)
                and not bool(full_budget_gate.get("accepted", True))
            )
        )
        if retry_triggered:
            try:
                activity_report = _activity_driven_support_report(
                    setup=setup,
                    T=T,
                    P=P,
                    b=b,
                    Pref=Pref,
                    gas_ln_n=result_ln_nk,
                    options=opts,
                    existing_support_indices=tuple(
                        int(index) for index in result_support_indices
                    ),
                )
                existing = {int(index) for index in result_support_indices}
                inactive_positive = tuple(
                    int(index)
                    for index in activity_report["inactive_positive_indices"]
                    if int(index) not in existing
                )
                add_count = _support_add_count(
                    inactive_count=len(inactive_positive),
                    support_count=len(tuple(result_support_indices)),
                    options=opts,
                )
                added = inactive_positive[:add_count]
                retry_report: Mapping[str, Any] = {
                    "retry_schema": (
                        "exogibbs_explicit_support_lifecycle_closure_retry_v1"
                    ),
                    "triggered": True,
                    "accepted": False,
                    "initial_selected_route": final_result.selected_route,
                    "initial_status": final_result.status,
                    "initial_support_count": len(tuple(result_support_indices)),
                    "initial_support_closure_gate": support_closure_gate,
                    "initial_full_condensate_budget_gate": full_budget_gate,
                    "inactive_positive_count": len(inactive_positive),
                    "added_support_indices": added,
                    "added_support_names": tuple(
                        setup.condensate_species[int(index)] for index in added
                    ),
                    "fastchem4_trace_public_runtime_constructor_inputs_used": False,
                }
                if added:
                    retry_support_indices = (
                        tuple(int(index) for index in result_support_indices) + added
                    )
                    retry_support_amounts = (
                        _positive_support_amounts_for_warm_start(
                            tuple(
                                float(value)
                                for value in jnp.asarray(result_support_amounts).tolist()
                            ),
                            min_seed_amount=opts.min_seed_amount,
                        )
                        + _budget_seed_for_support(
                            setup=setup,
                            b=b,
                            support_indices=added,
                            options=opts,
                        )
                    )
                    retry_options = replace(
                        opts,
                        case_id=None
                        if opts.case_id is None
                        else f"{opts.case_id}__explicit_support_closure_retry",
                        enable_support_outer_loop=False,
                        enable_lifecycle_final_state_support_growth=False,
                        enable_support_cap_retry=False,
                        enable_support_growth_staging_retry=False,
                        enable_head_route_scalar_step_control_retry=False,
                    )
                    retry_result = condensate_equilibrium(
                        setup,
                        T,
                        P,
                        b,
                        Pref=Pref,
                        support_indices=retry_support_indices,
                        support_amounts_init=retry_support_amounts,
                        options=retry_options,
                    )
                    retry_route_promoted = (
                        retry_result.selected_route
                        != "native_budget_seed_fallback_budget_tradeoff"
                    )
                    retry_gate = _support_closure_retry_gate_report(
                        setup=setup,
                        T=T,
                        P=P,
                        b=b,
                        Pref=Pref,
                        result=retry_result,
                        options=opts,
                    )
                    retry_full_budget_gate = (retry_result.diagnostics or {}).get(
                        "full_condensate_budget_residual_gate",
                        {},
                    )
                    retry_accepted = bool(
                        retry_route_promoted
                        and retry_result.converged
                        and retry_gate.get("accepted", False)
                        and (
                            not isinstance(retry_full_budget_gate, Mapping)
                            or retry_full_budget_gate.get("accepted", True)
                        )
                    )
                    retry_report = {
                        **dict(retry_report),
                        "accepted": retry_accepted,
                        "route_promoted": bool(retry_route_promoted),
                        "support_closure_accepted": bool(
                            retry_gate.get("accepted", False)
                        ),
                        "retry_selected_route": retry_result.selected_route,
                        "retry_status": retry_result.status,
                        "retry_support_count": len(
                            tuple(retry_result.condensate_support_names)
                        ),
                        "retry_support_closure_gate": retry_gate,
                        "retry_full_condensate_budget_gate": retry_full_budget_gate,
                    }
                    if retry_accepted:
                        return _with_support_closure_retry_selection_diagnostics(
                            result=retry_result,
                            selected_retry_key="explicit_support_closure_retry",
                            retry_report=retry_report,
                            selection_report={
                                "selection_schema": (
                                    "exogibbs_explicit_support_closure_retry_selection_v1"
                                ),
                                "triggered": True,
                                "accepted": True,
                                "selection_policy": (
                                    "single_lifecycle_final_state_support_closure_retry"
                                ),
                                "selected_retry_kind": (
                                    "explicit_support_closure_retry"
                                ),
                                "fastchem4_trace_public_runtime_constructor_inputs_used": (
                                    False
                                ),
                            },
                            return_diagnostics=opts.return_diagnostics,
                        )
                if opts.return_diagnostics:
                    diagnostics = dict(final_result.diagnostics or {})
                    diagnostics["explicit_support_closure_retry"] = retry_report
                    final_result = replace(final_result, diagnostics=diagnostics)
            except Exception as exc:  # noqa: BLE001 - optional retry diagnostics.
                if opts.return_diagnostics:
                    diagnostics = dict(final_result.diagnostics or {})
                    diagnostics["explicit_support_closure_retry"] = {
                        "retry_schema": (
                            "exogibbs_explicit_support_lifecycle_closure_retry_v1"
                        ),
                        "triggered": True,
                        "accepted": False,
                        "status": "exception",
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "fastchem4_trace_public_runtime_constructor_inputs_used": False,
                    }
                    final_result = replace(final_result, diagnostics=diagnostics)
    return final_result


def _resolve_condensate_profile_method(
    method: Optional[CondensateProfileMethod],
    initializer: Optional[CondensateEquilibriumInitializer],
    *,
    has_fixed_support_payload: bool = False,
) -> CondensateProfileMethod:
    if method is not None and method != "auto":
        return method
    if has_fixed_support_payload or initializer is not None:
        return "vmap_cold"
    return "scan_hot_from_top"


def _profile_has_complete_fixed_support_payload(
    explicit_inits: Sequence[CondensateEquilibriumInit | None],
    *,
    support_indices: Optional[Sequence[int]],
    support_amounts_init: Optional[Sequence[float]],
) -> bool:
    if support_indices is not None:
        return support_amounts_init is not None
    if not explicit_inits:
        return False
    return all(
        init is not None
        and init.gas_ln_n is not None
        and (
            (
                init.support_indices is not None
                and (
                    init.support_amounts is not None
                    or init.condensate_amounts is not None
                )
            )
            or init.condensate_amounts is not None
        )
        for init in explicit_inits
    )


def _resolve_condensate_initial_guess(
    initializer: Optional[CondensateEquilibriumInitializer],
    request: CondensateEquilibriumInitRequest,
) -> CondensateEquilibriumInit:
    active_initializer = initializer or _DEFAULT_CONDENSATE_INITIALIZER
    return active_initializer(request)


def _with_profile_layer_diagnostics(
    result: CondensateEquilibriumResult,
    *,
    profile_report: Mapping[str, Any],
    return_diagnostics: bool,
) -> CondensateEquilibriumResult:
    if not return_diagnostics:
        return result
    diagnostics = dict(result.diagnostics or {})
    diagnostics["condensate_profile_layer"] = dict(profile_report)
    return replace(result, diagnostics=diagnostics)


def _run_condensate_profile_layer(
    *,
    setup: CondensateChemicalSetup,
    T: float,
    P: float,
    b: Array,
    Pref: float,
    layer_index: int,
    base_options: CondensateEquilibriumOptions,
    init: CondensateEquilibriumInit,
    support_indices: Optional[Sequence[int]],
    support_amounts_init: Optional[Sequence[float]],
    warm_start_support_policy: CondensateProfileWarmStartSupportPolicy,
    return_diagnostics: bool,
) -> tuple[CondensateEquilibriumResult, Mapping[str, Any]]:
    use_explicit_support_for_warm_start = (
        warm_start_support_policy == "explicit_payload"
        and support_indices is not None
        and support_amounts_init is not None
        and init.gas_ln_n is not None
    )
    if use_explicit_support_for_warm_start:
        support_payload = (
            tuple(int(index) for index in support_indices),
            _positive_support_amounts_for_warm_start(
                support_amounts_init,
                min_seed_amount=base_options.min_seed_amount,
            ),
        )
    else:
        support_payload = _support_payload_from_condensate_init(
            init,
            setup=setup,
            min_seed_amount=base_options.min_seed_amount,
        )
    initialization_mode = "initializer"
    warm_start_attempted = support_payload is not None
    if use_explicit_support_for_warm_start:
        initialization_mode = "initializer_gas_with_explicit_support_payload"
    if support_payload is None and support_indices is not None:
        if support_amounts_init is None:
            raise ValueError(
                "support_amounts_init is required when support_indices is provided."
            )
        support_payload = (
            tuple(int(index) for index in support_indices),
            _positive_support_amounts_for_warm_start(
                support_amounts_init,
                min_seed_amount=base_options.min_seed_amount,
            ),
        )
        initialization_mode = "explicit_support_payload"
        warm_start_attempted = False
    case_id = (
        None
        if base_options.case_id is None
        else f"{base_options.case_id}__layer_{layer_index}"
    )
    layer_options = replace(
        base_options,
        case_id=case_id,
        return_diagnostics=return_diagnostics,
    )
    if support_payload is None:
        result = condensate_equilibrium(
            setup,
            T,
            P,
            b,
            Pref=Pref,
            options=layer_options,
        )
        report = {
            "profile_layer_schema": "exogibbs_condensate_profile_layer_v1",
            "layer_index": int(layer_index),
            "initialization_mode": "fresh",
            "warm_start_attempted": warm_start_attempted,
            "fresh_fallback_used": False,
            "accepted": bool(result.converged),
        }
        return _with_profile_layer_diagnostics(
            result,
            profile_report=report,
            return_diagnostics=return_diagnostics,
        ), report

    support_indices, support_amounts = support_payload
    warm_result = condensate_equilibrium(
        setup,
        T,
        P,
        b,
        Pref=Pref,
        support_indices=support_indices,
        support_amounts_init=support_amounts,
        init=init,
        options=layer_options,
    )
    if warm_result.converged:
        report = {
            "profile_layer_schema": "exogibbs_condensate_profile_layer_v1",
            "layer_index": int(layer_index),
            "initialization_mode": initialization_mode,
            "warm_start_attempted": warm_start_attempted,
            "fresh_fallback_used": False,
            "accepted": True,
            "support_count": len(support_indices),
        }
        return _with_profile_layer_diagnostics(
            warm_result,
            profile_report=report,
            return_diagnostics=return_diagnostics,
        ), report

    fresh_result = condensate_equilibrium(
        setup,
        T,
        P,
        b,
        Pref=Pref,
        options=layer_options,
    )
    report = {
        "profile_layer_schema": "exogibbs_condensate_profile_layer_v1",
        "layer_index": int(layer_index),
        "initialization_mode": f"{initialization_mode}_with_fresh_fallback",
        "warm_start_attempted": warm_start_attempted,
        "warm_start_status": warm_result.status,
        "fresh_fallback_used": True,
        "accepted": bool(fresh_result.converged),
        "support_count": len(support_indices),
    }
    return _with_profile_layer_diagnostics(
        fresh_result,
        profile_report=report,
        return_diagnostics=return_diagnostics,
    ), report


def _profile_result_from_fixed_support_batch_arrays(
    *,
    setup: CondensateChemicalSetup,
    arrays: Mapping[str, Any],
    b: Array,
    support_by_layer: Sequence[Sequence[int]],
    reports: Sequence[Mapping[str, Any]],
    max_iter: int,
    return_diagnostics: bool,
    opts: CondensateEquilibriumOptions,
    route_name: str,
) -> CondensateEquilibriumProfileResult | None:
    converged = np.asarray(jax.device_get(arrays["converged"]), dtype=bool)
    if converged.ndim != 1:
        raise ValueError("fixed-support profile arrays must be one-dimensional.")
    n_layers = int(converged.shape[0])
    if len(support_by_layer) != n_layers or len(reports) != n_layers:
        raise ValueError("fixed-support profile metadata must match layer count.")
    if not bool(np.all(converged)):
        return None

    gas_ln_n_batch = jnp.asarray(arrays["gas_ln_n"], dtype=jnp.float64)
    gas_n_batch = jnp.asarray(arrays["gas_n"], dtype=jnp.float64)
    gas_x_batch = jnp.asarray(arrays["gas_x"], dtype=jnp.float64)
    gas_ntot_batch = jnp.asarray(arrays["gas_ntot"], dtype=jnp.float64)
    condensate_amounts_batch = jnp.asarray(
        arrays["condensate_amounts"],
        dtype=jnp.float64,
    )
    n_iter = np.asarray(jax.device_get(arrays.get("n_iter", np.zeros(n_layers))))
    final_residual = np.asarray(
        jax.device_get(arrays.get("final_residual", np.zeros(n_layers))),
        dtype=np.float64,
    )
    fallback_rescue = arrays.get("fallback_rescue")
    fallback_replaced = None
    selected_candidate_label = None
    selected_support_count = None
    if isinstance(fallback_rescue, Mapping):
        if "replaced" in fallback_rescue:
            fallback_replaced = np.asarray(
                jax.device_get(fallback_rescue["replaced"]),
                dtype=bool,
            )
        selected_candidate_label = fallback_rescue.get("selected_candidate_label")
        selected_support_count = fallback_rescue.get("selected_support_count")
        if selected_support_count is not None:
            selected_support_count = np.asarray(
                jax.device_get(selected_support_count),
                dtype=np.int64,
            )

    def layer_selected_label(layer_index: int) -> str | None:
        if selected_candidate_label is None:
            return None
        value = selected_candidate_label[layer_index]
        return None if value is None else str(value)

    def layer_selected_support_count(layer_index: int) -> int | None:
        if selected_support_count is None or layer_selected_label(layer_index) is None:
            return None
        return int(selected_support_count[layer_index])

    batched_arrays = None
    if not return_diagnostics and not opts.enable_full_condensate_budget_residual_gate:
        batched_arrays = {
            "gas_ln_n": gas_ln_n_batch,
            "gas_n": gas_n_batch,
            "gas_x": gas_x_batch,
            "gas_ntot": gas_ntot_batch,
            "condensate_amounts": condensate_amounts_batch,
        }

    layer_results = []
    updated_reports = []
    for layer_index in range(n_layers):
        support_tuple = tuple(int(index) for index in support_by_layer[layer_index])
        support_index_array = jnp.asarray(support_tuple, dtype=jnp.int32)
        layer_report = {**dict(reports[layer_index]), "accepted": True}
        if fallback_replaced is not None:
            layer_report["fallback_rescue_replaced"] = bool(
                fallback_replaced[layer_index]
            )
            selected_label = layer_selected_label(layer_index)
            if selected_label is not None:
                layer_report["fallback_rescue_candidate"] = selected_label
                layer_report["fallback_rescue_support_count"] = (
                    layer_selected_support_count(layer_index)
                )
        if not return_diagnostics and not opts.enable_full_condensate_budget_residual_gate:
            result = CondensateEquilibriumResult(
                gas_ln_n=gas_ln_n_batch[layer_index],
                gas_n=gas_n_batch[layer_index],
                gas_x=gas_x_batch[layer_index],
                gas_ntot=gas_ntot_batch[layer_index],
                condensate_amounts=condensate_amounts_batch[layer_index],
                condensate_support_indices=support_index_array,
                condensate_support_names=tuple(
                    setup.condensate_species[int(index)] for index in support_tuple
                ),
                acceptance_tier="runtime_unclassified",
                selected_route=route_name,
                status=CONVERGED,
                converged=True,
                diagnostics=None,
            )
        else:
            diagnostics = {
                "experimental_profile_fixed_support_batch": {
                    "schema": "exogibbs_experimental_profile_fixed_support_batch_v1",
                    "enabled": True,
                    "layer_index": int(layer_index),
                    "solver_success": True,
                    "max_iter": int(max_iter),
                    "n_iter": int(n_iter[layer_index]),
                    "final_residual": float(final_residual[layer_index]),
                    "support_indices": support_tuple,
                    "fallback_rescue_replaced": bool(
                        False
                        if fallback_replaced is None
                        else fallback_replaced[layer_index]
                    ),
                    "fallback_rescue_candidate": layer_selected_label(layer_index),
                    "fallback_rescue_support_count": (
                        layer_selected_support_count(layer_index)
                    ),
                },
            }
            if isinstance(fallback_rescue, Mapping):
                diagnostics["experimental_profile_fixed_support_batch"][
                    "fallback_rescue"
                ] = fallback_rescue
            result = build_condensate_equilibrium_result_from_solver_payload(
                setup=setup,
                gas_ln_n=gas_ln_n_batch[layer_index],
                support_indices=support_tuple,
                support_amounts=condensate_amounts_batch[layer_index][
                    support_index_array
                ],
                selected_route=route_name,
                metric_status=None,
                solver_success=True,
                allow_caveat_tiers=opts.allow_caveat_tiers,
                diagnostics=diagnostics,
                element_inventory_target=b,
                enable_full_condensate_budget_residual_gate=(
                    opts.enable_full_condensate_budget_residual_gate
                ),
                full_condensate_budget_relative_tolerance=(
                    opts.full_condensate_budget_relative_tolerance
                ),
            )
        layer_report["accepted"] = bool(result.converged)
        updated_reports.append(layer_report)
        layer_results.append(
            _with_profile_layer_diagnostics(
                result,
                profile_report=layer_report,
                return_diagnostics=return_diagnostics,
            )
        )

    profile_diagnostics = None
    if return_diagnostics:
        profile_diagnostics = {
            "profile_schema": "exogibbs_condensate_equilibrium_profile_v1",
            "method": "vmap_cold",
            "layer_count": n_layers,
            "warm_start_attempt_count": n_layers,
            "fresh_fallback_count": 0,
            "experimental_profile_fixed_support_batch": {
                "schema": "exogibbs_experimental_profile_fixed_support_batch_profile_v1",
                "accepted": True,
                "bucket_count": None,
                "layer_count": n_layers,
                "route": route_name,
                "fallback_rescue": fallback_rescue,
            },
            "layers": tuple(updated_reports),
        }

    return CondensateEquilibriumProfileResult(
        layers=tuple(layer_results),
        method="vmap_cold",
        diagnostics=profile_diagnostics,
        batched_arrays=batched_arrays,
    )


def _run_experimental_profile_fixed_support_batch(
    *,
    setup: CondensateChemicalSetup,
    temperatures: np.ndarray,
    pressures: np.ndarray,
    b: Array,
    Pref: float,
    explicit_inits: Sequence[CondensateEquilibriumInit | None],
    initializer: Optional[CondensateEquilibriumInitializer],
    support_indices: Optional[Sequence[int]],
    support_amounts_init: Optional[Sequence[float]],
    opts: CondensateEquilibriumOptions,
    return_diagnostics: bool,
) -> CondensateEquilibriumProfileResult | None:
    if not opts.enable_experimental_profile_fixed_support_batch:
        return None
    if opts.restricted_reduced_coupling_mode != "pdipm_rgie_v11_activity_correction":
        return None
    if opts.profile_warm_start_support_policy != "explicit_payload":
        return None
    n_layers = int(temperatures.shape[0])
    solver_inits = []
    support_by_layer = []
    reports = []
    states = []
    for layer_index in range(n_layers):
        initial_guess = _resolve_condensate_initial_guess(
            initializer,
            CondensateEquilibriumInitRequest(
                setup=setup,
                T=float(temperatures[layer_index]),
                P=float(pressures[layer_index]),
                b=b,
                Pref=Pref,
                layer_index=layer_index,
                user_init=explicit_inits[layer_index],
                previous_solution=None,
            ),
        )
        if support_indices is not None:
            if support_amounts_init is None:
                return None
            support_payload = (
                tuple(int(index) for index in support_indices),
                _positive_support_amounts_for_warm_start(
                    support_amounts_init,
                    min_seed_amount=opts.min_seed_amount,
                ),
            )
        else:
            support_payload = _support_payload_from_condensate_init(
                initial_guess,
                setup=setup,
                min_seed_amount=opts.min_seed_amount,
            )
        if support_payload is None:
            return None
        layer_support_indices, layer_support_amounts = support_payload
        if len(layer_support_indices) == 0:
            return None
        solver_init = _solver_log_state_from_condensate_init(
            initial_guess,
            setup=setup,
            support_amounts_init=layer_support_amounts,
            source="exogibbs_experimental_profile_fixed_support_batch",
        )
        if solver_init is None:
            solver_init = _default_fixed_support_solver_log_state(
                setup=setup,
                T=float(temperatures[layer_index]),
                P=float(pressures[layer_index]),
                b=b,
                Pref=Pref,
                support_indices=layer_support_indices,
                support_amounts_init=layer_support_amounts,
                options=opts,
                source="exogibbs_experimental_profile_fixed_support_batch_default_gas_init",
            )
        solver_inits.append(solver_init)
        support_by_layer.append(layer_support_indices)
        states.append(
            ThermoState(
                temperature=float(temperatures[layer_index]),
                ln_normalized_pressure=_ln_normalized_pressure(
                    float(pressures[layer_index]),
                    Pref,
                ),
                element_vector=jnp.asarray(b, dtype=jnp.float64),
            )
        )
        reports.append(
            {
                "profile_layer_schema": "exogibbs_condensate_profile_layer_v1",
                "layer_index": int(layer_index),
                "initialization_mode": (
                    "experimental_profile_fixed_support_batch"
                ),
                "warm_start_attempted": True,
                "fresh_fallback_used": False,
                "support_count": len(layer_support_indices),
            }
        )
    from exogibbs.optimize.minimize_cond import (
        _prepare_pdipm_rgie_v11_activity_correction_profile_buckets,
        _run_pdipm_rgie_v11_activity_correction_prepared_profile_buckets,
    )

    max_iter = 100 if opts.max_inner_iterations is None else int(opts.max_inner_iterations)
    temperature_array = jnp.asarray(temperatures, dtype=jnp.float64)
    hvector_by_layer = jnp.asarray(
        setup.gas_setup.hvector_func(temperature_array),
        dtype=jnp.float64,
    )
    if hvector_by_layer.ndim != 2 or hvector_by_layer.shape[0] != n_layers:
        hvector_by_layer = None
    hvector_cond_by_layer = jnp.asarray(
        setup.condensate_setup.hvector_func(temperature_array),
        dtype=jnp.float64,
    )
    if hvector_cond_by_layer.ndim != 2 or hvector_cond_by_layer.shape[0] != n_layers:
        hvector_cond_by_layer = None
    buckets = _prepare_pdipm_rgie_v11_activity_correction_profile_buckets(
        states=tuple(states),
        init_states=tuple(solver_inits),
        support_indices_by_layer=tuple(support_by_layer),
        formula_matrix_cond=jnp.asarray(setup.formula_matrix_cond, dtype=jnp.float64),
        hvector_func=setup.gas_setup.hvector_func,
        hvector_cond_func=setup.condensate_setup.hvector_func,
        hvector_by_layer=hvector_by_layer,
        hvector_cond_by_layer=hvector_cond_by_layer,
    )
    if opts.enable_experimental_profile_fixed_support_fallback_rescue:
        plan = _ExperimentalProfileFixedSupportBatchPlan(
            setup=setup,
            buckets=buckets,
            formula_matrix=jnp.asarray(setup.formula_matrix, dtype=jnp.float64),
            max_iter=max_iter,
            n_layers=n_layers,
            condensate_count=len(setup.condensate_species),
            bucket_layer_index_arrays=tuple(
                jnp.asarray(bucket.layer_indices, dtype=jnp.int32)
                for bucket in buckets
            ),
        )
        arrays = run_experimental_profile_fixed_support_batch_plan_with_fallback_rescue(
            plan,
            rho_initialization=(
                opts.experimental_profile_fixed_support_rescue_rho_initialization
            ),
            lambda_initialization=(
                opts.experimental_profile_fixed_support_rescue_lambda_initialization
            ),
            residual_tolerance_multiplier=(
                opts.experimental_profile_fixed_support_rescue_residual_tolerance_multiplier
            ),
            prune_relative_floors=(
                opts.experimental_profile_fixed_support_rescue_prune_relative_floors
            ),
        )
        return _profile_result_from_fixed_support_batch_arrays(
            setup=setup,
            arrays=arrays,
            b=b,
            support_by_layer=support_by_layer,
            reports=reports,
            max_iter=max_iter,
            return_diagnostics=return_diagnostics,
            opts=opts,
            route_name="experimental_profile_fixed_support_batch_fallback_rescue",
        )
    bucket_results, batch_trace = (
        _run_pdipm_rgie_v11_activity_correction_prepared_profile_buckets(
            buckets=buckets,
            formula_matrix=jnp.asarray(setup.formula_matrix, dtype=jnp.float64),
            epsilon=-10.0,
            max_iter=max_iter,
        )
    )
    layer_solver_results: dict[int, Any] = {}
    for bucket, batch_result in zip(buckets, bucket_results):
        for local_index, layer_index in enumerate(bucket.layer_indices):
            layer_solver_results[int(layer_index)] = (
                batch_result,
                int(local_index),
            )
    if len(layer_solver_results) != n_layers:
        return None
    for batch_result, local_index in layer_solver_results.values():
        if not bool(batch_result.diagnostics.converged[local_index]):
            return None

    gas_ln_n_by_layer: dict[int, Array] = {}
    full_condensate_amounts_by_layer: dict[int, Array] = {}
    if not return_diagnostics and not opts.enable_full_condensate_budget_residual_gate:
        for bucket, batch_result in zip(buckets, bucket_results):
            support_index_array = jnp.asarray(bucket.support_indices, dtype=jnp.int32)
            support_amounts_batch = jnp.exp(batch_result.ln_mk)
            full_amounts_batch = jnp.zeros(
                (
                    len(bucket.layer_indices),
                    len(setup.condensate_species),
                ),
                dtype=support_amounts_batch.dtype,
            ).at[:, support_index_array].set(support_amounts_batch)
            for local_index, layer_index in enumerate(bucket.layer_indices):
                gas_ln_n_by_layer[int(layer_index)] = batch_result.ln_nk[local_index]
                full_condensate_amounts_by_layer[int(layer_index)] = (
                    full_amounts_batch[local_index]
                )
    batched_arrays = None
    if gas_ln_n_by_layer and len(gas_ln_n_by_layer) == n_layers:
        gas_ln_n_batch = jnp.stack(
            [gas_ln_n_by_layer[index] for index in range(n_layers)],
            axis=0,
        )
        gas_n_batch = jnp.exp(gas_ln_n_batch)
        gas_ntot_batch = jnp.sum(gas_n_batch, axis=1)
        cond_amounts_batch = jnp.stack(
            [full_condensate_amounts_by_layer[index] for index in range(n_layers)],
            axis=0,
        )
        batched_arrays = {
            "gas_ln_n": gas_ln_n_batch,
            "gas_n": gas_n_batch,
            "gas_x": gas_n_batch / jnp.clip(gas_ntot_batch[:, None], 1.0e-300),
            "gas_ntot": gas_ntot_batch,
            "condensate_amounts": cond_amounts_batch,
        }

    layer_results = []
    for layer_index in range(n_layers):
        batch_result, local_index = layer_solver_results[layer_index]
        support_tuple = support_by_layer[layer_index]
        support_amounts = jnp.exp(batch_result.ln_mk[local_index])
        solver_success = bool(batch_result.diagnostics.converged[local_index])
        if not return_diagnostics and not opts.enable_full_condensate_budget_residual_gate:
            gas_ln_n = jnp.asarray(batch_result.ln_nk[local_index], dtype=jnp.float64)
            gas_n = jnp.exp(gas_ln_n)
            gas_ntot = jnp.sum(gas_n)
            support_index_array = jnp.asarray(support_tuple, dtype=jnp.int32)
            condensate_amounts = full_condensate_amounts_by_layer[layer_index]
            result = CondensateEquilibriumResult(
                gas_ln_n=gas_ln_n,
                gas_n=gas_n,
                gas_x=gas_n / jnp.clip(gas_ntot, 1.0e-300),
                gas_ntot=gas_ntot,
                condensate_amounts=condensate_amounts,
                condensate_support_indices=support_index_array,
                condensate_support_names=tuple(
                    setup.condensate_species[int(index)] for index in support_tuple
                ),
                acceptance_tier="runtime_unclassified",
                selected_route="experimental_profile_fixed_support_batch",
                status=CONVERGED if solver_success else NOT_CONVERGED,
                converged=solver_success,
                diagnostics=None,
            )
        else:
            diagnostics = {
                "experimental_profile_fixed_support_batch": {
                    "schema": "exogibbs_experimental_profile_fixed_support_batch_v1",
                    "enabled": True,
                    "layer_index": int(layer_index),
                    "solver_success": solver_success,
                    "max_iter": max_iter,
                    "n_iter": int(batch_result.diagnostics.n_iter[local_index]),
                    "final_residual": float(
                        batch_result.diagnostics.final_residual[local_index]
                    ),
                    "support_indices": tuple(int(index) for index in support_tuple),
                },
                **batch_trace,
            }
            result = build_condensate_equilibrium_result_from_solver_payload(
                setup=setup,
                gas_ln_n=batch_result.ln_nk[local_index],
                support_indices=support_tuple,
                support_amounts=support_amounts,
                selected_route="experimental_profile_fixed_support_batch",
                metric_status=None,
                solver_success=solver_success,
                allow_caveat_tiers=opts.allow_caveat_tiers,
                diagnostics=diagnostics,
                element_inventory_target=b,
                enable_full_condensate_budget_residual_gate=(
                    opts.enable_full_condensate_budget_residual_gate
                ),
                full_condensate_budget_relative_tolerance=(
                    opts.full_condensate_budget_relative_tolerance
                ),
            )
        reports[layer_index] = {
            **reports[layer_index],
            "accepted": bool(result.converged),
        }
        layer_results.append(
            _with_profile_layer_diagnostics(
                result,
                profile_report=reports[layer_index],
                return_diagnostics=return_diagnostics,
            )
        )
    profile_diagnostics = None
    if return_diagnostics:
        profile_diagnostics = {
            "profile_schema": "exogibbs_condensate_equilibrium_profile_v1",
            "method": "vmap_cold",
            "layer_count": n_layers,
            "warm_start_attempt_count": n_layers,
            "fresh_fallback_count": 0,
            "experimental_profile_fixed_support_batch": {
                "schema": "exogibbs_experimental_profile_fixed_support_batch_profile_v1",
                "accepted": all(bool(result.converged) for result in layer_results),
                "bucket_count": len(buckets),
                "layer_count": n_layers,
            },
            "layers": tuple(reports),
        }
    return CondensateEquilibriumProfileResult(
        layers=tuple(layer_results),
        method="vmap_cold",
        diagnostics=profile_diagnostics,
        batched_arrays=batched_arrays,
    )


def _prepare_experimental_profile_fixed_support_batch_plan(
    *,
    setup: CondensateChemicalSetup,
    temperatures: np.ndarray,
    pressures: np.ndarray,
    b: Array,
    Pref: float,
    explicit_inits: Sequence[CondensateEquilibriumInit | None],
    initializer: Optional[CondensateEquilibriumInitializer],
    support_indices: Optional[Sequence[int]],
    support_amounts_init: Optional[Sequence[float]],
    opts: CondensateEquilibriumOptions,
) -> _ExperimentalProfileFixedSupportBatchPlan | None:
    if not opts.enable_experimental_profile_fixed_support_batch:
        return None
    if opts.restricted_reduced_coupling_mode != "pdipm_rgie_v11_activity_correction":
        return None
    if opts.profile_warm_start_support_policy != "explicit_payload":
        return None
    n_layers = int(temperatures.shape[0])
    solver_inits = []
    support_by_layer = []
    states = []
    for layer_index in range(n_layers):
        initial_guess = _resolve_condensate_initial_guess(
            initializer,
            CondensateEquilibriumInitRequest(
                setup=setup,
                T=float(temperatures[layer_index]),
                P=float(pressures[layer_index]),
                b=b,
                Pref=Pref,
                layer_index=layer_index,
                user_init=explicit_inits[layer_index],
                previous_solution=None,
            ),
        )
        if support_indices is not None:
            if support_amounts_init is None:
                return None
            support_payload = (
                tuple(int(index) for index in support_indices),
                _positive_support_amounts_for_warm_start(
                    support_amounts_init,
                    min_seed_amount=opts.min_seed_amount,
                ),
            )
        else:
            support_payload = _support_payload_from_condensate_init(
                initial_guess,
                setup=setup,
                min_seed_amount=opts.min_seed_amount,
            )
        if support_payload is None:
            return None
        layer_support_indices, layer_support_amounts = support_payload
        if len(layer_support_indices) == 0:
            return None
        solver_init = _solver_log_state_from_condensate_init(
            initial_guess,
            setup=setup,
            support_amounts_init=layer_support_amounts,
            source="exogibbs_experimental_profile_fixed_support_batch_plan",
        )
        if solver_init is None:
            return None
        solver_inits.append(solver_init)
        support_by_layer.append(layer_support_indices)
        states.append(
            ThermoState(
                temperature=float(temperatures[layer_index]),
                ln_normalized_pressure=_ln_normalized_pressure(
                    float(pressures[layer_index]),
                    Pref,
                ),
                element_vector=jnp.asarray(b, dtype=jnp.float64),
            )
        )
    from exogibbs.optimize.minimize_cond import (
        _prepare_pdipm_rgie_v11_activity_correction_profile_buckets,
    )

    temperature_array = jnp.asarray(temperatures, dtype=jnp.float64)
    hvector_by_layer = jnp.asarray(
        setup.gas_setup.hvector_func(temperature_array),
        dtype=jnp.float64,
    )
    if hvector_by_layer.ndim != 2 or hvector_by_layer.shape[0] != n_layers:
        hvector_by_layer = None
    hvector_cond_by_layer = jnp.asarray(
        setup.condensate_setup.hvector_func(temperature_array),
        dtype=jnp.float64,
    )
    if hvector_cond_by_layer.ndim != 2 or hvector_cond_by_layer.shape[0] != n_layers:
        hvector_cond_by_layer = None
    buckets = _prepare_pdipm_rgie_v11_activity_correction_profile_buckets(
        states=tuple(states),
        init_states=tuple(solver_inits),
        support_indices_by_layer=tuple(support_by_layer),
        formula_matrix_cond=jnp.asarray(setup.formula_matrix_cond, dtype=jnp.float64),
        hvector_func=setup.gas_setup.hvector_func,
        hvector_cond_func=setup.condensate_setup.hvector_func,
        hvector_by_layer=hvector_by_layer,
        hvector_cond_by_layer=hvector_cond_by_layer,
    )
    max_iter = 100 if opts.max_inner_iterations is None else int(opts.max_inner_iterations)
    return _ExperimentalProfileFixedSupportBatchPlan(
        setup=setup,
        buckets=buckets,
        formula_matrix=jnp.asarray(setup.formula_matrix, dtype=jnp.float64),
        max_iter=max_iter,
        n_layers=n_layers,
        condensate_count=len(setup.condensate_species),
        bucket_layer_index_arrays=tuple(
            jnp.asarray(bucket.layer_indices, dtype=jnp.int32)
            for bucket in buckets
        ),
    )


def _run_experimental_profile_fixed_support_batch_plan_arrays(
    plan: _ExperimentalProfileFixedSupportBatchPlan,
    *,
    element_inventory_target: Optional[Array] = None,
    rho_initialization: str = "unit_activity",
    lambda_initialization: str = "best_residual",
    residual_tolerance_multiplier: float = 1.0,
) -> Mapping[str, Array]:
    from exogibbs.optimize.minimize_cond import (
        _run_pdipm_rgie_v11_activity_correction_prepared_profile_buckets,
    )

    buckets = plan.buckets
    if element_inventory_target is not None:
        target = jnp.asarray(element_inventory_target, dtype=jnp.float64)
        n_elements = int(plan.formula_matrix.shape[0])
        if target.ndim == 1:
            if target.shape[0] != n_elements:
                raise ValueError("element_inventory_target length must match elements.")
            target = jnp.broadcast_to(target, (plan.n_layers, n_elements))
        elif target.ndim == 2:
            if target.shape != (plan.n_layers, n_elements):
                raise ValueError(
                    "element_inventory_target must have shape "
                    f"({plan.n_layers}, {n_elements})."
                )
        else:
            raise ValueError("element_inventory_target must be one- or two-dimensional.")
        buckets = tuple(
            replace(
                bucket,
                element_inventory_target=target[layer_indices],
            )
            for bucket, layer_indices in zip(
                plan.buckets,
                plan.bucket_layer_index_arrays,
            )
        )

    bucket_results, _batch_trace = (
        _run_pdipm_rgie_v11_activity_correction_prepared_profile_buckets(
            buckets=buckets,
            formula_matrix=plan.formula_matrix,
            epsilon=-10.0,
            max_iter=plan.max_iter,
            rho_initialization=rho_initialization,
            lambda_initialization=lambda_initialization,
            residual_tolerance_multiplier=residual_tolerance_multiplier,
        )
    )
    gas_species_count = int(plan.formula_matrix.shape[1])
    gas_ln_n_batch = jnp.zeros(
        (plan.n_layers, gas_species_count),
        dtype=jnp.float64,
    )
    condensate_amounts_batch = jnp.zeros(
        (plan.n_layers, plan.condensate_count),
        dtype=jnp.float64,
    )
    converged_batch = jnp.zeros((plan.n_layers,), dtype=bool)
    final_residual_batch = jnp.zeros((plan.n_layers,), dtype=jnp.float64)
    n_iter_batch = jnp.zeros((plan.n_layers,), dtype=jnp.int32)
    for bucket, batch_result, layer_indices in zip(
        plan.buckets,
        bucket_results,
        plan.bucket_layer_index_arrays,
    ):
        support_index_array = jnp.asarray(bucket.support_indices, dtype=jnp.int32)
        support_amounts_batch = jnp.exp(batch_result.ln_mk)
        full_amounts_batch = jnp.zeros(
            (
                len(bucket.layer_indices),
                plan.condensate_count,
            ),
            dtype=support_amounts_batch.dtype,
        ).at[:, support_index_array].set(support_amounts_batch)
        gas_ln_n_batch = gas_ln_n_batch.at[layer_indices].set(batch_result.ln_nk)
        condensate_amounts_batch = condensate_amounts_batch.at[layer_indices].set(
            full_amounts_batch
        )
        converged_batch = converged_batch.at[layer_indices].set(
            batch_result.diagnostics.converged
        )
        final_residual_batch = final_residual_batch.at[layer_indices].set(
            batch_result.diagnostics.final_residual
        )
        n_iter_batch = n_iter_batch.at[layer_indices].set(
            jnp.asarray(batch_result.diagnostics.n_iter, dtype=jnp.int32)
        )
    gas_n_batch = jnp.exp(gas_ln_n_batch)
    gas_ntot_batch = jnp.sum(gas_n_batch, axis=1)
    return {
        "gas_ln_n": gas_ln_n_batch,
        "gas_n": gas_n_batch,
        "gas_x": gas_n_batch / jnp.clip(gas_ntot_batch[:, None], 1.0e-300),
        "gas_ntot": gas_ntot_batch,
        "condensate_amounts": condensate_amounts_batch,
        "converged": converged_batch,
        "fallback_required": ~converged_batch,
        "final_residual": final_residual_batch,
        "n_iter": n_iter_batch,
        "lambda_candidate_labels": (
            "provided",
            "gas_lstsq",
            "gas_cond_lstsq",
            "damped_gas_lstsq",
            "damped_gas_cond_lstsq",
        ),
    }


def _fallback_layer_indices_from_fixed_support_arrays(
    arrays: Mapping[str, Any],
) -> tuple[int, ...]:
    fallback_required = np.asarray(
        jax.device_get(arrays.get("fallback_required", ~arrays["converged"])),
        dtype=bool,
    )
    if fallback_required.ndim == 1:
        indices = np.where(fallback_required)[0]
    else:
        indices = np.where(np.any(fallback_required, axis=0))[0]
    return tuple(int(index) for index in indices.tolist())


def _prepare_experimental_profile_fixed_support_prune_rescue_plan(
    plan: _ExperimentalProfileFixedSupportBatchPlan,
    fallback_layer_indices: Sequence[int],
    *,
    prune_relative_floors: Sequence[float],
) -> tuple[_ExperimentalProfileFixedSupportBatchPlan | None, Mapping[str, Any]]:
    fallback_set = {int(index) for index in fallback_layer_indices}
    floors = tuple(sorted({float(value) for value in prune_relative_floors}))
    if any(not math.isfinite(value) or value <= 0.0 for value in floors):
        raise ValueError("prune_relative_floors must contain positive finite values.")
    if not fallback_set:
        return None, {
            "schema": "exogibbs_experimental_profile_fixed_support_prune_rescue_v1",
            "mode": "none",
            "fallback_layer_indices": (),
            "expanded_to_original_layer": (),
            "candidate_labels": (),
            "candidate_support_counts": (),
            "expanded_layer_count": 0,
            "prune_relative_floors": floors,
        }

    from exogibbs.optimize.minimize_cond import _PDIPMActivityFixedSupportBucket

    candidates_by_key: dict[
        tuple[tuple[int, ...], bool],
        list[dict[str, Any]],
    ] = {}
    expanded_to_original: list[int] = []
    candidate_labels: list[str] = []
    candidate_support_counts: list[int] = []

    for bucket in plan.buckets:
        original_support = tuple(int(index) for index in bucket.support_indices)
        support_position = {
            int(index): local_index
            for local_index, index in enumerate(original_support)
        }
        for local_index, original_layer_index in enumerate(bucket.layer_indices):
            original_layer_index = int(original_layer_index)
            if original_layer_index not in fallback_set:
                continue
            ln_mk = jnp.asarray(bucket.ln_mk_init[local_index], dtype=jnp.float64)
            amounts = np.asarray(jax.device_get(jnp.exp(ln_mk)), dtype=np.float64)
            if amounts.size == 0:
                continue
            variants: list[tuple[str, tuple[int, ...], bool]] = [
                ("current", original_support, True),
            ]
            max_amount = float(np.max(amounts))
            for relative_floor in floors:
                threshold = max(1.0e-12, float(relative_floor) * max_amount)
                pruned = tuple(
                    support_index
                    for support_index, amount in zip(original_support, amounts)
                    if float(amount) >= threshold
                )
                if pruned and pruned != original_support:
                    variants.append(
                        (
                            f"prune_amount_ge_{relative_floor:g}_max",
                            pruned,
                            False,
                        )
                    )

            seen_supports: set[tuple[int, ...]] = set()
            for label, support, keep_rho in variants:
                if support in seen_supports:
                    continue
                seen_supports.add(support)
                positions = tuple(support_position[index] for index in support)
                candidate_index = len(expanded_to_original)
                expanded_to_original.append(original_layer_index)
                candidate_labels.append(label)
                candidate_support_counts.append(len(support))
                key = (support, keep_rho and bucket.rho_init is not None)
                candidates_by_key.setdefault(key, []).append(
                    {
                        "candidate_index": candidate_index,
                        "source_bucket": bucket,
                        "source_local_index": int(local_index),
                        "positions": positions,
                    }
                )

    rescue_buckets = []
    for (support, keep_rho), entries in candidates_by_key.items():
        first = entries[0]
        source_bucket = first["source_bucket"]
        positions = first["positions"]
        position_array = jnp.asarray(positions, dtype=jnp.int32)
        layer_indices = tuple(int(entry["candidate_index"]) for entry in entries)
        source_local_indices = tuple(int(entry["source_local_index"]) for entry in entries)
        source_local_array = jnp.asarray(source_local_indices, dtype=jnp.int32)
        rho_init = None
        if keep_rho and source_bucket.rho_init is not None:
            rho_init = source_bucket.rho_init[source_local_array][:, position_array]
        rescue_buckets.append(
            _PDIPMActivityFixedSupportBucket(
                support_indices=support,
                layer_indices=layer_indices,
                formula_matrix_cond_active=source_bucket.formula_matrix_cond_active[
                    :,
                    position_array,
                ],
                ln_nk_init=source_bucket.ln_nk_init[source_local_array],
                ln_mk_init=source_bucket.ln_mk_init[source_local_array][
                    :,
                    position_array,
                ],
                ln_ntot_init=source_bucket.ln_ntot_init[source_local_array],
                element_potential_init=None
                if source_bucket.element_potential_init is None
                else source_bucket.element_potential_init[source_local_array],
                rho_init=rho_init,
                barrier_epsilon_init=None
                if source_bucket.barrier_epsilon_init is None
                else source_bucket.barrier_epsilon_init[source_local_array],
                gas_stationarity_source_init=None
                if source_bucket.gas_stationarity_source_init is None
                else source_bucket.gas_stationarity_source_init[source_local_array],
                element_inventory_target=source_bucket.element_inventory_target[
                    source_local_array
                ],
                hvector=source_bucket.hvector[source_local_array],
                hvector_cond_active=source_bucket.hvector_cond_active[
                    source_local_array
                ][:, position_array],
                ln_normalized_pressure=source_bucket.ln_normalized_pressure[
                    source_local_array
                ],
            )
        )

    metadata = {
        "schema": "exogibbs_experimental_profile_fixed_support_prune_rescue_v1",
        "mode": "prune",
        "fallback_layer_indices": tuple(sorted(fallback_set)),
        "expanded_to_original_layer": tuple(expanded_to_original),
        "candidate_labels": tuple(candidate_labels),
        "candidate_support_counts": tuple(candidate_support_counts),
        "expanded_layer_count": len(expanded_to_original),
        "prune_relative_floors": floors,
    }
    if not rescue_buckets:
        return None, metadata
    return (
        _ExperimentalProfileFixedSupportBatchPlan(
            setup=plan.setup,
            buckets=tuple(rescue_buckets),
            formula_matrix=plan.formula_matrix,
            max_iter=plan.max_iter,
            n_layers=len(expanded_to_original),
            condensate_count=plan.condensate_count,
            bucket_layer_index_arrays=tuple(
                jnp.asarray(bucket.layer_indices, dtype=jnp.int32)
                for bucket in rescue_buckets
            ),
        ),
        metadata,
    )


def _merge_fixed_support_prune_rescue_arrays(
    base_arrays: Mapping[str, Any],
    rescue_arrays: Mapping[str, Any],
    rescue_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    expanded_to_original = tuple(
        int(index) for index in rescue_metadata["expanded_to_original_layer"]
    )
    candidate_labels = tuple(str(label) for label in rescue_metadata["candidate_labels"])
    candidate_support_counts = tuple(
        int(value) for value in rescue_metadata["candidate_support_counts"]
    )
    base_residual = np.asarray(
        jax.device_get(base_arrays["final_residual"]),
        dtype=np.float64,
    )
    base_fallback = np.asarray(
        jax.device_get(base_arrays.get("fallback_required", ~base_arrays["converged"])),
        dtype=bool,
    )
    rescue_residual = np.asarray(
        jax.device_get(rescue_arrays["final_residual"]),
        dtype=np.float64,
    )
    rescue_converged = np.asarray(
        jax.device_get(rescue_arrays["converged"]),
        dtype=bool,
    )
    if base_residual.ndim == 1:
        eval_count = 1
        base_residual_view = base_residual[None, :]
        base_fallback_view = base_fallback[None, :]
        rescue_residual_view = rescue_residual[None, :]
        rescue_converged_view = rescue_converged[None, :]
    else:
        eval_count = int(base_residual.shape[0])
        base_residual_view = base_residual
        base_fallback_view = base_fallback
        rescue_residual_view = rescue_residual
        rescue_converged_view = rescue_converged
    n_layers = int(base_residual_view.shape[1])
    replace_mask = np.zeros((eval_count, n_layers), dtype=bool)
    selected_expanded = np.full((eval_count, n_layers), -1, dtype=np.int32)
    selected_label = np.full((eval_count, n_layers), None, dtype=object)
    selected_support_count = np.zeros((eval_count, n_layers), dtype=np.int32)

    for original_index in sorted(set(expanded_to_original)):
        candidates = np.asarray(
            [
                candidate_index
                for candidate_index, mapped_index in enumerate(expanded_to_original)
                if mapped_index == original_index
            ],
            dtype=np.int64,
        )
        for eval_index in range(eval_count):
            candidate_converged = rescue_converged_view[eval_index, candidates]
            candidate_residual = rescue_residual_view[eval_index, candidates]
            score = np.where(candidate_converged, candidate_residual, np.inf)
            if not np.isfinite(score).any():
                score = np.where(np.isfinite(candidate_residual), candidate_residual, np.inf)
            selected = int(candidates[int(np.argmin(score))])
            rescue_value = rescue_residual_view[eval_index, selected]
            base_value = base_residual_view[eval_index, original_index]
            finite_improvement = np.isfinite(rescue_value) and (
                (not np.isfinite(base_value)) or rescue_value < base_value
            )
            should_replace = bool(
                base_fallback_view[eval_index, original_index]
                and (
                    rescue_converged_view[eval_index, selected]
                    or finite_improvement
                )
            )
            if should_replace:
                replace_mask[eval_index, original_index] = True
                selected_expanded[eval_index, original_index] = selected
                selected_label[eval_index, original_index] = candidate_labels[selected]
                selected_support_count[eval_index, original_index] = (
                    candidate_support_counts[selected]
                )

    def merge_layer_array(base_value: Any, rescue_value: Any) -> Any:
        base_array = np.asarray(jax.device_get(base_value))
        rescue_array = np.asarray(jax.device_get(rescue_value))
        if base_residual.ndim == 1:
            if base_array.ndim == 0 or base_array.shape[0] != n_layers:
                return base_value
            merged = base_array.copy()
            for original_index in range(n_layers):
                selected = int(selected_expanded[0, original_index])
                if selected >= 0:
                    merged[original_index] = rescue_array[selected]
            return merged
        if (
            base_array.ndim < 2
            or base_array.shape[0] != eval_count
            or base_array.shape[1] != n_layers
        ):
            return base_value
        merged = base_array.copy()
        row_indices = np.arange(eval_count)
        for original_index in range(n_layers):
            selected = selected_expanded[:, original_index]
            rows = selected >= 0
            if np.any(rows):
                merged[rows, original_index] = rescue_array[
                    row_indices[rows],
                    selected[rows],
                ]
        return merged

    merged = {
        key: merge_layer_array(value, rescue_arrays[key])
        for key, value in base_arrays.items()
        if key in rescue_arrays
        and key
        not in {
            "residual_components",
            "step_diagnostics",
            "lambda_candidate_labels",
        }
    }
    merged["residual_components"] = {
        key: merge_layer_array(value, rescue_arrays["residual_components"][key])
        for key, value in base_arrays.get("residual_components", {}).items()
        if key in rescue_arrays.get("residual_components", {})
    }
    merged["step_diagnostics"] = {
        key: merge_layer_array(value, rescue_arrays["step_diagnostics"][key])
        for key, value in base_arrays.get("step_diagnostics", {}).items()
        if key in rescue_arrays.get("step_diagnostics", {})
    }
    merged["fallback_rescue"] = {
        **dict(rescue_metadata),
        "replaced_count": int(np.count_nonzero(replace_mask)),
        "selected_expanded_layer_index": (
            selected_expanded[0] if base_residual.ndim == 1 else selected_expanded
        ),
        "selected_candidate_label": (
            selected_label[0].tolist()
            if base_residual.ndim == 1
            else selected_label.tolist()
        ),
        "selected_support_count": (
            selected_support_count[0]
            if base_residual.ndim == 1
            else selected_support_count
        ),
        "replaced": replace_mask[0] if base_residual.ndim == 1 else replace_mask,
    }
    if "lambda_candidate_labels" in base_arrays:
        merged["lambda_candidate_labels"] = base_arrays["lambda_candidate_labels"]
    return merged


def _attach_empty_fixed_support_rescue_metadata(
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {**dict(arrays), "fallback_rescue": {**dict(metadata), "replaced_count": 0}}


def prepare_experimental_profile_fixed_support_batch_plan(
    setup: CondensateChemicalSetup,
    T: Sequence[float] | Array,
    P: Sequence[float] | Array,
    b: Array,
    *,
    Pref: float = 1.0,
    support_indices: Optional[Sequence[int]] = None,
    support_amounts_init: Optional[Sequence[float]] = None,
    init: Optional[Sequence[CondensateEquilibriumInit | None]] = None,
    initializer: Optional[CondensateEquilibriumInitializer] = None,
    options: Optional[CondensateEquilibriumOptions] = None,
) -> ExperimentalCondensateProfileFixedSupportBatchPlan:
    """Prepare a reusable experimental fixed-support batch profile plan.

    The returned plan can be run repeatedly with
    :func:`run_experimental_profile_fixed_support_batch_plan` to avoid rebuilding
    support buckets and thermochemical vectors for every call.
    """

    validate_condensate_chemical_setup(setup)
    temperatures = np.asarray(T, dtype=np.float64)
    pressures = np.asarray(P, dtype=np.float64)
    if temperatures.ndim != 1 or pressures.ndim != 1:
        raise ValueError("T and P must be 1D arrays of equal length.")
    if temperatures.shape[0] != pressures.shape[0]:
        raise ValueError("T and P must have the same length.")
    n_layers = int(temperatures.shape[0])
    if init is None:
        explicit_inits: tuple[CondensateEquilibriumInit | None, ...] = (
            None,
        ) * n_layers
    else:
        explicit_inits = tuple(init)
        if len(explicit_inits) != n_layers:
            raise ValueError("init must have one entry per profile layer.")
    opts = replace(
        options or CondensateEquilibriumOptions(),
        profile_method="vmap_cold",
        profile_warm_start_support_policy="explicit_payload",
        enable_experimental_profile_fixed_support_batch=True,
    )
    _validate_options(opts)
    plan = _prepare_experimental_profile_fixed_support_batch_plan(
        setup=setup,
        temperatures=temperatures,
        pressures=pressures,
        b=b,
        Pref=Pref,
        explicit_inits=explicit_inits,
        initializer=initializer,
        support_indices=support_indices,
        support_amounts_init=support_amounts_init,
        opts=opts,
    )
    if plan is None:
        raise ValueError(
            "experimental fixed-support batch plan requires non-empty explicit "
            "support and solver log-state initialization for every layer."
        )
    return plan


def run_experimental_profile_fixed_support_batch_plan(
    plan: ExperimentalCondensateProfileFixedSupportBatchPlan,
    *,
    element_inventory_target: Optional[Array] = None,
    rho_initialization: str = "unit_activity",
    lambda_initialization: str = "best_residual",
    residual_tolerance_multiplier: float = 1.0,
) -> Mapping[str, Array]:
    """Run a prepared experimental fixed-support batch profile plan.

    ``element_inventory_target`` may be a single element vector shared by every
    layer or a ``(n_layers, n_elements)`` array for repeated fixed-support
    evaluations with updated composition.
    """

    if not isinstance(plan, ExperimentalCondensateProfileFixedSupportBatchPlan):
        raise TypeError(
            "plan must be an ExperimentalCondensateProfileFixedSupportBatchPlan."
        )
    return _run_experimental_profile_fixed_support_batch_plan_arrays(
        plan,
        element_inventory_target=element_inventory_target,
        rho_initialization=rho_initialization,
        lambda_initialization=lambda_initialization,
        residual_tolerance_multiplier=residual_tolerance_multiplier,
    )


def prepare_experimental_profile_fixed_support_prune_rescue_plan(
    plan: ExperimentalCondensateProfileFixedSupportBatchPlan,
    fallback_layer_indices: Sequence[int],
    *,
    prune_relative_floors: Sequence[float] = (1.0e-5, 1.0e-3),
) -> ExperimentalCondensateProfileFixedSupportPruneRescuePlan:
    """Prepare prune-rescue candidates for selected fallback layers.

    This separates host-side rescue plan construction from repeated GPU solves.
    The returned object may contain no executable rescue plan when
    ``fallback_layer_indices`` is empty or no pruned candidate changes support.
    """

    if not isinstance(plan, ExperimentalCondensateProfileFixedSupportBatchPlan):
        raise TypeError(
            "plan must be an ExperimentalCondensateProfileFixedSupportBatchPlan."
        )
    rescue_plan, rescue_metadata = (
        _prepare_experimental_profile_fixed_support_prune_rescue_plan(
            plan,
            fallback_layer_indices,
            prune_relative_floors=prune_relative_floors,
        )
    )
    return _ExperimentalProfileFixedSupportPruneRescuePlan(
        rescue_plan=rescue_plan,
        metadata=rescue_metadata,
    )


def _fixed_support_prune_rescue_cache_key(
    fallback_layer_indices: Sequence[int],
    prune_relative_floors: Sequence[float],
) -> tuple[Any, ...]:
    return (
        tuple(int(index) for index in fallback_layer_indices),
        tuple(float(floor) for floor in prune_relative_floors),
    )


def _get_cached_fixed_support_prune_rescue_plan(
    plan: ExperimentalCondensateProfileFixedSupportBatchPlan,
    cache: ExperimentalCondensateProfileFixedSupportPruneRescueCache,
    fallback_layer_indices: Sequence[int],
    *,
    prune_relative_floors: Sequence[float],
) -> ExperimentalCondensateProfileFixedSupportPruneRescuePlan:
    key = _fixed_support_prune_rescue_cache_key(
        fallback_layer_indices,
        prune_relative_floors,
    )
    cached = cache.plans.get(key)
    if cached is not None:
        cache.hit_count += 1
        return cached
    prepared = prepare_experimental_profile_fixed_support_prune_rescue_plan(
        plan,
        fallback_layer_indices,
        prune_relative_floors=prune_relative_floors,
    )
    cache.plans[key] = prepared
    cache.prepare_count += 1
    return prepared


def _fixed_support_prune_rescue_single_target(
    element_inventory_target: Optional[Array],
    rescue_metadata: Mapping[str, Any],
) -> Optional[Array]:
    if element_inventory_target is None:
        return None
    target = jnp.asarray(element_inventory_target, dtype=jnp.float64)
    if target.ndim == 2:
        expanded_to_original = jnp.asarray(
            rescue_metadata["expanded_to_original_layer"],
            dtype=jnp.int32,
        )
        return target[expanded_to_original, :]
    return target


def _fixed_support_prune_rescue_many_targets(
    element_inventory_targets: Array,
    rescue_metadata: Mapping[str, Any],
) -> Array:
    targets = jnp.asarray(element_inventory_targets, dtype=jnp.float64)
    if targets.ndim == 3:
        expanded_to_original = jnp.asarray(
            rescue_metadata["expanded_to_original_layer"],
            dtype=jnp.int32,
        )
        return targets[:, expanded_to_original, :]
    return targets


def run_experimental_profile_fixed_support_batch_plan_with_prepared_fallback_rescue(
    plan: ExperimentalCondensateProfileFixedSupportBatchPlan,
    rescue: ExperimentalCondensateProfileFixedSupportPruneRescuePlan,
    *,
    element_inventory_target: Optional[Array] = None,
    rho_initialization: str = "complementarity",
    lambda_initialization: str = "best_residual",
    residual_tolerance_multiplier: float = 1.0e9,
) -> Mapping[str, Any]:
    """Run a fixed-support plan with a pre-built prune-rescue plan."""

    if not isinstance(plan, ExperimentalCondensateProfileFixedSupportBatchPlan):
        raise TypeError(
            "plan must be an ExperimentalCondensateProfileFixedSupportBatchPlan."
        )
    if not isinstance(rescue, ExperimentalCondensateProfileFixedSupportPruneRescuePlan):
        raise TypeError(
            "rescue must be an "
            "ExperimentalCondensateProfileFixedSupportPruneRescuePlan."
        )
    base_arrays = run_experimental_profile_fixed_support_batch_plan(
        plan,
        element_inventory_target=element_inventory_target,
        rho_initialization=rho_initialization,
        lambda_initialization=lambda_initialization,
        residual_tolerance_multiplier=residual_tolerance_multiplier,
    )
    if rescue.rescue_plan is None:
        return _attach_empty_fixed_support_rescue_metadata(
            base_arrays,
            rescue.metadata,
        )
    rescue_arrays = run_experimental_profile_fixed_support_batch_plan(
        rescue.rescue_plan,
        element_inventory_target=_fixed_support_prune_rescue_single_target(
            element_inventory_target,
            rescue.metadata,
        ),
        rho_initialization=rho_initialization,
        lambda_initialization=lambda_initialization,
        residual_tolerance_multiplier=residual_tolerance_multiplier,
    )
    return _merge_fixed_support_prune_rescue_arrays(
        base_arrays,
        rescue_arrays,
        rescue.metadata,
    )


def run_experimental_profile_fixed_support_batch_plan_with_cached_fallback_rescue(
    plan: ExperimentalCondensateProfileFixedSupportBatchPlan,
    cache: ExperimentalCondensateProfileFixedSupportPruneRescueCache,
    *,
    element_inventory_target: Optional[Array] = None,
    rho_initialization: str = "complementarity",
    lambda_initialization: str = "best_residual",
    residual_tolerance_multiplier: float = 1.0e9,
    prune_relative_floors: Sequence[float] = (1.0e-5, 1.0e-3),
) -> Mapping[str, Any]:
    """Run fallback rescue while caching rescue plans for repeated layer sets."""

    if not isinstance(plan, ExperimentalCondensateProfileFixedSupportBatchPlan):
        raise TypeError(
            "plan must be an ExperimentalCondensateProfileFixedSupportBatchPlan."
        )
    if not isinstance(cache, ExperimentalCondensateProfileFixedSupportPruneRescueCache):
        raise TypeError(
            "cache must be an "
            "ExperimentalCondensateProfileFixedSupportPruneRescueCache."
        )
    base_arrays = run_experimental_profile_fixed_support_batch_plan(
        plan,
        element_inventory_target=element_inventory_target,
        rho_initialization=rho_initialization,
        lambda_initialization=lambda_initialization,
        residual_tolerance_multiplier=residual_tolerance_multiplier,
    )
    rescue = _get_cached_fixed_support_prune_rescue_plan(
        plan,
        cache,
        _fallback_layer_indices_from_fixed_support_arrays(base_arrays),
        prune_relative_floors=prune_relative_floors,
    )
    if rescue.rescue_plan is None:
        return _attach_empty_fixed_support_rescue_metadata(base_arrays, rescue.metadata)
    rescue_arrays = run_experimental_profile_fixed_support_batch_plan(
        rescue.rescue_plan,
        element_inventory_target=_fixed_support_prune_rescue_single_target(
            element_inventory_target,
            rescue.metadata,
        ),
        rho_initialization=rho_initialization,
        lambda_initialization=lambda_initialization,
        residual_tolerance_multiplier=residual_tolerance_multiplier,
    )
    return _merge_fixed_support_prune_rescue_arrays(
        base_arrays,
        rescue_arrays,
        rescue.metadata,
    )


def run_experimental_profile_fixed_support_batch_plan_with_fallback_rescue(
    plan: ExperimentalCondensateProfileFixedSupportBatchPlan,
    *,
    element_inventory_target: Optional[Array] = None,
    rho_initialization: str = "complementarity",
    lambda_initialization: str = "best_residual",
    residual_tolerance_multiplier: float = 1.0e9,
    prune_relative_floors: Sequence[float] = (1.0e-5, 1.0e-3),
) -> Mapping[str, Any]:
    """Run a prepared fixed-support plan, then prune-rescue fallback layers.

    The first pass is identical to
    :func:`run_experimental_profile_fixed_support_batch_plan`. Layers whose
    ``fallback_required`` flag is true are re-run through a smaller prepared
    plan containing ``current`` plus pruned-support candidates. The returned
    arrays keep the original layer shape and include a ``fallback_rescue``
    metadata entry describing any replacements.
    """

    if not isinstance(plan, ExperimentalCondensateProfileFixedSupportBatchPlan):
        raise TypeError(
            "plan must be an ExperimentalCondensateProfileFixedSupportBatchPlan."
        )
    base_arrays = run_experimental_profile_fixed_support_batch_plan(
        plan,
        element_inventory_target=element_inventory_target,
        rho_initialization=rho_initialization,
        lambda_initialization=lambda_initialization,
        residual_tolerance_multiplier=residual_tolerance_multiplier,
    )
    fallback_layer_indices = _fallback_layer_indices_from_fixed_support_arrays(
        base_arrays
    )
    rescue_plan, rescue_metadata = (
        _prepare_experimental_profile_fixed_support_prune_rescue_plan(
            plan,
            fallback_layer_indices,
            prune_relative_floors=prune_relative_floors,
        )
    )
    rescue = _ExperimentalProfileFixedSupportPruneRescuePlan(
        rescue_plan=rescue_plan,
        metadata=rescue_metadata,
    )
    if rescue.rescue_plan is None:
        return _attach_empty_fixed_support_rescue_metadata(base_arrays, rescue.metadata)
    rescue_arrays = run_experimental_profile_fixed_support_batch_plan(
        rescue.rescue_plan,
        element_inventory_target=_fixed_support_prune_rescue_single_target(
            element_inventory_target,
            rescue.metadata,
        ),
        rho_initialization=rho_initialization,
        lambda_initialization=lambda_initialization,
        residual_tolerance_multiplier=residual_tolerance_multiplier,
    )
    return _merge_fixed_support_prune_rescue_arrays(
        base_arrays,
        rescue_arrays,
        rescue.metadata,
    )


def run_experimental_profile_fixed_support_batch_plan_many(
    plan: ExperimentalCondensateProfileFixedSupportBatchPlan,
    element_inventory_targets: Array,
    *,
    rho_initialization: str = "unit_activity",
    lambda_initialization: str = "best_residual",
    residual_tolerance_multiplier: float = 1.0,
) -> Mapping[str, Array]:
    """Run a prepared fixed-support profile plan for multiple compositions.

    ``element_inventory_targets`` must have shape ``(n_eval, n_elements)`` for
    compositions shared by every layer, or ``(n_eval, n_layers, n_elements)`` for
    layer-specific compositions. Returned arrays have ``n_eval`` as the leading
    dimension.
    """

    if not isinstance(plan, ExperimentalCondensateProfileFixedSupportBatchPlan):
        raise TypeError(
            "plan must be an ExperimentalCondensateProfileFixedSupportBatchPlan."
        )
    targets = jnp.asarray(element_inventory_targets, dtype=jnp.float64)
    n_elements = int(plan.formula_matrix.shape[0])
    if targets.ndim == 2:
        if targets.shape[1] != n_elements:
            raise ValueError(
                "element_inventory_targets must have shape "
                "(n_eval, n_elements)."
            )
        layer_specific_targets = False
    elif targets.ndim == 3:
        if targets.shape[1:] != (plan.n_layers, n_elements):
            raise ValueError(
                "element_inventory_targets must have shape "
                f"(n_eval, {plan.n_layers}, {n_elements})."
            )
        layer_specific_targets = True
    else:
        raise ValueError(
            "element_inventory_targets must be two- or three-dimensional."
        )
    n_eval = int(targets.shape[0])
    if n_eval <= 0:
        raise ValueError("element_inventory_targets must contain at least one row.")

    from exogibbs.optimize.minimize_cond import (
        _solve_pdipm_rgie_v11_activity_correction_fixed_support_batch,
    )

    gas_species_count = int(plan.formula_matrix.shape[1])
    gas_ln_n = jnp.zeros(
        (n_eval, plan.n_layers, gas_species_count),
        dtype=jnp.float64,
    )
    condensate_amounts = jnp.zeros(
        (n_eval, plan.n_layers, plan.condensate_count),
        dtype=jnp.float64,
    )
    converged = jnp.zeros((n_eval, plan.n_layers), dtype=bool)
    final_residual = jnp.zeros((n_eval, plan.n_layers), dtype=jnp.float64)
    n_iter = jnp.zeros((n_eval, plan.n_layers), dtype=jnp.int32)
    step_diagnostics = {
        "accepted_iteration_count": jnp.zeros(
            (n_eval, plan.n_layers),
            dtype=jnp.int32,
        ),
        "normal_accepted_iteration_count": jnp.zeros(
            (n_eval, plan.n_layers),
            dtype=jnp.int32,
        ),
        "fallback_accepted_iteration_count": jnp.zeros(
            (n_eval, plan.n_layers),
            dtype=jnp.int32,
        ),
        "stationarity_restoration_accepted_iteration_count": jnp.zeros(
            (n_eval, plan.n_layers),
            dtype=jnp.int32,
        ),
        "initial_residual": jnp.zeros((n_eval, plan.n_layers), dtype=jnp.float64),
        "lambda_selection_index": jnp.zeros((n_eval, plan.n_layers), dtype=jnp.int32),
    }
    residual_components = {
        "gas": jnp.zeros((n_eval, plan.n_layers), dtype=jnp.float64),
        "condensate_stationarity": jnp.zeros(
            (n_eval, plan.n_layers),
            dtype=jnp.float64,
        ),
        "budget": jnp.zeros((n_eval, plan.n_layers), dtype=jnp.float64),
        "complementarity": jnp.zeros((n_eval, plan.n_layers), dtype=jnp.float64),
        "total_density": jnp.zeros((n_eval, plan.n_layers), dtype=jnp.float64),
    }

    for bucket, layer_indices in zip(plan.buckets, plan.bucket_layer_index_arrays):
        bucket_size = len(bucket.layer_indices)
        if layer_specific_targets:
            bucket_targets = targets[:, layer_indices, :]
        else:
            bucket_targets = jnp.broadcast_to(
                targets[:, None, :],
                (n_eval, bucket_size, n_elements),
            )
        flat_targets = jnp.reshape(bucket_targets, (n_eval * bucket_size, n_elements))
        flat_ln_nk_init = jnp.reshape(
            jnp.broadcast_to(
                bucket.ln_nk_init[None, :, :],
                (n_eval, bucket_size, bucket.ln_nk_init.shape[1]),
            ),
            (n_eval * bucket_size, bucket.ln_nk_init.shape[1]),
        )
        flat_ln_mk_init = jnp.reshape(
            jnp.broadcast_to(
                bucket.ln_mk_init[None, :, :],
                (n_eval, bucket_size, bucket.ln_mk_init.shape[1]),
            ),
            (n_eval * bucket_size, bucket.ln_mk_init.shape[1]),
        )
        flat_ln_ntot_init = jnp.reshape(
            jnp.broadcast_to(
                bucket.ln_ntot_init[None, :],
                (n_eval, bucket_size),
            ),
            (n_eval * bucket_size,),
        )
        flat_element_potential_init = None
        if bucket.element_potential_init is not None:
            flat_element_potential_init = jnp.reshape(
                jnp.broadcast_to(
                    bucket.element_potential_init[None, :, :],
                    (n_eval, bucket_size, bucket.element_potential_init.shape[1]),
                ),
                (n_eval * bucket_size, bucket.element_potential_init.shape[1]),
            )
        flat_rho_init = None
        if bucket.rho_init is not None:
            flat_rho_init = jnp.reshape(
                jnp.broadcast_to(
                    bucket.rho_init[None, :, :],
                    (n_eval, bucket_size, bucket.rho_init.shape[1]),
                ),
                (n_eval * bucket_size, bucket.rho_init.shape[1]),
            )
        flat_barrier_epsilon_init = None
        if bucket.barrier_epsilon_init is not None:
            flat_barrier_epsilon_init = jnp.reshape(
                jnp.broadcast_to(
                    bucket.barrier_epsilon_init[None, :],
                    (n_eval, bucket_size),
                ),
                (n_eval * bucket_size,),
            )
        flat_gas_stationarity_source_init = None
        if bucket.gas_stationarity_source_init is not None:
            flat_gas_stationarity_source_init = jnp.reshape(
                jnp.broadcast_to(
                    bucket.gas_stationarity_source_init[None, :, :],
                    (
                        n_eval,
                        bucket_size,
                        bucket.gas_stationarity_source_init.shape[1],
                    ),
                ),
                (n_eval * bucket_size, bucket.gas_stationarity_source_init.shape[1]),
            )
        flat_hvector = jnp.reshape(
            jnp.broadcast_to(
                bucket.hvector[None, :, :],
                (n_eval, bucket_size, bucket.hvector.shape[1]),
            ),
            (n_eval * bucket_size, bucket.hvector.shape[1]),
        )
        flat_hcond = jnp.reshape(
            jnp.broadcast_to(
                bucket.hvector_cond_active[None, :, :],
                (n_eval, bucket_size, bucket.hvector_cond_active.shape[1]),
            ),
            (n_eval * bucket_size, bucket.hvector_cond_active.shape[1]),
        )
        flat_pressure = jnp.reshape(
            jnp.broadcast_to(
                bucket.ln_normalized_pressure[None, :],
                (n_eval, bucket_size),
            ),
            (n_eval * bucket_size,),
        )
        batch_result, _batch_extra = (
            _solve_pdipm_rgie_v11_activity_correction_fixed_support_batch(
                ln_nk_init=flat_ln_nk_init,
                ln_mk_init=flat_ln_mk_init,
                ln_ntot_init=flat_ln_ntot_init,
                element_potential_init=flat_element_potential_init,
                rho_init=flat_rho_init,
                barrier_epsilon_init=flat_barrier_epsilon_init,
                gas_stationarity_source_init=flat_gas_stationarity_source_init,
                formula_matrix=plan.formula_matrix,
                formula_matrix_cond_active=bucket.formula_matrix_cond_active,
                element_inventory_target=flat_targets,
                hvector=flat_hvector,
                hvector_cond_active=flat_hcond,
                ln_normalized_pressure=flat_pressure,
                epsilon=-10.0,
                residual_tolerance_multiplier=residual_tolerance_multiplier,
                max_iter=plan.max_iter,
                rho_initialization=rho_initialization,
                lambda_initialization=lambda_initialization,
            )
        )
        batch_payload = _batch_extra[
            "pdipm_rgie_v11_activity_correction_fixed_support_batch"
        ]
        bucket_gas_ln_n = jnp.reshape(
            batch_result.ln_nk,
            (n_eval, bucket_size, gas_species_count),
        )
        support_amounts = jnp.reshape(
            jnp.exp(batch_result.ln_mk),
            (n_eval, bucket_size, len(bucket.support_indices)),
        )
        bucket_condensate_amounts = jnp.zeros(
            (n_eval, bucket_size, plan.condensate_count),
            dtype=support_amounts.dtype,
        ).at[:, :, jnp.asarray(bucket.support_indices, dtype=jnp.int32)].set(
            support_amounts
        )
        gas_ln_n = gas_ln_n.at[:, layer_indices, :].set(bucket_gas_ln_n)
        condensate_amounts = condensate_amounts.at[:, layer_indices, :].set(
            bucket_condensate_amounts
        )
        converged = converged.at[:, layer_indices].set(
            jnp.reshape(
                batch_result.diagnostics.converged,
                (n_eval, bucket_size),
            )
        )
        final_residual = final_residual.at[:, layer_indices].set(
            jnp.reshape(
                batch_result.diagnostics.final_residual,
                (n_eval, bucket_size),
            )
        )
        n_iter = n_iter.at[:, layer_indices].set(
            jnp.reshape(
                jnp.asarray(batch_result.diagnostics.n_iter, dtype=jnp.int32),
                (n_eval, bucket_size),
            )
        )
        component_sources = {
            "gas": "gas_residual_norm",
            "condensate_stationarity": "condensate_stationarity_residual_norm",
            "budget": "budget_residual_norm",
            "complementarity": "complementarity_residual_norm",
            "total_density": "total_density_residual_norm",
        }
        for component_name, payload_name in component_sources.items():
            residual_components[component_name] = residual_components[
                component_name
            ].at[:, layer_indices].set(
                jnp.reshape(
                    jnp.asarray(batch_payload[payload_name], dtype=jnp.float64),
                    (n_eval, bucket_size),
                )
            )
        step_sources = {
            "accepted_iteration_count": "accepted_iteration_count",
            "normal_accepted_iteration_count": "normal_accepted_iteration_count",
            "fallback_accepted_iteration_count": "fallback_accepted_iteration_count",
            "stationarity_restoration_accepted_iteration_count": (
                "stationarity_restoration_accepted_iteration_count"
            ),
            "initial_residual": "initial_residual",
            "lambda_selection_index": "lambda_selection_index",
        }
        for diagnostic_name, payload_name in step_sources.items():
            diagnostic_dtype = step_diagnostics[diagnostic_name].dtype
            step_diagnostics[diagnostic_name] = step_diagnostics[
                diagnostic_name
            ].at[:, layer_indices].set(
                jnp.reshape(
                    jnp.asarray(batch_payload[payload_name], dtype=diagnostic_dtype),
                    (n_eval, bucket_size),
                )
            )
    gas_n = jnp.exp(gas_ln_n)
    gas_ntot = jnp.sum(gas_n, axis=2)
    return {
        "gas_ln_n": gas_ln_n,
        "gas_n": gas_n,
        "gas_x": gas_n / jnp.clip(gas_ntot[:, :, None], 1.0e-300),
        "gas_ntot": gas_ntot,
        "condensate_amounts": condensate_amounts,
        "converged": converged,
        "fallback_required": ~converged,
        "final_residual": final_residual,
        "residual_components": residual_components,
        "step_diagnostics": step_diagnostics,
        "n_iter": n_iter,
        "lambda_candidate_labels": (
            "provided",
            "gas_lstsq",
            "gas_cond_lstsq",
            "damped_gas_lstsq",
            "damped_gas_cond_lstsq",
        ),
    }


def run_experimental_profile_fixed_support_batch_plan_many_with_prepared_fallback_rescue(
    plan: ExperimentalCondensateProfileFixedSupportBatchPlan,
    rescue: ExperimentalCondensateProfileFixedSupportPruneRescuePlan,
    element_inventory_targets: Array,
    *,
    rho_initialization: str = "complementarity",
    lambda_initialization: str = "best_residual",
    residual_tolerance_multiplier: float = 1.0e9,
) -> Mapping[str, Any]:
    """Run many fixed-support evaluations with a pre-built prune-rescue plan."""

    if not isinstance(plan, ExperimentalCondensateProfileFixedSupportBatchPlan):
        raise TypeError(
            "plan must be an ExperimentalCondensateProfileFixedSupportBatchPlan."
        )
    if not isinstance(rescue, ExperimentalCondensateProfileFixedSupportPruneRescuePlan):
        raise TypeError(
            "rescue must be an "
            "ExperimentalCondensateProfileFixedSupportPruneRescuePlan."
        )
    targets = jnp.asarray(element_inventory_targets, dtype=jnp.float64)
    base_arrays = run_experimental_profile_fixed_support_batch_plan_many(
        plan,
        targets,
        rho_initialization=rho_initialization,
        lambda_initialization=lambda_initialization,
        residual_tolerance_multiplier=residual_tolerance_multiplier,
    )
    if rescue.rescue_plan is None:
        return _attach_empty_fixed_support_rescue_metadata(
            base_arrays,
            rescue.metadata,
        )
    rescue_arrays = run_experimental_profile_fixed_support_batch_plan_many(
        rescue.rescue_plan,
        _fixed_support_prune_rescue_many_targets(targets, rescue.metadata),
        rho_initialization=rho_initialization,
        lambda_initialization=lambda_initialization,
        residual_tolerance_multiplier=residual_tolerance_multiplier,
    )
    return _merge_fixed_support_prune_rescue_arrays(
        base_arrays,
        rescue_arrays,
        rescue.metadata,
    )


def run_experimental_profile_fixed_support_batch_plan_many_with_cached_fallback_rescue(
    plan: ExperimentalCondensateProfileFixedSupportBatchPlan,
    cache: ExperimentalCondensateProfileFixedSupportPruneRescueCache,
    element_inventory_targets: Array,
    *,
    rho_initialization: str = "complementarity",
    lambda_initialization: str = "best_residual",
    residual_tolerance_multiplier: float = 1.0e9,
    prune_relative_floors: Sequence[float] = (1.0e-5, 1.0e-3),
) -> Mapping[str, Any]:
    """Run many fallback-rescue evaluations with cached rescue plans."""

    if not isinstance(plan, ExperimentalCondensateProfileFixedSupportBatchPlan):
        raise TypeError(
            "plan must be an ExperimentalCondensateProfileFixedSupportBatchPlan."
        )
    if not isinstance(cache, ExperimentalCondensateProfileFixedSupportPruneRescueCache):
        raise TypeError(
            "cache must be an "
            "ExperimentalCondensateProfileFixedSupportPruneRescueCache."
        )
    targets = jnp.asarray(element_inventory_targets, dtype=jnp.float64)
    base_arrays = run_experimental_profile_fixed_support_batch_plan_many(
        plan,
        targets,
        rho_initialization=rho_initialization,
        lambda_initialization=lambda_initialization,
        residual_tolerance_multiplier=residual_tolerance_multiplier,
    )
    rescue = _get_cached_fixed_support_prune_rescue_plan(
        plan,
        cache,
        _fallback_layer_indices_from_fixed_support_arrays(base_arrays),
        prune_relative_floors=prune_relative_floors,
    )
    if rescue.rescue_plan is None:
        return _attach_empty_fixed_support_rescue_metadata(
            base_arrays,
            rescue.metadata,
        )
    rescue_arrays = run_experimental_profile_fixed_support_batch_plan_many(
        rescue.rescue_plan,
        _fixed_support_prune_rescue_many_targets(targets, rescue.metadata),
        rho_initialization=rho_initialization,
        lambda_initialization=lambda_initialization,
        residual_tolerance_multiplier=residual_tolerance_multiplier,
    )
    return _merge_fixed_support_prune_rescue_arrays(
        base_arrays,
        rescue_arrays,
        rescue.metadata,
    )


def run_experimental_profile_fixed_support_batch_plan_many_with_fallback_rescue(
    plan: ExperimentalCondensateProfileFixedSupportBatchPlan,
    element_inventory_targets: Array,
    *,
    rho_initialization: str = "complementarity",
    lambda_initialization: str = "best_residual",
    residual_tolerance_multiplier: float = 1.0e9,
    prune_relative_floors: Sequence[float] = (1.0e-5, 1.0e-3),
) -> Mapping[str, Any]:
    """Run many fixed-support evaluations with fallback-only prune rescue."""

    if not isinstance(plan, ExperimentalCondensateProfileFixedSupportBatchPlan):
        raise TypeError(
            "plan must be an ExperimentalCondensateProfileFixedSupportBatchPlan."
        )
    targets = jnp.asarray(element_inventory_targets, dtype=jnp.float64)
    base_arrays = run_experimental_profile_fixed_support_batch_plan_many(
        plan,
        targets,
        rho_initialization=rho_initialization,
        lambda_initialization=lambda_initialization,
        residual_tolerance_multiplier=residual_tolerance_multiplier,
    )
    fallback_layer_indices = _fallback_layer_indices_from_fixed_support_arrays(
        base_arrays
    )
    rescue_plan, rescue_metadata = (
        _prepare_experimental_profile_fixed_support_prune_rescue_plan(
            plan,
            fallback_layer_indices,
            prune_relative_floors=prune_relative_floors,
        )
    )
    if rescue_plan is None:
        return _attach_empty_fixed_support_rescue_metadata(
            base_arrays,
            rescue_metadata,
        )
    rescue_arrays = run_experimental_profile_fixed_support_batch_plan_many(
        rescue_plan,
        _fixed_support_prune_rescue_many_targets(targets, rescue_metadata),
        rho_initialization=rho_initialization,
        lambda_initialization=lambda_initialization,
        residual_tolerance_multiplier=residual_tolerance_multiplier,
    )
    return _merge_fixed_support_prune_rescue_arrays(
        base_arrays,
        rescue_arrays,
        rescue_metadata,
    )


def condensate_equilibrium_profile(
    setup: CondensateChemicalSetup,
    T: Sequence[float] | Array,
    P: Sequence[float] | Array,
    b: Array,
    *,
    Pref: float = 1.0,
    support_indices: Optional[Sequence[int]] = None,
    support_amounts_init: Optional[Sequence[float]] = None,
    init: Optional[Sequence[CondensateEquilibriumInit | None]] = None,
    initializer: Optional[CondensateEquilibriumInitializer] = None,
    options: Optional[CondensateEquilibriumOptions] = None,
    method: Optional[CondensateProfileMethod] = None,
    return_diagnostics: bool = False,
) -> CondensateEquilibriumProfileResult:
    """Compute condensate equilibrium over a 1D T/P profile.

    ``method="vmap_cold"`` treats each layer independently except for any
    initializer-provided guess. ``scan_hot_from_top`` and
    ``scan_hot_from_bottom`` carry the previous accepted layer result as the
    next layer's initializer. Any failed warm-start attempt falls back to the
    standard fresh one-layer route for that layer.
    """

    validate_condensate_chemical_setup(setup)
    temperatures = np.asarray(T, dtype=np.float64)
    pressures = np.asarray(P, dtype=np.float64)
    if temperatures.ndim != 1 or pressures.ndim != 1:
        raise ValueError("T and P must be 1D arrays of equal length.")
    if temperatures.shape[0] != pressures.shape[0]:
        raise ValueError("T and P must have the same length.")
    opts = options or CondensateEquilibriumOptions()
    _validate_options(opts)
    n_layers = int(temperatures.shape[0])
    explicit_inits: tuple[CondensateEquilibriumInit | None, ...]
    if init is None:
        explicit_inits = (None,) * n_layers
    else:
        explicit_inits = tuple(init)
        if len(explicit_inits) != n_layers:
            raise ValueError("init must have one entry per profile layer.")
    requested_method = method if method is not None else opts.profile_method
    auto_method = requested_method is None or requested_method == "auto"
    has_fixed_support_payload = _profile_has_complete_fixed_support_payload(
        explicit_inits,
        support_indices=support_indices,
        support_amounts_init=support_amounts_init,
    )
    resolved_method = _resolve_condensate_profile_method(
        requested_method,
        initializer,
        has_fixed_support_payload=has_fixed_support_payload,
    )
    valid_methods = ("vmap_cold", "scan_hot_from_top", "scan_hot_from_bottom")
    if resolved_method not in valid_methods:
        raise ValueError(
            f"Unknown condensate profile solve method '{resolved_method}'. "
            f"Expected one of {valid_methods} or 'auto'."
        )
    if auto_method and resolved_method == "vmap_cold" and has_fixed_support_payload:
        opts = replace(
            opts,
            profile_method="auto",
            profile_warm_start_support_policy="explicit_payload",
            enable_experimental_profile_fixed_support_batch=True,
            enable_experimental_profile_fixed_support_fallback_rescue=True,
        )

    if resolved_method == "vmap_cold":
        experimental_batch_result = _run_experimental_profile_fixed_support_batch(
            setup=setup,
            temperatures=temperatures,
            pressures=pressures,
            b=b,
            Pref=Pref,
            explicit_inits=explicit_inits,
            initializer=initializer,
            support_indices=support_indices,
            support_amounts_init=support_amounts_init,
            opts=opts,
            return_diagnostics=return_diagnostics,
        )
        if experimental_batch_result is not None:
            return experimental_batch_result

    if resolved_method == "scan_hot_from_bottom":
        layer_order = tuple(reversed(range(n_layers)))
    else:
        layer_order = tuple(range(n_layers))

    previous_solution: CondensateEquilibriumInit | None = None
    layer_results: dict[int, CondensateEquilibriumResult] = {}
    layer_reports: dict[int, Mapping[str, Any]] = {}
    for layer_index in layer_order:
        initial_guess = _resolve_condensate_initial_guess(
            initializer,
            CondensateEquilibriumInitRequest(
                setup=setup,
                T=float(temperatures[layer_index]),
                P=float(pressures[layer_index]),
                b=b,
                Pref=Pref,
                layer_index=layer_index,
                user_init=explicit_inits[layer_index],
                previous_solution=(
                    None if resolved_method == "vmap_cold" else previous_solution
                ),
            ),
        )
        result, report = _run_condensate_profile_layer(
            setup=setup,
            T=float(temperatures[layer_index]),
            P=float(pressures[layer_index]),
            b=b,
            Pref=Pref,
            layer_index=layer_index,
            base_options=opts,
            init=initial_guess,
            support_indices=support_indices,
            support_amounts_init=support_amounts_init,
            warm_start_support_policy=opts.profile_warm_start_support_policy,
            return_diagnostics=return_diagnostics,
        )
        layer_results[layer_index] = result
        layer_reports[layer_index] = report
        previous_solution = _condensate_init_from_result(
            result,
            min_seed_amount=opts.min_seed_amount,
        )

    ordered_results = tuple(layer_results[index] for index in range(n_layers))
    ordered_reports = tuple(layer_reports[index] for index in range(n_layers))
    diagnostics = None
    if return_diagnostics:
        diagnostics = {
            "profile_schema": "exogibbs_condensate_equilibrium_profile_v1",
            "method": resolved_method,
            "layer_count": n_layers,
            "warm_start_attempt_count": sum(
                bool(report.get("warm_start_attempted", False))
                for report in ordered_reports
            ),
            "fresh_fallback_count": sum(
                bool(report.get("fresh_fallback_used", False))
                for report in ordered_reports
            ),
            "layers": ordered_reports,
        }
    return CondensateEquilibriumProfileResult(
        layers=ordered_results,
        method=resolved_method,
        diagnostics=diagnostics,
    )


__all__ = (
    "CondensateChemicalSetup",
    "CondensateEquilibriumInit",
    "CondensateEquilibriumInitRequest",
    "CondensateEquilibriumInitializer",
    "CondensateEquilibriumOptions",
    "CondensateEquilibriumProfileResult",
    "CondensateEquilibriumResult",
    "CondensateProfileMethod",
    "CondensateProfileWarmStartSupportPolicy",
    "DefaultCondensateEquilibriumInitializer",
    "ExperimentalCondensateProfileFixedSupportBatchPlan",
    "ExperimentalCondensateProfileFixedSupportPruneRescueCache",
    "ExperimentalCondensateProfileFixedSupportPruneRescuePlan",
    "build_condensate_chemical_setup",
    "build_condensate_equilibrium_result_from_solver_payload",
    "condensate_equilibrium",
    "condensate_equilibrium_profile",
    "prepare_experimental_profile_fixed_support_batch_plan",
    "prepare_experimental_profile_fixed_support_prune_rescue_plan",
    "run_experimental_profile_fixed_support_batch_plan",
    "run_experimental_profile_fixed_support_batch_plan_with_cached_fallback_rescue",
    "run_experimental_profile_fixed_support_batch_plan_with_prepared_fallback_rescue",
    "run_experimental_profile_fixed_support_batch_plan_with_fallback_rescue",
    "run_experimental_profile_fixed_support_batch_plan_many",
    "run_experimental_profile_fixed_support_batch_plan_many_with_cached_fallback_rescue",
    "run_experimental_profile_fixed_support_batch_plan_many_with_prepared_fallback_rescue",
    "run_experimental_profile_fixed_support_batch_plan_many_with_fallback_rescue",
    "validate_condensate_chemical_setup",
)
