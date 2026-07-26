"""Backward-compatible import path and structured API for condensate minimization."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from time import perf_counter
from typing import Any, Literal, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, tree_util
from scipy.optimize import least_squares

from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.core import _compute_gk
from exogibbs.optimize.fixed_support_batch import (
    FIXED_SUPPORT_BATCH_DEFAULT_EPSILON_SCHEDULE,
    FIXED_SUPPORT_BATCH_LAMBDA_INITIALIZATIONS,
    build_ipopt_current_iterate_filter_mask,
    build_fixed_support_batch_metadata,
)
from exogibbs.optimize.fixed_support_convergence import (
    fixed_support_batch_converged,
    fixed_support_budget_relative_max,
)
from exogibbs.optimize.fixed_support_charge import (
    retract_fixed_support_charge_neutrality,
)
from exogibbs.optimize.fixed_support_filter import (
    fixed_support_filter_acceptance,
    prepare_fixed_support_restoration_filter,
    update_fixed_support_filter,
)
from exogibbs.optimize.fixed_support_dual_step import (
    fixed_support_min_equality_dual_infeasibility_step,
)
from exogibbs.optimize.fixed_support_ipopt_soc import (
    fixed_support_reduced_direction_from_rhs_with_diagnostics,
    fixed_support_soc_constraint_rhs,
    fixed_support_soc_trial_from_current,
)
from exogibbs.optimize.fixed_support_kkt import (
    fixed_support_barrier_objective,
    fixed_support_barrier_objective_linearized_change,
    fixed_support_filter_theta,
    fixed_support_full_newton_linearized_residual,
)
from exogibbs.optimize.fixed_support_soc import (
    fixed_support_soc_correction_direction,
)
from exogibbs.optimize.fixed_support_restoration import (
    fixed_support_amount_space_restoration,
    fixed_support_full_restoration,
    fixed_support_ipopt_restoration_dual_return,
    fixed_support_restoration_phase_exit,
    fixed_support_restoration_phase_transition,
)
from exogibbs.optimize.stepsize import LOG_S_MAX
from exogibbs.optimize.pdipm_cond import minimize_gibbs_cond_core
from exogibbs.optimize.minimize import (
    build_minimize_gibbs_core_lnnk_output_source_trace,
    minimize_gibbs_core,
    minimize_gibbs_core_with_source_trace,
)

Array = jax.Array


def _call_pipm_rgie(symbol_name: str, *args, **kwargs):
    """Call one legacy PIPM/RGIE symbol without importing it at module load."""

    from exogibbs.optimize import pipm_rgie_cond

    return getattr(pipm_rgie_cond, symbol_name)(*args, **kwargs)


def _recompute_pi_for_residual(*args, **kwargs):
    return _call_pipm_rgie("_recompute_pi_for_residual", *args, **kwargs)


def build_rgie_condensate_init_from_policy(*args, **kwargs):
    return _call_pipm_rgie(
        "build_rgie_condensate_init_from_policy",
        *args,
        **kwargs,
    )


def compute_condensate_budget_limits(*args, **kwargs):
    return _call_pipm_rgie("compute_condensate_budget_limits", *args, **kwargs)


def select_conditional_capped_s_reduced_coupling_mode(*args, **kwargs):
    return _call_pipm_rgie(
        "select_conditional_capped_s_reduced_coupling_mode",
        *args,
        **kwargs,
    )


def summarize_rgie_inactive_driving(*args, **kwargs):
    return _call_pipm_rgie("summarize_rgie_inactive_driving", *args, **kwargs)


def _diagnose_full_vs_reduced_gie_direction_raw(*args, **kwargs):
    return _call_pipm_rgie(
        "diagnose_full_vs_reduced_gie_direction",
        *args,
        **kwargs,
    )


def _diagnose_pdipm_vs_pipm_direction_raw(*args, **kwargs):
    return _call_pipm_rgie(
        "diagnose_pdipm_vs_pipm_direction",
        *args,
        **kwargs,
    )


def _diagnose_pdipm_vs_pipm_fixed_epsilon_trajectories_raw(*args, **kwargs):
    return _call_pipm_rgie(
        "diagnose_pdipm_vs_pipm_fixed_epsilon_trajectories",
        *args,
        **kwargs,
    )


def _diagnose_reduced_solver_backend_experiments_raw(*args, **kwargs):
    return _call_pipm_rgie(
        "diagnose_reduced_solver_backend_experiments",
        *args,
        **kwargs,
    )


def _diagnose_gas_step_limiter_and_direction_raw(*args, **kwargs):
    return _call_pipm_rgie(
        "diagnose_gas_step_limiter_and_direction",
        *args,
        **kwargs,
    )


def _diagnose_iteration_lambda_trials_raw(*args, **kwargs):
    return _call_pipm_rgie(
        "diagnose_iteration_lambda_trials",
        *args,
        **kwargs,
    )


def _minimize_gibbs_cond_with_diagnostics_raw(*args, **kwargs):
    return _call_pipm_rgie(
        "minimize_gibbs_cond_with_diagnostics",
        *args,
        **kwargs,
    )


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
    use_external_gas_stationarity_source: jnp.ndarray,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond_active: jnp.ndarray,
    element_inventory_target: jnp.ndarray,
    hvector: jnp.ndarray,
    hvector_cond_active: jnp.ndarray,
    ln_normalized_pressure: jnp.ndarray,
    epsilon: jnp.ndarray,
    residual_tolerance_multiplier: jnp.ndarray,
    budget_relative_acceptance_floor: jnp.ndarray,
    budget_direction_projection_strength: jnp.ndarray,
    convergence_log_tolerance: jnp.ndarray,
    convergence_budget_relative_tolerance: jnp.ndarray,
    convergence_budget_relative_floor: jnp.ndarray,
    convergence_total_density_tolerance: jnp.ndarray,
    relaxed_stationarity_fallback_enabled: jnp.ndarray,
    relaxed_stationarity_fallback_factor: jnp.ndarray,
    adaptive_regularization_enabled: jnp.ndarray,
    adaptive_regularization_base: jnp.ndarray,
    second_order_correction_enabled: jnp.ndarray,
    second_order_correction_max_abs_step: jnp.ndarray,
    second_order_correction_interleave: jnp.ndarray,
    second_order_correction_budget_passes: int,
    second_order_correction_dual_repair: jnp.ndarray,
    second_order_correction_policy: str,
    second_order_correction_kappa_soc: jnp.ndarray,
    second_order_correction_alpha_y_policy: str,
    second_order_correction_charge_solve_policy: str,
    second_order_correction_reduced_mode_policy: str,
    second_order_correction_diagnostic_mode_vector_policy: str,
    budget_restoration_enabled: bool,
    budget_restoration_coordinate_policy: str,
    budget_restoration_dual_recenter: jnp.ndarray,
    budget_restoration_dual_recenter_policy: str,
    budget_restoration_proximity_weight: jnp.ndarray,
    budget_restoration_max_abs_step: jnp.ndarray,
    budget_restoration_passes: int,
    budget_restoration_phase_enabled: bool,
    budget_restoration_phase_theta_reduction: jnp.ndarray,
    budget_restoration_phase_cooldown_iterations: int,
    ipopt_filter_acceptance_enabled: jnp.ndarray,
    ipopt_filter_budget_relative_max: jnp.ndarray,
    ipopt_filter_policy: str,
    ipopt_filter_use_l1_theta: jnp.ndarray,
    line_search_candidate_selection_policy: str,
    use_legacy_capacity_epsilon: jnp.ndarray,
    use_scalar_step_control: jnp.ndarray,
    use_log_amount_boundary: jnp.ndarray,
    use_log_activity_boundary: jnp.ndarray,
    max_iter: int,
    tiny_step_consecutive_limit: int,
    rho_initialization: str = "unit_activity",
    lambda_initialization: str = "best_residual",
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

    def scaled_damped_lstsq(
        matrix: jnp.ndarray,
        rhs: jnp.ndarray,
    ) -> jnp.ndarray:
        column_scale = jnp.maximum(
            jnp.linalg.norm(matrix, axis=0),
            jnp.asarray(1.0e-300, dtype=matrix.dtype),
        )
        scaled_matrix = matrix / column_scale[None, :]
        normal_matrix = scaled_matrix.T @ scaled_matrix
        normal_rhs = scaled_matrix.T @ rhs
        mean_diagonal = jnp.mean(jnp.diag(normal_matrix))
        damping = jnp.maximum(
            jnp.asarray(1.0e-12, dtype=matrix.dtype) * mean_diagonal,
            jnp.asarray(1.0e-30, dtype=matrix.dtype),
        )
        solution_scaled = jnp.linalg.solve(
            normal_matrix + damping * jnp.eye(normal_matrix.shape[0], dtype=matrix.dtype),
            normal_rhs,
        )
        solution = solution_scaled / column_scale
        return jnp.nan_to_num(solution, nan=0.0, posinf=0.0, neginf=0.0)

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
        use_external_gas_source: jnp.ndarray,
        use_scalar_step: jnp.ndarray,
        residual_crit: jnp.ndarray,
        budget_relative_crit: jnp.ndarray,
        total_density_crit: jnp.ndarray,
        initial_filter_theta: jnp.ndarray,
        filter_phi_entries: jnp.ndarray,
        filter_theta_entries: jnp.ndarray,
        filter_valid_entries: jnp.ndarray,
        consecutive_filter_rejection_count: jnp.ndarray,
        filter_reset_count: jnp.ndarray,
        restoration_phase_active: jnp.ndarray,
        restoration_can_enter: jnp.ndarray,
        restoration_reference_q: jnp.ndarray,
        restoration_reference_r: jnp.ndarray,
        restoration_reference_qtot: jnp.ndarray,
    ) -> tuple[jnp.ndarray, ...]:
        gas_stationarity_source = jnp.where(
            use_external_gas_source,
            hgas_or_gas_stationarity_source,
            hgas_or_gas_stationarity_source + ln_pressure - qtot,
        )
        jac_mask = jnp.ones((r.shape[0],), dtype=bool)
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
        rhs_top = ag @ (n * geff) + ac @ (j_vec * hcond + m * t_vec - m) + delta_bhat
        rhs_bottom = jnp.dot(n, geff) - delta_ntot
        rhs = jnp.concatenate([rhs_top, jnp.asarray([rhs_bottom], dtype=qhat.dtype)])

        mean_qhat_diagonal = jnp.maximum(
            jnp.mean(jnp.diag(qhat)),
            jnp.asarray(1.0, dtype=qhat.dtype),
        )

        def algorithm_solution_for_regularization(
            absolute_regularization: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            regularization = absolute_regularization.astype(qhat.dtype)
            qhat_reg = qhat + regularization * jnp.eye(qhat.shape[0], dtype=qhat.dtype)
            matrix = jnp.block(
                [
                    [qhat_reg, gas_inventory[:, None]],
                    [
                        gas_inventory[None, :],
                        jnp.asarray([[delta_ntot]], dtype=qhat.dtype),
                    ],
                ]
            )
            solution = jnp.linalg.lstsq(matrix, rhs, rcond=None)[0]
            solution = jnp.nan_to_num(solution, nan=0.0, posinf=0.0, neginf=0.0)
            unregularized_matrix = jnp.block(
                [
                    [qhat, gas_inventory[:, None]],
                    [
                        gas_inventory[None, :],
                        jnp.asarray([[delta_ntot]], dtype=qhat.dtype),
                    ],
                ]
            )
            linear_residual = unregularized_matrix @ solution - rhs
            score = l2(linear_residual) + jnp.asarray(1.0e-18, dtype=qhat.dtype) * l2(
                solution
            )
            score = jnp.where(jnp.all(jnp.isfinite(solution)), score, jnp.inf)
            return solution, score

        base_reg = jnp.maximum(
            adaptive_regularization_base.astype(qhat.dtype),
            jnp.asarray(1.0e-14, dtype=qhat.dtype),
        )
        adaptive_reg_grid = jnp.asarray(
            (1.0e-14, 1.0, 1.0e2, 1.0e4),
            dtype=qhat.dtype,
        ) * base_reg * mean_qhat_diagonal
        disabled_reg_grid = jnp.full_like(
            adaptive_reg_grid,
            jnp.asarray(1.0e-14, dtype=qhat.dtype),
        )
        reg_grid = jnp.where(
            adaptive_regularization_enabled,
            adaptive_reg_grid,
            disabled_reg_grid,
        )
        candidate_solutions, candidate_scores = jax.vmap(
            algorithm_solution_for_regularization
        )(reg_grid)
        selected_regularization_index = jnp.asarray(
            jnp.argmin(candidate_scores),
            dtype=jnp.int32,
        )
        solution = candidate_solutions[selected_regularization_index]
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

        def budget_project_amount_direction(
            direction_q: jnp.ndarray,
            direction_r: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            budget_jacobian = jnp.concatenate(
                [
                    ag * n[None, :],
                    ac * m[None, :],
                ],
                axis=1,
            )
            direction_amount = jnp.concatenate([direction_q, direction_r], axis=0)
            direction_budget = budget_jacobian @ direction_amount
            correction_rhs = delta_bhat - direction_budget
            gram = budget_jacobian @ budget_jacobian.T
            mean_diagonal = jnp.maximum(
                jnp.mean(jnp.diag(gram)),
                jnp.asarray(1.0, dtype=gram.dtype),
            )
            damping = jnp.asarray(1.0e-14, dtype=gram.dtype) * mean_diagonal
            correction_dual = jnp.linalg.solve(
                gram + damping * jnp.eye(gram.shape[0], dtype=gram.dtype),
                correction_rhs,
            )
            correction = budget_jacobian.T @ correction_dual
            q_end = direction_q.shape[0]
            projected_q = jnp.clip(
                direction_q
                + budget_direction_projection_strength.astype(direction_q.dtype)
                * correction[:q_end],
                -2.0,
                2.0,
            )
            projected_r = jnp.clip(
                direction_r
                + budget_direction_projection_strength.astype(direction_r.dtype)
                * correction[q_end:],
                -5.0,
                5.0,
            )
            use_projection = budget_direction_projection_strength > 0.0
            return (
                jnp.where(use_projection, projected_q, direction_q),
                jnp.where(use_projection, projected_r, direction_r),
            )

        def joint_stationarity_restoration_direction() -> tuple[jnp.ndarray, ...]:
            q_size = q.shape[0]
            r_size = r.shape[0]
            lam_size = lam.shape[0]
            sqrt_budget_weight = jnp.sqrt(jnp.asarray(30.0, dtype=q.dtype))
            sqrt_total_weight = jnp.sqrt(jnp.asarray(30.0, dtype=q.dtype))
            gas_residual = q + gas_stationarity_source - ag.T @ lam
            cond_residual = hcond - ac.T @ lam - eta
            budget_residual = ag @ n + ac @ m - target
            ntot = jnp.exp(qtot)
            positive_target = jnp.where(target > 0.0, target, 0.0)
            target_scale = jnp.maximum(
                jnp.max(positive_target, initial=jnp.asarray(0.0, dtype=q.dtype)),
                jnp.asarray(1.0, dtype=q.dtype),
            )
            budget_floor = jnp.maximum(
                jnp.asarray(jnp.finfo(q.dtype).tiny, dtype=q.dtype),
                jnp.asarray(1.0e-300, dtype=q.dtype) * target_scale,
            )
            row_weights = jnp.where(
                target > 0.0,
                1.0 / jnp.maximum(jnp.abs(target), budget_floor),
                0.0,
            )
            row_weights = jnp.where(jnp.isfinite(row_weights), row_weights, 0.0)

            budget_rows = jnp.concatenate(
                [
                    sqrt_budget_weight * row_weights[:, None] * (ag * n[None, :]),
                    sqrt_budget_weight * row_weights[:, None] * (ac * m[None, :]),
                    jnp.zeros((target.shape[0], lam_size), dtype=q.dtype),
                    jnp.zeros((target.shape[0], r_size), dtype=q.dtype),
                    jnp.zeros((target.shape[0], 1), dtype=q.dtype),
                ],
                axis=1,
            )
            budget_rhs = -sqrt_budget_weight * row_weights * budget_residual
            total_row = jnp.concatenate(
                [
                    sqrt_total_weight * n,
                    jnp.zeros((r_size,), dtype=q.dtype),
                    jnp.zeros((lam_size,), dtype=q.dtype),
                    jnp.zeros((r_size,), dtype=q.dtype),
                    jnp.asarray([-sqrt_total_weight * ntot], dtype=q.dtype),
                ],
                axis=0,
            )[None, :]
            total_rhs = jnp.asarray(
                [-sqrt_total_weight * (jnp.sum(n) - ntot)],
                dtype=q.dtype,
            )
            gas_rows = jnp.concatenate(
                [
                    jnp.diag(n * (gas_residual + 1.0)),
                    jnp.zeros((q_size, r_size), dtype=q.dtype),
                    -(n[:, None] * ag.T),
                    jnp.zeros((q_size, r_size), dtype=q.dtype),
                    -n[:, None],
                ],
                axis=1,
            )
            gas_rhs = -n * gas_residual
            cond_rows = jnp.concatenate(
                [
                    jnp.zeros((r_size, q_size), dtype=q.dtype),
                    jnp.diag(m * cond_residual),
                    -(m[:, None] * ac.T),
                    jnp.diag(-m * eta),
                    jnp.zeros((r_size, 1), dtype=q.dtype),
                ],
                axis=1,
            )
            cond_rhs = -m * cond_residual
            restoration_matrix = jnp.concatenate(
                [budget_rows, total_row, gas_rows, cond_rows],
                axis=0,
            )
            restoration_rhs = jnp.concatenate(
                [budget_rhs, total_rhs, gas_rhs, cond_rhs],
                axis=0,
            )
            restoration_solution = jnp.linalg.lstsq(
                restoration_matrix,
                restoration_rhs,
                rcond=None,
            )[0]
            restoration_solution = jnp.nan_to_num(
                restoration_solution,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            q_end = q_size
            r_end = q_end + r_size
            lam_end = r_end + lam_size
            rho_end = lam_end + r_size
            rdq = jnp.clip(restoration_solution[:q_end], -2.0, 2.0)
            rdr = jnp.clip(restoration_solution[q_end:r_end], -5.0, 5.0)
            rdlam = jnp.clip(restoration_solution[r_end:lam_end], -100.0, 100.0)
            rdrho = jnp.clip(restoration_solution[lam_end:rho_end], -5.0, 5.0)
            rdqtot = restoration_solution[-1]
            return rdq, rdr, rdlam, rdrho, rdqtot

        (
            restoration_delta_q,
            restoration_delta_r,
            restoration_delta_lam,
            restoration_delta_rho,
            restoration_delta_qtot,
        ) = joint_stationarity_restoration_direction()
        delta_q, delta_r = budget_project_amount_direction(delta_q, delta_r)
        restoration_delta_q, restoration_delta_r = budget_project_amount_direction(
            restoration_delta_q,
            restoration_delta_r,
        )

        def direction_alpha_boundary(
            direction_r: jnp.ndarray,
            direction_rho: jnp.ndarray,
        ) -> jnp.ndarray:
            bounded_alpha_r = jnp.min(
                jnp.where(direction_r < 0.0, -1.0 / direction_r, 1.0),
                initial=jnp.asarray(1.0, dtype=jnp.float64),
            )
            local_alpha_r = jnp.where(
                use_log_amount_boundary,
                bounded_alpha_r,
                jnp.asarray(1.0, dtype=jnp.float64),
            )
            bounded_alpha_rho = jnp.min(
                jnp.where(direction_rho < 0.0, -1.0 / direction_rho, 1.0),
                initial=jnp.asarray(1.0, dtype=jnp.float64),
            )
            local_alpha_rho = jnp.where(
                use_log_activity_boundary,
                bounded_alpha_rho,
                jnp.asarray(1.0, dtype=jnp.float64),
            )
            return jnp.minimum(
                1.0,
                0.995 * jnp.minimum(local_alpha_r, local_alpha_rho),
            )
        bounded_alpha_r = jnp.min(
            jnp.where(delta_r < 0.0, -1.0 / delta_r, 1.0),
            initial=jnp.asarray(1.0, dtype=jnp.float64),
        )
        alpha_r = jnp.where(
            use_log_amount_boundary,
            bounded_alpha_r,
            jnp.asarray(1.0, dtype=jnp.float64),
        )
        bounded_alpha_rho = jnp.min(
            jnp.where(delta_rho < 0.0, -1.0 / delta_rho, 1.0),
            initial=jnp.asarray(1.0, dtype=jnp.float64),
        )
        alpha_rho = jnp.where(
            use_log_activity_boundary,
            bounded_alpha_rho,
            jnp.asarray(1.0, dtype=jnp.float64),
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

        def charge_retract_trial(
            trial_q: jnp.ndarray,
            trial_lam: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            if second_order_correction_charge_solve_policy == "neutrality_retraction":
                retracted_q, retracted_lam, _charge, _susceptibility = (
                    retract_fixed_support_charge_neutrality(
                        log_gas_amounts=trial_q,
                        element_potential=trial_lam,
                        charge_coefficients=ag[-1],
                    )
                )
                return retracted_q, retracted_lam
            return trial_q, trial_lam

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
        initial_gas, initial_cond, initial_budget, initial_comp, initial_total = (
            residual_components(q, r, lam, rho, qtot)
        )
        initial_gas_norm = l2(initial_gas)
        initial_cond_norm = l2(initial_cond)
        initial_budget_norm = l2(initial_budget)
        initial_comp_norm = l2(initial_comp)
        initial_total_norm = l2(initial_total)
        full_newton_linearized_residual_norm = l2(
            fixed_support_full_newton_linearized_residual(
                formula_matrix=ag,
                formula_matrix_cond_active=ac,
                q=q,
                r=r,
                rho=rho,
                qtot=qtot,
                gas_residual=initial_gas,
                condensate_stationarity_residual=initial_cond,
                budget_residual=initial_budget,
                complementarity_residual=initial_comp,
                total_density_residual=initial_total,
                delta_q=raw_delta_q,
                delta_r=raw_delta_r,
                delta_lambda=raw_delta_lam,
                delta_rho=raw_delta_rho,
                delta_qtot=delta_qtot,
            )
        )

        def relative_budget_max_abs(budget: jnp.ndarray) -> jnp.ndarray:
            return fixed_support_budget_relative_max(
                budget_residual=budget,
                target=target,
                relative_floor=jnp.maximum(
                    jnp.asarray(jnp.finfo(budget.dtype).tiny, dtype=budget.dtype),
                    convergence_budget_relative_floor.astype(budget.dtype),
                ),
            )

        initial_budget_relative_max = relative_budget_max_abs(initial_budget)
        def trial_for_direction(
            alpha: jnp.ndarray,
            direction_q: jnp.ndarray,
            direction_r: jnp.ndarray,
            direction_lam: jnp.ndarray,
            direction_rho: jnp.ndarray,
            direction_qtot: jnp.ndarray,
        ) -> tuple[jnp.ndarray, ...]:
            tq = q + alpha * direction_q
            tr = jnp.minimum(r + alpha * direction_r, r_cap)
            tlam = lam + alpha * direction_lam
            tq, tlam = charge_retract_trial(tq, tlam)
            trho = rho + alpha * direction_rho
            tqtot = qtot + alpha * direction_qtot
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
                relative_budget_max_abs(budget),
                l2(comp),
                l2(_total),
            )

        def single_second_order_kkt_correction(
            tq_in: jnp.ndarray,
            tr_in: jnp.ndarray,
            tlam_in: jnp.ndarray,
            trho_in: jnp.ndarray,
            tqtot_in: jnp.ndarray,
        ) -> tuple[jnp.ndarray, ...]:
            trial_n = jnp.exp(tq_in)
            trial_m = jnp.exp(tr_in)
            trial_ntot = jnp.exp(tqtot_in)
            trial_budget = ag @ trial_n + ac @ trial_m - target
            trial_total = jnp.sum(trial_n) - trial_ntot
            displacement_q = tq_in - q
            displacement_r = tr_in - r
            displacement_qtot = tqtot_in - qtot
            linearized_budget = (
                initial_budget
                + ag @ (n * displacement_q)
                + ac @ (m * displacement_r)
            )
            linearized_total = (
                initial_total[0]
                + jnp.dot(n, displacement_q)
                - jnp.exp(qtot) * displacement_qtot
            )
            budget_defect = trial_budget - linearized_budget
            total_defect = trial_total - linearized_total
            dq_soc, dr_soc, dlam_soc, drho_soc, dqtot_soc = (
                fixed_support_soc_correction_direction(
                    formula_matrix=ag,
                    formula_matrix_cond_active=ac,
                    gas_amounts=n,
                    condensate_amounts=m,
                    condensate_duals=eta,
                    gas_inventory=gas_inventory,
                    total_density_residual=delta_ntot,
                    budget_defect=budget_defect,
                    total_density_defect=total_defect,
                    max_abs_primal_step=second_order_correction_max_abs_step,
                )
            )
            return (
                tq_in + dq_soc,
                tr_in + dr_soc,
                tlam_in + dlam_soc,
                trho_in + drho_soc,
                tqtot_in + dqtot_soc,
            )

        def legacy_budget_projection_correction(
            tq_in: jnp.ndarray,
            tr_in: jnp.ndarray,
            tqtot_in: jnp.ndarray,
        ) -> tuple[jnp.ndarray, ...]:
            trial_n = jnp.exp(tq_in)
            trial_m = jnp.exp(tr_in)
            trial_ntot = jnp.exp(tqtot_in)
            trial_budget = ag @ trial_n + ac @ trial_m - target
            trial_total = jnp.sum(trial_n) - trial_ntot
            row_scale = jnp.where(
                target > 0.0,
                1.0
                / jnp.maximum(
                    jnp.abs(target),
                    convergence_budget_relative_floor.astype(tq_in.dtype),
                ),
                0.0,
            )
            budget_jacobian = jnp.concatenate(
                [
                    row_scale[:, None] * (ag * trial_n[None, :]),
                    row_scale[:, None] * (ac * trial_m[None, :]),
                    jnp.zeros((target.shape[0], 1), dtype=tq_in.dtype),
                ],
                axis=1,
            )
            total_jacobian = jnp.concatenate(
                [
                    trial_n,
                    jnp.zeros_like(trial_m),
                    jnp.asarray([-trial_ntot], dtype=tq_in.dtype),
                ]
            )[None, :]
            correction_matrix = jnp.concatenate(
                [budget_jacobian, total_jacobian], axis=0
            )
            correction_rhs = -jnp.concatenate(
                [row_scale * trial_budget, jnp.asarray([trial_total])]
            )
            column_scale = jnp.maximum(
                jnp.linalg.norm(correction_matrix, axis=0),
                jnp.asarray(1.0e-300, dtype=tq_in.dtype),
            )
            scaled_matrix = correction_matrix / column_scale[None, :]
            normal_matrix = scaled_matrix.T @ scaled_matrix
            damping = jnp.asarray(1.0e-12, dtype=tq_in.dtype) * jnp.maximum(
                jnp.mean(jnp.diag(normal_matrix)),
                jnp.asarray(1.0, dtype=tq_in.dtype),
            )
            correction_scaled = jnp.linalg.solve(
                normal_matrix
                + damping * jnp.eye(normal_matrix.shape[0], dtype=tq_in.dtype),
                scaled_matrix.T @ correction_rhs,
            )
            correction = jnp.nan_to_num(
                correction_scaled / column_scale,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            q_end = tq_in.shape[0]
            r_end = q_end + tr_in.shape[0]
            max_step = jnp.maximum(
                second_order_correction_max_abs_step.astype(tq_in.dtype),
                jnp.asarray(0.0, dtype=tq_in.dtype),
            )
            return (
                tq_in + jnp.clip(correction[:q_end], -max_step, max_step),
                jnp.minimum(
                    tr_in
                    + jnp.clip(correction[q_end:r_end], -max_step, max_step),
                    r_cap,
                ),
                tqtot_in + jnp.clip(correction[-1], -max_step, max_step),
            )

        def legacy_dual_repair(
            corrected_q: jnp.ndarray,
            corrected_r: jnp.ndarray,
            corrected_qtot: jnp.ndarray,
            fallback_lam: jnp.ndarray,
            fallback_rho: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            repaired_rho = epsilon_vec[0] - corrected_r
            stationarity_matrix = jnp.concatenate([ag.T, ac.T], axis=0)
            stationarity_rhs = jnp.concatenate(
                [
                    corrected_q
                    + gas_stationarity_source
                    + qtot
                    - corrected_qtot,
                    hcond - jnp.exp(repaired_rho),
                ]
            )
            column_scale = jnp.maximum(
                jnp.linalg.norm(stationarity_matrix, axis=0),
                jnp.asarray(1.0e-300, dtype=corrected_q.dtype),
            )
            scaled_matrix = stationarity_matrix / column_scale[None, :]
            normal_matrix = scaled_matrix.T @ scaled_matrix
            damping = jnp.asarray(1.0e-12, dtype=corrected_q.dtype) * jnp.maximum(
                jnp.mean(jnp.diag(normal_matrix)),
                jnp.asarray(1.0, dtype=corrected_q.dtype),
            )
            repaired_scaled = jnp.linalg.solve(
                normal_matrix
                + damping * jnp.eye(normal_matrix.shape[0], dtype=corrected_q.dtype),
                scaled_matrix.T @ stationarity_rhs,
            )
            repaired_lam = repaired_scaled / column_scale
            repair_finite = jnp.all(jnp.isfinite(repaired_lam))
            return (
                jnp.where(repair_finite, repaired_lam, fallback_lam),
                jnp.where(repair_finite, repaired_rho, fallback_rho),
            )

        def ipopt_restoration_dual_repair(
            corrected_q: jnp.ndarray,
            corrected_r: jnp.ndarray,
            corrected_qtot: jnp.ndarray,
            _fallback_lam: jnp.ndarray,
            fallback_rho: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            (
                repaired_lam,
                repaired_rho,
                _alpha_dual,
                _bound_reset,
                _equality_reset,
            ) = fixed_support_ipopt_restoration_dual_return(
                formula_matrix=ag,
                formula_matrix_cond_active=ac,
                restored_q=corrected_q,
                restored_r=corrected_r,
                restored_qtot=corrected_qtot,
                qtot_reference=qtot,
                gas_stationarity_source=gas_stationarity_source,
                condensate_standard_source=hcond,
                current_r=r,
                current_rho=fallback_rho,
                barrier=jnp.exp(epsilon_vec[0]),
            )
            return repaired_lam, repaired_rho

        def second_order_correct_trial(
            tq_in: jnp.ndarray,
            tr_in: jnp.ndarray,
            tlam_in: jnp.ndarray,
            trho_in: jnp.ndarray,
            tqtot_in: jnp.ndarray,
            alpha_test: jnp.ndarray,
        ) -> tuple[jnp.ndarray, ...]:
            def kkt_correction_body(
                _pass_index: int,
                state: tuple[jnp.ndarray, ...],
            ) -> tuple[jnp.ndarray, ...]:
                return single_second_order_kkt_correction(*state)

            def legacy_correction_body(
                _pass_index: int,
                state: tuple[jnp.ndarray, ...],
            ) -> tuple[jnp.ndarray, ...]:
                return legacy_budget_projection_correction(*state)

            if second_order_correction_policy in {
                "legacy_budget_projection",
                "legacy_budget_projection_triggered",
            }:
                corrected_q, corrected_r, corrected_qtot = lax.fori_loop(
                    0,
                    int(second_order_correction_budget_passes),
                    legacy_correction_body,
                    (tq_in, tr_in, tqtot_in),
                )
                corrected_lam, corrected_rho = lax.cond(
                    second_order_correction_dual_repair.astype(bool),
                    lambda _: legacy_dual_repair(
                        corrected_q,
                        corrected_r,
                        corrected_qtot,
                        tlam_in,
                        trho_in,
                    ),
                    lambda _: (tlam_in, trho_in),
                    operand=None,
                )
            else:
                (
                    corrected_q,
                    corrected_r,
                    corrected_lam,
                    corrected_rho,
                    corrected_qtot,
                ) = lax.fori_loop(
                    0,
                    int(second_order_correction_budget_passes),
                    kkt_correction_body,
                    (tq_in, tr_in, tlam_in, trho_in, tqtot_in),
                )
            gas, cond, budget_corr, comp, _total = residual_components(
                corrected_q,
                corrected_r,
                corrected_lam,
                corrected_rho,
                corrected_qtot,
            )
            return (
                corrected_q,
                corrected_r,
                corrected_lam,
                corrected_rho,
                corrected_qtot,
                residual_norm(
                    corrected_q,
                    corrected_r,
                    corrected_lam,
                    corrected_rho,
                    corrected_qtot,
                ),
                l2(gas),
                l2(cond),
                l2(budget_corr),
                relative_budget_max_abs(budget_corr),
                l2(comp),
                l2(_total),
            )

        def full_restoration_trial() -> tuple[jnp.ndarray, ...]:
            if budget_restoration_coordinate_policy == "amount":
                restoration_result = fixed_support_amount_space_restoration(
                    formula_matrix=ag,
                    formula_matrix_cond_active=ac,
                    element_inventory_target=target,
                    q_reference=q,
                    r_reference=r,
                    qtot_reference=qtot,
                    relative_floor=convergence_budget_relative_floor,
                    proximity_weight=budget_restoration_proximity_weight,
                    max_abs_primal_step=budget_restoration_max_abs_step,
                    passes=budget_restoration_passes,
                    slack_barrier=jnp.exp(epsilon_vec[0]),
                    q_proximity_reference=restoration_reference_q,
                    r_proximity_reference=restoration_reference_r,
                    qtot_proximity_reference=restoration_reference_qtot,
                )
            else:
                restoration_result = fixed_support_full_restoration(
                    formula_matrix=ag,
                    formula_matrix_cond_active=ac,
                    element_inventory_target=target,
                    q_reference=q,
                    r_reference=r,
                    qtot_reference=qtot,
                    relative_floor=convergence_budget_relative_floor,
                    proximity_weight=budget_restoration_proximity_weight,
                    max_abs_primal_step=budget_restoration_max_abs_step,
                    passes=budget_restoration_passes,
                )
            restored_q, restored_r, restored_qtot, _positive, _negative = (
                restoration_result
            )
            restored_lam, restored_rho = lax.cond(
                budget_restoration_dual_recenter.astype(bool)
                & (~jnp.asarray(budget_restoration_phase_enabled)),
                lambda _: (
                    ipopt_restoration_dual_repair(
                        restored_q,
                        restored_r,
                        restored_qtot,
                        lam,
                        rho,
                    )
                    if budget_restoration_dual_recenter_policy
                    == "ipopt_linearized"
                    else legacy_dual_repair(
                        restored_q,
                        restored_r,
                        restored_qtot,
                        lam,
                        rho,
                    )
                ),
                lambda _: (lam, rho),
                operand=None,
            )
            gas, cond, budget_restore, comp, total = residual_components(
                restored_q,
                restored_r,
                restored_lam,
                restored_rho,
                restored_qtot,
            )
            return (
                restored_q,
                restored_r,
                restored_lam,
                restored_rho,
                restored_qtot,
                residual_norm(
                    restored_q,
                    restored_r,
                    restored_lam,
                    restored_rho,
                    restored_qtot,
                ),
                l2(gas),
                l2(cond),
                l2(budget_restore),
                relative_budget_max_abs(budget_restore),
                l2(comp),
                l2(total),
            )

        def ipopt_soc_replacement_trial(
            tq_in: jnp.ndarray,
            tr_in: jnp.ndarray,
            tlam_in: jnp.ndarray,
            trho_in: jnp.ndarray,
            tqtot_in: jnp.ndarray,
            alpha_test: jnp.ndarray,
        ) -> tuple[jnp.ndarray, ...]:
            reference_grad_barrier_delta = (
                fixed_support_barrier_objective_linearized_change(
                    q=q,
                    r=r,
                    qtot=qtot,
                    gas_stationarity_source=gas_stationarity_source,
                    condensate_standard_source=hcond,
                    qtot_reference=qtot,
                    epsilon=epsilon_vec[0],
                    delta_q=delta_q,
                    delta_r=delta_r,
                    delta_qtot=delta_qtot,
                )
            )
            reference_linearized_change = (
                alpha_test * reference_grad_barrier_delta
            )
            current_theta = fixed_support_filter_theta(
                formula_matrix=ag,
                formula_matrix_cond_active=ac,
                element_inventory_target=target,
                q=q,
                r=r,
                qtot=qtot,
                relative_floor=convergence_budget_relative_floor,
                use_l1_norm=ipopt_filter_use_l1_theta,
            )
            current_phi = fixed_support_barrier_objective(
                q=q,
                r=r,
                qtot=qtot,
                gas_stationarity_source=gas_stationarity_source,
                condensate_standard_source=hcond,
                qtot_reference=qtot,
                epsilon=epsilon_vec[0],
            )

            def filter_accepts(
                trial_q: jnp.ndarray,
                trial_r: jnp.ndarray,
                trial_qtot: jnp.ndarray,
            ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                trial_theta = fixed_support_filter_theta(
                    formula_matrix=ag,
                    formula_matrix_cond_active=ac,
                    element_inventory_target=target,
                    q=trial_q,
                    r=trial_r,
                    qtot=trial_qtot,
                    relative_floor=convergence_budget_relative_floor,
                    use_l1_norm=ipopt_filter_use_l1_theta,
                )
                trial_phi = fixed_support_barrier_objective(
                    q=trial_q,
                    r=trial_r,
                    qtot=trial_qtot,
                    gas_stationarity_source=gas_stationarity_source,
                    condensate_standard_source=hcond,
                    qtot_reference=qtot,
                    epsilon=epsilon_vec[0],
                )
                accepted = fixed_support_filter_acceptance(
                    trial_phi=trial_phi[None],
                    trial_theta=trial_theta[None],
                    trial_alpha=alpha_test[None],
                    trial_linearized_change=reference_linearized_change[None],
                    finite=jnp.asarray([True]),
                    current_phi=current_phi,
                    current_theta=current_theta,
                    initial_theta=initial_filter_theta,
                    filter_phi=filter_phi_entries,
                    filter_theta=filter_theta_entries,
                    filter_valid=filter_valid_entries,
                )[0][0]
                return accepted, trial_theta, trial_phi

            normal_accepted, normal_theta, normal_phi = filter_accepts(
                tq_in, tr_in, tqtot_in
            )
            eligible = (~normal_accepted) & (normal_theta >= current_theta)
            trial_n = jnp.exp(tq_in)
            trial_m = jnp.exp(tr_in)
            trial_budget = ag @ trial_n + ac @ trial_m - target
            trial_total = jnp.sum(trial_n) - jnp.exp(tqtot_in)

            def soc_body(
                _pass_index: int,
                state: tuple[jnp.ndarray, ...],
            ) -> tuple[jnp.ndarray, ...]:
                (
                    previous_budget_rhs,
                    previous_total_rhs,
                    trial_q,
                    trial_r,
                    trial_lam,
                    trial_rho,
                    trial_qtot,
                    trial_budget_residual,
                    trial_total_residual,
                    alpha_soc_previous,
                    theta_old,
                    active,
                    accepted_before,
                    correction_count,
                    all_directions_finite,
                    kappa_stopped,
                    phi_old,
                    max_solve_linear_residual,
                    max_solve_solution_norm,
                    min_solve_singular_value,
                    max_solve_condition_estimate,
                    max_scaled_solve_condition_estimate,
                    max_relative_solve_linear_residual,
                    last_solve_solution_norm,
                    last_relative_solve_linear_residual,
                    last_solve_condition_estimate,
                    last_scaled_solve_condition_estimate,
                    last_solve_smallest_singular_value,
                    last_smallest_right_singular_vector,
                ) = state

                def take_soc(_unused: None) -> tuple[jnp.ndarray, ...]:
                    soc_budget_rhs, soc_total_rhs = fixed_support_soc_constraint_rhs(
                        trial_budget_residual=trial_budget_residual,
                        trial_total_density_residual=trial_total_residual,
                        previous_soc_budget_rhs=previous_budget_rhs,
                        previous_soc_total_density_rhs=previous_total_rhs,
                        alpha_soc=alpha_soc_previous,
                    )
                    direction_with_diagnostics = (
                        fixed_support_reduced_direction_from_rhs_with_diagnostics(
                        formula_matrix=ag,
                        formula_matrix_cond_active=ac,
                        gas_amounts=n,
                        condensate_amounts=m,
                        condensate_duals=eta,
                        total_gas_amount=jnp.exp(qtot),
                        gas_rhs=initial_gas,
                        condensate_rhs=initial_cond,
                        budget_rhs=soc_budget_rhs,
                        complementarity_rhs=initial_comp,
                        total_density_rhs=soc_total_rhs,
                        charge_solve_policy=(
                            "charge_schur"
                            if second_order_correction_charge_solve_policy
                            == "charge_schur"
                            else "coupled"
                        ),
                        charge_row_index=ag.shape[0] - 1,
                        reduced_mode_policy=(
                            second_order_correction_reduced_mode_policy
                        ),
                        diagnostic_mode_vector_policy=(
                            second_order_correction_diagnostic_mode_vector_policy
                        ),
                        )
                    )
                    direction = direction_with_diagnostics[:5]
                    raw_solve_finite = direction_with_diagnostics[5]
                    solve_linear_residual = direction_with_diagnostics[6]
                    solve_solution_norm = direction_with_diagnostics[7]
                    solve_smallest_singular = direction_with_diagnostics[8]
                    solve_largest_singular = direction_with_diagnostics[9]
                    scaled_solve_smallest_singular = direction_with_diagnostics[10]
                    scaled_solve_largest_singular = direction_with_diagnostics[11]
                    relative_solve_linear_residual = direction_with_diagnostics[12]
                    smallest_right_singular_vector = direction_with_diagnostics[13]
                    solve_condition_estimate = solve_largest_singular / jnp.maximum(
                        solve_smallest_singular,
                        jnp.asarray(jnp.finfo(q.dtype).tiny, dtype=q.dtype),
                    )
                    scaled_solve_condition_estimate = (
                        scaled_solve_largest_singular
                        / jnp.maximum(
                            scaled_solve_smallest_singular,
                            jnp.asarray(jnp.finfo(q.dtype).tiny, dtype=q.dtype),
                        )
                    )
                    bounded_primal_alpha = jnp.min(
                        jnp.where(direction[1] < 0.0, -1.0 / direction[1], 1.0),
                        initial=jnp.asarray(1.0, dtype=q.dtype),
                    )
                    bounded_dual_alpha = jnp.min(
                        jnp.where(direction[3] < 0.0, -1.0 / direction[3], 1.0),
                        initial=jnp.asarray(1.0, dtype=q.dtype),
                    )
                    alpha_soc = jnp.where(
                        use_log_amount_boundary,
                        0.995 * bounded_primal_alpha,
                        jnp.asarray(1.0, dtype=q.dtype),
                    )
                    alpha_dual = jnp.where(
                        use_log_activity_boundary,
                        0.995 * bounded_dual_alpha,
                        jnp.asarray(1.0, dtype=q.dtype),
                    )
                    alpha_soc = jnp.clip(alpha_soc, 0.0, 1.0)
                    alpha_dual = jnp.clip(alpha_dual, 0.0, 1.0)
                    provisional_q = q + alpha_soc * direction[0]
                    provisional_r = r + alpha_soc * direction[1]
                    if second_order_correction_alpha_y_policy == "full":
                        alpha_y = jnp.asarray(1.0, dtype=q.dtype)
                    elif second_order_correction_alpha_y_policy == "primal":
                        alpha_y = alpha_soc
                    else:
                        alpha_y = fixed_support_min_equality_dual_infeasibility_step(
                            formula_matrix=ag,
                            formula_matrix_cond_active=ac,
                            q_trial=provisional_q,
                            rho_trial=rho + alpha_dual * direction[3],
                            lambda_current=lam,
                            delta_lambda=direction[2],
                            gas_stationarity_source=gas_stationarity_source,
                            condensate_standard_source=hcond,
                        )
                    next_trial = fixed_support_soc_trial_from_current(
                        q=q,
                        r=r,
                        element_potential=lam,
                        rho=rho,
                        qtot=qtot,
                        delta_q=direction[0],
                        delta_r=direction[1],
                        delta_element_potential=direction[2],
                        delta_rho=direction[3],
                        delta_qtot=direction[4],
                        alpha_test=alpha_test,
                        alpha_soc=alpha_soc,
                        alpha_y=alpha_y,
                        alpha_dual=alpha_dual,
                    )
                    next_q, next_lam = charge_retract_trial(
                        next_trial[0], next_trial[2]
                    )
                    next_trial = (
                        next_q,
                        next_trial[1],
                        next_lam,
                        next_trial[3],
                        next_trial[4],
                        next_trial[5],
                    )
                    next_n = jnp.exp(next_trial[0])
                    next_m = jnp.exp(next_trial[1])
                    next_budget = ag @ next_n + ac @ next_m - target
                    next_total = jnp.sum(next_n) - jnp.exp(next_trial[4])
                    accepted, theta_new, phi_new = filter_accepts(
                        next_trial[0], next_trial[1], next_trial[4]
                    )
                    direction_finite = raw_solve_finite
                    continue_soc = (~accepted) & (
                        theta_new
                        <= second_order_correction_kappa_soc.astype(theta_new.dtype)
                        * theta_old
                    )
                    return (
                        soc_budget_rhs,
                        soc_total_rhs,
                        next_trial[0],
                        next_trial[1],
                        next_trial[2],
                        next_trial[3],
                        next_trial[4],
                        next_budget,
                        next_total,
                        alpha_soc,
                        theta_new,
                        continue_soc,
                        accepted,
                        correction_count + jnp.asarray(1, dtype=jnp.int32),
                        all_directions_finite & direction_finite,
                        (~accepted) & (~continue_soc),
                        phi_new,
                        jnp.maximum(
                            max_solve_linear_residual, solve_linear_residual
                        ),
                        jnp.maximum(max_solve_solution_norm, solve_solution_norm),
                        jnp.minimum(
                            min_solve_singular_value, solve_smallest_singular
                        ),
                        jnp.maximum(
                            max_solve_condition_estimate,
                            solve_condition_estimate,
                        ),
                        jnp.maximum(
                            max_scaled_solve_condition_estimate,
                            scaled_solve_condition_estimate,
                        ),
                        jnp.maximum(
                            max_relative_solve_linear_residual,
                            relative_solve_linear_residual,
                        ),
                        solve_solution_norm,
                        relative_solve_linear_residual,
                        solve_condition_estimate,
                        scaled_solve_condition_estimate,
                        solve_smallest_singular,
                        smallest_right_singular_vector,
                    )

                return lax.cond(active, take_soc, lambda _: state, operand=None)

            soc_state = lax.fori_loop(
                0,
                int(second_order_correction_budget_passes),
                soc_body,
                (
                    initial_budget,
                    initial_total[0],
                    tq_in,
                    tr_in,
                    tlam_in,
                    trho_in,
                    tqtot_in,
                    trial_budget,
                    trial_total,
                    alpha_test,
                    normal_theta,
                    eligible,
                    jnp.asarray(False),
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(True),
                    jnp.asarray(False),
                    normal_phi,
                    jnp.asarray(0.0, dtype=q.dtype),
                    jnp.asarray(0.0, dtype=q.dtype),
                    jnp.asarray(jnp.inf, dtype=q.dtype),
                    jnp.asarray(0.0, dtype=q.dtype),
                    jnp.asarray(0.0, dtype=q.dtype),
                    jnp.asarray(0.0, dtype=q.dtype),
                    jnp.asarray(0.0, dtype=q.dtype),
                    jnp.asarray(0.0, dtype=q.dtype),
                    jnp.asarray(0.0, dtype=q.dtype),
                    jnp.asarray(0.0, dtype=q.dtype),
                    jnp.asarray(0.0, dtype=q.dtype),
                    jnp.zeros((ag.shape[0] + 24,), dtype=q.dtype),
                ),
            )
            corrected_q = soc_state[2]
            corrected_r = soc_state[3]
            corrected_lam = soc_state[4]
            corrected_rho = soc_state[5]
            corrected_qtot = soc_state[6]
            gas, cond, budget, comp, total = residual_components(
                corrected_q,
                corrected_r,
                corrected_lam,
                corrected_rho,
                corrected_qtot,
            )
            return (
                corrected_q,
                corrected_r,
                corrected_lam,
                corrected_rho,
                corrected_qtot,
                residual_norm(
                    corrected_q,
                    corrected_r,
                    corrected_lam,
                    corrected_rho,
                    corrected_qtot,
                ),
                l2(gas),
                l2(cond),
                l2(budget),
                relative_budget_max_abs(budget),
                l2(comp),
                l2(total),
                eligible,
                soc_state[13],
                soc_state[14],
                soc_state[12],
                soc_state[15],
                normal_theta,
                soc_state[10],
                normal_phi,
                soc_state[16],
                soc_state[9],
                soc_state[17],
                soc_state[18],
                soc_state[19],
                soc_state[20],
                soc_state[21],
                soc_state[22],
                soc_state[23],
                soc_state[24],
                soc_state[25],
                soc_state[26],
                soc_state[27],
                soc_state[28],
            )

        bounded_alpha_grid = jnp.minimum(alpha_grid, alpha_boundary)
        restoration_alpha_boundary = direction_alpha_boundary(
            restoration_delta_r,
            restoration_delta_rho,
        )
        restoration_bounded_alpha_grid = jnp.minimum(
            alpha_grid,
            restoration_alpha_boundary,
        )
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
            budget_relative_maxes,
            comp_norms,
            total_norms,
        ) = jax.vmap(trial_for_direction, in_axes=(0, None, None, None, None, None))(
            bounded_alpha_grid,
            delta_q,
            delta_r,
            delta_lam,
            delta_rho,
            delta_qtot,
        )
        (
            restoration_tq,
            restoration_tr,
            restoration_tlam,
            restoration_trho,
            restoration_tqtot,
            restoration_norms,
            restoration_gas_norms,
            restoration_cond_norms,
            restoration_budget_norms,
            restoration_budget_relative_maxes,
            restoration_comp_norms,
            restoration_total_norms,
        ) = jax.vmap(trial_for_direction, in_axes=(0, None, None, None, None, None))(
            restoration_bounded_alpha_grid,
            restoration_delta_q,
            restoration_delta_r,
            restoration_delta_lam,
            restoration_delta_rho,
            restoration_delta_qtot,
        )
        tq = jnp.concatenate([tq, restoration_tq], axis=0)
        tr = jnp.concatenate([tr, restoration_tr], axis=0)
        tlam = jnp.concatenate([tlam, restoration_tlam], axis=0)
        trho = jnp.concatenate([trho, restoration_trho], axis=0)
        tqtot = jnp.concatenate([tqtot, restoration_tqtot], axis=0)
        trial_alphas = jnp.concatenate(
            [bounded_alpha_grid, restoration_bounded_alpha_grid],
            axis=0,
        )
        norms = jnp.concatenate([norms, restoration_norms], axis=0)
        gas_norms = jnp.concatenate([gas_norms, restoration_gas_norms], axis=0)
        cond_norms = jnp.concatenate([cond_norms, restoration_cond_norms], axis=0)
        budget_norms = jnp.concatenate([budget_norms, restoration_budget_norms], axis=0)
        budget_relative_maxes = jnp.concatenate(
            [budget_relative_maxes, restoration_budget_relative_maxes],
            axis=0,
        )
        comp_norms = jnp.concatenate([comp_norms, restoration_comp_norms], axis=0)
        total_norms = jnp.concatenate([total_norms, restoration_total_norms], axis=0)
        restoration_trial_flags = jnp.concatenate(
            [
                jnp.zeros_like(bounded_alpha_grid, dtype=bool),
                jnp.ones_like(restoration_bounded_alpha_grid, dtype=bool),
            ],
            axis=0,
        )
        budget_restoration_trial_flags = jnp.zeros_like(
            restoration_trial_flags,
            dtype=bool,
        )

        def append_budget_restoration_trials(_unused: None) -> tuple[jnp.ndarray, ...]:
            base_thetas = jax.vmap(
                lambda trial_q, trial_r, trial_qtot: fixed_support_filter_theta(
                    formula_matrix=ag,
                    formula_matrix_cond_active=ac,
                    element_inventory_target=target,
                    q=trial_q,
                    r=trial_r,
                    qtot=trial_qtot,
                    relative_floor=convergence_budget_relative_floor,
                    use_l1_norm=ipopt_filter_use_l1_theta,
                )
            )(tq, tr, tqtot)
            current_theta = fixed_support_filter_theta(
                formula_matrix=ag,
                formula_matrix_cond_active=ac,
                element_inventory_target=target,
                q=q,
                r=r,
                qtot=qtot,
                relative_floor=convergence_budget_relative_floor,
                use_l1_norm=ipopt_filter_use_l1_theta,
            )
            current_phi = fixed_support_barrier_objective(
                q=q,
                r=r,
                qtot=qtot,
                gas_stationarity_source=gas_stationarity_source,
                condensate_standard_source=hcond,
                qtot_reference=qtot,
                epsilon=epsilon_vec[0],
            )
            base_phis = jax.vmap(
                lambda trial_q, trial_r, trial_qtot: fixed_support_barrier_objective(
                    q=trial_q,
                    r=trial_r,
                    qtot=trial_qtot,
                    gas_stationarity_source=gas_stationarity_source,
                    condensate_standard_source=hcond,
                    qtot_reference=qtot,
                    epsilon=epsilon_vec[0],
                )
            )(tq, tr, tqtot)
            base_linearized_changes = jax.vmap(
                lambda trial_q, trial_r, trial_qtot: (
                    fixed_support_barrier_objective_linearized_change(
                        q=q,
                        r=r,
                        qtot=qtot,
                        gas_stationarity_source=gas_stationarity_source,
                        condensate_standard_source=hcond,
                        qtot_reference=qtot,
                        epsilon=epsilon_vec[0],
                        delta_q=trial_q - q,
                        delta_r=trial_r - r,
                        delta_qtot=trial_qtot - qtot,
                    )
                )
            )(tq, tr, tqtot)
            base_filter_accepted, _ftype, _armijo, base_history_accepted = (
                fixed_support_filter_acceptance(
                    trial_phi=base_phis,
                    trial_theta=base_thetas,
                    trial_alpha=trial_alphas,
                    trial_linearized_change=base_linearized_changes,
                    finite=jnp.isfinite(norms),
                    current_phi=current_phi,
                    current_theta=current_theta,
                    initial_theta=initial_filter_theta,
                    filter_phi=filter_phi_entries,
                    filter_theta=filter_theta_entries,
                    filter_valid=filter_valid_entries,
                )
            )
            base_soft_accepted = (
                jnp.isfinite(norms)
                & (~restoration_trial_flags)
                & base_history_accepted
                & (norms <= 0.9 * initial_norm)
            )
            normal_globalization_failed = (~jnp.any(base_filter_accepted)) & (
                ~jnp.any(base_soft_accepted)
            )
            restoration_needed = jnp.where(
                jnp.asarray(budget_restoration_phase_enabled),
                restoration_phase_active
                | (restoration_can_enter & normal_globalization_failed),
                normal_globalization_failed,
            )

            def skipped_restoration(_operand: None) -> tuple[jnp.ndarray, ...]:
                return (
                    q,
                    r,
                    lam,
                    rho,
                    qtot,
                    jnp.asarray(jnp.inf, dtype=q.dtype),
                    initial_gas_norm,
                    initial_cond_norm,
                    initial_budget_norm,
                    initial_budget_relative_max,
                    initial_comp_norm,
                    initial_total_norm,
                )

            (
                restored_q,
                restored_r,
                restored_lam,
                restored_rho,
                restored_qtot,
                restored_norm,
                restored_gas_norm,
                restored_cond_norm,
                restored_budget_norm,
                restored_budget_relative_max,
                restored_comp_norm,
                restored_total_norm,
            ) = lax.cond(
                restoration_needed,
                lambda _: full_restoration_trial(),
                skipped_restoration,
                operand=None,
            )
            restore_tq = tq.at[0].set(restored_q)
            restore_tr = tr.at[0].set(restored_r)
            restore_tlam = tlam.at[0].set(restored_lam)
            restore_trho = trho.at[0].set(restored_rho)
            restore_tqtot = tqtot.at[0].set(restored_qtot)
            restore_norms = jnp.full_like(norms, jnp.inf).at[0].set(restored_norm)
            restore_gas_norms = gas_norms.at[0].set(restored_gas_norm)
            restore_cond_norms = cond_norms.at[0].set(restored_cond_norm)
            restore_budget_norms = budget_norms.at[0].set(restored_budget_norm)
            restore_budget_relative_maxes = budget_relative_maxes.at[0].set(
                restored_budget_relative_max
            )
            restore_comp_norms = comp_norms.at[0].set(restored_comp_norm)
            restore_total_norms = total_norms.at[0].set(restored_total_norm)
            restore_alphas = trial_alphas.at[0].set(1.0)
            return (
                jnp.concatenate([tq, restore_tq], axis=0),
                jnp.concatenate([tr, restore_tr], axis=0),
                jnp.concatenate([tlam, restore_tlam], axis=0),
                jnp.concatenate([trho, restore_trho], axis=0),
                jnp.concatenate([tqtot, restore_tqtot], axis=0),
                jnp.concatenate([trial_alphas, restore_alphas], axis=0),
                jnp.concatenate([norms, restore_norms], axis=0),
                jnp.concatenate([gas_norms, restore_gas_norms], axis=0),
                jnp.concatenate([cond_norms, restore_cond_norms], axis=0),
                jnp.concatenate([budget_norms, restore_budget_norms], axis=0),
                jnp.concatenate(
                    [budget_relative_maxes, restore_budget_relative_maxes],
                    axis=0,
                ),
                jnp.concatenate([comp_norms, restore_comp_norms], axis=0),
                jnp.concatenate([total_norms, restore_total_norms], axis=0),
                jnp.concatenate(
                    [restoration_trial_flags, restoration_trial_flags],
                    axis=0,
                ),
                jnp.concatenate(
                    [
                        budget_restoration_trial_flags,
                        jnp.ones_like(budget_restoration_trial_flags, dtype=bool),
                    ],
                    axis=0,
                ),
            )

        def append_disabled_budget_restoration_trials(
            _unused: None,
        ) -> tuple[jnp.ndarray, ...]:
            return (
                tq,
                tr,
                tlam,
                trho,
                tqtot,
                trial_alphas,
                norms,
                gas_norms,
                cond_norms,
                budget_norms,
                budget_relative_maxes,
                comp_norms,
                total_norms,
                restoration_trial_flags,
                budget_restoration_trial_flags,
            )

        if budget_restoration_enabled:
            (
                tq,
                tr,
                tlam,
                trho,
                tqtot,
                trial_alphas,
                norms,
                gas_norms,
                cond_norms,
                budget_norms,
                budget_relative_maxes,
                comp_norms,
                total_norms,
                restoration_trial_flags,
                budget_restoration_trial_flags,
            ) = append_budget_restoration_trials(None)
        else:
            (
                tq,
                tr,
                tlam,
                trho,
                tqtot,
                trial_alphas,
                norms,
                gas_norms,
                cond_norms,
                budget_norms,
                budget_relative_maxes,
                comp_norms,
                total_norms,
                restoration_trial_flags,
                budget_restoration_trial_flags,
            ) = append_disabled_budget_restoration_trials(None)
        def append_soc_trials(_unused: None) -> tuple[jnp.ndarray, ...]:
            normal_trial_mask = (~restoration_trial_flags) & (
                ~budget_restoration_trial_flags
            )
            max_normal_alpha = jnp.max(
                jnp.where(normal_trial_mask, trial_alphas, -jnp.inf)
            )
            soc_base_index = jnp.argmax(
                normal_trial_mask & (trial_alphas == max_normal_alpha)
            )
            ipopt_soc_eligible = jnp.asarray(False)
            ipopt_soc_correction_count = jnp.asarray(0, dtype=jnp.int32)
            ipopt_soc_all_directions_finite = jnp.asarray(False)
            ipopt_soc_filter_accepted = jnp.asarray(False)
            ipopt_soc_kappa_stopped = jnp.asarray(False)
            ipopt_soc_normal_theta = jnp.asarray(0.0, dtype=q.dtype)
            ipopt_soc_final_theta = jnp.asarray(0.0, dtype=q.dtype)
            ipopt_soc_normal_phi = jnp.asarray(0.0, dtype=q.dtype)
            ipopt_soc_final_phi = jnp.asarray(0.0, dtype=q.dtype)
            ipopt_soc_final_alpha = jnp.asarray(0.0, dtype=q.dtype)
            ipopt_soc_max_solve_linear_residual = jnp.asarray(0.0, dtype=q.dtype)
            ipopt_soc_max_solve_solution_norm = jnp.asarray(0.0, dtype=q.dtype)
            ipopt_soc_min_solve_singular_value = jnp.asarray(jnp.inf, dtype=q.dtype)
            ipopt_soc_max_solve_condition_estimate = jnp.asarray(0.0, dtype=q.dtype)
            ipopt_soc_max_scaled_solve_condition_estimate = jnp.asarray(
                0.0, dtype=q.dtype
            )
            ipopt_soc_max_relative_solve_linear_residual = jnp.asarray(
                0.0, dtype=q.dtype
            )
            ipopt_soc_last_solve_solution_norm = jnp.asarray(0.0, dtype=q.dtype)
            ipopt_soc_last_relative_solve_linear_residual = jnp.asarray(
                0.0, dtype=q.dtype
            )
            ipopt_soc_last_solve_condition_estimate = jnp.asarray(
                0.0, dtype=q.dtype
            )
            ipopt_soc_last_scaled_solve_condition_estimate = jnp.asarray(
                0.0, dtype=q.dtype
            )
            ipopt_soc_last_solve_smallest_singular_value = jnp.asarray(
                0.0, dtype=q.dtype
            )
            ipopt_soc_last_smallest_right_singular_vector = jnp.zeros(
                (ag.shape[0] + 24,), dtype=q.dtype
            )
            if second_order_correction_policy == "legacy_budget_projection":
                corrected = jax.vmap(second_order_correct_trial)(
                    tq, tr, tlam, trho, tqtot, trial_alphas
                )
                (
                    soc_tq,
                    soc_tr,
                    soc_tlam,
                    soc_trho,
                    soc_tqtot,
                    corrected_norms,
                    soc_gas_norms,
                    soc_cond_norms,
                    soc_budget_norms,
                    soc_budget_relative_maxes,
                    soc_comp_norms,
                    soc_total_norms,
                ) = corrected
                soc_norms = jnp.where(
                    ~budget_restoration_trial_flags,
                    corrected_norms,
                    jnp.inf,
                )
            elif second_order_correction_policy == "ipopt_first_trial":
                (
                    corrected_q,
                    corrected_r,
                    corrected_lam,
                    corrected_rho,
                    corrected_qtot,
                    corrected_norm,
                    corrected_gas_norm,
                    corrected_cond_norm,
                    corrected_budget_norm,
                    corrected_budget_relative_max,
                    corrected_comp_norm,
                    corrected_total_norm,
                    ipopt_soc_eligible,
                    ipopt_soc_correction_count,
                    ipopt_soc_all_directions_finite,
                    ipopt_soc_filter_accepted,
                    ipopt_soc_kappa_stopped,
                    ipopt_soc_normal_theta,
                    ipopt_soc_final_theta,
                    ipopt_soc_normal_phi,
                    ipopt_soc_final_phi,
                    ipopt_soc_final_alpha,
                    ipopt_soc_max_solve_linear_residual,
                    ipopt_soc_max_solve_solution_norm,
                    ipopt_soc_min_solve_singular_value,
                    ipopt_soc_max_solve_condition_estimate,
                    ipopt_soc_max_scaled_solve_condition_estimate,
                    ipopt_soc_max_relative_solve_linear_residual,
                    ipopt_soc_last_solve_solution_norm,
                    ipopt_soc_last_relative_solve_linear_residual,
                    ipopt_soc_last_solve_condition_estimate,
                    ipopt_soc_last_scaled_solve_condition_estimate,
                    ipopt_soc_last_solve_smallest_singular_value,
                    ipopt_soc_last_smallest_right_singular_vector,
                ) = ipopt_soc_replacement_trial(
                    tq[soc_base_index],
                    tr[soc_base_index],
                    tlam[soc_base_index],
                    trho[soc_base_index],
                    tqtot[soc_base_index],
                    trial_alphas[soc_base_index],
                )
            else:
                (
                    corrected_q,
                    corrected_r,
                    corrected_lam,
                    corrected_rho,
                    corrected_qtot,
                    corrected_norm,
                    corrected_gas_norm,
                    corrected_cond_norm,
                    corrected_budget_norm,
                    corrected_budget_relative_max,
                    corrected_comp_norm,
                    corrected_total_norm,
                ) = second_order_correct_trial(
                    tq[soc_base_index],
                    tr[soc_base_index],
                    tlam[soc_base_index],
                    trho[soc_base_index],
                    tqtot[soc_base_index],
                    trial_alphas[soc_base_index],
                )
            current_theta_for_soc = fixed_support_filter_theta(
                formula_matrix=ag,
                formula_matrix_cond_active=ac,
                element_inventory_target=target,
                q=q,
                r=r,
                qtot=qtot,
                relative_floor=convergence_budget_relative_floor,
                use_l1_norm=ipopt_filter_use_l1_theta,
            )
            current_phi_for_soc = fixed_support_barrier_objective(
                q=q,
                r=r,
                qtot=qtot,
                gas_stationarity_source=gas_stationarity_source,
                condensate_standard_source=hcond,
                qtot_reference=qtot,
                epsilon=epsilon_vec[0],
            )
            base_filter_thetas = jax.vmap(
                lambda trial_q, trial_r, trial_qtot: fixed_support_filter_theta(
                    formula_matrix=ag,
                    formula_matrix_cond_active=ac,
                    element_inventory_target=target,
                    q=trial_q,
                    r=trial_r,
                    qtot=trial_qtot,
                    relative_floor=convergence_budget_relative_floor,
                    use_l1_norm=ipopt_filter_use_l1_theta,
                )
            )(tq, tr, tqtot)
            base_barrier_objectives = jax.vmap(
                lambda trial_q, trial_r, trial_qtot: fixed_support_barrier_objective(
                    q=trial_q,
                    r=trial_r,
                    qtot=trial_qtot,
                    gas_stationarity_source=gas_stationarity_source,
                    condensate_standard_source=hcond,
                    qtot_reference=qtot,
                    epsilon=epsilon_vec[0],
                )
            )(tq, tr, tqtot)
            base_linearized_changes = jax.vmap(
                lambda trial_q, trial_r, trial_qtot: (
                    fixed_support_barrier_objective_linearized_change(
                        q=q,
                        r=r,
                        qtot=qtot,
                        gas_stationarity_source=gas_stationarity_source,
                        condensate_standard_source=hcond,
                        qtot_reference=qtot,
                        epsilon=epsilon_vec[0],
                        delta_q=trial_q - q,
                        delta_r=trial_r - r,
                        delta_qtot=trial_qtot - qtot,
                    )
                )
            )(tq, tr, tqtot)
            base_filter_accepted = fixed_support_filter_acceptance(
                trial_phi=base_barrier_objectives,
                trial_theta=base_filter_thetas,
                trial_alpha=trial_alphas,
                trial_linearized_change=base_linearized_changes,
                finite=jnp.isfinite(norms),
                current_phi=current_phi_for_soc,
                current_theta=current_theta_for_soc,
                initial_theta=initial_filter_theta,
                filter_phi=filter_phi_entries,
                filter_theta=filter_theta_entries,
                filter_valid=filter_valid_entries,
            )[0]
            if second_order_correction_policy == "ipopt_first_trial":
                soc_eligible = ipopt_soc_eligible
            else:
                soc_eligible = jnp.where(
                    ipopt_filter_acceptance_enabled.astype(bool),
                    ~base_filter_accepted[soc_base_index],
                    base_filter_thetas[soc_base_index]
                    >= 0.99 * current_theta_for_soc,
                )
            if second_order_correction_policy != "legacy_budget_projection":
                soc_tq = tq.at[soc_base_index].set(corrected_q)
                soc_tr = tr.at[soc_base_index].set(corrected_r)
                soc_tlam = tlam.at[soc_base_index].set(corrected_lam)
                soc_trho = trho.at[soc_base_index].set(corrected_rho)
                soc_tqtot = tqtot.at[soc_base_index].set(corrected_qtot)
                soc_norms = jnp.full_like(norms, jnp.inf).at[soc_base_index].set(
                    jnp.where(soc_eligible, corrected_norm, jnp.inf)
                )
                soc_gas_norms = gas_norms.at[soc_base_index].set(
                    corrected_gas_norm
                )
                soc_cond_norms = cond_norms.at[soc_base_index].set(
                    corrected_cond_norm
                )
                soc_budget_norms = budget_norms.at[soc_base_index].set(
                    corrected_budget_norm
                )
                soc_budget_relative_maxes = budget_relative_maxes.at[
                    soc_base_index
                ].set(corrected_budget_relative_max)
                soc_comp_norms = comp_norms.at[soc_base_index].set(
                    corrected_comp_norm
                )
                soc_total_norms = total_norms.at[soc_base_index].set(
                    corrected_total_norm
                )
            def interleave_pair(base: jnp.ndarray, correction: jnp.ndarray) -> jnp.ndarray:
                stacked = jnp.stack([base, correction], axis=1)
                return jnp.reshape(
                    stacked,
                    (base.shape[0] * 2,) + base.shape[1:],
                )

            def combine_pair(base: jnp.ndarray, correction: jnp.ndarray) -> jnp.ndarray:
                return lax.cond(
                    second_order_correction_interleave.astype(bool),
                    lambda _: interleave_pair(base, correction),
                    lambda _: jnp.concatenate([base, correction], axis=0),
                    operand=None,
                )

            soc_false_flags = jnp.zeros((norms.shape[0],), dtype=bool)
            soc_true_flags = jnp.ones((norms.shape[0],), dtype=bool)
            return (
                combine_pair(tq, soc_tq),
                combine_pair(tr, soc_tr),
                combine_pair(tlam, soc_tlam),
                combine_pair(trho, soc_trho),
                combine_pair(tqtot, soc_tqtot),
                combine_pair(trial_alphas, trial_alphas),
                combine_pair(norms, soc_norms),
                combine_pair(gas_norms, soc_gas_norms),
                combine_pair(cond_norms, soc_cond_norms),
                combine_pair(budget_norms, soc_budget_norms),
                combine_pair(budget_relative_maxes, soc_budget_relative_maxes),
                combine_pair(comp_norms, soc_comp_norms),
                combine_pair(total_norms, soc_total_norms),
                combine_pair(restoration_trial_flags, restoration_trial_flags),
                combine_pair(
                    budget_restoration_trial_flags,
                    budget_restoration_trial_flags,
                ),
                combine_pair(soc_false_flags, soc_true_flags),
                ipopt_soc_eligible,
                ipopt_soc_correction_count,
                ipopt_soc_all_directions_finite,
                ipopt_soc_filter_accepted,
                ipopt_soc_kappa_stopped,
                ipopt_soc_normal_theta,
                ipopt_soc_final_theta,
                ipopt_soc_normal_phi,
                ipopt_soc_final_phi,
                ipopt_soc_final_alpha,
                ipopt_soc_max_solve_linear_residual,
                ipopt_soc_max_solve_solution_norm,
                ipopt_soc_min_solve_singular_value,
                ipopt_soc_max_solve_condition_estimate,
                ipopt_soc_max_scaled_solve_condition_estimate,
                ipopt_soc_max_relative_solve_linear_residual,
                ipopt_soc_last_solve_solution_norm,
                ipopt_soc_last_relative_solve_linear_residual,
                ipopt_soc_last_solve_condition_estimate,
                ipopt_soc_last_scaled_solve_condition_estimate,
                ipopt_soc_last_solve_smallest_singular_value,
                ipopt_soc_last_smallest_right_singular_vector,
            )

        def append_disabled_soc_trials(_unused: None) -> tuple[jnp.ndarray, ...]:
            disabled_norms = jnp.full_like(norms, jnp.inf)
            return (
                jnp.concatenate([tq, tq], axis=0),
                jnp.concatenate([tr, tr], axis=0),
                jnp.concatenate([tlam, tlam], axis=0),
                jnp.concatenate([trho, trho], axis=0),
                jnp.concatenate([tqtot, tqtot], axis=0),
                jnp.concatenate([trial_alphas, trial_alphas], axis=0),
                jnp.concatenate([norms, disabled_norms], axis=0),
                jnp.concatenate([gas_norms, gas_norms], axis=0),
                jnp.concatenate([cond_norms, cond_norms], axis=0),
                jnp.concatenate([budget_norms, budget_norms], axis=0),
                jnp.concatenate([budget_relative_maxes, budget_relative_maxes], axis=0),
                jnp.concatenate([comp_norms, comp_norms], axis=0),
                jnp.concatenate([total_norms, total_norms], axis=0),
                jnp.concatenate(
                    [restoration_trial_flags, restoration_trial_flags],
                    axis=0,
                ),
                jnp.concatenate(
                    [
                        budget_restoration_trial_flags,
                        budget_restoration_trial_flags,
                    ],
                    axis=0,
                ),
                jnp.concatenate(
                    [
                        jnp.zeros((norms.shape[0],), dtype=bool),
                        jnp.ones((norms.shape[0],), dtype=bool),
                    ],
                    axis=0,
                ),
                jnp.asarray(False),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(0.0, dtype=q.dtype),
                jnp.asarray(0.0, dtype=q.dtype),
                jnp.asarray(0.0, dtype=q.dtype),
                jnp.asarray(0.0, dtype=q.dtype),
                jnp.asarray(0.0, dtype=q.dtype),
                jnp.asarray(0.0, dtype=q.dtype),
                jnp.asarray(0.0, dtype=q.dtype),
                jnp.asarray(jnp.inf, dtype=q.dtype),
                jnp.asarray(0.0, dtype=q.dtype),
                jnp.asarray(0.0, dtype=q.dtype),
                jnp.asarray(0.0, dtype=q.dtype),
                jnp.asarray(0.0, dtype=q.dtype),
                jnp.asarray(0.0, dtype=q.dtype),
                jnp.asarray(0.0, dtype=q.dtype),
                jnp.asarray(0.0, dtype=q.dtype),
                jnp.asarray(0.0, dtype=q.dtype),
                jnp.zeros((ag.shape[0] + 24,), dtype=q.dtype),
            )

        (
            tq,
            tr,
            tlam,
            trho,
            tqtot,
            trial_alphas,
            norms,
            gas_norms,
            cond_norms,
            budget_norms,
            budget_relative_maxes,
            comp_norms,
            total_norms,
            restoration_trial_flags,
            budget_restoration_trial_flags,
            soc_trial_flags,
            ipopt_soc_eligible,
            ipopt_soc_correction_count,
            ipopt_soc_all_directions_finite,
            ipopt_soc_filter_accepted,
            ipopt_soc_kappa_stopped,
            ipopt_soc_normal_theta,
            ipopt_soc_final_theta,
            ipopt_soc_normal_phi,
            ipopt_soc_final_phi,
            ipopt_soc_final_alpha,
            ipopt_soc_max_solve_linear_residual,
            ipopt_soc_max_solve_solution_norm,
            ipopt_soc_min_solve_singular_value,
            ipopt_soc_max_solve_condition_estimate,
            ipopt_soc_max_scaled_solve_condition_estimate,
            ipopt_soc_max_relative_solve_linear_residual,
            ipopt_soc_last_solve_solution_norm,
            ipopt_soc_last_relative_solve_linear_residual,
            ipopt_soc_last_solve_condition_estimate,
            ipopt_soc_last_scaled_solve_condition_estimate,
            ipopt_soc_last_solve_smallest_singular_value,
            ipopt_soc_last_smallest_right_singular_vector,
        ) = lax.cond(
            second_order_correction_enabled.astype(bool)
            & (~restoration_phase_active),
            append_soc_trials,
            append_disabled_soc_trials,
            operand=None,
        )
        ipopt_soc_last_null_lambda_norm = jnp.linalg.norm(
            ipopt_soc_last_smallest_right_singular_vector[: ag.shape[0]]
        )
        ipopt_soc_last_null_qtot_abs = jnp.abs(
            ipopt_soc_last_smallest_right_singular_vector[ag.shape[0]]
        )
        ipopt_soc_last_null_dominant_lambda_index = jnp.asarray(
            jnp.argmax(
                jnp.abs(
                    ipopt_soc_last_smallest_right_singular_vector[: ag.shape[0]]
                )
            ),
            dtype=jnp.int32,
        )
        ipopt_soc_last_null_dominant_lambda_abs = jnp.max(
            jnp.abs(ipopt_soc_last_smallest_right_singular_vector[: ag.shape[0]])
        )
        filter_thetas = jax.vmap(
            lambda trial_q, trial_r, trial_qtot: fixed_support_filter_theta(
                formula_matrix=ag,
                formula_matrix_cond_active=ac,
                element_inventory_target=target,
                q=trial_q,
                r=trial_r,
                qtot=trial_qtot,
                relative_floor=convergence_budget_relative_floor,
                use_l1_norm=ipopt_filter_use_l1_theta,
            )
        )(tq, tr, tqtot)
        barrier_objectives = jax.vmap(
            lambda trial_q, trial_r, trial_qtot: fixed_support_barrier_objective(
                q=trial_q,
                r=trial_r,
                qtot=trial_qtot,
                gas_stationarity_source=gas_stationarity_source,
                condensate_standard_source=hcond,
                qtot_reference=qtot,
                epsilon=epsilon_vec[0],
            )
        )(tq, tr, tqtot)
        barrier_objective_linearized_changes = jax.vmap(
            lambda trial_q, trial_r, trial_qtot: (
                fixed_support_barrier_objective_linearized_change(
                    q=q,
                    r=r,
                    qtot=qtot,
                    gas_stationarity_source=gas_stationarity_source,
                    condensate_standard_source=hcond,
                    qtot_reference=qtot,
                    epsilon=epsilon_vec[0],
                    delta_q=trial_q - q,
                    delta_r=trial_r - r,
                    delta_qtot=trial_qtot - qtot,
                )
            )
        )(tq, tr, tqtot)
        finite = jnp.isfinite(norms)
        budget_relative_acceptance_limit = jnp.maximum(
            budget_relative_acceptance_floor.astype(initial_budget_relative_max.dtype),
            jnp.maximum(
                initial_budget_relative_max
                + jnp.asarray(1.0e-8, dtype=initial_budget_relative_max.dtype),
                jnp.asarray(1.001, dtype=initial_budget_relative_max.dtype)
                * initial_budget_relative_max,
            ),
        )
        relative_budget_not_worse = (
            budget_relative_maxes <= budget_relative_acceptance_limit
        )
        combined_improved = norms < initial_norm
        accepted_mask = finite & relative_budget_not_worse & (norms < initial_norm)
        any_accepted = jnp.any(accepted_mask)
        accepted_candidate_count = jnp.sum(
            accepted_mask.astype(jnp.int32),
            dtype=jnp.int32,
        )
        first_index = jnp.argmax(accepted_mask)
        best_index = jnp.argmin(jnp.where(finite, norms, jnp.inf))
        protected_theta = jnp.maximum(
            jnp.maximum(gas_norms, cond_norms),
            jnp.maximum(budget_norms, total_norms),
        )
        initial_protected_theta = jnp.maximum(
            jnp.maximum(initial_gas_norm, initial_cond_norm),
            jnp.maximum(initial_budget_norm, initial_total_norm),
        )
        fallback_merit = comp_norms
        stationarity_floor = jnp.asarray(1.0e-300, dtype=initial_norm.dtype)
        complementarity_target_ratio = jnp.minimum(
            1.0,
            jnp.maximum(residual_crit, stationarity_floor)
            / jnp.maximum(initial_comp_norm, stationarity_floor),
        )
        required_complementarity_factor = jnp.exp(
            jnp.log(complementarity_target_ratio)
            / jnp.asarray(max_iter, dtype=initial_norm.dtype)
        )
        current_iterate_filter_mask = build_ipopt_current_iterate_filter_mask(
            finite=finite,
            protected_theta=protected_theta,
            complementarity_merit=fallback_merit,
            initial_protected_theta=initial_protected_theta,
            initial_complementarity_merit=initial_comp_norm,
            required_complementarity_factor=required_complementarity_factor,
            relaxed_fallback_enabled=relaxed_stationarity_fallback_enabled,
            relaxed_fallback_factor=relaxed_stationarity_fallback_factor,
        )
        budget_not_broken = budget_norms <= jnp.maximum(
            1.05 * initial_budget_norm,
            initial_budget_norm + jnp.asarray(1.0e-10, dtype=initial_budget_norm.dtype),
        )
        budget_relative_not_broken = (
            budget_relative_maxes <= budget_relative_acceptance_limit
        )
        ipopt_filter_budget_relative_not_broken = budget_relative_maxes <= jnp.maximum(
            budget_relative_acceptance_limit,
            ipopt_filter_budget_relative_max.astype(budget_relative_maxes.dtype),
        )
        combined_not_worse = norms <= (
            initial_norm
            + jnp.maximum(
                jnp.asarray(1.0e-12, dtype=initial_norm.dtype),
                1.0e-6 * jnp.maximum(
                    initial_norm,
                    jnp.asarray(1.0, dtype=initial_norm.dtype),
                ),
            )
        )
        effective_combined_not_worse = jnp.where(
            relaxed_stationarity_fallback_enabled,
            jnp.asarray(True, dtype=combined_not_worse.dtype),
            combined_not_worse,
        )
        fallback_mask = (
            current_iterate_filter_mask
            & budget_not_broken
            & jnp.where(
                ipopt_filter_acceptance_enabled.astype(bool),
                ipopt_filter_budget_relative_not_broken,
                budget_relative_not_broken,
            )
            & effective_combined_not_worse
        )
        current_filter_theta = fixed_support_filter_theta(
            formula_matrix=ag,
            formula_matrix_cond_active=ac,
            element_inventory_target=target,
            q=q,
            r=r,
            qtot=qtot,
            relative_floor=convergence_budget_relative_floor,
            use_l1_norm=ipopt_filter_use_l1_theta,
        )
        current_barrier_objective = fixed_support_barrier_objective(
            q=q,
            r=r,
            qtot=qtot,
            gas_stationarity_source=gas_stationarity_source,
            condensate_standard_source=hcond,
            qtot_reference=qtot,
            epsilon=epsilon_vec[0],
        )
        (
            persistent_filter_mask,
            filter_f_type_mask,
            filter_armijo_mask,
            filter_history_mask,
        ) = fixed_support_filter_acceptance(
            trial_phi=barrier_objectives,
            trial_theta=filter_thetas,
            trial_alpha=trial_alphas,
            trial_linearized_change=barrier_objective_linearized_changes,
            finite=finite,
            current_phi=current_barrier_objective,
            current_theta=current_filter_theta,
            initial_theta=initial_filter_theta,
            filter_phi=filter_phi_entries,
            filter_theta=filter_theta_entries,
            filter_valid=filter_valid_entries,
        )
        local_filter_mask = fixed_support_filter_acceptance(
            trial_phi=barrier_objectives,
            trial_theta=filter_thetas,
            trial_alpha=trial_alphas,
            trial_linearized_change=barrier_objective_linearized_changes,
            finite=finite,
            current_phi=current_barrier_objective,
            current_theta=current_filter_theta,
            initial_theta=initial_filter_theta,
            filter_phi=filter_phi_entries,
            filter_theta=filter_theta_entries,
            filter_valid=jnp.zeros_like(filter_valid_entries),
        )[0]
        use_persistent_filter = (
            ipopt_filter_acceptance_enabled.astype(bool)
            & (ipopt_filter_policy == "persistent_phi_theta")
        )
        normal_candidates_enabled = ~restoration_phase_active
        original_candidate_mask = (
            ~budget_restoration_trial_flags
        ) & normal_candidates_enabled
        original_filter_mask = persistent_filter_mask & original_candidate_mask
        any_original_filter_accepted = jnp.any(original_filter_mask)
        soft_restoration_mask = (
            finite
            & original_candidate_mask
            & (~soc_trial_flags)
            & (~restoration_trial_flags)
            & filter_history_mask
            & (norms <= 0.9 * initial_norm)
        )
        soft_restoration_mask = soft_restoration_mask & (
            ~any_original_filter_accepted
        )
        any_soft_restoration = jnp.any(soft_restoration_mask)
        full_restoration_mask = (
            persistent_filter_mask
            & budget_restoration_trial_flags
            & (filter_thetas <= 0.9 * current_filter_theta)
            & (~any_original_filter_accepted)
            & (~any_soft_restoration)
        )
        restoration_progress_limit = jnp.minimum(
            (1.0 - jnp.asarray(1.0e-4, dtype=current_filter_theta.dtype))
            * current_filter_theta,
            current_filter_theta
            - jnp.asarray(1.0e-12, dtype=current_filter_theta.dtype),
        )
        phase_restoration_mask = (
            finite
            & budget_restoration_trial_flags
            & (filter_thetas <= restoration_progress_limit)
            & (
                restoration_phase_active
                | (
                    restoration_can_enter
                    & (~any_original_filter_accepted)
                    & (~any_soft_restoration)
                )
            )
        )
        full_restoration_mask = jnp.where(
            jnp.asarray(budget_restoration_phase_enabled),
            phase_restoration_mask,
            full_restoration_mask,
        )
        persistent_accepted_mask = original_filter_mask | full_restoration_mask
        current_iterate_filter_mask = jnp.where(
            use_persistent_filter,
            persistent_filter_mask,
            current_iterate_filter_mask,
        )
        accepted_mask = jnp.where(
            use_persistent_filter,
            persistent_accepted_mask,
            accepted_mask,
        )
        fallback_mask = jnp.where(
            use_persistent_filter,
            soft_restoration_mask,
            fallback_mask,
        )
        accepted_mask = jnp.where(
            jnp.asarray(budget_restoration_phase_enabled),
            jnp.where(
                restoration_phase_active,
                phase_restoration_mask,
                accepted_mask | phase_restoration_mask,
            ),
            accepted_mask,
        )
        fallback_mask = jnp.where(
            jnp.asarray(budget_restoration_phase_enabled)
            & restoration_phase_active,
            jnp.zeros_like(fallback_mask),
            fallback_mask,
        )
        any_accepted = jnp.any(accepted_mask)
        accepted_candidate_count = jnp.sum(
            accepted_mask.astype(jnp.int32),
            dtype=jnp.int32,
        )
        first_index = jnp.argmax(accepted_mask)
        soc_candidate_mask = finite & soc_trial_flags
        soc_candidate_count = jnp.sum(
            soc_candidate_mask.astype(jnp.int32),
            dtype=jnp.int32,
        )
        any_soc_candidate = jnp.any(soc_candidate_mask)
        best_soc_index = jnp.argmin(
            jnp.where(soc_candidate_mask, norms, jnp.inf)
        )
        best_soc_index = jnp.where(
            any_soc_candidate,
            best_soc_index,
            jnp.asarray(0, dtype=best_soc_index.dtype),
        )
        soc_accepted_candidate_count = jnp.sum(
            (soc_candidate_mask & accepted_mask).astype(jnp.int32),
            dtype=jnp.int32,
        )
        soc_fallback_candidate_count = jnp.sum(
            (soc_candidate_mask & fallback_mask).astype(jnp.int32),
            dtype=jnp.int32,
        )
        soc_budget_relative_not_worse_candidate_count = jnp.sum(
            (soc_candidate_mask & relative_budget_not_worse).astype(jnp.int32),
            dtype=jnp.int32,
        )
        soc_filter_candidate_count = jnp.sum(
            (soc_candidate_mask & current_iterate_filter_mask).astype(jnp.int32),
            dtype=jnp.int32,
        )
        finite_candidate_count = jnp.sum(
            finite.astype(jnp.int32),
            dtype=jnp.int32,
        )
        combined_improved_candidate_count = jnp.sum(
            (finite & combined_improved).astype(jnp.int32),
            dtype=jnp.int32,
        )
        budget_relative_not_worse_candidate_count = jnp.sum(
            (finite & relative_budget_not_worse).astype(jnp.int32),
            dtype=jnp.int32,
        )
        filter_candidate_count = jnp.sum(
            (finite & current_iterate_filter_mask).astype(jnp.int32),
            dtype=jnp.int32,
        )
        budget_not_broken_candidate_count = jnp.sum(
            (finite & budget_not_broken).astype(jnp.int32),
            dtype=jnp.int32,
        )
        budget_relative_not_broken_candidate_count = jnp.sum(
            (finite & budget_relative_not_broken).astype(jnp.int32),
            dtype=jnp.int32,
        )
        combined_not_worse_candidate_count = jnp.sum(
            (finite & effective_combined_not_worse).astype(jnp.int32),
            dtype=jnp.int32,
        )
        accepted_or_fallback_mask = accepted_mask | fallback_mask
        rejected_trial_count = jnp.sum(
            (finite & (~accepted_or_fallback_mask)).astype(jnp.int32),
            dtype=jnp.int32,
        )
        any_fallback = jnp.any(fallback_mask)
        fallback_candidate_count = jnp.sum(
            fallback_mask.astype(jnp.int32),
            dtype=jnp.int32,
        )
        fallback_index = jnp.argmin(jnp.where(fallback_mask, fallback_merit, jnp.inf))
        acceptable_mask = accepted_mask | fallback_mask
        any_acceptable = any_accepted | any_fallback
        first_acceptable_index = jnp.argmax(acceptable_mask)
        max_acceptable_alpha = jnp.max(
            jnp.where(acceptable_mask, trial_alphas, -jnp.inf)
        )
        max_alpha_mask = acceptable_mask & (trial_alphas == max_acceptable_alpha)
        max_alpha_soc_available = jnp.any(max_alpha_mask & soc_trial_flags)
        max_alpha_tie_mask = max_alpha_mask & jnp.where(
            max_alpha_soc_available,
            soc_trial_flags,
            jnp.ones_like(soc_trial_flags, dtype=bool),
        )
        max_alpha_index = jnp.argmin(jnp.where(max_alpha_tie_mask, norms, jnp.inf))
        legacy_selected = jnp.where(
            any_accepted,
            first_index,
            jnp.where(any_fallback, fallback_index, best_index),
        )
        if line_search_candidate_selection_policy == "legacy":
            selected = legacy_selected
        elif line_search_candidate_selection_policy == "ipopt_sequential":
            selected = jnp.where(any_acceptable, first_acceptable_index, best_index)
        elif line_search_candidate_selection_policy == "ipopt_vectorized_max_alpha":
            selected = jnp.where(any_acceptable, max_alpha_index, best_index)
        else:
            raise ValueError(
                "line_search_candidate_selection_policy must be 'legacy', "
                "'ipopt_sequential', or 'ipopt_vectorized_max_alpha'."
            )
        step_accepted = any_accepted | any_fallback
        selected_strict_accepted = step_accepted & accepted_mask[selected]
        selected_fallback_accepted = (
            step_accepted
            & fallback_mask[selected]
            & (~selected_strict_accepted)
        )
        selected_converged = fixed_support_batch_converged(
            gas_norm=gas_norms[selected],
            condensate_stationarity_norm=cond_norms[selected],
            complementarity_norm=comp_norms[selected],
            total_density_norm=total_norms[selected],
            budget_relative_max=budget_relative_maxes[selected],
            log_tolerance=residual_crit,
            budget_relative_tolerance=budget_relative_crit,
            total_density_tolerance=total_density_crit,
        )
        step_converged = step_accepted & selected_converged
        selected_soc_accepted = step_accepted & soc_trial_flags[selected]
        selected_h_type = ~(
            filter_f_type_mask[selected] & filter_armijo_mask[selected]
        )
        (
            next_filter_phi_entries,
            next_filter_theta_entries,
            next_filter_valid_entries,
        ) = update_fixed_support_filter(
            filter_phi=filter_phi_entries,
            filter_theta=filter_theta_entries,
            filter_valid=filter_valid_entries,
            current_phi=current_barrier_objective,
            current_theta=current_filter_theta,
            add_entry=(
                use_persistent_filter
                & selected_strict_accepted
                & selected_h_type
                & (~(
                    jnp.asarray(budget_restoration_phase_enabled)
                    & budget_restoration_trial_flags[selected]
                ))
            ),
        )
        filter_history_rejected = jnp.any(
            local_filter_mask
            & (~filter_history_mask)
            & (~budget_restoration_trial_flags)
        )
        next_consecutive_filter_rejection_count = jnp.where(
            step_accepted & filter_history_rejected,
            consecutive_filter_rejection_count + jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        reset_filter = (
            use_persistent_filter
            & step_accepted
            & (next_consecutive_filter_rejection_count >= 5)
            & (filter_reset_count < 5)
        )
        next_filter_phi_entries = jnp.where(
            reset_filter, jnp.zeros_like(next_filter_phi_entries), next_filter_phi_entries
        )
        next_filter_theta_entries = jnp.where(
            reset_filter,
            jnp.zeros_like(next_filter_theta_entries),
            next_filter_theta_entries,
        )
        next_filter_valid_entries = jnp.where(
            reset_filter,
            jnp.zeros_like(next_filter_valid_entries),
            next_filter_valid_entries,
        )
        next_consecutive_filter_rejection_count = jnp.where(
            reset_filter,
            jnp.asarray(0, dtype=jnp.int32),
            next_consecutive_filter_rejection_count,
        )
        next_filter_reset_count = filter_reset_count + reset_filter.astype(jnp.int32)
        return (
            jnp.where(step_accepted, tq[selected], q),
            jnp.where(step_accepted, tr[selected], r),
            jnp.where(step_accepted, tlam[selected], lam),
            jnp.where(step_accepted, trho[selected], rho),
            jnp.where(step_accepted, tqtot[selected], qtot),
            jnp.where(step_accepted, norms[selected], initial_norm),
            step_accepted,
            selected_strict_accepted,
            selected_fallback_accepted,
            step_accepted
            & (
                restoration_trial_flags[selected]
                | budget_restoration_trial_flags[selected]
            ),
            selected_soc_accepted,
            initial_norm,
            selected_regularization_index,
            jnp.where(step_accepted, trial_alphas[selected], jnp.asarray(0.0, dtype=q.dtype)),
            rejected_trial_count,
            step_converged,
            alpha_boundary,
            alpha_r,
            alpha_rho,
            jnp.asarray(selected, dtype=jnp.int32),
            trial_alphas[selected],
            norms[selected],
            accepted_candidate_count,
            fallback_candidate_count,
            jnp.asarray(best_index, dtype=jnp.int32),
            trial_alphas[best_index],
            norms[best_index],
            gas_norms[best_index],
            cond_norms[best_index],
            budget_norms[best_index],
            budget_relative_maxes[best_index],
            comp_norms[best_index],
            total_norms[best_index],
            finite_candidate_count,
            combined_improved_candidate_count,
            budget_relative_not_worse_candidate_count,
            filter_candidate_count,
            budget_not_broken_candidate_count,
            budget_relative_not_broken_candidate_count,
            combined_not_worse_candidate_count,
            finite[best_index],
            combined_improved[best_index],
            relative_budget_not_worse[best_index],
            current_iterate_filter_mask[best_index],
            budget_not_broken[best_index],
            budget_relative_not_broken[best_index],
            effective_combined_not_worse[best_index],
            accepted_mask[best_index],
            fallback_mask[best_index],
            soc_candidate_count,
            soc_accepted_candidate_count,
            soc_fallback_candidate_count,
            soc_budget_relative_not_worse_candidate_count,
            soc_filter_candidate_count,
            any_soc_candidate,
            jnp.asarray(best_soc_index, dtype=jnp.int32),
            jnp.where(any_soc_candidate, trial_alphas[best_soc_index], 0.0),
            jnp.where(any_soc_candidate, norms[best_soc_index], 0.0),
            jnp.where(any_soc_candidate, gas_norms[best_soc_index], 0.0),
            jnp.where(any_soc_candidate, cond_norms[best_soc_index], 0.0),
            jnp.where(any_soc_candidate, budget_norms[best_soc_index], 0.0),
            jnp.where(
                any_soc_candidate,
                budget_relative_maxes[best_soc_index],
                0.0,
            ),
            jnp.where(any_soc_candidate, comp_norms[best_soc_index], 0.0),
            jnp.where(any_soc_candidate, total_norms[best_soc_index], 0.0),
            jnp.where(any_soc_candidate, combined_improved[best_soc_index], False),
            jnp.where(
                any_soc_candidate,
                relative_budget_not_worse[best_soc_index],
                False,
            ),
            jnp.where(
                any_soc_candidate,
                current_iterate_filter_mask[best_soc_index],
                False,
            ),
            jnp.where(any_soc_candidate, accepted_mask[best_soc_index], False),
            jnp.where(any_soc_candidate, fallback_mask[best_soc_index], False),
            gas_norms[selected],
            cond_norms[selected],
            budget_norms[selected],
            budget_relative_maxes[selected],
            comp_norms[selected],
            total_norms[selected],
            (
                trial_alphas,
                norms,
                gas_norms,
                cond_norms,
                budget_norms,
                budget_relative_maxes,
                comp_norms,
                total_norms,
                finite,
                accepted_mask,
                fallback_mask,
                current_iterate_filter_mask,
                relative_budget_not_worse,
                budget_not_broken,
                budget_relative_not_broken,
                effective_combined_not_worse,
                soc_trial_flags,
                restoration_trial_flags,
                budget_restoration_trial_flags,
                filter_thetas,
                barrier_objectives,
                barrier_objective_linearized_changes,
                jnp.full_like(
                    trial_alphas,
                    full_newton_linearized_residual_norm,
                ),
                filter_f_type_mask,
                filter_armijo_mask,
                filter_history_mask,
                jnp.full_like(
                    trial_alphas,
                    jnp.sum(filter_valid_entries.astype(jnp.int32)),
                    dtype=jnp.int32,
                ),
                soft_restoration_mask,
                jnp.full_like(
                    trial_alphas, ipopt_soc_eligible, dtype=jnp.int32
                ),
                jnp.full_like(
                    trial_alphas, ipopt_soc_correction_count, dtype=jnp.int32
                ),
                jnp.full_like(
                    trial_alphas,
                    ipopt_soc_eligible & ipopt_soc_all_directions_finite,
                    dtype=jnp.int32,
                ),
                jnp.full_like(
                    trial_alphas, ipopt_soc_filter_accepted, dtype=jnp.int32
                ),
                jnp.full_like(
                    trial_alphas, ipopt_soc_kappa_stopped, dtype=jnp.int32
                ),
                jnp.full_like(trial_alphas, ipopt_soc_normal_theta),
                jnp.full_like(trial_alphas, ipopt_soc_final_theta),
                jnp.full_like(trial_alphas, ipopt_soc_normal_phi),
                jnp.full_like(trial_alphas, ipopt_soc_final_phi),
                jnp.full_like(trial_alphas, ipopt_soc_final_alpha),
                jnp.full_like(
                    trial_alphas, ipopt_soc_max_solve_linear_residual
                ),
                jnp.full_like(trial_alphas, ipopt_soc_max_solve_solution_norm),
                jnp.full_like(
                    trial_alphas, ipopt_soc_min_solve_singular_value
                ),
                jnp.full_like(
                    trial_alphas, ipopt_soc_max_solve_condition_estimate
                ),
                jnp.full_like(
                    trial_alphas,
                    ipopt_soc_max_scaled_solve_condition_estimate,
                ),
                jnp.full_like(
                    trial_alphas,
                    ipopt_soc_max_relative_solve_linear_residual,
                ),
                jnp.full_like(
                    trial_alphas,
                    next_consecutive_filter_rejection_count,
                    dtype=jnp.int32,
                ),
                jnp.full_like(
                    trial_alphas, next_filter_reset_count, dtype=jnp.int32
                ),
                jnp.full_like(
                    trial_alphas, selected_soc_accepted, dtype=jnp.int32
                ),
                jnp.full_like(
                    trial_alphas, ipopt_soc_last_solve_solution_norm
                ),
                jnp.full_like(
                    trial_alphas,
                    ipopt_soc_last_relative_solve_linear_residual,
                ),
                jnp.full_like(
                    trial_alphas, ipopt_soc_last_solve_condition_estimate
                ),
                jnp.full_like(
                    trial_alphas,
                    ipopt_soc_last_scaled_solve_condition_estimate,
                ),
                jnp.full_like(
                    trial_alphas,
                    ipopt_soc_last_solve_smallest_singular_value,
                ),
                jnp.full_like(trial_alphas, ipopt_soc_last_null_lambda_norm),
                jnp.full_like(trial_alphas, ipopt_soc_last_null_qtot_abs),
                jnp.full_like(
                    trial_alphas,
                    ipopt_soc_last_null_dominant_lambda_index,
                    dtype=jnp.int32,
                ),
                jnp.full_like(
                    trial_alphas,
                    ipopt_soc_last_null_dominant_lambda_abs,
                ),
                jnp.broadcast_to(
                    ipopt_soc_last_smallest_right_singular_vector,
                    (
                        trial_alphas.shape[0],
                        ipopt_soc_last_smallest_right_singular_vector.shape[0],
                    ),
                ),
            ),
            next_filter_phi_entries,
            next_filter_theta_entries,
            next_filter_valid_entries,
            next_consecutive_filter_rejection_count,
            next_filter_reset_count,
        )

    def run_one(
        q0: jnp.ndarray,
        r0: jnp.ndarray,
        qtot0: jnp.ndarray,
        lam_init: jnp.ndarray,
        rho_init_one: jnp.ndarray,
        gas_source_init: jnp.ndarray,
        use_external_gas_source_one: jnp.ndarray,
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
        use_requested_epsilon = use_solver_epsilon_one | (~use_legacy_capacity_epsilon)
        epsilon_vec = jnp.where(
            use_requested_epsilon,
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
            use_external_gas_source_one,
            gas_source_init,
            hgas + ln_pressure - qtot0,
        )
        gas_step_source = jnp.where(
            use_external_gas_source_one,
            gas_source_init,
            hgas,
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
        lam0_gas_damped = scaled_damped_lstsq(
            ag.T,
            q0 + gas_stationarity_source_init,
        )
        lam0_joint_damped = scaled_damped_lstsq(
            jnp.concatenate([ag.T, ac.T], axis=0),
            jnp.concatenate([q0 + gas_stationarity_source_init, hcond - eta0]),
        )

        def initial_residual_for_lambda(lami: jnp.ndarray) -> jnp.ndarray:
            ni = jnp.exp(q0)
            mi = jnp.exp(r0)
            etai = jnp.exp(rho0)
            gas = q0 + gas_stationarity_source_init - ag.T @ lami
            cond = hcond - ac.T @ lami - etai
            budget = ag @ ni + ac @ mi - target
            comp = r0 + rho0 - epsilon_vec
            total_density = jnp.asarray(
                [jnp.sum(ni) - jnp.exp(qtot0)],
                dtype=q0.dtype,
            )
            jac_mask = jnp.ones((r0.shape[0],), dtype=bool)
            return l2(
                jnp.concatenate(
                    [
                        gas,
                        jnp.where(jac_mask, cond, 0.0),
                        budget,
                        comp,
                        total_density,
                    ]
                )
            )

        if lambda_initialization == "provided":
            lam0 = lam_init
            lambda_selection_index = jnp.asarray(0, dtype=jnp.int32)
        elif lambda_initialization == "best_residual":
            lambda_candidates = jnp.stack(
                [
                    lam_init,
                    lam0_gas,
                    lam0_joint,
                    lam0_gas_damped,
                    lam0_joint_damped,
                ],
                axis=0,
            )
            lambda_candidate_residuals = jax.vmap(initial_residual_for_lambda)(
                lambda_candidates
            )
            lambda_selection_index = jnp.asarray(
                jnp.argmin(lambda_candidate_residuals),
                dtype=jnp.int32,
            )
            lam0 = lambda_candidates[lambda_selection_index]
        elif lambda_initialization == "gas_cond_lstsq":
            lam0 = lam0_joint
            lambda_selection_index = jnp.asarray(2, dtype=jnp.int32)
        else:
            lam0 = lam0_gas
            lambda_selection_index = jnp.asarray(1, dtype=jnp.int32)
        residual_crit = residual_tolerance_multiplier * convergence_log_tolerance
        budget_relative_crit = (
            residual_tolerance_multiplier * convergence_budget_relative_tolerance
        )
        total_density_crit = (
            residual_tolerance_multiplier * convergence_total_density_tolerance
        )
        initial_filter_theta = fixed_support_filter_theta(
            formula_matrix=ag,
            formula_matrix_cond_active=ac,
            element_inventory_target=target,
            q=q0,
            r=r0,
            qtot=qtot0,
            relative_floor=convergence_budget_relative_floor,
            use_l1_norm=ipopt_filter_use_l1_theta,
        )
        filter_phi_entries0 = jnp.zeros((max_iter,), dtype=q0.dtype)
        filter_theta_entries0 = jnp.zeros((max_iter,), dtype=q0.dtype)
        filter_valid_entries0 = jnp.zeros((max_iter,), dtype=bool)
        tiny_step_threshold = jnp.asarray(1.0e-3, dtype=q0.dtype)
        tiny_step_limit = jnp.asarray(tiny_step_consecutive_limit, dtype=jnp.int32)
        initial_step = step(
            q0,
            r0,
            lam0,
            rho0,
            qtot0,
            target,
            gas_step_source,
            hcond,
            ln_pressure,
            epsilon_vec,
            r_cap,
            use_external_gas_source_one,
            use_scalar_step_control,
            residual_crit,
            budget_relative_crit,
            total_density_crit,
            initial_filter_theta,
            filter_phi_entries0,
            filter_theta_entries0,
            filter_valid_entries0,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False, dtype=bool),
            jnp.asarray(False, dtype=bool),
            q0,
            r0,
            qtot0,
        )
        initial_residual = initial_step[11]
        initial_running = initial_residual > residual_crit

        def cond_fun(carry):
            (
                iteration,
                _q,
                _r,
                _lam,
                _rho,
                _qtot,
                _residual,
                _last_alpha,
                _rejected_count,
                _accepted_count,
                _normal_accepted_count,
                _fallback_accepted_count,
                _restoration_accepted_count,
                _soc_accepted_count,
                _adaptive_regularization_selected_count,
                _tiny_step_consecutive_count,
                _last_accepted,
                _last_alpha_boundary,
                _last_alpha_r,
                _last_alpha_rho,
                _last_selected_trial_index,
                _last_selected_trial_alpha,
                _last_selected_trial_residual,
                _last_accepted_candidate_count,
                _last_fallback_candidate_count,
                _last_best_trial_index,
                _last_best_trial_alpha,
                _last_best_trial_residual,
                _last_best_trial_gas_residual,
                _last_best_trial_cond_residual,
                _last_best_trial_budget_residual,
                _last_best_trial_budget_relative_max,
                _last_best_trial_comp_residual,
                _last_best_trial_total_residual,
                _last_finite_candidate_count,
                _last_combined_improved_candidate_count,
                _last_budget_relative_not_worse_candidate_count,
                _last_filter_candidate_count,
                _last_budget_not_broken_candidate_count,
                _last_budget_relative_not_broken_candidate_count,
                _last_combined_not_worse_candidate_count,
                _last_best_trial_finite,
                _last_best_trial_combined_improved,
                _last_best_trial_budget_relative_not_worse,
                _last_best_trial_filter_accepted,
                _last_best_trial_budget_not_broken,
                _last_best_trial_budget_relative_not_broken,
                _last_best_trial_combined_not_worse,
                _last_best_trial_accepted,
                _last_best_trial_fallback_accepted,
                _last_soc_candidate_count,
                _last_soc_accepted_candidate_count,
                _last_soc_fallback_candidate_count,
                _last_soc_budget_relative_not_worse_candidate_count,
                _last_soc_filter_candidate_count,
                _last_soc_best_trial_present,
                _last_soc_best_trial_index,
                _last_soc_best_trial_alpha,
                _last_soc_best_trial_residual,
                _last_soc_best_trial_gas_residual,
                _last_soc_best_trial_cond_residual,
                _last_soc_best_trial_budget_residual,
                _last_soc_best_trial_budget_relative_max,
                _last_soc_best_trial_comp_residual,
                _last_soc_best_trial_total_residual,
                _last_soc_best_trial_combined_improved,
                _last_soc_best_trial_budget_relative_not_worse,
                _last_soc_best_trial_filter_accepted,
                _last_soc_best_trial_accepted,
                _last_soc_best_trial_fallback_accepted,
                _last_selected_trial_gas_residual,
                _last_selected_trial_cond_residual,
                _last_selected_trial_budget_residual,
                _last_selected_trial_budget_relative_max,
                _last_selected_trial_comp_residual,
                _last_selected_trial_total_residual,
                _last_candidate_diagnostics,
                _filter_phi_entries,
                _filter_theta_entries,
                _filter_valid_entries,
                _consecutive_filter_rejection_count,
                _filter_reset_count,
                still_running,
                _restoration_phase_state,
            ) = carry
            return (iteration < jnp.asarray(max_iter, dtype=jnp.int32)) & still_running

        def body(carry):
            (
                iteration,
                q,
                r,
                lam,
                rho,
                qtot,
                residual,
                last_alpha,
                rejected_count,
                accepted_count,
                normal_accepted_count,
                fallback_accepted_count,
                restoration_accepted_count,
                soc_accepted_count,
                adaptive_regularization_selected_count,
                tiny_step_consecutive_count,
                _last_accepted,
                last_alpha_boundary,
                last_alpha_r,
                last_alpha_rho,
                last_selected_trial_index,
                last_selected_trial_alpha,
                last_selected_trial_residual,
                last_accepted_candidate_count,
                last_fallback_candidate_count,
                last_best_trial_index,
                last_best_trial_alpha,
                last_best_trial_residual,
                last_best_trial_gas_residual,
                last_best_trial_cond_residual,
                last_best_trial_budget_residual,
                last_best_trial_budget_relative_max,
                last_best_trial_comp_residual,
                last_best_trial_total_residual,
                last_finite_candidate_count,
                last_combined_improved_candidate_count,
                last_budget_relative_not_worse_candidate_count,
                last_filter_candidate_count,
                last_budget_not_broken_candidate_count,
                last_budget_relative_not_broken_candidate_count,
                last_combined_not_worse_candidate_count,
                last_best_trial_finite,
                last_best_trial_combined_improved,
                last_best_trial_budget_relative_not_worse,
                last_best_trial_filter_accepted,
                last_best_trial_budget_not_broken,
                last_best_trial_budget_relative_not_broken,
                last_best_trial_combined_not_worse,
                last_best_trial_accepted,
                last_best_trial_fallback_accepted,
                last_soc_candidate_count,
                last_soc_accepted_candidate_count,
                last_soc_fallback_candidate_count,
                last_soc_budget_relative_not_worse_candidate_count,
                last_soc_filter_candidate_count,
                last_soc_best_trial_present,
                last_soc_best_trial_index,
                last_soc_best_trial_alpha,
                last_soc_best_trial_residual,
                last_soc_best_trial_gas_residual,
                last_soc_best_trial_cond_residual,
                last_soc_best_trial_budget_residual,
                last_soc_best_trial_budget_relative_max,
                last_soc_best_trial_comp_residual,
                last_soc_best_trial_total_residual,
                last_soc_best_trial_combined_improved,
                last_soc_best_trial_budget_relative_not_worse,
                last_soc_best_trial_filter_accepted,
                last_soc_best_trial_accepted,
                last_soc_best_trial_fallback_accepted,
                last_selected_trial_gas_residual,
                last_selected_trial_cond_residual,
                last_selected_trial_budget_residual,
                last_selected_trial_budget_relative_max,
                last_selected_trial_comp_residual,
                last_selected_trial_total_residual,
                last_candidate_diagnostics,
                filter_phi_entries,
                filter_theta_entries,
                filter_valid_entries,
                consecutive_filter_rejection_count,
                filter_reset_count,
                still_running,
                restoration_phase_state,
            ) = carry
            (
                restoration_entry_q,
                restoration_entry_r,
                restoration_entry_rho,
                restoration_entry_qtot,
                restoration_entry_theta,
                restoration_phase_active,
                restoration_cooldown,
                amount_restoration_accepted_count,
                restoration_phase_entry_count,
                restoration_phase_exit_count,
                restoration_bound_reset_count,
                restoration_equality_reset_count,
                restoration_last_exit_theta,
                restoration_last_dual_alpha,
                restoration_entry_residual_vector,
                restoration_best_residual_vector,
                restoration_best_theta,
                restoration_last_exit_predual_residual_vector,
                restoration_last_exit_postdual_residual_vector,
                restoration_first_normal_residual_vector,
                restoration_first_normal_attempted,
                restoration_first_normal_accepted,
                restoration_first_normal_selected_type,
                restoration_return_probe_pending,
                restoration_active_accepted_count,
                restoration_last_active_accepted_count,
            ) = restoration_phase_state
            restoration_can_enter = (
                (~restoration_phase_active)
                & (restoration_cooldown <= jnp.asarray(0, dtype=jnp.int32))
            )
            restoration_reference_q = jnp.where(
                restoration_phase_active, restoration_entry_q, q
            )
            restoration_reference_r = jnp.where(
                restoration_phase_active, restoration_entry_r, r
            )
            restoration_reference_qtot = jnp.where(
                restoration_phase_active, restoration_entry_qtot, qtot
            )

            def diagnostic_residual_vector(
                diagnostic_q: jnp.ndarray,
                diagnostic_r: jnp.ndarray,
                diagnostic_lam: jnp.ndarray,
                diagnostic_rho: jnp.ndarray,
                diagnostic_qtot: jnp.ndarray,
            ) -> jnp.ndarray:
                diagnostic_n = jnp.exp(diagnostic_q)
                diagnostic_m = jnp.exp(diagnostic_r)
                diagnostic_eta = jnp.exp(diagnostic_rho)
                diagnostic_gas_source = jnp.where(
                    use_external_gas_source_one,
                    gas_step_source,
                    gas_step_source + ln_pressure - diagnostic_qtot,
                )
                diagnostic_gas = (
                    diagnostic_q
                    + diagnostic_gas_source
                    - ag.T @ diagnostic_lam
                )
                diagnostic_cond = (
                    hcond - ac.T @ diagnostic_lam - diagnostic_eta
                )
                diagnostic_budget = (
                    ag @ diagnostic_n + ac @ diagnostic_m - target
                )
                diagnostic_comp = diagnostic_r + diagnostic_rho - epsilon_vec
                diagnostic_total = jnp.asarray(
                    [jnp.sum(diagnostic_n) - jnp.exp(diagnostic_qtot)],
                    dtype=q.dtype,
                )
                diagnostic_full = l2(
                    jnp.concatenate(
                        [
                            diagnostic_gas,
                            diagnostic_cond,
                            diagnostic_budget,
                            diagnostic_comp,
                            diagnostic_total,
                        ]
                    )
                )
                diagnostic_budget_relative = fixed_support_budget_relative_max(
                    budget_residual=diagnostic_budget,
                    target=target,
                    relative_floor=jnp.maximum(
                        jnp.asarray(jnp.finfo(q.dtype).tiny, dtype=q.dtype),
                        convergence_budget_relative_floor.astype(q.dtype),
                    ),
                )
                return jnp.asarray(
                    [
                        diagnostic_full,
                        l2(diagnostic_gas),
                        l2(diagnostic_cond),
                        diagnostic_budget_relative,
                        l2(diagnostic_comp),
                        l2(diagnostic_total),
                    ],
                    dtype=q.dtype,
                )

            current_residual_vector = diagnostic_residual_vector(
                q,
                r,
                lam,
                rho,
                qtot,
            )
            (
                next_q,
                next_r,
                next_lam,
                next_rho,
                next_qtot,
                next_residual,
                accepted,
                normal_accepted,
                fallback_accepted,
                restoration_accepted,
                soc_accepted,
                _initial_residual,
                regularization_index,
                step_alpha,
                step_rejected_count,
                step_converged,
                step_alpha_boundary,
                step_alpha_r,
                step_alpha_rho,
                step_selected_trial_index,
                step_selected_trial_alpha,
                step_selected_trial_residual,
                step_accepted_candidate_count,
                step_fallback_candidate_count,
                step_best_trial_index,
                step_best_trial_alpha,
                step_best_trial_residual,
                step_best_trial_gas_residual,
                step_best_trial_cond_residual,
                step_best_trial_budget_residual,
                step_best_trial_budget_relative_max,
                step_best_trial_comp_residual,
                step_best_trial_total_residual,
                step_finite_candidate_count,
                step_combined_improved_candidate_count,
                step_budget_relative_not_worse_candidate_count,
                step_filter_candidate_count,
                step_budget_not_broken_candidate_count,
                step_budget_relative_not_broken_candidate_count,
                step_combined_not_worse_candidate_count,
                step_best_trial_finite,
                step_best_trial_combined_improved,
                step_best_trial_budget_relative_not_worse,
                step_best_trial_filter_accepted,
                step_best_trial_budget_not_broken,
                step_best_trial_budget_relative_not_broken,
                step_best_trial_combined_not_worse,
                step_best_trial_accepted,
                step_best_trial_fallback_accepted,
                step_soc_candidate_count,
                step_soc_accepted_candidate_count,
                step_soc_fallback_candidate_count,
                step_soc_budget_relative_not_worse_candidate_count,
                step_soc_filter_candidate_count,
                step_soc_best_trial_present,
                step_soc_best_trial_index,
                step_soc_best_trial_alpha,
                step_soc_best_trial_residual,
                step_soc_best_trial_gas_residual,
                step_soc_best_trial_cond_residual,
                step_soc_best_trial_budget_residual,
                step_soc_best_trial_budget_relative_max,
                step_soc_best_trial_comp_residual,
                step_soc_best_trial_total_residual,
                step_soc_best_trial_combined_improved,
                step_soc_best_trial_budget_relative_not_worse,
                step_soc_best_trial_filter_accepted,
                step_soc_best_trial_accepted,
                step_soc_best_trial_fallback_accepted,
                step_selected_trial_gas_residual,
                step_selected_trial_cond_residual,
                step_selected_trial_budget_residual,
                step_selected_trial_budget_relative_max,
                step_selected_trial_comp_residual,
                step_selected_trial_total_residual,
                step_candidate_diagnostics,
                step_filter_phi_entries,
                step_filter_theta_entries,
                step_filter_valid_entries,
                step_consecutive_filter_rejection_count,
                step_filter_reset_count,
            ) = step(
                q,
                r,
                lam,
                rho,
                qtot,
                target,
                gas_step_source,
                hcond,
                ln_pressure,
                epsilon_vec,
                r_cap,
                use_external_gas_source_one,
                use_scalar_step_control,
                residual_crit,
                budget_relative_crit,
                total_density_crit,
                initial_filter_theta,
                filter_phi_entries,
                filter_theta_entries,
                filter_valid_entries,
                consecutive_filter_rejection_count,
                filter_reset_count,
                restoration_phase_active,
                restoration_can_enter,
                restoration_reference_q,
                restoration_reference_r,
                restoration_reference_qtot,
            )
            apply_step = still_running & accepted
            selected_budget_restoration = (
                apply_step
                & step_candidate_diagnostics[18][step_selected_trial_index]
            )
            selected_phase_restoration = (
                jnp.asarray(budget_restoration_phase_enabled)
                & selected_budget_restoration
            )
            current_filter_theta = fixed_support_filter_theta(
                formula_matrix=ag,
                formula_matrix_cond_active=ac,
                element_inventory_target=target,
                q=q,
                r=r,
                qtot=qtot,
                relative_floor=convergence_budget_relative_floor,
                use_l1_norm=ipopt_filter_use_l1_theta,
            )
            phase_entered, _, _ = fixed_support_restoration_phase_transition(
                phase_active=restoration_phase_active,
                cooldown=restoration_cooldown,
                normal_iteration_attempted=(~selected_phase_restoration),
                selected_amount_restoration=selected_phase_restoration,
                phase_exit=jnp.asarray(False, dtype=bool),
                cooldown_iterations=budget_restoration_phase_cooldown_iterations,
            )
            current_barrier_objective = fixed_support_barrier_objective(
                q=q,
                r=r,
                qtot=qtot,
                gas_stationarity_source=jnp.where(
                    use_external_gas_source_one,
                    gas_step_source,
                    gas_step_source + ln_pressure - qtot,
                ),
                condensate_standard_source=hcond,
                qtot_reference=qtot,
                epsilon=epsilon_vec[0],
            )
            (
                prepared_filter_phi_entries,
                prepared_filter_theta_entries,
                prepared_filter_valid_entries,
            ) = prepare_fixed_support_restoration_filter(
                filter_phi=step_filter_phi_entries,
                filter_theta=step_filter_theta_entries,
                filter_valid=step_filter_valid_entries,
                current_phi=current_barrier_objective,
                current_theta=current_filter_theta,
                phase_entered=(
                    phase_entered
                    & ipopt_filter_acceptance_enabled.astype(bool)
                    & (ipopt_filter_policy == "persistent_phi_theta")
                ),
            )
            effective_entry_q = jnp.where(
                phase_entered,
                q,
                restoration_entry_q,
            )
            effective_entry_r = jnp.where(
                phase_entered,
                r,
                restoration_entry_r,
            )
            effective_entry_rho = jnp.where(
                phase_entered,
                rho,
                restoration_entry_rho,
            )
            effective_entry_qtot = jnp.where(
                phase_entered,
                qtot,
                restoration_entry_qtot,
            )
            effective_entry_theta = jnp.where(
                phase_entered,
                current_filter_theta,
                restoration_entry_theta,
            )
            selected_filter_theta = step_candidate_diagnostics[19][
                step_selected_trial_index
            ]
            selected_original_filter_accepted = step_candidate_diagnostics[11][
                step_selected_trial_index
            ]
            restoration_phase_exit = fixed_support_restoration_phase_exit(
                selected_amount_restoration=selected_phase_restoration,
                trial_theta=selected_filter_theta,
                entry_theta=effective_entry_theta,
                theta_reduction=budget_restoration_phase_theta_reduction,
                budget_relative_residual_max=(
                    step_selected_trial_budget_relative_max
                ),
                budget_relative_tolerance=budget_relative_crit,
                total_density_residual=step_selected_trial_total_residual,
                total_density_tolerance=total_density_crit,
                original_filter_accepted=selected_original_filter_accepted,
            )

            def return_original_duals(_unused: None):
                return fixed_support_ipopt_restoration_dual_return(
                    formula_matrix=ag,
                    formula_matrix_cond_active=ac,
                    restored_q=next_q,
                    restored_r=next_r,
                    restored_qtot=next_qtot,
                    qtot_reference=effective_entry_qtot,
                    gas_stationarity_source=jnp.where(
                        use_external_gas_source_one,
                        gas_step_source,
                        gas_step_source + ln_pressure - effective_entry_qtot,
                    ),
                    condensate_standard_source=hcond,
                    current_r=effective_entry_r,
                    current_rho=effective_entry_rho,
                    barrier=jnp.exp(epsilon_vec[0]),
                )

            def keep_restoration_duals(_unused: None):
                return (
                    next_lam,
                    next_rho,
                    jnp.asarray(1.0, dtype=q.dtype),
                    jnp.asarray(False, dtype=bool),
                    jnp.asarray(False, dtype=bool),
                )

            (
                returned_lam,
                returned_rho,
                _restoration_dual_alpha,
                restoration_bound_reset,
                restoration_equality_reset,
            ) = lax.cond(
                restoration_phase_exit,
                return_original_duals,
                keep_restoration_duals,
                operand=None,
            )

            returned_residual_vector = diagnostic_residual_vector(
                next_q,
                next_r,
                returned_lam,
                returned_rho,
                next_qtot,
            )
            returned_residual = jnp.where(
                restoration_phase_exit,
                returned_residual_vector[0],
                next_residual,
            )
            (
                _phase_entered_again,
                next_restoration_phase_active,
                next_restoration_cooldown,
            ) = fixed_support_restoration_phase_transition(
                phase_active=restoration_phase_active,
                cooldown=restoration_cooldown,
                normal_iteration_attempted=(~selected_phase_restoration),
                selected_amount_restoration=selected_phase_restoration,
                phase_exit=restoration_phase_exit,
                cooldown_iterations=budget_restoration_phase_cooldown_iterations,
            )
            selected_residual_vector = jnp.asarray(
                [
                    step_selected_trial_residual,
                    step_selected_trial_gas_residual,
                    step_selected_trial_cond_residual,
                    step_selected_trial_budget_relative_max,
                    step_selected_trial_comp_residual,
                    step_selected_trial_total_residual,
                ],
                dtype=q.dtype,
            )
            effective_entry_residual_vector = jnp.where(
                phase_entered,
                current_residual_vector,
                restoration_entry_residual_vector,
            )
            restoration_best_improved = selected_phase_restoration & (
                phase_entered | (selected_filter_theta < restoration_best_theta)
            )
            next_restoration_best_residual_vector = jnp.where(
                restoration_best_improved,
                selected_residual_vector,
                restoration_best_residual_vector,
            )
            next_restoration_best_theta = jnp.where(
                restoration_best_improved,
                selected_filter_theta,
                restoration_best_theta,
            )
            first_normal_probe = (
                restoration_return_probe_pending & (~selected_phase_restoration)
            )
            selected_stationarity_restoration = (
                step_candidate_diagnostics[17][step_selected_trial_index]
                & (~selected_budget_restoration)
            )
            first_normal_selected_type = jnp.where(
                step_candidate_diagnostics[16][step_selected_trial_index],
                jnp.asarray(2, dtype=jnp.int32),
                jnp.where(
                    selected_stationarity_restoration,
                    jnp.asarray(1, dtype=jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32),
                ),
            )
            current_phase_accepted_count = jnp.where(
                phase_entered,
                jnp.asarray(1, dtype=jnp.int32),
                restoration_active_accepted_count
                + selected_phase_restoration.astype(jnp.int32),
            )
            next_restoration_phase_state = (
                effective_entry_q,
                effective_entry_r,
                effective_entry_rho,
                effective_entry_qtot,
                effective_entry_theta,
                next_restoration_phase_active,
                next_restoration_cooldown,
                amount_restoration_accepted_count
                + selected_budget_restoration.astype(jnp.int32),
                restoration_phase_entry_count + phase_entered.astype(jnp.int32),
                restoration_phase_exit_count
                + restoration_phase_exit.astype(jnp.int32),
                restoration_bound_reset_count
                + restoration_bound_reset.astype(jnp.int32),
                restoration_equality_reset_count
                + restoration_equality_reset.astype(jnp.int32),
                jnp.where(
                    restoration_phase_exit,
                    selected_filter_theta,
                    restoration_last_exit_theta,
                ),
                jnp.where(
                    restoration_phase_exit,
                    _restoration_dual_alpha,
                    restoration_last_dual_alpha,
                ),
                effective_entry_residual_vector,
                next_restoration_best_residual_vector,
                next_restoration_best_theta,
                jnp.where(
                    restoration_phase_exit,
                    selected_residual_vector,
                    restoration_last_exit_predual_residual_vector,
                ),
                jnp.where(
                    restoration_phase_exit,
                    returned_residual_vector,
                    restoration_last_exit_postdual_residual_vector,
                ),
                jnp.where(
                    restoration_phase_exit,
                    jnp.zeros((6,), dtype=q.dtype),
                    jnp.where(
                        first_normal_probe,
                        selected_residual_vector,
                        restoration_first_normal_residual_vector,
                    ),
                ),
                jnp.where(
                    restoration_phase_exit,
                    jnp.asarray(False, dtype=bool),
                    restoration_first_normal_attempted | first_normal_probe,
                ),
                jnp.where(
                    restoration_phase_exit,
                    jnp.asarray(False, dtype=bool),
                    jnp.where(
                        first_normal_probe,
                        accepted,
                        restoration_first_normal_accepted,
                    ),
                ),
                jnp.where(
                    restoration_phase_exit,
                    jnp.asarray(3, dtype=jnp.int32),
                    jnp.where(
                        first_normal_probe,
                        first_normal_selected_type,
                        restoration_first_normal_selected_type,
                    ),
                ),
                (
                    restoration_return_probe_pending & (~first_normal_probe)
                )
                | restoration_phase_exit,
                jnp.where(
                    restoration_phase_exit,
                    jnp.asarray(0, dtype=jnp.int32),
                    current_phase_accepted_count,
                ),
                jnp.where(
                    restoration_phase_exit,
                    current_phase_accepted_count,
                    restoration_last_active_accepted_count,
                ),
            )
            next_tiny_step_consecutive_count = jnp.where(
                apply_step & (step_alpha <= tiny_step_threshold),
                tiny_step_consecutive_count + jnp.asarray(1, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            )
            tiny_step_stalled = next_tiny_step_consecutive_count >= tiny_step_limit
            cooldown_retry = (
                jnp.asarray(budget_restoration_phase_enabled)
                & still_running
                & (~restoration_phase_active)
                & (restoration_cooldown > 0)
                & (~accepted)
            )
            next_still_running = (
                apply_step & (~step_converged) & (~tiny_step_stalled)
            ) | cooldown_retry
            next_rejected_count = jnp.asarray(
                rejected_count + step_rejected_count,
                dtype=jnp.int32,
            )
            return (
                iteration + jnp.asarray(1, dtype=jnp.int32),
                jnp.where(apply_step, next_q, q),
                jnp.where(apply_step, next_r, r),
                jnp.where(apply_step, returned_lam, lam),
                jnp.where(apply_step, returned_rho, rho),
                jnp.where(apply_step, next_qtot, qtot),
                jnp.where(still_running, returned_residual, residual),
                jnp.where(apply_step, step_alpha, last_alpha),
                jnp.where(still_running, next_rejected_count, rejected_count),
                accepted_count + apply_step.astype(jnp.int32),
                normal_accepted_count
                + (
                    still_running & normal_accepted & (~restoration_accepted)
                    & (~soc_accepted)
                ).astype(jnp.int32),
                fallback_accepted_count
                + (still_running & fallback_accepted).astype(jnp.int32),
                restoration_accepted_count
                + (
                    still_running
                    & restoration_accepted
                    & (~selected_budget_restoration)
                ).astype(jnp.int32),
                soc_accepted_count + (still_running & soc_accepted).astype(jnp.int32),
                adaptive_regularization_selected_count
                + (still_running & accepted & (regularization_index > 0)).astype(
                    jnp.int32
                ),
                next_tiny_step_consecutive_count,
                still_running & accepted,
                jnp.where(still_running, step_alpha_boundary, last_alpha_boundary),
                jnp.where(still_running, step_alpha_r, last_alpha_r),
                jnp.where(still_running, step_alpha_rho, last_alpha_rho),
                jnp.where(
                    still_running,
                    step_selected_trial_index,
                    last_selected_trial_index,
                ),
                jnp.where(
                    still_running,
                    step_selected_trial_alpha,
                    last_selected_trial_alpha,
                ),
                jnp.where(
                    still_running,
                    step_selected_trial_residual,
                    last_selected_trial_residual,
                ),
                jnp.where(
                    still_running,
                    step_accepted_candidate_count,
                    last_accepted_candidate_count,
                ),
                jnp.where(
                    still_running,
                    step_fallback_candidate_count,
                    last_fallback_candidate_count,
                ),
                jnp.where(still_running, step_best_trial_index, last_best_trial_index),
                jnp.where(still_running, step_best_trial_alpha, last_best_trial_alpha),
                jnp.where(
                    still_running,
                    step_best_trial_residual,
                    last_best_trial_residual,
                ),
                jnp.where(
                    still_running,
                    step_best_trial_gas_residual,
                    last_best_trial_gas_residual,
                ),
                jnp.where(
                    still_running,
                    step_best_trial_cond_residual,
                    last_best_trial_cond_residual,
                ),
                jnp.where(
                    still_running,
                    step_best_trial_budget_residual,
                    last_best_trial_budget_residual,
                ),
                jnp.where(
                    still_running,
                    step_best_trial_budget_relative_max,
                    last_best_trial_budget_relative_max,
                ),
                jnp.where(
                    still_running,
                    step_best_trial_comp_residual,
                    last_best_trial_comp_residual,
                ),
                jnp.where(
                    still_running,
                    step_best_trial_total_residual,
                    last_best_trial_total_residual,
                ),
                jnp.where(
                    still_running,
                    step_finite_candidate_count,
                    last_finite_candidate_count,
                ),
                jnp.where(
                    still_running,
                    step_combined_improved_candidate_count,
                    last_combined_improved_candidate_count,
                ),
                jnp.where(
                    still_running,
                    step_budget_relative_not_worse_candidate_count,
                    last_budget_relative_not_worse_candidate_count,
                ),
                jnp.where(
                    still_running,
                    step_filter_candidate_count,
                    last_filter_candidate_count,
                ),
                jnp.where(
                    still_running,
                    step_budget_not_broken_candidate_count,
                    last_budget_not_broken_candidate_count,
                ),
                jnp.where(
                    still_running,
                    step_budget_relative_not_broken_candidate_count,
                    last_budget_relative_not_broken_candidate_count,
                ),
                jnp.where(
                    still_running,
                    step_combined_not_worse_candidate_count,
                    last_combined_not_worse_candidate_count,
                ),
                jnp.where(
                    still_running,
                    step_best_trial_finite,
                    last_best_trial_finite,
                ),
                jnp.where(
                    still_running,
                    step_best_trial_combined_improved,
                    last_best_trial_combined_improved,
                ),
                jnp.where(
                    still_running,
                    step_best_trial_budget_relative_not_worse,
                    last_best_trial_budget_relative_not_worse,
                ),
                jnp.where(
                    still_running,
                    step_best_trial_filter_accepted,
                    last_best_trial_filter_accepted,
                ),
                jnp.where(
                    still_running,
                    step_best_trial_budget_not_broken,
                    last_best_trial_budget_not_broken,
                ),
                jnp.where(
                    still_running,
                    step_best_trial_budget_relative_not_broken,
                    last_best_trial_budget_relative_not_broken,
                ),
                jnp.where(
                    still_running,
                    step_best_trial_combined_not_worse,
                    last_best_trial_combined_not_worse,
                ),
                jnp.where(
                    still_running,
                    step_best_trial_accepted,
                    last_best_trial_accepted,
                ),
                jnp.where(
                    still_running,
                    step_best_trial_fallback_accepted,
                    last_best_trial_fallback_accepted,
                ),
                jnp.where(
                    still_running,
                    step_soc_candidate_count,
                    last_soc_candidate_count,
                ),
                jnp.where(
                    still_running,
                    step_soc_accepted_candidate_count,
                    last_soc_accepted_candidate_count,
                ),
                jnp.where(
                    still_running,
                    step_soc_fallback_candidate_count,
                    last_soc_fallback_candidate_count,
                ),
                jnp.where(
                    still_running,
                    step_soc_budget_relative_not_worse_candidate_count,
                    last_soc_budget_relative_not_worse_candidate_count,
                ),
                jnp.where(
                    still_running,
                    step_soc_filter_candidate_count,
                    last_soc_filter_candidate_count,
                ),
                jnp.where(
                    still_running,
                    step_soc_best_trial_present,
                    last_soc_best_trial_present,
                ),
                jnp.where(
                    still_running,
                    step_soc_best_trial_index,
                    last_soc_best_trial_index,
                ),
                jnp.where(
                    still_running,
                    step_soc_best_trial_alpha,
                    last_soc_best_trial_alpha,
                ),
                jnp.where(
                    still_running,
                    step_soc_best_trial_residual,
                    last_soc_best_trial_residual,
                ),
                jnp.where(
                    still_running,
                    step_soc_best_trial_gas_residual,
                    last_soc_best_trial_gas_residual,
                ),
                jnp.where(
                    still_running,
                    step_soc_best_trial_cond_residual,
                    last_soc_best_trial_cond_residual,
                ),
                jnp.where(
                    still_running,
                    step_soc_best_trial_budget_residual,
                    last_soc_best_trial_budget_residual,
                ),
                jnp.where(
                    still_running,
                    step_soc_best_trial_budget_relative_max,
                    last_soc_best_trial_budget_relative_max,
                ),
                jnp.where(
                    still_running,
                    step_soc_best_trial_comp_residual,
                    last_soc_best_trial_comp_residual,
                ),
                jnp.where(
                    still_running,
                    step_soc_best_trial_total_residual,
                    last_soc_best_trial_total_residual,
                ),
                jnp.where(
                    still_running,
                    step_soc_best_trial_combined_improved,
                    last_soc_best_trial_combined_improved,
                ),
                jnp.where(
                    still_running,
                    step_soc_best_trial_budget_relative_not_worse,
                    last_soc_best_trial_budget_relative_not_worse,
                ),
                jnp.where(
                    still_running,
                    step_soc_best_trial_filter_accepted,
                    last_soc_best_trial_filter_accepted,
                ),
                jnp.where(
                    still_running,
                    step_soc_best_trial_accepted,
                    last_soc_best_trial_accepted,
                ),
                jnp.where(
                    still_running,
                    step_soc_best_trial_fallback_accepted,
                    last_soc_best_trial_fallback_accepted,
                ),
                jnp.where(
                    still_running,
                    step_selected_trial_gas_residual,
                    last_selected_trial_gas_residual,
                ),
                jnp.where(
                    still_running,
                    step_selected_trial_cond_residual,
                    last_selected_trial_cond_residual,
                ),
                jnp.where(
                    still_running,
                    step_selected_trial_budget_residual,
                    last_selected_trial_budget_residual,
                ),
                jnp.where(
                    still_running,
                    step_selected_trial_budget_relative_max,
                    last_selected_trial_budget_relative_max,
                ),
                jnp.where(
                    still_running,
                    step_selected_trial_comp_residual,
                    last_selected_trial_comp_residual,
                ),
                jnp.where(
                    still_running,
                    step_selected_trial_total_residual,
                    last_selected_trial_total_residual,
                ),
                tuple(
                    jnp.where(
                        still_running,
                        (
                            last_value + step_value
                            if 28 <= diagnostic_index <= 32
                            else (
                                jnp.maximum(last_value, step_value)
                                if diagnostic_index in (38, 39, 41, 42, 43)
                                else (
                                    jnp.minimum(last_value, step_value)
                                    if diagnostic_index == 40
                                    else (
                                        jnp.maximum(last_value, step_value)
                                        if diagnostic_index == 46
                                        else (
                                            jnp.where(
                                                (
                                                    last_candidate_diagnostics[46]
                                                    > 0
                                                )
                                                if diagnostic_index <= 55
                                                else (
                                                    last_candidate_diagnostics[46]
                                                    > 0
                                                )[:, None],
                                                last_value,
                                                jnp.where(
                                                    (
                                                        step_candidate_diagnostics[46]
                                                        > 0
                                                    )
                                                    if diagnostic_index <= 55
                                                    else (
                                                        step_candidate_diagnostics[46]
                                                        > 0
                                                    )[:, None],
                                                    step_value,
                                                    last_value,
                                                ),
                                            )
                                            if 47 <= diagnostic_index <= 56
                                            else step_value
                                        )
                                    )
                                )
                            )
                        ),
                        last_value,
                    )
                    for diagnostic_index, (step_value, last_value) in enumerate(
                        zip(
                            step_candidate_diagnostics,
                            last_candidate_diagnostics,
                        )
                    )
                ),
                jnp.where(
                    still_running,
                    prepared_filter_phi_entries,
                    filter_phi_entries,
                ),
                jnp.where(
                    still_running,
                    prepared_filter_theta_entries,
                    filter_theta_entries,
                ),
                jnp.where(
                    still_running,
                    prepared_filter_valid_entries,
                    filter_valid_entries,
                ),
                jnp.where(
                    still_running,
                    step_consecutive_filter_rejection_count,
                    consecutive_filter_rejection_count,
                ),
                jnp.where(
                    still_running,
                    step_filter_reset_count,
                    filter_reset_count,
                ),
                next_still_running,
                next_restoration_phase_state,
            )

        initial = (
            jnp.asarray(0, dtype=jnp.int32),
            q0,
            r0,
            lam0,
            rho0,
            qtot0,
            initial_residual,
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False, dtype=bool),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(initial_residual, dtype=q0.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(initial_residual, dtype=q0.dtype),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False, dtype=bool),
            jnp.asarray(False, dtype=bool),
            jnp.asarray(False, dtype=bool),
            jnp.asarray(False, dtype=bool),
            jnp.asarray(False, dtype=bool),
            jnp.asarray(False, dtype=bool),
            jnp.asarray(False, dtype=bool),
            jnp.asarray(False, dtype=bool),
            jnp.asarray(False, dtype=bool),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False, dtype=bool),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(False, dtype=bool),
            jnp.asarray(False, dtype=bool),
            jnp.asarray(False, dtype=bool),
            jnp.asarray(False, dtype=bool),
            jnp.asarray(False, dtype=bool),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0.0, dtype=q0.dtype),
            jnp.asarray(0.0, dtype=q0.dtype),
            initial_step[-6],
            filter_phi_entries0,
            filter_theta_entries0,
            filter_valid_entries0,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            initial_running,
            (
                q0,
                r0,
                rho0,
                qtot0,
                initial_filter_theta,
                jnp.asarray(False, dtype=bool),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0.0, dtype=q0.dtype),
                jnp.asarray(1.0, dtype=q0.dtype),
                jnp.zeros((6,), dtype=q0.dtype),
                jnp.zeros((6,), dtype=q0.dtype),
                jnp.asarray(jnp.inf, dtype=q0.dtype),
                jnp.zeros((6,), dtype=q0.dtype),
                jnp.zeros((6,), dtype=q0.dtype),
                jnp.zeros((6,), dtype=q0.dtype),
                jnp.asarray(False, dtype=bool),
                jnp.asarray(False, dtype=bool),
                jnp.asarray(3, dtype=jnp.int32),
                jnp.asarray(False, dtype=bool),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )
        final = lax.while_loop(cond_fun, body, initial)
        accepted_count = final[9]
        normal_accepted_count = final[10]
        fallback_accepted_count = final[11]
        restoration_accepted_count = final[12]
        soc_accepted_count = final[13]
        adaptive_regularization_selected_count = final[14]
        tiny_step_consecutive_count = final[15]
        rejected_trial_count = final[8]
        n_iter = jnp.minimum(accepted_count + 1, jnp.asarray(max_iter, dtype=jnp.int32))
        qf, rf, lamf, rhof, qtotf = final[1], final[2], final[3], final[4], final[5]
        final_step_size = final[7]
        gas_stationarity_source_final = jnp.where(
            use_external_gas_source_one,
            gas_source_init,
            hgas + ln_pressure - qtotf,
        )
        jac_mask_final = jnp.ones((rf.shape[0],), dtype=bool)
        nf = jnp.exp(qf)
        mf = jnp.exp(rf)
        etaf = jnp.exp(rhof)
        gas_component = (
            qf
            + gas_stationarity_source_final
            - ag.T @ lamf
        )
        cond_component = hcond - ac.T @ lamf - etaf
        budget_component = ag @ nf + ac @ mf - target
        complementarity_component = rf + rhof - epsilon_vec
        total_density_component = jnp.asarray(
            [jnp.sum(nf) - jnp.exp(qtotf)],
            dtype=qf.dtype,
        )
        budget_relative_final = fixed_support_budget_relative_max(
            budget_residual=budget_component,
            target=target,
            relative_floor=jnp.maximum(
                jnp.asarray(jnp.finfo(qf.dtype).tiny, dtype=qf.dtype),
                convergence_budget_relative_floor.astype(qf.dtype),
            ),
        )
        final_residual = l2(
            jnp.concatenate(
                [
                    gas_component,
                    jnp.where(jac_mask_final, cond_component, 0.0),
                    budget_component,
                    complementarity_component,
                    total_density_component,
                ]
            )
        )
        converged = fixed_support_batch_converged(
            gas_norm=l2(gas_component),
            condensate_stationarity_norm=l2(jnp.where(jac_mask_final, cond_component, 0.0)),
            complementarity_norm=l2(complementarity_component),
            total_density_norm=l2(total_density_component),
            budget_relative_max=budget_relative_final,
            log_tolerance=residual_crit,
            budget_relative_tolerance=budget_relative_crit,
            total_density_tolerance=total_density_crit,
        )
        residual_component_norms = jnp.asarray(
            [
                l2(gas_component),
                l2(jnp.where(jac_mask_final, cond_component, 0.0)),
                l2(budget_component),
                l2(complementarity_component),
                l2(total_density_component),
            ],
            dtype=qf.dtype,
        )
        dominant_residual_component_index = jnp.asarray(
            jnp.argmax(residual_component_norms),
            dtype=jnp.int32,
        )
        final_accepted_trial = final[16]
        hit_max_iter = (n_iter >= max_iter) & (~converged)
        tiny_step_stalled = (
            (~converged)
            & (tiny_step_consecutive_count >= tiny_step_limit)
            & (final_step_size <= tiny_step_threshold)
        )
        stop_reason_code = jnp.select(
            (
                ~jnp.isfinite(final_residual),
                converged,
                tiny_step_stalled,
                (~converged) & (~final_accepted_trial),
                hit_max_iter & (final_step_size <= tiny_step_threshold),
                hit_max_iter,
            ),
            (
                jnp.asarray(4, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(6, dtype=jnp.int32),
                jnp.asarray(3, dtype=jnp.int32),
                jnp.asarray(2, dtype=jnp.int32),
                jnp.asarray(1, dtype=jnp.int32),
            ),
            default=jnp.asarray(5, dtype=jnp.int32),
        )
        return (
            qf,
            rf,
            qtotf,
            n_iter,
            converged,
            hit_max_iter,
            final_residual,
            residual_crit,
            accepted_count,
            normal_accepted_count,
            fallback_accepted_count,
            restoration_accepted_count,
            soc_accepted_count,
            adaptive_regularization_selected_count,
            tiny_step_consecutive_count,
            initial_residual,
            lambda_selection_index,
            l2(gas_component),
            l2(jnp.where(jac_mask_final, cond_component, 0.0)),
            l2(budget_component),
            budget_relative_final,
            l2(complementarity_component),
            l2(total_density_component),
            final_step_size,
            rejected_trial_count,
            rhof,
            lamf,
            stop_reason_code,
            dominant_residual_component_index,
            final[17],
            final[18],
            final[19],
            final[20],
            final[21],
            final[22],
            final[23],
            final[24],
            final[25],
            final[26],
            final[27],
            final[28],
            final[29],
            final[30],
            final[31],
            final[32],
            final[33],
            final[34],
            final[35],
            final[36],
            final[37],
            final[38],
            final[39],
            final[40],
            final[41],
            final[42],
            final[43],
            final[44],
            final[45],
            final[46],
            final[47],
            final[48],
            final[49],
            final[50],
            final[51],
            final[52],
            final[53],
            final[54],
            final[55],
            final[56],
            final[57],
            final[58],
            final[59],
            final[60],
            final[61],
            final[62],
            final[63],
            final[64],
            final[65],
            final[66],
            final[67],
            final[68],
            final[69],
            final[70],
            final[71],
            final[72],
            final[73],
            final[74],
            final[75],
            final[76],
            final[-1][4],
            final[-1][5],
            final[-1][6],
            final[-1][7],
            final[-1][8],
            final[-1][9],
            final[-1][10],
            final[-1][11],
            final[-1][12],
            final[-1][13],
            final[-1][14],
            final[-1][15],
            final[-1][16],
            final[-1][17],
            final[-1][18],
            final[-1][19],
            final[-1][20],
            final[-1][21],
            final[-1][22],
            final[-1][23],
            final[-1][24],
            final[-1][25],
        )

    return jax.vmap(run_one)(
        ln_nk_init,
        ln_mk_init,
        ln_ntot_init,
        element_potential_init,
        rho_init,
        gas_stationarity_source_init,
        use_external_gas_stationarity_source,
        use_solver_epsilon,
        element_inventory_target,
        hvector,
        hvector_cond_active,
        ln_normalized_pressure,
        epsilon,
    )


_pdipm_activity_fixed_support_batch_core_jit = jax.jit(
    _pdipm_activity_fixed_support_batch_core,
    static_argnames=(
        "max_iter",
        "tiny_step_consecutive_limit",
        "second_order_correction_budget_passes",
        "second_order_correction_policy",
        "second_order_correction_alpha_y_policy",
        "second_order_correction_charge_solve_policy",
        "second_order_correction_reduced_mode_policy",
        "second_order_correction_diagnostic_mode_vector_policy",
        "budget_restoration_enabled",
        "budget_restoration_coordinate_policy",
        "budget_restoration_dual_recenter_policy",
        "budget_restoration_passes",
        "budget_restoration_phase_enabled",
        "budget_restoration_phase_cooldown_iterations",
        "line_search_candidate_selection_policy",
        "ipopt_filter_policy",
        "rho_initialization",
        "lambda_initialization",
    ),
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
    lambda_initialization: str = "best_residual",
) -> tuple[CondensateEquilibriumResult, dict[str, Any]]:
    """Run the experimental fixed-support activity-correction core for one bucket.

    The helper is intentionally private and currently does not alter the
    production route. It provides the GPU-friendly fixed-shape batch primitive
    used by the optimization experiments.
    """

    lambda_initialization = str(lambda_initialization)
    if lambda_initialization not in FIXED_SUPPORT_BATCH_LAMBDA_INITIALIZATIONS:
        raise ValueError(
            "lambda_initialization must be one of "
            f"{FIXED_SUPPORT_BATCH_LAMBDA_INITIALIZATIONS}."
        )

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
    effective_epsilon = float(
        os.environ.get(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_EPSILON",
            str(float(epsilon)),
        )
    )
    epsilon_array = (
        jnp.asarray(barrier_epsilon_init, dtype=jnp.float64)
        if barrier_epsilon_init is not None
        else jnp.full_like(
            ln_ntot_init_array,
            effective_epsilon,
            dtype=jnp.float64,
        )
    )
    use_solver_epsilon_array = jnp.full_like(
        ln_ntot_init_array,
        barrier_epsilon_init is not None,
        dtype=bool,
    )
    use_external_gas_stationarity_source_array = jnp.full_like(
        ln_ntot_init_array,
        gas_stationarity_source_init is not None,
        dtype=bool,
    )
    gas_stationarity_source_init_array = (
        jnp.asarray(gas_stationarity_source_init, dtype=jnp.float64)
        if gas_stationarity_source_init is not None
        else jnp.asarray(hvector, dtype=jnp.float64)
    )
    budget_relative_acceptance_floor = float(
        os.environ.get(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RELATIVE_LIMIT",
            "1.0e-3",
        )
    )
    budget_direction_projection_strength = float(
        os.environ.get(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_DIRECTION_PROJECTION",
            "0.0",
        )
    )
    convergence_log_tolerance = float(
        os.environ.get(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_CONVERGENCE_LOG_TOLERANCE",
            "1.0e-5",
        )
    )
    convergence_budget_relative_tolerance = float(
        os.environ.get(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_CONVERGENCE_BUDGET_RELATIVE_TOLERANCE",
            "1.0e-4",
        )
    )
    convergence_budget_relative_floor = float(
        os.environ.get(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_CONVERGENCE_BUDGET_RELATIVE_FLOOR",
            "1.0e-6",
        )
    )
    convergence_total_density_tolerance = float(
        os.environ.get(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_CONVERGENCE_TOTAL_DENSITY_TOLERANCE",
            "1.0e-5",
        )
    )
    tiny_step_consecutive_limit = int(
        os.environ.get(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_TINY_STEP_CONSECUTIVE_LIMIT",
            "50",
        )
    )
    if tiny_step_consecutive_limit < 1:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_TINY_STEP_CONSECUTIVE_LIMIT must be >= 1."
        )
    relaxed_stationarity_fallback_enabled = bool(
        int(
            os.environ.get(
                "EXOGIBBS_FIXED_SUPPORT_BATCH_RELAXED_STATIONARITY_FALLBACK",
                "0",
            )
        )
    )
    relaxed_stationarity_fallback_factor = float(
        os.environ.get(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_RELAXED_STATIONARITY_FACTOR",
            "0.999",
        )
    )
    adaptive_regularization_enabled = bool(
        int(
            os.environ.get(
                "EXOGIBBS_FIXED_SUPPORT_BATCH_ADAPTIVE_REGULARIZATION",
                "0",
            )
        )
    )
    adaptive_regularization_base = float(
        os.environ.get(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_REGULARIZATION_BASE",
            "1.0e-10",
        )
    )
    second_order_correction_enabled = bool(
        int(
            os.environ.get(
                "EXOGIBBS_FIXED_SUPPORT_BATCH_SECOND_ORDER_CORRECTION",
                "0",
            )
        )
    )
    second_order_correction_max_abs_step = float(
        os.environ.get(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_MAX_ABS_STEP",
            "1.0",
        )
    )
    second_order_correction_budget_passes = int(
        os.environ.get(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_MAX_CORRECTIONS",
            os.environ.get(
                "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_BUDGET_PASSES",
                "1",
            ),
        )
    )
    if second_order_correction_budget_passes < 1:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_MAX_CORRECTIONS (or legacy "
            "SOC_BUDGET_PASSES) must be >= 1."
        )
    second_order_correction_dual_repair = bool(
        int(
            os.environ.get(
                "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_DUAL_REPAIR",
                "0",
            )
        )
    )
    second_order_correction_policy = os.environ.get(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_POLICY",
        "reduced_kkt",
    )
    if second_order_correction_policy not in {
        "legacy_budget_projection",
        "legacy_budget_projection_triggered",
        "ipopt_first_trial",
        "reduced_kkt",
    }:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_POLICY must be "
            "'legacy_budget_projection', 'legacy_budget_projection_triggered', "
            "'ipopt_first_trial', or 'reduced_kkt'."
        )
    second_order_correction_kappa_soc = float(
        os.environ.get("EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_KAPPA", "0.99")
    )
    if not 0.0 < second_order_correction_kappa_soc < 1.0:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_KAPPA must be between 0 and 1."
        )
    second_order_correction_alpha_y_policy = os.environ.get(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_ALPHA_Y_POLICY", "full"
    )
    if second_order_correction_alpha_y_policy not in {
        "full",
        "min_dual_infeas",
        "primal",
    }:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_ALPHA_Y_POLICY must be 'full', "
            "'min_dual_infeas', or 'primal'."
        )
    second_order_correction_charge_solve_policy = os.environ.get(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_CHARGE_SOLVE_POLICY",
        "coupled",
    )
    if second_order_correction_charge_solve_policy not in {
        "coupled",
        "charge_schur",
        "neutrality_retraction",
    }:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_CHARGE_SOLVE_POLICY must be "
            "'coupled', 'charge_schur', or 'neutrality_retraction'."
        )
    second_order_correction_reduced_mode_policy = os.environ.get(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_REDUCED_MODE_POLICY",
        "full",
    )
    if second_order_correction_reduced_mode_policy not in {
        "full",
        "remove_smallest_mode",
    }:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_REDUCED_MODE_POLICY must be "
            "'full' or 'remove_smallest_mode'."
        )
    second_order_correction_diagnostic_mode_vector_policy = os.environ.get(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_DIAGNOSTIC_MODE_VECTOR_POLICY",
        "smallest_right_singular",
    )
    if second_order_correction_diagnostic_mode_vector_policy not in {
        "smallest_right_singular",
        "dominant_solution_component",
    }:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_DIAGNOSTIC_MODE_VECTOR_POLICY "
            "must be 'smallest_right_singular' or "
            "'dominant_solution_component'."
        )
    budget_restoration_enabled = bool(
        int(
            os.environ.get(
                "EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION",
                "0",
            )
        )
    )
    budget_restoration_coordinate_policy = os.environ.get(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION_COORDINATES",
        "log",
    )
    if budget_restoration_coordinate_policy not in {"log", "amount"}:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION_COORDINATES "
            "must be 'log' or 'amount'."
        )
    budget_restoration_dual_recenter = bool(
        int(
            os.environ.get(
                "EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION_DUAL_RECENTER",
                "0",
            )
        )
    )
    budget_restoration_dual_recenter_policy = os.environ.get(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION_DUAL_RECENTER_POLICY",
        "hard",
    )
    if budget_restoration_dual_recenter_policy not in {
        "hard",
        "ipopt_linearized",
    }:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION_DUAL_RECENTER_POLICY "
            "must be 'hard' or 'ipopt_linearized'."
        )
    budget_restoration_proximity_weight = float(
        os.environ.get(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION_PROXIMITY_WEIGHT",
            "1.0e-4",
        )
    )
    if budget_restoration_proximity_weight < 0.0:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION_PROXIMITY_WEIGHT "
            "must be non-negative."
        )
    budget_restoration_max_abs_step = float(
        os.environ.get(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION_MAX_ABS_STEP",
            "2.0",
        )
    )
    if budget_restoration_max_abs_step < 0.0:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION_MAX_ABS_STEP "
            "must be non-negative."
        )
    budget_restoration_passes = int(
        os.environ.get(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION_PASSES",
            "1",
        )
    )
    if budget_restoration_passes < 1:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION_PASSES must be >= 1."
        )
    budget_restoration_phase_enabled = bool(
        int(
            os.environ.get(
                "EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION_PHASE",
                "0",
            )
        )
    )
    budget_restoration_phase_theta_reduction = float(
        os.environ.get(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION_PHASE_THETA_REDUCTION",
            "0.9",
        )
    )
    if not 0.0 < budget_restoration_phase_theta_reduction < 1.0:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION_PHASE_THETA_REDUCTION "
            "must be in the interval (0, 1)."
        )
    budget_restoration_phase_cooldown_iterations = int(
        os.environ.get(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION_PHASE_COOLDOWN",
            "1",
        )
    )
    if budget_restoration_phase_cooldown_iterations < 0:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION_PHASE_COOLDOWN "
            "must be non-negative."
        )
    if budget_restoration_phase_enabled and not (
        budget_restoration_enabled
        and budget_restoration_coordinate_policy == "amount"
        and budget_restoration_dual_recenter
        and budget_restoration_dual_recenter_policy == "ipopt_linearized"
    ):
        raise ValueError(
            "The amount-restoration phase requires budget restoration, amount "
            "coordinates, dual recentering, and the ipopt_linearized return policy."
        )
    ipopt_filter_acceptance_enabled = bool(
        int(
            os.environ.get(
                "EXOGIBBS_FIXED_SUPPORT_BATCH_IPOPT_FILTER_ACCEPTANCE",
                "0",
            )
        )
    )
    ipopt_filter_policy = os.environ.get(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_IPOPT_FILTER_POLICY",
        "persistent_phi_theta",
    )
    if ipopt_filter_policy not in {
        "current_iterate",
        "persistent_phi_theta",
    }:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_IPOPT_FILTER_POLICY must be "
            "'current_iterate' or 'persistent_phi_theta'."
        )
    if budget_restoration_phase_enabled and not (
        ipopt_filter_acceptance_enabled
        and ipopt_filter_policy == "persistent_phi_theta"
    ):
        raise ValueError(
            "The amount-restoration phase requires the persistent Ipopt-style "
            "filter."
        )
    ipopt_filter_theta_norm = os.environ.get(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_IPOPT_FILTER_THETA_NORM",
        "max_scaled",
    )
    if ipopt_filter_theta_norm not in {"max_scaled", "l1_scaled"}:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_IPOPT_FILTER_THETA_NORM must be "
            "'max_scaled' or 'l1_scaled'."
        )
    ipopt_filter_budget_relative_max = float(
        os.environ.get(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_IPOPT_FILTER_BUDGET_RELATIVE_MAX",
            "0.25",
        )
    )
    if ipopt_filter_budget_relative_max < 0.0:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_IPOPT_FILTER_BUDGET_RELATIVE_MAX "
            "must be non-negative."
        )
    line_search_candidate_selection_policy = os.environ.get(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_SELECTION_POLICY",
        "ipopt_vectorized_max_alpha",
    )
    if line_search_candidate_selection_policy not in {
        "legacy",
        "ipopt_sequential",
        "ipopt_vectorized_max_alpha",
    }:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_SELECTION_POLICY must be 'legacy', "
            "'ipopt_sequential', or 'ipopt_vectorized_max_alpha'."
        )
    second_order_correction_trial_order = os.environ.get(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_TRIAL_ORDER",
        "append",
    )
    if second_order_correction_trial_order not in {"append", "interleave"}:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_TRIAL_ORDER must be "
            "'append' or 'interleave'."
        )
    second_order_correction_interleave = (
        second_order_correction_trial_order == "interleave"
    )
    use_legacy_capacity_epsilon = bool(
        int(
            os.environ.get(
                "EXOGIBBS_FIXED_SUPPORT_BATCH_USE_LEGACY_CAPACITY_EPSILON",
                "0",
            )
        )
    )
    use_log_amount_boundary = bool(
        int(
            os.environ.get(
                "EXOGIBBS_FIXED_SUPPORT_BATCH_LOG_AMOUNT_BOUNDARY",
                "1",
            )
        )
    )
    use_log_activity_boundary = bool(
        int(
            os.environ.get(
                "EXOGIBBS_FIXED_SUPPORT_BATCH_LOG_ACTIVITY_BOUNDARY",
                "1",
            )
        )
    )
    step_control_policy = os.environ.get(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_STEP_CONTROL",
        "scalar_fraction_to_boundary",
    )
    if step_control_policy not in {
        "scalar_fraction_to_boundary",
        "component_clip",
    }:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_STEP_CONTROL must be "
            "'scalar_fraction_to_boundary' or 'component_clip'."
        )
    use_scalar_step_control = step_control_policy == "scalar_fraction_to_boundary"

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
        normal_accepted_count,
        fallback_accepted_count,
        restoration_accepted_count,
        soc_accepted_count,
        adaptive_regularization_selected_count,
        tiny_step_consecutive_count,
        initial_residual,
        lambda_selection_index,
        gas_residual_norm,
        condensate_stationarity_residual_norm,
        budget_residual_norm,
        budget_relative_residual_max,
        complementarity_residual_norm,
        total_density_residual_norm,
        final_step_size,
        rejected_trial_count,
        final_log_activity_correction,
        final_element_potential,
        stop_reason_code,
        dominant_residual_component_index,
        line_search_alpha_boundary,
        line_search_alpha_r,
        line_search_alpha_rho,
        line_search_selected_trial_index,
        line_search_selected_trial_alpha,
        line_search_selected_trial_residual,
        line_search_accepted_candidate_count,
        line_search_fallback_candidate_count,
        line_search_best_trial_index,
        line_search_best_trial_alpha,
        line_search_best_trial_residual,
        line_search_best_trial_gas_residual,
        line_search_best_trial_condensate_stationarity_residual,
        line_search_best_trial_budget_residual,
        line_search_best_trial_budget_relative_residual_max,
        line_search_best_trial_complementarity_residual,
        line_search_best_trial_total_density_residual,
        line_search_finite_candidate_count,
        line_search_combined_improved_candidate_count,
        line_search_budget_relative_not_worse_candidate_count,
        line_search_filter_candidate_count,
        line_search_budget_not_broken_candidate_count,
        line_search_budget_relative_not_broken_candidate_count,
        line_search_combined_not_worse_candidate_count,
        line_search_best_trial_finite,
        line_search_best_trial_combined_improved,
        line_search_best_trial_budget_relative_not_worse,
        line_search_best_trial_filter_accepted,
        line_search_best_trial_budget_not_broken,
        line_search_best_trial_budget_relative_not_broken,
        line_search_best_trial_combined_not_worse,
        line_search_best_trial_accepted,
        line_search_best_trial_fallback_accepted,
        line_search_soc_candidate_count,
        line_search_soc_accepted_candidate_count,
        line_search_soc_fallback_candidate_count,
        line_search_soc_budget_relative_not_worse_candidate_count,
        line_search_soc_filter_candidate_count,
        line_search_soc_best_trial_present,
        line_search_soc_best_trial_index,
        line_search_soc_best_trial_alpha,
        line_search_soc_best_trial_residual,
        line_search_soc_best_trial_gas_residual,
        line_search_soc_best_trial_condensate_stationarity_residual,
        line_search_soc_best_trial_budget_residual,
        line_search_soc_best_trial_budget_relative_residual_max,
        line_search_soc_best_trial_complementarity_residual,
        line_search_soc_best_trial_total_density_residual,
        line_search_soc_best_trial_combined_improved,
        line_search_soc_best_trial_budget_relative_not_worse,
        line_search_soc_best_trial_filter_accepted,
        line_search_soc_best_trial_accepted,
        line_search_soc_best_trial_fallback_accepted,
        line_search_selected_trial_gas_residual,
        line_search_selected_trial_condensate_stationarity_residual,
        line_search_selected_trial_budget_residual,
        line_search_selected_trial_budget_relative_residual_max,
        line_search_selected_trial_complementarity_residual,
        line_search_selected_trial_total_density_residual,
        line_search_candidate_diagnostics,
        restoration_phase_entry_theta_at_stop,
        restoration_phase_active_at_stop,
        restoration_phase_cooldown_at_stop,
        amount_restoration_accepted_count,
        restoration_phase_entry_count,
        restoration_phase_exit_count,
        restoration_bound_multiplier_reset_count,
        restoration_equality_multiplier_reset_count,
        restoration_last_exit_theta,
        restoration_last_dual_alpha,
        restoration_entry_residual_vector,
        restoration_best_residual_vector,
        restoration_best_theta,
        restoration_last_exit_predual_residual_vector,
        restoration_last_exit_postdual_residual_vector,
        restoration_first_normal_residual_vector,
        restoration_first_normal_attempted,
        restoration_first_normal_accepted,
        restoration_first_normal_selected_type,
        restoration_return_probe_pending,
        restoration_active_accepted_count,
        restoration_last_active_accepted_count,
    ) = _pdipm_activity_fixed_support_batch_core_jit(
        ln_nk_init=ln_nk_init_array,
        ln_mk_init=ln_mk_init_array,
        ln_ntot_init=ln_ntot_init_array,
        element_potential_init=element_potential_init_array,
        rho_init=rho_init_array,
        gas_stationarity_source_init=gas_stationarity_source_init_array,
        use_external_gas_stationarity_source=(
            use_external_gas_stationarity_source_array
        ),
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
        budget_relative_acceptance_floor=jnp.asarray(
            budget_relative_acceptance_floor,
            dtype=jnp.float64,
        ),
        budget_direction_projection_strength=jnp.asarray(
            budget_direction_projection_strength,
            dtype=jnp.float64,
        ),
        convergence_log_tolerance=jnp.asarray(
            convergence_log_tolerance,
            dtype=jnp.float64,
        ),
        convergence_budget_relative_tolerance=jnp.asarray(
            convergence_budget_relative_tolerance,
            dtype=jnp.float64,
        ),
        convergence_budget_relative_floor=jnp.asarray(
            convergence_budget_relative_floor,
            dtype=jnp.float64,
        ),
        convergence_total_density_tolerance=jnp.asarray(
            convergence_total_density_tolerance,
            dtype=jnp.float64,
        ),
        relaxed_stationarity_fallback_enabled=jnp.asarray(
            relaxed_stationarity_fallback_enabled,
            dtype=bool,
        ),
        relaxed_stationarity_fallback_factor=jnp.asarray(
            relaxed_stationarity_fallback_factor,
            dtype=jnp.float64,
        ),
        adaptive_regularization_enabled=jnp.asarray(
            adaptive_regularization_enabled,
            dtype=bool,
        ),
        adaptive_regularization_base=jnp.asarray(
            adaptive_regularization_base,
            dtype=jnp.float64,
        ),
        second_order_correction_enabled=jnp.asarray(
            second_order_correction_enabled,
            dtype=bool,
        ),
        second_order_correction_max_abs_step=jnp.asarray(
            second_order_correction_max_abs_step,
            dtype=jnp.float64,
        ),
        second_order_correction_interleave=jnp.asarray(
            second_order_correction_interleave,
            dtype=bool,
        ),
        second_order_correction_budget_passes=int(
            second_order_correction_budget_passes
        ),
        second_order_correction_dual_repair=jnp.asarray(
            second_order_correction_dual_repair,
            dtype=bool,
        ),
        second_order_correction_policy=second_order_correction_policy,
        second_order_correction_kappa_soc=jnp.asarray(
            second_order_correction_kappa_soc, dtype=jnp.float64
        ),
        second_order_correction_alpha_y_policy=(
            second_order_correction_alpha_y_policy
        ),
        second_order_correction_charge_solve_policy=(
            second_order_correction_charge_solve_policy
        ),
        second_order_correction_reduced_mode_policy=(
            second_order_correction_reduced_mode_policy
        ),
        second_order_correction_diagnostic_mode_vector_policy=(
            second_order_correction_diagnostic_mode_vector_policy
        ),
        budget_restoration_enabled=budget_restoration_enabled,
        budget_restoration_coordinate_policy=budget_restoration_coordinate_policy,
        budget_restoration_dual_recenter=jnp.asarray(
            budget_restoration_dual_recenter,
            dtype=bool,
        ),
        budget_restoration_dual_recenter_policy=(
            budget_restoration_dual_recenter_policy
        ),
        budget_restoration_proximity_weight=jnp.asarray(
            budget_restoration_proximity_weight,
            dtype=jnp.float64,
        ),
        budget_restoration_max_abs_step=jnp.asarray(
            budget_restoration_max_abs_step,
            dtype=jnp.float64,
        ),
        budget_restoration_passes=int(budget_restoration_passes),
        budget_restoration_phase_enabled=budget_restoration_phase_enabled,
        budget_restoration_phase_theta_reduction=jnp.asarray(
            budget_restoration_phase_theta_reduction,
            dtype=jnp.float64,
        ),
        budget_restoration_phase_cooldown_iterations=int(
            budget_restoration_phase_cooldown_iterations
        ),
        ipopt_filter_acceptance_enabled=jnp.asarray(
            ipopt_filter_acceptance_enabled,
            dtype=bool,
        ),
        ipopt_filter_budget_relative_max=jnp.asarray(
            ipopt_filter_budget_relative_max,
            dtype=jnp.float64,
        ),
        ipopt_filter_policy=ipopt_filter_policy,
        ipopt_filter_use_l1_theta=jnp.asarray(
            ipopt_filter_theta_norm == "l1_scaled", dtype=bool
        ),
        line_search_candidate_selection_policy=(
            line_search_candidate_selection_policy
        ),
        use_legacy_capacity_epsilon=jnp.asarray(
            use_legacy_capacity_epsilon,
            dtype=bool,
        ),
        use_scalar_step_control=jnp.asarray(
            use_scalar_step_control,
            dtype=bool,
        ),
        use_log_amount_boundary=jnp.asarray(
            use_log_amount_boundary,
            dtype=bool,
        ),
        use_log_activity_boundary=jnp.asarray(
            use_log_activity_boundary,
            dtype=bool,
        ),
        max_iter=int(max_iter),
        tiny_step_consecutive_limit=int(tiny_step_consecutive_limit),
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
            "final_step_size": final_step_size,
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
        build_fixed_support_batch_metadata(
            accepted_count=accepted_count,
            normal_accepted_count=normal_accepted_count,
            fallback_accepted_count=fallback_accepted_count,
            restoration_accepted_count=restoration_accepted_count,
            soc_accepted_count=soc_accepted_count,
            adaptive_regularization_selected_count=(
                adaptive_regularization_selected_count
            ),
            rejected_trial_count=rejected_trial_count,
            tiny_step_consecutive_count=tiny_step_consecutive_count,
            final_step_size=final_step_size,
            stop_reason_code=stop_reason_code,
            dominant_residual_component_index=dominant_residual_component_index,
            final_log_activity_correction=final_log_activity_correction,
            final_element_potential=final_element_potential,
            initial_residual=initial_residual,
            lambda_selection_index=lambda_selection_index,
            line_search_alpha_boundary=line_search_alpha_boundary,
            line_search_alpha_r=line_search_alpha_r,
            line_search_alpha_rho=line_search_alpha_rho,
            line_search_selected_trial_index=line_search_selected_trial_index,
            line_search_selected_trial_alpha=line_search_selected_trial_alpha,
            line_search_selected_trial_residual=line_search_selected_trial_residual,
            line_search_accepted_candidate_count=(
                line_search_accepted_candidate_count
            ),
            line_search_fallback_candidate_count=(
                line_search_fallback_candidate_count
            ),
            line_search_best_trial_index=line_search_best_trial_index,
            line_search_best_trial_alpha=line_search_best_trial_alpha,
            line_search_best_trial_residual=line_search_best_trial_residual,
            line_search_best_trial_gas_residual=(
                line_search_best_trial_gas_residual
            ),
            line_search_best_trial_condensate_stationarity_residual=(
                line_search_best_trial_condensate_stationarity_residual
            ),
            line_search_best_trial_budget_residual=(
                line_search_best_trial_budget_residual
            ),
            line_search_best_trial_budget_relative_residual_max=(
                line_search_best_trial_budget_relative_residual_max
            ),
            line_search_best_trial_complementarity_residual=(
                line_search_best_trial_complementarity_residual
            ),
            line_search_best_trial_total_density_residual=(
                line_search_best_trial_total_density_residual
            ),
            line_search_finite_candidate_count=line_search_finite_candidate_count,
            line_search_combined_improved_candidate_count=(
                line_search_combined_improved_candidate_count
            ),
            line_search_budget_relative_not_worse_candidate_count=(
                line_search_budget_relative_not_worse_candidate_count
            ),
            line_search_filter_candidate_count=line_search_filter_candidate_count,
            line_search_budget_not_broken_candidate_count=(
                line_search_budget_not_broken_candidate_count
            ),
            line_search_budget_relative_not_broken_candidate_count=(
                line_search_budget_relative_not_broken_candidate_count
            ),
            line_search_combined_not_worse_candidate_count=(
                line_search_combined_not_worse_candidate_count
            ),
            line_search_best_trial_finite=line_search_best_trial_finite,
            line_search_best_trial_combined_improved=(
                line_search_best_trial_combined_improved
            ),
            line_search_best_trial_budget_relative_not_worse=(
                line_search_best_trial_budget_relative_not_worse
            ),
            line_search_best_trial_filter_accepted=(
                line_search_best_trial_filter_accepted
            ),
            line_search_best_trial_budget_not_broken=(
                line_search_best_trial_budget_not_broken
            ),
            line_search_best_trial_budget_relative_not_broken=(
                line_search_best_trial_budget_relative_not_broken
            ),
            line_search_best_trial_combined_not_worse=(
                line_search_best_trial_combined_not_worse
            ),
            line_search_best_trial_accepted=line_search_best_trial_accepted,
            line_search_best_trial_fallback_accepted=(
                line_search_best_trial_fallback_accepted
            ),
            line_search_soc_candidate_count=line_search_soc_candidate_count,
            line_search_soc_accepted_candidate_count=(
                line_search_soc_accepted_candidate_count
            ),
            line_search_soc_fallback_candidate_count=(
                line_search_soc_fallback_candidate_count
            ),
            line_search_soc_budget_relative_not_worse_candidate_count=(
                line_search_soc_budget_relative_not_worse_candidate_count
            ),
            line_search_soc_filter_candidate_count=(
                line_search_soc_filter_candidate_count
            ),
            line_search_soc_best_trial_present=line_search_soc_best_trial_present,
            line_search_soc_best_trial_index=line_search_soc_best_trial_index,
            line_search_soc_best_trial_alpha=line_search_soc_best_trial_alpha,
            line_search_soc_best_trial_residual=line_search_soc_best_trial_residual,
            line_search_soc_best_trial_gas_residual=(
                line_search_soc_best_trial_gas_residual
            ),
            line_search_soc_best_trial_condensate_stationarity_residual=(
                line_search_soc_best_trial_condensate_stationarity_residual
            ),
            line_search_soc_best_trial_budget_residual=(
                line_search_soc_best_trial_budget_residual
            ),
            line_search_soc_best_trial_budget_relative_residual_max=(
                line_search_soc_best_trial_budget_relative_residual_max
            ),
            line_search_soc_best_trial_complementarity_residual=(
                line_search_soc_best_trial_complementarity_residual
            ),
            line_search_soc_best_trial_total_density_residual=(
                line_search_soc_best_trial_total_density_residual
            ),
            line_search_soc_best_trial_combined_improved=(
                line_search_soc_best_trial_combined_improved
            ),
            line_search_soc_best_trial_budget_relative_not_worse=(
                line_search_soc_best_trial_budget_relative_not_worse
            ),
            line_search_soc_best_trial_filter_accepted=(
                line_search_soc_best_trial_filter_accepted
            ),
            line_search_soc_best_trial_accepted=(
                line_search_soc_best_trial_accepted
            ),
            line_search_soc_best_trial_fallback_accepted=(
                line_search_soc_best_trial_fallback_accepted
            ),
            line_search_selected_trial_gas_residual=(
                line_search_selected_trial_gas_residual
            ),
            line_search_selected_trial_condensate_stationarity_residual=(
                line_search_selected_trial_condensate_stationarity_residual
            ),
            line_search_selected_trial_budget_residual=(
                line_search_selected_trial_budget_residual
            ),
            line_search_selected_trial_budget_relative_residual_max=(
                line_search_selected_trial_budget_relative_residual_max
            ),
            line_search_selected_trial_complementarity_residual=(
                line_search_selected_trial_complementarity_residual
            ),
            line_search_selected_trial_total_density_residual=(
                line_search_selected_trial_total_density_residual
            ),
            line_search_candidate_diagnostics=line_search_candidate_diagnostics,
            gas_residual_norm=gas_residual_norm,
            condensate_stationarity_residual_norm=(
                condensate_stationarity_residual_norm
            ),
            budget_residual_norm=budget_residual_norm,
            budget_relative_residual_max=budget_relative_residual_max,
            complementarity_residual_norm=complementarity_residual_norm,
            total_density_residual_norm=total_density_residual_norm,
            rho_initialization=str(rho_initialization),
            lambda_initialization=str(lambda_initialization),
            effective_epsilon=effective_epsilon,
            budget_relative_acceptance_floor=budget_relative_acceptance_floor,
            budget_direction_projection_strength=(
                budget_direction_projection_strength
            ),
            convergence_log_tolerance=convergence_log_tolerance,
            convergence_budget_relative_tolerance=(
                convergence_budget_relative_tolerance
            ),
            convergence_budget_relative_floor=convergence_budget_relative_floor,
            convergence_total_density_tolerance=(
                convergence_total_density_tolerance
            ),
            tiny_step_consecutive_limit=tiny_step_consecutive_limit,
            relaxed_stationarity_fallback_enabled=(
                relaxed_stationarity_fallback_enabled
            ),
            relaxed_stationarity_fallback_factor=relaxed_stationarity_fallback_factor,
            adaptive_regularization_enabled=adaptive_regularization_enabled,
            adaptive_regularization_base=adaptive_regularization_base,
            second_order_correction_enabled=second_order_correction_enabled,
            second_order_correction_max_abs_step=(
                second_order_correction_max_abs_step
            ),
            second_order_correction_trial_order=(
                second_order_correction_trial_order
            ),
            second_order_correction_budget_passes=(
                second_order_correction_budget_passes
            ),
            second_order_correction_dual_repair=(
                second_order_correction_dual_repair
            ),
            second_order_correction_policy=second_order_correction_policy,
            second_order_correction_kappa_soc=second_order_correction_kappa_soc,
            second_order_correction_alpha_y_policy=(
                second_order_correction_alpha_y_policy
            ),
            second_order_correction_charge_solve_policy=(
                second_order_correction_charge_solve_policy
            ),
            second_order_correction_reduced_mode_policy=(
                second_order_correction_reduced_mode_policy
            ),
            second_order_correction_diagnostic_mode_vector_policy=(
                second_order_correction_diagnostic_mode_vector_policy
            ),
            budget_restoration_enabled=budget_restoration_enabled,
            budget_restoration_coordinate_policy=(
                budget_restoration_coordinate_policy
            ),
            budget_restoration_dual_recenter=budget_restoration_dual_recenter,
            budget_restoration_dual_recenter_policy=(
                budget_restoration_dual_recenter_policy
            ),
            budget_restoration_proximity_weight=(
                budget_restoration_proximity_weight
            ),
            budget_restoration_max_abs_step=budget_restoration_max_abs_step,
            budget_restoration_passes=budget_restoration_passes,
            budget_restoration_phase_enabled=budget_restoration_phase_enabled,
            budget_restoration_phase_theta_reduction=(
                budget_restoration_phase_theta_reduction
            ),
            budget_restoration_phase_cooldown_iterations=(
                budget_restoration_phase_cooldown_iterations
            ),
            restoration_phase_entry_theta_at_stop=(
                restoration_phase_entry_theta_at_stop
            ),
            restoration_phase_active_at_stop=restoration_phase_active_at_stop,
            restoration_phase_cooldown_at_stop=restoration_phase_cooldown_at_stop,
            amount_restoration_accepted_count=amount_restoration_accepted_count,
            restoration_phase_entry_count=restoration_phase_entry_count,
            restoration_phase_exit_count=restoration_phase_exit_count,
            restoration_bound_multiplier_reset_count=(
                restoration_bound_multiplier_reset_count
            ),
            restoration_equality_multiplier_reset_count=(
                restoration_equality_multiplier_reset_count
            ),
            restoration_last_exit_theta=restoration_last_exit_theta,
            restoration_last_dual_alpha=restoration_last_dual_alpha,
            restoration_entry_residual_vector=(
                restoration_entry_residual_vector
            ),
            restoration_best_residual_vector=(
                restoration_best_residual_vector
            ),
            restoration_best_theta=restoration_best_theta,
            restoration_last_exit_predual_residual_vector=(
                restoration_last_exit_predual_residual_vector
            ),
            restoration_last_exit_postdual_residual_vector=(
                restoration_last_exit_postdual_residual_vector
            ),
            restoration_first_normal_residual_vector=(
                restoration_first_normal_residual_vector
            ),
            restoration_first_normal_attempted=restoration_first_normal_attempted,
            restoration_first_normal_accepted=restoration_first_normal_accepted,
            restoration_first_normal_selected_type=(
                restoration_first_normal_selected_type
            ),
            restoration_return_probe_pending=restoration_return_probe_pending,
            restoration_active_accepted_count=restoration_active_accepted_count,
            restoration_last_active_accepted_count=(
                restoration_last_active_accepted_count
            ),
            ipopt_filter_acceptance_enabled=ipopt_filter_acceptance_enabled,
            ipopt_filter_policy=ipopt_filter_policy,
            ipopt_filter_theta_norm=ipopt_filter_theta_norm,
            ipopt_filter_budget_relative_max=ipopt_filter_budget_relative_max,
            line_search_candidate_selection_policy=(
                line_search_candidate_selection_policy
            ),
            use_legacy_capacity_epsilon=use_legacy_capacity_epsilon,
            use_log_amount_boundary=use_log_amount_boundary,
            use_log_activity_boundary=use_log_activity_boundary,
            step_control_policy=step_control_policy,
        ),
    )


def _parse_fixed_support_batch_epsilon_schedule(final_epsilon: float) -> tuple[float, ...]:
    """Return the explicit path-following schedule for fixed-support batch PD-IPM."""

    raw_schedule = os.environ.get("EXOGIBBS_FIXED_SUPPORT_BATCH_EPSILON_SCHEDULE")
    if raw_schedule is not None:
        schedule = tuple(
            float(value.strip())
            for value in raw_schedule.replace(":", ",").split(",")
            if value.strip()
        )
        if not schedule:
            raise ValueError(
                "EXOGIBBS_FIXED_SUPPORT_BATCH_EPSILON_SCHEDULE must contain at "
                "least one numeric stage."
            )
    else:
        schedule = FIXED_SUPPORT_BATCH_DEFAULT_EPSILON_SCHEDULE
    target = float(final_epsilon)
    if schedule[-1] != target:
        schedule = tuple(value for value in schedule if value > target) + (target,)
    for previous, current in zip(schedule, schedule[1:]):
        if current > previous:
            raise ValueError(
                "fixed-support batch epsilon schedule must be non-increasing."
            )
    return tuple(float(value) for value in schedule)


def _solve_pdipm_rgie_v11_activity_correction_fixed_support_batch_continuation(
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
    epsilon_schedule: Sequence[float],
    residual_tolerance_multiplier: float = 1.0,
    max_iter: int,
    rho_initialization: str = "unit_activity",
    lambda_initialization: str = "best_residual",
) -> tuple[CondensateEquilibriumResult, dict[str, Any]]:
    """Run fixed-support batch solves along an explicit log-barrier schedule."""

    schedule = tuple(float(value) for value in epsilon_schedule)
    if not schedule:
        raise ValueError("epsilon_schedule must contain at least one value.")
    continuation_tolerance_multiplier = float(
        os.environ.get(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_CONTINUATION_TOLERANCE_MULTIPLIER",
            "3.0",
        )
    )
    effective_continuation_tolerance_multiplier = max(
        float(residual_tolerance_multiplier),
        continuation_tolerance_multiplier,
    )
    continuation_recenter_policy = os.environ.get(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_CONTINUATION_RECENTER",
        "adaptive_amount",
    )
    if continuation_recenter_policy not in {
        "adaptive_amount",
        "rho",
        "amount",
        "none",
    }:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_CONTINUATION_RECENTER must be one of "
            "'adaptive_amount', 'rho', 'amount', or 'none'."
        )
    continuation_recenter_amount_relative_threshold = float(
        os.environ.get(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_CONTINUATION_RECENTER_AMOUNT_RELATIVE_THRESHOLD",
            "1.0e-8",
        )
    )
    if continuation_recenter_amount_relative_threshold <= 0.0:
        raise ValueError(
            "EXOGIBBS_FIXED_SUPPORT_BATCH_CONTINUATION_RECENTER_AMOUNT_RELATIVE_THRESHOLD "
            "must be positive."
        )
    current_ln_nk = jnp.asarray(ln_nk_init, dtype=jnp.float64)
    current_ln_mk = jnp.asarray(ln_mk_init, dtype=jnp.float64)
    current_ln_ntot = jnp.asarray(ln_ntot_init, dtype=jnp.float64)
    current_lambda = (
        None
        if element_potential_init is None
        else jnp.asarray(element_potential_init, dtype=jnp.float64)
    )
    current_rho = None if rho_init is None else jnp.asarray(rho_init, dtype=jnp.float64)
    current_rho_initialization = str(rho_initialization)
    stage_reports: list[dict[str, Any]] = []
    completed_stage_count = 0
    blocked_stage_index = None
    blocked_epsilon = None
    result = None
    extra = None
    final_result = None
    final_payload = None
    batch_size = current_ln_ntot.shape[0]
    active_mask = jnp.ones((batch_size,), dtype=bool)
    layer_completed_stage_count = jnp.zeros((batch_size,), dtype=jnp.int32)
    reached_final_epsilon = jnp.zeros((batch_size,), dtype=bool)
    restoration_stage_diagnostic_keys = (
        "amount_restoration_accepted_iteration_count",
        "restoration_phase_entry_count",
        "restoration_phase_exit_count",
        "restoration_phase_entry_theta_at_stop",
        "restoration_phase_active_at_stop",
        "restoration_phase_cooldown_at_stop",
        "restoration_bound_multiplier_reset_count",
        "restoration_equality_multiplier_reset_count",
        "restoration_last_exit_theta",
        "restoration_last_dual_alpha",
        "restoration_entry_residual_vector",
        "restoration_best_residual_vector",
        "restoration_best_theta",
        "restoration_last_exit_predual_residual_vector",
        "restoration_last_exit_postdual_residual_vector",
        "restoration_first_normal_residual_vector",
        "restoration_first_normal_attempted",
        "restoration_first_normal_accepted",
        "restoration_first_normal_selected_type",
        "restoration_return_probe_pending",
        "restoration_active_accepted_iteration_count_at_stop",
        "restoration_last_active_accepted_iteration_count",
    )

    def restoration_stage_value(
        payload: Mapping[str, Any],
        key: str,
    ) -> jnp.ndarray:
        if key in payload:
            return jnp.asarray(payload[key])
        shape = (batch_size, 6) if "residual_vector" in key else (batch_size,)
        if key in {
            "restoration_phase_active_at_stop",
            "restoration_first_normal_attempted",
            "restoration_first_normal_accepted",
            "restoration_return_probe_pending",
        }:
            return jnp.zeros(shape, dtype=bool)
        if (
            key.endswith("count")
            or key.endswith("count_at_stop")
            or key == "restoration_phase_cooldown_at_stop"
            or key == "restoration_first_normal_selected_type"
        ):
            default = 3 if key == "restoration_first_normal_selected_type" else 0
            return jnp.full(shape, default, dtype=jnp.int32)
        return jnp.zeros(shape, dtype=jnp.float64)

    def select_batch_rows(mask: jnp.ndarray, new: Any, old: Any) -> Any:
        """Select leading batch rows while preserving nested metadata."""

        if isinstance(new, Mapping) and isinstance(old, Mapping):
            return {
                key: select_batch_rows(mask, new[key], old[key])
                if key in old
                else new[key]
                for key in new
            }
        if not hasattr(new, "shape") or not hasattr(old, "shape"):
            return new
        new_array = jnp.asarray(new)
        old_array = jnp.asarray(old)
        if (
            new_array.ndim == 0
            or old_array.shape != new_array.shape
            or new_array.shape[0] != mask.shape[0]
        ):
            return new
        expanded_mask = jnp.reshape(mask, mask.shape + (1,) * (new_array.ndim - 1))
        return jnp.where(expanded_mask, new_array, old_array)

    def select_result_rows(
        mask: jnp.ndarray,
        new: CondensateEquilibriumResult,
        old: CondensateEquilibriumResult,
    ) -> CondensateEquilibriumResult:
        return CondensateEquilibriumResult(
            ln_nk=select_batch_rows(mask, new.ln_nk, old.ln_nk),
            ln_mk=select_batch_rows(mask, new.ln_mk, old.ln_mk),
            ln_ntot=select_batch_rows(mask, new.ln_ntot, old.ln_ntot),
            diagnostics=tree_util.tree_map(
                lambda new_value, old_value: select_batch_rows(
                    mask,
                    new_value,
                    old_value,
                ),
                new.diagnostics,
                old.diagnostics,
            ),
        )

    def recentered_for_epsilon_delta(
        ln_mk: jnp.ndarray,
        rho: Optional[jnp.ndarray],
        delta_epsilon: jnp.ndarray,
    ) -> tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
        if rho is None or continuation_recenter_policy == "none":
            return ln_mk, rho, None
        delta = jnp.asarray(delta_epsilon, dtype=ln_mk.dtype)
        if delta.ndim == 0:
            delta = jnp.full((ln_mk.shape[0],), delta, dtype=ln_mk.dtype)
        delta = delta[:, None]
        if continuation_recenter_policy == "rho":
            return ln_mk, rho + delta, jnp.ones_like(ln_mk, dtype=bool)
        if continuation_recenter_policy == "amount":
            return ln_mk + delta, rho, jnp.zeros_like(ln_mk, dtype=bool)

        positive_stoich = formula_matrix_cond_active[None, :, :] > 0.0
        safe_stoich = jnp.where(
            positive_stoich,
            formula_matrix_cond_active[None, :, :],
            jnp.asarray(1.0, dtype=formula_matrix_cond_active.dtype),
        )
        capacity = jnp.where(
            positive_stoich,
            element_inventory_target[:, :, None] / safe_stoich,
            jnp.inf,
        )
        condensate_capacity = jnp.min(capacity, axis=1)
        log_capacity = jnp.log(jnp.maximum(condensate_capacity, 1.0e-300))
        log_threshold = jnp.log(
            jnp.asarray(
                continuation_recenter_amount_relative_threshold,
                dtype=ln_mk.dtype,
            )
        )
        amount_significant = (ln_mk - log_capacity) > log_threshold
        return (
            jnp.where(amount_significant, ln_mk, ln_mk + delta),
            jnp.where(amount_significant, rho + delta, rho),
            amount_significant,
        )

    if current_rho is not None and barrier_epsilon_init is not None:
        previous_epsilon = jnp.asarray(barrier_epsilon_init, dtype=jnp.float64)
        current_ln_mk, current_rho, _initial_recenter_mask = recentered_for_epsilon_delta(
            current_ln_mk,
            current_rho,
            jnp.asarray(schedule[0], dtype=jnp.float64) - previous_epsilon,
        )
    for stage_index, stage_epsilon in enumerate(schedule):
        attempted_mask = active_mask
        barrier_epsilon_init = jnp.full_like(
            current_ln_ntot,
            stage_epsilon,
            dtype=jnp.float64,
        )
        result, extra = _solve_pdipm_rgie_v11_activity_correction_fixed_support_batch(
            ln_nk_init=current_ln_nk,
            ln_mk_init=current_ln_mk,
            ln_ntot_init=current_ln_ntot,
            element_potential_init=current_lambda,
            rho_init=current_rho,
            barrier_epsilon_init=barrier_epsilon_init,
            gas_stationarity_source_init=gas_stationarity_source_init,
            formula_matrix=formula_matrix,
            formula_matrix_cond_active=formula_matrix_cond_active,
            element_inventory_target=element_inventory_target,
            hvector=hvector,
            hvector_cond_active=hvector_cond_active,
            ln_normalized_pressure=ln_normalized_pressure,
            epsilon=stage_epsilon,
            residual_tolerance_multiplier=residual_tolerance_multiplier,
            max_iter=max_iter,
            rho_initialization=current_rho_initialization,
            lambda_initialization=lambda_initialization,
        )
        payload = extra["pdipm_rgie_v11_activity_correction_fixed_support_batch"]
        final_result = (
            result
            if final_result is None
            else select_result_rows(attempted_mask, result, final_result)
        )
        final_payload = (
            payload
            if final_payload is None
            else select_batch_rows(attempted_mask, payload, final_payload)
        )
        layer_completed_stage_count = layer_completed_stage_count + attempted_mask.astype(
            jnp.int32
        )
        if stage_index + 1 == len(schedule):
            reached_final_epsilon = attempted_mask
        continuation_log_tolerance = (
            effective_continuation_tolerance_multiplier
            * float(payload["convergence_log_tolerance"])
        )
        continuation_budget_relative_tolerance = (
            effective_continuation_tolerance_multiplier
            * float(payload["convergence_budget_relative_tolerance"])
        )
        continuation_total_density_tolerance = (
            effective_continuation_tolerance_multiplier
            * float(payload["convergence_total_density_tolerance"])
        )
        centered_raw = fixed_support_batch_converged(
            gas_norm=payload["gas_residual_norm"],
            condensate_stationarity_norm=payload[
                "condensate_stationarity_residual_norm"
            ],
            complementarity_norm=payload["complementarity_residual_norm"],
            total_density_norm=payload["total_density_residual_norm"],
            budget_relative_max=payload["budget_relative_residual_max"],
            log_tolerance=continuation_log_tolerance,
            budget_relative_tolerance=continuation_budget_relative_tolerance,
            total_density_tolerance=continuation_total_density_tolerance,
        )
        centered_for_continuation = attempted_mask & centered_raw
        stage_converged = attempted_mask & result.diagnostics.converged
        stage_reports.append(
            {
                "stage_index": stage_index,
                "epsilon": stage_epsilon,
                "attempted": attempted_mask,
                "converged": stage_converged,
                "centered_for_continuation": centered_for_continuation,
                "n_iter": jnp.where(attempted_mask, result.diagnostics.n_iter, 0),
                "final_residual": jnp.where(
                    attempted_mask,
                    result.diagnostics.final_residual,
                    jnp.inf,
                ),
                "residual_crit": jnp.where(
                    attempted_mask,
                    result.diagnostics.residual_crit,
                    jnp.nan,
                ),
                "continuation_log_tolerance": jnp.full_like(
                    result.diagnostics.final_residual,
                    continuation_log_tolerance,
                    dtype=jnp.float64,
                ),
                "continuation_budget_relative_tolerance": jnp.full_like(
                    result.diagnostics.final_residual,
                    continuation_budget_relative_tolerance,
                    dtype=jnp.float64,
                ),
                "continuation_total_density_tolerance": jnp.full_like(
                    result.diagnostics.final_residual,
                    continuation_total_density_tolerance,
                    dtype=jnp.float64,
                ),
                "continuation_recenter_policy": continuation_recenter_policy,
                "continuation_recenter_amount_relative_threshold": (
                    continuation_recenter_amount_relative_threshold
                ),
                "accepted_iteration_count": jnp.where(
                    attempted_mask,
                    payload["accepted_iteration_count"],
                    0,
                ),
                "rejected_trial_count": jnp.where(
                    attempted_mask,
                    payload["rejected_trial_count"],
                    0,
                ),
                "final_step_size": jnp.where(
                    attempted_mask,
                    payload["final_step_size"],
                    0.0,
                ),
                "stop_reason_code": jnp.where(
                    attempted_mask,
                    payload["stop_reason_code"],
                    5,
                ),
                "dominant_residual_component_index": jnp.where(
                    attempted_mask,
                    payload["dominant_residual_component_index"],
                    3,
                ),
                **{
                    key: select_batch_rows(
                        attempted_mask,
                        restoration_stage_value(payload, key),
                        jnp.zeros_like(restoration_stage_value(payload, key)),
                    )
                    for key in restoration_stage_diagnostic_keys
                },
            }
        )
        completed_stage_count += 1
        current_ln_nk = select_batch_rows(
            attempted_mask,
            result.ln_nk,
            current_ln_nk,
        )
        current_ln_mk = select_batch_rows(
            attempted_mask,
            result.ln_mk,
            current_ln_mk,
        )
        current_ln_ntot = select_batch_rows(
            attempted_mask,
            result.ln_ntot,
            current_ln_ntot,
        )
        current_lambda = (
            payload["final_element_potential"]
            if current_lambda is None
            else select_batch_rows(
                attempted_mask,
                payload["final_element_potential"],
                current_lambda,
            )
        )
        current_rho = (
            payload["final_log_activity_correction"]
            if current_rho is None
            else select_batch_rows(
                attempted_mask,
                payload["final_log_activity_correction"],
                current_rho,
            )
        )
        active_mask = centered_for_continuation
        if stage_index + 1 < len(schedule):
            if not bool(np.any(np.asarray(jax.device_get(active_mask)))):
                blocked_stage_index = stage_index
                blocked_epsilon = stage_epsilon
                break
            next_epsilon = schedule[stage_index + 1]
            recentered_ln_mk, recentered_rho, _recenter_mask = (
                recentered_for_epsilon_delta(
                    current_ln_mk,
                    current_rho,
                    jnp.asarray(
                        next_epsilon - stage_epsilon,
                        dtype=jnp.asarray(current_ln_mk).dtype,
                    ),
                )
            )
            current_ln_mk = select_batch_rows(
                active_mask,
                recentered_ln_mk,
                current_ln_mk,
            )
            if recentered_rho is not None and current_rho is not None:
                current_rho = select_batch_rows(
                    active_mask,
                    recentered_rho,
                    current_rho,
                )
        current_rho_initialization = "provided"
    if final_result is None or final_payload is None or extra is None:
        raise RuntimeError("epsilon continuation did not run any stages.")
    if len(stage_reports) < len(schedule):
        batch_shape = jnp.asarray(result.diagnostics.final_residual).shape
        for stage_index in range(len(stage_reports), len(schedule)):
            stage_epsilon = schedule[stage_index]
            stage_reports.append(
                {
                    "stage_index": stage_index,
                    "epsilon": stage_epsilon,
                    "attempted": jnp.zeros(batch_shape, dtype=bool),
                    "converged": jnp.zeros(batch_shape, dtype=bool),
                    "centered_for_continuation": jnp.zeros(
                        batch_shape,
                        dtype=bool,
                    ),
                    "n_iter": jnp.zeros(batch_shape, dtype=jnp.int32),
                    "final_residual": jnp.full(
                        batch_shape,
                        jnp.inf,
                        dtype=jnp.float64,
                    ),
                    "residual_crit": jnp.full(
                        batch_shape,
                        jnp.nan,
                        dtype=jnp.float64,
                    ),
                    "continuation_log_tolerance": jnp.full(
                        batch_shape,
                        jnp.nan,
                        dtype=jnp.float64,
                    ),
                    "continuation_budget_relative_tolerance": jnp.full(
                        batch_shape,
                        jnp.nan,
                        dtype=jnp.float64,
                    ),
                    "continuation_total_density_tolerance": jnp.full(
                        batch_shape,
                        jnp.nan,
                        dtype=jnp.float64,
                    ),
                    "continuation_recenter_policy": continuation_recenter_policy,
                    "continuation_recenter_amount_relative_threshold": (
                        continuation_recenter_amount_relative_threshold
                    ),
                    "accepted_iteration_count": jnp.zeros(
                        batch_shape,
                        dtype=jnp.int32,
                    ),
                    "rejected_trial_count": jnp.zeros(batch_shape, dtype=jnp.int32),
                    "final_step_size": jnp.zeros(batch_shape, dtype=jnp.float64),
                    "stop_reason_code": jnp.full(
                        batch_shape,
                        5,
                        dtype=jnp.int32,
                    ),
                    "dominant_residual_component_index": jnp.full(
                        batch_shape,
                        3,
                        dtype=jnp.int32,
                    ),
                    **{
                        key: jnp.zeros(
                            batch_shape + ((6,) if "residual_vector" in key else ()),
                            dtype=(
                                bool
                                if key
                                in {
                                    "restoration_phase_active_at_stop",
                                    "restoration_first_normal_attempted",
                                    "restoration_first_normal_accepted",
                                    "restoration_return_probe_pending",
                                }
                                else (
                                    jnp.int32
                                    if key.endswith("count")
                                    or key.endswith("count_at_stop")
                                    or key
                                    == "restoration_phase_cooldown_at_stop"
                                    or key == "restoration_first_normal_selected_type"
                                    else jnp.float64
                                )
                            ),
                        )
                        for key in restoration_stage_diagnostic_keys
                    },
                }
            )
    strict_final_converged = (
        final_result.diagnostics.converged & reached_final_epsilon
    )
    final_diagnostics_mapping = final_result.diagnostics.asdict()
    final_diagnostics_mapping.update(
        {
            "converged": strict_final_converged,
            "requested_epsilon": jnp.full_like(
                final_result.diagnostics.epsilon,
                schedule[-1],
            ),
            "reached_requested_epsilon": reached_final_epsilon,
        }
    )
    final_result = CondensateEquilibriumResult(
        ln_nk=final_result.ln_nk,
        ln_mk=final_result.ln_mk,
        ln_ntot=final_result.ln_ntot,
        diagnostics=CondensateEquilibriumDiagnostics.from_mapping(
            final_diagnostics_mapping
        ),
    )
    return (
        final_result,
        {
            **extra,
            "pdipm_rgie_v11_activity_correction_fixed_support_batch": final_payload,
            "pdipm_rgie_v11_activity_correction_fixed_support_batch_continuation": {
                "schema": (
                    "exogibbs_pdipm_rgie_v11_activity_correction_fixed_support_"
                    "batch_continuation_v1"
                ),
                "epsilon_schedule": schedule,
                "stage_count": len(schedule),
                "completed_stage_count": completed_stage_count,
                "stopped_early": completed_stage_count < len(schedule),
                "blocked_stage_index": blocked_stage_index,
                "blocked_epsilon": blocked_epsilon,
                "layer_completed_stage_count": layer_completed_stage_count,
                "reached_final_epsilon": reached_final_epsilon,
                "continuation_tolerance_multiplier": (
                    continuation_tolerance_multiplier
                ),
                "effective_continuation_tolerance_multiplier": (
                    effective_continuation_tolerance_multiplier
                ),
                "continuation_recenter_policy": continuation_recenter_policy,
                "continuation_recenter_amount_relative_threshold": (
                    continuation_recenter_amount_relative_threshold
                ),
                "stages": tuple(stage_reports),
            },
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
    lambda_initialization: str = "best_residual",
    residual_tolerance_multiplier: float = 1.0,
) -> tuple[tuple[CondensateEquilibriumResult, ...], dict[str, Any]]:
    """Run already-prepared profile buckets without per-layer materialization."""

    formula_matrix = jnp.asarray(formula_matrix, dtype=jnp.float64)
    epsilon_schedule = _parse_fixed_support_batch_epsilon_schedule(epsilon)
    results = []
    bucket_reports = []
    for bucket in buckets:
        batch_result, batch_extra = (
            _solve_pdipm_rgie_v11_activity_correction_fixed_support_batch_continuation(
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
                epsilon_schedule=epsilon_schedule,
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
        continuation_payload = batch_extra[
            "pdipm_rgie_v11_activity_correction_fixed_support_batch_continuation"
        ]
        bucket_reports.append(
            {
                "support_indices": bucket.support_indices,
                "layer_indices": bucket.layer_indices,
                "execution": "batch",
                "batch_size": len(bucket.layer_indices),
                "epsilon_schedule": continuation_payload["epsilon_schedule"],
                "continuation_stage_count": continuation_payload["stage_count"],
                "continuation_completed_stage_count": continuation_payload[
                    "completed_stage_count"
                ],
                "continuation_stopped_early": continuation_payload["stopped_early"],
                "continuation_blocked_stage_index": continuation_payload[
                    "blocked_stage_index"
                ],
                "continuation_blocked_epsilon": continuation_payload[
                    "blocked_epsilon"
                ],
                "continuation_layer_completed_stage_count": continuation_payload[
                    "layer_completed_stage_count"
                ],
                "continuation_reached_final_epsilon": continuation_payload[
                    "reached_final_epsilon"
                ],
                "continuation_stages": continuation_payload["stages"],
                "accepted_iteration_count": batch_payload[
                    "accepted_iteration_count"
                ],
                "stationarity_restoration_accepted_iteration_count": batch_payload[
                    "stationarity_restoration_accepted_iteration_count"
                ],
                "normal_accepted_iteration_count": batch_payload[
                    "normal_accepted_iteration_count"
                ],
                "fallback_accepted_iteration_count": batch_payload[
                    "fallback_accepted_iteration_count"
                ],
                "second_order_correction_accepted_iteration_count": batch_payload[
                    "second_order_correction_accepted_iteration_count"
                ],
                "amount_restoration_accepted_iteration_count": batch_payload[
                    "amount_restoration_accepted_iteration_count"
                ],
                "restoration_phase_entry_count": batch_payload[
                    "restoration_phase_entry_count"
                ],
                "restoration_phase_exit_count": batch_payload[
                    "restoration_phase_exit_count"
                ],
                "restoration_phase_entry_theta_at_stop": batch_payload[
                    "restoration_phase_entry_theta_at_stop"
                ],
                "restoration_phase_active_at_stop": batch_payload[
                    "restoration_phase_active_at_stop"
                ],
                "restoration_phase_cooldown_at_stop": batch_payload[
                    "restoration_phase_cooldown_at_stop"
                ],
                "restoration_bound_multiplier_reset_count": batch_payload[
                    "restoration_bound_multiplier_reset_count"
                ],
                "restoration_equality_multiplier_reset_count": batch_payload[
                    "restoration_equality_multiplier_reset_count"
                ],
                "restoration_last_exit_theta": batch_payload[
                    "restoration_last_exit_theta"
                ],
                "restoration_last_dual_alpha": batch_payload[
                    "restoration_last_dual_alpha"
                ],
                "restoration_entry_residual_vector": batch_payload[
                    "restoration_entry_residual_vector"
                ],
                "restoration_best_residual_vector": batch_payload[
                    "restoration_best_residual_vector"
                ],
                "restoration_best_theta": batch_payload[
                    "restoration_best_theta"
                ],
                "restoration_last_exit_predual_residual_vector": batch_payload[
                    "restoration_last_exit_predual_residual_vector"
                ],
                "restoration_last_exit_postdual_residual_vector": batch_payload[
                    "restoration_last_exit_postdual_residual_vector"
                ],
                "restoration_first_normal_residual_vector": batch_payload[
                    "restoration_first_normal_residual_vector"
                ],
                "restoration_first_normal_attempted": batch_payload[
                    "restoration_first_normal_attempted"
                ],
                "restoration_first_normal_accepted": batch_payload[
                    "restoration_first_normal_accepted"
                ],
                "restoration_first_normal_selected_type": batch_payload[
                    "restoration_first_normal_selected_type"
                ],
                "restoration_return_probe_pending": batch_payload[
                    "restoration_return_probe_pending"
                ],
                "restoration_active_accepted_iteration_count_at_stop": (
                    batch_payload[
                        "restoration_active_accepted_iteration_count_at_stop"
                    ]
                ),
                "restoration_last_active_accepted_iteration_count": batch_payload[
                    "restoration_last_active_accepted_iteration_count"
                ],
                "rejected_trial_count": batch_payload["rejected_trial_count"],
                "final_step_size": batch_payload["final_step_size"],
                "stop_reason_code": batch_payload["stop_reason_code"],
                "dominant_residual_component_index": batch_payload[
                    "dominant_residual_component_index"
                ],
                "initial_residual": batch_payload["initial_residual"],
                "lambda_selection_index": batch_payload["lambda_selection_index"],
                "use_log_amount_boundary": batch_payload[
                    "use_log_amount_boundary"
                ],
                "use_log_activity_boundary": batch_payload[
                    "use_log_activity_boundary"
                ],
                "line_search_alpha_boundary": batch_payload[
                    "line_search_alpha_boundary"
                ],
                "line_search_alpha_r": batch_payload["line_search_alpha_r"],
                "line_search_alpha_rho": batch_payload["line_search_alpha_rho"],
                "line_search_selected_trial_index": batch_payload[
                    "line_search_selected_trial_index"
                ],
                "line_search_selected_trial_alpha": batch_payload[
                    "line_search_selected_trial_alpha"
                ],
                "line_search_selected_trial_residual": batch_payload[
                    "line_search_selected_trial_residual"
                ],
                "line_search_accepted_candidate_count": batch_payload[
                    "line_search_accepted_candidate_count"
                ],
                "line_search_fallback_candidate_count": batch_payload[
                    "line_search_fallback_candidate_count"
                ],
                "line_search_best_trial_index": batch_payload[
                    "line_search_best_trial_index"
                ],
                "line_search_best_trial_alpha": batch_payload[
                    "line_search_best_trial_alpha"
                ],
                "line_search_best_trial_residual": batch_payload[
                    "line_search_best_trial_residual"
                ],
                "line_search_best_trial_gas_residual": batch_payload[
                    "line_search_best_trial_gas_residual"
                ],
                "line_search_best_trial_condensate_stationarity_residual": (
                    batch_payload[
                        "line_search_best_trial_condensate_stationarity_residual"
                    ]
                ),
                "line_search_best_trial_budget_residual": batch_payload[
                    "line_search_best_trial_budget_residual"
                ],
                "line_search_best_trial_budget_relative_residual_max": (
                    batch_payload[
                        "line_search_best_trial_budget_relative_residual_max"
                    ]
                ),
                "line_search_best_trial_complementarity_residual": batch_payload[
                    "line_search_best_trial_complementarity_residual"
                ],
                "line_search_best_trial_total_density_residual": batch_payload[
                    "line_search_best_trial_total_density_residual"
                ],
                "line_search_finite_candidate_count": batch_payload[
                    "line_search_finite_candidate_count"
                ],
                "line_search_combined_improved_candidate_count": batch_payload[
                    "line_search_combined_improved_candidate_count"
                ],
                "line_search_budget_relative_not_worse_candidate_count": (
                    batch_payload[
                        "line_search_budget_relative_not_worse_candidate_count"
                    ]
                ),
                "line_search_filter_candidate_count": batch_payload[
                    "line_search_filter_candidate_count"
                ],
                "line_search_budget_not_broken_candidate_count": batch_payload[
                    "line_search_budget_not_broken_candidate_count"
                ],
                "line_search_budget_relative_not_broken_candidate_count": (
                    batch_payload[
                        "line_search_budget_relative_not_broken_candidate_count"
                    ]
                ),
                "line_search_combined_not_worse_candidate_count": batch_payload[
                    "line_search_combined_not_worse_candidate_count"
                ],
                "line_search_best_trial_finite": batch_payload[
                    "line_search_best_trial_finite"
                ],
                "line_search_best_trial_combined_improved": batch_payload[
                    "line_search_best_trial_combined_improved"
                ],
                "line_search_best_trial_budget_relative_not_worse": batch_payload[
                    "line_search_best_trial_budget_relative_not_worse"
                ],
                "line_search_best_trial_filter_accepted": batch_payload[
                    "line_search_best_trial_filter_accepted"
                ],
                "line_search_best_trial_budget_not_broken": batch_payload[
                    "line_search_best_trial_budget_not_broken"
                ],
                "line_search_best_trial_budget_relative_not_broken": batch_payload[
                    "line_search_best_trial_budget_relative_not_broken"
                ],
                "line_search_best_trial_combined_not_worse": batch_payload[
                    "line_search_best_trial_combined_not_worse"
                ],
                "line_search_best_trial_accepted": batch_payload[
                    "line_search_best_trial_accepted"
                ],
                "line_search_best_trial_fallback_accepted": batch_payload[
                    "line_search_best_trial_fallback_accepted"
                ],
                "line_search_soc_candidate_count": batch_payload[
                    "line_search_soc_candidate_count"
                ],
                "line_search_soc_accepted_candidate_count": batch_payload[
                    "line_search_soc_accepted_candidate_count"
                ],
                "line_search_soc_fallback_candidate_count": batch_payload[
                    "line_search_soc_fallback_candidate_count"
                ],
                "line_search_soc_budget_relative_not_worse_candidate_count": (
                    batch_payload[
                        "line_search_soc_budget_relative_not_worse_candidate_count"
                    ]
                ),
                "line_search_soc_filter_candidate_count": batch_payload[
                    "line_search_soc_filter_candidate_count"
                ],
                "line_search_soc_best_trial_present": batch_payload[
                    "line_search_soc_best_trial_present"
                ],
                "line_search_soc_best_trial_index": batch_payload[
                    "line_search_soc_best_trial_index"
                ],
                "line_search_soc_best_trial_alpha": batch_payload[
                    "line_search_soc_best_trial_alpha"
                ],
                "line_search_soc_best_trial_residual": batch_payload[
                    "line_search_soc_best_trial_residual"
                ],
                "line_search_soc_best_trial_gas_residual": batch_payload[
                    "line_search_soc_best_trial_gas_residual"
                ],
                "line_search_soc_best_trial_condensate_stationarity_residual": (
                    batch_payload[
                        "line_search_soc_best_trial_condensate_stationarity_residual"
                    ]
                ),
                "line_search_soc_best_trial_budget_residual": batch_payload[
                    "line_search_soc_best_trial_budget_residual"
                ],
                "line_search_soc_best_trial_budget_relative_residual_max": (
                    batch_payload[
                        "line_search_soc_best_trial_budget_relative_residual_max"
                    ]
                ),
                "line_search_soc_best_trial_complementarity_residual": batch_payload[
                    "line_search_soc_best_trial_complementarity_residual"
                ],
                "line_search_soc_best_trial_total_density_residual": batch_payload[
                    "line_search_soc_best_trial_total_density_residual"
                ],
                "line_search_soc_best_trial_combined_improved": batch_payload[
                    "line_search_soc_best_trial_combined_improved"
                ],
                "line_search_soc_best_trial_budget_relative_not_worse": batch_payload[
                    "line_search_soc_best_trial_budget_relative_not_worse"
                ],
                "line_search_soc_best_trial_filter_accepted": batch_payload[
                    "line_search_soc_best_trial_filter_accepted"
                ],
                "line_search_soc_best_trial_accepted": batch_payload[
                    "line_search_soc_best_trial_accepted"
                ],
                "line_search_soc_best_trial_fallback_accepted": batch_payload[
                    "line_search_soc_best_trial_fallback_accepted"
                ],
                "line_search_selected_trial_gas_residual": batch_payload[
                    "line_search_selected_trial_gas_residual"
                ],
                "line_search_selected_trial_condensate_stationarity_residual": (
                    batch_payload[
                        "line_search_selected_trial_condensate_stationarity_residual"
                    ]
                ),
                "line_search_selected_trial_budget_residual": batch_payload[
                    "line_search_selected_trial_budget_residual"
                ],
                "line_search_selected_trial_budget_relative_residual_max": (
                    batch_payload[
                        "line_search_selected_trial_budget_relative_residual_max"
                    ]
                ),
                "line_search_selected_trial_complementarity_residual": (
                    batch_payload[
                        "line_search_selected_trial_complementarity_residual"
                    ]
                ),
                "line_search_selected_trial_total_density_residual": batch_payload[
                    "line_search_selected_trial_total_density_residual"
                ],
                "line_search_candidate_diagnostics": batch_payload[
                    "line_search_candidate_diagnostics"
                ],
                "gas_residual_norm": batch_payload["gas_residual_norm"],
                "condensate_stationarity_residual_norm": batch_payload[
                    "condensate_stationarity_residual_norm"
                ],
                "budget_residual_norm": batch_payload["budget_residual_norm"],
                "complementarity_residual_norm": batch_payload[
                    "complementarity_residual_norm"
                ],
                "total_density_residual_norm": batch_payload[
                    "total_density_residual_norm"
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
