"""Backward-compatible import path and structured API for condensate minimization."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from time import perf_counter
from typing import Any, Literal, Optional, Sequence

import jax
import jax.numpy as jnp
from jax import lax, tree_util
from scipy.optimize import least_squares

from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.core import _compute_gk
from exogibbs.optimize.stepsize import LOG_S_MAX
from exogibbs.optimize.pdipm_cond import minimize_gibbs_cond_core
from exogibbs.optimize.minimize import minimize_gibbs_core
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
    "candidate_selected_weighted_mask",
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

    def tree_flatten(self):
        children = (self.ln_nk, self.ln_mk, self.ln_ntot)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        ln_nk, ln_mk, ln_ntot = children
        return cls(ln_nk=ln_nk, ln_mk=ln_mk, ln_ntot=ln_ntot)


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
    )


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
        "candidate_selected_weighted_mask",
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
        },
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
):
    """Solve the gas-only subproblem and recover a practical dual vector."""

    ln_nk_init0 = jnp.zeros((formula_matrix.shape[1],), dtype=jnp.float64)
    ln_ntot_init0 = jnp.asarray(0.0, dtype=jnp.float64)
    hvector = jnp.asarray(hvector_func(state.temperature), dtype=jnp.float64)
    ln_nk, ln_ntot, n_iter, final_residual = minimize_gibbs_core(
        state,
        ln_nk_init0,
        ln_ntot_init0,
        formula_matrix,
        lambda _temperature: hvector,
        epsilon_crit=gas_epsilon_crit,
        max_iter=gas_max_iter,
    )
    nk = jnp.exp(jnp.asarray(ln_nk, dtype=jnp.float64))
    ntot = jnp.exp(jnp.asarray(ln_ntot, dtype=jnp.float64))
    gk = _compute_gk(state.temperature, ln_nk, ln_ntot, hvector, state.ln_normalized_pressure)
    qmat = formula_matrix @ (nk[:, None] * formula_matrix.T)
    rhs = formula_matrix @ (gk * nk)
    pi_vector = jnp.linalg.lstsq(qmat, rhs)[0]
    stationarity = formula_matrix.T @ pi_vector - gk
    return {
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


def _support_signature_export(
    condensate_species: Optional[Sequence[str]],
    element_names: Optional[Sequence[str]],
    formula_matrix_cond: jnp.ndarray,
    support_indices: jnp.ndarray,
) -> dict[str, Any]:
    names = (
        [str(condensate_species[int(index)]) for index in support_indices.tolist()]
        if condensate_species is not None
        else [str(int(index)) for index in support_indices.tolist()]
    )
    entries = []
    associated_element_coverage = set()
    for local_pos, cond_index in enumerate(support_indices.tolist()):
        stoich = jnp.asarray(formula_matrix_cond[:, int(cond_index)], dtype=jnp.float64)
        element_indices = [int(i) for i in range(stoich.shape[0]) if float(stoich[i]) > 0.0]
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
        "scalar_merit": scalar_merit,
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
    gas_epsilon_crit: float = 1.0e-12,
    gas_max_iter: int = 1000,
    epsilon: float = -10.0,
    max_iter: int = 100,
    startup_config: Optional[CondensateRGIEStartupConfig] = None,
    least_squares_max_nfev: int = 50,
):
    """Run the current RGIE local solve on a fixed candidate support."""

    del least_squares_max_nfev
    support_indices = jnp.asarray(support_indices, dtype=jnp.int32)
    hvector_cond_full = jnp.asarray(hvector_cond_func(state.temperature), dtype=jnp.float64)
    formula_matrix_cond_active = jnp.asarray(formula_matrix_cond[:, support_indices], dtype=jnp.float64)
    hvector_cond_active = jnp.asarray(hvector_cond_full[support_indices], dtype=jnp.float64)
    gas_start = solve_gas_equilibrium_with_duals(
        state,
        formula_matrix,
        hvector_func,
        gas_epsilon_crit=gas_epsilon_crit,
        gas_max_iter=gas_max_iter,
    )
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
    result = _minimize_gibbs_cond_legacy(
        state,
        init=CondensateEquilibriumInit(
            ln_nk=jnp.asarray(gas_start["ln_nk"], dtype=jnp.float64),
            ln_mk=jnp.log(jnp.maximum(support_amounts_init, 1.0e-300)),
            ln_ntot=jnp.asarray(gas_start["ln_ntot"], dtype=jnp.float64),
        ),
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
        reduced_coupling_config=None,
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
    b_eff = jnp.asarray(state.element_vector, dtype=jnp.float64) - formula_matrix_cond_active @ jnp.exp(result.ln_mk)
    return {
        "status": "ok",
        "raw_final_status": "ok",
        "solver_success": bool(result.diagnostics.converged),
        "solver_status": int(result.diagnostics.n_iter),
        "solver_message": "rgie_restricted_support",
        "support_size": int(support_indices.shape[0]),
        "support_indices": [int(i) for i in support_indices.tolist()],
        "support_names": metrics["support_names"],
        "active_support_count": int(jnp.sum(jnp.exp(result.ln_mk) > 0.0)),
        "m_support": jnp.exp(result.ln_mk),
        "ln_m_support": jnp.asarray(result.ln_mk, dtype=jnp.float64),
        "ln_nk": jnp.asarray(result.ln_nk, dtype=jnp.float64),
        "ln_ntot": jnp.asarray(result.ln_ntot, dtype=jnp.float64),
        "diagnostics": result.diagnostics.asdict(),
        "feasible_projection_alpha": 1.0,
        "restricted_kkt_gap_inf": metrics["scalar_merit"],
        "max_positive_inactive_driving": metrics["max_positive_inactive_driving"],
        "inactive_positive_count": metrics["inactive_positive_count"],
        "top_inactive_names": metrics["top_inactive_names"],
        "top_inactive_driving": metrics["top_inactive_driving"],
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
        "complementarity_residual_inf": metrics["complementarity_residual_inf"],
        "scalar_merit": metrics["scalar_merit"],
        "full_driving": metrics["full_driving"],
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
):
    """Diagnostic-only wrapper for one-step reduced-solver backend comparisons."""

    init_prepared = _prepare_condensate_init(init)
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
