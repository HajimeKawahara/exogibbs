"""Backward-compatible import path and structured API for condensate minimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.stepsize import LOG_S_MAX
from exogibbs.optimize.condensate_outer_diagnostics import (
    diagnose_dynamic_support_outer_objective_layer as _diagnose_dynamic_support_outer_objective_layer,
    diagnose_dynamic_support_outer_objective_layer_with_start_portfolio as _diagnose_dynamic_support_outer_objective_layer_with_start_portfolio,
    diagnose_augmented_semismooth_candidate_condensate_layer as _diagnose_augmented_semismooth_candidate_condensate_layer,
    diagnose_semismooth_candidate_condensate_layer as _diagnose_semismooth_candidate_condensate_layer,
    diagnose_smoothed_semismooth_candidate_condensate_layer as _diagnose_smoothed_semismooth_candidate_condensate_layer,
    diagnose_support_updating_active_set_layer as _diagnose_support_updating_active_set_layer,
    diagnose_condensate_outer_active_set_layer as _diagnose_condensate_outer_active_set_layer,
    diagnose_outer_objective_candidate_condensate_layer as _diagnose_outer_objective_candidate_condensate_layer,
    evaluate_outer_objective_on_candidate_support as _evaluate_outer_objective_on_candidate_support,
    optimize_outer_objective_on_candidate_support as _optimize_outer_objective_on_candidate_support,
    solve_augmented_semismooth_candidate_condensate_layer as _solve_augmented_semismooth_candidate_condensate_layer,
    solve_gas_equilibrium_with_duals as _solve_gas_equilibrium_with_duals,
    solve_restricted_support_condensate_layer as _solve_restricted_support_condensate_layer,
    solve_semismooth_candidate_condensate_layer as _solve_semismooth_candidate_condensate_layer,
    solve_smoothed_semismooth_candidate_condensate_layer as _solve_smoothed_semismooth_candidate_condensate_layer,
)
from exogibbs.optimize.pdipm_cond import minimize_gibbs_cond_core
from exogibbs.optimize.pipm_rgie_cond import (
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
    condensate_species: Optional[Sequence[str]] = None,
    top_k: int = 5,
):
    """Run one layer with an sk-feasibility-aware epsilon schedule."""

    current_init = _prepare_condensate_init(init)
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
) -> CondensateEquilibriumResult:
    """Run the active condensate solver using a structured init/result interface."""

    init_prepared = _prepare_condensate_init(init)
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
    )
    return CondensateEquilibriumResult(
        ln_nk=ln_nk,
        ln_mk=ln_mk,
        ln_ntot=ln_ntot,
        diagnostics=CondensateEquilibriumDiagnostics.from_mapping(diagnostics_raw),
    )


def minimize_gibbs_cond_with_diagnostics(*args, **kwargs) -> CondensateEquilibriumResult:
    """Alias of :func:`minimize_gibbs_cond` kept for explicit diagnostics-oriented callers."""

    return minimize_gibbs_cond(*args, **kwargs)


def solve_gas_equilibrium_with_duals(*args, **kwargs):
    """Diagnostic-only gas solve that also recovers the converged gas dual vector."""

    return _solve_gas_equilibrium_with_duals(*args, **kwargs)


def diagnose_condensate_outer_active_set_layer(*args, **kwargs):
    """Diagnostic-only condensate-outer / gas-inner active-set prototype for one layer."""

    return _diagnose_condensate_outer_active_set_layer(*args, **kwargs)


def solve_restricted_support_condensate_layer(*args, **kwargs):
    """Diagnostic-only restricted-support nonlinear condensate outer solve for one layer."""

    return _solve_restricted_support_condensate_layer(*args, **kwargs)


def diagnose_support_updating_active_set_layer(*args, **kwargs):
    """Diagnostic-only support-updating condensate active-set outer loop for one layer."""

    return _diagnose_support_updating_active_set_layer(*args, **kwargs)


def solve_semismooth_candidate_condensate_layer(*args, **kwargs):
    """Diagnostic-only semismooth complementarity solve on a restricted candidate set."""

    return _solve_semismooth_candidate_condensate_layer(*args, **kwargs)


def diagnose_semismooth_candidate_condensate_layer(*args, **kwargs):
    """Diagnostic-only LP-seeded semismooth complementarity prototype for one layer."""

    return _diagnose_semismooth_candidate_condensate_layer(*args, **kwargs)


def solve_smoothed_semismooth_candidate_condensate_layer(*args, **kwargs):
    """Diagnostic-only smoothed semismooth continuation solve on a restricted candidate set."""

    return _solve_smoothed_semismooth_candidate_condensate_layer(*args, **kwargs)


def diagnose_smoothed_semismooth_candidate_condensate_layer(*args, **kwargs):
    """Diagnostic-only LP-seeded smoothed semismooth continuation prototype for one layer."""

    return _diagnose_smoothed_semismooth_candidate_condensate_layer(*args, **kwargs)


def solve_augmented_semismooth_candidate_condensate_layer(*args, **kwargs):
    """Diagnostic-only augmented semismooth solve with active and inactive KKT residuals."""

    return _solve_augmented_semismooth_candidate_condensate_layer(*args, **kwargs)


def diagnose_augmented_semismooth_candidate_condensate_layer(*args, **kwargs):
    """Diagnostic-only LP-seeded augmented semismooth prototype for one layer."""

    return _diagnose_augmented_semismooth_candidate_condensate_layer(*args, **kwargs)


def evaluate_outer_objective_on_candidate_support(*args, **kwargs):
    """Diagnostic-only outer objective evaluation on a fixed condensate support."""

    return _evaluate_outer_objective_on_candidate_support(*args, **kwargs)


def optimize_outer_objective_on_candidate_support(*args, **kwargs):
    """Diagnostic-only constrained outer objective optimization on a fixed support."""

    return _optimize_outer_objective_on_candidate_support(*args, **kwargs)


def diagnose_outer_objective_candidate_condensate_layer(*args, **kwargs):
    """Diagnostic-only LP-seeded outer objective optimization prototype for one layer."""

    return _diagnose_outer_objective_candidate_condensate_layer(*args, **kwargs)


def diagnose_dynamic_support_outer_objective_layer(*args, **kwargs):
    """Diagnostic-only LP-seeded dynamic-support outer objective prototype."""

    return _diagnose_dynamic_support_outer_objective_layer(*args, **kwargs)


def diagnose_dynamic_support_outer_objective_layer_with_start_portfolio(*args, **kwargs):
    """Diagnostic-only dynamic-support outer loop with online local-start portfolio selection."""

    return _diagnose_dynamic_support_outer_objective_layer_with_start_portfolio(*args, **kwargs)


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

    if epsilon_schedule == "adaptive_sk_guard":
        def solve_layer_adaptive(
            temperature: Array,
            ln_normalized_pressure: Array,
            layer_init: CondensateEquilibriumInit,
            run_full_schedule: bool,
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
            for temperature, ln_normalized_pressure in zip(
                temperatures_scan.tolist(),
                ln_pressures_scan.tolist(),
            ):
                result = solve_layer_adaptive(
                    jnp.asarray(temperature),
                    jnp.asarray(ln_normalized_pressure),
                    carry_init,
                    run_full_schedule,
                )
                results.append(result)
                carry_init = result.to_init()
                run_full_schedule = not skip_rewind_after_first_layer
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
    ) -> CondensateEquilibriumResult:
        thermo_state = ThermoState(
            temperature=temperature,
            ln_normalized_pressure=ln_normalized_pressure,
            element_vector=element_vector,
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
            )
            return result.to_init()

        final_epsilon = epsilons[-1]
        prepared_init = _prepare_condensate_init(layer_init)
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
            ),
            out_axes=0,
        )(
            temperatures,
            ln_normalized_pressures,
            batched_init,
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
        def scan_body(carry, layer_inputs):
            carry_init, run_full_schedule = carry
            temperature, ln_normalized_pressure = layer_inputs
            result = solve_layer(
                temperature,
                ln_normalized_pressure,
                carry_init,
                run_full_schedule,
            )
            next_run_full_schedule = jnp.asarray(not skip_rewind_after_first_layer)
            return (result.to_init(), next_run_full_schedule), result

        init_carry = (
            init0,
            jnp.asarray(True),
        )
        _, result_seq = lax.scan(scan_body, init_carry, (temperatures_scan, ln_pressures_scan))
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
    "CondensateEquilibriumResult",
    "compute_sk_feasible_epsilon_floor",
    "solve_gas_equilibrium_with_duals",
    "diagnose_condensate_outer_active_set_layer",
    "solve_restricted_support_condensate_layer",
    "diagnose_support_updating_active_set_layer",
    "solve_semismooth_candidate_condensate_layer",
    "diagnose_semismooth_candidate_condensate_layer",
    "solve_smoothed_semismooth_candidate_condensate_layer",
    "diagnose_smoothed_semismooth_candidate_condensate_layer",
    "solve_augmented_semismooth_candidate_condensate_layer",
    "diagnose_augmented_semismooth_candidate_condensate_layer",
    "diagnose_dynamic_support_outer_objective_layer",
    "diagnose_dynamic_support_outer_objective_layer_with_start_portfolio",
    "minimize_gibbs_cond",
    "minimize_gibbs_cond_profile",
    "minimize_gibbs_cond_core",
    "minimize_gibbs_cond_with_diagnostics",
    "trace_adaptive_condensate_schedule",
    "trace_condensate_gas_limiter_diagnostics",
    "trace_condensate_iteration_lambda_trials",
    "trace_condensate_full_vs_reduced_gie_direction",
    "trace_condensate_pdipm_vs_pipm_direction",
    "trace_condensate_pdipm_vs_pipm_fixed_epsilon_trajectories",
    "trace_condensate_reduced_solver_backends",
    "trace_condensate_sk_stage_feasibility",
]
