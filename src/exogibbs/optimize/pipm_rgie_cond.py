import jax
import jax.numpy as jnp
import math
from jax import debug as jdebug
from jax.lax import while_loop
from jax.lax import cond
from jax.lax import fori_loop
from jax.scipy.linalg import cho_factor, cho_solve
from jax.scipy.linalg import lu_factor, lu_solve
from typing import Dict, Tuple, Optional, Sequence, Any
from time import perf_counter

from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.core import _A_diagn_At
from exogibbs.optimize.core import _compute_gk
from exogibbs.optimize.minimize import solve_gibbs_iteration_equations
from exogibbs.optimize.pipm_gie_cond import (
    solve_gibbs_iteration_equations_cond as solve_full_gibbs_iteration_equations_cond,
)

# heuristic step size functions for condensates
from exogibbs.optimize.stepsize import stepsize_cea_gas
from exogibbs.optimize.stepsize import stepsize_cond_heurstic
from exogibbs.optimize.stepsize import stepsize_sk
from exogibbs.optimize.stepsize import LOG_S_MAX

DEFAULT_REDUCED_SOLVER = "augmented_lu_row_scaled"
DEFAULT_REGULARIZATION_MODE = "none"
DEFAULT_REGULARIZATION_STRENGTH = 0.0


def diagnose_rgie_raw_condensate_update_block(
    ln_mk: jnp.ndarray,
    epsilon: float,
    formula_matrix_cond: jnp.ndarray,
    pi_vector: jnp.ndarray,
    hvector_cond: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Return the current RGIE raw condensate update anatomy.

    This is diagnostic-only. It exposes the decomposition behind the current
    production raw update

        raw_delta_ln_m = 1 + (m / nu) * (A_cond^T pi - h_cond)

    without changing the solver path.
    """

    ln_mk = jnp.asarray(ln_mk)
    epsilon = jnp.asarray(epsilon, dtype=ln_mk.dtype)
    nu = jnp.exp(epsilon)
    mk = jnp.exp(ln_mk)
    driving = formula_matrix_cond.T @ jnp.asarray(pi_vector) - jnp.asarray(hvector_cond)
    factor = mk / nu
    correction = factor * driving
    raw_delta_ln_mk_current = 1.0 + correction
    condensate_stationarity_residual = mk * driving + nu
    condensate_stationarity_residual_over_nu = condensate_stationarity_residual / nu
    return {
        "nu": nu,
        "mk": mk,
        "factor": factor,
        "driving": driving,
        "correction": correction,
        "raw_delta_ln_mk_current": raw_delta_ln_mk_current,
        "condensate_stationarity_residual": condensate_stationarity_residual,
        "condensate_stationarity_residual_over_nu": condensate_stationarity_residual_over_nu,
        "raw_identity_max_abs_diff": jnp.max(
            jnp.abs(raw_delta_ln_mk_current - condensate_stationarity_residual_over_nu)
        ),
    }


def build_rgie_condensate_direction_variant(
    raw_update: Dict[str, jnp.ndarray],
    variant_name: str,
) -> Dict[str, jnp.ndarray]:
    """Build a diagnostic condensate-direction variant from the raw RGIE update block."""

    raw_delta_ln_mk_current = jnp.asarray(raw_update["raw_delta_ln_mk_current"])
    correction = jnp.asarray(raw_update["correction"])

    if variant_name == "production_clipped_current":
        delta_ln_mk = jnp.clip(raw_delta_ln_mk_current, -0.1, 0.1)
    elif variant_name == "raw_current_no_clip":
        delta_ln_mk = raw_delta_ln_mk_current
    elif variant_name == "correction_only_no_clip":
        delta_ln_mk = correction
    elif variant_name == "gas_only":
        delta_ln_mk = jnp.zeros_like(raw_delta_ln_mk_current)
    elif variant_name == "correction_only_scalar_rescale_0p1":
        max_abs_correction = jnp.max(jnp.abs(correction))
        alpha = jnp.where(max_abs_correction <= 0.1, 1.0, 0.1 / max_abs_correction)
        delta_ln_mk = alpha * correction
    else:
        raise ValueError(
            "Unknown RGIE condensate direction variant "
            f"'{variant_name}'. Expected one of "
            "('production_clipped_current', 'raw_current_no_clip', "
            "'correction_only_no_clip', 'gas_only', "
            "'correction_only_scalar_rescale_0p1')."
        )

    return {
        "variant_name": variant_name,
        "delta_ln_mk": delta_ln_mk,
        "max_abs_delta_ln_mk": jnp.max(jnp.abs(delta_ln_mk)),
    }


def build_rgie_condensate_direction_transform_variant(
    raw_delta_ln_mk: jnp.ndarray,
    variant_name: str,
) -> Dict[str, jnp.ndarray]:
    """Build a diagnostic condensate direction transform from the raw RGIE update."""

    raw = jnp.asarray(raw_delta_ln_mk)
    if variant_name == "current_component_clip_0p1":
        limit = 0.1
        delta = jnp.clip(raw, -limit, limit)
        saturated = jnp.abs(raw) > limit + 1.0e-15
    elif variant_name == "component_clip_0p5":
        limit = 0.5
        delta = jnp.clip(raw, -limit, limit)
        saturated = jnp.abs(raw) > limit + 1.0e-15
    elif variant_name == "scalar_rescale_inf_0p1":
        limit = 0.1
        max_abs = jnp.max(jnp.abs(raw))
        alpha = jnp.where(max_abs <= limit, 1.0, limit / max_abs)
        delta = alpha * raw
        saturated = None
    elif variant_name == "scalar_rescale_inf_0p5":
        limit = 0.5
        max_abs = jnp.max(jnp.abs(raw))
        alpha = jnp.where(max_abs <= limit, 1.0, limit / max_abs)
        delta = alpha * raw
        saturated = None
    elif variant_name == "raw_no_clip":
        delta = raw
        saturated = None
    else:
        raise ValueError(
            "Unknown RGIE condensate direction-transform variant "
            f"'{variant_name}'. Expected one of "
            "('current_component_clip_0p1', 'component_clip_0p5', "
            "'scalar_rescale_inf_0p1', 'scalar_rescale_inf_0p5', 'raw_no_clip')."
        )

    return {
        "variant_name": variant_name,
        "delta_ln_mk": delta,
        "cosine_raw_vs_variant": jnp.where(
            jnp.linalg.norm(raw) * jnp.linalg.norm(delta) > 1.0e-300,
            jnp.clip(
                jnp.dot(raw, delta)
                / jnp.maximum(jnp.linalg.norm(raw) * jnp.linalg.norm(delta), 1.0e-300),
                -1.0,
                1.0,
            ),
            jnp.nan,
        ),
        "max_abs_variant_delta_ln_mk": jnp.max(jnp.abs(delta)),
        "saturated_fraction": None
        if saturated is None
        else jnp.mean(saturated.astype(jnp.float64)),
    }


def compute_rgie_lambda_cap_policy(
    policy_name: str,
    *,
    lam1_gas: jnp.ndarray,
    lam1_cond: jnp.ndarray,
    lam2_cond: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Compute a diagnostic lambda cap policy from existing RGIE heuristic ceilings."""

    if policy_name == "current_full_cap":
        lam_cap = jnp.minimum(1.0, jnp.minimum(lam1_gas, jnp.minimum(lam1_cond, lam2_cond)))
    elif policy_name == "no_cond_cap":
        lam_cap = jnp.minimum(1.0, jnp.minimum(lam1_gas, lam2_cond))
    elif policy_name == "no_sk_cap":
        lam_cap = jnp.minimum(1.0, jnp.minimum(lam1_gas, lam1_cond))
    elif policy_name == "gas_only_cap":
        lam_cap = jnp.minimum(1.0, lam1_gas)
    elif policy_name == "no_heuristic_cap":
        lam_cap = jnp.asarray(1.0, dtype=jnp.asarray(lam1_gas).dtype)
    else:
        raise ValueError(
            "Unknown RGIE lambda-cap policy "
            f"'{policy_name}'. Expected one of "
            "('current_full_cap', 'no_cond_cap', 'no_sk_cap', 'gas_only_cap', 'no_heuristic_cap')."
        )

    limiter_values = jnp.asarray([1.0, lam1_gas, lam1_cond, lam2_cond], dtype=jnp.asarray(lam1_gas).dtype)
    limiter_names = ("unit", "lam1_gas", "lam1_cond", "lam2_cond")
    limiting_index = int(jnp.argmin(limiter_values))
    return {
        "policy_name": policy_name,
        "lam_cap": jnp.clip(lam_cap, 0.0, 1.0),
        "lam1_gas": lam1_gas,
        "lam1_cond": lam1_cond,
        "lam2_cond": lam2_cond,
        "production_limiting_index": jnp.asarray(limiting_index, dtype=jnp.int32),
        "production_limiting_name": limiter_names[limiting_index],
    }


def build_rgie_condensate_init_from_policy(
    epsilon: float,
    support_indices: jnp.ndarray,
    startup_policy: str,
    *,
    driving: Optional[jnp.ndarray] = None,
    m0: Optional[float] = None,
    r0: Optional[float] = None,
    top_k: Optional[int] = None,
    tiny_fallback: float = 1.0e-30,
    dtype: Optional[jnp.dtype] = None,
) -> jnp.ndarray:
    """Build a diagnostic-only RGIE condensate initialization from a startup policy.

    The returned vector is defined only on the currently supported condensates.
    This helper does not change production defaults; callers must explicitly opt
    into using it.
    """

    support_indices = jnp.asarray(support_indices)
    if support_indices.ndim != 1:
        raise ValueError("support_indices must be a one-dimensional index array.")

    if dtype is None:
        dtype = jnp.float64

    eps = jnp.asarray(epsilon, dtype=dtype)
    n_support = int(support_indices.shape[0])
    if n_support == 0:
        return jnp.zeros((0,), dtype=dtype)

    if tiny_fallback <= 0.0:
        raise ValueError("tiny_fallback must be positive.")
    fallback_ln_m0 = jnp.log(jnp.asarray(tiny_fallback, dtype=dtype))

    if startup_policy == "absolute_uniform_m0":
        if m0 is None or m0 <= 0.0:
            raise ValueError("absolute_uniform_m0 requires a positive m0.")
        return jnp.full((n_support,), jnp.log(jnp.asarray(m0, dtype=dtype)), dtype=dtype)

    target_ln_m0 = None
    if startup_policy in (
        "ratio_uniform_r0",
        "ratio_positive_driving_r0",
        "ratio_topk_positive_driving_r0",
    ):
        if r0 is None or r0 <= 0.0:
            raise ValueError(f"{startup_policy} requires a positive r0.")
        target_ln_m0 = eps + jnp.log(jnp.asarray(r0, dtype=dtype))

    if startup_policy == "ratio_uniform_r0":
        return jnp.full((n_support,), target_ln_m0, dtype=dtype)

    if startup_policy in ("ratio_positive_driving_r0", "ratio_topk_positive_driving_r0"):
        if driving is None:
            raise ValueError(f"{startup_policy} requires driving values.")
        driving = jnp.asarray(driving, dtype=dtype)
        if driving.shape != (n_support,):
            raise ValueError(
                "driving must have the same shape as the supported condensate block "
                f"(got {driving.shape}, expected {(n_support,)})."
            )
        positive = driving > 0.0

        if startup_policy == "ratio_positive_driving_r0":
            selected = positive
        else:
            if top_k is None:
                raise ValueError("ratio_topk_positive_driving_r0 requires top_k.")
            if top_k < 0:
                raise ValueError("top_k must be non-negative.")
            if top_k == 0:
                selected = jnp.zeros((n_support,), dtype=bool)
            else:
                safe_driving = jnp.where(positive, driving, -jnp.inf)
                ranked = jnp.argsort(-safe_driving)
                top_indices = ranked[: min(top_k, n_support)]
                selected = jnp.zeros((n_support,), dtype=bool).at[top_indices].set(True)
                selected = selected & positive

        return jnp.where(selected, target_ln_m0, fallback_ln_m0)

    raise ValueError(
        "Unknown startup_policy "
        f"'{startup_policy}'. Expected one of "
        "('absolute_uniform_m0', 'ratio_uniform_r0', "
        "'ratio_positive_driving_r0', 'ratio_topk_positive_driving_r0')."
    )


def summarize_rgie_inactive_driving(
    full_driving: jnp.ndarray,
    support_indices: jnp.ndarray,
    *,
    condensate_species_names: Optional[Sequence[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Summarize inactive-driving violations for a current support."""

    full_driving = jnp.asarray(full_driving, dtype=jnp.float64)
    support_indices = jnp.asarray(support_indices, dtype=jnp.int32)
    n_cond = int(full_driving.shape[0])
    support_mask = jnp.zeros((n_cond,), dtype=bool).at[support_indices].set(True)
    inactive_indices = jnp.nonzero(~support_mask, size=n_cond, fill_value=-1)[0]
    inactive_indices = inactive_indices[inactive_indices >= 0]

    if inactive_indices.shape[0] == 0:
        return {
            "max_positive_inactive_driving": 0.0,
            "inactive_positive_count": 0,
            "top_inactive_indices": [],
            "top_inactive_names": [],
            "top_inactive_driving": [],
            "top_positive_inactive_indices": [],
        }

    inactive_driving = full_driving[inactive_indices]
    positive_mask = inactive_driving > 0.0
    positive_indices = inactive_indices[positive_mask]
    positive_driving = inactive_driving[positive_mask]

    if positive_indices.shape[0] == 0:
        return {
            "max_positive_inactive_driving": 0.0,
            "inactive_positive_count": 0,
            "top_inactive_indices": [],
            "top_inactive_names": [],
            "top_inactive_driving": [],
            "top_positive_inactive_indices": [],
        }

    ranked_order = jnp.argsort(-positive_driving)
    ranked_positive = positive_indices[ranked_order]
    top_indices = ranked_positive[: min(int(top_k), int(ranked_positive.shape[0]))]
    top_driving = full_driving[top_indices]
    if condensate_species_names is None:
        top_names = [str(int(index)) for index in jax.device_get(top_indices)]
    else:
        top_names = [
            str(condensate_species_names[int(index)]) for index in jax.device_get(top_indices)
        ]

    return {
        "max_positive_inactive_driving": float(jnp.max(positive_driving)),
        "inactive_positive_count": int(positive_indices.shape[0]),
        "top_inactive_indices": [int(index) for index in jax.device_get(top_indices)],
        "top_inactive_names": top_names,
        "top_inactive_driving": [float(value) for value in jax.device_get(top_driving)],
        "top_positive_inactive_indices": [int(index) for index in jax.device_get(ranked_positive)],
    }


def build_rgie_support_candidate_indices(
    support_indices: jnp.ndarray,
    *,
    full_driving: jnp.ndarray,
    active_ln_mk: jnp.ndarray,
    active_driving: jnp.ndarray,
    mechanism_name: str,
    inactive_positive_ranked: Optional[Sequence[int]] = None,
    semismooth_add_top_k: int = 1,
    smoothed_add_top_k: int = 3,
) -> Dict[str, Any]:
    """Build diagnostic-only support candidates from active/inactive scores."""

    support_indices = jnp.asarray(support_indices, dtype=jnp.int32)
    full_driving = jnp.asarray(full_driving, dtype=jnp.float64)
    active_ln_mk = jnp.asarray(active_ln_mk, dtype=jnp.float64)
    active_driving = jnp.asarray(active_driving, dtype=jnp.float64)

    if active_ln_mk.shape != support_indices.shape or active_driving.shape != support_indices.shape:
        raise ValueError("active_ln_mk and active_driving must match support_indices shape.")

    if inactive_positive_ranked is None:
        inactive_positive_ranked = summarize_rgie_inactive_driving(
            full_driving,
            support_indices,
            top_k=full_driving.shape[0],
        )["top_positive_inactive_indices"]

    support_set = {int(index) for index in jax.device_get(support_indices)}
    ranked_add = [int(index) for index in inactive_positive_ranked if int(index) not in support_set]

    weak_active_order = jnp.argsort(-(active_driving + jnp.maximum(-active_ln_mk, 0.0)))
    weak_active_ranked = [int(index) for index in jax.device_get(support_indices[weak_active_order])]

    candidate = sorted(support_set)
    added_indices: list[int] = []
    dropped_indices: list[int] = []

    if mechanism_name == "current_support_updating_active_set":
        pass
    elif mechanism_name == "semismooth_candidate":
        added_indices = ranked_add[: max(0, int(semismooth_add_top_k))]
        candidate = sorted(set(candidate).union(added_indices))
    elif mechanism_name == "smoothed_semismooth_candidate":
        added_indices = ranked_add[: max(0, int(smoothed_add_top_k))]
        candidate = sorted(set(candidate).union(added_indices))
    elif mechanism_name == "augmented_semismooth_candidate":
        added_indices = ranked_add[:1]
        candidate = sorted(set(candidate).union(added_indices))
        if len(candidate) > 1 and weak_active_ranked:
            drop_index = weak_active_ranked[0]
            if drop_index in candidate and drop_index not in added_indices:
                candidate = [index for index in candidate if index != drop_index]
                dropped_indices = [drop_index]
    else:
        raise ValueError(
            "Unknown RGIE support candidate mechanism "
            f"'{mechanism_name}'. Expected one of "
            "('current_support_updating_active_set', 'semismooth_candidate', "
            "'smoothed_semismooth_candidate', 'augmented_semismooth_candidate')."
        )

    return {
        "mechanism_name": mechanism_name,
        "support_indices": jnp.asarray(candidate, dtype=jnp.int32),
        "added_indices": added_indices,
        "dropped_indices": dropped_indices,
        "inactive_positive_ranked": ranked_add,
        "weak_active_ranked": weak_active_ranked,
    }


def _assemble_reduced_system_terms(
    nk: jnp.ndarray,
    mk: jnp.ndarray,
    ntotk: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    bk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    sk: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Assemble the reduced condensed system once for backend experiments."""

    resn = jnp.sum(nk) - ntotk
    Qk = _A_diagn_At(nk, formula_matrix) + _A_diagn_At(sk, formula_matrix_cond)
    Angk = formula_matrix @ (gk * nk)
    ngk = jnp.dot(nk, gk)
    delta_bk_hat = b - (bk + formula_matrix_cond @ mk)
    condvec = formula_matrix_cond @ (sk * hvector_cond - mk)
    rhs = Angk + condvec + delta_bk_hat
    scalar_rhs = ngk - resn
    assemble_mat = jnp.block([[Qk, bk[:, None]], [bk[None, :], jnp.array([[resn]])]])
    assemble_vec = jnp.concatenate([rhs, jnp.array([scalar_rhs])])
    return {
        "resn": resn,
        "Qk": Qk,
        "rhs": rhs,
        "scalar_rhs": scalar_rhs,
        "assemble_mat": assemble_mat,
        "assemble_vec": assemble_vec,
    }


def _regularize_q_block(
    q_block: jnp.ndarray,
    regularization_mode: str,
    regularization_strength: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply explicit optional regularization to the Q block only."""

    dtype = q_block.dtype
    reg_used = (
        jnp.asarray(regularization_strength, dtype=dtype)
        if regularization_mode == "diag_shift"
        else jnp.asarray(0.0, dtype=dtype)
    )
    if regularization_mode == "none":
        return q_block, reg_used
    if regularization_mode != "diag_shift":
        raise ValueError(
            f"Unknown regularization_mode '{regularization_mode}'. Expected 'none' or 'diag_shift'."
        )
    eye = jnp.eye(q_block.shape[0], dtype=dtype)
    return q_block + reg_used * eye, reg_used


def solve_reduced_gibbs_iteration_equations_cond(
    nk: jnp.ndarray,
    mk: jnp.ndarray,
    ntotk: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    bk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    sk: jnp.ndarray,
    *,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Tuple[jnp.ndarray, float]:
    """
        Solve the reduced Gibbs iteration equations with condensates using the Lagrange multipliers.
        This function computes the matrix and vector to solve the system of equations
        that arises from the Gibbs energy minimization problem.

        Args:
            nk: number of species vector (n_species,) for k-th iteration.
            mk: number of condensed species vector (n_cond,) for k-th iteration.
            ntotk: Total number of species for k-th iteration.
            formula_matrix: Gas Formula matrix for stoichiometric constraints (n_elements, n_species).
            formula_matrix_cond: Condensates Formula matrix for stoichiometric constraints (n_elements, n_cond).
            b: Element abundance vector (n_elements, ).
            gk: gk vector (n_species,) for k-th iteration.
            bk: (gas) formula_matrix @ nk vector (n_elements, ).
            hvector_cond: chemical_potentials for condensates divided by RT (n_cond, )
            sk: mk^2/nu (n_cond, )

        Returns:
            Tuple containing:
                - The pi vector (nelements, ).fastchem_elements = list(gas.elements)
                    element_indices = jnp.array([fastchem_elements.index(e) for e in elements])

                - The update of the  log total number of species (delta_ln_ntot).
    """

    pi_vector, delta_ln_ntot, _metrics = _solve_reduced_gibbs_iteration_equations_cond_with_metrics(
        nk,
        mk,
        ntotk,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        bk,
        hvector_cond,
        sk,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )
    return pi_vector, delta_ln_ntot


def _solve_reduced_gibbs_iteration_equations_cond_with_metrics(
    nk: jnp.ndarray,
    mk: jnp.ndarray,
    ntotk: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    bk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    sk: jnp.ndarray,
    *,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Tuple[jnp.ndarray, float, Dict[str, jnp.ndarray]]:
    """Solve the reduced system and expose scale metrics for debugging."""

    assembled = _assemble_reduced_system_terms(
        nk,
        mk,
        ntotk,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        bk,
        hvector_cond,
        sk,
    )
    resn = assembled["resn"]
    q_block = assembled["Qk"]
    rhs = assembled["rhs"]
    scalar_rhs = assembled["scalar_rhs"]
    assemble_mat = assembled["assemble_mat"]
    assemble_vec = assembled["assemble_vec"]

    reg_q_block, reg_used = _regularize_q_block(
        q_block,
        regularization_mode,
        regularization_strength,
    )

    row_scale = jnp.maximum(jnp.max(jnp.abs(assemble_mat), axis=1, keepdims=True), 1.0)
    col_scale = jnp.maximum(jnp.max(jnp.abs(assemble_mat), axis=0, keepdims=True), 1.0)
    row_scale_flat = row_scale[:, 0]
    col_scale_flat = col_scale[0, :]

    if reduced_solver == "augmented_lu_row_scaled":
        solve_mat = assemble_mat / row_scale
        solve_vec = assemble_vec / row_scale[:, 0]
        lu, piv = lu_factor(solve_mat)
        assemble_variable = lu_solve((lu, piv), solve_vec)
        factorization_succeeded = jnp.all(jnp.isfinite(assemble_variable))
    elif reduced_solver == "augmented_lu_rowcol_scaled":
        reg_assemble_mat = jnp.block(
            [[reg_q_block, bk[:, None]], [bk[None, :], jnp.array([[resn]])]]
        )
        row_scale = jnp.maximum(jnp.max(jnp.abs(reg_assemble_mat), axis=1, keepdims=True), 1.0)
        solve_mat_row = reg_assemble_mat / row_scale
        col_scale = jnp.maximum(jnp.max(jnp.abs(solve_mat_row), axis=0, keepdims=True), 1.0)
        solve_mat = solve_mat_row / col_scale
        solve_vec = assemble_vec / row_scale[:, 0]
        lu, piv = lu_factor(solve_mat)
        scaled_solution = lu_solve((lu, piv), solve_vec)
        assemble_variable = scaled_solution / col_scale[0, :]
        factorization_succeeded = jnp.all(jnp.isfinite(assemble_variable))
        row_scale_flat = row_scale[:, 0]
        col_scale_flat = col_scale[0, :]
        assemble_mat = reg_assemble_mat
    elif reduced_solver == "schur_cholesky_reg":
        cho, lower = cho_factor(reg_q_block)
        rhs_pair = jnp.stack((rhs, bk), axis=1)
        solved_pair = cho_solve((cho, lower), rhs_pair)
        qinv_rhs = solved_pair[:, 0]
        qinv_bk = solved_pair[:, 1]
        schur = resn - jnp.vdot(bk, qinv_bk)
        schur_safe = jnp.where(
            jnp.abs(schur) < 1.0e-300,
            jnp.where(schur < 0.0, -1.0e-300, 1.0e-300),
            schur,
        )
        delta_ln_ntot = (scalar_rhs - jnp.vdot(bk, qinv_rhs)) / schur_safe
        pi_vector = qinv_rhs - qinv_bk * delta_ln_ntot
        assemble_variable = jnp.concatenate([pi_vector, jnp.array([delta_ln_ntot])])
        factorization_succeeded = jnp.all(jnp.isfinite(assemble_variable))
    else:
        raise ValueError(
            "Unknown reduced_solver "
            f"'{reduced_solver}'. Expected one of "
            "('augmented_lu_row_scaled', 'augmented_lu_rowcol_scaled', 'schur_cholesky_reg')."
        )

    metrics = {
        "reduced_solver_backend": reduced_solver,
        "reduced_factorization_succeeded": factorization_succeeded,
        "reduced_regularization_mode": regularization_mode,
        "reduced_regularization_strength": jnp.asarray(regularization_strength, dtype=q_block.dtype),
        "reduced_regularization_used": reg_used,
        "reduced_resn": resn,
        "reduced_row_scale_min": jnp.min(row_scale_flat),
        "reduced_row_scale_max": jnp.max(row_scale_flat),
        "reduced_row_scale_ratio": jnp.max(row_scale_flat)
        / jnp.maximum(jnp.min(row_scale_flat), 1.0e-300),
        "reduced_col_scale_min": jnp.min(col_scale_flat),
        "reduced_col_scale_max": jnp.max(col_scale_flat),
        "reduced_col_scale_ratio": jnp.max(col_scale_flat)
        / jnp.maximum(jnp.min(col_scale_flat), 1.0e-300),
        "reduced_mat_maxabs": jnp.max(jnp.abs(assemble_mat)),
        "reduced_vec_maxabs": jnp.max(jnp.abs(assemble_vec)),
        "reduced_qk_maxabs": jnp.max(jnp.abs(q_block)),
        "reduced_qk_diag_min": jnp.min(jnp.diag(q_block)),
        "reduced_qk_diag_max": jnp.max(jnp.diag(q_block)),
    }
    return assemble_variable[:-1], assemble_variable[-1], metrics




def _compute_residuals(
    nk: jnp.ndarray,
    mk: jnp.ndarray,
    ntotk: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    nu: float,
    An: jnp.ndarray,
    Am: jnp.ndarray,
    pi_vector: jnp.ndarray,
) -> float:

    ress = nk * (formula_matrix.T @ pi_vector - gk)
    ress_squared = jnp.dot(ress, ress)

    resc = mk * (formula_matrix_cond.T @ pi_vector - hvector_cond) + nu
    resc_squared = jnp.dot(resc, resc)

    
    deltabhat = An + Am - b
    resj_squared = jnp.dot(deltabhat, deltabhat)

    resn = jnp.sum(nk) - ntotk
    resn_squared = jnp.dot(resn, resn)

    return jnp.sqrt(ress_squared + resc_squared + resj_squared + resn_squared)


def _compute_residual_component_metrics(
    nk: jnp.ndarray,
    mk: jnp.ndarray,
    ntotk: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    nu: float,
    pi_vector: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Return barrier residual components for diagnostics."""

    ress = nk * (formula_matrix.T @ pi_vector - gk)
    resc = mk * (formula_matrix_cond.T @ pi_vector - hvector_cond) + nu
    deltabhat = formula_matrix @ nk + formula_matrix_cond @ mk - b
    resn = jnp.sum(nk) - ntotk
    residual = jnp.sqrt(
        jnp.dot(ress, ress)
        + jnp.dot(resc, resc)
        + jnp.dot(deltabhat, deltabhat)
        + jnp.dot(resn, resn)
    )
    return {
        "fresh_residual": residual,
        "element_balance_residual_norm": jnp.linalg.norm(deltabhat),
        "ntot_residual": resn,
        "gas_stationarity_residual_norm": jnp.linalg.norm(ress),
        "cond_stationarity_residual_norm": jnp.linalg.norm(resc),
    }


def _compute_normalized_gibbs_energy(
    nk: jnp.ndarray,
    mk: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
) -> jnp.ndarray:
    """Return the current physical Gibbs energy normalized by RT."""

    return jnp.sum(nk * gk) + jnp.sum(mk * hvector_cond)


def _build_trial_lambda_grid(
    heuristic_lambda: float,
    lambda_trials: Optional[Sequence[float]] = None,
    lambda_multipliers: Sequence[float] = (1.0, 0.5, 0.2, 0.1, 0.05),
    extra_lambda_trials: Sequence[float] = (1.0, 0.5, 0.2, 0.1, 0.05),
) -> list[float]:
    """Build a de-duplicated trial grid around the current heuristic lambda."""

    if lambda_trials is not None:
        values = [float(x) for x in lambda_trials]
    else:
        values = [heuristic_lambda * float(scale) for scale in lambda_multipliers]
        values.extend(float(x) for x in extra_lambda_trials)
        values.append(float(heuristic_lambda))

    clipped = []
    for value in values:
        value = max(0.0, min(1.0, float(value)))
        if all(abs(value - existing) > 1.0e-12 for existing in clipped):
            clipped.append(value)
    return sorted(clipped, reverse=True)


def _compute_gas_limiter_species_diagnostics(
    ln_nk: jnp.ndarray,
    ln_ntot: float,
    delta_ln_nk: jnp.ndarray,
    delta_ln_ntot: float,
    heuristic_lam1_gas: float,
    *,
    species_names: Optional[Sequence[str]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Decompose the gas heuristic step limiter into per-species candidates."""

    ln_nk = jnp.asarray(ln_nk)
    ln_ntot = jnp.asarray(ln_ntot)
    delta_ln_nk = jnp.asarray(delta_ln_nk)
    delta_ln_ntot = jnp.asarray(delta_ln_ntot)

    nk = jnp.exp(ln_nk)
    vmr = jnp.exp(ln_nk - ln_ntot)
    common_ntot_cap = 2.0 / jnp.maximum(5.0 * jnp.abs(delta_ln_ntot), 1.0e-300)
    abs_delta_cap = 2.0 / jnp.maximum(jnp.abs(delta_ln_nk), 1.0e-300)
    cap_candidate = jnp.minimum(common_ntot_cap, abs_delta_cap)

    ln_xk = ln_nk - ln_ntot
    denom2 = delta_ln_nk - delta_ln_ntot
    small = (ln_xk <= -18.420681) & (delta_ln_nk >= 0.0)
    safe_trace = small & (denom2 > 0.0)
    trace_candidate = jnp.where(
        safe_trace,
        (-9.2103404 - ln_xk) / denom2,
        jnp.inf,
    )
    species_candidate = jnp.minimum(cap_candidate, trace_candidate)

    ranked = jnp.argsort(species_candidate)
    limit = min(int(species_candidate.shape[0]), top_k)
    top_indices = [int(i) for i in ranked[:limit]]

    species_records = []
    for rank, idx in enumerate(top_indices):
        species_records.append(
            {
                "rank": rank,
                "species_index": idx,
                "species_name": None if species_names is None else str(species_names[idx]),
                "ln_nk": float(ln_nk[idx]),
                "nk": float(nk[idx]),
                "delta_ln_nk": float(delta_ln_nk[idx]),
                "vmr": float(vmr[idx]),
                "ln_vmr": float(ln_xk[idx]),
                "common_ntot_cap_candidate": float(common_ntot_cap),
                "abs_delta_cap_candidate": float(abs_delta_cap[idx]),
                "trace_candidate_lambda": float(trace_candidate[idx]),
                "species_candidate_lambda": float(species_candidate[idx]),
                "trace_guard_active": bool(safe_trace[idx]),
                "is_within_top_k_smallest": True,
                "matches_heuristic_lam1_gas": bool(
                    abs(float(species_candidate[idx]) - float(heuristic_lam1_gas)) <= 1.0e-12
                ),
            }
        )

    global_abs_delta_index = int(jnp.argmax(jnp.abs(delta_ln_nk)))
    trace_ranked = jnp.argsort(trace_candidate)
    top_trace_indices = [
        int(i) for i in trace_ranked[: min(int(jnp.sum(safe_trace)), top_k)]
    ]
    return {
        "lam1_gas": float(heuristic_lam1_gas),
        "common_ntot_cap_candidate": float(common_ntot_cap),
        "max_abs_delta_species_index": global_abs_delta_index,
        "max_abs_delta_species_name": None
        if species_names is None
        else str(species_names[global_abs_delta_index]),
        "max_abs_delta_ln_nk": float(jnp.max(jnp.abs(delta_ln_nk))),
        "n_trace_guard_active": int(jnp.sum(safe_trace)),
        "top_trace_guard_indices": top_trace_indices,
        "top_trace_guard_names": None
        if species_names is None
        else [str(species_names[i]) for i in top_trace_indices],
        "top_species": species_records,
    }


def _compute_frozen_condensate_gas_direction_reference(
    nk: jnp.ndarray,
    mk: jnp.ndarray,
    ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Compute a gas-only reference direction with condensates frozen into b_eff."""

    b_eff = b - formula_matrix_cond @ mk
    An = formula_matrix @ nk
    pi_ref, delta_ln_ntot_ref = solve_gibbs_iteration_equations(
        nk,
        ntot,
        formula_matrix,
        b_eff,
        gk,
        An,
    )
    delta_ln_nk_ref = formula_matrix.T @ pi_ref + delta_ln_ntot_ref - gk
    return {
        "b_eff": b_eff,
        "pi_ref": pi_ref,
        "delta_ln_ntot_ref": delta_ln_ntot_ref,
        "delta_ln_nk_ref": delta_ln_nk_ref,
    }


def _compare_gas_directions(
    ln_nk: jnp.ndarray,
    ln_ntot: float,
    delta_ln_nk_full: jnp.ndarray,
    delta_ln_ntot_full: float,
    delta_ln_nk_ref: jnp.ndarray,
    delta_ln_ntot_ref: float,
    lam1_gas_full: float,
    *,
    species_names: Optional[Sequence[str]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Compare the coupled gas direction against a frozen-condensate gas-only reference."""

    delta_ln_nk_full = jnp.asarray(delta_ln_nk_full)
    delta_ln_nk_ref = jnp.asarray(delta_ln_nk_ref)
    diff = delta_ln_nk_full - delta_ln_nk_ref
    full_norm = jnp.linalg.norm(delta_ln_nk_full)
    ref_norm = jnp.linalg.norm(delta_ln_nk_ref)
    denom = jnp.maximum(full_norm * ref_norm, 1.0e-300)
    cosine_similarity = jnp.dot(delta_ln_nk_full, delta_ln_nk_ref) / denom
    cosine_similarity = jnp.clip(cosine_similarity, -1.0, 1.0)
    angle_degrees = jnp.degrees(jnp.arccos(cosine_similarity))
    lam1_gas_ref = stepsize_cea_gas(delta_ln_nk_ref, delta_ln_ntot_ref, ln_nk, ln_ntot)

    ranked = jnp.argsort(-jnp.abs(diff))
    limit = min(int(diff.shape[0]), top_k)
    top_indices = [int(i) for i in ranked[:limit]]
    disagreement_species = []
    for rank, idx in enumerate(top_indices):
        disagreement_species.append(
            {
                "rank": rank,
                "species_index": idx,
                "species_name": None if species_names is None else str(species_names[idx]),
                "delta_ln_nk_full": float(delta_ln_nk_full[idx]),
                "delta_ln_nk_ref": float(delta_ln_nk_ref[idx]),
                "delta_ln_nk_diff": float(diff[idx]),
                "abs_delta_ln_nk_diff": float(jnp.abs(diff[idx])),
            }
        )

    return {
        "norm_full": float(full_norm),
        "norm_ref": float(ref_norm),
        "cosine_similarity": float(cosine_similarity),
        "angle_degrees": float(angle_degrees),
        "max_abs_delta_ln_nk_diff": float(jnp.max(jnp.abs(diff))),
        "delta_ln_ntot_full": float(delta_ln_ntot_full),
        "delta_ln_ntot_ref": float(delta_ln_ntot_ref),
        "abs_delta_ln_ntot_diff": float(jnp.abs(delta_ln_ntot_full - delta_ln_ntot_ref)),
        "lam1_gas_full": float(lam1_gas_full),
        "lam1_gas_ref": float(lam1_gas_ref),
        "top_direction_disagreement_species": disagreement_species,
    }


def build_rgie_gas_direction_variant(
    variant_name: str,
    *,
    delta_ln_nk_current: jnp.ndarray,
    delta_ln_ntot_current: float,
    delta_ln_nk_ref: jnp.ndarray,
    delta_ln_ntot_ref: float,
) -> Dict[str, jnp.ndarray]:
    """Build a diagnostic gas-side direction variant for RGIE."""

    delta_ln_nk_current = jnp.asarray(delta_ln_nk_current)
    delta_ln_nk_ref = jnp.asarray(delta_ln_nk_ref)
    delta_ln_ntot_current = jnp.asarray(delta_ln_ntot_current, dtype=delta_ln_nk_current.dtype)
    delta_ln_ntot_ref = jnp.asarray(delta_ln_ntot_ref, dtype=delta_ln_nk_current.dtype)

    if variant_name == "current_full_direction":
        delta_ln_nk = delta_ln_nk_current
        delta_ln_ntot = delta_ln_ntot_current
    elif variant_name == "frozen_condensate_gas_only_reference":
        delta_ln_nk = delta_ln_nk_ref
        delta_ln_ntot = delta_ln_ntot_ref
    elif variant_name == "no_common_ntot_shift":
        delta_ln_nk = delta_ln_nk_current - delta_ln_ntot_current
        delta_ln_ntot = jnp.asarray(0.0, dtype=delta_ln_nk_current.dtype)
    elif variant_name == "partial_ntot_shift_0p25":
        scale = jnp.asarray(0.25, dtype=delta_ln_nk_current.dtype)
        delta_ln_nk = (delta_ln_nk_current - delta_ln_ntot_current) + scale * delta_ln_ntot_current
        delta_ln_ntot = scale * delta_ln_ntot_current
    elif variant_name == "partial_ntot_shift_0p5":
        scale = jnp.asarray(0.5, dtype=delta_ln_nk_current.dtype)
        delta_ln_nk = (delta_ln_nk_current - delta_ln_ntot_current) + scale * delta_ln_ntot_current
        delta_ln_ntot = scale * delta_ln_ntot_current
    elif variant_name == "gas_only_with_current_condensate_block":
        delta_ln_nk = delta_ln_nk_ref
        delta_ln_ntot = delta_ln_ntot_ref
    else:
        raise ValueError(
            "Unknown RGIE gas-direction variant "
            f"'{variant_name}'. Expected one of "
            "('current_full_direction', 'frozen_condensate_gas_only_reference', "
            "'no_common_ntot_shift', 'partial_ntot_shift_0p25', "
            "'partial_ntot_shift_0p5', 'gas_only_with_current_condensate_block')."
        )

    return {
        "variant_name": variant_name,
        "delta_ln_nk": delta_ln_nk,
        "delta_ln_ntot": delta_ln_ntot,
    }


def compute_rgie_lam1_gas_ignore_trace_diagnostics(
    ln_nk: jnp.ndarray,
    ln_ntot: float,
    delta_ln_nk: jnp.ndarray,
    delta_ln_ntot: float,
    vmr_floor: float,
) -> Dict[str, Any]:
    """Diagnostic-only lam1_gas recomputation that ignores ultra-trace species."""

    ln_nk = jnp.asarray(ln_nk)
    ln_ntot = jnp.asarray(ln_ntot)
    delta_ln_nk = jnp.asarray(delta_ln_nk)
    delta_ln_ntot = jnp.asarray(delta_ln_ntot, dtype=delta_ln_nk.dtype)
    vmr = jnp.exp(ln_nk - ln_ntot)
    active = vmr >= vmr_floor
    common_ntot_cap = 2.0 / jnp.maximum(5.0 * jnp.abs(delta_ln_ntot), 1.0e-300)
    abs_delta_cap = 2.0 / jnp.maximum(jnp.abs(delta_ln_nk), 1.0e-300)
    cap_candidate = jnp.minimum(common_ntot_cap, abs_delta_cap)
    ln_xk = ln_nk - ln_ntot
    denom2 = delta_ln_nk - delta_ln_ntot
    small = (ln_xk <= -18.420681) & (delta_ln_nk >= 0.0)
    safe_trace = small & (denom2 > 0.0)
    trace_candidate = jnp.where(
        safe_trace,
        (-9.2103404 - ln_xk) / denom2,
        jnp.inf,
    )
    species_candidate = jnp.minimum(cap_candidate, trace_candidate)
    active_candidate = jnp.where(active, species_candidate, jnp.inf)
    top_index = int(jnp.argmin(species_candidate))
    active_top_index = int(jnp.argmin(active_candidate))
    lam1_gas_ignore_trace = float(jnp.min(active_candidate))
    if not math.isfinite(lam1_gas_ignore_trace):
        lam1_gas_ignore_trace = float(common_ntot_cap)
    return {
        "vmr_floor": float(vmr_floor),
        "lam1_gas_ignore_trace": lam1_gas_ignore_trace,
        "current_top_limiter_species_index": top_index,
        "current_top_limiter_active_under_floor": bool(active[top_index]),
        "active_top_limiter_species_index": active_top_index,
        "ignored_species_count": int(jnp.sum(~active)),
    }


def diagnose_iteration_lambda_trials(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    *,
    element_indices: Optional[jnp.ndarray] = None,
    lambda_trials: Optional[Sequence[float]] = None,
    lambda_multipliers: Sequence[float] = (1.0, 0.5, 0.2, 0.1, 0.05),
    extra_lambda_trials: Sequence[float] = (1.0, 0.5, 0.2, 0.1, 0.05),
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Dict[str, Any]:
    """Evaluate fresh-residual trial lambdas along one fixed current direction.

    This helper is diagnostic-only. It computes the current reduced-system
    direction once, then evaluates multiple lambda values along that same
    direction without altering the normal solver update rule.
    """

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

    hvector = hvector_func(state.temperature)
    hvector_cond = hvector_cond_func(state.temperature)
    gk = _compute_gk(
        state.temperature,
        ln_nk,
        ln_ntot,
        hvector,
        state.ln_normalized_pressure,
    )
    step_metrics = _compute_iteration_step_metrics(
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        epsilon,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )

    heuristic_lambda = float(step_metrics["lam"])
    trial_grid = _build_trial_lambda_grid(
        heuristic_lambda,
        lambda_trials=lambda_trials,
        lambda_multipliers=lambda_multipliers,
        extra_lambda_trials=extra_lambda_trials,
    )

    delta_ln_nk = jnp.asarray(step_metrics["delta_ln_nk"])
    delta_ln_mk = jnp.asarray(step_metrics["delta_ln_mk"])
    delta_ln_ntot = jnp.asarray(step_metrics["delta_ln_ntot"])

    trials = []
    for lambda_trial in trial_grid:
        lambda_trial_arr = jnp.asarray(lambda_trial, dtype=jnp.asarray(ln_ntot).dtype)
        trial_ln_nk = jnp.asarray(ln_nk) + lambda_trial_arr * delta_ln_nk
        trial_ln_mk = jnp.asarray(ln_mk) + lambda_trial_arr * delta_ln_mk
        trial_ln_ntot = jnp.asarray(ln_ntot) + lambda_trial_arr * delta_ln_ntot

        trial_nk = jnp.exp(trial_ln_nk)
        trial_mk = jnp.exp(trial_ln_mk)
        trial_ntot = jnp.exp(trial_ln_ntot)
        trial_gk = _compute_gk(
            state.temperature,
            trial_ln_nk,
            trial_ln_ntot,
            hvector,
            state.ln_normalized_pressure,
        )

        invalid_numbers_detected = bool(
            _contains_invalid_numbers(
                trial_ln_nk,
                trial_ln_mk,
                trial_ln_ntot,
                trial_nk,
                trial_mk,
                trial_ntot,
                trial_gk,
            )
        )
        sk_margin = LOG_S_MAX + epsilon - 2.0 * trial_ln_mk
        min_sk_margin = jnp.min(sk_margin)
        is_sk_feasible = bool(jnp.all(sk_margin >= 0.0))

        if invalid_numbers_detected:
            pi_vector_resid = None
            residual_metrics = {
                "fresh_residual": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
                "element_balance_residual_norm": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
                "ntot_residual": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
                "gas_stationarity_residual_norm": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
                "cond_stationarity_residual_norm": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
            }
            normalized_gibbs = jnp.asarray(jnp.nan, dtype=trial_ntot.dtype)
        else:
            pi_vector_resid = _recompute_pi_for_residual(
                trial_nk,
                trial_mk,
                trial_ntot,
                formula_matrix,
                formula_matrix_cond,
                b,
                trial_gk,
                hvector_cond,
                epsilon,
                reduced_solver=reduced_solver,
                regularization_mode=regularization_mode,
                regularization_strength=regularization_strength,
            )
            residual_metrics = _compute_residual_component_metrics(
                trial_nk,
                trial_mk,
                trial_ntot,
                formula_matrix,
                formula_matrix_cond,
                b,
                trial_gk,
                hvector_cond,
                jnp.exp(epsilon),
                pi_vector_resid,
            )
            normalized_gibbs = _compute_normalized_gibbs_energy(
                trial_nk,
                trial_mk,
                trial_gk,
                hvector_cond,
            )

        if abs(lambda_trial - heuristic_lambda) <= 1.0e-12:
            relative_to_heuristic = "equal"
        elif lambda_trial > heuristic_lambda:
            relative_to_heuristic = "above"
        else:
            relative_to_heuristic = "below"

        trials.append(
            {
                "lambda_trial": float(lambda_trial),
                "relative_to_heuristic": relative_to_heuristic,
                "is_above_heuristic_lambda": bool(lambda_trial > heuristic_lambda + 1.0e-12),
                "is_below_heuristic_lambda": bool(lambda_trial < heuristic_lambda - 1.0e-12),
                "fresh_residual": float(residual_metrics["fresh_residual"]),
                "element_balance_residual_norm": float(
                    residual_metrics["element_balance_residual_norm"]
                ),
                "ntot_residual": float(residual_metrics["ntot_residual"]),
                "gas_stationarity_residual_norm": float(
                    residual_metrics["gas_stationarity_residual_norm"]
                ),
                "cond_stationarity_residual_norm": float(
                    residual_metrics["cond_stationarity_residual_norm"]
                ),
                "sk_feasibility_margin_min": float(min_sk_margin),
                "sk_feasibility_margin_worst": float(min_sk_margin),
                "sk_feasible": is_sk_feasible,
                "n_sk_infeasible": int(jnp.sum(sk_margin < 0.0)),
                "invalid_numbers_detected": invalid_numbers_detected,
                "all_finite": not invalid_numbers_detected,
                "max_abs_delta_ln_nk": float(jnp.max(jnp.abs(lambda_trial_arr * delta_ln_nk))),
                "max_abs_delta_ln_mk": float(jnp.max(jnp.abs(lambda_trial_arr * delta_ln_mk))),
                "abs_delta_ln_ntot": float(jnp.abs(lambda_trial_arr * delta_ln_ntot)),
                "normalized_gibbs_energy": float(normalized_gibbs),
            }
        )

    return {
        "epsilon": float(epsilon),
        "heuristic_lambda": heuristic_lambda,
        "trial_lambdas": [trial["lambda_trial"] for trial in trials],
        "step_metrics": {
            "lam": float(step_metrics["lam"]),
            "lam1_gas": float(step_metrics["lam1_gas"]),
            "lam1_cond": float(step_metrics["lam1_cond"]),
            "lam2_cond": float(step_metrics["lam2_cond"]),
            "limiting_index": int(step_metrics["limiting_index"]),
            "delta_ln_ntot": float(step_metrics["delta_ln_ntot"]),
            "max_abs_delta_ln_nk": float(step_metrics["max_abs_delta_ln_nk"]),
            "max_abs_raw_delta_ln_mk": float(step_metrics["max_abs_raw_delta_ln_mk"]),
            "max_abs_clipped_delta_ln_mk": float(step_metrics["max_abs_clipped_delta_ln_mk"]),
        },
        "trials": trials,
    }


def diagnose_gas_step_limiter_and_direction(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    *,
    element_indices: Optional[jnp.ndarray] = None,
    gas_species_names: Optional[Sequence[str]] = None,
    top_k: int = 10,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Dict[str, Any]:
    """Diagnostic-only gas limiter decomposition and frozen-condensate direction comparison."""

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

    hvector = hvector_func(state.temperature)
    hvector_cond = hvector_cond_func(state.temperature)
    gk = _compute_gk(
        state.temperature,
        ln_nk,
        ln_ntot,
        hvector,
        state.ln_normalized_pressure,
    )
    step_metrics = _compute_iteration_step_metrics(
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        epsilon,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )

    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntot = jnp.exp(ln_ntot)
    gas_limiter = _compute_gas_limiter_species_diagnostics(
        ln_nk,
        ln_ntot,
        step_metrics["delta_ln_nk"],
        step_metrics["delta_ln_ntot"],
        step_metrics["lam1_gas"],
        species_names=gas_species_names,
        top_k=top_k,
    )
    gas_ref = _compute_frozen_condensate_gas_direction_reference(
        nk,
        mk,
        ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
    )
    direction_comparison = _compare_gas_directions(
        ln_nk,
        ln_ntot,
        step_metrics["delta_ln_nk"],
        step_metrics["delta_ln_ntot"],
        gas_ref["delta_ln_nk_ref"],
        gas_ref["delta_ln_ntot_ref"],
        step_metrics["lam1_gas"],
        species_names=gas_species_names,
        top_k=top_k,
    )
    return {
        "epsilon": float(epsilon),
        "step_metrics": {
            "lam": float(step_metrics["lam"]),
            "lam1_gas": float(step_metrics["lam1_gas"]),
            "lam1_cond": float(step_metrics["lam1_cond"]),
            "lam2_cond": float(step_metrics["lam2_cond"]),
            "limiting_index": int(step_metrics["limiting_index"]),
        },
        "gas_limiter": gas_limiter,
        "gas_direction_reference": {
            "b_eff_norm": float(jnp.linalg.norm(gas_ref["b_eff"])),
            "delta_ln_ntot_ref": float(gas_ref["delta_ln_ntot_ref"]),
        },
        "direction_comparison": direction_comparison,
    }


def diagnose_reduced_solver_backend_experiments(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    *,
    element_indices: Optional[jnp.ndarray] = None,
    backend_configs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compare one-step reduced-solver backend experiments from the same state."""

    if backend_configs is None:
        backend_configs = (
            {
                "reduced_solver": DEFAULT_REDUCED_SOLVER,
                "regularization_mode": DEFAULT_REGULARIZATION_MODE,
                "regularization_strength": DEFAULT_REGULARIZATION_STRENGTH,
            },
            {
                "reduced_solver": "augmented_lu_rowcol_scaled",
                "regularization_mode": DEFAULT_REGULARIZATION_MODE,
                "regularization_strength": DEFAULT_REGULARIZATION_STRENGTH,
            },
            {
                "reduced_solver": "schur_cholesky_reg",
                "regularization_mode": "diag_shift",
                "regularization_strength": 1.0e-12,
            },
        )

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

    hvector = hvector_func(state.temperature)
    hvector_cond = hvector_cond_func(state.temperature)
    gk = _compute_gk(
        state.temperature,
        ln_nk,
        ln_ntot,
        hvector,
        state.ln_normalized_pressure,
    )
    An = formula_matrix @ jnp.exp(ln_nk)
    Am = formula_matrix_cond @ jnp.exp(ln_mk)

    comparisons = []
    baseline_delta_ln_nk = None
    baseline_backend = None
    for config in backend_configs:
        reduced_solver = config.get("reduced_solver", DEFAULT_REDUCED_SOLVER)
        regularization_mode = config.get("regularization_mode", DEFAULT_REGULARIZATION_MODE)
        regularization_strength = config.get(
            "regularization_strength", DEFAULT_REGULARIZATION_STRENGTH
        )
        start = perf_counter()
        (
            _ln_nk_new,
            _ln_mk_new,
            _ln_ntot_new,
            _gk_new,
            _An_new,
            _Am_new,
            residual,
            _lam,
            metrics,
        ) = _update_all_with_metrics(
            ln_nk,
            ln_mk,
            ln_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            state.temperature,
            state.ln_normalized_pressure,
            hvector,
            hvector_cond,
            gk,
            An,
            Am,
            jnp.inf,
            epsilon,
            iter_count=0,
            debug_nan=False,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
        )
        runtime_seconds = perf_counter() - start
        delta_ln_nk = jnp.asarray(metrics["delta_ln_nk"])
        if baseline_delta_ln_nk is None:
            baseline_delta_ln_nk = delta_ln_nk
            baseline_backend = reduced_solver
            cosine_vs_baseline = 1.0
            max_abs_diff_vs_baseline = 0.0
        else:
            denom = jnp.maximum(
                jnp.linalg.norm(delta_ln_nk) * jnp.linalg.norm(baseline_delta_ln_nk),
                1.0e-300,
            )
            cosine_vs_baseline = float(
                jnp.clip(jnp.dot(delta_ln_nk, baseline_delta_ln_nk) / denom, -1.0, 1.0)
            )
            max_abs_diff_vs_baseline = float(
                jnp.max(jnp.abs(delta_ln_nk - baseline_delta_ln_nk))
            )
        comparisons.append(
            {
                "backend": reduced_solver,
                "regularization_mode": regularization_mode,
                "regularization_strength": float(regularization_strength),
                "factorization_succeeded": bool(metrics["reduced_factorization_succeeded"]),
                "regularization_used": float(metrics["reduced_regularization_used"]),
                "reduced_row_scale_ratio": float(metrics["reduced_row_scale_ratio"]),
                "reduced_col_scale_ratio": float(metrics["reduced_col_scale_ratio"]),
                "pi_norm": float(metrics["pi_norm"]),
                "delta_ln_ntot": float(metrics["delta_ln_ntot"]),
                "lam1_gas": float(metrics["lam1_gas"]),
                "lam1_cond": float(metrics["lam1_cond"]),
                "lam2_cond": float(metrics["lam2_cond"]),
                "lam": float(metrics["lam"]),
                "fresh_post_update_residual": float(residual),
                "direction_norm": float(jnp.linalg.norm(delta_ln_nk)),
                "runtime_seconds": runtime_seconds,
                "invalid_numbers_detected": bool(
                    _contains_invalid_numbers(
                        metrics["pi_vector"],
                        metrics["delta_ln_nk"],
                        metrics["delta_ln_mk"],
                        metrics["delta_ln_ntot"],
                        residual,
                    )
                ),
                "cosine_similarity_vs_baseline": cosine_vs_baseline,
                "max_abs_delta_ln_nk_diff_vs_baseline": max_abs_diff_vs_baseline,
                "baseline_backend": baseline_backend,
            }
        )

    return {
        "epsilon": float(epsilon),
        "baseline_backend": baseline_backend,
        "comparisons": comparisons,
    }


def _evaluate_direction_with_existing_update_rule(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    hvector: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
    delta_ln_nk: jnp.ndarray,
    raw_delta_ln_mk: jnp.ndarray,
    delta_ln_ntot: float,
    *,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Dict[str, Any]:
    """Apply the current clipping, limiter, and fresh-residual evaluation to a direction."""

    clipped_delta_ln_mk = jnp.clip(raw_delta_ln_mk, -0.1, 0.1)
    lam1_gas = stepsize_cea_gas(delta_ln_nk, delta_ln_ntot, ln_nk, ln_ntot)
    lam1_cond = stepsize_cond_heurstic(clipped_delta_ln_mk)
    lam2_cond = stepsize_sk(clipped_delta_ln_mk, ln_mk, epsilon)
    lam = jnp.minimum(1.0, jnp.minimum(lam1_gas, jnp.minimum(lam1_cond, lam2_cond)))
    lam = jnp.clip(lam, 0.0, 1.0)

    trial_ln_nk = jnp.asarray(ln_nk) + lam * jnp.asarray(delta_ln_nk)
    trial_ln_mk = jnp.asarray(ln_mk) + lam * jnp.asarray(clipped_delta_ln_mk)
    trial_ln_ntot = jnp.asarray(ln_ntot) + lam * jnp.asarray(delta_ln_ntot)
    trial_nk = jnp.exp(trial_ln_nk)
    trial_mk = jnp.exp(trial_ln_mk)
    trial_ntot = jnp.exp(trial_ln_ntot)
    trial_gk = _compute_gk(
        state.temperature,
        trial_ln_nk,
        trial_ln_ntot,
        hvector,
        state.ln_normalized_pressure,
    )
    invalid_numbers_detected = bool(
        _contains_invalid_numbers(
            trial_ln_nk,
            trial_ln_mk,
            trial_ln_ntot,
            trial_nk,
            trial_mk,
            trial_ntot,
            trial_gk,
            delta_ln_nk,
            raw_delta_ln_mk,
            clipped_delta_ln_mk,
            delta_ln_ntot,
        )
    )
    if invalid_numbers_detected:
        pi_vector_resid = None
        residual_metrics = {
            "fresh_residual": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
            "element_balance_residual_norm": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
            "ntot_residual": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
            "gas_stationarity_residual_norm": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
            "cond_stationarity_residual_norm": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
        }
    else:
        pi_vector_resid = _recompute_pi_for_residual(
            trial_nk,
            trial_mk,
            trial_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            trial_gk,
            hvector_cond,
            epsilon,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
        )
        residual_metrics = _compute_residual_component_metrics(
            trial_nk,
            trial_mk,
            trial_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            trial_gk,
            hvector_cond,
            jnp.exp(epsilon),
            pi_vector_resid,
        )

    return {
        "raw_direction_norm": float(
            jnp.linalg.norm(
                jnp.concatenate(
                    [
                        jnp.ravel(jnp.asarray(delta_ln_nk)),
                        jnp.ravel(jnp.asarray(raw_delta_ln_mk)),
                        jnp.atleast_1d(jnp.asarray(delta_ln_ntot)),
                    ]
                )
            )
        ),
        "clipped_direction_norm": float(
            jnp.linalg.norm(
                jnp.concatenate(
                    [
                        jnp.ravel(jnp.asarray(delta_ln_nk)),
                        jnp.ravel(jnp.asarray(clipped_delta_ln_mk)),
                        jnp.atleast_1d(jnp.asarray(delta_ln_ntot)),
                    ]
                )
            )
        ),
        "gas_direction_norm": float(jnp.linalg.norm(jnp.asarray(delta_ln_nk))),
        "raw_cond_direction_norm": float(jnp.linalg.norm(jnp.asarray(raw_delta_ln_mk))),
        "clipped_cond_direction_norm": float(jnp.linalg.norm(jnp.asarray(clipped_delta_ln_mk))),
        "delta_ln_ntot": float(delta_ln_ntot),
        "lam1_gas": float(lam1_gas),
        "lam1_cond": float(lam1_cond),
        "lam2_cond": float(lam2_cond),
        "lam": float(lam),
        "fresh_post_update_residual": float(residual_metrics["fresh_residual"]),
        "element_balance_residual_norm": float(
            residual_metrics["element_balance_residual_norm"]
        ),
        "ntot_residual": float(residual_metrics["ntot_residual"]),
        "gas_stationarity_residual_norm": float(
            residual_metrics["gas_stationarity_residual_norm"]
        ),
        "cond_stationarity_residual_norm": float(
            residual_metrics["cond_stationarity_residual_norm"]
        ),
        "invalid_numbers_detected": invalid_numbers_detected,
        "all_finite": not invalid_numbers_detected,
        "max_abs_delta_ln_nk": float(jnp.max(jnp.abs(delta_ln_nk))),
        "max_abs_raw_delta_ln_mk": float(jnp.max(jnp.abs(raw_delta_ln_mk))),
        "max_abs_clipped_delta_ln_mk": float(jnp.max(jnp.abs(clipped_delta_ln_mk))),
    }


def _solve_full_pipm_gie_direction_with_metrics(
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
) -> Dict[str, Any]:
    """Solve the full PIPM GIE linearization directly for diagnostic comparison."""

    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntot = jnp.exp(ln_ntot)
    bk = formula_matrix @ nk
    nu = jnp.exp(epsilon)
    start = perf_counter()
    delta_ln_nk, delta_ln_mk, pi_vector, delta_ln_ntot = (
        solve_full_gibbs_iteration_equations_cond(
            nk,
            mk,
            ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            gk,
            hvector_cond,
            bk,
            nu,
        )
    )
    runtime_seconds = perf_counter() - start
    factorization_succeeded = bool(
        jnp.all(
            jnp.isfinite(
                jnp.concatenate(
                    [
                        jnp.ravel(jnp.asarray(delta_ln_nk)),
                        jnp.ravel(jnp.asarray(delta_ln_mk)),
                        jnp.ravel(jnp.asarray(pi_vector)),
                        jnp.atleast_1d(jnp.asarray(delta_ln_ntot)),
                    ]
                )
            )
        )
    )
    return {
        "delta_ln_nk": delta_ln_nk,
        "raw_delta_ln_mk": delta_ln_mk,
        "pi_vector": pi_vector,
        "delta_ln_ntot": delta_ln_ntot,
        "pi_norm": float(jnp.linalg.norm(jnp.asarray(pi_vector))),
        "runtime_seconds": runtime_seconds,
        "factorization_succeeded": factorization_succeeded,
    }


def _solve_full_pdipm_gie_direction_with_metrics(
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    rho: Optional[jnp.ndarray],
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
) -> Dict[str, Any]:
    """Solve the full PDIPM GIE linearization directly for one diagnostic state."""

    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntot = jnp.exp(ln_ntot)
    rho = epsilon - ln_mk if rho is None else jnp.asarray(rho)
    eta = jnp.exp(rho)

    nspecies = int(nk.shape[0])
    ncond = int(mk.shape[0])
    nelement = int(b.shape[0])

    y_gas = formula_matrix * nk
    y_cond = formula_matrix_cond * mk
    u = jnp.ones((nspecies, 1), dtype=nk.dtype)
    zeros_nm = jnp.zeros((nspecies, ncond), dtype=nk.dtype)
    zeros_en = jnp.zeros((nelement, nspecies), dtype=nk.dtype)
    zeros_eb = jnp.zeros((nelement, nelement), dtype=nk.dtype)
    zeros_em = jnp.zeros((nelement, ncond), dtype=nk.dtype)
    zeros_cn = jnp.zeros((ncond, nspecies), dtype=nk.dtype)
    zeros_cb = jnp.zeros((ncond, nelement), dtype=nk.dtype)
    zeros_c1 = jnp.zeros((ncond, 1), dtype=nk.dtype)
    zeros_1m = jnp.zeros((1, ncond), dtype=nk.dtype)
    zeros_1b = jnp.zeros((1, nelement), dtype=nk.dtype)

    row_fn = jnp.block(
        [
            jnp.eye(nspecies, dtype=nk.dtype),
            zeros_nm,
            -formula_matrix.T,
            jnp.zeros((nspecies, ncond), dtype=nk.dtype),
            -u,
        ]
    )
    row_fm = jnp.block(
        [
            zeros_cn,
            jnp.zeros((ncond, ncond), dtype=nk.dtype),
            formula_matrix_cond.T,
            jnp.diag(eta),
            zeros_c1,
        ]
    )
    row_flambda = jnp.block(
        [
            y_gas,
            y_cond,
            zeros_eb,
            zeros_em,
            jnp.zeros((nelement, 1), dtype=nk.dtype),
        ]
    )
    row_fc = jnp.block(
        [
            zeros_cn,
            jnp.eye(ncond, dtype=nk.dtype),
            zeros_cb,
            jnp.eye(ncond, dtype=nk.dtype),
            zeros_c1,
        ]
    )
    row_ftot = jnp.block(
        [
            nk[jnp.newaxis, :],
            zeros_1m,
            zeros_1b,
            zeros_1m,
            jnp.array([[-ntot]], dtype=nk.dtype),
        ]
    )
    assemble_mat = jnp.block(
        [
            [row_fn],
            [row_fm],
            [row_flambda],
            [row_fc],
            [row_ftot],
        ]
    )

    rhs_fn = -gk
    rhs_fm = hvector_cond - eta
    rhs_flambda = b - formula_matrix @ nk - formula_matrix_cond @ mk
    rhs_fc = jnp.full_like(mk, epsilon) - ln_mk - rho
    rhs_ftot = jnp.array([ntot - jnp.sum(nk)], dtype=nk.dtype)
    assemble_vec = jnp.concatenate(
        [rhs_fn, rhs_fm, rhs_flambda, rhs_fc, rhs_ftot]
    )

    start = perf_counter()
    assemble_variable = jnp.linalg.solve(assemble_mat, assemble_vec)
    runtime_seconds = perf_counter() - start

    delta_ln_nk = assemble_variable[:nspecies]
    delta_ln_mk = assemble_variable[nspecies : nspecies + ncond]
    pi_vector = assemble_variable[nspecies + ncond : nspecies + ncond + nelement]
    delta_rho = assemble_variable[
        nspecies + ncond + nelement : nspecies + 2 * ncond + nelement
    ]
    delta_ln_ntot = assemble_variable[-1]

    factorization_succeeded = bool(
        jnp.all(jnp.isfinite(assemble_variable))
    )
    return {
        "rho": rho,
        "eta": eta,
        "delta_ln_nk": delta_ln_nk,
        "raw_delta_ln_mk": delta_ln_mk,
        "pi_vector": pi_vector,
        "delta_rho": delta_rho,
        "delta_ln_ntot": delta_ln_ntot,
        "pi_norm": float(jnp.linalg.norm(jnp.asarray(pi_vector))),
        "runtime_seconds": runtime_seconds,
        "factorization_succeeded": factorization_succeeded,
    }


def _pdipm_total_residual_norm(residual_components: Dict[str, float]) -> float:
    """Combine PDIPM residual components into a single Euclidean norm."""

    return float(
        jnp.sqrt(
            residual_components["Fn_norm"] ** 2
            + residual_components["Fm_norm"] ** 2
            + residual_components["Flambda_norm"] ** 2
            + residual_components["Fc_norm"] ** 2
            + residual_components["Ftot_abs"] ** 2
        )
    )


def _compute_pdipm_barrier_residual_components(
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_eta: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    pi_vector: jnp.ndarray,
    epsilon: float,
) -> Dict[str, float]:
    """Compute nonlinear PDIPM residual component norms at a state."""

    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntot = jnp.exp(ln_ntot)
    eta = jnp.exp(ln_eta)
    fn = nk * (formula_matrix.T @ pi_vector - gk)
    fm = eta + formula_matrix_cond.T @ pi_vector - hvector_cond
    flambda = formula_matrix @ nk + formula_matrix_cond @ mk - b
    fc = ln_mk + ln_eta - epsilon
    ftot = jnp.sum(nk) - ntot
    return {
        "Fn_norm": float(jnp.linalg.norm(fn)),
        "Fm_norm": float(jnp.linalg.norm(fm)),
        "Flambda_norm": float(jnp.linalg.norm(flambda)),
        "Fc_norm": float(jnp.linalg.norm(fc)),
        "Ftot_abs": float(jnp.abs(ftot)),
    }


def _evaluate_pdipm_state(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    rho: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    hvector: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
    *,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Dict[str, Any]:
    """Evaluate PDIPM and primal residual diagnostics at a given state."""

    gk = _compute_gk(
        state.temperature,
        ln_nk,
        ln_ntot,
        hvector,
        state.ln_normalized_pressure,
    )
    pdipm_metrics = _solve_full_pdipm_gie_direction_with_metrics(
        ln_nk,
        ln_mk,
        rho,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        epsilon,
    )
    pdipm_residuals = _compute_pdipm_barrier_residual_components(
        ln_nk,
        ln_mk,
        rho,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        pdipm_metrics["pi_vector"],
        epsilon,
    )
    pdipm_total_residual = _pdipm_total_residual_norm(pdipm_residuals)

    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntot = jnp.exp(ln_ntot)
    invalid_numbers_detected = bool(
        _contains_invalid_numbers(ln_nk, ln_mk, rho, ln_ntot, nk, mk, ntot, gk)
    )
    if invalid_numbers_detected:
        primal_residual_metrics = {
            "fresh_residual": jnp.asarray(jnp.nan, dtype=ntot.dtype),
            "element_balance_residual_norm": jnp.asarray(jnp.nan, dtype=ntot.dtype),
            "ntot_residual": jnp.asarray(jnp.nan, dtype=ntot.dtype),
            "gas_stationarity_residual_norm": jnp.asarray(jnp.nan, dtype=ntot.dtype),
            "cond_stationarity_residual_norm": jnp.asarray(jnp.nan, dtype=ntot.dtype),
        }
    else:
        pi_vector_resid = _recompute_pi_for_residual(
            nk,
            mk,
            ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            gk,
            hvector_cond,
            epsilon,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
        )
        primal_residual_metrics = _compute_residual_component_metrics(
            nk,
            mk,
            ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            gk,
            hvector_cond,
            jnp.exp(epsilon),
            pi_vector_resid,
        )

    return {
        "gk": gk,
        "pdipm_metrics": pdipm_metrics,
        "pdipm_residuals": pdipm_residuals,
        "pdipm_total_residual": pdipm_total_residual,
        "primal_residual_metrics": primal_residual_metrics,
        "invalid_numbers_detected": invalid_numbers_detected,
    }


def _evaluate_direction_lambda_grid(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    hvector: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
    delta_ln_nk: jnp.ndarray,
    raw_delta_ln_mk: jnp.ndarray,
    delta_ln_ntot: float,
    *,
    delta_aux: Optional[jnp.ndarray] = None,
    aux_name: str = "delta_aux",
    lambda_trials: Sequence[float],
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> list[Dict[str, Any]]:
    """Evaluate a fixed direction on a shared lambda grid."""

    clipped_delta_ln_mk = jnp.clip(raw_delta_ln_mk, -0.1, 0.1)
    lam1_gas = stepsize_cea_gas(delta_ln_nk, delta_ln_ntot, ln_nk, ln_ntot)
    lam1_cond = stepsize_cond_heurstic(clipped_delta_ln_mk)
    lam2_cond = stepsize_sk(clipped_delta_ln_mk, ln_mk, epsilon)
    trials = []
    for lambda_trial in lambda_trials:
        lam = jnp.asarray(lambda_trial, dtype=jnp.asarray(ln_ntot).dtype)
        trial_ln_nk = jnp.asarray(ln_nk) + lam * jnp.asarray(delta_ln_nk)
        trial_ln_mk = jnp.asarray(ln_mk) + lam * jnp.asarray(clipped_delta_ln_mk)
        trial_ln_ntot = jnp.asarray(ln_ntot) + lam * jnp.asarray(delta_ln_ntot)
        trial_aux = None if delta_aux is None else lam * jnp.asarray(delta_aux)
        trial_nk = jnp.exp(trial_ln_nk)
        trial_mk = jnp.exp(trial_ln_mk)
        trial_ntot = jnp.exp(trial_ln_ntot)
        trial_gk = _compute_gk(
            state.temperature,
            trial_ln_nk,
            trial_ln_ntot,
            hvector,
            state.ln_normalized_pressure,
        )
        invalid_numbers_detected = bool(
            _contains_invalid_numbers(
                trial_ln_nk,
                trial_ln_mk,
                trial_ln_ntot,
                trial_nk,
                trial_mk,
                trial_ntot,
                trial_gk,
                delta_ln_nk,
                raw_delta_ln_mk,
                clipped_delta_ln_mk,
                delta_ln_ntot,
                trial_aux if trial_aux is not None else jnp.asarray(0.0),
            )
        )
        if invalid_numbers_detected:
            residual_metrics = {
                "fresh_residual": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
                "element_balance_residual_norm": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
                "ntot_residual": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
                "gas_stationarity_residual_norm": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
                "cond_stationarity_residual_norm": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
            }
        else:
            pi_vector_resid = _recompute_pi_for_residual(
                trial_nk,
                trial_mk,
                trial_ntot,
                formula_matrix,
                formula_matrix_cond,
                b,
                trial_gk,
                hvector_cond,
                epsilon,
                reduced_solver=reduced_solver,
                regularization_mode=regularization_mode,
                regularization_strength=regularization_strength,
            )
            residual_metrics = _compute_residual_component_metrics(
                trial_nk,
                trial_mk,
                trial_ntot,
                formula_matrix,
                formula_matrix_cond,
                b,
                trial_gk,
                hvector_cond,
                jnp.exp(epsilon),
                pi_vector_resid,
            )
        record = {
            "lambda_trial": float(lambda_trial),
            "fresh_post_update_residual": float(residual_metrics["fresh_residual"]),
            "element_balance_residual_norm": float(
                residual_metrics["element_balance_residual_norm"]
            ),
            "ntot_residual": float(residual_metrics["ntot_residual"]),
            "invalid_numbers_detected": invalid_numbers_detected,
            "all_finite": not invalid_numbers_detected,
            "max_abs_delta_ln_nk": float(jnp.max(jnp.abs(lam * jnp.asarray(delta_ln_nk)))),
            "max_abs_delta_ln_mk": float(
                jnp.max(jnp.abs(lam * jnp.asarray(clipped_delta_ln_mk)))
            ),
            "max_abs_delta_ln_ntot": float(jnp.abs(lam * jnp.asarray(delta_ln_ntot))),
            "lam1_gas": float(lam1_gas),
            "lam1_cond": float(lam1_cond),
            "lam2_cond": float(lam2_cond),
        }
        if delta_aux is not None:
            record[f"max_abs_{aux_name}"] = float(jnp.max(jnp.abs(trial_aux)))
        trials.append(record)
    return trials


def trace_pdipm_fixed_epsilon_trajectory(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    *,
    rho_offset: float = 0.0,
    max_iter: int = 10,
    min_lambda: float = 1.0e-6,
    backtrack_factor: float = 0.5,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
    element_indices: Optional[jnp.ndarray] = None,
) -> Dict[str, Any]:
    """Diagnostic-only fixed-epsilon full-PDIPM trajectory with independent rho."""

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

    hvector = hvector_func(state.temperature)
    hvector_cond = hvector_cond_func(state.temperature)
    rho = epsilon - jnp.asarray(ln_mk) + jnp.asarray(rho_offset, dtype=jnp.asarray(ln_mk).dtype)
    ln_nk_state = jnp.asarray(ln_nk)
    ln_mk_state = jnp.asarray(ln_mk)
    ln_ntot_state = jnp.asarray(ln_ntot)
    history = []
    start = perf_counter()

    for iter_count in range(max_iter):
        eval_current = _evaluate_pdipm_state(
            state,
            ln_nk_state,
            ln_mk_state,
            rho,
            ln_ntot_state,
            formula_matrix,
            formula_matrix_cond,
            b,
            hvector,
            hvector_cond,
            epsilon,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
        )
        current_total = eval_current["pdipm_total_residual"]
        current_primal = float(eval_current["primal_residual_metrics"]["fresh_residual"])
        metrics = eval_current["pdipm_metrics"]

        trial_lambda = 1.0
        accepted = False
        accepted_eval = None
        while trial_lambda >= min_lambda:
            lam = jnp.asarray(trial_lambda, dtype=ln_nk_state.dtype)
            trial_ln_nk = ln_nk_state + lam * jnp.asarray(metrics["delta_ln_nk"])
            trial_ln_mk = ln_mk_state + lam * jnp.asarray(metrics["raw_delta_ln_mk"])
            trial_rho = rho + lam * jnp.asarray(metrics["delta_rho"])
            trial_ln_ntot = ln_ntot_state + lam * jnp.asarray(metrics["delta_ln_ntot"])
            accepted_eval = _evaluate_pdipm_state(
                state,
                trial_ln_nk,
                trial_ln_mk,
                trial_rho,
                trial_ln_ntot,
                formula_matrix,
                formula_matrix_cond,
                b,
                hvector,
                hvector_cond,
                epsilon,
                reduced_solver=reduced_solver,
                regularization_mode=regularization_mode,
                regularization_strength=regularization_strength,
            )
            if (
                not accepted_eval["invalid_numbers_detected"]
                and accepted_eval["pdipm_total_residual"] <= current_total
            ):
                accepted = True
                break
            trial_lambda *= backtrack_factor

        record = {
            "iter": iter_count,
            "pdipm_total_residual": current_total,
            "primal_fresh_residual": current_primal,
            "Fn_norm": eval_current["pdipm_residuals"]["Fn_norm"],
            "Fm_norm": eval_current["pdipm_residuals"]["Fm_norm"],
            "Flambda_norm": eval_current["pdipm_residuals"]["Flambda_norm"],
            "Fc_norm": eval_current["pdipm_residuals"]["Fc_norm"],
            "Ftot_abs": eval_current["pdipm_residuals"]["Ftot_abs"],
            "delta_ln_ntot": float(metrics["delta_ln_ntot"]),
            "pi_norm": float(metrics["pi_norm"]),
            "max_abs_delta_ln_nk": float(jnp.max(jnp.abs(metrics["delta_ln_nk"]))),
            "max_abs_delta_ln_mk": float(jnp.max(jnp.abs(metrics["raw_delta_ln_mk"]))),
            "max_abs_delta_rho": float(jnp.max(jnp.abs(metrics["delta_rho"]))),
            "chosen_lambda": float(trial_lambda if accepted else 0.0),
            "accepted": accepted,
            "accepted_total_residual": float(
                accepted_eval["pdipm_total_residual"] if accepted_eval is not None else jnp.nan
            ),
            "accepted_primal_fresh_residual": float(
                accepted_eval["primal_residual_metrics"]["fresh_residual"]
                if accepted_eval is not None
                else jnp.nan
            ),
            "invalid_numbers_detected": bool(
                eval_current["invalid_numbers_detected"]
                or (accepted_eval["invalid_numbers_detected"] if accepted_eval is not None else False)
            ),
        }
        history.append(record)
        if not accepted:
            break
        lam = jnp.asarray(trial_lambda, dtype=ln_nk_state.dtype)
        ln_nk_state = ln_nk_state + lam * jnp.asarray(metrics["delta_ln_nk"])
        ln_mk_state = ln_mk_state + lam * jnp.asarray(metrics["raw_delta_ln_mk"])
        rho = rho + lam * jnp.asarray(metrics["delta_rho"])
        ln_ntot_state = ln_ntot_state + lam * jnp.asarray(metrics["delta_ln_ntot"])

    runtime_seconds = perf_counter() - start
    total_residuals = [rec["pdipm_total_residual"] for rec in history]
    monotonically_decreasing = all(
        total_residuals[i + 1] <= total_residuals[i] + 1.0e-12
        for i in range(len(total_residuals) - 1)
    )
    return {
        "rho_offset": float(rho_offset),
        "rho_initialization": "rho0 = epsilon - ln_mk + rho_offset",
        "initial_fc_norm": float(history[0]["Fc_norm"]) if history else float("nan"),
        "runtime_seconds": runtime_seconds,
        "monotonically_decreasing_total_residual": monotonically_decreasing,
        "history": history,
        "final_state": {
            "ln_nk": ln_nk_state,
            "ln_mk": ln_mk_state,
            "rho": rho,
            "ln_ntot": ln_ntot_state,
        },
    }


def diagnose_full_vs_reduced_gie_direction(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    *,
    element_indices: Optional[jnp.ndarray] = None,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Dict[str, Any]:
    """Compare the current reduced-GIE direction against a full-GIE direction on one state."""

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

    hvector = hvector_func(state.temperature)
    hvector_cond = hvector_cond_func(state.temperature)
    gk = _compute_gk(
        state.temperature,
        ln_nk,
        ln_ntot,
        hvector,
        state.ln_normalized_pressure,
    )
    reduced_metrics = _compute_iteration_step_metrics(
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        epsilon,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )
    full_metrics = _solve_full_pipm_gie_direction_with_metrics(
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        epsilon,
    )

    reduced_delta_ln_nk = jnp.asarray(reduced_metrics["delta_ln_nk"])
    reduced_raw_delta_ln_mk = jnp.asarray(reduced_metrics["raw_delta_ln_mk"])
    reduced_clipped_delta_ln_mk = jnp.asarray(reduced_metrics["delta_ln_mk"])
    full_delta_ln_nk = jnp.asarray(full_metrics["delta_ln_nk"])
    full_raw_delta_ln_mk = jnp.asarray(full_metrics["raw_delta_ln_mk"])
    full_clipped_delta_ln_mk = jnp.clip(full_raw_delta_ln_mk, -0.1, 0.1)

    gas_diff = reduced_delta_ln_nk - full_delta_ln_nk
    raw_cond_diff = reduced_raw_delta_ln_mk - full_raw_delta_ln_mk
    clipped_cond_diff = reduced_clipped_delta_ln_mk - full_clipped_delta_ln_mk
    reduced_gas_norm = jnp.linalg.norm(reduced_delta_ln_nk)
    full_gas_norm = jnp.linalg.norm(full_delta_ln_nk)
    cosine_denom = jnp.maximum(reduced_gas_norm * full_gas_norm, 1.0e-300)
    gas_cosine_similarity = jnp.clip(
        jnp.dot(reduced_delta_ln_nk, full_delta_ln_nk) / cosine_denom,
        -1.0,
        1.0,
    )
    gas_angle_degrees = jnp.degrees(jnp.arccos(gas_cosine_similarity))

    reduced_step_eval = _evaluate_direction_with_existing_update_rule(
        state,
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        hvector,
        hvector_cond,
        epsilon,
        reduced_delta_ln_nk,
        reduced_raw_delta_ln_mk,
        reduced_metrics["delta_ln_ntot"],
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )
    full_step_eval = _evaluate_direction_with_existing_update_rule(
        state,
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        hvector,
        hvector_cond,
        epsilon,
        full_delta_ln_nk,
        full_raw_delta_ln_mk,
        full_metrics["delta_ln_ntot"],
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )

    full_materially_better = bool(
        full_step_eval["all_finite"]
        and reduced_step_eval["all_finite"]
        and full_step_eval["fresh_post_update_residual"]
        <= 0.95 * reduced_step_eval["fresh_post_update_residual"]
    )

    return {
        "epsilon": float(epsilon),
        "reduced_backend": reduced_solver,
        "reduced_regularization_mode": regularization_mode,
        "reduced_regularization_strength": float(regularization_strength),
        "raw_direction_comparison": {
            "reduced_raw_direction_norm": float(
                jnp.linalg.norm(
                    jnp.concatenate(
                        [
                            reduced_delta_ln_nk,
                            reduced_raw_delta_ln_mk,
                            jnp.atleast_1d(jnp.asarray(reduced_metrics["delta_ln_ntot"])),
                        ]
                    )
                )
            ),
            "full_raw_direction_norm": float(
                jnp.linalg.norm(
                    jnp.concatenate(
                        [
                            full_delta_ln_nk,
                            full_raw_delta_ln_mk,
                            jnp.atleast_1d(jnp.asarray(full_metrics["delta_ln_ntot"])),
                        ]
                    )
                )
            ),
            "reduced_clipped_direction_norm": float(
                jnp.linalg.norm(
                    jnp.concatenate(
                        [
                            reduced_delta_ln_nk,
                            reduced_clipped_delta_ln_mk,
                            jnp.atleast_1d(jnp.asarray(reduced_metrics["delta_ln_ntot"])),
                        ]
                    )
                )
            ),
            "full_clipped_direction_norm": float(
                jnp.linalg.norm(
                    jnp.concatenate(
                        [
                            full_delta_ln_nk,
                            full_clipped_delta_ln_mk,
                            jnp.atleast_1d(jnp.asarray(full_metrics["delta_ln_ntot"])),
                        ]
                    )
                )
            ),
            "gas_cosine_similarity": float(gas_cosine_similarity),
            "gas_angle_degrees": float(gas_angle_degrees),
            "max_abs_delta_ln_nk_diff": float(jnp.max(jnp.abs(gas_diff))),
            "max_abs_raw_delta_ln_mk_diff": float(jnp.max(jnp.abs(raw_cond_diff))),
            "max_abs_clipped_delta_ln_mk_diff": float(jnp.max(jnp.abs(clipped_cond_diff))),
        },
        "reduced_direction": {
            "pi_norm": float(reduced_metrics["pi_norm"]),
            "delta_ln_ntot": float(reduced_metrics["delta_ln_ntot"]),
            "raw_direction_norm": reduced_step_eval["raw_direction_norm"],
            "clipped_direction_norm": reduced_step_eval["clipped_direction_norm"],
            "gas_direction_norm": reduced_step_eval["gas_direction_norm"],
            "raw_cond_direction_norm": reduced_step_eval["raw_cond_direction_norm"],
            "clipped_cond_direction_norm": reduced_step_eval["clipped_cond_direction_norm"],
            "lam1_gas": reduced_step_eval["lam1_gas"],
            "lam1_cond": reduced_step_eval["lam1_cond"],
            "lam2_cond": reduced_step_eval["lam2_cond"],
            "lam": reduced_step_eval["lam"],
            "fresh_post_update_residual": reduced_step_eval["fresh_post_update_residual"],
            "invalid_numbers_detected": reduced_step_eval["invalid_numbers_detected"],
            "max_abs_delta_ln_nk": reduced_step_eval["max_abs_delta_ln_nk"],
            "max_abs_raw_delta_ln_mk": reduced_step_eval["max_abs_raw_delta_ln_mk"],
            "max_abs_clipped_delta_ln_mk": reduced_step_eval["max_abs_clipped_delta_ln_mk"],
        },
        "full_gie_direction": {
            "factorization_succeeded": full_metrics["factorization_succeeded"],
            "runtime_seconds": full_metrics["runtime_seconds"],
            "pi_norm": full_metrics["pi_norm"],
            "delta_ln_ntot": float(full_metrics["delta_ln_ntot"]),
            "raw_direction_norm": full_step_eval["raw_direction_norm"],
            "clipped_direction_norm": full_step_eval["clipped_direction_norm"],
            "gas_direction_norm": full_step_eval["gas_direction_norm"],
            "raw_cond_direction_norm": full_step_eval["raw_cond_direction_norm"],
            "clipped_cond_direction_norm": full_step_eval["clipped_cond_direction_norm"],
            "lam1_gas": full_step_eval["lam1_gas"],
            "lam1_cond": full_step_eval["lam1_cond"],
            "lam2_cond": full_step_eval["lam2_cond"],
            "lam": full_step_eval["lam"],
            "fresh_post_update_residual": full_step_eval["fresh_post_update_residual"],
            "invalid_numbers_detected": full_step_eval["invalid_numbers_detected"],
            "max_abs_delta_ln_nk": full_step_eval["max_abs_delta_ln_nk"],
            "max_abs_raw_delta_ln_mk": full_step_eval["max_abs_raw_delta_ln_mk"],
            "max_abs_clipped_delta_ln_mk": full_step_eval["max_abs_clipped_delta_ln_mk"],
        },
        "full_gie_materially_better": full_materially_better,
        "material_better_criterion": "full fresh_post_update_residual <= 0.95 * reduced fresh_post_update_residual with both finite",
    }


def diagnose_pdipm_vs_pipm_direction(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    *,
    element_indices: Optional[jnp.ndarray] = None,
    lambda_trials: Optional[Sequence[float]] = None,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Dict[str, Any]:
    """Compare the current PIPM direction against an experimental full PDIPM direction."""

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

    hvector = hvector_func(state.temperature)
    hvector_cond = hvector_cond_func(state.temperature)
    gk = _compute_gk(
        state.temperature,
        ln_nk,
        ln_ntot,
        hvector,
        state.ln_normalized_pressure,
    )
    pipm_metrics = _compute_iteration_step_metrics(
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        epsilon,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )
    pdipm_metrics = _solve_full_pdipm_gie_direction_with_metrics(
        ln_nk,
        ln_mk,
        None,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        epsilon,
    )

    pipm_delta_ln_nk = jnp.asarray(pipm_metrics["delta_ln_nk"])
    pipm_raw_delta_ln_mk = jnp.asarray(pipm_metrics["raw_delta_ln_mk"])
    pdipm_delta_ln_nk = jnp.asarray(pdipm_metrics["delta_ln_nk"])
    pdipm_raw_delta_ln_mk = jnp.asarray(pdipm_metrics["raw_delta_ln_mk"])
    pipm_clipped_delta_ln_mk = jnp.asarray(pipm_metrics["delta_ln_mk"])
    pdipm_clipped_delta_ln_mk = jnp.clip(pdipm_raw_delta_ln_mk, -0.1, 0.1)

    pipm_gas_norm = jnp.linalg.norm(pipm_delta_ln_nk)
    pdipm_gas_norm = jnp.linalg.norm(pdipm_delta_ln_nk)
    gas_cosine_similarity = jnp.clip(
        jnp.dot(pipm_delta_ln_nk, pdipm_delta_ln_nk)
        / jnp.maximum(pipm_gas_norm * pdipm_gas_norm, 1.0e-300),
        -1.0,
        1.0,
    )
    gas_angle_degrees = jnp.degrees(jnp.arccos(gas_cosine_similarity))

    pipm_heuristic_lambda = float(pipm_metrics["lam"])
    if lambda_trials is None:
        lambda_trials = [
            pipm_heuristic_lambda,
            0.5 * pipm_heuristic_lambda,
            0.1 * pipm_heuristic_lambda,
            1.0e-4,
        ]
    lambda_grid = _build_trial_lambda_grid(
        pipm_heuristic_lambda,
        lambda_trials=lambda_trials,
    )

    pipm_trials = _evaluate_direction_lambda_grid(
        state,
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        hvector,
        hvector_cond,
        epsilon,
        pipm_delta_ln_nk,
        pipm_raw_delta_ln_mk,
        pipm_metrics["delta_ln_ntot"],
        lambda_trials=lambda_grid,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )
    pdipm_trials = _evaluate_direction_lambda_grid(
        state,
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        hvector,
        hvector_cond,
        epsilon,
        pdipm_delta_ln_nk,
        pdipm_raw_delta_ln_mk,
        pdipm_metrics["delta_ln_ntot"],
        delta_aux=pdipm_metrics["delta_rho"],
        aux_name="delta_rho",
        lambda_trials=lambda_grid,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )

    ln_eta = epsilon - jnp.asarray(ln_mk)
    pdipm_current_residuals = _compute_pdipm_barrier_residual_components(
        ln_nk,
        ln_mk,
        ln_eta,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        pdipm_metrics["pi_vector"],
        epsilon,
    )

    best_pipm_trial = min(
        pipm_trials, key=lambda rec: float("inf") if not rec["all_finite"] else rec["fresh_post_update_residual"]
    )
    best_pdipm_trial = min(
        pdipm_trials, key=lambda rec: float("inf") if not rec["all_finite"] else rec["fresh_post_update_residual"]
    )
    pdipm_materially_better = bool(
        best_pdipm_trial["all_finite"]
        and best_pipm_trial["all_finite"]
        and best_pdipm_trial["fresh_post_update_residual"]
        <= 0.95 * best_pipm_trial["fresh_post_update_residual"]
    )

    return {
        "epsilon": float(epsilon),
        "rho_initialization": "rho = epsilon - ln_mk, eta = exp(rho) = exp(epsilon - ln_mk)",
        "lambda_grid": [float(x) for x in lambda_grid],
        "direction_comparison": {
            "pipm_raw_direction_norm": float(
                jnp.linalg.norm(
                    jnp.concatenate(
                        [
                            pipm_delta_ln_nk,
                            pipm_raw_delta_ln_mk,
                            jnp.atleast_1d(jnp.asarray(pipm_metrics["delta_ln_ntot"])),
                        ]
                    )
                )
            ),
            "pdipm_raw_direction_norm": float(
                jnp.linalg.norm(
                    jnp.concatenate(
                        [
                            pdipm_delta_ln_nk,
                            pdipm_raw_delta_ln_mk,
                            jnp.ravel(jnp.asarray(pdipm_metrics["delta_rho"])),
                            jnp.atleast_1d(jnp.asarray(pdipm_metrics["delta_ln_ntot"])),
                        ]
                    )
                )
            ),
            "pipm_clipped_direction_norm": float(
                jnp.linalg.norm(
                    jnp.concatenate(
                        [
                            pipm_delta_ln_nk,
                            pipm_clipped_delta_ln_mk,
                            jnp.atleast_1d(jnp.asarray(pipm_metrics["delta_ln_ntot"])),
                        ]
                    )
                )
            ),
            "pdipm_clipped_direction_norm": float(
                jnp.linalg.norm(
                    jnp.concatenate(
                        [
                            pdipm_delta_ln_nk,
                            pdipm_clipped_delta_ln_mk,
                            jnp.ravel(jnp.asarray(pdipm_metrics["delta_rho"])),
                            jnp.atleast_1d(jnp.asarray(pdipm_metrics["delta_ln_ntot"])),
                        ]
                    )
                )
            ),
            "gas_cosine_similarity": float(gas_cosine_similarity),
            "gas_angle_degrees": float(gas_angle_degrees),
            "max_abs_delta_ln_nk_diff": float(
                jnp.max(jnp.abs(pipm_delta_ln_nk - pdipm_delta_ln_nk))
            ),
            "max_abs_raw_delta_ln_mk_diff": float(
                jnp.max(jnp.abs(pipm_raw_delta_ln_mk - pdipm_raw_delta_ln_mk))
            ),
            "max_abs_clipped_delta_ln_mk_diff": float(
                jnp.max(jnp.abs(pipm_clipped_delta_ln_mk - pdipm_clipped_delta_ln_mk))
            ),
            "abs_delta_ln_ntot_diff": float(
                jnp.abs(pipm_metrics["delta_ln_ntot"] - pdipm_metrics["delta_ln_ntot"])
            ),
        },
        "pipm_direction": {
            "pi_norm": float(pipm_metrics["pi_norm"]),
            "delta_ln_ntot": float(pipm_metrics["delta_ln_ntot"]),
            "lam1_gas": float(pipm_metrics["lam1_gas"]),
            "lam1_cond": float(pipm_metrics["lam1_cond"]),
            "lam2_cond": float(pipm_metrics["lam2_cond"]),
            "lam": float(pipm_metrics["lam"]),
            "max_abs_delta_ln_nk": float(jnp.max(jnp.abs(pipm_delta_ln_nk))),
            "max_abs_raw_delta_ln_mk": float(jnp.max(jnp.abs(pipm_raw_delta_ln_mk))),
            "max_abs_clipped_delta_ln_mk": float(jnp.max(jnp.abs(pipm_clipped_delta_ln_mk))),
            "lambda_trials": pipm_trials,
        },
        "pdipm_direction": {
            "factorization_succeeded": pdipm_metrics["factorization_succeeded"],
            "runtime_seconds": pdipm_metrics["runtime_seconds"],
            "pi_norm": pdipm_metrics["pi_norm"],
            "delta_ln_ntot": float(pdipm_metrics["delta_ln_ntot"]),
            "max_abs_delta_ln_nk": float(jnp.max(jnp.abs(pdipm_delta_ln_nk))),
            "max_abs_raw_delta_ln_mk": float(jnp.max(jnp.abs(pdipm_raw_delta_ln_mk))),
            "max_abs_clipped_delta_ln_mk": float(jnp.max(jnp.abs(pdipm_clipped_delta_ln_mk))),
            "max_abs_delta_rho": float(jnp.max(jnp.abs(pdipm_metrics["delta_rho"]))),
            "pdipm_barrier_residuals_current": pdipm_current_residuals,
            "lambda_trials": pdipm_trials,
        },
        "pdipm_materially_better": pdipm_materially_better,
        "material_better_criterion": "best PDIPM fresh_post_update_residual on shared lambda grid <= 0.95 * best PIPM fresh_post_update_residual with both finite",
    }


def diagnose_pdipm_vs_pipm_fixed_epsilon_trajectories(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    *,
    rho_offsets: Sequence[float] = (0.0, 1.0, -1.0),
    max_iter: int = 10,
    min_lambda: float = 1.0e-6,
    backtrack_factor: float = 0.5,
    element_indices: Optional[jnp.ndarray] = None,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Dict[str, Any]:
    """Compare fixed-epsilon full-PDIPM trajectories against the current PIPM trajectory."""

    pipm_trace = trace_minimize_gibbs_cond_iterations(
        state,
        ln_nk_init=ln_nk,
        ln_mk_init=ln_mk,
        ln_ntot_init=ln_ntot,
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=hvector_func,
        hvector_cond_func=hvector_cond_func,
        epsilon=epsilon,
        max_iter=max_iter,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )
    pipm_history = pipm_trace["history"]
    pipm_best_residual = min(
        (rec["residual"] for rec in pipm_history),
        default=float("inf"),
    )

    pdipm_runs = []
    for rho_offset in rho_offsets:
        run = trace_pdipm_fixed_epsilon_trajectory(
            state,
            ln_nk,
            ln_mk,
            ln_ntot,
            formula_matrix,
            formula_matrix_cond,
            hvector_func,
            hvector_cond_func,
            epsilon,
            rho_offset=rho_offset,
            max_iter=max_iter,
            min_lambda=min_lambda,
            backtrack_factor=backtrack_factor,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
            element_indices=element_indices,
        )
        best_primal_residual = min(
            (rec["accepted_primal_fresh_residual"] for rec in run["history"] if rec["accepted"]),
            default=float("inf"),
        )
        run["best_primal_residual"] = best_primal_residual
        run["beats_pipm_within_budget"] = bool(best_primal_residual < pipm_best_residual)
        pdipm_runs.append(run)

    return {
        "epsilon": float(epsilon),
        "previous_one_step_center_path_note": "the earlier one-step PDIPM diagnostic used rho = epsilon - ln_mk, so Fc = ln_mk + rho - epsilon was zero at the initial state",
        "pipm_trace": {
            "n_iter": len(pipm_history),
            "best_residual": pipm_best_residual,
            "history": pipm_history,
        },
        "pdipm_runs": pdipm_runs,
    }


def _recompute_pi_for_residual(
    nk: jnp.ndarray,
    mk: jnp.ndarray,
    ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
    *,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> jnp.ndarray:
    """Re-solve the reduced system on the current state for residual evaluation only.

    This solve is intentionally separate from the earlier solve that produced the
    update direction. It is not fed back into the primal update; it is only used
    to evaluate a post-update barrier residual on a self-consistent state.
    """

    bk = formula_matrix @ nk
    sk = mk * mk * jnp.exp(-epsilon)
    pi_vector, _delta_ln_ntot = solve_reduced_gibbs_iteration_equations_cond(
        nk,
        mk,
        ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        bk,
        hvector_cond,
        sk,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )
    return pi_vector


def _compute_iteration_step_metrics(
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
    *,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Dict[str, jnp.ndarray]:
    """Compute the current PIPM step components without changing the update rule."""

    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntot = jnp.exp(ln_ntot)
    ln_sk = 2.0 * ln_mk - epsilon
    bk = formula_matrix @ nk

    pi_vector, delta_ln_ntot, reduced_metrics = _solve_reduced_gibbs_iteration_equations_cond_with_metrics(
        nk,
        mk,
        ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        bk,
        hvector_cond,
        jnp.exp(ln_sk),
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )

    delta_ln_nk = formula_matrix.T @ pi_vector + delta_ln_ntot - gk
    factor = jnp.exp(ln_mk - epsilon)
    raw_delta_ln_mk = factor * (formula_matrix_cond.T @ pi_vector - hvector_cond) + 1.0

    max_step_m = 0.1
    delta_ln_mk = jnp.clip(raw_delta_ln_mk, -max_step_m, max_step_m)

    lam1_gas = stepsize_cea_gas(delta_ln_nk, delta_ln_ntot, ln_nk, ln_ntot)
    lam1_cond = stepsize_cond_heurstic(delta_ln_mk)
    lam2_cond = stepsize_sk(delta_ln_mk, ln_mk, epsilon)
    lam = jnp.minimum(1.0, jnp.minimum(lam1_gas, jnp.minimum(lam1_cond, lam2_cond)))
    lam = jnp.clip(lam, 0.0, 1.0)

    limiter_candidates = jnp.asarray([1.0, lam1_gas, lam1_cond, lam2_cond])
    limiting_index = jnp.argmin(limiter_candidates).astype(jnp.int32)

    metrics = {
        "pi_vector": pi_vector,
        "delta_ln_ntot": delta_ln_ntot,
        "delta_ln_nk": delta_ln_nk,
        "raw_delta_ln_mk": raw_delta_ln_mk,
        "delta_ln_mk": delta_ln_mk,
        "lam1_gas": lam1_gas,
        "lam1_cond": lam1_cond,
        "lam2_cond": lam2_cond,
        "lam": lam,
        "limiting_index": limiting_index,
        "pi_norm": jnp.linalg.norm(pi_vector),
        "max_abs_delta_ln_nk": jnp.max(jnp.abs(delta_ln_nk)),
        "max_abs_raw_delta_ln_mk": jnp.max(jnp.abs(raw_delta_ln_mk)),
        "max_abs_clipped_delta_ln_mk": jnp.max(jnp.abs(delta_ln_mk)),
    }
    metrics.update(reduced_metrics)
    return metrics


def _evaluate_trial_step(
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    lam: float,
    delta_ln_nk: jnp.ndarray,
    delta_ln_mk: jnp.ndarray,
    delta_ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    temperature: float,
    ln_normalized_pressure: float,
    hvector: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
    *,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Dict[str, jnp.ndarray]:
    """Evaluate one damped trial step on a fresh self-consistent residual."""

    lam = jnp.asarray(lam, dtype=jnp.asarray(ln_ntot).dtype)
    trial_ln_nk = jnp.asarray(ln_nk) + lam * jnp.asarray(delta_ln_nk)
    trial_ln_mk = jnp.asarray(ln_mk) + lam * jnp.asarray(delta_ln_mk)
    trial_ln_ntot = jnp.asarray(ln_ntot) + lam * jnp.asarray(delta_ln_ntot)

    trial_nk = jnp.exp(trial_ln_nk)
    trial_mk = jnp.exp(trial_ln_mk)
    trial_ntot = jnp.exp(trial_ln_ntot)
    trial_gk = _compute_gk(
        temperature,
        trial_ln_nk,
        trial_ln_ntot,
        hvector,
        ln_normalized_pressure,
    )
    trial_An = formula_matrix @ trial_nk
    trial_Am = formula_matrix_cond @ trial_mk

    invalid_state = _contains_invalid_numbers(
        trial_ln_nk,
        trial_ln_mk,
        trial_ln_ntot,
        trial_nk,
        trial_mk,
        trial_ntot,
        trial_gk,
        trial_An,
        trial_Am,
    )

    pi_placeholder = jnp.full_like(b, jnp.nan)
    residual_placeholder = jnp.asarray(jnp.inf, dtype=trial_ntot.dtype)

    def _eval_valid(_):
        pi_vector_resid = _recompute_pi_for_residual(
            trial_nk,
            trial_mk,
            trial_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            trial_gk,
            hvector_cond,
            epsilon,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
        )
        residual = _compute_residuals(
            trial_nk,
            trial_mk,
            trial_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            trial_gk,
            hvector_cond,
            jnp.exp(epsilon),
            trial_An,
            trial_Am,
            pi_vector_resid,
        )
        residual_is_finite = jnp.isfinite(residual) & (~_contains_invalid_numbers(pi_vector_resid))
        residual = jnp.where(residual_is_finite, residual, residual_placeholder)
        return pi_vector_resid, residual, residual_is_finite

    pi_vector_resid, fresh_residual, all_finite = cond(
        invalid_state,
        lambda _: (pi_placeholder, residual_placeholder, jnp.asarray(False)),
        _eval_valid,
        operand=0,
    )

    return {
        "lam": lam,
        "ln_nk": trial_ln_nk,
        "ln_mk": trial_ln_mk,
        "ln_ntot": trial_ln_ntot,
        "nk": trial_nk,
        "mk": trial_mk,
        "ntot": trial_ntot,
        "gk": trial_gk,
        "An": trial_An,
        "Am": trial_Am,
        "pi_vector_resid": pi_vector_resid,
        "fresh_residual": fresh_residual,
        "all_finite": all_finite,
    }


def _choose_lambda_by_residual_backtracking(
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    current_gk: jnp.ndarray,
    current_An: jnp.ndarray,
    current_Am: jnp.ndarray,
    current_residual: float,
    lam_init: float,
    delta_ln_nk: jnp.ndarray,
    delta_ln_mk: jnp.ndarray,
    delta_ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    temperature: float,
    ln_normalized_pressure: float,
    hvector: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
    *,
    beta: float = 0.5,
    max_backtracks: int = 8,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Dict[str, jnp.ndarray]:
    """Choose the damped step from fresh residuals on backtracked trial states."""

    dtype = jnp.asarray(ln_ntot).dtype
    int_dtype = jnp.int32
    lam_init = jnp.asarray(lam_init, dtype=dtype)
    lam_init = jnp.where(jnp.isfinite(lam_init), jnp.clip(lam_init, 0.0, 1.0), 0.0)
    beta = jnp.asarray(beta, dtype=dtype)
    current_residual = jnp.asarray(current_residual, dtype=dtype)
    inf_value = jnp.asarray(jnp.inf, dtype=dtype)

    init_carry = {
        "accepted": jnp.asarray(False),
        "accept_index": jnp.asarray(0, dtype=int_dtype),
        "accept_residual": inf_value,
        "accept_lam": jnp.asarray(0.0, dtype=dtype),
        "accept_ln_nk": jnp.asarray(ln_nk),
        "accept_ln_mk": jnp.asarray(ln_mk),
        "accept_ln_ntot": jnp.asarray(ln_ntot),
        "accept_gk": jnp.asarray(current_gk),
        "accept_An": jnp.asarray(current_An),
        "accept_Am": jnp.asarray(current_Am),
        "best_found": jnp.asarray(False),
        "best_index": jnp.asarray(0, dtype=int_dtype),
        "best_residual": inf_value,
        "best_lam": jnp.asarray(0.0, dtype=dtype),
        "best_ln_nk": jnp.asarray(ln_nk),
        "best_ln_mk": jnp.asarray(ln_mk),
        "best_ln_ntot": jnp.asarray(ln_ntot),
        "best_gk": jnp.asarray(current_gk),
        "best_An": jnp.asarray(current_An),
        "best_Am": jnp.asarray(current_Am),
    }

    def _loop_body(i, carry):
        lam_trial = lam_init * jnp.power(beta, jnp.asarray(i, dtype=dtype))
        trial = _evaluate_trial_step(
            ln_nk,
            ln_mk,
            ln_ntot,
            lam_trial,
            delta_ln_nk,
            delta_ln_mk,
            delta_ln_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            temperature,
            ln_normalized_pressure,
            hvector,
            hvector_cond,
            epsilon,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
        )

        finite_trial = jnp.isfinite(trial["fresh_residual"]) & trial["all_finite"]
        monotone_accept = finite_trial & (
            cond(
                jnp.isfinite(current_residual),
                lambda _: trial["fresh_residual"] <= current_residual,
                lambda _: jnp.asarray(True),
                operand=0,
            )
        )
        accept_now = (~carry["accepted"]) & monotone_accept
        better_best = finite_trial & (
            (~carry["best_found"]) | (trial["fresh_residual"] < carry["best_residual"])
        )

        carry = cond(
            accept_now,
            lambda c: {
                **c,
                "accepted": jnp.asarray(True),
                "accept_index": jnp.asarray(i, dtype=int_dtype),
                "accept_residual": trial["fresh_residual"],
                "accept_lam": trial["lam"],
                "accept_ln_nk": trial["ln_nk"],
                "accept_ln_mk": trial["ln_mk"],
                "accept_ln_ntot": trial["ln_ntot"],
                "accept_gk": trial["gk"],
                "accept_An": trial["An"],
                "accept_Am": trial["Am"],
            },
            lambda c: c,
            carry,
        )
        carry = cond(
            better_best,
            lambda c: {
                **c,
                "best_found": jnp.asarray(True),
                "best_index": jnp.asarray(i, dtype=int_dtype),
                "best_residual": trial["fresh_residual"],
                "best_lam": trial["lam"],
                "best_ln_nk": trial["ln_nk"],
                "best_ln_mk": trial["ln_mk"],
                "best_ln_ntot": trial["ln_ntot"],
                "best_gk": trial["gk"],
                "best_An": trial["An"],
                "best_Am": trial["Am"],
            },
            lambda c: c,
            carry,
        )
        return carry

    carry = fori_loop(0, max_backtracks + 1, _loop_body, init_carry)

    use_accept = carry["accepted"]
    use_best = (~use_accept) & carry["best_found"]
    n_backtracks = jnp.where(
        use_accept,
        carry["accept_index"],
        jnp.where(use_best, carry["best_index"], jnp.asarray(max_backtracks, dtype=int_dtype)),
    )
    accept_code = jnp.where(
        use_accept,
        jnp.asarray(0, dtype=int_dtype),
        jnp.where(use_best, jnp.asarray(1, dtype=int_dtype), jnp.asarray(2, dtype=int_dtype)),
    )

    return {
        "lam": jnp.where(use_accept, carry["accept_lam"], jnp.where(use_best, carry["best_lam"], 0.0)),
        "ln_nk": jnp.where(use_accept, carry["accept_ln_nk"], jnp.where(use_best, carry["best_ln_nk"], ln_nk)),
        "ln_mk": jnp.where(use_accept, carry["accept_ln_mk"], jnp.where(use_best, carry["best_ln_mk"], ln_mk)),
        "ln_ntot": jnp.where(use_accept, carry["accept_ln_ntot"], jnp.where(use_best, carry["best_ln_ntot"], ln_ntot)),
        "gk": jnp.where(use_accept, carry["accept_gk"], jnp.where(use_best, carry["best_gk"], current_gk)),
        "An": jnp.where(use_accept, carry["accept_An"], jnp.where(use_best, carry["best_An"], current_An)),
        "Am": jnp.where(use_accept, carry["accept_Am"], jnp.where(use_best, carry["best_Am"], current_Am)),
        "fresh_residual": jnp.where(
            use_accept,
            carry["accept_residual"],
            jnp.where(use_best, carry["best_residual"], current_residual),
        ),
        "n_backtracks": n_backtracks,
        "accept_code": accept_code,
    }


def _debug_array(label, array, iter_count, limit=None):
    arr = jnp.ravel(jnp.asarray(array))
    max_val = jnp.max(arr)
    min_val = jnp.min(arr)
    has_nan = jnp.any(jnp.isnan(arr))
    has_inf = jnp.any(jnp.isinf(arr))
    has_over = False if limit is None else (max_val > limit)
    predicate = has_nan | has_inf | has_over
    max_idx = jnp.argmax(arr)
    max_at = arr[max_idx]
    if limit is None:
        over_count = jnp.array(0, dtype=jnp.int32)
        first_over_idx = jnp.array(0, dtype=jnp.int32)
        first_over_val = jnp.array(0.0)
    else:
        over_mask = arr > limit
        over_count = jnp.sum(over_mask)
        first_over_idx = jnp.argmax(over_mask)
        first_over_val = arr[first_over_idx]

    def _print(_):
        jdebug.print(
            "iter {i} {label}: min {min_val} max {max_val} nan {nan} inf {inf} "
            "over {over} max_idx {max_idx} max_at {max_at} over_count {over_count} "
            "first_over_idx {first_over_idx} first_over_val {first_over_val}",
            i=iter_count,
            label=label,
            min_val=min_val,
            max_val=max_val,
            nan=has_nan,
            inf=has_inf,
            over=has_over,
            max_idx=max_idx,
            max_at=max_at,
            over_count=over_count,
            first_over_idx=first_over_idx,
            first_over_val=first_over_val,
        )
        return 0

    return cond(predicate, _print, lambda _: 0, operand=0)


def _update_all_core(
    ln_nk,
    ln_mk,
    ln_ntot,
    formula_matrix,
    formula_matrix_cond,
    b,
    T,
    ln_normalized_pressure,
    hvector,
    hvector_cond,
    gk,
    An,
    Am,
    current_residual,
    epsilon,
    iter_count,
    debug_nan=False,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
):
    exp_overflow_limit = 700.0
    if debug_nan:
        _debug_array("ln_nk pre-exp", ln_nk, iter_count, exp_overflow_limit)
        _debug_array("ln_mk pre-exp", ln_mk, iter_count, exp_overflow_limit)
        _debug_array(
            "ln_ntot pre-exp", jnp.array([ln_ntot]), iter_count, exp_overflow_limit
        )

    ln_sk = 2.0 * ln_mk - epsilon
    bk = formula_matrix @ jnp.exp(ln_nk)

    if debug_nan:
        _debug_array("ln_nk_scaled pre-exp", ln_nk, iter_count, exp_overflow_limit)
        _debug_array("ln_mk_scaled pre-exp", ln_mk, iter_count, exp_overflow_limit)
        _debug_array(
            "ln_ntot_scaled pre-exp",
            jnp.array([ln_ntot]),
            iter_count,
            exp_overflow_limit,
        )
        _debug_array("ln_sk_scaled pre-exp", ln_sk, iter_count, exp_overflow_limit)

    step_metrics = _compute_iteration_step_metrics(
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        epsilon,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )
    pi_vector = step_metrics["pi_vector"]
    delta_ln_ntot = step_metrics["delta_ln_ntot"]

    delta_ln_nk = step_metrics["delta_ln_nk"]
    # this breaks the results. we cannot clip here.
    # raw_delta_ln_nk = formula_matrix.T @ pi_vector + delta_ln_ntot - gk
    # MAX_STEP_N_UP = 10.0  # do not update larger than ln(n) 0.1e ~ 10%
    # MAX_STEP_N_LOW = 10.0
    # delta_ln_nk = jnp.clip(raw_delta_ln_nk, -MAX_STEP_N_LOW, MAX_STEP_N_UP)

    # log_m_over_nu = jnp.clip(ln_mk - epsilon, LOG_MIN, LOG_MAX)
    log_m_over_nu = ln_mk - epsilon
    if debug_nan:
        _debug_array(
            "log_m_over_nu pre-exp", log_m_over_nu, iter_count, exp_overflow_limit
        )

    raw_delta_ln_mk = step_metrics["raw_delta_ln_mk"]

    MAX_STEP_M_UP = 0.1  # do not update larger than ln(m) 0.1e ~ 10%
    MAX_STEP_M_LOW = 0.1
    delta_ln_mk = step_metrics["delta_ln_mk"]
    # delta_ln_mk = jnp.exp(ln_mk - epsilon) * (formula_matrix_cond.T @ pi_vector - hvector_cond) + 1.0

    line_search_result = _choose_lambda_by_residual_backtracking(
        ln_nk,
        ln_mk,
        ln_ntot,
        gk,
        An,
        Am,
        current_residual,
        step_metrics["lam"],
        delta_ln_nk,
        delta_ln_mk,
        delta_ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        T,
        ln_normalized_pressure,
        hvector,
        hvector_cond,
        epsilon,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )

    lam = line_search_result["lam"]
    ln_nk = line_search_result["ln_nk"]
    ln_mk = line_search_result["ln_mk"]
    ln_ntot = line_search_result["ln_ntot"]
    gk = line_search_result["gk"]
    An = line_search_result["An"]
    Am = line_search_result["Am"]
    residual = line_search_result["fresh_residual"]
    if debug_nan:
        _debug_array("residual", jnp.array([residual]), iter_count)
    numeric_metrics = dict(step_metrics)
    numeric_metrics["lam_heuristic"] = step_metrics["lam"]
    numeric_metrics["lam_selected"] = lam
    numeric_metrics["lam"] = lam
    numeric_metrics["n_backtracks"] = line_search_result["n_backtracks"]
    numeric_metrics["residual_before"] = jnp.asarray(current_residual, dtype=residual.dtype)
    numeric_metrics["residual_after"] = residual
    numeric_metrics["line_search_accept_code"] = line_search_result["accept_code"]
    numeric_metrics["residual"] = residual
    return ln_nk, ln_mk, ln_ntot, gk, An, Am, residual, lam, numeric_metrics


def _update_all(
    ln_nk,
    ln_mk,
    ln_ntot,
    formula_matrix,
    formula_matrix_cond,
    b,
    T,
    ln_normalized_pressure,
    hvector,
    hvector_cond,
    gk,
    An,
    Am,
    current_residual,
    epsilon,
    iter_count,
    debug_nan=False,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
):
    (
        ln_nk,
        ln_mk,
        ln_ntot,
        gk,
        An,
        Am,
        residual,
        lam,
        _numeric_metrics,
    ) = _update_all_core(
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        T,
        ln_normalized_pressure,
        hvector,
        hvector_cond,
        gk,
        An,
        Am,
        current_residual,
        epsilon,
        iter_count,
        debug_nan=debug_nan,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )
    return ln_nk, ln_mk, ln_ntot, gk, An, Am, residual, lam


def _update_all_with_metrics(
    ln_nk,
    ln_mk,
    ln_ntot,
    formula_matrix,
    formula_matrix_cond,
    b,
    T,
    ln_normalized_pressure,
    hvector,
    hvector_cond,
    gk,
    An,
    Am,
    current_residual,
    epsilon,
    iter_count,
    debug_nan=False,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
):
    (
        ln_nk,
        ln_mk,
        ln_ntot,
        gk,
        An,
        Am,
        residual,
        lam,
        trace_metrics,
    ) = _update_all_core(
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        T,
        ln_normalized_pressure,
        hvector,
        hvector_cond,
        gk,
        An,
        Am,
        current_residual,
        epsilon,
        iter_count,
        debug_nan=debug_nan,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )
    trace_metrics = dict(trace_metrics)
    trace_metrics["line_search_used"] = True
    accept_code = int(trace_metrics["line_search_accept_code"])
    if accept_code == 0:
        accept_kind = "monotone"
    elif accept_code == 1:
        accept_kind = "best_finite_fallback"
    else:
        accept_kind = "zero_step"
    trace_metrics["line_search_accept_kind"] = accept_kind
    return ln_nk, ln_mk, ln_ntot, gk, An, Am, residual, lam, trace_metrics


def _contains_invalid_numbers(*arrays) -> jnp.ndarray:
    invalid_flags = []
    for array in arrays:
        arr = jnp.asarray(array)
        invalid_flags.append(jnp.any(jnp.isnan(arr) | jnp.isinf(arr)))
    return jnp.any(jnp.stack(invalid_flags))


def _minimize_gibbs_cond_core_impl(
    state: ThermoState,
    ln_nk_init: jnp.ndarray,
    ln_mk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    residual_crit: float = 1.0e-11,
    max_iter: int = 1000,
    element_indices: Optional[jnp.ndarray] = None,
    debug_nan: bool = False,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Shared implementation for condensate solves and diagnostics wrappers."""

    n_elements = formula_matrix.shape[0]
    if formula_matrix_cond.shape[0] != n_elements:
        raise ValueError(
            "formula_matrix and formula_matrix_cond must have the same number of element rows."
        )

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

    hvector = hvector_func(state.temperature)
    hvector_cond = hvector_cond_func(state.temperature)

    def cond_fun(carry):
        *_, residual, counter, _last_step_size = carry
        return (residual > residual_crit) & (counter < max_iter)

    def body_fun(carry):
        ln_nk, ln_mk, ln_ntot, gk, An, Am, residual, counter, _last_step_size = carry
        (
            ln_nk_new,
            ln_mk_new,
            ln_ntot_new,
            gk,
            An,
            Am,
            residual,
            last_step_size,
        ) = _update_all(
            ln_nk,
            ln_mk,
            ln_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            state.temperature,
            state.ln_normalized_pressure,
            hvector,
            hvector_cond,
            gk,
            An,
            Am,
            residual,
            epsilon,
            counter,
            debug_nan,
            reduced_solver,
            regularization_mode,
            regularization_strength,
        )
        return (
            ln_nk_new,
            ln_mk_new,
            ln_ntot_new,
            gk,
            An,
            Am,
            residual,
            counter + 1,
            last_step_size,
        )

    gk = _compute_gk(
        state.temperature,
        ln_nk_init,
        ln_ntot_init,
        hvector,
        state.ln_normalized_pressure,
    )
    An_in = formula_matrix @ jnp.exp(ln_nk_init)
    Am_in = formula_matrix_cond @ jnp.exp(ln_mk_init)
    init_last_step_size = jnp.asarray(0.0, dtype=ln_nk_init.dtype)

    ln_nk, ln_mk, ln_ntot, _gk, _An, _Am, residual, counter, last_step_size = while_loop(
        cond_fun,
        body_fun,
        (
            ln_nk_init,
            ln_mk_init,
            ln_ntot_init,
            gk,
            An_in,
            Am_in,
            jnp.inf,
            0,
            init_last_step_size,
        ),
    )
    return ln_nk, ln_mk, ln_ntot, counter, residual, last_step_size


def minimize_gibbs_cond_core(
    state: ThermoState,
    ln_nk_init: jnp.ndarray,
    ln_mk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,  ### new argument
    residual_crit: float = 1.0e-11,
    max_iter: int = 1000,
    element_indices: Optional[jnp.ndarray] = None,
    debug_nan: bool = False,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, int]:
    """Compute log(number of species) by minimizing the Gibbs energy using the Lagrange multipliers method.

    Args:
        state: Thermodynamic state containing temperature, pressure, and element abundances.
        ln_nk_init: Initial log number of species vector (n_species,).
        ln_ntot_init: Initial log total number of species.
        formula_matrix: Stoichiometric formula matrix (n_elements, n_species).
        hvector: Chemical potential over RT vector (n_species,).
        residual_crit: Convergence tolerance for residual norm.
        max_iter: Maximum number of iterations allowed.
        element_indices: Optional indices mapping ``state.element_vector`` onto the
            element ordering used by ``formula_matrix``/``formula_matrix_cond``.
            Use this when ``state.element_vector`` stores a superset of elements.

    Returns:
        Tuple containing:
            - Final log number of species vector (n_species,).
            - Final log number of condensed species vector (n_condensed_species,).
            - Final log eta vector (n_condensed_species,).
            - Final log total number of species.
            - Number of iterations performed.
    """

    ln_nk, ln_mk, ln_ntot, counter, _residual, _last_step_size = _minimize_gibbs_cond_core_impl(
        state,
        ln_nk_init,
        ln_mk_init,
        ln_ntot_init,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        epsilon,
        residual_crit=residual_crit,
        max_iter=max_iter,
        element_indices=element_indices,
        debug_nan=debug_nan,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )
    return ln_nk, ln_mk, ln_ntot, counter


def minimize_gibbs_cond_with_diagnostics(
    state: ThermoState,
    ln_nk_init: jnp.ndarray,
    ln_mk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    residual_crit: float = 1.0e-11,
    max_iter: int = 1000,
    element_indices: Optional[jnp.ndarray] = None,
    debug_nan: bool = False,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Run the active condensate solver and return lightweight convergence diagnostics."""

    ln_nk, ln_mk, ln_ntot, n_iter, final_residual, last_step_size = _minimize_gibbs_cond_core_impl(
        state,
        ln_nk_init,
        ln_mk_init,
        ln_ntot_init,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        epsilon,
        residual_crit=residual_crit,
        max_iter=max_iter,
        element_indices=element_indices,
        debug_nan=debug_nan,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )

    residual_crit_used = jnp.asarray(residual_crit, dtype=final_residual.dtype)
    max_iter_used = jnp.asarray(max_iter, dtype=n_iter.dtype)
    epsilon_used = jnp.asarray(epsilon, dtype=final_residual.dtype)
    converged = final_residual <= residual_crit_used
    hit_max_iter = (n_iter >= max_iter_used) & (~converged)
    invalid_numbers_detected = _contains_invalid_numbers(
        ln_nk,
        ln_mk,
        ln_ntot,
        last_step_size,
    )

    diagnostics = {
        "n_iter": n_iter,
        "converged": converged,
        "hit_max_iter": hit_max_iter,
        "final_residual": final_residual,
        "residual_crit": residual_crit_used,
        "max_iter": max_iter_used,
        "epsilon": epsilon_used,
        "final_step_size": last_step_size,
        "invalid_numbers_detected": invalid_numbers_detected,
        "debug_nan": jnp.asarray(debug_nan),
        "reduced_solver": reduced_solver,
        "regularization_mode": regularization_mode,
        "regularization_strength": jnp.asarray(regularization_strength, dtype=final_residual.dtype),
    }
    return ln_nk, ln_mk, ln_ntot, diagnostics


def trace_minimize_gibbs_cond_iterations(
    state: ThermoState,
    ln_nk_init: jnp.ndarray,
    ln_mk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    residual_crit: float = 1.0e-11,
    max_iter: int = 1000,
    element_indices: Optional[jnp.ndarray] = None,
    tiny_step: float = 1.0e-14,
    trial_lambda_every_iter: bool = False,
    trial_lambda_iterations: Optional[Sequence[int]] = None,
    trial_lambda_values: Optional[Sequence[float]] = None,
    trial_lambda_multipliers: Sequence[float] = (1.0, 0.5, 0.2, 0.1, 0.05),
    extra_trial_lambda_values: Sequence[float] = (1.0, 0.5, 0.2, 0.1, 0.05),
    gas_species_names: Optional[Sequence[str]] = None,
    gas_limiter_every_iter: bool = False,
    gas_limiter_iterations: Optional[Sequence[int]] = None,
    gas_limiter_top_k: int = 10,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Dict[str, Any]:
    """Run one condensate layer with a full per-iteration trace for debugging."""

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

    hvector = hvector_func(state.temperature)
    hvector_cond = hvector_cond_func(state.temperature)
    ln_nk = jnp.asarray(ln_nk_init)
    ln_mk = jnp.asarray(ln_mk_init)
    ln_ntot = jnp.asarray(ln_ntot_init)
    gk = _compute_gk(
        state.temperature,
        ln_nk,
        ln_ntot,
        hvector,
        state.ln_normalized_pressure,
    )
    An = formula_matrix @ jnp.exp(ln_nk)
    Am = formula_matrix_cond @ jnp.exp(ln_mk)
    residual = jnp.inf
    history = []

    for iter_count in range(max_iter):
        if float(residual) <= float(residual_crit):
            break

        (
            ln_nk,
            ln_mk,
            ln_ntot,
            gk,
            An,
            Am,
            residual,
            lam,
            metrics,
        ) = _update_all_with_metrics(
            ln_nk,
            ln_mk,
            ln_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            state.temperature,
            state.ln_normalized_pressure,
            hvector,
            hvector_cond,
            gk,
            An,
            Am,
            residual,
            epsilon,
            iter_count,
            debug_nan=False,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
        )
        record = {
            "iter": iter_count,
            "residual": float(residual),
            "lam": float(metrics["lam"]),
            "lam_heuristic": float(metrics["lam_heuristic"]),
            "lam_selected": float(metrics["lam_selected"]),
            "lam1_gas": float(metrics["lam1_gas"]),
            "lam1_cond": float(metrics["lam1_cond"]),
            "lam2_cond": float(metrics["lam2_cond"]),
            "n_backtracks": int(metrics["n_backtracks"]),
            "residual_before": float(metrics["residual_before"]),
            "residual_after": float(metrics["residual_after"]),
            "line_search_used": bool(metrics["line_search_used"]),
            "line_search_accept_kind": metrics["line_search_accept_kind"],
            "limiting_index": int(metrics["limiting_index"]),
            "max_abs_delta_ln_nk": float(metrics["max_abs_delta_ln_nk"]),
            "max_abs_raw_delta_ln_mk": float(metrics["max_abs_raw_delta_ln_mk"]),
            "max_abs_clipped_delta_ln_mk": float(metrics["max_abs_clipped_delta_ln_mk"]),
            "delta_ln_ntot": float(metrics["delta_ln_ntot"]),
            "pi_norm": float(metrics["pi_norm"]),
            "reduced_resn": float(metrics["reduced_resn"]),
            "reduced_row_scale_min": float(metrics["reduced_row_scale_min"]),
            "reduced_row_scale_max": float(metrics["reduced_row_scale_max"]),
            "reduced_row_scale_ratio": float(metrics["reduced_row_scale_ratio"]),
            "reduced_mat_maxabs": float(metrics["reduced_mat_maxabs"]),
            "reduced_vec_maxabs": float(metrics["reduced_vec_maxabs"]),
            "reduced_qk_maxabs": float(metrics["reduced_qk_maxabs"]),
            "reduced_solver_backend": metrics["reduced_solver_backend"],
            "reduced_factorization_succeeded": bool(metrics["reduced_factorization_succeeded"]),
            "reduced_regularization_mode": metrics["reduced_regularization_mode"],
            "reduced_regularization_used": float(metrics["reduced_regularization_used"]),
        }
        should_record_trial_lambdas = trial_lambda_every_iter or (
            trial_lambda_iterations is not None and iter_count in set(trial_lambda_iterations)
        )
        if should_record_trial_lambdas:
            record["trial_lambda_diagnostics"] = diagnose_iteration_lambda_trials(
                state,
                ln_nk=ln_nk - metrics["lam"] * metrics["delta_ln_nk"],
                ln_mk=ln_mk - metrics["lam"] * metrics["delta_ln_mk"],
                ln_ntot=ln_ntot - metrics["lam"] * metrics["delta_ln_ntot"],
                formula_matrix=formula_matrix,
                formula_matrix_cond=formula_matrix_cond,
                hvector_func=hvector_func,
                hvector_cond_func=hvector_cond_func,
                epsilon=epsilon,
                element_indices=element_indices,
                lambda_trials=trial_lambda_values,
                lambda_multipliers=trial_lambda_multipliers,
                extra_lambda_trials=extra_trial_lambda_values,
                reduced_solver=reduced_solver,
                regularization_mode=regularization_mode,
                regularization_strength=regularization_strength,
            )
        should_record_gas_limiter = gas_limiter_every_iter or (
            gas_limiter_iterations is not None and iter_count in set(gas_limiter_iterations)
        )
        if should_record_gas_limiter:
            record["gas_limiter_diagnostics"] = diagnose_gas_step_limiter_and_direction(
                state,
                ln_nk=ln_nk - metrics["lam"] * metrics["delta_ln_nk"],
                ln_mk=ln_mk - metrics["lam"] * metrics["delta_ln_mk"],
                ln_ntot=ln_ntot - metrics["lam"] * metrics["delta_ln_ntot"],
                formula_matrix=formula_matrix,
                formula_matrix_cond=formula_matrix_cond,
                hvector_func=hvector_func,
                hvector_cond_func=hvector_cond_func,
                epsilon=epsilon,
                element_indices=element_indices,
                gas_species_names=gas_species_names,
                top_k=gas_limiter_top_k,
                reduced_solver=reduced_solver,
                regularization_mode=regularization_mode,
                regularization_strength=regularization_strength,
            )
        history.append(record)
        if record["lam"] <= tiny_step:
            break

    return {
        "epsilon": float(epsilon),
        "residual_crit": float(residual_crit),
        "n_iter": len(history),
        "final_residual": float(residual),
        "reduced_solver": reduced_solver,
        "regularization_mode": regularization_mode,
        "regularization_strength": float(regularization_strength),
        "converged": bool(float(residual) <= float(residual_crit)),
        "hit_max_iter": bool(len(history) >= max_iter and float(residual) > float(residual_crit)),
        "history": history,
        "ln_nk": ln_nk,
        "ln_mk": ln_mk,
        "ln_ntot": ln_ntot,
    }


def trace_minimize_gibbs_cond_epsilon_sweep(
    state: ThermoState,
    ln_nk_init: jnp.ndarray,
    ln_mk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilons: Sequence[float],
    max_iter: int = 1000,
    element_indices: Optional[jnp.ndarray] = None,
    tiny_step: float = 1.0e-14,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Dict[str, Any]:
    """Trace one layer over a fixed list of epsilon values and summarize stagnation."""

    traces = []
    limiter_names = {
        0: "none_or_full_step",
        1: "gas_step_limiter",
        2: "condensate_step_limiter",
        3: "sk_limiter",
    }
    for epsilon in epsilons:
        trace = trace_minimize_gibbs_cond_iterations(
            state,
            ln_nk_init,
            ln_mk_init,
            ln_ntot_init,
            formula_matrix,
            formula_matrix_cond,
            hvector_func,
            hvector_cond_func,
            epsilon=float(epsilon),
            residual_crit=float(jnp.exp(jnp.asarray(epsilon))),
            max_iter=max_iter,
            element_indices=element_indices,
            tiny_step=tiny_step,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
        )
        history = trace["history"]
        first_tiny = next((rec for rec in history if rec["lam"] <= tiny_step), None)
        first_tiny_iter = None if first_tiny is None else first_tiny["iter"]
        first_tiny_limiter = None if first_tiny is None else limiter_names.get(first_tiny["limiting_index"], "unknown")
        residuals = [rec["residual"] for rec in history]
        residual_decreased_before_stagnation = any(
            curr < prev for prev, curr in zip(residuals[:-1], residuals[1:])
        )
        row_scale_ratio = None if first_tiny is None else first_tiny["reduced_row_scale_ratio"]
        trace["summary"] = {
            "made_progress": residual_decreased_before_stagnation,
            "first_tiny_lam_iter": first_tiny_iter,
            "first_tiny_lam_limiter": first_tiny_limiter,
            "residual_decreased_before_stagnation": residual_decreased_before_stagnation,
            "appears_ill_scaled": False if row_scale_ratio is None else bool(row_scale_ratio > 1.0e12),
        }
        traces.append(trace)

    return {"epsilons": [float(eps) for eps in epsilons], "traces": traces}
