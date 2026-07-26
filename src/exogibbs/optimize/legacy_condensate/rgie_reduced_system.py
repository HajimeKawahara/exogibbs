"""Reduced-system primitives shared by explicit legacy condensate routes."""

from __future__ import annotations

from typing import Dict, Tuple

import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
from jax.scipy.linalg import lu_factor, lu_solve

from exogibbs.optimize.core import _A_diagn_At

DEFAULT_REDUCED_SOLVER = "augmented_lu_row_scaled"
DEFAULT_REGULARIZATION_MODE = "none"
DEFAULT_REGULARIZATION_STRENGTH = 0.0


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
    assemble_mat = jnp.block(
        [[Qk, bk[:, None]], [bk[None, :], jnp.array([[resn]])]]
    )
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
            f"Unknown regularization_mode '{regularization_mode}'. "
            "Expected 'none' or 'diag_shift'."
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
    """Solve the reduced Gibbs iteration equations for condensates.

    Returns:
        The element-potential vector and the total-density log update.
    """

    pi_vector, delta_ln_ntot, _metrics = (
        _solve_reduced_gibbs_iteration_equations_cond_with_metrics(
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
    include_system_trace: bool = False,
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
        row_scale = jnp.maximum(
            jnp.max(jnp.abs(reg_assemble_mat), axis=1, keepdims=True),
            1.0,
        )
        solve_mat_row = reg_assemble_mat / row_scale
        col_scale = jnp.maximum(
            jnp.max(jnp.abs(solve_mat_row), axis=0, keepdims=True),
            1.0,
        )
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
        "reduced_regularization_strength": jnp.asarray(
            regularization_strength,
            dtype=q_block.dtype,
        ),
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
    if include_system_trace:
        metrics.update(
            {
                "diagnostic_only_reduced_system_trace": jnp.asarray(True),
                "reduced_jacobian_matrix_before_scaling": assemble_mat,
                "reduced_rhs_vector_before_scaling": assemble_vec,
                "reduced_row_scaling_vector": row_scale_flat,
                "reduced_col_scaling_vector": col_scale_flat,
                "reduced_solve_matrix": solve_mat,
                "reduced_solve_rhs_vector": solve_vec,
                "reduced_raw_solver_result_vector": assemble_variable,
                "reduced_q_block": q_block,
                "reduced_rhs_element_block": rhs,
                "reduced_scalar_rhs": jnp.asarray(scalar_rhs),
                "reduced_element_condensate_block_before_scaling": (
                    jnp.asarray(formula_matrix_cond, dtype=jnp.float64)
                    * jnp.asarray(mk, dtype=jnp.float64)[None, :]
                ),
                "reduced_element_condensate_block_scaled_by_reduced_row_scaling": (
                    (
                        jnp.asarray(formula_matrix_cond, dtype=jnp.float64)
                        * jnp.asarray(mk, dtype=jnp.float64)[None, :]
                    )
                    / jnp.maximum(row_scale_flat[:-1], 1.0e-300)[:, None]
                ),
                "reduced_element_condensate_owner_mk_vector": jnp.asarray(
                    mk,
                    dtype=jnp.float64,
                ),
                "reduced_system_trace_basis": jnp.asarray(1, dtype=jnp.int32),
            }
        )
    return assemble_variable[:-1], assemble_variable[-1], metrics


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
