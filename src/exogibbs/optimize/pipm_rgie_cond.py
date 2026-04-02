import jax.numpy as jnp
from jax import debug as jdebug
from jax.lax import while_loop
from jax.lax import cond
from jax.scipy.linalg import lu_factor, lu_solve
from typing import Dict, Tuple, Optional, Sequence, Any

from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.core import _A_diagn_At
from exogibbs.optimize.core import _compute_gk

# heuristic step size functions for condensates
from exogibbs.optimize.stepsize import stepsize_cea_gas
from exogibbs.optimize.stepsize import stepsize_cond_heurstic
from exogibbs.optimize.stepsize import stepsize_sk


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

    resn = jnp.sum(nk) - ntotk
    Qk = _A_diagn_At(nk, formula_matrix) + _A_diagn_At(sk, formula_matrix_cond)
    Angk = formula_matrix @ (gk * nk)
    ngk = jnp.dot(nk, gk)

    delta_bk_hat = b - (bk + formula_matrix_cond @ mk)  # b - (Ag nk + Ac mk)
    condvec = formula_matrix_cond @ (sk * hvector_cond - mk)  # Ac(sk*ck - mk)

    # A) Row-wise scaling
    # row_scale = jnp.maximum(jnp.max(jnp.abs(Qk), axis=1, keepdims=True), 1.0)
    # Qk = Qk / row_scale
    # bk_scaled = bk / row_scale[:, 0]
    # Angk = Angk / row_scale[:, 0]
    # condvec = condvec / row_scale[:, 0]
    # delta_bk_hat = delta_bk_hat / row_scale[:, 0]
    # assemble_mat = jnp.block([[Qk, bk_scaled[:, None]], [bk[None, :], jnp.array([[resn]])]])

    assemble_mat = jnp.block([[Qk, bk[:, None]], [bk[None, :], jnp.array([[resn]])]])
    assemble_vec = jnp.concatenate(
        [Angk + condvec + delta_bk_hat, jnp.array([ngk - resn])]
    )

    # B) whole scaling
    row_scale = jnp.maximum(jnp.max(jnp.abs(assemble_mat), axis=1, keepdims=True), 1.0)
    assemble_mat = assemble_mat / row_scale
    assemble_vec = assemble_vec / row_scale[:, 0]

    # Solver
    # 1) direct solver
    # assemble_variable = jnp.linalg.solve(assemble_mat, assemble_vec)
    # 2) LU solver
    lu, piv = lu_factor(assemble_mat)
    assemble_variable = lu_solve((lu, piv), assemble_vec)

    return assemble_variable[:-1], assemble_variable[-1]


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
) -> Tuple[jnp.ndarray, float, Dict[str, jnp.ndarray]]:
    """Solve the reduced system and expose scale metrics for debugging."""

    resn = jnp.sum(nk) - ntotk
    Qk = _A_diagn_At(nk, formula_matrix) + _A_diagn_At(sk, formula_matrix_cond)
    Angk = formula_matrix @ (gk * nk)
    ngk = jnp.dot(nk, gk)

    delta_bk_hat = b - (bk + formula_matrix_cond @ mk)
    condvec = formula_matrix_cond @ (sk * hvector_cond - mk)

    assemble_mat = jnp.block([[Qk, bk[:, None]], [bk[None, :], jnp.array([[resn]])]])
    assemble_vec = jnp.concatenate(
        [Angk + condvec + delta_bk_hat, jnp.array([ngk - resn])]
    )

    row_scale = jnp.maximum(jnp.max(jnp.abs(assemble_mat), axis=1, keepdims=True), 1.0)
    assemble_mat_scaled = assemble_mat / row_scale
    assemble_vec_scaled = assemble_vec / row_scale[:, 0]

    lu, piv = lu_factor(assemble_mat_scaled)
    assemble_variable = lu_solve((lu, piv), assemble_vec_scaled)

    row_scale_flat = row_scale[:, 0]
    metrics = {
        "reduced_resn": resn,
        "reduced_row_scale_min": jnp.min(row_scale_flat),
        "reduced_row_scale_max": jnp.max(row_scale_flat),
        "reduced_row_scale_ratio": jnp.max(row_scale_flat) / jnp.maximum(jnp.min(row_scale_flat), 1.0e-300),
        "reduced_mat_maxabs": jnp.max(jnp.abs(assemble_mat)),
        "reduced_vec_maxabs": jnp.max(jnp.abs(assemble_vec)),
        "reduced_qk_maxabs": jnp.max(jnp.abs(Qk)),
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
    epsilon,
    iter_count,
    debug_nan=False,
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
        _metrics,
    ) = _update_all_with_metrics(
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
        epsilon,
        iter_count,
        debug_nan=debug_nan,
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
    epsilon,
    iter_count,
    debug_nan=False,
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

    # relaxation and update
    # lam = 0.0001  # need to reconsider

    lam = step_metrics["lam"]

    ln_ntot += lam * delta_ln_ntot
    ln_nk += lam * delta_ln_nk
    ln_mk += lam * delta_ln_mk

    # clip
    # ln_nk = jnp.clip(ln_nk, LOG_MIN, LOG_MAX)
    # ln_ntot = jnp.clip(ln_ntot, LOG_MIN, LOG_MAX)
    # ln_mk = jnp.clip(ln_mk, LOG_MIN, LOG_MAX)

    # Rebuild the thermodynamic state after the damped primal update.

    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntot = jnp.exp(ln_ntot)
    gk = _compute_gk(T, ln_nk, ln_ntot, hvector, ln_normalized_pressure)
    An = formula_matrix @ nk
    Am = formula_matrix_cond @ mk

    # The earlier reduced solve defined the update direction. Re-solve only to
    # evaluate the barrier residual on this updated state.
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
    )

    residual = _compute_residuals(
        nk,
        mk,
        ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        jnp.exp(epsilon),
        An,
        Am,
        pi_vector_resid,
    )
    if debug_nan:
        _debug_array("residual", jnp.array([residual]), iter_count)
    trace_metrics = dict(step_metrics)
    trace_metrics["residual"] = residual
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
            epsilon,
            counter,
            debug_nan,
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
            epsilon,
            iter_count,
            debug_nan=False,
        )
        record = {
            "iter": iter_count,
            "residual": float(residual),
            "lam": float(metrics["lam"]),
            "lam1_gas": float(metrics["lam1_gas"]),
            "lam1_cond": float(metrics["lam1_cond"]),
            "lam2_cond": float(metrics["lam2_cond"]),
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
        }
        history.append(record)
        if record["lam"] <= tiny_step:
            break

    return {
        "epsilon": float(epsilon),
        "residual_crit": float(residual_crit),
        "n_iter": len(history),
        "final_residual": float(residual),
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
