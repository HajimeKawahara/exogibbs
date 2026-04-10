import jax.numpy as jnp
from jax import debug as jdebug
from jax.lax import cond
from jax.lax import fori_loop
from jax.lax import while_loop
from typing import Tuple, Optional, Dict, Any

from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.core import _compute_gk

# heuristic step size functions for condensates
from exogibbs.optimize.stepsize import stepsize_cea_gas
from exogibbs.optimize.stepsize import stepsize_cond_heurstic
from exogibbs.optimize.stepsize import stepsize_sk


def solve_gibbs_iteration_equations_cond(
    nk: jnp.ndarray,
    mk: jnp.ndarray,
    ntotk: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    bk: jnp.ndarray,
    nuk: float,
) -> Tuple[jnp.ndarray, float]:
    """
    Solve the Gibbs iteration equations with condensates using the Lagrange multipliers.
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
        hvector_cond: hvector_cond (i.e. ck) vector (n_cond, ) for k-th iteration.
        bk: (gas) formula_matrix @ nk vector (n_elements, ).
        nuk: control parameter for PIPM/PDIPM.

    Returns:
        Tuple containing:
            - The pi vector (nelements, ).fastchem_elements = list(gas.elements)
                element_indices = jnp.array([fastchem_elements.index(e) for e in elements])

            - The update of the  log total number of species (delta_ln_ntot).
    """
    nspecies = nk.shape[0]
    ncond = mk.shape[0]
    nelement = b.shape[0]

    resn = jnp.sum(nk) - ntotk  # delta n_{tot,k}
    Yg = formula_matrix * nk
    Yc = formula_matrix_cond * mk
    delta_bk_hat = b - (bk + formula_matrix_cond @ mk)  # b - (Ag nk + Ac mk)

    EN = jnp.identity(nspecies)
    ZNM = jnp.zeros((nspecies, ncond))
    ZB = jnp.zeros((nelement, nelement))
    zvm = jnp.zeros((ncond, 1))
    zvb = jnp.zeros((nelement, 1))
    u = jnp.ones((nspecies, 1))

    assemble_mat = jnp.block(
        [
            [EN, ZNM, - formula_matrix.T, -u],
            [ZNM.T, nuk * jnp.diag(1.0 / mk), -formula_matrix_cond.T, zvm],
            [Yg, Yc, ZB, zvb],
            [nk[:, jnp.newaxis].T, zvm.T, zvb.T, jnp.array([[-ntotk]])],
        ]
    )
    assemble_vec = jnp.concatenate(
        [-gk, nuk / mk - hvector_cond, delta_bk_hat, jnp.array([- resn])]
    )

    # B) whole scaling
    # row_scale = jnp.maximum(jnp.max(jnp.abs(assemble_mat), axis=1, keepdims=True), 1.0)
    # assemble_mat = assemble_mat / row_scale
    # assemble_vec = assemble_vec / row_scale[:, 0]

    # Solver
    # 1) direct solver
    assemble_variable = jnp.linalg.solve(assemble_mat, assemble_vec)
    # 2) LU solver
    # lu, piv = lu_factor(assemble_mat)
    # assemble_variable = lu_solve((lu, piv), assemble_vec)

    return assemble_variable[:nspecies], assemble_variable[nspecies:nspecies+ncond], assemble_variable[nspecies+ncond:-1], assemble_variable[-1]


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
    """Re-solve the full GIE system on the trial state for residual evaluation only."""

    bk = formula_matrix @ nk
    _delta_ln_nk, _delta_ln_mk, pi_vector, _delta_ln_ntot = solve_gibbs_iteration_equations_cond(
        nk,
        mk,
        ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        bk,
        jnp.exp(epsilon),
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
    """Compute the current full-GIE PIPM step components without changing the update rule."""

    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntot = jnp.exp(ln_ntot)
    bk = formula_matrix @ nk
    nuk = jnp.exp(epsilon)

    delta_ln_nk, raw_delta_ln_mk, pi_vector, delta_ln_ntot = solve_gibbs_iteration_equations_cond(
        nk,
        mk,
        ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        bk,
        nuk,
    )

    max_step_m = 0.1
    delta_ln_mk = jnp.clip(raw_delta_ln_mk, -max_step_m, max_step_m)

    lam1_gas = stepsize_cea_gas(delta_ln_nk, delta_ln_ntot, ln_nk, ln_ntot)
    lam1_cond = stepsize_cond_heurstic(delta_ln_mk)
    lam2_cond = stepsize_sk(delta_ln_mk, ln_mk, epsilon)
    lam = jnp.minimum(1.0, jnp.minimum(lam1_gas, jnp.minimum(lam1_cond, lam2_cond)))
    lam = jnp.clip(lam, 0.0, 1.0)

    return {
        "delta_ln_nk": delta_ln_nk,
        "raw_delta_ln_mk": raw_delta_ln_mk,
        "delta_ln_mk": delta_ln_mk,
        "pi_vector": pi_vector,
        "delta_ln_ntot": delta_ln_ntot,
        "lam1_gas": lam1_gas,
        "lam1_cond": lam1_cond,
        "lam2_cond": lam2_cond,
        "lam": lam,
        "pi_norm": jnp.linalg.norm(pi_vector),
        "max_abs_delta_ln_nk": jnp.max(jnp.abs(delta_ln_nk)),
        "max_abs_raw_delta_ln_mk": jnp.max(jnp.abs(raw_delta_ln_mk)),
        "max_abs_clipped_delta_ln_mk": jnp.max(jnp.abs(delta_ln_mk)),
    }


def _contains_invalid_numbers(*arrays) -> jnp.ndarray:
    invalid_flags = []
    for array in arrays:
        arr = jnp.asarray(array)
        invalid_flags.append(jnp.any(jnp.isnan(arr) | jnp.isinf(arr)))
    return jnp.any(jnp.stack(invalid_flags))


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
) -> dict:
    """Evaluate one damped GIE trial step on a fresh self-consistent residual."""

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
        "gk": trial_gk,
        "An": trial_An,
        "Am": trial_Am,
        "fresh_residual": fresh_residual,
        "all_finite": all_finite,
        "pi_vector_resid": pi_vector_resid,
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
) -> dict:
    """Choose the GIE damped step from fresh residuals on backtracked trial states."""

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
        )
        finite_trial = jnp.isfinite(trial["fresh_residual"]) & trial["all_finite"]
        monotone_accept = finite_trial & cond(
            jnp.isfinite(current_residual),
            lambda _: trial["fresh_residual"] <= current_residual,
            lambda _: jnp.asarray(True),
            operand=0,
        )
        accept_now = (~carry["accepted"]) & monotone_accept
        better_best = finite_trial & ((~carry["best_found"]) | (trial["fresh_residual"] < carry["best_residual"]))

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
        "n_backtracks": jnp.where(
            use_accept,
            carry["accept_index"],
            jnp.where(use_best, carry["best_index"], jnp.asarray(max_backtracks, dtype=int_dtype)),
        ),
        "accept_code": jnp.where(
            use_accept,
            jnp.asarray(0, dtype=int_dtype),
            jnp.where(use_best, jnp.asarray(1, dtype=int_dtype), jnp.asarray(2, dtype=int_dtype)),
        ),
    }


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
):
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
    delta_ln_nk = step_metrics["delta_ln_nk"]
    raw_delta_ln_mk = step_metrics["raw_delta_ln_mk"]
    delta_ln_mk = step_metrics["delta_ln_mk"]
    delta_ln_ntot = step_metrics["delta_ln_ntot"]
    lam1_gas = step_metrics["lam1_gas"]
    lam1_cond = step_metrics["lam1_cond"]
    lam2_cond = step_metrics["lam2_cond"]
    lam = step_metrics["lam"]

    line_search_result = _choose_lambda_by_residual_backtracking(
        ln_nk,
        ln_mk,
        ln_ntot,
        gk,
        An,
        Am,
        current_residual,
        lam,
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
    )
    lam_selected = line_search_result["lam"]
    metrics = {
        "delta_ln_nk": delta_ln_nk,
        "raw_delta_ln_mk": raw_delta_ln_mk,
        "delta_ln_mk": delta_ln_mk,
        "delta_ln_ntot": delta_ln_ntot,
        "lam1_gas": lam1_gas,
        "lam1_cond": lam1_cond,
        "lam2_cond": lam2_cond,
        "lam_heuristic": lam,
        "lam_selected": lam_selected,
        "lam": lam_selected,
        "pi_norm": step_metrics["pi_norm"],
        "max_abs_delta_ln_nk": step_metrics["max_abs_delta_ln_nk"],
        "max_abs_raw_delta_ln_mk": step_metrics["max_abs_raw_delta_ln_mk"],
        "max_abs_clipped_delta_ln_mk": step_metrics["max_abs_clipped_delta_ln_mk"],
        "n_backtracks": line_search_result["n_backtracks"],
        "residual_before": jnp.asarray(current_residual, dtype=line_search_result["fresh_residual"].dtype),
        "residual_after": line_search_result["fresh_residual"],
        "line_search_accept_code": line_search_result["accept_code"],
    }
    return (
        line_search_result["ln_nk"],
        line_search_result["ln_mk"],
        line_search_result["ln_ntot"],
        line_search_result["gk"],
        line_search_result["An"],
        line_search_result["Am"],
        line_search_result["fresh_residual"],
        lam_selected,
        metrics,
    )


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
):
    (
        ln_nk,
        ln_mk,
        ln_ntot,
        gk,
        An,
        Am,
        residual,
        _lam_selected,
        _metrics,
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
    )
    return ln_nk, ln_mk, ln_ntot, gk, An, Am, residual


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
):
    (
        ln_nk,
        ln_mk,
        ln_ntot,
        gk,
        An,
        Am,
        residual,
        lam_selected,
        metrics,
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
    )
    trace_metrics = dict(metrics)
    trace_metrics["line_search_used"] = True
    accept_code = int(trace_metrics["line_search_accept_code"])
    if accept_code == 0:
        accept_kind = "monotone"
    elif accept_code == 1:
        accept_kind = "best_finite_fallback"
    else:
        accept_kind = "zero_step"
    trace_metrics["line_search_accept_kind"] = accept_kind
    return ln_nk, ln_mk, ln_ntot, gk, An, Am, residual, lam_selected, trace_metrics


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
        *_, residual, counter = carry
        return (residual > residual_crit) & (counter < max_iter)

    def body_fun(carry):
        ln_nk, ln_mk, ln_ntot, gk, An, Am, residual, counter = carry
        ln_nk_new, ln_mk_new, ln_ntot_new, gk, An, Am, residual = _update_all(
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

    ln_nk, ln_mk, ln_ntot, gk, Am, An, residual, counter = while_loop(
        cond_fun,
        body_fun,
        (ln_nk_init, ln_mk_init, ln_ntot_init, gk, An_in, Am_in, jnp.inf, 0),
    )
    return ln_nk, ln_mk, ln_ntot, counter


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
    """Run one full-GIE condensate layer with per-iteration line-search diagnostics."""

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
            residual,
            epsilon,
            iter_count,
            debug_nan=False,
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
            "max_abs_delta_ln_nk": float(jnp.max(jnp.abs(metrics["delta_ln_nk"]))),
            "max_abs_raw_delta_ln_mk": float(jnp.max(jnp.abs(metrics["raw_delta_ln_mk"]))),
            "max_abs_clipped_delta_ln_mk": float(jnp.max(jnp.abs(metrics["delta_ln_mk"]))),
            "delta_ln_ntot": float(metrics["delta_ln_ntot"]),
        }
        history.append(record)
        if record["lam"] <= tiny_step:
            break

    invalid_numbers_detected = bool(_contains_invalid_numbers(ln_nk, ln_mk, ln_ntot, residual))
    return {
        "epsilon": float(epsilon),
        "residual_crit": float(residual_crit),
        "n_iter": len(history),
        "final_residual": float(residual),
        "converged": bool(float(residual) <= float(residual_crit)),
        "hit_max_iter": bool(len(history) >= max_iter and float(residual) > float(residual_crit)),
        "invalid_numbers_detected": invalid_numbers_detected,
        "history": history,
        "ln_nk": ln_nk,
        "ln_mk": ln_mk,
        "ln_ntot": ln_ntot,
    }
