import time

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax import jacrev
from jax.lax import while_loop, stop_gradient
from jax.scipy.linalg import cho_factor
from jax.scipy.linalg import cho_solve
from functools import partial
from typing import Tuple, Callable, Dict

from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.core import _A_diagn_At
from exogibbs.optimize.core import _compute_gk
from exogibbs.optimize.vjpgibbs import vjp_temperature
from exogibbs.optimize.vjpgibbs import vjp_pressure
from exogibbs.optimize.vjpgibbs import vjp_elements

_CHO_EPS = 1.0e-18


def _minimize_gibbs_cond_fun(carry):
    (
        _ln_nk,
        _ln_ntot,
        _gk,
        _an,
        epsilon,
        counter,
        _formula_matrix,
        _element_vector,
        _temperature,
        _ln_normalized_pressure,
        _hvector,
        epsilon_crit,
        max_iter,
    ) = carry
    return (epsilon > epsilon_crit) & (counter < max_iter)


def _minimize_gibbs_body_fun(carry):
    (
        ln_nk,
        ln_ntot,
        gk,
        An,
        _epsilon,
        counter,
        formula_matrix,
        element_vector,
        temperature,
        ln_normalized_pressure,
        hvector,
        epsilon_crit,
        max_iter,
    ) = carry
    ln_nk_new, ln_ntot_new, epsilon, gk, An = update_all(
        ln_nk,
        ln_ntot,
        formula_matrix,
        element_vector,
        temperature,
        ln_normalized_pressure,
        hvector,
        gk,
        An,
    )
    # Keep cond/body at module scope and thread solver context through the
    # carry so repeated calls reuse the same while_loop callable identities.
    return (
        ln_nk_new,
        ln_ntot_new,
        gk,
        An,
        epsilon,
        counter + 1,
        formula_matrix,
        element_vector,
        temperature,
        ln_normalized_pressure,
        hvector,
        epsilon_crit,
        max_iter,
    )

def solve_gibbs_iteration_equations(
    nk: jnp.ndarray,
    ntotk: float,
    formula_matrix: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    An: jnp.ndarray,
) -> Tuple[jnp.ndarray, float]:
    """
    Solve the Gibbs iteration equations using the Lagrange multipliers.
    This function computes the matrix and vector to solve the system of equations
    that arises from the Gibbs energy minimization problem.

    Args:
        nk: number of species vector (n_species,) for k-th iteration.
        ntotk: Total number of species for k-th iteration.
        formula_matrix: Formula matrix for stoichiometric constraints (n_elements, n_species).
        b: Element abundance vector (n_elements, ).
        gk: gk vector (n_species,) for k-th iteration.
        An: formula_matrix @ nk vector (n_elements, ).

    Returns:
        Tuple containing:
            - The pi vector (nspecies, ).
            - The update of the  log total number of species (delta_ln_ntot).
    """
    resn = jnp.sum(nk) - ntotk
    bmatrix = _A_diagn_At(nk, formula_matrix)
    gk_nk = gk * nk
    Angk = formula_matrix @ gk_nk
    ngk = jnp.dot(nk, gk)
    rhs = Angk + b - An
    scalar_rhs = ngk - resn

    # Solve the bordered system through its Schur complement on
    # B = A diag(n) A^T to avoid assembling the dense (E+1)x(E+1) matrix.
    jitter = jnp.asarray(_CHO_EPS, dtype=bmatrix.dtype)
    eye = jnp.eye(bmatrix.shape[0], dtype=bmatrix.dtype)
    c_factor, lower = cho_factor(bmatrix + jitter * eye)

    rhs_pair = jnp.stack((rhs, An), axis=1)
    solved_pair = cho_solve((c_factor, lower), rhs_pair)
    binv_rhs = solved_pair[:, 0]
    binv_an = solved_pair[:, 1]

    schur = resn - jnp.vdot(An, binv_an)
    schur_safe = jnp.where(
        jnp.abs(schur) < jitter,
        jnp.where(schur < 0.0, -jitter, jitter),
        schur,
    )
    delta_ln_ntot = (scalar_rhs - jnp.vdot(An, binv_rhs)) / schur_safe
    pi_vector = binv_rhs - binv_an * delta_ln_ntot
    return pi_vector, delta_ln_ntot


def compute_residuals_with_at_pi(
    nk: jnp.ndarray,
    ntotk: float,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    An: jnp.ndarray,
    at_pi: jnp.ndarray,
) -> float:
    ress = nk * (at_pi - gk)
    ress_squared = jnp.dot(ress, ress)

    An_b = An - b
    resj_squared = jnp.dot(An_b, An_b)

    resn = jnp.sum(nk) - ntotk
    resn_squared = jnp.dot(resn, resn)
    return jnp.sqrt(ress_squared + resj_squared + resn_squared)

CEA_SIZE = 18.420681        # = -ln(1e-8)
LN_X_CAP = 9.2103404        # = -ln(1e-4)

def _cea_lambda(delta_ln_nk, delta_ln_ntot, ln_nk, ln_ntot, size=CEA_SIZE):
    # λ1: ensure |Δln n|<=0.4, |Δln n_k|<=2
    cap_ntot = 5.0 * jnp.abs(delta_ln_ntot)           # 1/0.4
    cap_sp   = jnp.max(jnp.abs(delta_ln_nk))
    denom1   = jnp.maximum(jnp.maximum(cap_ntot, cap_sp), 1e-300)
    lam1     = 2.0 / denom1

    # maintain x_k<=1e-4 if increasing when x_k<=1e-8
    ln_xk  = ln_nk - ln_ntot
    small  = (ln_xk <= -size) & (delta_ln_nk >= 0.0)
    denom2 = delta_ln_nk - delta_ln_ntot
    safe   = small & (denom2 > 0.0)
    cand   = ( -LN_X_CAP - ln_xk ) / denom2           # (-ln 1e-4 - ln xk)/(Δln nk - Δln n)
    lam2   = jnp.where(jnp.any(safe), jnp.min(jnp.where(safe, cand, jnp.inf)), jnp.inf)

    lam = jnp.minimum(1.0, jnp.minimum(lam1, lam2))
    # safe guard
    lam = jnp.clip(lam, 1e-6, 1.0)
    return lam


def _prepare_iteration_system(
    ln_nk,
    ln_ntot,
    formula_matrix,
    b,
    gk,
    An,
):
    nk = jnp.exp(ln_nk)
    ntot = jnp.exp(ln_ntot)
    resn = jnp.sum(nk) - ntot
    bmatrix = _A_diagn_At(nk, formula_matrix)
    gk_nk = gk * nk
    Angk = formula_matrix @ gk_nk
    ngk = jnp.dot(nk, gk)
    rhs = Angk + b - An
    scalar_rhs = ngk - resn
    return nk, ntot, resn, bmatrix, rhs, scalar_rhs


def _solve_iteration_system(bmatrix, rhs, An, resn):
    jitter = jnp.asarray(_CHO_EPS, dtype=bmatrix.dtype)
    eye = jnp.eye(bmatrix.shape[0], dtype=bmatrix.dtype)
    c_factor, lower = cho_factor(bmatrix + jitter * eye)

    rhs_pair = jnp.stack((rhs, An), axis=1)
    solved_pair = cho_solve((c_factor, lower), rhs_pair)
    binv_rhs = solved_pair[:, 0]
    binv_an = solved_pair[:, 1]

    schur = resn - jnp.vdot(An, binv_an)
    schur_safe = jnp.where(
        jnp.abs(schur) < jitter,
        jnp.where(schur < 0.0, -jitter, jitter),
        schur,
    )
    return binv_rhs, binv_an, schur_safe


def _finish_iteration_solve(binv_rhs, binv_an, An, scalar_rhs, schur_safe):
    delta_ln_ntot = (scalar_rhs - jnp.vdot(An, binv_rhs)) / schur_safe
    pi_vector = binv_rhs - binv_an * delta_ln_ntot
    return pi_vector, delta_ln_ntot


def _apply_iteration_step(
    ln_nk,
    ln_ntot,
    formula_matrix,
    gk,
    pi_vector,
    delta_ln_ntot,
):
    at_pi = formula_matrix.T @ pi_vector
    delta_ln_nk = at_pi + delta_ln_ntot - gk
    lam = _cea_lambda(delta_ln_nk, delta_ln_ntot, ln_nk, ln_ntot)
    ln_ntot_new = ln_ntot + lam * delta_ln_ntot
    ln_nk_new = ln_nk + lam * delta_ln_nk
    return ln_nk_new, ln_ntot_new, lam, delta_ln_nk, at_pi


def _evaluate_iteration_state(
    ln_nk,
    ln_ntot,
    formula_matrix,
    b,
    T,
    ln_normalized_pressure,
    hvector,
    gk_prev,
    lam,
    at_pi,
    pi_vector,
):
    nk = jnp.exp(ln_nk)
    ntot = jnp.exp(ln_ntot)
    del T, ln_normalized_pressure, hvector, pi_vector
    gk = gk_prev + lam * (at_pi - gk_prev)
    An = formula_matrix @ nk
    epsilon = compute_residuals_with_at_pi(nk, ntot, b, gk, An, at_pi)
    return gk, An, epsilon

def update_all(
    ln_nk,
    ln_ntot,
    formula_matrix,
    b,
    T,
    ln_normalized_pressure,
    hvector,
    gk,
    An,
):
    _, _, resn, bmatrix, rhs, scalar_rhs = _prepare_iteration_system(
        ln_nk, ln_ntot, formula_matrix, b, gk, An
    )
    binv_rhs, binv_an, schur_safe = _solve_iteration_system(bmatrix, rhs, An, resn)
    pi_vector, delta_ln_ntot = _finish_iteration_solve(
        binv_rhs, binv_an, An, scalar_rhs, schur_safe
    )
    ln_nk, ln_ntot, lam, _, at_pi = _apply_iteration_step(
        ln_nk, ln_ntot, formula_matrix, gk, pi_vector, delta_ln_ntot
    )
    gk, An, epsilon = _evaluate_iteration_state(
        ln_nk,
        ln_ntot,
        formula_matrix,
        b,
        T,
        ln_normalized_pressure,
        hvector,
        gk,
        lam,
        at_pi,
        pi_vector,
    )
    return ln_nk, ln_ntot, epsilon, gk, An


def profile_minimize_gibbs_iterations(
    state: ThermoState,
    ln_nk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    epsilon_crit: float = 1.0e-11,
    max_iter: int = 1000,
) -> Dict[str, object]:
    """Run the same Newton iterations in Python and time major sub-steps.

    This is a profiling helper, not the production solve path.
    """

    def _block(x):
        return jax.tree_util.tree_map(
            lambda y: y.block_until_ready() if hasattr(y, "block_until_ready") else y,
            x,
        )

    prepare_system = jax.jit(_prepare_iteration_system)
    solve_system = jax.jit(_solve_iteration_system)
    finish_solve = jax.jit(_finish_iteration_solve)
    apply_step = jax.jit(_apply_iteration_step)
    eval_state = jax.jit(_evaluate_iteration_state)

    hvector = hvector_func(state.temperature)
    _block(hvector)

    gk = _compute_gk(
        state.temperature,
        ln_nk_init,
        ln_ntot_init,
        hvector,
        state.ln_normalized_pressure,
    )
    An = formula_matrix @ jnp.exp(ln_nk_init)
    _block((gk, An))

    ln_nk = ln_nk_init
    ln_ntot = ln_ntot_init
    epsilon = jnp.asarray(jnp.inf, dtype=ln_nk.dtype)
    epsilon_host = float("inf")
    counter = 0

    # Compile the per-part kernels once outside the timed loop.
    _, _, resn0, bmatrix0, rhs0, scalar_rhs0 = prepare_system(
        ln_nk, ln_ntot, formula_matrix, state.element_vector, gk, An
    )
    _block((resn0, bmatrix0, rhs0, scalar_rhs0))
    binv_rhs0, binv_an0, schur_safe0 = solve_system(bmatrix0, rhs0, An, resn0)
    _block((binv_rhs0, binv_an0, schur_safe0))
    pi_vector0, delta_ln_ntot0 = finish_solve(
        binv_rhs0, binv_an0, An, scalar_rhs0, schur_safe0
    )
    _block((pi_vector0, delta_ln_ntot0))
    ln_nk1, ln_ntot1, lam1, _, at_pi1 = apply_step(
        ln_nk, ln_ntot, formula_matrix, gk, pi_vector0, delta_ln_ntot0
    )
    _block((ln_nk1, ln_ntot1))
    gk1, An1, epsilon1 = eval_state(
        ln_nk1,
        ln_ntot1,
        formula_matrix,
        state.element_vector,
        state.temperature,
        state.ln_normalized_pressure,
        hvector,
        gk,
        lam1,
        at_pi1,
        pi_vector0,
    )
    _block((gk1, An1, epsilon1))

    timings_s = {
        "prepare_system": 0.0,
        "linear_solve": 0.0,
        "finish_solve": 0.0,
        "step_update_damping": 0.0,
        "residual_evaluation": 0.0,
        "convergence_check": 0.0,
    }

    while True:
        t0 = time.perf_counter()
        keep_going = (epsilon_host > epsilon_crit) and (counter < max_iter)
        timings_s["convergence_check"] += time.perf_counter() - t0
        if not keep_going:
            break

        t0 = time.perf_counter()
        _, _, resn, bmatrix, rhs, scalar_rhs = prepare_system(
            ln_nk, ln_ntot, formula_matrix, state.element_vector, gk, An
        )
        _block((resn, bmatrix, rhs, scalar_rhs))
        timings_s["prepare_system"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        binv_rhs, binv_an, schur_safe = solve_system(bmatrix, rhs, An, resn)
        _block((binv_rhs, binv_an, schur_safe))
        timings_s["linear_solve"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        pi_vector, delta_ln_ntot = finish_solve(
            binv_rhs, binv_an, An, scalar_rhs, schur_safe
        )
        _block((pi_vector, delta_ln_ntot))
        timings_s["finish_solve"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        ln_nk, ln_ntot, lam, _, at_pi = apply_step(
            ln_nk, ln_ntot, formula_matrix, gk, pi_vector, delta_ln_ntot
        )
        _block((ln_nk, ln_ntot))
        timings_s["step_update_damping"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        gk, An, epsilon = eval_state(
            ln_nk,
            ln_ntot,
            formula_matrix,
            state.element_vector,
            state.temperature,
            state.ln_normalized_pressure,
            hvector,
            gk,
            lam,
            at_pi,
            pi_vector,
        )
        _block((gk, An, epsilon))
        epsilon_host = float(jax.device_get(epsilon))
        timings_s["residual_evaluation"] += time.perf_counter() - t0

        counter += 1

    total_profiled_s = sum(timings_s.values())
    average_iteration_s = total_profiled_s / counter if counter else 0.0
    average_breakdown_s = {
        key: value / counter if counter else 0.0 for key, value in timings_s.items()
    }

    return {
        "ln_nk": ln_nk,
        "ln_ntot": ln_ntot,
        "n_iter": counter,
        "final_residual": epsilon,
        "timings_s": timings_s,
        "average_iteration_s": average_iteration_s,
        "average_breakdown_s": average_breakdown_s,
        "total_profiled_s": total_profiled_s,
    }

def minimize_gibbs_core(
    state: ThermoState,
    ln_nk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    hvector_func,
    epsilon_crit: float = 1.0e-11,
    max_iter: int = 1000,
) -> Tuple[jnp.ndarray, float, int, jnp.ndarray]:
    """Compute log(number of species) by minimizing the Gibbs energy using the Lagrange multipliers method.

    Args:
        state: Thermodynamic state containing temperature, pressure, and element abundances.
        ln_nk_init: Initial log number of species vector (n_species,).
        ln_ntot_init: Initial log total number of species.
        formula_matrix: Stoichiometric formula matrix (n_elements, n_species).
        hvector: Chemical potential over RT vector (n_species,).
        epsilon_crit: Convergence tolerance for residual norm.
        max_iter: Maximum number of iterations allowed.

    Returns:
        Tuple containing:
            - Final log number of species vector (n_species,).
            - Final log total number of species.
            - Number of iterations performed.
            - Final residual norm used in convergence checks.
    """

    hvector = hvector_func(state.temperature)

    gk = _compute_gk(
        state.temperature,
        ln_nk_init,
        ln_ntot_init,
        hvector,
        state.ln_normalized_pressure,
    )
    An = formula_matrix @ jnp.exp(ln_nk_init)

    init_carry = (
        ln_nk_init,
        ln_ntot_init,
        gk,
        An,
        jnp.inf,
        0,
        formula_matrix,
        state.element_vector,
        state.temperature,
        state.ln_normalized_pressure,
        hvector,
        epsilon_crit,
        max_iter,
    )
    ln_nk, ln_tot, _, _, epsilon, counter, _, _, _, _, _, _, _ = while_loop(
        _minimize_gibbs_cond_fun,
        _minimize_gibbs_body_fun,
        init_carry,
    )
    return ln_nk, ln_tot, counter, epsilon


def _minimize_gibbs_solve_impl(
    state: ThermoState,
    ln_nk0: jnp.ndarray,
    ln_ntot0: float,
    formula_matrix: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    epsilon_crit: float,
    max_iter: int,
) -> jnp.ndarray:
    ln_nk, _, _, _ = minimize_gibbs_core(
        state,
        ln_nk0,
        ln_ntot0,
        formula_matrix,
        hvector_func,
        epsilon_crit,
        max_iter,
    )
    return ln_nk


# Keep the transformed solver at module scope so repeated calls reuse the same
# Python callable identity instead of rebuilding a new custom_vjp closure.
@partial(custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def _minimize_gibbs_solve(
    state: ThermoState,
    ln_nk0: jnp.ndarray,
    ln_ntot0: float,
    formula_matrix: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    epsilon_crit: float,
    max_iter: int,
) -> jnp.ndarray:
    return _minimize_gibbs_solve_impl(
        state,
        ln_nk0,
        ln_ntot0,
        formula_matrix,
        hvector_func,
        epsilon_crit,
        max_iter,
    )


def _minimize_gibbs_solve_fwd(
    state: ThermoState,
    ln_nk0: jnp.ndarray,
    ln_ntot0: float,
    formula_matrix: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    epsilon_crit: float,
    max_iter: int,
):
    ln_nk, ln_ntot, _, _ = minimize_gibbs_core(
        state,
        ln_nk0,
        ln_ntot0,
        formula_matrix,
        hvector_func,
        epsilon_crit,
        max_iter,
    )
    dfunc = jacrev(hvector_func)
    hdot = dfunc(state.temperature)
    residuals = (ln_nk, hdot, state.element_vector, ln_ntot)
    return ln_nk, residuals


def _minimize_gibbs_solve_bwd(
    formula_matrix: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    epsilon_crit: float,
    max_iter: int,
    res,
    g,
):
    del hvector_func, epsilon_crit, max_iter
    ln_nk, hdot, element_vector, ln_ntot = res

    nk = jnp.exp(ln_nk)
    ntot_result = jnp.exp(ln_ntot)

    Bmatrix = _A_diagn_At(nk, formula_matrix)
    c, lower = cho_factor(Bmatrix)
    alpha = cho_solve((c, lower), formula_matrix @ g)
    beta = cho_solve((c, lower), element_vector)
    beta_dot_b_element = jnp.vdot(beta, element_vector)

    cot_T = vjp_temperature(
        g,
        nk,
        formula_matrix,
        hdot,
        alpha,
        beta,
        element_vector,
        beta_dot_b_element,
    )
    cot_P = vjp_pressure(g, ntot_result, alpha, element_vector, beta_dot_b_element)
    cot_b = vjp_elements(g, alpha, beta, element_vector, beta_dot_b_element)
    # No gradients for initialization arguments.
    return (ThermoState(jnp.asarray(cot_T), jnp.asarray(cot_P), cot_b), None, None)


_minimize_gibbs_solve.defvjp(_minimize_gibbs_solve_fwd, _minimize_gibbs_solve_bwd)


def minimize_gibbs(
    state: ThermoState,
    ln_nk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    epsilon_crit: float = 1.0e-11,
    max_iter: int = 1000,
) -> jnp.ndarray:
    """Compute log(number of species) by minimizing the Gibbs energy using the Lagrange multipliers method.

    Args:
        state: Thermodynamic state containing temperature, pressure, and element abundances.
        ln_nk_init: Initial natural log number of species vector (n_species,).
        ln_ntot_init: Initial natural log total number of species.
        formula_matrix: Stoichiometric formula matrix (n_elements, n_species).
        hvector_func: Function that returns chemical potential over RT vector (n_species,).
        epsilon_crit: Convergence tolerance for residual norm.
        max_iter: Maximum number of iterations allowed.

    Returns:
        Final log number of species vector (n_species,).
    """
    # Treat initial guesses as non-differentiable inputs
    ln_nk0 = stop_gradient(ln_nk_init)
    ln_ntot0 = stop_gradient(ln_ntot_init)
    return _minimize_gibbs_solve(
        state,
        ln_nk0,
        ln_ntot0,
        formula_matrix,
        hvector_func,
        epsilon_crit,
        max_iter,
    )


def minimize_gibbs_with_diagnostics(
    state: ThermoState,
    ln_nk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    epsilon_crit: float = 1.0e-11,
    max_iter: int = 1000,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Run Gibbs minimization and return lightweight convergence diagnostics."""
    ln_nk, _, n_iter, final_residual = minimize_gibbs_core(
        state,
        ln_nk_init,
        ln_ntot_init,
        formula_matrix,
        hvector_func,
        epsilon_crit,
        max_iter,
    )
    epsilon_crit_used = jnp.asarray(epsilon_crit, dtype=final_residual.dtype)
    max_iter_used = jnp.asarray(max_iter, dtype=n_iter.dtype)
    converged = final_residual <= epsilon_crit_used
    hit_max_iter = (n_iter >= max_iter_used) & (~converged)

    diagnostics = {
        "n_iter": n_iter,
        "converged": converged,
        "hit_max_iter": hit_max_iter,
        "final_residual": final_residual,
        "epsilon_crit": epsilon_crit_used,
        "max_iter": max_iter_used,
    }
    return ln_nk, diagnostics
