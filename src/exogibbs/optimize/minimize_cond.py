import jax.numpy as jnp
from jax import custom_vjp
from jax import jacrev
from jax import jit
from jax.lax import while_loop, stop_gradient
from jax.scipy.linalg import cho_factor
from jax.scipy.linalg import cho_solve
from functools import partial
from typing import Tuple, Callable

from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.core import _A_diagn_At
from exogibbs.optimize.core import _compute_gk


def solve_gibbs_iteration_equations_cond(
    nk: jnp.ndarray,
    mk: jnp.ndarray,
    etak: jnp.ndarray,
    ntotk: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    bk: jnp.ndarray,
    ck: jnp.ndarray,
    sk: jnp.ndarray,
    eta_clip: float = 1e-30,
) -> Tuple[jnp.ndarray, float]:
    """
    Solve the Gibbs iteration equations with condensates using the Lagrange multipliers.
    This function computes the matrix and vector to solve the system of equations
    that arises from the Gibbs energy minimization problem.

    Args:
        nk: number of species vector (n_species,) for k-th iteration.
        ntotk: Total number of species for k-th iteration.
        formula_matrix: Gas Formula matrix for stoichiometric constraints (n_elements, n_species).
        formula_matrix_cond: Condensates Formula matrix for stoichiometric constraints (n_elements, n_cond).
        b: Element abundance vector (n_elements, ).
        gk: gk vector (n_species,) for k-th iteration.
        bk: (gas) formula_matrix @ nk vector (n_elements, ).
        ck: chemical_potentials for condensates divided by RT (n_cond, )
        sk: rk + rhok - epsilon (n_cond, )

    Returns:
        Tuple containing:
            - The pi vector (nspecies, ).
            - The update of the  log total number of species (delta_ln_ntot).
    """

    etak_clipped = jnp.clip(etak, a_min=eta_clip)
    jk = mk / etak_clipped

    resn = jnp.sum(nk) - ntotk
    Qk = _A_diagn_At(nk, formula_matrix) + _A_diagn_At(jk, formula_matrix_cond)
    Angk = formula_matrix @ (gk * nk)
    ngk = jnp.dot(nk, gk)

    delta_bk_hat = b - (bk + formula_matrix_cond @ mk)
    condvec = formula_matrix_cond @ (jk * ck + mk * sk - mk)

    assemble_mat = jnp.block([[Qk, bk[:, None]], [bk[None, :], jnp.array([[resn]])]])
    assemble_vec = jnp.concatenate(
        [Angk + condvec + delta_bk_hat, jnp.array([ngk - resn])]
    )
    assemble_variable = jnp.linalg.solve(assemble_mat, assemble_vec)
    return assemble_variable[:-1], assemble_variable[-1]


def compute_residuals(
    nk: jnp.ndarray,
    mk: jnp.ndarray,
    ntotk: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    ck: jnp.ndarray,
    sk: jnp.ndarray,
    An: jnp.ndarray,
    Am: jnp.ndarray,
    pi_vector: jnp.ndarray,
) -> float:

    ress = nk * (formula_matrix.T @ pi_vector - gk)
    ress_squared = jnp.dot(ress, ress)

    resc = mk * (formula_matrix_cond.T @ pi_vector - ck)
    resc_squared = jnp.dot(resc, resc)

    deltabhat = An + Am - b
    resj_squared = jnp.dot(deltabhat, deltabhat)

    resr_squared = jnp.dot(sk, sk)

    resn = jnp.sum(nk) - ntotk
    resn_squared = jnp.dot(resn, resn)

    return jnp.sqrt(ress_squared + resc_squared + resj_squared + resr_squared + resn_squared)


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

    pi_vector, delta_ln_ntot = solve_gibbs_iteration_equations(
        jnp.exp(ln_nk), jnp.exp(ln_ntot), formula_matrix, b, gk, An
    )
    delta_ln_nk = formula_matrix.T @ pi_vector + delta_ln_ntot - gk

    # relaxation and update
    #lam = 0.1  # need to reconsider
    lam = _cea_lambda(delta_ln_nk, delta_ln_ntot, ln_nk, ln_ntot)
    ln_ntot += lam * delta_ln_ntot
    ln_nk += lam * delta_ln_nk

    # computes new gk,An and residuals
    nk = jnp.exp(ln_nk)
    ntot = jnp.exp(ln_ntot)
    gk = _compute_gk(T, ln_nk, ln_ntot, hvector, ln_normalized_pressure)
    An = formula_matrix @ nk
    residual = compute_residuals(nk, ntot, formula_matrix, b, gk, An, pi_vector)
    return ln_nk, ln_ntot, residual, gk, An
