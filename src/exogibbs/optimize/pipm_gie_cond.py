import jax.numpy as jnp
from jax import debug as jdebug
from jax.lax import while_loop
from typing import Tuple, Optional

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
        [-gk, nuk / mk - hvector_cond, delta_bk_hat, jnp.array([ntotk - resn])]
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

    nuk = jnp.exp(epsilon)
    bk = formula_matrix @ jnp.exp(ln_nk)
    
    delta_ln_nk, delta_ln_mk, pi_vector, delta_ln_ntot = solve_gibbs_iteration_equations_cond(
        jnp.exp(ln_nk),
        jnp.exp(ln_mk),
        jnp.exp(ln_ntot),
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        bk,
        nuk,
    )

    
    
    #MAX_STEP_M_UP = 0.1  # do not update larger than ln(m) 0.1e ~ 10%
    #MAX_STEP_M_LOW = 0.1
    #delta_ln_mk = jnp.clip(raw_delta_ln_mk, -MAX_STEP_M_LOW, MAX_STEP_M_UP)
    
    # relaxation and update
    # lam = 0.0001  # need to reconsider

    lam1_gas = stepsize_cea_gas(delta_ln_nk, delta_ln_ntot, ln_nk, ln_ntot)
    lam1_cond = stepsize_cond_heurstic(delta_ln_mk)
    lam2_cond = stepsize_sk(delta_ln_mk, ln_mk, epsilon)
    lam = jnp.minimum(1.0, jnp.minimum(lam1_gas, jnp.minimum(lam1_cond, lam2_cond)))
    # Do not force a minimum step; allow very small values when needed.
    lam = jnp.clip(lam, 0.0, 1.0)

    ln_ntot += lam * delta_ln_ntot
    ln_nk += lam * delta_ln_nk
    ln_mk += lam * delta_ln_mk

    # clip
    # ln_nk = jnp.clip(ln_nk, LOG_MIN, LOG_MAX)
    # ln_ntot = jnp.clip(ln_ntot, LOG_MIN, LOG_MAX)
    # ln_mk = jnp.clip(ln_mk, LOG_MIN, LOG_MAX)

    # computes new gk,An and residuals

    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntot = jnp.exp(ln_ntot)
    gk = _compute_gk(T, ln_nk, ln_ntot, hvector, ln_normalized_pressure)
    An = formula_matrix @ nk
    Am = formula_matrix_cond @ mk

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
        pi_vector,
    )
    return ln_nk, ln_mk, ln_ntot, gk, An, Am, residual


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
