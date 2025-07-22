import jax.numpy as jnp
from jax import custom_vjp
from jax import jacrev
from jax.lax import while_loop
from jax import jit
from functools import partial
from typing import Tuple, Callable
from exogibbs.optimize.lagrange.core import _A_diagn_At
from exogibbs.optimize.lagrange.core import _compute_gk
from exogibbs.optimize.lagrange.derivative import derivative_temperature
from exogibbs.optimize.lagrange.derivative import derivative_pressure


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
    AnAt = _A_diagn_At(nk, formula_matrix)
    Angk = formula_matrix @ (gk * nk)
    ngk = jnp.dot(nk, gk)

    assemble_mat = jnp.block([[AnAt, An[:, None]], [An[None, :], jnp.array([[resn]])]])
    assemble_vec = jnp.concatenate([Angk + b - An, jnp.array([ngk - resn])])
    assemble_variable = jnp.linalg.solve(assemble_mat, assemble_vec)
    return assemble_variable[:-1], assemble_variable[-1]


def compute_residuals(
    nk: jnp.ndarray,
    ntotk: float,
    formula_matrix: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    An: jnp.ndarray,
    pi_vector: jnp.ndarray,
) -> float:

    ress = nk * (formula_matrix.T @ pi_vector - gk)
    ress_squared = jnp.dot(ress, ress)

    An_b = An - b
    resj_squared = jnp.dot(An_b, An_b)

    resn = jnp.sum(nk) - ntotk
    resn_squared = jnp.dot(resn, resn)
    return jnp.sqrt(ress_squared + resj_squared + resn_squared)


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
    under_relax = 0.1  # need to reconsider
    ln_ntot += under_relax * delta_ln_ntot
    ln_nk += under_relax * delta_ln_nk

    # computes new gk,An and residuals
    nk = jnp.exp(ln_nk)
    ntot = jnp.exp(ln_ntot)
    gk = _compute_gk(T, ln_nk, ln_ntot, hvector, ln_normalized_pressure)
    An = formula_matrix @ nk
    epsilon = compute_residuals(nk, ntot, formula_matrix, b, gk, An, pi_vector)
    return ln_nk, ln_ntot, epsilon, gk, An


def minimize_gibbs_core(
    temperature: float,
    ln_normalized_pressure: float,
    b_element_vector: jnp.ndarray,
    ln_nk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    hvector_func,
    epsilon_crit: float = 1.0e-11,
    max_iter: int = 1000,
) -> Tuple[jnp.ndarray, float, int]:
    """Compute log(number of species) by minimizing the Gibbs energy using the Lagrange multipliers method.

    Args:
        temperature: Temperature in Kelvin.
        ln_normalized_pressure: natural log of pressure normalized by reference pressure (P/Pref).
        b_element_vector: Element abundance vector (n_elements,).
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
    """

    hvector = hvector_func(temperature)

    def cond_fun(carry):
        _, _, _, _, epsilon, counter = carry
        return (epsilon > epsilon_crit) & (counter < max_iter)

    def body_fun(carry):
        ln_nk, ln_ntot, gk, An, _, counter = carry
        ln_nk_new, ln_ntot_new, epsilon, gk, An = update_all(
            ln_nk,
            ln_ntot,
            formula_matrix,
            b_element_vector,
            temperature,
            ln_normalized_pressure,
            hvector,
            gk,
            An,
        )
        return ln_nk_new, ln_ntot_new, gk, An, epsilon, counter + 1

    gk = _compute_gk(
        temperature,
        ln_nk_init,
        ln_ntot_init,
        hvector,
        ln_normalized_pressure,
    )
    An = formula_matrix @ jnp.exp(ln_nk_init)

    ln_nk, ln_tot, _, _, _, counter = while_loop(
        cond_fun, body_fun, (ln_nk_init, ln_ntot_init, gk, An, jnp.inf, 0)
    )
    return ln_nk, ln_tot, counter


@partial(custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8))
def minimize_gibbs(
    temperature: float,
    ln_normalized_pressure: float,
    b_element_vector: jnp.ndarray,
    ln_nk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    epsilon_crit: float = 1.0e-11,
    max_iter: int = 1000,
) -> jnp.ndarray:
    """Compute log(number of species) by minimizing the Gibbs energy using the Lagrange multipliers method.

    Args:
        temperature: Temperature in Kelvin.
        ln_normalized_pressure: natural log pressure normalized by the reference pressure jnp.log(P/Pref), or use optimize.lagrange.core.compute_ln_normalized_pressure.
        b_element_vector: Element abundance vector (n_elements,).
        ln_nk_init: Initial log number of species vector (n_species,).
        ln_ntot_init: Initial log total number of species.
        formula_matrix: Stoichiometric formula matrix (n_elements, n_species).
        hvector_func: Function that returns chemical potential over RT vector (n_species,).
        epsilon_crit: Convergence tolerance for residual norm.
        max_iter: Maximum number of iterations allowed.

    Returns:
        Final log number of species vector (n_species,).
    """
    
    ln_nk, _, _ = minimize_gibbs_core(
        temperature,
        ln_normalized_pressure,
        b_element_vector,
        ln_nk_init,
        ln_ntot_init,
        formula_matrix,
        hvector_func,
        epsilon_crit,
        max_iter,
    )

    return ln_nk


def minimize_gibbs_fwd(
    temperature,
    ln_normalized_pressure,
    b_element_vector,
    ln_nk_init,
    ln_ntot_init,
    formula_matrix,
    hvector_func,
    epsilon_crit=1.0e-11,
    max_iter=1000,
):
    ln_nk, ln_ntot, _ = minimize_gibbs_core(
        temperature,
        ln_normalized_pressure,
        b_element_vector,
        ln_nk_init,
        ln_ntot_init,
        formula_matrix,
        hvector_func,
        epsilon_crit,
        max_iter,
    )
    dfunc = jacrev(hvector_func)
    hdot = dfunc(temperature)

    residuals = (ln_nk, hdot, b_element_vector, ln_ntot)
    return ln_nk, residuals


def minimize_gibbs_bwd(
    ln_nk_init,
    ln_ntot_init,
    formula_matrix,
    hvector_func,
    epsilon_crit,
    max_iter,
    res,
    g,
):

    ln_nk, hdot, b_element_vector, ln_ntot = res
    nk_result = jnp.exp(ln_nk)
    ntot_result = jnp.exp(ln_ntot)
    Bmatrix = _A_diagn_At(nk_result, formula_matrix)
    nk_cdot_hdot = jnp.dot(nk_result, hdot)    
    
    #temperature derivative
    ln_nspecies_dT = derivative_temperature(nk_result, formula_matrix, hdot, nk_cdot_hdot, Bmatrix, b_element_vector)
    cot_T = jnp.dot(ln_nspecies_dT, g)

    #pressure derivative
    ln_nspecies_dlogp = derivative_pressure(ntot_result, formula_matrix, Bmatrix, b_element_vector)
    cot_P = jnp.dot(ln_nspecies_dlogp, g)
    return (cot_T, cot_P, None)


minimize_gibbs.defvjp(minimize_gibbs_fwd, minimize_gibbs_bwd)

if __name__ == "__main__":
    from exogibbs.test.analytic_hsystem import HSystem
    from exogibbs.optimize.lagrange.core import compute_ln_normalized_pressure
    import numpy as np
    from jax import jacrev
    from jax import config

    config.update("jax_enable_x64", True)

    # Initialize the analytic H system
    hsystem = HSystem()

    formula_matrix = jnp.array([[1.0, 2.0]])
    temperature = 3500.0
    P = 1.0  # bar
    Pref = 1.0  # bar, reference pressure
    ln_normalized_pressure = compute_ln_normalized_pressure(P, Pref)

    ln_nk = jnp.array([0.0, 0.0])
    ln_ntot = 0.0

    def hvector_func(temperature):
        return jnp.array([hsystem.hv_h(temperature), hsystem.hv_h2(temperature)])

    b_element_vector = jnp.array([1.0])  # Total hydrogen nuclei abundance

    # set criterions
    epsilon_crit = 1e-11
    max_iter = 1000

    # Run Gibbs minimization
    ln_nk_result, ln_ntot_result, counter = minimize_gibbs_core(
        temperature,
        ln_normalized_pressure,
        b_element_vector,
        ln_nk,
        ln_ntot,
        formula_matrix,
        hvector_func,
        epsilon_crit=epsilon_crit,
        max_iter=max_iter,
    )

    ln_nk_result = minimize_gibbs(
        temperature,
        ln_normalized_pressure,
        b_element_vector,
        ln_nk,
        ln_ntot,
        formula_matrix,
        hvector_func,
        epsilon_crit=epsilon_crit,
        max_iter=max_iter,
    )

    dln_dT = jacrev(
        lambda temperature_in: minimize_gibbs(
            temperature_in,
            ln_normalized_pressure,
            b_element_vector,
            ln_nk,
            ln_ntot,
            formula_matrix,
            hvector_func,
            epsilon_crit=epsilon_crit,
            max_iter=max_iter,
        )
    )(temperature)
    print(f"dln_dT: {dln_dT}")

    

    # Compare with analytical solution
    k = hsystem.compute_k(ln_normalized_pressure, temperature)
    refH = hsystem.ln_nH_dT(jnp.array([temperature]), ln_normalized_pressure)[0]
    refH2 = hsystem.ln_nH2_dT(jnp.array([temperature]), ln_normalized_pressure)[0]
    print(f"Reference ln_nH_dT: {refH}, Reference ln_nH2_dT: {refH2}")
    diff = refH - dln_dT[0]
    diff2 = refH2 - dln_dT[1]
    print(f"Difference for H: {diff}, Difference for H2: {diff2}")


    dln_dlogp = jacrev(
        lambda ln_normalized_pressure: minimize_gibbs(
            temperature,
            ln_normalized_pressure,
            b_element_vector,
            ln_nk,
            ln_ntot,
            formula_matrix,
            hvector_func,
            epsilon_crit=epsilon_crit,
            max_iter=max_iter,
        )
    )(ln_normalized_pressure)
    print(f"dln_dlogp: {dln_dlogp}")
    refH = hsystem.ln_nH_dlogp(jnp.array([temperature]), ln_normalized_pressure)[0]
    refH2 = hsystem.ln_nH2_dlogp(jnp.array([temperature]), ln_normalized_pressure)[0]
    print(f"Reference ln_nH_dlogp: {refH}, Reference ln_nH2_dlogp: {refH2}")
    diff = refH - dln_dlogp[0]
    diff2 = refH2 - dln_dlogp[1]
    print(f"Difference for H: {diff}, Difference for H2: {diff2}")
    exit()

    # Vectorized computation over temperature range
    from jax import vmap, jit

    Tarr = jnp.linspace(300.0, 6000.0, 300)
    
    ln_nk_init = jnp.array([0.0, 0.0])
    ln_ntot_init = 0.0

    vmap_minimize_gibbs = vmap(
        minimize_gibbs, in_axes=(0, None, None, None, None, None, None, None, None)
    )

    ln_nk_arr = vmap_minimize_gibbs(
        Tarr,
        ln_normalized_pressure,
        b_element_vector,
        ln_nk_init,
        ln_ntot_init,
        formula_matrix,
        hvector_func,
        epsilon_crit,
        max_iter,
    )
    vmap_minimize_gibbs_dT = vmap(
        jacrev(minimize_gibbs),
        in_axes=(0, None, None, None, None, None, None, None, None),
    )
    dln_dT_arr = vmap_minimize_gibbs_dT(
        Tarr,
        ln_normalized_pressure,
        b_element_vector,
        ln_nk_init,
        ln_ntot_init,
        formula_matrix,
        hvector_func,
        epsilon_crit,
        max_iter,
    )

    # vmapped analytical computation
    karr = vmap(hsystem.compute_k, in_axes=(None, 0))(ln_normalized_pressure, Tarr)

    n_H = jnp.exp(ln_nk_arr[:, 0])
    n_H2 = jnp.exp(ln_nk_arr[:, 1])
    ntot = n_H + n_H2
    vmrH = n_H / ntot
    vmrH2 = n_H2 / ntot


    diffH = vmrH - vmap(hsystem.vmr_h)(karr)
    diffH2 = vmrH2 - vmap(hsystem.vmr_h2)(karr)

    diff_dT_H = dln_dT_arr[:, 0] - hsystem.ln_nH_dT(Tarr, ln_normalized_pressure)
    diff_dT_H2 = dln_dT_arr[:, 1] - hsystem.ln_nH2_dT(Tarr, ln_normalized_pressure)

    print(f"Max difference in VMR for H: {jnp.max(jnp.abs(diffH))}")
    print(f"Max difference in VMR for H2: {jnp.max(jnp.abs(diffH2))}")
    print(f"Max difference in dln_dT for H: {jnp.max(jnp.abs(diff_dT_H))}")
    print(f"Max difference in dln_dT for H2: {jnp.max(jnp.abs(diff_dT_H2))}")

    # derivative computation

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(311)
    plt.plot(Tarr, vmrH, label="H", alpha=0.5)
    plt.plot(Tarr, vmrH2, label="H2", alpha=0.5)
    plt.plot(Tarr, vmap(hsystem.vmr_h)(karr), ls="dashed", label="analytical H")
    plt.plot(Tarr, vmap(hsystem.vmr_h2)(karr), ls="dashed", label="analytical H2")
    plt.ylabel("Species VMR")
    plt.legend()
    ax = fig.add_subplot(312)
    plt.plot(Tarr, vmrH, label="H", alpha=0.5)
    plt.plot(Tarr, vmrH2, label="H2", alpha=0.5)
    plt.plot(Tarr, vmap(hsystem.vmr_h)(karr), ls="dashed", label="analytical H")
    plt.plot(Tarr, vmap(hsystem.vmr_h2)(karr), ls="dashed", label="analytical H2")
    plt.yscale("log")
    plt.ylabel("Species VMR")
    plt.legend()
    ax = fig.add_subplot(313)
    plt.plot(Tarr, jnp.abs(dln_dT_arr[:, 0]), label="H", alpha=0.5)
    plt.plot(Tarr, jnp.abs(dln_dT_arr[:, 1]), label="H2", alpha=0.5)
    plt.plot(
        Tarr, jnp.abs(hsystem.ln_nH_dT(Tarr, ln_normalized_pressure)), ls="dashed", label="analytical H"
    )
    plt.plot(
        Tarr, jnp.abs(hsystem.ln_nH2_dT(Tarr, ln_normalized_pressure)), ls="dashed", label="analytical H2"
    )
    plt.yscale("log")
    plt.ylabel("|Derivative of log number|")
    plt.legend()
    plt.xlabel("Temperature (K)")

    plt.savefig("gibbs_minimization.png")
    plt.show()
