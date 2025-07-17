import jax.numpy as jnp
from jax import custom_vjp
from jax.lax import while_loop
from jax import jit
from typing import Tuple
from exogibbs.optimize.lagrange.core import _A_diagn_At
from exogibbs.optimize.lagrange.core import _compute_gk


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
        nk: Number density vector (n_species,) for k-th iteration.
        ntotk: Total number density for k-th iteration.
        formula_matrix: Formula matrix for stoichiometric constraints (n_elements, n_species).
        b: Element abundance vector (n_elements, ).
        gk: gk vector (n_species,) for k-th iteration.
        An: formula_matrix @ nk vector (n_elements, ).

    Returns:
        Tuple containing:
            - The pi vector (nspecies, ).
            - The update of the  log total number density (delta_ln_ntot).
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
    normalized_pressure,
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
    gk = _compute_gk(T, ln_nk, ln_ntot, hvector, normalized_pressure)
    An = formula_matrix @ nk
    epsilon = compute_residuals(nk, ntot, formula_matrix, b, gk, An, pi_vector)
    return ln_nk, ln_ntot, epsilon, gk, An


def minimize_gibbs_core(
    temperature: float,
    normalized_pressure: float,
    b_element_vector: jnp.ndarray,
    ln_nk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    hvector: jnp.ndarray,
    epsilon_crit: float = 1.0e-11,
    max_iter: int = 1000,
) -> Tuple[jnp.ndarray, float, int]:
    """Compute log(number of species) by minimizing the Gibbs energy using the Lagrange multipliers method.

    Args:
        temperature: Temperature in Kelvin.
        normalized_pressure: Pressure normalized by reference pressure (P/Pref).
        b_element_vector: Element abundance vector (n_elements,).
        ln_nk_init: Initial log number density vector (n_species,).
        ln_ntot_init: Initial log total number density.
        formula_matrix: Stoichiometric formula matrix (n_elements, n_species).
        hvector: Chemical potential over RT vector (n_species,).
        epsilon_crit: Convergence tolerance for residual norm.
        max_iter: Maximum number of iterations allowed.

    Returns:
        Tuple containing:
            - Final log number density vector (n_species,).
            - Final log total number density.
            - Number of iterations performed.
    """

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
            normalized_pressure,
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
        normalized_pressure,
    )
    An = formula_matrix @ jnp.exp(ln_nk_init)

    ln_nk, ln_tot, _, _, _, counter = while_loop(
        cond_fun, body_fun, (ln_nk_init, ln_ntot_init, gk, An, jnp.inf, 0)
    )
    return ln_nk, ln_tot, counter


# @custom_vjp
def minimize_gibbs(
    temperature,
    normalized_pressure,
    b_element_vector,
    ln_nk_init,
    ln_ntot_init,
    formula_matrix,
    chemical_potential_vec,
    epsilon_crit=1.0e-11,
    max_iter=1000,
):
    """compute log(number of species) by minimizing the Gibbs energy using the Lagrange multipliers method."""

    ln_nk, _, _ = minimize_gibbs_core(
        temperature,
        normalized_pressure,
        b_element_vector,
        ln_nk_init,
        ln_ntot_init,
        formula_matrix,
        chemical_potential_vec,
        epsilon_crit,
        max_iter,
    )

    return ln_nk


"""
def minimize_gibbs_fwd(
    temperature,
    normalized_pressure,
    b_element_vector,
    ln_nk_init,
    ln_ntot_init,
    formula_matrix,
    chemical_potential_vec,
    epsilon_crit=1.0e-11,
    max_iter=1000,
):
    ln_nk, ln_ntot, _ = minimize_gibbs(
        temperature,
        normalized_pressure,
        b_element_vector,
        ln_nk_init,
        ln_ntot_init,
        formula_matrix,
        chemical_potential_vec,
        epsilon_crit,
        max_iter,
    )

    return ln_nk, (None, )
"""

if __name__ == "__main__":
    from exogibbs.equilibrium.gibbs import interpolate_hvector_one
    from exogibbs.io.load_data import get_data_filepath
    from exogibbs.io.load_data import DEFAULT_JANAF_GIBBS_MATRICES
    import numpy as np

    from jax import config

    config.update("jax_enable_x64", True)

    path = get_data_filepath(DEFAULT_JANAF_GIBBS_MATRICES)
    gibbs_matrices = np.load(path, allow_pickle=True)["arr_0"].item()

    kJtoJ = 1000.0  # conversion factor from kJ to J
    T_h_table = gibbs_matrices["H1"]["T(K)"].to_numpy()
    mu_h_table = gibbs_matrices["H1"]["delta-f G"].to_numpy() * kJtoJ
    T_h2_table = gibbs_matrices["H2"]["T(K)"].to_numpy()
    mu_h2_table = gibbs_matrices["H2"]["delta-f G"].to_numpy() * kJtoJ

    """
    def mu_h(T):
        return interpolate_chemical_potential_one(T, T_h_table, mu_h_table, order=2)

    def mu_h2(T):
        return interpolate_chemical_potential_one(T, T_h2_table, mu_h2_table, order=2)
    """

    def hv_h(T):
        return interpolate_hvector_one(T, T_h_table, mu_h_table)

    def hv_h2(T):
        return interpolate_hvector_one(T, T_h2_table, mu_h2_table)

    def compute_k(P, T, Pref=1.0):
        delta_h = hv_h2(T) - 2.0 * hv_h(T)
        return np.exp(-delta_h) * P / Pref

    def nh(k):
        return 1.0 / np.sqrt(4.0 * k + 1.0)

    def nh2(k):
        return 0.5 * (1.0 - nh(k))

    def ntotal(k):
        return nh(k) + nh2(k)

    def vmr_h(k):
        return nh(k) / ntotal(k)

    def vmr_h2(k):
        return nh2(k) / ntotal(k)

    formula_matrix = jnp.array([[1.0, 2.0]])
    temperature = 3500.0
    P = 1.0  # bar
    P_ref = 1.0  # bar

    normalized_pressure = P / P_ref
    ln_nk = jnp.array([0.0, 0.0])
    ln_ntot = 0.0
    # chemical_potential_vector = jnp.array([hv_h(temperature), hv_h2(temperature)])
    hvector = jnp.array([hv_h(temperature), hv_h2(temperature)])
    b_element_vector = jnp.array([1.0])  # Assuming no element abundance constraints

    # minimize Gibbs energy
    ln_nk, ln_ntot, counter = minimize_gibbs_core(
        temperature,
        normalized_pressure,
        b_element_vector,
        ln_nk,
        ln_ntot,
        formula_matrix,
        hvector,
    )

    k = compute_k(P, temperature, P_ref)
    diff = jnp.log(nh(k)) - ln_nk[0]
    diff2 = jnp.log(nh2(k)) - ln_nk[1]
    print(f"Difference for H: {diff}, Difference for H2: {diff2}")

    # derivative
    from jax import grad
    from jax import vmap
    from exogibbs.optimize.lagrange.derivative import derivative_temperature

    def dot_hv_h(T):
        return grad(hv_h)(T)

    def dot_hv_h2(T):
        return grad(hv_h2)(T)

    def delta(T):
        return 2.0 * hv_h(T) - hv_h2(T)

    def delta_dT(Tarr):
        return vmap(grad(delta),in_axes=0)(Tarr)

    def ln_nH_dT(Tarr):
        k = compute_k(P, Tarr, P_ref)
        return -2.0 * nh2(k) * ntotal(k) * delta_dT(Tarr)

    def ln_nH2_dT(Tarr):
        k = compute_k(P, Tarr, P_ref)
        return nh(k) * ntotal(k) * delta_dT(Tarr)

    hdot = jnp.array([dot_hv_h(temperature), dot_hv_h2(temperature)])
    nk = jnp.exp(ln_nk)
    An = formula_matrix @ nk

    ln_nspecies_dT = derivative_temperature(nk, formula_matrix, hdot, An)

    diff = ln_nH_dT(jnp.array([temperature]))[0] - ln_nspecies_dT[0]
    diff2 = ln_nH2_dT(jnp.array([temperature]))[0] - ln_nspecies_dT[1]
    print(f"Difference for H: {diff}, Difference for H2: {diff2}")

    # does not work with vmap
    from jax import vmap, jit

    vmap_minimize_gibbs = jit(
        vmap(minimize_gibbs_core, in_axes=(0, None, None, None, None, None, 0))
    )
    vmap_derivative_temperature = jit(
        vmap(derivative_temperature, in_axes=(0, None, 0, 0))
    )

    Tarr = jnp.linspace(100.0, 6000.0, 300)

    muH = hv_h(Tarr)
    muH2 = hv_h2(Tarr)
    hvector = jnp.array([muH, muH2]).T

    ln_nk = jnp.array([0.0, 0.0])
    ln_ntot = 0.0

    ln_nk_arr, _, counter_arr = vmap_minimize_gibbs(
        Tarr,
        normalized_pressure,
        b_element_vector,
        ln_nk,
        ln_ntot,
        formula_matrix,
        hvector,
    )
    print(ln_nk_arr.shape)

    karr = compute_k(P, Tarr, P_ref)

    n_H = jnp.exp(ln_nk_arr[:, 0])
    n_H2 = jnp.exp(ln_nk_arr[:, 1])
    ntot = n_H + n_H2
    vmrH = n_H / ntot
    vmrH2 = n_H2 / ntot

    #derivative 
    vmap_dot_hv_h = vmap(dot_hv_h, in_axes=0)
    vmap_dot_hv_h2 = vmap(dot_hv_h2, in_axes=0)
    vdot_hv_h = vmap_dot_hv_h(Tarr)
    vdot_hv_h2 = vmap_dot_hv_h2(Tarr)

    
    hdot_arr = jnp.array([vdot_hv_h, vdot_hv_h2]).T
    
    nk_arr = jnp.exp(ln_nk_arr)
    An_arr = jnp.einsum("ij,kj -> ki", formula_matrix, nk_arr)
    ln_nspecies_dT = vmap_derivative_temperature(nk_arr, formula_matrix, hdot_arr, An_arr)
    
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(412)
    plt.plot(Tarr, vmrH, label="H", alpha=0.5)
    plt.plot(Tarr, vmrH2, label="H2", alpha=0.5)
    plt.plot(Tarr, vmr_h(karr), ls="dashed", label="analytical H")
    plt.plot(Tarr, vmr_h2(karr), ls="dashed", label="analytical H2")
    plt.ylabel("Species VMR")
    plt.legend()
    ax = fig.add_subplot(413)
    plt.plot(Tarr, vmrH, label="H", alpha=0.5)
    plt.plot(Tarr, vmrH2, label="H2", alpha=0.5)
    plt.plot(Tarr, vmr_h(karr), ls="dashed", label="analytical H")
    plt.plot(Tarr, vmr_h2(karr), ls="dashed", label="analytical H2")
    plt.yscale("log")
    plt.ylabel("Species VMR")
    plt.legend()
    ax = fig.add_subplot(411)
    plt.plot(Tarr, counter_arr, label="Counter")
    plt.ylabel("# of iterations")
    ax = fig.add_subplot(414)
    plt.plot(Tarr, jnp.abs(ln_nspecies_dT[:,0]), label="H", alpha=0.5)
    plt.plot(Tarr, jnp.abs(ln_nspecies_dT[:,1]), label="H2", alpha=0.5)
    plt.plot(Tarr, jnp.abs(ln_nH_dT(Tarr)), ls="dashed", label="analytical H")
    plt.plot(Tarr, jnp.abs(ln_nH2_dT(Tarr)), ls="dashed", label="analytical H2")
    plt.yscale("log")
    plt.ylabel("|Derivative of log number|")
    plt.legend()
    plt.xlabel("Temperature (K)")
    
    
    plt.savefig("gibbs_minimization.png")
    plt.show()
