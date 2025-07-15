import jax.numpy as jnp
from jax import custom_vjp
from jax.lax import while_loop
from typing import Tuple
from exogibbs.utils.constants import R_gas_constant_si

# def lagrange_newton(T, P, T_table, mu_table, x0, formula_matrix, b, stepsize=1e-2, num_steps=1000):


def _A_diagn_At(number_density_vector, formula_matrix):
    return jnp.einsum(
        "ik,k,jk->ij", formula_matrix, number_density_vector, formula_matrix
    )


# @custom_vjp


def compute_gk(
    T: float,
    ln_nk: jnp.ndarray,
    ln_ntot: float,
    chemical_potential_vec: jnp.ndarray,
    normalized_pressure: float,
) -> jnp.ndarray:
    """computes gk vector for the Gibbs iteration

    Args:
        T: temperature (K)
        ln_nk: log of number density vector (n_species, )
        ln_ntot: log of total number density
        chemical_potential_over_RT_vec: chemical potential over RT vector (n_species, )
        normalized_pressure: normalized pressure P/Pref

    Returns:
        chemical potential vector (n_species, )
    """
    RT = R_gas_constant_si * T
    return chemical_potential_vec / RT + ln_nk - ln_ntot + jnp.log(normalized_pressure)


def solve_gibbs_iteration_equations(
    nk: jnp.ndarray,
    ntotk: float,
    formula_matrix: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    An: jnp.ndarray,
) -> Tuple[jnp.ndarray, float]:
    """
    Solve the Gibbs iteration equations for the Lagrange multipliers.
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
    print(ress_squared, resj_squared, resn_squared)
    return jnp.sqrt(ress_squared + resj_squared + resn_squared)


def update_ln_nk(
    pi_vector: jnp.ndarray,
    delta_ln_ntot: float,
    formula_matrix: jnp.ndarray,
    gk: jnp.ndarray,
) -> jnp.ndarray:

    return formula_matrix.T @ pi_vector + delta_ln_ntot - gk


def update_all(
    ln_nk,
    ln_ntot,
    formula_matrix,
    b,
    T,
    normalized_pressure,
    chemical_potential_vec,
    gk,
    An,
):

    pi_vector, delta_ln_ntot = solve_gibbs_iteration_equations(
        jnp.exp(ln_nk), jnp.exp(ln_ntot), formula_matrix, b, gk, An
    )
    ln_ntot += delta_ln_ntot
    ln_nk += update_ln_nk(pi_vector, delta_ln_ntot, formula_matrix, gk)

    nk = jnp.exp(ln_nk)
    ntot = jnp.exp(ln_ntot)
    gk = compute_gk(T, ln_nk, ln_ntot, chemical_potential_vec, normalized_pressure)
    An = formula_matrix @ nk

    epsilon = compute_residuals(nk, ntot, formula_matrix, b, gk, An, pi_vector)
    return ln_nk, ln_ntot, epsilon, gk, An


def iteration_gibbs(max_iter=100):
    def cond_fun(carry):
        ln_nk, ln_ntot, counter = carry
        return jnp.abs(E - E_prev) > 1e-6 and counter < max_iter

    def body_fun(carry):
        ln_nk, ln_ntot, counter = carry
        ln_nk_new, ln_ntot_new = update_all(
            ln_nk,
            ln_ntot,
            formula_matrix,
            b,
            T,
            normalized_pressure,
            chemical_potential_vec,
        )
        return ln_nk_new, ln_ntot_new, counter + 1

    _, E_star = while_loop(cond_fun, body_fun, (jnp.inf, Eini))
    return E_star


if __name__ == "__main__":
    from exogibbs.equilibrium.gibbs import interpolate_chemical_potential_one
    from exogibbs.io.load_data import get_data_filepath
    from exogibbs.io.load_data import DEFAULT_JANAF_GIBBS_MATRICES
    import numpy as np

    from jax import config

    config.update("jax_enable_x64", True)

    path = get_data_filepath(DEFAULT_JANAF_GIBBS_MATRICES)
    gibbs_matrices = np.load(path, allow_pickle=True)["arr_0"].item()

    # df_molname = load_molname()
    # path_JANAF_data = "/home/kawahara/thermochemical_equilibrium/Equilibrium/JANAF"
    # gibbs_matrices = load_JANAF_molecules(df_molname, path_JANAF_data)

    kJtoJ = 1000.0  # conversion factor from kJ to J
    T_h_table = gibbs_matrices["H1"]["T(K)"].to_numpy()
    mu_h_table = gibbs_matrices["H1"]["delta-f G"].to_numpy() * kJtoJ
    T_h2_table = gibbs_matrices["H2"]["T(K)"].to_numpy()
    mu_h2_table = gibbs_matrices["H2"]["delta-f G"].to_numpy() * kJtoJ

    def mu_h(T):
        return interpolate_chemical_potential_one(T, T_h_table, mu_h_table, order=2)

    def mu_h2(T):
        return interpolate_chemical_potential_one(T, T_h2_table, mu_h2_table, order=2)

    def compute_k(P, T, Pref=1.0):
        delta_mu = mu_h2(T) - 2.0 * mu_h(T)
        RT = R_gas_constant_si * T
        return np.exp(-delta_mu / RT) * P / Pref

    def nh(k):
        return 1.0 / np.sqrt(4.0 * k + 1.0)

    def nh2(k):
        return 0.5 * (1.0 - nh(k))

    formula_matrix = jnp.array([[1.0, 2.0]])
    T = 3700.0
    P = 1.0  # bar
    P_ref = 1.0  # bar

    normalized_pressure = P / P_ref
    ln_nk = jnp.array([0.0, 0.0])
    ln_ntot = 0.0
    chemical_potential_vec = jnp.array([mu_h(T), mu_h2(T)])
    b = jnp.array([1.0])  # Assuming no element abundance constraints

    # requires to compute (initial) gk and An before the loop
    gk = compute_gk(T, ln_nk, ln_ntot, chemical_potential_vec, normalized_pressure)
    An = formula_matrix @ jnp.exp(ln_nk)

    for i in range(10):

        # Solve the Gibbs iteration equations to get pi_vector and delta_ln_ntot
        pi_vector, delta_ln_ntot = solve_gibbs_iteration_equations(
            jnp.exp(ln_nk), jnp.exp(ln_ntot), formula_matrix, b, gk, An
        )

        #update ln_nk and ln_ntot
        ln_ntot += delta_ln_ntot
        ln_nk += update_ln_nk(pi_vector, delta_ln_ntot, formula_matrix, gk)

        # evaluate the residual (epsilon)
        nk = jnp.exp(ln_nk)
        ntot = jnp.exp(ln_ntot)
        gk = compute_gk(T, ln_nk, ln_ntot, chemical_potential_vec, normalized_pressure)
        An = formula_matrix @ nk
        epsilon = compute_residuals(nk, ntot, formula_matrix, b, gk, An, pi_vector)
        
    
        print(
            f"Iteration {i}: ln_nk = {ln_nk}, ln_ntot = {ln_ntot}, epsilon = {epsilon}"
        )

    k = compute_k(P, T, P_ref)
    print(jnp.log(nh(k)), jnp.log(nh2(k)))
    diff = jnp.log(nh(k)) - ln_nk[0]
    diff2 = jnp.log(nh2(k)) - ln_nk[1]
    print(f"Difference for H: {diff}, Difference for H2: {diff2}")
