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
    ln_nk: jnp.ndarray, 
    ln_ntot: float, 
    chemical_potential_over_RT_vec: jnp.ndarray, 
    normalized_pressure: float
) -> jnp.ndarray:
    """ computes gk vector for the Gibbs iteration

    Args:
        ln_nk: log of number density vector (n_species, )
        ln_ntot: log of total number density
        chemical_potential_over_RT_vec: chemical potential over RT vector (n_species, )
        normalized_pressure: normalized pressure P/Pref

    Returns:
        chemical potential vector (n_species, )
    """
    return chemical_potential_over_RT_vec  + ln_nk - ln_ntot + jnp.log(normalized_pressure)


def solve_gibbs_iteration_equations(
    nk: jnp.ndarray,
    ntotk: float,
    formula_matrix: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
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
        gk: Chemical potential vector (n_species,) for k-th iteration.

    Returns:
        Tuple containing:
            - The pi vector (nspecies, ).
            - The update of the  log total number density (delta_ln_ntot).
    """
    AnAt = _A_diagn_At(nk, formula_matrix)
    An = formula_matrix @ nk
    Angk = formula_matrix @ (gk * nk)
    resn = jnp.sum(nk) - ntotk
    ngk = jnp.dot(nk, gk)

    assemble_mat = jnp.block([[AnAt, An[:, None]], [An[None, :], jnp.array([[resn]])]])
    assemble_vec = jnp.concatenate([Angk + b - An, jnp.array([ngk - resn])])
    assemble_variable = jnp.linalg.solve(assemble_mat, assemble_vec)
    return assemble_variable[:-1], assemble_variable[-1]


def update_ln_nk(
    pi_vector: jnp.ndarray,
    delta_ln_ntot: float,
    formula_matrix: jnp.ndarray,
    gk: jnp.ndarray,
) -> jnp.ndarray:
    
    return formula_matrix.T @ pi_vector + delta_ln_ntot - gk


def interation():
    def cond_fun(carry):
        E_prev, E = carry
        return jnp.abs(E - E_prev) > 1e-6

    def body_fun(carry):
        _, E = carry
        E_new = E - f(E, e, M) / dfdE(E, e, M)
        return E, E_new

    _, E_star = while_loop(cond_fun, body_fun, (jnp.inf, Eini))
    return E_star


if __name__ == "__main__":
    from exogibbs.io.load_data import load_molname, load_JANAF_molecules
    from exogibbs.equilibrium.gibbs import interpolate_chemical_potential_one
    import numpy as np
    from jax import config

    config.update("jax_enable_x64", True)

    df_molname = load_molname()
    path_JANAF_data = "/home/kawahara/thermochemical_equilibrium/Equilibrium/JANAF"
    gibbs_matrices = load_JANAF_molecules(df_molname, path_JANAF_data)

    T_h_table = gibbs_matrices["H1"]["T(K)"].to_numpy()
    mu_over_RT_h_table = gibbs_matrices["H1"]["delta-f G"].to_numpy()/R_gas_constant_si
    T_h2_table = gibbs_matrices["H2"]["T(K)"].to_numpy()
    mu_over_RT_h2_table = gibbs_matrices["H2"]["delta-f G"].to_numpy()/R_gas_constant_si

    def mu_over_RT_h(T):
        return interpolate_chemical_potential_one(T, T_h_table, mu_over_RT_h_table, order=2)

    def mu_over_RT_h2(T):
        return interpolate_chemical_potential_one(T, T_h2_table, mu_over_RT_h2_table, order=2)

    def compute_k(P, T, Pref=1.0):
        delta_mu = mu_over_RT_h2(T) - 2.0 * mu_over_RT_h(T)
        return np.exp(-delta_mu) * P / Pref

    def nh(k):
        return 1.0 / np.sqrt(4.0*k + 1.0)


    def nh2(k):
        return 0.5 * (1.0 - nh(k))


    
    formula_matrix = jnp.array([[1.0, 2.0]])
    T = 3700.0
    P = 1.0  # bar
    P_ref = 1.0  # bar

    normalized_pressure = P / P_ref
    ln_nk = jnp.array([0.0, 0.0])
    ln_ntot = 0.0
    chemical_potential_vec = jnp.array([mu_over_RT_h(T), mu_over_RT_h2(T)])
    b = jnp.array([1.0])  # Assuming no element abundance constraints
    
    for i in range(10):
        gk = compute_gk(ln_nk, ln_ntot, chemical_potential_vec, normalized_pressure)
        pi_vector, delta_ln_ntot = solve_gibbs_iteration_equations(
            jnp.exp(ln_nk), jnp.exp(ln_ntot), formula_matrix, b, gk
        )
        ln_ntot += delta_ln_ntot
        ln_nk += update_ln_nk(pi_vector, delta_ln_ntot, formula_matrix, gk)
        print(f"Iteration {i}: ln_nk = {ln_nk}, ln_ntot = {ln_ntot}")

    k = compute_k(P,T, P_ref)
    print(jnp.log(nh(k)), jnp.log(nh2(k)))