import jax.numpy as jnp
from typing import Tuple
from exogibbs.optimize.lagrange.core import _A_diagn_At 

def solve_gibbs_equations_temperature_derivative(
    nspecies: jnp.ndarray,
    formula_matrix: jnp.ndarray,
    hdot: jnp.ndarray,
    b_element_vector: jnp.ndarray,
) -> Tuple[jnp.ndarray, float]:
    """
    Solve the Gibbs equations for temperature derivative.
    This function computes the matrix and vector to solve the system of equations
    that arises from the Gibbs energy minimization problem.

    Args:
        nspecies: species number vector (n_species,) for k-th iteration.
        formula_matrix: Formula matrix for stoichiometric constraints (n_elements, n_species).
        hdot: temperature derivative of h(T) = mu^o(T)/RT.
        b_element_vector: element abundance vector (n_elements, ).

    Returns:
        Tuple containing:
            - The pi vector (nspecies, ).
            - The update of the  log total number of species (delta_ln_ntot).
    """
    AnAt = _A_diagn_At(nspecies, formula_matrix)
    Anh = formula_matrix @ (nspecies * hdot)
    nk_cdot_hdot = jnp.dot(nspecies, hdot)

    assemble_mat = jnp.block([[AnAt, b_element_vector[:, None]], [b_element_vector[None, :], jnp.array([[0.0]])]])
    assemble_vec = jnp.concatenate([Anh, jnp.array([nk_cdot_hdot])])
    assemble_variable = jnp.linalg.solve(assemble_mat, assemble_vec)
    return assemble_variable[:-1], assemble_variable[-1]

def derivative_temperature(
    nspecies: jnp.ndarray,
    formula_matrix: jnp.ndarray,
    hdot: jnp.ndarray,
    b_element_vector: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the temperature derivative of the Gibbs energy.

    Args:
        nspecies: species number vector (n_species,).
        formula_matrix: Formula matrix for stoichiometric constraints (n_elements, n_species).
        hdot: temperature derivative of h(T) = mu^o(T)/RT.
        b_element_vector: element abundance vector (n_elements, ).

    Returns:
        The temperature derivative of log species number (n_species,).
    """
    pi, ln_ntot_dT = solve_gibbs_equations_temperature_derivative(nspecies, formula_matrix, hdot, b_element_vector)
    return ln_ntot_dT + formula_matrix.T @ pi - hdot