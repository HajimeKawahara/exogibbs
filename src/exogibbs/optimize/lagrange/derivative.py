import jax.numpy as jnp
from typing import Tuple

def _solve_gibbs_equations_temperature_derivative(
    nspecies: jnp.ndarray,
    formula_matrix: jnp.ndarray,
    hdot: jnp.ndarray,
    nk_cdot_hdot: float,
    Bmatrix: jnp.ndarray,
    b_element_vector: jnp.ndarray,
) -> Tuple[jnp.ndarray, float]:
    """
    Solve the Gibbs equations for temperature derivative.
    This function computes the matrix and vector to solve the system of equations
    that arises from the Gibbs energy minimization problem.

    Args:
        nspecies: species number vector (n_species,) for k-th iteration.
        formula_matrix: Formula matrix A for stoichiometric constraints (n_elements, n_species).
        hdot: temperature derivative of h(T) = mu^o(T)/RT.
        nk_cdot_hdot: dot product of species number and hdot.
        Bmatrix: A (diag(n) A^T (n_elements, n_elements)
        b_element_vector: element abundance vector (n_elements, ).

    Returns:
        Tuple containing:
            - The pi vector (nspecies, ).
            - The update of the  log total number of species (delta_ln_ntot).
    """
    Anh = formula_matrix @ (nspecies * hdot)
    
    assemble_mat = jnp.block([[Bmatrix, b_element_vector[:, None]], [b_element_vector[None, :], jnp.array([[0.0]])]])
    assemble_vec = jnp.concatenate([Anh, jnp.array([nk_cdot_hdot])])
    assemble_variable = jnp.linalg.solve(assemble_mat, assemble_vec)
    return assemble_variable[:-1], assemble_variable[-1]

def derivative_temperature(
    nspecies: jnp.ndarray,
    formula_matrix: jnp.ndarray,
    hdot: jnp.ndarray,
    nk_cdot_hdot: float,
    Bmatrix: jnp.ndarray,
    b_element_vector: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the temperature derivative of the Gibbs energy.

    Args:
        nspecies: species number vector (n_species,).
        formula_matrix: Formula matrix for stoichiometric constraints (n_elements, n_species).
        hdot: temperature derivative of h(T) = mu^o(T)/RT.
        nk_cdot_hdot: dot product of species number and hdot.
        Bmatrix: A (diag(n) A^T (n_elements, n_elements)
        b_element_vector: element abundance vector (n_elements, ).

    Returns:
        The temperature derivative of log species number (n_species,).
    """
    
    pi, ln_ntot_dT = _solve_gibbs_equations_temperature_derivative(nspecies, formula_matrix, hdot, nk_cdot_hdot, Bmatrix, b_element_vector)
    return ln_ntot_dT + formula_matrix.T @ pi - hdot

def _solve_gibbs_equations_pressure_derivative(
    ntot: float,
    Bmatrix: jnp.ndarray,
    b_element_vector: jnp.ndarray,
) -> Tuple[jnp.ndarray, float]:
    """
    Solve the Gibbs equations for pressure derivative.
    This function computes the matrix and vector to solve the system of equations
    that arises from the Gibbs energy minimization problem.

    Args:
        ntot: total number of species.
        Bmatrix: A (diag(n) A^T (n_elements, n_elements)
        b_element_vector: element abundance vector (n_elements, ).

    Returns:
        Tuple containing:
            - The pi vector (nspecies, ).
            - The update of the  log total number of species (delta_ln_ntot).
    """
    
    assemble_mat = jnp.block([[Bmatrix, b_element_vector[:, None]], [b_element_vector[None, :], jnp.array([[0.0]])]])
    assemble_vec = jnp.concatenate([b_element_vector, jnp.array([ntot])])
    assemble_variable = jnp.linalg.solve(assemble_mat, assemble_vec)
    return assemble_variable[:-1], assemble_variable[-1]

def derivative_pressure(
    ntot: float,
    formula_matrix: jnp.ndarray,
    Bmatrix: jnp.ndarray,
    b_element_vector: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the temperature derivative of the Gibbs energy.

    Args:
        ntot: total number of species.
        formula_matrix: Formula matrix for stoichiometric constraints (n_elements, n_species).
        Bmatrix: A (diag(n) A^T (n_elements, n_elements)
        b_element_vector: element abundance vector (n_elements, ).

    Returns:
        The pressure derivative of log species number (n_species,).
    """

    L, ln_ntot_dlogp = _solve_gibbs_equations_pressure_derivative(ntot, Bmatrix, b_element_vector)

    return formula_matrix.T @ L + ln_ntot_dlogp - 1.0
