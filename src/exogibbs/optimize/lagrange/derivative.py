import jax.numpy as jnp
from typing import Tuple
from exogibbs.optimize.lagrange.core import _A_diagn_At 

def solve_gibbs_equations_temperature_derivative(
    nk: jnp.ndarray,
    formula_matrix: jnp.ndarray,
    hdot: jnp.ndarray,
    An: jnp.ndarray,
) -> Tuple[jnp.ndarray, float]:
    """
    Solve the Gibbs equations for temperature derivative.
    This function computes the matrix and vector to solve the system of equations
    that arises from the Gibbs energy minimization problem.

    Args:
        nk: Number density vector (n_species,) for k-th iteration.
        formula_matrix: Formula matrix for stoichiometric constraints (n_elements, n_species).
        hdot: temperature derivative of h(T) = mu^o(T)/RT.
        An: formula_matrix @ nk vector (n_elements, ).

    Returns:
        Tuple containing:
            - The pi vector (nspecies, ).
            - The update of the  log total number density (delta_ln_ntot).
    """
    AnAt = _A_diagn_At(nk, formula_matrix)
    Anh = formula_matrix @ (nk * hdot)
    nk_cdot_hdot = jnp.dot(An, hdot)

    assemble_mat = jnp.block([[AnAt, An[:, None]], [An[None, :], jnp.array([[0.0]])]])
    assemble_vec = jnp.concatenate([Anh, jnp.array([nk_cdot_hdot])])
    assemble_variable = jnp.linalg.solve(assemble_mat, assemble_vec)
    return assemble_variable[:-1], assemble_variable[-1]
