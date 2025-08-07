import jax.numpy as jnp
from jax import vmap
from typing import Tuple
from jax.scipy.linalg import cho_factor
from jax.scipy.linalg import cho_solve

def vjp_temperature(
    gvector: jnp.ndarray,
    nspecies: jnp.ndarray,
    formula_matrix: jnp.ndarray,
    hdot: jnp.ndarray,
    nk_cdot_hdot: float,
    Bmatrix: jnp.ndarray,
    b_element_vector: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the temperature vector-Jacobian product of the Gibbs energy.

    Args:
        gvector: vector for vjp (n_species,).
        nspecies: species number vector (n_species,).
        formula_matrix: Formula matrix for stoichiometric constraints (n_elements, n_species).
        hdot: temperature derivative of h(T) = mu^o(T)/RT.
        nk_cdot_hdot: dot product of species number and hdot.
        Bmatrix: A (diag(n) A^T (n_elements, n_elements) Semipsitive Definite matrix.
        b_element_vector: element abundance vector (n_elements, ).

    Returns:
        The temperature derivative of log species number (.
    """
    c, lower = cho_factor(Bmatrix)
    alpha = cho_solve((c, lower), formula_matrix@gvector)
    beta = cho_solve((c, lower), b_element_vector)
    bsquared_inverse = 1.0/jnp.vdot(b_element_vector, b_element_vector)
    Anh = formula_matrix @ (nspecies * hdot)
    dqtot_dT = bsquared_inverse * (jnp.vdot(beta, Anh) - nk_cdot_hdot)

    return cot_T
