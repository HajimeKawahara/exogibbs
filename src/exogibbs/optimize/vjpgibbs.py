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
        Bmatrix: A (diag(n) A^T (n_elements, n_elements), positive definite matrix.
        b_element_vector: element abundance vector (n_elements, ).

    Returns:
        The temperature VJP of log species number.
    """
    nk_cdot_hdot = jnp.vdot(nspecies, hdot)    
    etav = formula_matrix @ (nspecies * hdot)
    # solves the linear systems
    c, lower = cho_factor(Bmatrix)
    alpha = cho_solve((c, lower), formula_matrix@gvector)
    beta = cho_solve((c, lower), b_element_vector)
    # derives the temperature derivative of qtot
    dqtot_dT = (jnp.vdot(beta, etav) - nk_cdot_hdot)/jnp.vdot(beta, b_element_vector)
    # derives the g^T A^T Pi term
    gTATPi = jnp.vdot(alpha, etav - dqtot_dT*b_element_vector)

    return dqtot_dT*jnp.sum(gvector) + gTATPi - jnp.vdot(gvector, hdot)