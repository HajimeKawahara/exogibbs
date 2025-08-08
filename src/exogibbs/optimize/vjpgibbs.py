import jax.numpy as jnp
from jax import vmap
from typing import Tuple

def vjp_temperature(
    gvector: jnp.ndarray,
    nspecies: jnp.ndarray,
    formula_matrix: jnp.ndarray,
    hdot: jnp.ndarray,
    alpha_vector: jnp.ndarray,
    beta_vector: jnp.ndarray,
    b_element_vector: jnp.ndarray,
) -> float:
    """
    Compute the temperature vector-Jacobian product of the Gibbs energy.

    Args:
        gvector: vector for vjp (n_species,).
        nspecies: species number vector (n_species,).
        formula_matrix: Formula matrix for stoichiometric constraints (n_elements, n_species).
        hdot: temperature derivative of h(T) = mu^o(T)/RT.
        alpha_vector: (A (diag(n) A^T) @ alpha_vector = formula_matrix @ gvector
        beta_vector: (A (diag(n) A^T) @ beta_vector = b_element_vector
        b_element_vector: element abundance vector (n_elements, ).

    Returns:
        The temperature VJP of log species number.
    """
    nk_cdot_hdot = jnp.vdot(nspecies, hdot)    
    etav = formula_matrix @ (nspecies * hdot)
    # derives the temperature derivative of qtot
    dqtot_dT = (jnp.vdot(beta_vector, etav) - nk_cdot_hdot)/jnp.vdot(beta_vector, b_element_vector)
    # derives the g^T A^T Pi term
    gTATPi = jnp.vdot(alpha_vector, etav - dqtot_dT*b_element_vector)

    return dqtot_dT*jnp.sum(gvector) + gTATPi - jnp.vdot(gvector, hdot)