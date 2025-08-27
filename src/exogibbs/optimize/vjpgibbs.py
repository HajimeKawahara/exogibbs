import jax.numpy as jnp
from jax import jit

@jit
def vjp_temperature(
    gvector: jnp.ndarray,
    nspecies: jnp.ndarray,
    formula_matrix: jnp.ndarray,
    hdot: jnp.ndarray,
    alpha_vector: jnp.ndarray,
    beta_vector: jnp.ndarray,
    b_element_vector: jnp.ndarray,
    beta_dot_b_element: float,
) -> float:
    """
    Compute the temperature vector-Jacobian product of the Gibbs energy.

    Args:
        gvector: vector for vjp (n_species,).
        nspecies: species number vector (n_species,).
        formula_matrix: Formula matrix for stoichiometric constraints (n_elements, n_species).
        hdot: temperature derivative of h(T) = mu^o(T)/RT.
        alpha_vector: Solution to the linear system (A diag(n) A^T) @ alpha_vector = formula_matrix @ gvector.
        beta_vector: Solution to the linear system (A diag(n) A^T) @ beta_vector = b_element_vector.
        b_element_vector: element abundance vector (n_elements, ).
        beta_dot_b_element: dot product of beta_vector and b_element_vector, i.e. jnp.vdot(beta_vector, b_element_vector).

    Returns:
        The temperature VJP of log species number.
    """
    nk_cdot_hdot = jnp.vdot(nspecies, hdot)
    etav = formula_matrix @ (nspecies * hdot)
    # derives the temperature derivative of qtot
    dqtot_dT = (jnp.vdot(beta_vector, etav) - nk_cdot_hdot) / beta_dot_b_element
    
    # derives the g^T A^T Pi term
    gTATPi = jnp.vdot(alpha_vector, etav - dqtot_dT * b_element_vector) #original
    
    return dqtot_dT * jnp.sum(gvector) + gTATPi - jnp.vdot(gvector, hdot) #original
    
@jit
def vjp_pressure(
    gvector: jnp.ndarray,
    ntot: jnp.ndarray,
    alpha_vector: jnp.ndarray,
    b_element_vector: jnp.ndarray,
    beta_dot_b_element: float,
) -> float:
    """
    Compute the pressure vector-Jacobian product of the Gibbs energy.

    Args:
        gvector: vector for vjp (n_species,).
        ntot: total number of species (scalar).
        alpha_vector: (A (diag(n) A^T) @ alpha_vector = formula_matrix @ gvector
        b_element_vector: element abundance vector (n_elements, ).
        beta_dot_b_element: dot product of beta_vector and b_element_vector, i.e. jnp.vdot(beta_vector, b_element_vector).
    Returns:
        The pressure VJP of log species number.
    """
    eps = jnp.asarray(1e-20, dtype=beta_dot_b_element.dtype)
    denom = jnp.where(jnp.abs(beta_dot_b_element) < eps, eps, beta_dot_b_element)
    return ntot * (alpha_vector @ b_element_vector - jnp.sum(gvector)) / denom

@jit
def vjp_elements(
    gvector: jnp.ndarray,
    alpha_vector: jnp.ndarray,
    beta_vector: jnp.ndarray,
    b_element_vector: jnp.ndarray,
    beta_dot_b_element: float,
) -> jnp.ndarray:
    """
    Compute the elements vector-Jacobian product of the Gibbs energy.

    Args:
        gvector: vector for vjp (n_species,).
        alpha_vector: (A (diag(n) A^T) @ alpha_vector = formula_matrix @ gvector
        beta_vector: (A (diag(n) A^T) @ beta_vector = b_element_vector
        b_element_vector: element abundance vector (n_elements, ).
        beta_dot_b_element: dot product of beta_vector and b_element_vector, i.e. jnp.vdot(beta_vector, b_element_vector).
    Returns:
        The elements VJP of log species number.
    """

    eps = jnp.asarray(1e-20, dtype=beta_dot_b_element.dtype)
    denom = jnp.where(jnp.abs(beta_dot_b_element) < eps, eps, beta_dot_b_element)
    dqtot_db = beta_vector / denom
    Xmatrix = jnp.eye(len(b_element_vector)) - jnp.outer(b_element_vector, dqtot_db)
    return jnp.sum(gvector) * dqtot_db + alpha_vector @ Xmatrix
