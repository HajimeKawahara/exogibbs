import jax.numpy as jnp
from jax import grad
from jax import vmap

from typing import Tuple

def _A_diagn_At(number_density_vector, formula_matrix):
    return jnp.einsum(
        "ik,k,jk->ij", formula_matrix, number_density_vector, formula_matrix
    )


def _compute_gk(
    T: float,
    ln_nk: jnp.ndarray,
    ln_ntot: float,
    hvector: jnp.ndarray,
    normalized_pressure: float,
) -> jnp.ndarray:
    """computes gk vector for the Gibbs iteration

    Args:
        T: temperature (K)
        ln_nk: log of number of species vector (n_species, )
        ln_ntot: log of total number of species
        hvector: chemical potential over RT vector (n_species, )
        normalized_pressure: normalized pressure P/Pref

    Returns:
        chemical potential vector (n_species, )
    """
    return hvector + ln_nk - ln_ntot + jnp.log(normalized_pressure)


