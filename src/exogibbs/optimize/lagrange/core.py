import jax.numpy as jnp
from jax import grad
from jax import vmap

from typing import Tuple

def _A_diagn_At(number_density_vector, formula_matrix):
    return jnp.einsum(
        "ik,k,jk->ij", formula_matrix, number_density_vector, formula_matrix
    )

def compute_ln_normalized_pressure(
    P: float, Pref: float = 1.0
) -> float:
    """Computes the natural log of normalized pressure P/Pref.

    Args:
        P: Pressure (bar).
        Pref: Reference pressure (bar), default is 1.0.

    Returns:
        Natural log of normalized pressure P/Pref.
    """
    return jnp.log(P / Pref)

def _compute_gk(
    T: float,
    ln_nk: jnp.ndarray,
    ln_ntot: float,
    hvector: jnp.ndarray,
    ln_normalized_pressure: float,
) -> jnp.ndarray:
    """computes gk vector for the Gibbs iteration

    Args:
        T: temperature (K)
        ln_nk: natural log of number of species vector (n_species, )
        ln_ntot: natural log of total number of species
        hvector: chemical potential over RT vector (n_species, )
        ln_normalized_pressure: natural log of normalized pressure P/Pref

    Returns:
        chemical potential vector (n_species, )
    """
    return hvector + ln_nk - ln_ntot + ln_normalized_pressure


