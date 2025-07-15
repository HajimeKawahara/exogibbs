import jax.numpy as jnp
from typing import Tuple
from exogibbs.utils.constants import R_gas_constant_si

def _A_diagn_At(number_density_vector, formula_matrix):
    return jnp.einsum(
        "ik,k,jk->ij", formula_matrix, number_density_vector, formula_matrix
    )


def compute_gk(
    T: float,
    ln_nk: jnp.ndarray,
    ln_ntot: float,
    chemical_potential_vec: jnp.ndarray,
    normalized_pressure: float,
) -> jnp.ndarray:
    """computes gk vector for the Gibbs iteration

    Args:
        T: temperature (K)
        ln_nk: log of number density vector (n_species, )
        ln_ntot: log of total number density
        chemical_potential_over_RT_vec: chemical potential over RT vector (n_species, )
        normalized_pressure: normalized pressure P/Pref

    Returns:
        chemical potential vector (n_species, )
    """
    RT = R_gas_constant_si * T
    return chemical_potential_vec / RT + ln_nk - ln_ntot + jnp.log(normalized_pressure)
