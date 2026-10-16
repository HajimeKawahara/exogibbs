"""Extensive ideal-gas Gibbs energy and exact simplex insertion bound."""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike

R = 8.31446261815324


def gibbs_j(temperature_k: ArrayLike, pressure_pa: ArrayLike, amounts_mol: ArrayLike,
            standard_potentials_rt: ArrayLike) -> jax.Array:
    """Return G [J] for a one-bar standard, including ideal mixing once."""
    n = jnp.asarray(amounts_mol)
    log_x = jnp.log(jnp.where(n > 0, n/jnp.sum(n), 1.))
    return R*temperature_k*jnp.sum(n*(jnp.asarray(standard_potentials_rt)+log_x+jnp.log(pressure_pa/1e5)))


def insertion_lower_j_mol(temperature_k: ArrayLike, pressure_pa: ArrayLike,
                          standard_potentials_rt: ArrayLike, atom_matrix: ArrayLike,
                          element_potentials_j_mol: ArrayLike,
                          allowed_components: tuple[int, ...]) -> jax.Array:
    """Exact minimum of G-lambda*A*x per mole over the allowed simplex."""
    cost = (jnp.asarray(standard_potentials_rt)+jnp.log(pressure_pa/1e5)
            - jnp.asarray(atom_matrix).T @ element_potentials_j_mol/(R*temperature_k))
    return -R*temperature_k*logsumexp(-cost[jnp.asarray(allowed_components)])
