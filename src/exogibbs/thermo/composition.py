"""Composition helpers shared by equilibrium features."""

from typing import Iterable, Optional

import jax.numpy as jnp
from jax.typing import ArrayLike

from exogibbs.thermo.models import ChemicalSetup


def mass_fraction_to_dilute_mole_ratio(
    mass_fraction: ArrayLike,
    species_molar_mass_g_mol: ArrayLike,
    *,
    matrix_molar_mass_g_mol: ArrayLike,
) -> jnp.ndarray:
    """Convert mass fraction to a dilute solute-to-matrix mole ratio.

    The approximation is ``Y * M_matrix / M_species``. It does not
    renormalize the mixture and therefore is not an exact finite-concentration
    mole fraction. Invalid mass fractions or nonpositive molar masses return
    ``nan``.
    """

    mass_fraction_array = jnp.asarray(mass_fraction)
    species_molar_mass = jnp.asarray(species_molar_mass_g_mol)
    matrix_molar_mass = jnp.asarray(matrix_molar_mass_g_mol)
    result = mass_fraction_array * matrix_molar_mass / species_molar_mass
    valid = (
        jnp.isfinite(mass_fraction_array)
        & (mass_fraction_array >= 0.0)
        & (mass_fraction_array <= 1.0)
        & jnp.isfinite(species_molar_mass)
        & (species_molar_mass > 0.0)
        & jnp.isfinite(matrix_molar_mass)
        & (matrix_molar_mass > 0.0)
    )
    return jnp.where(valid, result, jnp.nan)


def ln_mass_fraction_to_ln_dilute_mole_ratio(
    ln_mass_fraction: ArrayLike,
    species_molar_mass_g_mol: ArrayLike,
    *,
    matrix_molar_mass_g_mol: ArrayLike,
) -> jnp.ndarray:
    """Convert ``ln(Y)`` to a log dilute solute-to-matrix mole ratio.

    The result is ``ln(Y) + ln(M_matrix) - ln(M_species)``.  ``-inf``
    represents a zero mass fraction and is preserved.  Inputs greater than
    zero cannot represent a mass fraction, and invalid inputs return ``nan``.
    """

    ln_mass_fraction_array = jnp.asarray(ln_mass_fraction)
    species_molar_mass = jnp.asarray(species_molar_mass_g_mol)
    matrix_molar_mass = jnp.asarray(matrix_molar_mass_g_mol)
    result = (
        ln_mass_fraction_array
        + jnp.log(matrix_molar_mass)
        - jnp.log(species_molar_mass)
    )
    valid_ln_mass_fraction = (
        jnp.isfinite(ln_mass_fraction_array)
        & (ln_mass_fraction_array <= 0.0)
    ) | jnp.isneginf(ln_mass_fraction_array)
    valid = (
        valid_ln_mass_fraction
        & jnp.isfinite(species_molar_mass)
        & (species_molar_mass > 0.0)
        & jnp.isfinite(matrix_molar_mass)
        & (matrix_molar_mass > 0.0)
    )
    return jnp.where(valid, result, jnp.nan)


def update_element_vector(
    element_vector_ref: jnp.ndarray,
    scale_indices: jnp.ndarray,
    scales: jnp.ndarray,
    *,
    set_indices: Optional[jnp.ndarray] = None,
    set_values: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Build a new element vector by scaling and overriding entries."""

    b0 = jnp.asarray(element_vector_ref)
    idx = jnp.asarray(scale_indices, dtype=jnp.int32)
    scale_values = jnp.asarray(scales, dtype=b0.dtype)
    out = b0
    if idx.size != 0:
        out = out.at[idx].set(b0[idx] * scale_values)

    if set_indices is not None and set_values is not None:
        override_indices = jnp.asarray(set_indices, dtype=jnp.int32)
        override_values = jnp.asarray(set_values, dtype=b0.dtype)
        if override_indices.size != 0:
            out = out.at[override_indices].set(override_values)
    return out


def element_indices_by_name(
    setup: ChemicalSetup,
    names: Iterable[str],
) -> jnp.ndarray:
    """Return element indices in the same order as ``names``."""

    if setup.elements is None:
        raise ValueError("setup.elements is not available for index lookup.")
    positions = {element: index for index, element in enumerate(setup.elements)}
    return jnp.asarray([positions[name] for name in names], dtype=jnp.int32)
