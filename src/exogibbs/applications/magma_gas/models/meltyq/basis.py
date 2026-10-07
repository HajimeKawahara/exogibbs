"""MELTYQ-specific dissolved-volatile basis conversions."""

from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike

from exogibbs.thermo.composition import (
    ln_mass_fraction_to_ln_dilute_mole_ratio,
    mass_fraction_to_dilute_mole_ratio,
)


MELTYQ_MEAN_MELT_MOLAR_MASS_G_MOL = 60.0

_H2O_MOLAR_MASS_G_MOL = 18.01528
_CO2_MOLAR_MASS_G_MOL = 44.0095
_C_ATOMIC_MASS_G_MOL = 12.0107
_N_ATOMIC_MASS_G_MOL = 14.0067


def _meltyq_mass_fraction_to_mole_ratio(
    mass_fraction: ArrayLike,
    species_molar_mass_g_mol: ArrayLike,
) -> jnp.ndarray:
    """Convert a mass fraction using MELTYQ's mean melt molar mass."""

    return mass_fraction_to_dilute_mole_ratio(
        mass_fraction,
        species_molar_mass_g_mol,
        matrix_molar_mass_g_mol=MELTYQ_MEAN_MELT_MOLAR_MASS_G_MOL,
    )


def _meltyq_ln_mass_fraction_to_ln_mole_ratio(
    ln_mass_fraction: ArrayLike,
    species_molar_mass_g_mol: ArrayLike,
) -> jnp.ndarray:
    """Convert a log mass fraction using MELTYQ's mean melt molar mass."""

    return ln_mass_fraction_to_ln_dilute_mole_ratio(
        ln_mass_fraction,
        species_molar_mass_g_mol,
        matrix_molar_mass_g_mol=MELTYQ_MEAN_MELT_MOLAR_MASS_G_MOL,
    )


def h2o_mass_fraction_to_mole_ratio(mass_fraction: ArrayLike) -> jnp.ndarray:
    """Convert dissolved H2O mass fraction to dilute melt mole ratio."""

    return _meltyq_mass_fraction_to_mole_ratio(
        mass_fraction,
        _H2O_MOLAR_MASS_G_MOL,
    )


def co2_mass_fraction_to_mole_ratio(mass_fraction: ArrayLike) -> jnp.ndarray:
    """Convert dissolved CO2 mass fraction to dilute melt mole ratio."""

    return _meltyq_mass_fraction_to_mole_ratio(
        mass_fraction,
        _CO2_MOLAR_MASS_G_MOL,
    )


def elemental_c_mass_fraction_to_mole_ratio(
    mass_fraction: ArrayLike,
) -> jnp.ndarray:
    """Convert the CO law's elemental-C mass fraction to melt mole ratio."""

    return _meltyq_mass_fraction_to_mole_ratio(
        mass_fraction,
        _C_ATOMIC_MASS_G_MOL,
    )


def elemental_c_ln_mass_fraction_to_ln_mole_ratio(
    ln_mass_fraction: ArrayLike,
) -> jnp.ndarray:
    """Convert the CO law's log elemental-C mass fraction to log mole ratio."""

    return _meltyq_ln_mass_fraction_to_ln_mole_ratio(
        ln_mass_fraction,
        _C_ATOMIC_MASS_G_MOL,
    )


def elemental_n_mass_fraction_to_mole_ratio(
    mass_fraction: ArrayLike,
) -> jnp.ndarray:
    """Convert total elemental-N mass fraction to atomic-N mole ratio."""

    return _meltyq_mass_fraction_to_mole_ratio(
        mass_fraction,
        _N_ATOMIC_MASS_G_MOL,
    )


def elemental_n_ln_mass_fraction_to_ln_mole_ratio(
    ln_mass_fraction: ArrayLike,
) -> jnp.ndarray:
    """Convert the N law's log elemental-N mass fraction to log mole ratio."""

    return _meltyq_ln_mass_fraction_to_ln_mole_ratio(
        ln_mass_fraction,
        _N_ATOMIC_MASS_G_MOL,
    )


__all__ = (
    "MELTYQ_MEAN_MELT_MOLAR_MASS_G_MOL",
    "co2_mass_fraction_to_mole_ratio",
    "elemental_c_ln_mass_fraction_to_ln_mole_ratio",
    "elemental_c_mass_fraction_to_mole_ratio",
    "elemental_n_ln_mass_fraction_to_ln_mole_ratio",
    "elemental_n_mass_fraction_to_mole_ratio",
    "h2o_mass_fraction_to_mole_ratio",
)
