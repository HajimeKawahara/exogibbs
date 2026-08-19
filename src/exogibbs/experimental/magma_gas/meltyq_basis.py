"""Compatibility exports for MELTYQ basis conversions."""

from exogibbs.magma_gas.models.meltyq.basis import (
    MELTYQ_MEAN_MELT_MOLAR_MASS_G_MOL,
    co2_mass_fraction_to_mole_ratio,
    elemental_c_ln_mass_fraction_to_ln_mole_ratio,
    elemental_c_mass_fraction_to_mole_ratio,
    elemental_n_ln_mass_fraction_to_ln_mole_ratio,
    elemental_n_mass_fraction_to_mole_ratio,
    h2o_mass_fraction_to_mole_ratio,
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
