"""Experimental MELTYQ-equivalent magma--gas interface."""

from exogibbs.experimental.magma_gas.meltyq_basis import (
    MELTYQ_MEAN_MELT_MOLAR_MASS_G_MOL,
    co2_mass_fraction_to_mole_ratio,
    elemental_c_mass_fraction_to_mole_ratio,
    elemental_n_mass_fraction_to_mole_ratio,
    h2o_mass_fraction_to_mole_ratio,
)
from exogibbs.experimental.magma_gas.setup import (
    CANONICAL_ELEMENTS,
    CANONICAL_SPECIES,
    PreparedMagmaGasChemistry,
    prepare_meltyq_chemistry,
)
from exogibbs.experimental.magma_gas.solve import (
    solve_magma_atmosphere_interface,
)
from exogibbs.experimental.magma_gas.types import (
    MELTYQ_MELT_QUANTITIES,
    MELTYQ_ROOT_RESIDUALS,
    MagmaAtmosphereInterfaceInit,
    MagmaAtmosphereInterfaceOptions,
    MagmaAtmosphereInterfaceState,
    MagmaGasRootDiagnostics,
)


__all__ = (
    "CANONICAL_ELEMENTS",
    "CANONICAL_SPECIES",
    "MELTYQ_MEAN_MELT_MOLAR_MASS_G_MOL",
    "MELTYQ_MELT_QUANTITIES",
    "MELTYQ_ROOT_RESIDUALS",
    "MagmaAtmosphereInterfaceInit",
    "MagmaAtmosphereInterfaceOptions",
    "MagmaAtmosphereInterfaceState",
    "MagmaGasRootDiagnostics",
    "PreparedMagmaGasChemistry",
    "co2_mass_fraction_to_mole_ratio",
    "elemental_c_mass_fraction_to_mole_ratio",
    "elemental_n_mass_fraction_to_mole_ratio",
    "h2o_mass_fraction_to_mole_ratio",
    "prepare_meltyq_chemistry",
    "solve_magma_atmosphere_interface",
)
