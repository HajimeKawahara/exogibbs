"""Built-in MELTYQ magma--gas model."""

from exogibbs.magma_gas.models.meltyq.basis import (
    MELTYQ_MEAN_MELT_MOLAR_MASS_G_MOL,
    co2_mass_fraction_to_mole_ratio,
    elemental_c_ln_mass_fraction_to_ln_mole_ratio,
    elemental_c_mass_fraction_to_mole_ratio,
    elemental_n_ln_mass_fraction_to_ln_mole_ratio,
    elemental_n_mass_fraction_to_mole_ratio,
    h2o_mass_fraction_to_mole_ratio,
)
from exogibbs.magma_gas.models.meltyq.model import (
    MeltyqMagmaGasModel,
    prepare_meltyq_problem,
)
from exogibbs.magma_gas.models.meltyq.setup import (
    CANONICAL_ELEMENTS,
    CANONICAL_SPECIES,
    MELTYQ_ELEMENTS,
    MELTYQ_SPECIES,
    PreparedMagmaGasChemistry,
    PreparedMeltyqChemistry,
    prepare_meltyq_chemistry,
)
from exogibbs.magma_gas.models.meltyq.types import (
    MELTYQ_MELT_QUANTITIES,
    MELTYQ_ROOT_RESIDUALS,
    MagmaAtmosphereInterfaceInit,
    MagmaAtmosphereInterfaceOptions,
    MagmaAtmosphereInterfaceState,
    MagmaGasRootDiagnostics,
    MeltyqMagmaGasInputs,
    MeltyqMagmaGasState,
)


__all__ = (
    "CANONICAL_ELEMENTS",
    "CANONICAL_SPECIES",
    "MELTYQ_ELEMENTS",
    "MELTYQ_MEAN_MELT_MOLAR_MASS_G_MOL",
    "MELTYQ_MELT_QUANTITIES",
    "MELTYQ_ROOT_RESIDUALS",
    "MELTYQ_SPECIES",
    "MagmaAtmosphereInterfaceInit",
    "MagmaAtmosphereInterfaceOptions",
    "MagmaAtmosphereInterfaceState",
    "MagmaGasRootDiagnostics",
    "MeltyqMagmaGasInputs",
    "MeltyqMagmaGasModel",
    "MeltyqMagmaGasState",
    "PreparedMagmaGasChemistry",
    "PreparedMeltyqChemistry",
    "co2_mass_fraction_to_mole_ratio",
    "elemental_c_ln_mass_fraction_to_ln_mole_ratio",
    "elemental_c_mass_fraction_to_mole_ratio",
    "elemental_n_ln_mass_fraction_to_ln_mole_ratio",
    "elemental_n_mass_fraction_to_mole_ratio",
    "h2o_mass_fraction_to_mole_ratio",
    "prepare_meltyq_chemistry",
    "prepare_meltyq_problem",
)
