"""Stable built-in presets for the generic magma--gas API."""

from exogibbs.applications.magma_gas.models.meltyq import (
    MELTYQ_ELEMENTS,
    MELTYQ_MEAN_MELT_MOLAR_MASS_G_MOL,
    MELTYQ_MELT_QUANTITIES,
    MELTYQ_ROOT_RESIDUALS,
    MELTYQ_SPECIES,
    MeltyqMagmaGasInputs,
    MeltyqMagmaGasState,
    prepare_meltyq_problem,
)


__all__ = (
    "MELTYQ_ELEMENTS",
    "MELTYQ_MEAN_MELT_MOLAR_MASS_G_MOL",
    "MELTYQ_MELT_QUANTITIES",
    "MELTYQ_ROOT_RESIDUALS",
    "MELTYQ_SPECIES",
    "MeltyqMagmaGasInputs",
    "MeltyqMagmaGasState",
    "prepare_meltyq_problem",
)
