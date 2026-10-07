"""Compatibility exports for MELTYQ chemistry preparation."""

from exogibbs.applications.magma_gas.models.meltyq.setup import (
    CANONICAL_ELEMENTS,
    CANONICAL_SPECIES,
    PreparedMagmaGasChemistry,
    prepare_meltyq_chemistry,
)

__all__ = (
    "CANONICAL_ELEMENTS",
    "CANONICAL_SPECIES",
    "PreparedMagmaGasChemistry",
    "prepare_meltyq_chemistry",
)
