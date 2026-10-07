"""Compatibility exports for the former experimental interface types."""

from exogibbs.applications.magma_gas.models.meltyq.types import (
    MELTYQ_MELT_QUANTITIES,
    MELTYQ_ROOT_RESIDUALS,
    MagmaAtmosphereInterfaceInit,
    MagmaAtmosphereInterfaceOptions,
    MagmaAtmosphereInterfaceState,
    MagmaGasRootDiagnostics,
)

__all__ = (
    "MELTYQ_MELT_QUANTITIES",
    "MELTYQ_ROOT_RESIDUALS",
    "MagmaAtmosphereInterfaceInit",
    "MagmaAtmosphereInterfaceOptions",
    "MagmaAtmosphereInterfaceState",
    "MagmaGasRootDiagnostics",
)
