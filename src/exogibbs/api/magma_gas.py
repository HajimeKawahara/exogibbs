"""Stable, model-neutral API for coupled magma--gas calculations."""

from exogibbs.applications.magma_gas.solve import solve
from exogibbs.applications.magma_gas.types import (
    MagmaGasConditions,
    MagmaGasDiagnostics,
    MagmaGasEquilibriumState,
    MagmaGasInit,
    MagmaGasModel,
    MagmaGasModelEvaluation,
    MagmaGasOptions,
    MagmaGasProblem,
    MagmaGasResult,
)


__all__ = (
    "MagmaGasConditions",
    "MagmaGasDiagnostics",
    "MagmaGasEquilibriumState",
    "MagmaGasInit",
    "MagmaGasModel",
    "MagmaGasModelEvaluation",
    "MagmaGasOptions",
    "MagmaGasProblem",
    "MagmaGasResult",
    "solve",
)
