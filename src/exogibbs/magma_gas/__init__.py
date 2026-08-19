"""Generic magma--gas coupling engine."""

from exogibbs.magma_gas.solve import solve
from exogibbs.magma_gas.types import (
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
