"""Model-neutral types for coupled magma--gas calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional, Protocol, runtime_checkable

import jax
from jax import tree_util

from exogibbs.equilibrium.gas import types as gas_types
from exogibbs.equilibrium.gas.types import EquilibriumOptions
from exogibbs.thermo.fugacity import LogFugacityCoefficientFunction
from exogibbs.thermo.models import ChemicalSetup


Array = jax.Array


class MagmaGasConditions(NamedTuple):
    """Dynamic conditions supplied to a magma--gas model."""

    temperature_k: Array
    pressure_bar: Array
    model_inputs: Any


class MagmaGasEquilibriumState(NamedTuple):
    """Gas-equilibrium quantities shared by all magma models."""

    equilibrium: gas_types.EquilibriumResult
    ln_fugacity_coefficients: Array
    log_mole_fractions: Array
    log_partial_pressures_bar: Array
    log_fugacities_bar: Array
    partial_pressures_bar: Array
    fugacities_bar: Array


class MagmaGasModelEvaluation(NamedTuple):
    """A model residual and its model-specific state."""

    residual: Array
    state: Any


@runtime_checkable
class MagmaGasModel(Protocol):
    """Minimal contract for a differentiable magma--gas boundary model."""

    def initial_root(self, conditions: MagmaGasConditions) -> Array:
        """Return a deterministic one-dimensional initial root vector."""

        ...

    def element_abundances(
        self,
        conditions: MagmaGasConditions,
        root_variables: Array,
    ) -> Array:
        """Map root variables to the gas solver's element vector."""

        ...

    def evaluate(
        self,
        conditions: MagmaGasConditions,
        root_variables: Array,
        gas: MagmaGasEquilibriumState,
    ) -> MagmaGasModelEvaluation:
        """Return a square residual and model-specific output state."""

        ...


@dataclass(frozen=True)
class MagmaGasProblem:
    """Static chemistry and model definition for one magma--gas problem."""

    setup: ChemicalSetup
    model: MagmaGasModel
    lnphi_func: Optional[LogFugacityCoefficientFunction] = None


@dataclass(frozen=True)
class MagmaGasInit:
    """Optional initial coordinates in the selected model's root basis."""

    root_variables: Optional[Array] = None


@dataclass(frozen=True)
class MagmaGasOptions:
    """Numerical options for the outer root and inner gas equilibrium."""

    root_tolerance: float = 1.0e-8
    max_iter: int = 30
    line_search_steps: int = 12
    backtracking_factor: float = 0.5
    max_step: float = 5.0
    equilibrium_options: EquilibriumOptions = field(
        default_factory=lambda: EquilibriumOptions(
            epsilon_crit=1.0e-11,
            max_iter=1000,
        )
    )


class MagmaGasDiagnostics(NamedTuple):
    """JAX-compatible convergence information for both nested solves."""

    converged: Array
    outer_converged: Array
    inner_converged: Array
    iterations: Array
    inner_iterations: Array
    residual: Array
    residual_norm: Array
    root_tolerance: Array
    inner_residual_norm: Array
    inner_tolerance: Array
    step_accepted: Array


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MagmaGasResult:
    """Coupled gas state, model state, root coordinates, and diagnostics."""

    element_abundances: Array
    root_variables: Array
    gas: MagmaGasEquilibriumState
    model_state: Any
    diagnostics: MagmaGasDiagnostics

    def tree_flatten(self):
        children = (
            self.element_abundances,
            self.root_variables,
            self.gas,
            self.model_state,
            self.diagnostics,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        (
            element_abundances,
            root_variables,
            gas,
            model_state,
            diagnostics,
        ) = children
        return cls(
            element_abundances=element_abundances,
            root_variables=root_variables,
            gas=gas,
            model_state=model_state,
            diagnostics=diagnostics,
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
)
