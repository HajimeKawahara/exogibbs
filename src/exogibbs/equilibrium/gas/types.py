"""State, option, initialization, and result types for gas equilibrium."""

from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Protocol, Union, runtime_checkable

import jax
import jax.numpy as jnp
from jax import tree_util

from exogibbs.thermo.models import ChemicalSetup


Array = jax.Array


@tree_util.register_pytree_node_class
@dataclass
class ThermoState:
    """Thermodynamic inputs traced by the gas equilibrium kernel."""

    temperature: float
    ln_normalized_pressure: float
    element_vector: jnp.ndarray

    def tree_flatten(self):
        children = (
            self.temperature,
            self.ln_normalized_pressure,
            self.element_vector,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        temperature, ln_normalized_pressure, element_vector = children
        return cls(temperature, ln_normalized_pressure, element_vector)


@dataclass(frozen=True)
class EquilibriumOptions:
    """Solver and profile scheduling options for gas equilibrium."""

    epsilon_crit: float = 1.0e-10
    max_iter: int = 1000
    method: Optional[
        Literal[
            "vmap_cold",
            "scan_hot_from_top",
            "scan_hot_from_bottom",
        ]
    ] = None


@dataclass(frozen=True)
class EquilibriumInit:
    """Optional initial gas log-amount state."""

    ln_nk: Optional[Array] = None
    ln_ntot: Optional[Array] = None


@dataclass(frozen=True)
class EquilibriumInitRequest:
    """Inputs available to a one-layer gas initializer."""

    setup: ChemicalSetup
    T: float
    P: float
    b: Array
    K: int
    explicit_log10_z_over_z_sun: Optional[float] = None
    user_init: Optional[EquilibriumInit] = None
    previous_solution: Optional[EquilibriumInit] = None


@runtime_checkable
class EquilibriumInitializer(Protocol):
    """Produce an initial guess for one gas-equilibrium layer."""

    def __call__(self, request: EquilibriumInitRequest) -> EquilibriumInit:
        ...


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EquilibriumResult:
    """Gas-equilibrium composition represented by JAX arrays."""

    ln_n: Array
    n: Array
    x: Array
    ntot: Array
    iterations: Optional[int] = None
    metadata: Optional[Mapping[str, Union[bool, float, int]]] = None

    def tree_flatten(self):
        children = (self.ln_n, self.n, self.x, self.ntot)
        aux = (self.iterations, self.metadata)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        iterations, metadata = aux_data
        ln_n, n, x, ntot = children
        return cls(
            ln_n=ln_n,
            n=n,
            x=x,
            ntot=ntot,
            iterations=iterations,
            metadata=metadata,
        )
