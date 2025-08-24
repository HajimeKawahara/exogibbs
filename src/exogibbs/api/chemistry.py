import jax.numpy as jnp
from jax import tree_util

from dataclasses import dataclass
from typing import Callable, Union, Tuple

@tree_util.register_pytree_node_class
@dataclass
class ThermoState:
    temperature: float
    ln_normalized_pressure: float
    b_element_vector: jnp.ndarray

    def tree_flatten(self):
        children = (
            self.temperature,
            self.ln_normalized_pressure,
            self.b_element_vector,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        temperature, ln_normalized_pressure, b_element_vector = children
        return cls(temperature, ln_normalized_pressure, b_element_vector)

@dataclass(frozen=True)
class ChemicalSetup:
    """
    A minimal, immutable container for thermochemical pre-setup.

    Fields:
        formula_matrix: Fixed stoichiometric constraint matrix A (E x K).
        b_element_vector: Elemental abundance vector b (E,).
        hvector_func: Callable h(T) used by the optimizer. MUST be JAX-differentiable w.r.t. T.
        metadata: Tuple of misc. info if needed (e.g., source tag).
    """
    formula_matrix: jnp.ndarray
    b_element_vector: jnp.ndarray
    hvector_func: Callable[[jnp.ndarray], jnp.ndarray]
    metadata: Tuple[str, ...] = ()