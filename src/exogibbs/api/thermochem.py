import jax.numpy as jnp
from jax import tree_util

from dataclasses import dataclass


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
