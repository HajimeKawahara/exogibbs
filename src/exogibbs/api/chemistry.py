import jax.numpy as jnp
from jax import tree_util

from dataclasses import dataclass
from typing import Callable
from typing import Tuple
from typing import Optional
from typing import Mapping


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


from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Mapping, Union
import jax.numpy as jnp

Array = jnp.ndarray

@dataclass(frozen=True)
class ChemicalSetup:
    """Minimal, immutable container for thermochemical pre-setup.

    Fields
    ------
    formula_matrix : (E, K) jnp.ndarray
        Fixed stoichiometric constraint matrix A.
    hvector_func : Callable[[float|Array], Array]
        h(T) used by the optimizer (JAX-differentiable).

    elements : Optional[tuple[str, ...]]
        Element symbols (E,) if available.
    species : Optional[tuple[str, ...]]
        Species names (K,) if available.
    b_element_vector_reference : Optional[np.ndarray]
        Sample elemental abundance b (E,) for reference only.
    metadata : Optional[Mapping[str, str]]
        Free-form provenance info (e.g., source="JANAF", preset="ykb4").
    """
    formula_matrix: Array
    hvector_func: Callable[[Union[float, Array]], Array]

    # Optional metadata (host-side; NOT traced)
    elements: Optional[Tuple[str, ...]] = None
    species: Optional[Tuple[str, ...]] = None
    b_element_vector_reference: Optional["np.ndarray"] = None  # host-side
    metadata: Optional[Mapping[str, str]] = None
