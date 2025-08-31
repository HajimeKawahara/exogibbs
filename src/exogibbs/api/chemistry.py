import jax.numpy as jnp
from jax import tree_util

from dataclasses import dataclass
from typing import Callable, Iterable
from typing import Tuple
from typing import Optional
from typing import Mapping


@tree_util.register_pytree_node_class
@dataclass
class ThermoState:
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
        temperature, ln_normalized_pressure, element_vector = children
        return cls(temperature, ln_normalized_pressure, element_vector)


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
    element_vector_reference : Optional[np.ndarray]
        Sample elemental abundance b (E,) for reference only.
    metadata : Optional[Mapping[str, str]]
        Free-form provenance info (e.g., source="JANAF", preset="ykb4").
    """
    formula_matrix: Array
    hvector_func: Callable[[Union[float, Array]], Array]

    # Optional metadata (host-side; NOT traced)
    elements: Optional[Tuple[str, ...]] = None
    species: Optional[Tuple[str, ...]] = None
    element_vector_reference: Optional["np.ndarray"] = None  # host-side
    metadata: Optional[Mapping[str, str]] = None


def update_element_vector(
    element_vector_ref: jnp.ndarray,
    scale_indices: jnp.ndarray,
    scales: jnp.ndarray,
    *,
    set_indices: Optional[jnp.ndarray] = None,
    set_values: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Build a new ``element_vector`` by scaling and/or overriding entries.

    This is JAX-jit safe (pure array API; no Python dicts) and avoids pitfalls
    with ``jnp.append``. Typical use is to scale selected elements (e.g., C, O)
    relative to a reference vector, and optionally set a fixed value for a slot
    (e.g., electron abundance).

    Args:
        element_vector_ref: Reference vector, shape (E,).
        scale_indices: Integer indices to scale, shape (M,).
        scales: Multipliers for those indices, shape (M,).
        set_indices: Optional integer indices to override, shape (L,).
        set_values: Values for overrides, shape (L,).

    Returns:
        Updated vector with scales and overrides applied.
    """
    b0 = jnp.asarray(element_vector_ref)
    idx = jnp.asarray(scale_indices, dtype=jnp.int32)
    s = jnp.asarray(scales, dtype=b0.dtype)
    out = b0
    out = out.at[idx].set(b0[idx] * s) if idx.size != 0 else out

    if set_indices is not None and set_values is not None:
        oidx = jnp.asarray(set_indices, dtype=jnp.int32)
        oval = jnp.asarray(set_values, dtype=b0.dtype)
        out = out.at[oidx].set(oval) if oidx.size != 0 else out
    return out


def element_indices_by_name(
    setup: ChemicalSetup, names: Iterable[str]
) -> jnp.ndarray:
    """Return integer indices for element symbols using ``setup.elements``.

    Note: call this outside JIT. Use the returned index array inside JAX code.

    Args:
        setup: ChemicalSetup providing the element ordering.
        names: Iterable of element symbols to locate (e.g., ["C", "O"]).

    Returns:
        jnp.ndarray of int32 indices in the same order as ``names``.
    """
    if setup.elements is None:
        raise ValueError("setup.elements is not available for index lookup.")
    pos = {e: i for i, e in enumerate(setup.elements)}
    idx = [pos[n] for n in names]
    return jnp.asarray(idx, dtype=jnp.int32)
