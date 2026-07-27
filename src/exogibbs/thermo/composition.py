"""Element-composition helpers shared by equilibrium features."""

from typing import Iterable, Optional

import jax.numpy as jnp

from exogibbs.thermo.models import ChemicalSetup


def update_element_vector(
    element_vector_ref: jnp.ndarray,
    scale_indices: jnp.ndarray,
    scales: jnp.ndarray,
    *,
    set_indices: Optional[jnp.ndarray] = None,
    set_values: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Build a new element vector by scaling and overriding entries."""

    b0 = jnp.asarray(element_vector_ref)
    idx = jnp.asarray(scale_indices, dtype=jnp.int32)
    scale_values = jnp.asarray(scales, dtype=b0.dtype)
    out = b0
    if idx.size != 0:
        out = out.at[idx].set(b0[idx] * scale_values)

    if set_indices is not None and set_values is not None:
        override_indices = jnp.asarray(set_indices, dtype=jnp.int32)
        override_values = jnp.asarray(set_values, dtype=b0.dtype)
        if override_indices.size != 0:
            out = out.at[override_indices].set(override_values)
    return out


def element_indices_by_name(
    setup: ChemicalSetup,
    names: Iterable[str],
) -> jnp.ndarray:
    """Return element indices in the same order as ``names``."""

    if setup.elements is None:
        raise ValueError("setup.elements is not available for index lookup.")
    positions = {element: index for index, element in enumerate(setup.elements)}
    return jnp.asarray([positions[name] for name in names], dtype=jnp.int32)
