"""Thermochemical input models shared by equilibrium features."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Tuple, Union

import jax.numpy as jnp


Array = jnp.ndarray


def setup_float_dtype() -> jnp.dtype:
    """Return the default floating dtype for setup-time solver inputs."""

    return jnp.asarray(1.0).dtype


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
    element_vector_reference : Optional[Array]
        Sample elemental abundance b (E,) for reference only.
    metadata : Optional[Mapping[str, Any]]
        Free-form provenance info (e.g., source="JANAF", preset="ykb4").
    temperature_validity_upper : Optional[tuple[float, ...]]
        Per-species upper temperature bounds used by phase eligibility logic.
    """

    formula_matrix: Array
    hvector_func: Callable[[Union[float, Array]], Array]
    elements: Optional[Tuple[str, ...]] = None
    species: Optional[Tuple[str, ...]] = None
    element_vector_reference: Optional[Array] = None
    metadata: Optional[Mapping[str, Any]] = None
    temperature_validity_upper: Optional[Tuple[float, ...]] = None
