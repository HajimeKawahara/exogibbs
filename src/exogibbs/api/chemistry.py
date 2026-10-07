"""Compatibility exports for thermochemical models and helpers."""

import jax.numpy as jnp

from exogibbs.equilibrium.gas.types import ThermoState
from exogibbs.thermo.composition import (
    element_indices_by_name,
    update_element_vector,
)
from exogibbs.thermo.fugacity import LogFugacityCoefficientFunction
from exogibbs.thermo.models import ChemicalSetup, setup_float_dtype


Array = jnp.ndarray


__all__ = (
    "Array",
    "ChemicalSetup",
    "LogFugacityCoefficientFunction",
    "ThermoState",
    "element_indices_by_name",
    "setup_float_dtype",
    "update_element_vector",
)
