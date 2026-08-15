"""JAX-compatible physical-unit conversions."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jax.typing import ArrayLike


PressureUnit = Literal["Pa", "bar", "GPa"]

_PRESSURE_IN_PA = {
    "Pa": 1.0,
    "bar": 1.0e5,
    "GPa": 1.0e9,
}


__all__ = ("PressureUnit", "convert_pressure")


def convert_pressure(
    pressure: ArrayLike,
    *,
    from_unit: PressureUnit,
    to_unit: PressureUnit,
) -> jnp.ndarray:
    """Convert pressure among Pa, bar, and GPa by pure scaling.

    Unit names are deliberately case-sensitive so that ``Pa`` and ``GPa``
    remain explicit. Physical-domain validation belongs to the caller.
    """

    try:
        factor = _PRESSURE_IN_PA[from_unit] / _PRESSURE_IN_PA[to_unit]
    except KeyError as error:
        valid_units = ", ".join(_PRESSURE_IN_PA)
        raise ValueError(
            f"pressure unit must be one of: {valid_units}"
        ) from error

    pressure_array = jnp.asarray(pressure)
    return pressure_array * factor
