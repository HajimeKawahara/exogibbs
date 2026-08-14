"""Gas-phase fugacity coefficient interfaces and helpers."""

from typing import Callable, Optional, Union

import jax.numpy as jnp

from exogibbs.thermo.models import ChemicalSetup


Array = jnp.ndarray
Scalar = Union[float, Array]
LogFugacityCoefficientFunction = Callable[[Scalar, Scalar, Optional[Array]], Array]


def effective_gas_hvector(
    setup: ChemicalSetup,
    temperature: Scalar,
    pressure_bar: Scalar,
    lnphi_func: Optional[LogFugacityCoefficientFunction] = None,
    *,
    mole_fractions: Optional[Array] = None,
) -> Array:
    """Return the ideal gas source corrected by ``ln(phi)``.

    ``lnphi_func`` receives temperature, physical pressure in bar, and gas mole
    fractions.  Pure-component mode passes ``None`` for the mole fractions.
    Its result must follow ``setup.species`` order and contain natural-log,
    dimensionless fugacity coefficients.
    """

    ideal_hvector = jnp.asarray(setup.hvector_func(temperature))
    expected_shape = (int(setup.formula_matrix.shape[1]),)
    if ideal_hvector.shape != expected_shape:
        raise ValueError(
            "hvector_func must return one value per gas species: "
            f"expected shape {expected_shape}, got {ideal_hvector.shape}."
        )
    if lnphi_func is None:
        return ideal_hvector

    lnphi = jnp.asarray(
        lnphi_func(temperature, pressure_bar, mole_fractions),
        dtype=ideal_hvector.dtype,
    )
    if lnphi.shape != expected_shape:
        raise ValueError(
            "lnphi_func must return one value per gas species in setup.species "
            f"order: expected shape {expected_shape}, got {lnphi.shape}."
        )
    return ideal_hvector + lnphi


__all__ = ["LogFugacityCoefficientFunction", "effective_gas_hvector"]
