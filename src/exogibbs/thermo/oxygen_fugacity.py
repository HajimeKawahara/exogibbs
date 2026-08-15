"""Oxygen-fugacity buffer models."""

from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike


# Hirschmann (2021), Table 1. Each row contains m0 through m4 for
# m(P) = m0 + m1 P + m2 P**2 + m3 P**3 + m4 sqrt(P), with P in GPa.
_FCC_BCC_COEFFICIENTS = (
    (6.844864, 1.175691e-1, 1.143873e-3, 0.0, 0.0),
    (5.791364e-4, -2.891434e-4, -2.737171e-7, 0.0, 0.0),
    (-7.971469e-5, 3.198005e-5, 0.0, 1.059554e-10, 2.014461e-7),
    (-2.769002e4, 5.285977e2, -2.919275, 0.0, 0.0),
)
_HCP_COEFFICIENTS = (
    (8.463095, -3.000307e-3, 7.213445e-5, 0.0, 0.0),
    (1.148738e-3, -9.352312e-5, 5.161592e-7, 0.0, 0.0),
    (-7.448624e-4, -6.329325e-6, 0.0, -1.407339e-10, 1.830014e-4),
    (-2.782082e4, 5.285977e2, -8.473231e-1, 0.0, 0.0),
)


__all__ = (
    "delta_iw_hirschmann2021",
    "log10_oxygen_fugacity_iw_hirschmann2021",
)


def _valid_nonnegative(value: jnp.ndarray) -> jnp.ndarray:
    """Return whether values are finite and nonnegative."""

    return jnp.isfinite(value) & (value >= 0.0)


def _pressure_polynomial(
    coefficients: tuple[float, float, float, float, float],
    pressure_gpa: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate one pressure-dependent coefficient from Table 1."""

    m0, m1, m2, m3, m4 = coefficients
    return (
        m0
        + m1 * pressure_gpa
        + m2 * pressure_gpa**2
        + m3 * pressure_gpa**3
        + m4 * jnp.sqrt(pressure_gpa)
    )


def _iw_branch(
    temperature_k: jnp.ndarray,
    pressure_gpa: jnp.ndarray,
    coefficients: tuple[
        tuple[float, float, float, float, float],
        tuple[float, float, float, float, float],
        tuple[float, float, float, float, float],
        tuple[float, float, float, float, float],
    ],
) -> jnp.ndarray:
    """Evaluate one Hirschmann (2021) iron-polymorph branch."""

    a, b, c, d = (
        _pressure_polynomial(row, pressure_gpa) for row in coefficients
    )
    return a + b * temperature_k + c * temperature_k * jnp.log(
        temperature_k
    ) + d / temperature_k


def log10_oxygen_fugacity_iw_hirschmann2021(
    temperature_k: ArrayLike,
    pressure_gpa: ArrayLike,
) -> jnp.ndarray:
    """Return ``log10(f_O2^IW / 1 bar)``.

    This implements both iron-polymorph expressions and their pressure
    boundary from Hirschmann (2021), Table 1
    (doi:10.1016/j.gca.2021.08.039). Inputs are not clipped to the published
    1000--3000 K and 0.0001--100 GPa calibration domain. Nonpositive
    temperatures and negative or non-finite pressures return ``nan``.
    """

    temperature = jnp.asarray(temperature_k)
    pressure = jnp.asarray(pressure_gpa)
    fcc_bcc = _iw_branch(temperature, pressure, _FCC_BCC_COEFFICIENTS)
    hcp = _iw_branch(temperature, pressure, _HCP_COEFFICIENTS)
    hcp_boundary_gpa = (
        -18.64 + 0.04359 * temperature - 5.069e-6 * temperature**2
    )
    result = jnp.where(pressure > hcp_boundary_gpa, hcp, fcc_bcc)
    valid = (
        jnp.isfinite(temperature)
        & (temperature > 0.0)
        & _valid_nonnegative(pressure)
    )
    return jnp.where(valid, result, jnp.nan)


def delta_iw_hirschmann2021(
    oxygen_fugacity_bar: ArrayLike,
    temperature_k: ArrayLike,
    pressure_gpa: ArrayLike,
) -> jnp.ndarray:
    """Return ``log10(f_O2 / f_O2^IW)`` for fugacity supplied in bar."""

    oxygen_fugacity = jnp.asarray(oxygen_fugacity_bar)
    log10_iw = log10_oxygen_fugacity_iw_hirschmann2021(
        temperature_k,
        pressure_gpa,
    )
    result = jnp.log10(oxygen_fugacity) - log10_iw
    valid = jnp.isfinite(oxygen_fugacity) & (oxygen_fugacity > 0.0)
    return jnp.where(valid, result, jnp.nan)
