"""Tests for unit-conversion helpers."""

import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.utils.units import convert_pressure


@pytest.mark.parametrize(
    ("pressure", "from_unit", "to_unit", "expected"),
    (
        (1.0e9, "Pa", "GPa", 1.0),
        (1.0e5, "Pa", "bar", 1.0),
        (1.0, "GPa", "bar", 1.0e4),
        (1.0, "bar", "Pa", 1.0e5),
        (0.0, "GPa", "Pa", 0.0),
    ),
)
def test_convert_pressure_between_supported_units(
    pressure: float,
    from_unit: str,
    to_unit: str,
    expected: float,
) -> None:
    calculated = convert_pressure(
        pressure,
        from_unit=from_unit,
        to_unit=to_unit,
    )

    np.testing.assert_allclose(calculated, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("pressure", "expected"),
    ((-1.0, -1.0e5), (jnp.inf, jnp.inf), (jnp.nan, jnp.nan)),
)
def test_convert_pressure_is_pure_scaling(
    pressure: float,
    expected: float,
) -> None:
    calculated = convert_pressure(pressure, from_unit="bar", to_unit="Pa")

    np.testing.assert_allclose(
        calculated,
        expected,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_convert_pressure_rejects_unknown_units() -> None:
    with pytest.raises(ValueError, match="pressure unit"):
        convert_pressure(1.0, from_unit="atm", to_unit="bar")
