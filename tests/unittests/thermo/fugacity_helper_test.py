"""Tests for gas fugacity coefficient helpers."""

import jax.numpy as jnp
import pytest

from exogibbs.thermo.fugacity import effective_gas_hvector
from exogibbs.thermo.models import ChemicalSetup


def _setup() -> ChemicalSetup:
    return ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0, 2.0]]),
        hvector_func=lambda temperature: jnp.asarray(
            [temperature, 2.0 * temperature]
        ),
        species=("A", "B"),
    )


def test_effective_gas_hvector_defaults_to_ideal() -> None:
    result = effective_gas_hvector(_setup(), 3.0, 10.0)

    assert jnp.allclose(result, jnp.asarray([3.0, 6.0]))


def test_effective_gas_hvector_adds_pure_lnphi() -> None:
    def lnphi_func(temperature, pressure_bar, mole_fractions):
        assert mole_fractions is None
        return jnp.asarray([temperature, pressure_bar])

    result = effective_gas_hvector(
        _setup(),
        3.0,
        10.0,
        lnphi_func,
    )

    assert jnp.allclose(result, jnp.asarray([6.0, 16.0]))


def test_effective_gas_hvector_rejects_wrong_lnphi_shape() -> None:
    with pytest.raises(ValueError, match="expected shape \\(2,\\)"):
        effective_gas_hvector(
            _setup(),
            3.0,
            10.0,
            lambda temperature, pressure_bar, mole_fractions: jnp.zeros(1),
        )


def test_effective_gas_hvector_promotes_integer_ideal_values() -> None:
    setup = ChemicalSetup(
        formula_matrix=jnp.asarray([[1, 1]]),
        hvector_func=lambda temperature: jnp.asarray([0, 1]),
    )

    result = effective_gas_hvector(
        setup,
        3.0,
        10.0,
        lambda temperature, pressure_bar, mole_fractions: jnp.asarray(
            [0.25, 0.5]
        ),
    )

    assert jnp.allclose(result, jnp.asarray([0.25, 1.5]))
