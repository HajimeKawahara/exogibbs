"""Tests for thermodynamic composition-basis helpers."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.thermo.composition import (
    ln_mass_fraction_to_ln_dilute_mole_ratio,
    mass_fraction_to_dilute_mole_ratio,
)


def test_mass_fraction_converts_to_dilute_mole_ratio() -> None:
    calculated = mass_fraction_to_dilute_mole_ratio(
        0.01,
        18.0,
        matrix_molar_mass_g_mol=60.0,
    )

    np.testing.assert_allclose(calculated, 0.01 * 60.0 / 18.0, rtol=1.0e-12)


@pytest.mark.parametrize(
    "calculated",
    (
        mass_fraction_to_dilute_mole_ratio(
            -0.01,
            18.0,
            matrix_molar_mass_g_mol=60.0,
        ),
        mass_fraction_to_dilute_mole_ratio(
            1.01,
            18.0,
            matrix_molar_mass_g_mol=60.0,
        ),
        mass_fraction_to_dilute_mole_ratio(
            jnp.inf,
            18.0,
            matrix_molar_mass_g_mol=60.0,
        ),
        mass_fraction_to_dilute_mole_ratio(
            0.01,
            0.0,
            matrix_molar_mass_g_mol=60.0,
        ),
        mass_fraction_to_dilute_mole_ratio(
            0.01,
            18.0,
            matrix_molar_mass_g_mol=-60.0,
        ),
    ),
)
def test_mole_ratio_conversion_rejects_invalid_inputs(
    calculated: jnp.ndarray,
) -> None:
    assert math.isnan(float(calculated))


@pytest.mark.parametrize("mass_fraction", (1.0e-12, 0.01, 1.0))
def test_log_conversion_matches_linear_conversion(mass_fraction: float) -> None:
    linear = mass_fraction_to_dilute_mole_ratio(
        mass_fraction,
        18.0,
        matrix_molar_mass_g_mol=60.0,
    )
    logarithmic = ln_mass_fraction_to_ln_dilute_mole_ratio(
        jnp.log(mass_fraction),
        18.0,
        matrix_molar_mass_g_mol=60.0,
    )

    np.testing.assert_allclose(logarithmic, jnp.log(linear), rtol=1.0e-6)


def test_log_conversion_preserves_zero_and_extreme_log_values() -> None:
    zero = ln_mass_fraction_to_ln_dilute_mole_ratio(
        -jnp.inf,
        18.0,
        matrix_molar_mass_g_mol=60.0,
    )
    extreme = ln_mass_fraction_to_ln_dilute_mole_ratio(
        -1000.0,
        18.0,
        matrix_molar_mass_g_mol=60.0,
    )

    assert jnp.isneginf(zero)
    assert jnp.isfinite(extreme)
    np.testing.assert_allclose(
        extreme,
        -1000.0 + np.log(60.0 / 18.0),
        rtol=1.0e-7,
    )


@pytest.mark.parametrize(
    "calculated",
    (
        ln_mass_fraction_to_ln_dilute_mole_ratio(
            0.01,
            18.0,
            matrix_molar_mass_g_mol=60.0,
        ),
        ln_mass_fraction_to_ln_dilute_mole_ratio(
            jnp.nan,
            18.0,
            matrix_molar_mass_g_mol=60.0,
        ),
        ln_mass_fraction_to_ln_dilute_mole_ratio(
            jnp.inf,
            18.0,
            matrix_molar_mass_g_mol=60.0,
        ),
        ln_mass_fraction_to_ln_dilute_mole_ratio(
            -1.0,
            0.0,
            matrix_molar_mass_g_mol=60.0,
        ),
        ln_mass_fraction_to_ln_dilute_mole_ratio(
            -1.0,
            18.0,
            matrix_molar_mass_g_mol=jnp.inf,
        ),
    ),
)
def test_log_mole_ratio_conversion_rejects_invalid_inputs(
    calculated: jnp.ndarray,
) -> None:
    assert math.isnan(float(calculated))


def test_log_conversion_supports_jit_and_automatic_differentiation() -> None:
    def convert(ln_mass_fraction):
        return ln_mass_fraction_to_ln_dilute_mole_ratio(
            ln_mass_fraction,
            18.0,
            matrix_molar_mass_g_mol=60.0,
        )

    converted = jax.jit(convert)(-5.0)
    derivative = jax.grad(convert)(-5.0)

    assert jnp.isfinite(converted)
    np.testing.assert_allclose(derivative, 1.0, rtol=0.0, atol=0.0)
