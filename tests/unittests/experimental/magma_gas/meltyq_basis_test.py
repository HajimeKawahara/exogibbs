"""Tests for MELTYQ-specific melt-basis adapters."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.experimental.magma_gas.meltyq_basis import (
    MELTYQ_MEAN_MELT_MOLAR_MASS_G_MOL,
    co2_mass_fraction_to_mole_ratio,
    elemental_c_ln_mass_fraction_to_ln_mole_ratio,
    elemental_c_mass_fraction_to_mole_ratio,
    elemental_n_ln_mass_fraction_to_ln_mole_ratio,
    elemental_n_mass_fraction_to_mole_ratio,
    h2o_mass_fraction_to_mole_ratio,
)


@pytest.mark.parametrize(
    ("converter", "molar_mass_g_mol"),
    (
        (h2o_mass_fraction_to_mole_ratio, 18.01528),
        (co2_mass_fraction_to_mole_ratio, 44.0095),
        (elemental_c_mass_fraction_to_mole_ratio, 12.0107),
        (elemental_n_mass_fraction_to_mole_ratio, 14.0067),
    ),
)
def test_native_mass_bases_convert_to_common_dilute_mole_ratio(
    converter,
    molar_mass_g_mol: float,
) -> None:
    calculated = converter(0.01)
    expected = 0.01 * MELTYQ_MEAN_MELT_MOLAR_MASS_G_MOL / molar_mass_g_mol

    np.testing.assert_allclose(calculated, expected, rtol=1.0e-12)


def test_elemental_converters_use_atomic_not_molecular_mass() -> None:
    carbon = elemental_c_mass_fraction_to_mole_ratio(0.01)
    nitrogen = elemental_n_mass_fraction_to_mole_ratio(0.01)

    np.testing.assert_allclose(carbon, 0.01 * 60.0 / 12.0107, rtol=1.0e-12)
    np.testing.assert_allclose(nitrogen, 0.01 * 60.0 / 14.0067, rtol=1.0e-12)


def test_zero_mass_fraction_has_zero_mole_ratio() -> None:
    assert float(h2o_mass_fraction_to_mole_ratio(0.0)) == 0.0


def test_basis_adapter_supports_jit_and_automatic_differentiation() -> None:
    converted = jax.jit(h2o_mass_fraction_to_mole_ratio)(0.01)
    derivative = jax.grad(h2o_mass_fraction_to_mole_ratio)(0.01)

    assert jnp.isfinite(converted)
    assert jnp.isfinite(derivative)


@pytest.mark.parametrize(
    ("linear_converter", "log_converter"),
    (
        (
            elemental_c_mass_fraction_to_mole_ratio,
            elemental_c_ln_mass_fraction_to_ln_mole_ratio,
        ),
        (
            elemental_n_mass_fraction_to_mole_ratio,
            elemental_n_ln_mass_fraction_to_ln_mole_ratio,
        ),
    ),
)
def test_elemental_log_adapters_match_linear_adapters(
    linear_converter,
    log_converter,
) -> None:
    mass_fraction = 0.01

    np.testing.assert_allclose(
        log_converter(np.log(mass_fraction)),
        jnp.log(linear_converter(mass_fraction)),
        rtol=1.0e-6,
    )


@pytest.mark.parametrize(
    "converter",
    (
        elemental_c_ln_mass_fraction_to_ln_mole_ratio,
        elemental_n_ln_mass_fraction_to_ln_mole_ratio,
    ),
)
def test_elemental_log_adapters_preserve_extreme_values_and_support_ad(
    converter,
) -> None:
    converted = jax.jit(converter)(-1000.0)
    derivative = jax.grad(converter)(-5.0)

    assert jnp.isfinite(converted)
    np.testing.assert_allclose(derivative, 1.0, rtol=0.0, atol=0.0)
