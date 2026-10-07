"""Tests for oxygen-fugacity buffer helpers."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.thermo.oxygen_fugacity import (
    delta_iw_hirschmann2021,
    log10_oxygen_fugacity_iw_hirschmann2021,
)


@pytest.mark.parametrize(
    ("temperature_k", "pressure_gpa", "expected"),
    (
        (1500.0, 0.0001, -11.620840572317741),
        (2000.0, 10.0, -4.231349226207147),
        (2000.0, 60.0, 7.7466570207713525),
        (1000.0, 30.0, -5.039184460796159),
    ),
)
def test_iw_buffer_matches_hirschmann_table_1(
    temperature_k: float,
    pressure_gpa: float,
    expected: float,
) -> None:
    calculated = log10_oxygen_fugacity_iw_hirschmann2021(
        temperature_k,
        pressure_gpa,
    )

    np.testing.assert_allclose(calculated, expected, rtol=1.0e-12)


def test_iw_buffer_broadcasts_and_selects_both_iron_branches() -> None:
    calculated = log10_oxygen_fugacity_iw_hirschmann2021(
        jnp.asarray([1500.0, 2000.0]),
        jnp.asarray([0.0001, 60.0]),
    )

    np.testing.assert_allclose(
        calculated,
        np.asarray([-11.620840572317741, 7.7466570207713525]),
        rtol=1.0e-12,
    )


def test_delta_iw_uses_bar_fugacity_and_log10_offset() -> None:
    calculated = delta_iw_hirschmann2021(1.0e-9, 1500.0, 0.0001)

    np.testing.assert_allclose(calculated, 2.620840572317741, rtol=1.0e-12)


@pytest.mark.parametrize(
    "calculated",
    (
        log10_oxygen_fugacity_iw_hirschmann2021(0.0, 1.0),
        log10_oxygen_fugacity_iw_hirschmann2021(1500.0, -1.0),
        log10_oxygen_fugacity_iw_hirschmann2021(jnp.inf, 1.0),
        delta_iw_hirschmann2021(0.0, 1500.0, 1.0),
        delta_iw_hirschmann2021(-1.0, 1500.0, 1.0),
    ),
)
def test_iw_helpers_reject_invalid_inputs(calculated: jnp.ndarray) -> None:
    assert math.isnan(float(calculated))


def test_iw_buffer_supports_jit_and_automatic_differentiation() -> None:
    log10_iw = jax.jit(log10_oxygen_fugacity_iw_hirschmann2021)(
        1500.0,
        1.0,
    )
    derivative = jax.grad(log10_oxygen_fugacity_iw_hirschmann2021)(
        1500.0,
        1.0,
    )

    assert jnp.isfinite(log10_iw)
    assert jnp.isfinite(derivative)
