"""Independent source-equation checks for differentiable local thermochemistry."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


EXAMPLE_DIRECTORY = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate"


def _load_module(name):
    spec = importlib.util.spec_from_file_location(
        f"metal_silicate_{name}", EXAMPLE_DIRECTORY / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFERENCE = _load_module("reference")
SOURCE = _load_module("source")
RECORD = REFERENCE.load_reference()


@pytest.mark.parametrize("case", RECORD["cases"], ids=lambda case: case["id"])
def test_jitted_standards_recover_frozen_source_reactions(case) -> None:
    standard = SOURCE.make_source_standard_potentials_rt(RECORD, case)
    compiled = jax.jit(standard)
    actual = compiled(case["T_K"], case["P_bar"])
    expected = REFERENCE.source_standard_potentials(RECORD, case)
    rt = RECORD["source"]["gas_constant_J_mol_K"] * case["T_K"]
    np.testing.assert_allclose(actual, expected / rt, rtol=5e-13, atol=5e-13)
    _, _, reactions = REFERENCE.component_matrices(RECORD)
    np.testing.assert_allclose(
        reactions @ actual, case["source_delta_g_over_rt"], rtol=5e-12, atol=5e-12,
    )
    np.testing.assert_array_equal(compiled(case["T_K"], 1.0e4), actual)


@pytest.mark.parametrize("case", RECORD["cases"], ids=lambda case: case["id"])
def test_standard_temperature_derivative_matches_independent_shomate(case) -> None:
    standard = SOURCE.make_source_standard_potentials_rt(RECORD, case)
    derivative = jax.jit(jax.jacrev(standard, argnums=0))(case["T_K"], 1.0)
    step = 0.02

    def independent(temperature):
        state = {**case, "T_K": temperature}
        rt = RECORD["source"]["gas_constant_J_mol_K"] * temperature
        return REFERENCE.source_standard_potentials(RECORD, state) / rt

    expected = (independent(case["T_K"] + step) - independent(case["T_K"] - step)) / (2 * step)
    np.testing.assert_allclose(derivative, expected, atol=2e-12, rtol=2e-9)


def _literal_source_solute_coefficients(temperature, composition):
    # Keep the original quotients to check the separate zero-safe JAX algebra.
    si, oxygen = composition[1:3]
    cross = -5.0 * 1873.0 / temperature
    ln_si = (
        -6.65 * 1873.0 / temperature
        - 12.41 * 1873.0 / temperature * np.log(1 - si)
        - cross * oxygen * (1 + np.log(1 - oxygen) / oxygen - 1 / (1 - si))
        + cross * oxygen**2 * si
        * (1 / (1 - si) + 1 / (1 - oxygen) + si / (2 * (1 - si)**2) - 1)
    )
    ln_o = (
        4.29 - 16500.0 / temperature
        + 1873.0 / temperature * np.log(1 - oxygen)
        - cross * si * (1 + np.log(1 - si) / si - 1 / (1 - oxygen))
        + cross * si**2 * oxygen
        * (1 / (1 - oxygen) + 1 / (1 - si) + oxygen / (2 * (1 - oxygen)**2) - 1)
    )
    return np.asarray([0.0, ln_si, ln_o, 0.0])


@pytest.mark.parametrize("temperature", [2350.0, 3000.0])
@pytest.mark.parametrize("composition", [[0.6, 0.07, 0.03, 0.3], [0.8, 0.15, 0.05, 0.0]])
def test_source_alloy_matches_literal_four_component_equations(temperature, composition) -> None:
    actual = jax.jit(SOURCE.source_metal_ln_gamma)(temperature, 1.0, jnp.asarray(composition))
    expected = _literal_source_solute_coefficients(temperature, composition)
    np.testing.assert_allclose(actual, expected, atol=5e-14, rtol=5e-14)
    # H dilutes the source Si/O fractions; it is not stripped before evaluation.
    ternary_normalized = np.asarray(composition)[:3] / np.sum(composition[:3])
    if composition[3] > 0:
        wrong = _literal_source_solute_coefficients(temperature, ternary_normalized)
        assert np.max(np.abs(actual - wrong)) > 0.1


@pytest.mark.parametrize("composition", [[0.8, 0.0, 0.0, 0.2], [0.9, 0.1, 0.0, 0.0], [0.9, 0.0, 0.1, 0.0]])
def test_zero_solute_limits_and_composition_derivatives_remain_finite(composition) -> None:
    composition = jnp.asarray(composition)
    actual = jax.jit(SOURCE.source_metal_ln_gamma)(2350.0, 1.0, composition)
    derivative = jax.jit(jax.jacrev(SOURCE.source_metal_ln_gamma, argnums=2))(
        2350.0, 1.0, composition,
    )
    assert np.all(np.isfinite(actual))
    assert np.all(np.isfinite(derivative))
    np.testing.assert_array_equal(np.asarray(actual)[[0, 3]], [0.0, 0.0])
    if composition[1] == composition[2] == 0:
        np.testing.assert_allclose(actual[1:3], [-6.65 * 1873 / 2350, 4.29 - 16500 / 2350])


def test_alloy_temperature_and_composition_derivatives_match_source_differences() -> None:
    temperature = 2350.0
    composition = np.asarray([0.6, 0.07, 0.03, 0.3])
    dt = 0.01
    derivative = jax.jit(jax.jacrev(SOURCE.source_metal_ln_gamma, argnums=0))(
        temperature, 1.0, composition,
    )
    expected = (
        _literal_source_solute_coefficients(temperature + dt, composition)
        - _literal_source_solute_coefficients(temperature - dt, composition)
    ) / (2 * dt)
    np.testing.assert_allclose(derivative, expected, atol=1e-12, rtol=1e-9)
    direction = np.asarray([-1.0, 0.4, 0.6, 0.0])
    derivative = jax.jacrev(SOURCE.source_metal_ln_gamma, argnums=2)(
        temperature, 1.0, composition,
    ) @ direction
    dx = 1e-6
    expected = (
        _literal_source_solute_coefficients(temperature, composition + dx * direction)
        - _literal_source_solute_coefficients(temperature, composition - dx * direction)
    ) / (2 * dx)
    np.testing.assert_allclose(derivative, expected, atol=2e-9, rtol=1e-9)


@pytest.mark.parametrize("pressure", [1.0, 123.0, 1e4])
def test_r14_exception_retains_fixed_source_dissolution_pressure(pressure) -> None:
    offsets = jax.jit(SOURCE.source_reaction_offsets)(2350.0, pressure)
    np.testing.assert_array_equal(np.delete(offsets, 14), np.zeros(17))
    # Ordinary gas chemical potentials carry ln(P/1 bar).
    ordinary = 12.500076 - np.log(pressure)
    np.testing.assert_allclose(ordinary + offsets[14], 12.500076 - np.log(1e4), atol=2e-15)
    slope = jax.grad(lambda log_p: SOURCE.source_reaction_offsets(2350.0, jnp.exp(log_p))[14])(
        jnp.log(pressure),
    )
    np.testing.assert_allclose(slope, 1.0, atol=1e-15)
