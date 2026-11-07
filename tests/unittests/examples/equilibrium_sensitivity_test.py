"""Focused scientific checks for the gas-only sensitivity example."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def example():
    path = Path(__file__).resolve().parents[3] / "examples/equilibrium_sensitivity.py"
    spec = importlib.util.spec_from_file_location("equilibrium_sensitivity_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def model(example):
    setup = example.build_setup()
    state = jax.jit(example.make_state_function(setup))
    response = jax.jit(jax.jacrev(lambda theta, t: state(theta, t)[0].ln_n))
    return SimpleNamespace(setup=setup, state=state, response=response)


def test_reaction_controls_conserve_atoms_and_independently_shift_logk(example, model):
    a = np.asarray(model.setup.formula_matrix)
    np.testing.assert_allclose(a @ example.REACTIONS, 0., atol=1.e-14)
    np.testing.assert_allclose(-example.REACTIONS.T @ example.LOGK_LIFT,
                               np.eye(2), atol=1.e-14)


@pytest.mark.parametrize("temperature,retained_temperature_paths", [
    (800., {0, 1}), (1000., {0}), (1200., {0}),
])
def test_response_and_selected_paths_match_equilibrium_checks(
    example, model, temperature, retained_temperature_paths,
):
    response = np.asarray(model.response(jnp.zeros(5), temperature))
    # Reuse the example's independent finite-difference and equilibrium checks.
    example.validate_response(model.setup, model.state, temperature, response)
    analysis = example.physical_control_analysis(model.setup, model.state, temperature, response)
    np.testing.assert_array_equal(np.asarray(analysis["drive_response"])[:, 1], [2., 0.])
    np.testing.assert_allclose(np.asarray(analysis["physical_response"])[:, 1],
                               2 * response[:, 3], atol=2.e-9, rtol=2.e-9)
    assert set(analysis["reductions"]["T"]["indices"]) == retained_temperature_paths
    assert set(analysis["reductions"]["P"]["indices"]) == {0}
    for reduced in analysis["reductions"].values():
        assert reduced["retained_absolute_fraction"] >= 0.99
        assert abs(reduced["full"] - reduced["retained"]) <= reduced["bound"] + 1.e-12


def test_path_reduction_reports_cancellation_and_an_absolute_error_bound(example):
    reduced = example.reduce_contributions(np.array([100., -99., 0.5]), retention=0.99)
    assert set(reduced["indices"]) == {0, 1}
    assert reduced["full"] == pytest.approx(1.5)
    assert reduced["retained"] == pytest.approx(1.)
    assert reduced["omitted"] == pytest.approx(0.5)
    assert reduced["bound"] == pytest.approx(0.5)
    assert reduced["retained_absolute_fraction"] >= 0.99
    # Retaining 99% of absolute contributions does not imply 1% relative error.
    assert reduced["relative_error"] == pytest.approx(1 / 3)
