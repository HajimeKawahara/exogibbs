"""Offline contract tests for the optional ExoEOS solution adapter."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.interop.exoeos import make_solution_lngamma_func


class _RegularSolution:
    components = ("A", "B", "C")
    activity_basis = "mole_fraction"
    standard_state_convention = "symmetric"

    def gex_RT(self, temperature, pressure, composition):
        a, b, c = composition
        scale = 1000.0 / temperature + pressure / 2.0e8
        return scale * (2.0 * a * b + 3.0 * b * c + 5.0 * a * c)


def _solution_state(model, temperature, pressure, composition):
    def extensive_gex(amounts):
        total = jnp.sum(amounts)
        return total * model.gex_RT(temperature, pressure, amounts / total)

    return SimpleNamespace(lngamma=jax.grad(extensive_gex)(composition))


def _install_fake_exoeos(monkeypatch, solution_state=_solution_state):
    module = ModuleType("exoeos")
    module.solution_state = solution_state
    monkeypatch.setitem(sys.modules, "exoeos", module)


def _analytic_lngamma(temperature, pressure_bar, composition):
    """Independent partial-molar formula in the provider's A, B, C order."""
    a, b, c = np.asarray(composition)
    excess = 2.0 * a * b + 3.0 * b * c + 5.0 * a * c
    partials = np.asarray(
        [2.0 * b + 5.0 * c, 2.0 * a + 3.0 * c, 3.0 * b + 5.0 * a]
    )
    return (1000.0 / temperature + pressure_bar / 2000.0) * (partials - excess)


def test_maps_three_cycle_component_order_and_converts_bar_once(monkeypatch):
    calls = []
    model = _RegularSolution()

    def state(model, temperature, pressure, composition):
        calls.append((model, temperature, pressure, composition))
        return _solution_state(model, temperature, pressure, composition)

    _install_fake_exoeos(monkeypatch, state)
    callback = make_solution_lngamma_func(
        source_components=("B", "C", "A"), model=model,
    )
    result = callback(1500.0, 25.0, jnp.asarray([0.3, 0.5, 0.2]))

    expected = _analytic_lngamma(1500.0, 25.0, [0.2, 0.3, 0.5])
    np.testing.assert_allclose(result, expected[[1, 2, 0]], rtol=1.0e-6)
    assert len(calls) == 1
    assert calls[0][0] is model
    np.testing.assert_allclose(calls[0][1], 1500.0)
    np.testing.assert_allclose(calls[0][2], 2.5e6)
    np.testing.assert_array_equal(calls[0][3], [0.2, 0.3, 0.5])


def test_unlabelled_ideal_model_requires_explicit_order_and_adds_no_terms(monkeypatch):
    _install_fake_exoeos(monkeypatch)
    model = SimpleNamespace(
        activity_basis="mole_fraction",
        standard_state_convention="symmetric",
        gex_RT=lambda temperature, pressure, composition: jnp.sum(composition) * 0.0,
    )
    with pytest.raises(ValueError, match="model_components is required"):
        make_solution_lngamma_func(source_components=("A", "B"), model=model)

    callback = make_solution_lngamma_func(
        source_components=("B", "A"), model=model, model_components=("A", "B"),
    )
    np.testing.assert_array_equal(
        callback(1700.0, 300.0, jnp.asarray([0.1, 0.9])), [0.0, 0.0]
    )


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float32, jnp.float64])
def test_promotes_integer_states_and_preserves_floating_dtype(monkeypatch, dtype):
    _install_fake_exoeos(monkeypatch)
    callback = make_solution_lngamma_func(
        source_components=("A", "B", "C"), model=_RegularSolution(),
    )
    temperature = jnp.asarray(1500, dtype=dtype)
    pressure = jnp.asarray(25, dtype=dtype)
    composition = jnp.asarray([1, 0, 0], dtype=dtype)

    result = jax.jit(callback)(temperature, pressure, composition)

    assert result.dtype == (jnp.float32 if dtype == jnp.int32 else dtype)
    np.testing.assert_allclose(
        result, _analytic_lngamma(1500, 25, [1, 0, 0]), rtol=1.0e-6
    )


def test_jit_vmap_and_state_gradients_match_independent_formula(monkeypatch):
    _install_fake_exoeos(monkeypatch)
    callback = make_solution_lngamma_func(
        source_components=("B", "C", "A"), model=_RegularSolution(),
    )
    weights = jnp.asarray([1.0, 2.0, 4.0])

    def objective(parameters):
        temperature, pressure, a = parameters
        composition = jnp.asarray([0.3, 0.7 - a, a])
        return weights @ callback(temperature, pressure, composition)

    def analytic_objective(parameters):
        temperature, pressure, a = np.asarray(parameters)
        values = _analytic_lngamma(temperature, pressure, [a, 0.3, 0.7 - a])
        return np.asarray(weights) @ values[[1, 2, 0]]

    parameters = jnp.asarray([1500.0, 25.0, 0.2], dtype=jnp.float64)
    np.testing.assert_allclose(
        jax.jit(objective)(parameters), analytic_objective(parameters), rtol=1.0e-12
    )
    parameter_profile = jnp.stack(
        [parameters, parameters * jnp.asarray([1.1, 2.0, 1.2])]
    )
    profile = jax.jit(jax.vmap(objective))(parameter_profile)
    expected_profile = [analytic_objective(point) for point in parameter_profile]
    np.testing.assert_allclose(profile, expected_profile, rtol=1.0e-12)

    finite_difference = []
    for index, step in enumerate([0.01, 0.001, 1.0e-6]):
        offset = np.zeros(3)
        offset[index] = step
        finite_difference.append(
            (analytic_objective(parameters + offset)
             - analytic_objective(parameters - offset)) / (2.0 * step)
        )
    np.testing.assert_allclose(
        jax.jit(jax.grad(objective))(parameters), finite_difference,
        rtol=1.0e-8, atol=1.0e-9,
    )


@pytest.mark.parametrize("field", ["source_components", "model_components"])
@pytest.mark.parametrize("labels", ["ABC", [], ["A", "A"], ["A", ""], ["A", 1]])
def test_rejects_invalid_component_labels(field, labels):
    arguments = {
        "source_components": ("A", "B", "C"), "model": _RegularSolution(),
    }
    arguments[field] = labels
    with pytest.raises(ValueError, match=field):
        make_solution_lngamma_func(**arguments)


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"source_components": None}, "source_components"),
        ({"source_components": ("A", "B")}, "identical sets"),
        ({"model_components": ("C", "B", "A")}, "agree with model.components"),
    ],
)
def test_rejects_ambiguous_component_mapping(arguments, message):
    arguments = {
        "source_components": ("A", "B", "C"),
        "model": _RegularSolution(),
        **arguments,
    }
    with pytest.raises(ValueError, match=message):
        make_solution_lngamma_func(**arguments)


@pytest.mark.parametrize(
    "attribute, value, message",
    [
        ("activity_basis", None, "activity_basis"),
        ("activity_basis", "mass_fraction", "activity_basis"),
        ("standard_state_convention", None, "standard_state_convention"),
        ("standard_state_convention", "infinite_dilution", "standard_state_convention"),
        ("gex_RT", None, "gex_RT"),
    ],
)
def test_rejects_incompatible_model_contract(attribute, value, message):
    model = _RegularSolution()
    setattr(model, attribute, value)
    with pytest.raises(ValueError, match=message):
        make_solution_lngamma_func(source_components=model.components, model=model)


@pytest.mark.parametrize("module", [None, ModuleType("exoeos")])
def test_reports_missing_or_outdated_optional_provider(monkeypatch, module):
    monkeypatch.setitem(sys.modules, "exoeos", module)
    with pytest.raises(ImportError, match="requires ExoEOS with the solution_state API"):
        make_solution_lngamma_func(
            source_components=("A", "B", "C"), model=_RegularSolution(),
        )


@pytest.mark.parametrize(
    "temperature, pressure, composition, message",
    [
        (1500.0, 1.0, None, "not None"),
        (1500.0, 1.0, [0.5, 0.5], "mole_fractions must have shape"),
        (1500.0, 1.0, [[0.2, 0.3, 0.5]], "mole_fractions must have shape"),
        ([1500.0], 1.0, [0.2, 0.3, 0.5], "must be scalars"),
        (1500.0, [1.0], [0.2, 0.3, 0.5], "must be scalars"),
    ],
)
def test_rejects_ambiguous_state_shapes(
    monkeypatch, temperature, pressure, composition, message,
):
    _install_fake_exoeos(monkeypatch)
    callback = make_solution_lngamma_func(
        source_components=("A", "B", "C"), model=_RegularSolution(),
    )
    with pytest.raises(ValueError, match=message):
        callback(temperature, pressure, composition)


@pytest.mark.parametrize("shape", [(), (2,), (1, 3)])
def test_rejects_wrong_provider_output_shape(monkeypatch, shape):
    def state(model, temperature, pressure, composition):
        return SimpleNamespace(lngamma=jnp.zeros(shape))

    _install_fake_exoeos(monkeypatch, state)
    callback = make_solution_lngamma_func(
        source_components=("A", "B", "C"), model=_RegularSolution(),
    )
    with pytest.raises(ValueError, match="ExoEOS lngamma must have shape"):
        callback(1500.0, 1.0, jnp.asarray([0.2, 0.3, 0.5]))
