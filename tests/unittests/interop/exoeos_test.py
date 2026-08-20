"""Tests for the optional ExoEOS fugacity adapter."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.interop.exoeos import make_pure_lnphi_func


class _FakeEOS:
    def __init__(self, value: float, component_count: int = 1) -> None:
        self.value = value
        self.component_count = component_count


def _install_fake_exoeos(monkeypatch, state_tp) -> None:
    module = ModuleType("exoeos")
    module.state_tp = state_tp
    monkeypatch.setitem(sys.modules, "exoeos", module)


def test_maps_by_source_order_and_adapts_scalar_states(monkeypatch) -> None:
    calls = []

    def state_tp(eos, temperature, pressure, composition, phase="vapor"):
        calls.append((eos, temperature, pressure, composition, phase))
        value = jnp.asarray(eos.value, dtype=composition.dtype)
        value = value + pressure / jnp.asarray(1.0e5, dtype=composition.dtype)
        return SimpleNamespace(lnphi=jnp.reshape(value, (1,)))

    _install_fake_exoeos(monkeypatch, state_tp)
    model_a = _FakeEOS(10.0)
    model_b = _FakeEOS(20.0)
    lnphi_func = make_pure_lnphi_func(
        source_species=("B", "IDEAL", "A"),
        eos_by_species={"A": model_a, "B": model_b},
        unspecified_species="ideal",
        phase="liquid",
    )

    result = lnphi_func(
        jnp.asarray(1200.0, dtype=jnp.float32),
        jnp.asarray(2.5, dtype=jnp.float32),
        None,
    )

    np.testing.assert_allclose(result, np.asarray([22.5, 0.0, 12.5]))
    assert result.dtype == jnp.float32
    assert [call[0] for call in calls] == [model_b, model_a]
    for _, temperature, pressure, composition, phase in calls:
        assert temperature.dtype == jnp.float32
        assert pressure.dtype == jnp.float32
        assert phase == "liquid"
        np.testing.assert_allclose(pressure, 2.5e5)
        np.testing.assert_array_equal(composition, np.asarray([1.0]))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_callback_supports_jit_vmap_grad_and_preserves_dtype(
    monkeypatch,
    dtype,
) -> None:
    def state_tp(eos, temperature, pressure, composition, phase="vapor"):
        del phase
        scale = jnp.asarray(eos.value, dtype=composition.dtype)
        pressure_scale = jnp.asarray(1.0e6, dtype=composition.dtype)
        value = scale * temperature + pressure / pressure_scale
        return SimpleNamespace(lnphi=jnp.reshape(value, (1,)))

    _install_fake_exoeos(monkeypatch, state_tp)
    lnphi_func = make_pure_lnphi_func(
        source_species=("active", "ideal"),
        eos_by_species={"active": _FakeEOS(0.25)},
        unspecified_species="ideal",
    )
    temperature = jnp.asarray(2.0, dtype=dtype)
    pressure = jnp.asarray(3.0, dtype=dtype)

    result = jax.jit(lambda T, P: lnphi_func(T, P, None))(
        temperature,
        pressure,
    )
    profile = jax.vmap(lambda P: lnphi_func(temperature, P, None))(
        jnp.asarray([1.0, 3.0], dtype=dtype)
    )
    pressure_gradient = jax.grad(
        lambda P: jnp.sum(lnphi_func(temperature, P, None))
    )(pressure)

    assert result.dtype == dtype
    assert profile.dtype == dtype
    np.testing.assert_allclose(result, np.asarray([0.8, 0.0]), rtol=1.0e-6)
    np.testing.assert_allclose(
        profile,
        np.asarray([[0.6, 0.0], [0.8, 0.0]]),
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(pressure_gradient, 0.1, rtol=1.0e-6)


def test_all_ideal_callback_returns_zeros_and_rejects_mixture_mode() -> None:
    lnphi_func = make_pure_lnphi_func(
        source_species=("He", "N2", "NH3"),
        eos_by_species={},
        unspecified_species="ideal",
    )

    result = lnphi_func(
        jnp.asarray(1000.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        None,
    )

    np.testing.assert_array_equal(result, np.zeros(3))
    assert result.dtype == jnp.float32
    with pytest.raises(ValueError, match="mole_fractions must be None"):
        lnphi_func(1000.0, 1.0, jnp.ones(3) / 3.0)


def test_validates_species_mapping_and_component_count() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        make_pure_lnphi_func(
            source_species=("H2", "H2"),
            eos_by_species={},
            unspecified_species="ideal",
        )
    with pytest.raises(ValueError, match="absent from source_species"):
        make_pure_lnphi_func(
            source_species=("H2",),
            eos_by_species={"O2": _FakeEOS(0.0)},
        )
    with pytest.raises(ValueError, match="missing source species"):
        make_pure_lnphi_func(
            source_species=("H2", "He"),
            eos_by_species={"H2": _FakeEOS(0.0)},
        )
    with pytest.raises(ValueError, match="must be 'error' or 'ideal'"):
        make_pure_lnphi_func(
            source_species=("H2",),
            eos_by_species={"H2": _FakeEOS(0.0)},
            unspecified_species="zero",
        )
    with pytest.raises(ValueError, match="got None"):
        make_pure_lnphi_func(
            source_species=("H2",),
            eos_by_species={"H2": None},
        )
    with pytest.raises(ValueError, match="one-component EOS"):
        make_pure_lnphi_func(
            source_species=("H2",),
            eos_by_species={"H2": _FakeEOS(0.0, component_count=2)},
        )


def test_validates_state_lnphi_shape_when_component_count_is_unavailable(
    monkeypatch,
) -> None:
    class ModelWithoutComponentCount:
        pass

    def state_tp(eos, temperature, pressure, composition, phase="vapor"):
        del eos, temperature, pressure, composition, phase
        return SimpleNamespace(lnphi=jnp.zeros((2,)))

    _install_fake_exoeos(monkeypatch, state_tp)
    lnphi_func = make_pure_lnphi_func(
        source_species=("H2",),
        eos_by_species={"H2": ModelWithoutComponentCount()},
    )

    with pytest.raises(ValueError, match="must return one fugacity coefficient"):
        lnphi_func(1000.0, 1.0, None)


def test_factory_reports_missing_or_outdated_exoeos(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "exoeos", ModuleType("exoeos"))

    with pytest.raises(ImportError, match="current ExoEOS checkout"):
        make_pure_lnphi_func(
            source_species=("H2",),
            eos_by_species={"H2": _FakeEOS(0.0)},
        )
