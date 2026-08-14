import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.equilibrium.gas.profile import equilibrium_profile
from exogibbs.equilibrium.gas.solve import equilibrium
from exogibbs.equilibrium.gas.types import EquilibriumOptions
from exogibbs.thermo.models import ChemicalSetup


jax.config.update("jax_enable_x64", True)

_B = jnp.asarray([1.0], dtype=jnp.float64)
_OPTIONS = EquilibriumOptions(epsilon_crit=1.0e-12, max_iter=200)


def _setup() -> ChemicalSetup:
    return ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0, 1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.zeros((2,), dtype=jnp.float64),
        elements=("E",),
        species=("A", "B"),
    )


def _expected_mole_fractions(delta):
    return jax.nn.softmax(jnp.asarray([-delta, delta]))


def test_none_and_zero_lnphi_preserve_ideal_solution():
    setup = _setup()
    ideal = equilibrium(setup, 900.0, 3.0, _B, options=_OPTIONS)
    zero = equilibrium(
        setup,
        900.0,
        3.0,
        _B,
        options=_OPTIONS,
        lnphi_func=lambda temperature, pressure, mole_fractions: jnp.zeros((2,)),
    )

    np.testing.assert_allclose(zero.ln_n, ideal.ln_n, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(ideal.x, jnp.asarray([0.5, 0.5]), atol=1.0e-12)


def test_pure_lnphi_changes_equilibrium_and_receives_physical_state():
    setup = _setup()
    calls = []

    def lnphi_func(temperature, pressure_bar, mole_fractions):
        calls.append((temperature, pressure_bar, mole_fractions))
        return jnp.asarray([0.4, -0.4])

    result = equilibrium(
        setup,
        850.0,
        7.5,
        _B,
        Pref=2.5,
        options=_OPTIONS,
        lnphi_func=lnphi_func,
    )

    assert calls == [(850.0, 7.5, None)]
    np.testing.assert_allclose(result.x, _expected_mole_fractions(0.4), atol=1.0e-12)
    assert not np.allclose(result.x, jnp.asarray([0.5, 0.5]))


def test_jit_grad_includes_lnphi_temperature_and_pressure_derivatives():
    setup = _setup()
    temperature_scale = 2.0e-3
    pressure_scale = 0.15

    def lnphi_func(temperature, pressure_bar, mole_fractions):
        del mole_fractions
        delta = temperature_scale * temperature + pressure_scale * pressure_bar
        return jnp.asarray([delta, -delta])

    @jax.jit
    def solve_ln_n(temperature, pressure_bar):
        return equilibrium(
            setup,
            temperature,
            pressure_bar,
            _B,
            options=_OPTIONS,
            lnphi_func=lnphi_func,
        ).ln_n

    temperature = 400.0
    pressure_bar = 2.0
    dln_dT, dln_dP = jax.jacrev(solve_ln_n, argnums=(0, 1))(
        temperature, pressure_bar
    )
    delta = temperature_scale * temperature + pressure_scale * pressure_bar
    response = jnp.asarray([-1.0 - jnp.tanh(delta), 1.0 - jnp.tanh(delta)])

    np.testing.assert_allclose(dln_dT, response * temperature_scale, atol=2.0e-11)
    np.testing.assert_allclose(dln_dP, response * pressure_scale, atol=2.0e-11)


def test_vmap_profile_propagates_layer_state_to_lnphi():
    setup = _setup()
    temperatures = jnp.asarray([300.0, 500.0, 700.0])
    pressures = jnp.asarray([1.0, 2.0, 4.0])

    def lnphi_func(temperature, pressure_bar, mole_fractions):
        del mole_fractions
        delta = 1.0e-3 * temperature + 0.1 * pressure_bar
        return jnp.asarray([delta, -delta])

    result = equilibrium_profile(
        setup,
        temperatures,
        pressures,
        _B,
        options=EquilibriumOptions(
            epsilon_crit=1.0e-12,
            max_iter=200,
            method="vmap_cold",
        ),
        lnphi_func=lnphi_func,
    )
    delta = 1.0e-3 * temperatures + 0.1 * pressures
    expected = jax.vmap(_expected_mole_fractions)(delta)

    np.testing.assert_allclose(result.x, expected, atol=1.0e-12)


def test_scan_profile_cache_distinguishes_lnphi_providers():
    setup = _setup()
    temperatures = jnp.asarray([500.0, 600.0, 700.0])
    pressures = jnp.asarray([1.0, 2.0, 3.0])
    options = EquilibriumOptions(
        epsilon_crit=1.0e-12,
        max_iter=200,
        method="scan_hot_from_bottom",
    )

    def positive_lnphi(temperature, pressure_bar, mole_fractions):
        del temperature, pressure_bar, mole_fractions
        return jnp.asarray([0.25, -0.25])

    def negative_lnphi(temperature, pressure_bar, mole_fractions):
        del temperature, pressure_bar, mole_fractions
        return jnp.asarray([-0.35, 0.35])

    positive = equilibrium_profile(
        setup,
        temperatures,
        pressures,
        _B,
        options=options,
        lnphi_func=positive_lnphi,
    )
    negative = equilibrium_profile(
        setup,
        temperatures,
        pressures,
        _B,
        options=options,
        lnphi_func=negative_lnphi,
    )

    np.testing.assert_allclose(
        positive.x,
        jnp.broadcast_to(_expected_mole_fractions(0.25), positive.x.shape),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        negative.x,
        jnp.broadcast_to(_expected_mole_fractions(-0.35), negative.x.shape),
        atol=1.0e-12,
    )


def test_lnphi_shape_must_match_gas_species():
    with pytest.raises(ValueError, match=r"expected shape \(2,\), got \(1,\)"):
        equilibrium(
            _setup(),
            900.0,
            1.0,
            _B,
            options=_OPTIONS,
            lnphi_func=lambda temperature, pressure, mole_fractions: jnp.zeros((1,)),
        )
