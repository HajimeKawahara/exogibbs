import jax
import jax.numpy as jnp
import pytest

from exogibbs.optimize.fixed_support_charge import (
    retract_fixed_support_charge_neutrality,
)

jax.config.update("jax_enable_x64", True)


def test_charge_retraction_enforces_neutrality_and_preserves_stationarity_coordinate():
    q = jnp.asarray([-80.0, -83.0, -2.0, -91.0])
    lam = jnp.asarray([0.4, 7.0])
    z = jnp.asarray([1.0, -1.0, 0.0, 2.0])
    invariant = q - z * lam[-1]

    retracted_q, retracted_lam, charge, susceptibility = (
        retract_fixed_support_charge_neutrality(
            log_gas_amounts=q,
            element_potential=lam,
            charge_coefficients=z,
        )
    )

    assert charge == pytest.approx(0.0, abs=5.0e-14)
    assert susceptibility > 0.0
    assert retracted_q - z * retracted_lam[-1] == pytest.approx(
        invariant, abs=2.0e-14
    )
    assert retracted_q[2] == pytest.approx(q[2])
    assert retracted_lam[0] == pytest.approx(lam[0])


def test_charge_retraction_is_invariant_to_pure_electron_null_shift():
    q = jnp.asarray([-50.0, -54.0, -3.0])
    lam = jnp.asarray([0.2, -4.0])
    z = jnp.asarray([1.0, -1.0, 0.0])
    shifted_q = q + 1.0e6 * z
    shifted_lam = lam.at[-1].add(1.0e6)

    base = retract_fixed_support_charge_neutrality(
        log_gas_amounts=q,
        element_potential=lam,
        charge_coefficients=z,
    )
    shifted = retract_fixed_support_charge_neutrality(
        log_gas_amounts=shifted_q,
        element_potential=shifted_lam,
        charge_coefficients=z,
    )

    assert shifted[0] == pytest.approx(base[0], abs=2.0e-9)
    assert shifted[1] == pytest.approx(base[1], abs=2.0e-9)


def test_charge_retraction_is_exact_noop_without_opposite_charge_carriers():
    q = jnp.asarray([-120.0, -2.0, -8.0])
    lam = jnp.asarray([0.4, 30.0])
    electron_stoichiometry = jnp.asarray([1.0, 0.0, 0.0])

    retracted = retract_fixed_support_charge_neutrality(
        log_gas_amounts=q,
        element_potential=lam,
        charge_coefficients=electron_stoichiometry,
    )

    assert jnp.array_equal(retracted[0], q)
    assert jnp.array_equal(retracted[1], lam)
