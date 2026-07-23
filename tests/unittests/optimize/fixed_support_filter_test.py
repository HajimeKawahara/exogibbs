import jax
import jax.numpy as jnp

from exogibbs.optimize.fixed_support_filter import (
    fixed_support_filter_acceptance,
    prepare_fixed_support_restoration_filter,
    update_fixed_support_filter,
)

jax.config.update("jax_enable_x64", True)


def test_filter_accepts_theta_or_barrier_progress_and_checks_history():
    accepted, f_type, armijo, history = fixed_support_filter_acceptance(
        trial_phi=jnp.asarray([10.1, 9.9, 9.8]),
        trial_theta=jnp.asarray([0.5, 1.1, 1.1]),
        trial_alpha=jnp.ones((3,)),
        trial_linearized_change=jnp.asarray([0.1, -0.1, -0.2]),
        finite=jnp.ones((3,), dtype=bool),
        current_phi=10.0,
        current_theta=1.0,
        initial_theta=1.0,
        filter_phi=jnp.asarray([9.85, 0.0]),
        filter_theta=jnp.asarray([0.6, 0.0]),
        filter_valid=jnp.asarray([True, False]),
    )

    assert accepted.tolist() == [True, False, True]
    assert f_type.tolist() == [False, False, False]
    assert armijo.tolist() == [False, True, True]
    assert history.tolist() == [True, False, True]


def test_filter_uses_armijo_for_near_feasible_f_type_step():
    accepted, f_type, armijo, _history = fixed_support_filter_acceptance(
        trial_phi=jnp.asarray([9.95, 10.01]),
        trial_theta=jnp.asarray([1.0e-6, 1.0e-6]),
        trial_alpha=jnp.ones((2,)),
        trial_linearized_change=jnp.asarray([-0.1, -0.1]),
        finite=jnp.ones((2,), dtype=bool),
        current_phi=10.0,
        current_theta=1.0e-6,
        initial_theta=1.0,
        filter_phi=jnp.zeros((2,)),
        filter_theta=jnp.zeros((2,)),
        filter_valid=jnp.zeros((2,), dtype=bool),
    )

    assert f_type.tolist() == [True, True]
    assert armijo.tolist() == [True, False]
    assert accepted.tolist() == [True, False]


def test_filter_update_removes_dominated_entry():
    phi, theta, valid = update_fixed_support_filter(
        filter_phi=jnp.asarray([12.0, 9.0, 0.0]),
        filter_theta=jnp.asarray([2.0, 0.5, 0.0]),
        filter_valid=jnp.asarray([True, True, False]),
        current_phi=10.0,
        current_theta=1.0,
        add_entry=True,
    )

    assert valid.tolist() == [True, True, False]
    assert float(phi[0]) < 10.0
    assert float(theta[0]) < 1.0
    assert float(phi[1]) == 9.0


def test_prepare_restoration_filter_adds_margin_adjusted_current_iterate():
    phi, theta, valid = prepare_fixed_support_restoration_filter(
        filter_phi=jnp.zeros((2,)),
        filter_theta=jnp.zeros((2,)),
        filter_valid=jnp.zeros((2,), dtype=bool),
        current_phi=10.0,
        current_theta=2.0,
        phase_entered=True,
    )

    assert valid.tolist() == [True, False]
    assert float(phi[0]) == 10.0 - 1.0e-8 * 2.0
    assert float(theta[0]) == (1.0 - 1.0e-5) * 2.0


def test_prepare_restoration_filter_is_noop_outside_phase_entry():
    initial_phi = jnp.asarray([8.0, 0.0])
    initial_theta = jnp.asarray([0.5, 0.0])
    initial_valid = jnp.asarray([True, False])

    phi, theta, valid = prepare_fixed_support_restoration_filter(
        filter_phi=initial_phi,
        filter_theta=initial_theta,
        filter_valid=initial_valid,
        current_phi=10.0,
        current_theta=2.0,
        phase_entered=False,
    )

    assert jnp.array_equal(phi, initial_phi)
    assert jnp.array_equal(theta, initial_theta)
    assert jnp.array_equal(valid, initial_valid)
