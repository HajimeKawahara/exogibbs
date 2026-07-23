import jax
import jax.numpy as jnp
import pytest

from exogibbs.optimize.fixed_support_kkt import (
    fixed_support_barrier_objective,
    fixed_support_barrier_objective_linearized_change,
    fixed_support_filter_theta,
    fixed_support_full_newton_linearized_residual,
)

jax.config.update("jax_enable_x64", True)


def test_filter_theta_contains_only_scaled_primal_constraints():
    theta = fixed_support_filter_theta(
        formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
        formula_matrix_cond_active=[[1.0], [1.0]],
        element_inventory_target=[0.4, 0.5],
        q=jnp.log(jnp.asarray([0.3, 0.4])),
        r=jnp.log(jnp.asarray([0.1])),
        qtot=jnp.log(0.7),
        relative_floor=1.0e-8,
    )

    assert float(theta) == pytest.approx(0.0, abs=1.0e-14)


def test_filter_theta_supports_ipopt_style_l1_norm():
    kwargs = {
        "formula_matrix": [[1.0, 0.0], [0.0, 1.0]],
        "formula_matrix_cond_active": [[1.0], [1.0]],
        "element_inventory_target": [0.4, 0.5],
        "q": jnp.log(jnp.asarray([0.28, 0.37])),
        "r": jnp.log(jnp.asarray([0.1])),
        "qtot": jnp.log(0.65),
        "relative_floor": 1.0e-8,
    }

    max_theta = fixed_support_filter_theta(**kwargs)
    l1_theta = fixed_support_filter_theta(**kwargs, use_l1_norm=True)

    assert float(max_theta) == pytest.approx(0.06)
    assert float(l1_theta) == pytest.approx(0.11)


def test_barrier_objective_linearization_matches_finite_difference():
    kwargs = {
        "q": jnp.log(jnp.asarray([0.31, 0.42])),
        "r": jnp.log(jnp.asarray([0.025])),
        "qtot": jnp.log(0.74),
        "gas_stationarity_source": jnp.asarray([0.12, -0.07]),
        "condensate_standard_source": jnp.asarray([0.31]),
        "qtot_reference": jnp.log(0.74),
        "epsilon": jnp.log(1.0e-7),
    }
    delta_q = jnp.asarray([0.4, -0.3])
    delta_r = jnp.asarray([0.2])
    delta_qtot = jnp.asarray(-0.15)
    predicted = fixed_support_barrier_objective_linearized_change(
        **kwargs,
        delta_q=delta_q,
        delta_r=delta_r,
        delta_qtot=delta_qtot,
    )
    step = 1.0e-6
    plus = fixed_support_barrier_objective(
        **{
            **kwargs,
            "q": kwargs["q"] + step * delta_q,
            "r": kwargs["r"] + step * delta_r,
            "qtot": kwargs["qtot"] + step * delta_qtot,
        }
    )
    minus = fixed_support_barrier_objective(
        **{
            **kwargs,
            "q": kwargs["q"] - step * delta_q,
            "r": kwargs["r"] - step * delta_r,
            "qtot": kwargs["qtot"] - step * delta_qtot,
        }
    )

    assert float(predicted) == pytest.approx(
        float((plus - minus) / (2.0 * step)), rel=1.0e-9, abs=1.0e-10
    )


def test_full_newton_linearized_residual_matches_residual_jvp():
    ag = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    ac = jnp.asarray([[1.0], [1.0]])
    target = jnp.asarray([0.34, 0.47])
    gas_source = jnp.asarray([0.12, -0.07])
    cond_source = jnp.asarray([0.31])
    epsilon = jnp.log(1.0e-7)
    q = jnp.log(jnp.asarray([0.31, 0.42]))
    r = jnp.log(jnp.asarray([0.025]))
    lam = jnp.asarray([0.08, -0.11])
    rho = jnp.log(jnp.asarray([0.004]))
    qtot = jnp.log(0.74)
    delta = (
        jnp.asarray([0.4, -0.3]),
        jnp.asarray([0.2]),
        jnp.asarray([-0.1, 0.15]),
        jnp.asarray([0.05]),
        jnp.asarray(-0.15),
    )

    def residual(state):
        qi, ri, lami, rhoi, qtoti = state
        ni = jnp.exp(qi)
        mi = jnp.exp(ri)
        gas = qi + gas_source + qtot - qtoti - ag.T @ lami
        cond = cond_source - ac.T @ lami - jnp.exp(rhoi)
        budget = ag @ ni + ac @ mi - target
        comp = ri + rhoi - epsilon
        total = jnp.asarray([jnp.sum(ni) - jnp.exp(qtoti)])
        return jnp.concatenate([gas, cond, budget, comp, total])

    state = (q, r, lam, rho, qtot)
    current, jacobian_delta = jax.jvp(residual, (state,), (delta,))
    actual = fixed_support_full_newton_linearized_residual(
        formula_matrix=ag,
        formula_matrix_cond_active=ac,
        q=q,
        r=r,
        rho=rho,
        qtot=qtot,
        gas_residual=current[:2],
        condensate_stationarity_residual=current[2:3],
        budget_residual=current[3:5],
        complementarity_residual=current[5:6],
        total_density_residual=current[6:7],
        delta_q=delta[0],
        delta_r=delta[1],
        delta_lambda=delta[2],
        delta_rho=delta[3],
        delta_qtot=delta[4],
    )

    assert actual == pytest.approx(current + jacobian_delta, abs=1.0e-14)
