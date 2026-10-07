"""Regression tests for fixed-support zero-barrier implicit autodiff."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from exogibbs.equilibrium.condensate.fixed_support.autodiff import (
    fixed_support_source_vjp,
    minimize_gibbs_fixed_support,
    minimize_gibbs_fixed_support_with_diagnostics,
    zero_barrier_residual_vector,
)
from exogibbs.equilibrium.gas.types import ThermoState


jax.config.update("jax_enable_x64", True)


_LOG_K = jnp.log(0.25)


def _gas_hvector(temperature):
    return jnp.asarray([0.2 * temperature, -0.1 * temperature])


def _condensate_hvector(temperature):
    # At T=2 and log(P/Pref)=0 this gives
    # h_cond - h_gas[0] = log(0.25).
    return jnp.asarray([0.05 * temperature + _LOG_K + 0.3])


def _fixture():
    gas_formula_matrix = jnp.eye(2)
    condensate_formula_matrix = jnp.asarray([[1.0], [0.0]])
    target = jnp.asarray([2.0, 1.0])
    gas_log_amounts_init = jnp.log(jnp.asarray([0.4, 0.9]))
    condensate_amounts_init = jnp.asarray([1.6])
    total_gas_log_amount_init = jnp.log(
        jnp.sum(jnp.exp(gas_log_amounts_init))
    )
    return (
        gas_formula_matrix,
        condensate_formula_matrix,
        target,
        gas_log_amounts_init,
        condensate_amounts_init,
        total_gas_log_amount_init,
    )


def _solve(
    temperature=2.0,
    log_pressure=0.0,
    target=None,
    gas_log_amounts_init=None,
    condensate_amounts_init=None,
    return_diagnostics=False,
):
    ag, ac, default_target, q0, m0, qtot0 = _fixture()
    if target is None:
        target = default_target
    if gas_log_amounts_init is None:
        gas_log_amounts_init = q0
    if condensate_amounts_init is None:
        condensate_amounts_init = m0
    solver = (
        minimize_gibbs_fixed_support_with_diagnostics
        if return_diagnostics
        else minimize_gibbs_fixed_support
    )
    return solver(
        ThermoState(temperature, log_pressure, target),
        gas_log_amounts_init,
        condensate_amounts_init,
        qtot0,
        ag,
        ac,
        _gas_hvector,
        _condensate_hvector,
        residual_crit=1.0e-12,
        max_iter=50,
    )


def test_fixed_support_zero_barrier_solver_matches_analytic_coexistence():
    result, diagnostics = _solve(return_diagnostics=True)
    expected_gas = jnp.asarray([1.0 / 3.0, 1.0])
    expected_condensate = jnp.asarray([5.0 / 3.0])

    assert jnp.exp(result.gas_log_amounts) == pytest.approx(expected_gas)
    assert result.condensate_amounts == pytest.approx(expected_condensate)
    assert bool(diagnostics.converged)
    assert diagnostics.residual_norm < 1.0e-12
    assert int(diagnostics.iterations) > 0


def test_reduced_source_vjp_matches_dense_transpose_jacobian():
    ag, ac, target, _q0, _m0, _qtot0 = _fixture()
    result = _solve()
    q = result.gas_log_amounts
    m = result.condensate_amounts
    qtot = jnp.log(jnp.sum(jnp.exp(q)))
    gamma = _gas_hvector(2.0)
    hcond = _condensate_hvector(2.0)
    element_potential = jnp.asarray(
        [hcond[0], q[1] + gamma[1] - qtot]
    )
    gas_cotangent = jnp.asarray([0.7, -0.4])
    condensate_cotangent = jnp.asarray([1.3])

    reduced = fixed_support_source_vjp(
        gas_cotangent,
        condensate_cotangent,
        q,
        m,
        qtot,
        ag,
        ac,
    )

    def residual_from_variables(variables):
        q_value = variables[:2]
        m_value = variables[2:3]
        potential_value = variables[3:5]
        qtot_value = variables[-1]
        return zero_barrier_residual_vector(
            target,
            gamma,
            hcond,
            q_value,
            m_value,
            potential_value,
            qtot_value,
            ag,
            ac,
        )

    variables = jnp.concatenate(
        [q, m, element_potential, qtot.reshape((1,))]
    )
    jacobian = jax.jacrev(residual_from_variables)(variables)
    output_cotangent = jnp.concatenate(
        [gas_cotangent, condensate_cotangent, jnp.zeros((3,))]
    )
    adjoint = jnp.linalg.solve(jacobian.T, output_cotangent)
    gas_count = q.shape[0]
    condensate_count = m.shape[0]
    element_count = target.shape[0]
    expected_gas_source = -adjoint[:gas_count]
    expected_condensate_source = -adjoint[
        gas_count : gas_count + condensate_count
    ]
    expected_target = adjoint[
        gas_count
        + condensate_count : gas_count
        + condensate_count
        + element_count
    ]

    assert reduced.gas_source == pytest.approx(expected_gas_source)
    assert reduced.condensate_standard_source == pytest.approx(
        expected_condensate_source
    )
    assert reduced.target_inventory == pytest.approx(expected_target)


def test_reduced_source_vjp_handles_nonsquare_gas_and_multiple_support():
    ag = jnp.asarray(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    ac = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    gas_amounts = jnp.asarray([0.2, 0.3, 0.4, 0.1])
    q = jnp.log(gas_amounts)
    m = jnp.asarray([0.7, 0.5])
    qtot = jnp.log(jnp.sum(gas_amounts))
    element_potential = jnp.asarray([0.4, -0.2, 0.1])
    gamma = ag.T @ element_potential - q + qtot
    hcond = ac.T @ element_potential
    target = ag @ gas_amounts + ac @ m
    gas_cotangent = jnp.asarray([0.7, -0.4, 0.2, 0.9])
    condensate_cotangent = jnp.asarray([1.3, -0.6])

    reduced = fixed_support_source_vjp(
        gas_cotangent,
        condensate_cotangent,
        q,
        m,
        qtot,
        ag,
        ac,
    )

    def residual_from_variables(variables):
        gas_count = ag.shape[1]
        condensate_count = ac.shape[1]
        element_count = ag.shape[0]
        m_start = gas_count
        potential_start = m_start + condensate_count
        qtot_index = potential_start + element_count
        return zero_barrier_residual_vector(
            target,
            gamma,
            hcond,
            variables[:gas_count],
            variables[m_start:potential_start],
            variables[potential_start:qtot_index],
            variables[qtot_index],
            ag,
            ac,
        )

    variables = jnp.concatenate(
        [q, m, element_potential, qtot.reshape((1,))]
    )
    jacobian = jax.jacrev(residual_from_variables)(variables)
    output_cotangent = jnp.concatenate(
        [gas_cotangent, condensate_cotangent, jnp.zeros((4,))]
    )
    adjoint = jnp.linalg.solve(jacobian.T, output_cotangent)
    gas_count = ag.shape[1]
    condensate_count = ac.shape[1]
    element_count = ag.shape[0]

    assert reduced.gas_source == pytest.approx(-adjoint[:gas_count])
    assert reduced.condensate_standard_source == pytest.approx(
        -adjoint[gas_count : gas_count + condensate_count]
    )
    assert reduced.target_inventory == pytest.approx(
        adjoint[
            gas_count
            + condensate_count : gas_count
            + condensate_count
            + element_count
        ]
    )


def test_reverse_mode_matches_analytic_temperature_pressure_and_budget_derivatives():
    _ag, _ac, target, _q0, _m0, _qtot0 = _fixture()
    gas_pressure_jacobian = jax.jacrev(
        lambda log_pressure: _solve(
            log_pressure=log_pressure
        ).gas_log_amounts
    )(0.0)
    condensate_pressure_jacobian = jax.jacrev(
        lambda log_pressure: _solve(
            log_pressure=log_pressure
        ).condensate_amounts
    )(0.0)
    gas_temperature_jacobian = jax.jacrev(
        lambda temperature: _solve(
            temperature=temperature
        ).gas_log_amounts
    )(2.0)
    condensate_temperature_jacobian = jax.jacrev(
        lambda temperature: _solve(
            temperature=temperature
        ).condensate_amounts
    )(2.0)
    gas_budget_jacobian = jax.jacrev(
        lambda inventory: _solve(target=inventory).gas_log_amounts
    )(target)
    condensate_budget_jacobian = jax.jacrev(
        lambda inventory: _solve(target=inventory).condensate_amounts
    )(target)

    assert gas_pressure_jacobian == pytest.approx([-4.0 / 3.0, 0.0])
    assert condensate_pressure_jacobian == pytest.approx([4.0 / 9.0])
    assert gas_temperature_jacobian == pytest.approx([-0.2, 0.0])
    assert condensate_temperature_jacobian == pytest.approx([1.0 / 15.0])
    assert gas_budget_jacobian == pytest.approx(
        jnp.asarray([[0.0, 1.0], [0.0, 1.0]])
    )
    assert condensate_budget_jacobian == pytest.approx(
        jnp.asarray([[1.0, -1.0 / 3.0]])
    )


def test_reverse_mode_differentiates_the_complete_physical_result_pytree():
    jacobian = jax.jacrev(lambda temperature: _solve(temperature=temperature))(
        2.0
    )

    assert jacobian.gas_log_amounts == pytest.approx([-0.2, 0.0])
    assert jacobian.condensate_amounts == pytest.approx([1.0 / 15.0])


def test_implicit_autodiff_stops_initialization_gradients_and_is_jittable():
    ag, ac, target, q0, m0, qtot0 = _fixture()

    def loss(q_init, m_init, qtot_init, potential_init, gas_matrix, cond_matrix):
        result = minimize_gibbs_fixed_support(
            ThermoState(2.0, 0.0, target),
            q_init,
            m_init,
            qtot_init,
            gas_matrix,
            cond_matrix,
            _gas_hvector,
            _condensate_hvector,
            element_potential_init=potential_init,
            residual_crit=1.0e-12,
            max_iter=50,
        )
        return jnp.sum(result.gas_log_amounts) + jnp.sum(
            result.condensate_amounts
        )

    potential0 = jnp.zeros((ag.shape[0],), dtype=q0.dtype)
    gradients = jax.grad(loss, argnums=(0, 1, 2, 3, 4, 5))(
        q0, m0, qtot0, potential0, ag, ac
    )
    _value, initialization_tangent = jax.jvp(
        loss,
        (q0, m0, qtot0, potential0, ag, ac),
        tuple(
            jnp.ones_like(value)
            for value in (q0, m0, qtot0, potential0, ag, ac)
        ),
    )
    compiled = jax.jit(
        lambda temperature: _solve(
            temperature=temperature
        ).gas_log_amounts
    )(2.0)
    compiled_gradient = jax.jit(
        jax.grad(
            lambda temperature: jnp.sum(
                _solve(temperature=temperature).gas_log_amounts
            )
        )
    )(2.0)
    compiled_jvp = jax.jit(
        lambda temperature: jax.jvp(
            lambda value: jnp.sum(
                _solve(temperature=value).gas_log_amounts
            ),
            (temperature,),
            (jnp.ones_like(temperature),),
        )[1]
    )(2.0)
    batched_gradients = jax.vmap(
        jax.grad(
            lambda temperature: jnp.sum(
                _solve(temperature=temperature).gas_log_amounts
            )
        )
    )(jnp.asarray([1.9, 2.0, 2.1]))
    batched_jvps = jax.vmap(
        lambda temperature: jax.jvp(
            lambda value: jnp.sum(
                _solve(temperature=value).gas_log_amounts
            ),
            (temperature,),
            (jnp.ones_like(temperature),),
        )[1]
    )(jnp.asarray([1.9, 2.0, 2.1]))

    for gradient, value in zip(
        gradients, (q0, m0, qtot0, potential0, ag, ac)
    ):
        assert gradient == pytest.approx(jnp.zeros_like(value))
    assert initialization_tangent == pytest.approx(0.0)
    assert jnp.all(jnp.isfinite(compiled))
    assert compiled_gradient == pytest.approx(-0.2)
    assert compiled_jvp == pytest.approx(-0.2)
    assert jnp.all(jnp.isfinite(batched_gradients))
    assert jnp.all(jnp.isfinite(batched_jvps))


def test_public_solver_can_be_jitted_directly_with_dynamic_config_values():
    ag, ac, target, q0, m0, qtot0 = _fixture()
    compiled_solver = jax.jit(
        minimize_gibbs_fixed_support,
        static_argnums=(6, 7),
    )

    result = compiled_solver(
        ThermoState(2.0, 0.0, target),
        q0,
        m0,
        qtot0,
        ag,
        ac,
        _gas_hvector,
        _condensate_hvector,
        residual_crit=1.0e-12,
        max_iter=50,
    )

    assert jnp.exp(result.gas_log_amounts) == pytest.approx([1.0 / 3.0, 1.0])
    assert result.condensate_amounts == pytest.approx([5.0 / 3.0])


def test_fixed_support_jvp_matches_analytic_and_finite_difference():
    _ag, _ac, target, _q0, _m0, _qtot0 = _fixture()
    temperature_tangent = 0.7
    pressure_tangent = -0.3
    inventory_tangent = jnp.asarray([0.2, -0.4])

    def solve_flat(temperature, log_pressure, inventory):
        result = _solve(
            temperature=temperature,
            log_pressure=log_pressure,
            target=inventory,
        )
        return jnp.concatenate(
            [result.gas_log_amounts, result.condensate_amounts]
        )

    primals = (2.0, 0.0, target)
    tangents = (
        temperature_tangent,
        pressure_tangent,
        inventory_tangent,
    )
    _result, tangent = jax.jvp(solve_flat, primals, tangents)
    expected = jnp.concatenate(
        [
            temperature_tangent * jnp.asarray([-0.2, 0.0])
            + pressure_tangent * jnp.asarray([-4.0 / 3.0, 0.0])
            + jnp.asarray([[0.0, 1.0], [0.0, 1.0]])
            @ inventory_tangent,
            temperature_tangent * jnp.asarray([1.0 / 15.0])
            + pressure_tangent * jnp.asarray([4.0 / 9.0])
            + jnp.asarray([[1.0, -1.0 / 3.0]])
            @ inventory_tangent,
        ]
    )
    step = 1.0e-5
    plus = solve_flat(
        primals[0] + step * tangents[0],
        primals[1] + step * tangents[1],
        primals[2] + step * tangents[2],
    )
    minus = solve_flat(
        primals[0] - step * tangents[0],
        primals[1] - step * tangents[1],
        primals[2] - step * tangents[2],
    )
    finite_difference = (plus - minus) / (2.0 * step)

    assert tangent == pytest.approx(expected, rel=1.0e-10, abs=1.0e-11)
    assert tangent == pytest.approx(
        finite_difference, rel=1.0e-7, abs=1.0e-8
    )


def test_fixed_support_generated_vjp_satisfies_adjoint_identity():
    _ag, _ac, target, _q0, _m0, _qtot0 = _fixture()
    inputs = (2.0, 0.0, target)
    direction = (0.4, -0.2, jnp.asarray([0.3, -0.1]))
    output_cotangent = jnp.asarray([0.6, -0.2, 1.1])

    def solve_flat(temperature, log_pressure, inventory):
        result = _solve(
            temperature=temperature,
            log_pressure=log_pressure,
            target=inventory,
        )
        return jnp.concatenate(
            [result.gas_log_amounts, result.condensate_amounts]
        )

    _result, output_tangent = jax.jvp(
        solve_flat, inputs, direction
    )
    _result, pullback = jax.vjp(solve_flat, *inputs)
    input_cotangents = pullback(output_cotangent)
    input_pairing = sum(
        jnp.vdot(cotangent, tangent)
        for cotangent, tangent in zip(input_cotangents, direction)
    )

    assert jnp.vdot(output_cotangent, output_tangent) == pytest.approx(
        input_pairing, rel=1.0e-11, abs=1.0e-11
    )


def test_failed_zero_barrier_solve_returns_nonfinite_source_gradient():
    ag, ac, target, q0, m0, qtot0 = _fixture()

    def solve(temperature, *, diagnostics=False):
        solver = (
            minimize_gibbs_fixed_support_with_diagnostics
            if diagnostics
            else minimize_gibbs_fixed_support
        )
        return solver(
            ThermoState(temperature, 0.0, target),
            q0,
            m0,
            qtot0,
            ag,
            ac,
            _gas_hvector,
            _condensate_hvector,
            residual_crit=1.0e-12,
            max_iter=0,
        )

    def loss(temperature):
        result = solve(temperature)
        return jnp.sum(result.gas_log_amounts)

    _result, diagnostics = solve(2.0, diagnostics=True)
    _value, tangent = jax.jvp(loss, (2.0,), (1.0,))
    assert not bool(diagnostics.converged)
    assert diagnostics.residual_norm > 1.0e-12
    assert jnp.isnan(tangent)
    assert jnp.isnan(jax.grad(loss)(2.0))


def test_float32_uses_a_representable_convergence_tolerance():
    ag = jnp.eye(2, dtype=jnp.float32)
    ac = jnp.asarray([[1.0], [0.0]], dtype=jnp.float32)
    target = jnp.asarray([2.0, 1.0], dtype=jnp.float32)
    q0 = jnp.log(jnp.asarray([0.4, 0.9], dtype=jnp.float32))
    m0 = jnp.asarray([1.6], dtype=jnp.float32)
    qtot0 = jnp.log(jnp.sum(jnp.exp(q0)))

    def loss(temperature):
        result = minimize_gibbs_fixed_support(
            ThermoState(temperature, jnp.float32(0.0), target),
            q0,
            m0,
            qtot0,
            ag,
            ac,
            _gas_hvector,
            _condensate_hvector,
        )
        return (
            jnp.sum(result.gas_log_amounts)
            + jnp.sum(result.condensate_amounts)
        )

    _diagnostic_result, diagnostics = (
        minimize_gibbs_fixed_support_with_diagnostics(
            ThermoState(jnp.float32(2.0), jnp.float32(0.0), target),
            q0,
            m0,
            qtot0,
            ag,
            ac,
            _gas_hvector,
            _condensate_hvector,
        )
    )
    gradient = jax.grad(loss)(jnp.float32(2.0))

    assert bool(diagnostics.converged)
    assert jnp.isfinite(gradient)
    assert gradient == pytest.approx(-2.0 / 15.0, rel=2.0e-5)


def test_integer_stoichiometric_matrices_promote_to_floating_solver_dtype():
    ag, ac, target, q0, m0, qtot0 = _fixture()
    result, diagnostics = minimize_gibbs_fixed_support_with_diagnostics(
        ThermoState(2.0, 0.0, target),
        q0,
        m0,
        qtot0,
        ag.astype(jnp.int32),
        ac.astype(jnp.int32),
        _gas_hvector,
        _condensate_hvector,
        residual_crit=1.0e-12,
    )

    assert jnp.issubdtype(result.gas_log_amounts.dtype, jnp.floating)
    assert bool(diagnostics.converged)
    assert jnp.exp(result.gas_log_amounts) == pytest.approx([1.0 / 3.0, 1.0])
