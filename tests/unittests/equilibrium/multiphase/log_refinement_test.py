"""An analytic many-element equilibrium spanning extreme trace abundances."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.api import multiphase as mp

jax.config.update('jax_enable_x64', True)
R = 8.31446261815324


def analytic_problem(decades=80):
    rng = np.random.default_rng(17)
    atoms = np.c_[np.eye(13), rng.integers(0, 4, size=(13, 40))]
    expected = np.r_[np.ones(13), np.geomspace(1e-3, 10.**-decades, 40)]
    dual = np.linspace(-100., 30., 13)
    standards = atoms.T@dual-np.log(expected/expected.sum())

    def ideal(t, p, n, parameters):
        return R*t*jnp.sum(n*(parameters+jnp.log(n/n.sum())))

    phase = mp.Phase('gas', tuple(f's{i}' for i in range(53)), atoms, ideal)
    problem = mp.Problem(tuple(f'e{i}' for i in range(13)), (phase,))
    args = dict(temperature_k=1000., pressure_pa=1e5,
                element_amounts_mol=atoms@expected, support=mp.Support(tuple(range(53))),
                init=expected*np.exp(.05*np.sin(np.arange(53))), parameters=jnp.asarray(standards))
    return problem, args, expected


def test_log_refinement_recovers_80_decades_without_removing_species():
    problem, args, expected = analytic_problem()
    result = mp.solve_fixed_support(problem, **args,
        options=mp.Options(max_iterations=2, max_refinement_iterations=40))
    assert result.converged and result.local_kkt_valid and result.minimum_valid
    assert result.support.components == tuple(range(53))
    assert float(result.stationarity_error) < 1e-10
    assert float(result.conservation_error) < 1e-9
    np.testing.assert_allclose(result.amounts_mol, expected, rtol=1e-9, atol=0.)
    assert not result.derivative_valid


def test_refinement_budget_is_optional_and_nonnegative():
    assert mp.Options().max_refinement_iterations == 0
    for value in (-1, 1.5, True):
        with pytest.raises(ValueError, match='max_refinement_iterations'):
            mp.Options(max_refinement_iterations=value)


def test_refinement_does_not_override_an_active_physical_face():
    def ideal(t, p, n, parameters):
        return R*t*jnp.sum(n*jnp.log(n/n.sum()))
    phase = mp.Phase('bounded', ('A', 'B'), [[1., 1.]], ideal,
                     domain_matrix=[[.3, -.7]])
    problem = mp.Problem(('X',), (phase,))
    result = mp.solve_fixed_support(problem, temperature_k=1000., pressure_pa=1e5,
        element_amounts_mol=jnp.array([2.]), init=jnp.array([1.4, .6]),
        support=mp.Support((0, 1), (0,)),
        options=mp.Options(max_refinement_iterations=40))
    assert result.converged and result.local_kkt_valid
    np.testing.assert_allclose(result.amounts_mol, [1.4, .6], rtol=1e-12)
    assert result.support.active_constraints == (0,)


def binary_result(budget, refinement_iterations):
    def ideal(t, p, n, parameters):
        return R*t*jnp.sum(n*(parameters+jnp.log(n/n.sum())))
    phase = mp.Phase('gas', ('A', 'B'), [[1., 1.]], ideal)
    return mp.solve_fixed_support(mp.Problem(('X',), (phase,)),
        temperature_k=1000., pressure_pa=1e5, element_amounts_mol=jnp.array([budget]),
        init=budget*jnp.array([.999, .001]), support=mp.Support((0, 1)),
        parameters=jnp.array([0., -1.]),
        options=mp.Options(max_iterations=1, max_refinement_iterations=refinement_iterations))


def test_refinement_exhaustion_does_not_promote_an_invalid_root():
    ordinary = binary_result(2., 0)
    limited = binary_result(2., 1)
    assert not limited.converged and not limited.local_kkt_valid
    np.testing.assert_allclose(limited.amounts_mol, ordinary.amounts_mol, rtol=1e-13)


def test_refined_root_keeps_the_implicit_derivative_contract():
    result = jax.jit(lambda b: binary_result(b, 40))(2.)
    assert result.converged and result.minimum_valid and result.derivative_valid
    fraction = 1./(1.+np.e)
    np.testing.assert_allclose(result.amounts_mol[0], 2.*fraction, rtol=1e-10)
    derivative = jax.grad(lambda b: binary_result(b, 40).amounts_mol[0])(2.)
    np.testing.assert_allclose(derivative, fraction, rtol=1e-10)
