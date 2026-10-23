"""Implicit derivatives against re-solved finite differences and exact ledgers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.api import multiphase as mp
from exogibbs.equilibrium.multiphase.types import R_SCALE as R


def material(t, p, n, parameters):
    total = n.sum()
    logx = jnp.log(jnp.where(n > 0, n/total, 1.))
    return (n @ (parameters["standard"] + p*jnp.array([1e-5, 3e-5]))
            + R*t*(n @ logx) + parameters["mixing"]*n[0]*n[1]/total)


PROBLEM = mp.Problem(("X",), (mp.Phase("liquid", ("A", "B"), [[1, 2]], material),))
THETA = jnp.array([1400., 5e7, 3., 20000., -5000., 800.])


def run(theta, *, scale=1., init=None, options=mp.Options()):
    b = theta[2:3]*scale
    return mp.solve_fixed_support(PROBLEM, temperature_k=theta[0], pressure_pa=theta[1],
                                  element_amounts_mol=b, support=mp.Support((0, 1)),
                                  init=jnp.array([1., 1.])*scale if init is None else init,
                                  parameters={"standard": theta[3:5], "mixing": theta[5]}, options=options)


def output(theta):
    result = run(theta)
    return jnp.concatenate((result.amounts_mol, result.gibbs_j[None]/1e5,
                            result.chemical_potentials_j_mol/1e5,
                            result.element_potentials_j_mol/1e5))


def test_refined_finite_differences_for_temperature_pressure_inventory_and_materials():
    assert run(THETA).derivative_valid
    evaluate, jacobian = jax.jit(output), jax.jit(jax.jacfwd(output))
    jac = jacobian(THETA)
    steps = np.array([10., 1e6, .2, 100., 100., 100.])
    errors = []
    for refinement in (1., .5, .25):
        finite = []
        for i, step in enumerate(steps*refinement):
            delta = jnp.zeros_like(THETA).at[i].set(step)
            finite.append((evaluate(THETA+delta)-evaluate(THETA-delta))/(2*step))
        scaled_error = np.max(np.abs(np.array(finite).T-np.asarray(jac))*steps)
        errors.append(scaled_error)
    assert errors[-1] < 2e-6
    assert errors[-1] < errors[0]/8


def test_jvp_vjp_transpose_conservation_and_absolute_scaling():
    direction = jnp.array([3., 2e5, .15, 20., -50., 10.])
    value, tangent = jax.jvp(output, (THETA,), (direction,))
    cotangent = jnp.linspace(-.7, 1.2, value.size)
    _, pullback = jax.vjp(output, THETA)
    np.testing.assert_allclose(cotangent @ tangent, direction @ pullback(cotangent)[0], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(PROBLEM.atom_matrix @ tangent[:2], direction[2:3], atol=1e-12)
    for scale in (1e-18, 1e24):
        _, dn = jax.jvp(lambda theta: run(theta, scale=scale).amounts_mol/scale,
                        (THETA,), (direction,))
        np.testing.assert_allclose(dn, tangent[:2], rtol=1e-10, atol=1e-12)


def test_initialization_is_not_a_physical_derivative_and_diagnostics_are_nondifferentiable():
    first = run(THETA, init=jnp.array([.2, 1.4]))
    second = run(THETA, init=jnp.array([2., .5]))
    np.testing.assert_allclose(first.amounts_mol, second.amounts_mol, atol=1e-9)
    derivative = jax.jacfwd(lambda initial: run(THETA, init=initial).amounts_mol)(jnp.ones(2))
    np.testing.assert_array_equal(derivative, np.zeros((2, 2)))
    _, tangent = jax.jvp(run, (THETA,), (jnp.ones_like(THETA),))
    assert tangent.minimum_curvature == 0.
    assert tangent.converged.dtype == jax.dtypes.float0


def test_strict_active_domain_derivative_and_zero_multiplier_transition():
    def energy(t, p, n, h):
        return R*t*(n @ jnp.log(n/n.sum()) + h*n[1])

    problem = mp.Problem(("X",), (mp.Phase("liquid", ("A", "B"), [[1, 1]], energy, [[.2, -.8]]),))

    def fixed(theta):
        return mp.solve_fixed_support(problem, temperature_k=1200., pressure_pa=1e5,
                                      element_amounts_mol=theta[:1], init=[.8, .2],
                                      parameters=theta[1], support=mp.Support((0, 1), (0,)))

    point = jnp.array([1., 0.])
    assert fixed(point).derivative_valid
    np.testing.assert_allclose(jax.jacfwd(lambda a: fixed(a).amounts_mol)(point), [[.8, 0.], [.2, 0.]], atol=1e-12)
    point = point.at[1].set(jnp.log(4.))
    assert not fixed(point).derivative_valid
    assert np.all(np.isnan(jax.jacfwd(lambda a: fixed(a).amounts_mol)(point)))
    assert np.all(np.isnan(jax.grad(lambda a: fixed(a).gibbs_j)(point)))


def test_phase_boundary_invalidates_both_modes_even_for_zero_jacobian_columns():
    def liquid(t, p, n, h):
        return R*t*(n @ jnp.log(n/n.sum()))

    problem = mp.Problem(("X", "Y"), (
        mp.Phase("liquid", ("X", "Y"), np.eye(2), liquid),
        mp.Phase("solid", ("X",), [[1], [0]], lambda t, p, n, h: R*t*h*n[0]),
    ))

    def fixed(h):
        return mp.solve_fixed_support(problem, temperature_k=1200., pressure_pa=1e5,
                                      element_amounts_mol=[.4, .6], init=[.4, .6, 0.],
                                      parameters=h, support=mp.Support((0, 1)))

    threshold = jnp.log(.4)
    assert fixed(threshold+.2).derivative_valid
    for h in (threshold, threshold-.2):
        assert not fixed(h).derivative_valid
        assert np.all(np.isnan(jax.jvp(lambda a: fixed(a).amounts_mol, (h,), (1.,))[1]))
        assert np.isnan(jax.grad(lambda a: fixed(a).amounts_mol[0])(h))


@pytest.mark.parametrize("failure", ["not_converged", "ill_conditioned", "material_domain"])
def test_invalid_derivatives_fail_closed(failure):
    if failure == "not_converged":
        fixed = lambda a: run(a, options=mp.Options(max_iterations=1))
        point = THETA
    else:
        def energy(t, p, n, a):
            x = n/n.sum()
            return R*t*n.sum()*((x[0]-.3)**2 + 1e-8*(x[1]-.3)**2)

        phase = mp.Phase("liquid", ("A", "B", "C"), [[1, 1, 1]], energy,
                          derivative_domain=(lambda t, p, n, a: t != 1200.) if failure == "material_domain" else None)
        problem = mp.Problem(("X",), (phase,))
        fixed = lambda a: mp.solve_fixed_support(
            problem, temperature_k=a[0], pressure_pa=1e5, element_amounts_mol=[1.],
            init=[.3, .3, .4], support=mp.Support((0, 1, 2)),
            options=mp.Options(curvature_tolerance=1e-12, derivative_rcond=1e-6 if failure == "ill_conditioned" else 1e-12),
        )
        point = jnp.array([1200.])
        assert fixed(point).minimum_valid
    assert not fixed(point).derivative_valid
    assert np.all(np.isnan(jax.jacfwd(lambda a: fixed(a).amounts_mol)(point)))
    assert np.all(np.isnan(jax.grad(lambda a: fixed(a).gibbs_j)(point)))


def test_redundant_element_rows_support_only_compatible_budget_variations():
    def energy(t, p, n, a):
        return R*t*(n @ jnp.log(n/n.sum()))

    problem = mp.Problem(("X", "Y", "Z"), (
        mp.Phase("liquid", ("A", "B"), [[1, 1], [2, 2], [0, 0]], energy),
    ))
    def fixed(scale):
        return mp.solve_fixed_support(problem, temperature_k=1200., pressure_pa=1e5,
                                      element_amounts_mol=jnp.array([1., 2., 0.])*scale,
                                      init=[.4, .6], support=mp.Support((0, 1)))
    assert fixed(1.).derivative_valid
    dn = jax.jacfwd(lambda a: fixed(a).amounts_mol)(1.)
    np.testing.assert_allclose(problem.atom_matrix @ dn, [1., 2., 0.], atol=1e-12)


def test_absent_solution_requires_a_strict_jax_insertion_bound():
    from dataclasses import replace
    from jax.scipy.special import logsumexp

    def energy(t, p, n, a):
        return R*t*(n @ jnp.log(n/n.sum()))

    def metal(t, p, n, a):
        return energy(t, p, n, a) + R*t*(n @ jnp.array([-1., 1.]))

    def bound(t, p, dual, a, allowed):
        return -R*t*logsumexp((dual/(R*t) - jnp.array([-1., 1.]))[jnp.asarray(allowed)])

    liquid = mp.Phase("liquid", ("X", "Y"), np.eye(2), energy)
    alloy = mp.Phase("metal", ("X", "Y"), np.eye(2), metal, insertion_lower_bound_j_mol=bound)

    def fixed(budget, phase=alloy):
        return mp.solve_fixed_support(mp.Problem(("X", "Y"), (liquid, phase)),
                                      temperature_k=1200., pressure_pa=1e5, element_amounts_mol=budget,
                                      init=[.1, .9, 0., 0.], support=mp.Support((0, 1)))

    point = jnp.array([.1, .9])
    assert jax.jit(fixed)(point).derivative_valid
    np.testing.assert_allclose(jax.jacfwd(lambda b: fixed(b).amounts_mol)(point),
                               [[1., 0.], [0., 1.], [0., 0.], [0., 0.]], atol=1e-12)
    assert not fixed(point, replace(alloy, insertion_lower_bound_j_mol=None)).derivative_valid
    x = 1/(1+np.e)
    boundary = jnp.array([x, 1-x])
    assert not fixed(boundary).derivative_valid
    assert np.all(np.isnan(jax.grad(lambda b: fixed(b).gibbs_j)(boundary)))
