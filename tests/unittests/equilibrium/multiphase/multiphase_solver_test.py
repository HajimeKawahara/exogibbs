"""Independent constrained-G checks for the public fixed-support primal."""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from exogibbs.api import multiphase as mp
from exogibbs.equilibrium.multiphase.types import R_SCALE as R


def ideal(temperature, pressure, amounts, parameters):
    fraction = amounts / amounts.sum()
    log_fraction = jnp.log(jnp.where(amounts > 0, fraction, 1.))
    return R * temperature * (amounts @ (log_fraction + parameters))


def binary(domain=()):
    return mp.Problem(("X",), (
        mp.Phase("solution", ("A", "B"), [[1, 1]], ideal, domain),
    ))


def run(problem, *, budget=(1.,), init=(.3, .7), parameters=None,
        support=None, options=mp.Options()):
    if parameters is None:
        parameters = jnp.zeros(2)
    if support is None and np.shape(budget) == (len(problem.elements),):
        reachable = ~np.any(problem.atom_matrix[np.asarray(budget) == 0] > 0, axis=0)
        if np.any(reachable):
            support = mp.Support(tuple(np.flatnonzero(reachable)))
    return mp.solve(problem, temperature_k=1200., pressure_pa=1e5,
                    element_amounts_mol=budget, init=init, parameters=parameters,
                    support=support, options=options)


def test_amount_scale_gauge_and_independent_scalar_reference():
    problem = binary()
    reference = minimize(lambda n: np.sum(n * (np.log(n) + [0., 1.])),
                         [.3, .7], method="SLSQP", bounds=[(1e-12, 1)] * 2,
                         constraints={"type": "eq", "fun": lambda n: n.sum() - 1},
                         options={"ftol": 1e-14})
    assert reference.success
    base = run(problem, parameters=jnp.array([0., 1.]))
    assert base.converged and base.local_kkt_valid and base.minimum_valid
    np.testing.assert_allclose(base.amounts_mol, reference.x, atol=1e-8)
    assert abs(float(base.gibbs_j) / (R * 1200) - reference.fun) < 1e-9
    for scale in (1e-18, 1e24):
        scaled = run(problem, budget=[scale], init=scale * np.array([.3, .7]),
                     parameters=jnp.array([0., 1.]))
        np.testing.assert_allclose(scaled.amounts_mol / scale, base.amounts_mol, atol=1e-12)
        np.testing.assert_allclose(scaled.gibbs_j / scale, base.gibbs_j, rtol=1e-12)
        np.testing.assert_allclose(scaled.element_potentials_j_mol,
                                   base.element_potentials_j_mol, atol=1e-8)
    shifted = run(problem, parameters=jnp.array([7., 8.]))
    np.testing.assert_allclose(shifted.amounts_mol, base.amounts_mol, atol=1e-12)
    np.testing.assert_allclose(shifted.element_potentials_j_mol,
                               base.element_potentials_j_mol + 7 * R * 1200, atol=1e-8)


def test_active_linear_domain_and_multiplier_sign():
    result = run(binary([[.2, -.8]]), init=(.8, .2),
                 support=mp.Support((0, 1), (0,)))
    assert result.converged and result.local_kkt_valid and result.minimum_valid
    np.testing.assert_allclose(result.amounts_mol, [.8, .2], atol=1e-15)
    np.testing.assert_allclose(result.constraint_multipliers_j_mol,
                               [R * 1200 * np.log(4.)], atol=1e-8)
    # A declared upper face with the wrong multiplier cannot be accepted.
    wrong_face = run(binary([[.8, -.2]]), init=(.2, .8),
                     support=mp.Support((0, 1), (0,)))
    assert not wrong_face.local_kkt_valid
    assert not wrong_face.support_valid


def test_inactive_domain_is_preserved_and_boundary_requires_new_support():
    result = run(binary([[.2, -.8]]), init=(.9, .1),
                 options=mp.Options(max_iterations=20))
    assert result.domain_slack[0] >= -1e-12
    assert not result.converged
    assert not result.support_valid


def test_zero_budget_removal_and_whole_absent_phase_are_exact():
    def absent(*args):
        raise AssertionError("An absent phase must not be evaluated.")

    problem = mp.Problem(("X", "Y"), (
        mp.Phase("solution", ("A", "B"), [[1, 1], [0, 0]], ideal),
        mp.Phase("excluded", ("Y",), [[0], [1]], absent),
    ))
    result = run(problem, budget=(1., 0.), init=(.3, .7, 0.))
    assert result.local_kkt_valid and result.amounts_mol[2] == 0
    assert result.phase_amounts_mol[1] == 0
    assert result.conservation_residual_mol[1] == 0
    assert jnp.isnan(result.chemical_potentials_j_mol[2])
    assert jnp.isnan(result.element_potentials_j_mol[1])
    assert jnp.all(jnp.isnan(result.insertion_lower))
    assert jnp.all(jnp.isnan(result.incipient_compositions[1]))
    assert result.stability_status == "unchecked" and not result.derivative_valid


def test_zero_component_in_present_phase_and_redundant_elements():
    problem = mp.Problem(("X", "Y", "Z"), (
        mp.Phase("solution", ("A", "B", "C"),
                 [[1, 1, 0], [0, 0, 1], [2, 2, 0]], ideal),
    ))
    result = run(problem, budget=(1., 0., 2.), init=(.3, .7, 0.),
                 parameters=jnp.zeros(3))
    assert result.converged and result.local_kkt_valid and not result.dual_unique
    np.testing.assert_allclose(result.amounts_mol, [.5, .5, 0.], atol=1e-12)


def test_nonconvex_stationary_maximum_is_not_accepted():
    def nonconvex(t, p, n, params):
        x = n[0] / n.sum() - .5
        return R * t * n.sum() * (x**4 - .1 * x**2)

    problem = mp.Problem(("X",), (mp.Phase("mix", ("A", "B"), [[1, 1]], nonconvex),))
    result = run(problem, init=(.5, .5))
    assert result.minimum_valid and result.converged
    assert result.minimum_curvature > 0
    assert result.gibbs_j < -1.
    assert abs(float(result.amounts_mol[0]) - .5) == pytest.approx(np.sqrt(.05), abs=1e-8)


def test_jit_dynamic_inputs_parameters_and_explicit_derivative_failure():
    problem = binary()
    support = mp.Support((0, 1))

    @jax.jit
    def solve(budget, parameters):
        return mp.solve_fixed_support(
            problem, temperature_k=1200., pressure_pa=1e5,
            element_amounts_mol=budget, init=jnp.array([.3, .7]) * budget[0],
            parameters=parameters, support=support,
        )

    first = solve(jnp.array([1.]), jnp.zeros(2))
    second = solve(jnp.array([3.]), jnp.array([0., 1.]))
    np.testing.assert_allclose(first.amounts_mol, [.5, .5], atol=1e-12)
    np.testing.assert_allclose(second.amounts_mol, 3 * jax.nn.softmax(jnp.array([0., -1.])), atol=1e-12)
    with pytest.raises(NotImplementedError, match="R4"):
        jax.jacfwd(lambda b: solve(b, jnp.zeros(2)).amounts_mol)(jnp.array([1.]))
    with pytest.raises(NotImplementedError, match="R4"):
        jax.grad(lambda p: solve(jnp.array([1.]), p).gibbs_j)(jnp.zeros(2))


@pytest.mark.parametrize("budget", [(-1.,), (0.,), (np.nan,), (1., 2.)])
def test_rejects_invalid_host_budgets(budget):
    with pytest.raises(ValueError, match="element_amounts"):
        run(binary(), budget=budget)


def test_invalid_fixed_support_input_and_inconsistent_redundant_ledger():
    result = mp.solve_fixed_support(binary(), temperature_k=-1., pressure_pa=1e5,
                                    element_amounts_mol=[1.], init=[.3, .7],
                                    parameters=jnp.zeros(2), support=mp.Support((0, 1)))
    assert not result.local_kkt_valid and not result.converged
    problem = mp.Problem(("X", "Y"), (
        mp.Phase("mix", ("A", "B"), [[1, 1], [2, 2]], ideal),
    ))
    assert not run(problem, budget=(1., 3.)).local_kkt_valid
    with pytest.raises(ValueError, match="independent"):
        run(binary([[1, 1]]), support=mp.Support((0, 1), (0,)))


def test_contracts_copy_static_arrays_and_reject_inconsistent_metadata():
    atoms = np.ones((1, 2))
    phase = mp.Phase("mix", ("A", "B"), atoms, ideal)
    atoms[0, 0] = 7.
    assert phase.atom_matrix[0][0] == 1.
    with pytest.raises(ValueError, match="unique"):
        mp.Phase("mix", ("A", "A"), [[1, 1]], ideal)
    with pytest.raises(ValueError, match="nonnegative"):
        mp.Phase("mix", ("A", "B"), [[1, -1]], ideal)
    with pytest.raises(ValueError, match="rows"):
        mp.Problem(("X", "Y"), (phase,))
    with pytest.raises(ValueError, match="finite matrix"):
        mp.Phase("mix", ("A", "B"), [[1, 1]], ideal, [[1]])
    with pytest.raises(ValueError, match="zero element"):
        problem = mp.Problem(("X", "Y"), (
            mp.Phase("mix", ("A", "B"), [[1, 0], [0, 1]], ideal),
        ))
        run(problem, budget=(1., 0.), support=mp.Support((0, 1)))


def test_flat_stationary_point_is_not_an_isolated_minimum():
    def quartic_maximum(t, p, n, parameters):
        return -R * t * n.sum() * (n[0] / n.sum() - .5)**4

    problem = mp.Problem(("X",), (
        mp.Phase("flat", ("A", "B"), [[1, 1]], quartic_maximum),
    ))
    result = run(problem, init=(.5, .5))
    assert result.local_kkt_valid
    assert not result.minimum_valid


def test_public_import_and_synthetic_solve_without_providers_or_examples():
    code = '''
import sys
class Block:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'exoeos', 'exoinventory', 'examples'}:
            raise AssertionError(name)
sys.meta_path.insert(0, Block())
from exogibbs.api import multiphase as m
p = m.Problem(('X',), (m.Phase('pure', ('X',), [[1]], lambda t,p,n,a: n[0]*t),))
r = m.solve(p, temperature_k=1000., pressure_pa=1e5, element_amounts_mol=[2.], init=[2.])
assert r.local_kkt_valid and r.amounts_mol[0] == 2.
'''
    subprocess.run([sys.executable, "-c", code], env=os.environ.copy(), check=True)
