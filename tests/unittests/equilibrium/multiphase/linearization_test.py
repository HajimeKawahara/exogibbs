"""Local log-coordinate responses, trace species and transposed sensitivities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.api import multiphase as mp
from exogibbs.equilibrium.multiphase.types import R_SCALE as R


def ideal(t, p, n, standards):
    return n @ (standards+p*jnp.array([1e-5, 2e-5])) + R*t*jnp.sum(n*jnp.log(n/n.sum()))


PROBLEM = mp.Problem(('X',), (mp.Phase('liquid', ('A', 'B'), [[1, 1]], ideal),))
SUPPORT = mp.Support((0, 1))


def amounts(t, p, b, standards):
    return b[0]*jax.nn.softmax(-(standards+p*jnp.array([1e-5, 2e-5]))/(R*t))


def linearization(t=1400., p=1e5, b=None, standards=None, n=None):
    b = jnp.array([2.]) if b is None else b
    standards = jnp.array([0., 60*R*t]) if standards is None else standards
    n = amounts(t, p, b, standards) if n is None else n
    return mp.linearize_fixed_support(PROBLEM, temperature_k=t, pressure_pa=p,
        element_amounts_mol=b, amounts_mol=n, support=SUPPORT, parameters=standards)


def test_trace_species_direction_matches_analytic_and_conservation():
    t, p, b, standards = 1400., 1e5, jnp.array([2.]), jnp.array([0., 60*R*1400.])
    lin = linearization(t, p, b, standards)
    assert lin.derivative_valid, lin.diagnostics
    assert lin.diagnostics['minimum_positive_normalized_amount'] < 1e-25
    direction = (jnp.array(2.), jnp.array(100.), jnp.array([.3]), jnp.array([1., 7.]))
    got = jax.jit(lin.jvp)(*direction)
    assert jax.jit(lin.linear_residual)(*direction) < 1e-12
    expected = jax.jvp(amounts, (jnp.array(t), jnp.array(p), b, standards), direction)[1]
    np.testing.assert_allclose(got.amounts_mol, expected, rtol=1e-12, atol=0.)
    np.testing.assert_allclose(got.log_amounts, expected/amounts(t,p,b,standards), rtol=1e-12)
    np.testing.assert_allclose(PROBLEM.atom_matrix @ got.amounts_mol, direction[2], atol=1e-14)
    np.testing.assert_allclose(got.conservation_residual_mol, 0., atol=1e-14)


def test_vjp_is_transpose_of_all_physical_outputs():
    lin = linearization()
    direction = (jnp.array(1.), jnp.array(20.), jnp.array([.1]), jnp.array([2., -3.]))
    push = lin.jvp(*direction)
    cotangent = jax.tree_util.tree_map(lambda x: jnp.full_like(x, .3), push)
    reverse = jax.jit(lin.vjp)(cotangent)
    dot = lambda a,b: sum(jnp.vdot(x,y) for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)))
    np.testing.assert_allclose(dot(cotangent,push),dot(reverse,direction),rtol=2e-12)


def test_amount_normalization_does_not_change_intensive_response():
    baseline = linearization().jvp(1., 10., jnp.array([.1]), jnp.array([2.,3.]))
    for scale in (1e-18, 1e24):
        lin = linearization(b=jnp.array([2.*scale]))
        assert lin.derivative_valid
        got = lin.jvp(1.,10.,jnp.array([.1*scale]),jnp.array([2.,3.]))
        np.testing.assert_allclose(got.amounts_mol/scale,baseline.amounts_mol,rtol=1e-12)
        np.testing.assert_allclose(got.log_amounts,baseline.log_amounts,rtol=1e-12)
        np.testing.assert_allclose(got.element_potentials_j_mol,baseline.element_potentials_j_mol,rtol=1e-12)


def test_refined_central_differences_of_independent_equilibria():
    t,p,b,standards = 1400.,1e5,jnp.array([2.]),jnp.array([0.,60*R*1400.])
    lin=linearization(t,p,b,standards)
    got=lin.jvp(1.,0.,jnp.zeros(1),jnp.zeros(2)).log_amounts
    errors=[]
    for h in (.4,.2,.1):
        fd=(jnp.log(amounts(t+h,p,b,standards))-jnp.log(amounts(t-h,p,b,standards)))/(2*h)
        errors.append(float(jnp.max(jnp.abs(fd-got))))
    assert errors[-1] < 1e-8 and errors[-1] < errors[0]/8


def test_active_domain_and_zero_multiplier_boundary():
    def energy(t,p,n,h):
        return R*t*(jnp.sum(n*jnp.log(n/n.sum()))+h*n[1])
    problem=mp.Problem(('X',),(mp.Phase('liquid',('A','B'),[[1,1]],energy,[[.2,-.8]]),))
    def make(h):
        return mp.linearize_fixed_support(problem,temperature_k=1200.,pressure_pa=1e5,
            element_amounts_mol=[1.],amounts_mol=[.8,.2],support=mp.Support((0,1),(0,)),parameters=jnp.array(h))
    lin=make(0.)
    assert lin.derivative_valid,lin.diagnostics
    np.testing.assert_allclose(lin.jvp(0.,0.,jnp.ones(1),jnp.array(0.)).amounts_mol,[.8,.2],atol=1e-14)
    edge=make(float(jnp.log(4.)))
    assert not edge.derivative_valid and 'strict_active_multipliers' in edge.rejection_reasons
    result=edge.jvp(0.,0.,jnp.ones(1),jnp.array(0.))
    assert np.all(np.isnan(result.amounts_mol))
    assert np.all(np.isnan(edge.vjp(jax.tree_util.tree_map(jnp.ones_like,result))[2]))


def test_nonstationary_input_and_material_boundary_are_rejected():
    invalid=linearization(n=jnp.ones(2))
    assert not invalid.derivative_valid and 'stationarity' in invalid.rejection_reasons
    phase=mp.Phase('liquid',('A','B'),[[1,1]],ideal,derivative_domain=lambda t,p,n,a: False)
    n=amounts(1400.,1e5,jnp.array([2.]),jnp.array([0.,0.]))
    result=mp.linearize_fixed_support(mp.Problem(('X',),(phase,)),temperature_k=1400.,pressure_pa=1e5,
        element_amounts_mol=[2.],amounts_mol=n,support=SUPPORT,parameters=jnp.zeros(2))
    assert not result.derivative_valid and 'material_branch' in result.rejection_reasons


def test_excluded_solution_phase_is_not_a_stability_certificate():
    absent=mp.Phase('omitted',('C','D'),[[1,1]],ideal)
    problem=mp.Problem(('X',),(PROBLEM.phases[0],absent))
    n=jnp.concatenate((amounts(1400.,1e5,jnp.array([2.]),jnp.zeros(2)),jnp.zeros(2)))
    lin=mp.linearize_fixed_support(problem,temperature_k=1400.,pressure_pa=1e5,
        element_amounts_mol=[2.],amounts_mol=n,support=SUPPORT,parameters=jnp.zeros(2))
    assert lin.derivative_valid
    assert lin.diagnostics['excluded_phase_stability_checked'] is False
    np.testing.assert_array_equal(lin.jvp(0.,0.,jnp.ones(1),jnp.zeros(2)).amounts_mol[2:],0.)


def test_unsupported_zero_budget_and_redundant_elements_are_explicit():
    with pytest.raises(ValueError,match='Positive finite'):
        linearization(b=jnp.array([0.]),n=jnp.ones(2))
    phase=mp.Phase('liquid',('A','B'),[[1,1],[2,2]],ideal)
    with pytest.raises(ValueError,match='Independent'):
        mp.linearize_fixed_support(mp.Problem(('X','Y'),(phase,)),temperature_k=1400.,pressure_pa=1e5,
            element_amounts_mol=[2.,4.],amounts_mol=[1.,1.],support=SUPPORT,parameters=jnp.zeros(2))


def test_nonideal_parameter_pytree_matches_existing_transformed_solver():
    def energy(t,p,n,a):
        total=n.sum()
        return (n @ (a['standard']+p*jnp.array([1e-5,3e-5]))
                + R*t*jnp.sum(n*jnp.log(n/total))+a['mixing']*n[0]*n[1]/total)
    problem=mp.Problem(('X',),(mp.Phase('liquid',('A','B'),[[1,2]],energy),))
    params={'standard':jnp.array([20000.,-5000.]),'mixing':jnp.array(800.)}
    primals=(jnp.array(1400.),jnp.array(5e7),jnp.array([3.]),params)
    direction=(jnp.array(3.),jnp.array(2e5),jnp.array([.15]),
               {'standard':jnp.array([20.,-50.]),'mixing':jnp.array(10.)})
    def solve(t,p,b,a):
        return mp.solve_fixed_support(problem,temperature_k=t,pressure_pa=p,
            element_amounts_mol=b,support=SUPPORT,init=jnp.ones(2),parameters=a)
    state,expected=jax.jvp(solve,primals,direction)
    assert state.derivative_valid
    lin=mp.linearize_fixed_support(problem,temperature_k=primals[0],pressure_pa=primals[1],
        element_amounts_mol=primals[2],amounts_mol=state.amounts_mol,support=SUPPORT,parameters=params)
    assert lin.derivative_valid,lin.diagnostics
    got=lin.jvp(*direction)
    for field in mp.EquilibriumTangent._fields[:-1]:
        np.testing.assert_allclose(getattr(got,field),getattr(expected,field),rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize('kind', ['nonfinite_value', 'nonboolean_domain', 'integer_parameter'])
def test_malformed_material_contracts_fail_explicitly(kind):
    phase=mp.Phase('liquid',('A','B'),[[1,1]],
        (lambda t,p,n,a: ideal(t,p,n,a)+jnp.nan) if kind=='nonfinite_value' else ideal,
        derivative_domain=(lambda t,p,n,a: jnp.nan) if kind=='nonboolean_domain' else None)
    params=jnp.array([0,0]) if kind=='integer_parameter' else jnp.zeros(2)
    n=amounts(1400.,1e5,jnp.array([2.]),jnp.zeros(2))
    match={'nonfinite_value':'Gibbs value','nonboolean_domain':'boolean predicate',
           'integer_parameter':'floating-point'}[kind]
    with pytest.raises(ValueError,match=match):
        mp.linearize_fixed_support(mp.Problem(('X',),(phase,)),temperature_k=1400.,pressure_pa=1e5,
            element_amounts_mol=[2.],amounts_mol=n,support=SUPPORT,parameters=params)



def test_vjp_preserves_floating_parameter_dtype():
    lin=linearization(standards=jnp.zeros(2,dtype=jnp.float32))
    assert lin.derivative_valid
    direction=(jnp.array(1.),jnp.array(0.),jnp.array([.1]),jnp.zeros(2,dtype=jnp.float32))
    pushed=lin.jvp(*direction)
    cotangent=jax.tree_util.tree_map(jnp.ones_like,pushed)
    pulled=lin.vjp(cotangent)
    assert pulled[3].dtype == jnp.float32
    np.testing.assert_allclose(sum(jnp.vdot(x,y) for x,y in zip(pulled,direction)),
        sum(jnp.sum(x) for x in pushed),rtol=1e-6)
