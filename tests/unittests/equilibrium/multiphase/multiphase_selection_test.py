"""Phase switches, exact absence and independent insertion certificates."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp

from exogibbs.api import multiphase as mp
from exogibbs.equilibrium.multiphase.types import R_SCALE as R


def ideal_phase(name, shift):
    def energy(t, p, n, parameters):
        logx = jnp.log(jnp.where(n > 0, n / n.sum(), 1.))
        return R * t * (n @ (logx + jnp.asarray(shift)))

    def bound(t, p, potentials, parameters, allowed):
        cost = potentials / (R * t) - jnp.asarray(shift)
        return -R * t * logsumexp(cost[jnp.asarray(allowed)])

    return mp.Phase(name, ("X", "Y"), np.eye(2), energy,
                    insertion_lower_bound_j_mol=bound)


def two_solutions():
    return mp.Problem(("X", "Y"), (
        ideal_phase("liquid", (0., 0.)), ideal_phase("metal", (-1., 1.)),
    ))


def solve(problem, budget, init, **kwargs):
    return mp.solve(problem, temperature_k=1300., pressure_pa=2e5,
                    element_amounts_mol=budget, init=init, **kwargs)


@pytest.mark.parametrize("x", [.1, .5, .9])
def test_whole_solution_appearance_and_natural_disappearance(x):
    problem = two_solutions()
    # Start without metal, including the case that requires its appearance.
    result = solve(problem, [x, 1-x], [x, 1-x, 0., 0.])
    assert result.converged and result.local_kkt_valid and result.minimum_valid
    assert result.stability_status == "certified"
    np.testing.assert_allclose(problem.atom_matrix @ result.amounts_mol, [x, 1-x], atol=1e-12)
    if x == .1:
        np.testing.assert_array_equal(result.amounts_mol[2:], [0., 0.])
        assert result.insertion_lower[1] > .5
        assert np.all(np.isnan(result.chemical_potentials_j_mol[2:]))
    elif x == .9:
        np.testing.assert_array_equal(result.amounts_mol[:2], [0., 0.])
        assert result.insertion_lower[0] > .5
    else:
        liquid_x = 1 / (1 + np.e)
        expected = .5 * np.array([liquid_x, 1-liquid_x, 1-liquid_x, liquid_x])
        np.testing.assert_allclose(result.amounts_mol, expected, atol=1e-9)
        np.testing.assert_allclose(result.insertion_upper, 0., atol=1e-9)
    for composition in result.incipient_compositions:
        assert np.sum(composition) == pytest.approx(1.)


def test_removal_is_a_fresh_face_solve_and_amount_scale_is_preserved():
    problem = two_solutions()
    for scale in (1e-18, 1e24):
        result = solve(problem, scale * np.array([.1, .9]), scale * np.array([.05, .45, .05, .45]))
        assert result.stability_status == "certified"
        np.testing.assert_allclose(result.amounts_mol / scale, [.1, .9, 0., 0.], atol=1e-10)
        assert result.phase_amounts_mol[1] == 0


def test_active_domain_is_discovered():
    def energy(t, p, n, parameters):
        return R*t*jnp.sum(n*jnp.log(jnp.where(n > 0, n/n.sum(), 1.)))

    phase = mp.Phase("liquid", ("A", "B"), [[1, 1]], energy, [[.2, -.8]])
    problem = mp.Problem(("X",), (phase,))
    result = solve(problem, [1.], [.9, .1])
    assert result.local_kkt_valid and result.support.active_constraints == (0,)
    np.testing.assert_allclose(result.amounts_mol, [.8, .2], atol=1e-12)
    assert result.constraint_multipliers_j_mol[0] > 0
    assert result.stability_status == "unresolved"  # No global bound was supplied.


def test_missing_or_inconclusive_proof_never_certifies_absence():
    problem = two_solutions()
    metal = replace(problem.phases[1], insertion_lower_bound_j_mol=None)
    result = solve(replace(problem, phases=(problem.phases[0], metal)), [.1, .9], [.1, .9, 0., 0.])
    assert result.local_kkt_valid and result.phase_amounts_mol[1] == 0
    assert np.isnan(result.insertion_lower[1]) and result.insertion_upper[1] > 0
    assert result.stability_status == "unresolved"
    metal = replace(metal, insertion_lower_bound_j_mol=lambda *args: -1e6)
    result = solve(replace(problem, phases=(problem.phases[0], metal)), [.1, .9], [.1, .9, 0., 0.])
    assert result.stability_status == "unresolved"


def test_negative_trial_rejects_an_absent_branch_and_search_limit_cannot_certify_it():
    problem = two_solutions()
    absent = solve(problem, [.5, .5], [.5, .5, 0., 0.], support=mp.Support((0, 1)))
    checked = mp.audit_stability(problem, absent, temperature_k=1300., pressure_pa=2e5,
                                 element_amounts_mol=[.5, .5])
    assert checked.insertion_upper[1] < -.4
    assert checked.stability_status == "unstable"
    limited = solve(problem, [.5, .5], [.5, .5, 0., 0.], options=mp.Options(max_supports=1))
    assert limited.stability_status == "unstable"


def test_audit_repolishes_at_changed_budget_and_elemental_gauge_cancels():
    problem = two_solutions()
    old = solve(problem, [.1, .9], [.1, .9, 0., 0.])
    current = mp.audit_stability(problem, old, temperature_k=1300., pressure_pa=2e5,
                                 element_amounts_mol=[.5, .5])
    np.testing.assert_allclose(current.amounts_mol, [.5, .5, 0., 0.], atol=1e-12)
    assert current.stability_status == "unstable"
    gauge = jnp.array([2e4, -1e4])

    def shifted(phase):
        def energy(t, p, n, parameters):
            return phase.gibbs_j(t, p, n, parameters) + gauge @ n

        def bound(t, p, dual, parameters, allowed):
            return phase.insertion_lower_bound_j_mol(t, p, dual-gauge, parameters, allowed)

        return replace(phase, gibbs_j=energy, insertion_lower_bound_j_mol=bound)

    changed = solve(replace(problem, phases=tuple(map(shifted, problem.phases))),
                    [.1, .9], [.05, .45, .05, .45])
    np.testing.assert_allclose(changed.amounts_mol, old.amounts_mol, atol=1e-10)
    np.testing.assert_allclose(changed.insertion_lower, old.insertion_lower, atol=1e-10)


def test_insertion_bound_receives_current_state_and_rejects_invalid_certificate():
    problem = two_solutions()
    phase = problem.phases[1]
    calls = []

    def bound(t, p, dual, parameters, allowed):
        calls.append((t, p, np.asarray(dual), parameters, allowed))
        return phase.insertion_lower_bound_j_mol(t, p, dual, parameters, allowed)

    problem = replace(problem, phases=(problem.phases[0], replace(phase, insertion_lower_bound_j_mol=bound)))
    result = solve(problem, [.1, .9], [.1, .9, 0., 0.], parameters={"material": 2.})
    t, p, dual, parameters, allowed = calls[-1]
    assert (t, p, allowed) == (1300., 2e5, (0, 1))
    np.testing.assert_allclose(dual, result.element_potentials_j_mol)
    assert parameters == {"material": 2.}
    broken = replace(phase, insertion_lower_bound_j_mol=lambda *args: 1e9)
    with pytest.raises(ValueError, match="contradicts"):
        solve(replace(problem, phases=(problem.phases[0], broken)), [.1, .9], [.1, .9, 0., 0.])


def test_zero_budget_phase_is_not_evaluated_and_pure_bounds_are_exact():
    def forbidden(*args):
        raise AssertionError("Zero-budget phases cannot be inserted.")

    problem = mp.Problem(("X", "Y"), (
        mp.Phase("pure", ("X",), [[1], [0]], lambda t, p, n, a: n[0]*t),
        mp.Phase("forbidden", ("Y",), [[0], [1]], forbidden),
    ))
    result = solve(problem, [1., 0.], [1., 0.])
    assert result.stability_status == "certified"
    assert result.insertion_lower[0] == pytest.approx(0., abs=1e-14)
    assert np.isposinf(result.insertion_lower[1])
    assert np.all(np.isnan(result.incipient_compositions[1]))


def test_nonconvex_insertion_finds_negative_witness_without_promoting_a_trial():
    def nonconvex(t, p, n, parameters):
        x = n[0]/n.sum()
        return R*t*n.sum()*(10*(x-.2)**2*(x-.8)**2-.02)

    problem = mp.Problem(("X",), (
        mp.Phase("carrier", ("X",), [[1]], lambda t, p, n, a: 0.*n[0]),
        mp.Phase("metal", ("A", "B"), [[1, 1]], nonconvex),
    ))
    absent = solve(problem, [1.], [1., 0., 0.], support=mp.Support((0,)))
    result = mp.audit_stability(problem, absent, temperature_k=1300., pressure_pa=2e5,
                                element_amounts_mol=[1.])
    assert result.insertion_upper[1] < -.019
    assert np.isnan(result.insertion_lower[1])
    assert result.stability_status == "unstable"


def test_disappearing_phase_preserves_trace_species_in_finite_phase():
    from exogibbs.equilibrium.multiphase.selection import _neighbors
    phase = mp.Phase("gas", ("major", "trace"), [[1., 1.]], lambda t, p, n, data: n.sum())
    solid = mp.Phase("solid", ("s",), [[1.]], lambda t, p, n, data: n[0])
    problem = mp.Problem(("X",), (phase, solid))
    faces = list(_neighbors(problem, mp.Support((0, 1, 2)), np.array([1., 1e-40, 1e-30]),
                            np.zeros(0), np.zeros(0), mp.Options()))
    assert faces == [mp.Support((0, 1))]


def test_boundary_warm_start_is_blended_into_a_new_interior():
    from exogibbs.equilibrium.multiphase.selection import _seed
    phase = mp.Phase("solution", ("A", "B", "trace"), [[1., 1., 1.]],
                     lambda t, p, n, data: n.sum(), [[.1, -.9, .1]])
    problem = mp.Problem(("X",), (phase,))
    initial = np.array([.9, .1, 1e-40])
    seed = _seed(problem, mp.Support((0, 1, 2)), np.ones(1), initial)
    assert np.all(seed > 0) and np.all(problem.domain_matrix @ seed > 0)
    np.testing.assert_allclose(seed.sum(), 1., atol=1e-15)
    assert np.max(np.abs(seed-initial)) < 1e-4
