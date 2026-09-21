"""Scalar-energy minima independently checked on a finite H-Si-O reference."""

import importlib
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar
from scipy.special import xlogy


DIRECTORY = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate"
NAMES = ("local", "full_potential", "common_gibbs")
previous = {name: sys.modules.get(name) for name in NAMES}
sys.path.insert(0, str(DIRECTORY))
try:
    for name in NAMES:
        sys.modules.pop(name, None)
    LOCAL, FULL, ENERGY = [importlib.import_module(name) for name in NAMES]
finally:
    sys.path.pop(0)
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


ETA = .02
K_REACTION = .01


def silica_phase(t, p, n):
    s, u = n
    if s <= 0:
        raise ValueError("This reference requires positive silica host.")
    with np.errstate(divide="ignore"):
        mu = np.array([-2 * u / s, 2 * np.log(u / (ETA * s))])
    return FULL.PhaseState(mu, float(2 * xlogy(u, u / (ETA * s)) - 2 * u))


def reference(scale=1., *, dissolved=True, vapor=True, oxygen=40., gauge=None):
    phases = {"silicate": ["SiO2_m", "W_m"], "gas": ["H2_g", "H2O_g", "SiO_g"]}
    formulas = {"SiO2_m": {"Si": 1, "O": 2}, "W_m": {"H": 2, "O": 1},
                "H2_g": {"H": 2}, "H2O_g": {"H": 2, "O": 1}, "SiO_g": {"Si": 1, "O": 1}}
    record = {"elements": ["H", "Si", "O", "He"], "phases": phases,
              "component_formulas": formulas, "reactions": []}
    budget = scale * np.array([2., 20., oxygen, 0.])
    chosen = {"silicate": phases["silicate"] if dissolved else ["SiO2_m"],
              "gas": phases["gas"] if vapor else ["H2_g", "H2O_g"]}
    problem = LOCAL.build_problem(record, budget, lambda t, p: np.zeros(5),
                                  phases=tuple(phases), phase_components=chosen)
    gas_standards = np.array([0., 0., -np.log(K_REACTION)])[:len(chosen["gas"])]
    callbacks = {"silicate": silica_phase if dissolved else FULL.ideal_phase(lambda t, p: np.zeros(1)),
                 "gas": FULL.ideal_phase(lambda t, p: gas_standards, gas=True)}
    if gauge is not None:
        for name, section in zip(problem.phases, problem.phase_slices):
            original = callbacks[name]
            shift = np.asarray(problem.full_formula_matrix)[:, problem.species_indices[section]].T @ gauge
            def shifted(t, p, n, original=original, shift=shift):
                state = original(t, p, n)
                return FULL.PhaseState(state.mu_rt + shift, state.gibbs_rt + n @ shift)
            callbacks[name] = shifted
    return problem, budget, callbacks


def reduced_energy(zu, *, oxygen=40.):
    z, u = zu
    d = oxygen - 40.
    gas = np.array([1. - d - z, d + z - u, z])
    s = 20. - z
    if np.any(gas < 0) or s <= 0 or u < 0:
        return np.inf
    return float(-z * np.log(K_REACTION) + np.sum(xlogy(gas, gas / gas.sum()))
                 + 2 * xlogy(u, u / (ETA * s)) - 2 * u)


def solve_reference(**kwargs):
    problem, budget, callbacks = reference(**kwargs)
    return ENERGY.minimize_gibbs(problem, 2000., 1., budget, callbacks)


def test_reference_matches_independent_reduced_energy_minimum():
    result = solve_reference()
    assert result.accepted, result.audit_reasons
    # These two variables and the explicit elimination formula do not use
    # LocalProblem, its reaction matrix, or the production minimizer.
    independent = minimize(reduced_energy, [.16, .07], method="Nelder-Mead",
                           options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 1000})
    assert independent.success
    z, u = independent.x
    expected = np.array([20 - z, u, 1 - z, z - u, z])
    np.testing.assert_allclose(result.component_amounts_mol, expected, atol=5e-8, rtol=0)
    assert abs(result.gibbs_rt - independent.fun) / 62 < 1e-9
    assert result.gibbs_rt <= result.initial_gibbs_rt
    assert result.derivative_error_rt < 5e-6
    assert np.max(np.abs(result.reaction_residual)) < 1e-8
    assert result.component_amounts_mol[1] / result.component_amounts_mol[0] < .05


@pytest.mark.parametrize("scale", [1e-6, 1., 1e6])
def test_inventory_scaling_and_elemental_gauge(scale):
    base = solve_reference()
    gauge = np.array([.3, -.7, .2, 9.])
    shifted = solve_reference(scale=scale, gauge=gauge)
    assert shifted.accepted, shifted.audit_reasons
    np.testing.assert_allclose(shifted.component_amounts_mol / scale, base.component_amounts_mol, atol=1e-9)
    np.testing.assert_allclose(shifted.gibbs_rt / scale, base.gibbs_rt + np.array([2., 20., 40., 0.]) @ gauge, atol=1e-10)
    assert shifted.element_residual[-1] == 0


def test_dry_and_host_transfer_controls_have_consistent_energy():
    dry = solve_reference(dissolved=False)
    wet = solve_reference()
    assert dry.accepted and wet.accepted
    assert dry.component_amounts_mol[1] == 0
    np.testing.assert_allclose(dry.component_amounts_mol[-1], np.sqrt(.01 / 1.01), atol=1e-10)
    assert wet.gibbs_rt < dry.gibbs_rt
    fixed_host = solve_reference(vapor=False, oxygen=40.4)
    released = solve_reference(oxygen=40.4)
    assert fixed_host.accepted and released.accepted
    assert fixed_host.component_amounts_mol[-1] == 0
    assert released.gibbs_rt < fixed_host.gibbs_rt


def test_silica_host_derivatives_cross_response_and_curvature():
    n = jnp.array([20., .1])
    scalar = lambda n: 2 * n[1] * (jnp.log(n[1] / (ETA * n[0])) - 1)
    state = silica_phase(2000., 1., np.asarray(n))
    np.testing.assert_allclose(jax.grad(scalar)(n), state.mu_rt, atol=1e-14)
    expected_hessian = np.array([[2 * n[1] / n[0]**2, -2 / n[0]], [-2 / n[0], 2 / n[1]]])
    np.testing.assert_allclose(jax.hessian(scalar)(n), expected_hessian, atol=1e-14)
    assert np.linalg.eigvalsh(expected_hessian).min() >= -1e-14
    for step in (1e-4, 5e-5):
        numerical = np.array([(scalar(n + jnp.eye(2)[i] * step) - scalar(n - jnp.eye(2)[i] * step)) / (2 * step) for i in range(2)])
        np.testing.assert_allclose(numerical, state.mu_rt, atol=4e-7)


def test_zero_component_and_zero_phase_energies_are_continuous():
    ideal = FULL.ideal_phase(lambda t, p: np.array([2., -1.]), gas=True)
    zero = ideal(2000., 3., np.zeros(2))
    assert zero.gibbs_rt == 0 and np.all(np.isnan(zero.mu_rt))
    endpoint = ideal(2000., 3., np.array([1., 0.]))
    assert np.isneginf(endpoint.mu_rt[1])
    np.testing.assert_allclose(endpoint.gibbs_rt, 2 + np.log(3))
    np.testing.assert_allclose(ideal(2000., 3., np.array([1., 1e-12])).gibbs_rt, endpoint.gibbs_rt, atol=1e-10)
    assert silica_phase(2000., 1., np.array([20., 0.])).gibbs_rt == 0


def test_fresh_audits_and_unfinished_optimization():
    problem, budget, callbacks = reference()
    seen = []
    original = callbacks["silicate"]
    def phase(t, p, n):
        seen.append(n.copy())
        return original(t, p, n)
    callbacks["silicate"] = phase
    result = ENERGY.minimize_gibbs(problem, 2000., 1., budget, callbacks, maxiter=1)
    assert not result.accepted
    assert "scalar minimizer did not converge" in result.audit_reasons
    np.testing.assert_array_equal(seen[-1], result.component_amounts_mol[:2])


def test_wrong_derivative_is_rejected_even_when_euler_identity_holds():
    problem, budget, callbacks = reference()
    original = callbacks["silicate"]
    def wrong(t, p, n):
        state = original(t, p, n)
        # This extensive scalar is consistent, but the supplied gradient has
        # a tangential error whose Euler contraction vanishes identically.
        error = .01 * np.array([n[1], -n[0]]) / n.sum()
        return FULL.PhaseState(state.mu_rt + error, state.gibbs_rt)
    callbacks["silicate"] = wrong
    result = ENERGY.minimize_gibbs(problem, 2000., 1., budget, callbacks, maxiter=30)
    assert not result.accepted
    assert "scalar energy derivative disagrees with the supplied potentials" in result.audit_reasons


def test_initial_guess_cannot_supply_extra_atoms_or_unsupported_elements():
    problem, budget, callbacks = reference()
    for initial in (np.ones(5), np.full(5, -1.)):
        with pytest.raises(ValueError, match="Initial amounts"):
            ENERGY.minimize_gibbs(problem, 2000., 1., budget, callbacks,
                                  initial_component_amounts_mol=initial)
    with pytest.raises(ValueError, match="support changed"):
        ENERGY.minimize_gibbs(problem, 2000., 1., budget + [0, 0, 0, 1], callbacks)


def test_actual_exoeos_alloy_energy_drives_finite_hydrogen_exchange():
    eos = pytest.importorskip("exoeos")
    if not hasattr(eos, "total_solution_state"):
        pytest.skip("Requires the ExoEOS common total-energy provider.")
    model = eos.MaFeSiOHLiquid()
    names = ["Fe_m", "Si_m", "O_m", "H_m", "H2_g", "He_g"]
    formulas = {"Fe_m": {"Fe": 1}, "Si_m": {"Si": 1}, "O_m": {"O": 1},
                "H_m": {"H": 1}, "H2_g": {"H": 2}, "He_g": {"He": 1}}
    elements = ["Fe", "Si", "O", "H", "He", "C"]
    matrix = np.array([[formulas[name].get(e, 0) for name in names] for e in elements])
    expected = np.array([1., .02, .01, .01, .03, .02])
    budget = matrix @ expected
    record = {"elements": elements, "phases": {"metal": names[:4], "gas": names[4:]},
              "component_formulas": formulas, "reactions": []}
    problem = LOCAL.build_problem(record, budget, lambda t, p: np.zeros(6), phases=("metal", "gas"))
    standards = np.array([0., 0., 0., .5 * np.log(.6) - np.log(.01 / 1.04)])
    evaluator = jax.jit(lambda n: eos.total_solution_state(model, 1873., 1e5, n, standards))
    def alloy(t, p, n):
        model.validate_state(t, p * 1e5, n / n.sum())
        state = evaluator(n)
        return FULL.PhaseState(np.asarray(state.mu_RT), float(state.gibbs_RT))
    result = ENERGY.minimize_gibbs(problem, 1873., 1., budget,
                                  {"metal": alloy, "gas": FULL.ideal_phase(lambda t, p: np.zeros(2), gas=True)})
    assert result.accepted, result.audit_reasons
    np.testing.assert_allclose(result.component_amounts_mol, expected, atol=1e-10)
    np.testing.assert_allclose(matrix @ result.component_amounts_mol, budget, rtol=1e-9, atol=0)
    assert result.gibbs_rt < result.initial_gibbs_rt
    assert result.derivative_error_rt < 5e-6


def test_equality_forced_zero_is_preserved_and_rank_deficiency_is_unresolved():
    record = {"elements": ["A", "B"], "phases": {"solid": ["AB", "A"]},
              "component_formulas": {"AB": {"A": 1, "B": 1}, "A": {"A": 1}}, "reactions": []}
    budget = np.ones(2)
    problem = LOCAL.build_problem(record, budget, lambda t, p: np.zeros(2), phases=("solid",))
    result = ENERGY.minimize_gibbs(problem, 1000., 1., budget,
                                  {"solid": FULL.ideal_phase(lambda t, p: np.zeros(2))})
    assert result.component_amounts_mol[1] == 0
    np.testing.assert_array_equal(np.asarray(problem.full_formula_matrix) @ result.component_amounts_mol, budget)
    assert not result.accepted
    assert "present components do not identify all elemental potentials" in result.audit_reasons


def test_provider_rejected_trial_is_retried_without_altering_the_energy():
    problem, budget, callbacks = reference()
    original = callbacks["silicate"]
    rejected = []
    def limited(t, p, n):
        if n[1] / n[0] < 1e-6:
            rejected.append(n.copy())
            raise ENERGY.PhaseEvaluationError("The provider cannot evaluate this endpoint trial.")
        return original(t, p, n)
    callbacks["silicate"] = limited
    result = ENERGY.minimize_gibbs(problem, 2000., 1., budget, callbacks)
    assert result.accepted, result.audit_reasons
    np.testing.assert_allclose(result.component_amounts_mol, solve_reference().component_amounts_mol, atol=1e-9)
    assert rejected
    # The endpoint energy itself still follows the unaltered scalar.
    assert result.gibbs_rt == pytest.approx(solve_reference().gibbs_rt)


def test_ideal_scalar_ad_resolves_trace_derivative_and_detects_bad_potentials():
    callback = FULL.ideal_phase(lambda t, p: np.array([2., -1.]), gas=True)
    n = np.array([1., 1e-20])
    energy, gradient = callback.energy_value_and_grad_rt(2000., 3., n)
    state = callback(2000., 3., n)
    np.testing.assert_allclose(energy, state.gibbs_rt, atol=1e-14)
    np.testing.assert_allclose(gradient, state.mu_rt, atol=1e-14)
    record = {"elements": ["A"], "phases": {"gas": ["a", "b"]},
              "component_formulas": {"a": {"A": 1}, "b": {"A": 1}}, "reactions": []}
    problem = LOCAL.build_problem(record, np.array([1.]), lambda t, p: np.zeros(2), phases=("gas",))
    def corrupt(t, p, n):
        state = callback(t, p, n)
        error = .01 * np.array([n[1], -n[0]]) / n.sum()
        return FULL.PhaseState(state.mu_rt + error, state.gibbs_rt)
    corrupt.energy_value_and_grad_rt = callback.energy_value_and_grad_rt
    result = ENERGY.minimize_gibbs(problem, 2000., 3., np.array([1.]), {"gas": corrupt}, maxiter=30)
    assert not result.accepted
    assert "scalar energy derivative disagrees with the supplied potentials" in result.audit_reasons


@pytest.mark.parametrize("unavailable", [np.nan, np.inf, -np.inf])
def test_nonfinite_energy_at_derivative_probe_cannot_be_accepted(unavailable):
    record = {"elements": ["A", "B"], "phases": {"solid": ["a", "b"]},
              "component_formulas": {"a": {"A": 1}, "b": {"B": 1}}, "reactions": []}
    budget = np.ones(2)
    problem = LOCAL.build_problem(record, budget, lambda t, p: np.zeros(2), phases=("solid",))
    ideal = FULL.ideal_phase(lambda t, p: np.zeros(2))
    def incomplete(t, p, n):
        state = ideal(t, p, n)
        # The inventory and scaled states are valid. Only the independent
        # coordinate probes lack a scalar value; do not expose the ideal AD.
        energy = state.gibbs_rt if n[0] == n[1] else unavailable
        return FULL.PhaseState(state.mu_rt, energy)
    result = ENERGY.minimize_gibbs(problem, 2000., 1., budget, {"solid": incomplete}, polish=False)
    assert not result.accepted
    assert np.isinf(result.derivative_error_rt)
    assert "scalar derivative is unavailable at a finite-difference point" in result.audit_reasons


@pytest.mark.parametrize("scale", [1e-9, 1., 1e24])
def test_composition_boundary_matches_independent_scalar_and_positive_dual(scale):
    record = {"elements": ["A", "B", "C"], "phases": {"host": ["Ah", "Bh"], "metal": ["Am", "Bm"]},
              "component_formulas": {"Ah": {"A": 1}, "Bh": {"B": 1}, "Am": {"A": 1}, "Bm": {"B": 1}},
              "reactions": []}
    budget = scale * np.array([1., 1., 0.])
    problem = LOCAL.build_problem(record, budget, lambda t, p: np.zeros(4), phases=("host", "metal"))
    x = np.array([.9, .1])
    host_standard = -np.log(np.array([.82, .98]) / 1.8)
    metal_standard = np.array([.2, -1.8]) - np.log(x)
    gauge = np.array([.7, -.4])
    callbacks = {"host": FULL.ideal_phase(lambda t, p: host_standard + gauge),
                 "metal": FULL.ideal_phase(lambda t, p: metal_standard + gauge)}
    result = ENERGY.minimize_gibbs(problem, 2000., 1., budget, callbacks,
                                  phase_composition_bounds={"metal": (np.zeros(2), np.array([1., .1]))})
    assert result.accepted, result.audit_reasons

    def independent(total):
        metal = total * x
        host = 1 - metal
        return (host @ host_standard + np.sum(xlogy(host, host / host.sum()))
                + metal @ metal_standard + np.sum(xlogy(metal, x)))

    minimum = minimize_scalar(independent, bounds=(0., 1 / .9), method="bounded",
                              options={"xatol": 1e-13})
    expected = np.r_[1 - minimum.x * x, minimum.x * x]
    np.testing.assert_allclose(result.component_amounts_mol / scale, expected, atol=5e-8, rtol=0)
    np.testing.assert_allclose(result.elemental_potentials_rt, np.r_[gauge, 0.], atol=1e-8)
    assert np.max(np.abs(result.reduced_potentials_rt)) > 1
    assert np.max(np.abs(result.constrained_kkt_residual_rt)) < 1e-8
    assert result.composition_constraints[0]["multiplier_rt"] == pytest.approx(2., abs=1e-8)
    assert abs(result.composition_constraints[0]["fraction_slack"]) < 1e-12
    assert result.gibbs_rt / scale == pytest.approx(minimum.fun + gauge.sum(), abs=1e-12)


def test_composition_bound_forces_exact_zero_without_evaluating_its_potential():
    record = {"elements": ["A"], "phases": {"metal": ["a", "b"]},
              "component_formulas": {"a": {"A": 1}, "b": {"A": 1}}, "reactions": []}
    problem = LOCAL.build_problem(record, np.ones(1), lambda t, p: np.zeros(2), phases=("metal",))
    callback = FULL.ideal_phase(lambda t, p: np.zeros(2))
    result = ENERGY.minimize_gibbs(problem, 2000., 1., np.ones(1), {"metal": callback},
                                  phase_composition_bounds={"metal": (np.zeros(2), np.array([1., 0.]))})
    assert result.accepted, result.audit_reasons
    np.testing.assert_array_equal(result.component_amounts_mol, [1., 0.])


def test_composition_domain_validates_inputs_and_initial_amounts():
    problem, budget, callbacks = reference()
    with pytest.raises(ValueError, match="active phases"):
        ENERGY.minimize_gibbs(problem, 2000., 1., budget, callbacks,
                              phase_composition_bounds={"metal": (np.zeros(2), np.ones(2))})
    for limits in (([-1., 0.], [1., 1.]), ([.9, .2], [1., 1.]), ([0., 0.], [.2, .2])):
        with pytest.raises(ValueError, match="feasible composition bounds"):
            ENERGY.minimize_gibbs(problem, 2000., 1., budget, callbacks,
                                  phase_composition_bounds={"silicate": limits})
    result = solve_reference()
    with pytest.raises(ValueError, match="phase composition bounds"):
        ENERGY.minimize_gibbs(problem, 2000., 1., budget, callbacks,
                              initial_component_amounts_mol=result.component_amounts_mol,
                              phase_composition_bounds={"silicate": ([0., 0.], [1., 0.])})


def test_atom_equalities_can_exclude_an_entire_supported_phase_without_a_floor():
    record = {"elements": ["A", "B"], "phases": {"host": ["AB"], "metal": ["a", "b"]},
              "component_formulas": {"AB": {"A": 1, "B": 1}, "a": {"A": 1}, "b": {"A": 1}},
              "reactions": []}
    problem = LOCAL.build_problem(record, np.ones(2), lambda t, p: np.zeros(3), phases=("host", "metal"))
    with pytest.raises(ValueError, match="declared phase is unreachable"):
        ENERGY.minimize_gibbs(problem, 2000., 1., np.ones(2),
                              {"host": FULL.ideal_phase(lambda t, p: np.zeros(1)),
                               "metal": FULL.ideal_phase(lambda t, p: np.zeros(2))},
                              phase_composition_bounds={"metal": ([.86, 0.], [1., .14])})
