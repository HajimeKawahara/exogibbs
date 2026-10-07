"""Mixed-alloy minimum certificates and exact metal phase disappearance."""

import importlib
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp


DIRECTORY = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate"
NAMES = ("local", "full_potential", "common_gibbs", "phase_selection")
previous = {name: sys.modules.get(name) for name in NAMES}
sys.path.insert(0, str(DIRECTORY))
try:
    for name in NAMES:
        sys.modules.pop(name, None)
    LOCAL, FULL, COMMON, PHASE = [importlib.import_module(name) for name in NAMES]
finally:
    sys.path.pop(0)
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


@pytest.mark.parametrize("scale", [1., 1e24])
def test_insertion_seed_preserves_trace_composition_and_atoms(scale):
    elements = ["Fe", "Cr", "Ca", "He"]
    host, metal = [e + "_host" for e in elements], [e + "_metal" for e in elements]
    record = {"elements": elements, "phases": {"host": host, "metal": metal},
              "component_formulas": {n: {e: 1} for names in (host, metal)
                                     for n, e in zip(names, elements)}}
    budget = scale * np.array([.9, .1, .01, 0.])
    original = np.r_[budget, np.zeros(4)]
    composition = np.array([.9, .1 - 1e-12, 1e-12, 0.])
    absent = SimpleNamespace(component_amounts_mol=original, gibbs_rt=0.)
    def host_state(t, p, n):
        return FULL.PhaseState(np.zeros(4), 0.)
    def metal_state(t, p, n):
        return FULL.PhaseState(-np.ones(4), -float(np.sum(n)))
    seed = PHASE._insertion_seed(record, budget, absent, composition,
                                 {"host": host_state, "metal": metal_state}, 2000., 1., .01)
    assert np.all(seed >= 0)
    assert seed[6] > 0
    assert seed[3] == seed[7] == 0
    np.testing.assert_allclose(seed[:4] + seed[4:], budget, rtol=1e-12, atol=0)
    np.testing.assert_allclose(seed[4:] / seed[4:].sum(), composition, rtol=5e-16, atol=0)
    np.testing.assert_allclose(original, absent.component_amounts_mol, rtol=0, atol=0)


def test_pure_endmembers_cannot_certify_absence_of_a_favorable_mixed_phase():
    standards = np.full(4, .4)
    result = PHASE.minimize_insertion(FULL.ideal_phase(lambda t, p: standards),
                                     2000., 1., np.eye(4), np.zeros(4), np.zeros(4),
                                     np.ones(4), curvature_lower_bound_rt=0.)
    assert result.minimum_certified
    # Every pure insertion costs +0.4, but ideal alloy mixing makes tau < 0.
    assert result.upper_bound_rt == pytest.approx(.4 - np.log(4), abs=1e-12)
    assert result.lower_bound_rt <= result.upper_bound_rt < 0
    np.testing.assert_allclose(result.composition, .25, atol=1e-10)


def test_global_ideal_reference_and_bounded_composition_minimum():
    standards = np.array([.4, 1., 1.8, 2.])
    callback = FULL.ideal_phase(lambda t, p: standards)
    ideal = PHASE.minimize_insertion(callback, 2000., 1., np.eye(4), np.zeros(4),
                                    np.zeros(4), np.ones(4), curvature_lower_bound_rt=0.)
    expected = -logsumexp(-standards)
    assert ideal.minimum_certified
    assert ideal.lower_bound_rt <= expected + 1e-12 <= ideal.upper_bound_rt + 2e-12
    np.testing.assert_allclose(ideal.composition, np.exp(-standards + expected), rtol=1e-6)
    bounded = PHASE.minimize_insertion(callback, 2000., 1., np.eye(4), np.zeros(4),
                                      np.array([.8, 0., 0., 0.]), np.ones(4),
                                      curvature_lower_bound_rt=0.)
    assert bounded.minimum_certified
    assert bounded.composition[0] == pytest.approx(.8, abs=1e-10)
    assert bounded.upper_bound_rt > ideal.upper_bound_rt


def test_numerical_composition_search_without_global_evidence_remains_unresolved():
    callback = FULL.ideal_phase(lambda t, p: np.ones(4) * 5)
    result = PHASE.minimize_insertion(callback, 2000., 1., np.eye(4), np.zeros(4),
                                     np.zeros(4), np.ones(4))
    assert result.upper_bound_rt > 0
    assert result.lower_bound_rt is None and not result.minimum_certified
    uncertain = PHASE.minimize_insertion(callback, 2000., 1., np.eye(4), np.zeros(4),
                                        np.zeros(4), np.ones(4), curvature_lower_bound_rt=-10.)
    assert not uncertain.minimum_certified and uncertain.uncertainty_rt > 1e-8


def test_restricted_callback_restores_zero_helium_and_preserves_scalar_gradient():
    record = {"elements": ["H", "He", "O"], "phases": {"atmosphere": ["H_a", "He_a", "O_a"]},
              "component_formulas": {"H_a": {"H": 1}, "He_a": {"He": 1}, "O_a": {"O": 1}},
              "reactions": []}
    budget = np.array([2., 0., 1.])
    problem = LOCAL.build_problem(record, budget, lambda t, p: np.zeros(3), phases=("atmosphere",))
    seen = []
    def callback(t, p, n):
        seen.append(n.copy())
        return FULL.PhaseState(np.array([3., -np.inf, 5.]), float(n @ [3., 0., 5.]))
    def scalar_gradient(t, p, n):
        seen.append(n.copy())
        return float(n @ [3., 0., 5.]), np.array([3., -np.inf, 5.])
    callback.energy_value_and_grad_rt = scalar_gradient
    restricted = PHASE.restrict_phase_callbacks(record, problem, {"atmosphere": callback})["atmosphere"]
    state = restricted(2000., 1., [2., 1.])
    np.testing.assert_array_equal(state.mu_rt, [3., 5.])
    assert state.gibbs_rt == 11.
    energy, gradient = restricted.energy_value_and_grad_rt(2000., 1., [2., 1.])
    assert energy == 11.
    np.testing.assert_array_equal(gradient, [3., 5.])
    for full in seen:
        np.testing.assert_array_equal(full, budget)


@pytest.mark.parametrize("status,reasons,certified,expected", [
    ("metal_present", [], True, True),
    ("metal_absent", [], True, True),
    ("unresolved", ["Host global stability is not established."], True, True),
    ("unresolved", ["Metal-bearing branch unavailable; see local attempts."], True, False),
    ("unresolved", ["Host global stability is not established."], False, False),
])
def test_local_selection_gate_distinguishes_global_host_evidence_from_failed_metal_search(
        status, reasons, certified, expected):
    selection = {"status": status, "reasons": reasons, "result": {"accepted": True},
                 "insertion": {"minimum_certified": certified}}
    assert PHASE.local_metal_selection_accepted(selection) is expected
    selection["result"]["accepted"] = False
    assert not PHASE.local_metal_selection_accepted(selection)


def _ideal_assemblage(metal_offset=0.):
    phases = {"silicate": ["SiO2_l", "FeO_l", "H2O_l", "H2_l"],
              "metal": ["Fe_m", "Si_m", "O_m", "H_m"],
              "gas": ["H2_g", "He_g", "H2O_g", "SiO_g"]}
    formulas = {"SiO2": {"Si": 1, "O": 2}, "FeO": {"Fe": 1, "O": 1},
                "H2O": {"H": 2, "O": 1}, "H2": {"H": 2},
                "Fe": {"Fe": 1}, "Si": {"Si": 1}, "O": {"O": 1}, "H": {"H": 1},
                "He": {"He": 1}, "SiO": {"Si": 1, "O": 1}}
    names = [name for group in phases.values() for name in group]
    record = {"elements": ["H", "He", "O", "Si", "Fe", "C"], "phases": phases,
              "component_formulas": {name: formulas[name.split("_")[0]] for name in names}, "reactions": []}
    matrix = np.array([[record["component_formulas"][name].get(e, 0) for name in names]
                       for e in record["elements"]])
    target = np.array([1., .5, .03, .05, .1, .003, .001, .01, .3, .1, .02, .002])
    callbacks, start = {}, 0
    for phase, components in phases.items():
        values = target[start:start + len(components)]
        standard = -np.log(values / values.sum()) + (metal_offset if phase == "metal" else 0.)
        callbacks[phase] = FULL.ideal_phase(lambda t, p, value=standard: value, gas=phase == "gas")
        start += len(components)
    return record, matrix @ target, callbacks, target


@pytest.mark.parametrize("scale", [1e-9, 1., 1e24])
def test_certified_mixed_metal_presence_and_absolute_amount_scaling(scale):
    record, budget, callbacks, target = _ideal_assemblage()
    result = PHASE.select_metal_phase(record, budget * scale, 2000., 1., callbacks,
                                      np.zeros(4), np.ones(4),
                                      convex_phase_bounds={phase: 0. for phase in callbacks})
    assert result.status == "metal_present", result.reasons
    assert result.insertion.minimum_certified
    assert result.metal_free_result.accepted
    assert result.metal_free_insertion.minimum_certified
    assert result.metal_free_insertion.upper_bound_rt < -1e-3
    assert abs(result.insertion.upper_bound_rt) < 1e-8
    np.testing.assert_array_equal(result.metal_free_result.component_amounts_mol[4:8], 0.)
    assert PHASE.local_metal_status(asdict(result)) == "metal_present"
    assert result.metal_amount_mol / scale == pytest.approx(target[4:8].sum(), rel=1e-6)
    np.testing.assert_allclose(result.result.component_amounts_mol / scale, target, rtol=1e-6)
    assert result.result.element_residual[-1] == 0.


def test_certified_metal_absence_is_exact_zero_and_keeps_an_incipient_composition():
    record, budget, callbacks, _ = _ideal_assemblage(metal_offset=10.)
    result = PHASE.select_metal_phase(record, budget, 2000., 1., callbacks,
                                      np.zeros(4), np.ones(4),
                                      convex_phase_bounds={phase: 0. for phase in callbacks})
    assert result.status == "metal_absent", result.reasons
    assert result.metal_amount_mol == 0. and result.metal_composition is None
    np.testing.assert_array_equal(result.result.component_amounts_mol[4:8], 0.)
    assert result.insertion.minimum_certified and result.insertion.lower_bound_rt > 0
    assert result.insertion.composition.sum() == pytest.approx(1.)
    assert result.metal_free_result is result.result
    assert result.metal_free_insertion is result.insertion
    assert PHASE.local_metal_status(asdict(result)) == "metal_absent"


def test_supplied_metal_free_start_keeps_the_analytic_finite_equilibrium():
    record, budget, callbacks, target = _ideal_assemblage()
    problem = LOCAL.build_problem(record, budget, lambda t, p: np.zeros(len(target)),
                                  phases=("silicate", "gas"))
    active = problem.species_indices
    matrix = np.asarray(problem.formula_matrix)
    base = COMMON._feasible_start(matrix, budget[problem.element_indices])
    direction = COMMON.null_space(matrix)[:, 0]
    nonzero = direction != 0
    distance = .2 * np.min(base[nonzero] / np.abs(direction[nonzero]))
    seed = np.zeros_like(target)
    seed[active] = base + distance * direction
    result = PHASE.select_metal_phase(
        record, budget, 2000., 1., callbacks, np.zeros(4), np.ones(4),
        convex_phase_bounds={phase: 0. for phase in callbacks},
        initial_component_amounts_mol=seed)
    assert result.status == "metal_present", result.reasons
    assert result.result.accepted and result.metal_free_result.accepted
    np.testing.assert_allclose(result.result.component_amounts_mol, target, rtol=1e-6)
    assert result.result.derivative_error_rt < 5e-6
    assert result.result.extensivity_error < 5e-9
    assert np.max(np.abs(result.result.element_residual)) < 1e-9
    invalid = PHASE.select_metal_phase(
        record, budget, 2000., 1., callbacks, np.zeros(4), np.ones(4),
        convex_phase_bounds={phase: 0. for phase in callbacks},
        initial_component_amounts_mol=seed * 1.001)
    assert invalid.status == "unresolved" and invalid.result is None
    assert "element inventory" in " ".join(invalid.reasons)


def test_melt_global_stability_cannot_be_inferred_from_local_closure_and_metal_test():
    record, budget, callbacks, _ = _ideal_assemblage(metal_offset=10.)
    result = PHASE.select_metal_phase(record, budget, 2000., 1., callbacks,
                                      np.zeros(4), np.ones(4), convex_phase_bounds={"metal": 0.})
    assert result.result.accepted and result.insertion.minimum_certified
    assert result.status == "unresolved"
    assert "Host global stability is not established." in result.reasons
    assert PHASE.local_metal_status(asdict(result)) == "metal_absent"


def test_exact_zero_hydrogen_removes_alloy_hydrogen_without_using_its_dual_potential():
    record, budget, callbacks, _ = _ideal_assemblage(metal_offset=10.)
    budget[0] = 0.
    result = PHASE.select_metal_phase(record, budget, 2000., 1., callbacks,
                                      np.zeros(4), np.ones(4),
                                      convex_phase_bounds={phase: 0. for phase in callbacks})
    assert result.status == "metal_absent", result.reasons
    assert result.insertion.composition[-1] == 0.
    assert result.result.element_residual[0] == 0.
    domain = result.metal_composition_domain
    assert domain["zero_budget_components"] == ("H_m",)
    assert domain["upper_atomic_fractions"][-1] == 1.
    assert domain["effective_upper_atomic_fractions"][-1] == 0.
    assert not domain["metal_free_incipient"]["composition_box_contact"]
    contacts = [row for row in domain["metal_free_incipient"]["active_bounds"] if row["component"] == "H_m"]
    assert len(contacts) == 2
    assert all(row["kind"] == "exact_zero_budget" for row in contacts)


def test_provider_domain_failure_is_unresolved_not_a_phase_boundary():
    record, budget, callbacks, _ = _ideal_assemblage()

    def unavailable(t, p, n):
        raise RuntimeError("Outside the supplied liquid domain.")

    callbacks["silicate"] = unavailable
    result = PHASE.select_metal_phase(record, budget, 2000., 1., callbacks,
                                      np.zeros(4), np.ones(4),
                                      convex_phase_bounds={phase: 0. for phase in callbacks})
    assert result.status == "unresolved" and result.result is None
    assert "Outside the supplied liquid domain" in result.reasons[0]


@pytest.mark.parametrize("lower,upper", [
    ([.9, .2, 0., 0.], [1., .3, .1, .1]),
    ([.5, 0., 0., 0.], [.6, .1, .1, .1]),
    ([.8, -.1, 0., 0.], [1., .1, .1, .1]),
    ([.8, 0., 0., 0.], [1., float("nan"), .1, .1]),
])
def test_invalid_domain_is_not_misclassified_as_absence_due_to_zero_budgets(lower, upper):
    record, budget, callbacks, _ = _ideal_assemblage()
    with pytest.raises(ValueError, match="feasible bounds"):
        PHASE.select_metal_phase(record, budget, 2000., 1., callbacks, lower, upper,
                                 convex_phase_bounds={phase: 0. for phase in callbacks})


def test_fixed_unsupported_component_is_never_perturbed_for_a_derivative():
    supported = FULL.ideal_phase(lambda t, p: np.array([.4, .4]))

    def phase(t, p, n):
        assert n[2] == 0., "An unsupported component was evaluated."
        state = supported(t, p, n[:2])
        return FULL.PhaseState(np.r_[state.mu_rt, np.nan], state.gibbs_rt)

    result = PHASE.minimize_insertion(phase, 2000., 1., np.eye(3), np.zeros(3),
                                     np.zeros(3), np.array([1., 1., 0.]), curvature_lower_bound_rt=0.)
    assert result.minimum_certified
    assert result.composition[2] == 0.


def test_real_ma_alloy_uses_the_provider_curvature_bound_and_full_composition():
    jax = pytest.importorskip("jax")
    eos = pytest.importorskip("exoeos")
    from exoeos.ma_interval import ma_alloy_curvature_lower_bound
    model = eos.MaFeSiOHLiquid()
    lower, upper = np.array([.86, 0., 0., 0.]), np.array([1., .08, .02, .04])
    target = np.array([.945, .025, .01, .02])
    excess = np.asarray(eos.solution_state(model, 2173.15, 1e5, target).lngamma)
    standards = .2 - np.log(target) - excess
    evaluate = jax.jit(lambda n: eos.total_solution_state(model, 2173.15, 1e5, n, standards))

    def callback(t, p, n):
        state = evaluate(n)
        return FULL.PhaseState(np.asarray(state.mu_RT), float(state.gibbs_RT))

    curvature = ma_alloy_curvature_lower_bound(model, 2173.15, lower, upper)
    result = PHASE.minimize_insertion(callback, 2173.15, 1., np.eye(4), np.zeros(4),
                                     lower, upper, curvature_lower_bound_rt=curvature)
    assert curvature > 0 and result.minimum_certified, result
    assert result.lower_bound_rt <= .2 <= result.upper_bound_rt + 1e-12
    np.testing.assert_allclose(result.composition, target, rtol=2e-6)


def test_present_phase_obeys_the_same_domain_as_its_insertion_certificate():
    record, budget, callbacks, _ = _ideal_assemblage()
    lower, upper = np.array([.86, 0., 0., 0.]), np.array([1., .08, .02, .04])
    result = PHASE.select_metal_phase(record, budget, 2000., 1., callbacks, lower, upper,
                                      convex_phase_bounds={phase: 0. for phase in callbacks})
    assert result.status == "metal_present", result.reasons
    assert result.result.accepted and result.insertion.minimum_certified
    assert result.metal_composition[-1] == pytest.approx(.04, abs=1e-10)
    assert np.all(result.metal_composition >= lower - 1e-12)
    assert np.all(result.metal_composition <= upper + 1e-12)
    assert np.max(np.abs(result.result.constrained_kkt_residual_rt)) < 1e-8
    assert np.max(np.abs(result.result.reduced_potentials_rt)) > .1
    assert result.local_attempts and result.local_attempts[-1]["result"].accepted
    assert any(item["multiplier_rt"] > .1 for item in result.result.composition_constraints)
    domain = result.metal_composition_domain
    assert domain["selected_metal"]["composition_box_contact"]
    np.testing.assert_allclose(domain["selected_metal"]["lower_fraction_slack"], result.metal_composition - lower)
    np.testing.assert_allclose(domain["selected_metal"]["upper_fraction_slack"], upper - result.metal_composition)
    assert any(row["component"] == "H_m" and row["bound"] == "upper"
               and row["kind"] == "composition_box" for row in domain["selected_metal"]["active_bounds"])
    assert domain["selected_composition_constraints"] == result.result.composition_constraints


def test_bounded_present_phase_keeps_host_stability_unresolved():
    record, budget, callbacks, _ = _ideal_assemblage()
    result = PHASE.select_metal_phase(record, budget, 2000., 1., callbacks,
                                      np.array([.86, 0., 0., 0.]), np.array([1., .08, .02, .04]),
                                      convex_phase_bounds={"metal": 0., "gas": 0.})
    assert result.result.accepted and result.insertion.minimum_certified
    assert result.status == "unresolved"
    assert result.reasons == ("Host global stability is not established.",)
    assert PHASE.local_metal_status(asdict(result)) == "metal_present"


def test_failed_present_branch_retains_negative_metal_free_insertion_without_claiming_absence(monkeypatch):
    record, budget, callbacks, _ = _ideal_assemblage()

    def unavailable(*args):
        raise RuntimeError("Metal-bearing seed is unavailable.")

    monkeypatch.setattr(PHASE, "_insertion_seed", unavailable)
    result = PHASE.select_metal_phase(record, budget, 2000., 1., callbacks,
                                      np.zeros(4), np.ones(4),
                                      convex_phase_bounds={phase: 0. for phase in callbacks})
    assert result.status == "unresolved"
    assert result.metal_free_result.accepted
    assert result.metal_free_insertion.upper_bound_rt < -1e-3
    assert result.metal_free_insertion.minimum_certified
    assert result.metal_amount_mol == 0.
    assert PHASE.local_metal_status(asdict(result)) == "unresolved"
    assert len(result.local_attempts) == 2


def test_constrained_absent_branch_keeps_instability_without_attempting_present_solve(monkeypatch):
    record, budget, callbacks, _ = _ideal_assemblage()
    def forbidden(*args, **kwargs):
        pytest.fail("A constrained absent evaluation must not seed a present solve.")
    monkeypatch.setattr(PHASE, "_insertion_seed", forbidden)
    result = PHASE.select_metal_phase(record, budget, 2000., 1., callbacks,
                                      np.zeros(4), np.ones(4), allow_metal=False,
                                      convex_phase_bounds={phase: 0. for phase in callbacks})
    assert result.result is result.metal_free_result and result.result.accepted
    assert result.insertion is result.metal_free_insertion
    assert result.insertion.upper_bound_rt < -1e-3
    assert result.insertion.minimum_certified
    assert not result.local_attempts
    assert result.status == "unresolved" and PHASE.local_metal_status(asdict(result)) == "unresolved"
    assert "favorable metal insertion" in result.reasons[0]
    np.testing.assert_array_equal(result.result.component_amounts_mol[4:8], 0.)


def test_constrained_absent_branch_still_accepts_certified_local_absence():
    record, budget, callbacks, _ = _ideal_assemblage(metal_offset=10.)
    result = PHASE.select_metal_phase(record, budget, 2000., 1., callbacks,
                                      np.zeros(4), np.ones(4), allow_metal=False,
                                      convex_phase_bounds={phase: 0. for phase in callbacks})
    assert result.status == "metal_absent"
    assert result.insertion.lower_bound_rt > 0


@pytest.mark.parametrize("value", [0, 1, None, "false", np.bool_(False)])
def test_metal_permission_requires_an_explicit_boolean(value):
    record, budget, callbacks, _ = _ideal_assemblage()
    with pytest.raises(ValueError, match="allow_metal"):
        PHASE.select_metal_phase(record, budget, 2000., 1., callbacks,
                                np.zeros(4), np.ones(4), allow_metal=value)


@pytest.mark.parametrize("amount,expected", [(0., "metal_absent"), (1., "metal_present"),
                                            (None, "unresolved"), (True, "unresolved"),
                                            ("1", "unresolved"), (float("nan"), "unresolved"),
                                            (float("inf"), "unresolved"), (-1., "unresolved")])
def test_local_status_uses_legacy_selection_gate_and_finite_amount(amount, expected):
    selection = {"status": "unresolved", "reasons": ["Host global stability is not established."],
                 "result": {"accepted": True}, "insertion": {"minimum_certified": True},
                 "metal_amount_mol": amount}
    assert PHASE.local_metal_status(selection) == expected
    selection["reasons"] = ["Metal-bearing branch unavailable; see local attempts."]
    assert PHASE.local_metal_status(selection) == "unresolved"


def test_opt_in_global_insertion_preserves_local_and_host_acceptance():
    record, budget, callbacks, _ = _ideal_assemblage(metal_offset=10.)
    calls = []
    def insertion(t, p, formula, plane, lo, hi, *, tolerance, maxiter):
        calls.append((t, p, plane.copy()))
        return PHASE.minimize_insertion(callbacks['metal'], t, p, formula, plane,
                                        lo, hi, curvature_lower_bound_rt=0.,
                                        tolerance=tolerance, maxiter=maxiter)
    result = PHASE.select_metal_phase(record, budget, 2000., 1., callbacks,
                                      np.zeros(4), np.ones(4),
                                      metal_insertion_minimizer=insertion)
    assert calls and calls[0][:2] == (2000., 1.)
    assert result.insertion.minimum_certified
    assert result.status == 'unresolved'
    assert result.reasons == ('Host global stability is not established.',)
    assert PHASE.local_metal_status(asdict(result)) == 'metal_absent'


@pytest.mark.parametrize('gap', [None, np.nan, 1e-4, -1.])
def test_opt_in_cannot_claim_certification_without_a_valid_error_bound(gap):
    record, budget, callbacks, _ = _ideal_assemblage(metal_offset=10.)
    def malformed(*args, **kwargs):
        return PHASE.InsertionMinimum(np.full(4, .25), 1., .99, gap, True, 'invalid')
    result = PHASE.select_metal_phase(record, budget, 2000., 1., callbacks,
                                      np.zeros(4), np.ones(4),
                                      metal_insertion_minimizer=malformed)
    assert result.status == 'unresolved'
    assert 'invalid certificate' in ' '.join(result.reasons)
    assert not PHASE.local_metal_selection_accepted(asdict(result))


@pytest.mark.parametrize("scale", [1., 1e24])
def test_pressure_initial_ledgers_still_solve_and_audit_both_branches(scale, monkeypatch):
    record, budget, callbacks, _ = _ideal_assemblage()
    budget *= scale
    bounds = {phase: 0. for phase in callbacks}
    prior = PHASE.select_metal_phase(record, budget, 2000., 1., callbacks,
                                    np.zeros(4), np.ones(4), convex_phase_bounds=bounds)
    assert prior.status == "metal_present"
    cold = PHASE.select_metal_phase(record, budget, 2000., 1.01, callbacks,
                                   np.zeros(4), np.ones(4), convex_phase_bounds=bounds)
    original = PHASE.minimize_gibbs
    calls = []
    def tracked(problem, t, p, b, providers, **kwargs):
        calls.append((problem.phases, p, kwargs["initial_component_amounts_mol"].copy()))
        return original(problem, t, p, b, providers, **kwargs)
    monkeypatch.setattr(PHASE, "minimize_gibbs", tracked)
    warm = PHASE.select_metal_phase(record, budget, 2000., 1.01, callbacks,
        np.zeros(4), np.ones(4), convex_phase_bounds=bounds,
        initial_component_amounts_mol=prior.metal_free_result.component_amounts_mol,
        initial_metal_present_component_amounts_mol=prior.result.component_amounts_mol)
    assert warm.status == cold.status == "metal_present", warm.reasons
    assert len(calls) == 2 and all(row[1] == 1.01 for row in calls)
    assert "metal" not in calls[0][0] and "metal" in calls[1][0]
    np.testing.assert_array_equal(calls[0][2], prior.metal_free_result.component_amounts_mol)
    np.testing.assert_array_equal(calls[1][2], prior.result.component_amounts_mol)
    np.testing.assert_allclose(warm.result.component_amounts_mol, cold.result.component_amounts_mol, rtol=2e-6)
    assert abs((warm.result.gibbs_rt-cold.result.gibbs_rt)/budget.sum()) < 1e-10
    assert warm.metal_free_insertion.minimum_certified and warm.insertion.minimum_certified
    assert warm.local_attempts[0]["initialization"] == "supplied_metal_present"
    assert warm.local_attempts[0]["selection_reasons"] == ()


@pytest.mark.parametrize("failure", ["local", "insertion"])
def test_supplied_present_seed_cannot_skip_failed_final_admission(monkeypatch, failure):
    from dataclasses import replace
    record, budget, callbacks, target = _ideal_assemblage()
    original_solve, original_insertion = PHASE.minimize_gibbs, PHASE.minimize_insertion
    calls = []
    def solve(problem, *args, **kwargs):
        result = original_solve(problem, *args, **kwargs)
        calls.append(problem.phases)
        if len(calls) == 2 and failure == "local":
            return replace(result, accepted=False, audit_reasons=("Deliberate local rejection.",))
        return result
    inserted = []
    def insertion(*args, **kwargs):
        result = original_insertion(*args, **kwargs)
        inserted.append(result)
        if len(inserted) == 2 and failure == "insertion":
            return replace(result, minimum_certified=False, reason="Deliberate global rejection.")
        return result
    monkeypatch.setattr(PHASE, "minimize_gibbs", solve)
    monkeypatch.setattr(PHASE, "minimize_insertion", insertion)
    result = PHASE.select_metal_phase(record, budget, 2000., 1., callbacks,
        np.zeros(4), np.ones(4), convex_phase_bounds={phase: 0. for phase in callbacks},
        initial_metal_present_component_amounts_mol=target)
    assert result.status == "metal_present", result.reasons
    assert len(calls) == 3
    assert result.local_attempts[0]["selection_reasons"]
    assert result.local_attempts[1]["insertion_fraction"] == .01


@pytest.mark.parametrize("mutation", ["shape", "nan", "negative", "atoms", "absent", "domain", "unsupported"])
def test_present_initial_ledger_rejects_invalid_inputs_before_chemistry(monkeypatch, mutation):
    record, budget, callbacks, target = _ideal_assemblage()
    lower, upper = np.zeros(4), np.ones(4)
    seed = target.copy()
    if mutation == "shape": seed = seed[:-1]
    elif mutation == "nan": seed[0] = np.nan
    elif mutation == "negative": seed[0] = -1.
    elif mutation == "atoms": seed[0] += .1
    elif mutation == "absent": seed[4:8] = 0.
    elif mutation == "domain": upper[0] = .5
    else:
        record["component_formulas"]["H_m"] = {"H": 1, "C": 1}
    def forbidden(*args, **kwargs):
        pytest.fail("An invalid ledger reached chemistry.")
    monkeypatch.setattr(PHASE, "minimize_gibbs", forbidden)
    with pytest.raises(ValueError, match="ledger|composition"):
        PHASE.select_metal_phase(record, budget, 2000., 1., callbacks, lower, upper,
                                initial_metal_present_component_amounts_mol=seed)


def test_fresh_absence_ignores_old_present_branch(monkeypatch):
    record, budget, callbacks, target = _ideal_assemblage(metal_offset=10.)
    original = PHASE.minimize_gibbs
    calls = []
    def tracked(problem, *args, **kwargs):
        calls.append(problem.phases)
        return original(problem, *args, **kwargs)
    monkeypatch.setattr(PHASE, "minimize_gibbs", tracked)
    result = PHASE.select_metal_phase(record, budget, 2000., 1., callbacks,
        np.zeros(4), np.ones(4), convex_phase_bounds={phase: 0. for phase in callbacks},
        initial_metal_present_component_amounts_mol=target)
    assert result.status == "metal_absent"
    assert len(calls) == 1 and "metal" not in calls[0]
    assert result.local_attempts == ()
