"""Mixed-alloy minimum certificates and exact metal phase disappearance."""

import importlib
from pathlib import Path
import sys

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


def test_melt_global_stability_cannot_be_inferred_from_local_closure_and_metal_test():
    record, budget, callbacks, _ = _ideal_assemblage(metal_offset=10.)
    result = PHASE.select_metal_phase(record, budget, 2000., 1., callbacks,
                                      np.zeros(4), np.ones(4), convex_phase_bounds={"metal": 0.})
    assert result.result.accepted and result.insertion.minimum_certified
    assert result.status == "unresolved"
    assert "Host global stability is not established." in result.reasons


def test_exact_zero_hydrogen_removes_alloy_hydrogen_without_using_its_dual_potential():
    record, budget, callbacks, _ = _ideal_assemblage(metal_offset=10.)
    budget[0] = 0.
    result = PHASE.select_metal_phase(record, budget, 2000., 1., callbacks,
                                      np.zeros(4), np.ones(4),
                                      convex_phase_bounds={phase: 0. for phase in callbacks})
    assert result.status == "metal_absent", result.reasons
    assert result.insertion.composition[-1] == 0.
    assert result.result.element_residual[0] == 0.


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


def test_bounded_present_phase_keeps_host_stability_unresolved():
    record, budget, callbacks, _ = _ideal_assemblage()
    result = PHASE.select_metal_phase(record, budget, 2000., 1., callbacks,
                                      np.array([.86, 0., 0., 0.]), np.array([1., .08, .02, .04]),
                                      convex_phase_bounds={"metal": 0., "gas": 0.})
    assert result.result.accepted and result.insertion.minimum_certified
    assert result.status == "unresolved"
    assert result.reasons == ("Host global stability is not established.",)
