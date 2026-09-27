"""One-sided phase insertion controls independent of the native runtime."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


EXAMPLES = Path(__file__).resolve().parents[3] / "examples/metal_silicate"
sys.path.insert(0, str(EXAMPLES))
from m2_host_stability import assess_host_candidates
from melts_coupled import COMMON_R, PROVIDER_MODEL_ID
from m2_stability_search import _search, search_competing_solutions, search_liquid_splitting


def properties(cost=-0.2):
    return {
        "model_id": PROVIDER_MODEL_ID, "status": "ok_supplied_liquid_properties",
        "T_K": 1000., "P_Pa": 1e5, "component_order": ["A", "B"],
        "component_moles": [2., 3.], "mu_RT": [-2., -3.],
        "oxide_molar_masses_g_mol": [1., 1.],
        "basis": {"component_oxide_matrix": [[1., 0.], [0., 1.]],
                  "component_element_matrix": [[1., 0.], [0., 1.]],
                  "element_order": ["A", "B"], "common_R_J_mol_K": COMMON_R},
        "phase_policy": {"oxygen_buffer": "None", "equilibrated": False},
        "saturation": {"candidate_order": ["AB"], "equilibrated": False,
                       "candidates": [{"phase": "AB", "native_affinity_J": cost * COMMON_R * 1000.,
                                       "status": "ok_candidate_properties", "reason": None,
                                       "oxide_mass_g": [1., 1.],
                                       "gibbs_J": (-5. + cost) * COMMON_R * 1000.}]},
    }


def test_negative_feasible_insertion_rejects_but_nonnegative_is_not_certificate():
    rejected = assess_host_candidates(properties(), 0.)
    row = rejected["candidates"][0]
    assert rejected["status"] == "rejected_by_feasible_trial"
    assert row["maximum_feasible_trial_scale"] == 2.
    assert row["insertion_rt_per_mol_atoms"] == pytest.approx(-0.1)
    unresolved = assess_host_candidates(properties(0.2), 0.)
    assert unresolved["status"] == "unresolved"
    assert not unresolved["global_stability_certified"]
    assert not row["minimum_certified"]


def test_h2_dilution_is_the_derivative_of_the_same_full_scalar():
    native = properties()
    h2 = 5.
    row = assess_host_candidates(native, h2)["candidates"][0]
    assert row["h2_dilution_correction_gibbs_rt"] == pytest.approx(2. * np.log(2.))
    # Directly differentiate the extensive ideal host/H2 mixing term while
    # consuming one mole of each host component to insert the candidate.
    def total(step):
        nhost = 5. - 2. * step
        dilution = nhost * np.log(nhost / (nhost + h2)) + h2 * np.log(h2 / (nhost + h2))
        return -0.2 * step + dilution
    step = 1e-5
    derivative = (total(step) - total(-step)) / (2 * step)
    assert row["insertion_rt_per_mol_atoms"] == pytest.approx(derivative / 2., abs=1e-10)
    assert row["status"] == "nonnegative_insertion_trial"


def test_insertion_is_invariant_to_element_gauge_and_amount_scale():
    reference = assess_host_candidates(properties(), 1.)["candidates"][0]
    shifted = properties()
    gauge = np.array([70., -20.])
    shifted["mu_RT"] = (np.asarray(shifted["mu_RT"]) + gauge).tolist()
    shifted["saturation"]["candidates"][0]["gibbs_J"] += gauge.sum() * COMMON_R * 1000.
    shifted["component_moles"] = [20., 30.]
    row = assess_host_candidates(shifted, 10.)["candidates"][0]
    assert row["insertion_rt_per_mol_atoms"] == pytest.approx(reference["insertion_rt_per_mol_atoms"], abs=1e-13)
    assert row["maximum_feasible_trial_scale"] == pytest.approx(10. * reference["maximum_feasible_trial_scale"])


def test_unavailable_and_infeasible_candidates_remain_unresolved():
    native = properties()
    native["component_moles"][0] = 0.
    native["mu_RT"][0] = None
    assert "absent" in assess_host_candidates(native, 0.)["candidates"][0]["reason"]
    native = properties()
    native["saturation"]["candidates"][0].update(status="unavailable", reason="No native estimate.")
    result = assess_host_candidates(native, 0.)
    assert result["unresolved_trial_phases"] == ["AB"]
    assert result["negative_trial_phases"] == []


def test_tiny_consumption_of_absent_component_cannot_create_a_false_certificate():
    native = properties(-100.)
    native["component_moles"][0] = 0.
    native["mu_RT"][0] = None
    native["saturation"]["candidates"][0]["oxide_mass_g"][0] = 1e-15
    result = assess_host_candidates(native, 0.)
    assert result["status"] == "unresolved"
    assert "absent" in result["candidates"][0]["reason"]


@pytest.mark.parametrize("change", ["buffer", "equilibrated", "reordered", "duplicate", "gas_constant"])
def test_changed_state_or_incomplete_catalog_is_rejected(change):
    native = properties()
    if change == "buffer":
        native["phase_policy"]["oxygen_buffer"] = "IW"
    elif change == "equilibrated":
        native["saturation"]["equilibrated"] = True
    elif change == "reordered":
        native["saturation"]["candidate_order"] = []
    elif change == "duplicate":
        native["saturation"]["candidate_order"] *= 2
        native["saturation"]["candidates"] *= 2
    else:
        native["basis"]["common_R_J_mol_K"] = 8.3143
    with pytest.raises(ValueError):
        assess_host_candidates(native, 0.)


def test_solution_search_uses_h2_dilution_and_does_not_certify_nonnegative_trials():
    native = properties()

    def evaluate(t, p, n, **kwargs):
        rows = []
        for request in kwargs.get("candidate_compositions", []):
            row = {"phase": "AB", "native_endmember_oxide_mass_g_per_mol": np.eye(2).tolist()}
            if "endmember_moles" in request:
                n = np.asarray(request["endmember_moles"])
                positive = n > 0
                energy = n @ [-2., -3.] + .4 * n.sum() + np.sum(n[positive] * np.log(n[positive] / n.sum()))
                row.update(status="ok_candidate_properties", reason=None, oxide_mass_g=n.tolist(),
                           returned_oxide_mass_g=n.tolist(), gibbs_J=energy * COMMON_R * t)
            rows.append(row)
        return {**native, "returned_component_moles": native["component_moles"],
                "candidate_evaluations": rows, "provenance": {"mock": True}}

    provider = SimpleNamespace(evaluate_liquid=evaluate)
    dry = search_competing_solutions(native, 0., evaluator=provider, runtime=None, python_executable=None)
    wet = search_competing_solutions(native, 5., evaluator=provider, runtime=None, python_executable=None)
    assert dry["status"] == "rejected_by_feasible_trial"
    assert dry["phases"][0]["best_fresh_trial"]["objective_rt"] == pytest.approx(.4 - np.log(2.), abs=1e-8)
    assert wet["status"] == "unresolved"
    assert wet["phases"][0]["best_fresh_trial"]["objective_rt"] == pytest.approx(.4, abs=1e-8)
    assert not wet["global_stability_certified"]
    assert wet["phases"][0]["lower_bound_rt"] is None
    assert wet["phases"][0]["best_fresh_trial"]["fresh_final"]


@pytest.mark.parametrize("interaction,expected", [(0., "unresolved"), (4., "negative_feasible_witness")])
def test_two_liquid_search_preserves_components_and_detects_nonconvex_host(interaction, expected):
    native = properties()
    native["component_moles"] = [.5, .5]
    native["mu_RT"] = (np.full(2, np.log(.5) + interaction / 4)).tolist()
    native["gibbs_J"] = (np.log(.5) + interaction / 4) * COMMON_R * native["T_K"]

    def evaluate(t, p, n, **kwargs):
        n = np.asarray(n)
        energy = np.sum(n * np.log(n / n.sum())) + interaction * np.prod(n) / n.sum()
        return {**native, "component_moles": n.tolist(), "returned_component_moles": n.tolist(),
                "gibbs_J": energy * COMMON_R * t}

    result = search_liquid_splitting(native, .1, evaluator=SimpleNamespace(evaluate_liquid=evaluate),
        runtime=None, python_executable=None, max_evaluations=200)
    assert result["status"] == expected
    assert result["lower_bound_rt"] is None and not result["minimum_certified"]
    daughters = result["best_fresh_trial"]["daughters"]
    np.testing.assert_allclose(np.sum([row["native_component_moles"] for row in daughters], axis=0), [.5, .5])
    assert sum(row["dissolved_h2_moles"] for row in daughters) == pytest.approx(.1)
    if interaction:
        assert result["best_fresh_trial"]["objective_rt"] < -.01


def test_failed_search_evaluations_never_become_finite_thermodynamic_evidence():
    def fail(point):
        raise ValueError("Native endpoint unavailable.")
    result = _search(fail, [np.array([.5])], [(0., 1.)], max_evaluations=5, tolerance_rt=1e-8)
    assert result["status"] == "unresolved" and result["best_fresh_trial"] is None
    assert not result["trials"] and result["failed_evaluations"]


def test_prescan_reaches_endpoint_before_local_gradient_uses_the_budget():
    result = _search(lambda x: {"objective_rt": float(x[0]) - .2}, [np.array([.5])], [(0., 1.)],
                     max_evaluations=2, tolerance_rt=1e-8,
                     initial_points=[np.array([.5]), np.array([0.])])
    assert result["status"] == "negative_feasible_witness"
    assert result["initial_points_attempted"] == result["initial_points_requested"] == 2
    assert result["best_fresh_trial"]["coordinates"] == [0.]
    assert result["budget_exhausted"]


def test_singleton_has_complete_composition_enumeration_but_no_global_certificate():
    native = properties(.4)

    def evaluate(t, p, n, **kwargs):
        row = {"phase": "AB", "native_endmember_oxide_mass_g_per_mol": [[1.], [1.]]}
        if "endmember_moles" in kwargs["candidate_compositions"][0]:
            row.update(native["saturation"]["candidates"][0])
        return {**native, "returned_component_moles": native["component_moles"],
                "candidate_evaluations": [row], "provenance": {"mock": True}}

    result = search_competing_solutions(native, 0., evaluator=SimpleNamespace(evaluate_liquid=evaluate),
                                       runtime=None, python_executable=None, max_evaluations=2)
    row = result["phases"][0]
    assert row["composition_domain_is_single_point"] and row["composition_minimum_enumerated"]
    assert row["status"] == "nonnegative_fixed_composition"
    assert len(row["trials"]) == 2 and not row["optimizer_attempts"]
    assert not row["minimum_certified"] and not result["global_stability_certified"]


def regular_liquid(interaction, *, scale=1., gauge=(0., 0.)):
    native = properties()
    gauge = np.asarray(gauge)

    def evaluate(t, p, n, **kwargs):
        n = np.asarray(n)
        x = n / n.sum()
        energy = np.sum(n * np.log(x)) + interaction * np.prod(n) / n.sum() + n @ gauge
        mu = np.log(x) + interaction * x[::-1] ** 2 + gauge
        return {**native, "component_moles": n.tolist(), "returned_component_moles": n.tolist(),
                "mu_RT": mu.tolist(), "gibbs_J": energy * COMMON_R * t}

    native = evaluate(native["T_K"], native["P_Pa"], np.array([.5, .5]) * scale)
    return native, SimpleNamespace(evaluate_liquid=evaluate)


@pytest.mark.parametrize("interaction,expected", [(0., "numerically_positive_local_curvature"),
                                                    (4., "negative_feasible_witness")])
def test_local_curvature_uses_complete_support_and_finite_energy_witness(interaction, expected):
    from m2_stability_search import assess_liquid_local_curvature
    native, provider = regular_liquid(interaction)
    result = assess_liquid_local_curvature(native, .1, evaluator=provider,
                                          runtime=None, python_executable=None)
    assert result["status"] == expected
    assert result["composition_subspace_dimension"] == 2
    assert len(result["provider_evaluations"]) == 1 + 4 * 3 + 6
    assert not result["global_stability_certified"] and not result["curvature_bound_certified"]
    for split in result["finite_split_trials"]:
        daughters = [result["provider_evaluations"][i] for i in split["daughter_evaluation_indices"]]
        np.testing.assert_allclose(np.sum([d["native_component_moles"] for d in daughters], axis=0),
                                   np.array([.5, .5]) / 1.1, rtol=0, atol=1e-16)
        assert sum(d["dissolved_h2_moles"] for d in daughters) == pytest.approx(.1 / 1.1)
    if interaction:
        assert min(row["objective_rt_per_mol_atoms"] for row in result["finite_split_trials"]) < -1e-4
    else:
        eig = result["finite_difference_assessments"][-1]["projected_eigenvalues"]
        np.testing.assert_allclose(eig, [1., 1.], rtol=1e-6)


def test_local_curvature_gauge_amount_invariance_and_exact_zero_h2():
    from m2_stability_search import assess_liquid_local_curvature
    results = []
    for scale, gauge in [(1., (0., 0.)), (10., (71., -23.))]:
        native, provider = regular_liquid(0., scale=scale, gauge=gauge)
        result = assess_liquid_local_curvature(native, 0., evaluator=provider,
                                              runtime=None, python_executable=None)
        assert result["active_component_indices"] == [0, 1]
        assert result["composition_subspace_dimension"] == 1
        results.append(result)
    assert results[0]["status"] == results[1]["status"] == "numerically_positive_local_curvature"
    np.testing.assert_allclose(results[0]["finite_difference_assessments"][-1]["projected_eigenvalues"],
                               results[1]["finite_difference_assessments"][-1]["projected_eigenvalues"], atol=1e-10)


def test_local_curvature_invalid_steps_and_changed_provider_host_are_rejected():
    from m2_stability_search import assess_liquid_local_curvature
    native, provider = regular_liquid(0.)
    with pytest.raises(ValueError, match="decreasing"):
        assess_liquid_local_curvature(native, 0., evaluator=provider, runtime=None,
                                      python_executable=None, relative_steps=(1e-3, 1e-2))
    provider.evaluate_liquid = lambda *args, **kwargs: {**native, "returned_component_moles": [9., 9.]}
    with pytest.raises(ValueError, match="changed"):
        assess_liquid_local_curvature(native, 0., evaluator=provider, runtime=None, python_executable=None)


def test_solution_search_stays_on_supported_face_without_faking_pure_phase_coverage():
    native = properties(.4)
    native["component_moles"] = [2., 0.]
    native["mu_RT"] = [-2., None]
    native["saturation"]["candidates"][0]["oxide_mass_g"] = [1., 0.]
    requests = []

    def evaluate(t, p, n, **kwargs):
        request = kwargs["candidate_compositions"][0]
        row = {"phase": "AB", "native_endmember_oxide_mass_g_per_mol": np.eye(2).tolist()}
        if "endmember_moles" in request:
            x = np.asarray(request["endmember_moles"])
            requests.append(x)
            assert x[1] == 0 and x[0] == 1
            row.update(status="ok_candidate_properties", reason=None, oxide_mass_g=x.tolist(),
                       gibbs_J=-1.5 * COMMON_R * t)
        return {**native, "returned_component_moles": native["component_moles"],
                "candidate_evaluations": [row], "provenance": {"mock": True}}

    result = search_competing_solutions(native, 0., evaluator=SimpleNamespace(evaluate_liquid=evaluate),
                                       runtime=None, python_executable=None)
    row = result["phases"][0]
    assert row["supported_face"]["supported_endmember_indices"] == [0]
    assert row["supported_face"]["excluded_endmember_indices"] == [1]
    assert len(requests) == 2
    assert not row["composition_domain_is_single_point"]
    assert not row["composition_minimum_enumerated"]
    assert row["status"] == "unresolved"


def test_published_host_uses_its_own_potential_and_separate_native_trials():
    from copy import deepcopy
    from m2_host_stability import evaluate_host_stability
    from melts_coupled import PUBLISHED_MODEL_ID

    native = properties()
    published = deepcopy(native)
    published.update(model_id=PUBLISHED_MODEL_ID, mu_RT=[-3., -4.])
    published.pop("saturation")
    calls = []
    def host_call(t, p, n, **options):
        assert "include_saturation" not in options
        calls.append("host")
        return published
    def candidate_call(t, p, n, **options):
        assert options["include_saturation"] is True
        calls.append("candidate")
        return native
    common = dict(COMPONENTS=["A", "B"], ELEMENTS=["A", "B"], FORMULA_MATRIX=np.eye(2))
    host = SimpleNamespace(**common, MODEL_ID=PUBLISHED_MODEL_ID, evaluate_liquid=host_call)
    provider = SimpleNamespace(**common, MODEL_ID=PROVIDER_MODEL_ID, evaluate_liquid=candidate_call)
    record = {"phases": {"silicate": ["A_melts", "B_melts", "H2_dissolved"]},
              "component_formulas": {"A_melts": {"A": 1.}, "B_melts": {"B": 1.}, "H2_dissolved": {"H": 2}}}
    result = evaluate_host_stability(record, [2., 3., 0.], 1000., 1., evaluator=host,
        candidate_evaluator=provider, runtime=None, python_executable=None, amount_scale=1.)
    assert calls == ["host", "candidate"]
    assert result["provider_properties"] is published
    assert result["native_candidate_properties"] is native
    assert result["candidates"][0]["insertion_rt_per_mol_atoms"] == pytest.approx(.9)
    assert result["negative_trial_phases"] == []
    assert not result["global_stability_certified"]
    with pytest.raises(ValueError, match="explicit native"):
        evaluate_host_stability(record, [2., 3., 0.], 1000., 1., evaluator=host,
            runtime=None, python_executable=None, amount_scale=1.)


@pytest.mark.parametrize("corruption", ["temperature", "pressure", "amounts", "basis", "model"])
def test_selected_host_rejects_mismatched_native_candidate_receipts(corruption):
    from copy import deepcopy
    from melts_coupled import PUBLISHED_MODEL_ID
    native = properties()
    selected = deepcopy(native)
    selected["model_id"] = PUBLISHED_MODEL_ID
    selected.pop("saturation")
    if corruption == "temperature":
        native["T_K"] += 1.
    elif corruption == "pressure":
        native["P_Pa"] += 1.
    elif corruption == "amounts":
        native["component_moles"][0] += 1e-12
    elif corruption == "basis":
        native["basis"]["component_oxide_matrix"][0][0] = 2.
    else:
        selected["model_id"] = "unknown"
    with pytest.raises(ValueError):
        assess_host_candidates(selected, 0., candidate_properties=native)


def test_saved_liquid_model_preserves_and_cross_checks_all_declarations():
    from melts_coupled import saved_liquid_model, PUBLISHED_MODEL_ID
    assert saved_liquid_model({}) == "native"
    source = {"arguments": {"liquid_model": "published"}, "source_metadata": {
        "liquid_model": "published", "host_ledger": {"model_id": PUBLISHED_MODEL_ID}}}
    assert saved_liquid_model(source) == "published"
    source["liquid_model"] = "native"
    with pytest.raises(ValueError, match="inconsistent"):
        saved_liquid_model(source)
    for value in ("unknown", None, True):
        with pytest.raises(ValueError, match="unknown"):
            saved_liquid_model({"liquid_model": value})
    with pytest.raises(ValueError, match="inconsistent"):
        saved_liquid_model({"source_metadata": {"host_ledger": {"model_id": PUBLISHED_MODEL_ID}}})
