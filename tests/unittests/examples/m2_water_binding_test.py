"""Final-source identity, common-plane and external-shift binding controls."""
import copy
import importlib
from pathlib import Path
import sys

import pytest

DIRECTORY = Path(__file__).resolve().parents[3]/"examples/metal_silicate"
sys.path.insert(0, str(DIRECTORY))
try:
    runner = importlib.import_module("run_m2_water_global")
finally:
    sys.path.pop(0)

from m2_water_global_test import saved_water


def fixture(monkeypatch):
    hydrogen = {"dissolved_h2_base_standard_rt": 4., "H2_dissolved_standard_offset_rt": 0.}
    monkeypatch.setattr(runner, "source_hydrogen_standard_receipt", lambda source: hydrogen)
    p = saved_water()
    pins = {"exogibbs": {"head": "g", "status": ""}, "exoeos": {"head": "e", "status": ""}}
    budget = {"elements": ["X", "O", "H"], "total_element_amounts_mol": [2., 5.1, .3]}
    source = {"temperature_K": 2000., "pressure_bar": 1.,
              "source_metadata": {"liquid_model": "published_water", "host_ledger": {
                  "model_id": p["model_id"], "water_standard_receipts": [
                      {"T_K": 2000., "P_Pa": 1e5, "H2O_gas_standard_RT": 2.,
                       "common_R_J_mol_K": 1., "gas_standard_pressure_Pa": 1e5}]},
                  "input": {"native_amount_scale": 1., "element_amounts_mol": budget["total_element_amounts_mol"]},
                  "provenance": {owner: {"commit": item["head"], "changed_tracked_file_sha256": {}} for owner, item in pins.items()}},
              "source_internal_record": {"elements": budget["elements"],
                  "phases": {"silicate": ["a_melts", "b_melts", "h2o_melts", "H2_dissolved"]},
                  "component_formulas": {"a_melts": {"X": 1., "O": 2.}, "b_melts": {"X": 1., "O": 3.},
                                         "h2o_melts": {"H": 2., "O": 1.}, "H2_dissolved": {"H": 2}}},
              "source_internal_result": {"accepted": True, "component_amounts_mol": [1., 1., .1, .05],
                                         "elemental_potentials_rt": [0., 0., 0.]}}
    state = {"source": source, "accepted": True, "global_closure_numerically_accepted": True,
             "pressure_closure_performed": True, "contact_accepted": True, "column_numerically_accepted": True,
             "layers": [{}], "temperature_base_k": 2000., "pressure_base_pa": 1e5}
    report = {"numerically_accepted": True, "checkouts": pins, "runs": [{"roots": [state]}],
              "inventory": budget, "provider_scenario": None}
    audit = {"dissolved_hydrogen_standard_receipt": dict(hydrogen),
             "source": {"kind": "global_closure_root", "sha256": "source-digest", "numerical_source_accepted": True,
                        "source_contact_accepted": True, "executed_checkouts": copy.deepcopy(pins),
                        "selection": {"run_index": 0, "root_index": 0, "layers": 1, "temperature_k": 2000., "pressure_bar": 1.}},
             "host_stability": {"temperature_K": 2000., "pressure_Pa": 1e5, "provider_properties": p,
                                "native_amount_scale": 1., "native_host_component_amounts_mol": [1., 1., .1],
                                "native_dissolved_h2_moles": .05}}
    return report, audit, source


def test_bound_problem_uses_the_exact_source_and_audited_host(monkeypatch):
    report, audit, _ = fixture(monkeypatch)
    p, n, receipt = runner.saved_water_problem(report, audit, "source-digest")
    assert n == [1., 1.]
    assert p["dissolved_h2_cost_rt"].lo == 4
    assert receipt["source_provider_checkouts"] == report["checkouts"]


@pytest.mark.parametrize("change", [
    lambda r,a,s: a["source"].update(sha256="changed"),
    lambda r,a,s: a["dissolved_hydrogen_standard_receipt"].update(dissolved_h2_base_standard_rt=4.1),
    lambda r,a,s: r.update(numerically_accepted=False),
    lambda r,a,s: r["checkouts"]["exogibbs"].update(status="modified"),
    lambda r,a,s: a["source"]["selection"].update(pressure_bar=2.),
    lambda r,a,s: a["host_stability"].update(native_dissolved_h2_moles=.06),
    lambda r,a,s: s["source_internal_result"].update(component_amounts_mol=[1., 1., .11, .05]),
    lambda r,a,s: s["source_internal_result"].update(elemental_potentials_rt=[0., 0.]),
    lambda r,a,s: s["source_internal_result"].update(elemental_potentials_rt=[0., 0., float("nan")]),
    lambda r,a,s: s["source_metadata"]["host_ledger"]["water_standard_receipts"][0].update(H2O_gas_standard_RT=2.1),
    lambda r,a,s: s["source_metadata"].update(provider_scenario={"standard_offsets_rt": {"h2o_melts": 1.}}),
])
def test_changed_source_binding_cannot_be_certified(monkeypatch, change):
    report, audit, source = fixture(monkeypatch)
    change(report, audit, source)
    with pytest.raises(ValueError):
        runner.saved_water_problem(report, audit, "source-digest")


def water_trial_scenario():
    # Preserve the raw-versus-normalized shape of the accepted 250 bar pilot.
    raw = {"standard_offsets_rt": {"O_metal": 1.6320817196241961,
                                   "H_metal": -0.4445191104224264}}
    normalized = runner.normalize_scenario(raw)
    del normalized["standard_offsets_rt"]["h2o_melts"]
    return raw, normalized


def test_original_water_trial_scenario_accepts_only_default_normalization(monkeypatch):
    report, audit, source = fixture(monkeypatch)
    raw, normalized = water_trial_scenario()
    report["provider_scenario"] = {"values": raw, "sha256": "original-scenario-bytes"}
    source["provider_scenario_sha256"] = "original-scenario-bytes"
    source["source_metadata"]["provider_scenario"] = normalized
    runner.saved_water_problem(report, audit, "source-digest")
    assert report["provider_scenario"]["values"] == raw
    assert source["source_metadata"]["provider_scenario"] == normalized


@pytest.mark.parametrize("change", [
    lambda s: s.update(provider_scenario_sha256="changed-bytes"),
    lambda s: s["source_metadata"]["provider_scenario"]["standard_offsets_rt"].update(h2o_melts=1.),
    lambda s: s["source_metadata"]["provider_scenario"]["standard_offsets_rt"].update(H2_dissolved=1.),
    lambda s: s["source_metadata"]["provider_scenario"]["metal_bounds"]["upper"].__setitem__(1, .07),
])
def test_normalization_preserves_scenario_bytes_and_nondefault_values(monkeypatch, change):
    report, audit, source = fixture(monkeypatch)
    raw, normalized = water_trial_scenario()
    report["provider_scenario"] = {"values": raw, "sha256": "original-scenario-bytes"}
    source["provider_scenario_sha256"] = "original-scenario-bytes"
    source["source_metadata"]["provider_scenario"] = normalized
    change(source)
    with pytest.raises(ValueError, match="scenario"):
        runner.saved_water_problem(report, audit, "source-digest")


@pytest.mark.parametrize("flag", ["pressure_closure_performed", "global_closure_numerically_accepted"])
def test_fixed_pressure_trial_cannot_pass_the_final_root_gate(monkeypatch, flag):
    report, audit, source = fixture(monkeypatch)
    raw, normalized = water_trial_scenario()
    report["provider_scenario"] = {"values": raw, "sha256": "original-scenario-bytes"}
    source["provider_scenario_sha256"] = "original-scenario-bytes"
    source["source_metadata"]["provider_scenario"] = normalized
    report["runs"][0]["roots"][0][flag] = False
    with pytest.raises(ValueError, match="unaccepted pressure trial"):
        runner.saved_water_problem(report, audit, "source-digest")


def fixed_pressure_fixture(monkeypatch):
    report, audit, source = fixture(monkeypatch)
    state = report.pop("runs")[0]["roots"][0]
    report.pop("numerically_accepted")
    state.pop("accepted")
    state.pop("global_closure_numerically_accepted")
    state["pressure_closure_performed"] = False
    state["layers"] = [{}, {}]
    state["connection_numerically_accepted"] = True
    report.update(model_id="m2_fixed_pressure_internal_source_v1", source_state=state,
                  fixed_pressure_numerically_accepted=True, pressure_closure_performed=False,
                  arguments={"nlayer": 2, "temperature_k": 2000., "bottom_pressure_bar": 1.})
    audit["source"].update(kind="fixed_pressure_internal_source", pressure_closure_performed=False,
                           selection={"layers": 2, "temperature_k": 2000., "pressure_bar": 1.})
    return report, audit, source


def test_fixed_pressure_requires_opt_in_and_reuses_unchanged_water_math(monkeypatch):
    root, root_audit, _ = fixture(monkeypatch)
    expected, reference, _ = runner.saved_water_problem(root, root_audit, "source-digest")
    report, audit, _ = fixed_pressure_fixture(monkeypatch)
    original = copy.deepcopy(report)
    with pytest.raises(ValueError, match="explicit opt-in"):
        runner.saved_water_problem(report, audit, "source-digest")
    actual, point, receipt = runner.saved_water_problem(
        report, audit, "source-digest", allow_fixed_pressure=True)
    def primitive(value):
        if hasattr(value, "lo") and hasattr(value, "hi"):
            return (value.lo, value.hi)
        if isinstance(value, dict):
            return {key: primitive(item) for key, item in value.items()}
        if isinstance(value, list):
            return [primitive(item) for item in value]
        return value
    assert primitive(actual) == primitive(expected)
    assert point == reference
    assert report == original
    assert "runs" not in report and "roots" not in report
    assert receipt["source_kind"] == "fixed_pressure_internal_source"
    assert receipt["pressure_closure_performed"] is False


@pytest.mark.parametrize("change", [
    lambda r,a,s: r.update(model_id="unrecognized"),
    lambda r,a,s: r.update(failure="failed-source"),
    lambda r,a,s: r["source_state"].update(failure="failed-column"),
    lambda r,a,s: r.update(fixed_pressure_numerically_accepted=False),
    lambda r,a,s: r.update(pressure_closure_performed=True),
    lambda r,a,s: r.update(numerically_accepted=True),
    lambda r,a,s: r.update(numerically_accepted=1),
    lambda r,a,s: r.update(numerically_accepted=0),
    lambda r,a,s: r.update(numerically_accepted=None),
    lambda r,a,s: r.update(accepted=True),
    lambda r,a,s: r.update(accepted=None),
    lambda r,a,s: r["source_state"].update(accepted=None),
    lambda r,a,s: r["source_state"].update(numerically_accepted=True),
    lambda r,a,s: r["source_state"].update(accepted=1),
    lambda r,a,s: r["source_state"].update(global_closure_numerically_accepted=0),
    lambda r,a,s: r.update(runs=[]),
    lambda r,a,s: r.update(roots=[]),
    lambda r,a,s: r["source_state"].update(accepted=True),
    lambda r,a,s: r["source_state"].update(global_closure_numerically_accepted=True),
    lambda r,a,s: r["source_state"].update(pressure_closure_performed=True),
    lambda r,a,s: r["source_state"].update(contact_accepted=False),
    lambda r,a,s: r["source_state"].update(column_numerically_accepted=False),
    lambda r,a,s: a["source"]["selection"].update(root_index=0),
    lambda r,a,s: a["source"]["selection"].update(run_index=0),
    lambda r,a,s: r["arguments"].update(nlayer=3),
    lambda r,a,s: r["arguments"].update(temperature_k=1999.),
    lambda r,a,s: r["arguments"].update(bottom_pressure_bar=2.),
    lambda r,a,s: r["source_state"].update(temperature_base_k=1999.),
    lambda r,a,s: r["source_state"].update(pressure_base_pa=2e5),
    lambda r,a,s: r["source_state"].update(connection_numerically_accepted=False),
    lambda r,a,s: a["source"].update(pressure_closure_performed=True),
    lambda r,a,s: a["source"].pop("pressure_closure_performed"),
    lambda r,a,s: a["source"].update(sha256="different-source"),
    lambda r,a,s: a["source"].update(numerical_source_accepted=False),
    lambda r,a,s: s["source_internal_result"].update(component_amounts_mol=[1., 1., .2, .05]),
    lambda r,a,s: s["source_internal_result"].update(component_amounts_mol=[1., 1.]),
    lambda r,a,s: s["source_internal_result"].update(component_amounts_mol=[1., 1., float("nan"), .05]),
    lambda r,a,s: s["source_internal_record"]["component_formulas"]["a_melts"].update(Unknown=1.),
    lambda r,a,s: r["inventory"].update(total_element_amounts_mol=[2., 5.1, .4]),
])
def test_fixed_pressure_binding_rejects_promotion_or_changed_primitives(monkeypatch, change):
    report, audit, source = fixed_pressure_fixture(monkeypatch)
    change(report, audit, source)
    with pytest.raises(ValueError):
        runner.saved_water_problem(report, audit, "source-digest", allow_fixed_pressure=True)


def test_fixed_pressure_preserves_absent_original_state_closure_flag(monkeypatch):
    report, audit, _ = fixed_pressure_fixture(monkeypatch)
    del report["source_state"]["pressure_closure_performed"]
    original = copy.deepcopy(report)
    _, _, receipt = runner.saved_water_problem(
        report, audit, "source-digest", allow_fixed_pressure=True)
    assert report == original
    assert receipt["pressure_closure_performed"] is False
