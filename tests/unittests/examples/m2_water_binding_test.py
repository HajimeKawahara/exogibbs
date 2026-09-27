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
