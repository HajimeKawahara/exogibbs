"""Saved-root binding rejects changed formulas, gauges and source conditions."""

import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


DIRECTORY = Path(__file__).resolve().parents[3] / "examples/metal_silicate"
sys.path.insert(0, str(DIRECTORY))
try:
    RUNNER = importlib.import_module("run_m2_alloy_insertion_bound")
finally:
    sys.path.pop(0)


def records():
    amounts = np.array([.96, .01, .01, .02])
    composition = (amounts / amounts.sum()).tolist()
    domain = {"component_order": list(RUNNER.COMPONENTS),
              "lower_atomic_fractions": [.86, 0., 0., 0.],
              "upper_atomic_fractions": [1., .08, .02, .04],
              "effective_upper_atomic_fractions": [1., .08, .02, .04],
              "selected_metal": {"composition": composition}}
    checkouts = {name: {"head": name + "-source", "status": ""} for name in ("exogibbs", "exoeos")}
    source = {"temperature_K": 2173.15, "pressure_bar": 250.,
              "source_metadata": {"provenance": {
                  name: {"commit": value["head"], "changed_tracked_file_sha256": {}}
                  for name, value in checkouts.items()}},
              "source_internal_record": {
                  "elements": ["O", "H", "Fe", "Si"],
                  "phases": {"metal": list(RUNNER.COMPONENTS)},
                  "component_formulas": {name: {element: 1.} for name, element
                                         in zip(RUNNER.COMPONENTS, RUNNER.ELEMENTS)}},
              "source_internal_result": {"accepted": True, "elemental_potentials_rt": [-10., -7., -8., 1.],
                                         "component_amounts_mol": amounts.tolist()},
              "metal_selection": {"metal_composition": composition, "metal_amount_mol": 1.,
                                  "metal_composition_domain": domain}}
    state = {"accepted": True, "global_closure_numerically_accepted": True,
             "pressure_closure_performed": True, "contact_accepted": True,
             "column_numerically_accepted": True, "temperature_base_k": 2173.15,
             "pressure_base_pa": 250e5, "layers": [{} for _ in range(16)], "source": source}
    report = {"numerically_accepted": True, "checkouts": checkouts,
              "runs": [{"roots": [state]}], "provider_scenario": None}
    audit = {"source": {"kind": "global_closure_root", "numerical_source_accepted": True,
                         "source_contact_accepted": True, "executed_checkouts": checkouts,
                         "selection": {"run_index": 0, "root_index": 0, "layers": 16,
                                       "temperature_k": 2173.15, "pressure_bar": 250.}},
             "host_stability": {"temperature_K": 2173.15, "pressure_Pa": 250e5}}
    return report, audit, state, source


def save(tmp_path, report, audit):
    closure, physical = tmp_path / "closure.json", tmp_path / "physical.json"
    closure.write_text(json.dumps(report))
    audit["source"]["sha256"] = RUNNER.sha256(closure)
    physical.write_text(json.dumps(audit))
    return closure, physical


def test_atomic_plane_is_selected_in_component_order_without_matrix_rounding(tmp_path):
    report, audit, _, _ = records()
    paths = save(tmp_path, report, audit)
    _, _, _, _, plane, composition = RUNNER.saved_inputs(*paths)
    np.testing.assert_array_equal(plane, [-8., 1., -10., -7.])
    np.testing.assert_array_equal(composition, [.96, .01, .01, .02])


@pytest.mark.parametrize("change", [
    lambda report, audit, state, source: report.update(numerically_accepted=False),
    lambda report, audit, state, source: state.update(pressure_closure_performed=False),
    lambda report, audit, state, source: audit["source"].update(kind="fixed_pressure_trial"),
    lambda report, audit, state, source: audit["source"]["selection"].update(pressure_bar=251.),
    lambda report, audit, state, source: source["source_internal_record"]["component_formulas"].update(Fe_metal={"Fe": 2.}),
    lambda report, audit, state, source: source["source_internal_result"].update(elemental_potentials_rt=[1., 2.]),
    lambda report, audit, state, source: source["source_internal_result"].update(component_amounts_mol=[1., 0., 0., 0.]),
    lambda report, audit, state, source: source["metal_selection"].update(metal_amount_mol=0.),
    lambda report, audit, state, source: source["metal_selection"]["metal_composition_domain"].update(upper_atomic_fractions=[1., .08, .03, .04]),
    lambda report, audit, state, source: source["source_metadata"]["provenance"]["exoeos"].update(changed_tracked_file_sha256={"model.py": "modified"}),
    lambda report, audit, state, source: report.update(provider_scenario={"sha256": "changed", "values": {"standard_offsets_rt": {"O_metal": 1.63}}}),
    lambda report, audit, state, source: source.update(provider_scenario_sha256="unmatched"),
])
def test_changed_root_or_mathematical_contract_cannot_inherit_a_proof(tmp_path, change):
    values = records()
    change(*values)
    with pytest.raises(ValueError):
        RUNNER.saved_inputs(*save(tmp_path, values[0], values[1]))


def test_audit_hash_must_bind_the_exact_source_bytes(tmp_path):
    report, audit, _, _ = records()
    closure, physical = save(tmp_path, report, audit)
    closure.write_text(closure.read_text() + "\n")
    with pytest.raises(ValueError, match="exact accepted"):
        RUNNER.saved_inputs(closure, physical)


def test_declared_oxygen_and_hydrogen_offsets_survive_input_binding(tmp_path):
    report, audit, _, source = records()
    values = {"standard_offsets_rt": {"O_metal": 1.63, "H_metal": -.44}}
    report["provider_scenario"] = {"values": values, "sha256": "saved-scenario"}
    source["source_metadata"]["provider_scenario"] = RUNNER.normalize_scenario(values)
    source["provider_scenario_sha256"] = "saved-scenario"
    result = RUNNER.saved_inputs(*save(tmp_path, report, audit))
    assert result[3]["standard_offsets_rt"] == {
        "O_metal": 1.63, "H_metal": -.44, "H2_dissolved": 0., "h2o_melts": 0.}
