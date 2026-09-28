"""Saved-root binding rejects changed formulas, gauges and source conditions."""

import importlib
import hashlib
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
    assert result[3]["standard_offsets_rt"] == {"O_metal": 1.63, "H_metal": -.44, "H2_dissolved": 0., "h2o_melts": 0.}


def test_extended_source_requires_its_explicit_basis_and_original_domain(tmp_path):
    report,audit,_,source=records()
    record=source['source_internal_record']
    record['elements'].append('P')
    record['phases']['metal'].append('P_metal')
    record['component_formulas']['P_metal']={'P':1.}
    result=source['source_internal_result']
    result['elemental_potentials_rt'].append(-2.)
    result['component_amounts_mol'].append(.001)
    composition=(np.asarray(result['component_amounts_mol'])/sum(result['component_amounts_mol'])).tolist()
    selection=source['metal_selection']
    selection['metal_composition']=composition
    selection['metal_amount_mol']=sum(result['component_amounts_mol'])
    domain=selection['metal_composition_domain']
    domain['component_order'].append('P_metal')
    domain['selected_metal']['composition']=composition
    domain['lower_atomic_fractions'].append(0.)
    domain['upper_atomic_fractions'].append(.02)
    domain['effective_upper_atomic_fractions'].append(.02)
    source['source_metadata'].update(metal_model='phosphorus',phosphorus_metal={
        'component_order':['Fe','Si','O','H','P'],
        'lower_atomic_fractions':domain['lower_atomic_fractions'][:],
        'upper_atomic_fractions':domain['upper_atomic_fractions'][:]})
    paths=save(tmp_path,report,audit)
    with pytest.raises(ValueError,match='explicitly selected'):
        RUNNER.saved_inputs(*paths)
    np.testing.assert_array_equal(RUNNER.saved_inputs(*paths,allow_extended=True)[4],[-10.,-7.,-8.,1.,-2.])
    domain['lower_atomic_fractions'][0]=.84
    with pytest.raises(ValueError,match='domain must agree'):
        RUNNER.saved_inputs(*save(tmp_path,report,audit),allow_extended=True)


def historical_recipe(tmp_path,monkeypatch):
    here=tmp_path/'gibbs/examples/metal_silicate'
    here.mkdir(parents=True)
    eos=tmp_path/'eos'
    original={name:(name+' original\n').encode() for name in
              ['run_bse_common_gibbs.py','source.py','reference.json','m2_common_gas.py',
               'm2_scenarios.py','phase_selection.py','melts_coupled.py','m2_expanded_source.py']}
    blobs={}
    for name,raw in original.items():
        (here/name).write_bytes(raw)
        blobs['gibbs-source:examples/metal_silicate/'+name]=raw
    for name in ['ma_fe_si_o.py','ma_fe_si_o_h.py','_arrays.py']:
        path=eos/'src/exoeos'/name;path.parent.mkdir(parents=True,exist_ok=True)
        raw=(name+' model\n').encode();path.write_bytes(raw)
        blobs['eos-source:src/exoeos/'+name]=raw
    monkeypatch.setattr(RUNNER,'HERE',here)
    monkeypatch.setattr(RUNNER.subprocess,'check_output',lambda command,**kwargs:blobs[command[-1]])
    provenance={'file_sha256':{name:hashlib.sha256(raw).hexdigest() for name,raw in original.items()},
                'exogibbs':{'commit':'gibbs-source'},'exoeos':{'commit':'eos-source'},
                'jax_version':RUNNER.jax.__version__,'numpy_version':np.__version__}
    return {'source_metadata':{'provenance':provenance}},eos,here,blobs


@pytest.mark.parametrize('name',['melts_coupled.py','m2_expanded_source.py','m2_scenarios.py','phase_selection.py'])
def test_historical_builder_requires_its_original_blob_and_explicit_opt_in(tmp_path,monkeypatch,name):
    source,eos,here,_=historical_recipe(tmp_path,monkeypatch)
    (here/name).write_text('new construction or verification control flow\n')
    with pytest.raises(ValueError,match='source recipe has changed'):
        RUNNER.verified_recipe(source,eos)
    receipt={}
    files=RUNNER.verified_recipe(source,eos,allow_historical_builders=True,receipt=receipt)
    assert receipt['changed_nonreplayed_builder_files']==[name]
    assert receipt['executed_git_blob_sha256'][name]==source['source_metadata']['provenance']['file_sha256'][name]
    assert receipt['proof_checkout_file_sha256'][name]==files[str(here/name)]
    assert receipt['proof_checkout_file_sha256'][name]!=receipt['executed_git_blob_sha256'][name]


@pytest.mark.parametrize('change',['thermochemical_file','git_blob','saved_hash','provider_model'])
def test_historical_policy_cannot_hide_changed_standards_or_fabricated_provenance(tmp_path,monkeypatch,change):
    source,eos,here,blobs=historical_recipe(tmp_path,monkeypatch)
    if change=='thermochemical_file':(here/'source.py').write_text('different thermochemistry')
    if change=='git_blob':blobs['gibbs-source:examples/metal_silicate/phase_selection.py']=b'different executed blob'
    if change=='saved_hash':source['source_metadata']['provenance']['file_sha256']['phase_selection.py']='0'*64
    if change=='provider_model':(eos/'src/exoeos/ma_fe_si_o.py').write_text('different model')
    with pytest.raises(ValueError):
        RUNNER.verified_recipe(source,eos,allow_historical_builders=True)
