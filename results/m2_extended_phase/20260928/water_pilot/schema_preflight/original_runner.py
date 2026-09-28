"""Read-only numerical/schema checks of an unclosed 250 bar water pilot.

Any root-shaped object below is a TEST FIXTURE ONLY. The real trial remains
unclosed; neither this program nor its output certifies a pressure root.
"""
import copy, datetime, hashlib, importlib.util, json, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
root=Path('/tmp/stage2-helium-common-plane-gibbs-20260928')
eos=Path('/tmp/stage2-helium-boundary-eos-20260928')
sys.path.insert(0,str(root/'examples/metal_silicate'))
from run_m2_water_global import saved_water_problem, saved_scenario_values
from run_m2_alloy_insertion_bound import saved_inputs, verified_recipe
from m2_host_standards import source_hydrogen_standard_receipt
from m2_finite_gas import atmosphere_gauge_rt, build_atmosphere_setup
from run_bse_common_gibbs import source_standards_rt
from m2_common_plane import solution_common_plane
from m2_solid_global import certify_solid_insertion
from melts_coupled import load_melts_evaluator, COMMON_R
from run_m2_solid_global import require_fresh_host
trial_path=Path('/tmp/stage2-reclosed-water-h1e24-16-20260928/trial_0000.json')
report_path=trial_path.with_name('closure.json')
properties_path=Path('/tmp/stage2-water-solid-native-check-20260928/fresh_properties.json')
old_proof_path=Path('/tmp/stage2-water-fixed-pressure-curvature-global-20260928/assessment.json')
output=Path('/tmp/stage2-water-actual-schema-preflight-20260928.json')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inputs={str(p):digest(p) for p in [trial_path,report_path,properties_path,old_proof_path]}
started=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
trial=json.loads(trial_path.read_text());report=json.loads(report_path.read_text())
state=trial['state'];source=state['source'];properties=json.loads(properties_path.read_text())
assert state['pressure_closure_performed'] is False
assert source['source_internal_result']['accepted'] and state['contact_accepted']
scenario=saved_scenario_values(report['provider_scenario'],source)
hydrogen=source_hydrogen_standard_receipt(source)
recipe={};verified_recipe(source,eos,allow_historical_builders=True,receipt=recipe)
record=source['source_internal_record'];result=source['source_internal_result'];metadata=source['source_metadata']
names=[name for phase in record['phases'].values() for name in phase]
h2=result['component_amounts_mol'][names.index('H2_dissolved')]*metadata['input']['native_amount_scale']
selector={'run_index':0,'root_index':0,'layers':len(state['layers']),'temperature_k':source['temperature_K'],'pressure_bar':source['pressure_bar']}
audit={'schema_fixture_only':True,'dissolved_hydrogen_standard_receipt':hydrogen,
       'source':{'kind':'global_closure_root','sha256':'schema-fixture-only','numerical_source_accepted':True,
                 'source_contact_accepted':True,'executed_checkouts':report['checkouts'],'selection':selector},
       'host_stability':{'temperature_K':source['temperature_K'],'pressure_Pa':properties['P_Pa'],
                         'provider_properties':properties,'native_amount_scale':metadata['input']['native_amount_scale'],
                         'native_host_component_amounts_mol':properties['component_moles'],'native_dissolved_h2_moles':h2}}
fixture=copy.deepcopy(report);fixture['schema_fixture_only']=True
fixture_state=copy.deepcopy(state);fixture_state.update(accepted=True,global_closure_numerically_accepted=True)
fixture['runs']=[{'roots':[fixture_state]}];fixture['numerically_accepted']=True
# First preserve the actual false pressure gate and verify rejection.
try:saved_water_problem(fixture,audit,'schema-fixture-only',exoeos_checkout=eos)
except ValueError as exc:
    assert 'unaccepted pressure trial' in str(exc);gate_error=str(exc)
else:raise AssertionError('An unclosed actual trial passed the final-root gate.')
# This single boolean change is ONLY an in-memory unit fixture, not evidence.
fixture_state['pressure_closure_performed']=True
parameters,reference,binding=saved_water_problem(fixture,audit,'schema-fixture-only',exoeos_checkout=eos)
def serial(value):
    if hasattr(value,'lo') and hasattr(value,'hi'):return {'lower':str(value.lo),'upper':str(value.hi)}
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value
oldproof=json.loads(old_proof_path.read_text())
assert serial(parameters)==oldproof['parameters']
with tempfile.TemporaryDirectory(prefix='stage2-water-schema-fixture-only-') as temporary:
    fp=Path(temporary)/'NOT_A_ROOT_schema_fixture.json';ap=Path(temporary)/'NOT_A_PHYSICAL_AUDIT_schema_fixture.json'
    fp.write_text(json.dumps(fixture));audit['source']['sha256']=digest(fp);ap.write_text(json.dumps(audit))
    saved_inputs(fp,ap,allow_extended=True)
reference_standards,_=source_standards_rt(source['temperature_K'],source['pressure_bar'])
setup=build_atmosphere_setup(source['gas_model']);parcel=source['source_atmosphere_parcel']
gauge=atmosphere_gauge_rt(setup,source['temperature_K'],reference_standards)
for phase,label in [(setup.gas_setup,'gas'),(setup.condensate_setup,'condensate')]:
    actual=np.asarray(phase.hvector_func(source['temperature_K']))+np.asarray(phase.formula_matrix).T@gauge
    assert list(phase.species)==parcel[label+'_species']
    assert np.array_equal(actual,parcel[label+'_standard_potentials_rt'])
assert np.array_equal(gauge,parcel['element_gauge_rt'])
path=eos/'examples/melts_solid_mixing.py';spec=importlib.util.spec_from_file_location('water_preflight_solid_provider',path)
provider=importlib.util.module_from_spec(spec);spec.loader.exec_module(provider)
models=json.loads(provider.PARAMETER_PATH.read_text())['models']
native=load_melts_evaluator(eos).evaluate_liquid(properties['T_K'],properties['P_Pa'],properties['component_moles'],
    candidate_standard_states=list(models),runtime=Path('/tmp/exoeos-pr3-melts/runtime/alphamelts-py-2.3.2-ubuntu_22_04-x86_64'),
    python_executable='/tmp/exoeos-pr3-env/bin/python',common_R=COMMON_R)
plane=dict(zip(record['elements'],result['elemental_potentials_rt']))
for element in properties['basis']['element_order']:plane.setdefault(element,0.)
rows=[]
for standards in native['candidate_standard_states']:
    phase=standards['phase'];declared=provider.solid_mixing_parameters(phase,properties['T_K'],properties['P_Pa'])
    proof=certify_solid_insertion(declared,standards,properties,h2,max_nodes=1)
    lower,atoms,details=solution_common_plane(proof,properties,h2,plane)
    rows.append({'phase':phase,'single_box_schema_checked':True,'unresolved_boxes':proof['unresolved_box_count']})
assert len(rows)==20 and set(row['phase'] for row in rows)==set(models)
assert all(digest(Path(p))==sha for p,sha in inputs.items())
receipt={'kind':'unit_schema_preflight_from_unclosed_fixed_pressure_pilot','scientific_root_certificate':False,
         'new_equilibrium_solve':False,'original_pressure_closure_performed':state['pressure_closure_performed'],
         'original_pressure_log_residual':state['pressure_log_residual'],'actual_final_root_gate_rejection':gate_error,
         'root_shaped_objects_are_test_fixtures_only':True,'temporary_fixture_files_removed':True,
         'source_scalar_parameters_equal_prior_pilot_proof':True,'unchanged_gas_standards_replayed_exactly':True,
         'water_and_legacy_alloy_root_schemas_checked':True,'twenty_solution_common_plane_schemas':rows,
         'no_twenty_phase_stability_claim':True,'recipe_binding':recipe,'original_input_sha256':inputs,
         'proof_head':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
         'provider_head':subprocess.check_output(['git','-C',str(eos),'rev-parse','HEAD'],text=True).strip(),
         'started_utc':utc,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started}
with output.open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
print(json.dumps(receipt),flush=True)
