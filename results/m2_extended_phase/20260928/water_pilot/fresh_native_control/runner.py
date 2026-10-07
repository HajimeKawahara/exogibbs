import datetime,hashlib,importlib.util,json,subprocess,sys,time
from pathlib import Path
root=Path('/tmp/stage2-helium-common-plane-gibbs-20260928')
eos=Path('/tmp/stage2-helium-boundary-eos-20260928')
sys.path.insert(0,str(root/'examples/metal_silicate'))
from melts_coupled import load_melts_evaluator,with_saved_water_standard_offset,COMMON_R
from run_m2_solid_global import require_fresh_host
from m2_solid_global import certify_solid_insertion
request_path=Path('/tmp/stage2-water-fixed-pressure-pilot-20260928/results/fresh_properties.json')
source_path=Path('/tmp/stage2-reclosed-water-h1e24-16-20260928/trial_0000.json')
output=Path('/tmp/stage2-water-solid-native-check-20260928')
output.mkdir(exist_ok=False)
request=json.loads(request_path.read_text());source=json.loads(source_path.read_text())['state']['source']
t,p=request['T_K'],request['P_Pa'];started=time.monotonic()
startutc=datetime.datetime.now(datetime.timezone.utc).isoformat()
def gas(a,b):
    assert a==t and b==p
    return request['water_reconstruction']['gas_H2O_standard_RT']
kwargs={'runtime':Path('/tmp/exoeos-pr3-melts/runtime/alphamelts-py-2.3.2-ubuntu_22_04-x86_64'),
        'python_executable':'/tmp/exoeos-pr3-env/bin/python','common_R':COMMON_R}
evaluator=load_melts_evaluator(eos,liquid_model='published_water',gas_water_standard_rt=gas,
                             runtime=kwargs['runtime'],python_executable=kwargs['python_executable'])
evaluator=with_saved_water_standard_offset(evaluator,request['water_reconstruction'].get('standard_offset_rt',0.))
fresh=evaluator.evaluate_liquid(t,p,request['component_moles'],**kwargs)
require_fresh_host(request,fresh)
native=load_melts_evaluator(eos).evaluate_liquid(t,p,request['component_moles'],candidate_standard_states=['alloy-solid'],**kwargs)
path=eos/'examples/melts_solid_mixing.py'
spec=importlib.util.spec_from_file_location('water_solid_native_provider',path)
provider=importlib.util.module_from_spec(spec);spec.loader.exec_module(provider)
parameters=provider.solid_mixing_parameters('alloy-solid',t,p)
record=source['source_internal_record'];result=source['source_internal_result']
names=[name for phase in record['phases'].values() for name in phase]
h2=result['component_amounts_mol'][names.index('H2_dissolved')]*source['source_metadata']['input']['native_amount_scale']
proof=certify_solid_insertion(parameters,native['candidate_standard_states'][0],fresh,h2)
for name,value in [('fresh_properties.json',fresh),('native_standard_state_receipt.json',native),('alloy-solid.json',proof)]:
    with (output/name).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
receipt={'kind':'fixed_pressure_pilot_water_reference_replay_and_one_native_candidate_control',
         'new_source_solve':False,'pressure_root':False,'started_utc':startutc,
         'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started,
         'proof_head':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
         'proof_status':subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip(),
         'provider_head':subprocess.check_output(['git','-C',str(eos),'rev-parse','HEAD'],text=True).strip(),
         'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
         'original_properties_sha256':hashlib.sha256(request_path.read_bytes()).hexdigest(),
         'fresh_expression_basis_G_mu_amounts_equal':True,
         'one_candidate_global_lower_rt':proof['lower_bound_rt_per_formula_unit'],
         'files':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()}}
(output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
