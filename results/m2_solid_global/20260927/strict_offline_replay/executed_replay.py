"""Recheck fixed native receipts with strict final proof code; no native calls."""
import hashlib, importlib.util, json, subprocess, sys, time
from pathlib import Path
gibbs=Path('/tmp/stage2-solid-bounds-20260927')
eos=Path('/tmp/stage2-solid-eos-20260927')
original=Path('/tmp/stage2-solid-published-m1-frozen')
out=Path('/tmp/stage2-solid-final-offline-replay')
out.mkdir(exist_ok=False)
sys.path.insert(0,str(gibbs/'examples/metal_silicate'))
from m2_solid_global import certify_solid_insertion
from m2_liquid_global import assess_liquid_global_tangent_plane, require_saved_liquid_expression
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path); value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
solid=module('saved_solid_provider',eos/'examples/melts_solid_mixing.py')
liquid=module('saved_liquid_provider',eos/'examples/melts_liquid_mixing.py')
def read(path):return json.loads(path.read_text())
def write(name,value):(out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
inputs=[original/'source_physical_audit.json',original/'host_properties.json',original/'native_standard_state_receipt.json']
source,host,native=map(read,inputs)
require_saved_liquid_expression(source['host_stability']['provider_properties'],host)
protocol={'gibbs_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=gibbs,text=True).strip(),'exoeos_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=eos,text=True).strip(),'command':sys.argv,'fresh_native_calls':False,'new_pressure_root':False,'input_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
write('protocol.json',protocol)
h2=source['host_stability']['native_dissolved_h2_moles'];start=time.time();rows=[]
for standards in native['candidate_standard_states']:
 phase=standards['phase'];params=solid.solid_mixing_parameters(phase,host['T_K'],host['P_Pa'])
 result=certify_solid_insertion(params,standards,host,h2,max_nodes=100000)
 write(phase+'.json',result)
 row={key:result[key] for key in ('phase','formal_global_insertion_bound_accepted','bound_within_requested_tolerance','lower_bound_rt_per_formula_unit','node_count')}
 rows.append(row);print(json.dumps(row),flush=True)
result=assess_liquid_global_tangent_plane(host,h2,mixing_model=liquid,max_nodes=20000)
write('liquid.json',result)
summary={**protocol,'completed':True,'elapsed_seconds':time.time()-start,'phases':rows,'liquid_formal_accepted':result['formal_two_liquid_bound_accepted'],'liquid_lower_bound_rt':result['lower_bound_rt']}
write('summary.json',summary);print(json.dumps(summary),flush=True)
