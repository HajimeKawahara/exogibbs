import hashlib,json,sys,time
from pathlib import Path
from fractions import Fraction
import numpy as np

gibbs=Path('/tmp/stage2-material-proof-gibbs-20260928')
eos=Path('/tmp/stage2-combined-eos-20260928')
sys.path.insert(0,str(gibbs/'examples/metal_silicate'))
from melts_coupled import load_melts_evaluator,with_saved_water_standard_offset,COMMON_R
from m2_host_standards import source_hydrogen_standard_receipt
from m2_water_global import parameters_from_saved_water,certify_water_common_plane,water_insertion_value
from m2_common_plane import _I,_dot,ideal_energy,interval_json

path=Path('/tmp/stage2-reclosed-water-h1e24-16-20260928/trial_0000.json')
out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
trial=json.loads(path.read_text());source=trial['source'];metadata=source['source_metadata']
assert source['source_internal_result']['accepted'] and trial['state']['connection_numerically_accepted']
temperature=source['temperature_K'];pressure=source['pressure_bar']
water_receipt=metadata['host_ledger']['water_standard_receipts'][0]
assert water_receipt['T_K']==temperature and water_receipt['P_Pa']==pressure*1e5
def gas(t,p):
 assert t==temperature and p==pressure*1e5
 return water_receipt['H2O_gas_standard_RT']
evaluator=load_melts_evaluator(eos,liquid_model='published_water',
 runtime=Path('/tmp/exoeos-pr3-melts/runtime/alphamelts-py-2.3.2-ubuntu_22_04-x86_64'),
 python_executable='/tmp/exoeos-pr3-env/bin/python',gas_water_standard_rt=gas)
offset=metadata['provider_scenario']['standard_offsets_rt'].get('h2o_melts',0.)
evaluator=with_saved_water_standard_offset(evaluator,offset)
record=source['source_internal_record'];result=source['source_internal_result']
names=[name for group in record['phases'].values() for name in group]
amount=dict(zip(names,result['component_amounts_mol']))
scale=metadata['input']['native_amount_scale']
n=[float(amount.get(name+'_melts',0.)*scale) for name in evaluator.COMPONENTS]
h2=float(amount['H2_dissolved']*scale)
properties=evaluator.evaluate_liquid(temperature,pressure*1e5,n,common_R=COMMON_R)
(out/'fresh_properties.json').write_text(json.dumps(properties,indent=2,allow_nan=False)+'\n')
hydrogen=source_hydrogen_standard_receipt(source)
plane=dict(zip(record['elements'],result['elemental_potentials_rt']))
inventory=dict(zip(record['elements'],metadata['input']['element_amounts_mol']))
for e in properties['basis']['element_order']:plane.setdefault(e,0.);inventory.setdefault(e,0.)
standard=_I(hydrogen['dissolved_h2_base_standard_rt'])+_I(hydrogen['H2_dissolved_standard_offset_rt'])
p,reference,binding=parameters_from_saved_water(properties,plane,inventory,standard,water_standard_offset_rt=offset)
source_mu0=metadata['host_ledger']['native_standard_state_receipts'][0]['native_properties']['mu0_J_mol']
dry=properties['water_reconstruction']['dry_properties']
assert all(source_mu0[i]==dry['mu0_J_mol'][i] for i in binding['active_dry_component_indices'])
water=n[evaluator.COMPONENTS.index('h2o')]
q=water_insertion_value(p,reference,water,h2)
native_energy=_I(properties['gibbs_RT'])+ideal_energy([sum(map(Fraction,n)),Fraction(h2)])+_I(h2)*standard
columns=properties['basis']['component_element_matrix'];elements=properties['basis']['element_order']
plane_energy=sum((_I(v)*_dot(row,[plane[e] for e in elements]) for v,row in zip(n,columns)),_I(0))+_I(h2)*2*_I(plane['H'])
delta=q-(native_energy-plane_energy)
assert max(abs(float(delta.lo)),abs(float(delta.hi)))<1e-11
max_nodes=int(sys.argv[2]);started=time.monotonic()
assessment=certify_water_common_plane(p,reference,max_nodes=max_nodes)
assessment.update(source_kind='accepted_fixed_pressure_source_trial',pressure_root=False,
 source_file=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),binding=binding,
 dissolved_hydrogen_standard_receipt=hydrogen,scalar_replay_difference_rt_mol=interval_json(delta),
 fresh_native_dry_standards_match_source=True,elapsed_seconds=time.monotonic()-started)
(out/'assessment.json').write_text(json.dumps(assessment,indent=2,allow_nan=False)+'\n')
print(assessment['lower_bound_rt_per_dry_component'],assessment['nodes_evaluated'],assessment['bound_within_requested_tolerance'],flush=True)
