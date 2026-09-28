import hashlib,json,sys,time
from pathlib import Path
from decimal import Decimal
from fractions import Fraction
sys.path.insert(0,'/tmp/stage2-water-curvature-gibbs-20260928/examples/metal_silicate')
from m2_water_global import _I,certify_water_common_plane
base=Path('/tmp/stage2-water-fixed-pressure-pilot-20260928/results')
proof=json.loads((base/'assessment.json').read_text());properties=json.loads((base/'fresh_properties.json').read_text())
def convert(value):
 if isinstance(value,dict) and set(value)=={'lower','upper'}:return _I(Decimal(value['lower']),Decimal(value['upper']))
 if isinstance(value,list):return list(map(convert,value))
 return value
parameters={k:convert(v) for k,v in proof['parameters'].items()}
amounts=[properties['component_moles'][i] for i in proof['binding']['active_dry_component_indices']]
files=[base/'assessment.json',base/'fresh_properties.json',Path(proof['source_file'])]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
assert hashes[proof['source_file']]==proof['source_sha256']
started=time.monotonic()
result=certify_water_common_plane(parameters,amounts,max_nodes=int(sys.argv[2]),
 progress_callback=lambda count,best:print(json.dumps({'nodes_evaluated':count,'best_value_upper_rt':best['value_upper_rt'],'global_lower_bound_rt':best['global_lower_bound_rt'],'elapsed_seconds':time.monotonic()-started}),flush=True))
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in hashes.items())
result.update(source_kind='accepted_fixed_pressure_source_trial',pressure_root=False,
 binding=proof['binding'],source_input_sha256=hashes,independent_scalar_check=proof['scalar_replay_difference_rt_mol'],
 elapsed_seconds=time.monotonic()-started)
with Path(sys.argv[1]).open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
print(result['lower_bound_rt_per_dry_component'],result['nodes_evaluated'],result['bound_within_requested_tolerance'],flush=True)
