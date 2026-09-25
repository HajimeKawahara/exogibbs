import hashlib,json,subprocess,sys,time
from pathlib import Path
root=Path('/tmp/exoinventory-m2-readiness-20260925')
gibbs=root/'gibbs-phases';eos=root/'eos-integration'
sys.path.insert(0,str(gibbs/'examples/metal_silicate'))
from m2_stability_search import search_liquid_splitting
from melts_coupled import load_melts_evaluator
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
source=root/'physical_reactions_h1e24.json'
output=root/'native_two_liquid_search_smoke.json'
if output.exists():raise RuntimeError('Preserve the existing receipt.')
a=json.loads(source.read_text())['host_stability'];p=a['provider_properties'];h=a['native_dissolved_h2_moles']
provider=load_melts_evaluator(eos)
assert sha(Path(provider.__file__))==p['provenance']['evaluator_sha256']
pins={name:subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip() for name,path in [('exogibbs',gibbs),('exoeos',eos)]}
assert pins['exogibbs']=='6329226'+subprocess.check_output(['git','rev-parse','HEAD'],cwd=gibbs,text=True).strip()[7:]
start=time.monotonic()
r=search_liquid_splitting(p,h,evaluator=provider,runtime=Path('/tmp/exoeos-pr3-melts/runtime/alphamelts-py-2.3.2-ubuntu_22_04-x86_64'),python_executable='/tmp/exoeos-pr3-env/bin/python',max_evaluations=30)
report={'scope':'Bounded two-liquid negative-witness search at one unchanged saved host; no global minimum or planetary reclosure claim.',
 'input':{'physical_assessment':str(source),'sha256':sha(source),'temperature_K':p['T_K'],'pressure_Pa':p['P_Pa'],'native_component_moles':p['component_moles'],'native_dissolved_h2_moles':h},
 'provenance':{'commits':pins,'evaluator_sha256':sha(Path(provider.__file__)),'runtime':provider.REFERENCE['backend'],'orchestration_sha256':sha(Path(__file__))},
 'elapsed_s':time.monotonic()-start,'result':r}
with output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps({'status':r['status'],'trials':len(r['trials']),'failed':len(r['failed_evaluations']),'best':None if r['best_fresh_trial'] is None else r['best_fresh_trial']['objective_rt'],'budget_exhausted':r['budget_exhausted'],'elapsed_s':report['elapsed_s']},allow_nan=False),flush=True)
