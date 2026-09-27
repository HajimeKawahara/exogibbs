import importlib.util,json,sys
from pathlib import Path
root=Path('/tmp/exoinventory-m2-readiness-20260925')
sys.path.insert(0,str(root/'gibbs-phases/examples/metal_silicate'))
from m2_stability_search import search_competing_solutions
from melts_coupled import load_melts_evaluator
p=json.loads((root/'candidate_basis_native_check.json').read_text())
a=Path('/tmp/exoinventory-m2-series-preflight-20260925/exoinventory/examples/subneptune_taxonomy/finite_melt/validation/20260925_m2_hydrogen_pilot/native_h_pilot/case_002/physical_root_000.json')
h=json.loads(a.read_text())['host_stability']['native_dissolved_h2_moles']
r=search_competing_solutions(p,h,evaluator=load_melts_evaluator(root/'eos-phases'),runtime=Path('/tmp/exoeos-pr3-melts/runtime/alphamelts-py-2.3.2-ubuntu_22_04-x86_64'),python_executable='/tmp/exoeos-pr3-env/bin/python',phases=['hornblende','orthoamphibole','kalsilite'],max_evaluations=8)
(root/'native_candidate_search_smoke.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
for row in r['phases']:
 b=row['best_fresh_trial'];print(row['phase'],row['status'],None if b is None else b['objective_rt'],len(row['trials']),len(row['failed_evaluations']),flush=True)
