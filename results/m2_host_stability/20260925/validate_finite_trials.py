import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path('/tmp/exoinventory-m2-material-stability-20260925');sys.path.insert(0,str(ROOT/'exoeos/examples'))
import melts_liquid_evaluator as e
runtime=Path('/tmp/exoeos-pr3-melts/runtime/alphamelts-py-2.3.2-ubuntu_22_04-x86_64')
receipts=[]
for path in sorted((ROOT/'stability_probe/native').glob('*.json')):
 r=json.loads(path.read_text());a=r['assessment'];p=a['provider_properties'];n=np.asarray(p['component_moles']);h=a['native_dissolved_h2_moles'];row=next(x for x in a['candidates'] if x['phase']=='olivine');c=np.asarray(row['host_component_coefficients_mol']);atoms=np.sum(row['element_amounts_mol']);rt=e.COMMON_R*p['T_K']
 def mixing(nhost):
  return nhost * np.log(nhost/(nhost+h)) + (h*np.log(h/(nhost+h)) if h else 0.)
 baseline=p['gibbs_J']/rt+mixing(n.sum()); checks=[]
 for fraction in [1e-4,5e-5]:
  step=fraction*row['maximum_feasible_trial_scale'];after=n-step*c
  assert np.all(after>=0)
  prop=e.evaluate_liquid(p['T_K'],p['P_Pa'],after,runtime=runtime,python_executable='/tmp/exoeos-pr3-env/bin/python')
  delta=prop['gibbs_J']/rt+mixing(after.sum())+step*row['candidate_gibbs_rt']-baseline
  slope=delta/(step*atoms)
  checks.append({'fraction_of_maximum_feasible_step':fraction,'candidate_scale':step,'full_gibbs_difference_rt':float(delta),'one_sided_rt_per_mol_atoms':float(slope),'analytic_rt_per_mol_atoms':row['insertion_rt_per_mol_atoms'],'error_rt_per_mol_atoms':float(slope-row['insertion_rt_per_mol_atoms'])})
 assert abs(checks[-1]['error_rt_per_mol_atoms'])<2e-5,checks
 assert checks[-1]['full_gibbs_difference_rt'] * row['insertion_rt_per_mol_atoms']>0
 receipts.append({'assessment_file':path.name,'assessment_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'phase':'olivine','checks':checks})
 print(path.name,checks[-1]['full_gibbs_difference_rt'],checks[-1]['error_rt_per_mol_atoms'])
output={'scope':'Fresh one-sided full-energy differences for the stored native incipient olivine trial, using the same formal ideal-H2 augmented scalar. No new finite-source equilibrium or global stability certificate.','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks':receipts}
(ROOT/'stability_probe/finite_trials.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
