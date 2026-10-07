import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path('/tmp/exoinventory-m2-material-stability-20260925');output=ROOT/'stability_probe/native';output.mkdir(exist_ok=True)
python='/home/kawahara/anaconda3/envs/myenv39/bin/python'
command=[python,str(ROOT/'exogibbs/examples/metal_silicate/run_m2_host_stability.py'),'--exoeos-checkout',str(ROOT/'exoeos'),'--runtime','/tmp/exoeos-pr3-melts/runtime/alphamelts-py-2.3.2-ubuntu_22_04-x86_64','--python','/tmp/exoeos-pr3-env/bin/python']
inputs=list((ROOT/'stability_probe/dry_controls').glob('*.json'))+[ROOT/'exogibbs/results/m2_expanded_contact/20260924/contact.json',Path('/tmp/exoinventory-m2-expanded-contact-20260924/exoinventory/examples/subneptune_taxonomy/finite_melt/validation/20260924_m2/metal_select.json')]
for path in inputs:
 target=output/(path.stem+'.json')
 subprocess.run(command+['--source-json',str(path),'--output',str(target)],check=True,env=dict(os.environ,PYTHONPATH=str(ROOT/'exogibbs/src'),JAX_ENABLE_X64='1',OPENBLAS_NUM_THREADS='1'))
 a=json.loads(target.read_text())['assessment']
 minerals=[r for r in a['candidates'] if r['role']=='competing_mineral' and r['status']!='unresolved']
 print(path.name,a['status'],a['negative_trial_phases'],len(a['unresolved_trial_phases']),min((r['insertion_rt_per_mol_atoms'],r['phase']) for r in minerals))
