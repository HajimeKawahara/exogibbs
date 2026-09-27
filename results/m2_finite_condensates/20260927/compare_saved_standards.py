"""Compare preserved native endpoint energies with eligible FastChem phases.

This performs no native evaluations and asserts no calibrated uncertainty.
"""

from pathlib import Path
import argparse
import sys
import importlib.util
import hashlib
import json
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/metal_silicate"))
from m2_finite_gas import build_atmosphere_setup, atmosphere_gauge_rt
from run_bse_common_gibbs import source_standards_rt
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--native-archive", type=Path, required=True)
parser.add_argument("--exoeos-evaluator", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
base = args.native_archive
host_path = base / "fresh_host_assessment.json"
host = json.loads(host_path.read_text())
T = host["assessment"]["temperature_K"]
P = host["assessment"]["pressure_Pa"] / 1e5
evaluator_path = args.exoeos_evaluator
spec=importlib.util.spec_from_file_location('native_evaluator',evaluator_path)
evaluator=importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)
s=build_atmosphere_setup('janaf_condensed')
table=Path(__file__).resolve().parents[3] / "src/exogibbs/data/FastChem4/logK/logK_condensates.dat"
lines=table.read_text().splitlines()
reference,_=source_standards_rt(T,P)
q=atmosphere_gauge_rt(s,T,reference)
a=np.asarray(s.condensate_setup.formula_matrix)
h=np.asarray(s.condensate_setup.hvector_func(T))+a.T@q
native_elements=list(evaluator.ELEMENTS)
rows=[native_elements.index(e) if e in native_elements else None for e in s.elements]
results=[]; files={host_path.name:hashlib.sha256(host_path.read_bytes()).hexdigest()}
for path in sorted(base.glob('phase_*.json')):
 doc=json.loads(path.read_text()); files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
 for phase in doc['phases']:
  seen=set()
  for trial in phase['trials']:
   x=np.asarray(trial['coordinates'])
   if np.count_nonzero(x)!=1 or x.sum()!=1.: continue
   key=tuple(x)
   if key in seen:continue
   seen.add(key)
   provider=trial.get('provider_candidate',{})
   if provider.get('gibbs_J') is None:continue
   oxide=np.asarray(provider['oxide_mass_g'])/np.asarray(evaluator.OXIDE_MASSES)
   full_atoms=np.asarray(evaluator.OXIDE_ELEMENTS).T@oxide
   atoms=np.array([full_atoms[i] if i is not None else 0. for i in rows])
   excluded=[i for i,e in enumerate(native_elements) if e not in s.elements]
   if np.any(np.abs(full_atoms[excluded])>1e-10):continue
   for j,name in enumerate(s.condensate_species):
    formula=a[:,j]
    scale=atoms.sum()/formula.sum()
    if np.max(np.abs(atoms-scale*formula))>1e-9*atoms.sum():continue
    rt=float(provider['gibbs_J'])/(8.31446261815324*T)/scale
    assert np.isclose(rt, trial['chemical_trial']['candidate_gibbs_rt']/scale, rtol=1e-12, atol=1e-12)
    assert np.allclose(full_atoms,trial['chemical_trial']['element_amounts_mol'],rtol=1e-9,atol=1e-10)
    if T>s.condensate_setup.temperature_validity_upper[j]: continue
    idx=next(i for i,line in enumerate(lines) if line.startswith(name+' '))
    header=lines[idx]
    nextlines=[line.strip() for line in lines[idx+1:] if line.strip() and not line.lstrip().startswith('#')]
    phase_code=nextlines[0]; limits=list(map(float,nextlines[1].split()))
    segment=next(k for k,bound in enumerate(limits) if T<=bound)
    phase_branch=phase_code[segment] if len(phase_code)==len(limits) else 'unspecified'
    results.append({'fastchem_header':header,'fastchem_phase_code':phase_code,'fastchem_selected_branch':phase_branch,'fastchem_segment_upper_temperatures_K':limits,'fastchem_selected_segment':segment,'comparison_kind':'equal_stoichiometry_different_constitutive_models','phase_identity_verified':False,'native_phase':phase['phase'],'native_coordinates':x.tolist(),'fastchem_condensate':name,'eligible_at_T':bool(T<=s.condensate_setup.temperature_validity_upper[j]),'native_gibbs_rt_per_formula':rt,'fastchem_gibbs_rt_per_formula':float(h[j]),'fastchem_minus_native_rt_per_formula':float(h[j]-rt),'fastchem_minus_native_rt_per_atom':float((h[j]-rt)/formula.sum()),'native_source_file':path.name,'candidate_formula_scale':float(scale)})
report={'assessment_id':'m2_saved_native_fastchem_condensate_reference_comparison_v1','temperature_K':T,'pressure_bar':P,'scope':'New standard comparison using preserved native candidate evaluations, not new native runs or calibrated phase boundaries. Native unit endmember compositions include any internal ordering. FastChem pure condensates have no pressure correction. Formula equality does not establish model equivalence.','source_archive':'https://github.com/HajimeKawahara/exogibbs/tree/c32a45ff3f2152c73d2269bb3566b8f4ae63cc7c/results/m2_stability_search/20260927/explicit_coordinates','source_file_sha256':files,'evaluator_sha256':hashlib.sha256(evaluator_path.read_bytes()).hexdigest(),'fastchem_table_sha256':hashlib.sha256(table.read_bytes()).hexdigest(),'source_gauge_rt':q.tolist(),'comparisons':results}
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({"comparisons": len(results), "temperature_K": T, "pressure_bar": P, "native_evaluations": 0}))
