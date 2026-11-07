"""Preserve the inherited thermal fit while evaluating a measured dimer D0."""

import argparse
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

import k2_source_review as source
from pending_candidate import R, reaction_properties, validate_definition

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"tests/data/validation/k2_candidate/20261106"
SOURCE=source.DATA
NAME,SPECIES=source.NAME,source.SPECIES
WAVENUMBER_J_MOL=6.62607015e-34*29979245800*6.02214076e23


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    """Read the immutable physical-energy definition."""
    return json.loads((DATA/'candidate.json').read_text())


def build() -> dict:
    """Use source physical D0, never a fit intercept or a well depth De."""
    old=json.loads((SOURCE/'literature.json').read_text());latest=json.loads((DATA/'sources.json').read_text());spect=old['spectroscopy']
    jd0=source.source_comparison(1500.)['native_janaf_D0_J_mol'];d0=spect['D0_cm_inverse']*WAVENUMBER_J_MOL
    lines=(ROOT/'src/exogibbs/data/FastChem4/logK/logK.dat').read_text().splitlines();i=next(i for i,line in enumerate(lines) if line.startswith(SPECIES+' '));coeff=list(map(float,lines[i+1].split()))
    spec={'kind':'retained_energy','temperature_range_K':[1500,4000],'coefficients':coeff,'delta_GH_J_mol':jd0-d0}
    control={**spec,'delta_GH_J_mol':jd0-latest['selected_D0_cm_inverse']*WAVENUMBER_J_MOL}
    files=[Path(__file__),DATA/'sources.json',ROOT/'mceg/reviews/pending_candidate.py',ROOT/'mceg/reviews/k2_source_review.py',ROOT/'src/exogibbs/data/FastChem4/logK/logK.dat']+[p for p in SOURCE.iterdir() if p.is_file()]
    if (DATA/'ground_vibrational_levels.npz').exists():files.append(DATA/'ground_vibrational_levels.npz')
    return {'id':'k2_measured_D0_retained_thermal_20261106','molecule':NAME,'species':SPECIES,'baseline_dataset':'curated-ch2-v1','temperature_range_K':[1500,4000],'tracked_elements':list(source.inputs()['composition']),'primary_variant':'measured_D0_retained_thermal','variants':[{'name':'measured_D0_retained_thermal','replacements':{SPECIES:spec}}],'selected_variants':[{'name':latest['selected_control_name'],'replacements':{SPECIES:control}}],'physical_energy':{'retained_JANAF_D0_J_mol':jd0,'measured_D0_J_mol':d0,'delta_D0_J_mol':d0-jd0,'reported_D0_uncertainty_J_mol':spect['reported_uncertainty_cm_inverse']*WAVENUMBER_J_MOL,'formal_fit_intercept_Ra_J_mol':R*coeff[0],'measurement':spect,'ZPE_added_J_mol':0.},'source_sha256':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},'residual_scope':'A constant physical D0 correction retains the entire inherited thermal fit, atomic-reference gauge and fit residual. S/Cp are exactly unchanged; no arbitrary 298K heat or ZPE is added.','thermal_scope':'No thermal adoption: BC2016/2022 normalization,2025 corrected-figure and official-supplement scope, and available modern-state completeness are separately audited.','curated_update':'none'}


def validate() -> dict:
    """Check energy identity, dense derivatives, and source-state inventory."""
    model=load_candidate();assert model==build();spec=model['variants'][0]['replacements'][SPECIES];old={**spec,'delta_GH_J_mol':0.};samples=[]
    for t in (1500.,2000.,2500.,3000.,4000.):
        before,after=reaction_properties(t,old),reaction_properties(t,spec)
        difference={k:after[k]-before[k] for k in before}
        assert abs(difference['G_J_mol']-spec['delta_GH_J_mol'])<1e-8
        assert abs(difference['H_J_mol']-spec['delta_GH_J_mol'])<1e-8
        assert difference['S_J_mol_K']==difference['Cp_J_mol_K']==0.
        samples.append({'temperature_K':t,'difference':difference,'fixed_atomic_activity_K_factor':float(np.exp(-spec['delta_GH_J_mol']/(R*t)))})
    result={'continuous':validate_definition(model),'samples':samples,'thermal_adoption':'none','BC_state_count':8,'BC_shallow_a3Sigma_u_state_present':False,'BC_quantities_are_node_diagnostics_not_continuous_GHSCp':True}
    if (DATA/'ground_vibrational_levels.npz').exists():
        with np.load(DATA/'ground_vibrational_levels.npz') as a:
            levels=a['E_v_cm_inverse'];de=float(a['De_cm_inverse']);conv=a['dE_conv_cm_inverse'];v=a['v']
            assert np.array_equal(v,np.arange(86)) and np.all(np.diff(levels)>0) and np.all(levels<de)
            result['ground_J0_audit']={'count':len(v),'De_cm_inverse':de,'E00_cm_inverse':float(levels[0]),'D0_cm_inverse':float(de-levels[0]),'D0_minus_primary_J_mol':float((de-levels[0])*WAVENUMBER_J_MOL-model['physical_energy']['measured_D0_J_mol']),'maximum_reported_grid_change_cm_inverse':float(conv.max()),'scope':'Complete listed J0 vibrational spectrum only; no claim of complete J>0/electronic/isotope partition function.','vibrational_J0_diagnostics':[]}
            theta=(levels-levels[0])*WAVENUMBER_J_MOL/R
            for t in (1500.,2000.,3000.,4000.):
                terms=np.exp(-theta/t);result['ground_J0_audit']['vibrational_J0_diagnostics'].append({'temperature_K':t,'Q_v_J0_unweighted':float(terms.sum()),'v_ge_70_fraction':float(terms[v>=70].sum()/terms.sum())})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');args=parser.parse_args()
    if args.write:(DATA/'candidate.json').write_text(json.dumps(build(),indent=2)+'\n')
    result=validate()
    if args.write:(DATA/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
