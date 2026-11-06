"""Evaluate a bounded, single-well cyanide RRHO sensitivity without adoption."""

import argparse
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

import nacn_source_review as source
from pending_candidate import R, formation_g as retained_g, reaction_properties, validate_definition

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"tests/data/validation/nacn_candidate/20261106"
SOURCE=source.DATA
NAME,SPECIES=source.NAME,source.SPECIES


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    """Load explicitly pinned reference, modes and statistical conventions."""
    return json.loads((DATA/"candidate.json").read_text())


def thermal_g(t, model: dict, xp):
    """Return G/R and derivatives for one specified RRHO molecular proxy."""
    k,h,c2=1.380649e-23,6.62607015e-34,1.4387768775039338
    log_trans=1.5*xp.log(2*math.pi*model['mass_u']*1.66053906660e-27*k*t/h**2)+xp.log(k*t/100000)
    if model['geometry']=='linear':
        log_rot=xp.log(t/(c2*model['B_cm_inverse']))-math.log(model['sigma'])
        power=3.5
    else:
        theta=np.array(model['ABC_MHz'])*1e6*h/k
        log_rot=.5*xp.log(math.pi*t**3/float(np.prod(theta)))-math.log(model['sigma'])
        power=4.
    energy=power*t
    entropy=log_trans+log_rot+math.log(model['electronic_weight'])+power
    cp=power+xp.zeros_like(t)
    for frequency in model['frequencies_cm_inverse']:
        x=c2*frequency/t
        occupation=1/xp.expm1(x)
        energy=energy+t*x*occupation
        entropy=entropy+x*occupation-xp.log(-xp.expm1(-x))
        cp=cp+x*x*xp.exp(-x)/xp.expm1(-x)**2
    return energy-t*entropy,-entropy,-cp/t


def formation_g(t, parameters: dict, xp):
    """Change only molecular thermal proxy, retaining physical D0 and residual."""
    old=retained_g(t,parameters['retained'],xp)
    new=thermal_g(t,parameters['modern'],xp)
    ref=thermal_g(t,parameters['reference'],xp)
    return tuple(a+b-c for a,b,c in zip(old,new,ref))


def build() -> dict:
    """Construct exactly one finite sensitivity and its low-mode control."""
    literature=json.loads((SOURCE/'literature.json').read_text());latest=json.loads((DATA/'sources.json').read_text())
    old=literature['janaf_printed'];rot=literature['modern_rotation']
    abc=[rot['A_minus_mean_BC_MHz']+rot['mean_BC_MHz'],rot['mean_BC_MHz']+2*rot['B_minus_C_over4_MHz'],rot['mean_BC_MHz']-2*rot['B_minus_C_over4_MHz']]
    reference={'geometry':'linear','mass_u':old['mass_u'],'B_cm_inverse':old['B0_cm_inverse'],'sigma':1,'electronic_weight':1,'frequencies_cm_inverse':old['vibrational_frequencies_expanded_cm_inverse']}
    modern={'geometry':'nonlinear','mass_u':old['mass_u'],'ABC_MHz':abc,'sigma':1,'electronic_weight':1,'frequencies_cm_inverse':latest['main_frequencies_cm_inverse']}
    lines=(ROOT/'src/exogibbs/data/FastChem4/logK/logK.dat').read_text().splitlines();i=next(i for i,line in enumerate(lines) if line.startswith(SPECIES+' '));coeff=list(map(float,lines[i+1].split()))
    retained={'kind':'retained_energy','coefficients':coeff,'delta_GH_J_mol':0.,'temperature_range_K':[1500,4000]}
    spec={'kind':'custom','module':'nacn_candidate','temperature_range_K':[1500,4000],'parameters':{'retained':retained,'reference':reference,'modern':modern}}
    control=json.loads(json.dumps(spec));control['parameters']['modern']['frequencies_cm_inverse']=latest['control_frequencies_cm_inverse']
    files=[Path(__file__),DATA/'sources.json',ROOT/'mceg/reviews/pending_candidate.py',ROOT/'mceg/reviews/nacn_source_review.py',ROOT/'src/exogibbs/data/FastChem4/logK/logK.dat']+[p for p in SOURCE.iterdir() if p.is_file()]
    jd0=source.source_comparison(1500.)['native_janaf_D0_J_mol']
    return {'id':'nacn_bounded_rrho_sensitivity_20261106','molecule':NAME,'species':SPECIES,'baseline_dataset':'curated-ch2-v1','temperature_range_K':[1500,4000],'tracked_elements':list(source.inputs()['composition']),'primary_variant':'t_shaped_rrho_sensitivity','variants':[{'name':'t_shaped_rrho_sensitivity','replacements':{SPECIES:spec}}],'selected_variants':[{'name':'alternate_low_mode_surrogate','replacements':{SPECIES:control}}],'physical_energy':{'retained_JANAF_D0_J_mol':jd0,'delta_D0_J_mol':0.,'formal_fit_intercept_Ra_J_mol':R*coeff[0],'ZPE_added_J_mol':0.},'source_sha256':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files},'residual_scope':'Preserve literal fit residual relative to the reconstructed original linear JANAF RRHO molecular proxy and its unchanged atomic reference. No physical energy or fitted intercept is reinterpreted.','thermal_scope':latest['scope'],'curated_update':'none'}


def validate() -> dict:
    """Reconstruct old source calorics independently and validate all derivatives."""
    model=load_candidate();assert model==build()
    primary=model['variants'][0]['replacements'][SPECIES];rows=[]
    for t in (1500.,2000.,2500.,3000.,4000.):
        old=thermal_g(t,primary['parameters']['reference'],np);new=thermal_g(t,primary['parameters']['modern'],np)
        props=lambda v:dict(zip(('G_J_mol','H_J_mol','S_J_mol_K','Cp_J_mol_K'),map(float,(R*v[0],R*(v[0]-t*v[1]),-R*v[1],-R*t*v[2]))))
        row={'temperature_K':t,'reference_molecular':props(old),'new_molecular':props(new),'reaction':reaction_properties(t,primary),'full_molecular_partition_ratio':math.exp((old[0]-new[0])/t)}
        if t!=2500.:
            original=source.modern_diagnostic(t,{},{});op=props(old)
            row['independent_old_model_error']={'H_J_mol':op['H_J_mol']-original['old_linear_RRHO_H_minus_H0_J_mol'],'S_J_mol_K':op['S_J_mol_K']-original['old_linear_RRHO_S_J_mol_K'],'Cp_J_mol_K':op['Cp_J_mol_K']-original['old_linear_RRHO_Cp_J_mol_K']}
            row['reference_minus_JANAF']={k:v for k,v in original.items() if k.startswith('old_RRHO_minus_JANAF')}
        rows.append(row)
    return {'continuous':validate_definition(model),'samples':rows,'minimum_main_molecular_Cp_J_mol_K':min(-R*t*thermal_g(float(t),primary['parameters']['modern'],np)[2] for t in np.arange(1500.,4001.)),'physical_accuracy':'unknown; finite sensitivity only, not a high-temperature complete coupled-state construction'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');args=parser.parse_args()
    if args.write:(DATA/'candidate.json').write_text(json.dumps(build(),indent=2)+'\n')
    result=validate()
    if args.write:(DATA/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
