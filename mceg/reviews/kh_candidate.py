"""Evaluate the bounded KH spectroscopic-energy candidate without adoption."""

import argparse
from functools import lru_cache
import json
import math
import shlex
import sys
from pathlib import Path

import numpy as np

import kh_source_review as source
from tio_candidate import _nasa

ROOT = source.ROOT
DATA = ROOT / 'tests/data/validation/kh_candidate/20261105'
R = source.R
THERMAL = ('retained', 'nasa')
INPUTS = (source.DATA/'inputs.json', source.DATA/'literature.json', source.FASTCHEM, source.CURATED)


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    """Load the exact energy-shift model and normalized NASA control."""
    return json.loads((DATA/'candidate.json').read_text())


def build() -> dict:
    """Shift the actual formation fit while preserving entropy and heat capacity."""
    inputs = json.loads(INPUTS[0].read_text())
    literature = json.loads(INPUTS[1].read_text())
    lines = source.FASTCHEM.read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith('H1K1 '))
    original = np.array(list(map(float, lines[index+1].split())))
    tables = inputs['janaf']
    h0 = {n:1000*(table['298.15'][4]+table['0.0'][3]) for n,table in tables.items()}
    old_d0 = h0['H']+h0['K']-h0['KH']
    wavenumber = literature['Zemke1988']['D0_cm-1']
    d0 = wavenumber*source.PLANCK*29979245800*6.02214076e23
    shift = (d0-old_d0)/R
    corrected = original.copy(); corrected[0] += shift
    records = inputs['nasa9']
    coefficients = {s:np.asarray(next(row['coefficients'] for row in records[s]['intervals']
                    if row['bounds_K'] == [1000.,6000.])) for s in ('KH','H','K')}
    nasa_d0 = source.h0(records['H'])+source.h0(records['K'])-source.h0(records['KH'])
    reaction = coefficients['KH']-coefficients['H']-coefficients['K']
    reaction[7] -= d0/R-nasa_d0/source.NASA_R
    return {'id':'kh_retained_thermal_spectroscopic_D0_candidate_v1','species':'H1K1',
            'reaction':'H(g)+K(g)->KH(g)','temperature_range_K':[1500,4000],
            'standard_pressure_Pa':100000,'baseline_dataset':'curated-oh-v1',
            'representation':'Exact current five-coefficient formation fit with one 1/T energy shift; no refit',
            'D0_cm-1':wavenumber,'D0_J_mol':d0,'retained_D0_J_mol':old_d0,
            'delta_D0_J_mol':d0-old_d0,'delta_D0_over_R_K':shift,
            'gas_constant_J_mol_K':R,'nasa_native_R_J_mol_K':source.NASA_R,
            'original_logK_coefficients':original.tolist(),
            'candidate_logK_coefficients':corrected.tolist(),
            'nasa_D0_J_mol':nasa_d0,'nasa_control_coefficients':reaction.tolist(),
            'nasa_control_scope':'Same spectroscopic D0 with NASA molecular and atomic thermal functions; a thermal-reference alternative, not an energy uncertainty interval',
            'source_limit':'1988 publisher abstract supplies D0; +/-0.6 cm-1 belongs to De, not a complete D0 uncertainty. Current JANAF/FastChem thermal fit and its atomic convention are preserved by the primary model.',
            'source_sha256':{str(p.relative_to(ROOT)):source.digest(p) for p in INPUTS}}


def _formation(t, thermal, xp):
    model = load_candidate()
    if thermal == 'retained':
        a,b,c,d,e = model['candidate_logK_coefficients']
        return -(a/t+b*xp.log(t)+c+d*t+e*t*t)
    if thermal == 'nasa':
        return _nasa(t,model['nasa_control_coefficients'],xp)[0]/t
    raise ValueError("thermal must be 'retained' or 'nasa'")


def formation_h(temperature_K: float, thermal: str = 'retained') -> float:
    """Return Gformation/(RT), with no extrapolation beyond the reviewed range."""
    if not math.isfinite(temperature_K) or not 1500 <= temperature_K <= 4000:
        raise ValueError('Temperature outside KH candidate domain')
    return float(_formation(float(temperature_K),thermal,np))


def formation_h_jax(temperature_K, thermal: str = 'retained'):
    """Evaluate scalar or array inputs; traced invalid temperatures return NaN."""
    import jax.numpy as jnp
    t=jnp.asarray(temperature_K,dtype=jnp.asarray(load_candidate()['candidate_logK_coefficients']).dtype)
    value=_formation(t,thermal,jnp)
    return jnp.where(jnp.isfinite(t)&(t>=1500)&(t<=4000),value,jnp.nan)


def reaction_properties(temperature_K: float, thermal: str = 'retained') -> dict:
    """Return analytic reaction G/H/S/Cp in the same formation convention."""
    t=float(temperature_K)
    g=R*t*formation_h(t,thermal)
    if thermal == 'retained':
        a,b,c,d,e=load_candidate()['candidate_logK_coefficients']
        h=R*(-a+b*t+d*t*t+2*e*t**3)
        entropy=R*(b*math.log(t)+b+c+2*d*t+3*e*t*t)
        cp=R*(b+2*d*t+6*e*t*t)
    else:
        value,slope,curvature=_nasa(t,load_candidate()['nasa_control_coefficients'],np)
        h,entropy,cp=R*(value-t*slope),-R*slope,-R*t*curvature
    return dict(G_J_mol=g,H_J_mol=float(h),S_J_mol_K=float(entropy),Cp_J_mol_K=float(cp))


def validate() -> dict:
    """Check dense independent source reconstruction and analytic/JAX derivatives."""
    import jax
    model=load_candidate()
    for name,expected in model['source_sha256'].items():
        if source.digest(ROOT/name)!=expected:raise ValueError('Changed candidate source: '+name)
    assert model==build()
    inputs=json.loads(INPUTS[0].read_text());ts=np.arange(1500.,4001.)
    current=np.array([source.curated(t,inputs)[0] for t in ts])
    exact=-current-model['delta_D0_over_R_K']/ts
    computed=np.array([formation_h(t) for t in ts])
    records=inputs['nasa9']
    nasa=[]
    for t in ts:
        p={s:source.nasa(records[s],t) for s in ('KH','H','K')}
        g=sum(p[s]['H_J_mol']/t-p[s]['S_J_mol_K'] for s in ('H','K'))-(p['KH']['H_J_mol']/t-p['KH']['S_J_mol_K'])
        nasa.append(-g/source.NASA_R-(model['D0_J_mol']/R-model['nasa_D0_J_mol']/source.NASA_R)/t)
    errors={'retained_h':float(np.max(abs(computed-exact))),
            'nasa_h':float(np.max(abs(np.array([formation_h(t,'nasa') for t in ts])-nasa)))}
    previous=jax.config.x64_enabled;jax.config.update('jax_enable_x64',True)
    try:
        probes=np.array([1500.,1637.,2253.,3000.,3871.,4000.]);jax_errors={}
        for thermal in THERMAL:
            f=jax.jit(lambda t:formation_h_jax(t,thermal));g=lambda t:R*t*f(t)
            slope=jax.jit(jax.vmap(jax.grad(g)))(probes)
            curvature=jax.jit(jax.vmap(jax.grad(jax.grad(g))))(probes)
            expected=np.array([[reaction_properties(t,thermal)[k] for k in ('H_J_mol','S_J_mol_K','Cp_J_mol_K')] for t in probes])
            actual=np.stack([R*probes*np.asarray(f(probes))-probes*slope,-slope,-probes*curvature],axis=-1)
            jax_errors[thermal]=dict(zip(('H_J_mol','S_J_mol_K','Cp_J_mol_K'),map(float,np.max(abs(actual-expected),axis=0))))
    finally:jax.config.update('jax_enable_x64',previous)
    rows=[]
    for t in (1500.,2000.,2500.,3000.,3500.,4000.):
        old_ln=source.curated(t,inputs)[0]
        rows.append({'temperature_K':t,'candidate_over_current_K':math.exp(-formation_h(t)-old_ln),
                     'nasa_control_over_primary_K':math.exp(formation_h(t)-formation_h(t,'nasa')),
                     'retained_reaction_properties':reaction_properties(t),
                     'nasa_control_reaction_properties':reaction_properties(t,'nasa')})
    return {'temperature_count':len(ts),'temperature_range_K':[1500,4000],
            'maximum_absolute_reconstruction_error':errors,'jax_float64_derivative_errors':jax_errors,
            'thermal_comparison':rows,'production_change':'none','fit_error':'No new fit; original inherited thermal fit retained'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build',action='store_true');args=parser.parse_args()
    DATA.mkdir(parents=True,exist_ok=True)
    if args.build:
        (DATA/'candidate.json').write_text(json.dumps(build(),indent=2)+'\n');load_candidate.cache_clear()
    report=validate();(DATA/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    receipt={'command':shlex.join([sys.executable, *sys.argv]),'result':'passed',
             'source_sha256':{str(p.relative_to(ROOT)):source.digest(p) for p in (*INPUTS,Path(__file__),Path(source.__file__),ROOT/'mceg/reviews/tio_candidate.py')},
             'output_sha256':{p.name:source.digest(p) for p in (DATA/'candidate.json',DATA/'validation.json')}}
    (DATA/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(report,indent=2))
