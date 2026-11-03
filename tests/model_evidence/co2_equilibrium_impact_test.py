"""Verify saved CO2 finite substitutions, budgets and independent stationarity."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'mceg/reviews'))
import co2_candidate as candidate
DATA=ROOT/'tests/data/validation/co2_equilibrium_impact/20261104'
BASE=ROOT/'tests/data/validation/curated_tio_adoption/20261103/ions.npz'


def load(name):
    with np.load(DATA/(name+'.npz')) as a:
        return {k:a[k] for k in a.files}


def logx(values):
    return values-logsumexp(values,axis=-1,keepdims=True)


def constraints(data, ln_n, h, budget, pressure):
    n=np.exp(ln_n); matrix=data['formula_matrix'];elements=data['elements'].tolist();species=data['species'].tolist()
    positive=np.asarray(budget).reshape(-1,len(elements))[0]>0
    assert np.max(abs(n@matrix.T-budget)[...,positive]/np.asarray(budget)[...,positive])<1e-5
    charge=matrix[elements.index('e-')]
    assert np.max(abs(n@charge)/(n@abs(charge)))<1e-12
    refs=[species.index('e1-' if e=='e-' else e+'1') for e in elements]
    logp=logx(ln_n)+np.log(pressure)[...,None]
    assert np.max(abs(logp-logp[...,refs]@matrix+h))<2e-11
    for e in ('C','O'):
        i=elements.index(e)
        assert n@matrix[i]==pytest.approx(np.broadcast_to(np.asarray(budget)[...,i],n.shape[:-1]),rel=3e-12,abs=0)


@pytest.mark.parametrize('domain,count',[('original',75),('hydrogen_poor',6)])
def test_only_co2_changes_and_all_saved_endpoints_obey_constraints(domain,count):
    d=load(domain);c=d['species'].tolist().index('C1O2')
    assert d['ln_n'].shape==d['h'].shape==(2,count,537)
    np.testing.assert_array_equal(np.delete(d['h'][1]-d['h'][0],c,axis=-1),0)
    np.testing.assert_allclose(d['h'][1,:,c],[candidate.formation_h(t) for t in d['temperature_K']],rtol=0,atol=1e-12)
    constraints(d,d['ln_n'],d['h'],d['budget'],d['pressure_bar'])
    assert np.all(d['ln_n'][1,:,c]>=d['ln_n'][0,:,c])
    if domain=='original':
        with np.load(BASE) as b:
            for key in ('budget','formula_matrix','species','elements','temperature_K','pressure_bar'):
                np.testing.assert_array_equal(d[key],b[key])
            np.testing.assert_array_equal(d['ln_n'][0],b['ln_n'])
    else:
        e=d['elements'].tolist();b=d['budget'][0]
        for atom,ratio in (('C',.5),('H',.01),('He',.001)):
            assert b[e.index(atom)]/b[e.index('O')]==pytest.approx(ratio)
        assert np.exp(logx(d['ln_n']))[0,5,c]==pytest.approx(.1345217150397039,rel=1e-10)
        assert np.exp(logx(d['ln_n']))[1,5,c]==pytest.approx(.13512195717874448,rel=1e-10)


def test_tight_solutions_and_monotone_finite_path():
    for domain in ('original','hydrogen_poor'):
        d=load(domain);i=d['tight_case_indices'];c=d['species'].tolist().index('C1O2')
        constraints(d,d['tight_ln_n'],d['h'][:,i],d['budget'][i],d['pressure_bar'][i])
        delta=logx(d['tight_ln_n'])-logx(d['ln_n'][:,i])
        assert abs(delta).max()<3e-5 and abs(delta[...,c]).max()<2e-12
    d=load('hydrogen_poor');path=load('path');i=int(path['case_index']);lam=path['lambda_values']
    h=d['h'][0,i]+lam[:,None]*(d['h'][1,i]-d['h'][0,i])
    constraints(d,path['ln_n'],h,d['budget'][i],d['pressure_bar'][i])
    c=d['species'].tolist().index('C1O2')
    assert np.all(np.diff(path['ln_n'][:,c])>0)
    np.testing.assert_allclose(logx(path['ln_n'][[0,-1]]),logx(d['ln_n'][:,i]),atol=1e-11,rtol=0)


def test_local_isotope_and_thermal_uncertainty_variants_remain_conserved():
    rows=json.loads((DATA/'local_sensitivities.json').read_text())
    assert len(rows)==9 and len({(r['domain'],r['case_index'],r['variant']) for r in rows})==9
    for row in rows:
        d=load(row['domain']);i=row['case_index'];h=np.array(row['h']);n=np.array(row['ln_n']);c=d['species'].tolist().index('C1O2')
        constraints(d,n,h,np.array(row['budget']),np.array(row['pressure_bar']))
        np.testing.assert_allclose(np.delete(h-d['h'][1,i],c),0,atol=1e-12,rtol=0)
        if row['variant']=='isotope_average':
            assert h[c]==pytest.approx(candidate.formation_h(row['temperature_K'],'isotope_average'),abs=1e-12)
        else:
            sign=-1 if row['variant']=='Q_minus_1u' else 1
            assert h[c]-d['h'][1,i,c]==pytest.approx(-np.log1p(sign*row['relative_u_Q']),abs=2e-13)
            assert sign*(n[c]-d['ln_n'][1,i,c])>0


def test_same_model_gibbs_ordering_accounts_for_feasibility():
    for domain in ('original','hydrogen_poor'):
        d=load(domain);logs=d['ln_n'].astype(np.longdouble);n=np.exp(logs)
        logp=logs-np.log(n.sum(-1))[...,None]+np.log(d['pressure_bar'].astype(np.longdouble))[None,:,None]
        refs=[d['species'].tolist().index('e1-' if e=='e-' else e+'1') for e in d['elements']]
        db=(n[1]-n[0])@d['formula_matrix'].T
        raw=[];fixed=[]
        for i in (0,1):
            potential=d['h'][i].astype(np.longdouble)
            delta=np.sum(n[1]*(logp[1]+potential),-1)-np.sum(n[0]*(logp[0]+potential),-1)
            raw.append(delta);fixed.append(delta-np.sum(db*(logp[i]+potential)[...,refs],-1))
        np.testing.assert_allclose(raw,d['raw_gibbs_exchange'],atol=2e-17,rtol=1e-6)
        np.testing.assert_allclose(fixed,d['corrected_gibbs_exchange'],atol=2e-17,rtol=1e-6)
        assert np.min(fixed[0])>-5e-14 and np.max(fixed[1])<5e-14


def test_execution_receipt_and_independent_comparison():
    result=json.loads((DATA/'summary.json').read_text())
    assert result['new_endpoint_solves']==87 and result['reused_baseline_cases']==75
    assert result['curated_update']=='none'
    assert len(result['independent_solver'])==2
    for r in result['independent_solver']:
        assert r['max_abs_delta_log10_x']<3e-9
        assert r['mass_action_max_abs_ln']<3e-10
    receipt=json.loads((DATA/'receipt.json').read_text())
    for name,digest in receipt['selected_file_sha256'].items():
        assert sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for name,digest in receipt['output_sha256'].items():
        assert sha256((DATA/name).read_bytes()).hexdigest()==digest,name
