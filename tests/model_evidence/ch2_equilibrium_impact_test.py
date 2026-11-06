"""Replay archived CH2 fixed-network constraints and bounded abundance responses."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'mceg/reviews'))
import ch2_candidate as candidate
from curated_adoption_history import historical_source

DATA=ROOT/'tests/data/validation/ch2_equilibrium_impact/20261106'
BASELINE=ROOT/'tests/data/validation/curated_tio2_adoption/20261106'


@pytest.fixture(scope='module',params=['ions','neutral'])
def saved(request):
    with np.load(DATA/f'{request.param}.npz') as archive:
        return request.param,{key:archive[key].copy() for key in archive.files}


def fractions(values):
    return values-logsumexp(values,axis=-1,keepdims=True)


def constraints(data,amounts,h,budget,pressure):
    matrix=data['formula_matrix'];n=np.exp(amounts)
    elements,species=data['elements'].tolist(),data['species'].tolist()
    positive=np.asarray(budget).reshape(-1,len(elements))[0]>0
    residual=abs(n@matrix.T-budget)
    assert np.isfinite(amounts).all() and np.all(n>0)
    assert np.max(residual[...,positive]/np.asarray(budget)[...,positive])<1e-5
    for name in ('H','C','O'):
        j=elements.index(name)
        assert np.max(residual[...,j]/np.asarray(budget)[...,j])<1e-9
    if 'e-' in elements:
        charge=matrix[elements.index('e-')]
        assert np.max(abs(n@charge)/(n@abs(charge)))<1e-12
    references=[species.index('e1-' if name=='e-' else name+'1') for name in elements]
    logp=fractions(amounts)+np.log(np.asarray(pressure))[...,None]
    assert np.max(abs(logp-logp[...,references]@matrix+h))<1e-8


def test_baseline_reuse_and_target_only_substitution(saved):
    name,data=saved
    assert data['ln_n'].shape==data['h'].shape==(3,75,537 if name=='ions' else 421)
    with np.load(BASELINE/f'{name}.npz') as baseline:
        for key in ('species','elements','formula_matrix','budget','temperature_K','pressure_bar','C_over_O'):
            np.testing.assert_array_equal(data[key],baseline[key])
        np.testing.assert_array_equal(data['ln_n'][0],baseline['ln_n'])
        np.testing.assert_allclose(data['h'][0],baseline['h'],rtol=0,atol=1e-12)
    j=data['species'].tolist().index(candidate.SPECIES)
    np.testing.assert_array_equal(np.delete(data['h'][1:]-data['h'][0],j,axis=-1),0)
    for vi,variant in enumerate((candidate.PRIMARY,candidate.THERMAL),start=1):
        np.testing.assert_allclose(data['h'][vi,:,j],[candidate.formation_h(t,variant) for t in data['temperature_K']],rtol=0,atol=2e-12)
    constraints(data,data['ln_n'],data['h'],data['budget'],data['pressure_bar'])


def test_tight_and_partial_species_controls(saved):
    _,data=saved
    target=data['species'].tolist().index(candidate.SPECIES)
    for vi,indices in enumerate(data['tight_case_indices'],start=1):
        actual=data['tight_ln_n'][vi-1]
        difference=fractions(actual)-fractions(data['ln_n'][vi,indices])
        assert abs(difference).max()<1e-4 and abs(difference[:,target]).max()<1e-8
        assert abs(np.exp(fractions(actual))-np.exp(fractions(data['ln_n'][vi,indices]))).max()<1e-9
        constraints(data,actual,data['h'][vi,indices],data['budget'][indices],data['pressure_bar'][indices])
    indices=np.array([4,49,74])
    constraints(data,data['baseline_tight_ln_n'],data['h'][0,indices],data['budget'][indices],data['pressure_bar'][indices])
    constraints(data,data['triplet_ln_n'],data['triplet_h'],data['budget'][indices],data['pressure_bar'][indices])
    np.testing.assert_allclose(np.delete(data['triplet_h']-data['h'][0,indices],target,axis=-1),0,atol=1e-12,rtol=0)
    np.testing.assert_allclose(data['triplet_h'][:,target],[candidate.formation_h(t,'triplet_only') for t in data['temperature_K'][indices]],atol=2e-12,rtol=0)


def test_paths_and_same_model_gibbs_resolution(saved):
    name,data=saved
    summary=json.loads((DATA/'summary.json').read_text())
    target=data['species'].tolist().index(candidate.SPECIES)
    for vi,variant in enumerate((candidate.PRIMARY,candidate.THERMAL),start=1):
        row=next(r for r in summary['networks'] if r['network']==name and r['variant']==variant)
        if name=='ions':
            case=row['highlighted_case']['case_index'];lambdas=data['lambda_values']
            np.testing.assert_array_equal(lambdas,np.linspace(0,1,5))
            initial,final=data['h'][[0,vi],case]
            path=data['lambda_ln_n'][vi-1]
            constraints(data,path,initial+lambdas[:,None]*(final-initial),data['budget'][case],data['pressure_bar'][case])
            np.testing.assert_allclose(fractions(path[[0,-1]]),fractions(data['ln_n'][[0,vi],case]),atol=1e-4,rtol=0)
            assert np.max(np.diff(path[:,target])*(final[target]-initial[target]))<1e-10
        logs=data['ln_n'][[0,vi]].astype(np.longdouble);n=np.exp(logs)
        logp=logs-np.log(n.sum(axis=-1))[...,None]+np.log(data['pressure_bar'].astype(np.longdouble))[None,:,None]
        h=data['h'][[0,vi]].astype(np.longdouble)
        raw=np.asarray([np.sum(n[1]*(h[i]+logp[1])-n[0]*(h[i]+logp[0]),axis=-1) for i in (0,1)])
        refs=[data['species'].tolist().index('e1-' if e=='e-' else e+'1') for e in data['elements']]
        corrected=raw-((n[1]-n[0])@data['formula_matrix'].T*(h+logp)[...,refs]).sum(axis=-1)
        np.testing.assert_allclose(raw,data['delta_g_raw'][vi-1],atol=5e-14,rtol=1e-7)
        np.testing.assert_allclose(corrected,data['delta_g_corrected'][vi-1],atol=5e-14,rtol=1e-7)
        assert corrected[0].min()>=-5e-14 and corrected[1].max()<=5e-14
        assert data['gibbs_ordering_resolved'][vi-1].sum()==row['gibbs_ordering_resolved_cases']
        assert row['gibbs_ordering_resolved_cases']<75


def test_quoted_responses_follow_saved_amounts(saved):
    name,data=saved
    summary=json.loads((DATA/'summary.json').read_text())
    logx=fractions(data['ln_n']);target=data['species'].tolist().index(candidate.SPECIES)
    for vi,variant in enumerate((candidate.PRIMARY,candidate.THERMAL),start=1):
        row=next(r for r in summary['networks'] if r['network']==name and r['variant']==variant)
        ratio=np.exp(logx[vi,:,target]-logx[0,:,target])
        assert row['CH2_x_ratio_min']==pytest.approx(ratio.min(),rel=1e-12,abs=0)
        assert row['CH2_x_ratio_max']==pytest.approx(ratio.max(),rel=1e-12,abs=0)
        for point in [row['highlighted_case'],row['largest_absolute_response'],*row['species_extrema']]:
            i,j=point['case_index'],data['species'].tolist().index(point['species'])
            assert point['x_before']==pytest.approx(np.exp(logx[0,i,j]),rel=1e-12,abs=0)
            assert point['x_after']==pytest.approx(np.exp(logx[vi,i,j]),rel=1e-12,abs=0)
    triplet=next(r for r in summary['networks'] if r['network']==name and r['variant']=='triplet_only_control')
    np.testing.assert_allclose(triplet['CH2_x_ratio_to_reconstructed_RRHO'],np.exp(fractions(data['triplet_ln_n'])[:,target]-logx[2,[4,49,74],target]),rtol=1e-12)


def test_pinned_execution_and_counts_do_not_double_count_reused_baselines():
    receipt=json.loads((DATA/'receipt.json').read_text())
    assert receipt['result']=='passed' and receipt['dtype']=='float64' and receipt['backend']=='cpu'
    for path,digest in receipt['selected_file_sha256'].items():
        assert sha256(historical_source(ROOT,path,digest).read_bytes()).hexdigest()==digest,path
    for path,digest in receipt['output_sha256'].items():
        assert sha256((DATA/path).read_bytes()).hexdigest()==digest,path
    assert 'src/exogibbs/presets/curated.py' not in receipt['selected_file_sha256']
    rows=json.loads((DATA/'diagnostics.json').read_text());controls=json.loads((DATA/'selected_diagnostics.json').read_text())
    assert len(rows)==300 and len(controls)==34 and all(row['converged'] for row in rows+controls)
    summary=json.loads((DATA/'summary.json').read_text())
    assert summary['reused_unique_baseline_cases']==150 and summary['curated_update']=='none'
    assert summary['selected_counts']=={'tight_baseline':6,'tight_candidate':12,'path':10,'triplet_only':6}
    assert len(summary['independent_solver'])==4
    for point in summary['independent_solver']:
        assert len(point['comparisons'])==1
        row=point['comparisons'][0]
        assert row['max_abs_delta_log10_x']<1e-4
        assert row['element_relative_residual']<1e-5 and row['charge_relative_residual']<1e-7
        assert row['mass_action_max_abs_ln']<1e-7
