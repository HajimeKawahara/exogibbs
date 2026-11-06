"""Check explicit mode count, physical gauge and bounded cyanide substitution."""
import json
from hashlib import sha256
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'mceg/reviews'))
import kcn_candidate as study
from pending_candidate import formation_h,formation_h_jax,reaction_properties
from curated_adoption_history import historical_source


@pytest.fixture(scope='module',autouse=True)
def float64():
    previous=jax.config.x64_enabled;jax.config.update('jax_enable_x64',True)
    yield
    jax.config.update('jax_enable_x64',previous)


def test_build_and_dense_ghscp_agree_without_fitting_source_values():
    result=study.validate()
    assert result['continuous']['temperature_count']==2752
    assert len(result['continuous']['models'])==2
    assert result['minimum_main_molecular_Cp_J_mol_K']>50
    for row in result['samples']:
        for key,value in row.get('independent_old_model_error',{}).items():
            assert abs(value)<(1e-8 if key=='H_J_mol' else 1e-11)
        for key,value in row.get('reference_minus_JANAF',{}).items():
            assert abs(value)<(3 if 'thermal_H' in key else .003)


def test_three_modes_and_statistical_conventions_replace_four_linear_modes():
    model=study.load_candidate();spec=model['variants'][0]['replacements'][study.SPECIES]
    p=spec['parameters'];ref,new=p['reference'],p['modern']
    assert len(ref['frequencies_cm_inverse'])==4
    assert len(new['frequencies_cm_inverse'])==3
    assert ref['mass_u']==new['mass_u']
    assert ref['sigma']==new['sigma']==ref['electronic_weight']==new['electronic_weight']==1
    assert model['physical_energy']['delta_D0_J_mol']==0
    assert model['physical_energy']['ZPE_added_J_mol']==0
    assert model['physical_energy']['retained_JANAF_D0_J_mol']!=model['physical_energy']['formal_fit_intercept_Ra_J_mol']
    assert 'sensitivity' in model['thermal_scope'].lower()
    assert model['curated_update']=='none'


def test_partition_ratio_and_independent_vibrational_moments():
    spec=study.load_candidate()['variants'][0]['replacements'][study.SPECIES];p=spec['parameters']
    h,k=6.62607015e-34,1.380649e-23
    for t in (1500.,1743.25,2500.,3983.15,4000.):
        theta_old=np.array(p['reference']['frequencies_cm_inverse'])*h*29979245800/k
        theta_new=np.array(p['modern']['frequencies_cm_inverse'])*h*29979245800/k
        rot_theta=np.array(p['modern']['ABC_MHz'])*1e6*h/k
        qr_new=np.sqrt(np.pi*t**3/np.prod(rot_theta));qr_old=t/(h*29979245800/k*p['reference']['B_cm_inverse'])
        qv_old=np.prod(1/(-np.expm1(-theta_old/t)));qv_new=np.prod(1/(-np.expm1(-theta_new/t)))
        delta_g=-study.R*t*np.log(qr_new*qv_new/(qr_old*qv_old))
        energy=lambda theta:np.sum(theta/np.expm1(theta/t))
        cp=lambda theta:np.sum((theta/t)**2*np.exp(-theta/t)/(-np.expm1(-theta/t))**2)
        expected_h=study.R*(.5*t+energy(theta_new)-energy(theta_old))
        expected_cp=study.R*(.5+cp(theta_new)-cp(theta_old))
        before=reaction_properties(t,p['retained']);after=reaction_properties(t,spec)
        assert after['G_J_mol']-before['G_J_mol']==pytest.approx(delta_g,abs=2e-9,rel=0)
        assert after['H_J_mol']-before['H_J_mol']==pytest.approx(expected_h,abs=2e-9,rel=0)
        assert after['Cp_J_mol_K']-before['Cp_J_mol_K']==pytest.approx(expected_cp,abs=1e-11,rel=0)


def test_scalar_batch_and_invalid_domain():
    spec=study.load_candidate()['variants'][0]['replacements'][study.SPECIES]
    ts=jnp.array([[1500.,1873.13],[3500.27,4000.]])
    expected=np.array([[formation_h(float(t),spec) for t in row] for row in ts])
    np.testing.assert_allclose(jax.jit(lambda t:formation_h_jax(t,spec))(ts),expected,atol=1e-13,rtol=0)
    assert np.isnan(formation_h_jax(jnp.array([1499.,4001.,np.nan,np.inf]),spec)).all()
    with pytest.raises(ValueError):formation_h(1499,spec)


def test_frozen_endpoint_and_control_replay():
    from pending_equilibrium_impact import replay
    s=replay(study.DATA/'candidate.json',ROOT/'tests/data/validation/kcn_equilibrium_impact/20261106')
    assert s['new_endpoint_solves']==150
    assert s['tighter_control_solves']==12 and s['path_control_solves']==10
    assert s['selected_variant_solves']==4 and s['independent_fastchem_runs']==4


def test_continuous_receipt_input_and_output_hashes():
    receipt=json.loads((study.DATA/'receipt.json').read_text())
    for name,digest in receipt['selected_file_sha256'].items():
        assert sha256(historical_source(ROOT,name,digest).read_bytes()).hexdigest()==digest
    for name,digest in receipt['output_sha256'].items():
        assert sha256((study.DATA/name).read_bytes()).hexdigest()==digest
