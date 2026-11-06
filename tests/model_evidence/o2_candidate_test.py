"""Check evaluated O2 source fitting, energy/gauge and single-potential calorics."""

import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'mceg/reviews'))
import o2_candidate as study
from pending_candidate import formation_g, formation_h, formation_h_jax, reaction_properties
from c2h_source_review import nasa_properties
from exogibbs.presets import curated_ch2


@pytest.fixture(scope='module',autouse=True)
def float64():
    previous=jax.config.x64_enabled;jax.config.update('jax_enable_x64',True)
    yield
    jax.config.update('jax_enable_x64',previous)


def test_fit_uses_251_cp_rows_and_holds_out_caloric_values():
    result=study.validate()
    assert result['source_rows']==2501 and result['withheld_Cp_rows']==2250
    assert result['fit_Cp_rows']==251
    assert result['maximum_source_errors']['G_J_mol']<4
    assert result['maximum_source_errors']['H_J_mol']<1.6
    assert result['maximum_source_errors']['S_J_mol_K']<.0013
    assert result['withheld_Cp_max_error_J_mol_K']<.006
    assert max(result['maximum_error_over_published_expanded_U'].values())<.08
    assert result['independent_source_quadrature_H_error_J_mol']<1
    assert result['independent_source_quadrature_S_error_J_mol_K']<.0011
    assert result['minimum_fit_Cp_J_mol_K']>36


def test_balanced_energy_is_distinct_from_fit_intercept_and_has_no_zpe_duplication():
    model=study.load_candidate();energy=model['physical_energy']
    assert energy['ATcT_version']=='1.228'
    assert energy['ATcT_D0_J_mol']==493686 and energy['retained_JANAF_D0_J_mol']==493579
    assert energy['delta_D0_J_mol']==107
    assert energy['formal_fit_intercept_Ra_J_mol']!=energy['retained_JANAF_D0_J_mol']
    base=curated_ch2.chemsetup();j=base.species.index('O2')
    control=model['selected_variants'][0]['replacements']['O2']
    for t in (1500.,1637.,2500.,3871.,4000.):
        assert study.R*t*(formation_h(t,control)-float(base.hvector_func(t)[j]))==pytest.approx(-107,abs=1e-8,rel=0)
        unchanged={**control,'delta_GH_J_mol':0}
        before,after=reaction_properties(t,unchanged),reaction_properties(t,control)
        for key in ('G_J_mol','H_J_mol'):assert after[key]-before[key]==pytest.approx(-107,abs=1e-8,rel=0)
        for key in ('S_J_mol_K','Cp_J_mol_K'):assert after[key]==before[key]


def test_source_proxy_residual_and_atomic_gauge_are_preserved_independently():
    model=study.load_candidate();primary=model['variants'][0]['replacements']['O2'];params=primary['parameters']
    records=json.loads((study.SOURCE/'inputs.json').read_text())['nasa9']
    for t in (1500.,2000.,2500.,3000.,4000.):
        # Evaluate each native NASA thermochemical function independently, then
        # remove its source H0 and normalize to the common SI gas constant.
        thermal={}
        for name,record in records.items():
            h,s,cp=nasa_properties(record,t,8.314510)
            h0=record['hf298_J_mol']-record['h298_minus_h0_J_mol']
            thermal[name]=(h-h0-t*s)*study.R/8.314510
        old=formation_g(t,{**params['retained'],'delta_GH_J_mol':0},np)[0]*study.R
        residual=old-(-493579+thermal['O2']-2*thermal['O'])
        modern=study._evaluated(t,params['evaluated'],np)[0]*study.R
        expected=residual-493686+modern-2*thermal['O']
        assert reaction_properties(t,primary)['G_J_mol']==pytest.approx(expected,abs=3e-9,rel=0)


def test_natural_isotope_table_is_used_once_without_extra_statistical_factors():
    model=study.load_candidate();params=model['variants'][0]['replacements']['O2']['parameters']
    with np.load(study.DATA/'published_tables.npz') as tb:
        assert params['evaluated']['anchor_H_over_R_K']*study.R==pytest.approx(tb['H_relative_J_mol'][0],abs=1e-10)
        assert params['evaluated']['anchor_S_over_R']*study.R==pytest.approx(tb['S_J_mol_K'][0],abs=1e-12)
        # The published natural entropy already includes its mass correction;
        # another R ln(3), R ln(2) or isotope mixing term would fail by orders.
        pure=tb['pure16__S/[J/(K mol)]']
        assert np.max(abs(tb['S_J_mol_K']-pure))<.02
        assert np.min(tb['S_J_mol_K']-pure)>0


def test_scalar_batch_jit_vmap_and_invalid_domain():
    spec=study.load_candidate()['variants'][0]['replacements']['O2']
    ts=jnp.array([[1500.,1873.13],[3500.27,4000.]])
    expected=np.array([[formation_h(float(t),spec) for t in row] for row in ts])
    np.testing.assert_allclose(jax.jit(lambda t:formation_h_jax(t,spec))(ts),expected,atol=1e-13,rtol=0)
    np.testing.assert_allclose(jax.vmap(lambda t:formation_h_jax(t,spec))(ts[0]),expected[0],atol=1e-13,rtol=0)
    assert np.isnan(formation_h_jax(jnp.array([1499.,4001.,np.nan,np.inf]),spec)).all()
    with pytest.raises(ValueError):formation_h(1499,spec)


def test_frozen_endpoint_and_control_replay():
    from pending_equilibrium_impact import replay
    summary=replay(study.DATA/'candidate.json',ROOT/'tests/data/validation/o2_equilibrium_impact/20261106')
    assert summary['new_endpoint_solves']==150
    assert summary['tighter_control_solves']==12 and summary['path_control_solves']==10
    assert summary['selected_variant_solves']==4 and summary['independent_fastchem_runs']==4


def test_continuous_receipt_preserves_input_and_output_bytes():
    from hashlib import sha256
    from curated_history import historical_source
    receipt=json.loads((study.DATA/'receipt.json').read_text())
    for name,digest in receipt['selected_file_sha256'].items():
        assert sha256(historical_source(ROOT,name).read_bytes()).hexdigest()==digest
    for name,digest in receipt['output_sha256'].items():
        assert sha256((study.DATA/name).read_bytes()).hexdigest()==digest
