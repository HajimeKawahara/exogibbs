"""Verify the KH candidate's energy normalization and continuous calorics."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'mceg/reviews'))
import kh_candidate as candidate


@pytest.fixture(scope='module')
def report():
    return candidate.validate()


def test_exact_energy_shift_preserves_the_inherited_thermal_coefficients():
    model=candidate.load_candidate()
    assert candidate.build()==model
    assert model['D0_cm-1']==14282.6
    assert model['D0_J_mol']==pytest.approx(170857.8386391254,abs=1e-8,rel=0)
    assert model['retained_D0_J_mol']==pytest.approx(180390.,abs=1e-8,rel=0)
    original=np.array(model['original_logK_coefficients'])
    actual=np.array(model['candidate_logK_coefficients'])
    np.testing.assert_array_equal(actual[1:],original[1:])
    assert actual[0]-original[0]==pytest.approx(model['delta_D0_over_R_K'],abs=1e-10,rel=0)
    assert model['delta_D0_J_mol']<0
    # The reported uncertainty attaches to De, not a new D0 error band.
    assert 'De' in model['source_limit'] and 'uncertainty' in model['source_limit']


def test_entropy_and_heat_capacity_stay_fixed_while_enthalpy_changes():
    model=candidate.load_candidate();a,b,c,d,e=model['original_logK_coefficients']
    for t in (1500.,1637.,2253.,3000.,3871.,4000.):
        old_h=candidate.R*(-a+b*t+d*t*t+2*e*t**3)
        old_s=candidate.R*(b*np.log(t)+b+c+2*d*t+3*e*t*t)
        old_cp=candidate.R*(b+2*d*t+6*e*t*t)
        p=candidate.reaction_properties(t)
        assert p['H_J_mol']-old_h==pytest.approx(-model['delta_D0_J_mol'],abs=1e-9,rel=0)
        assert p['S_J_mol_K']==pytest.approx(old_s,abs=1e-12,rel=0)
        assert p['Cp_J_mol_K']==pytest.approx(old_cp,abs=1e-12,rel=0)
        assert p['G_J_mol']==pytest.approx(p['H_J_mol']-t*p['S_J_mol_K'],abs=1e-8,rel=0)
        initial=a/t+b*np.log(t)+c+d*t+e*t*t
        assert -candidate.formation_h(t)-initial==pytest.approx(model['delta_D0_J_mol']/(candidate.R*t),abs=2e-13,rel=0)


def test_dense_source_and_automatic_derivative_checks(report):
    assert report['temperature_count']==2501
    for value in report['maximum_absolute_reconstruction_error'].values():assert value<1e-12
    for errors in report['jax_float64_derivative_errors'].values():
        assert errors['H_J_mol']<1e-8
        assert errors['S_J_mol_K']<1e-10
        assert errors['Cp_J_mol_K']<1e-10
    frozen=json.loads((candidate.DATA/'validation.json').read_text())
    for a,b in zip(report['thermal_comparison'],frozen['thermal_comparison']):
        assert a['candidate_over_current_K']==pytest.approx(b['candidate_over_current_K'],abs=1e-12,rel=0)
        assert a['nasa_control_over_primary_K']==pytest.approx(b['nasa_control_over_primary_K'],abs=1e-12,rel=0)
    assert report['production_change']=='none'


def test_scalar_array_jit_and_invalid_temperature_handling():
    import jax
    previous=jax.config.x64_enabled;jax.config.update('jax_enable_x64',True)
    try:
        for thermal in candidate.THERMAL:
            f=jax.jit(lambda t:candidate.formation_h_jax(t,thermal))
            t=np.array([1500.,2253.,4000.]);expected=[candidate.formation_h(x,thermal) for x in t]
            np.testing.assert_allclose(f(t),expected,atol=1e-12,rtol=0)
            np.testing.assert_allclose(jax.jit(jax.vmap(f))(t),expected,atol=1e-12,rtol=0)
            assert float(f(2500))==pytest.approx(candidate.formation_h(2500,thermal),abs=1e-12,rel=0)
            assert np.isnan(f(np.array([1499.,4001.,np.inf,np.nan]))).all()
        for t in (1499.,4001.,np.inf,np.nan):
            with pytest.raises(ValueError,match='outside'):candidate.formation_h(t)
        with pytest.raises(ValueError,match='thermal'):candidate.formation_h(2000,'unknown')
    finally:jax.config.update('jax_enable_x64',previous)


def test_nasa_control_uses_the_same_D0_and_native_reference_constant():
    model=candidate.load_candidate();source=candidate.source
    records=json.loads(candidate.INPUTS[0].read_text())['nasa9']
    raw=np.array(records['KH']['intervals'][1]['coefficients'])-np.array(records['H']['intervals'][1]['coefficients'])-np.array(records['K']['intervals'][1]['coefficients'])
    delta=np.array(model['nasa_control_coefficients'])-raw
    np.testing.assert_array_equal(np.delete(delta,7),0)
    assert delta[7]==pytest.approx(-(model['D0_J_mol']/candidate.R-model['nasa_D0_J_mol']/source.NASA_R),abs=1e-10,rel=0)
    wrong=(model['D0_J_mol']-model['nasa_D0_J_mol'])/candidate.R
    assert abs(-delta[7]-wrong)>.1


def test_candidate_sources_and_real_receipt_are_pinned():
    receipt=json.loads((candidate.DATA/'receipt.json').read_text())
    assert receipt['result']=='passed'
    for name,expected in receipt['source_sha256'].items():assert sha256((ROOT/name).read_bytes()).hexdigest()==expected
    for name,expected in receipt['output_sha256'].items():assert sha256((candidate.DATA/name).read_bytes()).hexdigest()==expected
