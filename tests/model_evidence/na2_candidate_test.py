"""Verify measured D0 sign, retained calorics, source limits and finite response."""
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
import na2_candidate as study
from pending_candidate import formation_h,formation_h_jax,reaction_properties
from curated_adoption_history import historical_source
from exogibbs.presets import curated_ch2


@pytest.fixture(scope='module',autouse=True)
def float64():
    previous=jax.config.x64_enabled;jax.config.update('jax_enable_x64',True)
    yield
    jax.config.update('jax_enable_x64',previous)


def test_independent_energy_cycle_preserves_thermal_and_actual_current_fit():
    model=study.load_candidate();assert model==study.build();spec=model['variants'][0]['replacements'][study.SPECIES]
    energy=model['physical_energy'];spect=energy['measurement']
    # Exact SI constants independently convert the ground-vibrational binding
    # energy. No half-frequency ZPE or potential well depth enters this cycle.
    d0=spect['D0_cm_inverse']*6.62607015e-34*299792458*100*6.02214076e23
    delta=energy['retained_JANAF_D0_J_mol']-d0
    assert delta==pytest.approx(spec['delta_GH_J_mol'],abs=2e-11,rel=0)
    assert energy['ZPE_added_J_mol']==0 and delta<0
    assert energy['formal_fit_intercept_Ra_J_mol']!=energy['retained_JANAF_D0_J_mol']
    base=curated_ch2.chemsetup();j=base.species.index(study.SPECIES);old={**spec,'delta_GH_J_mol':0}
    for t in (1500.,1637.25,2500.,3871.3,4000.):
        assert formation_h(t,old)==pytest.approx(float(base.hvector_func(t)[j]),abs=2e-14,rel=0)
        b,a=reaction_properties(t,old),reaction_properties(t,spec)
        for key in ('G_J_mol','H_J_mol'):assert a[key]-b[key]==pytest.approx(delta,abs=1e-8,rel=0)
        for key in ('S_J_mol_K','Cp_J_mol_K'):assert a[key]==b[key]


def test_dense_and_selected_ghscp_share_one_potential():
    result=study.validate();assert result['continuous']['temperature_count']==2752
    assert len(result['continuous']['models'])==2
    assert result['thermal_adoption']=='none'
    if 'ground_J0_audit' in result:
        a=result['ground_J0_audit'];assert a['count']==86
        assert a['D0_cm_inverse']==pytest.approx(4404.80862651,abs=1e-8,rel=0)
        assert abs(a['D0_minus_primary_J_mol'])<study.load_candidate()['physical_energy']['reported_D0_uncertainty_J_mol']
        assert a['maximum_reported_grid_change_cm_inverse']<1e-8
        assert all(0<x['v_ge_70_fraction']<1 for x in a['vibrational_J0_diagnostics'])


def test_primary_state_inventory_does_not_claim_complete_hot_q():
    lines=(study.SOURCE/'bc-state-input.txt').read_text().splitlines()
    states=[line for line in lines if ' HH ' in line or ' IR ' in line]
    assert len(states)==8
    assert any('^3 Pi' in line for line in states)
    assert not any('^3 Sigma' in line for line in states)
    facts=json.loads((study.DATA/'sources.json').read_text())
    assert facts['BC_audit']['thermal_nodes_K']==[1000,1500,2000,3000,4000,5000]
    assert len(facts['official_2025_SI'])==2
    assert study.load_candidate()['curated_update']=='none'


def test_scalar_jit_batch_and_domain():
    spec=study.load_candidate()['variants'][0]['replacements'][study.SPECIES]
    ts=jnp.array([[1500.,1873.13],[3500.27,4000.]])
    expected=np.array([[formation_h(float(t),spec) for t in row] for row in ts])
    np.testing.assert_allclose(jax.jit(lambda t:formation_h_jax(t,spec))(ts),expected,atol=1e-13,rtol=0)
    assert np.isnan(formation_h_jax(jnp.array([1499.,4001.,np.nan,np.inf]),spec)).all()
    with pytest.raises(ValueError):formation_h(1499,spec)


def test_frozen_endpoint_and_control_replay():
    from pending_equilibrium_impact import replay
    s=replay(study.DATA/'candidate.json',ROOT/'tests/data/validation/na2_equilibrium_impact/20261106')
    assert s['new_endpoint_solves']==150 and s['tighter_control_solves']==12
    assert s['path_control_solves']==10 and s['selected_variant_solves']==4
    assert s['independent_fastchem_runs']==4


def test_continuous_receipt_input_and_output_hashes():
    receipt=json.loads((study.DATA/'receipt.json').read_text())
    for name,digest in receipt['selected_file_sha256'].items():
        assert sha256(historical_source(ROOT,name,digest).read_bytes()).hexdigest()==digest
    for name,digest in receipt['output_sha256'].items():
        assert sha256((study.DATA/name).read_bytes()).hexdigest()==digest
