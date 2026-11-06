"""Check CH2 energy anchoring, table inconsistency and independent thermal sums."""
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'mceg/reviews'))
import ch2_candidate as candidate
import ch2_rrho_audit
from curated_adoption_history import historical_source


@pytest.fixture(scope='module')
def report():
    return candidate.validate()


def test_same_version_ground_energy_does_not_replace_the_fit_intercept():
    model=candidate.load_candidate()
    assert model==candidate.build()
    assert model['evaluated_D0_J_mol']==752392.
    assert model['original_source_D0_J_mol']==757340.
    assert model['delta_D0_J_mol']==-4948.
    assert model['formal_fit_1_over_T_energy_J_mol']-model['original_source_D0_J_mol']==pytest.approx(687.186027754,abs=1e-8,rel=0)
    for t in (1500.,2000.,3000.,4000.):
        old,new=(candidate.reaction_properties(t,v) for v in ('current',candidate.PRIMARY))
        for key in ('G_J_mol','H_J_mol'):
            assert new[key]-old[key]==pytest.approx(4948.,abs=1e-8,rel=0)
        for key in ('S_J_mol_K','Cp_J_mol_K'):
            assert new[key]==old[key]
        assert candidate.formation_h(t,'nasa_thermal')==candidate.formation_h(t,candidate.PRIMARY)


def test_balanced_nasa_proxy_gauge_and_complete_residual_are_preserved():
    raw=json.loads((candidate.source.DATA/'inputs.json').read_text())
    model=candidate.load_candidate()
    for t in (1500.,2253.,3000.,4000.):
        thermal={}
        for name,record in raw['nasa_records'].items():
            heat,entropy,_=candidate.source.nasa_properties(record,t,candidate.R)
            h0=(record['hf298_J_mol']-record['h298_minus_h0_J_mol'])*candidate.R/candidate.NASA_R
            thermal[name]=heat-h0-t*entropy
        atom_thermal=math.fsum(n*thermal[s] for s,n in raw['composition'].items())
        old_reference=(-model['original_source_D0_J_mol']+thermal['CH2']-atom_thermal)/(candidate.R*t)
        residual=candidate.formation_h(t,'current')-old_reference
        for variant,state in ((candidate.THERMAL,'equilibrated'),('triplet_only','triplet_only')):
            new_thermal=candidate.molecular_properties(t,state)['G_minus_H0_J_mol']
            new_reference=(-model['evaluated_D0_J_mol']+new_thermal-atom_thermal)/(candidate.R*t)
            assert candidate.formation_h(t,variant)-new_reference==pytest.approx(residual,abs=3e-12,rel=0)


def test_both_component_tables_reproduce_but_equilibrated_table_does_not(report):
    for name in ('triplet_only','singlet_only'):
        audit=report['table_audit'][name]
        assert len(audit['rows'])==16
        limits={'G_minus_H0_J_mol':2.,'H_minus_H0_J_mol':1.1,'S_J_mol_K':.0008,'Cp_J_mol_K':.0007}
        assert all(audit['maximum_absolute_difference'][key]<limit for key,limit in limits.items())
    combined=report['table_audit']['equilibrated']['maximum_absolute_difference']
    assert combined['H_minus_H0_J_mol']>2900
    assert combined['S_J_mol_K']>2.7
    # This uses only the two printed partial-species tables and the measured gap.
    for row in report['printed_component_recombination']:
        difference=row['minus_reconstructed_RRHO']
        assert abs(difference['G_J_mol'])<2
        assert abs(difference['H_J_mol'])<1.1
        assert abs(difference['S_J_mol_K'])<.0008
        assert abs(difference['Cp_J_mol_K'])<.0008
    row=next(r for r in report['printed_component_recombination'] if r['temperature_K']==3000)
    assert row['minus_printed_equilibrated']['H_J_mol']==pytest.approx(-2974.9379803,abs=2e-6,rel=0)
    assert row['minus_printed_equilibrated']['S_J_mol_K']==pytest.approx(-2.71671942686,abs=1e-10,rel=0)


def test_state_origins_weights_symmetry_and_no_extra_zpe():
    data=candidate.inputs();states=data['states']
    assert data['mass_u']==14.02658 and data['nuclear_spin_weight']==1
    assert [s['electronic_weight'] for s in states]==[3,1,1]
    assert [s['symmetry_number'] for s in states]==[2,2,2]
    assert [s['origin_cm-1'] for s in states]==[0,3147,11497]
    assert states[1]['vibration_cm-1'][0]==2806 # not the gap copied into Burcat's nu field
    assert candidate.C2==pytest.approx(1.4387768775039338,abs=1e-15,rel=0)
    t=298.15
    regular=candidate.molecular_properties(t)
    triplet=candidate.molecular_properties(t,'triplet_only')
    assert regular['state_fractions'][1]==pytest.approx(1.407503481e-7,rel=2e-8)
    assert 0<regular['H_minus_H0_J_mol']-triplet['H_minus_H0_J_mol']<.006
    assert regular['state_fractions'][2]<1e-20
    # The ground zero is already encoded by the evaluated D0; origins only enter populations.
    assert candidate.load_candidate()['evaluated_D0_J_mol']==752392


def test_dense_single_potential_and_domain_checks(report):
    import jax
    jax.config.update('jax_enable_x64',True)
    assert report['temperature_count']==2501 and report['current_h_maximum_error']<2e-12
    temperatures=np.arange(1500.,4001.)
    for variant,errors in report['jax_float64_derivative_errors'].items():
        assert errors['G_J_mol']<2e-8 and errors['H_J_mol']<2e-8
        assert errors['S_J_mol_K']<1e-10 and errors['Cp_J_mol_K']<1e-10
        actual=np.asarray(jax.jit(lambda t:candidate.formation_h_jax(t,variant))(temperatures))
        np.testing.assert_allclose(actual,[candidate.formation_h(t,variant) for t in temperatures],rtol=0,atol=2e-12)
        for t in (1499.,4001.,np.nan,np.inf):
            with pytest.raises(ValueError):candidate.formation_h(t,variant)
        invalid=np.asarray(candidate.formation_h_jax(np.array([1499.,4001.,np.nan]),variant))
        assert np.isnan(invalid).all()
    assert report['curated_change']=='none'


def test_independent_sums_bound_numeric_extension_without_claiming_anharmonic_accuracy():
    report=ch2_rrho_audit.audit()
    assert report['oscillator_maximum_quanta']==128
    assert report['rigid_rotor_maximum_J']==[100,140]
    for row in report['samples']:
        for difference in ('finite_oscillator_minus_analytic','quantum_rotor_140_minus_100'):
            limits={'G_minus_H0_J_mol':1e-6,'H_minus_H0_J_mol':2e-6,'S_J_mol_K':1e-9,'Cp_J_mol_K':1e-8}
            assert all(abs(row[difference][key])<limit for key,limit in limits.items())
    assert report['samples'][-1]['temperature_K']==4000
    assert 'nonrigidity' not in report['scope'] or 'unvalidated' in report['scope']


def test_pinned_input_and_output_integrity():
    receipt=json.loads((candidate.DATA/'receipt.json').read_text())
    for path,digest in receipt['selected_file_sha256'].items():
        assert sha256(historical_source(ROOT,path,digest).read_bytes()).hexdigest()==digest,path
    for path,digest in receipt['output_sha256'].items():
        assert sha256((candidate.DATA/path).read_bytes()).hexdigest()==digest,path
    assert 'src/exogibbs/presets/curated.py' not in receipt['selected_file_sha256']
