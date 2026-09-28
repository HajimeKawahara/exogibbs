"""Exact feasible-primal reconstruction and shared-plane binding for new models."""
import copy
from decimal import Decimal
from fractions import Fraction
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[3]/'examples/metal_silicate'
sys.path.insert(0,str(HERE))
try:
    W=importlib.import_module('m2_water_global')
    C=importlib.import_module('m2_extended_common_plane')
    A=importlib.import_module('m2_extended_alloy')
finally:
    sys.path.pop(0)


def water_fixture():
    p={'dissolved_h2_cost_rt':4.,'water_gas_cost_rt':2.,'log_capacity_prefactor':0.,
       'predictor_oxide_counts':[1.,1.],'predictor_temperature_weights_K':[0.,0.],
       'temperature_K':2000.,'water_mass_kg_mol':1.,'dry_masses_kg_mol':[1.,1.],
       'dry_oxygen_counts':[2.,1.],'dry_standard_costs_rt':[2.,3.],
       'entropy_coefficient':1.,'quadratic_matrix_rt':[[0.,0.],[0.,0.]]}
    b={'minimum_atoms_per_bound_unit_exact':'2','complete_element_supported_provider_domain':True,
       'external_h2o_melts_standard_offset_rt':0.,'unmodified_provider_expression':{'water_index':2},
       'active_dry_component_indices':[0,1]}
    properties={'component_order':['SiO2','MgO','H2O'],
                'basis':{'element_order':['Si','Mg','O','H'],
                         'component_element_matrix':[[1,0,2,0],[0,1,1,0],[0,0,1,2]]}}
    plane={'Si':.1,'Mg':.2,'O':.5,'H':-1.}
    return p,b,properties,plane


def test_water_primal_reconstructs_the_primitive_scalar():
    p,b,properties,plane=water_fixture()
    dry=np.array([1.1,.7]);water=.2;h2=.1
    values=[Fraction(float(v)) for v in [*dry,water]]
    actual=C.water_primal_energy(p,b,properties,plane,values,Fraction(h2))
    ideal=lambda n:sum(v*np.log(v/sum(n)) for v in n if v)
    dry_standards=np.array(p['dry_standard_costs_rt'])+np.array([1.1,.7])
    expected=dry@dry_standards+ideal(dry)+water*.5+2*ideal([sum(dry),water])
    expected+=h2*2.+ideal([sum(dry)+water,h2])
    assert abs(float(actual.lo)-expected)<1e-13
    assert float(actual.hi-actual.lo)<1e-40


def test_water_bound_uses_the_identical_plane_coefficients_and_all_retained_boxes():
    p,b,_,_=water_fixture()
    proof=W.certify_water_common_plane(p,[1.,1.],max_nodes=5)
    proof['binding']=b
    lower,atoms,details=C.water_common_plane(p,b,proof)
    assert lower.lo==Decimal(proof['lower_bound_rt_per_dry_component'])
    assert atoms.lo==2 and details['bound_unit']=='mole of dry declared components'
    changed=copy.deepcopy(p);changed['water_gas_cost_rt']+=.1
    with pytest.raises(ValueError,match='exact source'):
        C.water_common_plane(changed,b,proof)
    changed=copy.deepcopy(proof);changed['lower_bound_rt_per_dry_component']='0'
    with pytest.raises(ValueError,match='summary'):
        C.water_common_plane(p,b,changed)


def test_water_primal_keeps_the_actual_oxygen_capacity():
    p,b,properties,plane=water_fixture()
    with pytest.raises(ValueError):
        C.water_primal_energy(p,b,properties,plane,[Fraction(1),Fraction(1),Fraction(4)],Fraction(1))


def alloy_fixture():
    x=np.array([.7,.15,.15]);epsilon=100.
    mu=np.log(x)-epsilon*x[1]*x[2];mu[1]+=epsilon*x[2];mu[2]+=epsilon*x[1]
    standards=-mu
    names=['Fe_metal','O_metal','FeO_metal']
    formulas=[{'Fe':1.},{'O':1.},{'Fe':1.,'O':1.}]
    context={'excess':lambda y:epsilon*y[0]*y[1], 'standards':list(map(W._I,standards)),
             'saved_lo':np.array([.4,0.,0.]),'saved_hi':np.array([1.,.3,.3]),
             'proof_lo':np.array([.4,0.,0.]),'kappa':3.,'shifts':list(map(W._I,[100.,100.])),
             'model':SimpleNamespace(reference_model_id='test_scalar'),
             'provider':SimpleNamespace(COMPONENTS=['Fe','O','FeO'],FORMULAS=formulas),
             'names':names,'row':{'provider_recipe_file_sha256':{'fixture':'pinned'}}}
    source={'temperature_K':2000.,'pressure_bar':1.,
            'source_internal_record':{'elements':['Fe','O'],'phases':{'metal':names},
                                      'component_formulas':dict(zip(names,formulas))},
            'source_internal_result':{'accepted':True,'elemental_potentials_rt':[0.,0.],
                                     'component_amounts_mol':x.tolist(),'reduced_potentials_rt':[0.,0.,0.]}}
    return context,source


def test_saved_associated_primal_uses_species_atoms_and_preserves_negative_bounds():
    context,source=alloy_fixture()
    result=A.certify_saved_alloy(source,context,max_nodes=101)
    assert result['component_formulas'][-1]=={'Fe':1.,'O':1.}
    assert Decimal(result['source_insertion_interval_rt'][0])<Decimal('1e-13')
    assert Decimal(result['lower_bound_rt'])<-.1
    assert Decimal(result['source_search_error_upper_rt'])>.1
    assert not result['empirical_material_certified']


def test_changed_linear_standards_cannot_reuse_saved_source_potentials():
    context,source=alloy_fixture()
    context['standards'][0]+=W._I(.01)
    with pytest.raises(ValueError,match='potentials differ'):
        A.certify_saved_alloy(source,context,max_nodes=21)


def test_exact_primal_domain_is_not_relaxed_with_the_lower_bound():
    context,_=alloy_fixture()
    context['saved_lo'][0]=.86
    with pytest.raises(ValueError,match='outside'):
        A.alloy_energy_interval(context,[Fraction('.7'),Fraction('.15'),Fraction('.15')])


@pytest.mark.parametrize('kind', ['phosphorus','associated','associated_k'])
def test_actual_provider_recipe_replays_the_selected_scalar_and_all_potentials(tmp_path, kind):
    eos=pytest.importorskip('exoeos')
    root=Path(eos.__file__).resolve().parents[2]
    if not (root/'examples/m2_material/potassium_reference.py').is_file():
        pytest.skip('Requires the complete finite-alloy provider checkout.')
    sys.path.insert(0,str(HERE))
    try:
        source_module=importlib.import_module('m2_expanded_source')
        finite=importlib.import_module('m2_finite_gas')
        standards_module=importlib.import_module('run_bse_common_gibbs')
    finally:
        sys.path.pop(0)
    options={'potassium_standard_offset_rt':-8.} if kind=='associated_k' else {}
    record,_,callbacks,initial,metadata=source_module.build_expanded_bse_problem(
        root/'examples/m2_material/bse_inventory.json',root,tmp_path,sys.executable,
        gas_model='janaf_condensed',metal_model=kind,initialization='canonical',**options)
    setup=finite.build_atmosphere_setup('janaf_condensed')
    reference,_=standards_module.source_standards_rt(2173.15,1.)
    gauge=finite.atmosphere_gauge_rt(setup,2173.15,reference)
    gas=np.asarray(setup.gas_setup.hvector_func(2173.15))+np.asarray(setup.gas_setup.formula_matrix).T@gauge
    names=[name for phase in record['phases'].values() for name in phase]
    selected=record['phases']['metal']
    x=np.full(len(selected),1e-7)
    x[1:5]=[.003,.002,.015,.003]
    if len(x)>5:x[8]=.04
    if len(x)==19:x[-1]=.005
    x[0]=1.-sum(x[1:])
    actual=callbacks['metal'](2173.15,1.,x)
    amounts=np.zeros(len(names));mu=np.zeros(len(names))
    for i,name in enumerate(selected):
        amounts[names.index(name)]=x[i]
        mu[names.index(name)]=actual.mu_rt[i]
    source={'source_metadata':metadata,'temperature_K':2173.15,'pressure_bar':1.,
            'source_internal_record':record,
            'source_atmosphere_parcel':{'gas_species':list(setup.gas_species),
                                      'gas_standard_potentials_rt':gas.tolist()},
            'source_internal_result':{'accepted':True,'elemental_potentials_rt':[0.]*len(record['elements']),
                                     'component_amounts_mol':amounts.tolist(),'reduced_potentials_rt':mu.tolist()}}
    context=A.load_saved_alloy(source,root)
    energy=A.alloy_energy_interval(context,list(map(Fraction,x)))
    assert abs(float(energy.lo)-actual.gibbs_rt)<1e-11
    proof=A.certify_saved_alloy(source,context,max_nodes=1)
    assert Decimal(proof['maximum_saved_source_potential_difference_rt'])<Decimal('1e-10')
    assert proof['lower_bound_domain_relaxation']==(kind=='phosphorus')
    assert Decimal(proof['lower_bound_rt'])<=energy.hi
    assert metadata['numerical_execution']['native_melt_calls']==0
    changed=copy.deepcopy(source)
    changed['source_atmosphere_parcel']['gas_standard_potentials_rt'][list(setup.gas_species).index('P1')]+=.1
    with pytest.raises(ValueError,match='finite-P standard'):
        A.load_saved_alloy(changed,root)
