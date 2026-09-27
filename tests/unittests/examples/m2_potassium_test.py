"""Explicit K-standard sensitivity solves finite transfer without inventing data."""
import copy
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

DIRECTORY=Path(__file__).resolve().parents[3]/'examples/metal_silicate'
sys.path.insert(0,str(DIRECTORY))
try:
    SOURCE=importlib.import_module('m2_expanded_source')
    LOCAL=importlib.import_module('local')
    COMMON=importlib.import_module('common_gibbs')
    SELECTION=importlib.import_module('phase_selection')
    STATE=importlib.import_module('full_potential').PhaseState
finally:
    sys.path.pop(0)


@pytest.fixture(scope='module')
def eos_checkout():
    eos=pytest.importorskip('exoeos')
    checkout=Path(eos.__file__).resolve().parents[2]
    if not (checkout/'examples/m2_material/potassium_reference.py').is_file():
        pytest.skip('Requires the explicit K EOS provider.')
    return checkout


def build(checkout,tmp_path,offset):
    return SOURCE.build_expanded_bse_problem(checkout/'examples/m2_material/bse_inventory.json',
        checkout,tmp_path,sys.executable,gas_model='janaf_condensed',metal_model='associated_k',
        potassium_standard_offset_rt=offset,initialization='canonical')


def test_nineteen_species_keep_the_exact_thirteen_element_initial_budget(eos_checkout,tmp_path):
    record,budget,callbacks,initial,metadata=build(eos_checkout,tmp_path,9.)
    names=[name for group in record['phases'].values() for name in group]
    formula=np.array([[record['component_formulas'][name].get(e,0) for name in names] for e in record['elements']])
    np.testing.assert_allclose(formula@initial,budget,rtol=1e-12,atol=0)
    assert len(record['phases']['metal'])==19 and record['phases']['metal'][-1]=='K_metal'
    assert metadata['potassium_metal']['gas_to_metal_standard_offset_rt']==9.
    assert not metadata['potassium_metal']['empirically_calibrated']
    lo,hi,kappa=SOURCE.associated_metal_domain(metadata)
    assert lo.shape==hi.shape==(19,)
    assert .344<kappa<.345
    assert metadata['atmosphere']['missing_paths']==[
        'Na alloy component', 'He alloy dissolution', 'He silicate dissolution']
    with pytest.raises(ValueError,match='requires an explicit'):
        build(eos_checkout,tmp_path,None)


def test_k_standard_drives_an_actual_finite_partition_without_changing_atoms(eos_checkout,tmp_path):
    results=[]
    for offset in (8.,9.):
        record,_,callbacks,_,metadata=build(eos_checkout,tmp_path,offset)
        record=copy.deepcopy(record)
        record['phases']={'metal':record['phases']['metal'],'gas':['K_gas']}
        record['component_formulas']['K_gas']={'K':1.}
        mu_gas=metadata['potassium_metal']['gas_standard_rt']
        gas=lambda t,p,n:STATE(np.full(len(n),mu_gas),float(np.sum(n)*mu_gas))
        budget=np.zeros(len(record['elements']))
        budget[record['elements'].index('Fe')]=.98
        budget[record['elements'].index('K')]=.001
        problem=LOCAL.build_problem(record,budget,lambda t,p:np.zeros(20),phases=('metal','gas'))
        restricted=SELECTION.restrict_phase_callbacks(record,problem,{'metal':callbacks['metal'],'gas':gas})
        result=COMMON.minimize_gibbs(problem,2173.15,1.,budget,restricted,maxiter=200)
        assert result.accepted,result.audit_reasons
        values=dict(zip(problem.full_species,result.component_amounts_mol))
        assert values['K_metal']+values['K_gas']==pytest.approx(.001,abs=1e-14)
        # In the pure-Fe host limit, muK = muK0 + ln(xK).
        expected=.98*np.exp(-offset)/(1-np.exp(-offset))
        assert values['K_metal']==pytest.approx(expected,rel=1e-9)
        results.append(values['K_metal'])
    assert 0<results[1]<results[0]<.001
