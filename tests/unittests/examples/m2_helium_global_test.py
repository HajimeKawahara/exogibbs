"""He lower-bound elimination never replaces the finite feasible-primal scalar."""
import copy
import hashlib
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

from m2_water_binding_test import fixture, runner

DIRECTORY=Path(__file__).resolve().parents[3]/'examples/metal_silicate'
sys.path.insert(0,str(DIRECTORY))
try:
    H=importlib.import_module('m2_helium')
    G=importlib.import_module('m2_helium_global')
    C=importlib.import_module('m2_extended_common_plane')
    F=importlib.import_module('m2_finite_gas')
finally:
    sys.path.pop(0)


def helium_fixture(monkeypatch):
    eos=pytest.importorskip('exoeos')
    root=Path(eos.__file__).resolve().parents[2]
    if not (root/'examples/m2_material/helium_dissolution.py').exists():
        pytest.skip('Requires the declared He provider checkout.')
    report,audit,source=fixture(monkeypatch)
    bare=runner.saved_water_problem(report,audit,'source-digest')
    record=source['source_internal_record']
    selected=record['phases']['silicate']
    setup=F.build_atmosphere_setup('janaf')
    raw=np.asarray(setup.gas_setup.hvector_func(2000.))
    standard=float(raw[list(setup.gas_species).index('He1')])
    provider=H._provider(root)
    model=provider.make_helium_dissolution('guillot2012_morb',2000.,[.03,.04,0.,0.],standard)
    receipt={**model.receipt,'temperature_K':2000.,'pressure_bar':1.,'pressure_Pa':1e5,
             'host_component_order':selected[:],'component_order':selected+['He_dissolved'],
             'coupling_file_sha256':hashlib.sha256((DIRECTORY/'m2_helium.py').read_bytes()).hexdigest(),
             'gas_anchor':{'species':'He1','temperature_K':2000.,'pressure_standard_bar':1.,
                           'retained_raw_standard_rt':standard,'common_standard_rt':standard,
                           'elements':list(setup.elements),'element_gauge_rt':[0.]*len(setup.elements),
                           'formula':[float(e=='He') for e in setup.elements]}}
    record['elements'].append('He')
    report['inventory']['total_element_amounts_mol'].append(.001)
    report['inventory']['atomic_masses_kg_mol']=[.01,.01,.001,.004]
    selected.append('He_dissolved')
    record['component_formulas']['He_dissolved']={'He':1.}
    result=source['source_internal_result']
    result['component_amounts_mol'].append(.001)
    result['elemental_potentials_rt'].append(0.)
    source['source_metadata']['helium_dissolution']=receipt
    source['gas_model']='janaf'
    source['source_atmosphere_parcel']={'elements':list(setup.elements),'gas_species':list(setup.gas_species),
        'element_gauge_rt':[0.]*len(setup.elements),'gas_standard_potentials_rt':raw.tolist()}
    state=model.state(result['component_amounts_mol'])
    audit['host_stability']['helium_dissolution']={'receipt':copy.deepcopy(receipt),
        'native_dissolved_helium_moles':.001,'native_additional_gibbs_rt':state['gibbs_rt'],
        'native_host_mu_correction_rt':state['mu_rt'][:3].tolist(),'helium_mu_rt':float(state['mu_rt'][-1])}
    return root,report,audit,source,bare,model


def test_actual_he_scalar_replay_and_elimination_keep_the_finite_upper_bound(monkeypatch):
    root,report,audit,source,bare,model=helium_fixture(monkeypatch)
    prepared,_,binding=runner.saved_water_problem(report,audit,'source-digest',exoeos_checkout=root)
    old_parameters,_,old_binding=bare
    masses=binding['helium_elimination']['dry_masses_kg_mol']
    assert masses==[.03,.04]
    properties=audit['host_stability']['provider_properties']
    plane=binding['supporting_element_potentials_rt']
    actual=C.water_primal_energy(prepared,binding,properties,plane,[1.,1.,.1],.05,.001)
    original=C.water_primal_energy(old_parameters,old_binding,properties,plane,[1.,1.,.1],.05)
    expected=original+G._I(model.state([1.,1.,.1,.05,.001])['gibbs_rt'])
    assert abs(float(actual.lo-expected.lo))<1e-14
    assert float(actual.hi-actual.lo)<1e-40
    capacity=model.receipt['capacity']['He_mol_per_kg_dry_host_per_bar_fugacity']
    optimal=.07*capacity*np.exp(-model.receipt['gas_standard_rt'])
    eliminated=sum(float(G._I(a).lo-G._I(b).lo) for a,b in zip(prepared['dry_standard_costs_rt'],old_parameters['dry_standard_costs_rt']))
    assert model.state([1.,1.,.1,.05,optimal])['gibbs_rt']==pytest.approx(eliminated,rel=1e-12)
    # A finite nonoptimal He amount is never replaced by its smaller minimum.
    assert model.state([1.,1.,.1,.05,.001])['gibbs_rt']>eliminated
    zero=C.water_primal_energy(prepared,binding,properties,plane,[1.,1.,.1],.05,0.)
    assert abs(float(zero.lo-original.lo))<1e-40


@pytest.mark.parametrize('change',[
    lambda r,a,s:a['host_stability']['helium_dissolution'].update(native_dissolved_helium_moles=.002),
    lambda r,a,s:s['source_atmosphere_parcel']['gas_standard_potentials_rt'].__setitem__(
        s['source_atmosphere_parcel']['gas_species'].index('He1'),.1),
    lambda r,a,s:s['source_metadata']['helium_dissolution'].update(gas_standard_rt=.1),
    lambda r,a,s:a['host_stability']['helium_dissolution'].update(native_host_mu_correction_rt=[0.,0.,0.]),
    lambda r,a,s:s['source_internal_record']['component_formulas'].update(He_dissolved={'He':2.}),
    lambda r,a,s:r['inventory']['atomic_masses_kg_mol'].__setitem__(0,.02),
])
def test_changed_he_receipt_or_actual_ledger_is_rejected(monkeypatch,change):
    root,report,audit,source,_,_=helium_fixture(monkeypatch)
    change(report,audit,source)
    with pytest.raises(ValueError):
        runner.saved_water_problem(report,audit,'source-digest',exoeos_checkout=root)


def test_undeclared_or_negative_helium_cannot_enter_the_primal(monkeypatch):
    _,_,_,_,bare,_=helium_fixture(monkeypatch)
    with pytest.raises(ValueError,match='undeclared'):
        G.undo_helium_elimination(bare[0],bare[2],[1.,1.],.001)
    with pytest.raises(ValueError,match='negative'):
        G.undo_helium_elimination(bare[0],bare[2],[1.,1.],-.001)
