"""Finite Na uses the actual host-standard receipt and a separate atom column."""
import copy
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parents[3] / 'examples/metal_silicate'
sys.path.insert(0, str(DIRECTORY))
try:
    SOURCE = importlib.import_module('m2_expanded_source')
    ASSOC = importlib.import_module('m2_associated')
    LOCAL = importlib.import_module('local')
    COMMON = importlib.import_module('common_gibbs')
    SELECTION = importlib.import_module('phase_selection')
    STATE = importlib.import_module('full_potential').PhaseState
finally:
    sys.path.pop(0)
OPTIONS = {'projection': 'remove_potassium_0.25', 'temperature_policy': 'published_exchange_slope'}


@pytest.fixture
def builder(tmp_path, monkeypatch):
    eos = pytest.importorskip('exoeos')
    checkout = Path(eos.__file__).resolve().parents[2]
    if not (checkout/'examples/m2_material/sodium_metal.py').is_file():
        pytest.skip('Requires the finite sodium EOS provider.')
    original = SOURCE.build_bse_problem
    seen = []
    def synthetic_host(*args, **kwargs):
        # Use the real conserved source/catalog builder; replace only the
        # native property callback with explicitly synthetic standards.
        kwargs['liquid_model'] = 'native'
        record, budget, callbacks, initial, metadata = original(*args, **kwargs)
        receipts = []
        metadata['host_ledger'].update(liquid_model='published', native_standard_state_receipts=receipts)
        def host(t, p, n):
            seen.append((t, p, np.array(n)))
            properties = {'T_K': t, 'P_Pa': p*1e5, 'basis': {'common_R_J_mol_K': 8.31446261815324},
                'component_order': ['na2sio3', 'sio2', 'fe2sio4'], 'mu0_RT': [-12.+.01*p, -8., -5.],
                'model_id': 'synthetic_unit_test', 'provenance': {'scope': 'No native property evaluation.'}}
            receipts.append({'T_K': t, 'P_Pa': p*1e5, 'native_properties': properties})
            return STATE(np.zeros(len(n)), 0.)
        callbacks['silicate'] = host
        return record, budget, callbacks, initial, metadata
    monkeypatch.setattr(SOURCE, 'build_bse_problem', synthetic_host)
    def build(p=1., **kwargs):
        options = dict(gas_model='janaf_condensed', metal_model='associated_k_na', liquid_model='published',
                       potassium_standard_offset_rt=-.09476121345453237, sodium_options=OPTIONS,
                       initialization='canonical', pressure_bar=p)
        options.update(kwargs)
        return SOURCE.build_expanded_bse_problem(checkout/'examples/m2_material/bse_inventory.json',
                                                checkout, tmp_path, sys.executable, **options)
    return build, checkout, seen


def test_sodium_standard_reconstructs_from_exact_current_host_and_parent_fe(builder):
    build, checkout, seen = builder
    prior = None
    for pressure in (1., 270.):
        record, budget, callbacks, initial, metadata = build(pressure)
        names = [name for group in record['phases'].values() for name in group]
        matrix = np.array([[record['component_formulas'][name].get(e, 0.) for name in names] for e in record['elements']])
        np.testing.assert_allclose(matrix@initial, budget, rtol=1e-12, atol=0.)
        assert len(record['phases']['metal']) == 20 and record['phases']['metal'][-1] == 'Na_metal'
        assert record['component_formulas']['Na_metal'] == {'Na': 1.}
        row = metadata['sodium_metal']
        provider = ASSOC.load_associated_provider(checkout, potassium=True, sodium=True)
        standards = metadata['associated_metal']['standards']['standard_potentials_rt']
        value, reproduced = provider.sodium_standard_rt(2173.15, pressure*1e5, standards[0],
                                                       row['native_liquid_properties'], **OPTIONS)
        assert value == row['standard_rt'] == standards[-1]
        assert {k:v for k,v in row.items() if k != 'native_liquid_properties'} == reproduced
        assert not row['empirical_BSE_calibration']
        assert metadata['atmosphere']['missing_paths'] == ['He alloy dissolution', 'He silicate dissolution']
        lo, hi, kappa = SOURCE.associated_metal_domain(metadata)
        assert lo.shape == hi.shape == (20,) and hi[-1] == .02 and np.isfinite(kappa)
        assert 'examples/m2_material/sodium_metal.py' in metadata['associated_metal']['provider_recipe_file_sha256']
        if prior is not None:
            assert value-prior == pytest.approx(.5*.01*(pressure-1.), abs=1e-12)
        prior = value
    assert [row[1] for row in seen] == [1., 270.]


def test_twenty_species_scalar_gradient_and_actual_finite_na_partition(builder):
    build, _, _ = builder
    record, _, callbacks, _, metadata = build()
    record = copy.deepcopy(record)
    record['phases'] = {'metal': record['phases']['metal'], 'gas': ['Na_gas']}
    record['component_formulas']['Na_gas'] = {'Na': 1.}
    standard = metadata['sodium_metal']['standard_rt'] + np.log(.002)
    gas = lambda t,p,n: STATE(np.full(len(n), standard), float(np.sum(n)*standard))
    budget = np.zeros(len(record['elements']))
    budget[record['elements'].index('Fe')] = .98
    budget[record['elements'].index('Na')] = .004
    problem = LOCAL.build_problem(record, budget, lambda t,p: np.zeros(21), phases=('metal','gas'))
    providers = SELECTION.restrict_phase_callbacks(record, problem, {'metal': callbacks['metal'], 'gas': gas})
    result = COMMON.minimize_gibbs(problem, 2173.15, 1., budget, providers, maxiter=200)
    assert result.accepted, result.audit_reasons
    values = dict(zip(problem.full_species, result.component_amounts_mol))
    expected = .98*.002/(1-.002)
    assert values['Na_metal'] == pytest.approx(expected, rel=1e-9)
    assert values['Na_gas']+values['Na_metal'] == pytest.approx(.004, abs=1e-14)
    assert values['Fe_metal'] == pytest.approx(.98, abs=1e-14)


@pytest.mark.parametrize('change', [
    {'sodium_options': None}, {'sodium_options': {}}, {'sodium_options': {'projection': 'unknown', 'temperature_policy': 'constant_delta'}},
    {'sodium_options': {'projection': 'remove_potassium_0.25', 'temperature_policy': 'unknown'}},
    {'metal_model': 'associated_k'}, {'potassium_standard_offset_rt': None}, {'liquid_model': 'native'}])
def test_sodium_options_are_explicit_and_invalid_choices_never_evaluate_host(builder, change):
    build, _, seen = builder
    with pytest.raises(ValueError):
        build(**change)
    assert not seen
