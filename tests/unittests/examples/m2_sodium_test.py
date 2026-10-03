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


@pytest.mark.parametrize('upper', [.01, .02, .021, .03])
def test_sodium_p_bound_keeps_standards_and_has_a_matched_proof_context(builder, upper):
    build, checkout, _ = builder
    base = build()
    changed = build(phosphorus_options={'upper_mole_fraction': upper})
    row = changed[-1]['associated_metal']
    assert row['composition_basis'] == 'chemical_species_moles'
    assert row['upper_species_fractions'][4] == upper
    assert row['standards'] == base[-1]['associated_metal']['standards']
    n = np.asarray(base[-1]['associated_metal']['upper_species_fractions'])*.1
    n[0] = 1-n[1:].sum()
    a, b = (item[2]['metal'](2173.15, 1., n) for item in (base, changed))
    assert a.gibbs_rt == b.gibbs_rt
    np.testing.assert_array_equal(a.mu_rt, b.mu_rt)
    sys.path.insert(0, str(DIRECTORY))
    try:
        global_alloy = importlib.import_module('m2_associated_global')
        if upper == .03:
            # A feasible composition simplex does not guarantee that the
            # existing rectangular certificate encloses it with a valid split.
            with pytest.raises(ValueError, match='feasible Fe/Na split'):
                global_alloy.load_associated_expression(changed[-1], checkout)
            return
        context = global_alloy.load_associated_expression(changed[-1], checkout)
    finally:
        sys.path.pop(0)
    assert context['saved_hi'][4] == upper and context['kappa'] > 0
    if upper == .02:
        assert row == base[-1]['associated_metal']


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


def test_saved_sodium_proof_replays_host_recipe_and_preserves_exact_primal_atoms(builder):
    from fractions import Fraction
    from decimal import Decimal
    sys.path.insert(0, str(DIRECTORY))
    try:
        proof_module = importlib.import_module('m2_extended_alloy')
        importlib.import_module('m2_sodium_global')  # The source loader also loads this sibling explicitly.
        finite = importlib.import_module('m2_finite_gas')
        reference_module = importlib.import_module('run_bse_common_gibbs')
    finally:
        sys.path.pop(0)
    build, checkout, _ = builder
    record, _, callbacks, _, metadata = build(270.)
    setup = finite.build_atmosphere_setup('janaf_condensed')
    reference, _ = reference_module.source_standards_rt(2173.15,270.)
    gauge = finite.atmosphere_gauge_rt(setup,2173.15,reference)
    gas = np.asarray(setup.gas_setup.hvector_func(2173.15))+np.asarray(setup.gas_setup.formula_matrix).T@gauge
    names = [name for group in record['phases'].values() for name in group]
    metal = record['phases']['metal']
    x = np.asarray(metadata['associated_metal']['upper_species_fractions'])*.03
    x[0] = 1-sum(x[1:])
    actual = callbacks['metal'](2173.15,270.,x)
    amounts, potentials = np.zeros(len(names)), np.zeros(len(names))
    for i,name in enumerate(metal):
        amounts[names.index(name)] = x[i]
        potentials[names.index(name)] = actual.mu_rt[i]
    source = {'source_metadata':metadata,'temperature_K':2173.15,'pressure_bar':270.,
        'source_internal_record':record,
        'source_atmosphere_parcel':{'gas_species':list(setup.gas_species),'gas_standard_potentials_rt':gas.tolist()},
        'source_internal_result':{'accepted':True,'elemental_potentials_rt':[0.]*len(record['elements']),
            'component_amounts_mol':amounts.tolist(),'reduced_potentials_rt':potentials.tolist()}}
    context = proof_module.load_saved_alloy(source,checkout)
    energy = proof_module.alloy_energy_interval(context,list(map(Fraction,x)))
    assert abs(float(energy.lo)-actual.gibbs_rt) < 1e-11
    result = proof_module.certify_saved_alloy(source,context,max_nodes=1)
    assert Decimal(result['maximum_saved_source_potential_difference_rt']) < Decimal('1e-10')
    assert result['component_formulas'][-1] == {'Na':1.}
    assert Decimal(result['lower_bound_rt']) <= energy.hi
    changed = copy.deepcopy(source)
    changed['source_metadata']['sodium_metal']['native_liquid_properties']['mu0_RT'][0] += .1
    with pytest.raises(ValueError,match='native receipt'):
        proof_module.load_saved_alloy(changed,checkout)
    changed = copy.deepcopy(source)
    changed['source_metadata']['sodium_metal']['fe_metal_standard_rt'] += .1
    with pytest.raises(ValueError,match='sodium standard'):
        proof_module.load_saved_alloy(changed,checkout)
