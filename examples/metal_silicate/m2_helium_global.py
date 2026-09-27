"""Exact trace-He elimination and unreduced feasible-primal reconstruction."""
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
import hashlib
from pathlib import Path

import numpy as np

from m2_common_plane import _I, _dot, _exp, interval_json


def add_saved_helium(parameters, binding, source, host, exoeos_checkout):
    """Bind the EOS-owned He scalar and eliminate all nonnegative He amounts."""
    from m2_helium import reconstruct_helium_model

    metadata = source['source_metadata']
    receipt = metadata['helium_dissolution']
    audited = host['helium_dissolution']
    record = source['source_internal_record']
    selected = record['phases']['silicate']
    names = [name for phase in record['phases'].values() for name in phase]
    amounts = source['source_internal_result']['component_amounts_mol']
    if (receipt != audited['receipt'] or receipt['component_order'] != selected
            or selected[-1] != 'He_dissolved' or receipt['host_component_order'] != selected[:-1]
            or record['component_formulas']['He_dissolved'] != {'He':1.}
            or receipt['temperature_K'] != source['temperature_K']
            or receipt['pressure_bar'] != source['pressure_bar']
            or receipt['pressure_Pa'] != source['pressure_bar']*1e5
            or receipt['coupling_file_sha256'] != hashlib.sha256(Path(__file__).with_name('m2_helium.py').read_bytes()).hexdigest()):
        raise ValueError('The He scalar must match the actual source, audit and coupling recipe.')
    if exoeos_checkout is None:
        raise ValueError('The He bound requires its explicitly selected ExoEOS checkout.')
    model = reconstruct_helium_model(exoeos_checkout, receipt)
    parcel = source['source_atmosphere_parcel']
    anchor = receipt['gas_anchor']
    i = parcel['gas_species'].index('He1')
    if (anchor['elements'] != parcel['elements']
            or anchor['element_gauge_rt'] != parcel['element_gauge_rt']
            or receipt['gas_standard_rt'] != parcel['gas_standard_potentials_rt'][i]):
        raise ValueError('The He standard does not belong to the executed retained gas plane.')
    from m2_finite_gas import build_atmosphere_setup
    setup = build_atmosphere_setup(source['gas_model'])
    raw = float(np.asarray(setup.gas_setup.hvector_func(source['temperature_K']))[list(setup.gas_species).index('He1')])
    if raw != anchor['retained_raw_standard_rt']:
        raise ValueError('The retained He gas standard differs from the executed recipe.')
    scale = metadata['input']['native_amount_scale']
    native = [float(amounts[names.index(name)]*scale) for name in selected]
    replay = model.state(native)
    properties = host['provider_properties']
    correction = np.zeros(len(properties['component_order']))
    for i,name in enumerate(selected[:-1]):
        if name.endswith('_melts'):
            correction[properties['component_order'].index(name[:-6])] = replay['mu_rt'][i]
    if (native[-1] != audited['native_dissolved_helium_moles']
            or replay['gibbs_rt'] != audited['native_additional_gibbs_rt']
            or correction.tolist() != audited['native_host_mu_correction_rt']
            or (np.isfinite(replay['mu_rt'][-1]) and float(replay['mu_rt'][-1]) != audited['helium_mu_rt'])):
        raise ValueError('The independent He replay differs from the fresh host audit.')
    masses = receipt['dry_host_molar_masses_kg']
    if len(masses) != len(selected)-1 or any(not np.isfinite(v) or v < 0 for v in masses):
        raise ValueError('The He dry-mass vector must cover the original host components.')
    mass_by_name = dict(zip(selected[:-1], masses))
    if any(mass_by_name.get(name, 0.) != 0. for name in ('h2o_melts','H2_dissolved')):
        raise ValueError('The declared He host mass must exclude water and molecular H2.')
    projected = [mass_by_name[properties['component_order'][i]+'_melts']
                 for i in binding['active_dry_component_indices']]
    if len(projected) != len(parameters['dry_standard_costs_rt']) or any(value <= 0 for value in projected):
        raise ValueError('Every supported dry component requires positive He host mass.')
    capacity = receipt['capacity']['He_mol_per_kg_dry_host_per_bar_fugacity']
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError('The declared He capacity must be finite and positive.')
    plane = binding['supporting_element_potentials_rt']
    if 'He' not in plane or binding['element_budget_mol'].get('He',0.) <= 0:
        raise ValueError('The He phase requires the saved positive elemental budget and plane.')
    rate = _I(capacity)*_exp(_I(plane['He'])-_I(receipt['gas_standard_rt']))
    reduction = [_I(mass)*rate for mass in projected]
    prepared = deepcopy(parameters)
    prepared['dry_standard_costs_rt'] = [_I(cost)-shift for cost,shift in zip(parameters['dry_standard_costs_rt'],reduction)]
    declared = {**binding,'helium_elimination':{
        'receipt':receipt,'dry_masses_kg_mol':projected,
        'dry_cost_reduction_rt':[interval_json(value) for value in reduction],
        'minimum_helium_moles_per_kg_dry':interval_json(rate),
        'scope':'All nonnegative He amounts in the declared trace scalar. The original host/H2 denominator is unchanged; empirical finite-concentration accuracy is not inferred.'}}
    return prepared, declared


def undo_helium_elimination(parameters, binding, dry_amounts, helium_amount):
    """Restore bare water costs and evaluate the actual He amount for an upper bound."""
    amount = Fraction(helium_amount)
    if amount < 0:
        raise ValueError('The feasible primal cannot contain negative dissolved He.')
    row = binding.get('helium_elimination')
    if row is None:
        if amount:
            raise ValueError('An undeclared He component cannot enter the feasible primal.')
        return parameters, _I(0)
    if len(dry_amounts) != len(row['dry_masses_kg_mol']):
        raise ValueError('The He dry-host basis has changed.')
    restored = deepcopy(parameters)
    shifts = [_I(Decimal(value['lower']),Decimal(value['upper'])) for value in row['dry_cost_reduction_rt']]
    if len(shifts) != len(parameters['dry_standard_costs_rt']):
        raise ValueError('The eliminated He cost vector is incomplete.')
    restored['dry_standard_costs_rt'] = [_I(cost)+shift for cost,shift in zip(parameters['dry_standard_costs_rt'],shifts)]
    if not amount:
        return restored, _I(0)
    mass = _dot(row['dry_masses_kg_mol'],dry_amounts)
    if mass.lo <= 0:
        raise ValueError('Positive He requires positive dry-host mass.')
    receipt = row['receipt']
    capacity = _I(receipt['capacity']['He_mol_per_kg_dry_host_per_bar_fugacity'])
    energy = _I(amount)*(_I(receipt['gas_standard_rt'])+(_I(amount)/(mass*capacity)).log()-1)
    return restored, energy
