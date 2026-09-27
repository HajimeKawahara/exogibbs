"""Saved finite-P/associated alloy bounds and exact feasible-primal energies."""
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from m2_associated_global import (alloy_energy_interval, certify_associated_insertion,
                                  load_associated_expression, _scalar_gradient)
from m2_liquid_global import _I, _outward_float


def _verify_phosphorus_standard(source, provider, model):
    """Replay the original common-gas host and measured finite-P anchor."""
    row = source['source_metadata']['phosphorus_metal']
    temperature = source['temperature_K']
    names = list(model.components)
    parcel = source['source_atmosphere_parcel']
    gas_name = row['standard']['gas_species']
    gas_value = parcel['gas_standard_potentials_rt'][parcel['gas_species'].index(gas_name)]
    standard = provider.phosphorus_standard_rt(
        temperature, gas_value, gas_species=gas_name, allow_temperature_continuation=True,
        measurement_shift_kcal_mol=row['standard']['measurement_shift_kcal_mol'])
    from run_bse_common_gibbs import source_standards_rt
    reference, _ = source_standards_rt(temperature, source['pressure_bar'])
    expected_standards = np.r_[[reference[name+'_metal'] for name in names[:4]], standard['standard_rt']]
    expected_standards += np.asarray(model.standard_state_shift_RT(temperature))
    if standard != row['standard'] or expected_standards.tolist() != row['base_standard_potentials_rt']:
        raise ValueError("The saved finite-P standard differs from the executed source recipe.")
    return expected_standards


def load_saved_alloy(source, exoeos_checkout):
    """Reconstruct the exact selected scalar from provider-owned expressions."""
    metadata = deepcopy(source["source_metadata"])
    kind = metadata.get("metal_model", "ma")
    if kind in ("associated", "associated_k"):
        context = load_associated_expression(metadata, exoeos_checkout)
        context["proof_lo"] = context["saved_lo"].copy()
        provider = context['provider']
        associated = provider._associated() if kind == 'associated_k' else provider
        host = context['model'].host if kind == 'associated_k' else context['model']
        base = _verify_phosphorus_standard(source, associated._phosphorus(), host.host)
        parcel = source['source_atmosphere_parcel']
        gas = dict(zip(parcel['gas_species'], parcel['gas_standard_potentials_rt']))
        options = ({'potassium_standard_offset_rt': metadata['potassium_metal']['gas_to_metal_standard_offset_rt']}
                   if kind == 'associated_k' else {})
        standards, receipt = provider.associated_standards_rt(source['temperature_K'], base, gas, **options)
        if receipt != context['row']['standards'] or standards.tolist() != receipt['standard_potentials_rt']:
            raise ValueError("The associated standards differ from the executed gas/host recipe.")
    elif kind == "phosphorus":
        root = Path(exoeos_checkout).resolve()
        row = metadata["phosphorus_metal"]
        for relative, digest in row["provider_recipe_file_sha256"].items():
            path = root/relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("The saved finite-P provider recipe has changed: "+relative)
            if relative.startswith("src/exoeos/"):
                module = __import__(relative[len('src/'):-3].replace('/', '.'), fromlist=['__file__'])
                if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
                    raise ValueError("The imported finite-P dependency differs from its receipt.")
        name = '_m2_saved_phosphorus_expression'
        path = root/'examples/m2_material/phosphorus_reference.py'
        if name in sys.modules:
            provider = sys.modules[name]
            if Path(provider.__file__).resolve() != path:
                raise ValueError("Use one fixed finite-P provider per process.")
        else:
            spec = importlib.util.spec_from_file_location(name, path)
            provider = importlib.util.module_from_spec(spec)
            sys.modules[name] = provider
            spec.loader.exec_module(provider)
        temperature = source['temperature_K']
        interactions = provider.phosphorus_interactions(
            temperature, temperature_policy=row['interactions']['temperature_policy'])
        if interactions != row['interactions'] or row['standard']['temperature_K'] != temperature:
            raise ValueError("The selected finite-P interactions or standard have changed.")
        model = provider.MaPhosphorusLiquid(np.asarray(interactions['epsilon']))
        names = list(model.components)
        if names != row['component_order'] or model.reference_model_id != row['model_id']:
            raise ValueError("The saved finite-P composition basis has changed.")
        _verify_phosphorus_standard(source, provider, model)
        lo, hi = np.asarray(row['lower_atomic_fractions'], float), np.asarray(row['upper_atomic_fractions'], float)
        if (lo.shape != (5,) or hi.shape != lo.shape or np.any(~np.isfinite(lo))
                or np.any(~np.isfinite(hi)) or np.any(lo < 0) or np.any(lo > hi)
                or np.any(hi > 1) or lo[0] <= 0 or lo.sum() > 1 or hi.sum() < 1):
            raise ValueError("Require the finite declared phosphorus composition box.")
        # The Fe constraint couples independent solutes. Enclose it by a
        # larger rectangular domain only for the lower bound; the exact
        # feasible primal still obeys the original Fe >= lower constraint.
        proof_lo = lo.copy()
        remaining = _I(1-sum(Fraction(float(v)) for v in hi[1:]))
        proof_lo[0] = min(lo[0], _outward_float(remaining.lo, True))
        if proof_lo[0] <= 0:
            raise ValueError("The rectangular proof enclosure reaches a singular Fe boundary.")
        curvature = provider.phosphorus_curvature_lower_bound(model, temperature, proof_lo, hi)
        dry = np.asarray(model.host.dry_model.interaction_K).tolist()
        epsilon = np.asarray(model.epsilon).tolist()
        def excess(values):
            return provider.phosphorus_excess(temperature, dry, epsilon, values)
        offsets = (metadata.get('provider_scenario') or {}).get('standard_offsets_rt', {})
        standards = [_I(float(value))+_I(float(offsets.get(name+'_metal', 0.)))
                     for name, value in zip(names, row['base_standard_potentials_rt'])]
        if len(standards) != 5:
            raise ValueError("The finite-P standard vector is incomplete.")
        context = {'metadata':metadata, 'row':row, 'model':model,
                   'provider':SimpleNamespace(COMPONENTS=names, FORMULAS=[{name:1.} for name in names]),
                   'temperature':temperature, 'standards':standards, 'saved_lo':lo,
                   'saved_hi':hi, 'proof_lo':proof_lo, 'kappa':curvature,
                   'shifts':[_I(0)]*4, 'excess':excess, 'offsets':offsets}
    else:
        raise ValueError("Select the explicitly declared finite-P or associated alloy model.")
    record = source['source_internal_record']
    expected_names = [name+'_metal' for name in context['provider'].COMPONENTS]
    if (record['phases']['metal'] != expected_names
            or source['temperature_K'] != context['temperature']
            or record['elements'] != metadata['input']['elements']):
        raise ValueError("The source record and declared alloy expression differ.")
    for name, formula in zip(expected_names, context['provider'].FORMULAS):
        if record['component_formulas'][name] != formula:
            raise ValueError("The saved alloy atom columns differ from the provider basis.")
    context['names'] = expected_names
    return context


def certify_saved_alloy(source, context, *, tolerance=1e-8, max_nodes=2000):
    """Bound the exact saved external elemental plane over the full alloy box."""
    record, saved = source['source_internal_record'], source['source_internal_result']
    if not saved['accepted']:
        raise ValueError("A saved unaccepted source cannot supply a phase certificate.")
    elements = record['elements']
    plane = saved['elemental_potentials_rt']
    names = [name for phase in record['phases'].values() for name in phase]
    if (len(plane) != len(elements) or not np.all(np.isfinite(plane))
            or len(saved['component_amounts_mol']) != len(names)):
        raise ValueError("The source plane and primitive ledger must be complete.")
    columns = [[record['component_formulas'][name].get(element,0.) for element in elements]
               for name in context['names']]
    costs = [standard-sum((_I(float(c))*_I(float(l)) for c,l in zip(column,plane)),_I(0))
             for standard,column in zip(context['standards'],columns)]
    report = certify_associated_insertion(context['excess'],costs,context['proof_lo'],
        context['saved_hi'],context['kappa'],context['shifts'],tolerance=tolerance,max_nodes=max_nodes)
    amounts = [Fraction(saved['component_amounts_mol'][names.index(name)]) for name in context['names']]
    if any(value < 0 for value in amounts):
        raise ValueError("The source alloy contains a negative component amount.")
    source_amount = sum(amounts)
    selected = None
    selected_gap = None
    gradient_error = None
    if source_amount:
        # This separately checks the original, unrelaxed numerical box.
        energy = alloy_energy_interval(context,amounts)
        plane_work = sum((_I(n)*sum((_I(float(c))*_I(float(l)) for c,l in zip(col,plane)),_I(0))
                          for n,col in zip(amounts,columns)),_I(0))
        selected = (energy-plane_work)/_I(source_amount)
        selected_gap = selected-_I(Decimal(report['lower_bound_rt']))
        fractions = [n/source_amount for n in amounts]
        _, gradient = _scalar_gradient(context['excess'],costs,fractions[1:],intervals=True)
        dependent = selected-sum((_I(x)*g for x,g in zip(fractions[1:],gradient)),_I(0))
        expected = [dependent, *[dependent+g for g in gradient]]
        observed = saved['reduced_potentials_rt']
        if len(observed) != len(names):
            raise ValueError("Require the source's complete reduced-potential ledger.")
        differences = [expected[i]-_I(float(observed[names.index(name)]))
                       for i,name in enumerate(context['names']) if amounts[i] > 0]
        gradient_error = max(max(abs(v.lo),abs(v.hi)) for v in differences)
        if gradient_error > Decimal('1e-9'):
            raise ValueError("The replayed alloy potentials differ from the executed source ledger.")
    report.update(assessment_id='saved_extended_alloy_common_plane_v1',
        model_id=context['model'].reference_model_id, temperature_K=source['temperature_K'],
        pressure_bar=source['pressure_bar'], component_order=list(context['provider'].COMPONENTS),
        component_formulas=context['provider'].FORMULAS, element_order=elements,
        elemental_potentials_rt=plane, saved_component_amounts_mol=[str(v) for v in amounts],
        declared_domain_lower=context['saved_lo'].tolist(), declared_domain_upper=context['saved_hi'].tolist(),
        lower_bound_domain_relaxation=not np.array_equal(context['proof_lo'],context['saved_lo']),
        source_insertion_interval_rt=None if selected is None else [str(selected.lo),str(selected.hi)],
        source_search_error_upper_rt=None if selected_gap is None else str(selected_gap.hi),
        maximum_saved_source_potential_difference_rt=None if gradient_error is None else str(gradient_error),
        provider_recipe_file_sha256=context['row']['provider_recipe_file_sha256'],
        scope='Full declared composition box; any rectangular relaxation is used only for the lower bound. The saved source point and exact atom-repaired primal retain the original domain. No empirical calibration is inferred.')
    return report
