"""Bind reconstructed-water proofs to a common elemental plane and primal."""
from decimal import Decimal
from fractions import Fraction

from m2_common_plane import interval_json
from m2_liquid_global import _I
from m2_water_global import water_insertion_value


def water_parameter_snapshot(parameters):
    """Use the exact interval and binary64 declaration saved by the verifier."""
    return {key: [interval_json(_I(v)) for v in value] if key == 'dry_standard_costs_rt' else
            interval_json(value) if isinstance(value, _I) else value for key,value in parameters.items()}


def water_common_plane(parameters, binding, proof):
    """Use the verified full dry simplex plus analytically unlimited volatiles."""
    if (proof.get('assessment_id') != 'reconstructed_water_common_plane_global_bound_v1'
            or proof.get('complete_simplex_coverage') is not True
            or proof['parameters'] != water_parameter_snapshot(parameters)
            or proof['binding'] != binding):
        raise ValueError('The water proof does not use the exact source, plane and expression.')
    boxes = proof['proof_leaves']+proof['unresolved_boxes']
    if any('lower_bound_rt' not in row for row in boxes):
        raise ValueError('Every retained water box requires its lower bound.')
    bounds = [Decimal(row['lower_bound_rt']) for row in boxes]
    recorded = Decimal(proof['lower_bound_rt_per_dry_component'])
    if (not bounds or not recorded.is_finite() or any(not v.is_finite() for v in bounds)
            or min(bounds) != recorded):
        raise ValueError('The water summary differs from its complete retained lower bounds.')
    atoms = Fraction(binding['minimum_atoms_per_bound_unit_exact'])
    if atoms <= 0 or not binding['complete_element_supported_provider_domain']:
        raise ValueError('A complete elemental support and positive atom conversion are required.')
    return _I(recorded), _I(atoms), {
        'bound_unit':'mole of dry declared components',
        'volatile_domain':'all nonnegative dissolved H2; oxygen capacity relaxed only for the lower bound',
        'original_plane_strict_nonnegative': recorded >= 0,
        'water_standard_offset_rt':binding['external_h2o_melts_standard_offset_rt']}


def water_primal_energy(parameters, binding, properties, plane, amounts, dissolved_h2, dissolved_he=0):
    """Enclose G of the exact finite, atom-repaired wet-host composition."""
    values = [Fraction(v) for v in amounts]
    h2 = Fraction(dissolved_h2)
    expression = binding['unmodified_provider_expression']
    active = binding['active_dry_component_indices']
    water = expression['water_index']
    if (len(values) != len(properties['component_order']) or h2 < 0 or any(v < 0 for v in values)
            or any(v for i,v in enumerate(values) if i not in [*active,water])):
        raise ValueError('The feasible primal uses an unsupported water-host component.')
    from m2_helium_global import undo_helium_elimination
    original, helium_energy = undo_helium_elimination(parameters,binding,[values[i] for i in active],dissolved_he)
    insertion = water_insertion_value(original, [values[i] for i in active], values[water], h2)
    basis = properties['basis']
    columns = basis['component_element_matrix']
    elements = basis['element_order']
    if len(columns) != len(values) or any(len(col) != len(elements) for col in columns):
        raise ValueError('The water provider atom matrix has changed shape.')
    work = sum((_I(n)*sum((_I(Fraction(c))*_I(float(plane.get(e,0.)))
                for e,c in zip(elements,col)),_I(0)) for n,col in zip(values,columns)),_I(0))
    work += _I(2*h2)*_I(float(plane['H']))
    return insertion+work+helium_energy
