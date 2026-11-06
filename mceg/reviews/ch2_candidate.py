"""Audit CH2 thermal reconstruction and evaluate separate energy/thermal candidates."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

import ch2_source_review as source
from gas_source_comparison import file_hash
from tio_candidate import _nasa

ROOT = source.ROOT
DATA = ROOT / 'tests/data/validation/ch2_candidate/20261106'
SPECIES = 'C1H2'
R, NASA_R = source.R, source.NASA_R
C2 = 6.62607015e-34 * 29979245800 / 1.380649e-23
PRIMARY, THERMAL = 'energy_only', 'reconstructed_rrho'
VARIANTS = ('current', PRIMARY, THERMAL, 'triplet_only', 'nasa_thermal')
FASTCHEM = ROOT / 'src/exogibbs/data/FastChem4/logK/logK.dat'


@lru_cache(maxsize=1)
def inputs() -> dict:
    """Load inspected state constants and unaltered printed caloric tables."""
    return json.loads((DATA / 'thermal_inputs.json').read_text())


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    """Load the immutable candidate definition."""
    return json.loads((DATA / 'candidate.json').read_text())


def build() -> dict:
    """Keep the source D0, atomic thermal gauge and fit residual distinct."""
    raw = json.loads((source.DATA / 'inputs.json').read_text())
    rows = raw['atct']['species']
    energy = 1000 * (math.fsum(n * rows[s]['hf0_kJ_mol'] for s, n in raw['composition'].items()) - rows['CH2']['hf0_kJ_mol'])
    source_rows = json.loads((source.DATA / 'comparison.json').read_text())
    old_d0 = source_rows['janaf_D0_J_mol']
    record = raw['nasa_records']['CH2']
    h0 = (record['hf298_J_mol'] - record['h298_minus_h0_J_mol']) / NASA_R
    coefficients = np.array(record['intervals'][1]['coefficients'])
    assert record['intervals'][1]['bounds_K'] == [1000., 6000.]
    coefficients[7] -= h0
    lines = FASTCHEM.read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith(SPECIES + ' '))
    old_coefficients = list(map(float, lines[index + 1].split()))
    paths = [DATA / 'thermal_inputs.json', FASTCHEM, source.DATA / 'inputs.json', source.DATA / 'comparison.json',
             source.DATA / 'atct-selected.tsv', source.DATA / 'nasa-selected.txt', source.DATA / 'sources.json']
    return {'id': 'ch2_atct_and_reconstructed_rrho_candidates_v1', 'species': SPECIES,
            'baseline_dataset': 'curated-tio2-v1', 'temperature_range_K': [1500, 4000],
            'original_source_D0_J_mol': old_d0, 'evaluated_D0_J_mol': energy, 'delta_D0_J_mol': energy - old_d0,
            'ATcT_version': '1.228', 'original_logK_coefficients': old_coefficients,
            'formal_fit_1_over_T_energy_J_mol': R * old_coefficients[0],
            'nasa_molecular_thermal_coefficients': coefficients.tolist(), 'nasa_h0_div_R_K': h0,
            'source_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in paths},
            'energy_scope': 'Regular equilibrated CH2 ground energy; same-version neutral C/H atomization cycle. Marginal species uncertainties do not give reaction covariance.',
            'thermal_scope': 'The reconstructed three-state RRHO is not a reproduction of the IUPAC printed equilibrated table. It is a finite sensitivity, not recommended for adoption. Above3000K is explicit model extension.',
            'residual_scope': 'Keep the complete literal-fit residual relative to NASA molecular/atomic thermal functions combined with the JANAF physical D0. Neither the atomic gauge nor the residual is refitted.',
            'production_adoption': False}


def _rrho(t, xp, variant='equilibrated'):
    states = inputs()['states']
    if variant == 'triplet_only':
        states = states[:1]
    elif variant == 'singlet_only':
        states = states[1:]
    elif variant != 'equilibrated':
        raise ValueError('Unknown RRHO state selection')
    offset = states[0]['origin_cm-1'] if variant == 'singlet_only' else 0.
    logs, means, cvs = [], [], []
    for state in states:
        origin = (state['origin_cm-1'] - offset) * C2
        logq = math.log(state['electronic_weight'] / state['symmetry_number'])
        logq += .5 * xp.log(math.pi * t**3 / float(np.prod(C2 * np.array(state['rotation_cm-1'])))) - origin / t
        mean, cv = 1.5 * t + origin, 1.5 + xp.zeros_like(t)
        for frequency in state['vibration_cm-1']:
            x = C2 * frequency / t
            occupation = 1 / xp.expm1(x)
            logq = logq - xp.log(-xp.expm1(-x))
            mean = mean + t * x * occupation
            cv = cv + x*x * xp.exp(-x) / xp.expm1(-x)**2
        logs.append(logq)
        means.append(mean)
        cvs.append(cv)
    logs, means, cvs = (xp.stack(a, axis=-1) for a in (logs, means, cvs))
    largest = xp.max(logs, axis=-1, keepdims=True)
    weights = xp.exp(logs - largest)
    weights = weights / xp.sum(weights, axis=-1, keepdims=True)
    logq = largest[..., 0] + xp.log(xp.sum(xp.exp(logs - largest), axis=-1))
    mean = xp.sum(weights * means, axis=-1)
    cv = xp.sum(weights * cvs, axis=-1) + xp.sum(weights * (means - mean[..., None])**2, axis=-1) / t**2
    k, h = 1.380649e-23, 6.62607015e-34
    logtrans = 1.5 * xp.log(2 * math.pi * inputs()['mass_u'] * 1.66053906660e-27 * k * t / h**2) + xp.log(k*t/100000)
    heat, entropy, cp = 2.5*t + mean, logtrans + logq + 2.5 + mean/t, 2.5 + cv
    return heat - t*entropy, -entropy, -cp/t, weights


def molecular_properties(temperature_K: float, variant: str = 'equilibrated') -> dict:
    """Return independent statistical moments, without a formation-energy offset."""
    if not math.isfinite(temperature_K) or not 150 <= temperature_K <= 4000:
        raise ValueError('Temperature outside thermal diagnostic domain')
    t = float(temperature_K)
    g, dg, ddg, weights = _rrho(t, np, variant)
    return {'G_minus_H0_J_mol': float(R*g), 'H_minus_H0_J_mol': float(R*(g-t*dg)),
            'S_J_mol_K': float(-R*dg), 'Cp_J_mol_K': float(-R*t*ddg),
            'state_fractions': np.asarray(weights).tolist()}


def _formation(t, variant, xp):
    model = load_candidate()
    a,b,c,d,e = model['original_logK_coefficients']
    current = -(a/t + b*xp.log(t) + c + d*t + e*t*t)
    if variant == 'current':
        return current
    if variant not in VARIANTS:
        raise ValueError('Unknown candidate variant')
    value = current - model['delta_D0_J_mol'] / (R*t)
    if variant in (PRIMARY, 'nasa_thermal'):
        # NASA thermal is the reference defining the retained residual, so this
        # label is mathematically identical to the energy-only control.
        return value
    thermal = 'equilibrated' if variant == THERMAL else 'triplet_only'
    return value + (_rrho(t, xp, thermal)[0] - _nasa(t, model['nasa_molecular_thermal_coefficients'], xp)[0]) / t


def formation_h(temperature_K: float, variant: str = PRIMARY) -> float:
    """Evaluate only the declared1500--4000K finite candidate interval."""
    if not math.isfinite(temperature_K) or not 1500 <= temperature_K <= 4000:
        raise ValueError('Temperature outside CH2 candidate domain')
    return float(_formation(float(temperature_K), variant, np))


def formation_h_jax(temperature_K, variant: str = PRIMARY):
    """Provide the same C2 potential for vectorization, JIT and differentiation."""
    import jax.numpy as jnp
    t = jnp.asarray(temperature_K, dtype=jnp.float64)
    valid = jnp.isfinite(t) & (t >= 1500) & (t <= 4000)
    safe = jnp.where(valid, t, 1500.)
    return jnp.where(valid, _formation(safe, variant, jnp), jnp.nan)


def reaction_properties(temperature_K: float, variant: str = PRIMARY) -> dict:
    """Reconstruct analytic G/H/S/Cp separately from automatic differentiation."""
    t = float(temperature_K)
    g = R*t*formation_h(t, variant)
    a,b,c,d,e = load_candidate()['original_logK_coefficients']
    heat = R*(-a+b*t+d*t*t+2*e*t**3)
    entropy = R*(b*math.log(t)+b+c+2*d*t+3*e*t*t)
    cp = R*(b+2*d*t+6*e*t*t)
    if variant != 'current':
        heat -= load_candidate()['delta_D0_J_mol']
    if variant in (THERMAL, 'triplet_only'):
        state = 'equilibrated' if variant == THERMAL else 'triplet_only'
        delta = np.array(_rrho(t, np, state)[:3]) - np.array(_nasa(t, load_candidate()['nasa_molecular_thermal_coefficients'], np))
        heat += R*(delta[0]-t*delta[1])
        entropy -= R*delta[1]
        cp -= R*t*delta[2]
    return {'G_J_mol': g, 'H_J_mol': float(heat), 'S_J_mol_K': float(entropy), 'Cp_J_mol_K': float(cp)}


def validate() -> dict:
    """Expose the failed source-table reconstruction as well as numerical checks."""
    import jax
    from exogibbs.presets import curated_tio2
    jax.config.update('jax_enable_x64', True)
    assert build() == load_candidate()
    ts = np.arange(1500., 4001.)
    base = curated_tio2.chemsetup(ions=True)
    current = np.asarray(base.hvector_func(ts))[:, base.species.index(SPECIES)]
    current_error = float(np.max(abs(current - [formation_h(t, 'current') for t in ts])))
    assert current_error < 2e-12
    errors = {}
    for variant in VARIANTS:
        exact = np.array([list(reaction_properties(t, variant).values()) for t in ts])
        potential = lambda t: R*t*formation_h_jax(t, variant)
        g = np.asarray(jax.jit(jax.vmap(potential))(ts))
        dg = np.asarray(jax.jit(jax.vmap(jax.grad(potential)))(ts))
        ddg = np.asarray(jax.jit(jax.vmap(jax.grad(jax.grad(potential))))(ts))
        actual = np.stack((g, g-ts*dg, -dg, -ts*ddg), axis=-1)
        errors[variant] = dict(zip(reaction_properties(1500).keys(), np.max(abs(actual-exact), axis=0).tolist()))
    audit = {}
    for label, rows in inputs()['iupac2005']['tables'].items():
        rows_out = []
        for t, cp, entropy, heat in rows:
            calculated = molecular_properties(t, label)
            printed = {'Cp_J_mol_K': cp, 'S_J_mol_K': entropy, 'H_minus_H0_J_mol': heat*1000,
                       'G_minus_H0_J_mol': heat*1000-t*entropy}
            rows_out.append({'temperature_K': t, 'printed': printed, 'calculated': calculated,
                             'calculated_minus_printed': {key: calculated[key]-value for key,value in printed.items()}})
        audit[label] = {'rows': rows_out, 'maximum_absolute_difference': {
            key: max(abs(row['calculated_minus_printed'][key]) for row in rows_out) for key in rows_out[0]['printed']}}
    recombined = []
    tables = inputs()['iupac2005']['tables']
    gap = inputs()['states'][1]['origin_cm-1'] * C2 * R
    for triplet, singlet, printed_eq in zip(tables['triplet_only'], tables['singlet_only'], tables['equilibrated']):
        t = triplet[0]
        assert t == singlet[0] == printed_eq[0]
        hs = np.array([triplet[3]*1000, singlet[3]*1000+gap])
        ss = np.array([triplet[2], singlet[2]])
        cps = np.array([triplet[1], singlet[1]])
        logs = -(hs-t*ss)/(R*t)
        weights = np.exp(logs-logs.max()); weights /= weights.sum()
        g = -R*t*(logs.max()+math.log(np.exp(logs-logs.max()).sum()))
        heat = float(weights @ hs)
        cp = float(weights @ cps + weights @ ((hs-heat)**2)/(R*t*t))
        entropy = (heat-g)/t
        model = molecular_properties(t)
        recombined.append({'temperature_K': t, 'from_printed_components': {'G_minus_H0_J_mol': g,
            'H_minus_H0_J_mol': heat, 'S_J_mol_K': entropy, 'Cp_J_mol_K': cp},
            'minus_reconstructed_RRHO': {'G_J_mol': g-model['G_minus_H0_J_mol'],
                'H_J_mol': heat-model['H_minus_H0_J_mol'], 'S_J_mol_K': entropy-model['S_J_mol_K'],
                'Cp_J_mol_K': cp-model['Cp_J_mol_K']},
            'minus_printed_equilibrated': {'G_J_mol': g-(printed_eq[3]*1000-t*printed_eq[2]),
                'H_J_mol': heat-printed_eq[3]*1000, 'S_J_mol_K': entropy-printed_eq[2],
                'Cp_J_mol_K': cp-printed_eq[1]}})
    samples = []
    for t in (1500., 2000., 2500., 3000., 4000.):
        samples.append({'temperature_K': t, 'thermal': molecular_properties(t),
                        'K_over_current': {v: math.exp(formation_h(t, 'current')-formation_h(t, v)) for v in VARIANTS},
                        'reaction': {v: reaction_properties(t, v) for v in VARIANTS}})
    return {'temperature_count': len(ts), 'current_h_maximum_error': current_error,
            'jax_float64_derivative_errors': errors, 'table_audit': audit, 'printed_component_recombination': recombined, 'samples': samples,
            'NASA_control_identity': 'With the unchanged residual defined against the NASA thermal proxy, nasa_thermal equals energy_only exactly; no duplicate solves are counted.',
            'decision': 'Prefer the evaluated energy-only candidate; do not recommend reconstructed RRHO adoption because the published equilibrium table is not reproduced and high-temperature nonrigidity is unresolved.',
            'curated_change': 'none', 'physical_accuracy': 'unknown'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if args.write:
        (DATA / 'candidate.json').write_text(json.dumps(build(), indent=2)+'\n')
        (DATA / 'validation.json').write_text(json.dumps(validate(), indent=2, allow_nan=False)+'\n')
    else:
        print(json.dumps(validate(), indent=2, allow_nan=False))
