"""Reconstruct a limited published sulfur isomer RRHO set; no production model."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import platform

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = 8.31446261815324
KB, PLANCK, LIGHT = 1.380649e-23, 6.62607015e-34, 299792458.0
AVOGADRO, ELECTRON_VOLT, ATOMIC_MASS = 6.02214076e23, 1.602176634e-19, 1.66053906660e-27


def state_properties(record: dict, temperature: float, mass_u: float) -> dict:
    """Return nonlinear RRHO excitation properties using declared numerical facts."""
    positions = np.array(record['positions_angstrom'], dtype=float)
    positions = (positions-positions.mean(axis=0))*1e-10
    mass = mass_u*ATOMIC_MASS
    inertia = sum(mass*(np.dot(r,r)*np.eye(3)-np.outer(r,r)) for r in positions)
    moments = np.linalg.eigvalsh(inertia)
    frequencies = np.sort(record['all_frequencies_cm-1'])
    count = 3*len(positions)-6
    assert len(frequencies) == 3*len(positions)
    assert np.max(np.abs(frequencies[:-count])) < 2
    frequencies = frequencies[-count:]
    assert np.all(frequencies > 0) and np.all(moments > 0)
    theta = PLANCK*LIGHT*100*frequencies/KB
    x = theta/temperature
    vibrational_ln_q = -float(np.log(-np.expm1(-x)).sum())
    vibrational_h = R*float((theta/np.expm1(x)).sum())
    vibrational_cp = R*float((x*x*np.exp(-x)/(-np.expm1(-x))**2).sum())
    rotational_ln_q = (0.5*math.log(math.pi)-math.log(record['rotational_symmetry_number'])
        +1.5*math.log(8*math.pi**2*KB*temperature/PLANCK**2)+0.5*float(np.log(moments).sum()))
    return {'ln_q_excitation': rotational_ln_q+vibrational_ln_q,
            'vibrational_H_J_mol': vibrational_h, 'vibrational_Cp_J_mol_K': vibrational_cp,
            'ZPE_J_mol': 0.5*R*float(theta.sum()), 'moments_kg_m2': moments.tolist(),
            'vibrations_cm-1': frequencies.tolist()}


def properties(inputs: dict, temperature: float, selected: list[str] | None = None) -> dict:
    """Sum only the explicitly listed states with one electronic component each."""
    if not 250 <= temperature <= 4000:
        raise ValueError('Diagnostic temperature outside 250-4000 K')
    records = inputs['states']
    states = {name: state_properties(row,temperature,inputs['atomic_mass_u']) for name,row in records.items()}
    electronic_min = min(row['electronic_energy_eV'] for row in records.values())
    origins = {name: (row['electronic_energy_eV']-electronic_min)*ELECTRON_VOLT*AVOGADRO
                    +states[name]['ZPE_J_mol'] for name,row in records.items()}
    origin_min = min(origins.values())
    offsets = {name: value-origin_min for name,value in origins.items()}
    names = list(records) if selected is None else selected
    logs = np.array([states[name]['ln_q_excitation']-offsets[name]/(R*temperature) for name in names])
    maximum = float(logs.max())
    ln_q = maximum+math.log(float(np.exp(logs-maximum).sum()))
    weights = np.exp(logs-ln_q)
    energies = np.array([offsets[name]+states[name]['vibrational_H_J_mol'] for name in names])
    average = float(weights@energies)
    cp_vib = float(weights@np.array([states[name]['vibrational_Cp_J_mol_K'] for name in names]))
    cp_mixing = float(weights@((energies-average)**2))/(R*temperature**2)
    molecule_mass = inputs['atoms_per_molecule']*inputs['atomic_mass_u']*ATOMIC_MASS
    ln_translation = 1.5*math.log(2*math.pi*molecule_mass*KB*temperature/PLANCK**2)
    ln_translation += math.log(KB*temperature/inputs['pressure_Pa'])
    enthalpy = 4*R*temperature+average
    entropy = R*(ln_translation+ln_q+4)+average/temperature
    return {'H_above_lowest_archived_E0_J_mol': enthalpy, 'S_J_mol_K': entropy,
            'Cp_J_mol_K': 4*R+cp_vib+cp_mixing, 'isomer_mixing_Cp_J_mol_K': cp_mixing,
            'internal_ln_Q': ln_q, 'conditional_state_fractions': dict(zip(names,weights.tolist())),
            'state_offsets_E0_J_mol': offsets}


def review(slug: str) -> dict:
    """Compare molecular thermal functions; no atomization-energy adoption or solve."""
    directory = ROOT/f'tests/data/validation/{slug}_state_followup/20261105'
    inputs = json.loads((directory/'inputs.json').read_text())
    old = json.loads((ROOT/f'tests/data/validation/{slug}_source_review/20261105/comparison.json').read_text())
    offsets = properties(inputs,298.15)['state_offsets_E0_J_mol']
    lowest = min(offsets,key=offsets.get)
    rows = []
    for temperature in [298.15,500.,1000.,1500.,2000.,3000.,4000.]:
        single = properties(inputs,temperature,[lowest])
        ensemble = properties(inputs,temperature)
        for value,selection in ((single,[lowest]),(ensemble,None)):
            value['H_minus_H298_J_mol'] = value['H_above_lowest_archived_E0_J_mol']-properties(inputs,298.15,selection)['H_above_lowest_archived_E0_J_mol']
        row = {'temperature_K':temperature,'lowest_archived_state':single,'listed_state_sum':ensemble,
               'listed_over_lowest_Q':math.exp(ensemble['internal_ln_Q']-single['internal_ln_Q'])}
        match = next((x for x in old['source_samples'] if x['temperature_K']==temperature),None)
        if match:
            ref=match['properties'][slug.upper()]
            row['source_molecular_comparison']={k:ref[k] for k in ['janaf_S_J_mol_K','nasa_S_J_mol_K','janaf_Cp_J_mol_K','nasa_Cp_J_mol_K']}
        rows.append(row)
    result = {'molecule':slug.upper(),'lowest_archived_E0_state':lowest,'state_count':len(inputs['states']),
            'pressure_Pa':inputs['pressure_Pa'],'atomic_mass_u':inputs['atomic_mass_u'],
            'electronic_degeneracy_policy':'One component per archived record; no extra spin or enantiomer factor.',
            'scope':inputs['scope'],'rows':rows,'curated_change':'none','new_equilibrium_solves':0}
    if 'physical_energy_comparison' in inputs:
        energy = inputs['physical_energy_comparison']
        d298 = inputs['atoms_per_molecule']*energy['janaf_atom_Hf298_J_mol']-energy['janaf_molecule_Hf298_J_mol']
        molecular_increment = properties(inputs,298.15)['H_above_lowest_archived_E0_J_mol']
        hybrid_d0 = d298-inputs['atoms_per_molecule']*energy['janaf_atom_H298_minus_H0_J_mol']+molecular_increment
        physical_tae0 = energy['W4_physical_TAE0_kcal_mol']*4184
        result['energy_diagnostic'] = {
            'W4_physical_TAE0_J_mol': physical_tae0,
            'janaf_balanced_D298_J_mol': d298,
            'janaf_intrinsic_D0_J_mol': None,
            'assumed_Jackson_molecular_H298_minus_E0_J_mol': molecular_increment,
            'hybrid_JANAF_Jackson_D0_J_mol': hybrid_d0,
            'W4_minus_hybrid_D0_J_mol': physical_tae0-hybrid_d0,
            'interpretation': 'JANAF has no molecular 0 K node or increment in the inspected table. The hybrid uses Jackson molecular thermal assumptions; it is not an intrinsic JANAF D0 or an adopted correction.'}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('molecule',choices=['s4','s6'])
    args=parser.parse_args()
    slug=args.molecule
    directory=ROOT/f'tests/data/validation/{slug}_state_followup/20261105'
    (directory/'diagnostic.json').write_text(json.dumps(review(slug),indent=2,allow_nan=False)+'\n')
    files=[Path(__file__),directory/'inputs.json',directory/'sources.json',
           ROOT/f'tests/data/validation/{slug}_source_review/20261105/comparison.json']
    sha=lambda p:sha256(p.read_bytes()).hexdigest()
    receipt={'command':f'python mceg/reviews/sulfur_state_followup.py {slug}',
             'python':platform.python_version(),'numpy':np.__version__,
             'selected_input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},
             'output_sha256':{'diagnostic.json':sha(directory/'diagnostic.json')},
             'scope':'Independent mathematical reconstruction of selected published numerical facts, not execution of author code.'}
    (directory/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
