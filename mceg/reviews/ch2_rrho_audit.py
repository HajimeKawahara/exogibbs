"""Independent finite-oscillator and quantum-rigid-rotor diagnostics for CH2."""
from __future__ import annotations

import json
import math

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from scipy.special import logsumexp

import ch2_candidate as candidate


def rotor_levels(constants, maximum_j):
    """Diagonalize the asymmetric rotor in K-parity blocks; no spectral fit."""
    a,b,c = constants
    energies, weights = [], []
    for j in range(maximum_j+1):
        total = j*(j+1)
        for parity in (0,1):
            k = np.arange(-j+parity,j+1,2,dtype=float)
            if not len(k):
                continue
            diagonal = .5*(b+c)*(total-k*k)+a*k*k
            upper = .25*(b-c)*np.sqrt((total-k[:-1]*(k[:-1]+1))*(total-(k[:-1]+1)*(k[:-1]+2)))
            values = eigvalsh_tridiagonal(diagonal,upper) if len(k)>1 else diagonal
            energies.extend(values*candidate.C2)
            weights.extend([2*j+1]*len(values))
    return np.array(energies),np.array(weights,dtype=float)


def finite_thermals(t, states, rotors=None):
    """Sum oscillator quanta explicitly and combine state energy moments."""
    logs, means, variances = [], [], []
    for si,state in enumerate(states):
        origin = state['origin_cm-1']*candidate.C2
        if rotors is None:
            logq = .5*math.log(math.pi*t**3/np.prod(candidate.C2*np.array(state['rotation_cm-1'])))
            mean, variance = 1.5*t, 1.5*t*t
        else:
            energies,weights = rotors[si]
            boltzmann = weights*np.exp(-energies/t)
            norm = boltzmann.sum()
            mean = float(boltzmann@energies/norm)
            variance = float(boltzmann@((energies-mean)**2)/norm)
            logq = math.log(norm)
        logq += math.log(state['electronic_weight']/state['symmetry_number'])-origin/t
        mean += origin
        for frequency in state['vibration_cm-1']:
            energies = np.arange(129)*frequency*candidate.C2
            boltzmann = np.exp(-energies/t)
            norm = boltzmann.sum()
            vib_mean = float(boltzmann@energies/norm)
            variance += float(boltzmann@((energies-vib_mean)**2)/norm)
            mean += vib_mean
            logq += math.log(norm)
        logs.append(logq);means.append(mean);variances.append(variance)
    logq = logsumexp(logs)
    weights = np.exp(np.array(logs)-logq)
    mean = float(weights@means)
    variance = float(weights@(np.array(variances)+(np.array(means)-mean)**2))
    k,h = 1.380649e-23,6.62607015e-34
    trans = 1.5*math.log(2*math.pi*candidate.inputs()['mass_u']*1.66053906660e-27*k*t/h**2)+math.log(k*t/100000)
    heat = candidate.R*(2.5*t+mean)
    entropy = candidate.R*(trans+logq+2.5+mean/t)
    return {'G_minus_H0_J_mol':heat-t*entropy,'H_minus_H0_J_mol':heat,
            'S_J_mol_K':entropy,'Cp_J_mol_K':candidate.R*(2.5+variance/t**2)}


def audit() -> dict:
    """Bound numerical truncation while leaving physical nonrigidity unresolved."""
    states = candidate.inputs()['states']
    rotors = {j:[rotor_levels(state['rotation_cm-1'],j) for state in states] for j in (100,140)}
    rows=[]
    for t in (1500.,2000.,2500.,3000.,3500.,4000.):
        direct = candidate.molecular_properties(t)
        finite = finite_thermals(t,states)
        quantum = {j:finite_thermals(t,states,rotors[j]) for j in rotors}
        rows.append({'temperature_K':t,'finite_oscillator_minus_analytic':{key:finite[key]-direct[key] for key in finite},
            'quantum_rotor_140_minus_100':{key:quantum[140][key]-quantum[100][key] for key in finite},
            'quantum_rotor_minus_classical':{key:quantum[140][key]-finite[key] for key in finite}})
    return {'oscillator_maximum_quanta':128,'rigid_rotor_maximum_J':[100,140],'samples':rows,
        'scope':'Independent finite numerical sums verify the explicitly extended analytic RRHO model. The all-K rotor sum is divided by sigma2 as a spin-averaged control; it is not a state-resolved ortho/para treatment. Anharmonicity, the low triplet linearity barrier, Renner-Teller coupling and higher electronic states remain unvalidated.'}


if __name__=='__main__':
    print(json.dumps(audit(),indent=2,allow_nan=False))
