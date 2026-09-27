"""Independently recount archived insertion and finite splitting energies."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from scipy.special import xlogy


def audit(directory):
    directory = Path(directory)
    host = json.loads((directory / 'fresh_host_assessment.json').read_text())
    props = host['provider_properties']
    report = host['assessment']
    rt = props['basis']['common_R_J_mol_K'] * props['T_K']
    mu = np.asarray(props['mu_RT'], dtype=float)
    n = np.asarray(props['component_moles'])
    h = report['native_dissolved_h2_moles']
    host_matrix = np.asarray(props['basis']['component_oxide_matrix'])
    atoms_matrix = np.asarray(props['basis']['component_element_matrix'])
    masses = np.asarray(props['oxide_molar_masses_g_mol'])
    errors, covered, failures, fixed, checked = [], [], 0, 0, 0
    for path in sorted(directory.glob('phase_*.json')):
        phase = json.loads(path.read_text())['phases'][0]
        assert not phase['minimum_certified'] and phase['lower_bound_rt'] is None
        trials = phase.get('trials', [])
        if phase.get('composition_minimum_enumerated'):
            assert phase['composition_domain_is_single_point']
            assert np.asarray(phase['native_basis']['native_endmember_oxide_mass_g_per_mol']).shape[1] == 1
            fixed += 1
        if phase.get('initial_points_attempted') == phase.get('initial_points_requested'):
            covered.append(phase['phase'])
        failures += len(phase.get('failed_evaluations', []))
        for trial in trials:
            native = trial['provider_candidate']
            oxide = np.asarray(native['oxide_mass_g'])
            np.testing.assert_allclose(oxide, native['returned_oxide_mass_g'], rtol=5e-9, atol=1e-9)
            c = np.linalg.solve(host_matrix.T, oxide / masses)
            used = c != 0
            assert not np.any((c > 0) & (n == 0)) and np.all(np.isfinite(mu[used]))
            atoms = atoms_matrix.T @ c
            value = (native['gibbs_J'] / rt - c[used] @ mu[used]
                     + c.sum() * np.log1p(h / n.sum())) / atoms.sum()
            errors.append(abs(value - trial['objective_rt']))
            checked += 1
        best = phase.get('best_fresh_trial')
        if best is not None:
            assert best['fresh_final'] and best == trials[-1]
            assert best['coordinates'] == min(trials[:-1], key=lambda r: r['objective_rt'])['coordinates']
    curvature = json.loads((directory / 'local_curvature.json').read_text())
    assert not curvature['global_stability_certified'] and not curvature['curvature_bound_certified']
    parent = np.asarray(curvature['parent_component_moles'])
    denominator = float((atoms_matrix.T @ parent[:-1]).sum() + 2 * parent[-1])
    energies = []
    for evaluation in curvature['provider_evaluations']:
        n_local = np.asarray(evaluation['native_component_moles'])
        h_local = evaluation['dissolved_h2_moles']
        N = n_local.sum()
        p = evaluation['provider_properties']
        assert p['T_K'] == props['T_K'] and p['P_Pa'] == props['P_Pa']
        np.testing.assert_allclose(p['returned_component_moles'], n_local, rtol=5e-9, atol=0)
        energy = p['gibbs_J'] / rt + xlogy(N, N / (N + h_local)) + xlogy(h_local, h_local / (N + h_local))
        errors.append(abs(energy - evaluation['augmented_gibbs_rt']))
        energies.append(energy)
    conservation = []
    for trial in curvature['finite_split_trials']:
        a, b = trial['daughter_evaluation_indices']
        daughters = [curvature['provider_evaluations'][i] for i in (a, b)]
        total = sum((np.r_[r['native_component_moles'], r['dissolved_h2_moles']] for r in daughters))
        conservation.append(float(np.max(np.abs(total - parent))))
        value = (energies[a] + energies[b] - energies[0]) / denominator
        errors.append(abs(value - trial['objective_rt_per_mol_atoms']))
    protocol = json.loads((directory / 'protocol.json').read_text())
    assert hashlib.sha256((directory / 'source_physical_audit.json').read_bytes()).hexdigest() == protocol['source_sha256']
    assert len(covered) == len(report['candidate_order']) == 33
    assert max(errors) < 1e-10 and max(conservation) < 1e-14
    return {'accepted': True, 'insertion_trials_recounted': checked,
            'catalog_count': len(covered), 'complete_geometric_probe_attempts': len(covered),
            'failed_native_evaluations_preserved': failures, 'fixed_composition_minima_enumerated': fixed,
            'maximum_energy_recount_error_rt': max(errors),
            'maximum_finite_split_component_error_mol': max(conservation),
            'curvature_native_evaluations_recounted': len(energies),
            'scope': 'Primitive energy/composition recount only; neither global stability nor empirical calibration is certified.'}


if __name__ == '__main__':
    print(json.dumps(audit(sys.argv[1]), indent=2, allow_nan=False))
