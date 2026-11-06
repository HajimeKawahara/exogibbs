"""Run two CH2 candidates on unchanged curated-tio2-v1 ion/neutral networks."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import jax
import numpy as np
from scipy.special import logsumexp

from exogibbs.presets import curated_tio2
from c2h_equilibrium_impact import carbon_budget, independent_check
from co2_equilibrium_impact import gibbs_ordering
from curated_h2o_adoption import solve_cases
from gas_source_comparison import ROOT, file_hash
import ch2_candidate as candidate

BASELINE = ROOT / 'tests/data/validation/curated_tio2_adoption/20261106'
PRIMARY_TOLERANCE, STRICT_TOLERANCE = 5e-14, 2e-14
CANDIDATES = (candidate.PRIMARY, candidate.THERMAL)


def candidate_setup(base, variant=candidate.PRIMARY, fraction=1.0):
    """Substitute CH2 alone without changing the species or elemental budgets."""
    target = base.species.index(candidate.SPECIES)
    def hvector(t):
        h = base.hvector_func(t)
        replacement = candidate.formation_h_jax(t, variant)
        return h.at[..., target].set(h[..., target] + fraction*(replacement-h[..., target]))
    return replace(base, hvector_func=jax.jit(hvector), metadata={**base.metadata,
        'dataset': f'review-ch2-{variant}', 'production_adoption': False, 'substitution_fraction': fraction})


def fractions(values):
    return values-logsumexp(values, axis=-1, keepdims=True)


def constraints(matrix, elements, species, budget, amounts, h, pressure):
    """Check every element, charge and atom-balanced formation equation."""
    n = np.exp(amounts)
    residual = abs(n @ matrix.T-budget)
    positive = budget[0] > 0
    references = [species.index('e1-' if element == 'e-' else element+'1') for element in elements]
    logp = fractions(amounts)+np.log(pressure)[..., None]
    values = {'element_relative_residual': np.max(residual[..., positive]/budget[..., positive], axis=-1),
              'mass_action_max_abs_ln': np.max(abs(logp-logp[..., references]@matrix+h), axis=-1),
              'major_element_relative_residual': np.max(np.stack([
                  residual[..., elements.index(e)]/budget[..., elements.index(e)] for e in ('H','C','O')]), axis=0)}
    if 'e-' in elements:
        charge = matrix[elements.index('e-')]
        values['charge_relative_residual'] = abs(n@charge)/(n@abs(charge))
    limits = {'element_relative_residual': 1e-5, 'mass_action_max_abs_ln': 1e-8,
              'major_element_relative_residual': 1e-9, 'charge_relative_residual': 1e-12}
    assert np.isfinite(amounts).all() and np.all(n > 0)
    for name, value in values.items():
        assert np.isfinite(value).all() and np.max(value) < limits[name], (name, np.max(value))
    return values


def stability(actual, reference, target):
    delta = fractions(actual)-fractions(reference)
    result = {'all_max_abs_delta_ln_x': float(abs(delta).max()),
              'CH2_max_abs_delta_ln_x': float(abs(delta[..., target]).max()),
              'max_absolute_delta_x': float(abs(np.exp(fractions(actual))-np.exp(fractions(reference))).max())}
    assert result['all_max_abs_delta_ln_x'] < 1e-4
    assert result['CH2_max_abs_delta_ln_x'] < 1e-8
    assert result['max_absolute_delta_x'] < 1e-9
    return result


def run(output: Path, executable: Path) -> None:
    """Run300 fresh endpoints, finite strict/path/triplet controls and4FastChem comparisons."""
    jax.config.update('jax_enable_x64', True)
    output.mkdir(parents=True, exist_ok=False)
    summaries, diagnostics, controls, independent, baseline_checks = [], [], [], [], []
    baseline_receipt = json.loads((BASELINE/'receipt.json').read_text())
    for ions in (True, False):
        network = 'ions' if ions else 'neutral'
        path = BASELINE/f'{network}.npz'
        assert file_hash(path) == baseline_receipt['output_sha256'][path.name]
        with np.load(path) as archive:
            saved = {key: archive[key].copy() for key in archive.files}
        base = curated_tio2.chemsetup(ions=ions)
        assert base.metadata['dataset'] == 'curated-tio2-v1'
        species, elements = list(base.species), list(base.elements)
        matrix = np.asarray(base.formula_matrix)
        for value, key in ((species,'species'), (elements,'elements'), (matrix,'formula_matrix')):
            np.testing.assert_array_equal(value, saved[key])
        t,p,budgets = (saved[key] for key in ('temperature_K','pressure_bar','budget'))
        assert len(t) == 75 and len(species) == (537 if ions else 421) and saved['converged'].all()
        np.testing.assert_allclose(base.hvector_func(t), saved['h'], atol=1e-12, rtol=0)
        target, carbon = species.index(candidate.SPECIES), elements.index('C')
        for ratio in np.unique(saved['C_over_O']):
            indices = np.flatnonzero(saved['C_over_O'] == ratio)
            np.testing.assert_allclose(budgets[indices], np.broadcast_to(carbon_budget(base,float(ratio)), budgets[indices].shape), rtol=1e-15, atol=0)
        models = [base, *(candidate_setup(base, variant) for variant in CANDIDATES)]
        h = np.asarray([model.hvector_func(t) for model in models])
        np.testing.assert_array_equal(np.delete(h[1:]-h[0], target, axis=-1), 0)
        amounts = [saved['ln_n']]
        for variant, model in zip(CANDIDATES, models[1:]):
            solved = np.empty_like(saved['ln_n'])
            for ratio in np.unique(saved['C_over_O']):
                indices = np.flatnonzero(saved['C_over_O'] == ratio)
                values, rows = solve_cases(model, budgets[indices[0]], t[indices], p[indices], PRIMARY_TOLERANCE)
                solved[indices] = values
                diagnostics.extend({'network':network, 'variant':variant, 'case_index':int(i), **row} for i,row in zip(indices,rows))
                print(f'CH2 {variant} {network} C/O={ratio}:25 fresh endpoints passed', flush=True)
            amounts.append(solved)
        amounts = np.asarray(amounts)
        checks = constraints(matrix,elements,species,budgets,amounts,h,p)
        ln_x,x = fractions(amounts),np.exp(fractions(amounts))
        shares = np.exp(amounts)*matrix[carbon]/budgets[:,carbon,None]
        tight_indices, tight_amounts, paths, gibbs_raw, gibbs_corrected, scales, resolved_arrays = [], [], [], [], [], [], []
        baseline_tight=[]
        for i in (4,49,74):
            values, rows = solve_cases(base,budgets[i],t[i:i+1],p[i:i+1],STRICT_TOLERANCE)
            baseline_tight.append(values[0])
            controls.extend({'network':network,'kind':'tight_baseline','case_index':i,**row} for row in rows)
        baseline_check = stability(np.asarray(baseline_tight), amounts[0,[4,49,74]],target)
        baseline_checks.append({'network':network,'archive_sha256':file_hash(path),'selected_case_indices':[4,49,74],**baseline_check})
        for vi,(variant,model) in enumerate(zip(CANDIDATES,models[1:]),start=1):
            delta = x[vi]-x[0]
            ratio = np.exp(ln_x[vi,:,target]-ln_x[0,:,target])
            highlight = int(np.argmax(abs(shares[vi,:,target]-shares[0,:,target])))
            selected = {highlight,int(np.argmax(abs(np.log(ratio)))),int(np.argmax(checks['element_relative_residual'][vi]))}
            for i in (4,49,74):
                if len(selected)==3:break
                selected.add(i)
            selected=np.array(sorted(selected));assert len(selected)==3
            tight=[]
            for i in selected:
                values, rows=solve_cases(model,budgets[i],t[i:i+1],p[i:i+1],STRICT_TOLERANCE)
                tight.append(values[0])
                controls.extend({'network':network,'variant':variant,'kind':'tight_candidate','case_index':int(i),**row} for row in rows)
            tight=np.asarray(tight)
            strict=stability(tight,amounts[vi,selected],target)
            constraints(matrix,elements,species,budgets[selected],tight,h[vi,selected],p[selected])
            raw,corrected=gibbs_ordering(amounts[[0,vi]],h[[0,vi]],p,matrix,elements,species)
            g_scale=np.sum(np.exp(amounts[[0,vi]])*abs(h[[0,vi]]+ln_x[[0,vi]]+np.log(p)[None,:,None]),axis=-1)
            scale=100*np.finfo(np.longdouble).eps*np.max(g_scale,axis=0)
            resolved=(corrected[0]>scale)&(corrected[1]<-scale)
            path_amounts=[]
            if ions:
                for fraction in np.linspace(0,1,5):
                    path_model=candidate_setup(base,variant,float(fraction))
                    values,rows=solve_cases(path_model,budgets[highlight],t[highlight:highlight+1],p[highlight:highlight+1],STRICT_TOLERANCE)
                    path_amounts.append(values[0])
                    controls.extend({'network':network,'variant':variant,'kind':'path','lambda':float(fraction),'case_index':highlight,**row} for row in rows)
                path_amounts=np.asarray(path_amounts)
                path_h=h[0,highlight]+np.linspace(0,1,5)[:,None]*(h[vi,highlight]-h[0,highlight])
                constraints(matrix,elements,species,np.broadcast_to(budgets[highlight],(5,len(elements))),path_amounts,path_h,np.repeat(p[highlight],5))
                assert np.max(np.diff(path_amounts[:,target])*(h[vi,highlight,target]-h[0,highlight,target]))<1e-10
                exportable=np.flatnonzero(np.isin(t,[1500.,2000.,3000.,4000.]))
                fc_selected={int(exportable[np.argmax(abs(shares[vi,exportable,target]-shares[0,exportable,target]))]),int(exportable[np.argmax(abs(np.log(ratio[exportable])))] )}
                if len(fc_selected)<2:fc_selected.add(4 if 4 not in fc_selected else 74)
                for i in sorted(fc_selected):
                    independent.append({'network':network,'variant':variant,'case_index':i,'comparisons':independent_check([model],budgets[i],float(t[i]),float(p[i]),amounts[vi,i:i+1],executable.resolve())})
            def point(i,j):
                return {'case_index':int(i),'temperature_K':float(t[i]),'pressure_bar':float(p[i]),'C_over_O':float(saved['C_over_O'][i]),'species':species[j],
                    'x_before':float(x[0,i,j]),'x_after':float(x[vi,i,j]),'delta_x':float(delta[i,j]),
                    'ratio':float(np.exp(ln_x[vi,i,j]-ln_x[0,i,j])),
                    'carbon_share_before':float(shares[0,i,j]),'carbon_share_after':float(shares[vi,i,j])}
            observed=[s for s in ('C1H2','C2H2','C1H1','C1H3','C1H4','C1O1','C1O2','C1H1O1','C1','C2','C1N1','H1','H2','H2O1','e1-') if s in species]
            summary={'network':network,'variant':variant,'species':len(species),'cases':75,'CH2_x_ratio_min':float(ratio.min()),'CH2_x_ratio_max':float(ratio.max()),
                'highlighted_case':point(highlight,target),'largest_absolute_response':point(*np.unravel_index(np.argmax(abs(delta)),delta.shape)),
                'species_extrema':[point(int(np.argmax(abs(delta[:,species.index(s)]))),species.index(s)) for s in observed],
                'max_constraint_residuals':{key:float(value[vi].max()) for key,value in checks.items()},'tight_case_indices':selected.tolist(),'tolerance_stability':strict,
                'gibbs_ordering_resolved_cases':int(resolved.sum()),'gibbs_numerical_scale_max':float(scale.max()),
                'gibbs_corrected_current_min':float(corrected[0].min()),'gibbs_corrected_candidate_max':float(corrected[1].max()),
                'gibbs_scope':'Feasibility-corrected same-model exchanges; values below floating-point scale are unresolved, not scientific model selection.'}
            summaries.append(summary)
            tight_indices.append(selected);tight_amounts.append(tight);paths.append(path_amounts)
            gibbs_raw.append(raw);gibbs_corrected.append(corrected);scales.append(scale);resolved_arrays.append(resolved)
            print(json.dumps({'network':network,'variant':variant,'CH2_ratio':[ratio.min(),ratio.max()],'highlight':summary['highlighted_case'],'tight':strict}),flush=True)
        triplet_model=candidate_setup(base,'triplet_only');triplet=[]
        for i in (4,49,74):
            values,rows=solve_cases(triplet_model,budgets[i],t[i:i+1],p[i:i+1],STRICT_TOLERANCE)
            triplet.append(values[0])
            controls.extend({'network':network,'kind':'triplet_only','case_index':i,**row} for row in rows)
        triplet=np.asarray(triplet)
        triplet_h=np.asarray(triplet_model.hvector_func(t[[4,49,74]]))
        constraints(matrix,elements,species,budgets[[4,49,74]],triplet,triplet_h,p[[4,49,74]])
        summaries.append({'network':network,'variant':'triplet_only_control','cases':[4,49,74],
            'CH2_x_ratio_to_reconstructed_RRHO':np.exp(fractions(triplet)[:,target]-ln_x[2,[4,49,74],target]).tolist(),
            'scope':'A different partial thermodynamic species; selected sensitivity, not an interchangeable equilibrated species.'})
        np.savez_compressed(output/f'{network}.npz',variants=['current',*CANDIDATES],ln_n=amounts,h=h,formula_matrix=matrix,
            species=species,elements=elements,budget=budgets,temperature_K=t,pressure_bar=p,C_over_O=saved['C_over_O'],
            tight_case_indices=tight_indices,tight_ln_n=tight_amounts,baseline_tight_ln_n=baseline_tight,
            lambda_values=np.linspace(0,1,5) if ions else [],lambda_ln_n=paths,
            delta_g_raw=gibbs_raw,delta_g_corrected=gibbs_corrected,gibbs_numerical_scale=scales,gibbs_ordering_resolved=resolved_arrays,
            triplet_case_indices=[4,49,74],triplet_ln_n=triplet,triplet_h=triplet_h)
    summary={'baseline_dataset':'curated-tio2-v1','curated_update':'none','new_endpoint_solves':len(diagnostics),
        'reused_unique_baseline_cases':150,'new_selected_cold_solves':len(controls),
        'selected_counts':{kind:sum(row['kind']==kind for row in controls) for kind in ('tight_baseline','tight_candidate','path','triplet_only')},
        'primary_tolerance':PRIMARY_TOLERANCE,'strict_tolerance':STRICT_TOLERANCE,'networks':summaries,
        'baseline_reuse_verification':baseline_checks,'independent_solver':independent,
        'NASA_control':'Identical to energy-only under the retained NASA-proxy residual definition; no redundant solves counted.',
        'scope':'Fixed hot H-rich ideal gas. RRHO thermal sensitivity is not IUPAC table reproduction; 3000-4000K is explicit analytic extension, not validated high-temperature physical accuracy.'}
    assert len(diagnostics)==300 and len(controls)==34 and len(independent)==4
    for name,value in [('summary.json',summary),('diagnostics.json',diagnostics),('selected_diagnostics.json',controls)]:
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    paths={Path(__file__),Path(candidate.__file__)}
    paths.update(Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
        and Path(m.__file__).resolve().is_relative_to(ROOT) and Path(m.__file__).suffix=='.py'
        and Path(m.__file__).name not in ('exogibbs_version.py','curated.py'))
    paths.update((ROOT/'src/exogibbs/data/curated').glob('*.json'))
    paths.update(ROOT/'src/exogibbs/data'/p for p in ('FastChem4/logK/logK.dat','FastChem4/element_abundances/asplund_2021.dat'))
    paths.update(BASELINE/p for p in ('ions.npz','neutral.npz','receipt.json'))
    paths.update(candidate.DATA.iterdir())
    paths.update(ROOT/p for p in candidate.load_candidate()['source_sha256'])
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'result':'passed','python':platform.python_version(),
        'jax':jax.__version__,'numpy':np.__version__,'dtype':'float64','backend':jax.default_backend(),'command':[sys.executable,*sys.argv],
        'selected_file_sha256':{str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths) if p.is_file()},
        'output_sha256':{p.name:file_hash(p) for p in sorted(output.iterdir())},'fastchem_binary_sha256':file_hash(executable)}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--fastchem-executable',type=Path,required=True)
    args=parser.parse_args()
    run(args.output_dir,args.fastchem_executable)
