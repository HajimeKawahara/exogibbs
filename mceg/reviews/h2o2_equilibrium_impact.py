"""Measure H2O2 finite substitutions on fixed curated-cos-v1 gas networks."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import jax
import numpy as np
from scipy.special import logsumexp

from exogibbs.presets import curated_cos as curated
from c2h_equilibrium_impact import carbon_budget, independent_check
from co2_equilibrium_impact import gibbs_ordering
from cos_equilibrium_impact import constraints
from curated_h2o_adoption import solve_cases
from gas_source_comparison import ROOT, file_hash
import h2o2_candidate as candidate

BASELINE = ROOT / "tests/data/validation/curated_cos_adoption/20261105"
PRIMARY_TOLERANCE, STRICT_TOLERANCE = 5e-14, 2e-14


def candidate_setup(base, variant: str = candidate.PRIMARY, fraction: float = 1.0):
    """Replace the target alone, retaining species, elements, budgets and gauge."""
    col=base.species.index(candidate.source.SPECIES)
    def hvector(t):
        old=base.hvector_func(t)
        target=candidate.formation_h_jax(t,variant)
        return old.at[...,col].set(old[...,col]+fraction*(target-old[...,col]))
    return replace(base,hvector_func=jax.jit(hvector),metadata={**base.metadata,"dataset":"review-h2o2-"+variant,"production_adoption":False,"substitution_fraction":fraction})


def fractions(amounts):
    """Normalize logarithms without concentration floors."""
    return amounts-logsumexp(amounts,axis=-1,keepdims=True)


def run(output: Path, executable: Path | None = None):
    """Run candidates, tighter cold controls, finite paths and solver parity."""
    started=time.monotonic();jax.config.update("jax_enable_x64",True)
    output.mkdir(parents=True,exist_ok=False)
    names=["current",*candidate.VARIANTS]
    summaries=[];rows=[];controls=[];independent=[]
    for ions in [True,False]:
        mode="ions" if ions else "neutral"
        with np.load(BASELINE/f"{mode}.npz") as g:saved={key:g[key].copy() for key in g.files}
        base=curated.chemsetup(ions=ions)
        assert base.metadata["dataset"]=="curated-cos-v1"
        species,elements=list(base.species),list(base.elements);matrix=np.asarray(base.formula_matrix)
        for actual,key in [(species,"species"),(elements,"elements"),(matrix,"formula_matrix")]:np.testing.assert_array_equal(actual,saved[key])
        t,p,budget=(saved[k] for k in ["temperature_K","pressure_bar","budget"])
        assert len(t)==75 and saved["converged"].all()
        for ratio in np.unique(saved["C_over_O"]):
            indices=np.flatnonzero(saved["C_over_O"]==ratio)
            np.testing.assert_allclose(budget[indices],np.broadcast_to(carbon_budget(base,float(ratio)),budget[indices].shape),rtol=1e-15,atol=0)
        models=[base]+[candidate_setup(base,v) for v in candidate.VARIANTS]
        h=np.asarray([m.hvector_func(t) for m in models]);np.testing.assert_allclose(h[0],saved["h"],rtol=0,atol=1e-12)
        col=species.index(candidate.source.SPECIES);element=elements.index(candidate.source.inputs()["footprint_element"])
        np.testing.assert_array_equal(np.delete(h[1:]-h[0],col,axis=-1),0)
        amounts=np.empty((len(models),*saved["ln_n"].shape));amounts[0]=saved["ln_n"]
        for vi,model in enumerate(models[1:],1):
            for ratio in np.unique(saved["C_over_O"]):
                indices=np.flatnonzero(saved["C_over_O"]==ratio)
                values,extra=solve_cases(model,budget[indices[0]],t[indices],p[indices],PRIMARY_TOLERANCE)
                amounts[vi,indices]=values
                rows.extend({"network":mode,"variant":names[vi],"case_index":int(i),**r} for i,r in zip(indices,extra))
            print(json.dumps({"network":mode,"variant":names[vi],"new_endpoint_solves":75,"elapsed_seconds":time.monotonic()-started}),flush=True)
        checks=constraints(matrix,elements,species,budget,amounts,h,p)
        n=np.exp(amounts);ln_x=fractions(amounts);x=np.exp(ln_x)
        element_error=np.max(np.stack([abs(n@matrix[elements.index(e)]-budget[:,elements.index(e)])/budget[:,elements.index(e)] for e in candidate.source.COMPOSITION]),axis=0)
        assert element_error.max()<1e-9
        share=n[...,col]*matrix[element,col]/budget[:,element]
        primary=names.index(candidate.PRIMARY)
        highlight=int(np.argmax(abs(share[primary]-share[0])))
        selected=sorted({highlight,int(np.argmax(abs(ln_x[primary,:,col]-ln_x[0,:,col]))),int(np.argmax(checks["element_relative_residual"][primary]))})
        tight=np.empty((len(models),len(selected),len(species)))
        for vi,model in enumerate(models):
            for si,i in enumerate(selected):
                values,extra=solve_cases(model,budget[i],t[i:i+1],p[i:i+1],STRICT_TOLERANCE);tight[vi,si]=values[0]
                controls.extend({"network":mode,"variant":names[vi],"kind":"tight_cold","case_index":i,**r} for r in extra)
        tight_x=fractions(tight);difference=tight_x-ln_x[:,selected]
        visible=np.maximum(np.exp(tight_x),x[:,selected])>=1e-12
        stability={"all_delta_ln_x":float(abs(difference).max()),"target_delta_ln_x":float(abs(difference[...,col]).max()),"visible_delta_ln_x":float(abs(difference[visible]).max()),"absolute_delta_x":float(abs(np.exp(tight_x)-x[:,selected]).max())}
        for key,limit in zip(stability,[1e-4,1e-8,2e-6,1e-9]):assert stability[key]<limit,(mode,key,stability)
        summaries_variants=[];raws=[];correcteds=[]
        for vi in range(1,len(models)):
            raw,corrected=gibbs_ordering(amounts[[0,vi]],h[[0,vi]],p,matrix,elements,species);raws.append(raw);correcteds.append(corrected)
            delta=x[vi]-x[0];ratio=np.exp(ln_x[vi,:,col]-ln_x[0,:,col])
            mass_action_prediction=-(h[vi,:,col]-h[0,:,col])+sum(count*(ln_x[vi,:,species.index(e+"1")]-ln_x[0,:,species.index(e+"1")]) for e,count in candidate.source.COMPOSITION.items())
            closure=float(abs(ln_x[vi,:,col]-ln_x[0,:,col]-mass_action_prediction).max());assert closure<1e-8
            def point(i,j):
                return {"case_index":int(i),"temperature_K":float(t[i]),"pressure_bar":float(p[i]),"C_over_O":float(saved["C_over_O"][i]),"species":species[j],"x_before":float(x[0,i,j]),"x_after":float(x[vi,i,j]),"delta_x":float(delta[i,j]),"ratio":float(np.exp(ln_x[vi,i,j]-ln_x[0,i,j])),"element_share_before":float(n[0,i,j]*matrix[element,j]/budget[i,element]),"element_share_after":float(n[vi,i,j]*matrix[element,j]/budget[i,element])}
            overall=np.unravel_index(np.argmax(abs(delta)),delta.shape)
            target_i=int(np.argmax(abs(delta[:,col])))
            summaries_variants.append({"variant":names[vi],"target_ratio_min":float(ratio.min()),"target_ratio_max":float(ratio.max()),"target_maximum_absolute_response":point(target_i,col),"target_maximum_element_share_before":float(share[0].max()),"target_maximum_element_share_after":float(share[vi].max()),"largest_any_species_response":point(*overall),"finite_mass_action_closure_ln":closure,"gibbs_corrected_current_min":float(corrected[0].min()),"gibbs_corrected_candidate_max":float(corrected[1].max()),"gibbs_scope":"First-order feasibility-corrected ordering within 5e-14 numerical tolerance; a trace-species response below that scale is not a resolved Gibbs improvement."})
        lambdas=np.linspace(0,1,5);path=[]
        for fraction in lambdas:
            model=candidate_setup(base,candidate.PRIMARY,float(fraction))
            values,extra=solve_cases(model,budget[highlight],t[highlight:highlight+1],p[highlight:highlight+1],STRICT_TOLERANCE);path.append(values[0])
            controls.extend({"network":mode,"variant":candidate.PRIMARY,"kind":"finite_path","lambda":float(fraction),"case_index":highlight,**r} for r in extra)
        if ions and executable:
            exportable=np.flatnonzero(np.isin(t,[1500.,2000.,3000.,4000.]))
            chosen=sorted({int(exportable[np.argmax(abs(share[primary,exportable]-share[0,exportable]))]),int(exportable[np.argmax(abs(ln_x[primary,exportable,col]-ln_x[0,exportable,col]))])})
            for i in chosen:
                independent.append({"network":mode,"case_index":i,"comparisons":independent_check(models,budget[i],float(t[i]),float(p[i]),amounts[:,i],executable.resolve())})
        summary={"network":mode,"cases":75,"species_count":len(species),"variants":summaries_variants,"target_element":elements[element],"highlight_case_index":highlight,"tight_case_indices":selected,"tolerance_stability":stability,"maximum_target_elements_relative_residual":float(element_error.max()),"max_constraint_residuals":{k:float(v.max()) for k,v in checks.items()},"path_target_ratios":np.exp(fractions(np.asarray(path))[:,col]-ln_x[0,highlight,col]).tolist(),"actual_new_endpoint_solves":75*(len(models)-1)}
        summaries.append(summary)
        np.savez_compressed(output/f"{mode}.npz",variants=names,ln_n=amounts,h=h,species=species,elements=elements,formula_matrix=matrix,budget=budget,temperature_K=t,pressure_bar=p,C_over_O=saved["C_over_O"],tight_case_indices=selected,tight_ln_n=tight,highlighted_case_index=highlight,lambda_values=lambdas,lambda_ln_n=path,delta_g_raw=raws,delta_g_corrected=correcteds)
        print(json.dumps(summary),flush=True)
    summary={"baseline_dataset":"curated-cos-v1","production_change":"none","reused_baseline_cases":150,"new_endpoint_solves":len(rows),"new_tighter_cold_solves":sum(r["kind"]=="tight_cold" for r in controls),"new_finite_path_solves":sum(r["kind"]=="finite_path" for r in controls),"primary_tolerance":PRIMARY_TOLERANCE,"strict_tolerance":STRICT_TOLERANCE,"networks":summaries,"independent_solver":independent if executable else "not_run","elapsed_seconds":time.monotonic()-started,"scope":"Finite single-species substitutions in fixed ideal-gas networks and inventories, 1500-4000 K. No condensation, cooler-atmosphere, photochemical or complete physical validation."}
    for filename,value in [("summary.json",summary),("diagnostics.json",rows),("selected_diagnostics.json",controls)]:
        (output/filename).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    paths={Path(__file__),Path(candidate.__file__)}
    paths.update(ROOT/p for p in json.loads((candidate.DATA/"receipt.json").read_text())["selected_file_sha256"])
    paths.update(candidate.DATA.glob("*.json"));paths.update(BASELINE/f"{m}.npz" for m in ["ions","neutral"])
    paths.update(ROOT/"mceg/reviews"/(s+".py") for s in ["c2h_equilibrium_impact","co2_equilibrium_impact","cos_equilibrium_impact","cos_candidate","cos_source_review","curated_h2o_adoption","co2_candidate","co2_source_review","h2o_candidate","h2o_source_review","tio_equilibrium_impact"])
    paths.update((ROOT/"benchmarks/fastchem4").glob("*.py"));paths.update(ROOT/p for p in ["examples/comparisons/compare_gas_thermochemistry.py","examples/comparisons/select_atomic_ions.py","benchmarks/__init__.py"])
    receipt={"created_utc":datetime.now(timezone.utc).isoformat(),"command":sys.argv,"python":platform.python_version(),"numpy":np.__version__,"jax":jax.__version__,"backend":jax.default_backend(),"dtype":"float64","result":"passed","selected_file_sha256":{str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)},"output_sha256":{p.name:file_hash(p) for p in sorted(output.iterdir())},"fastchem_binary_sha256":file_hash(executable) if executable else None}
    (output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--fastchem-executable",type=Path);args=parser.parse_args()
    run(args.output_dir,args.fastchem_executable)
