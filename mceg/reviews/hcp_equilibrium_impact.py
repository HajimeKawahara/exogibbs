"""Measure two independent HCP energy substitutions on fixed ion/neutral networks."""

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
import hcp_candidate as candidate

BASELINE = ROOT / "tests/data/validation/curated_tio2_adoption/20261106"
PRIMARY_TOLERANCE, STRICT_TOLERANCE = 5e-14, 2e-14


def candidate_setup(base, anchor: str, offset_J_mol: float = 0.):
    """Replace HCP alone with one independent energy anchor, never cumulatively."""
    column = base.species.index(candidate.SPECIES)
    def hvector(t):
        initial = base.hvector_func(t)
        return initial.at[...,column].set(candidate.formation_h_jax(t,anchor,offset_J_mol))
    return replace(base,hvector_func=jax.jit(hvector),metadata={**base.metadata,
        "dataset":"review-hcp-"+anchor,"production_adoption":False,"energy_offset_J_mol":offset_J_mol})


def fractions(values):
    """Normalize the stored unfloored logarithmic amounts."""
    return values-logsumexp(values,axis=-1,keepdims=True)


def constraints(matrix, elements, species, budget, amounts, h, pressure):
    """Verify all elements, phosphorus, charge and formation mass action."""
    n = np.exp(amounts)
    residual = abs(n@matrix.T-budget)
    positive = np.asarray(budget).reshape(-1,len(elements))[0]>0
    refs = [species.index("e1-" if e=="e-" else e+"1") for e in elements]
    logp = fractions(amounts)+np.log(pressure)[...,None]
    values = {"element_relative_residual":np.max(residual[...,positive]/budget[...,positive],axis=-1),
              "HCNOPS_element_relative_residual":np.max(np.stack([residual[...,elements.index(e)]/budget[...,elements.index(e)] for e in ("H","C","N","O","P","S")]),axis=0),
              "phosphorus_relative_residual":residual[...,elements.index("P")]/budget[...,elements.index("P")],
              "mass_action_max_abs_ln":np.max(abs(logp-logp[...,refs]@matrix+h),axis=-1)}
    if "e-" in elements:
        charge = matrix[elements.index("e-")]
        values["charge_relative_residual"] = abs(n@charge)/(n@abs(charge))
    limits = {"element_relative_residual":1e-5,"HCNOPS_element_relative_residual":1e-9,
              "phosphorus_relative_residual":1e-9,"mass_action_max_abs_ln":1e-8,"charge_relative_residual":1e-12}
    assert np.isfinite(amounts).all() and np.all(n>0)
    for key,value in values.items():
        assert np.isfinite(value).all() and np.max(value)<limits[key],(key,np.max(value))
    return values


def run(output: Path, executable: Path | None = None) -> None:
    """Archive 300 fresh endpoints, 12 tight and 16 energy-uncertainty controls."""
    jax.config.update("jax_enable_x64",True)
    output.mkdir(parents=True,exist_ok=False)
    summaries,diagnostics,controls,independent = [],[],[],[]
    baseline_receipt = json.loads((BASELINE/"receipt.json").read_text())
    for ions in (True,False):
        name = "ions" if ions else "neutral"
        path = BASELINE/f"{name}.npz"
        assert file_hash(path)==baseline_receipt["output_sha256"][path.name]
        with np.load(path) as archive:
            saved = {key:archive[key].copy() for key in archive.files}
        base = curated_tio2.chemsetup(ions=ions)
        assert base.metadata["dataset"]=="curated-tio2-v1"
        species,elements,matrix = list(base.species),list(base.elements),np.asarray(base.formula_matrix)
        for value,key in ((species,"species"),(elements,"elements"),(matrix,"formula_matrix")):
            np.testing.assert_array_equal(value,saved[key])
        t,p,budget = (saved[key] for key in ("temperature_K","pressure_bar","budget"))
        assert len(t)==75 and len(species)==(537 if ions else 421) and saved["converged"].all()
        models = [base]+[candidate_setup(base,anchor) for anchor in candidate.ANCHORS]
        h = np.asarray([model.hvector_func(t) for model in models])
        np.testing.assert_allclose(h[0],saved["h"],rtol=0,atol=1e-12)
        target,phosphorus = species.index(candidate.SPECIES),elements.index("P")
        for hi in h[1:]:np.testing.assert_array_equal(np.delete(hi-h[0],target,axis=-1),0)
        amounts = np.empty((3,*saved["ln_n"].shape));amounts[0]=saved["ln_n"]
        for variant,anchor in enumerate(candidate.ANCHORS,1):
            for ratio in np.unique(saved["C_over_O"]):
                indices = np.flatnonzero(saved["C_over_O"]==ratio)
                expected = np.broadcast_to(carbon_budget(base,float(ratio)),budget[indices].shape)
                np.testing.assert_allclose(budget[indices],expected,rtol=1e-15,atol=0)
                values,extra = solve_cases(models[variant],budget[indices[0]],t[indices],p[indices],PRIMARY_TOLERANCE)
                amounts[variant,indices] = values
                diagnostics.extend({"network":name,"anchor":anchor,"case_index":int(i),**row} for i,row in zip(indices,extra))
                print(f"HCP {anchor} {name} C/O={ratio}: {len(indices)} new cold endpoints passed",flush=True)
        checks = constraints(matrix,elements,species,budget,amounts,h,p)
        logx = fractions(amounts);x=np.exp(logx)
        pshare = np.exp(amounts)*matrix[phosphorus]/budget[:,phosphorus,None]
        effect = np.max(abs(pshare[1:,:,target]-pshare[0,:,target]),axis=0)
        highlight = int(np.argmax(effect))
        selected = [highlight,int(np.argmax(pshare[0,:,target]))]
        if selected[0]==selected[1]:
            selected[1] = int(np.argsort(effect)[-2])
        selected = np.array(sorted(selected));assert len(np.unique(selected))==2
        tight = np.empty((3,len(selected),len(species)))
        for variant,model in enumerate(models):
            for k,i in enumerate(selected):
                values,extra = solve_cases(model,budget[i],t[i:i+1],p[i:i+1],STRICT_TOLERANCE)
                tight[variant,k] = values[0]
                controls.extend({"network":name,"kind":"tight","variant":variant,"case_index":int(i),**row} for row in extra)
        constraints(matrix,elements,species,budget[selected],tight,h[:,selected],p[selected])
        difference = fractions(tight)-logx[:,selected]
        visible = np.maximum(np.exp(fractions(tight)),x[:,selected])>=1e-12
        stability = {"max_abs_delta_ln_x":float(abs(difference).max()),"HCP_max_abs_delta_ln_x":float(abs(difference[...,target]).max()),
                     "visible_max_abs_delta_ln_x":float(abs(difference[visible]).max()),"max_absolute_delta_x":float(abs(np.exp(fractions(tight))-x[:,selected]).max())}
        assert stability["max_abs_delta_ln_x"]<1e-4 and stability["HCP_max_abs_delta_ln_x"]<1e-8
        assert stability["visible_max_abs_delta_ln_x"]<2e-6 and stability["max_absolute_delta_x"]<1e-9
        uncertainty = np.empty((2,2,len(selected),len(species)));uncertainty_h=np.empty_like(uncertainty)
        uncertainty_rows = []
        for ai,anchor in enumerate(candidate.ANCHORS):
            width = candidate.load_candidate()["anchors"][anchor]["reported_uncertainty_J_mol"]
            for oi,sign in enumerate((-1,1)):
                model = candidate_setup(base,anchor,sign*width)
                uncertainty_h[ai,oi]=np.asarray(model.hvector_func(t[selected]))
                for k,i in enumerate(selected):
                    values,extra = solve_cases(model,budget[i],t[i:i+1],p[i:i+1],STRICT_TOLERANCE)
                    uncertainty[ai,oi,k]=values[0]
                    controls.extend({"network":name,"kind":"quoted_energy_edge","anchor":anchor,"offset_J_mol":sign*width,"case_index":int(i),**row} for row in extra)
                    share = np.exp(values[0])*matrix[phosphorus]/budget[i,phosphorus]
                    top = np.argsort(share)[-5:][::-1]
                    uncertainty_rows.append({"anchor":anchor,"offset_J_mol":sign*width,"case_index":int(i),
                        "HCP_phosphorus_share":float(share[target]),"top_phosphorus_carriers":[{"species":species[j],"share":float(share[j])} for j in top]})
        constraints(matrix,elements,species,budget[selected],uncertainty,uncertainty_h,p[selected])
        assert np.max(uncertainty[:,1,:,target]-uncertainty[:,0,:,target])<1e-10
        raw,corrected,scales,resolved = [],[],[],[]
        anchor_summaries=[]
        def point(variant,i,j):
            return {"case_index":int(i),"temperature_K":float(t[i]),"pressure_bar":float(p[i]),"C_over_O":float(saved["C_over_O"][i]),"species":species[j],
                    "x_before":float(x[0,i,j]),"x_after":float(x[variant,i,j]),"ratio":float(np.exp(logx[variant,i,j]-logx[0,i,j])),
                    "phosphorus_share_before":float(pshare[0,i,j]),"phosphorus_share_after":float(pshare[variant,i,j])}
        for variant,anchor in enumerate(candidate.ANCHORS,1):
            ar,ac = gibbs_ordering(amounts[[0,variant]],h[[0,variant]],p,matrix,elements,species)
            scale = 100*np.finfo(np.longdouble).eps*np.max(np.sum(np.exp(amounts[[0,variant]])*abs(h[[0,variant]]+logx[[0,variant]]+np.log(p)[None,:,None]),axis=-1),axis=0)
            resolved_anchor = (ac[0]>scale)&(ac[1]<-scale)
            raw.append(ar);corrected.append(ac);scales.append(scale);resolved.append(resolved_anchor)
            ratio = np.exp(logx[variant,:,target]-logx[0,:,target]);delta=x[variant]-x[0]
            pi = int(np.argmax(abs(pshare[variant,:,target]-pshare[0,:,target])))
            carriers = np.flatnonzero(matrix[phosphorus]>0)
            observed = [species[j] for j in carriers if np.max(pshare[:, :, j])>=1e-4]
            anchor_summaries.append({"anchor":anchor,"HCP_ratio_min":float(ratio.min()),"HCP_ratio_max":float(ratio.max()),
                "highlighted_case":point(variant,pi,target),"largest_absolute_response":point(variant,*np.unravel_index(np.argmax(abs(delta)),delta.shape)),
                "phosphorus_carrier_extrema":[point(variant,int(np.argmax(abs(pshare[variant,:,species.index(s)]-pshare[0,:,species.index(s)]))),species.index(s)) for s in observed],
                "max_phosphorus_share_before":float(pshare[0,:,target].max()),"max_phosphorus_share_after":float(pshare[variant,:,target].max()),
                "gibbs_resolved_cases":int(np.sum(resolved_anchor)),"gibbs_scale_max":float(np.max(scale)),
                "gibbs_corrected_current_min":float(ac[0].min()),"gibbs_corrected_candidate_max":float(ac[1].max())})
            if ions and executable:
                exportable=np.flatnonzero(np.isin(t,[1500.,2000.,3000.,4000.]))
                i=int(exportable[np.argmax(abs(pshare[variant,exportable,target]-pshare[0,exportable,target]))])
                independent.append({"network":name,"anchor":anchor,"case_index":i,"comparisons":independent_check(
                    [base,models[variant]],budget[i],float(t[i]),float(p[i]),amounts[[0,variant],i],executable.resolve())})
        summary = {"network":name,"species":len(species),"cases":75,"anchors":anchor_summaries,
                   "max_constraint_residuals":{key:float(np.max(value)) for key,value in checks.items()},
                   "tight_case_indices":selected.tolist(),"tolerance_stability":stability,"quoted_energy_controls":uncertainty_rows,
                   "uncertainty_scope":"Only two selected conditions per network; source-reported widths, not confidence intervals or full thermal/network uncertainty.",
                   "gibbs_scope":"Same-model feasibility-corrected exchanges are resolved only above the displayed floating-point scale; unresolved trace cases are retained."}
        summaries.append(summary)
        np.savez_compressed(output/f"{name}.npz",variants=["current",*candidate.ANCHORS],ln_n=amounts,h=h,
            species=species,elements=elements,formula_matrix=matrix,budget=budget,temperature_K=t,pressure_bar=p,C_over_O=saved["C_over_O"],
            converged=np.ones((3,75),dtype=bool),tight_case_indices=selected,tight_ln_n=tight,
            uncertainty_ln_n=uncertainty,uncertainty_h=uncertainty_h,uncertainty_signs=[-1,1],
            delta_g_raw=raw,delta_g_corrected=corrected,gibbs_numerical_scale=scales,gibbs_ordering_resolved=resolved)
        print(json.dumps(summary),flush=True)
    summary = {"baseline_dataset":"curated-tio2-v1","curated_update":"none","new_endpoint_solves":len(diagnostics),"reused_baseline_cases":150,
               "new_control_solves":len(controls),"new_tighter_solves":sum(r["kind"]=="tight" for r in controls),
               "new_quoted_energy_edge_solves":sum(r["kind"]=="quoted_energy_edge" for r in controls),
               "independent_fastchem_solves":sum(len(r["comparisons"]) for r in independent),
               "primary_tolerance":PRIMARY_TOLERANCE,"strict_tolerance":STRICT_TOLERANCE,"networks":summaries,
               "independent_solver":independent if executable else "not_run",
               "scope":"Two independent HCP energy-only alternatives, each on the same 75-case ion and 75-case neutral baseline; no thermal replacement, adoption, condensation or full physical uncertainty."}
    assert summary["new_endpoint_solves"]==300 and summary["new_control_solves"]==28
    for name,value in (("summary.json",summary),("diagnostics.json",diagnostics),("selected_diagnostics.json",controls)):
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    paths = {Path(__file__).resolve(),Path(candidate.__file__).resolve(),BASELINE/"receipt.json"}
    paths.update(ROOT/p for p in candidate.load_candidate()["source_sha256"])
    paths.update(BASELINE/f"{n}.npz" for n in ("ions","neutral"))
    paths.update(p for p in candidate.DATA.iterdir() if p.is_file())
    paths.update(Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,"__file__",None)
                 and Path(m.__file__).resolve().is_relative_to(ROOT) and Path(m.__file__).suffix==".py"
                 and Path(m.__file__).name not in ("exogibbs_version.py","curated.py"))
    paths.update((ROOT/"src/exogibbs/data/curated").glob("*.json"))
    paths.update(ROOT/"src/exogibbs/data"/p for p in ("FastChem4/logK/logK.dat","FastChem4/element_abundances/asplund_2021.dat"))
    receipt = {"created_utc":datetime.now(timezone.utc).isoformat(),"command":[sys.executable,*sys.argv],"python":platform.python_version(),
               "jax":jax.__version__,"numpy":np.__version__,"dtype":"float64","backend":jax.default_backend(),"result":"passed",
               "selected_file_sha256":{str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)},
               "output_sha256":{p.name:file_hash(p) for p in sorted(output.iterdir()) if p.is_file()},
               "fastchem_binary_sha256":file_hash(executable) if executable else None}
    (output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--fastchem-executable",type=Path)
    args=parser.parse_args()
    run(args.output_dir,args.fastchem_executable)
