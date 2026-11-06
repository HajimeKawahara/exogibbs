"""Verify compact POPS PS production thermals on the cumulative CN base."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import jax
import numpy as np
from scipy.special import logsumexp

from exogibbs.presets import curated_cn, curated_ps
from c2h_equilibrium_impact import carbon_budget, independent_check
from curated_h2o_adoption import solve_cases
from ps_equilibrium_impact import candidate_setup, constraints


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tests/data/validation/curated_cn_adoption/20261106"
OLD_BASELINE = ROOT / "tests/data/validation/curated_cos_adoption/20261105"
CANDIDATE = ROOT / "tests/data/validation/ps_equilibrium_impact/20261106"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def fractions(logs):
    return logs-logsumexp(logs, axis=-1, keepdims=True)


def run(output: Path, executable: Path | None = None) -> None:
    """Run 150 fresh endpoints and six independent tighter candidate controls."""
    jax.config.update("jax_enable_x64", True)
    output.mkdir(parents=True, exist_ok=False)
    summaries, diagnostics, independent = [], [], []
    for ions in (True, False):
        name = "ions" if ions else "neutral"
        archives=[]
        for directory in (BASELINE,OLD_BASELINE):
            with np.load(directory/f"{name}.npz") as archive:
                archives.append({key:archive[key].copy() for key in archive.files})
        previous,original=archives
        setup,base=curated_ps.chemsetup(ions=ions),curated_cn.chemsetup(ions=ions)
        reference=candidate_setup(base)
        assert setup.metadata["dataset"] == "curated-ps-v1"
        assert base.metadata["dataset"] == "curated-cn-v1"
        assert setup.species == base.species and setup.elements == base.elements
        np.testing.assert_array_equal(setup.formula_matrix,base.formula_matrix)
        np.testing.assert_array_equal(setup.element_vector_reference,base.element_vector_reference)
        species,elements=list(setup.species),list(setup.elements)
        matrix=np.asarray(setup.formula_matrix)
        for value,key in ((species,"species"),(elements,"elements"),(matrix,"formula_matrix")):
            for saved in archives:np.testing.assert_array_equal(value,saved[key])
        t,p,budgets=(previous[key] for key in ("temperature_K","pressure_bar","budget"))
        for key in ("temperature_K","pressure_bar","budget","C_over_O"):
            np.testing.assert_array_equal(previous[key],original[key])
        assert len(t) == 75 and previous["converged"].all()
        h=np.asarray(setup.hvector_func(t))
        np.testing.assert_allclose(base.hvector_func(t),previous["h"],atol=1e-12,rtol=0)
        target,phosphorus=species.index("P1S1"),elements.index("P")
        np.testing.assert_array_equal(np.delete(h-previous["h"],target,axis=-1),0)
        dense_t=np.linspace(1500.,4000.,2501)
        # Bound the direct state-sum temporary arrays; production uses only 19 coefficients.
        dense_reference=np.concatenate([np.asarray(reference.hvector_func(chunk))
                                        for chunk in np.array_split(dense_t,80)])
        dense_error=float(abs(np.asarray(setup.hvector_func(dense_t))-dense_reference).max())
        assert dense_error < 1e-12
        amounts=np.empty_like(previous["ln_n"])
        for ratio in np.unique(previous["C_over_O"]):
            indices=np.flatnonzero(previous["C_over_O"] == ratio)
            expected=np.broadcast_to(carbon_budget(base,float(ratio)),budgets[indices].shape)
            np.testing.assert_allclose(budgets[indices],expected,rtol=1e-15,atol=0)
            values,checks=solve_cases(setup,budgets[indices[0]],t[indices],p[indices],5e-14)
            amounts[indices]=values
            diagnostics.extend({"network":name,"case_index":int(i),**row} for i,row in zip(indices,checks))
            print(f"PS production {name} C/O={ratio}: {len(indices)} endpoints passed",flush=True)
        closure=constraints(matrix,elements,species,budgets,amounts,h,p)
        logx,prior_x,old_x=map(fractions,(amounts,previous["ln_n"],original["ln_n"]))
        x=np.exp(logx)
        ratio=np.exp(logx[:,target]-prior_x[:,target])
        share_before=np.exp(previous["ln_n"][:,target])/budgets[:,phosphorus]
        share_after=np.exp(amounts[:,target])/budgets[:,phosphorus]
        highlight=int(np.argmax(abs(share_after-share_before)))
        selected={highlight,int(np.argmax(abs(np.log(ratio)))),int(np.argmax(closure["element_relative_residual"]))}
        for i in (4,46,70,0):
            if len(selected)==3:break
            selected.add(i)
        selected=np.array(sorted(selected));assert len(selected)==3
        tight=[]
        for i in selected:
            values,checks=solve_cases(reference,budgets[i],t[i:i+1],p[i:i+1],2e-14)
            tight.append(values[0])
            diagnostics.extend({"network":name,"case_index":int(i),"tighter_reference":True,**row} for row in checks)
        tight=np.asarray(tight);tight_x=fractions(tight)
        constraints(matrix,elements,species,budgets[selected],tight,h[selected],p[selected])
        difference=logx[selected]-tight_x
        stability={"max_abs_delta_ln_x":float(abs(difference).max()),
                   "max_ps_abs_delta_ln_x":float(abs(difference[:,target]).max()),
                   "max_absolute_delta_x":float(abs(x[selected]-np.exp(tight_x)).max())}
        assert stability["max_abs_delta_ln_x"]<1e-4
        assert stability["max_ps_abs_delta_ln_x"]<1e-8
        assert stability["max_absolute_delta_x"]<1e-9
        cumulative={}
        for label in ("C1N1","P1S1","P2","O1P1"):
            j=species.index(label)
            cumulative[label]={"production_over_old_cos_min":float(np.exp(logx[:,j]-old_x[:,j]).min()),
                               "production_over_old_cos_max":float(np.exp(logx[:,j]-old_x[:,j]).max()),
                               "max_absolute_delta_x":float(abs(x[:,j]-np.exp(old_x[:,j])).max())}
        summary={"ions":ions,"cases_passed":len(t),"species":len(species),
                 "max_dense_h_difference_from_candidate_on_cn":dense_error,
                 "tighter_case_indices":selected.tolist(),"tighter_reference":stability,
                 "closure":{key:float(np.max(value)) for key,value in closure.items()},
                 "ps_incremental_response":{"min_x_ratio":float(ratio.min()),"max_x_ratio":float(ratio.max()),
                     "highlighted_case_index":highlight,"temperature_K":float(t[highlight]),
                     "pressure_bar":float(p[highlight]),"C_over_O":float(previous["C_over_O"][highlight]),
                     "x_before":float(np.exp(prior_x[highlight,target])),"x_after":float(x[highlight,target]),
                     "phosphorus_share_before":float(share_before[highlight]),"phosphorus_share_after":float(share_after[highlight])},
                 "cumulative_response_from_curated_cos":cumulative}
        if ions:
            with np.load(CANDIDATE/"ions.npz") as archived:
                for key in ("species","elements","formula_matrix","budget","temperature_K","pressure_bar","C_over_O"):
                    np.testing.assert_array_equal(previous[key],archived[key])
                np.testing.assert_allclose(h[:,target],archived["h"][1,:,target],atol=1e-12,rtol=0)
                background=[species.index("C1N1")]
                np.testing.assert_allclose(np.delete(h-archived["h"][1],background+[target],axis=-1),0,atol=1e-12,rtol=0)
                archived_x=fractions(archived["ln_n"][1]);delta=logx-archived_x
                summary["background_effect_relative_to_old_ps_only_candidate"]={
                    "PS_ratio_min":float(np.exp(delta[:,target]).min()),"PS_ratio_max":float(np.exp(delta[:,target]).max()),
                    "max_absolute_delta_x_all_species":float(abs(x-np.exp(archived_x)).max()),
                    "scope":"CN background adoption changes equilibrium composition; this is a physical model difference, not a numerical reproduction error."}
            if executable:
                exportable=np.flatnonzero(np.isin(t,[1500.,2000.,3000.,4000.]))
                selected_fc=sorted({int(exportable[np.argmax(abs(share_after[exportable]-share_before[exportable]))]),
                                    int(exportable[np.argmax(abs(np.log(ratio[exportable])))] )})
                assert len(selected_fc)==2
                for i in selected_fc:
                    independent.append({"case_index":i,"comparisons":independent_check(
                        [setup],budgets[i],float(t[i]),float(p[i]),amounts[i:i+1],executable.resolve())})
        else:
            summary["background_effect_relative_to_old_ps_only_candidate"]="not_available: prior PS candidate solved the ion-inclusive network only"
        np.savez_compressed(output/f"{name}.npz",ln_n=amounts,ln_x=logx,h=h,
            formula_matrix=matrix,budget=budgets,species=species,elements=elements,
            temperature_K=t,pressure_bar=p,C_over_O=previous["C_over_O"],
            converged=np.ones(len(t),dtype=bool),tighter_case_indices=selected,tighter_reference_ln_n=tight)
        summaries.append(summary)
        print(json.dumps({"network":name,**summary}),flush=True)
    summary={"dataset":"curated-ps-v1","base_dataset":"curated-cn-v1",
             "epsilon_crit":5e-14,"strict_epsilon_crit":2e-14,"method":"vmap_cold",
             "new_production_solves":150,"tighter_reference_solves":6,
             "networks":summaries,"independent_solver":independent if executable else "not_run",
             "scope":"Compact reviewed PS retained-energy POPS thermal update on cumulative CN base; fixed1500-4000K ideal-gas networks. Old single-candidate abundance differences include CN background changes. Full physical accuracy remains unknown."}
    for filename,value in (("summary.json",summary),("diagnostics.json",diagnostics)):
        (output/filename).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    prior=json.loads((CANDIDATE/"receipt.json").read_text())
    paths={ROOT/name for name in prior["selected_file_sha256"]}
    paths.update((Path(__file__),ROOT/"mceg/reviews/build_ps_adoption.py"))
    paths.update(path for path in (ROOT/"src/exogibbs/presets").glob("curated*.py") if path.name!="curated.py")
    paths.update((ROOT/"src/exogibbs/data/curated").glob("*_v1.json"))
    for directory in (BASELINE,OLD_BASELINE):
        paths.update(directory/f"{name}.npz" for name in ("ions","neutral"))
    paths.add(CANDIDATE/"ions.npz")
    paths.update(ROOT/"tests/data/validation/curated_ps_adoption/20261106"/name
                 for name in ("fit_validation.json","fit_receipt.json"))
    receipt={"created_utc":datetime.now(timezone.utc).isoformat(),"result":"passed",
             "python":platform.python_version(),"numpy":np.__version__,"jax":jax.__version__,
             "backend":jax.default_backend(),"dtype":"float64","command":[sys.executable,*sys.argv],
             "selected_file_sha256":{str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)},
             "independent_executable_sha256":digest(executable) if executable else None,
             "output_sha256":{p.name:digest(p) for p in sorted(output.iterdir())}}
    (output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--fastchem-executable",type=Path)
    args=parser.parse_args()
    run(args.output_dir,args.fastchem_executable)
