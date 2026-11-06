"""Verify a retained-thermal energy adoption using its actual named factory."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import platform
import sys

import jax
import numpy as np
from scipy.special import logsumexp

from c2h_equilibrium_impact import carbon_budget, independent_check
from curated_h2o_adoption import solve_cases
from hcp_equilibrium_impact import constraints


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "tests/data/validation/curated_tio2_adoption/20261106"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def fractions(amounts):
    return amounts-logsumexp(amounts, axis=-1, keepdims=True)


def archive(directory: Path, network: str) -> dict:
    """Check the original execution output hash before loading an archive."""
    path=directory/f"{network}.npz"
    receipt=json.loads((directory/"receipt.json").read_text())
    assert digest(path) == receipt["output_sha256"][path.name]
    with np.load(path) as data:
        return {key:data[key].copy() for key in data.files}


def run(slug: str, base_slug: str, variant: str, element: str,
        output: Path, executable: Path | None = None) -> None:
    """Solve 150 fresh production cases and six tighter independent references."""
    jax.config.update("jax_enable_x64",True)
    output.mkdir(parents=True,exist_ok=False)
    factory=import_module(f"exogibbs.presets.curated_{slug}")
    base_factory=import_module(f"exogibbs.presets.curated_{base_slug}")
    candidate=import_module(f"{slug}_candidate")
    data_path=ROOT/f"src/exogibbs/data/curated/{slug}_v1.json"
    data=json.loads(data_path.read_text())
    baseline=ROOT/f"tests/data/validation/curated_{base_slug}_adoption/20261106"
    candidate_path=ROOT/f"tests/data/validation/{slug}_equilibrium_impact/20261106"
    summaries,diagnostics,independent=[],[],[]
    for ions in (True,False):
        network="ions" if ions else "neutral"
        previous,original=archive(baseline,network),archive(ORIGINAL,network)
        setup,base=factory.chemsetup(ions=ions),base_factory.chemsetup(ions=ions)
        assert setup.metadata["dataset"] == f"curated-{slug}-v1"
        assert base.metadata["dataset"] == data["base_dataset"]
        assert setup.species==base.species and setup.elements==base.elements
        np.testing.assert_array_equal(setup.formula_matrix,base.formula_matrix)
        np.testing.assert_array_equal(setup.element_vector_reference,base.element_vector_reference)
        species,elements=list(setup.species),list(setup.elements)
        matrix=np.asarray(setup.formula_matrix)
        for key,value in (("species",species),("elements",elements),("formula_matrix",matrix)):
            for saved in (previous,original):np.testing.assert_array_equal(saved[key],value)
        for key in ("temperature_K","pressure_bar","budget","C_over_O"):
            np.testing.assert_array_equal(previous[key],original[key])
        t,p,budgets=(previous[key] for key in ("temperature_K","pressure_bar","budget"))
        assert len(t)==75 and previous["converged"].all()
        h=np.asarray(setup.hvector_func(t))
        np.testing.assert_allclose(base.hvector_func(t),previous["h"],atol=1e-12,rtol=0)
        target=species.index(data["species"])
        np.testing.assert_array_equal(np.delete(h-previous["h"],target,axis=-1),0)
        def reference_h(temperature):
            return base.hvector_func(temperature).at[...,target].set(candidate.formation_h_jax(temperature,variant))
        reference=replace(base,hvector_func=jax.jit(reference_h))
        dense_t=np.linspace(1500.,4000.,2501)
        dense_error=float(abs(np.asarray(setup.hvector_func(dense_t))-np.asarray(reference.hvector_func(dense_t))).max())
        assert dense_error<1e-12
        amounts=np.empty_like(previous["ln_n"])
        for ratio in np.unique(previous["C_over_O"]):
            indices=np.flatnonzero(previous["C_over_O"]==ratio)
            np.testing.assert_allclose(budgets[indices],np.broadcast_to(carbon_budget(base,float(ratio)),budgets[indices].shape),rtol=1e-15,atol=0)
            values,checks=solve_cases(setup,budgets[indices[0]],t[indices],p[indices],5e-14)
            amounts[indices]=values
            diagnostics.extend({"network":network,"case_index":int(i),**row} for i,row in zip(indices,checks))
            print(f"{slug} production {network} C/O={ratio}: {len(indices)} endpoints passed",flush=True)
        closure=constraints(matrix,elements,species,budgets,amounts,h,p)
        logx,prior_x,old_x=map(fractions,(amounts,previous["ln_n"],original["ln_n"]))
        x=np.exp(logx);ratios=np.exp(logx[:,target]-prior_x[:,target])
        atom_index=elements.index(element);stoich=matrix[atom_index,target]
        share_before=stoich*np.exp(previous["ln_n"][:,target])/budgets[:,atom_index]
        share_after=stoich*np.exp(amounts[:,target])/budgets[:,atom_index]
        highlight=int(np.argmax(abs(share_after-share_before)))
        selected={highlight,int(np.argmax(abs(np.log(ratios)))),int(np.argmax(closure["element_relative_residual"]))}
        for i in (4,49,74,0):
            if len(selected)==3:break
            selected.add(i)
        selected=np.array(sorted(selected));assert len(selected)==3
        tight=[]
        for i in selected:
            values,checks=solve_cases(reference,budgets[i],t[i:i+1],p[i:i+1],2e-14)
            tight.append(values[0]);diagnostics.extend({"network":network,"case_index":int(i),"tighter_reference":True,**row} for row in checks)
        tight=np.asarray(tight);tight_x=fractions(tight)
        constraints(matrix,elements,species,budgets[selected],tight,h[selected],p[selected])
        difference=logx[selected]-tight_x
        stability={"max_abs_delta_ln_x":float(abs(difference).max()),"max_target_abs_delta_ln_x":float(abs(difference[:,target]).max()),"max_absolute_delta_x":float(abs(x[selected]-np.exp(tight_x)).max())}
        assert stability["max_abs_delta_ln_x"]<1e-4
        assert stability["max_target_abs_delta_ln_x"]<1e-8
        assert stability["max_absolute_delta_x"]<1e-9
        response={"min_x_ratio":float(ratios.min()),"max_x_ratio":float(ratios.max()),"highlighted_case_index":highlight,"temperature_K":float(t[highlight]),"pressure_bar":float(p[highlight]),"C_over_O":float(previous["C_over_O"][highlight]),"x_before":float(np.exp(prior_x[highlight,target])),"x_after":float(x[highlight,target]),"allocation_element":element,"element_share_before":float(share_before[highlight]),"element_share_after":float(share_after[highlight]),"maximum_element_share_after":float(share_after.max()),"maximum_element_share_after_case":int(np.argmax(share_after))}
        summary={"ions":ions,"cases_passed":len(t),"species":len(species),"max_dense_h_difference_from_candidate_on_current_base":dense_error,"tighter_case_indices":selected.tolist(),"tighter_reference":stability,"closure":{key:float(np.max(value)) for key,value in closure.items()},"incremental_response":response,"cumulative_response_from_curated_tio2":{label:{"min_x_ratio":float(np.exp(logx[:,j]-old_x[:,j]).min()),"max_x_ratio":float(np.exp(logx[:,j]-old_x[:,j]).max()),"max_absolute_delta_x":float(abs(x[:,j]-np.exp(old_x[:,j])).max())} for label in ("C1H1P1","C2H2","C1H2") for j in [species.index(label)]}}
        if ions and executable:
            eligible=np.flatnonzero(np.isin(t,[1500.,2000.,3000.,4000.]))
            chosen={int(eligible[np.argmax(abs(share_after[eligible]-share_before[eligible]))]),int(eligible[np.argmax(abs(np.log(ratios[eligible])))] )}
            for i in (4,74):
                if len(chosen)==2:break
                chosen.add(i)
            assert len(chosen)==2
            for i in sorted(chosen):
                independent.append({"case_index":i,"comparisons":independent_check([setup],budgets[i],float(t[i]),float(p[i]),amounts[i:i+1],executable.resolve())})
        np.savez_compressed(output/f"{network}.npz",ln_n=amounts,ln_x=logx,h=h,formula_matrix=matrix,budget=budgets,species=species,elements=elements,temperature_K=t,pressure_bar=p,C_over_O=previous["C_over_O"],converged=np.ones(len(t),dtype=bool),tighter_case_indices=selected,tighter_reference_ln_n=tight)
        summaries.append(summary)
        print(json.dumps({"network":network,**summary}),flush=True)
    summary={"dataset":data["id"],"base_dataset":data["base_dataset"],"species":data["species"],"energy_increase_J_mol":-data["delta_D0_J_mol"],"epsilon_crit":5e-14,"strict_epsilon_crit":2e-14,"method":"vmap_cold","new_production_solves":150,"tighter_reference_solves":6,"networks":summaries,"independent_solver":independent if executable else "not_run","scope":"Actual versioned production factory; selected energy only, unchanged thermals and source-fit residual. Incremental and cumulative responses use distinct baselines. Finite numerical verification does not establish complete physical accuracy."}
    for filename,value in (("summary.json",summary),("diagnostics.json",diagnostics)):
        (output/filename).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    prior=json.loads((candidate_path/"receipt.json").read_text())
    paths={ROOT/name for name in prior["selected_file_sha256"] if name!="src/exogibbs/presets/curated.py"}
    paths.update((Path(__file__),ROOT/f"mceg/reviews/curated_{slug}_adoption.py",ROOT/f"mceg/reviews/build_{slug}_adoption.py",data_path))
    for name in data["source_sha256"]:paths.add(ROOT/name)
    for module in list(sys.modules.values()):
        name=getattr(module,"__file__",None)
        if name:
            path=Path(name).resolve()
            if path.is_relative_to(ROOT) and path.suffix==".py" and path!=ROOT/"src/exogibbs/presets/curated.py":paths.add(path)
    paths.update(path for path in (ROOT/"src/exogibbs/presets").glob("curated*.py") if path.name!="curated.py")
    paths.update((ROOT/"src/exogibbs/data/curated").glob("*_v1.json"))
    for directory in (baseline,ORIGINAL):
        paths.update(directory/name for name in ("ions.npz","neutral.npz","receipt.json"))
    receipt={"created_utc":datetime.now(timezone.utc).isoformat(),"result":"passed","python":platform.python_version(),"numpy":np.__version__,"jax":jax.__version__,"backend":jax.default_backend(),"dtype":"float64","command":[sys.executable,*sys.argv],"selected_file_sha256":{str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)},"independent_executable_sha256":digest(executable) if executable else None,"output_sha256":{p.name:digest(p) for p in sorted(output.iterdir())}}
    (output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")


def main(slug: str, base: str, variant: str, element: str) -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--fastchem-executable",type=Path)
    args=parser.parse_args()
    run(slug,base,variant,element,args.output_dir,args.fastchem_executable)
