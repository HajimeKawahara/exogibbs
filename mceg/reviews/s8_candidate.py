"""Build an S8 W4L energy correction while retaining the current thermal fit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

from c2h_source_review import read_table
from gas_source_comparison import ROOT, file_hash
import s8_source_review as source

DATA = ROOT / "tests/data/validation/s8_candidate/20261105"
R = source.R
FASTCHEM = ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat"
INPUTS = (FASTCHEM, source.DATA / "inputs.json", source.DATA / "sources.json", source.ATOMIC_TABLE)


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    """Load the pinned source anchor and literal current five-coefficient fit."""
    return json.loads((DATA / "candidate.json").read_text())


def build() -> dict:
    """Balance the original zero-K atoms; use direct TAE0 rather than molecular Hf."""
    facts = json.loads((source.DATA / "inputs.json").read_text())
    atom = read_table(source.ATOMIC_TABLE)
    original_D0 = 1000*(8*atom[0][4]-facts["janaf_nodes"]["0"][4])
    target_D0 = facts["karton2021"]["TAE0_kcal_mol"]*4184
    lines = FASTCHEM.read_text().splitlines()
    index = next(i for i,line in enumerate(lines) if line.startswith("S8 "))
    coefficients = list(map(float, lines[index+1].split()))
    return {"id":"s8_retained_thermal_w4l_energy_candidate_v1", "species":"S8",
            "reaction":"8 S(g) -> S8(g)", "base_dataset":"curated-cos-v1",
            "temperature_range_K":[1500,4000], "standard_pressure_Pa":100000,
            "gas_constant_J_mol_K":R, "base_source_D0_J_mol":original_D0,
            "direct_W4L_TAE0_J_mol":target_D0, "delta_D0_J_mol":target_D0-original_D0,
            "delta_D0_over_R_K":(target_D0-original_D0)/R,
            "original_logK_coefficients":coefficients,
            "source_doi":"10.1016/j.chphi.2021.100047",
            "representation":"Current five-coefficient reaction with one additive delta(D0)/(R T) in ln K; no thermal refit or extra atomic shift",
            "anchor_scope":"Difference between direct W4L TAE0 and balanced same-version JANAF zero-K source rows; not a unique D0 extrapolated from the finite-range inherited fit. Its fitting residual and thermal uncertainty remain.",
            "uncertainty_scope":"Printed central values; no inferred reaction covariance, complete state-sum uncertainty or physical abundance bounds",
            "source_sha256":{str(p.relative_to(ROOT)):file_hash(p) for p in INPUTS}}


def _ln_k(t, updated, xp):
    a,b,c,d,e = load_candidate()["original_logK_coefficients"]
    shift = load_candidate()["delta_D0_over_R_K"] if updated else 0.0
    return (a+shift)/t+b*xp.log(t)+c+d*t+e*t*t


def formation_h(temperature_K: float, updated: bool = True) -> float:
    """Return -ln K on the production range, without extrapolating to zero K."""
    if not math.isfinite(temperature_K) or not 1500 <= temperature_K <= 4000:
        raise ValueError("Temperature outside S8 candidate domain")
    return float(-_ln_k(float(temperature_K), updated, np))


def formation_h_jax(temperature_K, updated: bool = True):
    """Evaluate scalars/arrays/JIT; invalid traced temperatures return NaN."""
    import jax.numpy as jnp
    t = jnp.asarray(temperature_K, dtype=jnp.asarray(load_candidate()["original_logK_coefficients"]).dtype)
    valid = jnp.isfinite(t) & (t >= 1500) & (t <= 4000)
    safe = jnp.where(valid,t,1500.0)
    return jnp.where(valid,-_ln_k(safe,updated,jnp),jnp.nan)


def reaction_properties(temperature_K: float, updated: bool = True) -> dict:
    """Use one continuous potential for reaction G/H/S/Cp; only G/H shift."""
    t = float(temperature_K)
    g = R*t*formation_h(t,updated)
    a,b,c,d,e = load_candidate()["original_logK_coefficients"]
    h = R*(-a+b*t+d*t*t+2*e*t**3)
    if updated:
        h -= load_candidate()["delta_D0_J_mol"]
    return {"G_J_mol":g, "H_J_mol":h,
            "S_J_mol_K":R*(b*math.log(t)+b+c+2*d*t+3*e*t*t),
            "Cp_J_mol_K":R*(b+2*d*t+6*e*t*t)}


def validate() -> dict:
    """Check the actual named runtime, caloric identities and source-fit residuals."""
    import jax
    from exogibbs.presets import curated_cos
    model=load_candidate()
    assert model==build()
    previous=jax.config.x64_enabled
    jax.config.update("jax_enable_x64",True)
    try:
        setup=curated_cos.chemsetup(ions=True)
        assert setup.metadata["dataset"]==model["base_dataset"]
        ts=np.arange(1500.,4001.)
        old=np.array([formation_h(t,False) for t in ts])
        new=np.array([formation_h(t) for t in ts])
        current=np.asarray(setup.hvector_func(ts))[:,setup.species.index("S8")]
        props=np.array([list(reaction_properties(t).values()) for t in ts])
        oldprops=np.array([list(reaction_properties(t,False).values()) for t in ts])
        expected=np.zeros_like(props); expected[:,:2]=-model["delta_D0_J_mol"]
        f=jax.jit(formation_h_jax);g=lambda t:R*t*f(t)
        slope=np.asarray(jax.jit(jax.vmap(jax.grad(g)))(ts))
        curvature=np.asarray(jax.jit(jax.vmap(jax.grad(jax.grad(g))))(ts))
        jg=R*ts*np.asarray(f(ts))
        jprops=np.stack([jg,jg-ts*slope,-slope,-ts*curvature],axis=-1)
        facts=json.loads((source.DATA/"inputs.json").read_text());atoms=read_table(source.ATOMIC_TABLE)
        samples=[]
        for t in (1500,2000,3000,4000):
            mol=facts["janaf_nodes"][str(t)];ref=facts["janaf_nodes"]["298.15"];atom=atoms[t];ar=atoms[298.15]
            table={"H_J_mol":1000*(ref[4]+mol[3]-8*(ar[4]+atom[3])),
                   "S_J_mol_K":mol[1]-8*atom[1],"Cp_J_mol_K":mol[0]-8*atom[0]}
            before=reaction_properties(t,False)
            samples.append({"temperature_K":t,"candidate_over_current_K":math.exp(model["delta_D0_over_R_K"]/t),
                "current_minus_JANAF_reaction":{key:before[key]-value for key,value in table.items()},
                "candidate_reaction_properties":reaction_properties(t)})
        keys=list(reaction_properties(1500))
        result={"temperature_count":len(ts),"range_K":[1500,4000],"step_K":1,
            "maximum_abs_current_h_replay_error":float(abs(old-current).max()),
            "maximum_abs_energy_shift_h_error":float(abs(new-current+model["delta_D0_over_R_K"]/ts).max()),
            "maximum_abs_energy_only_property_errors":dict(zip(keys,map(float,abs(props-oldprops-expected).max(axis=0)))),
            "maximum_abs_jax_property_errors":dict(zip(keys,map(float,abs(jprops-props).max(axis=0)))),
            "samples":samples,"production_change":"none","new_fit":"none"}
        assert result["maximum_abs_current_h_replay_error"]<1e-12
        assert result["maximum_abs_energy_shift_h_error"]<1e-12
        return result
    finally:
        jax.config.update("jax_enable_x64",previous)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build",action="store_true")
    args=parser.parse_args()
    if args.build:
        DATA.mkdir(parents=True,exist_ok=True)
        (DATA/"candidate.json").write_text(json.dumps(build(),indent=2)+"\n")
        load_candidate.cache_clear()
    result=validate()
    if args.build:
        import jax
        (DATA/"validation.json").write_text(json.dumps(result,indent=2)+"\n")
        files=[Path(__file__),ROOT/"mceg/reviews/s8_source_review.py",*INPUTS]
        files += [ROOT/p for p in json.loads((source.DATA/"receipt.json").read_text())["input_and_artifact_sha256"] if p.startswith("src/")]
        files += [ROOT/"mceg/reviews"/f"{n}.py" for n in ("c2h_source_review","gas_source_comparison","curated_history")]
        files += [ROOT/"examples/comparisons/audit_atomic_ions.py"]
        receipt={"created_utc":datetime.now(timezone.utc).isoformat(),"command":[sys.executable,*sys.argv],
            "python":platform.python_version(),"jax":jax.__version__,"numpy":np.__version__,"dtype":"float64","backend":"cpu","result":"passed",
            "selected_file_sha256":{str(p.relative_to(ROOT)):file_hash(p) for p in sorted(set(files))},
            "output_sha256":{name:file_hash(DATA/name) for name in ("candidate.json","validation.json")}}
        (DATA/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(result,indent=2))
