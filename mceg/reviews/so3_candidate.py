"""Evaluate SO3 ATcT1.228 energy with the literal current NASA thermal function."""

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

from gas_source_comparison import ROOT, file_hash
from tio_candidate import _nasa
import select_major_molecules as major

NAME, SPECIES = "SO3", "O3S1"
COMPOSITION = {"O":3,"S":1}
DATA = ROOT / "tests/data/validation/so3_candidate/20261105"
SOURCE = ROOT / "tests/data/validation/so3_source_review/20261105"
R = major.R
INPUTS = (ROOT/"src/exogibbs/data/curated/gas_v1.json", major.REFERENCE_PATH,
          SOURCE/"inputs.json", DATA/"sources.json")


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    """Load the literal current reaction and the balanced version-to-version shift."""
    return json.loads((DATA/"candidate.json").read_text())


def build() -> dict:
    """Balance molecule and all atoms within each ATcT version, then shift once."""
    production=json.loads(INPUTS[0].read_text())
    reference=json.loads(INPUTS[1].read_text())
    facts=json.loads(INPUTS[2].read_text())
    old={n:reference["atct"]["records"][n]["hf0_kJ_mol"] for n in (*COMPOSITION,NAME)}
    new={n:row["hf0_kJ_mol"] for n,row in facts["atct"].items()}
    balance=lambda records:sum(c*round(1000*records[n]) for n,c in COMPOSITION.items())-round(1000*records[NAME])
    before,after=balance(old),balance(new)
    return {"id":"so3_retained_nasa_thermal_atct1228_candidate_v1", "species":SPECIES,
            "reaction":"3 O(g)+S(g)->SO3(g)","composition":COMPOSITION,
            "temperature_range_K":[1500,4000],"standard_pressure_Pa":100000,
            "gas_constant_J_mol_K":R,"base_dataset":"curated-cos-v1",
            "baseline_ATcT_version":reference["atct"]["version"],"candidate_ATcT_version":"1.228",
            "baseline_hf0_kJ_mol":old,"candidate_hf0_kJ_mol":new,
            "base_D0_J_mol":before,"D0_J_mol":after,"delta_D0_J_mol":after-before,
            "delta_D0_over_R_K":(after-before)/R,
            "original_reaction_coefficients":production["nasa9_reactions"][SPECIES],
            "representation":"Retain the exact adopted NASA9 reaction; subtract only delta(D0/R) from G/R. No thermal refit, isotope change or double energy alignment.",
            "uncertainty_scope":"Evaluated central values in two correlated ATcT versions; version shift is not an independent measurement or a significance test. Reaction covariance and full thermal uncertainty are unavailable.",
            "source_sha256":{str(p.relative_to(ROOT)):file_hash(p) for p in INPUTS}}


def _formation(t,updated,xp):
    record=load_candidate()
    g,slope,curvature=_nasa(t,record["original_reaction_coefficients"],xp)
    return g-(record["delta_D0_over_R_K"] if updated else 0.0),slope,curvature


def formation_h(temperature_K: float, updated: bool = True) -> float:
    """Return one-bar Gformation/(RT) without extrapolating the production domain."""
    if not math.isfinite(temperature_K) or not 1500 <= temperature_K <= 4000:
        raise ValueError("Temperature outside SO3 candidate domain")
    t=float(temperature_K)
    return float(_formation(t,updated,np)[0]/t)


def formation_h_jax(temperature_K, updated: bool = True):
    """Support scalar/array/JIT/vmap evaluation and return NaN outside the domain."""
    import jax.numpy as jnp
    t=jnp.asarray(temperature_K,dtype=jnp.asarray(load_candidate()["original_reaction_coefficients"]).dtype)
    valid=jnp.isfinite(t)&(t>=1500)&(t<=4000)
    safe=jnp.where(valid,t,1500.0)
    return jnp.where(valid,_formation(safe,updated,jnp)[0]/safe,jnp.nan)


def reaction_properties(temperature_K: float, updated: bool = True) -> dict:
    """Use the continuous potential to obtain consistent reaction G/H/S/Cp."""
    formation_h(temperature_K,updated)
    t=float(temperature_K);g,slope,curvature=_formation(t,updated,np)
    return dict(zip(("G_J_mol","H_J_mol","S_J_mol_K","Cp_J_mol_K"),
                    map(float,(R*g,R*(g-t*slope),-R*slope,-R*t*curvature))))


def validate() -> dict:
    """Reconstruct native NASA thermal terms and verify the named current factory."""
    import jax
    from exogibbs.presets import curated_cos
    model=load_candidate();assert model==build()
    previous=jax.config.x64_enabled;jax.config.update("jax_enable_x64",True)
    try:
        setup=curated_cos.chemsetup(ions=True)
        assert setup.metadata["dataset"]==model["base_dataset"]
        ts=np.arange(1500.,4001.);reference=json.loads(INPUTS[1].read_text())
        current=np.asarray(setup.hvector_func(ts))[:,setup.species.index(SPECIES)]
        before=np.array([formation_h(t,False) for t in ts]);after=np.array([formation_h(t) for t in ts])
        independent=-np.array([major.formation_log_kp(NAME,t,"nasa9",model["D0_J_mol"]/R,reference) for t in ts])
        oldprops=np.array([list(reaction_properties(t,False).values()) for t in ts])
        props=np.array([list(reaction_properties(t).values()) for t in ts]);expected=np.zeros_like(props);expected[:,:2]=-model["delta_D0_J_mol"]
        f=jax.jit(formation_h_jax);g=lambda t:R*t*f(t)
        slope=np.asarray(jax.jit(jax.vmap(jax.grad(g)))(ts));curvature=np.asarray(jax.jit(jax.vmap(jax.grad(jax.grad(g))))(ts))
        jg=R*ts*np.asarray(f(ts));jprops=np.stack([jg,jg-ts*slope,-slope,-ts*curvature],axis=-1)
        keys=list(reaction_properties(1500))
        result={"temperature_count":len(ts),"range_K":[1500,4000],"step_K":1,
            "maximum_abs_current_h_replay_error":float(abs(before-current).max()),
            "maximum_abs_independent_native_source_h_error":float(abs(after-independent).max()),
            "maximum_abs_energy_shift_h_error":float(abs(after-current+model["delta_D0_over_R_K"]/ts).max()),
            "maximum_abs_energy_only_property_errors":dict(zip(keys,map(float,abs(props-oldprops-expected).max(axis=0)))),
            "maximum_abs_jax_property_errors":dict(zip(keys,map(float,abs(jprops-props).max(axis=0)))),
            "samples":[{"temperature_K":t,"candidate_over_current_K":math.exp(model["delta_D0_over_R_K"]/t),"reaction_properties":reaction_properties(t)} for t in (1500,2000,2500,3000,4000)],
            "production_change":"none","new_fit":"none"}
        assert max(result[k] for k in result if k.endswith('_h_error') or k.endswith('_replay_error'))<1e-12
        return result
    finally:
        jax.config.update("jax_enable_x64",previous)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--build",action="store_true");args=parser.parse_args()
    if args.build:
        DATA.mkdir(parents=True,exist_ok=True);(DATA/"candidate.json").write_text(json.dumps(build(),indent=2)+"\n");load_candidate.cache_clear()
    result=validate()
    if args.build:
        import jax
        (DATA/"validation.json").write_text(json.dumps(result,indent=2)+"\n")
        paths={Path(module.__file__).resolve() for module in list(sys.modules.values()) if getattr(module,"__file__",None) and Path(module.__file__).resolve().is_relative_to(ROOT) and Path(module.__file__).suffix=='.py' and Path(module.__file__).name!='exogibbs_version.py'}
        paths.update(INPUTS);paths.add(Path(__file__).resolve())
        paths.update((ROOT/"src/exogibbs/data/curated").glob('*.json'))
        paths.update(ROOT/"src/exogibbs/data"/n for n in ('FastChem4/logK/logK.dat','FastChem4/element_abundances/asplund_2021.dat'))
        receipt={"created_utc":datetime.now(timezone.utc).isoformat(),"command":[sys.executable,*sys.argv],"python":platform.python_version(),"jax":jax.__version__,"numpy":np.__version__,"dtype":"float64","backend":"cpu","result":"passed","selected_file_sha256":{str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)},"output_sha256":{name:file_hash(DATA/name) for name in ('candidate.json','validation.json')}}
        (DATA/'receipt.json').write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(result,indent=2))
