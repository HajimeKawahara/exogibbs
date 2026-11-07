"""Construct H2O2 candidates without changing the named curated-cos baseline."""

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

import h2o2_source_review as source
from gas_source_comparison import ROOT, file_hash
from tio_candidate import _nasa

DATA = ROOT / "tests/data/validation/h2o2_candidate/20261105"
R, NASA_R = source.R, source.NASA_R
VARIANTS = ('energy_only',)
PRIMARY = 'energy_only'
PROPERTY_KEYS = ("G_J_mol", "H_J_mol", "S_J_mol_K", "Cp_J_mol_K")


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    """Read exact source coefficients and balanced source energy references."""
    return json.loads((DATA / "candidate.json").read_text())


def build() -> dict:
    """Build one consistent NASA reaction and/or retain the literal thermal fit."""
    inputs = source.inputs()
    records = json.loads((source.DATA / "nasa_records.json").read_text())
    atct = inputs["atct"]
    d0 = 1000*(sum(n*atct[e]["hf0_kJ_mol"] for e,n in source.COMPOSITION.items())-atct[source.NAME]["hf0_kJ_mol"])
    report = source.review()
    old = report["inherited_source_D0_J_mol"]
    native = report["source_samples"][0]["table_source"]["native_nasa_D0_J_mol"]
    high = {s:np.array(next(row["coefficients"] for row in record["intervals"] if row["bounds_K"]==[1000.,6000.])) for s,record in records.items()}
    reaction = high[source.NAME]-sum(n*high[e] for e,n in source.COMPOSITION.items())
    reaction[7] -= d0/R-native/NASA_R
    path = ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat"
    lines = path.read_text().splitlines()
    index = next(i for i,line in enumerate(lines) if line.startswith(source.SPECIES+" "))
    current = list(map(float,lines[index+1].split()))
    inputs_paths = [source.DATA / p for p in ("inputs.json","nasa_records.json","comparison.json","sources.json")]+[path]
    return {"id":"h2o2_atct_1_228_candidates_v1","species":source.SPECIES,"molecule":source.NAME,
            "base_dataset":"curated-cos-v1","composition":source.COMPOSITION,
            "variants":list(VARIANTS),"primary":PRIMARY,"temperature_range_K":[1500,4000],
            "standard_pressure_Pa":100000,"gas_constant_J_mol_K":R,"nasa_native_gas_constant_J_mol_K":NASA_R,
            "atct_version":"1.228","same_version_hf0_kJ_mol":{s:r["hf0_kJ_mol"] for s,r in atct.items()},
            "D0_J_mol":d0,"documented_source_D0_J_mol":old,"native_NASA_D0_J_mol":native,
            "delta_D0_J_mol":d0-old,"delta_D0_over_R_K":(d0-old)/R,
            "NASA_energy_shift_over_R_K":d0/R-native/NASA_R,
            "current_fastchem_coefficients":current,"current_source_header":lines[index],
            "nasa_formation_coefficients":reaction.tolist(),
            "thermal_scope":"Exact continuous high NASA interval on 1500-4000 K, or unchanged current five-coefficient reaction; no new fit or internal join.",
            "energy_only_reference_scope":"Offset uses the documented molecular-table plus JANAF-atom energy anchor; no unique zero-temperature energy is inferred from the finite-range FastChem fit. Historical fit atom basis remains unrecovered.",
            "nasa_convention":"Retain dimensionless NASA molecular-minus-atoms thermal functions; remove source D0 with native NASA R, then insert same-version ATcT D0 with SI R exactly once.",
            "physical_scope":"Neutral ground-state elemental species; no added conformers, isotopic mixture, nuclear-spin factor, physical covariance or high-temperature accuracy certificate.",
            "source_sha256":{str(p.relative_to(ROOT)):file_hash(p) for p in inputs_paths}}


def _gibbs(t, variant, xp):
    model = load_candidate()
    if variant not in ("current",*VARIANTS):
        raise ValueError("Unknown candidate variant")
    if variant=="nasa_atct":
        return _nasa(t,model["nasa_formation_coefficients"],xp)
    a,b,c,d,e = model["current_fastchem_coefficients"]
    shift = model["delta_D0_over_R_K"] if variant=="energy_only" else 0.
    g = -a-b*t*xp.log(t)-c*t-d*t*t-e*t**3-shift
    first = -b*(xp.log(t)+1)-c-2*d*t-3*e*t*t
    second = -b/t-2*d-6*e*t
    return g,first,second


def _temperature(temperature):
    if not math.isfinite(temperature) or not 1500<=temperature<=4000:
        raise ValueError("Temperature outside candidate domain")
    return float(temperature)


def formation_h(temperature: float, variant: str = PRIMARY) -> float:
    """Return one-bar Gformation/RT=-lnK for neutral atoms forming H2O2."""
    t=_temperature(temperature)
    return float(_gibbs(t,variant,np)[0]/t)


def formation_h_jax(temperature, variant: str = PRIMARY):
    """Evaluate traced scalar/array temperatures with NaN outside the domain."""
    import jax.numpy as jnp
    t=jnp.asarray(temperature,dtype=jnp.asarray(load_candidate()["current_fastchem_coefficients"]).dtype)
    valid=jnp.isfinite(t)&(t>=1500)&(t<=4000)
    safe=jnp.where(valid,t,1500.)
    return jnp.where(valid,_gibbs(safe,variant,jnp)[0]/safe,jnp.nan)


def reaction_properties(temperature: float, variant: str = PRIMARY) -> dict:
    """Derive G/H/S/Cp from a single smooth Gibbs representation."""
    t=_temperature(temperature)
    g,first,second=_gibbs(t,variant,np)
    return dict(zip(PROPERTY_KEYS,map(float,(R*g,R*(g-t*first),-R*first,-R*t*second))))


def validate() -> dict:
    """Check dense production replay, independent source reconstruction and AD."""
    import jax
    from exogibbs.presets import curated_cos
    model=load_candidate()
    assert model==build()
    previous=jax.config.x64_enabled
    jax.config.update("jax_enable_x64",True)
    try:
        setup=curated_cos.chemsetup(ions=True)
        t=np.arange(1500.,4001.)
        current=np.asarray(setup.hvector_func(t))[:,setup.species.index(source.SPECIES)]
        replay=np.array([formation_h(x,"current") for x in t])
        assert abs(current-replay).max()<2e-12
        records=json.loads((source.DATA/"nasa_records.json").read_text())
        results={}
        for variant in VARIANTS:
            props=np.array([list(reaction_properties(x,variant).values()) for x in t])
            f=lambda temp:formation_h_jax(temp,variant)
            vals=np.asarray(jax.jit(f)(t))
            first=np.asarray(jax.jit(jax.vmap(jax.grad(f)))(t))
            second=np.asarray(jax.jit(jax.vmap(jax.grad(jax.grad(f))))(t))
            ad=np.stack((R*t*vals,-R*t*t*first,-R*(vals+t*first),-R*(2*t*first+t*t*second)),axis=-1)
            if variant=="energy_only":
                old=np.array([list(reaction_properties(x,"current").values()) for x in t])
                expected=old.copy();expected[:,:2]-=model["delta_D0_J_mol"]
                expected_h=current-model["delta_D0_over_R_K"]/t
            else:
                raw=np.array([np.array(source.nasa_properties(records[source.NAME],x,NASA_R))-sum(n*np.array(source.nasa_properties(records[e],x,NASA_R)) for e,n in source.COMPOSITION.items()) for x in t])
                h=raw[:,0]*R/NASA_R-R*model["NASA_energy_shift_over_R_K"]
                s=raw[:,1]*R/NASA_R;cp=raw[:,2]*R/NASA_R
                expected=np.stack((h-t*s,h,s,cp),axis=-1);expected_h=expected[:,0]/(R*t)
            err=dict(zip(PROPERTY_KEYS,map(float,abs(props-expected).max(axis=0))))
            aderr=dict(zip(PROPERTY_KEYS,map(float,abs(props-ad).max(axis=0))))
            assert max(err.values())<1e-7 and max(aderr.values())<1e-7
            assert abs(vals-expected_h).max()<2e-12
            results[variant]={"maximum_h_reconstruction_error":float(abs(vals-expected_h).max()),"maximum_property_reconstruction_errors":err,"maximum_autodiff_errors":aderr,
                              "samples":[{"temperature_K":x,"K_over_current":math.exp(formation_h(x,"current")-formation_h(x,variant)),"reaction_properties":reaction_properties(x,variant)} for x in [1500.,2000.,2500.,3000.,4000.]]}
        return {"base_dataset":"curated-cos-v1","temperature_count":len(t),"range_K":[1500,4000],"step_K":1,"maximum_current_h_replay_error":float(abs(current-replay).max()),"variants":results,"internal_joins":[],"production_change":"none","physical_accuracy":"unknown"}
    finally:
        jax.config.update("jax_enable_x64",previous)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--build",action="store_true");args=parser.parse_args()
    if args.build:
        DATA.mkdir(parents=True,exist_ok=True);(DATA/"candidate.json").write_text(json.dumps(build(),indent=2)+"\n");load_candidate.cache_clear()
    result=validate()
    if args.build:
        (DATA/"validation.json").write_text(json.dumps(result,indent=2)+"\n")
        paths={Path(__file__),Path(source.__file__),ROOT/"mceg/reviews/tio_candidate.py",ROOT/"mceg/reviews/c2h_source_review.py",ROOT/"mceg/reviews/gas_source_comparison.py",ROOT/"mceg/reviews/curated_history.py",ROOT/"examples/comparisons/select_major_molecules.py",ROOT/"examples/comparisons/audit_atomic_ions.py",ROOT/"examples/comparisons/data/major_molecule_reference.json"}
        paths.update(ROOT/p for p in json.loads((source.DATA/"receipt.json").read_text())["selected_input_sha256"])
        paths.update((ROOT/"src/exogibbs").rglob("*.py"));paths.update((ROOT/"src/exogibbs/data/curated").glob("*.json"))
        paths.update(ROOT/p for p in ["src/exogibbs/data/FastChem4/logK/logK.dat","src/exogibbs/data/FastChem4/element_abundances/asplund_2021.dat"])
        receipt={"created_utc":datetime.now(timezone.utc).isoformat(),"command":sys.argv,"python":platform.python_version(),"dtype":"float64","result":"passed","selected_file_sha256":{str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)},"output_sha256":{n:file_hash(DATA/n) for n in ["candidate.json","validation.json"]}}
        (DATA/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(result,indent=2))
