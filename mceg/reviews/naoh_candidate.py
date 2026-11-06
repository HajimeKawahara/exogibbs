"""Evaluate NaOH energy/thermal candidates without changing production data."""

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

import naoh_source_review as source
from gas_source_comparison import file_hash
from tio_candidate import _nasa

ROOT = source.ROOT
DATA = ROOT / "tests/data/validation/naoh_candidate/20261105"
NAME, SPECIES, ELEMENT = "NaOH", "H1Na1O1", "Na"
PRIMARY, CONTROL = "energy", "nasa"
R, NASA_R = source.R, source.NASA_R
FASTCHEM = ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat"
ATOM_INPUTS = (ROOT / "tests/data/validation/nao_source_review/20261105/inputs.json",
               ROOT / "tests/data/validation/kh_source_review/20261105/inputs.json")
INPUTS = (source.DATA / "inputs.json", source.DATA / "janaf.txt", FASTCHEM, *ATOM_INPUTS)
VARIANTS = ("current", "energy", "nasa", "nasa_retained")


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    """Load the fixed energy references and smooth candidate coefficients."""
    return json.loads((DATA / "candidate.json").read_text())


def build() -> dict:
    """Balance source zero-energy gauges without extrapolating a finite-range fit."""
    inputs = json.loads(INPUTS[0].read_text())
    molecule = source.read_table(INPUTS[1])
    atoms = {}
    for path in ATOM_INPUTS:
        atoms.update(json.loads(path.read_text())["janaf"])
    composition = inputs["composition"]
    old_h0 = {e: 1000*(atoms[e]["298.15"][4]+atoms[e]["0.0"][3]) for e in composition}
    old_h0[NAME] = 1000*(molecule[298.15][4]+molecule[0.0][3])
    old_d0 = sum(n*old_h0[e] for e,n in composition.items())-old_h0[NAME]
    records = inputs["nasa_records"]
    native_h0 = {e: records[e]["hf298_J_mol"]-records[e]["h298_minus_h0_J_mol"]
                 for e in (*composition, NAME)}
    nasa_d0 = sum(n*native_h0[e] for e,n in composition.items())-native_h0[NAME]
    # Use the evaluated printed molecular heat content, retaining its small
    # rounding/fitting difference from the NASA header explicitly.
    target_h0 = inputs["published_hf298_J_mol"]-inputs["published_h298_minus_h0_J_mol"]
    target_d0 = sum(n*native_h0[e] for e,n in composition.items())-target_h0
    lines = FASTCHEM.read_text().splitlines()
    index = next(i for i,line in enumerate(lines) if line.startswith(SPECIES+" "))
    coefficients = list(map(float,lines[index+1].split()))
    assert len(coefficients) == 5
    native = {e: np.array(next(row["coefficients"] for row in records[e]["intervals"]
                  if row["bounds_K"] == [1000.,6000.])) for e in (*composition, NAME)}
    reaction = native[NAME]-sum(n*native[e] for e,n in composition.items())
    nasa_coefficients = {}
    for key,d0 in (("nasa",target_d0),("nasa_retained",old_d0)):
        corrected = reaction.copy()
        corrected[7] -= d0/R-nasa_d0/NASA_R
        nasa_coefficients[key] = corrected.tolist()
    return {"id":"naoh_controlled_gurvich_candidate_v1", "molecule":NAME, "species":SPECIES,
            "baseline_dataset":"curated-cos-v1", "standard_pressure_Pa":100000,
            "temperature_range_K":[1500,4000], "primary":PRIMARY, "control":CONTROL,
            "composition":composition, "gas_constant_J_mol_K":R, "nasa_native_R_J_mol_K":NASA_R,
            "janaf_source_h0_gauge_J_mol":old_h0, "nasa_native_h0_gauge_J_mol":native_h0,
            "Gurvich_molecular_hf298_J_mol":inputs["published_hf298_J_mol"],
            "Gurvich_molecular_heat_content_J_mol":inputs["published_h298_minus_h0_J_mol"],
            "Gurvich_molecular_h0_gauge_J_mol":target_h0,
            "original_source_D0_J_mol":old_d0, "target_D0_J_mol":target_d0,
            "nasa_native_D0_J_mol":nasa_d0, "delta_D0_J_mol":target_d0-old_d0,
            "atomic_h0_change_J_mol":sum(n*(native_h0[e]-old_h0[e]) for e,n in composition.items()),
            "molecular_h0_change_J_mol":target_h0-old_h0[NAME],
            "original_logK_coefficients":coefficients,
            "nasa_coefficients":nasa_coefficients,
            "molecular_Hf298_evaluation_margin_J_mol":8000,
            "uncertainty_scope":"Published total evaluation margin, not an independent Gaussian error or balanced D0 covariance. Atom references, heat-content approximation and thermal-model limitations remain explicit.",
            "thermal_scope":"NASA/Gurvich molecular and atomic thermal functions are a declared reference alternative. The full-range OYT partition sum is not adopted or extrapolated.",
            "anchor_scope":"Original D0 is the documented JANAF source anchor, not an inferred unique zero-temperature limit of the finite-range inherited fit.",
            "source_sha256":{str(p.relative_to(ROOT)):file_hash(p) for p in INPUTS}}


def _formation(t, variant, xp):
    model = load_candidate()
    if variant in ("current", "energy"):
        a,b,c,d,e = model["original_logK_coefficients"]
        if variant == "energy":
            a += model["delta_D0_J_mol"]/R
        return -(a/t+b*xp.log(t)+c+d*t+e*t*t)
    if variant in model["nasa_coefficients"]:
        return _nasa(t,model["nasa_coefficients"][variant],xp)[0]/t
    raise ValueError("Unknown candidate variant")


def formation_h(temperature_K: float, variant: str = PRIMARY) -> float:
    """Return Gformation/(RT) on the declared 1500--4000 K interval."""
    if not math.isfinite(temperature_K) or not 1500 <= temperature_K <= 4000:
        raise ValueError("Temperature outside NaOH candidate domain")
    return float(_formation(float(temperature_K),variant,np))


def formation_h_jax(temperature_K, variant: str = PRIMARY):
    """Support JIT and differentiation; invalid traced input returns NaN."""
    import jax.numpy as jnp
    t = jnp.asarray(temperature_K,dtype=jnp.asarray(load_candidate()["original_logK_coefficients"]).dtype)
    valid = jnp.isfinite(t)&(t>=1500)&(t<=4000)
    safe = jnp.where(valid,t,1500.)
    return jnp.where(valid,_formation(safe,variant,jnp),jnp.nan)


def reaction_properties(temperature_K: float, variant: str = PRIMARY) -> dict:
    """Return analytic reaction G/H/S/Cp from one smooth Gibbs function."""
    t = float(temperature_K)
    g = R*t*formation_h(t,variant)
    if variant in ("current","energy"):
        a,b,c,d,e = load_candidate()["original_logK_coefficients"]
        if variant == "energy": a += load_candidate()["delta_D0_J_mol"]/R
        h = R*(-a+b*t+d*t*t+2*e*t**3)
        entropy = R*(b*math.log(t)+b+c+2*d*t+3*e*t*t)
        cp = R*(b+2*d*t+6*e*t*t)
    else:
        value,slope,curvature = _nasa(t,load_candidate()["nasa_coefficients"][variant],np)
        h,entropy,cp = R*(value-t*slope),-R*slope,-R*t*curvature
    return dict(G_J_mol=g,H_J_mol=float(h),S_J_mol_K=float(entropy),Cp_J_mol_K=float(cp))


def validate() -> dict:
    """Verify source reconstruction, unchanged calorics and automatic derivatives."""
    import jax
    from exogibbs.presets import curated_cos
    model = load_candidate()
    assert model == build()
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64",True)
    try:
        base = curated_cos.chemsetup(ions=True)
        ts = np.arange(1500.,4001.)
        current = np.asarray(base.hvector_func(ts))[:,base.species.index(SPECIES)]
        values = {v:np.array([formation_h(t,v) for t in ts]) for v in VARIANTS}
        reconstruction = {"current_h":float(abs(current-values["current"]).max()),
                          "energy_h":float(abs(values["energy"]-current+model["delta_D0_J_mol"]/(R*ts)).max())}
        records = json.loads(INPUTS[0].read_text())["nasa_records"]
        native = np.array([source.nasa_g(records[NAME],t)-sum(n*source.nasa_g(records[e],t)
                         for e,n in model["composition"].items()) for t in ts])
        for variant,d0 in (("nasa",model["target_D0_J_mol"]),("nasa_retained",model["original_source_D0_J_mol"])):
            expected = native-(d0/R-model["nasa_native_D0_J_mol"]/NASA_R)/ts
            reconstruction[variant+"_h"] = float(abs(values[variant]-expected).max())
        derivatives = {}
        probes = np.array([1500.,1637.,2253.,3000.,3871.,4000.])
        for variant in VARIANTS:
            f = lambda t:R*t*formation_h_jax(t,variant)
            g = np.asarray(jax.vmap(f)(probes))
            dg = np.asarray(jax.vmap(jax.grad(f))(probes))
            ddg = np.asarray(jax.vmap(jax.grad(jax.grad(f)))(probes))
            actual = np.stack((g,g-probes*dg,-dg,-probes*ddg),axis=-1)
            expected = np.array([list(reaction_properties(t,variant).values()) for t in probes])
            derivatives[variant] = dict(zip(reaction_properties(1500).keys(),map(float,abs(actual-expected).max(axis=0))))
        assert max(reconstruction.values()) < 2e-12
        return {"temperature_count":len(ts), "maximum_absolute_reconstruction_error":reconstruction,
                "jax_float64_derivative_errors":derivatives,
                "samples":[{"temperature_K":t,"K_over_current":{v:math.exp(formation_h(t,"current")-formation_h(t,v)) for v in VARIANTS},
                            "properties":{v:reaction_properties(t,v) for v in VARIANTS}} for t in (1500.,2000.,2500.,3000.,4000.)],
                "production_change":"none", "OYT_adopted":False}
    finally:
        jax.config.update("jax_enable_x64",previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build",action="store_true")
    args = parser.parse_args()
    if args.build:
        DATA.mkdir(parents=True,exist_ok=True)
        (DATA/"candidate.json").write_text(json.dumps(build(),indent=2)+"\n")
        load_candidate.cache_clear()
    report = validate()
    if args.build:
        (DATA/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
