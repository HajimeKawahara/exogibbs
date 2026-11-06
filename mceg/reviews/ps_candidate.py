"""Evaluate a retained-energy PS thermal candidate from finite POPS state sums."""

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

from c2h_source_review import nasa_properties
from gas_source_comparison import ROOT, file_hash
from ph3_source_review import AMU, KB, PLANCK, R, NASA_R, log_translation
from phosphorus_diatomic_source_review import C2
from tio_candidate import _nasa

SOURCE = ROOT / "tests/data/validation/ps_source_review/20261106"
DATA = ROOT / "tests/data/validation/ps_candidate/20261106"
FASTCHEM = ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat"
SPECIES = "P1S1"
VARIANTS = ("current", "pops", "full", "ground")
INPUTS = tuple(SOURCE/name for name in ("inputs.json", "states.npz", "state_inventory.json", "sources.json", "partition.pf")) + (FASTCHEM,)


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    """Load the pinned source definition and explicit physical conventions."""
    return json.loads((DATA/"candidate.json").read_text())


def build() -> dict:
    """Preserve the current fit while identifying the molecular thermal difference."""
    inputs = json.loads((SOURCE/"inputs.json").read_text())
    nasa = inputs["nasa_records"]["PS"]
    lines = FASTCHEM.read_text().splitlines()
    index = next(i for i,line in enumerate(lines) if line.startswith(SPECIES+" "))
    coefficients = list(map(float,lines[index+1].split()))
    assert len(coefficients) == 5
    return {"id":"ps_retained_lodders_pops_candidate_v1", "species":SPECIES,
            "baseline_dataset":"curated-cos-v1", "standard_pressure_Pa":100000,
            "temperature_range_K":[1500,4000], "primary":"pops", "composition":{"P":1,"S":1},
            "original_logK_coefficients":coefficients,
            "original_source_D0_J_mol":438000., "delta_D0_J_mol":0.,
            "D0_evaluation_margin_J_mol":10000.,
            "bound_cutoff_cm-1":438000/(R*C2),
            "NASA_molecular_coefficients":nasa["intervals"][1]["coefficients"],
            "NASA_molecular_H0_gauge_J_mol":nasa["hf298_J_mol"]-nasa["h298_minus_h0_J_mol"],
            "molecular_mass_u":inputs["natural_mass_u"], "isotope_mass_u":inputs["isotope_mass_u"],
            "nuclear_spin_divisor":2, "symmetry_number":1,
            "gas_constant_J_mol_K":R, "NASA_native_R_J_mol_K":NASA_R,
            "thermal_rule":"Current Gformation plus POPS molecular (G-H0) minus NASA/Gurvich molecular (G-H0); atomic functions and current fit/source residual are retained.",
            "energy_scope":"Retain the documented Lodders2004 D0=438 +/-10 kJ/mol source anchor. No zero-temperature extrapolation of the current finite-range fit, no energy correction, and no potential De substitution.",
            "cutoff_scope":"Fixed central-D0 bound-level cutoff. Full-distribution, lower-D0-cutoff and ground-electronic-only sums are sensitivities, not uncertainty bounds. Varying a cutoff is not a replacement for a correlated energy uncertainty calculation.",
            "isotope_scope":"31P32S internal levels with unchanged natural-mass translation form an element-aggregated proxy, not a natural-isotope mixture.",
            "source_sha256":{str(p.relative_to(ROOT)):file_hash(p) for p in INPUTS}}


@lru_cache(maxsize=None)
def states(variant: str = "pops") -> np.ndarray:
    """Return the fixed level set without adding spin, parity or electronic factors."""
    with np.load(SOURCE/"states.npz") as packet:
        levels = packet["states"].copy()
    if variant != "full":
        levels = levels[levels[:,0] <= load_candidate()["bound_cutoff_cm-1"]]
    if variant == "ground":
        labels = json.loads((SOURCE/"state_inventory.json").read_text())["electronic_labels"]
        levels = levels[levels[:,3] == labels.index("X2Pi")]
    if variant not in ("pops", "full", "ground"):
        raise ValueError("Unknown state-sum variant")
    levels.setflags(write=False)
    return levels


def _pops_gthermal(t, variant, xp):
    levels = states(variant)
    energy = xp.asarray(levels[:,0]*C2)
    weights = xp.asarray(levels[:,1]/load_candidate()["nuclear_spin_divisor"])
    q = xp.sum(weights*xp.exp(-energy/t[...,None]),axis=-1)
    mass = load_candidate()["molecular_mass_u"]
    lntrans = 1.5*xp.log(2*math.pi*mass*AMU*KB*t/PLANCK**2)+xp.log(KB*t/100000.)
    return -R*t*(xp.log(q)+lntrans)


def _formation(t, variant, xp):
    model = load_candidate()
    a,b,c,d,e = model["original_logK_coefficients"]
    current = -(a/t+b*xp.log(t)+c+d*t+e*t*t)
    if variant == "current":
        return current
    old = NASA_R*_nasa(t,model["NASA_molecular_coefficients"],xp)[0]-model["NASA_molecular_H0_gauge_J_mol"]
    return current+(_pops_gthermal(t,variant,xp)-old)/(R*t)


def formation_h(temperature_K: float, variant: str = "pops") -> float:
    """Return Gformation/(RT) inside the reviewed 1500--4000 K interval."""
    if not math.isfinite(temperature_K) or not 1500 <= temperature_K <= 4000:
        raise ValueError("Temperature outside PS candidate domain")
    return float(_formation(np.asarray(float(temperature_K)),variant,np))


def formation_h_jax(temperature_K, variant: str = "pops"):
    """Support JIT/vmap/grad; invalid traced temperatures return NaN."""
    import jax.numpy as jnp
    t = jnp.asarray(temperature_K,dtype=jnp.asarray(load_candidate()["original_logK_coefficients"]).dtype)
    valid = jnp.isfinite(t)&(t>=1500)&(t<=4000)
    safe = jnp.where(valid,t,1500.)
    return jnp.where(valid,_formation(safe,variant,jnp),jnp.nan)


def direct_thermal(temperature_K: float, levels: np.ndarray) -> dict:
    """Use independent first/second moments for molecular H-H0, S and Cp."""
    t = float(temperature_K)
    energy = levels[:,0]*C2
    weight = levels[:,1]*np.exp(-energy/t)/load_candidate()["nuclear_spin_divisor"]
    q = float(np.sum(weight))
    mean = float(np.sum(weight*energy)/q)
    variance = float(np.sum(weight*(energy-mean)**2)/q)
    h = R*(2.5*t+mean)
    entropy = R*(math.log(q)+log_translation(t,load_candidate()["molecular_mass_u"])+2.5+mean/t)
    return {"G_J_mol":h-t*entropy,"H_J_mol":h,"S_J_mol_K":entropy,
            "Cp_J_mol_K":R*(2.5+variance/t**2),"Q":q}


def reaction_properties(temperature_K: float, variant: str = "pops") -> dict:
    """Derive reaction calorics from the current fit and molecular state moments."""
    g = R*temperature_K*formation_h(temperature_K,variant)
    t = float(temperature_K)
    a,b,c,d,e = load_candidate()["original_logK_coefficients"]
    h = R*(-a+b*t+d*t*t+2*e*t**3)
    entropy = R*(b*math.log(t)+b+c+2*d*t+3*e*t*t)
    cp = R*(b+2*d*t+6*e*t*t)
    if variant != "current":
        record = json.loads((SOURCE/"inputs.json").read_text())["nasa_records"]["PS"]
        old_h,old_s,old_cp = nasa_properties(record,t,NASA_R)
        new = direct_thermal(t,states(variant))
        h += new["H_J_mol"]-(old_h-load_candidate()["NASA_molecular_H0_gauge_J_mol"])
        entropy += new["S_J_mol_K"]-old_s
        cp += new["Cp_J_mol_K"]-old_cp
    return {"G_J_mol":float(g),"H_J_mol":float(h),"S_J_mol_K":float(entropy),"Cp_J_mol_K":float(cp)}


def validate() -> dict:
    """Check the dense runtime replay, independent moments and truncation sensitivity."""
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
        old = np.array([formation_h(t,"current") for t in ts])
        dense = np.array([formation_h(t) for t in ts])
        function = jax.jit(formation_h_jax)
        # Bound temporary arrays while verifying every one-kelvin temperature.
        actual = np.concatenate([np.asarray(function(chunk)) for chunk in np.array_split(ts,80)])
        native = json.loads((SOURCE/"inputs.json").read_text())["nasa_records"]["PS"]
        reconstruction = []
        for t,initial,value in zip(ts,current,dense):
            h,s,_ = nasa_properties(native,t,NASA_R)
            direct = direct_thermal(t,states())
            expected = initial+(direct["H_J_mol"]-t*direct["S_J_mol_K"]-(h-model["NASA_molecular_H0_gauge_J_mol"]-t*s))/(R*t)
            reconstruction.append(abs(value-expected))
        probes = np.array([1500.,1637.,2253.,3000.,3871.,4000.])
        derivatives = {}
        for variant in VARIANTS:
            f = lambda t:R*t*formation_h_jax(t,variant)
            g = np.asarray(jax.vmap(f)(probes))
            dg = np.asarray(jax.vmap(jax.grad(f))(probes))
            ddg = np.asarray(jax.vmap(jax.grad(jax.grad(f)))(probes))
            actual_properties = np.stack((g,g-probes*dg,-dg,-probes*ddg),axis=-1)
            expected = np.array([list(reaction_properties(t,variant).values()) for t in probes])
            derivatives[variant] = dict(zip(reaction_properties(1500).keys(),map(float,abs(actual_properties-expected).max(axis=0))))
        full = states("full")
        low = full[full[:,0] <= (model["original_source_D0_J_mol"]-model["D0_evaluation_margin_J_mol"])/(R*C2)]
        labels = json.loads((SOURCE/"state_inventory.json").read_text())["electronic_labels"]
        pf = dict(np.loadtxt(SOURCE/"partition.pf"))
        samples = []
        for t in (1500.,2000.,2500.,3000.,4000.):
            baseline = direct_thermal(t,states())
            masks = {"distributed_full":full,"D0_minus_margin_cutoff":low,
                     "ground_electronic_only":states("ground"),
                     "E_le_30000_cm-1":full[full[:,0]<=30000],
                     "J_le_240_5":states()[states()[:,2]<=240.5]}
            controls = {}
            for key,level_set in masks.items():
                value = direct_thermal(t,level_set)
                controls[key] = {"levels":len(level_set),"Q_over_primary":value["Q"]/baseline["Q"],
                    "delta_H_J_mol":value["H_J_mol"]-baseline["H_J_mol"],
                    "delta_S_J_mol_K":value["S_J_mol_K"]-baseline["S_J_mol_K"],
                    "delta_Cp_J_mol_K":value["Cp_J_mol_K"]-baseline["Cp_J_mol_K"]}
            w = states()[:,1]*np.exp(-states()[:,0]*C2/t)
            samples.append({"temperature_K":t,
                "K_over_current":{v:math.exp(formation_h(t,"current")-formation_h(t,v)) for v in VARIANTS},
                "reaction_properties":{v:reaction_properties(t,v) for v in VARIANTS},
                "molecular_thermal":baseline,"sensitivity":controls,
                "electronic_Q_fractions":{label:float(w[states()[:,3]==i].sum()/w.sum()) for i,label in enumerate(labels)},
                "full_state_sum_over_distributed_pf":direct_thermal(t,full)["Q"]*2/pf[t]})
        return {"temperature_count":len(ts),"range_K":[1500,4000],"step_K":1,
                "maximum_abs_current_h_replay_error":float(abs(current-old).max()),
                "maximum_abs_scalar_jax_h_error":float(abs(dense-actual).max()),
                "maximum_abs_independent_moment_h_error":float(max(reconstruction)),
                "jax_float64_derivative_errors":derivatives,
                "state_counts":{v:len(states(v)) for v in VARIANTS if v!="current"},
                "omitted_above_central_D0_levels":len(full)-len(states()),
                "D0_plus_margin_cutoff_keeps_all_distributed_levels":bool(full[:,0].max()<(438000+10000)/(R*C2)),
                "isotope_translation_K_factor":(model["isotope_mass_u"]/model["molecular_mass_u"])**1.5,
                "samples":samples,"production_change":"none",
                "completeness":"Finite sums and within-file sensitivity only; unlisted states and a4Pi ab initio errors have no quantified global bound."}
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
        states.cache_clear()
    report = validate()
    if args.build:
        (DATA/"validation.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        paths = set(INPUTS)|{Path(__file__)}
        paths.update(Path(module.__file__).resolve() for module in list(sys.modules.values())
                     if getattr(module,"__file__",None) and Path(module.__file__).resolve().is_relative_to(ROOT)
                     and Path(module.__file__).suffix==".py" and Path(module.__file__).name!="exogibbs_version.py")
        paths.update((ROOT/"src/exogibbs/data/curated").glob("*.json"))
        receipt = {"created_utc":datetime.now(timezone.utc).isoformat(),"python":platform.python_version(),
                   "numpy":np.__version__,"dtype":"float64","result":"passed","command":[sys.executable,*sys.argv],
                   "selected_file_sha256":{str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)},
                   "output_sha256":{p.name:file_hash(p) for p in (DATA/"candidate.json",DATA/"validation.json")}}
        (DATA/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(report,indent=2,allow_nan=False))
