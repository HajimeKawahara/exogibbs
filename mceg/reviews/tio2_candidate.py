"""Evaluate a retained-energy TiO2 molecular thermal substitution offline."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

import tio2_source_review as source
from gas_source_comparison import BASELINE, file_hash
from tio_candidate import _nasa

ROOT = source.ROOT
DATA = ROOT / "tests/data/validation/tio2_candidate/20261106"
NAME, SPECIES, ELEMENT = "TiO2", "O2Ti1", "Ti"
PRIMARY, CONTROL = "rrho", "h298"
R, NASA_R = source.R, source.NASA_R
FASTCHEM = ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat"
INPUTS = (source.DATA/"inputs.json", source.DATA/"nasa-record.txt",
          source.DATA/"O-046.txt", source.DATA/"Ti-006.txt", source.OXYGEN_TABLE,
          BASELINE/"inputs.json", FASTCHEM)
VARIANTS = ("current", PRIMARY, CONTROL)


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    """Load the frozen physical conventions and continuous coefficients."""
    return json.loads((DATA/"candidate.json").read_text())


def build() -> dict:
    """Define a molecular thermal correction while preserving the inherited fit."""
    inputs = json.loads(INPUTS[0].read_text())
    record = json.loads((BASELINE/"inputs.json").read_text())["nasa9_records"][NAME]
    lines = INPUTS[1].read_text().splitlines()
    hf298 = float(lines[1].split()[-1])
    heat_content = float(lines[2].split()[-1])
    nasa_h0 = (hf298-heat_content)/NASA_R
    coefficients = np.array(record["intervals"][1]["coefficients"])
    assert record["intervals"][1]["bounds_K"] == [1000.,6000.]
    coefficients[7] -= nasa_h0
    old_ref = source.nasa_properties(record,298.15,R)[0]-R*nasa_h0
    new_ref = source.rrho(298.15,inputs)["H_minus_H0_J_mol"]
    tables = {NAME:source.read_table(INPUTS[2]), "Ti":source.read_table(INPUTS[3]),
              "O":source.read_table(INPUTS[4])}
    d0 = -1000*math.fsum(n*tables[name][0][4] for name,n in source.STOICHIOMETRY.items())
    lines = FASTCHEM.read_text().splitlines()
    i = next(i for i,line in enumerate(lines) if line.startswith(SPECIES+" "))
    original = list(map(float,lines[i+1].split()))
    assert len(original) == 5
    return {"id":"tio2_retained_energy_rrho_candidate_v1", "molecule":NAME, "species":SPECIES,
            "baseline_dataset":"curated-cos-v1", "standard_pressure_Pa":100000,
            "temperature_range_K":[1500,4000], "primary":PRIMARY, "control":CONTROL,
            "composition":{"Ti":1,"O":2}, "gas_constant_J_mol_K":R,
            "nasa_native_R_J_mol_K":NASA_R, "original_source_D0_J_mol":d0,
            "delta_D0_J_mol":0.0, "nasa_molecular_hf298_native_J_mol":hf298,
            "nasa_molecular_heat_content_native_J_mol":heat_content,
            "nasa_molecular_h0_div_R_K":nasa_h0,
            "nasa_molecular_thermal_coefficients":coefficients.tolist(),
            "original_logK_coefficients":original,
            "rotational_constants_MHz":inputs["brunken2008"]["rotational_constants_MHz"],
            "frequencies_cm-1":inputs["gas_fundamentals"]["frequencies_cm-1"],
            "mass_u":inputs["diagnostic_mass_u"], "symmetry_number":2,
            "electronic_degeneracy":1, "nuclear_spin_weight":1,
            "new_minus_reference_heat_content_298_J_mol":new_ref-old_ref,
            "anchor_scope":"Retain source D0 and the complete current-fit residual to the NASA reference. Only molecular G-H0 changes; no unique 0 K limit is inferred for the finite-range fit. H298 control subtracts the 298 K heat-content difference and therefore changes the source zero-energy offset.",
            "thermal_scope":"Single-ground-state rigid rotor/harmonic oscillator using measured 48Ti16O2 rotation and gas fundamentals as harmonic frequencies, JANAF mean mass, sigma=2 and spin weight=1. NASA is the continuous old-thermal proxy, not an exact JANAF interpolation.",
            "physical_scope":"No anharmonicity, centrifugal distortion, complete excited-state manifold, isotope mixture or independent energy reevaluation. ZPE is already included in D0 and is not added again.",
            "source_sha256":{str(p.relative_to(ROOT)):file_hash(p) for p in INPUTS}}


def _rrho(t, xp, frequencies=None):
    model = load_candidate()
    k, h = 1.380649e-23, 6.62607015e-34
    mass = model["mass_u"]*1.66053906660e-27
    theta = np.array(model["rotational_constants_MHz"])*1e6*h/k
    log_rot = .5*xp.log(math.pi*t**3/float(np.prod(theta)))-math.log(model["symmetry_number"])
    log_trans = 1.5*xp.log(2*math.pi*mass*k*t/h**2)+xp.log(k*t/100000)
    enthalpy, entropy, cp = 4*t, log_trans+log_rot+4, 4+xp.zeros_like(t)
    for frequency in model["frequencies_cm-1"] if frequencies is None else frequencies:
        x = h*29979245800*frequency/(k*t)
        occupation = 1/xp.expm1(x)
        enthalpy = enthalpy+t*x*occupation
        entropy = entropy+x*occupation-xp.log(-xp.expm1(-x))
        cp = cp+x*x*xp.exp(-x)/xp.expm1(-x)**2
    return enthalpy-t*entropy, -entropy, -cp/t


def _formation(t, variant, xp):
    model = load_candidate()
    a,b,c,d,e = model["original_logK_coefficients"]
    current = -(a/t+b*xp.log(t)+c+d*t+e*t*t)
    if variant == "current":
        return current
    if variant not in (PRIMARY,CONTROL):
        raise ValueError("Unknown candidate variant")
    delta = _rrho(t,xp)[0]-_nasa(t,model["nasa_molecular_thermal_coefficients"],xp)[0]
    if variant == CONTROL:
        delta -= model["new_minus_reference_heat_content_298_J_mol"]/R
    return current+delta/t


def formation_h(temperature_K: float, variant: str = PRIMARY) -> float:
    """Return Gformation/(RT) on the declared continuous 1500--4000 K interval."""
    if not math.isfinite(temperature_K) or not 1500 <= temperature_K <= 4000:
        raise ValueError("Temperature outside TiO2 candidate domain")
    return float(_formation(float(temperature_K),variant,np))


def formation_h_jax(temperature_K, variant: str = PRIMARY):
    """Support JIT/derivatives and return NaN for invalid traced inputs."""
    import jax.numpy as jnp
    t = jnp.asarray(temperature_K,dtype=jnp.asarray(load_candidate()["original_logK_coefficients"]).dtype)
    valid = jnp.isfinite(t)&(t>=1500)&(t<=4000)
    safe = jnp.where(valid,t,1500.)
    return jnp.where(valid,_formation(safe,variant,jnp),jnp.nan)


def reaction_properties(temperature_K: float, variant: str = PRIMARY) -> dict:
    """Return analytic reaction G/H/S/Cp from the single candidate Gibbs function."""
    t = float(temperature_K)
    g = R*t*formation_h(t,variant)
    a,b,c,d,e = load_candidate()["original_logK_coefficients"]
    h = R*(-a+b*t+d*t*t+2*e*t**3)
    entropy = R*(b*math.log(t)+b+c+2*d*t+3*e*t*t)
    cp = R*(b+2*d*t+6*e*t*t)
    if variant != "current":
        difference = np.array(_rrho(t,np))-np.array(_nasa(t,load_candidate()["nasa_molecular_thermal_coefficients"],np))
        h += R*(difference[0]-t*difference[1])
        entropy -= R*difference[1]
        cp -= R*t*difference[2]
        if variant == CONTROL:
            h -= load_candidate()["new_minus_reference_heat_content_298_J_mol"]
    return dict(G_J_mol=g,H_J_mol=float(h),S_J_mol_K=float(entropy),Cp_J_mol_K=float(cp))


def validate() -> dict:
    """Independently reconstruct thermal changes and check automatic derivatives."""
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
        inputs = json.loads(INPUTS[0].read_text())
        record = json.loads((BASELINE/"inputs.json").read_text())["nasa9_records"][NAME]
        rrho = [source.rrho(t,inputs) for t in ts]
        nasa = [source.nasa_properties(record,t,R) for t in ts]
        difference = np.array([new["H_minus_H0_J_mol"]-old[0]+R*model["nasa_molecular_h0_div_R_K"]
                               -t*(new["S_J_mol_K"]-old[1]) for t,new,old in zip(ts,rrho,nasa)])
        reconstruction = {"current_h":float(abs(current-values["current"]).max()),
                          "rrho_h":float(abs(current+difference/(R*ts)-values[PRIMARY]).max()),
                          "h298_h":float(abs(values[PRIMARY]-model["new_minus_reference_heat_content_298_J_mol"]/(R*ts)-values[CONTROL]).max())}
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
        sensitivities = []
        for t in (1500.,2000.,2500.,3000.,4000.):
            ground_g = _rrho(t,np)[0]
            # Controlled approximation: same rotational partition function for A
            # and X, measured A origin and harmonic frequencies. Not a bound on
            # unobserved triplets or all other excited rovibronic manifolds.
            a_frequencies = [876.,184.,316.]
            vib_ratio = math.exp(-( _rrho(t,np,a_frequencies)[0]-ground_g)/t)
            a_weight = math.exp(-1.4387768775039338*17593/t)*vib_ratio
            sensitivities.append({"temperature_K":t,
                "bend_316_over_322_K":math.exp(-(_rrho(t,np,[959.,316.,944.])[0]-ground_g)/t),
                "bend_328_over_322_K":math.exp(-(_rrho(t,np,[959.,328.,944.])[0]-ground_g)/t),
                "legacy_triplet_equal_rovib_over_ground_weight":3*math.exp(-1.4387768775039338*18880/t),
                "A_singlet_equal_rotation_weight":a_weight,
                "A_singlet_equal_rotation_K_multiplier":1+a_weight})
        return {"temperature_count":len(ts), "maximum_absolute_reconstruction_error":reconstruction,
                "jax_float64_derivative_errors":derivatives,
                "samples":[{"temperature_K":t,"K_over_current":{v:math.exp(formation_h(t,"current")-formation_h(t,v)) for v in VARIANTS},
                            "properties":{v:reaction_properties(t,v) for v in VARIANTS}} for t in (1500.,2000.,2500.,3000.,4000.)],
                "thermal_sensitivity":sensitivities,
                "sensitivity_scope":"Bend interval and selected electronic approximations only, not a physical uncertainty interval. A-state harmonic constants and origin come from Zhuang2010 primary abstract, DOI10.1039/C0CP00861C; equal rotation is assumed. Fundamentals used as ground-state harmonic frequencies remain an approximation.",
                "production_change":"none", "energy_replacement":False}
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
