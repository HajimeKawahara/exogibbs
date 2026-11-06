"""Compare two HCP formation-energy anchors while retaining current thermals."""

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
from hcp_source_review import R, curated_properties

SPECIES = "C1H1P1"
ANCHORS = ("cccbdb", "theory_2001")
DATA = ROOT / "tests/data/validation/hcp_candidate/20261106"
SOURCE = ROOT / "tests/data/validation/hcp_source_review/20261106"
FIT = ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat"


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    """Read both frozen independent anchors on the same retained reference."""
    return json.loads((DATA / "candidate.json").read_text())


def build() -> dict:
    """Separate physical source energies from the formal fit intercept."""
    config = json.loads((SOURCE / "inputs.json").read_text())
    facts = json.loads((SOURCE / "literature.json").read_text())
    paths = {name: SOURCE / path for name, path in config["tables"].items()}
    tables = {name: read_table(path) for name, path in paths.items()}
    offsets = {name: round(1000*(tb[298.15][4]+tb[0.][3])) for name,tb in tables.items()}
    d0 = sum(offsets[n] for n in ("C", "H", "P"))-offsets["HCP"]
    hf0 = round(1000*tables["HCP"][0.][4])
    phosphorus = tables["P"]
    elemental_heat = -1000*phosphorus[0.][3]-1000*(phosphorus[298.15][4]-phosphorus[0.][4])
    assert abs(elemental_heat-facts["elemental_reference_verification"]["P_H298_minus_H0_J_mol"]) < 1e-8
    lines = FIT.read_text().splitlines()
    i = next(i for i,line in enumerate(lines) if line.startswith(SPECIES+" "))
    coefficients = list(map(float,lines[i+1].split()))
    anchors = {}
    for name in ANCHORS:
        source = facts[name]
        delta = source["Hf0_J_mol"]-hf0
        anchors[name] = {"Hf0_J_mol":source["Hf0_J_mol"],"reported_uncertainty_J_mol":source["reported_uncertainty_J_mol"],
                         "delta_Hf0_J_mol":delta,"common_JANAF_D0_J_mol":d0-delta,
                         "source":source.get("doi",source["url"])}
    inputs = [FIT, SOURCE/"inputs.json", SOURCE/"literature.json", SOURCE/"sources.json", *paths.values()]
    return {"id":"hcp_retained_thermal_two_energy_anchors_v1","species":SPECIES,
            "reaction":"C(g) + H(g) + P(g) -> HCP(g)","composition":{"C":1,"H":1,"P":1},
            "temperature_range_K":[1500,4000],"standard_pressure_Pa":100000,"gas_constant_J_mol_K":R,
            "base_dataset":"curated-tio2-v1","base_Hf0_J_mol":hf0,"base_D0_J_mol":d0,
            "janaf_fixed_reference_offsets_J_mol":offsets,"anchors":anchors,
            "white_phosphorus_elemental_heat_J_mol":elemental_heat,
            "theory_own_atomic_TAE0_J_mol":sum(facts["theory_2001"]["atomic_Hf0_J_mol"].values())-facts["theory_2001"]["Hf0_J_mol"],
            "original_record":lines[i],"original_coefficients":coefficients,
            "formal_fit_intercept_Ra_J_mol":R*coefficients[0],
            "representation":"G_candidate = G_current + delta(Hf0); retain all thermal coefficients, atomic gauge and fit/source residual. No evaluation of the empirical fit at zero K.",
            "energy_scope":"Both Hf0 alternatives use the selected JANAF elemental reference (white P). Physical D0 is a balanced atomization energy, not R*a, H298 or a single H-CP bond energy. The 2001 own-atom TAE0=1018 kJ/mol is a separate rounded-gauge control, not the adopted common-reference D0=1017.774 kJ/mol.",
            "uncertainty_scope":"Reported +/-20 and +/-8 kJ/mol are separate source sensitivities, not independent Gaussian errors or a complete uncertainty/covariance. The 2001 body explicitly quotes +/-8 for HCP, distinct from its abstract's broader +/-10 wording. CCCBDB is a CP-based JANAF adjustment, not an independent HCP measurement.",
            "thermal_scope":"Retained historical thermals only. The 2024 PES/DMS supplement does not supply states or Q; no 2024 thermal model or 4000 K validity claim is inferred.",
            "source_sha256":{str(p.relative_to(ROOT)):file_hash(p) for p in inputs}}


def energy_shift(anchor: str | None = "cccbdb", offset_J_mol: float = 0.) -> float:
    """Return the energy-only change, optionally at a reported uncertainty edge."""
    if anchor is None:
        if offset_J_mol != 0:
            raise ValueError("Current reference cannot have an uncertainty offset")
        return 0.
    if anchor not in ANCHORS or not math.isfinite(offset_J_mol):
        raise ValueError("Invalid HCP anchor or energy offset")
    return load_candidate()["anchors"][anchor]["delta_Hf0_J_mol"]+offset_J_mol


def _formation(t, anchor, offset, xp):
    a,b,c,d,e = load_candidate()["original_coefficients"]
    g = -a-b*t*xp.log(t)-c*t-d*t*t-e*t**3+energy_shift(anchor,offset)/R
    slope = -b*(xp.log(t)+1)-c-2*d*t-3*e*t*t
    curvature = -b/t-2*d-6*e*t
    return g,slope,curvature


def formation_h(temperature_K: float, anchor: str | None = "cccbdb", offset_J_mol: float = 0.) -> float:
    """Return continuous Gformation/(RT) on the bounded candidate domain."""
    if not math.isfinite(temperature_K) or not 1500 <= temperature_K <= 4000:
        raise ValueError("Temperature outside HCP candidate domain")
    return float(_formation(float(temperature_K),anchor,offset_J_mol,np)[0]/temperature_K)


def formation_h_jax(temperature_K, anchor: str | None = "cccbdb", offset_J_mol: float = 0.):
    """Support scalar, array, JIT and vmap evaluation; mask invalid temperatures."""
    import jax.numpy as jnp
    t = jnp.asarray(temperature_K,dtype=jnp.asarray(load_candidate()["original_coefficients"]).dtype)
    valid = jnp.isfinite(t)&(t>=1500)&(t<=4000)
    safe = jnp.where(valid,t,1500.)
    return jnp.where(valid,_formation(safe,anchor,offset_J_mol,jnp)[0]/safe,jnp.nan)


def reaction_properties(temperature_K: float, anchor: str | None = "cccbdb", offset_J_mol: float = 0.) -> dict:
    """Differentiate one potential for consistent formation G/H/S/Cp."""
    formation_h(temperature_K,anchor,offset_J_mol)
    t = float(temperature_K)
    g,slope,curvature = _formation(t,anchor,offset_J_mol,np)
    return dict(zip(("G_J_mol","H_J_mol","S_J_mol_K","Cp_J_mol_K"),map(float,(R*g,R*(g-t*slope),-R*slope,-R*t*curvature))))


def validate() -> dict:
    """Audit both continuous candidates against the actual named production base."""
    import jax
    from exogibbs.presets import curated_tio2
    assert build() == load_candidate()
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64",True)
    try:
        setup = curated_tio2.chemsetup(ions=True)
        assert setup.metadata["dataset"] == "curated-tio2-v1"
        ts = np.arange(1500.,4001.)
        current = np.asarray(setup.hvector_func(ts))[:,setup.species.index(SPECIES)]
        before = np.array([formation_h(t,None) for t in ts])
        oldprops = np.array([list(reaction_properties(t,None).values()) for t in ts])
        result = {"temperature_count":len(ts),"range_K":[1500,4000],"step_K":1,
                  "maximum_abs_current_h_replay_error":float(abs(before-current).max()),"anchors":{},"production_change":"none"}
        source = json.loads((SOURCE/"comparison.json").read_text())["source_samples"]
        for anchor in ANCHORS:
            after = np.array([formation_h(t,anchor) for t in ts])
            independent = -np.array([curated_properties(t)["ln_K"] for t in ts])+energy_shift(anchor)/(R*ts)
            props = np.array([list(reaction_properties(t,anchor).values()) for t in ts])
            expected = np.zeros_like(props);expected[:,:2] = energy_shift(anchor)
            fun = jax.jit(lambda t:formation_h_jax(t,anchor))
            g = lambda t:R*t*fun(t)
            slope = np.asarray(jax.jit(jax.vmap(jax.grad(g)))(ts))
            curvature = np.asarray(jax.jit(jax.vmap(jax.grad(jax.grad(g))))(ts))
            jg = R*ts*np.asarray(fun(ts))
            jprops = np.stack((jg,jg-ts*slope,-slope,-ts*curvature),axis=-1)
            residuals = []
            for row in source:
                t = row["temperature_K"];ln_k = row["source"]["reaction"]["janaf"]["ln_K"]
                old = -formation_h(t,None)-ln_k
                new = -formation_h(t,anchor)-(ln_k-energy_shift(anchor)/(R*t))
                residuals.append({"temperature_K":t,"current_minus_JANAF_ln_K":old,
                                  "candidate_minus_shifted_JANAF_ln_K":new,"preservation_error_ln_K":new-old})
            result["anchors"][anchor] = {"maximum_abs_independent_h_error":float(abs(after-independent).max()),
                "maximum_abs_energy_shift_h_error":float(abs(after-current-energy_shift(anchor)/(R*ts)).max()),
                "maximum_abs_energy_only_property_errors":dict(zip(reaction_properties(1500),map(float,abs(props-oldprops-expected).max(axis=0)))),
                "maximum_abs_jax_property_errors":dict(zip(reaction_properties(1500),map(float,abs(jprops-props).max(axis=0)))),
                "source_fit_residuals":residuals,"samples":[{"temperature_K":t,"candidate_over_current_K":math.exp(-energy_shift(anchor)/(R*t)),"reaction_properties":reaction_properties(t,anchor)} for t in (1500,2000,2500,3000,4000)]}
            assert result["anchors"][anchor]["maximum_abs_energy_shift_h_error"] < 1e-12
        assert result["maximum_abs_current_h_replay_error"] < 1e-12
        return result
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
    result = validate()
    if args.build:
        import jax
        (DATA/"validation.json").write_text(json.dumps(result,indent=2)+"\n")
        paths = {Path(__file__).resolve(),SOURCE/"comparison.json",ROOT/"mceg/reviews/hcp_source_review.py",ROOT/"mceg/reviews/c2h_source_review.py",ROOT/"mceg/reviews/gas_source_comparison.py"}
        paths.update(ROOT/p for p in load_candidate()["source_sha256"])
        paths.update((ROOT/"src/exogibbs/presets").glob("curated_*.py"))
        paths.update((ROOT/"src/exogibbs/data/curated").glob("*.json"))
        receipt = {"created_utc":datetime.now(timezone.utc).isoformat(),"command":[sys.executable,*sys.argv],
                   "python":platform.python_version(),"jax":jax.__version__,"numpy":np.__version__,"dtype":"float64","backend":"cpu","result":"passed",
                   "selected_file_sha256":{str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)},
                   "output_sha256":{p:file_hash(DATA/p) for p in ("candidate.json","validation.json")}}
        (DATA/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(result,indent=2))
