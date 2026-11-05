"""Evaluate a COS energy candidate on the unchanged curated JANAF thermal fit."""

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

from gas_source_comparison import ROOT, file_hash
from tio_candidate import _nasa, _polynomial
import cos_source_review as source


DATA = ROOT / "tests/data/validation/cos_candidate/20261105"
R = source.major.R
INPUTS = (ROOT / "src/exogibbs/data/curated/gas_v1.json", source.DATA / "inputs.json",
          source.major.REFERENCE_PATH)


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    """Load the pinned original thermal coefficients and balanced energy shift."""
    return json.loads((DATA / "candidate.json").read_text())


def build() -> dict:
    """Preserve the production representation and balance same-version ATcT atoms."""
    production = json.loads(INPUTS[0].read_text())
    inputs = json.loads(INPUTS[1].read_text())
    reference = json.loads(INPUTS[2].read_text())
    records = inputs["atct_1.228"]
    old = source.major.atomization_kelvin("COS", "janaf", reference) * R
    new = 1000 * (sum(n*records[e] for e,n in inputs["composition"].items()) - records["COS"])
    return {"id": "cos_janaf_thermal_atct_1_228_candidate_v1", "species": "C1O1S1",
            "reaction": "C(g)+O(g)+S(g)->OCS(g)", "temperature_range_K": [1500,4000],
            "standard_pressure_Pa": 100000, "gas_constant_J_mol_K": R,
            "base_dataset": "curated-oh-v1", "base_D0_J_mol": old, "D0_J_mol": new,
            "delta_D0_J_mol": new-old, "delta_D0_over_R_K": (new-old)/R,
            "atct_version": "1.228", "same_version_hf0_kJ_mol": records,
            "thermal_record": production["janaf_reactions"]["C1O1S1"],
            "representation": "Unchanged production JANAF piecewise quintic and NASA atomic background; subtract delta(D0/R)/T once; no refit",
            "uncertainty_scope": "Same-version evaluated central energies; no synthesized covariance or physical uncertainty band. Historical thermal and isotope limitations retained.",
            "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in INPUTS}}


def _formation(temperature, updated: bool, xp):
    model = load_candidate()
    record = model["thermal_record"]
    molecule = _polynomial(temperature, record, xp)
    atoms = _nasa(temperature, record["background_nasa9"], xp)
    shift = model["delta_D0_over_R_K"] if updated else 0.0
    return molecule[0]+atoms[0]-shift, molecule[1]+atoms[1], molecule[2]+atoms[2]


def formation_h(temperature_K: float, updated: bool = True) -> float:
    """Return -ln K for the one-bar neutral-atom reaction on 1500--4000 K."""
    if not math.isfinite(temperature_K) or not 1500 <= temperature_K <= 4000:
        raise ValueError("Temperature outside COS candidate domain")
    t = float(temperature_K)
    return float(_formation(t, updated, np)[0]/t)


def formation_h_jax(temperature_K, updated: bool = True):
    """Support JIT, vectorization and derivatives; invalid temperatures yield NaN."""
    import jax.numpy as jnp

    t = jnp.asarray(temperature_K, dtype=jnp.asarray(load_candidate()["thermal_record"]["coefficients"]).dtype)
    valid = jnp.isfinite(t) & (t >= 1500) & (t <= 4000)
    safe = jnp.where(valid, t, 1500.0)
    return jnp.where(valid, _formation(safe, updated, jnp)[0]/safe, jnp.nan)


def reaction_properties(temperature_K: float, updated: bool = True) -> dict:
    """Derive reaction G/H/S/Cp from one continuous Gibbs representation."""
    if not math.isfinite(temperature_K) or not 1500 <= temperature_K <= 4000:
        raise ValueError("Temperature outside COS candidate domain")
    t = float(temperature_K)
    g, first, second = _formation(t, updated, np)
    return dict(zip(("G_J_mol", "H_J_mol", "S_J_mol_K", "Cp_J_mol_K"),
                    map(float, (R*g, R*(g-t*first), -R*first, -R*t*second))))


def validate() -> dict:
    """Check dense production replay, exact energy-only calorics and knot matching."""
    import jax
    from exogibbs.presets import curated

    model = load_candidate()
    assert model == build()
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        base = curated.chemsetup(ions=True)
        assert base.metadata["dataset"] == model["base_dataset"]
        t = np.arange(1500.,4001.)
        column = base.species.index(model["species"])
        original = np.asarray(base.hvector_func(t))[:,column]
        before = np.array([formation_h(x, False) for x in t])
        after = np.array([formation_h(x) for x in t])
        scalar = np.array([list(reaction_properties(x).values()) for x in t])
        old = np.array([list(reaction_properties(x, False).values()) for x in t])
        expected = np.zeros_like(old); expected[:,:2] = -model["delta_D0_J_mol"]
        f = jax.jit(formation_h_jax)
        g = lambda temp: R*temp*f(temp)
        first = np.asarray(jax.jit(jax.vmap(jax.grad(g)))(t))
        second = np.asarray(jax.jit(jax.vmap(jax.grad(jax.grad(g))))(t))
        jg = np.asarray(jax.jit(jax.vmap(g))(t))
        jprops = np.stack([jg,jg-t*first,-first,-t*second],axis=-1)
        jumps = []
        record = model["thermal_record"]
        for i,knot in enumerate(record["knots_K"][1:-1]):
            left,right = np.array(record["coefficients"])[i:i+2]
            width1=knot-record["knots_K"][i];width2=record["knots_K"][i+2]-knot
            def ends(c,u,width):
                f=np.polynomial.polynomial.polyval(u,c)
                d=np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyder(c))/width
                dd=np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyder(c,2))/width**2
                return np.array([R*f,-R*d,-R*knot*dd])
            jump=ends(right,0,width2)-ends(left,1,width1)
            jumps.append({"temperature_K":knot,"G_jump_J_mol":float(jump[0]),"S_jump_J_mol_K":float(jump[1]),"Cp_jump_J_mol_K":float(jump[2])})
        report={"temperature_count":len(t),"range_K":[1500,4000],"step_K":1,
                "maximum_abs_current_h_replay_error":float(abs(before-original).max()),
                "maximum_abs_energy_shift_h_error":float(abs(after-original+model["delta_D0_over_R_K"]/t).max()),
                "maximum_abs_energy_only_property_errors":dict(zip(reaction_properties(1500).keys(),map(float,abs(scalar-old-expected).max(axis=0)))),
                "maximum_abs_jax_property_errors":dict(zip(reaction_properties(1500).keys(),map(float,abs(jprops-scalar).max(axis=0)))),
                "internal_knot_jumps":jumps,
                "samples":[{"temperature_K":x,"candidate_over_current_K":math.exp(formation_h(x,False)-formation_h(x)),"reaction_properties":reaction_properties(x)} for x in (1500.,2000.,2500.,3000.,4000.)],
                "production_change":"none","scope":"Numerical consistency of an energy-only candidate, not an independent physical uncertainty assessment"}
        assert report["maximum_abs_current_h_replay_error"] < 1e-12
        assert report["maximum_abs_energy_shift_h_error"] < 1e-12
        return report
    finally:
        jax.config.update("jax_enable_x64", previous)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build",action="store_true")
    args=parser.parse_args()
    if args.build:
        DATA.mkdir(parents=True,exist_ok=True)
        (DATA/"candidate.json").write_text(json.dumps(build(),indent=2)+"\n")
        load_candidate.cache_clear()
    report=validate()
    if args.build:
        (DATA/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
