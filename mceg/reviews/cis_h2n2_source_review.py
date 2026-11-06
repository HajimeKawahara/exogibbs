"""Replay cis-H2N2 neutral-gas sources against the final curated-tio2-v1 archive."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import ROOT

DATA = ROOT / "tests/data/validation/cis_h2n2_source_review/20261106"
GRID = ROOT / "tests/data/validation/curated_tio2_adoption/20261106/ions.npz"
R, NASA_R = 8.31446261815324, 8.314510
NAME = 'cis-H2N2'


@lru_cache(maxsize=1)
def inputs() -> dict:
    """Read the neutral identity and frozen source records."""
    return json.loads((DATA / "inputs.json").read_text())


def curated_properties(t: float) -> dict:
    """Differentiate the literal inherited fit; its intercept is not physical D0."""
    if not 1500 <= t <= 4000:
        raise ValueError("Temperature outside reviewed curated domain")
    a, b, c, d, e = inputs()["fastchem_record"]["coefficients"]
    ln_k = a/t+b*math.log(t)+c+d*t+e*t*t
    h = R*(-a+b*t+d*t*t+2*e*t**3)
    s = R*(b*math.log(t)+b+c+2*d*t+3*e*t*t)
    return {"ln_K": ln_k, "G_J_mol": -R*t*ln_k, "H_J_mol": h,
            "S_J_mol_K": s, "Cp_J_mol_K": R*(b+2*d*t+6*e*t*t)}


def footprint() -> dict:
    """Account for every participating elemental budget in 75 existing endpoints."""
    config = inputs()
    with np.load(GRID) as grid:
        col = grid["species"].tolist().index(config["species_id"])
        n = np.exp(grid["ln_n"])
        shares = {}
        for element, count in config["composition"].items():
            e = grid["elements"].tolist().index(element)
            if grid["formula_matrix"][e, col] != count:
                raise ValueError("Archived identity differs from selected neutral molecule")
            share = count*n[:, col]/grid["budget"][:, e]
            i = int(np.argmax(share))
            shares[element] = {"maximum_share": float(share[i]), "case_index": i,
                "mole_fraction": float(np.exp(grid["ln_x"][i, col])),
                "temperature_K": float(grid["temperature_K"][i]),
                "pressure_bar": float(grid["pressure_bar"][i]),
                "C_over_O": float(grid["C_over_O"][i]),
                "maximum_budget_residual": float(np.max(abs(n@grid["formula_matrix"][e]/grid["budget"][:, e]-1)))}
        return {"model": "curated-tio2-v1", "cases": len(n), "species_count": n.shape[1],
            "all_converged": bool(np.all(grid["converged"])), "element_shares": shares,
            "maximum_mole_fraction": float(np.max(np.exp(grid["ln_x"][:, col]))),
            "maximum_abs_h_error": max(abs(h+curated_properties(t)["ln_K"])
                for t,h in zip(grid["temperature_K"],grid["h"][:,col])),
            "scope": "Finite hot H-rich ideal-gas equilibrium inventory; not a replacement response, observation or oxidizing/photochemical atmosphere validation."}


def review() -> dict:
    """Compare the exact isomer; keep the NASA mixture outside this comparison."""
    config = inputs()
    comp = config["composition"]
    nasa = json.loads((DATA/"nasa_records.json").read_text())
    tables = {n:read_table(DATA/p) for n,p in config["tables"].items()}
    jh0 = {n:1000*(tb[298.15][4]+tb[0.0][3]) for n,tb in tables.items()}
    jd0 = sum(c*jh0[n] for n,c in comp.items())-jh0[NAME]
    atct = config["atct"]
    ad0 = 1000*(sum(c*atct[n]["hf0_kJ_mol"] for n,c in comp.items())-atct[NAME]["hf0_kJ_mol"])
    rows = []
    for t in config["temperatures_K"]:
        jp = {n:(1000*(tb[298.15][4]+tb[t][3]),tb[t][1],tb[t][0]) for n,tb in tables.items()}
        h,s,cp = [jp[NAME][i]-sum(c*jp[n][i] for n,c in comp.items()) for i in range(3)]
        lj = -(h/t-s)/R
        reaction = {"ln_K":lj,"G_J_mol":h-t*s,"H_J_mol":h,"S_J_mol_K":s,"Cp_J_mol_K":cp}
        mh,ms,mcp = jp[NAME]
        # This standalone NASA mixture is retained to make the rejected mapping auditable.
        nh,ns,ncp = nasa_properties(nasa[config["nasa_record_key"]],t,NASA_R)
        current = curated_properties(t)
        rows.append({"temperature_K":t,"native_reactions":{"janaf":reaction},"curated":current,
            "molecular":{"janaf":{"G_J_mol":mh-t*ms,"H_J_mol":mh,"S_J_mol_K":ms,"Cp_J_mol_K":mcp}},
            "excluded_nasa_mixture":{"G_J_mol":nh-t*ns,"H_J_mol":nh,"S_J_mol_K":ns,"Cp_J_mol_K":ncp,
                "source_record":config["nasa_record_key"],"scope":"Different equilibrium mixture; not a same-isomer comparator or replacement."},
            "janaf_over_nasa":None,"curated_over_nasa":None,"energy_aligned_janaf_over_nasa":None,
            "curated_over_janaf":math.exp(current["ln_K"]-lj),
            "ATcT_energy_only_over_current":math.exp((ad0-jd0)/(R*t)),
            "NASA_thermal_ATcT_over_current":None,
            "curated_minus_janaf_reaction":{k:current[k]-v for k,v in reaction.items() if k!="ln_K"}})
    return {"molecule":NAME,"species_id":config["species_id"],"composition":comp,
        "standard_pressure_Pa":100000,"gas_constant_J_mol_K":R,"native_NASA_R_J_mol_K":NASA_R,
        "balanced_D0_J_mol":{"janaf":jd0,"nasa":None,"atct_1.228":ad0},
        "ATcT_minus_JANAF_D0_J_mol":ad0-jd0,"source_samples":rows,"footprint":footprint(),
        "identity_resolution":"JANAF and current fit identify a single neutral isomer. The NASA record is a different mixture and is excluded from native and aligned comparisons.",
        "energy_only_scope":"Exact-isomer ATcT-minus-JANAF physical atomization energy added to the current fit, preserving thermal function and fitting residual; not inferred from the fitted intercept.",
        "physical_accuracy":"unknown","production_change":"none","equilibrium_substitution":"not_run"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    text = json.dumps(review(),indent=2,allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text,end="")
