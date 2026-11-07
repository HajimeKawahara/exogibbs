"""Replay N2O neutral-gas sources against the final curated-tio2-v1 archive."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import ROOT

DATA = ROOT / "tests/data/validation/n2o_source_review/20261106"
GRID = ROOT / "tests/data/validation/curated_tio2_adoption/20261106/ions.npz"
R, NASA_R = 8.31446261815324, 8.314510
NAME = "N2O"


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


def partition_diagnostic(t: float, nasa: dict) -> dict:
    """Compare exact TIPS nodes after one nuclear-spin removal; never extrapolate."""
    cfg = json.loads((DATA/"tips2025_nodes.json").read_text())
    raw = cfg["raw_Q_nodes"].get(str(float(t)))
    if raw is None:
        return {"temperature_K":t,"available":False,"source_bounds_K":cfg["bounds_K"]}
    h,s,_ = nasa_properties(nasa,t,NASA_R)
    h0 = nasa["hf298_J_mol"]-nasa["h298_minus_h0_J_mol"]
    lt = (1.5*math.log(2*math.pi*nasa["mass_u"]*1.66053906660e-27*1.380649e-23*t/6.62607015e-34**2)
          +math.log(1.380649e-23*t/100000))
    qn = math.exp(-(h-h0)/(NASA_R*t)+s/NASA_R-lt)
    q = raw/cfg["nuclear_spin_divisor"]
    return {"temperature_K":t,"available":True,"raw_Q":raw,"spin_free_Q":q,
            "NASA_Q":qn,"same_energy_atoms_natural_mass_K_ratio":q/qn,
            "delta_thermal_G_J_mol":-R*t*math.log(q/qn),
            "scope":"Principal-isotope internal states with natural-mass translation proxy; sparse nodes do not supply H/S/Cp or a continuous replacement."}


def review() -> dict:
    """Separate native source energy, molecular thermal and atomic thermal terms."""
    config = inputs()
    comp = config["composition"]
    nasa = json.loads((DATA/"nasa_records.json").read_text())
    tables = {n:read_table(DATA/p) for n,p in config["tables"].items()}
    nh0 = {n:r["hf298_J_mol"]-r["h298_minus_h0_J_mol"] for n,r in nasa.items()}
    jh0 = {n:1000*(tb[298.15][4]+tb[0.0][3]) for n,tb in tables.items()}
    nd0 = sum(c*nh0[n] for n,c in comp.items())-nh0[NAME]
    jd0 = sum(c*jh0[n] for n,c in comp.items())-jh0[NAME]
    atct = config["atct"]
    ad0 = 1000*(sum(c*atct[n]["hf0_kJ_mol"] for n,c in comp.items())-atct[NAME]["hf0_kJ_mol"])
    rows = []
    for t in config["temperatures_K"]:
        jp = {n:(1000*(tb[298.15][4]+tb[t][3]),tb[t][1],tb[t][0]) for n,tb in tables.items()}
        np_ = {n:nasa_properties(r,t,NASA_R) for n,r in nasa.items()}
        phi_j = {n:(h-jh0[n])/(R*t)-s/R for n,(h,s,cp) in jp.items()}
        phi_n = {n:(h-nh0[n])/(NASA_R*t)-s/NASA_R for n,(h,s,cp) in np_.items()}
        terms = {"energy":(jd0/R-nd0/NASA_R)/t,
                 "molecular_thermal":phi_n[NAME]-phi_j[NAME],
                 "atomic_thermal":sum(c*(phi_j[n]-phi_n[n]) for n,c in comp.items())}
        reactions = {}
        for label,props,gas_r in [("janaf",jp,R),("nasa",np_,NASA_R)]:
            h,s,cp = [props[NAME][i]-sum(c*props[n][i] for n,c in comp.items()) for i in range(3)]
            reactions[label] = {"ln_K":-(h/t-s)/gas_r,"G_J_mol":h-t*s,
                "H_J_mol":h,"S_J_mol_K":s,"Cp_J_mol_K":cp}
        lj,ln = reactions['janaf']['ln_K'],reactions['nasa']['ln_K']
        current = curated_properties(t)
        rows.append({"temperature_K":t,"native_reactions":reactions,"curated":current,
            "molecular":{label:{"G_J_mol":p[0]-t*p[1],"H_J_mol":p[0],"S_J_mol_K":p[1],"Cp_J_mol_K":p[2]}
                         for label,p in [('janaf',jp[NAME]),('nasa',np_[NAME])]},
            "janaf_over_nasa":math.exp(lj-ln),"curated_over_nasa":math.exp(current['ln_K']-ln),
            "curated_over_janaf":math.exp(current['ln_K']-lj),
            "janaf_minus_nasa_ln_K_components":terms,
            "decomposition_closure_ln_K":lj-ln-sum(terms.values()),
            "energy_aligned_janaf_over_nasa":math.exp(terms['molecular_thermal']+terms['atomic_thermal']),
            "ATcT_energy_only_over_current":math.exp((ad0-jd0)/(R*t)),
            "NASA_thermal_ATcT_over_current":math.exp(ln+(ad0/R-nd0/NASA_R)/t-current['ln_K']),
            "curated_minus_janaf_reaction":{k:current[k]-v for k,v in reactions['janaf'].items() if k!='ln_K'}})
    return {"molecule":NAME,"species_id":config['species_id'],"composition":comp,
        "standard_pressure_Pa":100000,"gas_constant_J_mol_K":R,"native_NASA_R_J_mol_K":NASA_R,
        "balanced_D0_J_mol":{"janaf":jd0,"nasa":nd0,"atct_1.228":ad0},
        "ATcT_minus_JANAF_D0_J_mol":ad0-jd0,"source_samples":rows,"footprint":footprint(),
        "energy_only_scope":"Diagnostic physical ATcT-minus-JANAF D0 shift added to literal current fit; thermal and fitting residual retained, no inference of D0 from fit intercept. For a non-JANAF inherited source this is only a JANAF-proxy sensitivity, not its physical energy update.",
        "TIPS2025_diagnostics":[partition_diagnostic(t,nasa[NAME]) for t in [1000,1500,2000,3000,4000]],
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
