"""Replay P4S3 source comparisons and its archived final curated footprint."""

from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import ROOT

DATA = ROOT / "tests/data/validation/p4s3_source_review/20261106"
GRID = ROOT / "tests/data/validation/curated_tio2_adoption/20261106/ions.npz"
R, NASA_R = 8.31446261815324, 8.314510
NAME, SPECIES = "P4S3", "P4S3"


@lru_cache(maxsize=1)
def inputs() -> dict:
    """Load the compact pinned neutral-gas source inputs."""
    return json.loads((DATA / "inputs.json").read_text())


def curated_properties(temperature: float) -> dict:
    """Evaluate the literal retained formation fit and analytic caloric functions."""
    if not 1500 <= temperature <= 4000:
        raise ValueError("Temperature outside reviewed curated domain")
    t = temperature
    lines = (ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat").read_text().splitlines()
    i = next(i for i, line in enumerate(lines) if line.startswith(SPECIES + " "))
    a, b, c, d, e = map(float, lines[i + 1].split())
    ln_k = a/t + b*math.log(t) + c + d*t + e*t*t
    enthalpy = R*(-a + b*t + d*t*t + 2*e*t**3)
    entropy = R*(b*math.log(t) + b + c + 2*d*t + 3*e*t*t)
    return {"ln_K": ln_k, "G_J_mol": -R*t*ln_k, "H_J_mol": enthalpy,
            "S_J_mol_K": entropy, "Cp_J_mol_K": R*(b + 2*d*t + 6*e*t*t)}


def source_comparison(temperature: float) -> dict:
    """Use actual neutral JANAF references without inventing a NASA molecule."""
    t, config = temperature, inputs()
    tables = {n: read_table(DATA / f) for n,f in config["tables"].items()}
    h0 = {n: 1000*tb[298.15][4] for n,tb in tables.items()}
    props = {n: {"H_J_mol":1000*(tb[298.15][4]+tb[t][3]),
                 "S_J_mol_K":tb[t][1],"Cp_J_mol_K":tb[t][0]} for n,tb in tables.items()}
    for p in props.values(): p["G_J_mol"] = p["H_J_mol"]-t*p["S_J_mol_K"]
    reaction = {k:props[NAME][k]-sum(c*props[n][k] for n,c in config["composition"].items()) for k in props[NAME]}
    reaction["ln_K"] = -reaction["G_J_mol"]/(R*t)
    d0 = sum(c*h0[n] for n,c in config["composition"].items())-h0[NAME]
    phi = {n:(p["H_J_mol"]-h0[n])/(R*t)-p["S_J_mol_K"]/R for n,p in props.items()}
    components = {"energy":d0/(R*t),"molecular_thermal":-phi[NAME],
                  "atomic_thermal":sum(c*phi[n] for n,c in config["composition"].items())}
    raw = -1000*(tables[NAME][t][5]-sum(c*tables[n][t][5] for n,c in config["composition"].items()))/(R*t)
    return {"native_janaf_D0_J_mol":None,"janaf_atomization_H298_J_mol":d0,
            "energy_alignment":"H298","native_nasa_D0_J_mol":None,
            "nasa_comparison_status":config["nasa_comparison_status"],
            "janaf_formation_ln_K_components":components,
            "closure_ln_K":reaction["ln_K"]-sum(components.values()),
            "janaf_raw_formation_minus_HS_ln_K":raw-reaction["ln_K"],
            "molecular":props[NAME],"atoms":{n:props[n] for n in config["composition"]},
            "reaction":{"janaf":reaction},"source_H298_J_mol":h0,
            "janaf_Hf0_J_mol":None,
            "janaf_Hf298_J_mol":1000*tables[NAME][298.15][4]}


def footprint() -> dict:
    """Re-read 75 existing curated-tio2-v1 endpoints; no new solve is performed."""
    config = inputs()
    with np.load(GRID) as g:
        j = g["species"].tolist().index(SPECIES)
        element_rows = {}
        for element, count in config["composition"].items():
            e = g["elements"].tolist().index(element)
            assert g["formula_matrix"][e, j] == count
            share = count*np.exp(g["ln_n"][:, j])/g["budget"][:, e]
            i = int(np.argmax(share))
            element_rows[element] = {"maximum_share": float(share[i]), "case_index": i,
                "temperature_K": float(g["temperature_K"][i]), "pressure_bar": float(g["pressure_bar"][i]),
                "C_over_O": float(g["C_over_O"][i]), "mole_fraction": float(np.exp(g["ln_x"][i, j]))}
        return {"dataset": "curated-tio2-v1", "cases": len(g["temperature_K"]), "species": len(g["species"]),
                "all_converged": bool(np.all(g["converged"])), "element_shares": element_rows,
                "maximum_mole_fraction": float(np.exp(g["ln_x"][:, j]).max()),
                "maximum_abs_h_error": max(abs(h + curated_properties(float(t))["ln_K"])
                    for t, h in zip(g["temperature_K"], g["h"][:, j])),
                "scope": "Archived hot H-rich ideal-gas equilibrium; not an abundance error or a cold/oxidizing/photochemical validation."}


def modern_diagnostic(t: float, source: dict, current: dict) -> dict:
    """Report the molecule-specific inspected numerical evidence separately."""
    facts = json.loads((DATA / "literature.json").read_text())
    old,new = facts["janaf_printed"],facts["kroeger_karton2019"]
    table = read_table(DATA / inputs()["tables"][NAME])
    alternatives = {}
    for key in ["preferred", "atomization_route"]:
        delta = new[key+"_Hf298_J_mol"]-old["Hf298_J_mol"]
        alternatives[key] = {"H298_shift_J_mol":delta,"energy_only_K_factor":math.exp(-delta/(R*t)),
            "delta_G_J_mol":delta,"delta_H_J_mol":delta,"delta_S_J_mol_K":0.,"delta_Cp_J_mol_K":0.,
            "common_JANAF_atomization_H298_J_mol":source["janaf_atomization_H298_J_mol"]-delta}
    reference_check = {}
    for atom in ["P","S"]:
        tb = read_table(DATA / inputs()["tables"][atom])
        reference_check[atom] = -1000*tb[0.][3]-1000*(tb[298.15][4]-tb[0.][4])
    return {"constant_Cp_H_increment_minus_table_J_mol":old["constant_Cp_J_mol_K"]*(t-298.15)-1000*table[t][3],
            "constant_Cp_S_minus_table_J_mol_K":old["S298_J_mol_K"]+old["constant_Cp_J_mol_K"]*math.log(t/298.15)-table[t][1],
            "constant_Cp_minus_nonlinear_RRHO_high_T_limit_J_mol_K":old["constant_Cp_J_mol_K"]-19*R,
            "elemental_reference_heat_contents_J_mol":reference_check,
            "energy_alternatives":alternatives,
            "cycle_3_to_9_spread_J_mol":max(new["cycles_3_to_9_Hf298_J_mol"])-min(new["cycles_3_to_9_Hf298_J_mol"])}


def review() -> dict:
    """Replay the bounded source audit without changing a production coefficient."""
    rows = []
    for t in inputs()["temperatures_K"]:
        source = source_comparison(t)
        current = curated_properties(t)
        rows.append({"temperature_K": t, "source": source, "curated": current,
            "curated_over_nasa": None,
            "curated_over_janaf": math.exp(current["ln_K"]-source["reaction"]["janaf"]["ln_K"]),
            "curated_minus_janaf_reaction": {k: current[k]-v for k, v in source["reaction"]["janaf"].items()},
            "modern": modern_diagnostic(t, source, current)})
    return {"molecule": NAME, "species_id": SPECIES, "composition": inputs()["composition"],
            "standard_pressure_Pa": 100000, "JANAF_R_J_mol_K": R, "NASA_R_J_mol_K": NASA_R,
            "source_samples": rows, "footprint": footprint(), "decision": inputs()["decision"],
            "curated_change": "none", "new_equilibrium_substitution": "not_run", "physical_accuracy": "unknown"}


if __name__ == "__main__":
    print(json.dumps(review(), indent=2, allow_nan=False))
