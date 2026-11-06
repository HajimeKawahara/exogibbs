"""Replay S7 source comparisons and its archived final curated footprint."""

from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import ROOT

DATA = ROOT / "tests/data/validation/s7_source_review/20261106"
GRID = ROOT / "tests/data/validation/curated_tio2_adoption/20261106/ions.npz"
R, NASA_R = 8.31446261815324, 8.314510
NAME, SPECIES = "S7", "S7"


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
    """Separate native source energy from molecular and atomic thermal effects."""
    config, t = inputs(), temperature
    composition, nasa = config["composition"], config["nasa9"]
    tables = {n: read_table(DATA / path) for n, path in config["tables"].items()}
    # The molecular 0 K JANAF row is blank: use an explicit H298 anchor.
    nh0 = {n: r["hf298_J_mol"] - r["h298_minus_h0_J_mol"] for n, r in nasa.items()}
    jh0 = {n: 1000*tb[298.15][4] for n, tb in tables.items()}
    nhref = {n: r["hf298_J_mol"] for n, r in nasa.items()}
    nd0 = sum(c*nh0[n] for n, c in composition.items()) - nh0[NAME]
    nanchor = sum(c*nhref[n] for n, c in composition.items()) - nhref[NAME]
    janchor = sum(c*jh0[n] for n, c in composition.items()) - jh0[NAME]
    jp = {n: (1000*(tb[298.15][4] + tb[t][3]), tb[t][1], tb[t][0]) for n, tb in tables.items()}
    np_ = {n: nasa_properties(r, t, NASA_R) for n, r in nasa.items()}
    phi_j = {n: (h-jh0[n])/(R*t) - s/R for n, (h, s, cp) in jp.items()}
    phi_n = {n: (h-nhref[n])/(NASA_R*t) - s/NASA_R for n, (h, s, cp) in np_.items()}
    terms = {"energy": (janchor/R - nanchor/NASA_R)/t,
             "molecular_thermal": phi_n[NAME] - phi_j[NAME],
             "atomic_thermal": sum(c*(phi_j[n] - phi_n[n]) for n, c in composition.items())}
    reaction = {}
    for label, props, gas_r in (("janaf", jp, R), ("nasa", np_, NASA_R)):
        p = {key: props[NAME][i] - sum(c*props[n][i] for n, c in composition.items())
             for i, key in enumerate(("H_J_mol", "S_J_mol_K", "Cp_J_mol_K"))}
        p["G_J_mol"] = p["H_J_mol"] - t*p["S_J_mol_K"]
        p["ln_K"] = -p["G_J_mol"]/(gas_r*t)
        reaction[label] = p
    jln, nln = reaction["janaf"]["ln_K"], reaction["nasa"]["ln_K"]
    raw = -1000*(tables[NAME][t][5] - sum(c*tables[n][t][5] for n, c in composition.items()))/(R*t)
    molecular = {f"{label}_{key}": value for label, p in (("janaf", jp[NAME]), ("nasa", np_[NAME]))
                 for key, value in zip(("H_J_mol", "S_J_mol_K", "Cp_J_mol_K"), p)}
    molecular.update(janaf_G_J_mol=jp[NAME][0]-t*jp[NAME][1], nasa_G_J_mol=np_[NAME][0]-t*np_[NAME][1])
    return {"native_janaf_D0_J_mol": None, "energy_alignment": "H298",
            "janaf_atomization_H298_J_mol": janchor, "nasa_atomization_H298_J_mol": nanchor, "native_nasa_D0_J_mol": nd0,
            "janaf_over_nasa": math.exp(jln-nln),
            "H298_aligned_janaf_over_nasa": math.exp(terms["molecular_thermal"] + terms["atomic_thermal"]),
            "delta_ln_K_components": terms, "closure_ln_K": jln-nln-sum(terms.values()),
            "janaf_raw_formation_minus_HS_ln_K": raw-jln,
            "molecular": molecular, "reaction": reaction,
            "nasa_molecular_H_minus_H0_J_mol": np_[NAME][0]-nh0[NAME],
            "nasa_molecular_H298_thermal_G_over_RT": phi_n[NAME]}


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
    old = facts["janaf_printed"]
    gap = facts["jackson2016"]["illustrative_gap_eV"]
    return {"printed_298_free_function_minus_S_J_mol_K":
                old["printed_free_energy_function_298_J_mol_K"]-old["S298_J_mol_K"],
            "H298_uncertainty_only_K_factor": math.exp(1000*old["Hf298_uncertainty_kJ_mol"]/(R*t)),
            "S298_uncertainty_only_K_factor": math.exp(old["S298_uncertainty_J_mol_K"]/R),
            "equal_thermal_isomer_Boltzmann_weight_not_population":
                math.exp(-gap*96485.33212331002/(R*t)) if gap is not None else None}



def review() -> dict:
    """Replay the bounded source audit without changing a production coefficient."""
    rows = []
    for t in inputs()["temperatures_K"]:
        source = source_comparison(t)
        current = curated_properties(t)
        rows.append({"temperature_K": t, "source": source, "curated": current,
            "curated_over_nasa": math.exp(current["ln_K"]-source["reaction"]["nasa"]["ln_K"]),
            "curated_over_janaf": math.exp(current["ln_K"]-source["reaction"]["janaf"]["ln_K"]),
            "curated_minus_janaf_reaction": {k: current[k]-v for k, v in source["reaction"]["janaf"].items()},
            "modern": modern_diagnostic(t, source, current)})
    return {"molecule": NAME, "species_id": SPECIES, "composition": inputs()["composition"],
            "standard_pressure_Pa": 100000, "JANAF_R_J_mol_K": R, "NASA_R_J_mol_K": NASA_R,
            "source_samples": rows, "footprint": footprint(), "decision": inputs()["decision"],
            "curated_change": "none", "new_equilibrium_substitution": "not_run", "physical_accuracy": "unknown"}


if __name__ == "__main__":
    print(json.dumps(review(), indent=2, allow_nan=False))
