"""Replay K2C2N2 source comparisons and its archived final curated footprint."""

from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import ROOT

DATA = ROOT / "tests/data/validation/k2c2n2_source_review/20261106"
GRID = ROOT / "tests/data/validation/curated_tio2_adoption/20261106/ions.npz"
R, NASA_R = 8.31446261815324, 8.314510
NAME, SPECIES = "K2C2N2", "C2K2N2"


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
    nh0 = {n: r["hf298_J_mol"] - r["h298_minus_h0_J_mol"] for n, r in nasa.items()}
    jh0 = {n: 1000*(tb[298.15][4] + tb[0.0][3]) for n, tb in tables.items()}
    nd0 = sum(c*nh0[n] for n, c in composition.items()) - nh0[NAME]
    jd0 = sum(c*jh0[n] for n, c in composition.items()) - jh0[NAME]
    jp = {n: (1000*(tb[298.15][4] + tb[t][3]), tb[t][1], tb[t][0]) for n, tb in tables.items()}
    np_ = {n: nasa_properties(r, t, NASA_R) for n, r in nasa.items()}
    phi_j = {n: (h-jh0[n])/(R*t) - s/R for n, (h, s, cp) in jp.items()}
    phi_n = {n: (h-nh0[n])/(NASA_R*t) - s/NASA_R for n, (h, s, cp) in np_.items()}
    terms = {"energy": (jd0/R - nd0/NASA_R)/t,
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
    return {"native_janaf_D0_J_mol": jd0, "native_nasa_D0_J_mol": nd0,
            "janaf_over_nasa": math.exp(jln-nln),
            "D0_aligned_janaf_over_nasa": math.exp(terms["molecular_thermal"] + terms["atomic_thermal"]),
            "delta_ln_K_components": terms, "closure_ln_K": jln-nln-sum(terms.values()),
            "janaf_raw_formation_minus_HS_ln_K": raw-jln,
            "molecular": molecular, "reaction": reaction,
            "nasa_molecular_H_minus_H0_J_mol": np_[NAME][0]-nh0[NAME],
            "nasa_molecular_thermal_G_over_RT": phi_n[NAME]}


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


def old_rrho(t: float, frequencies=None) -> dict:
    """Reconstruct the listed nonlinear, ground-state harmonic source model."""
    old = json.loads((DATA / "literature.json").read_text())["janaf_printed"]
    k, h = 1.380649e-23, 6.62607015e-34
    theta = np.asarray(old["vibrational_frequencies_expanded_cm_inverse"] if frequencies is None else frequencies)*1.4387768775039338
    x = theta/t
    qtrans = (2*math.pi*old["mass_u"]*1.66053906660e-27*k*t/h**2)**1.5*k*t/100000
    qrot = math.sqrt(math.pi)/old["sigma"]*(8*math.pi**2*k*t/h**2)**1.5*math.sqrt(old["inertia_product_g3_cm6"]*1e-21)
    return {"Hthermal_J_mol":float(R*(4*t+np.sum(theta/np.expm1(x)))),
            "S_J_mol_K":float(R*(math.log(qtrans*qrot)+4+np.sum(x/np.expm1(x)-np.log(-np.expm1(-x))))),
            "Cp_J_mol_K":float(R*(4+np.sum(x*x*np.exp(x)/np.expm1(x)**2)))}


def modern_diagnostic(t: float, source: dict, current: dict) -> dict:
    """Report the molecule-specific inspected numerical evidence separately."""
    facts = json.loads((DATA / "literature.json").read_text())
    table = read_table(DATA / inputs()["tables"][NAME])
    monomer = read_table(DATA / inputs()["monomer_table"])
    p,ref = old_rrho(t),old_rrho(298.15)
    result = {"printed_RRHO_minus_JANAF_H_increment_J_mol":p["Hthermal_J_mol"]-ref["Hthermal_J_mol"]-1000*table[t][3],
        "printed_RRHO_minus_JANAF_S_J_mol_K":p["S_J_mol_K"]-table[t][1],
        "printed_RRHO_minus_JANAF_Cp_J_mol_K":p["Cp_J_mol_K"]-table[t][0],
        "source_dimer_dissociation_H298_J_mol":1000*(2*monomer[298.15][4]-table[298.15][4]),
        "source_dimer_dissociation_H1000_J_mol":1000*(2*(monomer[298.15][4]+monomer[1000.][3])-(table[298.15][4]+table[1000.][3])),
        "source_dimer_dissociation_D0_J_mol":1000*(2*(monomer[298.15][4]+monomer[0.][3])-(table[298.15][4]+table[0.][3]))}
    alternative = facts["janaf_printed"].get("diagnostic_alternative_frequencies_cm_inverse")
    if alternative is not None:
        q,qref = old_rrho(t,alternative),old_rrho(298.15,alternative)
        delta = facts["primary_energy"]["Hf298_dimer_J_mol"]-1000*table[298.15][4]
        result.update(alternative_RRHO_minus_JANAF_H_increment_J_mol=q["Hthermal_J_mol"]-qref["Hthermal_J_mol"]-1000*table[t][3],
            alternative_RRHO_minus_JANAF_S_J_mol_K=q["S_J_mol_K"]-table[t][1],
            alternative_RRHO_minus_JANAF_Cp_J_mol_K=q["Cp_J_mol_K"]-table[t][0],
            abstract_H298_shift_J_mol=delta, abstract_energy_only_K_factor=math.exp(-delta/(R*t)))
    else:
        result["source_minus_primary_dimer_dissociation_H1000_J_mol"] = result["source_dimer_dissociation_H1000_J_mol"]-facts["primary_energy"]["dimer_to_two_monomers_H_J_mol"]
    return result



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
