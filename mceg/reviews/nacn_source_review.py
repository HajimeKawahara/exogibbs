"""Replay NaCN source comparisons and its archived final curated footprint."""

from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import ROOT

DATA = ROOT / "tests/data/validation/nacn_source_review/20261106"
GRID = ROOT / "tests/data/validation/curated_tio2_adoption/20261106/ions.npz"
R, NASA_R = 8.31446261815324, 8.314510
NAME, SPECIES = "NaCN", "C1N1Na1"


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


def modern_diagnostic(t: float, source: dict, current: dict) -> dict:
    """Report the molecule-specific inspected numerical evidence separately."""
    literature = json.loads((DATA / "literature.json").read_text())
    old, new = literature["janaf_printed"], literature["modern_rotation"]
    kb, planck, amu = 1.380649e-23, 6.62607015e-34, 1.66053906660e-27
    c2 = planck*29979245800/kb
    theta = np.asarray(old["vibrational_frequencies_expanded_cm_inverse"])*c2
    x = theta/t
    log_vibration = -float(np.log(-np.expm1(-x)).sum())
    mean_vibration = float((theta/np.expm1(x)).sum())
    vibration_cp = float((x*x*np.exp(-x)/(-np.expm1(-x))**2).sum())
    log_translation = 1.5*math.log(2*math.pi*old["mass_u"]*amu*kb*t/planck**2)+math.log(kb*t/100000)
    old_qrot = t/(c2*old["B0_cm_inverse"])
    old_h = R*(3.5*t+mean_vibration)
    old_s = R*(log_translation+math.log(old_qrot)+log_vibration+3.5+mean_vibration/t)
    old_cp = R*(3.5+vibration_cp)
    a = new["A_minus_mean_BC_MHz"]+new["mean_BC_MHz"]
    b = new["mean_BC_MHz"]+2*new["B_minus_C_over4_MHz"]
    c = new["mean_BC_MHz"]-2*new["B_minus_C_over4_MHz"]
    theta_rot = np.asarray([a,b,c])*1e6*planck/kb
    new_qrot = math.sqrt(math.pi)*t**1.5/math.sqrt(float(np.prod(theta_rot)))
    table = read_table(DATA/inputs()["tables"][NAME])
    return {"old_linear_RRHO_H_minus_H0_J_mol": old_h,
            "old_linear_RRHO_S_J_mol_K": old_s, "old_linear_RRHO_Cp_J_mol_K": old_cp,
            "old_RRHO_minus_JANAF_thermal_H_J_mol": old_h-1000*(table[t][3]-table[0.][3]),
            "old_RRHO_minus_JANAF_S_J_mol_K": old_s-table[t][1],
            "old_RRHO_minus_JANAF_Cp_J_mol_K": old_cp-table[t][0],
            "measured_ABC_MHz": [a,b,c], "old_linear_qrot": old_qrot,
            "measured_nonlinear_qrot": new_qrot,
            "rotation_only_q_ratio_not_full_K": new_qrot/old_qrot,
            "nonlinear_minus_linear_rotational_Cp_J_mol_K": R/2,
            "scope": "Old linear RRHO reconstructs the source assumption; measured-rotation-only diagnostic does not supply a full nonlinear thermal model or formation-K correction."}


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
