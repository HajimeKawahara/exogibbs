"""Separate formaldehyde source energies, thermal functions and saved abundances."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import BASELINE, ROOT, file_hash

DATA = ROOT / "tests/data/validation/ch2o_source_review/20261105"
GRID = ROOT / "tests/data/validation/curated_cos_adoption/20261105"
R, NASA_R = 8.31446261815324, 8.314510
KB, PLANCK, AMU = 1.380649e-23, 6.62607015e-34, 1.66053906892e-27
SPECIES = "C1H2O1"
COMPOSITION = {"C": 1, "H": 2, "O": 1}


@lru_cache(maxsize=1)
def setup():
    """Pin the reviewed production version, independent of future defaults."""
    import jax
    from exogibbs.presets import curated_cos
    jax.config.update("jax_enable_x64", True)
    return curated_cos.chemsetup(ions=True)


def curated_properties(temperature: float) -> dict:
    """Differentiate the inherited five-coefficient formation ln K exactly."""
    if not 1500 <= temperature <= 4000:
        raise ValueError("Temperature outside reviewed curated domain")
    lines = (ROOT / "src/exogibbs/data/fastchem/logK/logK.dat").read_text().splitlines()
    i = next(i for i, line in enumerate(lines) if line.startswith(SPECIES + " "))
    a, b, c, d, e = map(float, lines[i+1].split())
    t = temperature
    return {"ln_K": a/t+b*math.log(t)+c+d*t+e*t*t,
            "H_J_mol": R*(-a+b*t+d*t*t+2*e*t**3),
            "S_J_mol_K": R*(b*math.log(t)+b+c+2*d*t+3*e*t*t),
            "Cp_J_mol_K": R*(b+2*d*t+6*e*t*t)}


def log_translation(temperature: float, mass_u: float) -> float:
    """Return one-bar ideal-gas translation with an explicit mass convention."""
    kt = KB*temperature
    return 1.5*math.log(2*math.pi*mass_u*AMU*kt/PLANCK**2)+math.log(kt/100000)


def partition_calorics(table: np.ndarray, temperature: float, step: float) -> dict:
    """Five-point diagnostics; rounding bounds do not cover missing states."""
    nodes = temperature + step*np.arange(-2, 3)
    rows = table[np.isin(table[:, 0], nodes)]
    if len(rows) != 5:
        raise ValueError("Partition stencil outside table")
    first = np.array([1, -8, 0, 8, -1])/(12*step)
    second = np.array([-1, 16, -30, 16, -1])/(12*step**2)
    d1, d2 = first @ np.log(rows[:, 1]), second @ np.log(rows[:, 1])
    t = temperature
    rounding = .00005/rows[:, 1]  # Downloaded Q has four decimal places.
    return {"H_increment_J_mol": float(R*(2.5*t+t*t*d1)),
            "Cp_J_mol_K": float(R*(2.5+2*t*d1+t*t*d2)),
            "Cp_rounding_bound_J_mol_K": float(R*((2*t*abs(first)+t*t*abs(second)) @ rounding))}


def footprint(mode: str) -> dict:
    """Read the adopted COS grid; this is not a replacement calculation."""
    with np.load(GRID / (mode+".npz")) as g:
        col = g["species"].tolist().index(SPECIES)
        el = g["elements"].tolist().index("C")
        n = np.exp(g["ln_n"])
        shares = g["formula_matrix"][el, col]*n[:, col]/g["budget"][:, el]
        i = int(np.argmax(shares))
        return {"model": "curated-cos-v1", "mode": mode, "cases": len(n),
                "species_count": n.shape[1], "all_converged": bool(g["converged"].all()),
                "maximum_carbon_share": float(shares[i]), "case_index": i,
                "mole_fraction_at_maximum_share": float(np.exp(g["ln_x"][i, col])),
                "temperature_K": float(g["temperature_K"][i]),
                "pressure_bar": float(g["pressure_bar"][i]), "C_over_O": float(g["C_over_O"][i]),
                "maximum_abs_h_error": max(abs(h+curated_properties(t)["ln_K"])
                    for t, h in zip(g["temperature_K"], g["h"][:, col])),
                "maximum_relative_carbon_residual": float(np.max(abs(
                    n @ g["formula_matrix"][el]/g["budget"][:, el]-1)))}


def review() -> dict:
    """Replay balanced source comparisons without extrapolating curated or PF."""
    sources = json.loads((DATA / "sources.json").read_text())
    for base, entries in ((ROOT, sources["existing_inputs"]), (DATA, sources["archived_files"])):
        for name, expected in entries.items():
            if file_hash(base/name) != expected:
                raise ValueError("Input hash mismatch: "+name)
    inputs = json.loads((DATA / "inputs.json").read_text())
    nasa = inputs["nasa_records"]
    frozen = json.loads((BASELINE / "inputs.json").read_text())
    for record in nasa.values():
        old = frozen["nasa9_records"][record["source_name"]]
        if [r["coefficients"] for r in old["intervals"]] != [r["coefficients"] for r in record["intervals"]]:
            raise ValueError("Frozen NASA coefficient mismatch")
    tables = {n: read_table(ROOT/p) for n, p in inputs["atomic_janaf_tables"].items()}
    tables["CH2O"] = read_table(DATA / "C-061.txt")
    nh0 = {n: r["hf298_J_mol"]-r["h298_minus_h0_J_mol"] for n, r in nasa.items()}
    jh0 = {n: 1000*(tb[298.15][4]+tb[0.0][3]) for n, tb in tables.items()}
    balanced = lambda h: sum(c*h[n] for n, c in COMPOSITION.items())-h["CH2O"]
    nd0, jd0 = balanced(nh0), balanced(jh0)
    ad0 = 1000*balanced({n: r["hf0_kJ_mol"] for n, r in inputs["atct_1.228"].items()})
    pf = np.loadtxt(DATA / "AYTY.pf")
    model, rows = setup(), []
    for t in inputs["temperatures_K"]:
        jp = {n: (1000*(tb[298.15][4]+tb[t][3]), tb[t][1], tb[t][0]) for n, tb in tables.items()}
        np_ = {n: nasa_properties(record, t, NASA_R) for n, record in nasa.items()}
        phi_j = {n: (h-jh0[n])/(R*t)-s/R for n, (h, s, _) in jp.items()}
        phi_n = {n: (h-nh0[n])/(NASA_R*t)-s/NASA_R for n, (h, s, _) in np_.items()}
        reaction = {key: jp["CH2O"][i]-sum(c*jp[n][i] for n, c in COMPOSITION.items())
                    for i, key in enumerate(("H_J_mol", "S_J_mol_K", "Cp_J_mol_K"))}
        ln_j = -(reaction["H_J_mol"]/t-reaction["S_J_mol_K"])/R
        ln_n = sum(c*phi_n[n] for n, c in COMPOSITION.items())-phi_n["CH2O"]+nd0/(NASA_R*t)
        terms = {"energy": (jd0/R-nd0/NASA_R)/t,
                 "molecular_thermal": phi_n["CH2O"]-phi_j["CH2O"],
                 "atomic_thermal": sum(c*(phi_j[n]-phi_n[n]) for n, c in COMPOSITION.items())}
        current = curated_properties(t) if t >= 1500 else None
        row = {"temperature_K": t, "janaf_ln_K": ln_j, "nasa_ln_K": ln_n,
               "janaf_minus_nasa_ln_K_components": terms,
               "decomposition_closure_ln_K": ln_j-ln_n-sum(terms.values()),
               "molecular_janaf_Cp_J_mol_K": jp["CH2O"][2],
               "molecular_nasa_common_R_Cp_J_mol_K": np_["CH2O"][2]*R/NASA_R,
               "curated": current, "curated_over_nasa": None if current is None else math.exp(current["ln_K"]-ln_n),
               "curated_minus_janaf_reaction": None if current is None else {k: current[k]-v for k, v in reaction.items()},
               "curated_fit_runtime_closure_ln_K": None if current is None else current["ln_K"]+float(model.hvector_func(float(t))[model.species.index(SPECIES)]),
               "ATcT_energy_only_K_over_current": math.exp((ad0-jd0)/(R*t)),
               "nasa_thermal_ATcT_K_over_current": None if current is None else math.exp(ln_n+(ad0/R-nd0/NASA_R)/t-current["ln_K"])}
        if t <= pf[-1, 0]:
            q = float(pf[pf[:, 0] == t, 1][0])
            qn = math.exp(-phi_n["CH2O"]-log_translation(t, nasa["CH2O"]["mass_u"]))
            normalized = q/inputs["partition"]["spin_divisor"]
            row["ayty"] = {"Q_printed": q, "Q_spin_removed": normalized,
                           "Q_over_nasa": normalized/qn,
                           "isotope_translation_K_proxy": normalized/qn*(inputs["partition"]["isotope_mass_u"]/nasa["CH2O"]["mass_u"])**1.5,
                           "calorics": None}
            if t+40 <= pf[-1, 0]:
                calorics = partition_calorics(pf, t, 10)
                calorics["Cp_step20_J_mol_K"] = partition_calorics(pf, t, 20)["Cp_J_mol_K"]
                row["ayty"]["calorics"] = calorics
        else:
            row["ayty"] = None
        rows.append(row)
    return {"molecule": "CH2O", "species_id": SPECIES, "standard_pressure_Pa": 100000,
            "native_R_J_mol_K": {"nasa": NASA_R, "janaf_and_curated": R},
            "balanced_D0_J_mol": {"nasa": nd0, "janaf": jd0, "atct_1.228": ad0},
            "ATcT_minus_JANAF_D0_J_mol": ad0-jd0,
            "source_samples": rows, "footprints": [footprint(m) for m in ("ions", "neutral")],
            "production_change": "none", "equilibrium_substitution": "not_run", "decision": inputs["decision"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(review(), indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
