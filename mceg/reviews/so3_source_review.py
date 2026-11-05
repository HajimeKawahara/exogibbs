"""Replay SO3 source thermochemistry and its archived curated-cos-v1 footprint."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import ROOT, file_hash
import select_major_molecules as major

DATA = ROOT / "tests/data/validation/so3_source_review/20261105"
GRID = ROOT / "tests/data/validation/curated_cos_adoption/20261105/ions.npz"
R, NASA_R = 8.31446261815324, 8.314510
NAME = "SO3"
SPECIES = "O3S1"
COMPOSITION = {'S': 1, 'O': 3}


@lru_cache(maxsize=1)
def inputs() -> dict:
    """Load only the pinned neutral molecular identity and numerical inputs."""
    return json.loads((DATA / "inputs.json").read_text())


def curated_properties(temperature: float) -> dict:
    """Evaluate the inherited literal production function, within its domain."""
    if not 1500 <= temperature <= 4000:
        raise ValueError("Temperature outside reviewed curated domain")
    t = temperature
    gas = json.loads((ROOT / "src/exogibbs/data/curated/gas_v1.json").read_text())
    if SPECIES in gas["nasa9_reactions"]:
        coefficients = gas["nasa9_reactions"][SPECIES]
        h, s, cp = nasa_properties({"intervals": [{"bounds_K": [1500, 4000],
                                                   "coefficients": coefficients}]}, t, R)
        value = -(h/t-s)/R
    else:
        lines = (ROOT / "src/exogibbs/data/fastchem/logK/logK.dat").read_text().splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith(SPECIES+" "))
        a, b, c, d, e = map(float, lines[index+1].split())
        value = a/t+b*math.log(t)+c+d*t+e*t*t
        h = R*(-a+b*t+d*t*t+2*e*t**3)
        s = R*(b*math.log(t)+b+c+2*d*t+3*e*t*t)
        cp = R*(b+2*d*t+6*e*t*t)
    return {"ln_K": value, "H_J_mol": h, "S_J_mol_K": s, "Cp_J_mol_K": cp}


def table_properties(table: dict, temperature: float) -> tuple:
    """Return H, S, Cp at a literal knot with one fixed 298 K formation anchor."""
    node = table[temperature]
    return 1000*(table[298.15][4]+node[3]), node[1], node[0]


def source_comparison(tables: dict, nasa: dict, temperature: float) -> dict:
    """Separate molecular thermal, atomic thermal and balanced energy terms."""
    t = temperature
    nh0 = {n: r["hf298_J_mol"]-r["h298_minus_h0_J_mol"] for n, r in nasa.items()}
    jh0 = {n: 1000*(tb[298.15][4]+tb[0.0][3]) for n, tb in tables.items()}
    nd0 = sum(c*nh0[n] for n, c in COMPOSITION.items())-nh0[NAME]
    jd0 = sum(c*jh0[n] for n, c in COMPOSITION.items())-jh0[NAME]
    jp = {n: table_properties(tb, t) for n, tb in tables.items()}
    np_ = {n: nasa_properties(r, t, NASA_R) for n, r in nasa.items()}
    phi_j = {n: (h-jh0[n])/(R*t)-s/R for n, (h,s,cp) in jp.items()}
    phi_n = {n: (h-nh0[n])/(NASA_R*t)-s/NASA_R for n,(h,s,cp) in np_.items()}
    terms = {"energy": (jd0/R-nd0/NASA_R)/t,
             "molecular_thermal": phi_n[NAME]-phi_j[NAME],
             "atomic_thermal": sum(c*(phi_j[n]-phi_n[n]) for n,c in COMPOSITION.items())}
    ln_n = sum(c*(np_[n][0]/t-np_[n][1])/NASA_R for n,c in COMPOSITION.items())-(np_[NAME][0]/t-np_[NAME][1])/NASA_R
    reaction = {key: jp[NAME][i]-sum(c*jp[n][i] for n,c in COMPOSITION.items())
                for i,key in enumerate(("H_J_mol","S_J_mol_K","Cp_J_mol_K"))}
    ln_j = -(reaction["H_J_mol"]/t-reaction["S_J_mol_K"])/R
    return {"ln_K": ln_j, "nasa_ln_K": ln_n, "source_over_nasa": math.exp(ln_j-ln_n),
            "source_minus_nasa_ln_K_components": terms,
            "decomposition_closure_ln_K": ln_j-ln_n-sum(terms.values()),
            "energy_aligned_source_over_nasa": math.exp(terms["molecular_thermal"]+terms["atomic_thermal"]),
            "native_source_D0_J_mol": jd0, "native_nasa_D0_J_mol": nd0,
            "molecular": {"source_H_J_mol": jp[NAME][0], "source_S_J_mol_K": jp[NAME][1],
                          "source_Cp_J_mol_K": jp[NAME][2], "nasa_H_J_mol": np_[NAME][0],
                          "nasa_S_J_mol_K": np_[NAME][1], "nasa_Cp_J_mol_K": np_[NAME][2]},
            "reaction": reaction}


def partition_diagnostic(temperature: float, record: dict) -> dict | None:
    """Compare tabulated Q at exact knots; never extrapolate the finite sum."""
    config = inputs()["partition"]
    if config is None:
        return None
    table = np.loadtxt(DATA / config["file"])
    match = table[table[:, 0] == temperature]
    if len(match) != 1:
        return None
    t = temperature
    h, s, _ = nasa_properties(record, t, NASA_R)
    h0 = record["hf298_J_mol"]-record["h298_minus_h0_J_mol"]
    phi = (h-h0)/(NASA_R*t)-s/NASA_R
    kt = 1.380649e-23*t
    ln_trans = 1.5*math.log(2*math.pi*record["mass_u"]*major.ATOMIC_MASS_KG*kt/6.62607015e-34**2)+math.log(kt/100000)
    q_nasa = math.exp(-phi-ln_trans)
    raw_q = float(match[0, 1])
    q = raw_q/config["nuclear_spin_divisor"]
    return {"raw_Q": raw_q, "spin_normalized_Q": q, "nasa_internal_Q": q_nasa,
            "spin_normalized_Q_over_nasa": q/q_nasa,
            "isotope_translation_K_proxy": q/q_nasa*(config["mass_u"]/record["mass_u"])**1.5,
            "tabulated_range_K": [float(table[0, 0]), float(table[-1, 0])],
            "isotope_scope": config["scope"],
            "caloric_derivatives": "not inferred from the rounded partition table"}


def footprint() -> dict:
    """Read 75 existing ion-inclusive endpoints; no replacement solve is run."""
    with np.load(GRID) as grid:
        col = grid["species"].tolist().index(SPECIES)
        el = grid["elements"].tolist().index(inputs()["footprint_element"])
        n = np.exp(grid["ln_n"])
        share = grid["formula_matrix"][el, col]*n[:, col]/grid["budget"][:, el]
        i = int(np.argmax(share))
        return {"model": "curated-cos-v1", "cases": len(n), "species_count": n.shape[1],
                "all_converged": bool(np.all(grid["converged"])), "element": inputs()["footprint_element"],
                "atoms_per_molecule": float(grid["formula_matrix"][el, col]),
                "maximum_element_share": float(share[i]), "case_index": i,
                "mole_fraction_at_maximum_share": float(np.exp(grid["ln_x"][i, col])),
                "maximum_mole_fraction": float(np.exp(grid["ln_x"][:, col]).max()),
                "temperature_K": float(grid["temperature_K"][i]),
                "pressure_bar": float(grid["pressure_bar"][i]), "C_over_O": float(grid["C_over_O"][i]),
                "maximum_abs_h_error": max(abs(h+curated_properties(t)["ln_K"])
                    for t, h in zip(grid["temperature_K"], grid["h"][:, col])),
                "maximum_relative_element_residual": float(np.max(abs(
                    n @ grid["formula_matrix"][el]/grid["budget"][:, el]-1)))}


def review() -> dict:
    """Reproduce source diagnostics without promoting a candidate to production."""
    config = inputs()
    nasa = json.loads((DATA / "nasa_records.json").read_text())
    tables = {n: read_table(ROOT/p) for n, p in config["atomic_tables"].items()}
    tables[NAME] = read_table(DATA / config["molecular_janaf_table"])
    selected = dict(tables)
    if NAME == "H2O2":
        selected[NAME] = {float(t): row for t, row in json.loads((DATA/"dorofeeva2003_nodes.json").read_text())["rows"].items()}
    atct = config["atct"]
    new_d0 = 1000*(sum(c*atct[n]["hf0_kJ_mol"] for n,c in COMPOSITION.items())-atct[NAME]["hf0_kJ_mol"])
    first = source_comparison(selected, nasa, 300)
    old_d0 = first["native_source_D0_J_mol"]
    if NAME == "SO3":
        reference = json.loads(major.REFERENCE_PATH.read_text())
        old_d0 = R*major.atomization_kelvin(NAME, "atct", reference)
    rows = []
    for t in config["temperatures_K"]:
        source = source_comparison(selected, nasa, t)
        current = curated_properties(t) if t >= 1500 else None
        candidate = source["nasa_ln_K"]+(new_d0/R-source["native_nasa_D0_J_mol"]/NASA_R)/t
        adopted_nasa = source["nasa_ln_K"]+(old_d0/R-source["native_nasa_D0_J_mol"]/NASA_R)/t
        row = {"temperature_K": t, "table_source": source,
               "adopted_NASA_ATcT_reconstruction_ln_K": adopted_nasa if NAME == "SO3" else None,
               "legacy_janaf": source_comparison(tables,nasa,t) if t in tables[NAME] else None,
               "curated": current,
               "curated_over_nasa": None if current is None else math.exp(current["ln_K"]-source["nasa_ln_K"]),
               "curated_minus_table_reaction": None if current is None else {k:current[k]-v for k,v in source["reaction"].items()},
               "same_thermal_ATcT_energy_only_K_ratio": math.exp((new_d0-old_d0)/(R*t)),
               "NASA_thermal_ATcT_K_over_curated": None if current is None else math.exp(candidate-current["ln_K"]),
               "partition": partition_diagnostic(t,nasa[NAME])}
        rows.append(row)
    result = {"molecule": NAME, "species_id": SPECIES, "composition": COMPOSITION,
              "standard_pressure_Pa": 100000, "source_R_J_mol_K": R, "nasa_R_J_mol_K": NASA_R,
              "curated_model": "curated-cos-v1", "atct_version": "1.228",
              "balanced_ATcT_D0_J_mol": new_d0, "inherited_source_D0_J_mol": old_d0,
              "energy_only_delta_D0_J_mol": new_d0-old_d0, "source_samples": rows,
              "footprint": footprint(), "production_change": "none", "equilibrium_substitution": "not_run",
              "physical_accuracy": "unknown"}
    if NAME == "N2O3":
        props = {n:nasa_properties(nasa[n],300,NASA_R) for n in [NAME,"NO","NO2"]}
        result["dissociation_at_300K"] = {"reaction":"N2O3 -> NO + NO2",
            "NASA_delta_G_kJ_mol":sum(sign*(props[n][0]-300*props[n][1]) for n,sign in [("NO",1),("NO2",1),(NAME,-1)])/1000,
            "ATcT_network_input_delta_G_kJ_mol":-1.84,"ATcT_input_prior_uncertainty_kJ_mol":0.07,
            "source":"Beattie1957 third-law evaluation apud Gurvich, ATcT1.228 TN1573.10; not independent of ATcT"}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(review(), indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
