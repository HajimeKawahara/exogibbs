"""Audit OH source energy, electronic weights and finite-state calorics offline."""

import argparse
import bz2
import json
import math
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import BASELINE, ROOT, file_hash, nasa_g
import select_major_molecules as major


DATA = ROOT / "tests/data/validation/oh_source_review/20261105"
GRID = ROOT / "tests/data/validation/curated_h2o_adoption/20261104/ions.npz"
FASTCHEM = ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat"
DATASETS = ("MYTHOS", "MoLLIST-OH")
TEMPERATURES = (298.15, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000)
KB = 1.380649e-23
PLANCK = 6.62607015e-34
C2 = PLANCK * 29979245800 / KB
MASS = 17.00734
ISOTOPE_MASS = 17.00273965


def extract(raw: Path, output: Path) -> dict:
    """Preserve every downloaded level without adjusting energies or weights."""
    arrays, inventory = {}, {}
    for name in DATASETS:
        path = raw / (name + ".states.bz2")
        with bz2.open(path, "rt") as stream:
            rows = [line.split() for line in stream]
        labels = {"MYTHOS": (9, 10), "MoLLIST-OH": (7, 5)}[name]
        electronic = sorted({row[labels[0]] for row in rows})
        array = np.array([[float(row[i]) for i in (1, 2, 3)]
                          + [electronic.index(row[labels[0]]), float(row[labels[1]])]
                          for row in rows])
        arrays[name] = array
        pf = np.loadtxt(raw / (name + ".pf"))
        temperatures = set(TEMPERATURES) | {1, 300, 8000, 10000}
        selected = pf[np.isin(pf[:, 0], list(temperatures))]
        inventory[name] = {
            "states_sha256": file_hash(path), "state_count": len(rows),
            "electronic_labels": electronic, "maximum_energy_cm-1": float(array[:, 0].max()),
            "maximum_J": float(array[:, 2].max()),
            "gtot_over_2Jplus1": np.unique(array[:, 1]/(2*array[:, 2]+1)).tolist(),
            "pf_sha256": file_hash(raw / (name + ".pf")),
            "published_pf_rows": selected.tolist(),
        }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "states.npz", **arrays)
    result = {"columns": ["energy_cm-1", "gtot", "J", "electronic_index", "v"],
              "datasets": inventory}
    (output / "state_inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def log_translation(temperature: float, mass: float = MASS) -> float:
    """Return the molecular translation factor at one bar."""
    kt = KB*temperature
    return (1.5*math.log(2*math.pi*mass*major.ATOMIC_MASS_KG*kt/PLANCK**2)
            + math.log(kt/100000))


def moments(states: np.ndarray, temperature: float, ground_index: int,
            d0_kelvin: float) -> dict:
    """Remove only nuclear spin 2; explicit fine structure and parity remain."""
    energy = states[:, 0]*C2
    pop = states[:, 1]/2*np.exp(-energy/temperature)
    q = float(pop.sum())
    mean = float(pop @ energy/q)
    variance = float(pop @ (energy-mean)**2/q)
    bound = energy <= d0_kelvin
    bound_q = float(pop[bound].sum())
    bound_mean = float(pop[bound] @ energy[bound]/bound_q)
    bound_variance = float(pop[bound] @ (energy[bound]-bound_mean)**2/bound_q)
    return {"Q": q, "mean_energy_K": mean, "variance_energy_K2": variance,
            "excited_electronic_Q_fraction": float(pop[states[:, 3] != ground_index].sum()/q),
            "above_D0_Q_fraction": float(pop[energy > d0_kelvin].sum()/q),
            "Cp_change_on_D0_cut_J_mol_K": major.R*(bound_variance-variance)/temperature**2,
            "highest_J_shell_Q_fraction": float(pop[states[:, 2] == states[:, 2].max()].sum()/q),
            "J_ge_45_5_Q_fraction": float(pop[states[:, 2] >= 45.5].sum()/q),
            "v_ge_10_Q_fraction": float(pop[states[:, 4] >= 10].sum()/q)}


def curated_ln_k(temperature: float) -> float:
    """Evaluate the inherited FastChem column in the curated production domain."""
    if not 1500 <= temperature <= 4000:
        raise ValueError("Temperature outside curated domain")
    lines = FASTCHEM.read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("H1O1 "))
    a, b, c, d, e = map(float, lines[index+1].split())
    return a/temperature+b*math.log(temperature)+c+d*temperature+e*temperature**2


def footprint() -> dict:
    """Describe existing endpoints; this is not a substitution or a new solve."""
    with np.load(GRID) as grid:
        column = grid["species"].tolist().index("H1O1")
        oxygen = grid["elements"].tolist().index("O")
        n = np.exp(grid["ln_n"])
        shares = n[:, column]/grid["budget"][:, oxygen]
        i = int(np.argmax(shares))
        return {"model": "curated-h2o-v1", "cases": len(n), "species": n.shape[1],
                "all_converged": bool(np.all(grid["converged"])), "case_index": i,
                "maximum_oxygen_fraction": float(shares[i]),
                "mole_fraction_at_maximum_share": float(np.exp(grid["ln_x"][i, column])),
                "maximum_mole_fraction": float(np.exp(grid["ln_x"][:, column]).max()),
                "temperature_K": float(grid["temperature_K"][i]),
                "pressure_bar": float(grid["pressure_bar"][i]),
                "C_over_O": float(grid["C_over_O"][i]),
                "maximum_abs_h_error": max(abs(h+curated_ln_k(t)) for t, h in
                    zip(grid["temperature_K"], grid["h"][:, column])),
                "maximum_relative_oxygen_residual": float(np.max(
                    np.abs(n @ grid["formula_matrix"][oxygen]/grid["budget"][:, oxygen]-1)))}


def review() -> dict:
    """Separate fit residuals, atomic anchors, molecular Q and caloric differences."""
    inputs = json.loads((BASELINE / "inputs.json").read_text())
    literature = json.loads((DATA / "literature.json").read_text())
    reference = json.loads(major.REFERENCE_PATH.read_text())
    inventory = json.loads((DATA / "state_inventory.json").read_text())["datasets"]
    tables = {s: read_table(DATA / filename) for s, filename in
              {"OH": "H-038.txt", "H": "H-001.txt", "O": "O-001.txt"}.items()}
    headers = literature["nasa_headers"]
    h0_nasa = {s: r["hf298_J_mol"]-r["h298_minus_h0_J_mol"] for s, r in headers.items()}
    h0_janaf = {s: 1000*(table[298.15][4]+table[0][3]) for s, table in tables.items()}
    d0_nasa = h0_nasa["H"]+h0_nasa["O"]-h0_nasa["OH"]
    d0_janaf = h0_janaf["H"]+h0_janaf["O"]-h0_janaf["OH"]
    d0_atct = 1000*(sum(reference["atct"]["records"][s]["hf0_kJ_mol"] for s in ("H", "O"))
                    - literature["atct"]["OH_hf0_kJ_mol"])
    energy_shift = d0_atct/major.R-d0_nasa/major.NASA_R
    with np.load(DATA / "states.npz") as archive:
        states = {name: archive[name] for name in DATASETS}
    samples = []
    for t in TEMPERATURES:
        ng = {s: nasa_g(inputs["nasa9_records"][s], t) for s in headers}
        nthermal = {s: ng[s]-h0_nasa[s]/(major.NASA_R*t) for s in headers}
        jthermal = {s: 1000*(tab[t][3]-tab[0][3])/(major.R*t)-tab[t][1]/major.R
                    for s, tab in tables.items()}
        native_janaf = sum(sign*(jthermal[s]+h0_janaf[s]/(major.R*t))
                           for s, sign in (("H", 1), ("O", 1), ("OH", -1)))
        ln_nasa = ng["H"]+ng["O"]-ng["OH"]
        hybrid = ln_nasa+energy_shift/t
        q_nasa = math.exp(-nthermal["OH"]-log_translation(t))
        nh, ns, ncp = nasa_properties(inputs["nasa9_records"]["OH"], t, major.NASA_R)
        sources = {}
        for name in DATASETS:
            ground = inventory[name]["electronic_labels"].index("X2Pi" if name == "MYTHOS" else "X(2PI)")
            m = moments(states[name], t, ground, d0_atct/major.R)
            h_increment = major.R*(2.5*t+m["mean_energy_K"])
            entropy = major.R*(math.log(m["Q"])+log_translation(t)+2.5+m["mean_energy_K"]/t)
            cp = major.R*(2.5+m["variance_energy_K2"]/t**2)
            pf = dict(inventory[name]["published_pf_rows"]).get(t)
            ln_k = hybrid+math.log(m["Q"]/q_nasa)
            sources[name] = {**m, "published_Q_with_spin": pf,
                "states_over_published_Q": 2*m["Q"]/pf if pf else None,
                "Q_over_nasa_internal": m["Q"]/q_nasa,
                "H_increment_J_mol": h_increment, "S_natural_mass_J_mol_K": entropy,
                "Cp_J_mol_K": cp, "delta_H_from_nasa_common_R_J_mol": h_increment-(nh-h0_nasa["OH"])*major.R/major.NASA_R,
                "delta_S_from_nasa_common_R_J_mol_K": entropy-ns*major.R/major.NASA_R,
                "delta_Cp_from_nasa_common_R_J_mol_K": cp-ncp*major.R/major.NASA_R,
                "ATcT_same_atoms_ln_K": ln_k,
                "ATcT_same_atoms_over_curated": math.exp(ln_k-curated_ln_k(t)) if t >= 1500 else None}
        samples.append({"temperature_K": t, "nasa_ln_K": ln_nasa,
            "janaf_ln_K": native_janaf, "janaf_over_nasa": math.exp(native_janaf-ln_nasa),
            "curated_ln_K": curated_ln_k(t) if t >= 1500 else None,
            "curated_over_nasa": math.exp(curated_ln_k(t)-ln_nasa) if t >= 1500 else None,
            "curated_over_janaf": math.exp(curated_ln_k(t)-native_janaf) if t >= 1500 else None,
            "nasa_ATcT_ln_K": hybrid,
            "nasa_ATcT_over_curated": math.exp(hybrid-curated_ln_k(t)) if t >= 1500 else None,
            "janaf_minus_nasa_ln_K_components": {
                "balanced_energy": (d0_janaf/major.R-d0_nasa/major.NASA_R)/t,
                "molecular_thermal": nthermal["OH"]-jthermal["OH"],
                "atomic_thermal": sum(jthermal[s]-nthermal[s] for s in ("H", "O"))},
            "Q_nasa": q_nasa, "nasa_Cp_common_R_J_mol_K": ncp*major.R/major.NASA_R,
            "janaf_OH_H_increment_J_mol": 1000*(tables["OH"][t][3]-tables["OH"][0][3]),
            "janaf_OH_Cp_J_mol_K": tables["OH"][t][0], "modern": sources})
    return {"reaction": "H(g)+O(g)->OH(g)", "standard_pressure_Pa": 100000,
        "balanced_D0_J_mol": {"atct_1_172": d0_atct, "nasa": d0_nasa, "janaf": d0_janaf},
        "ATcT_minus_nasa_D0_over_k_K": energy_shift,
        "isotope_to_natural_translation_factor": (ISOTOPE_MASS/MASS)**1.5,
        "source_samples": samples, "footprint": footprint(),
        "decision": "Prioritize a replacement candidate; known JANAF defects make indefinite retention scientifically unjustified. MYTHOS/ATcT and NASA/ATcT are candidate alternatives, pending continuous fit and finite substitution.",
        "production_change": "none", "equilibrium_substitution": "not_run"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = extract(args.extract, DATA) if args.extract else review()
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        print(serialized, end="")
