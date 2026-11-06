"""Audit CO thermal functions from frozen states and an unchanged energy anchor.

The default command is offline and performs no equilibrium substitution.
--extract accepts the original ExoMol downloads to regenerate the compact packet.
"""

import argparse
import bz2
import json
import math
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties
from gas_source_comparison import ROOT, file_hash
import select_major_molecules as major


DATA = ROOT / "tests/data/validation/co_source_review/20261103"
GRID = ROOT / "tests/data/validation/curated_tio_adoption/20261103/ions.npz"
CURATED = ROOT / "src/exogibbs/data/curated/gas_v1.json"
TEMPERATURES = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000)
KB = 1.380649e-23
PLANCK = 6.62607015e-34
C2 = PLANCK * 29979245800 / KB
ISOTOPE_MASS = 27.99491462
DATASETS = ("4thplus", "Li2015")


def extract(raw: Path, output: Path) -> dict:
    """Retain every state's energy, degeneracy, J and electronic-state flag."""
    arrays, facts, samples = {}, {}, {}
    for dataset in DATASETS:
        path = raw / (dataset + ".states.bz2")
        with bz2.open(path, "rt") as stream:
            rows = [line.split() for line in stream]
        values = np.array([[float(row[i]) for i in (1, 2, 3)] for row in rows])
        excited = np.array([row[8] == "A(1Pi)" if dataset == "4thplus" else False
                            for row in rows])
        arrays[dataset] = np.column_stack((values, excited.astype(int)))
        table = np.loadtxt(raw / (dataset + ".pf"))
        # Preserve actual printed temperatures around each diagnostic node.
        indices = {0, len(table)-1}
        for t in TEMPERATURES:
            i = int(np.searchsorted(table[:, 0], t))
            indices.update((max(0, i-1), min(i, len(table)-1)))
        samples[dataset] = table[sorted(indices)].tolist()
        facts[dataset] = {
            "states_sha256": file_hash(path), "states": len(rows),
            "maximum_energy_cm-1": float(values[:, 0].max()),
            "maximum_J": float(values[:, 2].max()),
            "gtot_over_2Jplus1": np.unique(values[:, 1]/(2*values[:, 2]+1)).tolist(),
            "excited_states": int(excited.sum()),
            "pf_sha256": file_hash(raw / (dataset + ".pf")),
            "pf_rows": len(table), "pf_first_last_T_K": table[[0, -1], 0].tolist(),
        }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "states.npz", **arrays)
    result = {"columns": ["energy_cm-1", "gtot", "J", "is_A1Pi"],
              "datasets": facts, "published_pf_rows": samples}
    (output / "state_inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def moments(states: np.ndarray, temperature: float) -> dict:
    """Use exact Boltzmann moments, avoiding derivatives of rounded PF tables."""
    energy = states[:, 0]*C2
    population = states[:, 1]*np.exp(-energy/temperature)
    q = population.sum()
    mean = population @ energy/q
    variance = population @ (energy-mean)**2/q
    return {"Q": float(q), "mean_energy_K": float(mean),
            "variance_energy_K2": float(variance),
            "A1Pi_fraction": float(population[states[:, 3] == 1].sum()/q)}


def log_translation(temperature: float, mass: float) -> float:
    """Return the ideal-gas molecular translation factor at one bar."""
    kt = KB*temperature
    return (1.5*math.log(2*math.pi*mass*major.ATOMIC_MASS_KG*kt/PLANCK**2)
            + math.log(kt/100000))


def curated_ln_k(temperature: float) -> float:
    """Evaluate the unchanged CO column shared by all three curated versions."""
    data = json.loads(CURATED.read_text())
    lower, upper = data["temperature_range_K"]
    if not lower <= temperature <= upper:
        raise ValueError("Temperature outside curated review range")
    return -major.audit.nasa9_g_rt(temperature, {
        "coefficients": data["nasa9_reactions"]["C1O1"], "bounds_K": [lower, upper]})


def footprint() -> dict:
    """Describe saved current-model carbon/oxygen allocation, not a response."""
    with np.load(GRID) as grid:
        column = grid["species"].tolist().index("C1O1")
        n = np.exp(grid["ln_n"])
        points = {}
        for element in ("C", "O"):
            e = grid["elements"].tolist().index(element)
            shares = n[:, column]/grid["budget"][:, e]
            i = int(np.argmax(shares))
            points[element] = {
                "case_index": i, "element_fraction": float(shares[i]),
                "mole_fraction": float(np.exp(grid["ln_x"][i, column])),
                "temperature_K": float(grid["temperature_K"][i]),
                "pressure_bar": float(grid["pressure_bar"][i]),
                "C_over_O": float(grid["C_over_O"][i]),
                "maximum_relative_conservation_residual": float(np.max(
                    np.abs(n @ grid["formula_matrix"][e]/grid["budget"][:, e]-1))),
            }
        return {"model": "curated-tio-v1", "cases": len(n), "species": n.shape[1],
                "all_converged": bool(np.all(grid["converged"])),
                "maximum_abs_h_error": max(abs(h+curated_ln_k(t)) for t, h in
                    zip(grid["temperature_K"], grid["h"][:, column])),
                "maximum_element_shares": points}


def review() -> dict:
    """Separate energy, thermal, isotope and data-distribution differences."""
    reference = json.loads(major.REFERENCE_PATH.read_text())
    inventory = json.loads((DATA / "state_inventory.json").read_text())
    energy = {s: major.atomization_kelvin("CO", s, reference)
              for s in ("atct", "nasa9", "janaf")}
    record = reference["nasa9_records"]["CO"]
    mass_factor = (ISOTOPE_MASS/record["mass_u"])**1.5
    with np.load(DATA / "states.npz") as archive:
        states = {d: archive[d] for d in DATASETS}
    rows = []
    for t in TEMPERATURES:
        nasa_g = major.thermal_g_rt("CO", t, "nasa9", reference)
        h, s, cp = nasa_properties(record, t, major.NASA_R)
        nasa_h = (h-major.assigned_h0("CO", "nasa9", reference))/major.NASA_R
        nasa_s, nasa_cp = s/major.NASA_R, cp/major.NASA_R
        translation = log_translation(t, record["mass_u"])
        q_nasa = math.exp(-nasa_g-translation)
        nasa_ln_k = major.formation_log_kp("CO", t, "nasa9", energy["nasa9"], reference)
        hybrid = major.formation_log_kp("CO", t, "nasa9", energy["atct"], reference)
        sources = {}
        for dataset in DATASETS:
            state = states[dataset]
            moment = moments(state, t)
            published = np.array(inventory["published_pf_rows"][dataset])
            pf = float(np.interp(t, published[:, 0], published[:, 1]))
            state_h = 2.5*t+moment["mean_energy_K"]
            state_s = math.log(moment["Q"])+translation+state_h/t
            state_cp = 2.5+moment["variance_energy_K2"]/t**2
            high = state[:, 0]*C2 > energy["atct"]
            pop = state[:, 1]*np.exp(-state[:, 0]*C2/t)
            sources[dataset] = {
                **moment, "published_Q_interpolated_actual_T": pf,
                "states_over_published_Q": moment["Q"]/pf,
                # Sensitivity to half a printed 0.1 K digit, not a source error bar.
                "printed_T_rounding_relative_scale": 0.05*moment["mean_energy_K"]/t**2,
                "printed_Q_rounding_relative_bound": 0.00005/pf,
                "equivalent_T_offset_K": math.log(moment["Q"]/pf)*t**2/moment["mean_energy_K"],
                "Q_over_nasa_internal": moment["Q"]/q_nasa,
                "same_D0_atoms_natural_mass_K_ratio": moment["Q"]/q_nasa,
                "same_D0_atoms_isotope_mass_K_proxy": moment["Q"]/q_nasa*mass_factor,
                "H_increment_J_mol": major.R*state_h,
                "S_natural_mass_J_mol_K": major.R*state_s,
                "Cp_J_mol_K": major.R*state_cp,
                "delta_H_J_mol": major.R*(state_h-nasa_h),
                "delta_S_J_mol_K": major.R*(state_s-nasa_s),
                "delta_Cp_J_mol_K": major.R*(state_cp-nasa_cp),
                "above_adopted_D0_Q_fraction": float(pop[high].sum()/pop.sum()),
                "edge_J_ge_140_Q_fraction": float(pop[state[:, 2] >= 140].sum()/pop.sum()),
            }
        janaf = None
        if str(float(t)) in reference["janaf"]["CO"]["rows"]:
            janaf = {
                "native_ln_K": major.formation_log_kp("CO", t, "janaf", energy["janaf"], reference, "janaf"),
                "same_D0_atoms_K_ratio": math.exp(nasa_g-major.thermal_g_rt("CO", t, "janaf", reference)),
                "Cp_J_mol_K": reference["janaf"]["CO"]["rows"][str(float(t))][0],
                "native_atomic_thermal_shift_ln_K": sum(
                    major.thermal_g_rt(a, t, "janaf", reference)
                    - major.thermal_g_rt(a, t, "nasa9", reference) for a in ("C", "O")),
            }
        rows.append({"temperature_K": t, "nasa_ln_K": nasa_ln_k,
                     "curated_ln_K": curated_ln_k(t) if t >= 1500 else None,
                     "reconstructed_hybrid_ln_K": hybrid,
                     "curated_over_nasa": math.exp(hybrid-nasa_ln_k) if t >= 1500 else None,
                     "Q_nasa_natural_mass": q_nasa, "nasa_source_Cp_J_mol_K": cp,
                     "nasa_common_R_Cp_J_mol_K": major.R*nasa_cp,
                     "janaf": janaf, "modern": sources})
        if t in reference["temperature_K"]:
            i = reference["temperature_K"].index(t)
            q_bc = reference["barklem_collet"]["CO"]["partition"]["values"][i]
            rows[-1]["BC2016_Q"] = q_bc
            rows[-1]["states_over_BC2016_Q"] = sources["4thplus"]["Q"]/q_bc
    return {
        "reaction": "C(g) + O(g) -> CO(g)", "standard_pressure_Pa": 100000,
        "atct_version": reference["atct"]["version"],
        "balanced_D0_J_mol": {s: e*(major.NASA_R if s == "nasa9" else major.R) for s, e in energy.items()},
        "balanced_D0_over_k_K": energy,
        "adopted_minus_nasa_D0_over_k_K": energy["atct"]-energy["nasa9"],
        "isotope_to_natural_translation_factor": mass_factor,
        "source_samples": rows, "footprint": footprint(),
        "interpretation": "Natural-mass comparisons use 12C16O internal levels as an elemental proxy. Isotope mass is separate; neither is a complete isotope reaction. Source NASA R is preserved in dimensionless functions, including the D0/k anchor; the 218.995 J/mol raw energy difference alone does not give the hybrid shift by division by modern R. Caloric differences use common modern R. PF rounding sensitivities do not identify the cause of the residual, especially for Li2015 whose declared grid has integer temperatures.",
        "production_change": "none", "equilibrium_substitution": "not_run",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path, help="Directory of original ExoMol states and PF downloads")
    parser.add_argument("--packet", type=Path, default=DATA)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = extract(args.extract, args.packet) if args.extract else review()
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        print(serialized, end="")
