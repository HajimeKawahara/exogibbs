"""Replay the CS source audit offline; no production or equilibrium changes."""

import argparse
import bz2
import json
import math
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties
from gas_source_comparison import ROOT, file_hash
import select_major_molecules as major


DATA = ROOT / "tests/data/validation/cs_source_review/20261105"
GRID = ROOT / "tests/data/validation/curated_h2o_adoption/20261104/ions.npz"
CURATED = ROOT / "src/exogibbs/data/curated/gas_v1.json"
TEMPERATURES = (300, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000)
KB = 1.380649e-23
PLANCK = 6.62607015e-34
C2 = PLANCK * 29979245800 / KB
ISOTOPE_MASS = 43.972071


def extract(raw: Path, output: Path) -> dict:
    """Compact the original download, rejecting an unreviewed upstream change."""
    sources = json.loads((DATA / "sources.json").read_text())
    for name in ("JnK.states.bz2", "JnK.pf", "JnK.def.json"):
        if file_hash(raw / name) != sources["downloads"][name]["sha256"]:
            raise ValueError("Unreviewed original download: " + name)
    with bz2.open(raw / "JnK.states.bz2", "rt") as stream:
        states = np.array([[float(row[i]) for i in (1, 2, 3, 6)]
                           for row in map(str.split, stream)])
    pf = np.loadtxt(raw / "JnK.pf")
    definition = json.loads((raw / "JnK.def.json").read_text())
    result = {
        "columns": ["energy_cm-1", "gtot", "J", "v"],
        "number_of_states": len(states),
        "maximum_energy_cm-1": float(states[:, 0].max()),
        "maximum_J": float(states[:, 2].max()),
        "maximum_v": float(states[:, 3].max()),
        "gtot_over_2Jplus1": np.unique(states[:, 1]/(2*states[:, 2]+1)).tolist(),
        "metadata_maximum_energy_cm-1": definition["dataset"]["states"]["max_energy"],
        "metadata_maximum_temperature_K": definition["dataset"]["max_temperature"],
        "metadata_version": definition["dataset"]["version"],
        "published_pf_first_last_T_K": pf[[0, -1], 0].tolist(),
        "published_pf_samples": pf[np.isin(pf[:, 0], (1, 150)+TEMPERATURES)].tolist(),
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "states.npz", states=states)
    (output / "state_inventory.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def moments(states: np.ndarray, temperature: float) -> dict:
    """Calculate Boltzmann moments with the supplied degeneracies and zero."""
    energy = states[:, 0]*C2
    population = states[:, 1]*np.exp(-energy/temperature)
    q = population.sum()
    mean = population @ energy/q
    variance = population @ (energy-mean)**2/q
    return {"Q": float(q), "mean_energy_K": float(mean),
            "variance_energy_K2": float(variance),
            "Cp_J_mol_K": float(major.R*(2.5+variance/temperature**2))}


def log_translation(temperature: float, mass: float) -> float:
    """Return the one-bar ideal-gas translation factor."""
    kt = KB*temperature
    return (1.5*math.log(2*math.pi*mass*major.ATOMIC_MASS_KG*kt/PLANCK**2)
            + math.log(kt/100000))


def curated_ln_k(temperature: float) -> float:
    """Evaluate the unchanged CS column within its declared domain."""
    data = json.loads(CURATED.read_text())
    lower, upper = data["temperature_range_K"]
    if not lower <= temperature <= upper:
        raise ValueError("Temperature outside curated review range")
    return -major.audit.nasa9_g_rt(temperature, {
        "coefficients": data["nasa9_reactions"]["C1S1"], "bounds_K": [lower, upper]})


def electronic_scale(temperature: float, constants: list) -> dict:
    """Illustrate electronic sensitivity assuming identical rovibrational sums.

    Te is a potential minimum, not a rovibronic origin. This deliberately coarse
    model omits zero-point differences, spin-orbit structure and state-dependent
    rotation/vibration. Its result is neither a correction nor an error bound.
    """
    energy = np.array([0]+[row[1]*C2 for row in constants])
    weight = np.array([1]+[row[2] for row in constants])*np.exp(-energy/temperature)
    q = weight.sum()
    mean = weight @ energy/q
    return {"electronic_factor": float(q),
            "Cp_increment_J_mol_K": float(major.R*(weight @ (energy-mean)**2/q)/temperature**2)}


def footprint() -> dict:
    """Replay current sulfur allocation; this is not a substitution response."""
    with np.load(GRID) as grid:
        column = grid["species"].tolist().index("C1S1")
        sulfur = grid["elements"].tolist().index("S")
        n = np.exp(grid["ln_n"])
        shares = n[:, column]/grid["budget"][:, sulfur]
        i = int(np.argmax(shares))
        return {
            "model": "curated-h2o-v1", "cases": len(n), "species": n.shape[1],
            "all_converged": bool(np.all(grid["converged"])),
            "maximum_abs_h_error": max(abs(h+curated_ln_k(t)) for t, h in
                zip(grid["temperature_K"], grid["h"][:, column])),
            "maximum_sulfur_fraction": float(shares[i]), "case_index": i,
            "mole_fraction_at_maximum_sulfur_share": float(np.exp(grid["ln_x"][i, column])),
            "temperature_K": float(grid["temperature_K"][i]),
            "pressure_bar": float(grid["pressure_bar"][i]),
            "C_over_O": float(grid["C_over_O"][i]),
            "maximum_relative_sulfur_conservation_residual": float(np.max(
                np.abs(n @ grid["formula_matrix"][sulfur]/grid["budget"][:, sulfur]-1))),
        }


def review() -> dict:
    """Separate energy-version, state-sum, caloric and convention differences."""
    reference = json.loads(major.REFERENCE_PATH.read_text())
    literature = json.loads((DATA / "literature.json").read_text())
    inventory = json.loads((DATA / "state_inventory.json").read_text())
    energy = {s: major.atomization_kelvin("CS", s, reference)
              for s in ("atct", "nasa9", "janaf")}
    latest = literature["atct_1.228"]["records"]
    latest_d0 = 1000*(latest["C"][0]+latest["S"][0]-latest["CS"][0])
    record = reference["nasa9_records"]["CS"]
    pf = dict(inventory["published_pf_samples"])
    with np.load(DATA / "states.npz") as archive:
        states = archive["states"]
    rows = []
    for t in TEMPERATURES:
        nasa_g = major.thermal_g_rt("CS", t, "nasa9", reference)
        h, s, cp = nasa_properties(record, t, major.NASA_R)
        nasa_h = (h-major.assigned_h0("CS", "nasa9", reference))/major.NASA_R
        translation = log_translation(t, record["mass_u"])
        q_nasa = math.exp(-nasa_g-translation)
        nasa_ln_k = major.formation_log_kp("CS", t, "nasa9", energy["nasa9"], reference)
        hybrid = major.formation_log_kp("CS", t, "nasa9", energy["atct"], reference)
        moment = moments(states, t)
        state_h = 2.5*t+moment["mean_energy_K"]
        state_s = math.log(moment["Q"])+translation+state_h/t
        population = states[:, 1]*np.exp(-states[:, 0]*C2/t)
        cut_diagnostics = {}
        for name, keep in {
            "below_adopted_D0": states[:, 0]*C2 <= energy["atct"],
            "J_below_240": states[:, 2] < 240,
            "v_below_45": states[:, 3] < 45,
        }.items():
            cut = moments(states[keep], t)
            cut_diagnostics[name] = {
                "removed_Q_fraction": float(population[~keep].sum()/moment["Q"]),
                "full_minus_cut_Cp_J_mol_K": moment["Cp_J_mol_K"]-cut["Cp_J_mol_K"],
            }
        janaf = None
        if str(float(t)) in reference["janaf"]["CS"]["rows"]:
            janaf = {
                "native_ln_K": major.formation_log_kp("CS", t, "janaf", energy["janaf"], reference, "janaf"),
                "same_D0_atoms_thermal_K_ratio": math.exp(nasa_g-major.thermal_g_rt("CS", t, "janaf", reference)),
                "Cp_J_mol_K": reference["janaf"]["CS"]["rows"][str(float(t))][0],
            }
        rows.append({
            "temperature_K": t, "nasa_ln_K": nasa_ln_k,
            "curated_ln_K": curated_ln_k(t) if t >= 1500 else None,
            "reconstructed_hybrid_ln_K": hybrid,
            "curated_over_nasa": math.exp(hybrid-nasa_ln_k) if t >= 1500 else None,
            "atct_1.228_energy_only_K_ratio": math.exp((latest_d0/major.R-energy["atct"])/t),
            "Q_nasa_natural_mass": q_nasa, "nasa_source_Cp_J_mol_K": cp,
            "nasa_common_R_Cp_J_mol_K": cp*major.R/major.NASA_R,
            "JnK": {
                **moment, "published_Q": pf.get(t),
                "states_over_published_Q": moment["Q"]/pf[t] if t in pf else None,
                "Q_over_nasa_internal": moment["Q"]/q_nasa,
                "isotope_mass_K_proxy": moment["Q"]/q_nasa*(ISOTOPE_MASS/record["mass_u"])**1.5,
                "H_increment_J_mol": major.R*state_h,
                "S_natural_mass_J_mol_K": major.R*state_s,
                "delta_H_J_mol": major.R*(state_h-nasa_h),
                "delta_S_J_mol_K": major.R*(state_s-s/major.NASA_R),
                "delta_Cp_J_mol_K": moment["Cp_J_mol_K"]-cp*major.R/major.NASA_R,
                "cut_diagnostics": cut_diagnostics,
            },
            "identical_rovibration_electronic_scale": electronic_scale(t, literature["electronic_scale_constants"]),
            "janaf": janaf,
        })
        if t in reference["temperature_K"]:
            i = reference["temperature_K"].index(t)
            q_bc = reference["barklem_collet"]["CS"]["partition"]["values"][i]
            rows[-1]["BC2016_Q"] = q_bc
            rows[-1]["JnK_over_BC2016_Q"] = moment["Q"]/q_bc
    return {
        "reaction": "C(g) + S(g) -> CS(g)", "standard_pressure_Pa": 100000,
        "adopted_atct_version": "1.172",
        "balanced_D0_J_mol": {s: e*(major.NASA_R if s == "nasa9" else major.R) for s, e in energy.items()},
        "adopted_minus_nasa_D0_over_k_K": energy["atct"]-energy["nasa9"],
        "atct_1.228_balanced_D0_J_mol": latest_d0,
        "atct_1.228_minus_adopted_D0_J_mol": latest_d0-energy["atct"]*major.R,
        "above_adopted_D0_states": int(np.sum(states[:, 0]*C2 > energy["atct"])),
        "source_samples": rows, "footprint": footprint(),
        "production_change": "none", "equilibrium_substitution": "not_run",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path, help="Directory of original ExoMol downloads")
    parser.add_argument("--packet", type=Path, default=DATA)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = extract(args.extract, args.packet) if args.extract else review()
    serialized = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        print(serialized, end="")
