"""Review CO2 thermal sources without changing production or solving new equilibria.

The default replay is offline. ``--extract DIRECTORY`` regenerates compact
NIST tables and full ExoMol state moments from separately downloaded originals.
"""

import argparse
import bz2
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import BinaryIO

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/comparisons"))
import select_major_molecules as major


DATA = ROOT / "tests/data/validation/co2_source_review/20261103"
GRID = ROOT / "tests/data/validation/curated_tio_adoption/20261103/ions.npz"
TEMPERATURES = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000)
CODES = ("626", "627", "628", "636", "637", "638", "727", "728", "737", "738", "828", "838")
KB = 1.380649e-23
PLANCK = 6.62607015e-34
C2 = PLANCK * 299792458 * 100 / KB
MASS_626 = 43.989829239132
NIST_C2 = 1.43877736


def file_hash(path: Path) -> str:
    """Hash an original or compact artifact without loading large files at once."""
    with path.open("rb") as stream:
        return sha256_stream(stream)


def sha256_stream(stream: BinaryIO) -> str:
    result = sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(chunk)
    return result.hexdigest()


def ln_translation(temperature: float, mass_u: float) -> float:
    """Translational single-molecule factor at standard pressure 1 bar."""
    kt = KB * temperature
    return (1.5 * math.log(2 * math.pi * mass_u * major.ATOMIC_MASS_KG * kt / PLANCK**2)
            + math.log(kt / 100000))


def curated_ln_k(temperature: float) -> float:
    """Read the retained CO2 polynomial, shared by all three curated versions."""
    data = json.loads((ROOT / "src/exogibbs/data/curated/gas_v1.json").read_text())
    lo, hi = data["temperature_range_K"]
    if not lo <= temperature <= hi:
        raise ValueError("Temperature outside curated runtime interval")
    return -major.audit.nasa9_g_rt(temperature, {
        "coefficients": data["nasa9_reactions"]["C1O2"], "bounds_K": [lo, hi]})


def state_moments(path: Path) -> dict:
    """Sum every current state; transition files and their intensity cuts are unused."""
    with bz2.open(path, "rt") as stream:
        values = np.loadtxt(stream, usecols=(1, 2, 3))
    energy, weights, angular = values.T
    if energy.min() != 0 or np.any(weights != 2 * angular + 1):
        raise ValueError("Unexpected 626 ground-zero or nuclear-spin convention")
    epsilon = energy * C2
    rows = []
    for t in TEMPERATURES:
        population = weights * np.exp(-epsilon / t)
        q = float(population.sum())
        mean = float(population @ epsilon / q)
        variance = float(population @ ((epsilon - mean)**2) / q)
        rows.append({"temperature_K": t, "Q": q, "mean_energy_K": mean,
                     "variance_energy_K2": variance,
                     "J_above_202_Q_fraction": float(population[angular > 202].sum() / q),
                     "energy_above_30000_Q_fraction": float(population[energy > 30000].sum() / q)})
        if "UCL-4000" in path.name:
            restricted = {}
            for label, mask in (("J_le_202", angular <= 202), ("E_le_26000", energy <= 26000),
                                ("J_le_202_E_le_26000", (angular <= 202) & (energy <= 26000)),
                                ("E_le_30000", energy <= 30000)):
                restricted_q = float(population[mask].sum())
                restricted_mean = float(population[mask] @ epsilon[mask] / restricted_q)
                restricted_var = float(population[mask] @ ((epsilon[mask] - restricted_mean)**2) / restricted_q)
                restricted[label] = {"Q": restricted_q, "Cp_R": 2.5 + restricted_var / t**2}
            rows[-1]["restricted_states"] = restricted
    return {"file": path.name, "sha256": file_hash(path), "states": len(values),
            "J_max": int(angular.max()), "energy_min_cm-1": float(energy.min()),
            "energy_max_cm-1": float(energy.max()), "rows": rows}


def extract(directory: Path) -> dict:
    """Create a small, deterministic packet from complete official source files."""
    isotope_tables = {code: np.loadtxt(directory / ("tcfun_" + code + ".txt"), skiprows=9)
                      for code in CODES}
    natural = np.loadtxt(directory / "CO2-total.txt", skiprows=5)
    exomol = {name: np.loadtxt(directory / ("12C-16O2__" + name + ".pf"))
              for name in ("UCL-4000", "Dozen")}
    cp = np.loadtxt(directory / "12C-16O2__UCL-4000.cp")
    result = {"nist_isotopologue_columns": ["T_K", "Q", "u_Q", "Cp_R", "u_Cp_R",
                                            "S_R", "u_S_R", "H_RT", "u_H_RT"],
              "nist_natural_columns": ["T_K", "Cp_R", "u_Cp_R", "S_R", "u_S_R", "H_RT", "u_H_RT"],
              "nist_isotopologues": {}, "nist_natural": [], "exomol": {},
              "nist_626_derivative_windows": []}
    for code, values in isotope_tables.items():
        result["nist_isotopologues"][code] = [values[values[:, 0] == t][0].tolist() for t in TEMPERATURES]
    result["nist_natural"] = [natural[natural[:, 0] == t][0].tolist() for t in TEMPERATURES]
    for t in TEMPERATURES:
        values = isotope_tables["626"]
        result["nist_626_derivative_windows"].append({
            "temperature_K": t, "step_K": 10,
            "rows": [values[values[:, 0] == t + offset][0].tolist() for offset in (-20, -10, 0, 10, 20)]})
    for name, values in exomol.items():
        result["exomol"][name] = state_moments(directory / ("12C-16O2__" + name + ".states.bz2"))
        result["exomol"][name]["published_pf"] = [values[values[:, 0] == t][0].tolist() for t in TEMPERATURES]
    result["exomol"]["UCL-4000"]["published_cp"] = [cp[cp[:, 0] == t][0].tolist() for t in TEMPERATURES]
    result["source_sha256"] = {path.name: file_hash(path) for path in sorted(directory.iterdir())
                               if path.name in {"CO2-total.txt", "12C-16O2__UCL-4000.cp"}
                               or path.name.startswith("tcfun_") and path.suffix == ".txt"
                               or path.name.startswith("12C-16O2__") and path.suffix in (".pf", ".def")}
    return result


def footprint() -> dict:
    """Verify the current saved gas solution before measuring CO2 element shares."""
    with np.load(GRID) as grid:
        column = grid["species"].tolist().index("C1O2")
        n = np.exp(grid["ln_n"])
        output = {"source": str(GRID.relative_to(ROOT)), "sha256": file_hash(GRID),
                  "dataset": "curated-tio-v1", "cases": len(n), "species": len(grid["species"]),
                  "all_converged": bool(np.all(grid["converged"])), "elements": {}}
        for element in ("C", "O"):
            row = grid["elements"].tolist().index(element)
            residual = float(np.max(abs(n @ grid["formula_matrix"][row] - grid["budget"][:, row])
                                    / grid["budget"][:, row]))
            shares = n[:, column] * grid["formula_matrix"][row, column] / grid["budget"][:, row]
            case = int(np.argmax(shares))
            output["elements"][element] = {
                "max_relative_budget_residual": residual, "case_index": case,
                "temperature_K": float(grid["temperature_K"][case]),
                "pressure_bar": float(grid["pressure_bar"][case]),
                "C_over_O": float(grid["C_over_O"][case]), "fraction": float(shares[case]),
                "mole_fraction": float(np.exp(grid["ln_x"][case, column]))}
        output["max_abs_potential_difference"] = float(max(
            abs(h + curated_ln_k(float(t))) for t, h in zip(grid["temperature_K"], grid["h"][:, column])))
        return output


def nist_consistency(packet: dict) -> dict:
    """Check published tables independently of the printed variance formula."""
    oxygen = {"6": .99757, "7": .00038, "8": .00205}
    carbon = {"2": .9893, "3": .0107}
    weights = {s: oxygen[s[0]] * carbon[s[1]] * oxygen[s[2]]
               * (1 if s[0] == s[2] else 2) for s in CODES}
    average = sum(weights[s] * np.array(packet["nist_isotopologues"][s])[:, [3, 5, 7]] for s in CODES)
    natural = np.array(packet["nist_natural"])[:, [1, 3, 5]]
    checks = []
    for window in packet["nist_626_derivative_windows"]:
        t, h = window["temperature_K"], window["step_K"]
        rows = np.array(window["rows"])
        def derivative(values):
            return float(np.dot([1, -8, 0, 8, -1], values) / (12 * h))
        cp = rows[2, 3]
        h_rt = rows[2, 7]
        checks.append({"temperature_K": t,
                       "Cp_R_from_H_derivative": derivative(rows[:, 0] * rows[:, 7]),
                       "Cp_R_published": cp,
                       "Cp_R_from_S_derivative": t * derivative(rows[:, 5]),
                       "H_RT_from_Q_derivative": 2.5 + t * derivative(np.log(rows[:, 1])),
                       "H_RT_published": h_rt,
                       "printed_plus_formula_excess_Cp_R": 2 * (h_rt - 2.5)**2})
    return {"natural_isotope_weights": weights, "sum_isotope_weights": sum(weights.values()),
            "max_abs_weighted_minus_natural_Cp_S_H": np.max(abs(average - natural), axis=0).tolist(),
            "mean_log_nuclear_spin_weight": sum(weights[s] * math.log((2 if s[1] == "3" else 1)
                * (6 if s[0] == "7" else 1) * (6 if s[2] == "7" else 1)) for s in CODES),
            "derivative_checks": checks}


def review() -> dict:
    """Separate energy, thermal shape, normalization, completeness and application."""
    data = json.loads(major.REFERENCE_PATH.read_text())
    packet = json.loads((DATA / "source_nodes.json").read_text())
    energies = {name: major.atomization_kelvin("CO2", name, data) for name in ("nasa9", "janaf", "atct")}
    mass = data["nasa9_records"]["CO2"]["mass_u"]
    rows = []
    for i, t in enumerate(TEMPERATURES):
        nist = packet["nist_isotopologues"]["626"][i]
        natural = packet["nist_natural"][i]
        thermal = major.thermal_g_rt("CO2", t, "nasa9", data)
        q_nasa = math.exp(-thermal - ln_translation(t, mass))
        coefficients = major.nasa_interval("CO2", t, data)["coefficients"]
        nasa_cp_r = coefficients[0] / t**2 + coefficients[1] / t + sum(coefficients[j] * t**(j-2) for j in range(2, 7))
        raw = {"nasa9": major.formation_log_kp("CO2", t, "nasa9", energies["nasa9"], data, "nasa9")}
        aligned = {"nasa9": major.formation_log_kp("CO2", t, "nasa9", energies["atct"], data)}
        has_janaf = str(float(t)) in data["janaf"]["CO2"]["rows"]
        raw["janaf"] = major.formation_log_kp("CO2", t, "janaf", energies["janaf"], data, "janaf") if has_janaf else None
        aligned["janaf"] = major.formation_log_kp("CO2", t, "janaf", energies["atct"], data) if has_janaf else None
        exomol = {}
        for name, dataset in packet["exomol"].items():
            moments = dataset["rows"][i]
            q = moments["Q"]
            cp_r = 2.5 + moments["variance_energy_K2"] / t**2
            exomol[name] = {"Q_states": q, "Q_published": dataset["published_pf"][i][1],
                            "state_over_published_Q": q / dataset["published_pf"][i][1],
                            "state_over_NIST_626_Q": q / nist[1],
                            "state_over_NASA_Q_same_mass_proxy": q / q_nasa,
                            "Cp_R_states": cp_r, "H_RT_states": 2.5 + moments["mean_energy_K"] / t,
                            "S_R_states_at_626_mass": math.log(q) + ln_translation(t, MASS_626) + 2.5 + moments["mean_energy_K"] / t,
                            "Cp_J_mol_K_published": dataset.get("published_cp", [[0, None]] * len(TEMPERATURES))[i][1],
                            "restricted_states": moments.get("restricted_states", {}),
                            "J_above_202_Q_fraction": moments["J_above_202_Q_fraction"]}
        rows.append({"temperature_K": t, "raw_nasa_ln_K": raw["nasa9"], "raw_janaf_ln_K": raw["janaf"],
                     "curated_ln_K": curated_ln_k(t) if 1500 <= t <= 4000 else None,
                     "reconstructed_hybrid_ln_K": aligned["nasa9"],
                     "curated_over_raw_NASA": math.exp(aligned["nasa9"] - raw["nasa9"]) if 1500 <= t <= 4000 else None,
                     "same_D0_same_atoms_JANAF_over_NASA": math.exp(aligned["janaf"] - aligned["nasa9"]) if has_janaf else None,
                     "NASA_internal_Q_natural_mass_convention": q_nasa,
                     "NIST_626_over_NASA_Q_same_mass_proxy": nist[1] / q_nasa,
                     "NIST_626_source_native_Q": nist[1],
                     "NIST_626_Q_standard_uncertainty": nist[2],
                     "NASA_Cp_R": nasa_cp_r, "NIST_natural_Cp_R": natural[1],
                     "NIST_natural_Cp_R_standard_uncertainty": natural[2],
                     "NIST_natural_minus_NASA_Cp_common_R_J_mol_K": major.R * (natural[1] - nasa_cp_r),
                     "NASA_native_R_minus_common_R_Cp_J_mol_K": (major.NASA_R - major.R) * nasa_cp_r,
                     "NIST_626_Cp_R": nist[3], "NIST_626_H_RT": nist[7],
                     "NIST_constant_rescaling_first_order_delta_ln_Q": (nist[7] - 2.5) * (NIST_C2 / C2 - 1),
                     "exomol": exomol})
    return {"reaction": "C(g) + 2 O(g) -> CO2(g)", "standard_pressure_Pa": 100000,
            "balanced_D0_J_mol": {s: e * (major.NASA_R if s == "nasa9" else major.R) for s, e in energies.items()},
            "balanced_D0_over_k_K": energies,
            "adopted_minus_native_NASA_D0_over_k_K": energies["atct"] - energies["nasa9"],
            "translational_mass_factor_natural_over_626": (mass / MASS_626)**1.5,
            "source_samples": rows, "nist_consistency": nist_consistency(packet), "footprint": footprint(),
            "equilibrium_substitution": "not_run", "production_change": "none",
            "interpretation": "Same-mass internal-Q ratios isolate a representative-isotope thermal proxy with fixed ATcT energy and NASA atoms. They are neither natural-mixture replacement nor abundance ratios."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path, help="Original source directory; no network access")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = extract(args.extract) if args.extract else review()
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
