"""Compare H2O thermochemical evaluations without changing production data.

Replay is offline. --extract accepts the hashed original source directory;
neither full papers nor the two large level lists are required for replay.
"""

import argparse
import bz2
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

from c2h_source_review import nasa_properties
from gas_source_comparison import ROOT, file_hash


sys.path.insert(0, str(ROOT / "examples/comparisons"))
import select_major_molecules as major


DATA = ROOT / "tests/data/validation/h2o_source_review/20261103"
GRID = ROOT / "tests/data/validation/curated_tio_adoption/20261103/ions.npz"
CURATED = ROOT / "src/exogibbs/data/curated/gas_v1.json"
TEMPERATURES = (298.15, 300., 500., 1000., 1500., 2000., 2500., 3000., 3500., 4000.)
R = major.R
KB = 1.380649e-23
PLANCK = 6.62607015e-34
C2 = PLANCK * 299792458 * 100 / KB
ISOTOPE_MASS_U = 18.01056468403
NATURAL_MASS_U = 18.01528
# First H + OH threshold, not the full H2O -> 2 H + O atomization energy.
FIRST_THRESHOLD_CM = 41145.92


def translation_log_q(temperature: float, mass_u: float) -> float:
    """Ideal-gas translational factor at the shared 1 bar standard pressure."""
    kt = KB * temperature
    return (1.5 * math.log(2 * math.pi * mass_u * major.ATOMIC_MASS_KG * kt / PLANCK**2)
            + math.log(kt / 100000))


def level_summary(energy_cm: np.ndarray, j: np.ndarray, spin: np.ndarray) -> dict:
    """Sum every supplied level; separately diagnose a first-threshold cut."""
    if energy_cm.min() != 0 or not set(np.unique(spin)) <= {1, 3}:
        raise ValueError("Unexpected ground zero or nuclear-spin normalization")
    energy = energy_cm * C2
    weights = spin * (2 * j + 1)
    below = energy_cm < FIRST_THRESHOLD_CM
    rows = []
    for t in TEMPERATURES:
        population = weights * np.exp(-energy / t)
        q = float(population.sum())
        mean = float(population @ energy / q)
        variance = float(population @ ((energy - mean)**2) / q)
        qcut = float(population[below].sum())
        meancut = float(population[below] @ energy[below] / qcut)
        varcut = float(population[below] @ ((energy[below] - meancut)**2) / qcut)
        rows.append({"temperature_K": t, "Q": q, "mean_energy_K": mean,
                     "variance_energy_K2": variance,
                     "ortho_over_para": float(population[spin == 3].sum() / population[spin == 1].sum()),
                     "above_first_threshold_Q_fraction": float(population[~below].sum() / q),
                     "full_minus_cut_Cp_J_mol_K": R * (variance - varcut) / t**2})
    return {"levels": len(energy), "maximum_energy_cm-1": float(energy_cm.max()),
            "maximum_J": int(j.max()), "para_ground_cm-1": float(energy_cm[spin == 1].min()),
            "ortho_ground_cm-1": float(energy_cm[spin == 3].min()),
            "para_levels": int((spin == 1).sum()), "ortho_levels": int((spin == 3).sum()),
            "above_first_threshold_levels": int((~below).sum()), "samples": rows}


def extract_sources(directory: Path) -> dict:
    """Rebuild compact state summaries and exact table rows from pinned files."""
    sources = json.loads((DATA / "sources.json").read_text())
    required = ("pokazatel.states.bz2", "PW2024.txt", "pokazatel.pf", "pokazatel.cp",
                "H216O_PartitionFunctions.csv", "H216O_ThermoFunctions.csv")
    for name in required:
        expected = next(s["sha256"] for s in sources["downloads"] if s["file"] == name)
        if file_hash(directory / name) != expected:
            raise ValueError("Changed original source: " + name)
    with bz2.open(directory / required[0], "rt") as stream:
        energy, weights, j = np.loadtxt(stream, usecols=(1, 2, 3)).T
    pokazatel = level_summary(energy, j, weights / (2 * j + 1))
    levels = np.genfromtxt(directory / "PW2024.txt", skip_header=1,
                          dtype=[("energy", float), ("j", int), ("spin", "U1"), ("uncertainty", float)])
    if not set(levels["spin"]) <= {"O", "P"}:
        raise ValueError("Unknown PW2024 ortho/para label")
    pw2024 = level_summary(levels["energy"], levels["j"], np.where(levels["spin"] == "O", 3, 1))
    pf, cp = (dict(np.loadtxt(directory / name)) for name in ("pokazatel.pf", "pokazatel.cp"))
    published = {"pokazatel": [{"temperature_K": t, "Q": pf[t], "Cp_J_mol_K": cp[t]}
                               for t in TEMPERATURES if t in pf and t in cp]}
    for label, name in (("partition_2025", "H216O_PartitionFunctions.csv"),
                        ("thermo_2025", "H216O_ThermoFunctions.csv")):
        with (directory / name).open(newline="") as stream:
            # Preserve the original headers, including the erroneous H unit label.
            published[label] = [{k: float(v) for k, v in row.items()}
                                for row in csv.DictReader(stream)
                                if float(row["T/K"]) in (*TEMPERATURES, 1.)]
    return {"state_samples.json": {"pokazatel": pokazatel, "pw2024": pw2024},
            "published_samples.json": published}


def molecular_properties(temperature: float, moment: dict, mass_u: float) -> dict:
    """G-H0, H-H0, S and Cp with overall nuclear-spin product four removed.

    State-dependent ortho/para weights remain. Permutation symmetry is already
    represented by the allowed levels: no second division by two is applied.
    """
    t = temperature
    logq = math.log(moment["Q"] / 4) + translation_log_q(t, mass_u)
    h = R * (2.5 * t + moment["mean_energy_K"])
    g = -R * t * logq
    return {"Grel_J_mol": g, "Hrel_J_mol": h, "S_J_mol_K": (h - g) / t,
            "Cp_J_mol_K": R * (2.5 + moment["variance_energy_K2"] / t**2)}


def curated_ln_k(temperature: float) -> float:
    """Read the unchanged H2O polynomial inherited by curated-tio-v1."""
    data = json.loads(CURATED.read_text())
    lo, hi = data["temperature_range_K"]
    if not lo <= temperature <= hi:
        raise ValueError("Temperature outside the curated review domain")
    return -major.audit.nasa9_g_rt(temperature, {
        "coefficients": data["nasa9_reactions"]["H2O1"], "bounds_K": [lo, hi]})


def footprint() -> dict:
    """Use the dated current-default archive, conserving oxygen across all cases."""
    with np.load(GRID) as grid:
        column = grid["species"].tolist().index("H2O1")
        oxygen = grid["elements"].tolist().index("O")
        n = np.exp(grid["ln_n"])
        shares = n[:, column] / grid["budget"][:, oxygen]
        i = int(np.argmax(shares))
        return {"source": str(GRID.relative_to(ROOT)), "source_sha256": file_hash(GRID),
                "dataset": "curated-tio-v1", "cases": len(shares), "species": len(grid["species"]),
                "all_converged": bool(grid["converged"].all()),
                "maximum_oxygen_share": {"case_index": i, "fraction": float(shares[i]),
                    "temperature_K": float(grid["temperature_K"][i]),
                    "pressure_bar": float(grid["pressure_bar"][i]),
                    "C_over_O": float(grid["C_over_O"][i]),
                    "mole_fraction": float(np.exp(grid["ln_x"][i, column]))},
                "maximum_mole_fraction": float(np.exp(grid["ln_x"][:, column]).max())}


def review() -> dict:
    """Separate legacy energy, modern thermal evidence and archived importance."""
    reference = json.loads(major.REFERENCE_PATH.read_text())
    states = json.loads((DATA / "state_samples.json").read_text())
    published = json.loads((DATA / "published_samples.json").read_text())
    energy = {s: major.atomization_kelvin("H2O", s, reference) for s in ("nasa9", "janaf", "atct")}
    rows = []
    for moment in states["pw2024"]["samples"]:
        t = moment["temperature_K"]
        pok = next(r for r in states["pokazatel"]["samples"] if r["temperature_K"] == t)
        pub = next(r for r in published["partition_2025"] if r["T/K"] == t)
        therm = next(r for r in published["thermo_2025"] if r["T/K"] == t)
        ng = major.thermal_g_rt("H2O", t, "nasa9", reference)
        nh, ns, ncp = nasa_properties(reference["nasa9_records"]["H2O"], t, major.NASA_R)
        # The runtime uses dimensionless NASA functions: express all caloric
        # comparisons with modern R while preserving native R for source D0.
        nh = (nh - major.assigned_h0("H2O", "nasa9", reference)) * R / major.NASA_R
        ns, ncp = ns * R / major.NASA_R, ncp * R / major.NASA_R
        nasa_q = math.exp(-ng - translation_log_q(t, NATURAL_MASS_U))
        direct = molecular_properties(t, moment, NATURAL_MASS_U)
        isotope = molecular_properties(t, moment, ISOTOPE_MASS_U)
        mass_entropy = 1.5 * R * math.log(NATURAL_MASS_U / ISOTOPE_MASS_U)
        # CSV H values are J/mol despite kJ/mol headers. Its S already excludes
        # the spin product. Both conventions are independently tested below.
        evaluated_g_rt = therm["H/kJ/mol"] / (R * t) - (therm["S/J/K mol"] + mass_entropy) / R
        pokpub = next((r for r in published["pokazatel"] if r["temperature_K"] == t), None)
        families = ["nasa9"]
        if str(float(t)) in reference["janaf"]["H2O"]["rows"]:
            families.append("janaf")
        raw = {s: major.formation_log_kp("H2O", t, s, energy[s], reference, s) for s in families}
        aligned = {s: major.formation_log_kp("H2O", t, s, energy["atct"], reference) for s in families}
        rows.append({"temperature_K": t, "curated_ln_K": curated_ln_k(t) if t >= 1500 else None,
                     "reconstructed_hybrid_ln_K": aligned["nasa9"],
                     "curated_over_raw_NASA": math.exp((energy["atct"] - energy["nasa9"]) / t),
                     "raw_JANAF_over_NASA": math.exp(raw["janaf"] - raw["nasa9"]) if "janaf" in raw else None,
                     "same_anchor_JANAF_over_NASA": math.exp(aligned["janaf"] - aligned["nasa9"]) if "janaf" in aligned else None,
                     "NASA_internal_Q": nasa_q,
                     "PW2024_direct_over_NASA_internal_Q": moment["Q"] / (4 * nasa_q),
                     "evaluated_2025_over_NASA_internal_Q": pub["Q"] / (4 * nasa_q),
                     "same_anchor_2025_over_current_K_from_H_S": math.exp(ng - evaluated_g_rt),
                     "PW2024_direct_over_published_Q": moment["Q"] / pub["Q"],
                     "POKAZATEL_direct_over_published_Q": None if pokpub is None else pok["Q"] / pokpub["Q"],
                     "PW2024_minus_POKAZATEL_direct_Q_relative": moment["Q"] / pok["Q"] - 1,
                     "PW2024_direct_natural_mass": direct, "PW2024_direct_isotope_mass": isotope,
                     "NASA_Hrel_J_mol": nh, "NASA_S_J_mol_K": ns, "NASA_Cp_J_mol_K": ncp,
                     "evaluated_minus_NASA_Cp_J_mol_K": therm["C_p/J/K mol"] - ncp,
                     "evaluated_minus_direct_Cp_J_mol_K": therm["C_p/J/K mol"] - direct["Cp_J_mol_K"],
                     "evaluated_Cp_expanded_uncertainty_J_mol_K": therm["U_ULB(C_p)/J/K mol"],
                     "evaluated_Q_expanded_relative_uncertainty": pub["U_ULB(Q)"] / pub["Q"]})
    return {"reaction": "2 H(g) + O(g) -> H2O(g)", "standard_pressure_Pa": 100000,
            "balanced_atomization_J_mol": {s: v * (major.NASA_R if s == "nasa9" else R)
                                           for s, v in energy.items()},
            "adopted_minus_native_NASA_D0_over_k_K": energy["atct"] - energy["nasa9"],
            "mass_entropy_shift_J_mol_K": mass_entropy, "samples": rows, "footprint": footprint(),
            "production_change": "none", "equilibrium_substitution": "not_run",
            "decision": "Retain production for this investigation; prioritize a consistent 2025 thermal candidate at the current ATcT energy anchor."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path, help="Directory containing hashed downloaded sources")
    parser.add_argument("--output", type=Path, help="Replay JSON, or output directory with --extract")
    args = parser.parse_args()
    if args.extract:
        output = args.output or DATA
        output.mkdir(parents=True, exist_ok=True)
        for name, value in extract_sources(args.extract).items():
            (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    else:
        text = json.dumps(review(), indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(text)
        else:
            print(text, end="")
