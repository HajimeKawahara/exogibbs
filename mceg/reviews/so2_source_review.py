"""Replay SO2 energy, thermal completeness and archived sulphur importance.

Offline replay uses compact energy moments. --extract verifies and resums the
original complete ExoAmes states; no transitions or network access are needed.
"""

import argparse
import ast
import bz2
import json
import math
from pathlib import Path
import re

import numpy as np

from c2h_source_review import nasa_properties
from gas_source_comparison import ROOT, file_hash
import select_major_molecules as major


DATA = ROOT / "tests/data/validation/so2_source_review/20261104"
GRID = ROOT / "tests/data/validation/curated_h2o_adoption/20261104/ions.npz"
CURATED = ROOT / "src/exogibbs/data/curated/gas_v1.json"
TEMPERATURES = (300., 500., 1000., 1500., 2000., 2500., 3000., 3500., 4000.)
R = major.R
KB = 1.380649e-23
PLANCK = 6.62607015e-34
C2 = PLANCK * 299792458 * 100 / KB
NATURAL_MASS_U = 64.0638
ISOTOPE_MASS_U = 63.96190024


def translation_log_q(temperature: float, mass_u: float) -> float:
    """Use the shared one-bar standard pressure and explicit mass convention."""
    kt = KB * temperature
    return (1.5 * math.log(2 * math.pi * mass_u * major.ATOMIC_MASS_KG * kt / PLANCK**2)
            + math.log(kt / 100000))


def moments(energy_cm: np.ndarray, weights: np.ndarray, temperature: float) -> dict:
    """Direct moments of all supplied states, including their actual weights."""
    energy = energy_cm * C2
    population = weights * np.exp(-energy / temperature)
    q = float(population.sum())
    mean = float(population @ energy / q)
    return {"Q": q, "mean_energy_K": mean,
            "variance_energy_K2": float(population @ ((energy - mean)**2) / q)}


def extract_sources(directory: Path) -> dict:
    """Resum the hash-pinned state file, diagnosing energy and rotational tails."""
    sources = json.loads((DATA / "sources.json").read_text())
    name = "exoames.states.bz2"
    expected = next(s["sha256"] for s in sources["downloads"] if s["file"] == name)
    if file_hash(directory / name) != expected:
        raise ValueError("Changed original source: " + name)
    with bz2.open(directory / name, "rt") as stream:
        energy, weights, j = np.loadtxt(stream, usecols=(1, 2, 3)).T
    if energy.min() != 0 or not np.all(weights == 2 * j + 1):
        raise ValueError("Unexpected ground zero or spin weights")
    cuts = {"E_below_8000": energy < 8000, "E_below_12000": energy < 12000,
            "E_below_14000": energy < 14000, "J_at_most_140": j <= 140,
            "J_at_most_160": j <= 160}
    samples = []
    for t in TEMPERATURES:
        full = moments(energy, weights, t)
        samples.append({"temperature_K": t, **full,
                        "cuts": {key: moments(energy[mask], weights[mask], t)
                                 for key, mask in cuts.items()}})
    return {"states": len(energy), "minimum_energy_cm-1": float(energy.min()),
            "maximum_energy_cm-1": float(energy.max()), "maximum_J": int(j.max()),
            "all_weights_equal_2J_plus_1": bool(np.all(weights == 2*j+1)),
            "ground_degeneracy": float(weights[np.argmin(energy)]),
            "cut_state_counts": {key: int(mask.sum()) for key, mask in cuts.items()},
            "samples": samples}


def molecular_properties(temperature: float, moment: dict, mass_u: float) -> dict:
    """Ground-relative ideal-gas properties; spin product is one.

    Oxygen permutation symmetry is already represented by allowed states. Do
    not divide by two again, or remove the rotational (2J+1) degeneracy.
    """
    t = temperature
    lnq = math.log(moment["Q"]) + translation_log_q(t, mass_u)
    h = R * (2.5 * t + moment["mean_energy_K"])
    g = -R * t * lnq
    return {"Grel_J_mol": g, "Hrel_J_mol": h, "S_J_mol_K": (h - g) / t,
            "Cp_J_mol_K": R * (2.5 + moment["variance_energy_K2"] / t**2)}


def extract_tips(path: Path) -> dict:
    """Extract literal SO2 tables from pinned HAPI without executing it."""
    sources = json.loads((DATA / "sources.json").read_text())
    expected = next(s["sha256"] for s in sources["downloads"] if s["file"] == "hapi.py")
    if file_hash(path) != expected:
        raise ValueError("Changed original source: hapi.py")
    text = path.read_text()
    result = {}
    for year in (2021, 2025):
        block = text.split(f"M = 9\nI = 1\nTIPS_{year}_ISOT_HASH")[1]
        grid = int(re.search(rf"TIPS_{year}_ISOT\[(\d+)\]", block).group(1))
        q = ast.literal_eval(re.search(r"float64\((\[.*?\])\)", block, re.S).group(1))
        t = ast.literal_eval(re.search(
            rf"TIPS_{year}_ISOT\[{grid}\] = float64\((\[.*?\])\)", text, re.S).group(1))
        result[str(year)] = {"temperature_K": t, "Q": q}
    return result


def tips_moment(table: dict, temperature: float, half_window: int = 5) -> dict:
    """Estimate derivatives from rounded PF values; this is not published Cp.

    A fourth-order local polynomial in log(T/T0) spans +/-50 K by default.
    The +/-100 K result records derivative/window sensitivity separately.
    """
    t = np.array(table["temperature_K"])
    q = np.array(table["Q"])
    index = int(np.flatnonzero(t == temperature)[0])
    selection = slice(index-half_window, index+half_window+1)
    coeff = np.polynomial.polynomial.polyfit(np.log(t[selection]/temperature),
                                           np.log(q[selection]), 4)
    return {"Q": float(q[index]), "mean_energy_K": float(temperature*coeff[1]),
            "variance_energy_K2": float(temperature**2*(coeff[1]+2*coeff[2]))}


def curated_ln_k(temperature: float) -> float:
    """Read the unchanged SO2 polynomial inherited by curated-h2o-v1."""
    data = json.loads(CURATED.read_text())
    lo, hi = data["temperature_range_K"]
    if not lo <= temperature <= hi:
        raise ValueError("Temperature outside the curated review domain")
    return -major.audit.nasa9_g_rt(temperature, {
        "coefficients": data["nasa9_reactions"]["O2S1"], "bounds_K": [lo, hi]})


def footprint() -> dict:
    """Read current saved endpoints; no candidate or abundance response is run."""
    with np.load(GRID) as grid:
        species = grid["species"].tolist()
        sulfur = grid["elements"].tolist().index("S")
        n = np.exp(grid["ln_n"])
        total = n @ grid["formula_matrix"][sulfur]
        carriers = {}
        for label in ("O2S1", "H2S1", "S1", "S2", "O1S1", "C1O1S1", "C1S1"):
            column = species.index(label)
            share = n[:, column] * grid["formula_matrix"][sulfur, column] / grid["budget"][:, sulfur]
            i = int(np.argmax(share))
            carriers[label] = {"maximum_sulfur_fraction": float(share[i]), "case_index": i,
                               "temperature_K": float(grid["temperature_K"][i]),
                               "pressure_bar": float(grid["pressure_bar"][i]),
                               "C_over_O": float(grid["C_over_O"][i]),
                               "mole_fraction_at_maximum_share": float(np.exp(grid["ln_x"][i, column])),
                               "maximum_mole_fraction": float(np.exp(grid["ln_x"][:, column]).max())}
        column = species.index("O2S1")
        error = [abs(grid["h"][i, column] + curated_ln_k(float(t)))
                 for i, t in enumerate(grid["temperature_K"])]
        return {"source": str(GRID.relative_to(ROOT)), "source_sha256": file_hash(GRID),
                "model": "curated-h2o-v1", "cases": len(n), "species": len(species),
                "all_converged": bool(grid["converged"].all()),
                "maximum_relative_sulfur_budget_error": float(np.max(abs(total/grid["budget"][:, sulfur]-1))),
                "maximum_abs_SO2_h_error": max(error), "sulfur_carriers": carriers}


def review() -> dict:
    """Separate balanced energy, thermal diagnostics and physical completeness."""
    reference = json.loads(major.REFERENCE_PATH.read_text())
    states = json.loads((DATA / "state_samples.json").read_text())
    tables = json.loads((DATA / "tips_tables.json").read_text())
    published = dict(np.loadtxt(DATA / "exoames.pf"))
    d0 = {s: major.atomization_kelvin("SO2", s, reference) for s in ("nasa9", "janaf", "atct")}
    rows = []
    for state in states["samples"]:
        t = state["temperature_K"]
        ng = major.thermal_g_rt("SO2", t, "nasa9", reference)
        nh, ns, ncp = nasa_properties(reference["nasa9_records"]["SO2"], t, major.NASA_R)
        nasa = {"Grel_J_mol": ng*R*t,
                "Hrel_J_mol": (nh-major.assigned_h0("SO2", "nasa9", reference))*R/major.NASA_R,
                "S_J_mol_K": ns*R/major.NASA_R, "Cp_J_mol_K": ncp*R/major.NASA_R}
        nasaq = math.exp(-ng-translation_log_q(t, NATURAL_MASS_U))
        modern = {}
        for name, moment in (("ExoAmes", state), ("TIPS2021", tips_moment(tables["2021"], t)),
                             ("TIPS2025", tips_moment(tables["2025"], t))):
            props = molecular_properties(t, moment, NATURAL_MASS_U)
            modern[name] = {"Q": moment["Q"], "Q_over_NASA": moment["Q"]/nasaq,
                            "same_D0_atoms_natural_mass_K_ratio": moment["Q"]/nasaq,
                            "same_D0_atoms_isotope_mass_K_proxy": moment["Q"]/nasaq*(ISOTOPE_MASS_U/NATURAL_MASS_U)**1.5,
                            "properties": props,
                            "delta_from_NASA": {key: props[key]-nasa[key] for key in nasa}}
            if name.startswith("TIPS"):
                wide = molecular_properties(t, tips_moment(tables[name[4:]], t, 10), NATURAL_MASS_U)
                modern[name]["Cp_50K_minus_100K_window_J_mol_K"] = props["Cp_J_mol_K"]-wide["Cp_J_mol_K"]
        cuts = {key: {"Q_fraction": moment["Q"]/state["Q"],
                      "full_minus_cut_Cp_J_mol_K": modern["ExoAmes"]["properties"]["Cp_J_mol_K"]
                      - molecular_properties(t, moment, NATURAL_MASS_U)["Cp_J_mol_K"]}
                for key, moment in state["cuts"].items()}
        row = {"temperature_K": t, "curated_ln_K": curated_ln_k(t) if t >= 1500 else None,
               "reconstructed_hybrid_ln_K": major.formation_log_kp("SO2", t, "nasa9", d0["atct"], reference),
               "native_NASA_ln_K": major.formation_log_kp("SO2", t, "nasa9", d0["nasa9"], reference),
               "curated_over_native_NASA_K": math.exp((d0["atct"]-d0["nasa9"])/t),
               "NASA_internal_Q": nasaq, "NASA_properties": nasa, "modern": modern,
               "ExoAmes_direct_over_published_Q": state["Q"]/published[t] if t in published else None,
               "ExoAmes_within_distributed_PF_range": t <= 2000, "cutoff_diagnostics": cuts}
        sq = math.exp(-major.thermal_g_rt("S", t, "nasa9", reference)
                      - translation_log_q(t, reference["nasa9_records"]["S"]["mass_u"]))
        five = moments(np.array([0., 396.055, 573.640, 9238.609, 22179.954]),
                       np.array([5., 3., 1., 5., 1.]), t)["Q"]
        row["S_atom_diagnostic"] = {"NASA_internal_Q": sq, "NIST_five_level_Q": five,
                                     "NIST_five_level_over_NASA_Q": five/sq}
        jrow = reference["janaf"]["SO2"]["rows"].get(str(float(t)))
        if jrow:
            row["JANAF"] = {"Cp_J_mol_K": jrow[0],
                "same_D0_atoms_K_over_NASA": math.exp(ng-major.thermal_g_rt("SO2", t, "janaf", reference)),
                "native_ln_K": major.formation_log_kp("SO2", t, "janaf", d0["janaf"], reference, "janaf")}
        rows.append(row)
    return {"reaction": "S(g) + 2 O(g) -> SO2(g)", "standard_pressure_Pa": 100000,
            "balanced_D0_J_mol": {s: value*(major.NASA_R if s == "nasa9" else R) for s, value in d0.items()},
            "adopted_minus_NASA_D0_over_k_K": d0["atct"]-d0["nasa9"],
            "mass_entropy_shift_J_mol_K": 1.5*R*math.log(NATURAL_MASS_U/ISOTOPE_MASS_U),
            "source_samples": rows, "footprint": footprint(), "production_change": "none",
            "equilibrium_substitution": "not_run",
            "decision": "Retain NASA thermal plus ATcT energy pending a converged high-temperature anharmonic thermal evaluation; the available modern sources do not establish a production replacement."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path, help="Directory with hash-pinned original states")
    parser.add_argument("--extract-tips", type=Path, help="Hash-pinned original hapi.py")
    parser.add_argument("--output", type=Path, help="JSON output path")
    args = parser.parse_args()
    result = (extract_sources(args.extract) if args.extract else
              extract_tips(args.extract_tips) if args.extract_tips else review())
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")
