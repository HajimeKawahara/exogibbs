"""Replay S2 energy and finite-state diagnostics without changing production.

The default command is offline. --extract accepts the original official files
and regenerates the compact numerical packet after checking their hashes.
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


DATA = ROOT / "tests/data/validation/s2_source_review/20261105"
GRID = ROOT / "tests/data/validation/curated_h2o_adoption/20261104/ions.npz"
CURATED = ROOT / "src/exogibbs/data/curated/gas_v1.json"
TEMPERATURES = (300, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000)
KB = 1.380649e-23
PLANCK = 6.62607015e-34
C2 = PLANCK * 29979245800 / KB
ISOTOPE_MASS = 63.944142
RAW_LABELS = ("X(3Sigma-g)", "B(3Sigma+g)", "B''(3Pig)")


def extract(raw: Path, output: Path) -> dict:
    """Preserve every distributed energy/weight and the raw state labels."""
    sources = json.loads((DATA / "sources.json").read_text())
    for source in sources["downloads"]:
        if source["file"] in ("Gomez.states.bz2", "Gomez.pf", "q152.txt"):
            if file_hash(raw / source["file"]) != source["sha256"]:
                raise ValueError("Original source hash mismatch: " + source["file"])
    with bz2.open(raw / "Gomez.states.bz2", "rt") as stream:
        rows = [line.split() for line in stream]
    states = np.array([[float(row[i]) for i in (1, 2, 3, 7, 8)]
                       + [RAW_LABELS.index(row[6])] for row in rows])
    pf = np.loadtxt(raw / "Gomez.pf")
    hitran = np.loadtxt(raw / "q152.txt")
    if not np.array_equal(pf, hitran):
        raise ValueError("Published Gomez and HITRAN tables differ")
    inventory = {
        "columns": ["energy_cm-1", "gtot", "J", "v", "N", "raw_label_index"],
        "raw_labels": list(RAW_LABELS), "states": len(states),
        "maximum_energy_cm-1": float(states[:, 0].max()),
        "maximum_J": float(states[:, 2].max()),
        "gtot_over_2Jplus1": np.unique(states[:, 1]/(2*states[:, 2]+1)).tolist(),
        "by_raw_label": {
            label: {
                "states": int(np.count_nonzero(states[:, 5] == i)),
                "maximum_v": int(states[states[:, 5] == i, 3].max()),
                "minimum_energy_cm-1": float(states[states[:, 5] == i, 0].min()),
                "N_parities": np.unique(states[states[:, 5] == i, 4] % 2).tolist(),
            } for i, label in enumerate(RAW_LABELS)},
        "published_pf": {"rows": len(pf), "first_last_T_K": pf[[0, -1], 0].tolist(),
                         "Gomez_equals_HITRAN_every_row": True},
        "label_note": "The distributed excited-state g/u and Sigma sign labels disagree with Gomez 2024's B3Sigma-u and B''3Piu names. Raw labels, energies and weights are preserved; no relabeling changes the sum.",
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "states.npz", states=states, published_pf=pf)
    (output / "state_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return inventory


def moments(states: np.ndarray, temperature: float) -> dict:
    """Compute Q and caloric moments with already selected permutation states."""
    energy = C2*states[:, 0]
    population = states[:, 1]*np.exp(-energy/temperature)
    q = population.sum()
    mean = population @ energy/q
    variance = population @ (energy-mean)**2/q
    return {"Q": float(q), "mean_energy_K": float(mean),
            "variance_energy_K2": float(variance),
            "X_v10_Q_fraction": float(population[(states[:, 5] == 0)
                                                & (states[:, 3] == 10)].sum()/q),
            "J_ge_145_Q_fraction": float(population[states[:, 2] >= 145].sum()/q),
            "B_and_Bpp_Q_fraction": float(population[states[:, 5] != 0].sum()/q)}


def log_translation(temperature: float, mass: float) -> float:
    """Return ideal-gas translation at the shared one-bar standard pressure."""
    kt = KB*temperature
    return (1.5*math.log(2*math.pi*mass*major.ATOMIC_MASS_KG*kt/PLANCK**2)
            + math.log(kt/100000))


def properties(moment: dict, temperature: float, mass: float) -> dict:
    """Return molecular thermal functions relative to the occupied ground state."""
    h = 2.5*temperature+moment["mean_energy_K"]
    s = math.log(moment["Q"])+log_translation(temperature, mass)+h/temperature
    cp = 2.5+moment["variance_energy_K2"]/temperature**2
    return {"H_increment_J_mol": major.R*h, "S_J_mol_K": major.R*s,
            "Cp_J_mol_K": major.R*cp, "G_minus_H0_J_mol": major.R*(h-temperature*s)}


def curated_ln_k(temperature: float) -> float:
    """Evaluate the unchanged S2 column shared by all curated revisions."""
    data = json.loads(CURATED.read_text())
    lower, upper = data["temperature_range_K"]
    if not lower <= temperature <= upper:
        raise ValueError("Temperature outside curated review range")
    return -major.audit.nasa9_g_rt(temperature, {
        "coefficients": data["nasa9_reactions"]["S2"], "bounds_K": [lower, upper]})


def footprint() -> dict:
    """Describe existing S2 allocation; this is not a finite substitution."""
    with np.load(GRID) as grid:
        j = grid["species"].tolist().index("S2")
        e = grid["elements"].tolist().index("S")
        n = np.exp(grid["ln_n"])
        shares = 2*n[:, j]/grid["budget"][:, e]
        i = int(np.argmax(shares))
        return {
            "model": "curated-h2o-v1", "cases": len(n), "species": n.shape[1],
            "all_converged": bool(np.all(grid["converged"])),
            "maximum_abs_h_error": max(abs(h+curated_ln_k(t)) for t, h in
                zip(grid["temperature_K"], grid["h"][:, j])),
            "maximum_relative_sulfur_residual": float(np.max(
                np.abs(n @ grid["formula_matrix"][e]/grid["budget"][:, e]-1))),
            "maximum_sulfur_fraction": float(shares[i]), "case_index": i,
            "mole_fraction_at_maximum_share": float(np.exp(grid["ln_x"][i, j])),
            "maximum_mole_fraction": float(np.exp(grid["ln_x"][:, j]).max()),
            "temperature_K": float(grid["temperature_K"][i]),
            "pressure_bar": float(grid["pressure_bar"][i]),
            "C_over_O": float(grid["C_over_O"][i]),
        }


def review() -> dict:
    """Separate balanced energy changes from incomplete UV thermal coverage."""
    reference = json.loads(major.REFERENCE_PATH.read_text())
    literature = json.loads((DATA / "literature.json").read_text())
    energy = {s: major.atomization_kelvin("S2", s, reference)
              for s in ("atct", "nasa9", "janaf")}
    latest = literature["ATcT_1.228"]["records"]
    latest_d0 = 1000*(2*latest["S"]["hf0_kJ_mol"]-latest["S2"]["hf0_kJ_mol"])
    record = reference["nasa9_records"]["S2"]
    with np.load(DATA / "states.npz") as archive:
        states, pf = archive["states"], archive["published_pf"]
    rows = []
    for t in TEMPERATURES:
        nasa_g = major.thermal_g_rt("S2", t, "nasa9", reference)
        h, s, cp = nasa_properties(record, t, major.NASA_R)
        nasa = {"H_increment_J_mol": major.R*(h-major.assigned_h0("S2", "nasa9", reference))/major.NASA_R,
                "S_J_mol_K": major.R*s/major.NASA_R,
                "Cp_J_mol_K": major.R*cp/major.NASA_R,
                "G_minus_H0_J_mol": major.R*t*nasa_g}
        q_nasa = math.exp(-nasa_g-log_translation(t, record["mass_u"]))
        moment = moments(states, t)
        modern = properties(moment, t, record["mass_u"])
        ratio = moment["Q"]/q_nasa
        hybrid = major.formation_log_kp("S2", t, "nasa9", energy["atct"], reference)
        nasa_ln_k = major.formation_log_kp("S2", t, "nasa9", energy["nasa9"], reference)
        pop = states[:, 1]*np.exp(-C2*states[:, 0]/t)
        row = {
            "temperature_K": t, "nasa_ln_K": nasa_ln_k,
            "reconstructed_hybrid_ln_K": hybrid,
            "curated_ln_K": curated_ln_k(t) if t >= 1500 else None,
            "curated_over_nasa": math.exp(hybrid-nasa_ln_k) if t >= 1500 else None,
            "latest_energy_only_over_current": math.exp((latest_d0/major.R-energy["atct"])/t),
            "Q_nasa_natural_mass": q_nasa, "nasa_common_R": nasa,
            "Gomez": {**moment, **modern},
            "published_Q": float(pf[pf[:, 0] == t, 1][0]),
            "states_over_published_Q": moment["Q"]/float(pf[pf[:, 0] == t, 1][0]),
            "same_D0_atoms_natural_mass_K_ratio": ratio,
            "same_D0_atoms_isotope_mass_K_proxy": ratio*(ISOTOPE_MASS/record["mass_u"])**1.5,
            "delta_Gomez_minus_nasa": {key: modern[key]-nasa[key] for key in nasa},
            "above_adopted_D0_Q_fraction": float(pop[C2*states[:, 0] > energy["atct"]].sum()/pop.sum()),
        }
        if str(float(t)) in reference["janaf"]["S2"]["rows"]:
            row["janaf"] = {
                "native_ln_K": major.formation_log_kp("S2", t, "janaf", energy["janaf"], reference, "janaf"),
                "same_D0_atoms_K_ratio": math.exp(nasa_g-major.thermal_g_rt("S2", t, "janaf", reference)),
                "Cp_J_mol_K": reference["janaf"]["S2"]["rows"][str(float(t))][0],
                "native_atomic_thermal_shift_ln_K": 2*(major.thermal_g_rt("S", t, "janaf", reference)
                                                        - major.thermal_g_rt("S", t, "nasa9", reference)),
            }
        if t in reference["temperature_K"]:
            i = reference["temperature_K"].index(t)
            row["BC2016_Q"] = reference["barklem_collet"]["S2"]["partition"]["values"][i]
            row["BC2016_over_nasa_Q"] = row["BC2016_Q"]/q_nasa
        rows.append(row)
    d0 = {s: e*(major.NASA_R if s == "nasa9" else major.R) for s, e in energy.items()}
    return {
        "reaction": "2S(g) -> S2(g)", "standard_pressure_Pa": 100000,
        "atct_version": reference["atct"]["version"], "balanced_D0_J_mol": d0,
        "balanced_D0_over_k_K": energy,
        "adopted_minus_nasa_D0_over_k_K": energy["atct"]-energy["nasa9"],
        "latest_ATcT_balanced_D0_J_mol": latest_d0,
        "latest_minus_current_D0_J_mol": latest_d0-d0["atct"],
        "ATcT_D0_cm-1": d0["atct"]/(major.R*C2),
        "source_samples": rows, "footprint": footprint(),
        "interpretation": "Nuclear spin product is one. Allowed N parity and triplet components are already in the state list, so no extra symmetry division or spin factor is applied. Molecular comparison uses 32S2 internal states with natural translation as an elemental proxy; it is not a complete isotope reaction. NASA dimensionless functions retain native R=8.31451. Shell populations and missing electronic manifolds diagnose limited coverage, not the exact error in NASA. Latest energy differences use rounded same-version ATcT rows, not propagated covariance.",
        "production_change": "none", "equilibrium_substitution": "not_run",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path)
    parser.add_argument("--packet", type=Path, default=DATA)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = extract(args.extract, args.packet) if args.extract else review()
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        print(serialized, end="")
