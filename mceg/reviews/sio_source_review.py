"""Replay SiO energy selection, normalized thermal comparisons and a saved grid.

No network access, production change or new equilibrium solve is performed.
The optional state-moment helper accepts an independently downloaded ExoMol file.
"""

import argparse
import bz2
import json
import math
from pathlib import Path
import sys

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import BASELINE, ROOT, file_hash


sys.path.insert(0, str(ROOT / "examples/comparisons"))
import select_major_molecules as major


DATA = ROOT / "tests/data/validation/sio_source_review/20261103"
GRID = ROOT / "tests/data/validation/curated_default_adoption/20261102/ions.npz"
CURATED = ROOT / "src/exogibbs/data/curated/gas_v1.json"
KB = 1.380649e-23
PLANCK = 6.62607015e-34
C2 = PLANCK * 299792458 * 100 / KB
TEMPERATURES = (500, 1000, 1500, 2000, 3000, 4000)


def state_moments(path: Path) -> list:
    """Summarize all states using their tabulated ground-state-relative energies.

    Regeneration requires the hashed SiOUVenIR states file, not transitions.
    These summaries are frozen separately from the published PF and Cp values.
    """
    with bz2.open(path, "rt") as stream:
        fields = [line.split() for line in stream]
    energy = np.array([float(row[1]) for row in fields]) * C2
    weights = np.array([int(row[2]) for row in fields])
    j = np.array([int(row[3]) for row in fields])
    excited = np.array([row[9] != "X(1SIGMA+)" for row in fields])
    if len(fields) != 174250 or np.any(weights != 2*j+1) or energy.min() != 0:
        raise ValueError("Unexpected SiOUVenIR state normalization")
    rows = []
    for t in TEMPERATURES:
        population = weights * np.exp(-energy/t)
        q = float(population.sum())
        mean = float(population @ energy / q)
        variance = float(population @ ((energy-mean)**2) / q)
        rows.append({"temperature_K": t, "Q_states": q,
                     "mean_energy_K": mean, "variance_energy_K2": variance,
                     "excited_state_fraction": float(population[excited].sum()/q)})
    return rows


def curated_ln_k(temperature: float) -> float:
    """Read the actual current SiO reaction polynomial inside its runtime domain."""
    data = json.loads(CURATED.read_text())
    lo, hi = data["temperature_range_K"]
    if not lo <= temperature <= hi:
        raise ValueError("Temperature outside curated review range")
    coefficients = data["nasa9_reactions"]["O1Si1"]
    return -major.audit.nasa9_g_rt(temperature, {"coefficients": coefficients,
                                               "bounds_K": [lo, hi]})


def internal_q(temperature: float, thermal_g_rt: float, mass_u: float) -> float:
    """Remove ideal-gas translation at 1 bar, without adding a zero-point shift."""
    kt = KB*temperature
    ln_translation = (1.5*math.log(2*math.pi*mass_u*major.ATOMIC_MASS_KG*kt/PLANCK**2)
                      + math.log(kt/100000))
    return math.exp(-thermal_g_rt-ln_translation)


def footprint() -> dict:
    """Use conserved silicon, not the abundance of an individual atomic species."""
    with np.load(GRID) as grid:
        column = grid["species"].tolist().index("O1Si1")
        silicon = grid["elements"].tolist().index("Si")
        n = np.exp(grid["ln_n"])
        shares = grid["formula_matrix"][silicon, column]*n[:, column]/grid["budget"][:, silicon]
        case = int(np.argmax(shares))
        return {
            "source": str(GRID.relative_to(ROOT)), "source_sha256": file_hash(GRID),
            "model": "curated-c2h-v1", "cases": len(shares),
            "species_count": len(grid["species"]),
            "all_cases_converged": bool(np.all(grid["converged"])),
            "maximum_silicon_share": {
                "case_index": case, "temperature_K": float(grid["temperature_K"][case]),
                "pressure_bar": float(grid["pressure_bar"][case]),
                "C_over_O": float(grid["C_over_O"][case]),
                "mole_fraction": float(np.exp(grid["ln_x"][case, column])),
                "silicon_fraction": float(shares[case]),
            },
            "interpretation": "Hydrogen-rich ideal-gas archive; not a lava-atmosphere calculation or substitution response.",
        }


def review() -> dict:
    """Keep balanced energy, molecular thermal comparison and application distinct."""
    data = json.loads(major.REFERENCE_PATH.read_text())
    tables = {n: read_table(DATA/f) for n, f in
              (("SiO", "O-012.txt"), ("Si", "Si-005.txt"), ("O", "O-001.txt"))}
    # Fresh official tables are compared to the original adoption inputs in tests.
    for name, table in tables.items():
        data["janaf"][name]["rows"] = {str(float(t)): values for t, values in table.items()}
    energies = {source: major.atomization_kelvin("SiO", source, data)
                for source in ("nasa9", "janaf", "atct")}
    baseline = json.loads((BASELINE/"comparison.json").read_text())
    moments = json.loads((DATA/"partition_samples.json").read_text())
    rows = []
    for moment in moments:
        t = moment["temperature_K"]
        raw = {s: major.formation_log_kp("SiO", t, s, energies[s], data, s)
               for s in ("nasa9", "janaf")}
        aligned = {s: major.formation_log_kp("SiO", t, s, energies["atct"], data)
                   for s in ("nasa9", "janaf")}
        thermal = major.thermal_g_rt("SiO", t, "nasa9", data)
        q_nasa = internal_q(t, thermal, data["nasa9_records"]["SiO"]["mass_u"])
        _, _, nasa_cp = nasa_properties(data["nasa9_records"]["SiO"], t, major.NASA_R)
        cp_states = major.R*(2.5+moment["variance_energy_K2"]/t**2)
        old = next((r for r in baseline["samples"]
                    if r["species_id"] == "O1Si1" and r["temperature_K"] == t), None)
        closure = None if old is None else raw["janaf"]-raw["nasa9"]-(
            old["ln_Kp"]["janaf"]-old["ln_Kp"]["nasa"])
        rows.append({
            "temperature_K": t, "nasa_ln_K": raw["nasa9"], "janaf_ln_K": raw["janaf"],
            "curated_ln_K": curated_ln_k(t) if 1500 <= t <= 4000 else None,
            "reconstructed_hybrid_ln_K": aligned["nasa9"],
            "curated_over_nasa": math.exp(aligned["nasa9"]-raw["nasa9"]) if t >= 1500 else None,
            "janaf_over_nasa": math.exp(raw["janaf"]-raw["nasa9"]),
            "same_D0_same_atoms_janaf_over_nasa": math.exp(aligned["janaf"]-aligned["nasa9"]),
            "baseline_closure_ln_K": closure,
            "Q_nasa_natural_mass_convention": q_nasa,
            "Q_exomol_28Si16O": moment["Q_published"],
            "exomol_over_nasa_internal_Q": moment["Q_published"]/q_nasa,
            "Q_states_over_published": moment["Q_states"]/moment["Q_published"],
            "excited_state_fraction": moment["excited_state_fraction"],
            "Cp_states_J_mol_K": cp_states, "Cp_nasa_J_mol_K": nasa_cp,
            "Cp_exomol_published_J_mol_K": moment["Cp_published_J_mol_K"],
        })
    return {
        "reaction": "Si(g) + O(g) -> SiO(g)", "standard_pressure_Pa": 100000,
        "balanced_D0_J_mol": {s: e*(major.NASA_R if s == "nasa9" else major.R)
                               for s, e in energies.items()},
        "balanced_D0_over_k_K": energies,
        "adopted_minus_native_nasa_D0_over_k_K": energies["atct"]-energies["nasa9"],
        "source_samples": rows, "footprint": footprint(),
        "normalization": "Ground rovibronic zero, nuclear-spin factor1, translation removed at1bar. ExoMol28Si16O and NASA natural-mass internal isotope effects remain distinct.",
        "energy_convention": "NASA source energies use R=8.314510; ATcT/JANAF use modern R. D0 includes molecular ZPE; De is not substituted.",
        "equilibrium_substitution": "not_run", "production_change": "none",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--states", type=Path, help="Summarize the original compressed SiOUVenIR states")
    arguments = parser.parse_args()
    result = state_moments(arguments.states) if arguments.states else review()
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.write_text(text)
    else:
        print(text, end="")
