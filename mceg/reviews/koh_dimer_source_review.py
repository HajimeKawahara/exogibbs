"""Replay the neutral potassium hydroxide dimer source review offline.

The vibrational and energy-alignment calculations are diagnostics, not adopted
thermochemical models. The abundance footprint reuses an existing solved grid.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from c2h_source_review import nasa_properties, read_table
from gas_source_comparison import janaf_g, nasa_g


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data/validation/koh_dimer_source_review/20261103"
BASELINE = ROOT / "tests/data/validation/gas_source_comparison/baseline-001"
GRID = ROOT / "tests/data/validation/curated_default_adoption/20261102/ions.npz"
FASTCHEM = ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat"
ATOMS = {"H": "H1", "K": "K1", "O": "O1"}


def current_ln_k(temperature: float) -> float:
    """Evaluate the unchanged inherited fit only in the declared review domain."""
    if not 1500 <= temperature <= 4000:
        raise ValueError("outside the declared review range")
    lines = FASTCHEM.read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("H2K2O2 "))
    a, b, c, d, e = map(float, lines[index + 1].split())
    t = temperature
    return float(a / t + b * np.log(t) + c + d * t + e * t**2)


def vibrational_properties(frequencies: list, temperature: float, r: float) -> tuple:
    """Return harmonic vibrational entropy and Cp, with each mode counted once."""
    theta = 6.62607015e-34 * 299792458 * 100 * np.asarray(frequencies) / 1.380649e-23
    x = theta / temperature
    em = np.exp(-x)
    denominator = -np.expm1(-x)
    entropy = r * np.sum(x * em / denominator - np.log(denominator))
    cp = r * np.sum(x**2 * em / denominator**2)
    return float(entropy), float(cp)


def footprint() -> dict:
    """Summarize archived solved endpoints without a new equilibrium solve."""
    with np.load(GRID) as grid:
        species, elements = grid["species"].tolist(), grid["elements"].tolist()
        column, potassium = species.index("H2K2O2"), elements.index("K")
        shares = (grid["formula_matrix"][potassium, column] * np.exp(grid["ln_n"][:, column])
                  / grid["budget"][:, potassium])
        case = int(np.argmax(shares))
        return {
            "source": str(GRID.relative_to(ROOT)),
            "source_sha256": sha256(GRID.read_bytes()).hexdigest(),
            "model": "curated-c2h-v1", "cases": len(shares),
            "species_count": len(species), "all_cases_converged": bool(np.all(grid["converged"])),
            "maximum_potassium_share": {
                "case_index": case, "temperature_K": float(grid["temperature_K"][case]),
                "pressure_bar": float(grid["pressure_bar"][case]),
                "C_over_O": float(grid["C_over_O"][case]),
                "mole_fraction": float(np.exp(grid["ln_x"][case, column])),
                "potassium_fraction": float(shares[case]),
            },
            "interpretation": "Finite ideal-gas model footprint, not observed abundance or a substitution response.",
        }


def review() -> dict:
    """Separate molecular anchor, thermal functions, atomic references and association."""
    inputs = json.loads((DATA / "inputs.json").read_text())
    baseline = json.loads((BASELINE / "inputs.json").read_text())
    screening = json.loads((BASELINE / "comparison.json").read_text())
    table = read_table(DATA / "H-054.txt")
    r = baseline["gas_constant_J_mol_K"]
    molecule = baseline["nasa9_records"]["K2O2H2"]
    nh0, _, _ = nasa_properties(molecule, 298.15, r)
    jh0 = 1000 * table[298.15][4]
    samples = []
    for t in inputs["temperatures_K"]:
        node = table[t]
        nh, ns, ncp = nasa_properties(molecule, t, r)
        jh, js = jh0 + 1000 * node[3], node[1]
        molecular = {"anchor": jh0 - nh0,
                     "enthalpy_increment": 1000 * node[3] - (nh - nh0),
                     "entropy": -t * (js - ns)}
        atomic_g = -2 * sum((janaf_g(baseline["janaf_records"][jname], t, r)
                            - nasa_g(baseline["nasa9_records"][name], t)) * r * t
                           for name, jname in ATOMS.items())
        atomic_anchor = -2 * sum(1000 * baseline["janaf_records"][jname]["hf298_kJ_mol"]
                                - nasa_properties(baseline["nasa9_records"][name], 298.15, r)[0]
                                for name, jname in ATOMS.items())
        delta_g = sum(molecular.values()) + atomic_g
        old = next(row for row in screening["samples"]
                   if row["species_id"] == "H2K2O2" and row["temperature_K"] == t)
        expected = old["ln_Kp"]["janaf"] - old["ln_Kp"]["nasa"]
        jmono = janaf_g(baseline["janaf_records"]["H1K1O1"], t, r)
        nmono = nasa_g(baseline["nasa9_records"]["KOH"], t)
        jdimer = (jh - t * js) / (r * t)
        ndimer = (nh - t * ns) / (r * t)
        curated = old["ln_Kp"]["curated"]
        samples.append({
            "temperature_K": t, "molecular_JANAF_minus_NASA_G_components_J_mol": molecular,
            "atomic_contribution_J_mol": atomic_g,
            "reaction_JANAF_minus_NASA_G_J_mol": delta_g,
            "reaction_anchor_difference_J_mol": molecular["anchor"] + atomic_anchor,
            "janaf_over_nasa": float(np.exp(-delta_g / (r * t))),
            "energy_aligned_janaf_over_nasa": float(np.exp(
                -(delta_g - molecular["anchor"] - atomic_anchor) / (r * t))),
            "curated_over_nasa": None if curated is None else float(np.exp(curated - old["ln_Kp"]["nasa"])),
            "current_fit_minus_JANAF_ln_K": None if curated is None else current_ln_k(t) - old["ln_Kp"]["janaf"],
            "baseline_closure_ln_K": -delta_g / (r * t) - expected,
            "dimer_Cp_J_mol_K": {"janaf": node[0], "nasa": ncp},
            "dimer_S_J_mol_K": {"janaf": js, "nasa": ns},
            "association_ln_K": {"janaf": -jdimer + 2 * jmono, "nasa": -ndimer + 2 * nmono},
            "association_JANAF_over_NASA": float(np.exp(-jdimer + 2 * jmono + ndimer - 2 * nmono)),
        })
    printed_checks = []
    constants = inputs["gurvich1997"]["table_24"]
    historical = inputs["janaf1974"]
    rotation_delta_s = r * (
        0.5 * np.log(constants["adopted_inertia_product_1e-113_g3_cm6"]
                     / historical["inertia_product_1e-113_g3_cm6"])
        - np.log(constants["adopted_symmetry_number"] / historical["symmetry_number"]))
    for row in inputs["gurvich1997"]["table_28"]:
        t = row["temperature_K"]
        h, s, cp = nasa_properties(molecule, t, r)
        printed_h = 1000 * (inputs["gurvich1997"]["hf298_kJ_mol"] + row["H_minus_H0_kJ_mol"] - 22.385)
        new = vibrational_properties(constants["adopted_frequencies_cm-1"], t, r)
        old = vibrational_properties(constants["janaf1985_frequencies_cm-1"], t, r)
        printed_checks.append({
            "temperature_K": t, "NASA_minus_printed_H_J_mol": h - printed_h,
            "NASA_minus_printed_S_J_mol_K": s - row["S_J_mol_K"],
            "NASA_minus_printed_Cp_J_mol_K": cp - row["Cp_J_mol_K"],
            "harmonic_new_minus_old_Cp_J_mol_K": new[1] - old[1],
            "harmonic_new_minus_old_S_J_mol_K": new[0] - old[0],
            "RRHO_new_minus_old_S_J_mol_K": float(new[0] - old[0] + rotation_delta_s),
            "printed_new_minus_JANAF_S_J_mol_K": row["S_J_mol_K"] - table[t][1],
            "printed_new_minus_JANAF_Cp_J_mol_K": row["Cp_J_mol_K"] - table[t][0],
        })
    return {
        "reaction": "2 K(g) + 2 O(g) + 2 H(g) -> (KOH)2(g)",
        "association_reaction": "2 KOH(g) -> (KOH)2(g)",
        "standard_pressure_Pa": 100000, "gas_constant_J_mol_K": r,
        "delta_convention": "JANAF minus NASA; atom contributions subtracted twice",
        "energy_alignment": "Reaction H298 diagnostic only; not an adopted model",
        "samples": samples, "gurvich1997_replay": printed_checks,
        "rotation_new_minus_old_S_J_mol_K": float(rotation_delta_s),
        "harmonic_diagnostic": "Twelve mode harmonic comparison plus classical nonlinear rotor entropy difference; common mass/pressure, no low-T quantum corrections or anharmonicity.",
        "footprint": footprint(), "equilibrium_substitution": "not_run", "production_change": "none",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    text = json.dumps(review(), indent=2, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.write_text(text)
    else:
        print(text, end="")
