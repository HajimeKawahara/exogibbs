"""Replay the C3H provenance, caloric comparison and archived carbon footprint.

This is an offline review, not a new thermochemical provider. The Burcat
comparison is conditional on identifying the inherited species with linear C3H.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

from c2h_source_review import nasa_properties


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data/validation/c3h_source_review/20261102"
BASELINE = ROOT / "tests/data/validation/gas_source_comparison/baseline-001/inputs.json"
GRID = ROOT / "tests/data/validation/curated_default_adoption/20261102/ions.npz"
FASTCHEM = ROOT / "src/exogibbs/data/FastChem4/logK/logK.dat"


def fastchem_coefficients() -> list:
    """Read the unchanged C3H1 record, without inferring its isomer from formula."""
    lines = FASTCHEM.read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("C3H1 "))
    return [float(value) for value in lines[index + 1].split()]


def burcat_record() -> dict:
    """Read the nine coefficients per interval from the inspected PDF excerpt."""
    lines = (DATA / "burcat_c3h_nasa9.txt").read_text().splitlines()
    intervals = []
    for i in (2, 5, 8):
        bounds = [float(value) for value in lines[i].split()[:2]]
        values = [float(value.replace("D", "E")) for value in re.findall(
            r"[+-]?\d+\.\d+D[+-]\d+", " ".join(lines[i + 1:i + 3]))]
        if len(values) != 10 or values[7] != 0:
            raise ValueError("Unexpected printed NASA9 record layout")
        intervals.append({"bounds_K": bounds, "coefficients": values[:7] + values[8:]})
    return {"intervals": intervals}


def tsuji_ln_k(temperature: float, coefficients: list) -> float:
    """Convert printed log10 dissociation Kp in (dyn/cm2)^3 to formation K at 1 bar."""
    if not 1500 <= temperature <= 4000:
        raise ValueError("outside the declared review range")
    theta = 5040 / temperature
    log10_dissociation = sum(value * theta**i for i, value in enumerate(coefficients))
    # Four atoms -> one molecule; 1 bar = 10^6 dyn/cm2.
    return float(np.log(10) * (3 * 6 - log10_dissociation))


def fit_properties(temperature: float, coefficients: list, r: float) -> tuple:
    """Return ln K and reaction H/Cp using analytical van't Hoff derivatives."""
    if not 1500 <= temperature <= 4000:
        raise ValueError("outside the declared review range")
    a, b, c, d, e = coefficients
    t = temperature
    ln_k = a / t + b * np.log(t) + c + d * t + e * t**2
    enthalpy = r * (-a + b * t + d * t**2 + 2 * e * t**3)
    cp = r * (b + 2 * d * t + 6 * e * t**2)
    return float(ln_k), float(enthalpy), float(cp)


def footprint() -> dict:
    """Summarize existing solved endpoints; do not run a new equilibrium solve."""
    with np.load(GRID) as grid:
        species = grid["species"].tolist()
        elements = grid["elements"].tolist()
        column, carbon = species.index("C3H1"), elements.index("C")
        stoichiometry = grid["formula_matrix"][carbon, column]
        shares = stoichiometry * np.exp(grid["ln_n"][:, column]) / grid["budget"][:, carbon]
        case = int(np.argmax(shares))
        return {
            "source": str(GRID.relative_to(ROOT)),
            "source_sha256": sha256(GRID.read_bytes()).hexdigest(),
            "model": "curated-c2h-v1", "cases": len(shares),
            "species_count": len(species), "all_cases_converged": bool(np.all(grid["converged"])),
            "maximum_carbon_share": {
                "case_index": case, "temperature_K": float(grid["temperature_K"][case]),
                "pressure_bar": float(grid["pressure_bar"][case]),
                "C_over_O": float(grid["C_over_O"][case]),
                "mole_fraction": float(np.exp(grid["ln_x"][case, column])),
                "carbon_fraction": float(shares[case]),
            },
            "interpretation": "Finite ideal-gas model footprint, not observed abundance or a substitution response.",
        }


def review() -> dict:
    """Return the reproducible source comparison without modifying production data."""
    inputs = json.loads((DATA / "inputs.json").read_text())
    baseline = json.loads(BASELINE.read_text())
    r = baseline["gas_constant_J_mol_K"]
    fit, candidate = fastchem_coefficients(), burcat_record()
    samples = []
    for t in inputs["temperatures_K"]:
        ln_k, reaction_h, reaction_cp = fit_properties(t, fit, r)
        printed = tsuji_ln_k(t, inputs["tsuji1973"]["coefficients"])
        h, _, cp = nasa_properties(candidate, t, r)
        c_atom = nasa_properties(baseline["nasa9_records"]["C"], t, r)
        h_atom = nasa_properties(baseline["nasa9_records"]["H"], t, r)
        atomic_h = 3 * c_atom[0] + h_atom[0]
        atomic_cp = 3 * c_atom[2] + h_atom[2]
        samples.append({
            "temperature_K": t, "current_ln_K": ln_k, "tsuji_ln_K_1bar": printed,
            "current_minus_tsuji_ln_K": ln_k - printed,
            "printed_coefficient_rounding_bound_ln_K": float(np.log(10) * sum(
                value * (5040 / t)**i for i, value in enumerate(
                    inputs["tsuji1973"]["half_last_digit_bounds"]))),
            "current_reaction_H_J_mol": reaction_h,
            "burcat_with_nasa_atoms_reaction_H_J_mol": h - atomic_h,
            "current_minus_burcat_reaction_H_J_mol": reaction_h - h + atomic_h,
            "current_with_nasa_atoms_Cp_J_mol_K": reaction_cp + atomic_cp,
            "burcat_Cp_J_mol_K": cp,
        })
    atoms = inputs["atct_1_172"]["atoms"]
    atomization = {}
    for name in ("linear", "cyclic"):
        molecule = inputs["atct_1_172"][name]
        atomization[name] = {f"reaction_H{temperature}_kJ_mol": molecule[f"hf{temperature}_kJ_mol"]
                            - 3 * atoms["C"][f"hf{temperature}_kJ_mol"]
                            - atoms["H"][f"hf{temperature}_kJ_mol"]
                            for temperature in (0, 298)}
    return {
        "reaction": "3 C(g) + H(g) -> C3H(g)", "standard_pressure_Pa": 100000,
        "gas_constant_J_mol_K": r, "samples": samples,
        "max_abs_source_residual_ln_K": max(abs(x["current_minus_tsuji_ln_K"]) for x in samples),
        "atct_balanced_reactions": atomization,
        "linear_minus_cyclic_H0_kJ_mol": inputs["atct_1_172"]["linear"]["hf0_kJ_mol"]
                                       - inputs["atct_1_172"]["cyclic"]["hf0_kJ_mol"],
        "footprint": footprint(),
        "candidate_scope": "Conditional linear-C3H caloric comparison; original Tsuji isomer/state construction unresolved.",
        "equilibrium_substitution": "not_run", "production_change": "none",
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
