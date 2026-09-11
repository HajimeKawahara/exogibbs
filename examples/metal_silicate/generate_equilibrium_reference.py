"""Independent local equilibrium reference generation
=====================================================

Freeze NumPy/SciPy mass-action solutions independently of JAX and ExoEOS.
The full Young/GCE component set is checked before the dry ternary reduction.
These isothermal, fixed-phase cases use formal liquid standards; they are
not measurements, planetary solutions, or phase-stability validations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from reference import component_matrices, load_reference, metal_standard_shift


def source_metal_ln_gamma(temperature: float, fractions: np.ndarray) -> np.ndarray:
    """Transcribe the pinned GCE author-code Si/O expressions independently."""
    _, si, oxygen, _ = fractions
    cross = -5.0 * 1873.0 / temperature
    si_value = (
        -6.65 * 1873.0 / temperature
        - 12.41 * 1873.0 / temperature * np.log1p(-si)
        - cross * (oxygen + np.log1p(-oxygen) - oxygen / (1 - si))
        + cross * oxygen**2 * si
        * (1 / (1 - si) + 1 / (1 - oxygen) + si / (2 * (1 - si)**2) - 1)
    )
    oxygen_value = (
        4.29 - 16500.0 / temperature + 1873.0 / temperature * np.log1p(-oxygen)
        - cross * (si + np.log1p(-si) - si / (1 - oxygen))
        + cross * si**2 * oxygen
        * (1 / (1 - oxygen) + 1 / (1 - si) + oxygen / (2 * (1 - oxygen)**2) - 1)
    )
    return np.array([0.0, si_value, oxygen_value, 0.0])


def completed_metal_ln_gamma(temperature: float, fractions: np.ndarray) -> np.ndarray:
    """Analytic partial derivatives of Ma's completed formal excess energy."""
    _, si, oxygen = fractions
    a, b, c = np.array([12.41 * 1873.0, -16500.0, -5.0 * 1873.0]) / temperature
    d = 1 / (1 - si) + 1 / (1 - oxygen) - 1
    pair = (
        -si * oxygen - si * np.log1p(-oxygen) - oxygen * np.log1p(-si)
        + (si * oxygen)**2 * d / 2
    )
    g = a * (1 - si) * np.log1p(-si) + b * (1 - oxygen) * np.log1p(-oxygen) + c * pair
    ds = -a * (np.log1p(-si) + 1) + c * (
        -oxygen - np.log1p(-oxygen) + oxygen / (1 - si)
        + si * oxygen**2 * d + (si * oxygen)**2 / (2 * (1 - si)**2)
    )
    do = -b * (np.log1p(-oxygen) + 1) + c * (
        -si - np.log1p(-si) + si / (1 - oxygen)
        + oxygen * si**2 * d + (si * oxygen)**2 / (2 * (1 - oxygen)**2)
    )
    solvent = g - si * ds - oxygen * do
    return np.array([solvent, solvent + ds, solvent + do])


def generate_case(record: dict, case: dict, *, completed: bool) -> dict:
    """Solve a square independent reaction/element system in log amounts."""
    species, formula, reactions = component_matrices(record)
    temperature, pressure = case["T_K"], 1.0
    budget = np.array([record["original_element_amounts_mol"][e] for e in record["elements"]])
    if completed:
        budget[4:] = 0.0
        active = np.array([0, 1, 2, 3, 4, 11, 12, 13])
        reaction_indices = np.array([1, 2, 3, 5])
        groups = (np.arange(5), np.arange(5, 8))
        initial = np.array([40., 25., 130., 4., 2., 170., 15., 1.])
    else:
        active = np.arange(25)
        reaction_indices = np.arange(18)
        groups = (np.arange(11), np.arange(11, 15), np.arange(15, 25))
        initial = np.r_[
            np.array([.12, .02, .66, .10, .02, .00001, .002, .02, .02, .00001, .00001]) * 250,
            np.array([.88, .06, .02, .04]) * 200,
            np.array([.7, .001, .00001, .00001, .00001, .25, .001, .001, .01, .04]) * 30,
        ]
    positive = budget > 0
    a = formula[positive][:, active]
    nu = reactions[reaction_indices][:, active]
    dg = np.array(case["source_delta_g_over_rt"])[reaction_indices]
    if completed:
        dg = dg + nu[:, groups[1]] @ metal_standard_shift(temperature)
    scale = budget.sum()

    def quantities(root):
        amounts = scale * np.exp(root)
        fractions = np.empty_like(amounts)
        for group in groups:
            fractions[group] = amounts[group] / amounts[group].sum()
        activity = np.log(fractions)
        if completed:
            activity[groups[1]] += completed_metal_ln_gamma(temperature, fractions[groups[1]])
        else:
            activity[groups[1]] += source_metal_ln_gamma(temperature, fractions[groups[1]])
            activity[groups[2]] += np.log(pressure)
        chemical = dg + nu @ activity
        if not completed:
            chemical[14] += np.log(pressure / 1e4)
        element = (a @ amounts) / budget[positive] - 1
        return amounts, fractions, chemical, element

    def residual(root):
        _, _, chemical, element = quantities(root)
        return np.r_[chemical, element]

    result = least_squares(
        residual, np.log(initial / scale), xtol=1e-13, ftol=1e-13, gtol=1e-13,
        max_nfev=1000,
    )
    amounts, fractions, chemical, element = quantities(result.x)
    if not result.success or np.max(np.abs(residual(result.x))) > 1e-10:
        raise RuntimeError(f"Independent reference did not converge: {result.message}")
    full = np.zeros(25)
    full[active] = amounts
    return {
        "id": ("completed_dry_" if completed else "source_full_") + str(int(temperature)),
        "model": "ma2001_fe_si_o_young2023_printed_v1" if completed else "gce_young_2023_author_code",
        "T_K": temperature, "P_bar": pressure,
        "element_amounts_mol": budget.tolist(),
        "active_components": [species[i] for i in active],
        "reaction_ids": [record["reactions"][i]["id"] for i in reaction_indices],
        "component_amounts_mol": full.tolist(),
        "phase_amounts_mol": [float(amounts[group].sum()) for group in groups],
        "mole_fractions_active": fractions.tolist(),
        "reaction_residual": chemical.tolist(), "relative_element_residual_positive": element.tolist(),
        "initial_component_amounts_mol_active": initial.tolist(),
        "scipy_evaluations": result.nfev,
    }


def generate_reference() -> dict:
    """Return two full-source and two completed, dry reference solutions."""
    record = load_reference()
    return {
        "description": "Independent NumPy/SciPy fixed-phase local mass-action solutions; formal liquid standards.",
        "gce_commit": record["source"]["commit"],
        "exoeos_commit": "9c62197d3a7c1aa882e246c9de3fbc4102410d81",
        "source_record": "reference.json",
        "scope": [
            "All reactions use one supplied temperature; pressure is prescribed locally with the stated R14 exception.",
            "Full25 cases retain GCE author-code metal activities, gamma_Fe=gamma_H=1, and the R14 pressure prescription.",
            "Dry8 cases use ideal silicate plus the completed native ternary metal model and matching formal standard shift.",
            "H, Na and C budgets in dry8 are exactly zero; gas is excluded by the declared phase assemblage.",
            "Both temperatures extrapolate source MgO liquid standards; no stable liquid or calibrated joint domain is established.",
            "Comparison tolerance: relative amounts 1e-8, absolute dimensionless residuals 1e-10.",
        ],
        "cases": [generate_case(record, case, completed=completed)
                  for completed in (False, True) for case in record["cases"]],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    rendered = json.dumps(generate_reference(), indent=2, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
