"""Metal--silicate thermochemistry audit
===========================================

Audit the pinned Young reaction thermochemistry without solving equilibrium.
This example uses NumPy and local reference data only. It does not import
GCE, ExoEOS, or a planetary pressure closure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


REFERENCE_PATH = Path(__file__).with_name("reference.json")


def load_reference(path: Path = REFERENCE_PATH) -> dict[str, Any]:
    """Load the example's fixed component and thermochemistry record."""
    return json.loads(path.read_text(encoding="utf-8"))


def component_matrices(record: dict[str, Any]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Return species, element matrix A[E,K], and reaction matrix nu[R,K]."""
    species = tuple(name for phase in record["phases"].values() for name in phase)
    formula = record["component_formulas"]
    elements = np.asarray([
        [formula[name].get(element, 0) for name in species]
        for element in record["elements"]
    ], dtype=np.float64)
    reactions = np.asarray([
        [reaction["stoichiometry"].get(name, 0.0) for name in species]
        for reaction in record["reactions"]
    ], dtype=np.float64)
    return species, elements, reactions


def shomate_standard_potential(temperature_k: float, coefficients: list[float]) -> float:
    """Evaluate H(298) + the Shomate enthalpy increment - T*S, in J/mol.

    The snapshot fixes the source's selected coefficient branch. This is not
    an interpolation or a claim of validity outside that reference state.
    """
    dh, a, b, c, d, e, f, g, h = coefficients
    t = temperature_k / 1000.0
    enthalpy_kj = dh + a*t + b*t**2/2 + c*t**3/3 + d*t**4/4 - e/t + f - h
    entropy = a*np.log(t) + b*t + c*t**2/2 + d*t**3/3 - e/(2*t**2) + g
    return float(1000.0 * enthalpy_kj - temperature_k * entropy)


def source_standard_potentials(record: dict[str, Any], case: dict[str, Any]) -> np.ndarray:
    """Reconstruct all 25 compatible standards from the source reaction basis.

    Empirical Na/Fe silicate reactions override the unused standalone source
    endmember fits. No missing potential is assigned an arbitrary zero.
    """
    temperature = case["T_K"]
    rt = record["source"]["gas_constant_J_mol_K"] * temperature
    log10_to_ln = record["source"]["log10_to_ln"]
    potentials = {
        name: shomate_standard_potential(temperature, coefficients)
        for name, coefficients in case["shomate"].items()
    }

    # Published empirical reaction fits in the pinned GCE implementation.
    dg_na = rt * log10_to_ln * (-1.33 + 13870.0 / temperature)
    dg_fe = rt * log10_to_ln * (-0.63 + 3103.0 / temperature)
    dg_si = -rt * log10_to_ln * (2.97 - 21800.0 / temperature)
    dg_o = -rt * log10_to_ln * (2.736 - 11439.0 / temperature)
    dg_h2 = rt * (12.5 + 0.76e-4)
    dg_h2o = rt * (14.21 - 2565.0 / temperature)
    dg_co2 = 5200.0 + 119.77 * temperature
    dg_co = dg_co2 + rt * log10_to_ln * np.log10(3.0)
    dg_okuchi = 143589.7 - 69.1 * temperature

    potentials["Na2SiO3_silicate"] = (
        potentials["Na2O_silicate"] + potentials["SiO2_silicate"] - dg_na
    )
    potentials["FeSiO3_silicate"] = (
        potentials["FeO_silicate"] + potentials["SiO2_silicate"] - dg_fe
    )
    potentials["H2_silicate"] = potentials["H2_gas"] + dg_h2
    potentials["H2O_silicate"] = potentials["H2O_gas"] + dg_h2o
    potentials["CO_silicate"] = potentials["CO_gas"] + dg_co
    potentials["CO2_silicate"] = potentials["CO2_gas"] + dg_co2
    potentials["Si_metal"] = (
        dg_si - 2 * potentials["FeO_silicate"]
        + 2 * potentials["Fe_metal"] + potentials["SiO2_silicate"]
    )
    potentials["O_metal"] = (
        dg_o + potentials["FeO_silicate"] - potentials["Fe_metal"]
    )
    potentials["H_metal"] = 0.5 * (
        dg_okuchi - potentials["FeO_silicate"]
        + potentials["Fe_metal"] + potentials["H2O_silicate"]
    )
    species, _, _ = component_matrices(record)
    return np.asarray([potentials[name] for name in species], dtype=np.float64)


def metal_standard_shift(temperature_k: float) -> np.ndarray:
    """Return (mu0_formal - mu0_source)/RT for ExoEOS's Fe, Si, O record.

    Pair this with ln_gamma_formal = ln_gamma_source - shift. The conversion
    concerns the completed ternary model, not GCE's four-component alloy.
    """
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("temperature_k must be finite and positive.")
    return np.asarray([
        0.0, 5.76 * 1873.0 / temperature_k, 4.29 - 33000.0 / temperature_k,
    ], dtype=np.float64)


def phase_element_amounts(
    record: dict[str, Any], component_amounts_mol: np.ndarray,
) -> dict[str, np.ndarray]:
    """Count atoms in each phase from absolute component amounts in mol.

    Metal components count atomic moles; gas and silicate components count
    formula-unit moles. Exact zeros are permitted, including an absent phase.
    """
    species, formula, _ = component_matrices(record)
    amounts = np.asarray(component_amounts_mol, dtype=np.float64)
    if amounts.shape != (len(species),):
        raise ValueError(f"component_amounts_mol must have shape {(len(species),)}.")
    if not np.all(np.isfinite(amounts)) or np.any(amounts < 0):
        raise ValueError("component_amounts_mol must be finite and nonnegative.")
    contributions = {}
    start = 0
    for phase, names in record["phases"].items():
        stop = start + len(names)
        contributions[phase] = formula[:, start:stop] @ amounts[start:stop]
        start = stop
    return contributions


def audit_reference(record: dict[str, Any]) -> dict[str, Any]:
    """Check source constants against independently reconstructed potentials."""
    species, formula, reactions = component_matrices(record)
    np.testing.assert_array_equal(formula @ reactions.T, 0.0)
    if np.linalg.matrix_rank(formula) != 7 or np.linalg.matrix_rank(reactions) != 18:
        raise ValueError("The reference must retain seven elements and 18 independent reactions.")
    cases = []
    for case in record["cases"]:
        rt = record["source"]["gas_constant_J_mol_K"] * case["T_K"]
        computed = reactions @ source_standard_potentials(record, case) / rt
        source = np.asarray(case["source_delta_g_over_rt"])
        np.testing.assert_allclose(computed, source, rtol=5e-12, atol=5e-12)
        cases.append({
            "id": case["id"],
            "max_abs_delta_g_over_rt_difference": float(np.max(np.abs(computed - source))),
            "ln_K_source": (-source).tolist(),
        })
    return {
        "status": "thermochemistry reference audit; no equilibrium solve",
        "component_count": len(species),
        "element_rank": 7,
        "reaction_rank": 18,
        "cases": cases,
    }


if __name__ == "__main__":
    print(json.dumps(audit_reference(load_reference()), indent=2, allow_nan=False))
