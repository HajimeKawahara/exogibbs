"""Finite C, S and N scans on the separately pinned source hosts
=============================================================

Local temperature and pressure are prescribed. Planetary reservoir scaling
and the atmospheric pressure root belong to the Inventory consumer.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from sulfur_source import component_matrices, load_reference, solve_reduced_source


ATOMIC_MASS_G_MOL = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "Na": 22.98976928, "Mg": 24.305, "Si": 28.085, "S": 32.06, "Fe": 55.845,
}


def summarize_state(network: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Report formula-unit gas mass and atom inventories from solved amounts.

    Silicate/gas amounts count named molecules or formula units. Metal
    amounts count atoms. No empirical fraction is treated as an absolute
    reservoir, and absent C/N/S elements are reported as exact zeros.
    """
    species, formula, _ = component_matrices(network)
    amounts = np.asarray(state["component_amounts_mol"])
    atomic_masses = np.array([ATOMIC_MASS_G_MOL[e] for e in network["elements"]])
    component_masses = atomic_masses @ formula
    phase_elements = {}
    phase_masses = {}
    for phase, names in network["phases"].items():
        indices = [species.index(name) for name in names]
        totals = formula[:, indices] @ amounts[indices]
        phase_elements[phase] = dict(zip(network["elements"], totals.tolist()))
        phase_masses[phase] = float(component_masses[indices] @ amounts[indices])
    gas = [species.index(name) for name in network["phases"]["gas"]]
    gas_amount = float(amounts[gas].sum())
    gas_fractions = dict(zip(network["phases"]["gas"], (amounts[gas] / gas_amount).tolist()))
    return {
        "phase_element_amounts_mol": phase_elements,
        "phase_cns_amounts_mol": {
            phase: {element: totals.get(element, 0.0) for element in ("C", "N", "S")}
            for phase, totals in phase_elements.items()
        },
        "phase_masses_g": phase_masses,
        "gas_mass_g": phase_masses["gas"],
        "gas_mean_molar_mass_g_mol": phase_masses["gas"] / gas_amount,
        "gas_mole_fractions": gas_fractions,
        "gas_h2o_h2_ratio": gas_fractions["H2O_gas"] / gas_fractions["H2_gas"],
    }


def scan_source(
    network: dict[str, Any], case: dict[str, Any], element: str,
    total_element_amounts_mol: Sequence[float], *,
    element_amounts_mol: Any = None, pressure_bar: float | None = None,
) -> dict[str, Any]:
    """Vary one finite C/N/S budget, retaining every other local atom budget.

    The supplied order is preserved, including returns to exact zero. Gas,
    silicate and metal remain present; this does not assess phase stability.
    S/N with S=N=0 retains its own host and standards and is not the Carbon
    source model. A failed local root propagates instead of entering a scan.
    """
    if element not in ("C", "N", "S") or element not in network["elements"]:
        raise ValueError("Select a C/N/S element present in this source network.")
    values = np.asarray(total_element_amounts_mol, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Supply a nonempty sequence of finite nonnegative atom amounts.")
    budget = np.asarray(network["element_amounts_mol"] if element_amounts_mol is None
                        else element_amounts_mol, dtype=float).copy()
    if (budget.shape != (len(network["elements"]),) or not np.all(np.isfinite(budget))
            or np.any(budget < 0)):
        raise ValueError("Budgets must be finite, nonnegative and use the complete source element order.")
    species, formula, _ = component_matrices(network)
    seed = np.asarray(case["component_amounts_mol"])
    previous = None
    states = []
    for value in values:
        budget[network["elements"].index(element)] = value
        supported = np.all(formula[budget == 0] == 0, axis=0)
        initial = np.zeros(len(species))
        # Positive guesses for newly activated solutes seed an optimization;
        # excluded components always remain exactly zero in its support.
        guess = seed * budget.sum() / sum(network["element_amounts_mol"])
        if previous is not None:
            guess = np.where(previous > 0, previous, guess)
        initial[supported] = guess[supported]
        state = solve_reduced_source(
            network, case, element_amounts_mol=budget,
            initial_component_amounts_mol=initial, pressure_bar=pressure_bar,
        )
        previous = np.asarray(state["component_amounts_mol"])
        states.append({"total_scanned_element_mol": float(value),
                       "local_state": state, "summary": summarize_state(network, state)})
    return {
        "model_id": network["model_id"], "case_id": case["id"],
        "scanned_element": element, "evidence_level": "conditional source-host mechanism",
        "pressure_policy": "prescribed local pressure; no atmospheric column closure",
        "excluded_reservoirs": ["separate sulfide", "graphite", "carbides", "metal N", "nitrides"],
        "limitations": (
            "Original source host and reaction standards; no MELTS mapping or common alloy Gibbs energy. "
            "No partition calibration, metal-N omission bound, or competing-phase stability assessment. "
            "Frozen source temperatures extrapolate the MgO liquid fit."
        ),
        "states": states,
    }


def run_controls() -> dict[str, Any]:
    """Run C-only, S-only, C/S and N-with-C/S source-host controls at 2350 K."""
    reference = load_reference()
    carbon = reference["networks"]["carbon"]
    sulfur_nitrogen = reference["networks"]["sulfur_nitrogen"]
    controls = {}
    for name, network, element, absent in (
        ("carbon_only", carbon, "C", ()),
        ("sulfur_only", sulfur_nitrogen, "S", ("C", "N")),
        ("carbon_sulfur", sulfur_nitrogen, "S", ("N",)),
        ("nitrogen_with_carbon_sulfur", sulfur_nitrogen, "N", ()),
    ):
        budget = np.asarray(network["element_amounts_mol"]).copy()
        for zero in absent:
            budget[network["elements"].index(zero)] = 0.0
        values = budget[network["elements"].index(element)] * np.array([0.0, 0.25, 1.0, 2.0])
        controls[name] = scan_source(network, network["cases"][0], element, values,
                                     element_amounts_mol=budget)
    return {"source": reference["source"], "controls": controls}


if __name__ == "__main__":
    print(json.dumps(run_controls(), indent=2, allow_nan=False))
