"""Frozen 2350 K mass-action contact control
=========================================

This example changes declared reaction standards, then retains the source's
empirical deep activities. Local equation closure is not a Gibbs-minimum,
phase-stability, empirical-calibration, or thermochemical-error-bound claim.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import time
from typing import Any

import jax
import numpy as np
from scipy.optimize import least_squares

_SPEC = importlib.util.spec_from_file_location(
    "_m1_control_chemistry", Path(__file__).with_name("m1_chemistry.py"),
)
CHEMISTRY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CHEMISTRY)

MODEL_ID = "m1_frozen_source_common_parcel_mass_action_v1"
ANCHOR_SPECIES = ("H2_gas", "O2_gas", "Fe_gas", "Mg_gas", "SiO_gas", "Na_gas", "He_gas")
BOUNDARY_CONTRACT = {
    "id": "m1_frozen_source_common_parcel_mass_action_v1",
    "kind": "declared_mass_action_contact_control",
    "continuous": ["temperature_K", "total_pressure_bar", "gas_partial_pressures_bar",
                   "element_amounts_mol", "declared_source_reaction_equilibrium"],
    "amount_basis": "Atmospheric atom moles include upper gas and retained pure condensates.",
    "contact_phases": "Positive source silicate and metal contact the upper gas and retained cloud.",
    "unconstrained": ["enthalpy", "entropy", "energy_balance", "global_phase_stability"],
    "acceptance": "Common declared mass-action equations; no global Gibbs minimum, calibration, or M1 acceptance.",
}


def build_control(network: dict, case: dict, upper: Any) -> dict:
    """Retain seven declared source standards and replace two formation data.

    The reaction matrix determines source standards only up to an elemental
    gauge. Match the seven named gas standards to the upper table, then replace
    H2O and SiH4. The anchor choice declares retained physical data; changing
    anchors is a different counterfactual, not merely a gauge transformation.
    Only the source's fixed 2350 K temperature is supported.
    """
    if (case["T_K"] != 2350.0 or upper.gas_setup.elements != CHEMISTRY.ELEMENTS
            or upper.gas_species != CHEMISTRY.EXPANDED_GAS_SPECIES
            or upper.condensate_species != CHEMISTRY.CONDENSATE_SPECIES):
        raise ValueError("The control requires the seven-element upper model at 2350 K.")
    species, formula, reactions = CHEMISTRY.SOURCE.component_matrices(network)
    excluded = [i for i, element in enumerate(network["elements"]) if element not in CHEMISTRY.ELEMENTS]
    supported = np.all(formula[excluded] == 0, axis=0)
    components = np.flatnonzero(supported)
    rows = np.flatnonzero(np.all(reactions[:, ~supported] == 0, axis=1))
    names = tuple(species[i] for i in components)
    gas = np.array([names.index(name) for name in CHEMISTRY.SHARED_SOURCE_SPECIES])
    basis = np.array([names.index(name) for name in ANCHOR_SPECIES])
    nu = reactions[np.ix_(rows, components)]
    upper_h = np.asarray(upper.gas_setup.hvector_func(case["T_K"]))
    shared_h = upper_h[[upper.gas_species.index(name) for name in CHEMISTRY.SHARED_GAS_SPECIES]]
    anchor_h = shared_h[[CHEMISTRY.SHARED_SOURCE_SPECIES.index(name) for name in ANCHOR_SPECIES]]
    matrix = np.vstack((nu, np.eye(len(components))[basis]))
    if matrix.shape != (23, 23) or np.linalg.matrix_rank(matrix) != 23:
        raise ValueError("The declared source requires 16 independent reactions and seven anchors.")
    rhs = np.r_[np.asarray(case["source_delta_g_over_rt"])[rows], anchor_h]
    source_h = np.linalg.solve(matrix, rhs)
    delta = shared_h - source_h[gas]
    new_case = deepcopy(case)
    new_case["id"] = case["id"] + "_upper_gas_standard_control"
    full_gas = [species.index(name) for name in CHEMISTRY.SHARED_SOURCE_SPECIES]
    new_case["source_delta_g_over_rt"] = (
        np.asarray(case["source_delta_g_over_rt"]) + reactions[:, full_gas] @ delta
    ).tolist()
    gas_only = np.all(nu[:, np.setdiff1d(np.arange(len(components)), gas)] == 0, axis=1)
    return {
        "model_id": MODEL_ID, "T_K": 2350.0, "case": new_case,
        "active_component_indices": components.tolist(), "active_source_reaction_indices": rows.tolist(),
        "contact_reaction_indices": rows[~gas_only].tolist(),
        "source_species": list(names), "anchor_species": list(ANCHOR_SPECIES),
        "source_standard_potentials_rt": source_h.tolist(),
        "shared_gas_standard_shift_rt": delta.tolist(),
        "standard_reconstruction_max_abs_rt": float(np.max(np.abs(matrix @ source_h - rhs))),
        "standard_policy": "Seven declared gas anchors retain source data; H2O and SiH4 use the upper table.",
        "temperature_policy": "Frozen 2350 K source only; upper table retains its original temperature dependence.",
        "phase_policy": "Positive source silicate and metal contact the upper gas and retained pure condensates.",
        "limitations": "Empirical source activities; no common excess Gibbs energy, global phase stability, calibration, or error bound.",
    }



def solve_standard_control(network: dict, case: dict, upper: Any, **kwargs) -> dict:
    """Change the two gas standards while retaining the original gas catalog."""
    control = build_control(network, case, upper)
    report = CHEMISTRY.SOURCE.solve_reduced_source(network, control["case"], **kwargs)
    report["model_id"] = "m1_frozen_source_upper_gas_standards_v1"
    report["thermochemical_control"] = control
    return report


def audit_control(network: dict, case: dict, upper: Any, report: dict) -> dict:
    """Recount finite atoms and all 16 source reactions from primitive arrays."""
    control = build_control(network, case, upper)
    species, formula, reactions = CHEMISTRY.SOURCE.component_matrices(network)
    budget = np.asarray(report["element_amounts_mol"], dtype=float)
    amounts = np.asarray(report["deep_component_amounts_mol"], dtype=float)
    active = np.asarray(control["active_component_indices"])
    active_names = tuple(species[i] for i in active)
    element_rows = [network["elements"].index(e) for e in CHEMISTRY.ELEMENTS]
    atmosphere = np.asarray(report["atmosphere_element_amounts_mol"])
    parcel = report["bottom_parcel"]
    if (tuple(parcel["gas_species"]) != upper.gas_species
            or tuple(parcel["condensate_species"]) != upper.condensate_species):
        raise ValueError("The parcel species orders must match the upper setup.")
    if (budget.shape != (len(network["elements"]),) or np.any(~np.isfinite(budget))
            or np.any(budget[element_rows] <= 0)
            or np.any(budget[np.setdiff1d(np.arange(len(budget)), element_rows)] != 0)
            or atmosphere.shape != (7,) or np.any(~np.isfinite(atmosphere)) or np.any(atmosphere <= 0)):
        raise ValueError("The control requires seven finite positive budgets and exact-zero C/N/S.")
    if report["T_K"] != 2350.0 or parcel["T_K"] != report["T_K"] or parcel["P_bar"] != report["P_bar"]:
        raise ValueError("The boundary requires common temperature and pressure.")
    if report["deep_source_species"] != list(species) or report["elements"] != network["elements"]:
        raise ValueError("The source species and element orders must match the network.")
    deep_columns = [species.index(name) for phase in ("silicate", "metal")
                    for name in network["phases"][phase] if name in active_names]
    excluded = np.setdiff1d(np.arange(len(species)), deep_columns)
    if (amounts.shape != (len(species),) or np.any(~np.isfinite(amounts))
            or np.any(amounts[deep_columns] <= 0) or np.any(amounts[excluded] != 0)):
        raise ValueError("Require positive active deep components and exact zero elsewhere.")
    audited = CHEMISTRY.audit_parcel(
        upper.gas_setup, report["T_K"], report["P_bar"], atmosphere,
        parcel["gas_amounts_mol"], condensate_setup=upper.condensate_setup,
        condensate_amounts=parcel["condensate_amounts_mol"],
    )
    phase_atoms = {}
    fractions = amounts.copy()
    for phase in ("silicate", "metal"):
        columns = [species.index(name) for name in network["phases"][phase]]
        fractions[columns] /= amounts[columns].sum()
        phase_atoms[phase] = (formula[:, columns] @ amounts[columns]).tolist()
    for phase, key in (("gas", "gas_element_amounts_mol"), ("retained_cloud", "cloud_element_amounts_mol")):
        atoms = np.zeros(len(network["elements"]))
        atoms[element_rows] = audited[key]
        phase_atoms[phase] = atoms.tolist()
    balance = np.sum(list(phase_atoms.values()), axis=0) - budget
    balance[budget > 0] /= budget[budget > 0]
    gas_active = [active_names.index(name) for name in CHEMISTRY.SHARED_SOURCE_SPECIES]
    deep_active = [i for i in range(len(active)) if i not in gas_active]
    ln_x = np.empty(len(active))
    ln_x[deep_active] = np.log(fractions[active[deep_active]])
    shared_columns = [upper.gas_species.index(name) for name in CHEMISTRY.SHARED_GAS_SPECIES]
    ln_x[gas_active] = np.log(np.asarray(audited["gas_mole_fractions"])[shared_columns])
    chemistry = CHEMISTRY.SOURCE._make_reaction_residual(
        network, control["case"], active, np.asarray(control["active_source_reaction_indices"]),
    )
    chemical = np.asarray(chemistry(ln_x, report["P_bar"]))
    return {
        "accepted": bool(audited["accepted"] and np.max(np.abs(balance)) < 1e-9
                         and np.max(np.abs(chemical)) < 1e-8),
        "relative_element_residual": balance.tolist(),
        "reaction_residual": chemical.tolist(),
        "source_reaction_indices": control["active_source_reaction_indices"],
        "phase_element_amounts_mol": phase_atoms,
        "upper_audit": audited,
        "contract": deepcopy(BOUNDARY_CONTRACT),
        "common_gas_contact_equilibrium": bool(audited["accepted"] and np.max(np.abs(chemical)) < 1e-8),
        "global_phase_stability_certified": False,
        "m1_a_accepted": False,
    }

def solve_control(
    network: dict, case: dict, upper: Any, *, element_amounts_mol: Any,
    pressure_bar: float, initial_source: dict | None = None,
    initial_control: dict | None = None, max_nfev: int = 80,
) -> dict:
    """Close finite deep amounts and one retained atmospheric parcel locally.

    The 21 logarithmic unknowns are 14 deep component amounts and seven
    atmospheric atom amounts. Existing parcel equilibrium supplies the shared
    partial pressures; 14 contact reactions and seven finite budgets close
    the root. Pure condensates belong to the atmospheric parcel, not magma.
    """
    if not jax.config.x64_enabled:
        raise ValueError("The control requires JAX_ENABLE_X64=1.")
    control = build_control(network, case, upper)
    species, formula, _ = CHEMISTRY.SOURCE.component_matrices(network)
    budget = np.asarray(element_amounts_mol, dtype=float)
    element_rows = np.array([network["elements"].index(e) for e in CHEMISTRY.ELEMENTS])
    excluded = np.setdiff1d(np.arange(len(network["elements"])), element_rows)
    if (budget.shape != (len(network["elements"]),) or np.any(~np.isfinite(budget))
            or np.any(budget[element_rows] <= 0) or np.any(budget[excluded] != 0)):
        raise ValueError("Supply positive seven-element budgets and exact-zero C/N/S.")
    if not np.isfinite(pressure_bar) or pressure_bar <= 0:
        raise ValueError("pressure_bar must be finite and positive.")
    if initial_source is not None and initial_control is not None:
        raise ValueError("Supply only one initial source or control result.")
    initialization = "previous_control" if initial_control is not None else "source_reactions"
    if initial_control is not None and initial_control["P_bar"] != pressure_bar:
        # A pressure jump can create transient cloud phases in an old parcel.
        # Keep that phase search in the provider and choose a nearby source seed.
        guess_atoms = (np.asarray(initial_control["atmosphere_element_amounts_mol"])
                       * budget.sum() / np.sum(initial_control["element_amounts_mol"]))
        try:
            probe = CHEMISTRY.solve_parcel(upper, 2350.0, pressure_bar, guess_atoms)
            old_cloud = np.asarray(initial_control["bottom_parcel"]["condensate_amounts_mol"]) > 0
            new_cloud = np.asarray(probe["condensate_amounts_mol"]) > 0
            changed = not probe["accepted"] or not np.array_equal(old_cloud, new_cloud)
        except (ValueError, RuntimeError):
            changed = True
        if changed:
            initial_control = None
            initialization = "source_after_changed_or_failed_warm_parcel"
    source = None if initial_control is not None else initial_source or CHEMISTRY.SOURCE.solve_reduced_source(
        network, control["case"], element_amounts_mol=budget, pressure_bar=pressure_bar,
    )
    active = np.array(control["active_component_indices"])
    active_names = tuple(species[i] for i in active)
    deep_names = tuple(name for phase in ("silicate", "metal")
                       for name in network["phases"][phase] if name in active_names)
    deep = np.array([species.index(name) for name in deep_names])
    deep_active = np.array([active_names.index(name) for name in deep_names])
    gas_active = np.array([active_names.index(name) for name in CHEMISTRY.SHARED_SOURCE_SPECIES])
    shared_columns = [upper.gas_species.index(name) for name in CHEMISTRY.SHARED_GAS_SPECIES]
    groups = [np.array([deep_names.index(name) for name in network["phases"][phase]
                        if name in deep_names]) for phase in ("silicate", "metal")]
    chemistry = jax.jit(CHEMISTRY.SOURCE._make_reaction_residual(
        network, control["case"], active, np.array(control["contact_reaction_indices"]),
    ))
    scale = float(budget.sum())
    if initial_control is None:
        initial_deep = np.asarray(source["component_amounts_mol"])[deep]
        initial_atmosphere = CHEMISTRY.source_gas_inventory(network, source)
    else:
        if initial_control["deep_source_species"] != list(species):
            raise ValueError("The initial control uses a different source component order.")
        guess_scale = scale / np.sum(initial_control["element_amounts_mol"])
        initial_deep = guess_scale * np.asarray(initial_control["deep_component_amounts_mol"])[deep]
        initial_atmosphere = guess_scale * np.asarray(initial_control["atmosphere_element_amounts_mol"])
    if (initial_atmosphere.shape != (7,) or np.any(~np.isfinite(initial_deep))
            or np.any(initial_deep <= 0) or np.any(~np.isfinite(initial_atmosphere))
            or np.any(initial_atmosphere <= 0)):
        raise ValueError("Initial deep and atmospheric amounts must be finite and positive.")
    initial = np.log(np.r_[initial_deep, initial_atmosphere] / scale)
    calls = 0
    cached = {}
    started = time.perf_counter()

    def evaluate(log_amounts):
        nonlocal calls
        key = np.asarray(log_amounts).tobytes()
        if key in cached:
            return cached[key]
        amounts = scale * np.exp(log_amounts[:len(deep)])
        atmosphere = scale * np.exp(log_amounts[len(deep):])
        parcel = CHEMISTRY.solve_parcel(upper, 2350.0, pressure_bar, atmosphere)
        calls += 1
        if not parcel["accepted"]:
            raise RuntimeError("The nested control parcel failed independent acceptance.")
        fractions = amounts.copy()
        for group in groups:
            fractions[group] /= amounts[group].sum()
        ln_x = np.empty(len(active))
        ln_x[deep_active] = np.log(fractions)
        ln_x[gas_active] = np.log(np.asarray(parcel["gas_mole_fractions"])[shared_columns])
        chemical = np.asarray(chemistry(ln_x, pressure_bar))
        total = formula[element_rows][:, deep] @ amounts + atmosphere
        balance = total / budget[element_rows] - 1.0
        value = (np.r_[chemical, balance], amounts, atmosphere, parcel)
        cached.clear()
        cached[key] = value
        return value

    # Cloud-bearing atom coordinates can differ greatly in chemical sensitivity.
    root = least_squares(lambda x: evaluate(x)[0], initial, diff_step=1e-5, x_scale="jac",
                         bounds=(-650.0, 10.0), method="dogbox",
                         xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=max_nfev)
    cached.clear()
    residual, amounts, atmosphere, parcel = evaluate(root.x)
    accepted = bool(root.success and np.max(np.abs(residual[:14])) < 1e-8
                    and np.max(np.abs(residual[14:])) < 1e-9)
    if not accepted:
        raise RuntimeError(f"Control failed independent acceptance: {root.message}; {np.max(abs(residual))}")
    full_deep = np.zeros(len(species))
    full_deep[deep] = amounts
    phase_atoms = {}
    for phase in ("silicate", "metal"):
        columns = [species.index(name) for name in network["phases"][phase]]
        phase_atoms[phase] = (formula[:, columns] @ full_deep[columns]).tolist()
    for phase, key in (("gas", "gas_element_amounts_mol"), ("retained_cloud", "cloud_element_amounts_mol")):
        atoms = np.zeros(len(network["elements"]))
        atoms[element_rows] = parcel[key]
        phase_atoms[phase] = atoms.tolist()
    report = {
        "model_id": MODEL_ID, "control": control, "accepted": accepted,
        "boundary_contract": deepcopy(BOUNDARY_CONTRACT),
        "T_K": 2350.0, "P_bar": pressure_bar, "elements": network["elements"],
        "element_amounts_mol": budget.tolist(), "deep_source_species": list(species),
        "deep_component_amounts_mol": full_deep.tolist(),
        "atmosphere_elements": list(CHEMISTRY.ELEMENTS),
        "atmosphere_element_amounts_mol": atmosphere.tolist(),
        "phase_element_amounts_mol": phase_atoms, "bottom_parcel": parcel,
        "source_formula_matrix": formula.tolist(),
        "source_phase_species": {p: network["phases"][p] for p in ("silicate", "metal")},
        "parcel_evaluations": calls, "outer_nfev": root.nfev,
        "initialization": initialization,
        "elapsed_seconds": time.perf_counter() - started,
        "common_gas_contact_equilibrium": True,
        "global_phase_stability_certified": False,
    }
    audit = audit_control(network, case, upper, report)
    if not audit["accepted"]:
        raise RuntimeError("The common-contact control failed fresh atom and reaction audits.")
    report.update({key: audit[key] for key in (
        "reaction_residual", "source_reaction_indices", "relative_element_residual", "phase_element_amounts_mol",
    )})
    report["independent_audit"] = audit
    return report
