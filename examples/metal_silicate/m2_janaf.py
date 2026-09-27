"""Pinned JANAF atomic references and finite-source trace-demand diagnostics.

These paper-specific, opt-in inputs fill six missing gas reference energies.
They neither recalibrate MELTS nor bound a re-equilibrated planetary response.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.special import logsumexp

from m2_omitted_gas import OMITTED_ELEMENTS, screen_source


DATA_PATH = Path(__file__).with_name("data") / "janaf_atomic.json"
DATA_SHA256 = "4a9d51e8dd58f513b334b5025f0de49476ec5359e1ca811f9e9fd3daab829518"
CONTROL_ELEMENTS = ("Mg", "Fe", "Na")


def atomic_standard_audit(temperature_k: float) -> dict:
    """Reconstruct atomic G from Hf(298), enthalpy increments and absolute S.

    Cubic Hermite interpolation of G uses the tabulated -S derivatives; it
    does not interpolate across the unsampled 298--2100 K interval or beyond
    2400 K. A linear-H/S comparison diagnoses interpolation sensitivity, not
    a statistical uncertainty or rigorous error bound.
    """
    raw = DATA_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("The pinned JANAF numeric excerpt changed.")
    data = json.loads(raw)
    if (isinstance(temperature_k, (bool, np.bool_)) or not np.isfinite(temperature_k)
            or not data["evaluation_temperature_interval_K"][0] <= temperature_k <= data["evaluation_temperature_interval_K"][1]):
        raise ValueError("JANAF excerpt evaluation requires 2100 <= T/K <= 2400; no extrapolation.")
    t, r = float(temperature_k), data["common_R_J_mol_K"]
    values = {}
    for element, row in data["rows"].items():
        grid, entropy, g_function, dh = np.asarray(row["data"], dtype=float).T
        hf = row["formation_enthalpy_298_kJ_mol"] * 1000.
        # delta-f G(T) uses the elements' temperature-dependent reference
        # phases and is deliberately not used as this absolute G convention.
        gibbs = hf + dh * 1000. - grid * entropy
        spline = CubicHermiteSpline(grid, gibbs, -entropy, extrapolate=False)
        value = float(spline(t)) / (r * t)
        linear = (hf + np.interp(t, grid, dh) * 1000. - t * np.interp(t, grid, entropy)) / (r * t)
        values[element] = {
            "value_rt": value, "linear_h_s_value_rt": float(linear),
            "hermite_minus_linear_rt": float(value - linear), "source": row["url"],
            "g_function_reconstruction_max_abs_J_mol": float(np.max(np.abs(dh * 1000. - grid * entropy + grid * g_function))),
        }
    return {"temperature_K": t, "pressure_standard_bar": 1., "atomic_standards": values,
            "reference_convention": "Hf(298.15 K) + H(T)-H(298.15 K) - T*S(T), with absolute entropy and p0=1 bar; not delta-f G(T).",
            "temperature_interval_K": data["evaluation_temperature_interval_K"],
            "interpolation": "Cubic Hermite G(T), derivative -S at each tabulated node.",
            "interpolation_error_bounded": False, "empirical_uncertainty_available": False,
            "physical_cross_phase_calibration_accepted": False,
            "data_sha256": DATA_SHA256, "data_extraction": data["extraction"],
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def janaf_atomic_references(temperature_k: float) -> dict:
    """Return the existing omitted-gas input schema for the six missing atoms."""
    audit = atomic_standard_audit(temperature_k)
    return {"temperature_K": audit["temperature_K"], "pressure_standard_bar": 1.,
            "elements": {element: {"value_rt": audit["atomic_standards"][element]["value_rt"],
                "source": audit["atomic_standards"][element]["source"] + "; " + audit["reference_convention"]}
                for element in OMITTED_ELEMENTS}}


def _representable_exp(log_value):
    return float(np.exp(log_value)) if -700 <= log_value <= 700 else None


def fixed_reservoir_demand(formula_matrix, log_mole_fractions, gas_amount_mol,
                           element_budget_mol, *, elements, species, mole_fraction_target):
    """Compare unnormalized trace demands with the SAME finite source ledger.

    ``n_i = n_gas(saved) exp(log_x_i)`` is a fixed-reservoir diagnostic on the
    saved gas amount scale, not an equilibrium addition or a conserved state.
    Finite source budgets reveal inconsistency; they are not response bounds.
    """
    a, log_x, budget = (np.asarray(v, dtype=float) for v in
                        (formula_matrix, log_mole_fractions, element_budget_mol))
    if (not elements or not species or len(set(elements)) != len(elements) or len(set(species)) != len(species)
            or a.shape != (len(elements), len(species)) or log_x.shape != (len(species),)
            or budget.shape != (len(elements),) or np.any(a < 0) or np.any(a.sum(axis=0) <= 0)
            or np.any(budget < 0) or any(np.any(~np.isfinite(v)) for v in (a, log_x, budget))
            or not np.isfinite(gas_amount_mol) or gas_amount_mol <= 0
            or not np.isfinite(mole_fraction_target) or not 0 < mole_fraction_target < 1):
        raise ValueError("Require finite aligned neutral formulas, log fractions and nonnegative source budgets.")
    log_sum = float(logsumexp(log_x))
    rows, exceeded = [], []
    for index, element in enumerate(elements):
        support = a[index] > 0
        log_demand = float(np.log(gas_amount_mol) + logsumexp(log_x[support] + np.log(a[index, support]))) if np.any(support) else None
        exceeds = bool(log_demand is not None and (budget[index] == 0 or log_demand > np.log(budget[index])))
        ratio_log = None if log_demand is None or budget[index] == 0 else float(log_demand - np.log(budget[index]))
        rows.append({"element": element, "source_budget_mol": float(budget[index]),
                     "log_demand_mol": log_demand,
                     "demand_mol": 0. if log_demand is None else _representable_exp(log_demand),
                     "log_demand_over_source_budget": ratio_log,
                     "demand_over_source_budget": 0. if log_demand is None and budget[index] > 0 else None if ratio_log is None else _representable_exp(ratio_log),
                     "source_budget_exceeded": exceeds})
        if exceeds:
            exceeded.append(element)
    target_exceeded = [name for name, value in zip(species, log_x) if value > np.log(mole_fraction_target)]
    reasons = []
    if log_sum >= 0:
        reasons.append("Omitted-gas fraction sum is at least one at the saved chemical potentials.")
    if exceeded:
        reasons.append("Fixed-reservoir demands exceed the finite source budgets for: " + ", ".join(exceeded) + ".")
    if target_exceeded:
        reasons.append("One or more omitted gases exceed the declared species trace target.")
    return {"gas_amount_scale_mol": float(gas_amount_mol), "log_sum_mole_fractions": log_sum,
            "sum_mole_fractions": _representable_exp(log_sum),
            "fraction_sum_at_least_one": bool(log_sum >= 0),
            "element_demands": rows, "source_budget_exceeded_elements": exceeded,
            "species_trace_target_exceeded": target_exceeded,
            "trace_screen_passed": not reasons, "reasons": reasons,
            "is_equilibrated_finite_state": False, "accepted_omission_bound": False,
            "scope": "Unnormalized demands at unchanged chemical potentials and saved gas amount scale. The denominator is reconstructed from the same local source, not the planetary inventory. Passing is only a trace screen; depletion, O/H exchange, alloy response, condensates and pressure feedback remain unbounded."}


def screen_source_janaf(source, *, mole_fraction_target=1e-8):
    """Assess a preserved accepted source with pinned optional JANAF inputs."""
    from exogibbs.presets.fastchem4_cond import condensate_chemical_setup

    t = source["temperature_K"]
    audit = atomic_standard_audit(t)
    report = screen_source(source, mole_fraction_target=mole_fraction_target,
                           additional_atomic_references=janaf_atomic_references(t))
    parcel = source.get("source_atmosphere_parcel", source.get("source_atmosphere"))
    record = source.get("source_record", source.get("record"))
    if (not isinstance(parcel, dict) or parcel.get("accepted") is not True
            or source["source_result"].get("accepted") is not True
            or parcel["T_K"] != t or parcel["P_bar"] != source["pressure_bar"]):
        raise ValueError("Require an accepted saved public source and atmosphere at unchanged T/P.")
    elements = report["elements"]
    names = [name for phase in record["phases"].values() for name in phase]
    amounts = np.asarray(source["source_result"]["component_amounts_mol"], dtype=float)
    gas = np.asarray(parcel["gas_amounts_mol"], dtype=float)
    if (len(names) != len(set(names)) or amounts.shape != (len(names),) or gas.ndim != 1
            or any(np.any(~np.isfinite(v)) or np.any(v < 0) for v in (amounts, gas))
            or any(set(record["component_formulas"][name]) - set(elements) for name in names)):
        raise ValueError("Require a finite complete public source ledger without unknown elements.")
    matrix = np.array([[record["component_formulas"][name].get(e, 0.) for name in names] for e in elements])
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("Invalid saved source formula matrix.")
    gas_names = record["phases"].get("gas", [])
    aliases = record.get("gas_species_aliases", {})
    parcel_names = parcel.get("gas_species", [])
    if (not gas_names or set(aliases) != set(gas_names)
            or len(set(parcel_names)) != len(parcel_names) or gas.shape != (len(parcel_names),)
            or set(aliases.values()) != set(parcel_names) or len(set(aliases.values())) != len(gas_names)):
        raise ValueError("The saved gas species and public source aliases disagree.")
    expected_gas = np.array([amounts[names.index(name)] for name in gas_names])
    aligned_gas = np.array([gas[parcel_names.index(aliases[name])] for name in gas_names])
    if not np.allclose(aligned_gas, expected_gas, rtol=1e-12, atol=0):
        raise ValueError("The saved gas amount scale differs from the public source ledger.")
    budget = matrix @ amounts
    rows = report["gas_candidates"]
    formula = [[row["formula"].get(e, 0.) for row in rows] for e in elements]
    report["fixed_reservoir_demand"] = fixed_reservoir_demand(
        formula, [row["log_mole_fraction"] for row in rows], float(gas.sum()), budget,
        elements=elements, species=[row["species"] for row in rows], mole_fraction_target=mole_fraction_target)
    # A complete check on three pre-existing atomic anchors exposes differing
    # source data. Never shift the saved seven anchors to make this audit pass.
    catalog = condensate_chemical_setup(silent=True).gas_setup
    raw = np.asarray(catalog.hvector_func(t))
    # FastChem's neutral monatomic references are exactly zero. Refuse a
    # different convention instead of silently interpreting absolute G as a
    # gauge shift with the wrong origin.
    for element in OMITTED_ELEMENTS:
        if raw[catalog.species.index(element + "1")] != 0:
            raise ValueError("The packaged gas atomic-zero reference changed.")
    for name in gas_names:
        column = catalog.species.index(aliases[name])
        expected_formula = {e: float(catalog.formula_matrix[i, column]) for i, e in enumerate(catalog.elements)
                            if catalog.formula_matrix[i, column]}
        if expected_formula != record["component_formulas"][name]:
            raise ValueError("The saved gas formulas differ from the pinned gas catalog.")
    controls = []
    for element in CONTROL_ELEMENTS:
        value = float(raw[catalog.species.index(element + "1")] + report["atomic_gauge_rt"][element])
        janaf = audit["atomic_standards"][element]["value_rt"]
        controls.append({"element": element, "saved_atomic_standard_rt": value,
                         "janaf_atomic_standard_rt": janaf, "janaf_minus_saved_rt": janaf - value})
    audit["existing_anchor_controls"] = controls
    audit["control_scope"] = "Independent Mg/Fe/Na data comparison only; no fitted gauge offset or cross-phase calibration. H/He/O/Si controls are not evaluated by this numeric excerpt."
    report["janaf_reference_audit"] = audit
    report["assessment_id"] = "m2_janaf_omitted_gas_demand_v1"
    return report
