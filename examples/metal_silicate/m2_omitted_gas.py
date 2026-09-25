"""Fixed-reservoir screening of neutral gases omitted from the M2 atmosphere.

An ideal trace gas has a finite equilibrium fugacity, not a pure-phase
zero-amount insertion test. Missing atomic energy references remain unknown;
the screen records the reference inequalities required for a declared trace
target instead of silently using FastChem's atomic-zero gauge as absolute G.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


OMITTED_ELEMENTS = ("Al", "Ca", "K", "Ti", "Cr", "P")


def trace_gas_requirements(formula_matrix, raw_standard_rt, elemental_potentials_rt,
                           element_gauge_rt, pressure_bar, *, mole_fraction_target):
    """Evaluate log(x) at fixed reservoir potentials, or a missing-gauge inequality.

    ``formula_matrix`` is elements by species. Potentials and standards use the
    same RT and p0=1 bar convention. ``None`` gauge entries are unavailable,
    not zero. An optional supplied gauge enables a conditional trace estimate;
    neither the inequality nor that estimate bounds coupled finite inventories.
    """
    a, h, lam = (np.asarray(value, dtype=float) for value in
                 (formula_matrix, raw_standard_rt, elemental_potentials_rt))
    if (a.ndim != 2 or h.shape != (a.shape[1],) or lam.shape != (a.shape[0],)
            or len(element_gauge_rt) != a.shape[0] or np.any(a < 0)
            or np.any(a.sum(axis=0) <= 0) or any(np.any(~np.isfinite(v)) for v in (a, h, lam))
            or not np.isfinite(pressure_bar) or pressure_bar <= 0
            or not np.isfinite(mole_fraction_target) or not 0 < mole_fraction_target < 1):
        raise ValueError("Require finite neutral formulas, matching potentials, positive bar pressure and a trace target in (0,1).")
    known = np.array([value is not None for value in element_gauge_rt])
    gauge = np.array([float(value) if present else 0. for value, present in zip(element_gauge_rt, known)])
    if np.any(~np.isfinite(gauge)):
        raise ValueError("Known atomic gauges must be finite; use None for unavailable references.")
    rhs = a.T @ lam - h - a[known].T @ gauge[known] - np.log(pressure_bar)
    rows = []
    for index, value in enumerate(rhs):
        missing = a[:, index] * ~known
        unavailable = bool(np.any(missing))
        log_x = None if unavailable else float(value)
        rows.append({
            "missing_gauge_coefficients": missing.tolist(),
            "required_missing_gauge_dot_lower_rt": float(value - np.log(mole_fraction_target)),
            "inequality": "dot(missing_gauge_coefficients, atomic_gauge_rt) >= required_missing_gauge_dot_lower_rt implies x <= target at fixed elemental potentials",
            "log_mole_fraction": log_x,
            "mole_fraction": (float(np.exp(value)) if not unavailable and -700 <= value <= 700 else None),
            "fixed_reservoir_trace_target_passed": None if unavailable else bool(value <= np.log(mole_fraction_target)),
            "status": "missing_atomic_reference" if unavailable else "conditional_fixed_reservoir_estimate",
        })
    return rows


def screen_source(source, *, mole_fraction_target=1e-8, additional_atomic_references=None):
    """Screen a saved provider source against the packaged neutral gas table.

    No native solve or ExoInventory import occurs. Additional reference inputs
    are temperature-specific hypotheses with mandatory provenance; supplying
    them never establishes cross-phase calibration. The source's original seven
    anchors cannot be overridden through this missing-reference interface.
    """
    from exogibbs.presets.fastchem4_cond import condensate_chemical_setup
    from m1_chemistry import provenance

    internal, result = source["source_internal_record"], source["source_internal_result"]
    metadata = source["source_metadata"]
    elements = list(internal["elements"])
    if result.get("accepted") is not True or len(set(elements)) != len(elements):
        raise ValueError("Require an accepted provider source with a unique element order.")
    lam = np.asarray(result["elemental_potentials_rt"], dtype=float)
    t, p = source["temperature_K"], source["pressure_bar"]
    if metadata["standards"]["evaluation_temperature_K"] != t:
        raise ValueError("Saved standards and source temperature disagree.")
    anchors = metadata["standards"]["common_gas"]
    gauge = dict(zip(anchors["elements"], anchors["element_gauge_rt"]))
    supplied = additional_atomic_references
    if supplied is not None:
        if (set(supplied) != {"temperature_K", "pressure_standard_bar", "elements"}
                or supplied["temperature_K"] != t or supplied["pressure_standard_bar"] != 1.
                or not isinstance(supplied["elements"], dict)):
            raise ValueError("Additional atomic references must match source temperature and p0=1 bar.")
        for element, row in supplied["elements"].items():
            if (element not in OMITTED_ELEMENTS or element in gauge or not isinstance(row, dict)
                    or set(row) != {"value_rt", "source"} or not isinstance(row["source"], str)
                    or not row["source"].strip() or isinstance(row["value_rt"], bool)
                    or not np.isfinite(row["value_rt"])):
                raise ValueError("Each missing atomic reference requires a finite value and nonempty source.")
            gauge[element] = float(row["value_rt"])
    catalog = condensate_chemical_setup(silent=True).gas_setup
    matrix = np.asarray(catalog.formula_matrix)
    excluded = [i for i, name in enumerate(catalog.elements) if name not in elements]
    omitted = [catalog.elements.index(name) for name in OMITTED_ELEMENTS]
    columns = np.flatnonzero(np.all(matrix[excluded] == 0, axis=0)
                            & np.any(matrix[omitted] > 0, axis=0))
    rows = [catalog.elements.index(name) for name in elements]
    formula = matrix[np.ix_(rows, columns)]
    raw = np.asarray(catalog.hvector_func(t))[columns]
    estimates = trace_gas_requirements(formula, raw, lam, [gauge.get(name) for name in elements], p,
                                      mole_fraction_target=mole_fraction_target)
    for index, item in enumerate(estimates):
        item["species"] = catalog.species[columns[index]]
        item["formula"] = {element: float(formula[row, index]) for row, element in enumerate(elements)
                           if formula[row, index]}
    # Metal activity coefficients are not inferred from gas thermochemistry.
    amounts = np.asarray(source["source_result"]["component_amounts_mol"])
    record = source["source_record"]
    names = [name for group in record["phases"].values() for name in group]
    metal_amount = float(sum(amounts[names.index(name)] for name in record["phases"]["metal"]))
    metal = [{"element": element, "elemental_potential_rt": float(lam[elements.index(element)]),
              "metal_component_amount_mol": metal_amount,
              "status": "missing_alloy_activity_and_standard",
              "incipient_alloy_effect_bounded": False}
             for element in ("Mg", *OMITTED_ELEMENTS)]
    return {
        "assessment_id": "m2_omitted_gas_reference_screen_v1", "temperature_K": t, "pressure_bar": p,
        "elements": elements, "mole_fraction_target": mole_fraction_target,
        "target_basis": "Declared screening threshold; not an empirical error tolerance.",
        "atomic_gauge_rt": {element: gauge.get(element) for element in elements},
        "additional_atomic_references": supplied,
        "gas_candidates": estimates, "omitted_metal_paths": metal,
        "source_scope": "Saved local source only; no assertion of planetary closure or host global stability.",
        "response_scope": "Fixed chemical potentials and ideal gas at p0=1 bar. Trace estimates omit reservoir depletion and chemical/pressure feedback; coupled errors remain unbounded.",
        "accepted_omission_bound": False, "scientific_acceptance": "pending",
        "provenance": {"gas_table": provenance(),
                       "screen_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }
