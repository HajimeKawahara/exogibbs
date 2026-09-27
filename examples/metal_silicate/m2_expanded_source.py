"""Finite BSE with a retained gas/cloud atmosphere
===============================================
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from common_gibbs import _feasible_start
from full_potential import PhaseState, ideal_phase
from local import build_problem
from m1_chemistry import build_setups
from m2_atmosphere import make_atmosphere_phase
from m2_common_gas import anchored_standards_rt, source_gas_names
from m2_finite_gas import atmosphere_gauge_rt, build_atmosphere_setup, catalog_sha256
from m2_janaf import DATA_PATH, atomic_standard_audit
from m2_phosphorus import add_phosphorus_metal, phosphorus_metal_domain
from m2_associated import add_associated_metal, associated_metal_domain
from m2_scenarios import apply_standard_offsets, metal_selection_domain, normalize_scenario
from run_bse_common_gibbs import build_bse_problem, source_standards_rt


def build_expanded_bse_problem(inventory_path, exoeos_checkout, runtime, python_executable,
                               *, temperature_k=2173.15, pressure_bar=1., scenario=None, gas_model="m1",
                               initialization="lp", liquid_model="native", metal_model="ma",
                               phosphorus_options=None, potassium_standard_offset_rt=None):
    """Return a finite source with seven or thirteen atmosphere atom carriers.

    ``m1`` retains 35 gases; opt-in ``janaf`` includes 41 background-element
    gases on six additional atomic references, retaining 26 condensates.
    ``janaf_condensed`` adds all 41 neutral background-element condensates.
    Carrier amounts are conserved coordinates, never added atomic gases.
    ``initialization='canonical'`` exposes a conserved metal-free starting
    vector in metadata. The returned canonical ledger remains unchanged.
    """
    if initialization not in ("lp", "canonical"):
        raise ValueError("initialization must be lp or canonical.")
    if metal_model not in ("ma", "phosphorus", "associated", "associated_k") or (metal_model == "ma" and phosphorus_options is not None):
        raise ValueError("Select metal_model ma, phosphorus, associated or associated_k; P options require an extended model.")
    if (metal_model == "associated_k") != (potassium_standard_offset_rt is not None):
        raise ValueError("Only associated_k requires an explicit potassium_standard_offset_rt.")
    normalized = normalize_scenario(scenario)
    setup = build_atmosphere_setup(gas_model)
    record, budget, callbacks, initial, metadata = build_bse_problem(
        inventory_path, exoeos_checkout, runtime, python_executable,
        temperature_k=temperature_k, pressure_bar=pressure_bar, gas_model="m1_expanded",
        liquid_model=liquid_model)

    def gauge(t, p):
        reference, _ = source_standards_rt(t, p)
        return atmosphere_gauge_rt(setup, t, reference)

    initial_gauge = gauge(temperature_k, pressure_bar)

    atmosphere = make_atmosphere_phase(setup, gauge)
    gas_names = record["phases"].pop("gas")
    start = len(initial) - len(gas_names)
    atom_names = [element + "_atmosphere_atom" for element in setup.gas_setup.elements]
    gas_formula = np.array([[record["component_formulas"][name].get(element, 0.) for name in gas_names]
                            for element in setup.gas_setup.elements])
    initial = np.r_[initial[:start], gas_formula @ initial[start:]]
    record["phases"]["atmosphere"] = atom_names
    for name in gas_names:
        del record["component_formulas"][name]
    record["component_formulas"].update({name: {element: 1.} for name, element in
                                          zip(atom_names, setup.gas_setup.elements)})
    record.pop("gas_species_aliases")
    record["atmosphere_element_order"] = list(setup.gas_setup.elements)
    record["atmosphere_gas_model"] = gas_model
    if metal_model in ("phosphorus", "associated", "associated_k"):
        if scenario is not None and "metal_bounds" in scenario:
            raise ValueError("Four-component scenario bounds cannot define the five-component P domain.")
        initial = add_phosphorus_metal(record, initial, callbacks, metadata, setup, initial_gauge,
                                       exoeos_checkout, temperature_k, pressure_bar, phosphorus_options)
        if metal_model in ("associated", "associated_k"):
            initial = add_associated_metal(record, initial, callbacks, metadata, setup, initial_gauge,
                                           exoeos_checkout, temperature_k, pressure_bar, phosphorus_options,
                                           potassium_standard_offset_rt=potassium_standard_offset_rt)
    metadata["numerical_initialization"] = {
        "strategy": initialization,
        "canonical_interior_fraction": 1e-4 if initialization == "canonical" else None,
        "initial_component_amounts_mol": (canonical_interior_seed(record, budget, initial).tolist()
                                           if initialization == "canonical" else None),
        "scope": "Numerical metal-free starting point only; no atom floor, new phase constraint, thermodynamic change, or relaxed acceptance tolerance.",
    }
    callbacks.pop("gas")
    callbacks["atmosphere"] = atmosphere
    if scenario is not None:
        callbacks = apply_standard_offsets(record, callbacks, normalized)
        metadata["provider_scenario"] = normalized
        metadata["provider_scenario_interpretation"] = (
            "Declared model sensitivity only; offsets and alloy bounds are not calibrated uncertainties.")
    metadata["model_id"] = "bse_melts_ma_retained_atmosphere_conditional_v1"
    metadata["standards"]["gas_model"] = gas_model + "_retained"
    metadata["standards"]["common_gas"].update(
        elements=list(setup.elements), element_gauge_rt=initial_gauge.tolist(),
        species=list(setup.gas_species))
    metadata["atmosphere"] = {
        "gas_species": list(setup.gas_species), "condensate_species": list(setup.condensate_species),
        "gas_model": gas_model, "elements": list(setup.elements),
        "catalog_sha256": catalog_sha256(setup),
        "source_reference_sha256": hashlib.sha256(json.dumps({
            "temperature_K": temperature_k, "pressure_standard_bar": 1.,
            "elements": list(setup.elements), "element_gauge_rt": initial_gauge.tolist(),
        }, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "amount_basis": "Finite atomic amounts, internally minimized into gas plus retained cloud.",
        "retention": "Full retention; atmospheric Fe condensate is distinct from the deep alloy.",
        "pressure": "Gas partial pressures sum to total pressure; clouds add mass without gas pressure.",
        "upper_reference_policy": "Re-evaluate raw FastChem reactions at every layer T/P. Conserved-element gauges cancel from isolated parcel composition; no low-temperature JANAF extrapolation or frozen source chemical potentials.",
        "missing_paths": (["Al/Ca/K/Ti/Cr/P retained condensates"]
                          if gas_model != "janaf_condensed" else [])
                         + ([] if metal_model == "associated_k" else ["K alloy component" if metal_model == "associated" else
                            "Mg/Al/Ca/K/Ti/Cr alloy components" if metal_model == "phosphorus"
                            else "Mg/Al/Ca/K/Ti/Cr/P alloy components"]),
        "pure_phase_reference_policy": "FastChem pure condensates and native MELTS phases are independent thermochemical models. Matching formulas do not establish matching energies or a calibrated phase boundary.",
    }
    metadata["provenance"]["file_sha256"].update({name: hashlib.sha256(
        Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("m2_expanded_source.py", "m2_atmosphere.py", "m1_chemistry.py", "m2_scenarios.py",
                     "m2_finite_gas.py", "m2_janaf.py", "m2_omitted_gas.py", "phase_selection.py",
                     "m2_phosphorus.py", "m2_associated.py")})
    metadata["provenance"]["file_sha256"]["data/janaf_atomic.json"] = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    metadata["metal_model"] = metal_model
    if gas_model in ("janaf", "janaf_condensed"):
        metadata["standards"]["janaf_atomic_reference"] = atomic_standard_audit(temperature_k)
        metadata["standards"]["common_gas"]["policy"] = (
            "Keep the seven lower anchors and add six pinned JANAF atomic energies; no cross-phase calibration.")
        metadata["standards"]["common_gas"]["reaction_model"] = (
            "Packaged FastChem4 76 neutral gases on thirteen elements; continuous temperature evaluation")
    return record, budget, callbacks, initial, metadata


def canonical_interior_seed(record, budget, canonical, *, interior_fraction=1e-4):
    """Mix a feasible canonical ledger with 1e-4 of its metal-free LP interior.

    The fraction selects a starting point, not a lower amount bound. Exact-zero
    global elements remain absent, and all outer equilibrium audits are kept.
    """
    if (isinstance(interior_fraction, bool) or not np.isfinite(interior_fraction)
            or not 0 < interior_fraction <= 1):
        raise ValueError("The interior fraction must be finite and in (0, 1].")
    canonical = np.asarray(canonical, dtype=float)
    budget = np.asarray(budget, dtype=float)
    phases = tuple(phase for phase in record["phases"] if phase != "metal")
    problem = build_problem(record, budget, lambda t, p: np.zeros(len(canonical)), phases=phases)
    active = problem.species_indices
    excluded = np.ones(len(problem.full_species), dtype=bool)
    excluded[active] = False
    if (canonical.shape != excluded.shape or np.any(~np.isfinite(canonical))
            or np.any(canonical < 0) or np.any(canonical[excluded] != 0)
            or not np.allclose(np.asarray(problem.full_formula_matrix) @ canonical,
                               budget, rtol=1e-12, atol=0)):
        raise ValueError("The canonical seed must preserve the metal-free elemental ledger exactly.")
    scale = float(budget.sum())
    interior = scale * _feasible_start(np.asarray(problem.formula_matrix),
                                       budget[problem.element_indices] / scale)
    seed = np.zeros_like(canonical)
    seed[active] = (1. - interior_fraction) * canonical[active] + interior_fraction * interior
    return seed


def conserved_source_seed(prior_record, prior_amounts, record, budget, *, interior_fraction=1e-4):
    """Map a metal-free source into a larger catalog as a numerical seed.

    The caller must require an accepted prior result. This helper validates
    every component formula and the identical finite elemental budget; it
    neither evaluates a new equilibrium nor reuses prior chemical potentials.
    """
    if prior_record["elements"] != record["elements"]:
        raise ValueError("The prior and target elemental orders must match.")
    prior_names = [(phase, name) for phase, names in prior_record["phases"].items() for name in names]
    names = [(phase, name) for phase, values in record["phases"].items() for name in values]
    prior_amounts = np.asarray(prior_amounts, dtype=float)
    if (prior_amounts.shape != (len(prior_names),) or np.any(~np.isfinite(prior_amounts))
            or np.any(prior_amounts < 0)):
        raise ValueError("Prior component amounts must be finite and nonnegative.")
    mapped = np.zeros(len(names))
    for i, pair in enumerate(prior_names):
        if pair not in names:
            raise ValueError("The target lacks a prior component: " + str(pair))
        formula = prior_record["component_formulas"][pair[1]]
        target_formula = record["component_formulas"][pair[1]]
        if any(formula.get(e, 0.) != target_formula.get(e, 0.)
               for e in set(formula) | set(target_formula)):
            raise ValueError("The prior and target component formulas differ: " + str(pair))
        mapped[names.index(pair)] = prior_amounts[i]
    return canonical_interior_seed(record, budget, mapped, interior_fraction=interior_fraction)


def unpack_expanded_source(record, result, callbacks, temperature_k, pressure_bar):
    """Restore independently auditable gas and cloud amounts from atom carriers.

    Solver residuals still belong to the internal record. The returned public
    result contains primitive amounts and freshly evaluated energy only;
    callers preserve the internal record/result separately.
    """
    if list(record["phases"])[-1] != "atmosphere":
        raise ValueError("The internal atmosphere must be the final declared phase.")
    atmosphere = callbacks["atmosphere"]
    setup = atmosphere.setup
    carrier_names = record["phases"]["atmosphere"]
    if (tuple(record.get("atmosphere_element_order", ())) != setup.gas_setup.elements
            or len(carrier_names) != len(setup.gas_setup.elements)
            or any(record["component_formulas"][name] != {element: 1.}
                   for name, element in zip(carrier_names, setup.gas_setup.elements))):
        raise ValueError("Internal atmosphere carriers must preserve the declared element order and identity formulas.")
    n = np.asarray(result["component_amounts_mol"], dtype=float)
    names = [name for group in record["phases"].values() for name in group]
    if n.shape != (len(names),) or np.any(n < 0) or not np.all(np.isfinite(n)):
        raise ValueError("Supply finite nonnegative internal component amounts.")
    count = len(carrier_names)
    parcel = atmosphere.parcel(temperature_k, pressure_bar, n[-count:])
    public = copy.deepcopy(record)
    for name in public["phases"].pop("atmosphere"):
        del public["component_formulas"][name]
    public.pop("atmosphere_element_order")
    gas_names = list(source_gas_names(setup.gas_setup))
    cloud_names = [name + "_retained" for name in setup.condensate_species]
    public["phases"].update(gas=gas_names, cloud=cloud_names)
    public["gas_species_aliases"] = dict(zip(gas_names, setup.gas_species))
    public["retained_condensate_components"] = dict(zip(cloud_names, setup.condensate_species))
    for phase_names, phase_setup in ((gas_names, setup.gas_setup), (cloud_names, setup.condensate_setup)):
        public["component_formulas"].update({name: {element: float(phase_setup.formula_matrix[row, column])
            for row, element in enumerate(phase_setup.elements) if phase_setup.formula_matrix[row, column]}
            for column, name in enumerate(phase_names)})
    amounts = np.r_[n[:-count], parcel["gas_amounts_mol"], parcel["condensate_amounts_mol"]]
    public_callbacks = {name: callback for name, callback in callbacks.items() if name != "atmosphere"}

    def standards(t, p, condensed=False):
        q = atmosphere.element_gauge_rt
        gauge = np.asarray(q(t, p) if callable(q) else q)
        selected = setup.condensate_setup if condensed else setup.gas_setup
        return np.asarray(selected.hvector_func(t)) + np.asarray(selected.formula_matrix).T @ gauge

    public_callbacks["gas"] = ideal_phase(standards, gas=True)

    def cloud(t, p, values):
        h = standards(t, p, True)
        present = np.asarray(values) > 0
        if np.any(present & (t > np.asarray(setup.condensate_setup.temperature_validity_upper))):
            raise ValueError("A retained condensate lies outside its temperature domain.")
        return PhaseState(h, float(np.asarray(values)[present] @ h[present]))

    public_callbacks["cloud"] = cloud
    totals, atoms, energy, offset = [], [], 0., 0
    for phase, phase_names in public["phases"].items():
        values = amounts[offset:offset + len(phase_names)]
        formula = np.array([[public["component_formulas"][name].get(element, 0.) for name in phase_names]
                            for element in public["elements"]])
        totals.append(float(values.sum()))
        atoms.append((formula @ values).tolist())
        energy += public_callbacks[phase](temperature_k, pressure_bar, values).gibbs_rt
        offset += len(phase_names)
    if not np.isclose(energy, result["gibbs_rt"], rtol=5e-9, atol=1e-10 * n.sum()):
        raise ValueError("Primitive atmospheric energy does not reproduce the internal source.")
    public_result = {"accepted": bool(result.get("accepted") is True and parcel.get("accepted") is True),
                     "component_amounts_mol": amounts.tolist(), "gibbs_rt": float(energy),
                     "phase_order": list(public["phases"]), "phase_amounts_mol": totals,
                     "phase_element_amounts_mol": atoms}
    return public, public_result, public_callbacks, parcel
