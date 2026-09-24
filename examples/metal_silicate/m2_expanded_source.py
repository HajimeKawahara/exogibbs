"""Finite BSE with a retained gas/cloud atmosphere
===============================================
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np

from full_potential import PhaseState, ideal_phase
from m1_chemistry import build_setups
from m2_atmosphere import make_atmosphere_phase
from m2_common_gas import anchored_standards_rt, source_gas_names
from run_bse_common_gibbs import build_bse_problem, source_standards_rt


def build_expanded_bse_problem(inventory_path, exoeos_checkout, runtime, python_executable,
                               *, temperature_k=2173.15, pressure_bar=1.):
    """Return the finite source using seven internal atmosphere atom carriers.

    The atmosphere callback minimizes all 35 gases and 26 retained pure
    condensates at its current finite atomic allocation. Its seven amounts
    are bookkeeping coordinates, never atomic gases or extra matter.
    """
    record, budget, callbacks, initial, metadata = build_bse_problem(
        inventory_path, exoeos_checkout, runtime, python_executable,
        temperature_k=temperature_k, pressure_bar=pressure_bar, gas_model="m1_expanded")
    _, setup = build_setups()

    def gauge(t, p):
        reference, _ = source_standards_rt(t, p)
        return anchored_standards_rt(setup.gas_setup, t, reference)[1]

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
    callbacks.pop("gas")
    callbacks["atmosphere"] = atmosphere
    metadata["model_id"] = "bse_melts_ma_retained_atmosphere_conditional_v1"
    metadata["standards"]["gas_model"] = "m1_retained"
    metadata["atmosphere"] = {
        "gas_species": list(setup.gas_species), "condensate_species": list(setup.condensate_species),
        "amount_basis": "Seven finite atomic amounts, internally minimized into gas plus retained cloud.",
        "retention": "Full retention; atmospheric Fe condensate is distinct from the deep alloy.",
        "pressure": "Gas partial pressures sum to total pressure; clouds add mass without gas pressure.",
    }
    metadata["provenance"]["file_sha256"].update({name: hashlib.sha256(
        Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("m2_expanded_source.py", "m2_atmosphere.py", "m1_chemistry.py")})
    return record, budget, callbacks, initial, metadata


def unpack_expanded_source(record, result, callbacks, temperature_k, pressure_bar):
    """Restore independently auditable gas and cloud amounts from atom carriers.

    Solver residuals still belong to the internal record. The returned public
    result contains primitive amounts and freshly evaluated energy only;
    callers preserve the internal record/result separately.
    """
    if list(record["phases"])[-1] != "atmosphere":
        raise ValueError("The internal atmosphere must be the final declared phase.")
    _, setup = build_setups()
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
    parcel = callbacks["atmosphere"].parcel(temperature_k, pressure_bar, n[-7:])
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
    amounts = np.r_[n[:-7], parcel["gas_amounts_mol"], parcel["condensate_amounts_mol"]]
    public_callbacks = {name: callback for name, callback in callbacks.items() if name != "atmosphere"}

    def standards(t, p, condensed=False):
        reference, _ = source_standards_rt(t, p)
        gas, gauge = anchored_standards_rt(setup.gas_setup, t, reference)
        return (np.asarray(setup.condensate_setup.hvector_func(t))
                + np.asarray(setup.condensate_setup.formula_matrix).T @ gauge) if condensed else gas

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
