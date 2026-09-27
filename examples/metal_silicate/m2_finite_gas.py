"""Finite background-element gases on the conditional JANAF reference
===================================================================

The opt-in catalog adds every neutral Al/Ca/K/Ti/Cr/P gas in the packaged
FastChem table to the existing 35 gases. ``janaf`` retains the original 26
condensates; ``janaf_condensed`` also includes the 41 neutral background-element
condensates. Alloy components remain unchanged. Atomic references align
conventions, not materials or independently tabulated pure phases.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from exogibbs.api.condensate import build_condensate_chemical_setup
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup

from m1_chemistry import CONDENSATE_SPECIES, ELEMENTS, EXPANDED_GAS_SPECIES, build_setups, subset_setup
from m2_common_gas import anchored_standards_rt
from m2_janaf import atomic_standard_audit
from m2_omitted_gas import OMITTED_ELEMENTS


FINITE_ELEMENTS = ELEMENTS + OMITTED_ELEMENTS


def build_atmosphere_setup(gas_model="m1"):
    """Return the explicit retained catalog; the historical model is default."""
    if gas_model == "m1":
        return build_setups()[1]
    if gas_model not in ("janaf", "janaf_condensed"):
        raise ValueError("gas_model must be m1, janaf, or janaf_condensed.")
    full = condensate_chemical_setup(silent=True)
    gas = full.gas_setup
    def additions(selected):
        matrix = np.asarray(selected.formula_matrix)
        excluded = [i for i, name in enumerate(selected.elements) if name not in FINITE_ELEMENTS]
        background = [selected.elements.index(name) for name in OMITTED_ELEMENTS]
        columns = np.flatnonzero(np.all(matrix[excluded] == 0, axis=0)
                                & np.any(matrix[background] > 0, axis=0))
        return tuple(selected.species[i] for i in columns)

    names = EXPANDED_GAS_SPECIES + additions(gas)
    clouds = CONDENSATE_SPECIES + (additions(full.condensate_setup)
                                    if gas_model == "janaf_condensed" else ())
    return build_condensate_chemical_setup(
        gas_setup=subset_setup(gas, names, elements=FINITE_ELEMENTS),
        condensate_setup=subset_setup(full.condensate_setup, clouds,
                                     elements=FINITE_ELEMENTS),
    )


def atmosphere_gauge_rt(setup, temperature_k, reference_standards):
    """Keep the seven source anchors; add six absolute JANAF atomic energies."""
    shared = subset_setup(setup.gas_setup, EXPANDED_GAS_SPECIES)
    _, original = anchored_standards_rt(shared, temperature_k, reference_standards)
    if tuple(setup.elements) == ELEMENTS:
        return original
    if tuple(setup.elements) != FINITE_ELEMENTS:
        raise ValueError("Unknown finite-atmosphere element order.")
    audit = atomic_standard_audit(temperature_k)
    raw = np.asarray(setup.gas_setup.hvector_func(temperature_k))
    atoms = [setup.gas_species.index(element + "1") for element in OMITTED_ELEMENTS]
    if not np.array_equal(raw[atoms], np.zeros(len(atoms))):
        raise ValueError("The adopted JANAF atomic references require atomic-zero FastChem standards.")
    return np.r_[original, [audit["atomic_standards"][name]["value_rt"]
                           for name in OMITTED_ELEMENTS]]


def catalog_sha256(setup):
    """Pin ordered species, elemental formulas, and condensate validity."""
    payload = {"elements": list(setup.elements)}
    for phase, selected in (("gas", setup.gas_setup), ("condensate", setup.condensate_setup)):
        payload[phase] = {"species": list(selected.species),
                          "formula_matrix": np.asarray(selected.formula_matrix).tolist(),
                          "temperature_validity_upper": selected.temperature_validity_upper}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
