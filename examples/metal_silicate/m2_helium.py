"""Finite He dissolution with the EOS-owned dry-host scalar."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from full_potential import PhaseState


HELIUM_MODELS = ("gas_only", "guillot2012_olivine", "guillot2012_morb", "guillot2012_rhyolite")


def _provider(checkout):
    path = Path(checkout).resolve() / "examples/m2_material/helium_dissolution.py"
    spec = importlib.util.spec_from_file_location("_exoeos_m2_helium", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reconstruct_helium_model(exoeos_checkout, receipt):
    """Rebuild a saved EOS scalar and verify every provider-owned recipe field."""
    provider = _provider(exoeos_checkout)
    dissolution = provider.make_helium_dissolution(
        receipt["model"], receipt["temperature_K"],
        receipt["dry_host_molar_masses_kg"], receipt["gas_standard_rt"])
    for key, value in dissolution.receipt.items():
        if receipt.get(key) != value:
            raise ValueError("Saved He provider recipe differs: " + key)
    if "gas_anchor" in receipt:
        anchor = receipt["gas_anchor"]
        standard = anchor["retained_raw_standard_rt"] + np.dot(anchor["formula"], anchor["element_gauge_rt"])
        if (anchor["species"] != "He1" or anchor["temperature_K"] != receipt["temperature_K"]
                or anchor["pressure_standard_bar"] != 1.
                or anchor["common_standard_rt"] != receipt["gas_standard_rt"]
                or standard != receipt["gas_standard_rt"]):
            raise ValueError("Saved He retained-gas anchor is inconsistent")
    return dissolution


def wrap_helium_phase(host_phase, exoeos_checkout, receipt):
    """Reconstruct the saved scalar for a fresh supplied host-phase callback.

    ``host_phase`` keeps its original native/water/H2 model identity and basis.
    He is the final added component. Linear source scenarios remain external.
    """
    dissolution = reconstruct_helium_model(exoeos_checkout, receipt)
    count = len(receipt["dry_host_molar_masses_kg"])

    def check(t, p, n):
        if t != receipt["temperature_K"] or p != receipt["pressure_bar"]:
            raise ValueError("Rebuild the He callback after changing source T/P")
        n = np.asarray(n, dtype=float)
        if n.shape != (count + 1,):
            raise ValueError("The He callback needs its complete host-plus-He basis")
        return n

    def evaluate(t, p, n):
        n = check(t, p, n)
        added = dissolution.state(n)
        if not np.any(n):
            return PhaseState(added["mu_rt"], added["gibbs_rt"])
        host = host_phase(t, p, n[:-1])
        return PhaseState(np.r_[host.mu_rt, 0.] + added["mu_rt"],
                          float(host.gibbs_rt + added["gibbs_rt"]))

    if hasattr(host_phase, "energy_value_and_grad_rt"):
        def gradient(t, p, n):
            n = check(t, p, n)
            energy, chemical_potentials = dissolution.energy_value_and_grad_rt(n)
            if not np.any(n):
                return energy, chemical_potentials
            host_energy, host_mu = host_phase.energy_value_and_grad_rt(t, p, n[:-1])
            return float(host_energy + energy), np.r_[host_mu, 0.] + chemical_potentials
        evaluate.energy_value_and_grad_rt = gradient
    return evaluate


def add_helium_silicate(record, initial, callbacks, metadata, setup, gauge,
                        inventory_path, exoeos_checkout, temperature_k, pressure_bar, model):
    """Add conserved atomic He after the existing silicate components."""
    import json

    provider = _provider(exoeos_checkout)
    names = record["phases"]["silicate"]
    if "He_dissolved" in names or "He1" not in setup.gas_species:
        raise ValueError("Require the original silicate and retained atomic-He gas")
    inventory = json.loads(Path(inventory_path).read_text())
    masses = dict(zip(inventory["elements"], inventory["atomic_masses_kg_mol"]))
    dry_masses = [0. if name in ("h2o_melts", "H2_dissolved") else
                  sum(masses[element] * amount for element, amount in record["component_formulas"][name].items())
                  for name in names]
    gas_index = list(setup.gas_species).index("He1")
    raw_standard = np.asarray(setup.gas_setup.hvector_func(temperature_k))
    formula = np.asarray(setup.gas_setup.formula_matrix)
    gas_standard = float(raw_standard[gas_index] + formula[:, gas_index] @ np.asarray(gauge))
    dissolution = provider.make_helium_dissolution(model, temperature_k, dry_masses, gas_standard)
    receipt = {**dissolution.receipt, "temperature_K": temperature_k,
               "pressure_bar": pressure_bar, "pressure_Pa": pressure_bar * 1e5,
               "host_component_order": list(names),
               "component_order": list(names) + ["He_dissolved"],
               "gas_anchor": {"species": "He1", "temperature_K": temperature_k,
                              "pressure_standard_bar": 1.,
                              "retained_raw_standard_rt": float(raw_standard[gas_index]),
                              "elements": list(setup.gas_setup.elements),
                              "formula": formula[:, gas_index].tolist(),
                              "element_gauge_rt": np.asarray(gauge).tolist(),
                              "common_standard_rt": gas_standard,
                              "recipe": "retained gas hvector(T) + formula.T @ actual source element gauge"},
               "coupling_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "gas_fugacity_policy": "Ideal retained gas: muHe = muHe_gas0 + ln(pHe / 1 bar), using the complete gas denominator."}
    flattened = [name for phase in record["phases"].values() for name in phase]
    initial = np.insert(initial, flattened.index(names[-1]) + 1, 0.)
    names.append("He_dissolved")
    record["component_formulas"]["He_dissolved"] = {"He": 1.}
    callbacks["silicate"] = wrap_helium_phase(callbacks["silicate"], exoeos_checkout, receipt)
    metadata["helium_dissolution"] = receipt
    return initial
