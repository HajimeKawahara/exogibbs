"""Finite atomic budgets with an EOS-owned chemical-species alloy scalar."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from full_potential import PhaseState


def load_associated_provider(exoeos_checkout, *, potassium=False, sodium=False):
    filename = "sodium_metal.py" if sodium else "potassium_reference.py" if potassium else "associate_reference.py"
    path = Path(exoeos_checkout).resolve() / "examples/m2_material" / filename
    name = "_exoeos_m2_associated_" + ("sodium" if sodium else "potassium_reference" if potassium else "reference")
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path:
            raise ValueError("Use one associated-alloy provider checkout per process.")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def add_associated_metal(record, initial, callbacks, metadata, setup, gauge,
                         exoeos_checkout, temperature_k, pressure_bar, options=None,
                         potassium_standard_offset_rt=None, hydrogen_oxygen_model="omitted", sodium_options=None):
    """Extend the finite P host by five free metals and eight O associates."""
    from exoeos import total_gex_RT, total_solution_state

    potassium = potassium_standard_offset_rt is not None
    sodium = sodium_options is not None
    provider = load_associated_provider(exoeos_checkout, potassium=potassium, sodium=sodium)
    if sodium and (not potassium or not isinstance(sodium_options, dict)
            or set(sodium_options) != {"projection", "temperature_policy"}
            or sodium_options["projection"] not in provider.PROJECTIONS
            or sodium_options["temperature_policy"] not in provider.TEMPERATURE_POLICIES):
        raise ValueError("Finite Na requires explicit supported projection/temperature options and K standard.")
    size = len(provider.COMPONENTS)
    p = metadata["phosphorus_metal"]
    names = record["phases"]["metal"]
    if names != [element + "_metal" for element in provider.COMPONENTS[:5]]:
        raise ValueError("Associated metal requires the declared finite-P host first.")
    options = {} if options is None else options
    model, interactions = provider.make_associated_model(
        temperature_k, temperature_policy=options.get("temperature_policy", "constant"),
        hydrogen_oxygen_model=hydrogen_oxygen_model)
    gas = np.asarray(setup.gas_setup.hvector_func(temperature_k)) + np.asarray(
        setup.gas_setup.formula_matrix).T @ np.asarray(gauge)
    standard_options = ({"potassium_standard_offset_rt": potassium_standard_offset_rt}
                        if potassium else {})
    standard_provider = load_associated_provider(exoeos_checkout, potassium=potassium) if sodium else provider
    standards, reference = standard_provider.associated_standards_rt(
        temperature_k, p["base_standard_potentials_rt"], dict(zip(setup.gas_species, gas)),
        **standard_options)
    if sodium:
        # Populate the selected published host's exact-T/P native standard
        # receipt once. This evaluates a property callback, not an equilibrium.
        host = metadata["host_ledger"]
        if host.get("liquid_model") not in ("published", "published_water"):
            raise ValueError("Finite Na requires the declared published dry-host standard receipt.")
        callbacks["silicate"](temperature_k, pressure_bar, initial[:len(record["phases"]["silicate"])])
        receipts = host["native_standard_state_receipts"]
        if (len(receipts) != 1 or receipts[0]["T_K"] != temperature_k
                or receipts[0]["P_Pa"] != pressure_bar*1e5):
            raise ValueError("Finite Na requires one native standard receipt at the actual source T/P.")
        properties = copy.deepcopy(receipts[0]["native_properties"])
        sodium_standard, sodium_reference = provider.sodium_standard_rt(
            temperature_k, pressure_bar*1e5, float(standards[0]), properties, **sodium_options)
        standards = np.r_[standards, sodium_standard]
        reference.update(component_order=list(provider.COMPONENTS), component_formulas=list(provider.FORMULAS),
                         standard_potentials_rt=standards.tolist(), sodium=sodium_reference)
        metadata["sodium_metal"] = {**sodium_reference, "native_liquid_properties": properties}
    # This is a numerical species simplex, not an empirical material domain.
    lower = np.r_[.75, np.zeros(size-1)]
    upper = np.array([1., .02, .01, .04, .02, .002, .00002, .0001, .12, .00001,
                      .003, .00002, .0001, .005, .00001, .000001, .002, .000001])
    if potassium:
        upper = np.r_[upper, .02]
    if sodium:
        upper = np.r_[upper, .02]
    curvature = provider.associated_curvature_lower_bound(model, temperature_k, lower, upper)
    flattened = [name for phase in record["phases"].values() for name in phase]
    insertion = flattened.index(names[-1]) + 1
    initial = np.insert(initial, insertion, np.zeros(size-5))
    names.extend(species + "_metal" for species in provider.COMPONENTS[5:])
    for name, formula in zip(names, provider.FORMULAS):
        record["component_formulas"][name] = dict(formula)
    evaluate = jax.jit(lambda n: total_solution_state(model, temperature_k, pressure_bar*1e5, n, standards))
    derivatives = {}

    def check(t, p):
        if t != temperature_k or p != pressure_bar:
            raise ValueError("Rebuild the associated callback after changing T/P.")

    def alloy(t, p, n):
        check(t, p)
        n = np.asarray(n)
        if n.sum() > 0:
            model.validate_state(t, p*1e5, n/n.sum())
        state = evaluate(n)
        return PhaseState(np.asarray(state.mu_RT), float(state.gibbs_RT))

    def energy_value_and_grad_rt(t, p, n):
        check(t, p)
        n = np.asarray(n)
        support = tuple(np.flatnonzero(n > 0))
        if not support:
            return 0., np.full(size, np.nan)
        if support not in derivatives:
            indices = np.asarray(support)

            def scalar(active):
                full = jnp.zeros(size, dtype=active.dtype).at[indices].set(active)
                # Differentiate the continuous scalar on this exact support.
                # xlogy(0, 0) has no derivative along an absent component;
                # excluding that constant term avoids 0/0 AD contamination.
                ideal = jnp.sum(active*jnp.log(active/jnp.sum(active)))
                return (jnp.dot(full, standards) + ideal +
                        total_gex_RT(model, temperature_k, pressure_bar*1e5, full))

            derivatives[support] = jax.jit(jax.value_and_grad(scalar))
        energy, active_gradient = derivatives[support](n[list(support)])
        gradient = np.full(size, -np.inf)
        gradient[list(support)] = np.asarray(active_gradient)
        return energy, gradient

    alloy.energy_value_and_grad_rt = energy_value_and_grad_rt
    callbacks["metal"] = alloy
    recipe = dict(p["provider_recipe_file_sha256"])
    root = Path(exoeos_checkout).resolve()
    for path in (Path(provider.__file__),
                 Path(provider.__file__).with_name("associate_reference.py"),
                 Path(provider.__file__).with_name("associate_sources.json")):
        recipe[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if sodium:
        for path in (Path(standard_provider.__file__), provider.CALIBRATION_PATH):
            recipe[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if hydrogen_oxygen_model != "omitted":
        path = Path(provider.__file__).with_name("hydrogen_oxygen_sources.json")
        recipe[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata["associated_metal"] = {
        "model_id": model.reference_model_id, "component_order": list(provider.COMPONENTS),
        "component_formulas": [dict(formula) for formula in provider.FORMULAS],
        "composition_basis": "chemical_species_moles",
        "atom_counts_per_component": [sum(formula.values()) for formula in provider.FORMULAS],
        "standards": reference, "interactions": interactions,
        "lower_species_fractions": lower.tolist(), "upper_species_fractions": upper.tolist(),
        "curvature_lower_bound_rt": curvature, "provider_recipe_file_sha256": recipe,
        "linear_scenario_offsets_applied_separately": True,
        "scenario_offset_scope": "Existing named O_metal/H_metal offsets change those free atomic species only. They are not automatically applied to MO/M2O species, whose independent Jung/gas-anchored standards are retained.",
        "scope": "Finite conserved Mg/Ca/Al/Cr/Ti/P transfer under the declared associated scalar. Numerical curvature and inactive numerical bounds do not establish empirical coupled calibration. " + ("K has an explicit uncalibrated finite-standard sensitivity." if potassium else "K metal transfer remains omitted.")}
    if sodium:
        metadata["associated_metal"]["scope"] += " Na follows the explicit projected native-standard continuation; its numerical box is not an empirical bound."
    if potassium:
        metadata["potassium_metal"] = reference["potassium"]
    return initial


def associated_metal_domain(metadata):
    """Return the actual selected species numerical simplex and matched bound."""
    row = metadata["associated_metal"]
    return (np.asarray(row["lower_species_fractions"]),
            np.asarray(row["upper_species_fractions"]), row["curvature_lower_bound_rt"])
