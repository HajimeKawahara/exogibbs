"""Finite atomic budgets with an EOS-owned chemical-species alloy scalar."""
import hashlib
import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from full_potential import PhaseState


def load_associated_provider(exoeos_checkout):
    path = Path(exoeos_checkout).resolve() / "examples/m2_material/associate_reference.py"
    name = "_exoeos_m2_associated_reference"
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
                         exoeos_checkout, temperature_k, pressure_bar, options=None):
    """Extend the finite P host by five free metals and eight O associates."""
    from exoeos import total_gex_RT, total_solution_state

    provider = load_associated_provider(exoeos_checkout)
    p = metadata["phosphorus_metal"]
    names = record["phases"]["metal"]
    if names != [element + "_metal" for element in provider.COMPONENTS[:5]]:
        raise ValueError("Associated metal requires the declared finite-P host first.")
    options = {} if options is None else options
    model, interactions = provider.make_associated_model(
        temperature_k, temperature_policy=options.get("temperature_policy", "constant"))
    gas = np.asarray(setup.gas_setup.hvector_func(temperature_k)) + np.asarray(
        setup.gas_setup.formula_matrix).T @ np.asarray(gauge)
    standards, reference = provider.associated_standards_rt(
        temperature_k, p["base_standard_potentials_rt"], dict(zip(setup.gas_species, gas)))
    # This is a numerical species simplex, not an empirical material domain.
    lower = np.r_[.75, np.zeros(17)]
    upper = np.array([1., .02, .01, .04, .02, .002, .00002, .0001, .12, .00001,
                      .003, .00002, .0001, .005, .00001, .000001, .002, .000001])
    curvature = provider.associated_curvature_lower_bound(model, temperature_k, lower, upper)
    flattened = [name for phase in record["phases"].values() for name in phase]
    insertion = flattened.index(names[-1]) + 1
    initial = np.insert(initial, insertion, np.zeros(13))
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
            return 0., np.full(18, np.nan)
        if support not in derivatives:
            indices = np.asarray(support)

            def scalar(active):
                full = jnp.zeros(18, dtype=active.dtype).at[indices].set(active)
                # Differentiate the continuous scalar on this exact support.
                # xlogy(0, 0) has no derivative along an absent component;
                # excluding that constant term avoids 0/0 AD contamination.
                ideal = jnp.sum(active*jnp.log(active/jnp.sum(active)))
                return (jnp.dot(full, standards) + ideal +
                        total_gex_RT(model, temperature_k, pressure_bar*1e5, full))

            derivatives[support] = jax.jit(jax.value_and_grad(scalar))
        energy, active_gradient = derivatives[support](n[list(support)])
        gradient = np.full(18, -np.inf)
        gradient[list(support)] = np.asarray(active_gradient)
        return energy, gradient

    alloy.energy_value_and_grad_rt = energy_value_and_grad_rt
    callbacks["metal"] = alloy
    recipe = dict(p["provider_recipe_file_sha256"])
    root = Path(exoeos_checkout).resolve()
    for path in (Path(provider.__file__), provider.DATA_PATH):
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
        "scope": "Finite conserved Mg/Ca/Al/Cr/Ti/P transfer under the declared associated scalar. Numerical curvature and inactive numerical bounds do not establish empirical coupled calibration. K metal transfer remains omitted."}
    return initial


def associated_metal_domain(metadata):
    """Return the actual eighteen-species numerical simplex and matched bound."""
    row = metadata["associated_metal"]
    return (np.asarray(row["lower_species_fractions"]),
            np.asarray(row["upper_species_fractions"]), row["curvature_lower_bound_rt"])
