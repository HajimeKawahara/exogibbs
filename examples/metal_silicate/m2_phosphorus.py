"""Finite phosphorus exchange with an independently supplied alloy scalar."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from full_potential import PhaseState
from run_metal_selection import METAL_LOWER, METAL_UPPER
from run_bse_common_gibbs import source_standards_rt


def load_phosphorus_provider(exoeos_checkout):
    path = Path(exoeos_checkout).resolve() / "examples/m2_material/phosphorus_reference.py"
    name = "_exoeos_m2_phosphorus_reference"
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path:
            raise ValueError("Use one phosphorus provider checkout per process.")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def add_phosphorus_metal(record, initial, callbacks, metadata, setup, gauge,
                         exoeos_checkout, temperature_k, pressure_bar, options=None):
    """Append conserved P_metal and keep all existing atmosphere candidates.

    The caller rebuilds source equations from the returned primitive record.
    P is a finite elemental component, never an extra imposed reservoir.
    Defaults explicitly select a source-published conditional continuation;
    alternative gas references and measurement shifts remain separate runs.
    """
    from exoeos import total_gex_RT, total_solution_state

    options = {} if options is None else dict(options)
    if set(options) - {"gas_reference", "temperature_policy", "standard_shift_kcal_mol"}:
        raise ValueError("Unknown phosphorus model option.")
    gas_reference = options.get("gas_reference", "P1")
    policy = options.get("temperature_policy", "constant")
    shift = options.get("standard_shift_kcal_mol", 0.)
    if isinstance(shift, bool) or not isinstance(shift, (int, float)) or not np.isfinite(shift):
        raise ValueError("Phosphorus standard shift must be a finite number.")
    provider = load_phosphorus_provider(exoeos_checkout)
    if gas_reference not in setup.gas_species or "P" not in setup.elements:
        raise ValueError("Finite P metal requires the thirteen-element gas catalog.")
    gas_standards = np.asarray(setup.gas_setup.hvector_func(temperature_k)) + np.asarray(
        setup.gas_setup.formula_matrix).T @ np.asarray(gauge)
    standard = provider.phosphorus_standard_rt(
        temperature_k, gas_standards[list(setup.gas_species).index(gas_reference)],
        gas_species=gas_reference, allow_temperature_continuation=True,
        measurement_shift_kcal_mol=shift)
    interactions = provider.phosphorus_interactions(temperature_k, temperature_policy=policy)
    model = provider.MaPhosphorusLiquid(np.asarray(interactions["epsilon"]))
    names = record["phases"]["metal"]
    if names != ["Fe_metal", "Si_metal", "O_metal", "H_metal"]:
        raise ValueError("Finite P extension requires the declared four-component Ma host.")
    source_standards, _ = source_standards_rt(temperature_k, pressure_bar)
    standards = np.r_[[source_standards[name] for name in names], standard["standard_rt"]]
    standards = standards + np.asarray(model.standard_state_shift_RT(temperature_k))
    lower, upper = np.r_[METAL_LOWER, 0.], np.r_[METAL_UPPER, .02]
    curvature = provider.phosphorus_curvature_lower_bound(model, temperature_k, lower, upper)
    flattened = [name for phase in record["phases"].values() for name in phase]
    insertion = flattened.index(names[-1]) + 1
    initial = np.insert(initial, insertion, 0.)
    names.append("P_metal")
    record["component_formulas"]["P_metal"] = {"P": 1.}
    evaluate = jax.jit(lambda n: total_solution_state(model, temperature_k, pressure_bar * 1e5, n, standards))
    derivatives = {}

    def alloy(t, p, n):
        if t != temperature_k or p != pressure_bar:
            raise ValueError("Rebuild the phosphorus callback after changing T/P.")
        n = np.asarray(n)
        if n.sum() > 0:
            model.validate_state(t, p * 1e5, n / n.sum())
        state = evaluate(n)
        return PhaseState(np.asarray(state.mu_RT), float(state.gibbs_RT))

    def energy_value_and_grad_rt(t, p, n):
        if t != temperature_k or p != pressure_bar:
            raise ValueError("Rebuild the phosphorus callback after changing T/P.")
        n = np.asarray(n)
        support = tuple(np.flatnonzero(n > 0))
        if not support:
            return 0., np.full(len(names), np.nan)
        if support not in derivatives:
            indices = np.asarray(support)

            def scalar(active):
                full = jnp.zeros(len(names), dtype=active.dtype).at[indices].set(active)
                # Differentiate only present amounts; absent entropy terms
                # vanish and have no finite derivative into their component.
                ideal = jnp.sum(active * jnp.log(active / jnp.sum(active)))
                return (jnp.dot(full, standards) + ideal +
                        total_gex_RT(model, temperature_k, pressure_bar * 1e5, full))

            derivatives[support] = jax.jit(jax.value_and_grad(scalar))
        energy, active_gradient = derivatives[support](n[list(support)])
        gradient = np.full(len(names), -np.inf)
        gradient[list(support)] = np.asarray(active_gradient)
        return energy, gradient

    alloy.energy_value_and_grad_rt = energy_value_and_grad_rt
    callbacks["metal"] = alloy
    metadata["phosphorus_metal"] = {
        "model_id": model.reference_model_id, "component_order": list(model.components),
        "standard": standard, "interactions": interactions,
        "base_standard_potentials_rt": standards.tolist(),
        "linear_scenario_offsets_applied_separately": True,
        "lower_atomic_fractions": lower.tolist(), "upper_atomic_fractions": upper.tolist(),
        "curvature_lower_bound_rt": curvature,
        "provider_file_sha256": hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
        "provider_recipe_file_sha256": {str(path.relative_to(Path(exoeos_checkout).resolve())):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                Path(provider.__file__), Path(provider.__file__).with_name("phosphorus_sources.json"),
                *(Path(exoeos_checkout).resolve() / "src/exoeos" / name for name in
                  ("ma_fe_si_o.py", "ma_fe_si_o_h.py", "ma_interval.py", "gibbs_excess.py", "solution_gibbs.py")))},
        "scope": "Finite conserved P exchange under the declared dilute-data continuation. This composition box and curvature certify only the selected scalar, not physical calibration or omitted-path errors."}
    return initial


def phosphorus_metal_domain(metadata):
    """Return the actual five-component numerical domain and matched bound."""
    row = metadata["phosphorus_metal"]
    return (np.asarray(row["lower_atomic_fractions"]),
            np.asarray(row["upper_atomic_fractions"]), row["curvature_lower_bound_rt"])
