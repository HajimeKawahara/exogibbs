"""Small review-only evaluators for explicitly anchored molecular candidates.

Source selection and energy alignment belong to each molecule's definition.
In particular, NASA coefficients passed here already remove their native-R
energy and insert the selected SI-R energy exactly once. Nothing here chooses
a source, changes a production preset, or interprets a fit intercept as D0.
"""

from __future__ import annotations

from dataclasses import replace
import importlib
import math

import numpy as np

from tio_candidate import _nasa

R = 8.31446261815324
PROPERTY_KEYS = ("G_J_mol", "H_J_mol", "S_J_mol_K", "Cp_J_mol_K")


def formation_g(temperature, spec: dict, xp):
    """Return G/R and its first two T derivatives from one continuous model.

    A custom molecule module implements ``formation_g(T, parameters, xp)``
    with precisely this contract. It owns its physical state conventions and
    source validation; this entry point supplies no implicit energy alignment.
    """
    kind = spec["kind"]
    if kind == "retained_energy":
        a, b, c, d, e = spec["coefficients"]
        t = temperature
        value = -a-b*t*xp.log(t)-c*t-d*t*t-e*t**3+spec["delta_GH_J_mol"]/R
        slope = -b*(xp.log(t)+1)-c-2*d*t-3*e*t*t
        curvature = -b/t-2*d-6*e*t
        return value, slope, curvature
    if kind == "nasa9":
        return _nasa(temperature, spec["coefficients"], xp)
    if kind == "custom":
        module = importlib.import_module(spec["module"])
        return module.formation_g(temperature, spec["parameters"], xp)
    raise ValueError(f"Unknown candidate representation: {kind}")


def formation_h(temperature_K: float, spec: dict) -> float:
    """Evaluate dimensionless G/(RT) only inside the declared model interval."""
    lower, upper = spec.get("temperature_range_K", (1500., 4000.))
    if not math.isfinite(temperature_K) or not lower <= temperature_K <= upper:
        raise ValueError("Temperature outside candidate interval")
    t = float(temperature_K)
    return float(formation_g(t, spec, np)[0]/t)


def formation_h_jax(temperature_K, spec: dict):
    """Evaluate the same potential with JAX; invalid temperatures yield NaN."""
    import jax.numpy as jnp

    t = jnp.asarray(temperature_K, dtype=jnp.float64)
    lower, upper = spec.get("temperature_range_K", (1500., 4000.))
    valid = jnp.isfinite(t) & (t >= lower) & (t <= upper)
    safe = jnp.where(valid, t, lower)
    return jnp.where(valid, formation_g(safe, spec, jnp)[0]/safe, jnp.nan)


def reaction_properties(temperature_K: float, spec: dict) -> dict:
    """Derive consistent formation G/H/S/Cp from the same potential."""
    formation_h(temperature_K, spec)
    t = float(temperature_K)
    g, dg, ddg = formation_g(t, spec, np)
    return dict(zip(PROPERTY_KEYS, map(float, (R*g, R*(g-t*dg), -R*dg, -R*t*ddg))))


def candidate_setup(base, replacements: dict, fraction: float = 1.0, name: str = "candidate"):
    """Replace only the declared species, including explicitly joint variants."""
    import jax

    if not replacements or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Nonempty replacements and a finite fraction in [0, 1] are required")
    columns = {species: base.species.index(species) for species in replacements}

    def hvector(t):
        original = base.hvector_func(t)
        result = original
        for species, spec in replacements.items():
            j = columns[species]
            target = formation_h_jax(t, spec)
            result = result.at[..., j].set(original[..., j]+fraction*(target-original[..., j]))
        return result

    return replace(base, hvector_func=jax.jit(hvector), metadata={
        **base.metadata, "dataset": "review-"+name,
        "production_adoption": False, "substitution_fraction": fraction,
        "replaced_species": list(replacements)})


def validate_definition(definition: dict) -> dict:
    """Check dense/off-grid scalar/JAX derivatives; source checks remain local."""
    import jax

    jax.config.update("jax_enable_x64", True)
    temperatures = np.unique(np.r_[np.arange(1500., 4001.), np.linspace(1500.137, 3999.813, 251)])
    rows = []
    for variant in definition["variants"]+definition.get("selected_variants", []):
        for species, spec in variant["replacements"].items():
            actual = np.array([list(reaction_properties(t, spec).values()) for t in temperatures])
            potential = lambda t: R*t*formation_h_jax(t, spec)
            g = np.asarray(jax.jit(jax.vmap(potential))(temperatures))
            dg = np.asarray(jax.jit(jax.vmap(jax.grad(potential)))(temperatures))
            ddg = np.asarray(jax.jit(jax.vmap(jax.grad(jax.grad(potential))))(temperatures))
            jax_properties = np.stack((g, g-temperatures*dg, -dg, -temperatures*ddg), axis=-1)
            errors = dict(zip(PROPERTY_KEYS, map(float, abs(jax_properties-actual).max(axis=0))))
            assert errors["G_J_mol"] < 2e-8 and errors["H_J_mol"] < 2e-8
            assert errors["S_J_mol_K"] < 2e-10 and errors["Cp_J_mol_K"] < 2e-10
            rows.append({"variant": variant["name"], "species": species,
                         "jax_scalar_derivative_errors": errors,
                         "samples": [{"temperature_K": t, **reaction_properties(t, spec)}
                                     for t in (1500., 2000., 2500., 3000., 4000.)]})
    return {"temperature_count": len(temperatures), "offgrid_count": 251,
            "range_K": [1500, 4000], "models": rows,
            "scope": "Numerical single-potential consistency only; source and physical accuracy require molecule-specific checks."}
