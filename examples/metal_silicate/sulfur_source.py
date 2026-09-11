"""Full sulfur/nitrogen and carbon source controls
=================================================

Reproduce the two pinned GCE mass-action networks with finite local amounts.
These fixed-phase source controls do not select sulfide saturation, establish
an integrable alloy free energy, or provide empirical partition calibration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
from scipy.optimize import least_squares


REFERENCE_PATH = Path(__file__).with_name("sulfur_reference.json")


def load_reference(path: Path = REFERENCE_PATH) -> dict[str, Any]:
    """Load the full, separately identified S/N and S-free carbon networks."""
    return json.loads(path.read_text(encoding="utf-8"))


def component_matrices(network: dict[str, Any]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Return component order, atom matrix and balanced source reaction matrix."""
    species = tuple(name for phase in network["phases"].values() for name in phase)
    formula = np.array([[network["component_formulas"][name].get(element, 0.0)
                         for name in species] for element in network["elements"]])
    reactions = np.array([[row["stoichiometry"].get(name, 0.0) for name in species]
                          for row in network["reactions"]])
    return species, formula, reactions


def make_reaction_residual(
    network: dict[str, Any], case: dict[str, Any],
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Return ``residual(ln_phase_mole_fractions, P_bar)`` at frozen case T.

    Inputs retain every source component in phase order. Each phase must have
    normalized, strictly positive fractions and the metal must contain Fe.
    This is an empirical reaction callback, not a phase activity provider.
    In particular, the source S correction depends on the silicate host.
    """
    species, _, reactions = component_matrices(network)
    nu = jnp.asarray(reactions)
    constants = jnp.asarray(case["source_delta_g_over_rt"])
    temperature = float(case["T_K"])
    indices = {name: i for i, name in enumerate(species)}
    gas = jnp.asarray([indices[name] for name in network["phases"]["gas"]])
    sulfur = network["version"] == "Sulfur_Nitrogen_Version"
    if not sulfur and network["version"] != "Carbon_Version":
        raise ValueError("Only the two pinned source versions are supported.")

    def residual(ln_mole_fractions: jax.Array, pressure_bar: jax.Array) -> jax.Array:
        ln_x = jnp.asarray(ln_mole_fractions)
        pressure = jnp.asarray(pressure_bar)
        if ln_x.shape != (len(species),) or pressure.ndim != 0:
            raise ValueError("Expected the full source component vector and scalar pressure_bar.")
        x = jnp.exp(ln_x)
        si, oxygen = x[indices["Si_metal"]], x[indices["O_metal"]]
        inverse_si, inverse_o = 1 / (1 - si), 1 / (1 - oxygen)
        cross = -5.0 * 1873.0 / temperature
        ln_si = (
            -6.65 * 1873.0 / temperature
            - 12.41 * 1873.0 / temperature * jnp.log1p(-si)
            - cross * (oxygen + jnp.log1p(-oxygen) - oxygen * inverse_si)
            + cross * oxygen**2 * si
            * (inverse_si + inverse_o + si * inverse_si**2 / 2 - 1)
        )
        ln_o = (
            4.29 - 16500.0 / temperature
            + 1873.0 / temperature * jnp.log1p(-oxygen)
            - cross * (si + jnp.log1p(-si) - si * inverse_o)
            + cross * si**2 * oxygen
            * (inverse_o + inverse_si + oxygen * inverse_o**2 / 2 - 1)
        )
        ln_c = -2.303 * 19.5 * jnp.log1p(-oxygen)
        potentials = (ln_x.at[gas].add(jnp.log(pressure))
                      .at[indices["Si_metal"]].add(ln_si)
                      .at[indices["O_metal"]].add(ln_o)
                      .at[indices["C_metal"]].add(ln_c))
        result = constants + nu @ potentials
        if sulfur:
            log_c_s = (-5.704 + 3.15 * x[indices["FeO_silicate"]]
                       + 0.12 * x[indices["MgO_silicate"]]
                       + 0.75 * x[indices["Na2O_silicate"]])
            host_correction = -2.302585093 * (-log_c_s + ln_x[indices["FeO_silicate"]])
            result = result.at[21].add(host_correction)
        else:
            # The separately pinned Carbon equation explicitly omits this
            # coefficient in its CO -> C + O reaction, unlike the S/N source.
            result = result.at[18].add(-ln_o)
        return result

    return residual


def solve_source(
    network: dict[str, Any], case: dict[str, Any], *,
    element_amounts_mol: Any = None,
    initial_component_amounts_mol: Any = None,
    pressure_bar: float | None = None,
) -> dict[str, Any]:
    """Solve the full positive source assemblage and audit absolute balances.

    All nine (S/N) or seven (Carbon) budgets must be positive. Exact zero
    inventories require a separately declared branch and are rejected here;
    no trace floor substitutes for absent elements or phases. The frozen
    thermochemistry only supports ``case['T_K']``. Pressure is local and in
    bar, with a 1 bar gas standard; neither network has the Young R14 offset.
    """
    species, formula, _ = component_matrices(network)
    budget = np.asarray(network["element_amounts_mol"] if element_amounts_mol is None
                        else element_amounts_mol, dtype=np.float64)
    initial = np.asarray(case["initial_component_amounts_mol"]
                         if initial_component_amounts_mol is None
                         else initial_component_amounts_mol, dtype=np.float64)
    pressure = float(case["P_bar"] if pressure_bar is None else pressure_bar)
    if (budget.shape != (len(network["elements"]),) or not np.all(np.isfinite(budget))
            or np.any(budget <= 0)):
        raise ValueError("The full source network requires finite, strictly positive element budgets.")
    if (initial.shape != (len(species),) or not np.all(np.isfinite(initial))
            or np.any(initial <= 0)):
        raise ValueError("Initial component amounts must be finite and strictly positive.")
    if not np.isfinite(pressure) or pressure <= 0:
        raise ValueError("pressure_bar must be finite and positive.")
    if not jax.config.x64_enabled:
        raise ValueError("The source acceptance tolerances require JAX_ENABLE_X64=1.")
    groups = [jnp.asarray([species.index(name) for name in names])
              for names in network["phases"].values()]
    chemistry = make_reaction_residual(network, case)
    scale = budget.sum()

    def residual(log_amounts):
        ln_x = log_amounts
        for group in groups:
            ln_x = ln_x.at[group].add(-logsumexp(log_amounts[group]))
        amounts = scale * jnp.exp(log_amounts)
        return jnp.concatenate((chemistry(ln_x, pressure),
                                jnp.asarray(formula) @ amounts / jnp.asarray(budget) - 1))

    compiled = jax.jit(residual)
    jacobian = jax.jit(jax.jacfwd(residual))
    root = least_squares(compiled, np.log(initial / scale), jac=jacobian,
                         bounds=(-650, 10), xtol=1e-13, ftol=1e-13,
                         gtol=1e-13, max_nfev=2000)
    amounts = scale * np.exp(root.x)
    phase_amounts = np.array([amounts[np.asarray(group)].sum() for group in groups])
    fractions = np.empty_like(amounts)
    contributions = []
    for group, phase_amount in zip(groups, phase_amounts):
        group = np.asarray(group)
        fractions[group] = amounts[group] / phase_amount
        contributions.append(formula[:, group] @ amounts[group])
    chemical = np.asarray(chemistry(np.log(fractions), pressure))
    balance = (formula @ amounts - budget) / budget
    accepted = (root.success and np.max(np.abs(chemical)) < 1e-8
                and np.max(np.abs(balance)) < 1e-9)
    if not accepted:
        raise RuntimeError(f"Source root failed independent acceptance: {root.message}")
    return {"model_id": network["model_id"], "evidence_level": "source reproduction",
            "T_K": case["T_K"], "P_bar": pressure,
            "species": list(species), "elements": network["elements"],
            "component_amounts_mol": amounts.tolist(),
            "phase_amounts_mol": phase_amounts.tolist(),
            "phase_element_amounts_mol": np.asarray(contributions).tolist(),
            "element_amounts_mol": budget.tolist(),
            "relative_element_residual": balance.tolist(),
            "reaction_residual": chemical.tolist(), "accepted": bool(accepted),
            "candidate_phases": list(network["phases"]),
            "phase_stability": "Not assessed; fixed positive source assemblage."}


if __name__ == "__main__":
    reference = load_reference()
    results = [solve_source(network, case) for network in reference["networks"].values()
               for case in network["cases"]]
    print(json.dumps({"source": reference["source"], "audit": reference["audit"],
                      "cases": results}, indent=2, allow_nan=False))
