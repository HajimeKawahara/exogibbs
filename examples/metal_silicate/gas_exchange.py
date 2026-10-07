"""Finite local hydrogen exchange through the magma--gas service
===============================================================

The full Young/GCE source model conserves finite Si, Mg, O, Fe, H, Na and C
inventories across prescribed silicate, metal and gas phases. Temperature
and pressure are supplied locally. The source's four-component Fe-Si-O-H
activities and R14 dissolution closure are retained explicitly; this is
not an extension of the native ternary ExoEOS model or a phase search.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from exogibbs.applications.magma_gas import (
    MagmaGasConditions,
    MagmaGasEquilibriumState,
    MagmaGasInit,
    MagmaGasModelEvaluation,
    MagmaGasOptions,
    MagmaGasProblem,
    MagmaGasResult,
    solve as solve_magma_gas,
)
from exogibbs.thermo.models import ChemicalSetup

from reference import component_matrices, load_reference
from source import (
    make_source_standard_potentials_rt,
    source_metal_ln_gamma,
    source_reaction_offsets,
)


_PHASE_SLICES = (slice(0, 11), slice(11, 15), slice(15, 25))
_OXYGEN_INDEX = 2
_GAS_RATIO_INDICES = np.asarray([0, 1, 3, 4, 5, 6])
_OUTER_REACTION_INDICES = np.asarray([i for i in range(18) if i not in (7, 8, 9)])
_SOURCE_PHASES = {
    "silicate": ("MgO", "SiO2", "MgSiO3", "FeO", "FeSiO3", "Na2O", "Na2SiO3", "H2", "H2O", "CO", "CO2"),
    "metal": ("Fe", "Si", "O", "H"),
    "gas": ("H2", "CO", "CO2", "CH4", "O2", "H2O", "Fe", "Mg", "SiO", "Na"),
}


class GasExchangeState(NamedTuple):
    """Physical amounts and independent audits in the full source order.

    Phase rows follow silicate, metal, gas. Element columns follow
    Si, Mg, O, Fe, H, Na, C. Gas solver amounts have an arbitrary elemental
    normalization; ``component_amounts_mol`` uses the finite physical gas
    amount determined by the outer conservation equations.
    """

    component_amounts_mol: jax.Array
    mole_fractions: jax.Array
    phase_amounts_mol: jax.Array
    phase_element_amounts_mol: jax.Array
    reaction_residual: jax.Array
    element_residual: jax.Array


@dataclass(frozen=True)
class SourceGasExchangeModel:
    """Twenty-two outer roots with seven finite elemental inventories."""

    formula_matrix: jax.Array
    reaction_matrix: jax.Array
    standard_potentials_rt: Callable[[jax.Array, jax.Array], jax.Array]
    reference_root: jax.Array

    def initial_root(self, conditions: MagmaGasConditions) -> jax.Array:
        return jnp.asarray(self.reference_root, dtype=conditions.temperature_k.dtype)

    def element_abundances(
        self, conditions: MagmaGasConditions, root_variables: jax.Array,
    ) -> jax.Array:
        del conditions
        # Oxygen is only a gas normalization anchor, never an imposed buffer.
        return jnp.ones((7,), dtype=root_variables.dtype).at[_GAS_RATIO_INDICES].set(
            jnp.exp(root_variables[15:21])
        )

    def evaluate(
        self, conditions: MagmaGasConditions, root_variables: jax.Array,
        gas: MagmaGasEquilibriumState,
    ) -> MagmaGasModelEvaluation:
        budget = conditions.model_inputs
        scale = jnp.sum(budget)
        condensed_amounts = scale * jnp.exp(root_variables[:15])
        gas_amount = scale * jnp.exp(root_variables[21])
        log_silicate = root_variables[:11] - logsumexp(root_variables[:11])
        log_metal = root_variables[11:15] - logsumexp(root_variables[11:15])
        log_fractions = jnp.concatenate((log_silicate, log_metal, gas.log_mole_fractions))
        fractions = jnp.exp(log_fractions)
        amounts = jnp.concatenate((condensed_amounts, gas_amount * fractions[15:]))
        chemical_potentials = self.standard_potentials_rt(
            conditions.temperature_k, conditions.pressure_bar,
        ) + log_fractions
        chemical_potentials = chemical_potentials.at[11:15].add(source_metal_ln_gamma(
            conditions.temperature_k, conditions.pressure_bar, fractions[11:15],
        ))
        chemical_potentials = chemical_potentials.at[15:].add(
            jnp.log(conditions.pressure_bar)
        )
        reaction_residual = self.reaction_matrix @ chemical_potentials
        reaction_residual = reaction_residual + source_reaction_offsets(
            conditions.temperature_k, conditions.pressure_bar,
        )
        phase_elements = jnp.stack([
            self.formula_matrix[:, section] @ amounts[section]
            for section in _PHASE_SLICES
        ])
        balance_ratio = jnp.sum(phase_elements, axis=0) / budget
        relative_balance = balance_ratio - 1
        state = GasExchangeState(
            amounts, fractions,
            jnp.stack([jnp.sum(amounts[section]) for section in _PHASE_SLICES]),
            phase_elements, reaction_residual, relative_balance,
        )
        residual = jnp.concatenate((
            reaction_residual[_OUTER_REACTION_INDICES], jnp.log(balance_ratio),
        ))
        return MagmaGasModelEvaluation(residual, state)


def build_problem(
    record: dict[str, Any], thermochemistry_case: dict[str, Any],
    reference_case: dict[str, Any],
) -> MagmaGasProblem:
    """Build the full source model with an explicit nearby initial solution.

    The reference fixes Shomate branches and supplies initial coordinates,
    not equilibrium constraints. Positive budgets may vary during tracing.
    Exact zero inventories require a different static component support;
    use the dry ``local.py`` example for that reduced model.
    """
    species, formula, reactions = component_matrices(record)
    if tuple(record["elements"]) != ("Si", "Mg", "O", "Fe", "H", "Na", "C"):
        raise ValueError("The full source element order must be Si, Mg, O, Fe, H, Na, C.")
    if tuple(record["phases"]) != tuple(_SOURCE_PHASES) or any(
        tuple(record["phases"][phase]) != tuple(f"{name}_{phase}" for name in names)
        for phase, names in _SOURCE_PHASES.items()
    ):
        raise ValueError("Phase and component orders must match the full25 source record.")
    if (tuple(item["id"] for item in record["reactions"]) != tuple(f"R{i}" for i in range(18))
            or np.linalg.matrix_rank(formula[:, 15:]) != 7):
        raise ValueError("The source requires ordered R0--R17 and seven independent gas elements.")
    pure_gas = np.flatnonzero(np.all(reactions[:, :15] == 0, axis=1))
    if (not np.array_equal(pure_gas, [7, 8, 9])
            or np.linalg.matrix_rank(reactions[pure_gas, 15:]) != 3):
        raise ValueError("The source pure-gas reactions must be R7, R8 and R9.")
    if not np.allclose(formula @ reactions.T, 0.0, atol=1e-12, rtol=0):
        raise ValueError("Source reactions must conserve all elements.")
    amounts = np.asarray(reference_case["component_amounts_mol"], dtype=np.float64)
    budget = np.asarray(reference_case["element_amounts_mol"], dtype=np.float64)
    if tuple(reference_case["active_components"]) != species:
        raise ValueError("Initial component amounts must use the full source component order.")
    if amounts.shape != (25,) or np.any(amounts <= 0) or not np.all(np.isfinite(amounts)):
        raise ValueError("The initial full source solution must contain 25 finite positive amounts.")
    _validate_budget(budget)
    gas_elements = formula[:, 15:] @ amounts[15:]
    gas_ratios = gas_elements[_GAS_RATIO_INDICES] / gas_elements[_OXYGEN_INDEX]
    scale = budget.sum()
    reference_root = np.concatenate((
        np.log(amounts[:15] / scale), np.log(gas_ratios),
        np.atleast_1d(np.log(amounts[15:].sum() / scale)),
    ))
    standards = make_source_standard_potentials_rt(record, thermochemistry_case)
    setup = ChemicalSetup(
        formula_matrix=jnp.asarray(formula[:, 15:]),
        hvector_func=lambda temperature: standards(temperature, 1.0)[15:],
        elements=tuple(record["elements"]), species=species[15:],
        metadata={"source": "Young/GCE full25; ideal gas; frozen Shomate branches"},
    )
    model = SourceGasExchangeModel(
        jnp.asarray(formula), jnp.asarray(reactions), standards, jnp.asarray(reference_root),
    )
    return MagmaGasProblem(setup=setup, model=model)


def _validate_budget(element_amounts_mol: Any) -> None:
    if np.shape(element_amounts_mol) != (7,):
        raise ValueError("element_amounts_mol must have shape (7,).")
    if not isinstance(element_amounts_mol, jax.core.Tracer):
        values = np.asarray(element_amounts_mol)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("The full source model requires seven finite positive element amounts.")


def solve(
    problem: MagmaGasProblem, temperature_k: Any, pressure_bar: Any,
    element_amounts_mol: Any, *, options: Optional[MagmaGasOptions] = None,
    init: Optional[MagmaGasInit] = None,
) -> MagmaGasResult:
    """Solve finite exchange at supplied K/bar, retaining nested diagnostics.

    All seven inventories, including oxygen, are conserved. No redox buffer
    or planetary pressure law is imposed. Callers must maintain positive
    finite conditions while tracing and stay near the fixed source branch.
    """
    budget = jnp.asarray(element_amounts_mol)
    _validate_budget(budget)
    if problem.lnphi_func is not None:
        raise ValueError("The full source exchange example assumes ideal gas activities.")
    for name, value in (("temperature_k", temperature_k), ("pressure_bar", pressure_bar)):
        value = jnp.asarray(value)
        if value.ndim != 0:
            raise ValueError(f"{name} must be scalar.")
        if not isinstance(value, jax.core.Tracer) and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive.")
    return solve_magma_gas(
        problem, temperature_k, pressure_bar, budget,
        options=options or MagmaGasOptions(root_tolerance=1e-10), init=init,
    )


def main() -> None:
    """Run one frozen source case and report physical amounts and residuals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("source_full_2350", "source_full_3000"), default="source_full_2350",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    record = load_reference()
    reference = json.loads(Path(__file__).with_name("equilibrium_reference.json").read_text())
    case = next(item for item in reference["cases"] if item["id"] == arguments.case)
    thermo = next(item for item in record["cases"] if item["T_K"] == case["T_K"])
    problem = build_problem(record, thermo, case)
    result = solve(problem, case["T_K"], case["P_bar"], case["element_amounts_mol"])
    state = result.model_state
    reaction_error = float(jnp.max(jnp.abs(state.reaction_residual)))
    element_error = float(jnp.max(jnp.abs(state.element_residual)))
    accepted = bool(result.diagnostics.converged) and reaction_error <= 1e-8 and element_error <= 1e-9
    output = {
        "case": arguments.case,
        "model": "GCE full25 isothermal mass action; ideal silicate/gas; source Fe-Si-O-H activities",
        "scope": "Fixed phases and formal 1 bar standards with MgO extrapolation; no Gibbs minimum or phase stability claim.",
        "accepted": accepted,
        "converged": bool(result.diagnostics.converged),
        "outer_converged": bool(result.diagnostics.outer_converged),
        "inner_converged": bool(result.diagnostics.inner_converged),
        "component_amounts_mol": np.asarray(state.component_amounts_mol).tolist(),
        "phase_amounts_mol": np.asarray(state.phase_amounts_mol).tolist(),
        "phase_element_amounts_mol": np.asarray(state.phase_element_amounts_mol).tolist(),
        "maximum_reaction_residual": reaction_error,
        "maximum_relative_element_residual": element_error,
    }
    rendered = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if not accepted:
        raise SystemExit("Finite exchange failed convergence or independent residual acceptance.")


if __name__ == "__main__":
    main()
