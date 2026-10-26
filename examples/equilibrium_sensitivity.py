"""Gas equilibrium sensitivity
===========================

MCA-inspired local responses of a five-species gas equilibrium.

Run with ``python examples/equilibrium_sensitivity.py``. Plotting additionally
requires Matplotlib, NetworkX and the Graphviz ``dot`` executable. All chemical
data are packaged; no external equilibrium provider or download is needed.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Callable

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/exogibbs_matplotlib")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from exogibbs.api.gas import EquilibriumOptions, EquilibriumResult, solve
from exogibbs.presets import fastchem4
from exogibbs.thermo.models import ChemicalSetup

SPECIES = ("H2", "H2O", "CO", "CO2", "CH4")
CONTROLS = ("H", "C", "O", "K1", "K2")
BUDGET = np.array([1.0, 0.03, 0.06])
PRESSURE_BAR = 1.0
# Products minus reactants: methanation and water-gas shift, respectively.
REACTIONS = np.array([[-3., 1.], [1., -1.], [-1., -1.], [0., 1.], [1., 0.]])
LOGK_LIFT = -REACTIONS @ np.linalg.inv(REACTIONS.T @ REACTIONS)
OPTIONS = EquilibriumOptions(epsilon_crit=1.e-14, max_iter=1000)


def build_setup() -> ChemicalSetup:
    """Select five neutral gases from the packaged FastChem4 thermochemistry."""
    base = fastchem4.chemsetup(silent=True)
    columns = jnp.array([base.species.index(name) for name in
                         ("H2", "H2O1", "C1O1", "C1O2", "C1H4")])
    rows = jnp.array([base.elements.index(name) for name in ("H", "C", "O")])
    return ChemicalSetup(
        formula_matrix=base.formula_matrix[rows[:, None], columns],
        hvector_func=lambda temperature: base.hvector_func(temperature)[columns],
        elements=("H", "C", "O"), species=SPECIES,
        element_vector_reference=jnp.asarray(BUDGET),
        metadata={"source": "packaged FastChem4 five-species subset",
                  "standard_pressure_bar": 1.0},
    )


def make_state_function(setup: ChemicalSetup, budget: np.ndarray = BUDGET) -> Callable:
    """Return a traced solve with independent log-budget and log-K controls.

    ``theta[:3]`` multiplies each elemental amount by ``exp(theta)`` without
    renormalization. ``theta[3:]`` adds to ln(K1), ln(K2) at fixed temperature.
    All derivatives describe species amounts, not mole fractions.
    """
    def state(theta, temperature, pressure=PRESSURE_BAR):
        perturbed = replace(
            setup,
            hvector_func=lambda t: setup.hvector_func(t) + jnp.asarray(LOGK_LIFT) @ theta[3:],
        )
        return solve(perturbed, temperature, pressure,
                     jnp.asarray(budget) * jnp.exp(theta[:3]),
                     options=OPTIONS, return_diagnostics=True)

    return state


def require_converged(result: EquilibriumResult, diagnostics: dict) -> None:
    """Refuse to interpret a nonconverged or nonfinite equilibrium response."""
    if (not bool(diagnostics["converged"])
            or not np.isfinite(float(diagnostics["final_residual"]))
            or not np.all(np.isfinite(result.ln_n))):
        raise RuntimeError("The gas solve did not produce a finite converged equilibrium.")


def validate_response(
    setup: ChemicalSetup, state: Callable, temperature: float,
    response: np.ndarray, budget: np.ndarray = BUDGET,
) -> dict:
    """Check AD against finite differences and independent equilibrium equations."""
    zero = jnp.zeros(5)
    result, diagnostics = state(zero, temperature)
    require_converged(result, diagnostics)
    n, x, a = np.asarray(result.n), np.asarray(result.x), np.asarray(setup.formula_matrix)
    step = 1.e-4
    differences = []
    for direction in np.eye(5):
        plus, dp = state(jnp.asarray(step * direction), temperature)
        minus, dm = state(jnp.asarray(-step * direction), temperature)
        require_converged(plus, dp)
        require_converged(minus, dm)
        differences.append(np.asarray((plus.ln_n - minus.ln_n) / (2 * step)))
    finite_difference = np.stack(differences, axis=1)
    # Differentiate A n = b and S.T ln(x P/Pref) = ln(K), independently
    # of the solver's implicit JVP implementation.
    tangent = np.vstack((a * n[None, :] / budget[:, None],
                         REACTIONS.T @ (np.eye(5) - x[None, :])))
    reference = np.linalg.solve(tangent, np.eye(5))
    logk = -REACTIONS.T @ np.asarray(setup.hvector_func(temperature))
    equilibrium_residual = REACTIONS.T @ np.log(x * PRESSURE_BAR) - logk
    np.testing.assert_allclose(a @ n / budget, 1., atol=2.e-10, rtol=0)
    np.testing.assert_allclose(equilibrium_residual, 0., atol=2.e-10, rtol=0)
    np.testing.assert_allclose(response, finite_difference, atol=2.e-7, rtol=2.e-6)
    np.testing.assert_allclose(response, reference, atol=2.e-9, rtol=2.e-9)
    np.testing.assert_allclose(response[:, :3].sum(axis=1), 1., atol=2.e-9, rtol=0)
    return {
        "temperature_K": temperature,
        "budget_H_C_O": np.asarray(budget).tolist(),
        "max_fd_absolute_error": float(np.max(np.abs(response - finite_difference))),
        "max_tangent_absolute_error": float(np.max(np.abs(response - reference))),
        "max_relative_budget_error": float(np.max(np.abs(a @ n / budget - 1))),
        "max_log_equilibrium_error": float(np.max(np.abs(equilibrium_residual))),
    }


def run(output: Path) -> dict:
    """Compute, validate and render the fixed gas-only concept demonstration."""
    if not jax.config.x64_enabled:
        raise RuntimeError("Run this sensitivity example with JAX_ENABLE_X64=1.")
    setup = build_setup()
    state = jax.jit(make_state_function(setup))
    log_amounts = lambda theta, temperature: state(theta, temperature)[0].ln_n
    jacobian = jax.jit(jax.jacrev(log_amounts, argnums=0))
    thermal_response = jax.jit(jax.jacrev(log_amounts, argnums=1))
    logk_derivative = jax.jit(jax.jacrev(lambda t: -jnp.asarray(REACTIONS).T @ setup.hvector_func(t)))
    temperatures = np.linspace(600., 1600., 51)
    amounts, responses, contributions, snapshots = [], [], [], []
    checks = []
    max_thermal_error = 0.0
    max_budget_error = 0.0
    zero = jnp.zeros(5)
    for temperature in temperatures:
        result, diagnostics = state(zero, temperature)
        require_converged(result, diagnostics)
        n = np.asarray(result.n)
        response = np.asarray(jacobian(zero, temperature))
        contribution = response[:, 3:] * np.asarray(logk_derivative(temperature) * temperature)
        direct_thermal = np.asarray(thermal_response(zero, temperature) * temperature)
        np.testing.assert_allclose(contribution.sum(axis=1), direct_thermal, atol=2.e-9, rtol=2.e-9)
        np.testing.assert_allclose(np.asarray(setup.formula_matrix) @ n / BUDGET, 1., atol=2.e-10, rtol=0)
        max_thermal_error = max(max_thermal_error, float(np.max(np.abs(contribution.sum(axis=1) - direct_thermal))))
        max_budget_error = max(max_budget_error, float(np.max(np.abs(np.asarray(setup.formula_matrix) @ n / BUDGET - 1))))
        amounts.append(n)
        responses.append(response)
        contributions.append(contribution)
        if temperature in (800., 1000., 1200.):
            checks.append(validate_response(setup, state, temperature, response))
            snapshots.append({"temperature": temperature, "amounts": n, "response": response})

    # A dilute comparison shows why the enriched pedagogical mixture was chosen.
    dilute_budget = np.array([1., 2.7e-4, 4.9e-4])
    dilute_state = jax.jit(make_state_function(setup, dilute_budget))
    dilute_result, dilute_diagnostics = dilute_state(zero, 1200.)
    require_converged(dilute_result, dilute_diagnostics)
    dilute_response = np.asarray(jax.jacrev(lambda theta: dilute_state(theta, 1200.)[0].ln_n)(zero))
    checks.append(validate_response(setup, dilute_state, 1200., dilute_response, dilute_budget))
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "model": "packaged FastChem4, restricted to five ideal gases, no condensation",
        "species": SPECIES, "controls": CONTROLS, "budget_H_C_O": BUDGET.tolist(),
        "pressure_bar": PRESSURE_BAR, "reference_pressure_bar": 1.0,
        "reaction_columns": REACTIONS.tolist(), "finite_difference_log_step": 1.e-4,
        "interpretation": "Local amount responses at fixed other controls; not kinetic paths or uncertainty contributions.",
        "network_threshold": 0.15,
        "temperature_samples": len(temperatures),
        "max_thermal_chain_absolute_error": max_thermal_error,
        "max_scan_relative_budget_error": max_budget_error,
        "versions": {"jax": jax.__version__, "numpy": np.__version__},
        "checks": checks,
        "snapshots": [{"temperature_K": s["temperature"], "amounts": s["amounts"].tolist(),
                       "response": s["response"].tolist()} for s in snapshots],
        "dilute_comparison": {"budget_H_C_O": dilute_budget.tolist(), "temperature_K": 1200.,
                              "CH4_response": dilute_response[-1].tolist()},
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Preserve every coefficient, including edges suppressed by visualization.
    rows = [[t, i, j, n[i], c[i, j]] for t, n, c in zip(temperatures, amounts, responses)
            for i in range(5) for j in range(5)]
    np.savetxt(output / "response_matrices.csv", rows, delimiter=",",
               header="temperature_K,species_index,control_index,amount,response", comments="")
    from _equilibrium_sensitivity_plot import render_figures
    render_figures(output, temperatures, np.asarray(amounts), np.asarray(responses),
                   np.asarray(contributions), snapshots)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/equilibrium_sensitivity")
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
