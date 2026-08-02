"""Validate the public gas solver against the analytical H-C-O system.

This restores the historical four-species comparison for
``CO + 3 H2 <-> CH4 + H2O`` with the current ``exogibbs.api.gas`` interface.
The equilibrium composition and the elemental derivatives of ``ln(n_CO)`` are
checked independently before a compact comparison figure is written.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/exogibbs_matplotlib")

import jax
from jax import config
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.api.gas import EquilibriumOptions, solve
from exogibbs.test.analytic_hcosystem import (
    HCOSystem,
    derivative_dlnnCO_db,
    function_equilibrium,
)


config.update("jax_enable_x64", True)

TEMPERATURE_K = 1500.0
PRESSURE_BAR = 1.5
REFERENCE_PRESSURE_BAR = 1.0
ELEMENT_VECTOR = jnp.array([0.5, 0.2, 0.3], dtype=jnp.float64)
OPTIONS = EquilibriumOptions(epsilon_crit=1.0e-11, max_iter=1000)

EQUILIBRIUM_ABSOLUTE_TOLERANCE = 1.0e-10
COMPOSITION_RELATIVE_TOLERANCE = 2.0e-9
DERIVATIVE_RELATIVE_TOLERANCE = 1.0e-5
BUDGET_ABSOLUTE_TOLERANCE = 2.0e-10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate ExoGibbs equilibrium and elemental derivatives "
            "against the analytical H-C-O system."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "results"
            / "comparisons"
            / "comparison_with_hcosystem.png"
        ),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    return parser.parse_args()


def _build_setup() -> tuple[ChemicalSetup, HCOSystem]:
    analytic = HCOSystem()
    formula_matrix = jnp.array(
        [
            [2.0, 0.0, 4.0, 2.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=jnp.float64,
    )
    setup = ChemicalSetup(
        formula_matrix=formula_matrix,
        hvector_func=analytic.hv_hco,
        elements=("H", "C", "O"),
        species=("H2", "CO", "CH4", "H2O"),
        element_vector_reference=ELEMENT_VECTOR,
        metadata={"source": "JANAF", "validation": "analytic_hcosystem"},
    )
    return setup, analytic


def _require_converged(
    diagnostics: Mapping[str, jax.Array],
) -> None:
    converged = bool(np.asarray(jax.device_get(diagnostics["converged"])))
    residual = float(
        np.asarray(jax.device_get(diagnostics["final_residual"]))
    )
    if not converged or not np.isfinite(residual):
        raise RuntimeError(
            "H-C-O calculation did not converge: "
            f"converged={converged}, residual={residual:.3e}."
        )


def _bisect_analytic_co(
    equilibrium_constant: float,
    *,
    b_hydrogen: float,
    b_carbon: float,
    b_oxygen: float,
) -> float:
    lower = max(
        0.0,
        (2.0 * b_carbon + b_oxygen - 0.5 * b_hydrogen) / 3.0,
    )
    upper = min(b_carbon, b_oxygen)

    def evaluate(n_co: float) -> float:
        return float(
            function_equilibrium(
                n_co,
                equilibrium_constant,
                b_carbon,
                b_hydrogen,
                b_oxygen,
            )
        )

    lower_value = evaluate(lower)
    upper_value = evaluate(upper)
    if lower_value == 0.0:
        return lower
    if upper_value == 0.0:
        return upper
    if lower_value * upper_value > 0.0:
        raise RuntimeError(
            "The analytical H-C-O root is not bracketed in the physical "
            f"interval [{lower:.6g}, {upper:.6g}]."
        )

    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        midpoint_value = evaluate(midpoint)
        if midpoint_value == 0.0:
            return midpoint
        if lower_value * midpoint_value <= 0.0:
            upper = midpoint
            upper_value = midpoint_value
        else:
            lower = midpoint
            lower_value = midpoint_value
    return 0.5 * (lower + upper)


def _analytic_amounts(
    n_co: float,
    *,
    b_hydrogen: float,
    b_carbon: float,
    b_oxygen: float,
) -> np.ndarray:
    n_ch4 = b_carbon - n_co
    n_h2o = b_oxygen - n_co
    n_h2 = 0.5 * (b_hydrogen - 4.0 * n_ch4 - 2.0 * n_h2o)
    amounts = np.array([n_h2, n_co, n_ch4, n_h2o], dtype=np.float64)
    if np.any(~np.isfinite(amounts)) or np.any(amounts <= 0.0):
        raise RuntimeError(
            f"The analytical H-C-O solution is not physical: {amounts}."
        )
    return amounts


def main() -> None:
    args = _parse_args()
    setup, analytic = _build_setup()
    formula_matrix = np.asarray(setup.formula_matrix, dtype=np.float64)
    if np.linalg.matrix_rank(formula_matrix) != formula_matrix.shape[0]:
        raise RuntimeError("The H-C-O formula matrix is not row-full-rank.")

    result, diagnostics = solve(
        setup,
        TEMPERATURE_K,
        PRESSURE_BAR,
        ELEMENT_VECTOR,
        Pref=REFERENCE_PRESSURE_BAR,
        options=OPTIONS,
        return_diagnostics=True,
    )
    _require_converged(diagnostics)

    amounts = np.asarray(jax.device_get(result.n), dtype=np.float64)
    if np.any(~np.isfinite(amounts)) or np.any(amounts <= 0.0):
        raise RuntimeError(
            f"ExoGibbs returned non-physical H-C-O amounts: {amounts}."
        )

    budget = formula_matrix @ amounts
    budget_error = float(
        np.max(np.abs(budget - np.asarray(ELEMENT_VECTOR)))
    )
    if budget_error > BUDGET_ABSOLUTE_TOLERANCE:
        raise RuntimeError(
            f"H-C-O elemental-budget error {budget_error:.3e} exceeds "
            f"{BUDGET_ABSOLUTE_TOLERANCE:.1e}."
        )

    b_hydrogen, b_carbon, b_oxygen = (
        float(value) for value in np.asarray(ELEMENT_VECTOR)
    )
    equilibrium_constant = float(
        analytic.equilibrium_constant(
            TEMPERATURE_K, PRESSURE_BAR / REFERENCE_PRESSURE_BAR
        )
    )
    equilibrium_residual = abs(
        float(
            function_equilibrium(
                amounts[1],
                equilibrium_constant,
                b_carbon,
                b_hydrogen,
                b_oxygen,
            )
        )
    )
    if equilibrium_residual > EQUILIBRIUM_ABSOLUTE_TOLERANCE:
        raise RuntimeError(
            f"H-C-O equilibrium residual {equilibrium_residual:.3e} exceeds "
            f"{EQUILIBRIUM_ABSOLUTE_TOLERANCE:.1e}."
        )

    analytic_n_co = _bisect_analytic_co(
        equilibrium_constant,
        b_hydrogen=b_hydrogen,
        b_carbon=b_carbon,
        b_oxygen=b_oxygen,
    )
    analytic_amounts = _analytic_amounts(
        analytic_n_co,
        b_hydrogen=b_hydrogen,
        b_carbon=b_carbon,
        b_oxygen=b_oxygen,
    )
    composition_relative_error = float(
        np.max(np.abs(amounts / analytic_amounts - 1.0))
    )
    if (
        not np.isfinite(composition_relative_error)
        or composition_relative_error > COMPOSITION_RELATIVE_TOLERANCE
    ):
        raise RuntimeError(
            "H-C-O analytical composition validation failed: "
            f"maximum relative error={composition_relative_error:.3e}."
        )

    numerical_derivative = jax.jacrev(
        lambda element_vector: solve(
            setup,
            TEMPERATURE_K,
            PRESSURE_BAR,
            element_vector,
            Pref=REFERENCE_PRESSURE_BAR,
            options=OPTIONS,
        ).ln_n
    )(ELEMENT_VECTOR)[1, :]
    analytical_derivative = derivative_dlnnCO_db(
        result.ln_n[1],
        b_carbon,
        b_hydrogen,
        b_oxygen,
        equilibrium_constant,
    )
    numerical_derivative_np = np.asarray(
        jax.device_get(numerical_derivative), dtype=np.float64
    )
    analytical_derivative_np = np.asarray(
        jax.device_get(analytical_derivative), dtype=np.float64
    )
    derivative_relative_error_by_element = np.abs(
        numerical_derivative_np / analytical_derivative_np - 1.0
    )
    derivative_relative_error = float(
        np.max(derivative_relative_error_by_element)
    )
    if (
        np.any(~np.isfinite(derivative_relative_error_by_element))
        or derivative_relative_error > DERIVATIVE_RELATIVE_TOLERANCE
    ):
        raise RuntimeError(
            "H-C-O elemental-derivative validation failed: "
            f"relative errors={derivative_relative_error_by_element}."
        )

    species_labels = (r"H$_2$", "CO", r"CH$_4$", r"H$_2$O")
    element_labels = ("H", "C", "O")
    species_positions = np.arange(len(species_labels))
    element_positions = np.arange(len(element_labels))
    bar_width = 0.36

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    axes[0].bar(
        species_positions - bar_width / 2.0,
        amounts,
        width=bar_width,
        label="ExoGibbs",
        color="tab:blue",
    )
    axes[0].bar(
        species_positions + bar_width / 2.0,
        analytic_amounts,
        width=bar_width,
        label="Analytical",
        color="tab:orange",
        alpha=0.8,
    )
    axes[0].set(
        xticks=species_positions,
        xticklabels=species_labels,
        ylabel="Equilibrium amount",
        title=f"Composition at {TEMPERATURE_K:.0f} K, {PRESSURE_BAR:g} bar",
    )
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        element_positions - bar_width / 2.0,
        numerical_derivative_np,
        width=bar_width,
        label="ExoGibbs AD",
        color="tab:blue",
    )
    axes[1].bar(
        element_positions + bar_width / 2.0,
        analytical_derivative_np,
        width=bar_width,
        label="Analytical",
        color="tab:orange",
        alpha=0.8,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(
        xticks=element_positions,
        xticklabels=element_labels,
        xlabel="Elemental budget component",
        ylabel=r"$\partial\ln n_{\rm CO}/\partial b$",
        title=r"Elemental derivatives of $\ln n_{\rm CO}$",
    )
    axes[1].set_yscale("symlog", linthresh=1.0e-5)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    figure.suptitle(
        r"ExoGibbs validation for "
        r"$\mathrm{CO}+3\mathrm{H}_2\rightleftharpoons"
        r"\mathrm{CH}_4+\mathrm{H}_2\mathrm{O}$"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(figure)

    print("H-C-O analytical validation passed.")
    print(
        "  Equilibrium: "
        f"residual={equilibrium_residual:.3e}, "
        f"composition relative error={composition_relative_error:.3e}, "
        f"budget error={budget_error:.3e}"
    )
    print(
        "  dln(n_CO)/db relative errors [H, C, O]: "
        + np.array2string(
            derivative_relative_error_by_element,
            precision=3,
            suppress_small=False,
        )
    )
    print(f"  Figure: {args.output}")


if __name__ == "__main__":
    main()
