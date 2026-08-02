"""Validate the public gas solver against the analytical H/H2 system.

This restores the historical hydrogen-system comparison with the current
``exogibbs.api.gas`` interface.  It checks equilibrium amounts and reverse-mode
derivatives at one point and across temperature and pressure sweeps.
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
from exogibbs.test.analytic_hsystem import HSystem


config.update("jax_enable_x64", True)

TEMPERATURE_K = 3500.0
PRESSURE_BAR = 1.0
REFERENCE_PRESSURE_BAR = 1.0
TEMPERATURES_K = jnp.linspace(300.0, 6000.0, 121)
PRESSURES_BAR = jnp.logspace(-3.0, 3.0, 101)
ELEMENT_VECTOR = jnp.array([1.0], dtype=jnp.float64)
OPTIONS = EquilibriumOptions(epsilon_crit=1.0e-11, max_iter=1000)

POINT_ABSOLUTE_TOLERANCE = 2.0e-10
SWEEP_ABSOLUTE_TOLERANCE = 1.0e-9
BUDGET_ABSOLUTE_TOLERANCE = 2.0e-10
PLOT_FLOOR = 1.0e-30


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate ExoGibbs equilibrium and reverse-mode derivatives "
            "against the analytical H/H2 system."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "results"
            / "comparisons"
            / "comparison_with_hsystem.png"
        ),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    return parser.parse_args()


def _build_setup() -> tuple[ChemicalSetup, HSystem]:
    analytic = HSystem()

    def hvector_func(temperature: jax.Array) -> jax.Array:
        return jnp.array(
            [analytic.hv_h(temperature), analytic.hv_h2(temperature)]
        )

    setup = ChemicalSetup(
        formula_matrix=jnp.array([[1.0, 2.0]], dtype=jnp.float64),
        hvector_func=hvector_func,
        elements=("H",),
        species=("H", "H2"),
        element_vector_reference=ELEMENT_VECTOR,
        metadata={"source": "JANAF", "validation": "analytic_hsystem"},
    )
    return setup, analytic


def _require_converged(
    diagnostics: Mapping[str, jax.Array],
    *,
    label: str,
) -> None:
    converged = np.asarray(jax.device_get(diagnostics["converged"]), dtype=bool)
    residual = np.asarray(
        jax.device_get(diagnostics["final_residual"]), dtype=np.float64
    )
    if not np.all(converged):
        failed = np.flatnonzero(~converged.reshape(-1)).tolist()
        raise RuntimeError(f"{label} did not converge at indices {failed}.")
    if not np.all(np.isfinite(residual)):
        raise RuntimeError(f"{label} returned non-finite convergence residuals.")


def _solve_with_diagnostics(
    setup: ChemicalSetup,
    temperature: jax.Array,
    pressure: jax.Array,
):
    return solve(
        setup,
        temperature,
        pressure,
        ELEMENT_VECTOR,
        Pref=REFERENCE_PRESSURE_BAR,
        options=OPTIONS,
        return_diagnostics=True,
    )


def _ln_n_at_temperature(
    setup: ChemicalSetup,
    temperature: jax.Array,
) -> jax.Array:
    return solve(
        setup,
        temperature,
        PRESSURE_BAR,
        ELEMENT_VECTOR,
        Pref=REFERENCE_PRESSURE_BAR,
        options=OPTIONS,
    ).ln_n


def _ln_n_at_log_pressure(
    setup: ChemicalSetup,
    log_pressure: jax.Array,
) -> jax.Array:
    pressure = REFERENCE_PRESSURE_BAR * jnp.exp(log_pressure)
    return solve(
        setup,
        TEMPERATURE_K,
        pressure,
        ELEMENT_VECTOR,
        Pref=REFERENCE_PRESSURE_BAR,
        options=OPTIONS,
    ).ln_n


def _maximum_absolute_error(left: jax.Array, right: jax.Array) -> float:
    difference = np.asarray(jax.device_get(left - right), dtype=np.float64)
    return float(np.max(np.abs(difference)))


def _validate_budget(
    setup: ChemicalSetup,
    amounts: jax.Array,
    *,
    label: str,
) -> float:
    closure = amounts @ setup.formula_matrix.T
    error = _maximum_absolute_error(closure, ELEMENT_VECTOR)
    if not np.isfinite(error) or error > BUDGET_ABSOLUTE_TOLERANCE:
        raise RuntimeError(
            f"{label} elemental-budget error {error:.3e} exceeds "
            f"{BUDGET_ABSOLUTE_TOLERANCE:.1e}."
        )
    return error


def _positive(values: jax.Array) -> np.ndarray:
    array = np.asarray(jax.device_get(values), dtype=np.float64)
    return np.clip(np.abs(array), PLOT_FLOOR, None)


def main() -> None:
    args = _parse_args()
    setup, analytic = _build_setup()
    log_pressure = jnp.log(PRESSURE_BAR / REFERENCE_PRESSURE_BAR)

    point_result, point_diagnostics = _solve_with_diagnostics(
        setup,
        jnp.asarray(TEMPERATURE_K),
        jnp.asarray(PRESSURE_BAR),
    )
    _require_converged(point_diagnostics, label="Single-point calculation")

    point_k = analytic.compute_k(log_pressure, TEMPERATURE_K)
    point_reference = jnp.array(
        [analytic.nh(point_k), analytic.nh2(point_k)]
    )
    point_amount_error = _maximum_absolute_error(point_result.n, point_reference)
    point_budget_error = _validate_budget(
        setup, point_result.n[None, :], label="Single-point calculation"
    )

    temperature_gradient = jax.jacrev(
        lambda value: _ln_n_at_temperature(setup, value)
    )(jnp.asarray(TEMPERATURE_K))
    temperature_gradient_reference = jnp.array(
        [
            analytic.ln_nH_dT(jnp.array([TEMPERATURE_K]), log_pressure)[0],
            analytic.ln_nH2_dT(jnp.array([TEMPERATURE_K]), log_pressure)[0],
        ]
    )
    point_temperature_gradient_error = _maximum_absolute_error(
        temperature_gradient, temperature_gradient_reference
    )

    pressure_gradient = jax.jacrev(
        lambda value: _ln_n_at_log_pressure(setup, value)
    )(log_pressure)
    pressure_gradient_reference = jnp.array(
        [
            analytic.ln_nH_dlogp(
                jnp.array([TEMPERATURE_K]), log_pressure
            )[0],
            analytic.ln_nH2_dlogp(
                jnp.array([TEMPERATURE_K]), log_pressure
            )[0],
        ]
    )
    point_pressure_gradient_error = _maximum_absolute_error(
        pressure_gradient, pressure_gradient_reference
    )

    (
        temperature_result,
        temperature_diagnostics,
    ) = jax.vmap(
        lambda value: _solve_with_diagnostics(
            setup, value, jnp.asarray(PRESSURE_BAR)
        )
    )(TEMPERATURES_K)
    _require_converged(
        temperature_diagnostics, label="Temperature sweep"
    )
    temperature_budget_error = _validate_budget(
        setup, temperature_result.n, label="Temperature sweep"
    )

    temperature_k = jax.vmap(
        lambda value: analytic.compute_k(log_pressure, value)
    )(TEMPERATURES_K)
    temperature_vmr_reference = jnp.column_stack(
        (analytic.vmr_h(temperature_k), analytic.vmr_h2(temperature_k))
    )
    temperature_vmr_error = _maximum_absolute_error(
        temperature_result.x, temperature_vmr_reference
    )
    temperature_gradients = jax.vmap(
        jax.jacrev(lambda value: _ln_n_at_temperature(setup, value))
    )(TEMPERATURES_K)
    temperature_gradient_references = jnp.column_stack(
        (
            analytic.ln_nH_dT(TEMPERATURES_K, log_pressure),
            analytic.ln_nH2_dT(TEMPERATURES_K, log_pressure),
        )
    )
    temperature_gradient_error = _maximum_absolute_error(
        temperature_gradients, temperature_gradient_references
    )

    log_pressures = jnp.log(PRESSURES_BAR / REFERENCE_PRESSURE_BAR)
    pressure_result, pressure_diagnostics = jax.vmap(
        lambda value: _solve_with_diagnostics(
            setup, jnp.asarray(TEMPERATURE_K), value
        )
    )(PRESSURES_BAR)
    _require_converged(pressure_diagnostics, label="Pressure sweep")
    pressure_budget_error = _validate_budget(
        setup, pressure_result.n, label="Pressure sweep"
    )

    pressure_k = jax.vmap(
        lambda value: analytic.compute_k(value, TEMPERATURE_K)
    )(log_pressures)
    pressure_vmr_reference = jnp.column_stack(
        (analytic.vmr_h(pressure_k), analytic.vmr_h2(pressure_k))
    )
    pressure_vmr_error = _maximum_absolute_error(
        pressure_result.x, pressure_vmr_reference
    )
    pressure_gradients = jax.vmap(
        jax.jacrev(lambda value: _ln_n_at_log_pressure(setup, value))
    )(log_pressures)
    temperatures = jnp.full_like(log_pressures, TEMPERATURE_K)
    pressure_gradient_references = jnp.column_stack(
        (
            analytic.ln_nH_dlogp(temperatures, log_pressures),
            analytic.ln_nH2_dlogp(temperatures, log_pressures),
        )
    )
    pressure_gradient_error = _maximum_absolute_error(
        pressure_gradients, pressure_gradient_references
    )

    point_errors = (
        point_amount_error,
        point_temperature_gradient_error,
        point_pressure_gradient_error,
    )
    if not np.all(np.isfinite(point_errors)) or max(point_errors) > (
        POINT_ABSOLUTE_TOLERANCE
    ):
        raise RuntimeError(
            "Single-point analytical validation failed: "
            f"amount={point_amount_error:.3e}, "
            f"dln(n)/dT={point_temperature_gradient_error:.3e}, "
            f"dln(n)/dln(P)={point_pressure_gradient_error:.3e}."
        )

    sweep_errors = (
        temperature_vmr_error,
        temperature_gradient_error,
        pressure_vmr_error,
        pressure_gradient_error,
    )
    if not np.all(np.isfinite(sweep_errors)) or max(sweep_errors) > (
        SWEEP_ABSOLUTE_TOLERANCE
    ):
        raise RuntimeError(
            "Sweep analytical validation failed: "
            f"T VMR={temperature_vmr_error:.3e}, "
            f"T derivative={temperature_gradient_error:.3e}, "
            f"P VMR={pressure_vmr_error:.3e}, "
            f"P derivative={pressure_gradient_error:.3e}."
        )

    temperatures_np = np.asarray(TEMPERATURES_K)
    pressures_np = np.asarray(PRESSURES_BAR)
    temperature_x_np = np.asarray(temperature_result.x)
    temperature_ref_np = np.asarray(temperature_vmr_reference)
    pressure_x_np = np.asarray(pressure_result.x)
    pressure_ref_np = np.asarray(pressure_vmr_reference)

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))
    labels = ("H", r"H$_2$")
    colors = ("tab:orange", "tab:blue")
    for species_index, (label, color) in enumerate(zip(labels, colors)):
        axes[0, 0].plot(
            temperatures_np,
            temperature_x_np[:, species_index],
            color=color,
            label=f"{label}, ExoGibbs",
        )
        axes[0, 0].plot(
            temperatures_np,
            temperature_ref_np[:, species_index],
            color=color,
            linestyle="--",
            label=f"{label}, analytical",
        )
        axes[0, 1].plot(
            temperatures_np,
            _positive(temperature_gradients[:, species_index]),
            color=color,
            label=f"{label}, ExoGibbs AD",
        )
        axes[0, 1].plot(
            temperatures_np,
            _positive(
                temperature_gradient_references[:, species_index]
            ),
            color=color,
            linestyle="--",
            label=f"{label}, analytical",
        )
        axes[1, 0].plot(
            pressures_np,
            pressure_x_np[:, species_index],
            color=color,
            label=f"{label}, ExoGibbs",
        )
        axes[1, 0].plot(
            pressures_np,
            pressure_ref_np[:, species_index],
            color=color,
            linestyle="--",
            label=f"{label}, analytical",
        )
        axes[1, 1].plot(
            pressures_np,
            _positive(pressure_gradients[:, species_index]),
            color=color,
            label=f"{label}, ExoGibbs AD",
        )
        axes[1, 1].plot(
            pressures_np,
            _positive(pressure_gradient_references[:, species_index]),
            color=color,
            linestyle="--",
            label=f"{label}, analytical",
        )

    axes[0, 0].set(
        xlabel="Temperature (K)",
        ylabel="Volume mixing ratio",
        title=f"Composition at {PRESSURE_BAR:g} bar",
    )
    axes[0, 1].set(
        xlabel="Temperature (K)",
        ylabel=r"$|\partial\ln n/\partial T|$ (K$^{-1}$)",
        title=f"Temperature derivatives at {PRESSURE_BAR:g} bar",
    )
    axes[1, 0].set(
        xlabel="Pressure (bar)",
        ylabel="Volume mixing ratio",
        title=f"Composition at {TEMPERATURE_K:.0f} K",
    )
    axes[1, 1].set(
        xlabel="Pressure (bar)",
        ylabel=r"$|\partial\ln n/\partial\ln P|$",
        title=f"Pressure derivatives at {TEMPERATURE_K:.0f} K",
    )
    for axis in axes.flat:
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[1, 0].set_xscale("log")
    axes[1, 1].set_xscale("log")

    figure.suptitle(
        r"ExoGibbs validation against the analytical $2\mathrm{H}"
        r"\rightleftharpoons\mathrm{H}_2$ system"
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(figure)

    print("H/H2 analytical validation passed.")
    print(
        "  Single point: "
        f"amount error={point_amount_error:.3e}, "
        f"dln(n)/dT error={point_temperature_gradient_error:.3e}, "
        f"dln(n)/dln(P) error={point_pressure_gradient_error:.3e}"
    )
    print(
        "  Temperature sweep: "
        f"VMR error={temperature_vmr_error:.3e}, "
        f"derivative error={temperature_gradient_error:.3e}, "
        f"budget error={temperature_budget_error:.3e}"
    )
    print(
        "  Pressure sweep: "
        f"VMR error={pressure_vmr_error:.3e}, "
        f"derivative error={pressure_gradient_error:.3e}, "
        f"budget error={pressure_budget_error:.3e}"
    )
    print(f"  Single-point budget error={point_budget_error:.3e}")
    print(f"  Figure: {args.output}")


if __name__ == "__main__":
    main()
