"""Plot ExoGibbs-only gas output for the high-temperature no-condensate family."""

from __future__ import annotations

import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/exogibbs_matplotlib")

from jax import config
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from exogibbs.api.condensate_equilibrium import (
    CondensateEquilibriumOptions,
    condensate_equilibrium,
)
from exogibbs.presets.fastchem_cond import condensate_chemical_setup

config.update("jax_enable_x64", True)

FAMILY = "solar_highT_no_condensate_gas_regression"
TEMPERATURE_K = 2200.0
PRESSURES_BAR = np.logspace(-6.0, 2.0, 18)
GAS_SPECIES = ("H2", "H1", "H2O1", "C1O1", "C1O2", "C1H4", "Mg1", "Si1", "Fe1")


def _case_id(pressure: float) -> str:
    pressure_label = f"{pressure:g}".replace(".", "p").replace("-", "m")
    return f"{FAMILY}__T{int(TEMPERATURE_K)}_P{pressure_label}"


def _run_profile(setup):
    budget = jnp.asarray(setup.gas_setup.element_vector_reference, dtype=jnp.float64)
    rows = []
    for pressure in PRESSURES_BAR:
        options = CondensateEquilibriumOptions(
            case_id=_case_id(float(pressure)),
            return_diagnostics=True,
            allow_caveat_tiers=True,
        )
        try:
            result = condensate_equilibrium(
                setup,
                TEMPERATURE_K,
                float(pressure),
                budget,
                support_indices=(),
                support_amounts_init=(),
                options=options,
            )
            rows.append((float(pressure), result, None))
        except Exception as exc:  # noqa: BLE001 - demo scripts annotate failures.
            rows.append((float(pressure), None, f"{type(exc).__name__}: {exc}"))
    return rows


def _series(rows, setup, species: str) -> np.ndarray:
    if species not in setup.gas_species:
        return np.full((len(rows),), np.nan)
    index = setup.gas_species.index(species)
    values = []
    for _, result, _ in rows:
        if result is None:
            values.append(np.nan)
        else:
            value = float(result.gas_x[index])
            values.append(value if np.isfinite(value) and value > 0.0 else np.nan)
    return np.asarray(values, dtype=float)


def _plot_profile(rows, setup) -> Path:
    pressures = np.asarray([row[0] for row in rows], dtype=float)
    output_path = Path(__file__).with_suffix(".png")
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.0))
    for species in GAS_SPECIES:
        values = _series(rows, setup, species)
        if np.any(np.isfinite(values)):
            ax.plot(values, pressures, marker="o", ms=3, label=species)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.grid(alpha=0.25, which="both")
    ax.set_xlabel("Gas mixing ratio")
    ax.set_ylabel("Pressure (bar)")
    ax.set_title(
        f"{FAMILY}\n"
        f"T = {TEMPERATURE_K:g} K, empty condensate support, failures: {sum(row[2] is not None for row in rows)}"
    )
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> None:
    setup = condensate_chemical_setup(silent=True)
    rows = _run_profile(setup)
    print(f"wrote {_plot_profile(rows, setup)}")


if __name__ == "__main__":
    main()
