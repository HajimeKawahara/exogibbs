"""Shared plotting helpers for ExoGibbs curated condensate demos.

The default demo path is intentionally self-contained: it builds a small
profile-like pressure/temperature grid from native curated-family definitions
and calls the production fixed-support v2 API directly.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/exogibbs_matplotlib")

from jax import config
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from exogibbs.api.condensate import (
    CondensateEquilibriumOptions,
    CondensateEquilibriumResult,
    solve as condensate_equilibrium,
)
from exogibbs.api.gas import EquilibriumOptions, solve as equilibrium
from benchmarks.fixed_support_v2.curated_profiles import (
    CuratedProfileDefinition,
    case_id_for_profile,
    element_budget_for_profile,
    fresh_profile_definition,
    pressure_label,
    support_payload_for_profile,
)
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup

config.update("jax_enable_x64", True)

ACCEPTED_STATUSES = {"converged", "converged_with_caveat"}


def curated_output_path(script_path: str | Path) -> Path:
    """Return the standard results path for a curated demo script."""

    output_directory = REPOSITORY_ROOT / "results" / "condensates_curated_demo"
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory / Path(script_path).with_suffix(".png").name


def _pressure_label(pressure: float) -> str:
    return pressure_label(pressure)


def _case_id_for_profile(definition: CuratedProfileDefinition, temperature: float, pressure: float) -> str:
    return case_id_for_profile(definition, temperature, pressure)


def _support_payload_for_profile(
    setup: Any,
    definition: CuratedProfileDefinition,
    budget: jnp.ndarray,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    return support_payload_for_profile(setup, definition, budget)


def run_fresh_curated_profile(
    setup: Any,
    definition: CuratedProfileDefinition,
) -> tuple[list[CondensateEquilibriumResult | None], list[np.ndarray | None], list[str]]:
    """Run one fresh profile through the production fixed-support v2 API."""

    budget = element_budget_for_profile(setup, definition)
    support_indices, support_amounts_init = _support_payload_for_profile(
        setup,
        definition,
        budget,
    )
    results: list[CondensateEquilibriumResult | None] = []
    gas_plot_x: list[np.ndarray | None] = []
    errors: list[str] = []
    for temperature, pressure in zip(definition.temperatures, definition.pressures):
        options = CondensateEquilibriumOptions(
            return_diagnostics=True,
        )
        try:
            result = condensate_equilibrium(
                setup,
                float(temperature),
                float(pressure),
                budget,
                support_indices=support_indices,
                support_amounts_init=support_amounts_init,
                options=options,
            )
        except Exception as exc:  # noqa: BLE001 - demos annotate layer failures.
            result = None
            errors.append(
                f"{_case_id_for_profile(definition, temperature, pressure)}: "
                f"{type(exc).__name__}: {exc}"
            )
        results.append(result)
        if result is not None:
            result_gas_x = np.asarray(result.gas_x, dtype=float)
            if np.any(np.isfinite(result_gas_x) & (result_gas_x > 0.0)):
                gas_plot_x.append(result_gas_x)
                continue
        try:
            gas_result = equilibrium(
                setup.gas_setup,
                float(temperature),
                float(pressure),
                budget,
                Pref=1.0,
                options=EquilibriumOptions(),
            )
            gas_plot_x.append(np.asarray(gas_result.x, dtype=float))
        except Exception:  # noqa: BLE001 - gas panel can remain sparse if fallback fails.
            gas_plot_x.append(None)
    return results, gas_plot_x, errors


def _finite_positive(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.where(np.isfinite(array) & (array > 0.0), array, np.nan)


def _series(
    results: Sequence[CondensateEquilibriumResult | None],
    setup: Any,
    species: str,
    *,
    condensate: bool,
) -> np.ndarray:
    names = setup.condensate_species if condensate else setup.gas_species
    if species not in names:
        return np.full((len(results),), np.nan)
    index = names.index(species)
    values = []
    for result in results:
        if result is None:
            values.append(np.nan)
        elif condensate:
            values.append(float(result.condensate_amounts[index]))
        else:
            values.append(float(result.gas_x[index]))
    return _finite_positive(values)


def _gas_series(
    gas_plot_x: Sequence[np.ndarray | None],
    setup: Any,
    species: str,
) -> np.ndarray:
    if species not in setup.gas_species:
        return np.full((len(gas_plot_x),), np.nan)
    index = setup.gas_species.index(species)
    values = []
    for gas_x in gas_plot_x:
        if gas_x is None:
            values.append(np.nan)
        else:
            values.append(float(gas_x[index]))
    return _finite_positive(values)


def _auto_gas_species(
    gas_plot_x: Sequence[np.ndarray | None],
    setup: Any,
    preferred: Sequence[str],
    *,
    max_species: int,
) -> tuple[str, ...]:
    scores: list[tuple[float, str]] = []
    for species in setup.gas_species:
        values = _gas_series(gas_plot_x, setup, species)
        max_value = np.nanmax(values) if np.any(np.isfinite(values)) else 0.0
        if max_value > 1.0e-30:
            scores.append((float(max_value), species))
    ordered = list(preferred)
    for _, species in sorted(scores, reverse=True):
        if species not in ordered:
            ordered.append(species)
        if len(ordered) >= max_species:
            break
    return tuple(ordered)


def _auto_condensate_species(
    results: Sequence[CondensateEquilibriumResult | None],
    setup: Any,
    preferred: Sequence[str],
    *,
    max_species: int,
) -> tuple[str, ...]:
    scores: list[tuple[float, str]] = []
    for species in setup.condensate_species:
        values = _series(results, setup, species, condensate=True)
        max_value = np.nanmax(values) if np.any(np.isfinite(values)) else 0.0
        if max_value > 0.0:
            scores.append((float(max_value), species))
    ordered = list(preferred)
    for _, species in sorted(scores, reverse=True):
        if species not in ordered:
            ordered.append(species)
        if len(ordered) >= max_species:
            break
    return tuple(ordered)


def plot_curated_family(
    *,
    family: str,
    display_name: Optional[str] = None,
    preferred_gas_species: Sequence[str],
    preferred_condensates: Sequence[str],
    output_path: Path,
    title_suffix: str = "",
    max_gas_species: int = 10,
    max_condensates: int = 8,
) -> Path:
    """Run and plot one curated family through the production v2 API."""

    setup = condensate_chemical_setup(silent=True)
    definition = fresh_profile_definition(family)
    results, gas_plot_x, errors = run_fresh_curated_profile(setup, definition)
    temperatures = list(definition.temperatures)
    case_labels = [f"T={temperature:g} K" for temperature in temperatures]

    pressure_array = np.asarray(definition.pressures, dtype=float)
    order = np.argsort(pressure_array)
    pressure_array = pressure_array[order]
    temperatures = [temperatures[index] for index in order]
    case_labels = [case_labels[index] for index in order]
    results = [results[index] for index in order]
    gas_plot_x = [gas_plot_x[index] for index in order]

    gas_species = _auto_gas_species(gas_plot_x, setup, preferred_gas_species, max_species=max_gas_species)
    condensates = _auto_condensate_species(
        results,
        setup,
        preferred_condensates,
        max_species=max_condensates,
    )

    fig, (ax_gas, ax_cond) = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=True)
    cmap = plt.get_cmap("tab20")

    for species_index, species in enumerate(gas_species):
        values = _gas_series(gas_plot_x, setup, species)
        if np.any(np.isfinite(values)):
            ax_gas.plot(
                values,
                pressure_array,
                marker="o",
                ms=4,
                linewidth=1.5,
                color=cmap(species_index % 20),
                label=species,
            )

    plotted_condensate = False
    for species_index, species in enumerate(condensates):
        values = _series(results, setup, species, condensate=True)
        if np.any(np.isfinite(values)):
            plotted_condensate = True
            ax_cond.plot(
                np.clip(values, 1.0e-300, None),
                pressure_array,
                marker="o",
                ms=4,
                linewidth=1.5,
                color=cmap(species_index % 20),
                label=species,
            )
    if not plotted_condensate:
        ax_cond.text(
            0.5,
            0.5,
            "No positive condensate amount in plotted rows",
            ha="center",
            va="center",
            transform=ax_cond.transAxes,
        )

    for pressure, label in zip(pressure_array, case_labels):
        ax_gas.text(0.02, pressure, label, transform=ax_gas.get_yaxis_transform(), fontsize=7, va="center")

    for axis in (ax_gas, ax_cond):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(alpha=0.25, which="both")
    ax_gas.invert_yaxis()
    ax_gas.set_xlabel("Gas mixing ratio")
    ax_cond.set_xlabel("Condensate amount")
    ax_gas.set_ylabel("Pressure (bar)")
    suffix = f"\n{title_suffix}" if title_suffix else ""
    ax_gas.set_title(f"{display_name or family}{suffix}")
    converged_count = sum(1 for result in results if result is not None and result.converged)
    ax_cond.set_title(
        f"Fixed-support v2 layers: {len(results)}, converged: {converged_count}, errors: {len(errors)}"
    )
    if ax_gas.get_legend_handles_labels()[0]:
        ax_gas.legend(fontsize=7)
    if plotted_condensate:
        ax_cond.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path
