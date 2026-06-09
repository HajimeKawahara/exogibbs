"""Shared helpers for curated support-selection condensate demos.

These demos keep the curated pressure/temperature families from
``condensates_curated_demo`` but deliberately do not pass explicit condensate
support to the public API. They are intended for auditing native support
selection behavior.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/exogibbs_matplotlib")

DEMO_DIR = Path(__file__).resolve().parents[1] / "condensates_curated_demo"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from jax import config
import matplotlib.pyplot as plt
import numpy as np

from _curated_demo_common import (  # noqa: E402
    _auto_condensate_species,
    _auto_gas_species,
    _gas_series,
    _series,
)
from exogibbs.api.condensate_equilibrium import (  # noqa: E402
    CondensateEquilibriumOptions,
    CondensateEquilibriumResult,
    condensate_equilibrium,
)
from exogibbs.api.equilibrium import EquilibriumOptions, equilibrium  # noqa: E402
from exogibbs.condensates.curated_profiles import (  # noqa: E402
    case_id_for_profile,
    element_budget_for_profile,
    fresh_profile_definition,
)
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup  # noqa: E402

config.update("jax_enable_x64", True)


def _selected_support_names(result: CondensateEquilibriumResult | None) -> tuple[str, ...]:
    if result is None or result.diagnostics is None:
        return ()
    support_selection = result.diagnostics.get("support_selection", {})
    if not isinstance(support_selection, Mapping):
        return ()
    solver_inputs = support_selection.get("solver_inputs", {})
    if isinstance(solver_inputs, Mapping) and "support_names" in solver_inputs:
        return tuple(str(name) for name in solver_inputs["support_names"])
    plan = support_selection.get("plan", {})
    if isinstance(plan, Mapping) and "positive_support_names" in plan:
        return tuple(str(name) for name in plan["positive_support_names"])
    return tuple(str(name) for name in result.condensate_support_names)


def _support_selection_payload(result: CondensateEquilibriumResult | None) -> Mapping[str, Any] | None:
    if result is None or result.diagnostics is None:
        return None
    support_selection = result.diagnostics.get("support_selection")
    return support_selection if isinstance(support_selection, Mapping) else None


def run_support_select_curated_profile(
    setup: Any,
    definition: Any,
    *,
    max_positive_support_count: int | None = None,
    max_inner_iterations: int | None = None,
) -> tuple[list[CondensateEquilibriumResult | None], list[np.ndarray | None], list[str]]:
    """Run one curated profile with native API support selection enabled."""

    budget = element_budget_for_profile(setup, definition)
    results: list[CondensateEquilibriumResult | None] = []
    gas_plot_x: list[np.ndarray | None] = []
    errors: list[str] = []
    for temperature, pressure in zip(definition.temperatures, definition.pressures):
        option_kwargs: dict[str, Any] = {}
        if max_positive_support_count is not None:
            option_kwargs["max_positive_support_count"] = int(max_positive_support_count)
        options = CondensateEquilibriumOptions(
            case_id=f"{case_id_for_profile(definition, temperature, pressure)}__support_select",
            return_diagnostics=True,
            allow_caveat_tiers=True,
            max_inner_iterations=(
                definition.max_inner_iterations
                if max_inner_iterations is None
                else int(max_inner_iterations)
            ),
            allow_empty_positive_support=True,
            enable_head_route_warm_start=True,
            enable_depleted_gas_refresh=True,
            **option_kwargs,
        )
        try:
            result = condensate_equilibrium(
                setup,
                float(temperature),
                float(pressure),
                budget,
                options=options,
            )
        except Exception as exc:  # noqa: BLE001 - demos annotate layer failures.
            result = None
            errors.append(
                f"{case_id_for_profile(definition, temperature, pressure)}: "
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


def _write_summary_json(
    *,
    output_path: Path,
    family: str,
    definition: Any,
    results: Sequence[CondensateEquilibriumResult | None],
    errors: Sequence[str],
) -> Path:
    rows = []
    status_counts = Counter("exception" if result is None else str(result.status) for result in results)
    route_counts = Counter(
        "exception" if result is None else str(result.selected_route) for result in results
    )
    support_counts = Counter()
    for layer_index, (temperature, pressure, result) in enumerate(
        zip(definition.temperatures, definition.pressures, results)
    ):
        selected_support = _selected_support_names(result)
        support_counts.update(selected_support)
        rows.append(
            {
                "layer_index": int(layer_index),
                "temperature": float(temperature),
                "pressure": float(pressure),
                "status": "exception" if result is None else str(result.status),
                "selected_route": "exception" if result is None else str(result.selected_route),
                "selected_support_names": selected_support,
                "result_support_names": ()
                if result is None
                else tuple(str(name) for name in result.condensate_support_names),
                "support_selection": _support_selection_payload(result),
            }
        )
    payload = {
        "family": family,
        "mode": "native_support_selection",
        "manual_support_from_source_profile": tuple(definition.support_species),
        "empty_condensate_support_from_source_profile": bool(definition.empty_condensate_support),
        "status_counts": dict(status_counts),
        "route_counts": dict(route_counts),
        "selected_support_frequency": dict(support_counts),
        "errors": tuple(errors),
        "layers": rows,
    }
    summary_path = output_path.with_suffix(".support_selection.json")
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def plot_support_select_family(
    *,
    family: str,
    preferred_gas_species: Sequence[str],
    preferred_condensates: Sequence[str],
    output_path: Path,
    title_suffix: str = "",
    max_gas_species: int = 10,
    max_condensates: int = 10,
    max_positive_support_count: int | None = None,
) -> tuple[Path, Path]:
    """Run and plot one curated family with native support selection enabled."""

    setup = condensate_chemical_setup(silent=True)
    definition = fresh_profile_definition(family)
    results, gas_plot_x, errors = run_support_select_curated_profile(
        setup,
        definition,
        max_positive_support_count=max_positive_support_count,
    )
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
    selected_support_by_layer = [_selected_support_names(result) for result in results]

    fig, (ax_gas, ax_cond, ax_support) = plt.subplots(
        1,
        3,
        figsize=(15.0, 5.2),
        sharey=True,
        gridspec_kw={"width_ratios": (1.0, 1.0, 0.85)},
    )
    cmap = plt.get_cmap("tab20")

    for species_index, species in enumerate(gas_species):
        values = _gas_series(gas_plot_x, setup, species)
        if np.any(np.isfinite(values)):
            ax_gas.plot(
                values,
                pressure_array,
                marker="o",
                ms=4,
                linewidth=1.4,
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
                linewidth=1.4,
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

    for layer_index, (pressure, label, support_names) in enumerate(
        zip(pressure_array, case_labels, selected_support_by_layer)
    ):
        support_label = ", ".join(support_names[:4])
        if len(support_names) > 4:
            support_label = f"{support_label}, +{len(support_names) - 4}"
        if not support_label:
            support_label = "(empty)"
        ax_support.text(
            0.02,
            pressure,
            f"{label}\n{support_label}",
            transform=ax_support.get_yaxis_transform(),
            fontsize=6.5,
            va="center",
        )

    for axis in (ax_gas, ax_cond):
        axis.set_xscale("log")
        axis.grid(alpha=0.25, which="both")
    for axis in (ax_gas, ax_cond, ax_support):
        axis.set_yscale("log")
    ax_gas.invert_yaxis()
    ax_gas.set_xlabel("Gas mixing ratio")
    ax_cond.set_xlabel("Condensate amount")
    ax_gas.set_ylabel("Pressure (bar)")
    suffix = f"\n{title_suffix}" if title_suffix else ""
    ax_gas.set_title(f"{family}{suffix}")
    converged_count = sum(1 for result in results if result is not None and result.converged)
    ax_cond.set_title(
        f"Support-select layers: {len(results)}, converged: {converged_count}, errors: {len(errors)}"
    )
    support_title = (
        "Selected support, API default"
        if max_positive_support_count is None
        else f"Selected support, top {max_positive_support_count}"
    )
    ax_support.set_title(support_title)
    ax_support.set_xlim(0.0, 1.0)
    ax_support.set_xticks([])
    ax_support.grid(alpha=0.15, axis="y")
    if ax_gas.get_legend_handles_labels()[0]:
        ax_gas.legend(fontsize=7)
    if plotted_condensate:
        ax_cond.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    summary_path = _write_summary_json(
        output_path=output_path,
        family=family,
        definition=definition,
        results=results,
        errors=errors,
    )
    return output_path, summary_path
