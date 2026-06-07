"""Shared plotting helpers for ExoGibbs curated condensate demos.

The default demo path is intentionally self-contained: it builds a small
profile-like pressure/temperature grid from native curated-family definitions
and calls the public HEAD route API directly.  Older result-artifact replay
helpers are kept for development diagnostics, but the public demos do not
depend on ``results/``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
    CondensateEquilibriumResult,
    condensate_equilibrium,
)
from exogibbs.api.equilibrium import EquilibriumOptions, equilibrium
from exogibbs.condensates.initialization_policy import (
    recommend_budget_preserving_seed_amounts,
)
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup

config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
HEAD_ROUTE_TABLE = RESULTS / "fastchem4_milestone4381_uniform_post_solver_residual_table.json"
PAYLOAD_READINESS = RESULTS / "fastchem4_milestone4385_tier1_callsite_payload_readiness.json"
M1492_TRACE = RESULTS / "fastchem4_milestone1492_iterative_driver_frontier_expansion_trace.json"
T500_REFRESH = RESULTS / "fastchem4_milestone4379_t500_refresh_policy_live_payload_validation.json"
STATIC_FORMULA_AUDIT = RESULTS / "fastchem4_milestone002_formula_matrix_audit.json"

ACCEPTED_STATUSES = {"converged", "converged_with_caveat"}


@dataclass(frozen=True)
class CuratedProfileDefinition:
    """Small native profile definition for one curated demo family."""

    family: str
    temperatures: tuple[float, ...]
    pressures: tuple[float, ...]
    support_species: tuple[str, ...] = ()
    carbon_to_oxygen_ratio: float | None = None
    empty_condensate_support: bool = False
    seed_fraction: float = 1.0e-3
    max_seed_amount: float = 1.0e-3
    max_inner_iterations: int = 40


def _logspace(start: float, stop: float, count: int) -> tuple[float, ...]:
    return tuple(float(value) for value in np.logspace(start, stop, count))


def _linspace(start: float, stop: float, count: int) -> tuple[float, ...]:
    return tuple(float(value) for value in np.linspace(start, stop, count))


FRESH_CURATED_PROFILES: Mapping[str, CuratedProfileDefinition] = {
    "solar_highT_no_condensate_gas_regression": CuratedProfileDefinition(
        family="solar_highT_no_condensate_gas_regression",
        temperatures=tuple(2200.0 for _ in range(18)),
        pressures=_logspace(-6.0, 2.0, 18),
        empty_condensate_support=True,
    ),
    "solar_silicate_first_condensation": CuratedProfileDefinition(
        family="solar_silicate_first_condensation",
        temperatures=_linspace(1600.0, 1300.0, 9),
        pressures=_logspace(-2.0, 1.0, 9),
        support_species=("MgSiO3(s,l)", "Mg2SiO4(s,l)", "SiO2(s,l)"),
    ),
    "solar_water_condensation": CuratedProfileDefinition(
        family="solar_water_condensation",
        temperatures=_linspace(360.0, 240.0, 9),
        pressures=_logspace(-3.0, 1.0, 9),
        support_species=("H2O(s,l)",),
    ),
    "solar_metal_sulfide_or_Fe_Ni_S_region": CuratedProfileDefinition(
        family="solar_metal_sulfide_or_Fe_Ni_S_region",
        temperatures=_linspace(850.0, 600.0, 9),
        pressures=_logspace(-3.0, 1.0, 9),
        support_species=("Fe(s,l)", "FeS(s,l)", "Ni(s,l)", "NiS(s,l)"),
    ),
    "carbon_rich_graphite_window": CuratedProfileDefinition(
        family="carbon_rich_graphite_window",
        temperatures=_linspace(1500.0, 1100.0, 9),
        pressures=_logspace(-3.0, 1.0, 9),
        support_species=("C(s)",),
        carbon_to_oxygen_ratio=2.0,
    ),
    "carbon_rich_CaS_MgS_AlN_window": CuratedProfileDefinition(
        family="carbon_rich_CaS_MgS_AlN_window",
        temperatures=_linspace(850.0, 600.0, 9),
        pressures=_logspace(-3.0, 1.0, 9),
        support_species=("CaS(s)", "MgS(s)", "AlN(s)"),
        carbon_to_oxygen_ratio=2.0,
    ),
    "SiO_s_condensate_window": CuratedProfileDefinition(
        family="SiO_s_condensate_window",
        temperatures=_linspace(1050.0, 750.0, 9),
        pressures=_logspace(-2.0, 1.0, 9),
        support_species=("SiO(s)",),
    ),
    "lowT_strong_condensation_budget_stress": CuratedProfileDefinition(
        family="lowT_strong_condensation_budget_stress",
        temperatures=_linspace(600.0, 350.0, 9),
        pressures=_logspace(-3.0, 1.0, 9),
        support_species=("H2O(s,l)", "MgSiO3(s,l)", "Mg2SiO4(s,l)", "Fe(s,l)", "FeS(s,l)"),
    ),
    "near_phase_boundary_support_sensitivity": CuratedProfileDefinition(
        family="near_phase_boundary_support_sensitivity",
        temperatures=_linspace(1550.0, 1450.0, 9),
        pressures=_logspace(-1.0, 1.0, 9),
        support_species=("MgSiO3(s,l)", "Mg2SiO4(s,l)", "Fe(s,l)", "CaTiO3(s)", "TiO2(s,l)"),
    ),
    "complex_heavy_element_or_boron_titanium_zirconium_case": CuratedProfileDefinition(
        family="complex_heavy_element_or_boron_titanium_zirconium_case",
        temperatures=_linspace(1250.0, 950.0, 9),
        pressures=_logspace(-2.0, 1.0, 9),
        support_species=("TiO2(s,l)", "TiC(s,l)", "TiN(s,l)", "CaTiO3(s)"),
    ),
}


def fresh_profile_definition(family: str) -> CuratedProfileDefinition:
    """Return the native fresh-profile definition for a curated family."""

    try:
        return FRESH_CURATED_PROFILES[family]
    except KeyError as exc:
        raise ValueError(f"No fresh curated profile definition for family={family!r}.") from exc


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required curated demo evidence file is missing: {path}. "
            "Run the demo from a workspace that contains the HEAD route evidence artifacts."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _unique_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    unique: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        unique.setdefault(str(row["case_id"]), row)
    return list(unique.values())


def rows_for_family(family: str) -> list[Mapping[str, Any]]:
    """Return unique saved HEAD route rows for one curated family."""

    payload = _read_json(HEAD_ROUTE_TABLE)
    rows = [row for row in payload["rows"] if row["family"] == family]
    if not rows:
        raise ValueError(f"No curated HEAD route rows found for family={family!r}.")
    return _unique_rows(rows)


def _rows_by_case_id(path: Path) -> dict[str, list[Mapping[str, Any]]]:
    payload = _read_json(path)
    rows: dict[str, list[Mapping[str, Any]]] = {}

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            case_id = node.get("case_id")
            if isinstance(case_id, str):
                rows.setdefault(case_id, []).append(node)
            for value in node.values():
                if isinstance(value, (dict, list)):
                    visit(value)
        elif isinstance(node, list):
            for value in node:
                if isinstance(value, (dict, list)):
                    visit(value)

    visit(payload)
    return rows


@lru_cache(maxsize=1)
def _static_condensate_species_order() -> tuple[str, ...]:
    payload = _read_json(STATIC_FORMULA_AUDIT)
    return tuple(str(name) for name in payload["Ac_static"]["species_order"])


def _remap_static_payload_to_setup(payload: Mapping[str, Any], setup: Any) -> dict[str, Any] | None:
    setup_index = {name: index for index, name in enumerate(setup.condensate_species)}
    remapped_indices: list[int] = []
    for static_index in payload["support_indices"]:
        species = _static_condensate_species_order()[int(static_index)]
        if species not in setup_index:
            return None
        remapped_indices.append(setup_index[species])
    return {
        **payload,
        "support_indices": remapped_indices,
        "payload_policy": f"{payload.get('payload_policy', 'unknown')}_species_remapped",
    }


def payload_by_case_id(setup: Any) -> dict[str, dict[str, Any]]:
    """Build setup-indexed explicit support payloads when saved payloads exist."""

    payloads: dict[str, dict[str, Any]] = {}
    if PAYLOAD_READINESS.exists():
        for row in _read_json(PAYLOAD_READINESS)["rows"]:
            if row.get("payload_reconstructable"):
                payload = _remap_static_payload_to_setup(row["payload"], setup)
                if payload is not None:
                    payloads.setdefault(str(row["case_id"]), payload)
    if M1492_TRACE.exists():
        for case_id, rows in _rows_by_case_id(M1492_TRACE).items():
            for row in rows:
                if "support_indices" in row and "support_amounts_init" in row:
                    payload = _remap_static_payload_to_setup(
                        {
                            "payload_policy": "recorded_seed_payload",
                            "support_indices": row["support_indices"],
                            "support_amounts_init": row["support_amounts_init"],
                        },
                        setup,
                    )
                    if payload is not None:
                        payloads.setdefault(case_id, payload)
                    break
    return payloads


def route_evidence_by_case_id() -> dict[str, list[Mapping[str, Any]]]:
    """Load saved route-selection evidence for rows without full API payloads."""

    evidence: dict[str, list[Mapping[str, Any]]] = {}
    if not T500_REFRESH.exists():
        return evidence
    for row in _read_json(T500_REFRESH)["rows"]:
        evidence.setdefault(str(row["case_id"]), []).append(
            {
                "primary_summary": row["route_selection_report"]["primary_summary"],
                "refresh_policy_summary": row["refresh_policy_report"],
            }
        )
    return evidence


def options_for_row(
    row: Mapping[str, Any],
    t500_evidence: dict[str, list[Mapping[str, Any]]],
) -> CondensateEquilibriumOptions:
    """Build API options that replay one saved HEAD route row."""

    route = str(row["selected_route"])
    kwargs: dict[str, Any] = {
        "case_id": str(row["case_id"]),
        "return_diagnostics": True,
        "metric_status": row["metric_status"],
        "selected_route": route,
        "max_inner_iterations": 40,
        "max_positive_support_count": 1,
    }
    if route in {"m4309_promoted_high_start_callsite_policy", "m4310_full_promoted_policy_route"}:
        kwargs["head_route_primary_summary"] = {
            "row_status": "centered",
            "converged_at_final_barrier": True,
            "source": "prevalidated_head_route_evidence",
        }
    elif route == "adaptive_refresh_selector_default_depleted_refresh_budget_tradeoff":
        evidence_rows = t500_evidence[str(row["case_id"])]
        evidence = evidence_rows.pop(0) if len(evidence_rows) > 1 else evidence_rows[0]
        kwargs["head_route_primary_summary"] = evidence["primary_summary"]
        kwargs["head_route_refresh_policy_summary"] = evidence["refresh_policy_summary"]
    elif route == "fastchem4_style_electron_refresh_route":
        kwargs["head_route_primary_summary"] = {
            "row_status": "not_centered",
            "converged_at_final_barrier": False,
            "source": "prevalidated_head_route_evidence",
        }
        kwargs["head_route_refresh_policy_summary"] = {
            "accepted": True,
            "selected_policy": "fastchem4_style_electron_refresh_route",
            "source": "prevalidated_head_route_evidence",
        }
    elif route == "adaptive_floor_frontier_repair":
        kwargs["head_route_primary_summary"] = {
            "row_status": "not_centered",
            "converged_at_final_barrier": False,
            "source": "prevalidated_head_route_evidence",
        }
        kwargs["head_route_refresh_policy_summary"] = {
            "accepted": True,
            "selected_policy": "adaptive_floor_frontier_repair",
            "source": "prevalidated_head_route_evidence",
        }
    return CondensateEquilibriumOptions(**kwargs)


def temperature_pressure(case_id: str) -> tuple[float, float]:
    """Parse temperature and pressure from a curated case identifier."""

    match = re.search(r"__T([0-9]+)_P([0-9p]+)", case_id)
    if match is None:
        raise ValueError(f"Cannot parse temperature and pressure from case_id={case_id!r}.")
    return float(match.group(1)), float(match.group(2).replace("p", "."))


def element_budget_for_row(setup: Any, row: Mapping[str, Any]) -> jnp.ndarray:
    """Build the native element budget used by one curated row."""

    budget = jnp.asarray(setup.gas_setup.element_vector_reference, dtype=jnp.float64)
    element_index = {name: index for index, name in enumerate(setup.elements)}
    if row["family"] in {"carbon_rich_graphite_window", "carbon_rich_CaS_MgS_AlN_window"}:
        budget = budget.at[element_index["C"]].set(2.0 * budget[element_index["O"]])
    return budget


def element_budget_for_profile(setup: Any, definition: CuratedProfileDefinition) -> jnp.ndarray:
    """Build the native element budget used by one fresh curated profile."""

    budget = jnp.asarray(setup.gas_setup.element_vector_reference, dtype=jnp.float64)
    if definition.carbon_to_oxygen_ratio is not None:
        element_index = {name: index for index, name in enumerate(setup.elements)}
        budget = budget.at[element_index["C"]].set(
            float(definition.carbon_to_oxygen_ratio) * budget[element_index["O"]]
        )
    return budget


def _pressure_label(pressure: float) -> str:
    return f"{pressure:g}".replace(".", "p").replace("-", "m")


def _case_id_for_profile(definition: CuratedProfileDefinition, temperature: float, pressure: float) -> str:
    return f"{definition.family}__T{int(round(float(temperature)))}_P{_pressure_label(float(pressure))}"


def _support_payload_for_profile(
    setup: Any,
    definition: CuratedProfileDefinition,
    budget: jnp.ndarray,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if definition.empty_condensate_support:
        return (), ()
    species_index = {name: index for index, name in enumerate(setup.condensate_species)}
    missing = [name for name in definition.support_species if name not in species_index]
    if missing:
        raise ValueError(
            f"Fresh curated profile {definition.family!r} references unknown condensates: {missing}"
        )
    support_indices = tuple(species_index[name] for name in definition.support_species)
    seed = recommend_budget_preserving_seed_amounts(
        formula_matrix_cond=setup.formula_matrix_cond,
        element_inventory_target=budget,
        condensate_species_order=setup.condensate_species,
        support_indices=support_indices,
        seed_fraction=definition.seed_fraction,
        max_seed_amount=definition.max_seed_amount,
        min_seed_amount=1.0e-300,
        field_provenance={
            "formula_matrix_cond": "exogibbs_condensate_chemical_setup",
            "element_inventory_target": "exogibbs_fresh_curated_profile_budget",
        },
    )
    return support_indices, tuple(float(value) for value in seed.recommended_amounts)


def run_fresh_curated_profile(
    setup: Any,
    definition: CuratedProfileDefinition,
) -> tuple[list[CondensateEquilibriumResult | None], list[np.ndarray | None], list[str]]:
    """Run one fresh profile through the public HEAD route API."""

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
            case_id=_case_id_for_profile(definition, temperature, pressure),
            return_diagnostics=True,
            allow_caveat_tiers=True,
            max_inner_iterations=definition.max_inner_iterations,
            enable_head_route_warm_start=True,
            enable_depleted_gas_refresh=True,
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


def replay_row(
    setup: Any,
    row: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, Any]],
    t500_evidence: dict[str, list[Mapping[str, Any]]],
) -> CondensateEquilibriumResult:
    """Replay one saved HEAD route row through the public API."""

    temperature, pressure = temperature_pressure(str(row["case_id"]))
    payload = payloads.get(str(row["case_id"]))
    return condensate_equilibrium(
        setup,
        temperature,
        pressure,
        element_budget_for_row(setup, row),
        support_indices=None if payload is None else payload["support_indices"],
        support_amounts_init=None if payload is None else payload["support_amounts_init"],
        options=options_for_row(row, t500_evidence),
    )


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
    preferred_gas_species: Sequence[str],
    preferred_condensates: Sequence[str],
    output_path: Path,
    title_suffix: str = "",
    max_gas_species: int = 10,
    max_condensates: int = 8,
) -> Path:
    """Run and plot one curated family through the public HEAD route API."""

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
    ax_gas.set_title(f"{family}{suffix}")
    converged_count = sum(1 for result in results if result is not None and result.converged)
    ax_cond.set_title(
        f"Fresh HEAD route layers: {len(results)}, converged: {converged_count}, errors: {len(errors)}"
    )
    if ax_gas.get_legend_handles_labels()[0]:
        ax_gas.legend(fontsize=7)
    if plotted_condensate:
        ax_cond.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path
