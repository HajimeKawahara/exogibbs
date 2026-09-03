"""Compare propagated ExoGibbs and FastChem rainout with Ito et al. (2025).

Ito Layer 1 is used only as the lower-boundary gas inventory.  The shared
H/O/Si network is then solved recursively from Layer 2 upward: ExoGibbs uses
``rainout=True`` and FastChem uses its native ``cr`` calculation mode.  Layer
1 itself is not re-solved because Ito's ground layer includes the separate
magma-contact water-solubility condition and excludes condensed SiO.

The workbook is ordered from the ground upward.  FastChem consumes that order
directly.  ExoGibbs profile inputs and outputs follow its top-to-bottom API,
so this example reverses Layers 2+ before and after the bottom scan.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/exogibbs_matplotlib")

import jax
from jax import config
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from benchmarks.fastchem4.comparison import align_species_values
from benchmarks.fastchem4.fastchem_executable import run_fastchem_executable
from examples.comparisons.comparison_with_ito_2025 import (
    COMMON_GAS_SPECIES,
    CONDENSATE_SPECIES,
    ELEMENTS,
    EXOGIBBS_GAS_SPECIES,
    FIXED_POINT_MAX_ITERATIONS,
    FIXED_POINT_RTOL,
    HE_TO_H2_RATIO,
    ITO_SPECIES,
    ItoProfile,
    PLOT_FLOOR,
    _convert_reactive_gas,
    _difference_summary,
    _ito_plateau_stops,
    _plateau_summary,
    _plot_values,
    _sha256,
    _validate_fastchem_catalog,
    _write_fastchem_abundance,
    _write_filtered_fastchem_inputs,
    build_ito_exogibbs_setup,
    load_ito_profile,
    reactive_element_abundances,
)
from exogibbs.api.condensate import (
    CondensateEquilibriumOptions,
    solve_profile as solve_condensate_profile,
)
from exogibbs.equilibrium.condensate.acceptance import (
    least_squares_element_potential,
)
from exogibbs.equilibrium.condensate.policy import (
    fixed_support_v2_production_policy,
)


config.update("jax_enable_x64", True)

DEFAULT_INPUT = REPOSITORY_ROOT / "external_data" / "Ito_2025.xlsx"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "ito_2025_rainout"
DEFAULT_FIGURE = DEFAULT_OUTPUT_ROOT / "ito_2025_rainout_comparison.png"
DEFAULT_TABLE = DEFAULT_OUTPUT_ROOT / "ito_2025_rainout_comparison.csv"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_ROOT / "ito_2025_rainout_comparison.json"
DEFAULT_ARCHIVE = DEFAULT_OUTPUT_ROOT / "ito_2025_rainout_comparison.npz"
SUMMARY_SCHEMA = "exogibbs_ito_2025_propagated_rainout_v2"
SIO_CONDENSATE = "SiO(s)"
SIO_ACTIVE_LOG_SATURATION_TOLERANCE = 1.0e-5
INVENTORY_RELATIVE_MISMATCH_TOLERANCE = 1.0e-3


@dataclass(frozen=True)
class RainoutSolution:
    """One solver's propagated profile in Ito's ground-to-top order."""

    gas_fractions: np.ndarray
    atomic_gas_fractions: np.ndarray
    condensate_amounts: np.ndarray
    reactive_pressure_bar: np.ndarray
    converged: np.ndarray
    status: tuple[str, ...]
    acceptance_tier: tuple[str, ...]
    fixed_point_iterations: int
    solver_iterations: np.ndarray
    element_inventory_target: np.ndarray | None = None
    gas_element_inventory: np.ndarray | None = None
    element_inventory_out: np.ndarray | None = None
    abundance_scale: np.ndarray | None = None
    elements_conserved: np.ndarray | None = None
    sio_log_saturation_ratio: np.ndarray | None = None
    sio_saturation_ratio: np.ndarray | None = None
    sio_support_active: np.ndarray | None = None
    sio_condensate_positive: np.ndarray | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Propagate the Ito Layer 1 gas inventory through ExoGibbs and "
            "native FastChem rainout calculations."
        )
    )
    parser.add_argument(
        "--fastchem-executable",
        required=True,
        type=Path,
        help="Path to the audited FastChem 4 standalone executable.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Optional bottom-up Layer 2+ limit for smoke tests.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after writing all outputs.",
    )
    return parser.parse_args()


def _target_profile(profile: ItoProfile, max_layers: int | None) -> ItoProfile:
    available = profile.layer.size - 1
    if max_layers is None:
        count = available
    else:
        if isinstance(max_layers, bool) or max_layers <= 0:
            raise ValueError("max_layers must be a positive integer.")
        count = min(int(max_layers), available)
    stop = 1 + count
    return ItoProfile(
        layer=profile.layer[1:stop].copy(),
        pressure_bar=profile.pressure_bar[1:stop].copy(),
        temperature_k=profile.temperature_k[1:stop].copy(),
        gas_fractions=profile.gas_fractions[1:stop].copy(),
        atomic_fractions=profile.atomic_fractions[1:stop].copy(),
    )


def _initial_reactive_pressures(target: ItoProfile) -> np.ndarray:
    helium = target.gas_fractions[:, ITO_SPECIES.index("He")]
    pressures = target.pressure_bar * (1.0 - helium)
    if np.any(~np.isfinite(pressures)) or np.any(pressures <= 0.0):
        raise ValueError("Initial reactive pressures must be finite and positive.")
    return pressures


def _profile_pressure_converged(
    current: np.ndarray,
    updated: np.ndarray,
) -> tuple[bool, float]:
    maximum = float(np.max(np.abs(np.log(updated / current))))
    return maximum <= FIXED_POINT_RTOL, maximum


def _convert_reactive_profile(
    fractions: np.ndarray,
    species: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_rows = []
    atomic_rows = []
    pressure_fractions = []
    for row in np.asarray(fractions, dtype=np.float64):
        total, atomic, pressure_fraction = _convert_reactive_gas(row, species)
        total_rows.append(total)
        atomic_rows.append(atomic)
        pressure_fractions.append(pressure_fraction)
    return (
        np.asarray(total_rows, dtype=np.float64),
        np.asarray(atomic_rows, dtype=np.float64),
        np.asarray(pressure_fractions, dtype=np.float64),
    )


def _mask_exactly_depleted_gases(
    gas_fractions: np.ndarray,
    element_inventory_target: np.ndarray,
    formula_matrix: np.ndarray,
) -> np.ndarray:
    """Remove gases containing an exactly depleted element and renormalize."""

    fractions = np.asarray(gas_fractions, dtype=np.float64)
    inventory = np.asarray(element_inventory_target, dtype=np.float64)
    formula = np.asarray(formula_matrix, dtype=np.float64)
    if fractions.ndim != 2 or inventory.ndim != 2 or formula.ndim != 2:
        raise ValueError("Gas fractions, inventories, and formula matrix must be 2D.")
    if fractions.shape[0] != inventory.shape[0]:
        raise ValueError("Gas fractions and inventories must have the same rows.")
    if formula.shape != (inventory.shape[1], fractions.shape[1]):
        raise ValueError("Formula matrix shape is inconsistent with the profile.")
    if (
        np.any(~np.isfinite(fractions))
        or np.any(fractions < 0.0)
        or np.any(~np.isfinite(inventory))
        or np.any(inventory < 0.0)
    ):
        raise ValueError("Gas fractions and inventories must be finite and non-negative.")

    masked = fractions.copy()
    absent_elements = inventory == 0.0
    species_use_element = formula > 0.0
    remove_species = np.any(
        absent_elements[:, :, None] & species_use_element[None, :, :],
        axis=1,
    )
    masked[remove_species] = 0.0
    row_sums = np.sum(masked, axis=1)
    if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
        raise RuntimeError(
            "Exact-depletion masking removed every reactive gas in a layer."
        )
    return masked / row_sums[:, None]


def _exact_depletion_summary(
    layers: np.ndarray,
    element_inventory_target: np.ndarray,
) -> dict[str, Any]:
    """Summarize the first exact propagated depletion in ground-up order."""

    layer_values = np.asarray(layers, dtype=np.int64)
    inventory = np.asarray(element_inventory_target, dtype=np.float64)
    if inventory.shape != (layer_values.size, len(ELEMENTS)):
        raise ValueError("Unexpected element-inventory profile shape.")
    by_element: dict[str, int | None] = {}
    first_indices = []
    for index, element in enumerate(ELEMENTS):
        depleted = np.flatnonzero(inventory[:, index] == 0.0)
        if depleted.size:
            first_index = int(depleted[0])
            first_indices.append(first_index)
            by_element[element] = int(layer_values[first_index])
        else:
            by_element[element] = None
    first_index = min(first_indices) if first_indices else None
    first_layer = None if first_index is None else int(layer_values[first_index])
    first_elements = (
        []
        if first_index is None
        else [
            element
            for element, layer in by_element.items()
            if layer == first_layer
        ]
    )
    return {
        "first_exact_depletion_layer": first_layer,
        "first_exact_depletion_elements": first_elements,
        "first_exact_depletion_layer_by_element": by_element,
    }


def _sio_saturation_diagnostics(
    setup,
    temperatures: np.ndarray,
    reactive_pressures: np.ndarray,
    layers: Sequence[Any],
    *,
    Pref: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return SiO log saturation, saturation ratio, and solver support state.

    The element potential is reconstructed with the same gas and condensate
    thermochemistry and the same least-squares relation used by the production
    support lifecycle.  The natural-log saturation ratio is authoritative when
    exponentiation underflows.
    """

    temperature_values = np.asarray(temperatures, dtype=np.float64)
    pressure_values = np.asarray(reactive_pressures, dtype=np.float64)
    if temperature_values.ndim != 1 or pressure_values.shape != temperature_values.shape:
        raise ValueError("Temperature and pressure profiles must be same-length vectors.")
    if len(layers) != temperature_values.size:
        raise ValueError("Layer results must match the thermodynamic profile length.")
    if not math.isfinite(float(Pref)) or float(Pref) <= 0.0:
        raise ValueError("Pref must be finite and positive.")
    if (
        np.any(~np.isfinite(temperature_values))
        or np.any(~np.isfinite(pressure_values))
        or np.any(pressure_values <= 0.0)
    ):
        raise ValueError("Thermodynamic profiles must be finite with positive pressure.")

    gas_ln_n = np.stack(
        [
            np.asarray(jax.device_get(layer.gas_ln_n), dtype=np.float64)
            for layer in layers
        ]
    )
    gas_ntot = np.asarray(
        [float(np.asarray(jax.device_get(layer.gas_ntot))) for layer in layers],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(gas_ln_n)) or np.any(~np.isfinite(gas_ntot)):
        raise ValueError("Gas equilibrium states must be finite.")
    if np.any(gas_ntot <= 0.0):
        raise ValueError("Gas total amounts must be positive.")

    gas_standard_source = np.stack(
        [
            np.asarray(setup.gas_setup.hvector_func(float(temperature)))
            for temperature in temperature_values
        ]
    )
    gas_stationarity_source = (
        gas_standard_source
        + np.log(pressure_values / float(Pref))[:, None]
        - np.log(gas_ntot)[:, None]
    )
    formula_matrix = jnp.asarray(setup.formula_matrix, dtype=jnp.float64)
    element_potential = jax.vmap(
        lambda q, source: least_squares_element_potential(
            formula_matrix=formula_matrix,
            gas_ln_n=q,
            gas_stationarity_source=source,
        )
    )(
        jnp.asarray(gas_ln_n, dtype=jnp.float64),
        jnp.asarray(gas_stationarity_source, dtype=jnp.float64),
    )
    element_potential = np.asarray(
        jax.device_get(element_potential), dtype=np.float64
    )
    condensate_standard_source = np.stack(
        [
            np.asarray(setup.condensate_setup.hvector_func(float(temperature)))
            for temperature in temperature_values
        ]
    )
    sio_index = setup.condensate_species.index(SIO_CONDENSATE)
    sio_formula = np.asarray(
        setup.formula_matrix_cond[:, sio_index], dtype=np.float64
    )
    log_saturation_ratio = (
        element_potential @ sio_formula
        - condensate_standard_source[:, sio_index]
    )
    if np.any(~np.isfinite(log_saturation_ratio)):
        raise RuntimeError("SiO log saturation ratio is not finite.")

    float_info = np.finfo(np.float64)
    minimum_log = math.log(np.nextafter(0.0, 1.0))
    maximum_log = math.log(float_info.max)
    saturation_ratio = np.exp(
        np.clip(log_saturation_ratio, minimum_log, maximum_log)
    )
    saturation_ratio[log_saturation_ratio < minimum_log] = 0.0
    support_active = np.asarray(
        [
            sio_index
            in set(
                np.asarray(
                    jax.device_get(layer.condensate_support_indices),
                    dtype=np.int64,
                ).tolist()
            )
            for layer in layers
        ],
        dtype=bool,
    )
    return log_saturation_ratio, saturation_ratio, support_active


def _exogibbs_profile_diagnostics(
    solution: RainoutSolution,
) -> dict[str, np.ndarray]:
    """Derive non-mutating profile diagnostics from public solver outputs."""

    required = {
        "element_inventory_target": solution.element_inventory_target,
        "gas_element_inventory": solution.gas_element_inventory,
        "element_inventory_out": solution.element_inventory_out,
        "abundance_scale": solution.abundance_scale,
        "sio_log_saturation_ratio": solution.sio_log_saturation_ratio,
        "sio_saturation_ratio": solution.sio_saturation_ratio,
        "sio_support_active": solution.sio_support_active,
        "sio_condensate_positive": solution.sio_condensate_positive,
    }
    missing = tuple(name for name, value in required.items() if value is None)
    if missing:
        raise ValueError(
            "ExoGibbs diagnostics require populated solution fields: "
            f"{missing!r}."
        )
    target = np.asarray(solution.element_inventory_target, dtype=np.float64)
    gas = np.asarray(solution.gas_element_inventory, dtype=np.float64)
    inventory_out = np.asarray(solution.element_inventory_out, dtype=np.float64)
    abundance_scale = np.asarray(solution.abundance_scale, dtype=np.float64)
    layer_count = target.shape[0]
    expected_inventory_shape = (layer_count, len(ELEMENTS))
    if target.shape != expected_inventory_shape or gas.shape != expected_inventory_shape:
        raise ValueError("Unexpected ExoGibbs element-inventory shape.")
    if inventory_out.shape != expected_inventory_shape:
        raise ValueError("Unexpected ExoGibbs propagated-inventory shape.")
    if abundance_scale.shape != (layer_count,):
        raise ValueError("Unexpected ExoGibbs abundance-scale shape.")
    if (
        np.any(~np.isfinite(target))
        or np.any(~np.isfinite(gas))
        or np.any(~np.isfinite(inventory_out))
        or np.any(~np.isfinite(abundance_scale))
        or np.any(target < 0.0)
        or np.any(gas < 0.0)
        or np.any(inventory_out < 0.0)
        or np.any(abundance_scale <= 0.0)
    ):
        raise ValueError("ExoGibbs inventory diagnostics require finite non-negative data.")

    setup = build_ito_exogibbs_setup()
    condensate_amounts = np.asarray(
        solution.condensate_amounts, dtype=np.float64
    )
    expected_condensate_shape = (layer_count, len(CONDENSATE_SPECIES))
    if condensate_amounts.shape != expected_condensate_shape:
        raise ValueError("Unexpected ExoGibbs condensate profile shape.")
    condensate_inventory = condensate_amounts @ np.asarray(
        setup.formula_matrix_cond, dtype=np.float64
    ).T
    conservative_gas_inventory = target - condensate_inventory
    mismatch = gas - conservative_gas_inventory
    absolute_mismatch = np.abs(mismatch)
    positive_target = target > 0.0
    positive_relative_mismatch = np.full_like(target, np.nan)
    positive_relative_mismatch[positive_target] = (
        absolute_mismatch[positive_target] / target[positive_target]
    )

    policy = fixed_support_v2_production_policy()
    caller_amount_scale = np.sum(
        np.where(positive_target, target, 0.0), axis=1
    )
    caller_budget_floor = (
        float(policy.budget_relative_floor) * caller_amount_scale
    )[:, None]
    floor_scaled_mismatch = absolute_mismatch / np.maximum(
        np.abs(target), caller_budget_floor
    )
    target_exact_depleted = target == 0.0
    exact_depletion_event = positive_target & (inventory_out == 0.0)
    positive_target_below_budget_floor = positive_target & (
        target <= caller_budget_floor
    )
    gas_reintroduced_into_exact_zero_target = target_exact_depleted & (gas > 0.0)

    sio_log_saturation = np.asarray(
        solution.sio_log_saturation_ratio, dtype=np.float64
    )
    sio_saturation = np.asarray(solution.sio_saturation_ratio, dtype=np.float64)
    sio_support = np.asarray(solution.sio_support_active)
    sio_positive = np.asarray(solution.sio_condensate_positive)
    for name, values in (
        ("sio_log_saturation_ratio", sio_log_saturation),
        ("sio_saturation_ratio", sio_saturation),
        ("sio_support_active", sio_support),
        ("sio_condensate_positive", sio_positive),
    ):
        if values.shape != (layer_count,):
            raise ValueError(f"Unexpected {name} profile shape.")
    if sio_support.dtype.kind != "b" or sio_positive.dtype.kind != "b":
        raise ValueError("SiO support diagnostics must be boolean.")
    if np.any(~np.isfinite(sio_log_saturation)) or np.any(~np.isfinite(sio_saturation)):
        raise ValueError("SiO saturation diagnostics must be finite.")
    sio_support_state = np.full(layer_count, "unsupported_zero", dtype="<U20")
    sio_support_state[sio_support & sio_positive] = "supported_positive"
    sio_support_state[sio_support & ~sio_positive] = "supported_zero"
    sio_support_state[~sio_support & sio_positive] = "unsupported_positive"

    return {
        "sio_log_saturation_ratio": sio_log_saturation,
        "sio_saturation_ratio": sio_saturation,
        "sio_support_active": sio_support,
        "sio_condensate_positive": sio_positive,
        "sio_support_state": sio_support_state,
        "condensate_element_inventory": condensate_inventory,
        "conservative_gas_element_inventory": conservative_gas_inventory,
        "gas_vs_conservative_inventory_mismatch": mismatch,
        "gas_vs_conservative_inventory_absolute_mismatch": absolute_mismatch,
        "gas_vs_conservative_inventory_positive_relative_mismatch": (
            positive_relative_mismatch
        ),
        "gas_vs_conservative_inventory_floor_scaled_mismatch": (
            floor_scaled_mismatch
        ),
        "solver_budget_floor_caller_gauge": np.broadcast_to(
            caller_budget_floor, target.shape
        ).copy(),
        "target_exact_depleted_element_mask": target_exact_depleted,
        "exact_depletion_event_element_mask": exact_depletion_event,
        "positive_target_below_budget_floor_element_mask": (
            positive_target_below_budget_floor
        ),
        "gas_reintroduced_into_exact_zero_target_element_mask": (
            gas_reintroduced_into_exact_zero_target
        ),
    }


def solve_exogibbs_rainout(
    profile: ItoProfile,
    *,
    layer1_abundance: np.ndarray,
) -> RainoutSolution:
    """Run the ExoGibbs bottom scan inside a whole-profile He fixed point."""

    setup = build_ito_exogibbs_setup()
    reactive_pressure = _initial_reactive_pressures(profile)
    final_profile = None
    total_fractions = None
    atomic_fractions = None
    for fixed_point_iteration in range(1, FIXED_POINT_MAX_ITERATIONS + 1):
        print(
            "ExoGibbs rainout He-pressure fixed point "
            f"{fixed_point_iteration}/{FIXED_POINT_MAX_ITERATIONS}",
            flush=True,
        )
        final_profile = solve_condensate_profile(
            setup,
            T=profile.temperature_k[::-1],
            P=reactive_pressure[::-1],
            b=jnp.asarray(layer1_abundance, dtype=jnp.float64),
            options=CondensateEquilibriumOptions(
                rainout=True,
                profile_method="scan_hot_from_bottom",
                return_diagnostics=False,
            ),
        )
        jax.block_until_ready(
            (
                final_profile.batched_arrays["gas_x"],
                final_profile.batched_arrays["condensate_amounts"],
            )
        )
        reactive_gas = np.asarray(
            final_profile.batched_arrays["gas_x"], dtype=np.float64
        )[::-1]
        element_inventory_target = np.asarray(
            final_profile.element_inventory_target, dtype=np.float64
        )[::-1]
        reactive_gas = _mask_exactly_depleted_gases(
            reactive_gas,
            element_inventory_target,
            np.asarray(setup.formula_matrix, dtype=np.float64),
        )
        total_fractions, atomic_fractions, pressure_fraction = (
            _convert_reactive_profile(reactive_gas, EXOGIBBS_GAS_SPECIES)
        )
        updated_pressure = profile.pressure_bar * pressure_fraction
        converged, residual = _profile_pressure_converged(
            reactive_pressure, updated_pressure
        )
        print(f"  maximum pressure log residual: {residual:.3e}", flush=True)
        if converged:
            break
        reactive_pressure = updated_pressure
    else:
        raise RuntimeError(
            "ExoGibbs whole-profile helium-pressure fixed point did not "
            "converge."
        )
    assert final_profile is not None
    assert total_fractions is not None
    assert atomic_fractions is not None
    layers = tuple(reversed(final_profile.layers))
    condensate_amounts = np.asarray(
        final_profile.batched_arrays["condensate_amounts"],
        dtype=np.float64,
    )[::-1]
    (
        sio_log_saturation_ratio,
        sio_saturation_ratio,
        sio_support_active,
    ) = _sio_saturation_diagnostics(
        setup,
        profile.temperature_k,
        reactive_pressure,
        layers,
    )
    sio_index = setup.condensate_species.index(SIO_CONDENSATE)
    return RainoutSolution(
        gas_fractions=total_fractions,
        atomic_gas_fractions=atomic_fractions,
        condensate_amounts=condensate_amounts,
        reactive_pressure_bar=reactive_pressure.copy(),
        converged=np.asarray([layer.converged for layer in layers], dtype=bool),
        status=tuple(layer.status for layer in layers),
        acceptance_tier=tuple(layer.acceptance_tier for layer in layers),
        fixed_point_iterations=fixed_point_iteration,
        solver_iterations=np.full(profile.layer.size, -1, dtype=np.int64),
        element_inventory_target=np.asarray(
            final_profile.element_inventory_target, dtype=np.float64
        )[::-1],
        gas_element_inventory=np.asarray(
            final_profile.gas_element_inventory, dtype=np.float64
        )[::-1],
        element_inventory_out=np.asarray(
            final_profile.rainout_element_inventory_out, dtype=np.float64
        )[::-1],
        abundance_scale=np.asarray(
            final_profile.rainout_abundance_scale, dtype=np.float64
        )[::-1],
        sio_log_saturation_ratio=sio_log_saturation_ratio,
        sio_saturation_ratio=sio_saturation_ratio,
        sio_support_active=sio_support_active,
        sio_condensate_positive=condensate_amounts[:, sio_index] > 0.0,
    )


def solve_fastchem_rainout(
    profile: ItoProfile,
    *,
    layer1_abundance: np.ndarray,
    executable: Path,
) -> RainoutSolution:
    """Run one native FastChem ``cr`` profile per He fixed-point step."""

    reactive_pressure = _initial_reactive_pressures(profile)
    final_result = None
    total_fractions = None
    atomic_fractions = None
    aligned_condensates = None
    with tempfile.TemporaryDirectory(
        prefix="exogibbs_ito_rainout_fastchem4_"
    ) as directory:
        work = Path(directory)
        gas_logk, condensate_logk = _write_filtered_fastchem_inputs(work)
        abundance_file = work / "layer1_element_abundances.dat"
        _write_fastchem_abundance(abundance_file, layer1_abundance)
        for fixed_point_iteration in range(1, FIXED_POINT_MAX_ITERATIONS + 1):
            print(
                "FastChem native rainout He-pressure fixed point "
                f"{fixed_point_iteration}/{FIXED_POINT_MAX_ITERATIONS}",
                flush=True,
            )
            final_result = run_fastchem_executable(
                executable=executable,
                temperatures=profile.temperature_k,
                pressures=reactive_pressure,
                element_abundance_file=abundance_file,
                gas_logk_file=gas_logk,
                condensate_logk_file=condensate_logk,
                chemistry_mode="rainout_condensation",
            )
            _validate_fastchem_catalog(
                final_result.gas_names, final_result.condensate_names
            )
            if not np.all(final_result.converged):
                failed = np.flatnonzero(~final_result.converged)
                raise RuntimeError(
                    f"FastChem rainout failed at profile rows {failed.tolist()}."
                )
            if not np.all(final_result.elements_conserved):
                failed = np.flatnonzero(~final_result.elements_conserved)
                raise RuntimeError(
                    "FastChem rainout failed element conservation at rows "
                    f"{failed.tolist()}."
                )
            aligned_gas = np.stack(
                [
                    align_species_values(
                        COMMON_GAS_SPECIES,
                        final_result.gas_names,
                        row,
                    )
                    for row in final_result.gas_number_densities
                ]
            )
            total_fractions, atomic_fractions, pressure_fraction = (
                _convert_reactive_profile(aligned_gas, COMMON_GAS_SPECIES)
            )
            aligned_condensates = np.stack(
                [
                    align_species_values(
                        CONDENSATE_SPECIES,
                        final_result.condensate_names,
                        row,
                    )
                    for row in final_result.condensate_number_densities
                ]
            ) / final_result.total_element_density[:, None]
            updated_pressure = profile.pressure_bar * pressure_fraction
            converged, residual = _profile_pressure_converged(
                reactive_pressure, updated_pressure
            )
            print(f"  maximum pressure log residual: {residual:.3e}", flush=True)
            if converged:
                break
            reactive_pressure = updated_pressure
        else:
            raise RuntimeError(
                "FastChem whole-profile helium-pressure fixed point did not "
                "converge."
            )
    assert final_result is not None
    assert total_fractions is not None
    assert atomic_fractions is not None
    assert aligned_condensates is not None
    return RainoutSolution(
        gas_fractions=total_fractions,
        atomic_gas_fractions=atomic_fractions,
        condensate_amounts=aligned_condensates,
        reactive_pressure_bar=reactive_pressure.copy(),
        converged=final_result.converged.copy(),
        status=tuple(str(value) for value in final_result.status),
        acceptance_tier=("native_cr",) * profile.layer.size,
        fixed_point_iterations=fixed_point_iteration,
        solver_iterations=final_result.iterations.copy(),
        elements_conserved=final_result.elements_conserved.copy(),
    )


def _solution_archive(
    target: ItoProfile,
    layer1_abundance: np.ndarray,
    exogibbs: RainoutSolution,
    fastchem: RainoutSolution,
) -> dict[str, np.ndarray]:
    arrays = {
        "layers": target.layer,
        "pressure_bar": target.pressure_bar,
        "temperature_k": target.temperature_k,
        "layer1_element_abundance": layer1_abundance,
        "ito_gas_fractions": target.gas_fractions,
    }
    for name, solution in (("exogibbs", exogibbs), ("fastchem", fastchem)):
        arrays[f"{name}_gas_fractions"] = solution.gas_fractions
        arrays[f"{name}_atomic_gas_fractions"] = (
            solution.atomic_gas_fractions
        )
        arrays[f"{name}_condensate_amounts"] = solution.condensate_amounts
        arrays[f"{name}_reactive_pressure_bar"] = (
            solution.reactive_pressure_bar
        )
        arrays[f"{name}_converged"] = solution.converged
        arrays[f"{name}_acceptance_tier"] = np.asarray(
            solution.acceptance_tier,
            dtype=np.str_,
        )
        arrays[f"{name}_solver_iterations"] = solution.solver_iterations
        if solution.elements_conserved is not None:
            arrays[f"{name}_elements_conserved"] = (
                solution.elements_conserved
            )
    assert exogibbs.element_inventory_target is not None
    assert exogibbs.gas_element_inventory is not None
    assert exogibbs.element_inventory_out is not None
    assert exogibbs.abundance_scale is not None
    assert fastchem.elements_conserved is not None
    arrays["exogibbs_element_inventory_target"] = (
        exogibbs.element_inventory_target
    )
    arrays["exogibbs_gas_element_inventory"] = exogibbs.gas_element_inventory
    arrays["exogibbs_element_inventory_out"] = exogibbs.element_inventory_out
    arrays["exogibbs_abundance_scale"] = exogibbs.abundance_scale
    diagnostics = _exogibbs_profile_diagnostics(exogibbs)
    for name, values in diagnostics.items():
        arrays[f"exogibbs_{name}"] = values
    return arrays


def _write_table(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    exogibbs: RainoutSolution,
    fastchem: RainoutSolution,
) -> None:
    assert exogibbs.element_inventory_target is not None
    assert exogibbs.gas_element_inventory is not None
    assert exogibbs.element_inventory_out is not None
    assert exogibbs.abundance_scale is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["layer", "pressure_bar", "temperature_k"]
    for source in ("ito", "exogibbs", "fastchem"):
        header.extend(f"{source}_{species}_fraction" for species in ITO_SPECIES)
    header.extend(
        (
            "exogibbs_reactive_pressure_bar",
            "fastchem_reactive_pressure_bar",
        )
    )
    for source in ("exogibbs", "fastchem"):
        header.extend(
            f"{source}_{species}_local_amount"
            for species in CONDENSATE_SPECIES
        )
    header.extend(
        (
            "exogibbs_sio_saturation_ratio",
            "exogibbs_sio_log_saturation_ratio",
            "exogibbs_sio_support_active",
            "exogibbs_sio_condensate_positive",
            "exogibbs_sio_support_state",
        )
    )
    header.extend(f"exogibbs_target_{element}" for element in ELEMENTS)
    header.extend(f"exogibbs_gas_inventory_{element}" for element in ELEMENTS)
    header.extend(f"exogibbs_out_{element}" for element in ELEMENTS)
    diagnostic_element_fields = (
        "condensate_element_inventory",
        "conservative_gas_element_inventory",
        "gas_vs_conservative_inventory_mismatch",
        "gas_vs_conservative_inventory_absolute_mismatch",
        "gas_vs_conservative_inventory_positive_relative_mismatch",
        "gas_vs_conservative_inventory_floor_scaled_mismatch",
        "solver_budget_floor_caller_gauge",
        "target_exact_depleted_element_mask",
        "exact_depletion_event_element_mask",
        "positive_target_below_budget_floor_element_mask",
        "gas_reintroduced_into_exact_zero_target_element_mask",
    )
    for field in diagnostic_element_fields:
        header.extend(f"exogibbs_{field}_{element}" for element in ELEMENTS)
    header.extend(
        (
            "exogibbs_abundance_scale",
            "exogibbs_converged",
            "fastchem_converged",
            "fastchem_elements_conserved",
            "exogibbs_status",
            "fastchem_status",
            "exogibbs_acceptance_tier",
            "fastchem_iterations",
        )
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(arrays["layers"].size):
            row: list[Any] = [
                int(arrays["layers"][index]),
                float(arrays["pressure_bar"][index]),
                float(arrays["temperature_k"][index]),
            ]
            for key in (
                "ito_gas_fractions",
                "exogibbs_gas_fractions",
                "fastchem_gas_fractions",
            ):
                row.extend(float(value) for value in arrays[key][index])
            row.extend(
                (
                    float(exogibbs.reactive_pressure_bar[index]),
                    float(fastchem.reactive_pressure_bar[index]),
                )
            )
            row.extend(
                float(value) for value in exogibbs.condensate_amounts[index]
            )
            row.extend(
                float(value) for value in fastchem.condensate_amounts[index]
            )
            row.extend(
                (
                    float(arrays["exogibbs_sio_saturation_ratio"][index]),
                    float(arrays["exogibbs_sio_log_saturation_ratio"][index]),
                    bool(arrays["exogibbs_sio_support_active"][index]),
                    bool(arrays["exogibbs_sio_condensate_positive"][index]),
                    str(arrays["exogibbs_sio_support_state"][index]),
                )
            )
            row.extend(
                float(value)
                for value in exogibbs.element_inventory_target[index]
            )
            row.extend(
                float(value)
                for value in exogibbs.gas_element_inventory[index]
            )
            row.extend(
                float(value)
                for value in exogibbs.element_inventory_out[index]
            )
            for field in diagnostic_element_fields:
                values = arrays[f"exogibbs_{field}"][index]
                if np.issubdtype(np.asarray(values).dtype, np.bool_):
                    row.extend(bool(value) for value in values)
                else:
                    row.extend(float(value) for value in values)
            row.extend(
                (
                    float(exogibbs.abundance_scale[index]),
                    bool(exogibbs.converged[index]),
                    bool(fastchem.converged[index]),
                    bool(fastchem.elements_conserved[index]),
                    exogibbs.status[index],
                    fastchem.status[index],
                    exogibbs.acceptance_tier[index],
                    int(fastchem.solver_iterations[index]),
                )
            )
            writer.writerow(row)


def _maximum_profile_metric(
    values: np.ndarray,
    layers: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Locate a finite maximum in a layer or layer-by-element metric."""

    metric = np.asarray(values, dtype=np.float64)
    layer_values = np.asarray(layers, dtype=np.int64)
    if metric.ndim not in {1, 2} or metric.shape[0] != layer_values.size:
        raise ValueError("Metric must have one row per profile layer.")
    selected = np.ones(metric.shape, dtype=bool)
    if mask is not None:
        mask_values = np.asarray(mask)
        if mask_values.shape != metric.shape:
            raise ValueError("Metric mask must match the metric shape.")
        if mask_values.dtype.kind != "b":
            raise ValueError("Metric mask must be boolean.")
        selected &= mask_values
    nonfinite = selected & ~np.isfinite(metric)
    included = selected & ~nonfinite
    all_selected_values_finite = not np.any(nonfinite)
    if not np.any(included):
        return {
            "value": None,
            "layer": None,
            "element": None,
            "all_selected_values_finite": all_selected_values_finite,
            "nonfinite_value_count": int(np.count_nonzero(nonfinite)),
        }
    ranked = np.where(included, metric, -np.inf)
    flat_index = int(np.argmax(ranked))
    indices = np.unravel_index(flat_index, metric.shape)
    element = None if metric.ndim == 1 else ELEMENTS[int(indices[1])]
    return {
        "value": float(metric[indices]),
        "layer": int(layer_values[int(indices[0])]),
        "element": element,
        "all_selected_values_finite": all_selected_values_finite,
        "nonfinite_value_count": int(np.count_nonzero(nonfinite)),
    }


def _mask_profile_summary(
    mask: np.ndarray,
    layers: np.ndarray,
) -> dict[str, Any]:
    """Summarize a layer-by-element numerical diagnostic mask."""

    values = np.asarray(mask, dtype=bool)
    layer_values = np.asarray(layers, dtype=np.int64)
    if values.shape != (layer_values.size, len(ELEMENTS)):
        raise ValueError("Diagnostic mask must have one column per element.")
    affected_rows = np.flatnonzero(np.any(values, axis=1))
    first_by_element: dict[str, int | None] = {}
    counts_by_element: dict[str, int] = {}
    for element_index, element in enumerate(ELEMENTS):
        indices = np.flatnonzero(values[:, element_index])
        counts_by_element[element] = int(indices.size)
        first_by_element[element] = (
            None if indices.size == 0 else int(layer_values[int(indices[0])])
        )
    return {
        "affected_layer_count": int(affected_rows.size),
        "affected_entry_count": int(np.count_nonzero(values)),
        "first_layer": (
            None
            if affected_rows.size == 0
            else int(layer_values[int(affected_rows[0])])
        ),
        "last_layer": (
            None
            if affected_rows.size == 0
            else int(layer_values[int(affected_rows[-1])])
        ),
        "first_layer_by_element": first_by_element,
        "entry_count_by_element": counts_by_element,
    }


def _stability_diagnostic_summary(
    target: ItoProfile,
    solution: RainoutSolution,
    diagnostics: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Build compact numerical-stability diagnostics and rerun criteria."""

    layers = target.layer
    sio_support = np.asarray(diagnostics["sio_support_active"])
    sio_positive = np.asarray(diagnostics["sio_condensate_positive"])
    converged = np.asarray(solution.converged)
    expected_boolean_shape = (layers.size,)
    for name, values in (
        ("sio_support_active", sio_support),
        ("sio_condensate_positive", sio_positive),
        ("converged", converged),
    ):
        if values.dtype.kind != "b" or values.shape != expected_boolean_shape:
            raise ValueError(
                f"{name} must be a boolean array with one entry per layer."
            )
    active_log_saturation = np.abs(
        np.asarray(diagnostics["sio_log_saturation_ratio"], dtype=np.float64)
    )
    active_mask = sio_support & sio_positive
    active_saturation_maximum = _maximum_profile_metric(
        active_log_saturation,
        layers,
        mask=active_mask,
    )
    positive_relative_maximum = _maximum_profile_metric(
        diagnostics[
            "gas_vs_conservative_inventory_positive_relative_mismatch"
        ],
        layers,
        mask=np.asarray(solution.element_inventory_target) > 0.0,
    )
    floor_scaled_maximum = _maximum_profile_metric(
        diagnostics["gas_vs_conservative_inventory_floor_scaled_mismatch"],
        layers,
    )
    absolute_maximum = _maximum_profile_metric(
        diagnostics["gas_vs_conservative_inventory_absolute_mismatch"],
        layers,
    )
    depletion_event_summary = _mask_profile_summary(
        diagnostics["exact_depletion_event_element_mask"], layers
    )
    exact_target_summary = _mask_profile_summary(
        diagnostics["target_exact_depleted_element_mask"], layers
    )
    below_floor_summary = _mask_profile_summary(
        diagnostics["positive_target_below_budget_floor_element_mask"],
        layers,
    )
    reintroduction_summary = _mask_profile_summary(
        diagnostics[
            "gas_reintroduced_into_exact_zero_target_element_mask"
        ],
        layers,
    )
    support_state_counts = Counter(
        str(value) for value in diagnostics["sio_support_state"]
    )
    trace_capacity_count = solution.acceptance_tier.count(
        "rainout_trace_capacity_accepted"
    )

    active_saturation_passed = bool(
        active_saturation_maximum["all_selected_values_finite"]
        and (
            active_saturation_maximum["value"] is None
            or active_saturation_maximum["value"]
            <= SIO_ACTIVE_LOG_SATURATION_TOLERANCE
        )
    )
    positive_relative_passed = bool(
        positive_relative_maximum["all_selected_values_finite"]
        and (
            positive_relative_maximum["value"] is None
            or positive_relative_maximum["value"]
            <= INVENTORY_RELATIVE_MISMATCH_TOLERANCE
        )
    )
    floor_scaled_passed = bool(
        floor_scaled_maximum["all_selected_values_finite"]
        and (
            floor_scaled_maximum["value"] is None
            or floor_scaled_maximum["value"]
            <= INVENTORY_RELATIVE_MISMATCH_TOLERANCE
        )
    )
    profile_metrics_are_finite = bool(
        all(
            summary["all_selected_values_finite"]
            for summary in (
                active_saturation_maximum,
                positive_relative_maximum,
                floor_scaled_maximum,
                absolute_maximum,
            )
        )
    )
    checks = {
        "all_layers_converged": bool(np.all(converged)),
        "profile_metrics_are_finite": profile_metrics_are_finite,
        "positive_sio_condensate_has_solver_support": bool(
            np.all(~sio_positive | sio_support)
        ),
        "active_sio_is_saturated": bool(active_saturation_passed),
        "positive_target_inventory_is_conservative": bool(
            positive_relative_passed
        ),
        "floor_scaled_inventory_is_conservative": bool(floor_scaled_passed),
        "no_positive_to_exact_zero_depletion_event": (
            depletion_event_summary["affected_entry_count"] == 0
        ),
        "no_gas_reintroduction_after_exact_zero": (
            reintroduction_summary["affected_entry_count"] == 0
        ),
        "no_trace_capacity_acceptance": trace_capacity_count == 0,
    }
    return {
        "schema": "exogibbs_ito_2025_rainout_stability_diagnostics_v1",
        "sio_saturation": {
            "definition": "S = exp(A_cond.T @ element_potential - h_cond)",
            "element_potential_source": (
                "least-squares reconstruction from the accepted ExoGibbs gas "
                "state using the production gas thermochemistry"
            ),
            "logarithm_base": "natural",
            "supported_layer_count": int(np.count_nonzero(sio_support)),
            "positive_condensate_layer_count": int(
                np.count_nonzero(sio_positive)
            ),
            "support_transition_count": int(
                np.count_nonzero(sio_support[1:] != sio_support[:-1])
            ),
            "support_state_counts": {
                name: int(count)
                for name, count in sorted(support_state_counts.items())
            },
            "maximum_absolute_log_saturation_when_supported_positive": (
                active_saturation_maximum
            ),
        },
        "inventory_consistency": {
            "conservative_definition": "b_conservative = b_target - A_cond @ m_cond",
            "gas_definition": "b_gas = A_gas @ n_gas",
            "maximum_absolute_mismatch": absolute_maximum,
            "maximum_positive_target_relative_mismatch": (
                positive_relative_maximum
            ),
            "maximum_solver_budget_floor_scaled_mismatch": (
                floor_scaled_maximum
            ),
        },
        "numerical_depletion": {
            "positive_to_exact_zero_events": depletion_event_summary,
            "exact_zero_targets": exact_target_summary,
            "positive_targets_at_or_below_solver_budget_floor": (
                below_floor_summary
            ),
            "gas_reintroduced_into_exact_zero_targets": (
                reintroduction_summary
            ),
        },
        "rerun_pass_criteria": {
            "thresholds": {
                "maximum_absolute_log_saturation_when_supported_positive": (
                    SIO_ACTIVE_LOG_SATURATION_TOLERANCE
                ),
                "maximum_inventory_relative_mismatch": (
                    INVENTORY_RELATIVE_MISMATCH_TOLERANCE
                ),
            },
            "checks": checks,
            "overall_passed": bool(all(checks.values())),
        },
    }


def make_comparison_figure(
    target: ItoProfile,
    exogibbs: RainoutSolution,
    fastchem: RainoutSolution,
):
    """Plot six gas species and two local condensate profiles."""

    trace_capacity_count = exogibbs.acceptance_tier.count(
        "rainout_trace_capacity_accepted"
    )
    assert exogibbs.element_inventory_target is not None
    diagnostics = _exogibbs_profile_diagnostics(exogibbs)
    stability = _stability_diagnostic_summary(target, exogibbs, diagnostics)
    saturation_summary = stability["sio_saturation"]
    inventory_summary = stability["inventory_consistency"]
    numerical_depletion = stability["numerical_depletion"]
    active_log_saturation_maximum = saturation_summary[
        "maximum_absolute_log_saturation_when_supported_positive"
    ]["value"]
    active_log_saturation_text = (
        "n/a"
        if active_log_saturation_maximum is None
        else f"{active_log_saturation_maximum:.3g}"
    )
    positive_relative_mismatch = inventory_summary[
        "maximum_positive_target_relative_mismatch"
    ]["value"]
    positive_relative_mismatch_text = (
        "n/a"
        if positive_relative_mismatch is None
        else f"{positive_relative_mismatch:.3g}"
    )
    depletion = _exact_depletion_summary(
        target.layer,
        exogibbs.element_inventory_target,
    )
    depletion_layer = depletion["first_exact_depletion_layer"]
    if depletion_layer is None:
        depletion_note = "No element reaches exact propagated depletion."
    else:
        depletion_elements = ", ".join(
            depletion["first_exact_depletion_elements"]
        )
        depletion_note = (
            f"First exact depletion: Layer {depletion_layer} "
            f"({depletion_elements}); affected gases are masked."
        )
    fig, axes = plt.subplots(4, 2, figsize=(12.0, 15.0), sharey=True)
    gas_styles = (
        ("Ito et al. (2025)", target.gas_fractions, "black", "-", 2.2),
        ("ExoGibbs rainout", exogibbs.gas_fractions, "tab:blue", "--", 1.8),
        ("FastChem 4 rainout", fastchem.gas_fractions, "tab:orange", ":", 2.0),
    )
    for species_index, (axis, species) in enumerate(
        zip(axes.flat[:6], ITO_SPECIES)
    ):
        for _label, values, color, linestyle, linewidth in gas_styles:
            axis.plot(
                _plot_values(values[:, species_index]),
                target.pressure_bar,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
        axis.set_title(species)
        axis.set_xlabel("Total-pressure gas mole fraction")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    for condensate_index, (axis, species) in enumerate(
        zip(axes.flat[6:], CONDENSATE_SPECIES)
    ):
        axis.plot(
            _plot_values(exogibbs.condensate_amounts[:, condensate_index]),
            target.pressure_bar,
            color="tab:blue",
            linestyle="--",
            linewidth=1.8,
        )
        axis.plot(
            _plot_values(fastchem.condensate_amounts[:, condensate_index]),
            target.pressure_bar,
            color="tab:orange",
            linestyle=":",
            linewidth=2.0,
        )
        axis.set_title(f"{species} local condensate amount")
        axis.set_xlabel("Amount / total element inventory")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    for index, axis in enumerate(axes.flat):
        if index % 2 == 0:
            axis.set_ylabel("Total pressure [bar]")
    axes.flat[0].set_ylim(
        float(np.max(target.pressure_bar)) * 1.02,
        float(np.min(target.pressure_bar)) / 1.02,
    )
    handles = [
        Line2D(
            (0,),
            (0,),
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
        for label, _values, color, linestyle, linewidth in gas_styles
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=3,
    )
    fig.suptitle(
        "Ito et al. (2025) propagated rainout comparison\n"
        f"Layer 1 gas boundary; Layers {target.layer[0]}--{target.layer[-1]}",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.014,
        "Layer 1 is not re-solved. Each solver propagates its own gas-phase "
        "H/O/Si inventory upward; He/H2 is fixed outside chemistry.\n"
        "ExoGibbs uses Ito's five molecules; FastChem also requires elemental "
        "H/O/Si gases. "
        f"{depletion_note}\n"
        f"SiO support transitions: {saturation_summary['support_transition_count']}; "
        f"max supported-positive |ln S|: {active_log_saturation_text}. "
        "Max positive-target gas/conservative mismatch: "
        f"{positive_relative_mismatch_text}.\n"
        "Positive-to-zero depletion events: "
        f"{numerical_depletion['positive_to_exact_zero_events']['affected_entry_count']}; "
        "gas reintroduction entries after exact zero: "
        f"{numerical_depletion['gas_reintroduced_into_exact_zero_targets']['affected_entry_count']}. "
        f"Trace-capacity tier: {trace_capacity_count} "
        f"layers. Stability criteria passed: "
        f"{stability['rerun_pass_criteria']['overall_passed']}. "
        f"Plot floor: {PLOT_FLOOR:.0e}.",
        ha="center",
        fontsize=8.0,
    )
    fig.tight_layout(rect=(0.02, 0.072, 0.98, 0.925), h_pad=1.7, w_pad=1.3)
    return fig


def _maximum_adjacency_error(solution: RainoutSolution) -> float:
    assert solution.element_inventory_target is not None
    assert solution.element_inventory_out is not None
    if solution.element_inventory_target.shape[0] < 2:
        return 0.0
    return float(
        np.max(
            np.abs(
                solution.element_inventory_target[1:]
                - solution.element_inventory_out[:-1]
            )
        )
    )


def _write_summary(
    path: Path,
    *,
    input_path: Path,
    executable: Path,
    target: ItoProfile,
    layer1_abundance: np.ndarray,
    exogibbs: RainoutSolution,
    fastchem: RainoutSolution,
) -> dict[str, Any]:
    plateau_stops = _ito_plateau_stops(target.gas_fractions)
    acceptance_counts = Counter(exogibbs.acceptance_tier)
    assert exogibbs.element_inventory_target is not None
    assert fastchem.elements_conserved is not None
    depletion = _exact_depletion_summary(
        target.layer,
        exogibbs.element_inventory_target,
    )
    diagnostics = _exogibbs_profile_diagnostics(exogibbs)
    stability = _stability_diagnostic_summary(target, exogibbs, diagnostics)
    payload = {
        "schema": SUMMARY_SCHEMA,
        "input": {
            "workbook": str(input_path.resolve()),
            "workbook_sha256": _sha256(input_path),
            "fastchem_executable": str(executable.resolve()),
            "fastchem_executable_sha256": _sha256(executable),
            "source_paper": "Ito et al. 2025, ApJ 987, 174",
        },
        "boundary": {
            "source": "Ito Layer 1 gas fractions",
            "layer1_is_not_resolved": True,
            "elements": list(ELEMENTS),
            "sum_normalized_abundance": layer1_abundance.tolist(),
        },
        "profile": {
            "first_layer": int(target.layer[0]),
            "last_layer": int(target.layer[-1]),
            "layer_count": int(target.layer.size),
            "ito_exact_trailing_plateaus": _plateau_summary(
                target.gas_fractions, target.layer
            ),
        },
        "chemistry": {
            "ito_gases": list(ITO_SPECIES),
            "exogibbs_reactive_gases": list(EXOGIBBS_GAS_SPECIES),
            "fastchem_reactive_gases": list(COMMON_GAS_SPECIES),
            "condensates": list(CONDENSATE_SPECIES),
            "helium_to_h2_number_ratio": HE_TO_H2_RATIO,
            "helium_pressure_fixed_point_rtol": FIXED_POINT_RTOL,
        },
        "exogibbs": {
            "method": "scan_hot_from_bottom",
            "rainout": True,
            "converged_layers": int(np.count_nonzero(exogibbs.converged)),
            "acceptance_tiers": {
                name: int(count)
                for name, count in sorted(acceptance_counts.items())
            },
            "trace_capacity_accepted_layers": int(
                acceptance_counts.get("rainout_trace_capacity_accepted", 0)
            ),
            "fixed_point_iterations": exogibbs.fixed_point_iterations,
            "maximum_inventory_adjacency_error": (
                _maximum_adjacency_error(exogibbs)
            ),
            "maximum_abundance_scale": float(np.max(exogibbs.abundance_scale)),
            "stability_diagnostics": stability,
            **depletion,
            "difference_from_ito": {
                "full_profile": _difference_summary(
                    exogibbs.gas_fractions, target.gas_fractions
                ),
                "before_ito_trailing_plateau": _difference_summary(
                    exogibbs.gas_fractions,
                    target.gas_fractions,
                    stop_indices=plateau_stops,
                ),
            },
        },
        "fastchem4": {
            "method": "native_cr",
            "converged_layers": int(np.count_nonzero(fastchem.converged)),
            "elements_conserved_layers": int(
                np.count_nonzero(fastchem.elements_conserved)
            ),
            "fixed_point_iterations": fastchem.fixed_point_iterations,
            "difference_from_ito": {
                "full_profile": _difference_summary(
                    fastchem.gas_fractions, target.gas_fractions
                ),
                "before_ito_trailing_plateau": _difference_summary(
                    fastchem.gas_fractions,
                    target.gas_fractions,
                    stop_indices=plateau_stops,
                ),
            },
        },
        "exogibbs_vs_fastchem4": {
            "full_profile": _difference_summary(
                exogibbs.gas_fractions, fastchem.gas_fractions
            ),
            "before_ito_trailing_plateau": _difference_summary(
                exogibbs.gas_fractions,
                fastchem.gas_fractions,
                stop_indices=plateau_stops,
            ),
        },
        "interpretation_limits": [
            "Ito Layer 1 uses a different ground-interface equilibrium system.",
            "Ito does not provide local condensate amounts.",
            (
                "FastChem includes elemental H/O/Si reference gases absent "
                "from the exact Ito/ExoGibbs five-molecule network."
            ),
            (
                "Exact trailing Ito plateaus are reported separately from "
                "the physically informative lower profile."
            ),
            (
                "When a propagated element reaches exact numerical zero, "
                "reported gas species containing that element are masked and "
                "renormalized. Raw solver gas-element inventories remain in "
                "the archive for numerical audit."
            ),
            (
                "The SiO saturation ratio is reconstructed diagnostically "
                "from accepted solver states and does not alter equilibrium "
                "or rainout propagation."
            ),
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _print_key_differences(payload: Mapping[str, Any]) -> None:
    print("Maximum absolute dex before each Ito trailing plateau:")
    for source in ("exogibbs", "fastchem4"):
        rows = payload[source]["difference_from_ito"][
            "before_ito_trailing_plateau"
        ]
        values = ", ".join(
            f"{species}={rows[species]['maximum_absolute_dex']:.4g}"
            for species in ITO_SPECIES
        )
        print(f"  {source} vs Ito: {values}")
    rows = payload["exogibbs_vs_fastchem4"][
        "before_ito_trailing_plateau"
    ]
    values = ", ".join(
        f"{species}={rows[species]['maximum_absolute_dex']:.4g}"
        for species in ITO_SPECIES
    )
    print(f"  ExoGibbs vs FastChem4: {values}")
    stability = payload["exogibbs"]["stability_diagnostics"]
    checks = stability["rerun_pass_criteria"]["checks"]
    failed = tuple(name for name, passed in checks.items() if not passed)
    print(
        "ExoGibbs rainout stability criteria: "
        f"passed={stability['rerun_pass_criteria']['overall_passed']}; "
        f"failed={failed!r}"
    )


def _validate_release_criteria(payload: Mapping[str, Any]) -> None:
    """Fail when the recorded rainout release criteria do not pass."""

    layer_count = int(payload["profile"]["layer_count"])
    count_checks = {
        "all ExoGibbs layers converged": (
            int(payload["exogibbs"]["converged_layers"]) == layer_count
        ),
        "all FastChem layers converged": (
            int(payload["fastchem4"]["converged_layers"]) == layer_count
        ),
        "all FastChem layers conserved elements": (
            int(payload["fastchem4"]["elements_conserved_layers"])
            == layer_count
        ),
    }
    stability = payload["exogibbs"]["stability_diagnostics"]
    stability_checks = stability["rerun_pass_criteria"]["checks"]
    failed = [name for name, passed in count_checks.items() if not passed]
    for name, passed in stability_checks.items():
        if type(passed) is not bool:
            failed.append(f"{name} (non-boolean result)")
        elif not passed:
            failed.append(name)
    overall_passed = stability["rerun_pass_criteria"]["overall_passed"]
    if type(overall_passed) is not bool:
        failed.append("overall rainout stability (non-boolean result)")
    elif not overall_passed:
        failed.append("overall rainout stability")
    if failed:
        raise RuntimeError(
            "Ito rainout release criteria failed: " + ", ".join(failed)
        )


def main() -> None:
    args = _parse_args()
    input_path = args.input.resolve(strict=True)
    executable_path = args.fastchem_executable.resolve(strict=True)
    if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
        raise ValueError(
            f"FastChem executable is not an executable file: {executable_path}."
        )
    profile = load_ito_profile(input_path)
    target = _target_profile(profile, args.max_layers)
    layer1_abundance = reactive_element_abundances(profile.gas_fractions[0])
    args.figure.unlink(missing_ok=True)
    print(
        f"Boundary: Ito Layer 1; solving Layers {target.layer[0]}--"
        f"{target.layer[-1]} ({target.layer.size} layers)",
        flush=True,
    )
    exogibbs = solve_exogibbs_rainout(
        target,
        layer1_abundance=layer1_abundance,
    )
    fastchem = solve_fastchem_rainout(
        target,
        layer1_abundance=layer1_abundance,
        executable=executable_path,
    )
    arrays = _solution_archive(target, layer1_abundance, exogibbs, fastchem)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.archive, **arrays)
    _write_table(args.table, arrays, exogibbs, fastchem)
    payload = _write_summary(
        args.summary,
        input_path=input_path,
        executable=executable_path,
        target=target,
        layer1_abundance=layer1_abundance,
        exogibbs=exogibbs,
        fastchem=fastchem,
    )
    _validate_release_criteria(payload)
    figure = make_comparison_figure(target, exogibbs, fastchem)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=180)
    _print_key_differences(payload)
    print(f"figure: {args.figure}")
    print(f"table: {args.table}")
    print(f"summary: {args.summary}")
    print(f"archive: {args.archive}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
