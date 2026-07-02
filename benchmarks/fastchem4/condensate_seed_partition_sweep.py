#!/usr/bin/env python
"""Single-layer condensate seed/support sweep for major gas discrepancies."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
from jax import config
import jax.numpy as jnp

from exogibbs.api.condensate_equilibrium import (
    CondensateEquilibriumInit,
    CondensateEquilibriumOptions,
    prepare_experimental_profile_fixed_support_batch_plan,
    run_experimental_profile_fixed_support_batch_plan_with_fallback_rescue,
)
from exogibbs.api.equilibrium import EquilibriumOptions, equilibrium
from exogibbs.condensates.curated_profiles import (
    FRESH_CURATED_PROFILES,
    element_budget_for_profile,
)
from exogibbs.condensates.initialization_policy import recommend_budget_preserving_seed_amounts
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup

from fastchem4_vmap_cold_rescue_compare import (
    _aligned_fastchem_amounts,
    _jsonable,
    _major_species_log10_ratio_metrics,
    _normal_name_map,
    _run_fastchem_profile,
)


DEFAULT_OUTPUT_DIR = Path("volatiles_artifacts/fastchem4_major_species_investigation")
DEFAULT_CASES = (
    "carbon_rich_graphite_window:3",
    "carbon_rich_CaS_MgS_AlN_window:0",
)
DEFAULT_SEED_FRACTIONS = (1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1, 3.0e-1, 6.0e-1)
TARGET_SPECIES = ("C1H4", "H2O1", "C1O1", "H2")
AMOUNT_FLOOR = 1.0e-300


def _parse_case(case: str) -> tuple[str, int]:
    if ":" not in case:
        raise ValueError(f"Case must be FAMILY:LAYER_INDEX, got {case!r}")
    family, layer = case.rsplit(":", 1)
    return family, int(layer)


def _one_layer_definition(family: str, layer_index: int) -> Any:
    definition = FRESH_CURATED_PROFILES[family]
    return replace(
        definition,
        temperatures=(float(definition.temperatures[layer_index]),),
        pressures=(float(definition.pressures[layer_index]),),
    )


def _curated_support_indices(setup: Any, definition: Any) -> tuple[int, ...]:
    index_by_name = {str(name): index for index, name in enumerate(setup.condensate_species)}
    return tuple(int(index_by_name[name]) for name in definition.support_species)


def _fastchem_active_support_indices(setup: Any, fastchem: dict[str, Any]) -> tuple[int, ...]:
    fc_name_map = _normal_name_map(tuple(fastchem["condensate_names"]))
    fc_amounts = np.asarray(fastchem["condensate_number_densities"], dtype=np.float64)
    support = []
    for index, name in enumerate(setup.condensate_species):
        fc_index = fc_name_map.get(str(name))
        if fc_index is not None and np.any(fc_amounts[:, fc_index] > 0.0):
            support.append(int(index))
    return tuple(support)


def _support_indices_for_mode(
    *,
    setup: Any,
    definition: Any,
    fastchem: dict[str, Any],
    mode: str,
) -> tuple[int, ...]:
    curated = _curated_support_indices(setup, definition)
    if mode == "curated":
        return curated
    fc_active = _fastchem_active_support_indices(setup, fastchem)
    if mode == "fastchem_active":
        return fc_active
    if mode == "curated_plus_fastchem_active":
        return tuple(dict.fromkeys(curated + fc_active))
    raise ValueError(f"Unknown support mode: {mode}")


def _budget_seed_amounts(
    *,
    setup: Any,
    budget: jax.Array,
    support_indices: Sequence[int],
    seed_fraction: float,
) -> tuple[float, ...]:
    seed = recommend_budget_preserving_seed_amounts(
        formula_matrix_cond=setup.formula_matrix_cond,
        element_inventory_target=budget,
        condensate_species_order=setup.condensate_species,
        support_indices=tuple(int(index) for index in support_indices),
        seed_fraction=float(seed_fraction),
        max_seed_amount=1.0,
        min_seed_amount=AMOUNT_FLOOR,
        preserve_budget_fraction=True,
        field_provenance={
            "formula_matrix_cond": "exogibbs_condensate_chemical_setup",
            "element_inventory_target": "exogibbs_seed_partition_sweep_budget",
        },
    )
    return tuple(float(value) for value in seed.recommended_amounts)


def _fastchem_scaled_amounts(
    *,
    setup: Any,
    budget: np.ndarray,
    fastchem: dict[str, Any],
    support_indices: Sequence[int],
) -> tuple[float, ...]:
    scaled = _aligned_fastchem_amounts(
        setup=setup,
        fastchem=fastchem,
        layer_index=0,
        target=budget,
    )
    amounts = np.asarray(scaled["condensate_amounts"], dtype=np.float64)
    return tuple(max(float(amounts[int(index)]), AMOUNT_FLOOR) for index in support_indices)


def _species_x(
    *,
    names: Sequence[str],
    values: np.ndarray,
    species: str,
) -> float | None:
    name_map = _normal_name_map(tuple(names))
    index = name_map.get(species)
    if index is None:
        return None
    return float(values[index])


def _log10_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a <= 0.0 or b <= 0.0:
        return None
    return float(np.log10(a / b))


def _element_fraction_rows(
    *,
    setup: Any,
    budget: np.ndarray,
    condensates: np.ndarray,
) -> list[dict[str, Any]]:
    ac = np.asarray(setup.formula_matrix_cond, dtype=np.float64)
    condensed = ac @ np.asarray(condensates, dtype=np.float64)
    rows = []
    element_index = {str(name): index for index, name in enumerate(setup.elements)}
    for element in ("H", "C", "O", "Mg", "Si", "S", "Fe"):
        index = element_index.get(element)
        if index is None:
            continue
        total = float(budget[index])
        amount = float(condensed[index])
        rows.append(
            {
                "element": element,
                "condensed": amount,
                "condensed_fraction": None if total <= 0.0 else amount / total,
            }
        )
    return rows


def _run_seed_variant(
    *,
    setup: Any,
    definition: Any,
    budget: jax.Array,
    fastchem: dict[str, Any],
    support_indices: tuple[int, ...],
    support_amounts: tuple[float, ...],
    max_inner_iterations: int,
    depleted_gas_init: bool,
) -> dict[str, Any]:
    gas_budget = budget
    if depleted_gas_init:
        full_support = np.zeros((len(setup.condensate_species),), dtype=np.float64)
        full_support[np.asarray(support_indices, dtype=np.int64)] = np.asarray(
            support_amounts,
            dtype=np.float64,
        )
        gas_budget_np = np.asarray(jax.device_get(budget), dtype=np.float64) - (
            np.asarray(setup.formula_matrix_cond, dtype=np.float64) @ full_support
        )
        gas_budget = jnp.asarray(np.maximum(gas_budget_np, 0.0), dtype=jnp.float64)
    gas_result = equilibrium(
        setup.gas_setup,
        float(definition.temperatures[0]),
        float(definition.pressures[0]),
        gas_budget,
        Pref=1.0,
        options=EquilibriumOptions(),
    )
    init = (
        CondensateEquilibriumInit(
            gas_ln_n=gas_result.ln_n,
            gas_ntot=gas_result.ntot,
            support_indices=support_indices,
            support_amounts=support_amounts,
        ),
    )
    options = CondensateEquilibriumOptions(
        profile_method="vmap_cold",
        profile_warm_start_support_policy="explicit_payload",
        return_diagnostics=True,
        allow_caveat_tiers=True,
        max_inner_iterations=int(max_inner_iterations),
        enable_head_route_warm_start=True,
        enable_depleted_gas_refresh=True,
        enable_full_condensate_budget_residual_gate=True,
    )
    plan = prepare_experimental_profile_fixed_support_batch_plan(
        setup,
        jnp.asarray(definition.temperatures, dtype=jnp.float64),
        jnp.asarray(definition.pressures, dtype=jnp.float64),
        budget,
        init=init,
        options=options,
    )
    arrays = run_experimental_profile_fixed_support_batch_plan_with_fallback_rescue(
        plan,
        rho_initialization=options.experimental_profile_fixed_support_rescue_rho_initialization,
        lambda_initialization=options.experimental_profile_fixed_support_rescue_lambda_initialization,
        residual_tolerance_multiplier=(
            options.experimental_profile_fixed_support_rescue_residual_tolerance_multiplier
        ),
        prune_relative_floors=options.experimental_profile_fixed_support_rescue_prune_relative_floors,
    )
    gas_x = np.asarray(jax.device_get(arrays["gas_x"][0]), dtype=np.float64)
    gas_n = np.asarray(jax.device_get(arrays["gas_n"][0]), dtype=np.float64)
    condensates = np.asarray(jax.device_get(arrays["condensate_amounts"][0]), dtype=np.float64)
    final_residual = float(np.asarray(jax.device_get(arrays["final_residual"]))[0])
    converged = bool(np.asarray(jax.device_get(arrays["converged"]))[0])
    budget_np = np.asarray(jax.device_get(budget), dtype=np.float64)
    fc_gas_x = np.asarray(fastchem["gas_x"][0], dtype=np.float64)
    metrics = _major_species_log10_ratio_metrics(
        exo_names=tuple(setup.gas_species),
        exo_values=gas_x,
        fc_names=tuple(fastchem["gas_names"]),
        fc_values=fc_gas_x,
        abundance_threshold=1.0e-4,
        ratio_floor=1.0e-300,
        selection_mode="overlap",
        top_n=8,
    )
    ac = np.asarray(setup.formula_matrix_cond, dtype=np.float64)
    ag = np.asarray(setup.formula_matrix, dtype=np.float64)
    budget_residual = ag @ gas_n + ac @ condensates - budget_np
    positive_budget = np.abs(budget_np) > 0.0
    rel = np.abs(budget_residual[positive_budget]) / np.maximum(
        np.abs(budget_np[positive_budget]),
        1.0e-300,
    )
    species = []
    for name in TARGET_SPECIES:
        exo_value = _species_x(names=setup.gas_species, values=gas_x, species=name)
        fc_value = _species_x(names=fastchem["gas_names"], values=fc_gas_x, species=name)
        species.append(
            {
                "species": name,
                "exogibbs_x": exo_value,
                "fastchem4_x": fc_value,
                "dex": _log10_ratio(exo_value, fc_value),
            }
        )
    return {
        "converged": converged,
        "final_residual": final_residual,
        "max_budget_relative_residual": float(np.max(rel)) if rel.size else 0.0,
        "major_overlap_ge_1e-4": metrics,
        "species": species,
        "condensate_sum": float(np.sum(condensates)),
        "element_fractions": _element_fraction_rows(
            setup=setup,
            budget=budget_np,
            condensates=condensates,
        ),
        "top_condensates": [
            {
                "name": str(setup.condensate_species[int(index)]),
                "amount": float(condensates[int(index)]),
            }
            for index in np.argsort(-condensates)[:10]
            if float(condensates[int(index)]) > 0.0
        ],
    }


def _analyze_case(args: argparse.Namespace, setup: Any, case: str) -> dict[str, Any]:
    family, layer_index = _parse_case(case)
    full_definition = FRESH_CURATED_PROFILES[family]
    definition = _one_layer_definition(family, layer_index)
    budget = element_budget_for_profile(setup, full_definition)
    budget_np = np.asarray(jax.device_get(budget), dtype=np.float64)
    fastchem = _run_fastchem_profile(args, definition)
    fc_scaled = _aligned_fastchem_amounts(
        setup=setup,
        fastchem=fastchem,
        layer_index=0,
        target=budget_np,
    )
    fc_condensates = np.asarray(fc_scaled["condensate_amounts"], dtype=np.float64)
    rows = []
    for support_mode in args.support_modes:
        support_indices = _support_indices_for_mode(
            setup=setup,
            definition=full_definition,
            fastchem=fastchem,
            mode=support_mode,
        )
        if not support_indices:
            continue
        for seed_fraction in args.seed_fractions:
            support_amounts = _budget_seed_amounts(
                setup=setup,
                budget=budget,
                support_indices=support_indices,
                seed_fraction=float(seed_fraction),
            )
            rows.append(
                {
                    "support_mode": support_mode,
                    "init_mode": "budget_seed",
                    "seed_fraction": float(seed_fraction),
                    "support_count": len(support_indices),
                    "support_names": tuple(
                        str(setup.condensate_species[int(index)]) for index in support_indices
                    ),
                    "init_sum": float(np.sum(support_amounts)),
                    **_run_seed_variant(
                        setup=setup,
                        definition=definition,
                        budget=budget,
                        fastchem=fastchem,
                        support_indices=support_indices,
                        support_amounts=support_amounts,
                        max_inner_iterations=args.max_inner_iterations,
                        depleted_gas_init=args.depleted_gas_init,
                    ),
                }
            )
        if args.include_fastchem_scaled_init:
            support_amounts = _fastchem_scaled_amounts(
                setup=setup,
                budget=budget_np,
                fastchem=fastchem,
                support_indices=support_indices,
            )
            rows.append(
                {
                    "support_mode": support_mode,
                    "init_mode": "fastchem_scaled",
                    "seed_fraction": None,
                    "support_count": len(support_indices),
                    "support_names": tuple(
                        str(setup.condensate_species[int(index)]) for index in support_indices
                    ),
                    "init_sum": float(np.sum(support_amounts)),
                    **_run_seed_variant(
                        setup=setup,
                        definition=definition,
                        budget=budget,
                        fastchem=fastchem,
                        support_indices=support_indices,
                        support_amounts=support_amounts,
                        max_inner_iterations=args.max_inner_iterations,
                        depleted_gas_init=args.depleted_gas_init,
                    ),
                }
            )
    return {
        "family": family,
        "layer_index": int(layer_index),
        "temperature": float(definition.temperatures[0]),
        "pressure": float(definition.pressures[0]),
        "fastchem4_condensate_sum": float(np.sum(fc_condensates)),
        "fastchem4_element_fractions": _element_fraction_rows(
            setup=setup,
            budget=budget_np,
            condensates=fc_condensates,
        ),
        "fastchem4_top_condensates": [
            {
                "name": str(setup.condensate_species[int(index)]),
                "amount": float(fc_condensates[int(index)]),
            }
            for index in np.argsort(-fc_condensates)[:10]
            if float(fc_condensates[int(index)]) > 0.0
        ],
        "variants": rows,
    }


def _metric_value(row: dict[str, Any]) -> float | None:
    value = row["major_overlap_ge_1e-4"]["max_abs_log10_ratio"]
    return None if value is None else float(value)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Condensate Seed Partition Sweep",
        "",
        f"- timestamp UTC: `{report['timestamp_utc']}`",
        f"- JAX backend: `{report['environment']['jax_default_backend']}`",
        "",
        "| family | layer | support | init | seed_fraction | support | init sum | final sum | C frac | O frac | CH4 dex | major >=1e-4 dex | budget rel | residual |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        for row in case["variants"]:
            fractions = {
                item["element"]: item["condensed_fraction"]
                for item in row["element_fractions"]
            }
            species = {item["species"]: item["dex"] for item in row["species"]}
            metric = _metric_value(row)
            lines.append(
                "| {family} | {layer} | {support_mode} | {init_mode} | {seed} | {support_count} | {init_sum:.3g} | {final_sum:.3g} | {cfrac:.3g} | {ofrac:.3g} | {ch4} | {major} | {budget:.3g} | {resid:.3g} |".format(
                    family=case["family"],
                    layer=case["layer_index"],
                    support_mode=row["support_mode"],
                    init_mode=row["init_mode"],
                    seed=(
                        "n/a"
                        if row["seed_fraction"] is None
                        else f"{float(row['seed_fraction']):.3g}"
                    ),
                    support_count=row["support_count"],
                    init_sum=row["init_sum"],
                    final_sum=row["condensate_sum"],
                    cfrac=float(fractions.get("C") or 0.0),
                    ofrac=float(fractions.get("O") or 0.0),
                    ch4=(
                        "n/a"
                        if species.get("C1H4") is None
                        else f"{float(species['C1H4']):.3g}"
                    ),
                    major="n/a" if metric is None else f"{metric:.3g}",
                    budget=row["max_budget_relative_residual"],
                    resid=row["final_residual"],
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument(
        "--support-modes",
        nargs="+",
        default=("curated", "curated_plus_fastchem_active"),
        choices=("curated", "fastchem_active", "curated_plus_fastchem_active"),
    )
    parser.add_argument("--seed-fractions", nargs="+", type=float, default=DEFAULT_SEED_FRACTIONS)
    parser.add_argument("--include-fastchem-scaled-init", action="store_true")
    parser.add_argument("--depleted-gas-init", action="store_true")
    parser.add_argument("--max-inner-iterations", type=int, default=150)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="cpu_condensate_seed_partition_sweep")
    parser.add_argument("--fastchem-root", default="FastChem")
    parser.add_argument("--fastchem-element-file", default="input/element_abundances/asplund_2021.dat")
    parser.add_argument("--fastchem-gas-logk", default="input/logK/logK_wo_ions.dat")
    parser.add_argument("--fastchem-cond-logk", default="input/logK/logK_condensates.dat")
    parser.add_argument("--fastchem-condensation", default="equilibrium")
    parser.add_argument("--fastchem-verbosity", type=int, default=0)
    parser.add_argument("--exogibbs-element-file", default="FastChem4/element_abundances/asplund_2021.dat")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.update("jax_enable_x64", True)
    setup = condensate_chemical_setup(
        species_defalt_elements=False,
        element_file=args.exogibbs_element_file,
        silent=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "exogibbs_condensate_seed_partition_sweep_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "jax_default_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "settings": {
            "cases": tuple(args.cases),
            "support_modes": tuple(args.support_modes),
            "seed_fractions": tuple(float(value) for value in args.seed_fractions),
            "include_fastchem_scaled_init": bool(args.include_fastchem_scaled_init),
            "depleted_gas_init": bool(args.depleted_gas_init),
            "max_inner_iterations": int(args.max_inner_iterations),
        },
        "cases": [_analyze_case(args, setup, case) for case in args.cases],
    }
    json_path = args.output_dir / f"{args.output_prefix}.json"
    md_path = args.output_dir / f"{args.output_prefix}.md"
    json_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
