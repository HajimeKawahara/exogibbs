"""Diagnostic-only RGIE support / inactive-complementarity audit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmarks.common import current_timestamp_utc
from benchmarks.common import device_for_platform
from benchmarks.common import to_python
from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.core import _compute_gk
from exogibbs.optimize.minimize import minimize_gibbs_core
from exogibbs.optimize.minimize_cond import CondensateEquilibriumInit
from exogibbs.optimize.minimize_cond import CondensateRGIEStartupConfig
from exogibbs.optimize.minimize_cond import minimize_gibbs_cond
from exogibbs.optimize.pipm_rgie_cond import _recompute_pi_for_residual
from exogibbs.optimize.pipm_rgie_cond import build_rgie_condensate_init_from_policy
from exogibbs.optimize.pipm_rgie_cond import build_rgie_support_candidate_indices
from exogibbs.optimize.pipm_rgie_cond import summarize_rgie_inactive_driving
from exogibbs.presets.fastchem import chemsetup as gas_chemsetup
from exogibbs.presets.fastchem_cond import chemsetup as cond_chemsetup


DEFAULT_LAYER_INDICES = (0, 45, 90)
DEFAULT_EPSILONS = (0.0, -5.0, -10.0)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "rgie_support_inactive_audit.json"
DEFAULT_TRACES_OUTPUT = REPO_ROOT / "results" / "rgie_support_inactive_traces.json"

STARTUP_POLICIES = (
    {"name": "ratio_uniform_r0_1e-2", "policy": "ratio_uniform_r0", "kwargs": {"r0": 1.0e-2}},
    {"name": "legacy_absolute_m0_1e-30", "policy": "absolute_uniform_m0", "kwargs": {"m0": 1.0e-30}},
)

SUPPORT_MECHANISMS = (
    "current_support_updating_active_set",
    "semismooth_candidate",
    "smoothed_semismooth_candidate",
    "augmented_semismooth_candidate",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYER_INDICES))
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(DEFAULT_EPSILONS))
    parser.add_argument("--gas-max-iter", type=int, default=1000)
    parser.add_argument("--rgie-max-iter", type=int, default=60)
    parser.add_argument("--top-inactive-k", type=int, default=3)
    parser.add_argument("--top-drop-k", type=int, default=3)
    parser.add_argument(
        "--include-legacy-control",
        action="store_true",
        help="Include the legacy absolute-m0 startup as a secondary control.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--traces-output", type=Path, default=DEFAULT_TRACES_OUTPUT)
    return parser


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _fraction(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(sum(1 for value in values if value) / len(values))


def _load_support_case(layer_index: int) -> dict[str, Any]:
    sparse_path = REPO_ROOT / "results" / f"sparse_layer{layer_index}_top1.json"
    if sparse_path.exists():
        payload = json.loads(sparse_path.read_text())
        layer_result = payload["results"][0]
        support_cfg = layer_result["configs"]["sparse_lp20_top1"]
        return {
            "temperature_K": float(payload["benchmark"]["temperature_K"]),
            "pressure_bar": float(layer_result["pressure_bar"]),
            "support_names": list(support_cfg["final_support_names"]),
            "support_source_json": str(sparse_path),
        }

    fallback_path = REPO_ROOT / "results" / "condensate_pipm_line_search_convergence_audit.json"
    payload = json.loads(fallback_path.read_text())
    runs = payload["runs"]
    for run in runs:
        if int(run["layer_index"]) == int(layer_index):
            return {
                "temperature_K": float(run["temperature_K"]),
                "pressure_bar": float(run["pressure_bar"]),
                "support_names": list(run["support_names"]),
                "support_source_json": str(fallback_path),
            }
    raise FileNotFoundError(f"Missing support metadata for layer {layer_index}.")


def _build_profile_states(element_vector: jax.Array, layer_indices: list[int]) -> list[dict[str, Any]]:
    states = []
    for layer_index in layer_indices:
        support_case = _load_support_case(layer_index)
        state = ThermoState(
            temperature=jnp.asarray(support_case["temperature_K"], dtype=jnp.float64),
            ln_normalized_pressure=jnp.log(jnp.asarray(support_case["pressure_bar"], dtype=jnp.float64)),
            element_vector=jnp.asarray(element_vector, dtype=jnp.float64),
        )
        states.append(
            {
                "layer_index": int(layer_index),
                "temperature_K": support_case["temperature_K"],
                "pressure_bar": support_case["pressure_bar"],
                "support_names": support_case["support_names"],
                "support_source_json": support_case["support_source_json"],
                "state": state,
            }
        )
    return states


def _build_shared_gas_init(
    state: ThermoState,
    gas_setup: Any,
    gas_hvector: jax.Array,
    gas_max_iter: int,
) -> dict[str, Any]:
    ln_nk_init0 = jnp.zeros((gas_setup.formula_matrix.shape[1],), dtype=jnp.float64)
    ln_ntot_init0 = jnp.asarray(0.0, dtype=jnp.float64)
    ln_nk_gas, ln_ntot_gas, gas_n_iter, gas_final_residual = minimize_gibbs_core(
        state,
        ln_nk_init0,
        ln_ntot_init0,
        gas_setup.formula_matrix,
        lambda _temperature: gas_hvector,
        epsilon_crit=1.0e-12,
        max_iter=gas_max_iter,
    )
    return {
        "ln_nk": jnp.asarray(ln_nk_gas, dtype=jnp.float64),
        "ln_ntot": jnp.asarray(ln_ntot_gas, dtype=jnp.float64),
        "gas_init_n_iter": int(gas_n_iter),
        "gas_init_final_residual": float(gas_final_residual),
    }


def _startup_seed_ln_mk(
    startup_policy: dict[str, Any],
    support_indices: jax.Array,
    epsilon: float,
) -> jax.Array:
    return build_rgie_condensate_init_from_policy(
        epsilon=epsilon,
        support_indices=support_indices,
        startup_policy=startup_policy["policy"],
        dtype=jnp.float64,
        **startup_policy["kwargs"],
    )


def _startup_config(startup_policy: dict[str, Any]) -> CondensateRGIEStartupConfig | None:
    if startup_policy["policy"] == "ratio_uniform_r0":
        return CondensateRGIEStartupConfig(policy="ratio_uniform_r0", r0=startup_policy["kwargs"]["r0"])
    return None


def _support_names(cond_setup: Any, support_indices: jax.Array) -> list[str]:
    return [str(cond_setup.species[int(index)]) for index in jax.device_get(support_indices)]


def _support_signature_export(cond_setup: Any, support_indices: jax.Array) -> dict[str, Any]:
    support_indices = jnp.asarray(support_indices, dtype=jnp.int32)
    species = _support_names(cond_setup, support_indices)
    entries = []
    element_coverage = set()
    for local_pos, cond_index in enumerate(jax.device_get(support_indices)):
        stoich = jnp.asarray(cond_setup.formula_matrix[:, int(cond_index)])
        element_indices = [int(i) for i in range(stoich.shape[0]) if float(stoich[i]) > 0.0]
        elements = [str(cond_setup.elements[i]) for i in element_indices]
        for element in elements:
            element_coverage.add(element)
        entries.append(
            {
                "species": species[local_pos],
                "associated_elements": elements,
                "family_signature": "+".join(sorted(elements)),
            }
        )
    return {
        "support_names": species,
        "family_signatures": sorted({entry["family_signature"] for entry in entries}),
        "associated_element_coverage": sorted(element_coverage),
        "entries": entries,
    }


def _jaccard_overlap(lhs: set[str], rhs: set[str]) -> float | None:
    union = lhs | rhs
    if not union:
        return None
    return float(len(lhs & rhs) / len(union))


def _load_fastchem_condensate_support_table(
    cond_output_path: Path,
    *,
    n_elements: int,
    positive_density_floor: float = 0.0,
) -> list[dict[str, Any]]:
    if not cond_output_path.exists():
        return []

    lines = [line.strip() for line in cond_output_path.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        return []

    header = lines[0].split("\t")
    condensate_names = [token.strip() for token in header[2 + n_elements :]]
    rows = []
    for row_index, line in enumerate(lines[1:]):
        fields = line.split("\t")
        if len(fields) != len(header):
            continue
        pressure_bar = float(fields[0])
        temperature_k = float(fields[1])
        densities = [float(value) for value in fields[2 + n_elements :]]
        support_names = [
            name for name, density in zip(condensate_names, densities) if density > positive_density_floor
        ]
        rows.append(
            {
                "row_index": int(row_index),
                "pressure_bar": pressure_bar,
                "temperature_K": temperature_k,
                "support_names": support_names,
            }
        )
    return rows


def _match_fastchem_condensate_support_row(
    rows: list[dict[str, Any]],
    *,
    pressure_bar: float,
    temperature_k: float,
) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (
            abs(float(row["pressure_bar"]) - float(pressure_bar)),
            abs(float(row["temperature_K"]) - float(temperature_k)),
        ),
    )


def _build_candidate_warm_init(
    *,
    candidate_indices: jax.Array,
    base_support_indices: jax.Array,
    base_ln_mk: jax.Array,
    ln_nk: jax.Array,
    ln_ntot: jax.Array,
    startup_policy: dict[str, Any],
    epsilon: float,
) -> CondensateEquilibriumInit:
    candidate_indices = jnp.asarray(candidate_indices, dtype=jnp.int32)
    base_support_indices = jnp.asarray(base_support_indices, dtype=jnp.int32)
    seed_ln_mk = _startup_seed_ln_mk(startup_policy, candidate_indices, epsilon)
    values = {int(index): float(value) for index, value in zip(jax.device_get(base_support_indices), jax.device_get(base_ln_mk))}
    candidate_ln_mk = []
    for local_pos, cond_index in enumerate(jax.device_get(candidate_indices)):
        candidate_ln_mk.append(values.get(int(cond_index), float(seed_ln_mk[local_pos])))
    return CondensateEquilibriumInit(
        ln_nk=jnp.asarray(ln_nk, dtype=jnp.float64),
        ln_mk=jnp.asarray(candidate_ln_mk, dtype=jnp.float64),
        ln_ntot=jnp.asarray(ln_ntot, dtype=jnp.float64),
    )


def _compute_local_support_metrics(
    *,
    state: ThermoState,
    result: Any,
    support_indices: jax.Array,
    formula_matrix: jax.Array,
    formula_matrix_cond_active: jax.Array,
    formula_matrix_cond_full: jax.Array,
    b: jax.Array,
    hvector: jax.Array,
    hvector_cond_active: jax.Array,
    hvector_cond_full: jax.Array,
    epsilon: float,
    cond_setup: Any,
    runtime_seconds: float,
) -> dict[str, Any]:
    ln_nk = jnp.asarray(result.ln_nk, dtype=jnp.float64)
    ln_mk = jnp.asarray(result.ln_mk, dtype=jnp.float64)
    ln_ntot = jnp.asarray(result.ln_ntot, dtype=jnp.float64)
    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntot = jnp.exp(ln_ntot)
    gk = _compute_gk(state.temperature, ln_nk, ln_ntot, hvector, state.ln_normalized_pressure)
    pi = _recompute_pi_for_residual(
        nk,
        mk,
        ntot,
        formula_matrix,
        formula_matrix_cond_active,
        b,
        gk,
        hvector_cond_active,
        epsilon,
    )
    active_driving = formula_matrix_cond_active.T @ pi - hvector_cond_active
    full_driving = formula_matrix_cond_full.T @ pi - hvector_cond_full
    gas_stationarity = formula_matrix.T @ pi - gk
    feasibility_vector = formula_matrix @ nk + formula_matrix_cond_active @ mk - b
    ntot_residual = jnp.sum(nk) - ntot
    complementarity = mk * active_driving + jnp.exp(jnp.asarray(epsilon, dtype=jnp.float64))
    inactive_summary = summarize_rgie_inactive_driving(
        full_driving,
        support_indices,
        condensate_species_names=cond_setup.species,
        top_k=5,
    )
    feasibility_residual_inf = float(
        max(
            float(jnp.max(jnp.abs(feasibility_vector))),
            abs(float(ntot_residual)),
        )
    )
    true_stationarity_residual_inf = float(
        max(
            float(jnp.max(jnp.abs(gas_stationarity))),
            float(jnp.max(jnp.abs(active_driving))) if active_driving.size else 0.0,
        )
    )
    complementarity_residual_inf = float(
        jnp.max(jnp.abs(complementarity)) if complementarity.size else 0.0
    )
    scalar_merit = float(
        max(
            feasibility_residual_inf,
            true_stationarity_residual_inf,
            complementarity_residual_inf,
            inactive_summary["max_positive_inactive_driving"],
        )
    )
    diagnostics = result.diagnostics.asdict()
    return {
        "support_indices": [int(index) for index in jax.device_get(support_indices)],
        "support_names": _support_names(cond_setup, support_indices),
        "support_size": int(support_indices.shape[0]),
        "solver_success": bool(result.diagnostics.converged),
        "converged": bool(result.diagnostics.converged),
        "n_iter": int(result.diagnostics.n_iter),
        "final_residual": float(result.diagnostics.final_residual),
        "feasibility_residual_inf": feasibility_residual_inf,
        "true_stationarity_residual_inf": true_stationarity_residual_inf,
        "complementarity_residual_inf": complementarity_residual_inf,
        "max_positive_inactive_driving": float(inactive_summary["max_positive_inactive_driving"]),
        "inactive_positive_count": int(inactive_summary["inactive_positive_count"]),
        "top_inactive_names": list(inactive_summary["top_inactive_names"]),
        "top_inactive_driving": list(inactive_summary["top_inactive_driving"]),
        "top_positive_inactive_indices": list(inactive_summary["top_positive_inactive_indices"]),
        "active_driving": to_python(jax.device_get(active_driving)),
        "active_ln_mk": to_python(jax.device_get(ln_mk)),
        "full_driving": to_python(jax.device_get(full_driving)),
        "gas_stationarity_inf": float(jnp.max(jnp.abs(gas_stationarity))),
        "scalar_merit": scalar_merit,
        "runtime_seconds": float(runtime_seconds),
        "diagnostics": to_python(jax.device_get(diagnostics)),
        "support_signature_export": _support_signature_export(cond_setup, support_indices),
    }


def _solve_support(
    *,
    state: ThermoState,
    init: CondensateEquilibriumInit,
    support_indices: jax.Array,
    gas_setup: Any,
    cond_setup: Any,
    gas_hvector: jax.Array,
    cond_hvector_full: jax.Array,
    epsilon: float,
    max_iter: int,
    startup_config: CondensateRGIEStartupConfig | None,
) -> tuple[Any, dict[str, Any]]:
    support_indices = jnp.asarray(support_indices, dtype=jnp.int32)
    formula_matrix_cond_active = jnp.asarray(cond_setup.formula_matrix[:, support_indices], dtype=jnp.float64)
    hvector_cond_active = jnp.asarray(cond_hvector_full[support_indices], dtype=jnp.float64)
    start = perf_counter()
    result = minimize_gibbs_cond(
        state,
        init=init,
        formula_matrix=jnp.asarray(gas_setup.formula_matrix, dtype=jnp.float64),
        formula_matrix_cond=formula_matrix_cond_active,
        hvector_func=lambda _temperature: jnp.asarray(gas_hvector, dtype=jnp.float64),
        hvector_cond_func=lambda _temperature: hvector_cond_active,
        epsilon=epsilon,
        residual_crit=float(math.exp(epsilon)),
        max_iter=max_iter,
        startup_config=startup_config,
    )
    runtime_seconds = perf_counter() - start
    metrics = _compute_local_support_metrics(
        state=state,
        result=result,
        support_indices=support_indices,
        formula_matrix=jnp.asarray(gas_setup.formula_matrix, dtype=jnp.float64),
        formula_matrix_cond_active=formula_matrix_cond_active,
        formula_matrix_cond_full=jnp.asarray(cond_setup.formula_matrix, dtype=jnp.float64),
        b=jnp.asarray(state.element_vector, dtype=jnp.float64),
        hvector=jnp.asarray(gas_hvector, dtype=jnp.float64),
        hvector_cond_active=hvector_cond_active,
        hvector_cond_full=jnp.asarray(cond_hvector_full, dtype=jnp.float64),
        epsilon=epsilon,
        cond_setup=cond_setup,
        runtime_seconds=runtime_seconds,
    )
    return result, metrics


def _proposal_record(
    label: str,
    proposal_kind: str,
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    tol = 1.0e-12
    return {
        "proposal_label": label,
        "proposal_kind": proposal_kind,
        "support_names": candidate_metrics["support_names"],
        "support_size": candidate_metrics["support_size"],
        "feasibility_residual_inf": candidate_metrics["feasibility_residual_inf"],
        "true_stationarity_residual_inf": candidate_metrics["true_stationarity_residual_inf"],
        "complementarity_residual_inf": candidate_metrics["complementarity_residual_inf"],
        "max_positive_inactive_driving": candidate_metrics["max_positive_inactive_driving"],
        "scalar_merit": candidate_metrics["scalar_merit"],
        "improves_feasibility": candidate_metrics["feasibility_residual_inf"] < baseline_metrics["feasibility_residual_inf"] - tol,
        "improves_stationarity": candidate_metrics["true_stationarity_residual_inf"] < baseline_metrics["true_stationarity_residual_inf"] - tol,
        "improves_complementarity": candidate_metrics["complementarity_residual_inf"] < baseline_metrics["complementarity_residual_inf"] - tol,
        "improves_inactive_max_positive_driving": candidate_metrics["max_positive_inactive_driving"] < baseline_metrics["max_positive_inactive_driving"] - tol,
        "improves_scalar_merit": candidate_metrics["scalar_merit"] < baseline_metrics["scalar_merit"] - tol,
    }


def _fastchem_support_summary(
    current_metrics: dict[str, Any],
    *,
    cond_setup: Any,
    fastchem_row: dict[str, Any] | None,
) -> dict[str, Any]:
    exo_export = current_metrics["support_signature_export"]
    if fastchem_row is None:
        return {
            "available": False,
            "message": "FastChem condensate output row was not available for this state.",
            "exact_support_overlap": None,
            "jaccard_overlap": None,
            "family_overlap": None,
            "associated_element_overlap": None,
            "exo_support_export": exo_export,
        }

    matched_fastchem_names = [name for name in fastchem_row["support_names"] if name in cond_setup.species]
    fastchem_indices = [cond_setup.species.index(name) for name in matched_fastchem_names]
    fastchem_export = _support_signature_export(
        cond_setup,
        jnp.asarray(fastchem_indices, dtype=jnp.int32),
    )
    exo_set = set(exo_export["support_names"])
    fastchem_set = set(fastchem_export["support_names"])
    exact_overlap = sorted(exo_set & fastchem_set)

    return {
        "available": True,
        "message": "FastChem condensate support parsed from local FastChem output.",
        "matched_row_index": int(fastchem_row["row_index"]),
        "matched_pressure_bar": float(fastchem_row["pressure_bar"]),
        "matched_temperature_K": float(fastchem_row["temperature_K"]),
        "fastchem_support_names": fastchem_export["support_names"],
        "exact_support_overlap": exact_overlap,
        "exact_support_overlap_fraction": (
            None if not exo_export["support_names"] else float(len(exact_overlap) / len(exo_export["support_names"]))
        ),
        "jaccard_overlap": _jaccard_overlap(exo_set, fastchem_set),
        "family_overlap": _jaccard_overlap(
            set(exo_export["family_signatures"]),
            set(fastchem_export["family_signatures"]),
        ),
        "associated_element_overlap": _jaccard_overlap(
            set(exo_export["associated_element_coverage"]),
            set(fastchem_export["associated_element_coverage"]),
        ),
        "exo_only_support": sorted(exo_set - fastchem_set),
        "fastchem_only_support": sorted(fastchem_set - exo_set),
        "exo_support_export": exo_export,
        "fastchem_support_export": fastchem_export,
    }


def main() -> None:
    args = build_parser().parse_args()
    device_for_platform(args.platform)

    gas_setup = gas_chemsetup()
    cond_setup = cond_chemsetup()
    startup_policies = STARTUP_POLICIES if args.include_legacy_control else STARTUP_POLICIES[:1]
    fastchem_cond_rows = _load_fastchem_condensate_support_table(
        REPO_ROOT / "FastChem" / "output" / "condensates.dat",
        n_elements=len(cond_setup.elements),
    )

    profile_states = _build_profile_states(gas_setup.element_vector_reference, args.layers)
    cases = []
    summary_rows = []

    for startup_policy in startup_policies:
        startup_case_records = []
        for layer_meta in profile_states:
            state = layer_meta["state"]
            gas_hvector = jnp.asarray(gas_setup.hvector_func(state.temperature), dtype=jnp.float64)
            cond_hvector_full = jnp.asarray(cond_setup.hvector_func(state.temperature), dtype=jnp.float64)
            gas_init = _build_shared_gas_init(state, gas_setup, gas_hvector, args.gas_max_iter)
            base_support_indices = jnp.asarray(
                [cond_setup.species.index(name) for name in layer_meta["support_names"]],
                dtype=jnp.int32,
            )
            for epsilon in args.epsilons:
                base_init = CondensateEquilibriumInit(
                    ln_nk=jnp.asarray(gas_init["ln_nk"], dtype=jnp.float64),
                    ln_mk=_startup_seed_ln_mk(startup_policy, base_support_indices, epsilon),
                    ln_ntot=jnp.asarray(gas_init["ln_ntot"], dtype=jnp.float64),
                )
                startup_config = _startup_config(startup_policy)
                fastchem_row = _match_fastchem_condensate_support_row(
                    fastchem_cond_rows,
                    pressure_bar=layer_meta["pressure_bar"],
                    temperature_k=layer_meta["temperature_K"],
                )
                base_result, base_metrics = _solve_support(
                    state=state,
                    init=base_init,
                    support_indices=base_support_indices,
                    gas_setup=gas_setup,
                    cond_setup=cond_setup,
                    gas_hvector=gas_hvector,
                    cond_hvector_full=cond_hvector_full,
                    epsilon=epsilon,
                    max_iter=args.rgie_max_iter,
                    startup_config=startup_config,
                )
                base_metrics["fastchem_support_comparison"] = _fastchem_support_summary(
                    base_metrics,
                    cond_setup=cond_setup,
                    fastchem_row=fastchem_row,
                )
                mechanism_results = {
                    "current_support_updating_active_set": base_metrics,
                }

                candidate_builds = {
                    name: build_rgie_support_candidate_indices(
                        base_support_indices,
                        full_driving=jnp.asarray(base_metrics["full_driving"], dtype=jnp.float64),
                        active_ln_mk=jnp.asarray(base_metrics["active_ln_mk"], dtype=jnp.float64),
                        active_driving=jnp.asarray(base_metrics["active_driving"], dtype=jnp.float64),
                        mechanism_name=name,
                        inactive_positive_ranked=base_metrics["top_positive_inactive_indices"],
                    )
                    for name in SUPPORT_MECHANISMS[1:]
                }

                for mechanism_name, candidate in candidate_builds.items():
                    candidate_support = jnp.asarray(candidate["support_indices"], dtype=jnp.int32)
                    if jnp.array_equal(candidate_support, base_support_indices):
                        mechanism_results[mechanism_name] = {
                            **base_metrics,
                            "support_indices": [int(index) for index in jax.device_get(candidate_support)],
                            "support_names": _support_names(cond_setup, candidate_support),
                            "candidate_added_indices": list(candidate["added_indices"]),
                            "candidate_dropped_indices": list(candidate["dropped_indices"]),
                            "candidate_reused_current_support": True,
                            "fastchem_support_comparison": _fastchem_support_summary(
                                base_metrics,
                                cond_setup=cond_setup,
                                fastchem_row=fastchem_row,
                            ),
                        }
                        continue

                    candidate_init = _build_candidate_warm_init(
                        candidate_indices=candidate_support,
                        base_support_indices=base_support_indices,
                        base_ln_mk=base_result.ln_mk,
                        ln_nk=base_result.ln_nk,
                        ln_ntot=base_result.ln_ntot,
                        startup_policy=startup_policy,
                        epsilon=epsilon,
                    )
                    _candidate_result, candidate_metrics = _solve_support(
                        state=state,
                        init=candidate_init,
                        support_indices=candidate_support,
                        gas_setup=gas_setup,
                        cond_setup=cond_setup,
                        gas_hvector=gas_hvector,
                        cond_hvector_full=cond_hvector_full,
                        epsilon=epsilon,
                        max_iter=args.rgie_max_iter,
                        startup_config=None,
                    )
                    candidate_metrics["candidate_added_indices"] = list(candidate["added_indices"])
                    candidate_metrics["candidate_dropped_indices"] = list(candidate["dropped_indices"])
                    candidate_metrics["candidate_reused_current_support"] = False
                    candidate_metrics["fastchem_support_comparison"] = _fastchem_support_summary(
                        candidate_metrics,
                        cond_setup=cond_setup,
                        fastchem_row=fastchem_row,
                    )
                    mechanism_results[mechanism_name] = candidate_metrics

                inactive_positive_ranked = list(base_metrics["top_positive_inactive_indices"])
                weak_active_ranked = build_rgie_support_candidate_indices(
                    base_support_indices,
                    full_driving=jnp.asarray(base_metrics["full_driving"], dtype=jnp.float64),
                    active_ln_mk=jnp.asarray(base_metrics["active_ln_mk"], dtype=jnp.float64),
                    active_driving=jnp.asarray(base_metrics["active_driving"], dtype=jnp.float64),
                    mechanism_name="current_support_updating_active_set",
                    inactive_positive_ranked=inactive_positive_ranked,
                )["weak_active_ranked"]

                proposals = []
                for add_index in inactive_positive_ranked[: args.top_inactive_k]:
                    candidate_support = jnp.sort(
                        jnp.unique(jnp.asarray(list(base_metrics["support_indices"]) + [int(add_index)], dtype=jnp.int32))
                    )
                    candidate_init = _build_candidate_warm_init(
                        candidate_indices=candidate_support,
                        base_support_indices=base_support_indices,
                        base_ln_mk=base_result.ln_mk,
                        ln_nk=base_result.ln_nk,
                        ln_ntot=base_result.ln_ntot,
                        startup_policy=startup_policy,
                        epsilon=epsilon,
                    )
                    _candidate_result, candidate_metrics = _solve_support(
                        state=state,
                        init=candidate_init,
                        support_indices=candidate_support,
                        gas_setup=gas_setup,
                        cond_setup=cond_setup,
                        gas_hvector=gas_hvector,
                        cond_hvector_full=cond_hvector_full,
                        epsilon=epsilon,
                        max_iter=args.rgie_max_iter,
                        startup_config=None,
                    )
                    proposals.append(
                        _proposal_record(
                            f"add:{cond_setup.species[int(add_index)]}",
                            "single_add",
                            candidate_metrics,
                            base_metrics,
                        )
                    )

                for drop_index in weak_active_ranked[: args.top_drop_k]:
                    candidate_support = jnp.asarray(
                        [idx for idx in base_metrics["support_indices"] if int(idx) != int(drop_index)],
                        dtype=jnp.int32,
                    )
                    if candidate_support.shape[0] == 0:
                        continue
                    candidate_init = _build_candidate_warm_init(
                        candidate_indices=candidate_support,
                        base_support_indices=base_support_indices,
                        base_ln_mk=base_result.ln_mk,
                        ln_nk=base_result.ln_nk,
                        ln_ntot=base_result.ln_ntot,
                        startup_policy=startup_policy,
                        epsilon=epsilon,
                    )
                    _candidate_result, candidate_metrics = _solve_support(
                        state=state,
                        init=candidate_init,
                        support_indices=candidate_support,
                        gas_setup=gas_setup,
                        cond_setup=cond_setup,
                        gas_hvector=gas_hvector,
                        cond_hvector_full=cond_hvector_full,
                        epsilon=epsilon,
                        max_iter=args.rgie_max_iter,
                        startup_config=None,
                    )
                    proposals.append(
                        _proposal_record(
                            f"drop:{cond_setup.species[int(drop_index)]}",
                            "single_drop",
                            candidate_metrics,
                            base_metrics,
                        )
                    )

                if inactive_positive_ranked and weak_active_ranked:
                    add_index = int(inactive_positive_ranked[0])
                    for drop_index in weak_active_ranked[: min(2, len(weak_active_ranked))]:
                        candidate_indices = [
                            idx for idx in base_metrics["support_indices"] if int(idx) != int(drop_index)
                        ]
                        candidate_indices.append(add_index)
                        candidate_support = jnp.sort(jnp.unique(jnp.asarray(candidate_indices, dtype=jnp.int32)))
                        candidate_init = _build_candidate_warm_init(
                            candidate_indices=candidate_support,
                            base_support_indices=base_support_indices,
                            base_ln_mk=base_result.ln_mk,
                            ln_nk=base_result.ln_nk,
                            ln_ntot=base_result.ln_ntot,
                            startup_policy=startup_policy,
                            epsilon=epsilon,
                        )
                        _candidate_result, candidate_metrics = _solve_support(
                            state=state,
                            init=candidate_init,
                            support_indices=candidate_support,
                            gas_setup=gas_setup,
                            cond_setup=cond_setup,
                            gas_hvector=gas_hvector,
                            cond_hvector_full=cond_hvector_full,
                            epsilon=epsilon,
                            max_iter=args.rgie_max_iter,
                            startup_config=None,
                        )
                        proposals.append(
                            _proposal_record(
                                f"swap:{cond_setup.species[int(drop_index)]}->{cond_setup.species[int(add_index)]}",
                                "single_swap",
                                candidate_metrics,
                                base_metrics,
                            )
                        )

                case_record = {
                    "layer_index": int(layer_meta["layer_index"]),
                    "epsilon": float(epsilon),
                    "startup_policy_name": startup_policy["name"],
                    "support_source_json": layer_meta["support_source_json"],
                    "mechanisms": mechanism_results,
                    "support_move_probes": proposals,
                    "fastchem_support_comparison": base_metrics["fastchem_support_comparison"],
                }
                cases.append(case_record)
                startup_case_records.append(case_record)

        for mechanism_name in SUPPORT_MECHANISMS:
            mechanism_metrics = [case["mechanisms"][mechanism_name] for case in startup_case_records]
            current_metrics = [case["mechanisms"]["current_support_updating_active_set"] for case in startup_case_records]
            summary_rows.append(
                {
                    "startup_policy_name": startup_policy["name"],
                    "mechanism_name": mechanism_name,
                    "mean_support_size": _mean([row["support_size"] for row in mechanism_metrics]),
                    "mean_max_positive_inactive_driving": _mean(
                        [row["max_positive_inactive_driving"] for row in mechanism_metrics]
                    ),
                    "mean_scalar_merit": _mean([row["scalar_merit"] for row in mechanism_metrics]),
                    "fraction_beats_current_scalar_merit": _fraction(
                        [
                            row["scalar_merit"] < current["scalar_merit"] - 1.0e-12
                            for row, current in zip(mechanism_metrics, current_metrics)
                        ]
                    ),
                    "mean_fastchem_jaccard_overlap": _mean(
                        [
                            case["mechanisms"][mechanism_name]["fastchem_support_comparison"]["jaccard_overlap"]
                            for case in startup_case_records
                            if case["mechanisms"][mechanism_name]["fastchem_support_comparison"]["available"]
                        ]
                    ),
                }
            )

    all_proposals = [proposal for case in cases for proposal in case["support_move_probes"]]
    semismooth_rows = [row for row in summary_rows if row["mechanism_name"] in SUPPORT_MECHANISMS[1:]]
    fastchem_rows = [
        case["fastchem_support_comparison"]
        for case in cases
        if case["fastchem_support_comparison"]["available"]
    ]
    support_moves_help = _fraction([proposal["improves_scalar_merit"] for proposal in all_proposals]) or 0.0
    semismooth_help = _mean(
        [row["fraction_beats_current_scalar_merit"] for row in semismooth_rows if row["fraction_beats_current_scalar_merit"] is not None]
    ) or 0.0
    fastchem_jaccard_mean = _mean([row["jaccard_overlap"] for row in fastchem_rows]) or 0.0
    fastchem_family_mean = _mean([row["family_overlap"] for row in fastchem_rows]) or 0.0

    if support_moves_help >= 0.4:
        decision_messages = ["support transition is the main remaining bottleneck"]
        next_move = "develop support handling"
    elif semismooth_help >= 0.4:
        decision_messages = ["inactive-complementarity handling is the next development target"]
        next_move = "develop support handling"
    else:
        decision_messages = ["support is probably not the main remaining issue; inspect Newton-model fidelity next"]
        next_move = "inspect Newton-model fidelity next"
    if fastchem_rows:
        decision_messages.append(
            "FastChem support overlap is now measured directly from local FastChem condensate output."
        )

    payload = {
        "timestamp_utc": current_timestamp_utc(),
        "platform": args.platform,
        "layers": list(args.layers),
        "epsilons": list(args.epsilons),
        "startup_policies": [policy["name"] for policy in startup_policies],
        "support_mechanisms": list(SUPPORT_MECHANISMS),
        "summary_rows": summary_rows,
        "cases": cases,
        "decision": {
            "messages": decision_messages,
            "next_move": next_move,
            "support_move_improvement_fraction": support_moves_help,
            "semismooth_outperformance_fraction": semismooth_help,
            "mean_fastchem_jaccard_overlap": fastchem_jaccard_mean,
            "mean_fastchem_family_overlap": fastchem_family_mean,
        },
    }
    traces_payload = {
        "timestamp_utc": payload["timestamp_utc"],
        "cases": cases,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.traces_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_python(payload), indent=2))
    args.traces_output.write_text(json.dumps(to_python(traces_payload), indent=2))

    print(
        f"{'startup':>28} {'mechanism':>34} {'mean max inactive':>18} {'mean merit':>14} {'beat current':>14} {'fc jaccard':>12}"
    )
    for row in summary_rows:
        print(
            f"{row['startup_policy_name']:>28} "
            f"{row['mechanism_name']:>34} "
            f"{row['mean_max_positive_inactive_driving']:18.3e} "
            f"{row['mean_scalar_merit']:14.3e} "
            f"{row['fraction_beats_current_scalar_merit']:14.3f} "
            f"{row['mean_fastchem_jaccard_overlap']:12.3f}"
        )
    print()
    print(f"decision: {decision_messages[0]}")
    print(f"next_move: {next_move}")
    print(f"summary_json: {args.output}")
    print(f"traces_json: {args.traces_output}")


if __name__ == "__main__":
    main()
