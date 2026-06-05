"""Curated API replay tests for the condensate equilibrium HEAD route."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from jax import config
import jax.numpy as jnp
from exogibbs.api.condensate_equilibrium import (
    CondensateEquilibriumOptions,
    condensate_equilibrium,
)
from exogibbs.condensates.head_route_standard_gate import (
    CONVERGED,
    CONVERGED_WITH_CAVEAT,
)
from exogibbs.presets.fastchem_cond import condensate_chemical_setup

config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
HEAD_ROUTE_TABLE = ROOT / "results" / "fastchem4_milestone4381_uniform_post_solver_residual_table.json"
PAYLOAD_READINESS = ROOT / "results" / "fastchem4_milestone4385_tier1_callsite_payload_readiness.json"
M1492_TRACE = ROOT / "results" / "fastchem4_milestone1492_iterative_driver_frontier_expansion_trace.json"
T500_REFRESH = ROOT / "results" / "fastchem4_milestone4379_t500_refresh_policy_live_payload_validation.json"
STATIC_FORMULA_AUDIT = ROOT / "results" / "fastchem4_milestone002_formula_matrix_audit.json"


def _load_head_route_rows() -> list[dict]:
    payload = json.loads(HEAD_ROUTE_TABLE.read_text(encoding="utf-8"))
    rows = payload["rows"]
    assert len(rows) == 14
    return rows


def _rows_by_case_id(path: Path) -> dict[str, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, list[dict]] = {}

    def visit(node) -> None:
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


def _remap_static_payload_to_setup(payload: dict, setup) -> dict | None:
    static_species_order = tuple(
        json.loads(STATIC_FORMULA_AUDIT.read_text(encoding="utf-8"))["Ac_static"]["species_order"]
    )
    setup_index = {name: index for index, name in enumerate(setup.condensate_species)}
    remapped_indices = []
    for static_index in payload["support_indices"]:
        species = static_species_order[int(static_index)]
        if species not in setup_index:
            return None
        remapped_indices.append(setup_index[species])
    return {
        **payload,
        "support_indices": remapped_indices,
        "payload_policy": f"{payload.get('payload_policy', 'unknown')}_species_remapped",
    }


def _payload_by_case_id(setup) -> dict[str, dict]:
    payloads: dict[str, dict] = {}
    for row in json.loads(PAYLOAD_READINESS.read_text(encoding="utf-8"))["rows"]:
        if row.get("payload_reconstructable"):
            payload = _remap_static_payload_to_setup(row["payload"], setup)
            if payload is not None:
                payloads.setdefault(str(row["case_id"]), payload)
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


def _route_evidence_by_case_id() -> dict[str, list[dict]]:
    evidence: dict[str, list[dict]] = {}
    for row in json.loads(T500_REFRESH.read_text(encoding="utf-8"))["rows"]:
        evidence.setdefault(str(row["case_id"]), []).append(
            {
                "primary_summary": row["route_selection_report"]["primary_summary"],
                "refresh_policy_summary": row["refresh_policy_report"],
            }
        )
    return evidence


def _options_for_row(row: dict, t500_evidence: dict[str, list[dict]]) -> CondensateEquilibriumOptions:
    route = str(row["selected_route"])
    kwargs = {
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


def _temperature_pressure(case_id: str) -> tuple[float, float]:
    match = re.search(r"__T([0-9]+)_P([0-9p]+)", case_id)
    if match is None:
        raise ValueError(f"Cannot parse temperature and pressure from case_id={case_id!r}.")
    temperature = float(match.group(1))
    pressure = float(match.group(2).replace("p", "."))
    return temperature, pressure


def _element_budget_for_row(setup, row: dict) -> jnp.ndarray:
    budget = jnp.asarray(setup.gas_setup.element_vector_reference, dtype=jnp.float64)
    element_index = {name: index for index, name in enumerate(setup.elements)}
    if row["family"] in {"carbon_rich_graphite_window", "carbon_rich_CaS_MgS_AlN_window"}:
        budget = budget.at[element_index["C"]].set(2.0 * budget[element_index["O"]])
    return budget


def test_condensate_equilibrium_api_replays_defined_head_route_evidence_for_14_curated_rows() -> None:
    setup = condensate_chemical_setup(silent=True)
    rows = _load_head_route_rows()
    payloads = _payload_by_case_id(setup)
    t500_evidence = _route_evidence_by_case_id()

    results = []
    for row in rows:
        temperature, pressure = _temperature_pressure(row["case_id"])
        payload = payloads.get(row["case_id"])
        result = condensate_equilibrium(
            setup,
            temperature,
            pressure,
            _element_budget_for_row(setup, row),
            support_indices=None if payload is None else payload["support_indices"],
            support_amounts_init=None if payload is None else payload["support_amounts_init"],
            options=_options_for_row(row, t500_evidence),
        )
        results.append(result)

    assert len(results) == 14
    assert all(result.status in {CONVERGED, CONVERGED_WITH_CAVEAT} for result in results)
    assert all(result.converged for result in results)
    assert all(result.condensate_support_indices.ndim == 1 for result in results)
    assert all(result.condensate_amounts.shape == (len(setup.condensate_species),) for result in results)
    assert all(jnp.all(jnp.isfinite(result.gas_ln_n)) for result in results)
    assert all(jnp.all(jnp.isfinite(result.condensate_amounts)) for result in results)
    assert sum(
        result.diagnostics is not None and bool(result.diagnostics["solver_success"])
        for result in results
    ) == 14
    assert all(
        result.diagnostics is not None
        and result.diagnostics["acceptance_tier"] != "runtime_solver_failed"
        for result in results
    )
    assert all(
        result.diagnostics is not None
        and result.diagnostics["head_route_lifecycle"]["route_result"]["standard_path_status"]
        in {CONVERGED, CONVERGED_WITH_CAVEAT}
        for result in results
    )
    assert all(
        result.diagnostics is not None
        and result.diagnostics["head_route_lifecycle"][
            "fastchem4_trace_public_runtime_constructor_inputs_used"
        ]
        is False
        for result in results
    )
    assert all(
        result.diagnostics is not None
        and result.diagnostics["support_selection"]["fastchem4_trace_values_used"] is False
        and result.diagnostics["support_selection"][
            "fastchem4_public_values_used_as_constructor_inputs"
        ]
        is False
        for result in results
    )
