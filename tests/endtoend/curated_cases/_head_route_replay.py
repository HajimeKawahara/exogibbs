"""Shared helpers for curated condensate HEAD route API replay tests."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from jax import config
import jax.numpy as jnp

from exogibbs.api.condensate_equilibrium import (
    CondensateEquilibriumOptions,
    CondensateEquilibriumResult,
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

ACCEPTED_STATUSES = {CONVERGED, CONVERGED_WITH_CAVEAT}


def load_head_route_rows() -> list[dict[str, Any]]:
    """Load the fixed 14-row HEAD route evidence table."""

    rows = json.loads(HEAD_ROUTE_TABLE.read_text(encoding="utf-8"))["rows"]
    if len(rows) != 14:
        raise AssertionError(f"Expected 14 curated HEAD route rows, got {len(rows)}.")
    return rows


def rows_for_family(family: str) -> list[dict[str, Any]]:
    """Return all HEAD route rows for one curated family."""

    rows = [row for row in load_head_route_rows() if row["family"] == family]
    if not rows:
        raise AssertionError(f"No curated HEAD route rows found for family={family!r}.")
    return rows


def _rows_by_case_id(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, list[dict[str, Any]]] = {}

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
    payload = json.loads(STATIC_FORMULA_AUDIT.read_text(encoding="utf-8"))
    return tuple(str(name) for name in payload["Ac_static"]["species_order"])


def _remap_static_payload_to_setup(payload: dict[str, Any], setup: Any) -> dict[str, Any] | None:
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
    """Build API-indexed explicit support payloads when the setup contains all species."""

    payloads: dict[str, dict[str, Any]] = {}
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


def route_evidence_by_case_id() -> dict[str, list[dict[str, Any]]]:
    """Load saved route-selection evidence for rows without full API payloads."""

    evidence: dict[str, list[dict[str, Any]]] = {}
    for row in json.loads(T500_REFRESH.read_text(encoding="utf-8"))["rows"]:
        evidence.setdefault(str(row["case_id"]), []).append(
            {
                "primary_summary": row["route_selection_report"]["primary_summary"],
                "refresh_policy_summary": row["refresh_policy_report"],
            }
        )
    return evidence


def options_for_row(
    row: dict[str, Any],
    t500_evidence: dict[str, list[dict[str, Any]]],
) -> CondensateEquilibriumOptions:
    """Build API options that replay the saved HEAD route evidence for one row."""

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
    """Parse temperature and pressure from the curated case identifier."""

    match = re.search(r"__T([0-9]+)_P([0-9p]+)", case_id)
    if match is None:
        raise ValueError(f"Cannot parse temperature and pressure from case_id={case_id!r}.")
    return float(match.group(1)), float(match.group(2).replace("p", "."))


def element_budget_for_row(setup: Any, row: dict[str, Any]) -> jnp.ndarray:
    """Build the native element budget used by the saved curated row."""

    budget = jnp.asarray(setup.gas_setup.element_vector_reference, dtype=jnp.float64)
    element_index = {name: index for index, name in enumerate(setup.elements)}
    if row["family"] in {"carbon_rich_graphite_window", "carbon_rich_CaS_MgS_AlN_window"}:
        budget = budget.at[element_index["C"]].set(2.0 * budget[element_index["O"]])
    return budget


def replay_row(
    setup: Any,
    row: dict[str, Any],
    payloads: dict[str, dict[str, Any]],
    t500_evidence: dict[str, list[dict[str, Any]]],
) -> CondensateEquilibriumResult:
    """Replay one saved HEAD route row through the public API."""

    temperature, pressure = temperature_pressure(row["case_id"])
    payload = payloads.get(row["case_id"])
    return condensate_equilibrium(
        setup,
        temperature,
        pressure,
        element_budget_for_row(setup, row),
        support_indices=None if payload is None else payload["support_indices"],
        support_amounts_init=None if payload is None else payload["support_amounts_init"],
        options=options_for_row(row, t500_evidence),
    )


def assert_replay_result_matches_row(result: CondensateEquilibriumResult, row: dict[str, Any], setup: Any) -> None:
    """Assert one API result reproduces the saved HEAD route standard decision."""

    assert result.status in ACCEPTED_STATUSES
    assert result.converged is True
    assert result.selected_route == row["selected_route"]
    assert result.condensate_support_indices.ndim == 1
    assert result.condensate_amounts.shape == (len(setup.condensate_species),)
    assert bool(jnp.all(jnp.isfinite(result.gas_ln_n)))
    assert bool(jnp.all(jnp.isfinite(result.condensate_amounts)))
    assert result.diagnostics is not None
    assert result.diagnostics["solver_success"] is True
    assert result.diagnostics["acceptance_tier"] != "runtime_solver_failed"
    assert result.diagnostics["head_route_lifecycle"]["route_result"]["standard_path_status"] in ACCEPTED_STATUSES
    assert result.diagnostics["head_route_lifecycle"]["fastchem4_trace_public_runtime_constructor_inputs_used"] is False
    assert result.diagnostics["support_selection"]["fastchem4_trace_values_used"] is False
    assert result.diagnostics["support_selection"]["fastchem4_public_values_used_as_constructor_inputs"] is False


def replay_family(family: str) -> list[CondensateEquilibriumResult]:
    """Replay all saved HEAD route rows for one curated family."""

    setup = condensate_chemical_setup(silent=True)
    rows = rows_for_family(family)
    payloads = payload_by_case_id(setup)
    t500_evidence = route_evidence_by_case_id()
    results = [replay_row(setup, row, payloads, t500_evidence) for row in rows]
    for result, row in zip(results, rows):
        assert_replay_result_matches_row(result, row, setup)
    return results


def replay_all_curated_rows() -> list[CondensateEquilibriumResult]:
    """Replay all 14 saved curated HEAD route rows through the public API."""

    setup = condensate_chemical_setup(silent=True)
    rows = load_head_route_rows()
    payloads = payload_by_case_id(setup)
    t500_evidence = route_evidence_by_case_id()
    results = [replay_row(setup, row, payloads, t500_evidence) for row in rows]
    for result, row in zip(results, rows):
        assert_replay_result_matches_row(result, row, setup)
    return results
