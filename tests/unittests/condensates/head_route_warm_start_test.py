"""Tests for condensate HEAD route warm-start candidate generation."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from exogibbs.api.chemistry import ThermoState
from exogibbs.condensates.head_route_warm_start import (
    build_condensate_head_route_warm_start_report,
)


def _state() -> ThermoState:
    return ThermoState(
        temperature=300.0,
        ln_normalized_pressure=jnp.asarray(0.0),
        element_vector=jnp.asarray([1.0, 1.0]),
    )


def _base_kwargs():
    return {
        "explicit_opt_in": True,
        "state": _state(),
        "formula_matrix": [[1.0, 0.0], [0.0, 1.0]],
        "formula_matrix_cond": [[0.1], [0.1]],
        "hvector_func": lambda _temperature: jnp.asarray([0.0, 0.0]),
        "support_indices": [0],
        "support_amounts_init": [1.0e-3],
        "field_provenance": {
            "formula_matrix": "exogibbs_native",
            "formula_matrix_cond": "exogibbs_native",
            "element_budget": "exogibbs_native",
            "ln_mk": "exogibbs_native_derived",
            "hvector_func": "exogibbs_native",
        },
    }


def test_builds_baseline_and_depleted_refresh_candidates() -> None:
    report = build_condensate_head_route_warm_start_report(**_base_kwargs())

    assert report.report_schema == "exogibbs_condensate_head_route_warm_start_report_v1"
    assert report.candidate_count == 2
    assert report.production_behavior_change is False
    assert report.fastchem4_trace_public_runtime_constructor_inputs_used is False
    assert report.candidates[0].candidate_kind == "baseline"
    assert report.candidates[0].initial_log_state_override is None
    assert report.candidates[1].candidate_kind == "depleted_gas_refresh"
    assert report.candidates[1].initial_log_state_override is not None
    assert report.candidates[1].refresh_report is not None
    assert report.candidates[1].finite_solver_inputs is True
    assert report.as_dict()["candidates"][1]["initial_log_state_override"]["ln_nk_shape"] == [2]


def test_can_disable_depleted_refresh_candidate() -> None:
    inputs = _base_kwargs()
    inputs["enable_depleted_gas_refresh"] = False

    report = build_condensate_head_route_warm_start_report(**inputs)

    assert report.candidate_count == 1
    assert report.candidates[0].candidate_kind == "baseline"


def test_rejects_forbidden_provenance() -> None:
    inputs = _base_kwargs()
    inputs["field_provenance"] = {"ln_mk": "fastchem4_trace"}

    with pytest.raises(ValueError, match="forbidden"):
        build_condensate_head_route_warm_start_report(**inputs)
