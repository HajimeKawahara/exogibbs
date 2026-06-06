"""Tests for native condensate support-boundary construction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from exogibbs.condensates.support_boundary import (
    build_condensate_support_boundary,
)


def _fixture_kwargs():
    return {
        "explicit_opt_in": True,
        "ln_nk": [math.log(0.8), math.log(0.2)],
        "support_indices": [1],
        "support_amounts": [1.0e-6],
        "formula_matrix": [[1.0, 0.0], [0.0, 1.0]],
        "formula_matrix_cond": [[2.0, 1.0], [0.0, 3.0]],
        "element_inventory_target": [0.800001, 0.200003],
        "field_provenance": {
            "ln_nk": "exogibbs_native",
            "support_indices": "exogibbs_native",
            "support_amounts": "exogibbs_native",
        },
    }


def test_builds_support_boundary_and_budget_residual() -> None:
    inputs = _fixture_kwargs()
    boundary = build_condensate_support_boundary(**inputs)

    expected_full = np.asarray([0.0, 1.0e-6])
    expected_gas = np.asarray(inputs["formula_matrix"]) @ np.exp(np.asarray(inputs["ln_nk"]))
    expected_cond = np.asarray(inputs["formula_matrix_cond"]) @ expected_full
    expected_budget = expected_gas + expected_cond - np.asarray(inputs["element_inventory_target"])

    assert boundary.boundary_schema == "exogibbs_condensate_support_boundary_v1"
    assert boundary.production_behavior_change is False
    assert boundary.production_return_signature_change is False
    assert boundary.preset_default_wiring_change is False
    assert boundary.fastchem4_trace_public_runtime_constructor_inputs_used is False
    assert boundary.support_indices == (1,)
    assert boundary.full_condensate_amounts == pytest.approx(tuple(expected_full))
    assert boundary.ln_mk == pytest.approx((math.log(1.0e-6),))
    assert boundary.formula_matrix_cond_active == ((1.0,), (3.0,))
    assert boundary.gas_element_inventory == pytest.approx(tuple(expected_gas))
    assert boundary.condensate_element_inventory == pytest.approx(tuple(expected_cond))
    assert boundary.budget_residual == pytest.approx(tuple(expected_budget))
    assert boundary.budget_residual_l2 == pytest.approx(float(np.linalg.norm(expected_budget)))
    assert boundary.as_dict()["support_amounts"] == pytest.approx((1.0e-6,))


def test_support_boundary_allows_empty_support() -> None:
    inputs = _fixture_kwargs()
    inputs["support_indices"] = []
    inputs["support_amounts"] = []

    boundary = build_condensate_support_boundary(**inputs)

    assert boundary.support_indices == ()
    assert boundary.support_amounts == ()
    assert boundary.ln_mk == ()
    assert boundary.formula_matrix_cond_active == ((), ())
    assert boundary.full_condensate_amounts == pytest.approx((0.0, 0.0))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("explicit_opt_in", False, "explicit_opt_in"),
        ("support_indices", [1, 1], "unique"),
        ("support_indices", [2], "out-of-range"),
        ("support_amounts", [0.0], "positive"),
        ("formula_matrix", [[1.0], [0.0]], "column count"),
        ("formula_matrix_cond", [[1.0, 0.0]], "row count"),
        ("element_inventory_target", [1.0], "element_inventory_target length"),
        ("amount_floor", 0.0, "amount_floor"),
    ],
)
def test_rejects_invalid_boundary_inputs(field, value, match) -> None:
    inputs = _fixture_kwargs()
    if field == "support_indices" and value == [1, 1]:
        inputs["support_amounts"] = [1.0e-6, 2.0e-6]
    inputs[field] = value

    with pytest.raises(ValueError, match=match):
        build_condensate_support_boundary(**inputs)


@pytest.mark.parametrize(
    "provenance",
    [
        "fastchem4_trace",
        "fastchem4_public",
        "fastchem4_runtime_value",
        "branch_replay",
        "reference_fit_scalar",
        "unknown_reference",
    ],
)
def test_rejects_forbidden_boundary_provenance(provenance) -> None:
    inputs = _fixture_kwargs()
    inputs["field_provenance"] = {"support_amounts": provenance}

    with pytest.raises(ValueError, match="forbidden"):
        build_condensate_support_boundary(**inputs)
