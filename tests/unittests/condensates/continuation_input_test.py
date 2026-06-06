"""Tests for native condensate continuation input construction."""

from __future__ import annotations

import math

import pytest

from exogibbs.condensates.continuation_input import (
    build_condensate_continuation_input,
)


def _fixture_kwargs():
    return {
        "explicit_opt_in": True,
        "ln_nk": [math.log(0.8), math.log(0.2)],
        "ln_mk": [math.log(1.0e-8)],
        "element_potential": [math.log(0.8) + 0.1, math.log(0.2) - 0.05],
        "support_indices": [3],
        "formula_matrix": [[1.0, 0.0], [0.0, 1.0]],
        "formula_matrix_cond_active": [[1.0], [1.0]],
        "element_inventory_target": [0.80000001, 0.20000001],
        "gas_stationarity_source": [0.1, -0.05],
        "condensate_standard_source": [0.2],
        "epsilon": math.log(1.0e-12),
        "field_provenance": {
            "ln_nk": "exogibbs_native",
            "ln_mk": "exogibbs_native",
            "element_potential": "exogibbs_native_derived",
        },
    }


def test_builds_continuation_input_and_infers_rho_from_epsilon() -> None:
    inputs = _fixture_kwargs()
    report = build_condensate_continuation_input(**inputs)

    assert report.input_schema == "exogibbs_condensate_continuation_input_v1"
    assert report.production_behavior_change is False
    assert report.production_return_signature_change is False
    assert report.preset_default_wiring_change is False
    assert report.fastchem4_trace_public_runtime_constructor_inputs_used is False
    assert report.support_indices == (3,)
    assert report.inferred_rho_from_epsilon is True
    assert report.gas_lambda_gauge_residual_l2 == pytest.approx(0.0, abs=1.0e-14)
    assert report.gas_lambda_gauge_residual_max_abs == pytest.approx(0.0, abs=1.0e-14)
    assert report.state.rho == pytest.approx((math.log(1.0e-12) - math.log(1.0e-8),))
    assert report.as_dict()["state"]["state_schema"] == "exogibbs_pdipm_rgie_condensate_state_v1"


def test_uses_eta_when_provided() -> None:
    inputs = _fixture_kwargs()
    inputs.pop("epsilon")
    inputs["eta"] = [1.0e-4]

    report = build_condensate_continuation_input(**inputs)

    assert report.inferred_rho_from_epsilon is False
    assert report.state.rho == pytest.approx((math.log(1.0e-4),))
    assert report.state.eta == pytest.approx((1.0e-4,))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("explicit_opt_in", False, "explicit_opt_in"),
        ("support_indices", [3, 3], "unique"),
        ("gas_stationarity_source", [0.1], "gas_stationarity_source length"),
        ("formula_matrix_cond_active", [[1.0, 0.0], [1.0, 0.0]], "column count"),
        ("rho", [0.1, 0.2], "rho length"),
    ],
)
def test_rejects_invalid_inputs(field, value, match) -> None:
    inputs = _fixture_kwargs()
    if field == "rho":
        inputs.pop("epsilon")
    if field == "support_indices":
        inputs["ln_mk"] = [math.log(1.0e-8), math.log(2.0e-8)]
        inputs["formula_matrix_cond_active"] = [[1.0, 0.0], [1.0, 1.0]]
        inputs["condensate_standard_source"] = [0.2, 0.3]
    inputs[field] = value

    with pytest.raises(ValueError, match=match):
        build_condensate_continuation_input(**inputs)


@pytest.mark.parametrize(
    "provenance",
    [
        "fastchem4_trace",
        "fastchem4_trace_snapshot",
        "fastchem4_public",
        "fastchem4_runtime_value",
        "branch_replay",
        "reference_fit_scalar",
        "unknown_reference",
    ],
)
def test_rejects_forbidden_provenance(provenance) -> None:
    inputs = _fixture_kwargs()
    inputs["field_provenance"] = {"ln_nk": provenance}

    with pytest.raises(ValueError, match="forbidden"):
        build_condensate_continuation_input(**inputs)


def test_requires_dual_carrier_source() -> None:
    inputs = _fixture_kwargs()
    inputs.pop("epsilon")

    with pytest.raises(ValueError, match="rho, eta, or epsilon"):
        build_condensate_continuation_input(**inputs)
