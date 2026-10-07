"""Tests for native condensate residual bundle validation."""

from __future__ import annotations

import pytest

from exogibbs.condensates.native_bundle import (
    build_native_condensate_residual_bundle,
)


def _valid_arguments() -> dict[str, object]:
    return {
        "ln_nk": [0.0],
        "ln_mk": [0.0],
        "ln_ntot": 0.0,
        "formula_matrix": [[1.0]],
        "formula_matrix_cond": [[1.0]],
        "element_inventory_target": [1.0],
        "gk": [0.0],
        "hvector_cond": [0.0],
        "eta": [0.0],
        "epsilon_or_nu": 0.0,
        "element_order": ("A",),
        "gas_species_order": ("A",),
        "condensate_species_order": ("A(s)",),
        "temperature": 1000.0,
        "pressure": 1.0,
        "active_amount_floor": 0.0,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("pressure", float("nan")),
        ("pressure", float("inf")),
        ("active_amount_floor", float("nan")),
        ("active_amount_floor", float("inf")),
        ("ln_ntot", float("nan")),
        ("ln_ntot", float("inf")),
        ("epsilon_or_nu", float("nan")),
        ("epsilon_or_nu", float("inf")),
    ),
)
def test_rejects_nonfinite_scalar_inputs(name: str, value: float) -> None:
    arguments = _valid_arguments()
    arguments[name] = value

    with pytest.raises(ValueError, match=name):
        build_native_condensate_residual_bundle(**arguments)


@pytest.mark.parametrize("name", ("ln_nk", "ln_mk"))
def test_rejects_log_amounts_that_overflow(name: str) -> None:
    arguments = _valid_arguments()
    arguments[name] = [1000.0]

    with pytest.raises(ValueError, match=name):
        build_native_condensate_residual_bundle(**arguments)


def test_rejects_total_log_amount_that_overflows() -> None:
    arguments = _valid_arguments()
    arguments["ln_ntot"] = 1000.0

    with pytest.raises(ValueError, match="ln_ntot"):
        build_native_condensate_residual_bundle(**arguments)


@pytest.mark.parametrize(
    ("matrix_name", "log_amount_name", "message"),
    (
        ("formula_matrix", "ln_nk", "gas inventory"),
        ("formula_matrix_cond", "ln_mk", "condensate inventory"),
    ),
)
def test_rejects_nonfinite_phase_inventory(
    matrix_name: str,
    log_amount_name: str,
    message: str,
) -> None:
    arguments = _valid_arguments()
    arguments[matrix_name] = [[1.0e308]]
    arguments[log_amount_name] = [2.0]

    with pytest.raises(ValueError, match=message):
        build_native_condensate_residual_bundle(**arguments)


def test_rejects_nonfinite_total_inventory() -> None:
    arguments = _valid_arguments()
    arguments["formula_matrix"] = [[1.0e308]]
    arguments["formula_matrix_cond"] = [[1.0e308]]

    with pytest.raises(ValueError, match="total inventory"):
        build_native_condensate_residual_bundle(**arguments)
