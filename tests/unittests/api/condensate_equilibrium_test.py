"""Focused tests for the production v2 condensate API."""

from __future__ import annotations

import importlib
import math
import sys

import jax.numpy as jnp
import pytest

import exogibbs.api.condensate_equilibrium as condmod
from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.api.condensate_equilibrium import (
    CONDENSATE_HEAD_V2_ROUTE_NAME,
    CONDENSATE_HEAD_V2_ROUTE_VERSION,
    HEAD_ROUTE_V2,
    CondensateChemicalSetup,
    CondensateEquilibriumOptions,
    _polish_gas_log_amounts_for_full_condensate_budget_gate,
    build_condensate_chemical_setup,
    build_condensate_equilibrium_result_from_solver_payload,
    validate_condensate_chemical_setup,
)


def _setup_pair() -> tuple[
    ChemicalSetup,
    ChemicalSetup,
    CondensateChemicalSetup,
]:
    gas = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        hvector_func=lambda temperature: jnp.asarray([0.0, 0.0]),
        elements=("H", "O"),
        species=("H", "O"),
        metadata={"source": "unit-test-gas"},
    )
    condensate = ChemicalSetup(
        formula_matrix=jnp.asarray([[2.0], [1.0]]),
        hvector_func=lambda temperature: jnp.asarray([0.0]),
        elements=("H", "O"),
        species=("H2O_s",),
        metadata={"source": "unit-test-condensate"},
    )
    return (
        gas,
        condensate,
        build_condensate_chemical_setup(
            gas_setup=gas,
            condensate_setup=condensate,
        ),
    )


def _build_v2_result(
    setup: CondensateChemicalSetup,
    *,
    gas_ln_n: tuple[float, float],
    support_amount: float,
    element_inventory_target,
):
    return build_condensate_equilibrium_result_from_solver_payload(
        setup=setup,
        gas_ln_n=gas_ln_n,
        support_indices=(0,),
        support_amounts=(support_amount,),
        selected_route=CONDENSATE_HEAD_V2_ROUTE_NAME,
        solver_success=True,
        route=HEAD_ROUTE_V2,
        head_route_version=CONDENSATE_HEAD_V2_ROUTE_VERSION,
        head_route_name=CONDENSATE_HEAD_V2_ROUTE_NAME,
        element_inventory_target=element_inventory_target,
    )


def test_build_condensate_chemical_setup_validates_element_order() -> None:
    gas, condensate, setup = _setup_pair()

    validate_condensate_chemical_setup(setup)

    assert setup.gas_setup is gas
    assert setup.condensate_setup is condensate
    assert setup.elements == ("H", "O")
    assert setup.gas_species == ("H", "O")
    assert setup.condensate_species == ("H2O_s",)


def test_condensate_chemical_setup_rejects_element_order_mismatch() -> None:
    gas, condensate, _setup = _setup_pair()
    invalid = ChemicalSetup(
        formula_matrix=condensate.formula_matrix,
        hvector_func=condensate.hvector_func,
        elements=("O", "H"),
        species=condensate.species,
    )

    with pytest.raises(ValueError, match="element orders"):
        build_condensate_chemical_setup(
            gas_setup=gas,
            condensate_setup=invalid,
        )


def test_v2_result_uses_full_condensate_vector_and_metadata() -> None:
    _gas, _condensate, setup = _setup_pair()

    result = _build_v2_result(
        setup,
        gas_ln_n=(0.0, 0.0),
        support_amount=1.0,
        element_inventory_target=jnp.asarray([3.0, 2.0]),
    )

    assert result.status == "converged"
    assert result.converged
    assert result.selected_route == CONDENSATE_HEAD_V2_ROUTE_NAME
    assert result.head_route_version == "v2.0"
    assert result.head_route_name == CONDENSATE_HEAD_V2_ROUTE_NAME
    assert result.condensate_support_names == ("H2O_s",)
    assert result.condensate_amounts.tolist() == pytest.approx([1.0])
    assert result.diagnostics["route"] == HEAD_ROUTE_V2
    assert result.diagnostics[
        "full_condensate_budget_residual_gate"
    ]["accepted"]


def test_v2_result_rejects_failed_full_budget_gate() -> None:
    _gas, _condensate, setup = _setup_pair()

    result = _build_v2_result(
        setup,
        gas_ln_n=(0.0, 0.0),
        support_amount=1.0,
        element_inventory_target=jnp.asarray([1.0, 1.0]),
    )

    assert result.status == "not_converged"
    assert not result.converged
    assert result.acceptance_tier == (
        "full_condensate_element_budget_residual_failed"
    )
    gate = result.diagnostics["full_condensate_budget_residual_gate"]
    assert not gate["accepted"]
    assert gate["max_abs_relative_residual_element"] == "H"


def test_full_budget_gate_uses_trace_relative_floor() -> None:
    _gas, _condensate, setup = _setup_pair()
    result = build_condensate_equilibrium_result_from_solver_payload(
        setup=setup,
        gas_ln_n=(math.log(1.04e-8), 0.0),
        support_indices=(),
        support_amounts=(),
        selected_route=CONDENSATE_HEAD_V2_ROUTE_NAME,
        solver_success=True,
        route=HEAD_ROUTE_V2,
        head_route_version=CONDENSATE_HEAD_V2_ROUTE_VERSION,
        head_route_name=CONDENSATE_HEAD_V2_ROUTE_NAME,
        element_inventory_target=jnp.asarray([1.0e-8, 1.0]),
    )

    gate = result.diagnostics["full_condensate_budget_residual_gate"]
    assert result.converged
    assert gate["relative_floor"] == pytest.approx(1.0e-6)
    assert gate["max_abs_relative_residual"] == pytest.approx(4.0e-4)


def test_gas_log_amount_polish_repairs_trace_budget_residual() -> None:
    gas = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        hvector_func=lambda temperature: jnp.asarray([0.0, 0.0]),
        elements=("H", "Ge"),
        species=("H", "Ge"),
    )
    condensate = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0], [0.0]]),
        hvector_func=lambda temperature: jnp.asarray([0.0]),
        elements=("H", "Ge"),
        species=("H_s",),
    )
    setup = build_condensate_chemical_setup(
        gas_setup=gas,
        condensate_setup=condensate,
    )

    polished, report = _polish_gas_log_amounts_for_full_condensate_budget_gate(
        setup=setup,
        gas_ln_n=jnp.log(jnp.asarray([0.5, 1.0e-6])),
        condensate_amounts=jnp.asarray([0.5]),
        element_inventory_target=jnp.asarray([1.0, 1.0e-9]),
        relative_tolerance=1.0e-3,
    )

    assert report is not None
    assert report["accepted"]
    assert float(jnp.exp(polished)[1]) < 2.0e-9


def test_options_select_only_the_promoted_v2_route() -> None:
    options = CondensateEquilibriumOptions()

    assert options.route == HEAD_ROUTE_V2
    assert options.fixed_support_v2_preset == "validated_2026_07"
    with pytest.raises(ValueError, match="Unsupported condensate route"):
        condmod._validate_options(
            CondensateEquilibriumOptions(route="head_v1")
        )


def test_api_init_does_not_import_condensate_equilibrium_by_default() -> None:
    sys.modules.pop("exogibbs.api", None)
    sys.modules.pop("exogibbs.api.condensate_equilibrium", None)

    module = importlib.import_module("exogibbs.api")

    assert "exogibbs.api.condensate_equilibrium" not in sys.modules
    resolved = getattr(module, "CondensateEquilibriumOptions")
    assert resolved.__name__ == "CondensateEquilibriumOptions"
    assert "exogibbs.api.condensate_equilibrium" in sys.modules
