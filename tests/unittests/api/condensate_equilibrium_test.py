"""Focused tests for the production v2 condensate API."""

from __future__ import annotations

from dataclasses import replace
import importlib
import math
import sys

import jax.numpy as jnp
import numpy as np
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
from exogibbs.equilibrium.condensate.setup import (
    condensate_temperature_validity_upper,
)
from exogibbs.equilibrium.condensate.acceptance import (
    full_condensate_element_budget_residual_report,
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
        diagnostics={
            "fixed_support_v2": {
                "zero_barrier_active_support_polish": {"accepted": True}
            }
        },
    )


def test_build_condensate_chemical_setup_validates_element_order() -> None:
    gas, condensate, setup = _setup_pair()

    validate_condensate_chemical_setup(setup)

    assert setup.gas_setup is gas
    assert setup.condensate_setup is condensate
    assert setup.elements == ("H", "O")
    assert setup.gas_species == ("H", "O")
    assert setup.condensate_species == ("H2O_s",)


def test_one_layer_solver_forwards_initializer(monkeypatch) -> None:
    _gas, _condensate, setup = _setup_pair()
    initializer = object()
    captured = {}

    def fake_profile(**kwargs):
        captured.update(kwargs)
        return type("Profile", (), {"layers": ("layer",)})()

    monkeypatch.setattr(condmod, "_run_head_v2_profile", fake_profile)

    result = condmod.condensate_equilibrium(
        setup,
        T=1000.0,
        P=1.0,
        b=jnp.asarray([1.0, 1.0]),
        initializer=initializer,
    )

    assert result == "layer"
    assert captured["initializer"] is initializer


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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "gas_species",
            ("O", "H"),
            "gas_species must match",
        ),
        (
            "formula_matrix",
            jnp.asarray([[0.0, 1.0], [1.0, 0.0]]),
            "formula_matrix must match",
        ),
        (
            "condensate_species",
            ("other",),
            "condensate_species must match",
        ),
        (
            "formula_matrix_cond",
            jnp.asarray([[1.0], [2.0]]),
            "formula_matrix_cond must match",
        ),
    ),
)
def test_condensate_setup_rejects_inconsistent_duplicated_fields(
    field,
    value,
    message,
) -> None:
    _gas, _condensate, setup = _setup_pair()

    with pytest.raises(ValueError, match=message):
        validate_condensate_chemical_setup(
            replace(setup, **{field: value})
        )


def test_condensate_temperature_validity_is_typed_and_metadata_optional() -> None:
    gas, condensate, setup = _setup_pair()

    assert condensate.metadata == {"source": "unit-test-condensate"}
    assert condensate_temperature_validity_upper(setup) is None

    typed_condensate = ChemicalSetup(
        formula_matrix=condensate.formula_matrix,
        hvector_func=condensate.hvector_func,
        elements=condensate.elements,
        species=condensate.species,
        metadata=None,
        temperature_validity_upper=(500.0,),
    )
    typed_setup = build_condensate_chemical_setup(
        gas_setup=gas,
        condensate_setup=typed_condensate,
    )

    assert condensate_temperature_validity_upper(typed_setup) == (500.0,)


def test_condensate_setup_validates_typed_temperature_validity() -> None:
    gas, condensate, _setup = _setup_pair()
    invalid_condensate = replace(
        condensate,
        temperature_validity_upper=(500.0, 600.0),
    )

    with pytest.raises(ValueError, match="one value per condensate"):
        build_condensate_chemical_setup(
            gas_setup=gas,
            condensate_setup=invalid_condensate,
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


def test_disabled_full_budget_gate_reports_disabled_state() -> None:
    _gas, _condensate, setup = _setup_pair()

    result = build_condensate_equilibrium_result_from_solver_payload(
        setup=setup,
        gas_ln_n=(0.0, 0.0),
        support_indices=(0,),
        support_amounts=(1.0,),
        selected_route=CONDENSATE_HEAD_V2_ROUTE_NAME,
        solver_success=True,
        route=HEAD_ROUTE_V2,
        head_route_version=CONDENSATE_HEAD_V2_ROUTE_VERSION,
        head_route_name=CONDENSATE_HEAD_V2_ROUTE_NAME,
        element_inventory_target=jnp.asarray([1.0, 1.0]),
        enable_full_condensate_budget_residual_gate=False,
        diagnostics={
            "fixed_support_v2": {
                "zero_barrier_active_support_polish": {"accepted": True}
            }
        },
    )

    gate = result.diagnostics["full_condensate_budget_residual_gate"]
    assert result.converged
    assert gate["enabled"] is False
    assert gate["accepted"] is False


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
    assert gate["gate_schema"] == (
        "exogibbs_full_condensate_element_budget_residual_gate_v2"
    )
    assert gate["relative_floor"] == pytest.approx(1.0e-6)
    assert gate["max_abs_relative_residual"] == pytest.approx(4.0e-4)


def test_full_budget_gate_rejects_overflowed_amount_scale() -> None:
    _gas, _condensate, setup = _setup_pair()
    report = full_condensate_element_budget_residual_report(
        setup=setup,
        gas_n=np.zeros(2, dtype=np.float64),
        condensate_amounts=np.zeros(1, dtype=np.float64),
        element_inventory_target=np.asarray([1.0e308, 1.0e308]),
        relative_tolerance=1.0e-6,
    )

    assert not report["accepted"]
    assert not report["amount_gauge_scale_finite"]
    assert report["max_abs_relative_residual"] == pytest.approx(1.0)


def test_result_mole_fractions_are_stable_in_a_tiny_amount_gauge() -> None:
    _gas, _condensate, setup = _setup_pair()
    amount_scale = 1.0e-305
    result = build_condensate_equilibrium_result_from_solver_payload(
        setup=setup,
        gas_ln_n=(
            math.log(0.5 * amount_scale),
            math.log(0.5 * amount_scale),
        ),
        support_indices=(),
        support_amounts=(),
        selected_route=CONDENSATE_HEAD_V2_ROUTE_NAME,
        solver_success=True,
        element_inventory_target=jnp.asarray(
            [0.5 * amount_scale, 0.5 * amount_scale]
        ),
    )

    assert result.converged
    assert result.gas_x.tolist() == pytest.approx([0.5, 0.5])
    gate = result.diagnostics["full_condensate_budget_residual_gate"]
    assert gate["absolute_floor"] / amount_scale == pytest.approx(1.0e-6)


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


def test_gas_budget_transform_is_disabled_and_raw_state_is_rejected() -> None:
    _gas, _condensate, setup = _setup_pair()

    result = build_condensate_equilibrium_result_from_solver_payload(
        setup=setup,
        gas_ln_n=(math.log(2.0e-12), 0.0),
        support_indices=(),
        support_amounts=(),
        selected_route=CONDENSATE_HEAD_V2_ROUTE_NAME,
        solver_success=True,
        route=HEAD_ROUTE_V2,
        head_route_version=CONDENSATE_HEAD_V2_ROUTE_VERSION,
        head_route_name=CONDENSATE_HEAD_V2_ROUTE_NAME,
        element_inventory_target=jnp.asarray([1.0e-12, 1.0]),
        full_condensate_budget_relative_floor=1.0e-12,
    )

    polish = result.diagnostics[
        "full_condensate_budget_gas_log_amount_polish"
    ]
    assert not result.converged
    assert not polish["triggered"]
    assert polish["skip_reason"] == (
        "gas_only_budget_transform_disabled_requires_resolve"
    )
    assert float(result.gas_n[0]) == pytest.approx(2.0e-12)


def test_positive_condensate_requires_zero_barrier_physical_audit() -> None:
    _gas, _condensate, setup = _setup_pair()

    result = build_condensate_equilibrium_result_from_solver_payload(
        setup=setup,
        gas_ln_n=(0.0, 0.0),
        support_indices=(0,),
        support_amounts=(1.0,),
        selected_route=CONDENSATE_HEAD_V2_ROUTE_NAME,
        solver_success=True,
        element_inventory_target=jnp.asarray([3.0, 2.0]),
    )

    assert not result.converged
    assert result.acceptance_tier == "physical_condensate_kkt_audit_failed"


def test_gas_budget_transform_never_publishes_a_stationarity_breaking_state() -> None:
    gas = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0, 2.0]]),
        hvector_func=lambda temperature: jnp.zeros(2),
        elements=("A",),
        species=("A", "A2"),
    )
    condensate = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0]]),
        hvector_func=lambda temperature: jnp.zeros(1),
        elements=("A",),
        species=("A_s",),
    )
    setup = build_condensate_chemical_setup(
        gas_setup=gas,
        condensate_setup=condensate,
    )
    atomic_fraction = 0.5 * (math.sqrt(5.0) - 1.0)
    gas_ln_n = jnp.log(
        jnp.asarray([atomic_fraction, atomic_fraction**2])
    )

    result = build_condensate_equilibrium_result_from_solver_payload(
        setup=setup,
        gas_ln_n=gas_ln_n,
        support_indices=(),
        support_amounts=(),
        selected_route="head_v2_gas_only_no_candidate",
        solver_success=True,
        element_inventory_target=jnp.asarray([1.3]),
        full_condensate_budget_relative_tolerance=1.0e-10,
    )

    polish = result.diagnostics[
        "full_condensate_budget_gas_log_amount_polish"
    ]
    assert not result.converged
    assert not polish["triggered"]
    assert not polish["accepted"]
    assert polish["skip_reason"] == (
        "gas_only_budget_transform_disabled_requires_resolve"
    )
    assert result.gas_ln_n.tolist() == pytest.approx(gas_ln_n.tolist())


def test_options_select_only_the_promoted_v2_route() -> None:
    options = CondensateEquilibriumOptions()

    assert options.route == HEAD_ROUTE_V2
    assert options.fixed_support_v2_preset == "validated_2026_07"
    with pytest.raises(ValueError, match="Unsupported condensate route"):
        condmod._validate_options(
            CondensateEquilibriumOptions(route="head_v1")
        )


def test_rainout_option_preserves_existing_positional_field_order() -> None:
    options = CondensateEquilibriumOptions(
        "head_v2",
        "validated_2026_07",
        "auto",
        True,
        False,
        2.0e-3,
        2.0e-6,
    )

    assert options.return_diagnostics is True
    assert options.enable_full_condensate_budget_residual_gate is False
    assert options.full_condensate_budget_relative_tolerance == 2.0e-3
    assert options.full_condensate_budget_relative_floor == 2.0e-6
    assert options.rainout is False


def test_api_init_does_not_import_condensate_equilibrium_by_default() -> None:
    sys.modules.pop("exogibbs.api", None)
    sys.modules.pop("exogibbs.api.condensate_equilibrium", None)

    module = importlib.import_module("exogibbs.api")

    assert "exogibbs.api.condensate_equilibrium" not in sys.modules
    resolved = getattr(module, "CondensateEquilibriumOptions")
    assert resolved.__name__ == "CondensateEquilibriumOptions"
    assert "exogibbs.api.condensate_equilibrium" in sys.modules
