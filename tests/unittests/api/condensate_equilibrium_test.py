"""Tests for the condensate equilibrium production-facing API shell."""

from __future__ import annotations

import importlib
import sys

import jax.numpy as jnp
import pytest

from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.api.condensate_equilibrium import (
    CONDENSATE_HEAD_ROUTE_NAME,
    CONDENSATE_HEAD_ROUTE_VERSION,
    CondensateChemicalSetup,
    CondensateEquilibriumOptions,
    _polish_support_amounts_for_full_condensate_budget_gate,
    build_condensate_chemical_setup,
    build_condensate_equilibrium_result_from_solver_payload,
    condensate_equilibrium,
    validate_condensate_chemical_setup,
)
from exogibbs.condensates.head_route_standard_gate import (
    BUDGET_TRADEOFF_STATUS,
    CONVERGED,
    CONVERGED_WITH_CAVEAT,
    HEAD_ROUTE_STANDARD,
    NOT_CONVERGED,
    TIGHT_RESIDUAL_STATUS,
)


def _setup_pair() -> tuple[ChemicalSetup, ChemicalSetup, CondensateChemicalSetup]:
    gas = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        hvector_func=lambda T: jnp.asarray([0.0, 0.0]),
        elements=("H", "O"),
        species=("H", "O"),
        metadata={"source": "unit-test-gas"},
    )
    cond = ChemicalSetup(
        formula_matrix=jnp.asarray([[2.0], [1.0]]),
        hvector_func=lambda T: jnp.asarray([0.0]),
        elements=("H", "O"),
        species=("H2O_s",),
        metadata={"source": "unit-test-condensate"},
    )
    return gas, cond, build_condensate_chemical_setup(gas_setup=gas, condensate_setup=cond)


def _setup_pair_with_condensate_hvalue(hvalue: float) -> CondensateChemicalSetup:
    gas = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        hvector_func=lambda T: jnp.asarray([0.0, 0.0]),
        elements=("H", "O"),
        species=("H", "O"),
        metadata={"source": "unit-test-gas"},
    )
    cond = ChemicalSetup(
        formula_matrix=jnp.asarray([[2.0], [1.0]]),
        hvector_func=lambda T: jnp.asarray([hvalue]),
        elements=("H", "O"),
        species=("H2O_s",),
        metadata={"source": "unit-test-condensate"},
    )
    return build_condensate_chemical_setup(gas_setup=gas, condensate_setup=cond)


def _setup_pair_with_two_condensates() -> CondensateChemicalSetup:
    gas = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        hvector_func=lambda T: jnp.asarray([0.0, 0.0]),
        elements=("H", "O"),
        species=("H", "O"),
        metadata={"source": "unit-test-gas"},
    )
    cond = ChemicalSetup(
        formula_matrix=jnp.asarray([[2.0, 1.0], [1.0, 1.0]]),
        hvector_func=lambda T: jnp.asarray([-1.0, -1.0]),
        elements=("H", "O"),
        species=("H2O_s", "HO_s"),
        metadata={"source": "unit-test-condensate"},
    )
    return build_condensate_chemical_setup(gas_setup=gas, condensate_setup=cond)


def test_build_condensate_chemical_setup_validates_element_order() -> None:
    gas, cond, setup = _setup_pair()

    validate_condensate_chemical_setup(setup)
    assert setup.gas_setup is gas
    assert setup.condensate_setup is cond
    assert setup.elements == ("H", "O")
    assert setup.gas_species == ("H", "O")
    assert setup.condensate_species == ("H2O_s",)


def test_condensate_chemical_setup_rejects_element_order_mismatch() -> None:
    gas, cond, _setup = _setup_pair()
    bad_cond = ChemicalSetup(
        formula_matrix=cond.formula_matrix,
        hvector_func=cond.hvector_func,
        elements=("O", "H"),
        species=cond.species,
    )

    with pytest.raises(ValueError, match="element orders"):
        build_condensate_chemical_setup(gas_setup=gas, condensate_setup=bad_cond)


def test_build_result_from_solver_payload_uses_full_condensate_vector() -> None:
    _gas, _cond, setup = _setup_pair()

    result = build_condensate_equilibrium_result_from_solver_payload(
        setup=setup,
        gas_ln_n=[0.0, -1.0],
        support_indices=[0],
        support_amounts=[1.0e-6],
        selected_route="m4309_promoted_high_start_callsite_policy",
        metric_status=TIGHT_RESIDUAL_STATUS,
        solver_success=True,
        diagnostics={"budget_residual": 1.0e-8},
    )

    assert result.status == CONVERGED
    assert result.converged is True
    assert result.selected_route == "m4309_promoted_high_start_callsite_policy"
    assert result.condensate_support_names == ("H2O_s",)
    assert result.condensate_amounts.shape == (1,)
    assert float(result.condensate_amounts[0]) == pytest.approx(1.0e-6)
    assert result.diagnostics is not None
    assert result.diagnostics["route"] == HEAD_ROUTE_STANDARD
    assert result.diagnostics["fastchem4_trace_public_runtime_constructor_inputs_used"] is False


def test_build_result_accepts_full_condensate_budget_residual_gate() -> None:
    _gas, _cond, setup = _setup_pair()

    result = build_condensate_equilibrium_result_from_solver_payload(
        setup=setup,
        gas_ln_n=[0.0, 0.0],
        support_indices=[0],
        support_amounts=[1.0],
        selected_route="m4310_full_promoted_policy_route",
        metric_status=TIGHT_RESIDUAL_STATUS,
        solver_success=True,
        element_inventory_target=jnp.asarray([3.0, 2.0]),
    )

    assert result.status == CONVERGED
    assert result.converged is True
    assert result.diagnostics is not None
    gate = result.diagnostics["full_condensate_budget_residual_gate"]
    assert gate["accepted"] is True
    assert gate["max_abs_relative_residual"] == pytest.approx(0.0)


def test_build_result_restores_external_condensates_before_budget_gate() -> None:
    setup = _setup_pair_with_two_condensates()

    result = build_condensate_equilibrium_result_from_solver_payload(
        setup=setup,
        gas_ln_n=[0.0, 0.0],
        support_indices=[1],
        support_amounts=[1.0],
        external_condensate_amounts=jnp.asarray([0.5, 0.0]),
        selected_route="m4310_full_promoted_policy_route",
        metric_status=TIGHT_RESIDUAL_STATUS,
        solver_success=True,
        element_inventory_target=jnp.asarray([3.0, 2.5]),
    )

    assert result.status == CONVERGED
    assert result.converged is True
    assert result.condensate_support_names == ("HO_s",)
    assert result.condensate_amounts.tolist() == pytest.approx([0.5, 1.0])
    assert result.diagnostics is not None
    gate = result.diagnostics["full_condensate_budget_residual_gate"]
    assert gate["accepted"] is True
    assert gate["max_abs_relative_residual"] == pytest.approx(0.0)


def test_build_result_rejects_full_condensate_budget_residual_gate() -> None:
    _gas, _cond, setup = _setup_pair()

    result = build_condensate_equilibrium_result_from_solver_payload(
        setup=setup,
        gas_ln_n=[0.0, 0.0],
        support_indices=[0],
        support_amounts=[1.0],
        selected_route="m4310_full_promoted_policy_route",
        metric_status=TIGHT_RESIDUAL_STATUS,
        solver_success=True,
        element_inventory_target=jnp.asarray([1.0, 1.0]),
    )

    assert result.status == NOT_CONVERGED
    assert result.converged is False
    assert result.acceptance_tier == "full_condensate_element_budget_residual_failed"
    assert result.diagnostics is not None
    gate = result.diagnostics["full_condensate_budget_residual_gate"]
    assert gate["accepted"] is False
    assert gate["max_abs_relative_residual_element"] == "H"
    assert result.diagnostics["pre_full_condensate_budget_gate_status"] == CONVERGED


def test_full_condensate_budget_residual_gate_ignores_electron_row() -> None:
    gas = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        hvector_func=lambda T: jnp.asarray([0.0, 0.0]),
        elements=("H", "e-"),
        species=("H", "e-"),
        metadata={"source": "unit-test-gas"},
    )
    cond = ChemicalSetup(
        formula_matrix=jnp.asarray([[0.0], [0.0]]),
        hvector_func=lambda T: jnp.asarray([0.0]),
        elements=("H", "e-"),
        species=("dummy_s",),
        metadata={"source": "unit-test-condensate"},
    )
    setup = build_condensate_chemical_setup(gas_setup=gas, condensate_setup=cond)

    result = build_condensate_equilibrium_result_from_solver_payload(
        setup=setup,
        gas_ln_n=[0.0, 0.0],
        support_indices=[],
        support_amounts=[],
        selected_route="m4310_full_promoted_policy_route",
        metric_status=TIGHT_RESIDUAL_STATUS,
        solver_success=True,
        element_inventory_target=jnp.asarray([1.0, 0.0]),
    )

    assert result.status == CONVERGED
    assert result.diagnostics is not None
    gate = result.diagnostics["full_condensate_budget_residual_gate"]
    assert gate["accepted"] is True
    assert gate["ignored_element_names"] == ("e-",)
    assert gate["max_abs_relative_residual_element"] == "H"


def test_full_condensate_budget_amount_polish_accepts_capacity_top_up() -> None:
    _, _, setup = _setup_pair()

    polished, report = _polish_support_amounts_for_full_condensate_budget_gate(
        setup=setup,
        gas_ln_n=jnp.asarray([-1000.0, -1000.0]),
        support_indices=(0,),
        support_amounts=jnp.asarray([0.499]),
        element_inventory_target=jnp.asarray([1.0, 0.5]),
        relative_tolerance=1.0e-3,
    )

    assert polished == pytest.approx([0.5])
    assert report is not None
    assert report["triggered"] is True
    assert report["accepted"] is True
    assert report["initial_full_condensate_budget_gate"]["accepted"] is False
    assert report["final_full_condensate_budget_gate"]["accepted"] is True


def test_build_result_preserves_caveat_status_when_allowed() -> None:
    _gas, _cond, setup = _setup_pair()

    result = build_condensate_equilibrium_result_from_solver_payload(
        setup=setup,
        gas_ln_n=[0.0, -1.0],
        support_indices=[0],
        support_amounts=[1.0e-6],
        selected_route="adaptive_refresh_selector_default_depleted_refresh_budget_tradeoff",
        metric_status=BUDGET_TRADEOFF_STATUS,
        solver_success=True,
        allow_caveat_tiers=True,
    )

    assert result.status == CONVERGED_WITH_CAVEAT
    assert result.converged is True
    assert result.diagnostics is not None
    assert result.diagnostics["warning_messages"]


def test_build_result_can_reject_caveat_tiers() -> None:
    _gas, _cond, setup = _setup_pair()

    result = build_condensate_equilibrium_result_from_solver_payload(
        setup=setup,
        gas_ln_n=[0.0, -1.0],
        support_indices=[0],
        support_amounts=[1.0e-6],
        selected_route="adaptive_refresh_selector_default_depleted_refresh_budget_tradeoff",
        metric_status=BUDGET_TRADEOFF_STATUS,
        solver_success=True,
        allow_caveat_tiers=False,
    )

    assert result.status == "not_converged"
    assert result.converged is False


def test_condensate_equilibrium_auto_selects_positive_support_and_calls_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    setup = _setup_pair_with_condensate_hvalue(-1.0)
    captured: dict[str, object] = {}

    def fake_solve_restricted_support_condensate_layer(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {
            "solver_success": True,
            "ln_nk": jnp.asarray([0.0, -1.0]),
            "support_indices": tuple(kwargs["support_indices"]),
            "m_support": jnp.asarray(kwargs["support_amounts_init"]),
        }

    import exogibbs.optimize.minimize_cond as minimize_cond

    monkeypatch.setattr(
        minimize_cond,
        "solve_restricted_support_condensate_layer",
        fake_solve_restricted_support_condensate_layer,
    )

    result = condensate_equilibrium(
        setup,
        300.0,
        1.0,
        jnp.asarray([1.0, 1.0]),
        options=CondensateEquilibriumOptions(
            return_diagnostics=True,
            metric_status=TIGHT_RESIDUAL_STATUS,
            max_positive_support_count=1,
            enable_full_condensate_budget_residual_gate=False,
        ),
    )

    assert result.status == NOT_CONVERGED
    assert result.converged is False
    assert result.head_route_version == CONDENSATE_HEAD_ROUTE_VERSION
    assert result.head_route_name == CONDENSATE_HEAD_ROUTE_NAME
    assert result.condensate_support_names == ("H2O_s",)
    assert float(result.condensate_amounts[0]) > 0.0
    assert captured["kwargs"] is not None
    kwargs = captured["kwargs"]
    assert tuple(kwargs["support_indices"]) == (0,)
    assert tuple(float(value) for value in kwargs["support_amounts_init"]) == pytest.approx((0.5,))
    assert result.diagnostics is not None
    assert result.diagnostics["head_route_version"] == CONDENSATE_HEAD_ROUTE_VERSION
    assert result.diagnostics["head_route_name"] == CONDENSATE_HEAD_ROUTE_NAME
    assert result.diagnostics["support_selection"]["solver_inputs"]["support_indices"] == (0,)
    assert (
        result.diagnostics["support_selection"]["solver_inputs"]["amount_gauge"]
        == "element_inventory_target_fraction"
    )
    assert (
        result.diagnostics["support_selection"]["solver_inputs"][
            "fastchem4_first_step_equivalent_gauge"
        ]
        == "number_density_divided_by_initial_gas_phase_total_element_density"
    )
    inactive_driving = result.diagnostics["inactive_condensate_driving"]
    assert inactive_driving["report_schema"] == (
        "exogibbs_inactive_condensate_driving_report_v1"
    )
    assert inactive_driving["diagnostic_only"] is True
    assert inactive_driving["production_behavior_change"] is False


def test_condensate_equilibrium_options_default_to_head_route_v1_8() -> None:
    options = CondensateEquilibriumOptions()

    assert options.max_positive_support_count is None
    assert options.max_support_add_per_round is None
    assert options.seed_initialization_policy == "max_density"
    assert options.restricted_reduced_coupling_mode == "pdipm_rgie_v11_activity_correction"
    assert options.support_cap_retry_counts == (34, 48, 80, 128)
    assert options.support_growth_staging_retry_add_per_rounds == (64, 32, 16, 8)
    assert options.enable_head_route_soft_restoration_retry is False
    assert options.enable_head_route_ipopt_h_type_retry is False
    assert options.enable_head_route_condensate_budget_correction_retry is True
    assert options.enable_support_closure_retry_gate is True
    assert options.support_closure_max_positive_inactive_count is None
    assert options.enable_full_condensate_budget_residual_gate is True
    assert options.full_condensate_budget_relative_tolerance == pytest.approx(1.0e-3)


def test_support_outer_loop_does_not_grow_from_native_seed_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup_pair_with_two_condensates()
    activity_calls: list[tuple[int, ...]] = []

    def fake_activity_report(**kwargs):
        existing = tuple(int(index) for index in kwargs.get("existing_support_indices", ()))
        activity_calls.append(existing)
        if not existing:
            return {
                "positive_support_indices": (0,),
                "positive_support_names": ("H2O_s",),
                "inactive_positive_indices": (),
                "inactive_positive_names": (),
            }
        return {
            "positive_support_indices": (0, 1),
            "positive_support_names": ("H2O_s", "HO_s"),
            "inactive_positive_indices": (1,),
            "inactive_positive_names": ("HO_s",),
        }

    def fake_solve_restricted_support_condensate_layer(*args, **kwargs):
        return {
            "solver_success": False,
            "ln_nk": jnp.asarray([0.0, -1.0]),
            "support_indices": tuple(kwargs["support_indices"]),
            "m_support": jnp.asarray(kwargs["support_amounts_init"]),
            "diagnostics": {"final_residual": 1.0, "n_iter": 1, "hit_max_iter": True},
        }

    import exogibbs.api.condensate_equilibrium as condensate_api
    import exogibbs.optimize.minimize_cond as minimize_cond

    monkeypatch.setattr(
        condensate_api,
        "_activity_driven_support_report",
        fake_activity_report,
    )
    monkeypatch.setattr(
        minimize_cond,
        "solve_restricted_support_condensate_layer",
        fake_solve_restricted_support_condensate_layer,
    )

    result = condensate_equilibrium(
        setup,
        300.0,
        1.0,
        jnp.asarray([1.0, 1.0]),
        options=CondensateEquilibriumOptions(
            return_diagnostics=True,
            allow_caveat_tiers=True,
            max_positive_support_count=None,
            max_support_add_per_round=None,
            enable_support_cap_retry=False,
            enable_support_growth_staging_retry=False,
        ),
    )

    assert activity_calls == [()]
    assert result.selected_route == "native_budget_seed_fallback_budget_tradeoff"
    assert result.condensate_support_names == ("H2O_s",)
    assert result.diagnostics is not None
    support_selection = result.diagnostics["support_selection"]
    assert support_selection["solver_inputs"]["support_indices"] == (0,)
    assert (
        support_selection["outer_loop"]["terminated_reason"]
        == "support_growth_stopped_after_unaccepted_head_route_result"
    )


def test_support_outer_loop_preserves_solver_amounts_when_growing_from_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup_pair_with_two_condensates()
    solve_calls: list[tuple[tuple[int, ...], tuple[float, ...]]] = []

    def fake_activity_report(**kwargs):
        existing = tuple(int(index) for index in kwargs.get("existing_support_indices", ()))
        if not existing:
            return {
                "positive_support_indices": (0,),
                "positive_support_names": ("H2O_s",),
                "inactive_positive_indices": (),
                "inactive_positive_names": (),
            }
        if existing == (0,):
            assert kwargs.get("element_potential_override") is not None
            return {
                "positive_support_indices": (0, 1),
                "positive_support_names": ("H2O_s", "HO_s"),
                "inactive_positive_indices": (1,),
                "inactive_positive_names": ("HO_s",),
            }
        assert existing == (0, 1)
        return {
            "positive_support_indices": (0, 1),
            "positive_support_names": ("H2O_s", "HO_s"),
            "inactive_positive_indices": (),
            "inactive_positive_names": (),
        }

    def fake_condensate_equilibrium(*args, **kwargs):
        support = tuple(int(index) for index in kwargs["support_indices"])
        amounts = tuple(float(value) for value in kwargs["support_amounts_init"])
        solve_calls.append((support, amounts))
        full_amounts = jnp.zeros((2,), dtype=jnp.float64)
        for index, amount in zip(support, amounts):
            full_amounts = full_amounts.at[index].set(amount)
        if len(solve_calls) == 1:
            diagnostics = {
                "restricted_solver_payload_for_support_growth": {
                    "ln_nk": jnp.asarray([0.0, -1.0], dtype=jnp.float64),
                    "support_indices": (0,),
                    "m_support": jnp.asarray([0.25], dtype=jnp.float64),
                    "pi_vector": jnp.asarray([1.0, 1.0], dtype=jnp.float64),
                }
            }
            route = "native_budget_seed_fallback_budget_tradeoff"
            status = CONVERGED_WITH_CAVEAT
            converged = True
        else:
            diagnostics = {}
            route = "m4310_full_promoted_policy_route"
            status = CONVERGED
            converged = True
        return condensate_api.CondensateEquilibriumResult(
            gas_ln_n=jnp.asarray([0.0, -1.0], dtype=jnp.float64),
            gas_n=jnp.asarray([1.0, 0.1], dtype=jnp.float64),
            gas_x=jnp.asarray([0.9, 0.1], dtype=jnp.float64),
            gas_ntot=jnp.asarray(1.1, dtype=jnp.float64),
            condensate_amounts=full_amounts,
            condensate_support_indices=jnp.asarray(support, dtype=jnp.int32),
            condensate_support_names=tuple(
                setup.condensate_species[index] for index in support
            ),
            acceptance_tier="tier_1_tight_residual_production_adjacent_candidate",
            selected_route=route,
            status=status,
            converged=converged,
            diagnostics=diagnostics,
        )

    import exogibbs.api.condensate_equilibrium as condensate_api

    monkeypatch.setattr(
        condensate_api,
        "_activity_driven_support_report",
        fake_activity_report,
    )
    monkeypatch.setattr(
        condensate_api,
        "condensate_equilibrium",
        fake_condensate_equilibrium,
    )

    result = condensate_api._run_activity_driven_support_outer_loop(
        setup=setup,
        T=300.0,
        P=1.0,
        b=jnp.asarray([1.0, 1.0]),
        Pref=1.0,
        options=CondensateEquilibriumOptions(
            return_diagnostics=True,
            max_positive_support_count=None,
            max_support_add_per_round=None,
            seed_initialization_policy="max_density",
        ),
    )

    assert solve_calls[0][0] == (0,)
    assert solve_calls[0][1] == pytest.approx((0.5,))
    assert solve_calls[1][0] == (0, 1)
    assert solve_calls[1][1] == pytest.approx((0.25, 1.0))
    assert tuple(int(index) for index in result.condensate_support_indices) == (0, 1)
    assert result.diagnostics is not None
    assert (
        result.diagnostics["support_selection"]["outer_loop"]["terminated_reason"]
        == "no_inactive_positive_support"
    )
    assert result.diagnostics["support_selection"]["fastchem4_trace_values_used"] is False
    assert result.diagnostics["support_selection"]["fastchem4_public_values_used_as_constructor_inputs"] is False


def test_support_outer_loop_floors_zero_solver_amounts_before_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup_pair_with_two_condensates()
    solve_calls: list[tuple[tuple[int, ...], tuple[float, ...]]] = []

    def fake_activity_report(**kwargs):
        existing = tuple(int(index) for index in kwargs.get("existing_support_indices", ()))
        if not existing:
            return {
                "positive_support_indices": (0,),
                "positive_support_names": ("H2O_s",),
                "inactive_positive_indices": (),
                "inactive_positive_names": (),
            }
        return {
            "positive_support_indices": (0, 1),
            "positive_support_names": ("H2O_s", "HO_s"),
            "inactive_positive_indices": (1,),
            "inactive_positive_names": ("HO_s",),
        }

    def fake_condensate_equilibrium(*args, **kwargs):
        support = tuple(int(index) for index in kwargs["support_indices"])
        amounts = tuple(float(value) for value in kwargs["support_amounts_init"])
        solve_calls.append((support, amounts))
        full_amounts = jnp.zeros((2,), dtype=jnp.float64)
        for index, amount in zip(support, amounts):
            full_amounts = full_amounts.at[index].set(amount)
        if len(solve_calls) == 1:
            diagnostics = {
                "restricted_solver_payload_for_support_growth": {
                    "ln_nk": jnp.asarray([0.0, -1.0], dtype=jnp.float64),
                    "support_indices": (0,),
                    "m_support": jnp.asarray([0.0], dtype=jnp.float64),
                    "pi_vector": jnp.asarray([1.0, 1.0], dtype=jnp.float64),
                }
            }
            route = "native_budget_seed_fallback_budget_tradeoff"
            status = CONVERGED_WITH_CAVEAT
        else:
            diagnostics = {}
            route = "m4310_full_promoted_policy_route"
            status = CONVERGED
        return condensate_api.CondensateEquilibriumResult(
            gas_ln_n=jnp.asarray([0.0, -1.0], dtype=jnp.float64),
            gas_n=jnp.asarray([1.0, 0.1], dtype=jnp.float64),
            gas_x=jnp.asarray([0.9, 0.1], dtype=jnp.float64),
            gas_ntot=jnp.asarray(1.1, dtype=jnp.float64),
            condensate_amounts=full_amounts,
            condensate_support_indices=jnp.asarray(support, dtype=jnp.int32),
            condensate_support_names=tuple(setup.condensate_species[index] for index in support),
            acceptance_tier="tier_1_tight_residual_production_adjacent_candidate",
            selected_route=route,
            status=status,
            converged=True,
            diagnostics=diagnostics,
        )

    import exogibbs.api.condensate_equilibrium as condensate_api

    monkeypatch.setattr(
        condensate_api,
        "_activity_driven_support_report",
        fake_activity_report,
    )
    monkeypatch.setattr(
        condensate_api,
        "condensate_equilibrium",
        fake_condensate_equilibrium,
    )

    result = condensate_api._run_activity_driven_support_outer_loop(
        setup=setup,
        T=300.0,
        P=1.0,
        b=jnp.asarray([1.0, 1.0]),
        Pref=1.0,
        options=CondensateEquilibriumOptions(
            return_diagnostics=True,
            max_positive_support_count=None,
            max_support_add_per_round=None,
            seed_initialization_policy="max_density",
            min_seed_amount=1.0e-200,
        ),
    )

    assert solve_calls[0][0] == (0,)
    assert solve_calls[1][0] == (0, 1)
    assert solve_calls[1][1] == pytest.approx((1.0e-200, 1.0))
    assert tuple(int(index) for index in result.condensate_support_indices) == (0, 1)


def test_support_outer_loop_tries_support_cap_retry_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gas = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        hvector_func=lambda T: jnp.asarray([0.0, 0.0]),
        elements=("H", "O"),
        species=("H", "O"),
        metadata={"source": "unit-test-gas"},
    )
    cond = ChemicalSetup(
        formula_matrix=jnp.asarray(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
        ),
        hvector_func=lambda T: jnp.asarray([-1.0, -1.0, -1.0, -1.0]),
        elements=("H", "O"),
        species=("C0_s", "C1_s", "C2_s", "C3_s"),
        metadata={"source": "unit-test-condensate"},
    )
    setup = build_condensate_chemical_setup(gas_setup=gas, condensate_setup=cond)
    calls: list[int | None] = []

    def fake_activity_report(**kwargs):
        return {
            "positive_support_indices": (0, 1, 2, 3),
            "positive_support_names": ("C0_s", "C1_s", "C2_s", "C3_s"),
            "inactive_positive_indices": (),
            "inactive_positive_names": (),
        }

    def fake_condensate_equilibrium(*args, **kwargs):
        options = kwargs["options"]
        cap = options.max_positive_support_count
        calls.append(cap)
        if cap == 3:
            support = (0, 1, 2)
            route = "m4310_full_promoted_policy_route"
            status = CONVERGED
            converged = True
        elif cap == 1:
            support = (0,)
            route = "native_budget_seed_fallback_budget_tradeoff"
            status = CONVERGED_WITH_CAVEAT
            converged = True
        else:
            support = tuple(int(index) for index in kwargs["support_indices"])
            route = "native_budget_seed_fallback_budget_tradeoff"
            status = CONVERGED_WITH_CAVEAT
            converged = True
        full_amounts = jnp.zeros((4,), dtype=jnp.float64)
        for index in support:
            full_amounts = full_amounts.at[index].set(0.1)
        return condensate_api.CondensateEquilibriumResult(
            gas_ln_n=jnp.asarray([0.0, -1.0], dtype=jnp.float64),
            gas_n=jnp.asarray([1.0, 0.1], dtype=jnp.float64),
            gas_x=jnp.asarray([0.9, 0.1], dtype=jnp.float64),
            gas_ntot=jnp.asarray(1.1, dtype=jnp.float64),
            condensate_amounts=full_amounts,
            condensate_support_indices=jnp.asarray(support, dtype=jnp.int32),
            condensate_support_names=tuple(setup.condensate_species[index] for index in support),
            acceptance_tier="tier_1_tight_residual_production_adjacent_candidate",
            selected_route=route,
            status=status,
            converged=converged,
            diagnostics={},
        )

    import exogibbs.api.condensate_equilibrium as condensate_api

    monkeypatch.setattr(
        condensate_api,
        "_activity_driven_support_report",
        fake_activity_report,
    )
    monkeypatch.setattr(
        condensate_api,
        "condensate_equilibrium",
        fake_condensate_equilibrium,
    )

    result = condensate_api._run_activity_driven_support_outer_loop(
        setup=setup,
        T=300.0,
        P=1.0,
        b=jnp.asarray([1.0, 1.0]),
        Pref=1.0,
        options=CondensateEquilibriumOptions(
            return_diagnostics=True,
            max_positive_support_count=None,
            support_cap_retry_counts=(1, 3),
        ),
    )

    assert calls == [None, 1, 3]
    assert result.selected_route == "m4310_full_promoted_policy_route"
    assert result.diagnostics is not None
    retry = result.diagnostics["support_cap_retry"]
    assert retry["support_cap"] == 3
    assert retry["accepted"] is True
    assert retry["route_promoted"] is True
    assert retry["support_cap_sequence"] == (1, 3)
    assert [attempt["support_cap"] for attempt in retry["attempts"]] == [1, 3]
    assert [attempt["route_promoted"] for attempt in retry["attempts"]] == [False, True]


def test_support_cap_retry_skips_promoted_candidate_when_closure_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gas = ChemicalSetup(
        formula_matrix=jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        hvector_func=lambda T: jnp.asarray([0.0, 0.0]),
        elements=("H", "O"),
        species=("H", "O"),
        metadata={"source": "unit-test-gas"},
    )
    cond = ChemicalSetup(
        formula_matrix=jnp.asarray(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
        ),
        hvector_func=lambda T: jnp.asarray([-1.0, -1.0, -1.0, -1.0]),
        elements=("H", "O"),
        species=("C0_s", "C1_s", "C2_s", "C3_s"),
        metadata={"source": "unit-test-condensate"},
    )
    setup = build_condensate_chemical_setup(gas_setup=gas, condensate_setup=cond)
    calls: list[int | None] = []

    def fake_activity_report(**kwargs):
        return {
            "positive_support_indices": (0, 1, 2, 3),
            "positive_support_names": ("C0_s", "C1_s", "C2_s", "C3_s"),
            "inactive_positive_indices": (),
            "inactive_positive_names": (),
        }

    def fake_condensate_equilibrium(*args, **kwargs):
        options = kwargs["options"]
        cap = options.max_positive_support_count
        calls.append(cap)
        support = tuple(range(4)) if cap is None else tuple(range(int(cap)))
        route = (
            "native_budget_seed_fallback_budget_tradeoff"
            if cap is None
            else "m4310_full_promoted_policy_route"
        )
        status = CONVERGED if cap is not None else NOT_CONVERGED
        full_amounts = jnp.zeros((4,), dtype=jnp.float64)
        for index in support:
            full_amounts = full_amounts.at[index].set(0.1)
        return condensate_api.CondensateEquilibriumResult(
            gas_ln_n=jnp.asarray([0.0, -1.0], dtype=jnp.float64),
            gas_n=jnp.asarray([1.0, 0.1], dtype=jnp.float64),
            gas_x=jnp.asarray([0.9, 0.1], dtype=jnp.float64),
            gas_ntot=jnp.asarray(1.1, dtype=jnp.float64),
            condensate_amounts=full_amounts,
            condensate_support_indices=jnp.asarray(support, dtype=jnp.int32),
            condensate_support_names=tuple(setup.condensate_species[index] for index in support),
            acceptance_tier="tier_1_tight_residual_production_adjacent_candidate",
            selected_route=route,
            status=status,
            converged=cap is not None,
            diagnostics={},
        )

    def fake_support_closure_gate(**kwargs):
        result = kwargs["result"]
        support_count = len(tuple(result.condensate_support_names))
        accepted = support_count >= 3
        return {
            "gate_schema": "unit_test_support_closure_gate",
            "accepted": accepted,
            "max_positive_inactive_driving": 10.0 if accepted else 1000.0,
            "positive_inactive_count": 0 if accepted else 2,
        }

    import exogibbs.api.condensate_equilibrium as condensate_api

    monkeypatch.setattr(
        condensate_api,
        "_activity_driven_support_report",
        fake_activity_report,
    )
    monkeypatch.setattr(
        condensate_api,
        "condensate_equilibrium",
        fake_condensate_equilibrium,
    )
    monkeypatch.setattr(
        condensate_api,
        "_support_closure_retry_gate_report",
        fake_support_closure_gate,
    )

    result = condensate_api._run_activity_driven_support_outer_loop(
        setup=setup,
        T=300.0,
        P=1.0,
        b=jnp.asarray([1.0, 1.0]),
        Pref=1.0,
        options=CondensateEquilibriumOptions(
            return_diagnostics=True,
            max_positive_support_count=None,
            support_cap_retry_counts=(1, 3),
            enable_support_growth_staging_retry=False,
        ),
    )

    assert calls == [None, 1, 3]
    assert result.selected_route == "m4310_full_promoted_policy_route"
    assert result.diagnostics is not None
    retry = result.diagnostics["support_cap_retry"]
    assert retry["support_cap"] == 3
    assert retry["support_closure_accepted"] is True
    assert [attempt["support_cap"] for attempt in retry["attempts"]] == [1, 3]
    assert [attempt["route_promoted"] for attempt in retry["attempts"]] == [True, True]
    assert [attempt["support_closure_accepted"] for attempt in retry["attempts"]] == [
        False,
        True,
    ]


def test_support_closure_gate_rejects_residual_inactive_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup_pair_with_two_condensates()

    def fake_activity_report(**kwargs):
        return {
            "inactive_positive_indices": (1,),
            "candidate_driving": {
                "H2O_s": -1.0,
                "HO_s": 10.0,
            },
        }

    import exogibbs.api.condensate_equilibrium as condensate_api

    monkeypatch.setattr(
        condensate_api,
        "_activity_driven_support_report",
        fake_activity_report,
    )
    result = condensate_api.CondensateEquilibriumResult(
        gas_ln_n=jnp.asarray([0.0, -1.0], dtype=jnp.float64),
        gas_n=jnp.asarray([1.0, 0.1], dtype=jnp.float64),
        gas_x=jnp.asarray([0.9, 0.1], dtype=jnp.float64),
        gas_ntot=jnp.asarray(1.1, dtype=jnp.float64),
        condensate_amounts=jnp.asarray([0.1, 0.0], dtype=jnp.float64),
        condensate_support_indices=jnp.asarray([0], dtype=jnp.int32),
        condensate_support_names=("H2O_s",),
        acceptance_tier="tier_1_tight_residual_production_adjacent_candidate",
        selected_route="m4310_full_promoted_policy_route",
        status=CONVERGED,
        converged=True,
        diagnostics={},
    )

    gate = condensate_api._support_closure_retry_gate_report(
        setup=setup,
        T=300.0,
        P=1.0,
        b=jnp.asarray([1.0, 1.0]),
        Pref=1.0,
        result=result,
        options=CondensateEquilibriumOptions(
            support_closure_max_positive_inactive_driving=500.0,
            support_closure_max_positive_inactive_count=0,
        ),
    )

    assert gate["accepted"] is False
    assert gate["max_positive_inactive_driving_accepted"] is True
    assert gate["positive_inactive_count"] == 1
    assert gate["positive_inactive_count_tolerance"] == 0
    assert gate["positive_inactive_count_accepted"] is False


def test_support_outer_loop_tries_staged_support_growth_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup_pair_with_two_condensates()
    calls: list[tuple[int | None, bool]] = []

    def fake_activity_report(**kwargs):
        return {
            "positive_support_indices": (0, 1),
            "positive_support_names": ("H2O_s", "HO_s"),
            "inactive_positive_indices": (),
            "inactive_positive_names": (),
        }

    def fake_condensate_equilibrium(*args, **kwargs):
        options = kwargs["options"]
        add_per_round = options.max_support_add_per_round
        calls.append((add_per_round, options.enable_support_growth_staging_retry))
        support = tuple(int(index) for index in kwargs.get("support_indices", (0, 1)))
        route = (
            "m4310_full_promoted_policy_route"
            if add_per_round == 2
            else "native_budget_seed_fallback_budget_tradeoff"
        )
        status = (
            CONVERGED
            if route == "m4310_full_promoted_policy_route"
            else CONVERGED_WITH_CAVEAT
        )
        full_amounts = jnp.zeros((2,), dtype=jnp.float64)
        for index in support:
            full_amounts = full_amounts.at[index].set(0.1)
        return condensate_api.CondensateEquilibriumResult(
            gas_ln_n=jnp.asarray([0.0, -1.0], dtype=jnp.float64),
            gas_n=jnp.asarray([1.0, 0.1], dtype=jnp.float64),
            gas_x=jnp.asarray([0.9, 0.1], dtype=jnp.float64),
            gas_ntot=jnp.asarray(1.1, dtype=jnp.float64),
            condensate_amounts=full_amounts,
            condensate_support_indices=jnp.asarray(support, dtype=jnp.int32),
            condensate_support_names=tuple(
                setup.condensate_species[index] for index in support
            ),
            acceptance_tier="tier_1_tight_residual_production_adjacent_candidate",
            selected_route=route,
            status=status,
            converged=True,
            diagnostics={
                "support_outer_loop": {
                    "terminated_reason": "max_support_outer_iterations_reached"
                }
            },
        )

    import exogibbs.api.condensate_equilibrium as condensate_api

    monkeypatch.setattr(
        condensate_api,
        "_activity_driven_support_report",
        fake_activity_report,
    )
    monkeypatch.setattr(
        condensate_api,
        "condensate_equilibrium",
        fake_condensate_equilibrium,
    )

    result = condensate_api._run_activity_driven_support_outer_loop(
        setup=setup,
        T=300.0,
        P=1.0,
        b=jnp.asarray([1.0, 1.0]),
        Pref=1.0,
        options=CondensateEquilibriumOptions(
            return_diagnostics=True,
            enable_support_cap_retry=False,
            support_growth_staging_retry_add_per_rounds=(2, 1),
        ),
    )

    assert calls == [(None, True), (2, False), (1, False)]
    assert result.selected_route == "m4310_full_promoted_policy_route"
    assert result.diagnostics is not None
    retry = result.diagnostics["support_growth_staging_retry"]
    assert retry["triggered"] is True
    assert retry["accepted"] is True
    assert retry["route_promoted"] is True
    assert retry["max_support_add_per_round"] == 2
    assert retry["max_support_add_per_round_sequence"] == (2, 1)
    assert retry["selection_policy"] == "best_support_closure_score"
    assert retry["initial_selected_route"] == "native_budget_seed_fallback_budget_tradeoff"
    assert retry["retry_selected_route"] == "m4310_full_promoted_policy_route"
    assert retry["attempts"][0]["support_outer_terminated_reason"] == (
        "max_support_outer_iterations_reached"
    )


def test_condensate_equilibrium_capacity_fraction_seed_is_api_selectable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup_pair_with_two_condensates()
    captured: dict[str, object] = {}

    def fake_solve_restricted_support_condensate_layer(*args, **kwargs):
        captured["kwargs"] = kwargs
        return {
            "solver_success": False,
            "ln_nk": jnp.asarray([0.0, -1.0]),
            "support_indices": tuple(kwargs["support_indices"]),
            "m_support": jnp.asarray(kwargs["support_amounts_init"]),
            "diagnostics": {"final_residual": 1.0, "n_iter": 1, "hit_max_iter": True},
        }

    import exogibbs.optimize.minimize_cond as minimize_cond

    monkeypatch.setattr(
        minimize_cond,
        "solve_restricted_support_condensate_layer",
        fake_solve_restricted_support_condensate_layer,
    )

    result = condensate_equilibrium(
        setup,
        300.0,
        1.0,
        jnp.asarray([1.0, 1.0]),
        options=CondensateEquilibriumOptions(
            return_diagnostics=True,
            metric_status=TIGHT_RESIDUAL_STATUS,
            enable_support_outer_loop=False,
            max_positive_support_count=2,
            seed_initialization_policy="capacity_fraction",
            seed_fraction=1.0e-3,
            max_seed_amount=1.0,
        ),
    )

    kwargs = captured["kwargs"]
    assert tuple(kwargs["support_indices"]) == (0, 1)
    assert tuple(float(value) for value in kwargs["support_amounts_init"]) == pytest.approx(
        (5.0e-4, 1.0e-3)
    )
    assert result.diagnostics is not None
    assert (
        result.diagnostics["support_selection"]["solver_inputs"]["seed_initialization_policy"]
        == "capacity_fraction"
    )
    assert (
        result.diagnostics["support_selection"]["solver_inputs"]["uses_b_not_b_normalized_by_sum_b"]
        is True
    )


def test_condensate_equilibrium_max_density_seed_is_api_selectable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup_pair_with_two_condensates()
    captured: dict[str, object] = {}

    def fake_solve_restricted_support_condensate_layer(*args, **kwargs):
        captured["kwargs"] = kwargs
        return {
            "solver_success": False,
            "ln_nk": jnp.asarray([0.0, -1.0]),
            "support_indices": tuple(kwargs["support_indices"]),
            "m_support": jnp.asarray(kwargs["support_amounts_init"]),
            "diagnostics": {"final_residual": 1.0, "n_iter": 1, "hit_max_iter": True},
        }

    import exogibbs.optimize.minimize_cond as minimize_cond

    monkeypatch.setattr(
        minimize_cond,
        "solve_restricted_support_condensate_layer",
        fake_solve_restricted_support_condensate_layer,
    )

    condensate_equilibrium(
        setup,
        300.0,
        1.0,
        jnp.asarray([1.0, 1.0]),
        options=CondensateEquilibriumOptions(
            return_diagnostics=True,
            metric_status=TIGHT_RESIDUAL_STATUS,
            enable_support_outer_loop=False,
            max_positive_support_count=2,
            seed_initialization_policy="max_density",
            seed_fraction=1.0e-3,
            max_seed_amount=1.0e-3,
            enable_head_route_warm_start=False,
        ),
    )

    kwargs = captured["kwargs"]
    assert tuple(kwargs["support_indices"]) == (0, 1)
    assert tuple(float(value) for value in kwargs["support_amounts_init"]) == pytest.approx(
        (5.0e-1, 1.0)
    )


def test_condensate_equilibrium_soft_restoration_retry_can_accept_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup_pair_with_condensate_hvalue(-1.0)
    policies: list[dict[str, object]] = []

    def fake_solve_restricted_support_condensate_layer(*args, **kwargs):
        return {
            "solver_success": True,
            "ln_nk": jnp.asarray([0.0, -1.0]),
            "support_indices": tuple(kwargs["support_indices"]),
            "m_support": jnp.asarray(kwargs["support_amounts_init"]),
            "pi_vector": jnp.asarray([0.0, 0.0]),
            "diagnostics": {"final_residual": 1.0, "n_iter": 1, "hit_max_iter": False},
        }

    class FakeRouteResult:
        def __init__(self, converged: bool):
            self.converged = converged
            self.selected_route = (
                "m4310_full_promoted_policy_route"
                if converged
                else "head_v1_restricted_support"
            )
            self.metric_status = (
                TIGHT_RESIDUAL_STATUS if converged else "runtime_not_accepted"
            )

    class FakeLifecycleReport:
        def __init__(self, converged: bool):
            self.route_result = FakeRouteResult(converged)

        def as_dict(self):
            return {
                "primary_execution_report": {
                    "continuation_report": {
                        "stopped_reason": "no_p_armijo_trial",
                        "outer_records": (),
                    }
                },
                "route_result": {
                    "selected_route": self.route_result.selected_route,
                    "metric_status": self.route_result.metric_status,
                    "converged": self.route_result.converged,
                },
            }

    def fake_lifecycle(**kwargs):
        policy = dict(kwargs["primary_continuation_policy"])
        policies.append(policy)
        return FakeLifecycleReport(
            bool(policy.get("enable_native_soft_restoration_fallback", False))
        )

    import exogibbs.api.condensate_equilibrium as condensate_api
    import exogibbs.optimize.minimize_cond as minimize_cond

    monkeypatch.setattr(
        minimize_cond,
        "solve_restricted_support_condensate_layer",
        fake_solve_restricted_support_condensate_layer,
    )
    monkeypatch.setattr(
        condensate_api,
        "_run_lifecycle_from_restricted_solver_state",
        fake_lifecycle,
    )

    result = condensate_equilibrium(
        setup,
        300.0,
        1.0,
        jnp.asarray([1.0, 1.0]),
        support_indices=(0,),
        support_amounts_init=(0.5,),
        options=CondensateEquilibriumOptions(
            return_diagnostics=True,
            enable_support_outer_loop=False,
            enable_head_route_soft_restoration_retry=True,
            enable_full_condensate_budget_residual_gate=False,
        ),
    )

    assert result.status == CONVERGED
    assert result.selected_route == "m4310_full_promoted_policy_route"
    assert len(policies) == 2
    assert policies[-1]["center_tolerance_multiplier"] == 1.0e11
    assert policies[-1]["enable_native_soft_restoration_fallback"] is True
    assert policies[-1]["soft_restoration_component_weights"] == {
        "budget": 1.0,
        "total_density": 1.0,
        "amount_weighted_gas": 1.0,
        "amount_weighted_condensate": 1.0,
    }
    assert result.diagnostics is not None
    retry = result.diagnostics["head_route_soft_restoration_retry"]
    assert retry["triggered"] is True
    assert retry["accepted"] is True
    assert retry["center_tolerance_multiplier"] == 1.0e11
    assert retry["retry_selected_route"] == "m4310_full_promoted_policy_route"


def test_condensate_equilibrium_ipopt_h_type_retry_can_accept_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup_pair_with_condensate_hvalue(-1.0)
    policies: list[dict[str, object]] = []

    def fake_solve_restricted_support_condensate_layer(*args, **kwargs):
        return {
            "solver_success": True,
            "ln_nk": jnp.asarray([0.0, 0.0]),
            "m_support": jnp.asarray([0.5]),
            "support_indices": (0,),
            "pi_vector": jnp.asarray([0.0, 0.0]),
        }

    class FakeRouteResult:
        def __init__(self, converged: bool):
            self.converged = converged
            self.selected_route = (
                "m4310_full_promoted_policy_route"
                if converged
                else "head_v1_restricted_support"
            )
            self.metric_status = (
                TIGHT_RESIDUAL_STATUS if converged else "runtime_not_accepted"
            )

    class FakeLifecycleReport:
        def __init__(self, converged: bool):
            self.route_result = FakeRouteResult(converged)

        def as_dict(self):
            return {
                "primary_execution_report": {
                    "continuation_report": {
                        "stopped_reason": "no_p_armijo_trial",
                        "outer_records": (),
                    }
                },
                "route_result": {
                    "selected_route": self.route_result.selected_route,
                    "metric_status": self.route_result.metric_status,
                    "converged": self.route_result.converged,
                },
            }

    def fake_lifecycle(**kwargs):
        policy = dict(kwargs["primary_continuation_policy"])
        policies.append(policy)
        return FakeLifecycleReport(
            policy.get("trial_acceptance_policy") == "ipopt_persistent_h_type"
        )

    import exogibbs.api.condensate_equilibrium as condensate_api
    import exogibbs.optimize.minimize_cond as minimize_cond

    monkeypatch.setattr(
        minimize_cond,
        "solve_restricted_support_condensate_layer",
        fake_solve_restricted_support_condensate_layer,
    )
    monkeypatch.setattr(
        condensate_api,
        "_run_lifecycle_from_restricted_solver_state",
        fake_lifecycle,
    )

    result = condensate_equilibrium(
        setup,
        300.0,
        1.0,
        jnp.asarray([1.0, 1.0]),
        support_indices=(0,),
        support_amounts_init=(0.5,),
        options=CondensateEquilibriumOptions(
            return_diagnostics=True,
            enable_support_outer_loop=False,
            enable_head_route_ipopt_h_type_retry=True,
            enable_full_condensate_budget_residual_gate=False,
        ),
    )

    assert result.status == CONVERGED
    assert result.selected_route == "m4310_full_promoted_policy_route"
    assert len(policies) == 2
    assert policies[-1]["trial_acceptance_policy"] == "ipopt_persistent_h_type"
    assert policies[-1]["center_tolerance_multiplier"] == 1.0e11
    assert policies[-1]["require_residual_nonworsening"] is False
    assert policies[-1]["filter_component_weights"] == {
        "budget": 1.0,
        "total_density": 1.0,
        "amount_weighted_gas": 1.0,
        "amount_weighted_condensate": 1.0,
        "complementarity": 1.0,
    }
    assert policies[-1]["ipopt_h_type_protected_components"] == (
        "budget",
        "total_density",
    )
    assert result.diagnostics is not None
    retry = result.diagnostics["head_route_ipopt_h_type_retry"]
    assert retry["triggered"] is True
    assert retry["accepted"] is True
    assert retry["trial_acceptance_policy"] == "ipopt_persistent_h_type"
    assert retry["retry_selected_route"] == "m4310_full_promoted_policy_route"


def test_condensate_equilibrium_retries_restricted_solver_with_refresh_warm_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup_pair_with_condensate_hvalue(-1.0)
    attempts: list[dict[str, object]] = []

    def fake_solve_restricted_support_condensate_layer(*args, **kwargs):
        initial_override = kwargs.get("initial_log_state_override")
        attempts.append(
            {
                "support_indices": tuple(kwargs["support_indices"]),
                "has_initial_override": initial_override is not None,
            }
        )
        attempt_success = len(attempts) > 1
        return {
            "solver_success": attempt_success,
            "ln_nk": jnp.asarray([0.0, -1.0]),
            "support_indices": tuple(kwargs["support_indices"]),
            "m_support": jnp.asarray(kwargs["support_amounts_init"]),
        }

    import exogibbs.optimize.minimize_cond as minimize_cond

    monkeypatch.setattr(
        minimize_cond,
        "solve_restricted_support_condensate_layer",
        fake_solve_restricted_support_condensate_layer,
    )

    result = condensate_equilibrium(
        setup,
        300.0,
        1.0,
        jnp.asarray([1.0, 1.0]),
        options=CondensateEquilibriumOptions(
            return_diagnostics=True,
            metric_status=TIGHT_RESIDUAL_STATUS,
            max_positive_support_count=1,
            enable_full_condensate_budget_residual_gate=False,
        ),
    )

    assert len(attempts) == 2
    assert attempts[0]["has_initial_override"] is True
    assert attempts[1]["has_initial_override"] is True
    assert result.diagnostics is not None
    assert result.diagnostics["head_route_solver_attempts"][0]["candidate_kind"] == "baseline"
    assert result.diagnostics["head_route_solver_attempts"][1]["candidate_kind"] == "depleted_gas_refresh"
    selected = result.diagnostics["selected_warm_start_candidate"]
    assert selected["candidate_kind"] == "depleted_gas_refresh"
    assert selected["fastchem4_trace_public_runtime_constructor_inputs_used"] is False


def test_condensate_equilibrium_passes_api_gas_state_to_baseline_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup_pair_with_condensate_hvalue(-1.0)
    gas_ln_n = jnp.asarray([2.0, -3.0])
    gas_ntot = jnp.asarray(4.0)
    captured: dict[str, object] = {}

    class FakeGasResult:
        ln_n = gas_ln_n
        ntot = gas_ntot

    def fake_equilibrium(*_args, **_kwargs):
        return FakeGasResult()

    def fake_solve_restricted_support_condensate_layer(*args, **kwargs):
        captured["initial_log_state_override"] = kwargs["initial_log_state_override"]
        return {
            "solver_success": False,
            "ln_nk": jnp.asarray(kwargs["initial_log_state_override"].ln_nk),
            "support_indices": tuple(kwargs["support_indices"]),
            "m_support": jnp.asarray(kwargs["support_amounts_init"]),
            "diagnostics": {"final_residual": 1.0, "n_iter": 1, "hit_max_iter": True},
            "restricted_reduced_coupling_config_mode": kwargs[
                "reduced_coupling_config"
            ].reduced_coupling_mode,
        }

    import exogibbs.api.equilibrium as api_equilibrium
    import exogibbs.optimize.minimize_cond as minimize_cond

    monkeypatch.setattr(api_equilibrium, "equilibrium", fake_equilibrium)
    monkeypatch.setattr(
        minimize_cond,
        "solve_restricted_support_condensate_layer",
        fake_solve_restricted_support_condensate_layer,
    )

    condensate_equilibrium(
        setup,
        300.0,
        1.0,
        jnp.asarray([1.0, 1.0]),
        options=CondensateEquilibriumOptions(
            return_diagnostics=True,
            max_positive_support_count=1,
            enable_head_route_warm_start=False,
        ),
    )

    init = captured["initial_log_state_override"]
    assert init is not None
    assert tuple(float(value) for value in init.ln_nk) == pytest.approx(
        tuple(float(value) for value in gas_ln_n)
    )
    assert float(init.ln_ntot) == pytest.approx(float(jnp.log(gas_ntot)))
    assert init.ln_nk_source_trace["source"] == "exogibbs_api_fresh_gas_equilibrium"


def test_condensate_equilibrium_passes_reduced_coupling_mode_to_restricted_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup_pair_with_condensate_hvalue(-1.0)
    captured: dict[str, object] = {}

    def fake_solve_restricted_support_condensate_layer(*args, **kwargs):
        captured["config"] = kwargs["reduced_coupling_config"]
        return {
            "solver_success": False,
            "ln_nk": jnp.asarray([0.0, -1.0]),
            "support_indices": tuple(kwargs["support_indices"]),
            "m_support": jnp.asarray(kwargs["support_amounts_init"]),
            "diagnostics": {"final_residual": 1.0, "n_iter": 1, "hit_max_iter": True},
            "restricted_reduced_coupling_config_mode": kwargs[
                "reduced_coupling_config"
            ].reduced_coupling_mode,
        }

    import exogibbs.optimize.minimize_cond as minimize_cond

    monkeypatch.setattr(
        minimize_cond,
        "solve_restricted_support_condensate_layer",
        fake_solve_restricted_support_condensate_layer,
    )

    result = condensate_equilibrium(
        setup,
        300.0,
        1.0,
        jnp.asarray([1.0, 1.0]),
        options=CondensateEquilibriumOptions(
            return_diagnostics=True,
            max_positive_support_count=1,
            restricted_reduced_coupling_mode=(
                "candidate_selected_active_plus_near_jacobian_with_rem_inventory"
            ),
        ),
    )

    config = captured["config"]
    assert (
        config.reduced_coupling_mode
        == "candidate_selected_active_plus_near_jacobian_with_rem_inventory"
    )
    assert result.diagnostics is not None
    attempt = result.diagnostics["head_route_solver_attempts"][0]
    assert (
        attempt["restricted_reduced_coupling_config_mode"]
        == "candidate_selected_active_plus_near_jacobian_with_rem_inventory"
    )


def test_condensate_equilibrium_passes_pdipm_activity_correction_mode_to_restricted_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup_pair_with_condensate_hvalue(-1.0)
    captured: dict[str, object] = {}

    def fake_solve_restricted_support_condensate_layer(*args, **kwargs):
        captured["config"] = kwargs["reduced_coupling_config"]
        return {
            "solver_success": False,
            "ln_nk": jnp.asarray([0.0, -1.0]),
            "support_indices": tuple(kwargs["support_indices"]),
            "m_support": jnp.asarray(kwargs["support_amounts_init"]),
            "diagnostics": {"final_residual": 1.0, "n_iter": 1, "hit_max_iter": True},
            "restricted_reduced_coupling_config_mode": kwargs[
                "reduced_coupling_config"
            ].reduced_coupling_mode,
        }

    import exogibbs.optimize.minimize_cond as minimize_cond

    monkeypatch.setattr(
        minimize_cond,
        "solve_restricted_support_condensate_layer",
        fake_solve_restricted_support_condensate_layer,
    )

    result = condensate_equilibrium(
        setup,
        300.0,
        1.0,
        jnp.asarray([1.0, 1.0]),
        options=CondensateEquilibriumOptions(
            return_diagnostics=True,
            max_positive_support_count=1,
            restricted_reduced_coupling_mode="pdipm_rgie_v11_activity_correction",
        ),
    )

    config = captured["config"]
    assert config.reduced_coupling_mode == "pdipm_rgie_v11_activity_correction"
    assert result.diagnostics is not None
    attempt = result.diagnostics["head_route_solver_attempts"][0]
    assert (
        attempt["restricted_reduced_coupling_config_mode"]
        == "pdipm_rgie_v11_activity_correction"
    )


def test_condensate_equilibrium_accepts_successful_head_lifecycle_after_solver_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup_pair_with_condensate_hvalue(-1.0)

    def fake_solve_restricted_support_condensate_layer(*args, **kwargs):
        return {
            "solver_success": False,
            "ln_nk": jnp.asarray([0.0, -1.0]),
            "support_indices": tuple(kwargs["support_indices"]),
            "m_support": jnp.asarray(kwargs["support_amounts_init"]),
        }

    class FakeLifecycleReport:
        route_result = type(
            "RouteResult",
            (),
            {
                "selected_route": "m4310_full_promoted_policy_route",
                "metric_status": TIGHT_RESIDUAL_STATUS,
                "converged": True,
            },
        )()

        def as_dict(self):
            return {
                "report_schema": "exogibbs_condensate_head_route_lifecycle_report_v1",
                "explicit_opt_in": True,
                "fastchem4_trace_public_runtime_constructor_inputs_used": False,
                "primary_execution_report": {
                    "continuation_report": {
                        "final_state": {
                            "ln_nk": [0.0, -1.0],
                            "ln_mk": [jnp.log(1.0e-4)],
                        },
                        "converged_at_final_barrier": True,
                    }
                },
                "route_result": {
                    "selected_route": "m4310_full_promoted_policy_route",
                    "metric_status": TIGHT_RESIDUAL_STATUS,
                    "converged": True,
                    "standard_path_status": CONVERGED,
                },
            }

    def fake_run_condensate_head_route_lifecycle(*args, **kwargs):
        return FakeLifecycleReport()

    import exogibbs.optimize.minimize_cond as minimize_cond
    import exogibbs.condensates.head_route_lifecycle as lifecycle_module

    monkeypatch.setattr(
        minimize_cond,
        "solve_restricted_support_condensate_layer",
        fake_solve_restricted_support_condensate_layer,
    )
    monkeypatch.setattr(
        lifecycle_module,
        "run_condensate_head_route_lifecycle",
        fake_run_condensate_head_route_lifecycle,
    )

    result = condensate_equilibrium(
        setup,
        300.0,
        1.0,
        jnp.asarray([1.0, 1.0]),
        options=CondensateEquilibriumOptions(
            return_diagnostics=True,
            metric_status=TIGHT_RESIDUAL_STATUS,
            max_positive_support_count=1,
            enable_full_condensate_budget_residual_gate=False,
        ),
    )

    assert result.status == CONVERGED
    assert result.converged is True
    assert result.selected_route == "m4310_full_promoted_policy_route"
    assert float(result.condensate_amounts[0]) == pytest.approx(1.0e-4)
    assert result.diagnostics is not None
    assert result.diagnostics["restricted_solver_success"] is False
    assert result.diagnostics["solver_success"] is True


def test_condensate_equilibrium_reflects_lifecycle_final_state_with_solver_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup_pair_with_two_condensates()
    captured = {}

    def fake_solve_restricted_support_condensate_layer(*args, **kwargs):
        captured["candidate_support_indices"] = tuple(kwargs["support_indices"])
        return {
            "solver_success": True,
            "ln_nk": jnp.asarray([0.0, -1.0]),
            "support_indices": (1,),
            "m_support": jnp.asarray([1.0e-6]),
        }

    class FakeLifecycleReport:
        route_result = type(
            "RouteResult",
            (),
            {
                "selected_route": "m4310_full_promoted_policy_route",
                "metric_status": TIGHT_RESIDUAL_STATUS,
                "converged": True,
            },
        )()

        def __init__(self, support_indices):
            self._support_indices = tuple(support_indices)

        def as_dict(self):
            return {
                "report_schema": "exogibbs_condensate_head_route_lifecycle_report_v1",
                "explicit_opt_in": True,
                "fastchem4_trace_public_runtime_constructor_inputs_used": False,
                "continuation_input": {
                    "support_indices": self._support_indices,
                },
                "primary_execution_report": {
                    "continuation_report": {
                        "final_state": {
                            "ln_nk": [jnp.log(0.75), jnp.log(0.5)],
                            "ln_mk": [jnp.log(2.0e-4)],
                        },
                        "converged_at_final_barrier": True,
                    }
                },
                "route_result": {
                    "selected_route": "m4310_full_promoted_policy_route",
                    "metric_status": TIGHT_RESIDUAL_STATUS,
                    "converged": True,
                    "standard_path_status": CONVERGED,
                },
            }

    def fake_run_condensate_head_route_lifecycle(*args, **kwargs):
        captured["lifecycle_support_indices"] = tuple(kwargs["support_indices"])
        return FakeLifecycleReport(kwargs["support_indices"])

    import exogibbs.optimize.minimize_cond as minimize_cond
    import exogibbs.condensates.head_route_lifecycle as lifecycle_module

    monkeypatch.setattr(
        minimize_cond,
        "solve_restricted_support_condensate_layer",
        fake_solve_restricted_support_condensate_layer,
    )
    monkeypatch.setattr(
        lifecycle_module,
        "run_condensate_head_route_lifecycle",
        fake_run_condensate_head_route_lifecycle,
    )

    result = condensate_equilibrium(
        setup,
        300.0,
        1.0,
        jnp.asarray([1.0, 1.0]),
        options=CondensateEquilibriumOptions(
            return_diagnostics=True,
            metric_status=TIGHT_RESIDUAL_STATUS,
            max_positive_support_count=2,
            enable_full_condensate_budget_residual_gate=False,
        ),
    )

    assert captured["candidate_support_indices"] == (1, 0)
    assert captured["lifecycle_support_indices"] == (1,)
    assert tuple(int(index) for index in result.condensate_support_indices) == (1,)
    assert float(result.gas_n[0]) == pytest.approx(0.75)
    assert float(result.gas_n[1]) == pytest.approx(0.5)
    assert float(result.condensate_amounts[0]) == pytest.approx(0.0)
    assert float(result.condensate_amounts[1]) == pytest.approx(2.0e-4)
    assert result.diagnostics is not None
    assert result.diagnostics["restricted_solver_success"] is True
    assert result.diagnostics["solver_success"] is True


def test_condensate_budget_correction_retry_starts_from_lifecycle_final_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup_pair_with_two_condensates()
    calls = []

    def fake_solve_restricted_support_condensate_layer(*args, **kwargs):
        return {
            "solver_success": True,
            "ln_nk": jnp.asarray([0.0, 0.0]),
            "support_indices": (1,),
            "m_support": jnp.asarray([1.0e-6]),
        }

    class FakeLifecycleReport:
        def __init__(self, *, ln_nk, ln_mk):
            self._ln_nk = list(ln_nk)
            self._ln_mk = list(ln_mk)
            self.route_result = type(
                "RouteResult",
                (),
                {
                    "selected_route": "m4310_full_promoted_policy_route",
                    "metric_status": TIGHT_RESIDUAL_STATUS,
                    "converged": True,
                },
            )()

        def as_dict(self):
            return {
                "report_schema": "exogibbs_condensate_head_route_lifecycle_report_v1",
                "explicit_opt_in": True,
                "fastchem4_trace_public_runtime_constructor_inputs_used": False,
                "continuation_input": {
                    "support_indices": (1,),
                },
                "primary_execution_report": {
                    "continuation_report": {
                        "final_state": {
                            "ln_nk": self._ln_nk,
                            "ln_mk": self._ln_mk,
                        },
                        "converged_at_final_barrier": True,
                    }
                },
                "route_result": {
                    "selected_route": "m4310_full_promoted_policy_route",
                    "metric_status": TIGHT_RESIDUAL_STATUS,
                    "converged": True,
                    "standard_path_status": CONVERGED,
                },
            }

    def fake_run_condensate_head_route_lifecycle(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return FakeLifecycleReport(
                ln_nk=[jnp.log(0.1), jnp.log(0.1)],
                ln_mk=[jnp.log(0.1)],
            )
        return FakeLifecycleReport(
            ln_nk=[jnp.log(0.1), jnp.log(0.1)],
            ln_mk=[jnp.log(0.9)],
        )

    import exogibbs.optimize.minimize_cond as minimize_cond
    import exogibbs.condensates.head_route_lifecycle as lifecycle_module

    monkeypatch.setattr(
        minimize_cond,
        "solve_restricted_support_condensate_layer",
        fake_solve_restricted_support_condensate_layer,
    )
    monkeypatch.setattr(
        lifecycle_module,
        "run_condensate_head_route_lifecycle",
        fake_run_condensate_head_route_lifecycle,
    )

    result = condensate_equilibrium(
        setup,
        300.0,
        1.0,
        jnp.asarray([1.0, 1.0]),
        options=CondensateEquilibriumOptions(
            return_diagnostics=True,
            max_positive_support_count=2,
        ),
    )

    assert len(calls) == 2
    assert calls[0]["field_provenance"]["ln_nk"] == "exogibbs_restricted_support_solver_output"
    assert calls[1]["field_provenance"]["ln_nk"] == "exogibbs_lifecycle_final_state"
    assert calls[1]["ln_nk"] == pytest.approx([jnp.log(0.1), jnp.log(0.1)])
    assert calls[1]["support_amounts"] == pytest.approx([0.1])
    assert calls[1]["support_indices"] == (1,)
    retry_policy = calls[1]["primary_continuation_policy"]
    assert retry_policy["direction_policy"] == "joint_budget_amount_gas_linearized_no_prior"
    assert retry_policy["budget_row_scaling_policy"] == "relative_target"
    assert retry_policy["filter_component_weights"]["relative_budget_max"] == pytest.approx(1.0)
    assert result.status == CONVERGED
    assert result.diagnostics is not None
    retry = result.diagnostics["head_route_condensate_budget_correction_retry"]
    assert retry["accepted"] is True
    assert retry["retry_start_state"] == "lifecycle_final_state"


def test_condensate_equilibrium_can_disable_native_seed_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup_pair_with_condensate_hvalue(-1.0)

    def fake_solve_restricted_support_condensate_layer(*args, **kwargs):
        return {
            "solver_success": False,
            "ln_nk": jnp.asarray([0.0, -1.0]),
            "support_indices": tuple(kwargs["support_indices"]),
            "m_support": jnp.asarray(kwargs["support_amounts_init"]),
        }

    import exogibbs.optimize.minimize_cond as minimize_cond

    monkeypatch.setattr(
        minimize_cond,
        "solve_restricted_support_condensate_layer",
        fake_solve_restricted_support_condensate_layer,
    )

    result = condensate_equilibrium(
        setup,
        300.0,
        1.0,
        jnp.asarray([1.0, 1.0]),
        options=CondensateEquilibriumOptions(
            return_diagnostics=True,
            max_positive_support_count=1,
            enable_native_seed_fallback=False,
        ),
    )

    assert result.status == NOT_CONVERGED
    assert result.converged is False
    assert result.diagnostics is not None
    assert "native_seed_fallback" not in result.diagnostics


def test_condensate_equilibrium_empty_positive_support_uses_gas_only_path(monkeypatch: pytest.MonkeyPatch) -> None:
    setup = _setup_pair_with_condensate_hvalue(1.0)

    def fake_equilibrium(*args, **kwargs):
        from exogibbs.api.equilibrium import EquilibriumResult

        return EquilibriumResult(
            ln_n=jnp.asarray([0.0, -1.0]),
            n=jnp.asarray([1.0, jnp.exp(-1.0)]),
            x=jnp.asarray([0.73105858, 0.26894142]),
            ntot=jnp.asarray(1.0 + jnp.exp(-1.0)),
            iterations=1,
            metadata={"fake": True},
        )

    import exogibbs.api.equilibrium as equilibrium_module

    monkeypatch.setattr(equilibrium_module, "equilibrium", fake_equilibrium)

    result = condensate_equilibrium(
        setup,
        300.0,
        1.0,
        jnp.asarray([1.0, 1.0]),
        options=CondensateEquilibriumOptions(
            return_diagnostics=True,
            enable_full_condensate_budget_residual_gate=False,
        ),
    )

    assert result.status == CONVERGED
    assert result.converged is True
    assert result.head_route_version == CONDENSATE_HEAD_ROUTE_VERSION
    assert result.head_route_name == CONDENSATE_HEAD_ROUTE_NAME
    assert result.condensate_support_names == ()
    assert result.condensate_support_indices.shape == (0,)
    assert result.condensate_amounts.shape == (1,)
    assert float(result.condensate_amounts[0]) == 0.0
    assert result.selected_route == "head_v1_empty_positive_support_gas_only"
    assert result.diagnostics is not None
    assert result.diagnostics["head_route_version"] == CONDENSATE_HEAD_ROUTE_VERSION
    assert result.diagnostics["head_route_name"] == CONDENSATE_HEAD_ROUTE_NAME
    assert result.diagnostics["support_selection"]["solver_inputs"]["empty_positive_support"] is True
    inactive_driving = result.diagnostics["inactive_condensate_driving"]
    assert inactive_driving["all_condensates"]["positive_inactive_count"] == 0
    assert inactive_driving["temperature_valid_condensates"]["positive_inactive_count"] == 0
    assert (
        inactive_driving["fastchem4_trace_public_runtime_constructor_inputs_used"]
        is False
    )


def test_condensate_equilibrium_rejects_invalid_positive_support_options() -> None:
    _gas, _cond, setup = _setup_pair()

    with pytest.raises(ValueError, match="max_positive_support_count"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(max_positive_support_count=0),
        )
    with pytest.raises(ValueError, match="max_activity_support_count"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(max_activity_support_count=0),
        )
    with pytest.raises(ValueError, match="max_support_add_per_round"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(max_support_add_per_round=0),
        )
    with pytest.raises(ValueError, match="restricted_reduced_coupling_mode"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                restricted_reduced_coupling_mode="not_a_mode",
            ),
        )
    with pytest.raises(ValueError, match="head_route_primary_center_tolerance_multiplier"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                head_route_primary_center_tolerance_multiplier=float("nan"),
            ),
        )
    with pytest.raises(ValueError, match="head_route_primary_residual_worsening_tolerance"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                head_route_primary_residual_worsening_tolerance=float("inf"),
            ),
        )
    with pytest.raises(TypeError, match="head_route_primary_require_residual_nonworsening"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                head_route_primary_require_residual_nonworsening="False",
            ),
        )
    with pytest.raises(ValueError, match="head_route_primary_acceptance_guard"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                head_route_primary_acceptance_guard="not_a_guard",
            ),
        )
    with pytest.raises(ValueError, match="head_route_primary_guard_max_budget"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                head_route_primary_guard_max_budget=float("nan"),
            ),
        )
    with pytest.raises(TypeError, match="enable_support_cap_retry"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                enable_support_cap_retry="False",
            ),
        )
    with pytest.raises(TypeError, match="enable_head_route_soft_restoration_retry"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                enable_head_route_soft_restoration_retry="False",
            ),
        )
    with pytest.raises(TypeError, match="enable_full_condensate_budget_residual_gate"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                enable_full_condensate_budget_residual_gate="False",
            ),
        )
    with pytest.raises(
        TypeError,
        match="enable_head_route_condensate_budget_correction_retry",
    ):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                enable_head_route_condensate_budget_correction_retry="False",
            ),
        )
    with pytest.raises(ValueError, match="full_condensate_budget_relative_tolerance"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                full_condensate_budget_relative_tolerance=float("nan"),
            ),
        )
    with pytest.raises(ValueError, match="head_route_soft_restoration_proximity_weight"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                head_route_soft_restoration_proximity_weight=float("nan"),
            ),
        )
    with pytest.raises(ValueError, match="head_route_soft_restoration_max_proximity"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                head_route_soft_restoration_max_proximity=-1.0,
            ),
        )
    with pytest.raises(TypeError, match="enable_head_route_ipopt_h_type_retry"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                enable_head_route_ipopt_h_type_retry="False",
            ),
        )
    with pytest.raises(ValueError, match="head_route_ipopt_h_type_theta_reduction_fraction"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                head_route_ipopt_h_type_theta_reduction_fraction=1.0,
            ),
        )
    with pytest.raises(
        ValueError,
        match="head_route_ipopt_h_type_protected_component_max_normalized_increase",
    ):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                head_route_ipopt_h_type_protected_component_max_normalized_increase=-1.0,
            ),
        )
    with pytest.raises(ValueError, match="support_cap_retry_count"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                support_cap_retry_count=0,
            ),
        )
    with pytest.raises(ValueError, match="support_cap_retry_counts"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                support_cap_retry_counts=(),
            ),
        )
    with pytest.raises(ValueError, match="support_cap_retry_counts"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                support_cap_retry_counts=(34, 0),
            ),
        )
    with pytest.raises(TypeError, match="enable_support_growth_staging_retry"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                enable_support_growth_staging_retry="False",
            ),
        )
    with pytest.raises(ValueError, match="support_growth_staging_retry_add_per_rounds"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                support_growth_staging_retry_add_per_rounds=(),
            ),
        )
    with pytest.raises(ValueError, match="support_growth_staging_retry_add_per_rounds"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                support_growth_staging_retry_add_per_rounds=(64, 0),
            ),
        )
    with pytest.raises(ValueError, match="support_closure_max_positive_inactive_count"):
        condensate_equilibrium(
            setup,
            300.0,
            1.0,
            jnp.asarray([1.0, 1.0]),
            options=CondensateEquilibriumOptions(
                support_closure_max_positive_inactive_count=-1,
            ),
        )


def test_api_init_does_not_import_condensate_equilibrium_by_default() -> None:
    sys.modules.pop("exogibbs.api", None)
    sys.modules.pop("exogibbs.api.condensate_equilibrium", None)

    module = importlib.import_module("exogibbs.api")

    assert "exogibbs.api.condensate_equilibrium" not in sys.modules
    resolved = getattr(module, "CondensateEquilibriumOptions")
    assert resolved.__name__ == "CondensateEquilibriumOptions"
    assert "exogibbs.api.condensate_equilibrium" in sys.modules
