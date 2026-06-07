"""Tests for the condensate equilibrium production-facing API shell."""

from __future__ import annotations

import importlib
import sys

import jax.numpy as jnp
import pytest

from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.api.condensate_equilibrium import (
    CondensateChemicalSetup,
    CondensateEquilibriumOptions,
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
        ),
    )

    assert result.status == NOT_CONVERGED
    assert result.converged is False
    assert result.condensate_support_names == ("H2O_s",)
    assert float(result.condensate_amounts[0]) > 0.0
    assert captured["kwargs"] is not None
    kwargs = captured["kwargs"]
    assert tuple(kwargs["support_indices"]) == (0,)
    assert tuple(float(value) for value in kwargs["support_amounts_init"]) == pytest.approx((5.0e-4,))
    assert result.diagnostics is not None
    assert result.diagnostics["support_selection"]["solver_inputs"]["support_indices"] == (0,)
    assert result.diagnostics["support_selection"]["fastchem4_trace_values_used"] is False
    assert result.diagnostics["support_selection"]["fastchem4_public_values_used_as_constructor_inputs"] is False
    assert result.diagnostics["head_route_lifecycle"]["primary_execution_report"] is not None
    assert result.diagnostics["head_route_lifecycle"]["route_result"]["standard_path_status"] == NOT_CONVERGED


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
        return {
            "solver_success": initial_override is not None,
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
        ),
    )

    assert len(attempts) == 2
    assert attempts[0]["has_initial_override"] is False
    assert attempts[1]["has_initial_override"] is True
    assert result.diagnostics is not None
    assert result.diagnostics["head_route_solver_attempts"][0]["candidate_kind"] == "baseline"
    assert result.diagnostics["head_route_solver_attempts"][1]["candidate_kind"] == "depleted_gas_refresh"
    selected = result.diagnostics["selected_warm_start_candidate"]
    assert selected["candidate_kind"] == "depleted_gas_refresh"
    assert selected["fastchem4_trace_public_runtime_constructor_inputs_used"] is False


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
        ),
    )

    assert result.status == CONVERGED
    assert result.converged is True
    assert result.selected_route == "m4310_full_promoted_policy_route"
    assert float(result.condensate_amounts[0]) == pytest.approx(1.0e-4)
    assert result.diagnostics is not None
    assert result.diagnostics["restricted_solver_success"] is False
    assert result.diagnostics["solver_success"] is True


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
        options=CondensateEquilibriumOptions(return_diagnostics=True),
    )

    assert result.status == CONVERGED
    assert result.converged is True
    assert result.condensate_support_names == ()
    assert result.condensate_support_indices.shape == (0,)
    assert result.condensate_amounts.shape == (1,)
    assert float(result.condensate_amounts[0]) == 0.0
    assert result.selected_route == "head_v1_empty_positive_support_gas_only"
    assert result.diagnostics is not None
    assert result.diagnostics["support_selection"]["solver_inputs"]["empty_positive_support"] is True


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


def test_api_init_does_not_import_condensate_equilibrium_by_default() -> None:
    sys.modules.pop("exogibbs.api", None)
    sys.modules.pop("exogibbs.api.condensate_equilibrium", None)

    module = importlib.import_module("exogibbs.api")

    assert "exogibbs.api.condensate_equilibrium" not in sys.modules
    resolved = getattr(module, "CondensateEquilibriumOptions")
    assert resolved.__name__ == "CondensateEquilibriumOptions"
    assert "exogibbs.api.condensate_equilibrium" in sys.modules
