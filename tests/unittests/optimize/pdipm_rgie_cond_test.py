"""Tests for explicit experimental PD-IPM R-GIE condensate helpers."""

from __future__ import annotations

import numpy as np
import pytest

from exogibbs.optimize.pdipm_rgie_cond import (
    _stable_l2_norm,
    algorithm_v11_active_support_residual_jacobian,
    audit_algorithm_v11_reduced_direction_against_gie,
    build_pdipm_rgie_dual_carrier_callsite_init,
    build_pdipm_rgie_condensate_state,
    propose_pdipm_rgie_restricted_trial_step,
    run_pdipm_rgie_dual_carrier_solver_step,
    solve_pdipm_rgie_algorithm_v11_reduced_step,
)


def _state_kwargs() -> dict:
    return {
        "ln_nk": [np.log(0.25), np.log(0.75)],
        "ln_mk": [np.log(1.0e-12)],
        "element_potential": [0.1, -0.2],
        "rho": [np.log(1.0e-3)],
        "eta": [1.0e-3],
        "field_provenance": {
            "ln_nk": "exogibbs_native_or_experimental",
            "ln_mk": "exogibbs_native_or_experimental",
            "element_potential": "exogibbs_native_or_experimental",
            "rho": "exogibbs_native_or_experimental",
            "eta": "exogibbs_native_or_experimental",
        },
    }


def test_pdipm_rgie_state_is_default_off_and_explicit() -> None:
    state = build_pdipm_rgie_condensate_state(**_state_kwargs())

    assert state.default_off is True
    assert state.diagnostic_only is True
    assert state.production_behavior_change is False
    assert state.production_return_signature_change is False
    assert state.preset_default_wiring_change is False
    assert state.fastchem4_trace_public_runtime_constructor_inputs_used is False
    assert state.rho is not None
    assert state.eta is not None
    assert state.rho == pytest.approx((np.log(1.0e-3),))
    assert state.eta == pytest.approx((1.0e-3,))


def test_pdipm_rgie_state_infers_log_dual_counterpart() -> None:
    rho_only = build_pdipm_rgie_condensate_state(
        ln_nk=[np.log(0.25)],
        ln_mk=[np.log(1.0e-12)],
        element_potential=[0.0],
        rho=[np.log(1.0e-4)],
    )
    eta_only = build_pdipm_rgie_condensate_state(
        ln_nk=[np.log(0.25)],
        ln_mk=[np.log(1.0e-12)],
        element_potential=[0.0],
        eta=[1.0e-4],
    )

    assert rho_only.eta == pytest.approx((1.0e-4,))
    assert eta_only.rho == pytest.approx((np.log(1.0e-4),))


def test_pdipm_rgie_state_rejects_inconsistent_rho_eta() -> None:
    kwargs = _state_kwargs()
    kwargs["rho"] = [0.0]
    kwargs["eta"] = [1.0e-3]

    with pytest.raises(ValueError, match="log\\(eta\\)"):
        build_pdipm_rgie_condensate_state(**kwargs)


def test_pdipm_rgie_state_rejects_forbidden_provenance() -> None:
    kwargs = _state_kwargs()
    kwargs["field_provenance"] = {"element_potential": "fastchem4_trace"}

    with pytest.raises(ValueError, match="forbidden"):
        build_pdipm_rgie_condensate_state(**kwargs)


def test_stable_l2_norm_keeps_large_finite_components_finite() -> None:
    values = np.asarray([1.0e288, -1.0e288], dtype=np.float64)

    assert not np.isfinite(np.linalg.norm(values))
    assert _stable_l2_norm(values) == pytest.approx(np.sqrt(2.0) * 1.0e288)


def test_algorithm_v11_active_support_jacobian_matches_finite_difference() -> None:
    formula_matrix = [[1.0, 0.0], [0.0, 1.0]]
    formula_matrix_cond_active = [[1.0], [1.0]]
    element_inventory_target = [0.34, 0.47]
    external_condensate_budget = [0.01, 0.02]
    gas_source = [0.12, -0.07]
    condensate_source = [0.31]
    q = np.asarray([np.log(0.31), np.log(0.42)], dtype=np.float64)
    r = np.asarray([np.log(0.025)], dtype=np.float64)
    lam = np.asarray([0.08, -0.11], dtype=np.float64)
    rho = np.asarray([np.log(0.004)], dtype=np.float64)
    qtot = float(np.log(0.74))
    epsilon = float(np.log(1.0e-7))

    residual, jacobian = algorithm_v11_active_support_residual_jacobian(
        formula_matrix=formula_matrix,
        formula_matrix_cond_active=formula_matrix_cond_active,
        element_inventory_target=element_inventory_target,
        external_condensate_budget=external_condensate_budget,
        gas_stationarity_source=gas_source,
        condensate_standard_source=condensate_source,
        q=q,
        r=r,
        lam=lam,
        rho=rho,
        qtot=qtot,
        epsilon=epsilon,
        qtot_reference=qtot,
    )
    base = np.concatenate([q, r, lam, rho, np.asarray([qtot])])

    def residual_at(values: np.ndarray) -> np.ndarray:
        q_next = values[:2]
        r_next = values[2:3]
        lam_next = values[3:5]
        rho_next = values[5:6]
        qtot_next = float(values[6])
        next_residual, _ = algorithm_v11_active_support_residual_jacobian(
            formula_matrix=formula_matrix,
            formula_matrix_cond_active=formula_matrix_cond_active,
            element_inventory_target=element_inventory_target,
            external_condensate_budget=external_condensate_budget,
            gas_stationarity_source=gas_source,
            condensate_standard_source=condensate_source,
            q=q_next,
            r=r_next,
            lam=lam_next,
            rho=rho_next,
            qtot=qtot_next,
            epsilon=epsilon,
            qtot_reference=qtot,
        )
        return next_residual

    fd_jacobian = np.zeros_like(jacobian)
    step = 1.0e-6
    for column in range(base.shape[0]):
        offset = np.zeros_like(base)
        offset[column] = step
        fd_jacobian[:, column] = (residual_at(base + offset) - residual_at(base - offset)) / (
            2.0 * step
        )

    assert residual.shape == (7,)
    assert jacobian.shape == (7, 7)
    assert jacobian == pytest.approx(fd_jacobian, rel=1.0e-7, abs=1.0e-8)
    assert jacobian[-1, -1] == pytest.approx(-np.exp(qtot))


def test_algorithm_v11_reduced_direction_satisfies_gie_linearization() -> None:
    state = build_pdipm_rgie_condensate_state(
        ln_nk=[np.log(0.31), np.log(0.42)],
        ln_mk=[np.log(0.025)],
        element_potential=[0.08, -0.11],
        ln_ntot=float(np.log(0.74)),
        rho=[np.log(0.004)],
        eta=[0.004],
        field_provenance={
            "ln_nk": "synthetic_control",
            "ln_mk": "synthetic_control",
            "element_potential": "synthetic_control",
            "rho": "synthetic_control",
            "eta": "synthetic_control",
        },
    )

    audit = audit_algorithm_v11_reduced_direction_against_gie(
        explicit_opt_in=True,
        state=state,
        formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
        formula_matrix_cond_active=[[1.0], [1.0]],
        element_inventory_target=[0.34, 0.47],
        external_condensate_budget=[0.01, 0.02],
        gas_stationarity_source=[0.12, -0.07],
        condensate_standard_source=[0.31],
        epsilon=float(np.log(1.0e-7)),
    )

    assert audit.default_off is True
    assert audit.diagnostic_only is True
    assert audit.production_behavior_change is False
    assert audit.production_return_signature_change is False
    assert audit.preset_default_wiring_change is False
    assert audit.fastchem4_trace_public_runtime_constructor_inputs_used is False
    assert audit.variable_order == ("q", "r", "lambda", "rho", "qtot")
    assert audit.reduced_linear_system_residual_l2 == pytest.approx(0.0, abs=1.0e-12)
    assert audit.full_linearized_residual_l2 == pytest.approx(0.0, abs=1.0e-12)
    assert audit.full_linearized_residual_max_abs == pytest.approx(0.0, abs=1.0e-12)
    assert audit.clipping_changed_direction is False
    assert audit.clipped_full_linearized_residual_l2 == pytest.approx(0.0, abs=1.0e-12)


def test_algorithm_v11_component_clipping_breaks_gie_newton_linearization() -> None:
    state = build_pdipm_rgie_condensate_state(
        ln_nk=[np.log(0.31), np.log(0.42)],
        ln_mk=[np.log(0.025)],
        element_potential=[0.08, -0.11],
        ln_ntot=float(np.log(0.74)),
        rho=[np.log(0.004)],
        eta=[0.004],
        field_provenance={
            "ln_nk": "synthetic_control",
            "ln_mk": "synthetic_control",
            "element_potential": "synthetic_control",
            "rho": "synthetic_control",
            "eta": "synthetic_control",
        },
    )

    audit = audit_algorithm_v11_reduced_direction_against_gie(
        explicit_opt_in=True,
        state=state,
        formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
        formula_matrix_cond_active=[[1.0], [1.0]],
        element_inventory_target=[0.34, 0.47],
        external_condensate_budget=[0.01, 0.02],
        gas_stationarity_source=[0.12, -0.07],
        condensate_standard_source=[0.31],
        epsilon=float(np.log(1.0e-7)),
        max_abs_delta_q=0.01,
        max_abs_delta_r=0.01,
        max_abs_delta_rho=0.01,
        max_abs_delta_lambda=0.01,
    )

    assert audit.full_linearized_residual_l2 == pytest.approx(0.0, abs=1.0e-12)
    assert audit.clipping_changed_direction is True
    assert audit.clipped_full_linearized_residual_l2 > 1.0e-3
    assert audit.clipped_full_linearized_residual_max_abs > 1.0e-3


def test_algorithm_v11_scalar_fraction_to_boundary_preserves_raw_direction() -> None:
    state = build_pdipm_rgie_condensate_state(
        ln_nk=[np.log(0.31), np.log(0.42)],
        ln_mk=[np.log(0.025)],
        element_potential=[0.08, -0.11],
        ln_ntot=float(np.log(0.74)),
        rho=[np.log(0.004)],
        eta=[0.004],
        field_provenance={
            "ln_nk": "synthetic_control",
            "ln_mk": "synthetic_control",
            "element_potential": "synthetic_control",
            "rho": "synthetic_control",
            "eta": "synthetic_control",
        },
    )
    audit = audit_algorithm_v11_reduced_direction_against_gie(
        explicit_opt_in=True,
        state=state,
        formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
        formula_matrix_cond_active=[[1.0], [1.0]],
        element_inventory_target=[0.34, 0.47],
        external_condensate_budget=[0.01, 0.02],
        gas_stationarity_source=[0.12, -0.07],
        condensate_standard_source=[0.31],
        epsilon=float(np.log(1.0e-7)),
    )

    report = solve_pdipm_rgie_algorithm_v11_reduced_step(
        explicit_opt_in=True,
        state=state,
        formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
        formula_matrix_cond_active=[[1.0], [1.0]],
        element_inventory_target=[0.34, 0.47],
        external_condensate_budget=[0.01, 0.02],
        gas_stationarity_source=[0.12, -0.07],
        condensate_standard_source=[0.31],
        epsilon=float(np.log(1.0e-7)),
        alpha_candidates=[1.0],
        max_abs_delta_q=0.01,
        max_abs_delta_r=0.01,
        max_abs_delta_rho=0.01,
        max_abs_delta_lambda=0.01,
        step_control_policy="scalar_fraction_to_boundary",
        fraction_to_boundary_safety=0.5,
    )

    raw_components = np.concatenate(
        [
            np.asarray(audit.delta_q),
            np.asarray(audit.delta_r),
            np.asarray(audit.delta_lambda),
            np.asarray(audit.delta_rho),
        ]
    )
    assert np.max(np.abs(raw_components)) > 0.01
    assert report.delta_q == pytest.approx(audit.delta_q)
    assert report.delta_r == pytest.approx(audit.delta_r)
    assert report.delta_lambda == pytest.approx(audit.delta_lambda)
    assert report.delta_rho == pytest.approx(audit.delta_rho)


def test_algorithm_v11_reduced_step_is_default_off_and_moves_full_state() -> None:
    state = build_pdipm_rgie_condensate_state(
        ln_nk=[np.log(0.8)],
        ln_mk=[np.log(1.0e-6)],
        element_potential=[0.0],
        rho=[np.log(1.0e-6)],
        eta=[1.0e-6],
        field_provenance={
            "ln_nk": "synthetic_control",
            "ln_mk": "synthetic_control",
            "element_potential": "synthetic_control",
            "rho": "synthetic_control",
            "eta": "synthetic_control",
        },
    )
    report = solve_pdipm_rgie_algorithm_v11_reduced_step(
        explicit_opt_in=True,
        state=state,
        formula_matrix=[[1.0]],
        formula_matrix_cond_active=[[1.0]],
        element_inventory_target=[1.0],
        gas_stationarity_source=[0.0],
        condensate_standard_source=[0.0],
        epsilon=float(np.log(1.0e-12)),
        qhat_regularization=1.0e-12,
    )

    assert report.default_off is True
    assert report.diagnostic_only is True
    assert report.production_behavior_change is False
    assert report.production_return_signature_change is False
    assert report.preset_default_wiring_change is False
    assert report.fastchem4_trace_public_runtime_constructor_inputs_used is False
    assert report.equation_family == "exogibbs_algorithm_v1_1_pdipm_reduced_rgie"
    assert report.finite_trial_step is True
    assert report.trial_step_accepted is True
    assert report.candidate_combined_residual_l2 < report.initial_combined_residual_l2
    assert report.candidate_state.ln_nk != state.ln_nk
    assert report.candidate_state.ln_mk != state.ln_mk
    assert report.candidate_state.element_potential != state.element_potential
    assert report.candidate_state.rho != state.rho


def test_algorithm_v11_reduced_step_includes_external_condensate_budget() -> None:
    state = build_pdipm_rgie_condensate_state(
        ln_nk=[np.log(0.8)],
        ln_mk=[np.log(1.0e-6)],
        element_potential=[0.0],
        rho=[np.log(1.0e-6)],
        eta=[1.0e-6],
        field_provenance={
            "ln_nk": "synthetic_control",
            "ln_mk": "synthetic_control",
            "element_potential": "synthetic_control",
            "rho": "synthetic_control",
            "eta": "synthetic_control",
        },
    )

    report = solve_pdipm_rgie_algorithm_v11_reduced_step(
        explicit_opt_in=True,
        state=state,
        formula_matrix=[[1.0]],
        formula_matrix_cond_active=[[1.0]],
        element_inventory_target=[1.0],
        external_condensate_budget=[0.199999],
        gas_stationarity_source=[0.0],
        condensate_standard_source=[0.0],
        epsilon=float(np.log(1.0e-12)),
        qhat_regularization=1.0e-12,
    )

    assert report.initial_budget_l2 == pytest.approx(0.0, abs=1.0e-14)


def test_algorithm_v11_reduced_step_reports_large_condensate_norm_stably() -> None:
    state = build_pdipm_rgie_condensate_state(
        ln_nk=[np.log(0.8)],
        ln_mk=[np.log(1.0e-300), np.log(1.0e-300)],
        element_potential=[0.0],
        rho=[np.log(1.0e288), np.log(1.0e288)],
        eta=[1.0e288, 1.0e288],
        field_provenance={
            "ln_nk": "synthetic_control",
            "ln_mk": "synthetic_control",
            "element_potential": "synthetic_control",
            "rho": "synthetic_control",
            "eta": "synthetic_control",
        },
    )

    report = solve_pdipm_rgie_algorithm_v11_reduced_step(
        explicit_opt_in=True,
        state=state,
        formula_matrix=[[1.0]],
        formula_matrix_cond_active=[[1.0, 1.0]],
        element_inventory_target=[1.0],
        gas_stationarity_source=[0.0],
        condensate_standard_source=[0.0, 0.0],
        epsilon=float(np.log(1.0e-12)),
        alpha_candidates=[1.0],
        qhat_regularization=1.0e-12,
    )

    assert np.isfinite(report.initial_condensate_stationarity_l2)
    assert report.initial_condensate_stationarity_l2 == pytest.approx(
        np.sqrt(2.0) * 1.0e288
    )
    assert np.isfinite(report.initial_combined_residual_l2)


def test_algorithm_v11_reduced_step_requires_rho() -> None:
    state = build_pdipm_rgie_condensate_state(
        ln_nk=[np.log(0.8)],
        ln_mk=[np.log(1.0e-6)],
        element_potential=[0.0],
        field_provenance={"ln_nk": "synthetic_control"},
    )

    with pytest.raises(ValueError, match="rho"):
        solve_pdipm_rgie_algorithm_v11_reduced_step(
            explicit_opt_in=True,
            state=state,
            formula_matrix=[[1.0]],
            formula_matrix_cond_active=[[1.0]],
            element_inventory_target=[1.0],
            gas_stationarity_source=[0.0],
            condensate_standard_source=[0.0],
            epsilon=float(np.log(1.0e-12)),
        )


def test_algorithm_v11_reduced_step_rejects_forbidden_provenance() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        state = build_pdipm_rgie_condensate_state(
            ln_nk=[np.log(0.8)],
            ln_mk=[np.log(1.0e-6)],
            element_potential=[0.0],
            rho=[np.log(1.0e-6)],
            field_provenance={"rho": "fastchem4_trace"},
        )
        solve_pdipm_rgie_algorithm_v11_reduced_step(
            explicit_opt_in=True,
            state=state,
            formula_matrix=[[1.0]],
            formula_matrix_cond_active=[[1.0]],
            element_inventory_target=[1.0],
            gas_stationarity_source=[0.0],
            condensate_standard_source=[0.0],
            epsilon=float(np.log(1.0e-12)),
        )


def test_algorithm_v11_reduced_step_honors_budget_nonworsening_guard() -> None:
    state = build_pdipm_rgie_condensate_state(
        ln_nk=[np.log(0.99)],
        ln_mk=[np.log(1.0e-6)],
        element_potential=[0.0],
        rho=[np.log(1.0e-6)],
        eta=[1.0e-6],
        field_provenance={
            "ln_nk": "synthetic_control",
            "ln_mk": "synthetic_control",
            "element_potential": "synthetic_control",
            "rho": "synthetic_control",
            "eta": "synthetic_control",
        },
    )
    report = solve_pdipm_rgie_algorithm_v11_reduced_step(
        explicit_opt_in=True,
        state=state,
        formula_matrix=[[1.0]],
        formula_matrix_cond_active=[[1.0]],
        element_inventory_target=[1.0],
        gas_stationarity_source=[0.0],
        condensate_standard_source=[0.0],
        epsilon=float(np.log(1.0e-12)),
        qhat_regularization=1.0e-12,
        require_budget_nonworsening=True,
    )

    assert report.require_budget_nonworsening is True
    assert report.candidate_budget_l2 <= report.initial_budget_l2 + 1.0e-15


def test_algorithm_v11_reduced_step_recovers_pdf_delta_rho_formula() -> None:
    state = build_pdipm_rgie_condensate_state(
        ln_nk=[np.log(0.8)],
        ln_mk=[np.log(1.0e-6)],
        element_potential=[0.0],
        rho=[np.log(2.0)],
        eta=[2.0],
        field_provenance={
            "ln_nk": "synthetic_control",
            "ln_mk": "synthetic_control",
            "element_potential": "synthetic_control",
            "rho": "synthetic_control",
            "eta": "synthetic_control",
        },
    )
    report = solve_pdipm_rgie_algorithm_v11_reduced_step(
        explicit_opt_in=True,
        state=state,
        formula_matrix=[[1.0]],
        formula_matrix_cond_active=[[1.0]],
        element_inventory_target=[1.0],
        gas_stationarity_source=[0.0],
        condensate_standard_source=[3.0],
        epsilon=float(np.log(1.0e-6)),
        alpha_candidates=[1.0],
        max_abs_delta_q=1.0e9,
        max_abs_delta_r=1.0e9,
        max_abs_delta_rho=1.0e9,
        qhat_regularization=1.0e-12,
    )
    pi = np.asarray(report.pi_vector)
    expected_delta_rho = (3.0 - pi[0]) / 2.0 - 1.0

    assert report.delta_rho == pytest.approx((expected_delta_rho,))


def test_algorithm_v11_paired_update_honors_max_density_cap() -> None:
    state = build_pdipm_rgie_condensate_state(
        ln_nk=[np.log(0.8)],
        ln_mk=[np.log(0.5)],
        element_potential=[0.0],
        rho=[0.0],
        eta=[1.0],
        field_provenance={
            "ln_nk": "synthetic_control",
            "ln_mk": "synthetic_control",
            "element_potential": "synthetic_control",
            "rho": "synthetic_control",
            "eta": "synthetic_control",
        },
    )

    report = solve_pdipm_rgie_algorithm_v11_reduced_step(
        explicit_opt_in=True,
        state=state,
        formula_matrix=[[1.0]],
        formula_matrix_cond_active=[[1.0]],
        element_inventory_target=[1.0],
        gas_stationarity_source=[0.0],
        condensate_standard_source=[0.0],
        epsilon=float(np.log(1.0e-15)),
        alpha_candidates=[1.0],
        max_abs_delta_r=5.0,
        max_abs_delta_rho=5.0,
        qhat_regularization=1.0e-12,
        paired_density_activity_update=True,
        max_log_condensate_density=[np.log(0.5)],
    )

    assert report.finite_trial_step is True
    assert report.delta_r[0] <= 0.0
    assert report.delta_rho == pytest.approx((-5.0,))
    assert report.fastchem4_trace_public_runtime_constructor_inputs_used is False


def test_pdipm_rgie_trial_step_improves_combined_residual() -> None:
    state = build_pdipm_rgie_condensate_state(**_state_kwargs())
    report = propose_pdipm_rgie_restricted_trial_step(
        explicit_opt_in=True,
        state=state,
        formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
        formula_matrix_cond_active=[[1.0], [0.0]],
        element_inventory_target=[0.251, 0.75],
        mass_action_constants=[0.0, 0.0],
        hvector_cond_active=[0.1],
        barrier_parameter=1.0e-15,
    )

    assert report.default_off is True
    assert report.diagnostic_only is True
    assert report.production_behavior_change is False
    assert report.production_return_signature_change is False
    assert report.fastchem4_trace_public_runtime_constructor_inputs_used is False
    assert report.finite_trial_step is True
    assert report.candidate_combined_residual_l2 <= report.initial_combined_residual_l2
    assert report.candidate_state.element_potential != state.element_potential


def test_pdipm_rgie_trial_step_requires_opt_in() -> None:
    state = build_pdipm_rgie_condensate_state(**_state_kwargs())

    with pytest.raises(ValueError, match="explicit_opt_in"):
        propose_pdipm_rgie_restricted_trial_step(
            explicit_opt_in=False,
            state=state,
            formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
            formula_matrix_cond_active=[[1.0], [0.0]],
            element_inventory_target=[0.251, 0.75],
            mass_action_constants=[0.0, 0.0],
            hvector_cond_active=[0.1],
        )


def test_pdipm_rgie_trial_step_validates_shapes() -> None:
    state = build_pdipm_rgie_condensate_state(**_state_kwargs())

    with pytest.raises(ValueError, match="element_inventory_target"):
        propose_pdipm_rgie_restricted_trial_step(
            explicit_opt_in=True,
            state=state,
            formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
            formula_matrix_cond_active=[[1.0], [0.0]],
            element_inventory_target=[1.0],
            mass_action_constants=[0.0, 0.0],
            hvector_cond_active=[0.1],
        )


def test_pdipm_rgie_dual_carrier_preserves_lambda_and_rho() -> None:
    state = build_pdipm_rgie_condensate_state(**_state_kwargs())
    carrier = build_pdipm_rgie_dual_carrier_callsite_init(
        explicit_opt_in=True,
        state=state,
        support_indices=[3],
    )

    assert carrier.default_off is True
    assert carrier.diagnostic_only is True
    assert carrier.production_behavior_change is False
    assert carrier.production_return_signature_change is False
    assert carrier.preset_default_wiring_change is False
    assert carrier.fastchem4_trace_public_runtime_constructor_inputs_used is False
    assert carrier.carries_ln_nk is True
    assert carrier.carries_ln_mk is True
    assert carrier.carries_ln_ntot is True
    assert carrier.carries_element_potential is True
    assert carrier.carries_rho is True
    assert carrier.carries_eta is True
    assert carrier.support_amounts_init[0] > 0.0


def test_pdipm_rgie_dual_carrier_validates_support_length() -> None:
    state = build_pdipm_rgie_condensate_state(**_state_kwargs())

    with pytest.raises(ValueError, match="support_indices"):
        build_pdipm_rgie_dual_carrier_callsite_init(
            explicit_opt_in=True,
            state=state,
            support_indices=[3, 4],
        )


def test_pdipm_rgie_dual_carrier_solver_step_runs_without_legacy_init() -> None:
    state = build_pdipm_rgie_condensate_state(**_state_kwargs())
    carrier = build_pdipm_rgie_dual_carrier_callsite_init(
        explicit_opt_in=True,
        state=state,
        support_indices=[3],
    )
    report = run_pdipm_rgie_dual_carrier_solver_step(
        explicit_opt_in=True,
        carrier=carrier,
        formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
        formula_matrix_cond_active=[[1.0], [0.0]],
        element_inventory_target=[0.251, 0.75],
        mass_action_constants=[0.0, 0.0],
        hvector_cond_active=[0.1],
        barrier_parameter=1.0e-15,
    )

    assert report.default_off is True
    assert report.diagnostic_only is True
    assert report.production_behavior_change is False
    assert report.production_return_signature_change is False
    assert report.initial_state.element_potential == state.element_potential
    assert report.finite_trial_step is True
    assert report.candidate_combined_residual_l2 <= report.initial_combined_residual_l2


def test_pdipm_rgie_dual_carrier_solver_step_accepts_budget_merit_policy() -> None:
    state = build_pdipm_rgie_condensate_state(**_state_kwargs())
    carrier = build_pdipm_rgie_dual_carrier_callsite_init(
        explicit_opt_in=True,
        state=state,
        support_indices=[3],
    )
    report = run_pdipm_rgie_dual_carrier_solver_step(
        explicit_opt_in=True,
        carrier=carrier,
        formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
        formula_matrix_cond_active=[[1.0], [0.0]],
        element_inventory_target=[0.251, 0.75],
        mass_action_constants=[0.0, 0.0],
        hvector_cond_active=[0.1],
        barrier_parameter=1.0e-15,
        merit_component_weights={"budget": 10.0},
        require_budget_nonworsening=True,
    )

    assert report.merit_component_weights["budget"] == 10.0
    assert report.require_budget_nonworsening is True
    assert report.candidate_budget_l2 <= report.initial_budget_l2 + 1.0e-15


def test_pdipm_rgie_dual_carrier_solver_step_accepts_budget_rhs_sign_control() -> None:
    state = build_pdipm_rgie_condensate_state(**_state_kwargs())
    carrier = build_pdipm_rgie_dual_carrier_callsite_init(
        explicit_opt_in=True,
        state=state,
        support_indices=[3],
    )
    report = run_pdipm_rgie_dual_carrier_solver_step(
        explicit_opt_in=True,
        carrier=carrier,
        formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
        formula_matrix_cond_active=[[1.0], [0.0]],
        element_inventory_target=[0.251, 0.75],
        mass_action_constants=[0.0, 0.0],
        hvector_cond_active=[0.1],
        barrier_parameter=1.0e-15,
        budget_rhs_sign=-1.0,
    )

    assert report.budget_rhs_sign == -1.0
    assert report.finite_trial_step is True


def test_pdipm_rgie_dual_carrier_solver_step_accepts_linear_system_weights() -> None:
    state = build_pdipm_rgie_condensate_state(**_state_kwargs())
    carrier = build_pdipm_rgie_dual_carrier_callsite_init(
        explicit_opt_in=True,
        state=state,
        support_indices=[3],
    )
    report = run_pdipm_rgie_dual_carrier_solver_step(
        explicit_opt_in=True,
        carrier=carrier,
        formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
        formula_matrix_cond_active=[[1.0], [0.0]],
        element_inventory_target=[0.251, 0.75],
        mass_action_constants=[0.0, 0.0],
        hvector_cond_active=[0.1],
        barrier_parameter=1.0e-15,
        linear_system_component_weights={"budget": 100.0},
    )

    assert report.linear_system_component_weights["budget"] == 100.0
    assert report.finite_trial_step is True


def test_pdipm_rgie_dual_carrier_solver_step_accepts_stationarity_guards() -> None:
    state = build_pdipm_rgie_condensate_state(**_state_kwargs())
    carrier = build_pdipm_rgie_dual_carrier_callsite_init(
        explicit_opt_in=True,
        state=state,
        support_indices=[3],
    )
    report = run_pdipm_rgie_dual_carrier_solver_step(
        explicit_opt_in=True,
        carrier=carrier,
        formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
        formula_matrix_cond_active=[[1.0], [0.0]],
        element_inventory_target=[0.251, 0.75],
        mass_action_constants=[0.0, 0.0],
        hvector_cond_active=[0.1],
        barrier_parameter=1.0e-15,
        max_gas_stationarity_worsening_ratio=2.0,
        max_condensate_stationarity_worsening_ratio=2.0,
    )

    assert report.max_gas_stationarity_worsening_ratio == 2.0
    assert report.max_condensate_stationarity_worsening_ratio == 2.0
    assert report.finite_trial_step is True


def test_pdipm_rgie_solver_step_accepts_fastchem4_style_row_scaling() -> None:
    state = build_pdipm_rgie_condensate_state(**_state_kwargs())
    carrier = build_pdipm_rgie_dual_carrier_callsite_init(
        explicit_opt_in=True,
        state=state,
        support_indices=[3],
    )
    report = run_pdipm_rgie_dual_carrier_solver_step(
        explicit_opt_in=True,
        carrier=carrier,
        formula_matrix=[[1000.0, 0.0], [0.0, 1.0]],
        formula_matrix_cond_active=[[1000.0], [0.0]],
        element_inventory_target=[251.0, 0.75],
        mass_action_constants=[0.0, 0.0],
        hvector_cond_active=[0.1],
        barrier_parameter=1.0e-15,
        linear_system_row_scaling="fastchem4_row_max_abs",
    )

    assert report.linear_system_row_scaling == "fastchem4_row_max_abs"
    assert report.linear_system_row_scale_max > 1.0
    assert report.linear_system_row_scale_min > 0.0
    assert report.finite_trial_step is True
    assert report.production_behavior_change is False
    assert report.production_return_signature_change is False


def test_pdipm_rgie_solver_step_accepts_floor_one_row_scaling() -> None:
    state = build_pdipm_rgie_condensate_state(**_state_kwargs())
    carrier = build_pdipm_rgie_dual_carrier_callsite_init(
        explicit_opt_in=True,
        state=state,
        support_indices=[3],
    )
    report = run_pdipm_rgie_dual_carrier_solver_step(
        explicit_opt_in=True,
        carrier=carrier,
        formula_matrix=[[1000.0, 0.0], [0.0, 1.0]],
        formula_matrix_cond_active=[[1000.0], [0.0]],
        element_inventory_target=[251.0, 0.75],
        mass_action_constants=[0.0, 0.0],
        hvector_cond_active=[0.1],
        barrier_parameter=1.0e-15,
        linear_system_row_scaling="row_max_abs_floor_1",
    )

    assert report.linear_system_row_scaling == "row_max_abs_floor_1"
    assert report.linear_system_row_scale_min >= 1.0
    assert report.linear_system_row_scale_max > 1.0
    assert report.finite_trial_step is True


def test_pdipm_rgie_solver_step_rejects_unknown_row_scaling_policy() -> None:
    state = build_pdipm_rgie_condensate_state(**_state_kwargs())
    carrier = build_pdipm_rgie_dual_carrier_callsite_init(
        explicit_opt_in=True,
        state=state,
        support_indices=[3],
    )

    with pytest.raises(ValueError, match="linear_system_row_scaling"):
        run_pdipm_rgie_dual_carrier_solver_step(
            explicit_opt_in=True,
            carrier=carrier,
            formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
            formula_matrix_cond_active=[[1.0], [0.0]],
            element_inventory_target=[0.251, 0.75],
            mass_action_constants=[0.0, 0.0],
            hvector_cond_active=[0.1],
            barrier_parameter=1.0e-15,
            linear_system_row_scaling="unknown",
        )


def test_pdipm_rgie_solver_step_accepts_dynamic_budget_priority() -> None:
    state = build_pdipm_rgie_condensate_state(**_state_kwargs())
    carrier = build_pdipm_rgie_dual_carrier_callsite_init(
        explicit_opt_in=True,
        state=state,
        support_indices=[3],
    )
    report = run_pdipm_rgie_dual_carrier_solver_step(
        explicit_opt_in=True,
        carrier=carrier,
        formula_matrix=[[1000.0, 0.0], [0.0, 1.0]],
        formula_matrix_cond_active=[[1000.0], [0.0]],
        element_inventory_target=[251.0, 0.75],
        mass_action_constants=[0.0, 0.0],
        hvector_cond_active=[0.1],
        barrier_parameter=1.0e-15,
        linear_system_row_scaling="fastchem4_row_max_abs",
        linear_system_budget_priority_policy="budget_priority_normalized",
        linear_system_budget_priority=10.0,
    )

    assert report.linear_system_budget_priority_policy == "budget_priority_normalized"
    assert report.linear_system_budget_priority == 10.0
    assert report.linear_system_budget_priority_effective_weight > 0.0
    assert report.linear_system_budget_priority_reference_norm > 0.0
    assert report.linear_system_budget_priority_budget_norm > 0.0
    assert report.finite_trial_step is True
    assert report.production_behavior_change is False
    assert report.production_return_signature_change is False


def test_pdipm_rgie_solver_step_rejects_unknown_budget_priority_policy() -> None:
    state = build_pdipm_rgie_condensate_state(**_state_kwargs())
    carrier = build_pdipm_rgie_dual_carrier_callsite_init(
        explicit_opt_in=True,
        state=state,
        support_indices=[3],
    )

    with pytest.raises(ValueError, match="linear_system_budget_priority_policy"):
        run_pdipm_rgie_dual_carrier_solver_step(
            explicit_opt_in=True,
            carrier=carrier,
            formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
            formula_matrix_cond_active=[[1.0], [0.0]],
            element_inventory_target=[0.251, 0.75],
            mass_action_constants=[0.0, 0.0],
            hvector_cond_active=[0.1],
            barrier_parameter=1.0e-15,
            linear_system_budget_priority_policy="unknown",
        )


def test_pdipm_rgie_solver_step_rejects_nonpositive_budget_priority() -> None:
    state = build_pdipm_rgie_condensate_state(**_state_kwargs())
    carrier = build_pdipm_rgie_dual_carrier_callsite_init(
        explicit_opt_in=True,
        state=state,
        support_indices=[3],
    )

    with pytest.raises(ValueError, match="linear_system_budget_priority"):
        run_pdipm_rgie_dual_carrier_solver_step(
            explicit_opt_in=True,
            carrier=carrier,
            formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
            formula_matrix_cond_active=[[1.0], [0.0]],
            element_inventory_target=[0.251, 0.75],
            mass_action_constants=[0.0, 0.0],
            hvector_cond_active=[0.1],
            barrier_parameter=1.0e-15,
            linear_system_budget_priority_policy="budget_priority_normalized",
            linear_system_budget_priority=0.0,
        )
