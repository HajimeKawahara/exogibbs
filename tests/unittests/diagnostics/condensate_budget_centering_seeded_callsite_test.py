"""Tests for explicit budget-centering seeded callsite diagnostics."""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from exogibbs.api.chemistry import ThermoState
from exogibbs.diagnostics.condensate_budget_centering_seeded_callsite import (
    LOWDIM_ALPHA_M,
    LOWDIM_ALPHA_Q,
    build_case_adaptive_lowdim_reduced_coupling_config,
    run_explicit_budget_centering_seeded_callsite,
)
from exogibbs.optimize.minimize_cond import CondensateRGIEReducedCouplingConfig


def toy_state() -> ThermoState:
    return ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(math.log(1.0), dtype=jnp.float64),
        element_vector=jnp.asarray([2.0, 1.0, 5.0], dtype=jnp.float64),
    )


def hvector(_temperature):
    return jnp.zeros((3,), dtype=jnp.float64)


def hvector_cond(_temperature):
    return jnp.asarray([-0.5, -0.1], dtype=jnp.float64)


def native_kwargs():
    return {
        "state": toy_state(),
        "formula_matrix": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "formula_matrix_cond": ((1.0, 1.0), (1.0, 0.0), (3.0, 1.0)),
        "hvector_func": hvector,
        "hvector_cond_func": hvector_cond,
        "condensate_species_order": ("MgSiO3_s", "MgO_s"),
        "element_order": ("Mg", "Si", "O"),
        "support_indices": (0,),
        "support_amounts_init": (1.0e-3,),
        "field_provenance": {
            "support_indices": "exogibbs_native_budget_centering_path",
            "support_amounts_init": "exogibbs_native_budget_centering_path",
        },
    }


def test_case_adaptive_lowdim_config_builds_baseline_variant():
    config = build_case_adaptive_lowdim_reduced_coupling_config(
        "baseline_current_reduced_callsite",
        base_gas_step_scale=0.01,
    )

    assert config.gas_step_scale == pytest.approx(0.01)
    assert config.ntot_step_scale is None
    assert config.condensate_step_scale == pytest.approx(1.0)
    assert config.initial_residual_policy == "computed_fresh"


def test_case_adaptive_lowdim_config_builds_lowdim_variant():
    config = build_case_adaptive_lowdim_reduced_coupling_config(
        "lowdim_step_factor_candidate",
        base_gas_step_scale=0.01,
    )

    assert config.gas_step_scale == pytest.approx(0.01 * LOWDIM_ALPHA_Q)
    assert config.ntot_step_scale == pytest.approx(0.01)
    assert config.condensate_step_scale == pytest.approx(LOWDIM_ALPHA_M)
    assert config.initial_residual_policy == "computed_fresh"


def test_case_adaptive_lowdim_config_rejects_unknown_variant():
    with pytest.raises(ValueError, match="selected_variant"):
        build_case_adaptive_lowdim_reduced_coupling_config("unsupported_variant")


def test_seeded_callsite_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="explicit_opt_in"):
        run_explicit_budget_centering_seeded_callsite(
            explicit_opt_in=False,
            **native_kwargs(),
        )


def test_seeded_callsite_runs_default_off():
    report = run_explicit_budget_centering_seeded_callsite(
        explicit_opt_in=True,
        **native_kwargs(),
    )

    assert report.diagnostic_only is True
    assert report.default_off is True
    assert report.explicit_opt_in is True
    assert report.production_behavior_change is False
    assert report.production_return_signature_change is False
    assert report.preset_default_wiring_change is False
    assert report.support_indices_shape_matches is True
    assert report.support_amounts_init_shape_matches is True
    assert report.finite_solver_inputs is True
    assert report.solver_called is True
    assert report.post_solver_budget_residual is not None
    assert report.post_solver_kkt_residual_diagnostic is not None
    assert report.line_search_selection_policy == "first_monotone_with_best_finite_fallback"
    assert report.fastchem4_trace_values_used is False
    assert report.fastchem4_public_values_used_as_constructor_inputs is False


def test_seeded_callsite_accepts_explicit_best_trial_line_search_policy():
    report = run_explicit_budget_centering_seeded_callsite(
        explicit_opt_in=True,
        line_search_selection_policy="best_finite_nonincreasing",
        **native_kwargs(),
    )

    assert report.line_search_selection_policy == "best_finite_nonincreasing"
    assert report.solver_called is True
    assert report.production_behavior_change is False


def test_seeded_callsite_accepts_explicit_charge_aware_line_search_policy():
    report = run_explicit_budget_centering_seeded_callsite(
        explicit_opt_in=True,
        line_search_selection_policy="charge_aware_composite_nonincreasing",
        line_search_charge_row_name="Mg",
        line_search_charge_weight=1.0,
        **native_kwargs(),
    )

    assert report.line_search_selection_policy == "charge_aware_composite_nonincreasing"
    assert report.solver_called is True
    assert report.production_behavior_change is False


def test_seeded_callsite_accepts_explicit_reduced_coupling_config():
    report = run_explicit_budget_centering_seeded_callsite(
        explicit_opt_in=True,
        reduced_coupling_config=CondensateRGIEReducedCouplingConfig(
            reduced_coupling_mode="capped_s_only_fixed_alpha",
            alpha_s=0.1,
            gas_step_scale=0.01,
            gas_step_direction_sign=-1.0,
            ntot_step_scale=0.02,
            condensate_step_scale=0.5,
            initial_residual_policy="computed_fresh",
        ),
        **native_kwargs(),
    )

    assert report.reduced_coupling_config_mode == "capped_s_only_fixed_alpha"
    assert report.reduced_coupling_selected_mode == "capped_s_only"
    assert report.reduced_coupling_selected_alpha_s == pytest.approx(0.1)
    assert report.gas_step_scale == pytest.approx(0.01)
    assert report.gas_step_direction_sign == pytest.approx(-1.0)
    assert report.ntot_step_scale == pytest.approx(0.02)
    assert report.condensate_step_scale == pytest.approx(0.5)
    assert report.initial_residual_policy == "computed_fresh"
    assert report.production_behavior_change is False


def test_reduced_coupling_config_rejects_invalid_gas_step_scale():
    with pytest.raises(ValueError, match="gas_step_scale"):
        run_explicit_budget_centering_seeded_callsite(
            explicit_opt_in=True,
            reduced_coupling_config=CondensateRGIEReducedCouplingConfig(
                gas_step_scale=0.0,
            ),
            **native_kwargs(),
        )


def test_reduced_coupling_config_rejects_invalid_ntot_step_scale():
    with pytest.raises(ValueError, match="ntot_step_scale"):
        run_explicit_budget_centering_seeded_callsite(
            explicit_opt_in=True,
            reduced_coupling_config=CondensateRGIEReducedCouplingConfig(
                ntot_step_scale=0.0,
            ),
            **native_kwargs(),
        )


def test_reduced_coupling_config_rejects_invalid_condensate_step_scale():
    with pytest.raises(ValueError, match="condensate_step_scale"):
        run_explicit_budget_centering_seeded_callsite(
            explicit_opt_in=True,
            reduced_coupling_config=CondensateRGIEReducedCouplingConfig(
                condensate_step_scale=0.0,
            ),
            **native_kwargs(),
        )


def test_reduced_coupling_config_rejects_invalid_initial_residual_policy():
    with pytest.raises(ValueError, match="initial_residual_policy"):
        run_explicit_budget_centering_seeded_callsite(
            explicit_opt_in=True,
            reduced_coupling_config=CondensateRGIEReducedCouplingConfig(
                initial_residual_policy="trace_seeded",
            ),
            **native_kwargs(),
        )


def test_reduced_coupling_config_rejects_invalid_gas_step_direction_sign():
    with pytest.raises(ValueError, match="gas_step_direction_sign"):
        run_explicit_budget_centering_seeded_callsite(
            explicit_opt_in=True,
            reduced_coupling_config=CondensateRGIEReducedCouplingConfig(
                gas_step_direction_sign=0.5,
            ),
            **native_kwargs(),
        )


def test_seeded_callsite_reports_inactive_driving_candidates():
    report = run_explicit_budget_centering_seeded_callsite(
        explicit_opt_in=True,
        **native_kwargs(),
    )

    assert report.inactive_positive_count is not None
    assert isinstance(report.top_positive_inactive_indices, tuple)
    assert isinstance(report.top_inactive_names, tuple)
    assert isinstance(report.top_inactive_driving, tuple)


def test_seeded_callsite_handles_empty_support_boundary():
    kwargs = native_kwargs()
    kwargs["support_indices"] = ()
    kwargs["support_amounts_init"] = ()
    report = run_explicit_budget_centering_seeded_callsite(
        explicit_opt_in=True,
        **kwargs,
    )

    assert report.solver_called is False
    assert report.solver_success is None
    assert report.support_size == 0


def test_seeded_callsite_rejects_shape_mismatch():
    kwargs = native_kwargs()
    kwargs["support_amounts_init"] = (1.0e-3, 2.0e-3)

    with pytest.raises(ValueError, match="matching length"):
        run_explicit_budget_centering_seeded_callsite(
            explicit_opt_in=True,
            **kwargs,
        )


def test_seeded_callsite_rejects_forbidden_provenance():
    kwargs = native_kwargs()
    kwargs["field_provenance"] = {"support_amounts_init": "fastchem4_public"}

    with pytest.raises(ValueError, match="provenance is forbidden"):
        run_explicit_budget_centering_seeded_callsite(
            explicit_opt_in=True,
            **kwargs,
        )


def test_seeded_callsite_rejects_budget_fraction_over_limit():
    kwargs = native_kwargs()
    kwargs["support_amounts_init"] = (10.0,)

    with pytest.raises(ValueError, match="max_budget_fraction"):
        run_explicit_budget_centering_seeded_callsite(
            explicit_opt_in=True,
            max_budget_fraction=0.1,
            **kwargs,
        )
