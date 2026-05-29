"""Tests for the algorithm-v1.1 thermo-valid diagnostic callsite."""

from __future__ import annotations

import math

import pytest

from exogibbs.diagnostics.condensate_algorithm_v11_callsite import (
    run_algorithm_v11_thermo_valid_reduced_callsite,
)
from exogibbs.optimize.pdipm_rgie_cond import build_pdipm_rgie_condensate_state


def _state():
    return build_pdipm_rgie_condensate_state(
        ln_nk=[math.log(0.8), math.log(0.2)],
        ln_mk=[math.log(1.0e-8), math.log(1.0e-6)],
        element_potential=[0.0, 0.0],
        rho=[math.log(1.0e-5), math.log(1.0e-4)],
        field_provenance={
            "ln_nk": "synthetic_control",
            "ln_mk": "synthetic_control",
            "element_potential": "synthetic_control",
            "rho": "synthetic_control",
            "eta": "synthetic_control",
        },
    )


def test_algorithm_v11_thermo_valid_callsite_filters_and_runs() -> None:
    report = run_algorithm_v11_thermo_valid_reduced_callsite(
        explicit_opt_in=True,
        state=_state(),
        support_indices=[0, 1],
        formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
        formula_matrix_cond_active=[[1.0, 0.0], [0.0, 1.0]],
        element_inventory_target=[0.80000001, 0.200001],
        gas_stationarity_source=[0.1, -0.05],
        condensate_standard_source=[1.0e20, 0.2],
        epsilon=math.log(1.0e-11),
        species_names=["bad", "good"],
        alpha_candidates=[1.0, 0.1, 0.01],
        max_abs_delta_q=1.0e300,
        max_abs_delta_r=1.0e300,
        max_abs_delta_rho=1.0e300,
        max_abs_delta_lambda=1.0e300,
        field_provenance={
            "ln_mk": "synthetic_control",
            "rho": "synthetic_control",
            "eta": "synthetic_control",
        },
    )

    assert report.default_off is True
    assert report.diagnostic_only is True
    assert report.production_behavior_change is False
    assert report.production_return_signature_change is False
    assert report.preset_default_wiring_change is False
    assert report.fastchem4_trace_public_runtime_constructor_inputs_used is False
    assert report.original_support_count == 2
    assert report.filtered_support_count == 1
    assert report.removed_support_count == 1
    assert report.filter_report.removed_species_names == ("bad",)
    assert report.reduced_step_report.finite_trial_step is True
    assert len(report.reduced_step_report.delta_r) == 1


def test_algorithm_v11_thermo_valid_callsite_requires_opt_in() -> None:
    with pytest.raises(ValueError, match="explicit_opt_in"):
        run_algorithm_v11_thermo_valid_reduced_callsite(
            explicit_opt_in=False,
            state=_state(),
            support_indices=[0],
            formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
            formula_matrix_cond_active=[[1.0], [0.0]],
            element_inventory_target=[0.8, 0.2],
            gas_stationarity_source=[0.1, -0.05],
            condensate_standard_source=[0.2],
            epsilon=math.log(1.0e-11),
        )


def test_algorithm_v11_thermo_valid_callsite_rejects_forbidden_provenance() -> None:
    with pytest.raises(ValueError):
        run_algorithm_v11_thermo_valid_reduced_callsite(
            explicit_opt_in=True,
            state=_state(),
            support_indices=[0],
            formula_matrix=[[1.0, 0.0], [0.0, 1.0]],
            formula_matrix_cond_active=[[1.0], [0.0]],
            element_inventory_target=[0.8, 0.2],
            gas_stationarity_source=[0.1, -0.05],
            condensate_standard_source=[0.2],
            epsilon=math.log(1.0e-11),
            field_provenance={"ln_mk": "fastchem4_public"},
        )

