"""Tests for component-safe restricted callsite diagnostics."""

from __future__ import annotations

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import pytest

from exogibbs.api.chemistry import ThermoState
from exogibbs.diagnostics.condensate_component_safe_restricted_callsite import (
    run_component_safe_restricted_callsite_experiment,
)
from exogibbs.diagnostics.condensate_residual_balanced_direction import (
    build_component_safe_curated_policy,
    build_component_safe_policy_payload,
)


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
            "support_indices": "exogibbs_native_component_safe_payload",
            "support_amounts_init": "exogibbs_native_component_safe_payload",
        },
    }


def update_payload():
    policy = build_component_safe_curated_policy(
        case_id="solar_water_condensation__T300_P1",
        native_support_size=1,
        explicit_opt_in=True,
    )
    return build_component_safe_policy_payload(
        policy=policy,
        explicit_opt_in=True,
        delta_ln_nk=(0.0, 0.0, 0.0),
        delta_ln_mk=(0.2,),
        delta_ln_ntot=0.0,
        lambda_trial=0.5,
    )


def skip_payload():
    policy = build_component_safe_curated_policy(
        case_id="solar_highT_no_condensate_gas_regression__T2200_P1",
        native_support_size=0,
        explicit_opt_in=True,
    )
    return build_component_safe_policy_payload(policy=policy, explicit_opt_in=True)


def test_component_safe_restricted_callsite_runs_solver_for_update_payload():
    report = run_component_safe_restricted_callsite_experiment(
        explicit_opt_in=True,
        payload=update_payload(),
        **native_kwargs(),
    )

    assert report.diagnostic_only is True
    assert report.default_off is True
    assert report.explicit_opt_in is True
    assert report.production_behavior_change is False
    assert report.production_return_signature_change is False
    assert report.preset_default_wiring_change is False
    assert report.normal_default_path_unchanged is True
    assert report.fastchem4_trace_public_runtime_constructor_inputs_used is False
    assert report.solver_call_policy == "call_restricted_solver_explicit_opt_in"
    assert report.solver_called is True
    assert report.solver_status is not None
    assert report.post_solver_budget_residual is not None
    assert report.post_solver_kkt_residual_diagnostic is not None
    assert report.support_indices_shape_matches is True
    assert report.support_amounts_init_shape_matches is True
    assert report.finite_solver_inputs is True


def test_component_safe_restricted_callsite_skips_solver_for_classified_skip_payload():
    kwargs = native_kwargs()
    kwargs["support_indices"] = ()
    kwargs["support_amounts_init"] = ()
    report = run_component_safe_restricted_callsite_experiment(
        explicit_opt_in=True,
        payload=skip_payload(),
        **kwargs,
    )

    assert report.solver_call_policy == "skip_solver_classified_skip"
    assert report.solver_called is False
    assert report.seeded_callsite_report is None
    assert report.post_solver_budget_residual is None
    assert report.finite_solver_inputs is True


def test_component_safe_restricted_callsite_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="explicit_opt_in"):
        run_component_safe_restricted_callsite_experiment(
            explicit_opt_in=False,
            payload=update_payload(),
            **native_kwargs(),
        )


def test_component_safe_restricted_callsite_rejects_forbidden_provenance():
    kwargs = native_kwargs()
    kwargs["field_provenance"] = {"support_indices": "fastchem4_public"}
    with pytest.raises(ValueError, match="forbidden|Forbidden"):
        run_component_safe_restricted_callsite_experiment(
            explicit_opt_in=True,
            payload=update_payload(),
            **kwargs,
        )
