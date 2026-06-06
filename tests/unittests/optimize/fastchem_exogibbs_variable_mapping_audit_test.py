"""Tests for the diagnostic FastChem/ExoGibbs variable-mapping audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

from exogibbs.api.chemistry import ThermoState
import exogibbs.optimize.minimize_cond as condmod


_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "comparisons"
    / "audit_fastchem_exogibbs_variable_mapping.py"
)
_SPEC = importlib.util.spec_from_file_location("audit_fastchem_exogibbs_variable_mapping", _SCRIPT)
audit = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = audit
_SPEC.loader.exec_module(audit)


def test_stage_aligned_trace_bookkeeping_records_update_and_flags():
    records = [
        {
            "record_type": "condensate",
            "stage": "post_calculate_entry_seeding",
            "condensate_index": 1,
            "condensate": "NaOH",
            "log_activity": 0.25,
            "maxDensity": 4.0,
            "number_density_before": 0.0,
            "number_density_after": 1.0,
            "activity_correction_before": 0.0,
            "activity_correction_after": 2.0,
            "newly_active": True,
            "cap_fired": False,
        },
        {
            "record_type": "condensate",
            "stage": "after_first_correctValues_update",
            "condensate_index": 1,
            "condensate": "NaOH",
            "log_activity": 0.1,
            "maxDensity": 4.0,
            "number_density_before": 1.0,
            "number_density_after": 3.0,
            "activity_correction_before": 2.0,
            "activity_correction_after": 6.0,
            "newly_active": True,
            "cap_fired": True,
        },
        {
            "record_type": "condensate",
            "stage": "post_final_removal",
            "condensate_index": 1,
            "condensate": "NaOH",
            "log_activity": -0.2,
            "maxDensity": 4.0,
            "number_density_before": 3.0,
            "number_density_after": 0.0,
            "activity_correction_before": 6.0,
            "activity_correction_after": 0.0,
            "removed": True,
        },
    ]
    exogibbs = {
        "post_calculate_entry_seeding": {
            1: {
                "number_density_exogibbs": 1.0,
                "lambda_exogibbs": 2.0,
                "chi_exogibbs": 0.6931471805599453,
                "newly_active": True,
            }
        },
        "after_first_correctValues_update": {
            1: {
                "number_density_exogibbs": 2.0,
                "lambda_exogibbs": 4.0,
                "chi_exogibbs": 1.3862943611198906,
                "update_factor": 2.0,
                "cap_fired_exogibbs": False,
            }
        },
        "post_final_removal": {1: {"number_density_exogibbs": 0.0, "lambda_exogibbs": 0.0, "removed": True}},
    }

    rows = audit.stage_aligned_rows(
        fastchem_records=records,
        exogibbs_by_stage=exogibbs,
        cond_species=["MgCO3", "NaOH"],
        top_k=4,
    )

    first = rows["after_first_correctValues_update"][0]
    assert first["condensate"] == "NaOH"
    assert first["update_factor_fastchem"] == pytest.approx(3.0)
    assert first["update_factor_exogibbs"] == pytest.approx(2.0)
    assert first["cap_fired_fastchem"] is True
    assert first["cap_fired_exogibbs"] is False
    assert first["stage_value_mismatch_score"] > 0.0
    assert rows["post_final_removal"][0]["removed"] is True


def test_variable_mapping_diagnostic_bookkeeping_prefers_direct_lambda():
    out = audit.mapping_diagnostic_bookkeeping(
        [
            {"activity_correction_fastchem": 2.0, "lambda_exogibbs": 2.0, "chi_exogibbs": 0.6931471805599453},
            {"activity_correction_fastchem": 4.0, "lambda_exogibbs": 4.0, "chi_exogibbs": 1.3862943611198906},
        ]
    )

    assert out["best_mapping"] == "activity_correction~lambda"
    assert out["pair_count"] == 2
    assert out["mean_abs_log_scores"]["activity_correction~lambda"] == pytest.approx(0.0)


def test_variable_mapping_audit_does_not_change_default_production_api(monkeypatch):
    called = {}

    def fake_legacy(*args, **kwargs):
        del args, kwargs
        called["legacy"] = True
        return condmod.CondensateEquilibriumResult(
            ln_nk=jnp.asarray([0.0], dtype=jnp.float64),
            ln_mk=jnp.asarray([-1.0], dtype=jnp.float64),
            ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
            diagnostics=condmod.CondensateEquilibriumDiagnostics(
                n_iter=jnp.asarray(1, dtype=jnp.int32),
                converged=jnp.asarray(True),
                hit_max_iter=jnp.asarray(False),
                final_residual=jnp.asarray(1.0e-12, dtype=jnp.float64),
                residual_crit=jnp.asarray(1.0e-9, dtype=jnp.float64),
                max_iter=jnp.asarray(10, dtype=jnp.int32),
                epsilon=jnp.asarray(-5.0, dtype=jnp.float64),
                final_step_size=jnp.asarray(1.0, dtype=jnp.float64),
                invalid_numbers_detected=jnp.asarray(False),
                debug_nan=jnp.asarray(False),
            ),
        )

    monkeypatch.setattr(condmod, "_minimize_gibbs_cond_legacy", fake_legacy)
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.asarray([1.0], dtype=jnp.float64),
    )

    condmod.minimize_gibbs_cond(
        state,
        init=condmod.CondensateEquilibriumInit(
            ln_nk=jnp.asarray([0.0], dtype=jnp.float64),
            ln_mk=jnp.asarray([-10.0], dtype=jnp.float64),
            ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        ),
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        epsilon=-5.0,
    )

    assert called["legacy"] is True
