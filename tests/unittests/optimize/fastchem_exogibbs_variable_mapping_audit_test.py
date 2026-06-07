"""Regression tests for the condensate optimizer API boundary."""

from __future__ import annotations

import jax.numpy as jnp

from exogibbs.api.chemistry import ThermoState
import exogibbs.optimize.minimize_cond as condmod


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
