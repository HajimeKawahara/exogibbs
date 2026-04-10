import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.pipm_rgie_cond import build_rgie_condensate_init_from_policy
from exogibbs.optimize.pipm_rgie_cond import minimize_gibbs_cond_with_diagnostics


def test_build_rgie_condensate_init_from_policy_absolute_uniform():
    ln_mk = build_rgie_condensate_init_from_policy(
        epsilon=-5.0,
        support_indices=jnp.asarray([3, 7, 11], dtype=jnp.int32),
        startup_policy="absolute_uniform_m0",
        m0=1.0e-20,
    )

    assert ln_mk.shape == (3,)
    assert jnp.allclose(ln_mk, jnp.log(jnp.asarray(1.0e-20, dtype=jnp.float64)))


def test_build_rgie_condensate_init_from_policy_ratio_uniform():
    ln_mk = build_rgie_condensate_init_from_policy(
        epsilon=-10.0,
        support_indices=jnp.asarray([1, 2], dtype=jnp.int32),
        startup_policy="ratio_uniform_r0",
        r0=1.0e-6,
    )

    expected = -10.0 + jnp.log(jnp.asarray(1.0e-6, dtype=jnp.float64))
    assert jnp.allclose(ln_mk, expected)


def test_build_rgie_condensate_init_from_policy_positive_driving_selective():
    ln_mk = build_rgie_condensate_init_from_policy(
        epsilon=-5.0,
        support_indices=jnp.asarray([0, 4, 9], dtype=jnp.int32),
        startup_policy="ratio_positive_driving_r0",
        r0=1.0e-6,
        driving=jnp.asarray([-1.0, 0.2, 0.0], dtype=jnp.float64),
        tiny_fallback=1.0e-30,
    )

    target = -5.0 + jnp.log(jnp.asarray(1.0e-6, dtype=jnp.float64))
    fallback = jnp.log(jnp.asarray(1.0e-30, dtype=jnp.float64))
    assert jnp.allclose(ln_mk, jnp.asarray([fallback, target, fallback], dtype=jnp.float64))


def test_build_rgie_condensate_init_from_policy_topk_positive_driving_selective():
    ln_mk = build_rgie_condensate_init_from_policy(
        epsilon=0.0,
        support_indices=jnp.asarray([2, 5, 8, 13], dtype=jnp.int32),
        startup_policy="ratio_topk_positive_driving_r0",
        r0=1.0e-6,
        top_k=2,
        driving=jnp.asarray([0.4, -0.2, 1.1, 0.7], dtype=jnp.float64),
        tiny_fallback=1.0e-30,
    )

    target = jnp.log(jnp.asarray(1.0e-6, dtype=jnp.float64))
    fallback = jnp.log(jnp.asarray(1.0e-30, dtype=jnp.float64))
    assert jnp.allclose(
        ln_mk,
        jnp.asarray([fallback, fallback, target, target], dtype=jnp.float64),
    )


def test_rgie_startup_policy_helper_is_diagnostic_only(monkeypatch):
    def _should_not_run(*args, **kwargs):
        raise AssertionError("startup policy helper should not be called by the production path")

    monkeypatch.setattr(
        "exogibbs.optimize.pipm_rgie_cond.build_rgie_condensate_init_from_policy",
        _should_not_run,
    )

    formula_matrix = jnp.array([[1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[1.0]], dtype=jnp.float64)
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )

    ln_nk, ln_mk, ln_ntot, diagnostics = minimize_gibbs_cond_with_diagnostics(
        state,
        ln_nk_init=jnp.array([0.0], dtype=jnp.float64),
        ln_mk_init=jnp.array([0.0], dtype=jnp.float64),
        ln_ntot_init=jnp.asarray(0.0, dtype=jnp.float64),
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=lambda temperature: jnp.array([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.array([2.0], dtype=jnp.float64),
        epsilon=-5.0,
        residual_crit=1.0e-8,
        max_iter=0,
    )

    assert ln_nk.shape == (1,)
    assert ln_mk.shape == (1,)
    assert ln_ntot.shape == ()
    assert "n_iter" in diagnostics
