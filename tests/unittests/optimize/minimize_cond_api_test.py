import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from exogibbs.api.chemistry import ThermoState
import exogibbs.optimize.minimize_cond as condmod
from exogibbs.optimize.pipm_rgie_cond import minimize_gibbs_cond_with_diagnostics as raw_minimize_gibbs_cond_with_diagnostics


def test_minimize_gibbs_cond_structured_wrapper(monkeypatch):
    captured = {}

    def stub_raw(state, ln_nk_init, ln_mk_init, ln_ntot_init, **kwargs):
        captured["ln_nk_init"] = ln_nk_init
        captured["ln_mk_init"] = ln_mk_init
        captured["ln_ntot_init"] = ln_ntot_init
        captured["epsilon"] = kwargs["epsilon"]
        return (
            jnp.asarray([0.1, 0.2], dtype=jnp.float64),
            jnp.asarray([0.3], dtype=jnp.float64),
            jnp.asarray(1.7, dtype=jnp.float64),
            {
                "n_iter": jnp.asarray(7, dtype=jnp.int32),
                "converged": jnp.asarray(True),
                "hit_max_iter": jnp.asarray(False),
                "final_residual": jnp.asarray(1.0e-12, dtype=jnp.float64),
                "residual_crit": jnp.asarray(kwargs["residual_crit"], dtype=jnp.float64),
                "max_iter": jnp.asarray(kwargs["max_iter"], dtype=jnp.int32),
                "epsilon": jnp.asarray(kwargs["epsilon"], dtype=jnp.float64),
                "final_step_size": jnp.asarray(0.5, dtype=jnp.float64),
                "invalid_numbers_detected": jnp.asarray(False),
                "debug_nan": jnp.asarray(kwargs["debug_nan"]),
            },
        )

    monkeypatch.setattr(
        condmod,
        "_minimize_gibbs_cond_with_diagnostics_raw",
        stub_raw,
    )

    init = condmod.CondensateEquilibriumInit(
        ln_nk=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        ln_mk=jnp.asarray([3.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(4.0, dtype=jnp.float64),
    )
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.asarray([1.0], dtype=jnp.float64),
    )

    result = condmod.minimize_gibbs_cond(
        state,
        init=init,
        formula_matrix=jnp.asarray([[1.0, 1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        epsilon=-6.0,
        residual_crit=1.0e-9,
        max_iter=25,
        debug_nan=False,
    )

    assert jnp.allclose(captured["ln_nk_init"], init.ln_nk)
    assert jnp.allclose(captured["ln_mk_init"], init.ln_mk)
    assert jnp.allclose(captured["ln_ntot_init"], init.ln_ntot)
    assert captured["epsilon"] == -6.0

    assert isinstance(result, condmod.CondensateEquilibriumResult)
    assert isinstance(result.diagnostics, condmod.CondensateEquilibriumDiagnostics)
    assert result.ln_nk.shape == (2,)
    assert result.ln_mk.shape == (1,)
    assert result.ln_ntot.shape == ()
    assert int(result.diagnostics.n_iter) == 7
    assert bool(result.diagnostics.converged)
    assert not bool(result.diagnostics.hit_max_iter)


def test_condensate_result_to_init_roundtrip():
    diagnostics = condmod.CondensateEquilibriumDiagnostics(
        n_iter=jnp.asarray(3, dtype=jnp.int32),
        converged=jnp.asarray(True),
        hit_max_iter=jnp.asarray(False),
        final_residual=jnp.asarray(1.0e-12, dtype=jnp.float64),
        residual_crit=jnp.asarray(1.0e-10, dtype=jnp.float64),
        max_iter=jnp.asarray(100, dtype=jnp.int32),
        epsilon=jnp.asarray(-5.0, dtype=jnp.float64),
        final_step_size=jnp.asarray(0.25, dtype=jnp.float64),
        invalid_numbers_detected=jnp.asarray(False),
        debug_nan=jnp.asarray(False),
    )
    result = condmod.CondensateEquilibriumResult(
        ln_nk=jnp.asarray([0.1, 0.2], dtype=jnp.float64),
        ln_mk=jnp.asarray([0.3], dtype=jnp.float64),
        ln_ntot=jnp.asarray(0.4, dtype=jnp.float64),
        diagnostics=diagnostics,
    )

    init = result.to_init()

    assert isinstance(init, condmod.CondensateEquilibriumInit)
    assert jnp.allclose(init.ln_nk, result.ln_nk)
    assert jnp.allclose(init.ln_mk, result.ln_mk)
    assert jnp.allclose(init.ln_ntot, result.ln_ntot)


def test_minimize_gibbs_cond_structured_smoke():
    formula_matrix = jnp.array([[1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[1.0]], dtype=jnp.float64)
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )
    init = condmod.CondensateEquilibriumInit(
        ln_nk=jnp.array([0.0], dtype=jnp.float64),
        ln_mk=jnp.array([0.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
    )

    result = condmod.minimize_gibbs_cond_with_diagnostics(
        state,
        init=init,
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=lambda temperature: jnp.array([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.array([2.0], dtype=jnp.float64),
        epsilon=-5.0,
        residual_crit=1.0e-8,
        max_iter=0,
    )

    assert isinstance(result, condmod.CondensateEquilibriumResult)
    assert result.ln_nk.shape == (1,)
    assert result.ln_mk.shape == (1,)
    assert result.ln_ntot.shape == ()
    assert int(result.diagnostics.n_iter) == 0
    assert bool(result.diagnostics.hit_max_iter)
    assert not bool(result.diagnostics.converged)


def test_raw_phase0_api_still_available():
    formula_matrix = jnp.array([[1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[1.0]], dtype=jnp.float64)
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )

    ln_nk, ln_mk, ln_ntot, diagnostics = raw_minimize_gibbs_cond_with_diagnostics(
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
    assert isinstance(diagnostics, dict)
    assert "n_iter" in diagnostics


def test_minimize_gibbs_cond_profile_scan_hot_from_bottom_carries_structured_state(monkeypatch):
    def stub_minimize_gibbs_cond(state, init, **kwargs):
        return condmod.CondensateEquilibriumResult(
            ln_nk=jnp.asarray(init.ln_nk) + 1.0,
            ln_mk=jnp.asarray(init.ln_mk) + 2.0,
            ln_ntot=jnp.asarray(init.ln_ntot) + 3.0,
            diagnostics=condmod.CondensateEquilibriumDiagnostics(
                n_iter=jnp.asarray(4, dtype=jnp.int32),
                converged=jnp.asarray(True),
                hit_max_iter=jnp.asarray(False),
                final_residual=jnp.asarray(1.0e-12, dtype=jnp.float64),
                residual_crit=jnp.asarray(kwargs["residual_crit"], dtype=jnp.float64),
                max_iter=jnp.asarray(kwargs["max_iter"], dtype=jnp.int32),
                epsilon=jnp.asarray(kwargs["epsilon"], dtype=jnp.float64),
                final_step_size=jnp.asarray(0.5, dtype=jnp.float64),
                invalid_numbers_detected=jnp.asarray(False),
                debug_nan=jnp.asarray(kwargs["debug_nan"]),
            ),
        )

    monkeypatch.setattr(condmod, "minimize_gibbs_cond", stub_minimize_gibbs_cond)

    init = condmod.CondensateEquilibriumInit(
        ln_nk=jnp.asarray([[10.0], [20.0], [30.0]], dtype=jnp.float64),
        ln_mk=jnp.asarray([[1.0], [2.0], [3.0]], dtype=jnp.float64),
        ln_ntot=jnp.asarray([100.0, 200.0, 300.0], dtype=jnp.float64),
    )

    result = condmod.minimize_gibbs_cond_profile(
        temperatures=jnp.asarray([1000.0, 1100.0, 1200.0], dtype=jnp.float64),
        ln_normalized_pressures=jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float64),
        element_vector=jnp.asarray([1.0], dtype=jnp.float64),
        init=init,
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        n_step=1,
        max_iter=25,
        method="scan_hot_from_bottom",
    )

    # Each layer runs one scheduled step plus one final epsilon_crit solve.
    assert jnp.allclose(result.ln_nk[:, 0], jnp.asarray([36.0, 34.0, 32.0], dtype=jnp.float64))
    assert jnp.allclose(result.ln_mk[:, 0], jnp.asarray([15.0, 11.0, 7.0], dtype=jnp.float64))
    assert jnp.allclose(result.ln_ntot, jnp.asarray([318.0, 312.0, 306.0], dtype=jnp.float64))
    assert result.diagnostics.n_iter.shape == (3,)
    assert jnp.all(result.diagnostics.converged)


def test_minimize_gibbs_cond_profile_scan_hot_from_top_runs_in_input_order(monkeypatch):
    def stub_minimize_gibbs_cond(state, init, **kwargs):
        return condmod.CondensateEquilibriumResult(
            ln_nk=jnp.asarray(init.ln_nk) + 1.0,
            ln_mk=jnp.asarray(init.ln_mk) + 2.0,
            ln_ntot=jnp.asarray(init.ln_ntot) + 3.0,
            diagnostics=condmod.CondensateEquilibriumDiagnostics(
                n_iter=jnp.asarray(4, dtype=jnp.int32),
                converged=jnp.asarray(True),
                hit_max_iter=jnp.asarray(False),
                final_residual=jnp.asarray(1.0e-12, dtype=jnp.float64),
                residual_crit=jnp.asarray(kwargs["residual_crit"], dtype=jnp.float64),
                max_iter=jnp.asarray(kwargs["max_iter"], dtype=jnp.int32),
                epsilon=jnp.asarray(kwargs["epsilon"], dtype=jnp.float64),
                final_step_size=jnp.asarray(0.5, dtype=jnp.float64),
                invalid_numbers_detected=jnp.asarray(False),
                debug_nan=jnp.asarray(kwargs["debug_nan"]),
            ),
        )

    monkeypatch.setattr(condmod, "minimize_gibbs_cond", stub_minimize_gibbs_cond)

    result = condmod.minimize_gibbs_cond_profile(
        temperatures=jnp.asarray([1000.0, 1100.0, 1200.0], dtype=jnp.float64),
        ln_normalized_pressures=jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float64),
        element_vector=jnp.asarray([1.0], dtype=jnp.float64),
        init=condmod.CondensateEquilibriumInit(
            ln_nk=jnp.asarray([[10.0], [20.0], [30.0]], dtype=jnp.float64),
            ln_mk=jnp.asarray([[1.0], [2.0], [3.0]], dtype=jnp.float64),
            ln_ntot=jnp.asarray([100.0, 200.0, 300.0], dtype=jnp.float64),
        ),
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        n_step=1,
        max_iter=25,
        method="scan_hot_from_top",
    )

    # Output order remains the same as the input profile order.
    assert jnp.allclose(result.ln_nk[:, 0], jnp.asarray([12.0, 14.0, 16.0], dtype=jnp.float64))
    assert jnp.allclose(result.ln_mk[:, 0], jnp.asarray([5.0, 9.0, 13.0], dtype=jnp.float64))
    assert jnp.allclose(result.ln_ntot, jnp.asarray([106.0, 112.0, 118.0], dtype=jnp.float64))
    assert result.diagnostics.n_iter.shape == (3,)
    assert result.diagnostics.final_residual.shape == (3,)
    assert result.diagnostics.epsilon.shape == (3,)
    assert jnp.all(result.diagnostics.converged)


def test_minimize_gibbs_cond_profile_vmap_cold_still_available(monkeypatch):
    def stub_minimize_gibbs_cond(state, init, **kwargs):
        return condmod.CondensateEquilibriumResult(
            ln_nk=jnp.asarray(init.ln_nk) + 1.0,
            ln_mk=jnp.asarray(init.ln_mk) + 1.0,
            ln_ntot=jnp.asarray(init.ln_ntot) + 1.0,
            diagnostics=condmod.CondensateEquilibriumDiagnostics(
                n_iter=jnp.asarray(2, dtype=jnp.int32),
                converged=jnp.asarray(True),
                hit_max_iter=jnp.asarray(False),
                final_residual=jnp.asarray(1.0e-12, dtype=jnp.float64),
                residual_crit=jnp.asarray(kwargs["residual_crit"], dtype=jnp.float64),
                max_iter=jnp.asarray(kwargs["max_iter"], dtype=jnp.int32),
                epsilon=jnp.asarray(kwargs["epsilon"], dtype=jnp.float64),
                final_step_size=jnp.asarray(0.25, dtype=jnp.float64),
                invalid_numbers_detected=jnp.asarray(False),
                debug_nan=jnp.asarray(kwargs["debug_nan"]),
            ),
        )

    monkeypatch.setattr(condmod, "minimize_gibbs_cond", stub_minimize_gibbs_cond)

    result = condmod.minimize_gibbs_cond_profile(
        temperatures=jnp.asarray([1000.0, 1100.0, 1200.0], dtype=jnp.float64),
        ln_normalized_pressures=jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float64),
        element_vector=jnp.asarray([1.0], dtype=jnp.float64),
        init=condmod.CondensateEquilibriumInit(
            ln_nk=jnp.asarray([[10.0], [20.0], [30.0]], dtype=jnp.float64),
            ln_mk=jnp.asarray([[1.0], [2.0], [3.0]], dtype=jnp.float64),
            ln_ntot=jnp.asarray([100.0, 200.0, 300.0], dtype=jnp.float64),
        ),
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        n_step=1,
        max_iter=25,
        method="vmap_cold",
    )

    assert jnp.allclose(result.ln_nk[:, 0], jnp.asarray([12.0, 22.0, 32.0], dtype=jnp.float64))
    assert jnp.allclose(result.ln_mk[:, 0], jnp.asarray([3.0, 4.0, 5.0], dtype=jnp.float64))
    assert jnp.allclose(result.ln_ntot, jnp.asarray([102.0, 202.0, 302.0], dtype=jnp.float64))


def test_minimize_gibbs_cond_profile_scan_hot_from_top_final_only_skips_rewind_after_first_layer(monkeypatch):
    def stub_minimize_gibbs_cond(state, init, **kwargs):
        return condmod.CondensateEquilibriumResult(
            ln_nk=jnp.asarray(init.ln_nk) + 1.0,
            ln_mk=jnp.asarray(init.ln_mk) + 2.0,
            ln_ntot=jnp.asarray(init.ln_ntot) + 3.0,
            diagnostics=condmod.CondensateEquilibriumDiagnostics(
                n_iter=jnp.asarray(4, dtype=jnp.int32),
                converged=jnp.asarray(True),
                hit_max_iter=jnp.asarray(False),
                final_residual=jnp.asarray(1.0e-12, dtype=jnp.float64),
                residual_crit=jnp.asarray(kwargs["residual_crit"], dtype=jnp.float64),
                max_iter=jnp.asarray(kwargs["max_iter"], dtype=jnp.int32),
                epsilon=jnp.asarray(kwargs["epsilon"], dtype=jnp.float64),
                final_step_size=jnp.asarray(0.5, dtype=jnp.float64),
                invalid_numbers_detected=jnp.asarray(False),
                debug_nan=jnp.asarray(kwargs["debug_nan"]),
            ),
        )

    monkeypatch.setattr(condmod, "minimize_gibbs_cond", stub_minimize_gibbs_cond)

    result = condmod.minimize_gibbs_cond_profile(
        temperatures=jnp.asarray([1000.0, 1100.0, 1200.0], dtype=jnp.float64),
        ln_normalized_pressures=jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float64),
        element_vector=jnp.asarray([1.0], dtype=jnp.float64),
        init=condmod.CondensateEquilibriumInit(
            ln_nk=jnp.asarray([[10.0], [20.0], [30.0]], dtype=jnp.float64),
            ln_mk=jnp.asarray([[1.0], [2.0], [3.0]], dtype=jnp.float64),
            ln_ntot=jnp.asarray([100.0, 200.0, 300.0], dtype=jnp.float64),
        ),
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        n_step=1,
        max_iter=25,
        method="scan_hot_from_top_final_only",
    )

    assert jnp.allclose(result.ln_nk[:, 0], jnp.asarray([12.0, 13.0, 14.0], dtype=jnp.float64))
    assert jnp.allclose(result.ln_mk[:, 0], jnp.asarray([5.0, 7.0, 9.0], dtype=jnp.float64))
    assert jnp.allclose(result.ln_ntot, jnp.asarray([106.0, 109.0, 112.0], dtype=jnp.float64))
    assert jnp.all(result.diagnostics.converged)


def test_minimize_gibbs_cond_profile_scan_hot_from_bottom_final_only_skips_rewind_after_first_layer(monkeypatch):
    def stub_minimize_gibbs_cond(state, init, **kwargs):
        return condmod.CondensateEquilibriumResult(
            ln_nk=jnp.asarray(init.ln_nk) + 1.0,
            ln_mk=jnp.asarray(init.ln_mk) + 2.0,
            ln_ntot=jnp.asarray(init.ln_ntot) + 3.0,
            diagnostics=condmod.CondensateEquilibriumDiagnostics(
                n_iter=jnp.asarray(4, dtype=jnp.int32),
                converged=jnp.asarray(True),
                hit_max_iter=jnp.asarray(False),
                final_residual=jnp.asarray(1.0e-12, dtype=jnp.float64),
                residual_crit=jnp.asarray(kwargs["residual_crit"], dtype=jnp.float64),
                max_iter=jnp.asarray(kwargs["max_iter"], dtype=jnp.int32),
                epsilon=jnp.asarray(kwargs["epsilon"], dtype=jnp.float64),
                final_step_size=jnp.asarray(0.5, dtype=jnp.float64),
                invalid_numbers_detected=jnp.asarray(False),
                debug_nan=jnp.asarray(kwargs["debug_nan"]),
            ),
        )

    monkeypatch.setattr(condmod, "minimize_gibbs_cond", stub_minimize_gibbs_cond)

    result = condmod.minimize_gibbs_cond_profile(
        temperatures=jnp.asarray([1000.0, 1100.0, 1200.0], dtype=jnp.float64),
        ln_normalized_pressures=jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float64),
        element_vector=jnp.asarray([1.0], dtype=jnp.float64),
        init=condmod.CondensateEquilibriumInit(
            ln_nk=jnp.asarray([[10.0], [20.0], [30.0]], dtype=jnp.float64),
            ln_mk=jnp.asarray([[1.0], [2.0], [3.0]], dtype=jnp.float64),
            ln_ntot=jnp.asarray([100.0, 200.0, 300.0], dtype=jnp.float64),
        ),
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        n_step=1,
        max_iter=25,
        method="scan_hot_from_bottom_final_only",
    )

    assert jnp.allclose(result.ln_nk[:, 0], jnp.asarray([34.0, 33.0, 32.0], dtype=jnp.float64))
    assert jnp.allclose(result.ln_mk[:, 0], jnp.asarray([11.0, 9.0, 7.0], dtype=jnp.float64))
    assert jnp.allclose(result.ln_ntot, jnp.asarray([312.0, 309.0, 306.0], dtype=jnp.float64))
    assert jnp.all(result.diagnostics.converged)


def test_minimize_gibbs_cond_profile_broadcasts_single_cold_start(monkeypatch):
    def stub_minimize_gibbs_cond(state, init, **kwargs):
        return condmod.CondensateEquilibriumResult(
            ln_nk=jnp.asarray(init.ln_nk),
            ln_mk=jnp.asarray(init.ln_mk),
            ln_ntot=jnp.asarray(init.ln_ntot),
            diagnostics=condmod.CondensateEquilibriumDiagnostics(
                n_iter=jnp.asarray(1, dtype=jnp.int32),
                converged=jnp.asarray(True),
                hit_max_iter=jnp.asarray(False),
                final_residual=jnp.asarray(1.0e-12, dtype=jnp.float64),
                residual_crit=jnp.asarray(kwargs["residual_crit"], dtype=jnp.float64),
                max_iter=jnp.asarray(kwargs["max_iter"], dtype=jnp.int32),
                epsilon=jnp.asarray(kwargs["epsilon"], dtype=jnp.float64),
                final_step_size=jnp.asarray(1.0, dtype=jnp.float64),
                invalid_numbers_detected=jnp.asarray(False),
                debug_nan=jnp.asarray(kwargs["debug_nan"]),
            ),
        )

    monkeypatch.setattr(condmod, "minimize_gibbs_cond", stub_minimize_gibbs_cond)

    result = condmod.minimize_gibbs_cond_profile(
        temperatures=jnp.asarray([1000.0, 1100.0], dtype=jnp.float64),
        ln_normalized_pressures=jnp.asarray([-1.0, 0.0], dtype=jnp.float64),
        element_vector=jnp.asarray([1.0], dtype=jnp.float64),
        init=condmod.CondensateEquilibriumInit(
            ln_nk=jnp.asarray([5.0], dtype=jnp.float64),
            ln_mk=jnp.asarray([6.0], dtype=jnp.float64),
            ln_ntot=jnp.asarray(7.0, dtype=jnp.float64),
        ),
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        n_step=1,
        method="vmap_cold",
    )

    assert result.ln_nk.shape == (2, 1)
    assert result.ln_mk.shape == (2, 1)
    assert result.ln_ntot.shape == (2,)
    assert jnp.allclose(result.ln_nk[:, 0], 5.0)
    assert jnp.allclose(result.ln_mk[:, 0], 6.0)
    assert jnp.allclose(result.ln_ntot, 7.0)


def test_trace_adaptive_condensate_schedule_reports_guard_and_plateau(monkeypatch):
    def stub_minimize_gibbs_cond(state, init, **kwargs):
        return condmod.CondensateEquilibriumResult(
            ln_nk=jnp.asarray(init.ln_nk),
            ln_mk=jnp.asarray(init.ln_mk),
            ln_ntot=jnp.asarray(init.ln_ntot),
            diagnostics=condmod.CondensateEquilibriumDiagnostics(
                n_iter=jnp.asarray(1, dtype=jnp.int32),
                converged=jnp.asarray(False),
                hit_max_iter=jnp.asarray(False),
                final_residual=jnp.asarray(1.0, dtype=jnp.float64),
                residual_crit=jnp.asarray(kwargs["residual_crit"], dtype=jnp.float64),
                max_iter=jnp.asarray(kwargs["max_iter"], dtype=jnp.int32),
                epsilon=jnp.asarray(kwargs["epsilon"], dtype=jnp.float64),
                final_step_size=jnp.asarray(0.1, dtype=jnp.float64),
                invalid_numbers_detected=jnp.asarray(False),
                debug_nan=jnp.asarray(kwargs["debug_nan"]),
            ),
        )

    monkeypatch.setattr(condmod, "minimize_gibbs_cond", stub_minimize_gibbs_cond)

    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.asarray([1.0], dtype=jnp.float64),
    )
    trace = condmod.trace_adaptive_condensate_schedule(
        state,
        init=condmod.CondensateEquilibriumInit(
            ln_nk=jnp.asarray([0.0], dtype=jnp.float64),
            ln_mk=jnp.asarray([6.8], dtype=jnp.float64),
            ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        ),
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        epsilon_start=0.0,
        epsilon_crit=-1.0,
        n_step=1,
        max_iter=1,
        condensate_species=["guarded_cond"],
    )

    assert trace["stages"][0]["stage_kind"] == "sk-guard-limited"
    assert trace["stages"][1]["stage_kind"] == "plateau-stopped"
    assert not trace["reached_requested_epsilon"]
    assert trace["plateaued"]


def test_minimize_gibbs_cond_profile_adaptive_reports_actual_epsilon(monkeypatch):
    def stub_minimize_gibbs_cond(state, init, **kwargs):
        return condmod.CondensateEquilibriumResult(
            ln_nk=jnp.asarray(init.ln_nk),
            ln_mk=jnp.asarray(init.ln_mk),
            ln_ntot=jnp.asarray(init.ln_ntot),
            diagnostics=condmod.CondensateEquilibriumDiagnostics(
                n_iter=jnp.asarray(1, dtype=jnp.int32),
                converged=jnp.asarray(False),
                hit_max_iter=jnp.asarray(False),
                final_residual=jnp.asarray(1.0, dtype=jnp.float64),
                residual_crit=jnp.asarray(kwargs["residual_crit"], dtype=jnp.float64),
                max_iter=jnp.asarray(kwargs["max_iter"], dtype=jnp.int32),
                epsilon=jnp.asarray(kwargs["epsilon"], dtype=jnp.float64),
                final_step_size=jnp.asarray(0.1, dtype=jnp.float64),
                invalid_numbers_detected=jnp.asarray(False),
                debug_nan=jnp.asarray(kwargs["debug_nan"]),
            ),
        )

    monkeypatch.setattr(condmod, "minimize_gibbs_cond", stub_minimize_gibbs_cond)

    result = condmod.minimize_gibbs_cond_profile(
        temperatures=jnp.asarray([1000.0], dtype=jnp.float64),
        ln_normalized_pressures=jnp.asarray([0.0], dtype=jnp.float64),
        element_vector=jnp.asarray([1.0], dtype=jnp.float64),
        init=condmod.CondensateEquilibriumInit(
            ln_nk=jnp.asarray([[0.0]], dtype=jnp.float64),
            ln_mk=jnp.asarray([[6.8]], dtype=jnp.float64),
            ln_ntot=jnp.asarray([0.0], dtype=jnp.float64),
        ),
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        epsilon_start=0.0,
        epsilon_crit=-1.0,
        n_step=1,
        max_iter=1,
        method="vmap_cold",
        epsilon_schedule="adaptive_sk_guard",
    )

    assert bool(result.diagnostics.plateaued[0])
    assert not bool(result.diagnostics.reached_requested_epsilon[0])
    assert float(result.diagnostics.actual_epsilon[0]) > -1.0
    assert float(result.diagnostics.requested_epsilon[0]) == -1.0
