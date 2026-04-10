import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.core import _compute_gk
from exogibbs.optimize.pipm_gie_cond import _choose_lambda_by_residual_backtracking
from exogibbs.optimize.pipm_gie_cond import _compute_iteration_step_metrics
from exogibbs.optimize.pipm_gie_cond import _compute_residuals
from exogibbs.optimize.pipm_gie_cond import _recompute_pi_for_residual
from exogibbs.optimize.pipm_gie_cond import _update_all
from exogibbs.optimize.pipm_gie_cond import minimize_gibbs_cond_core
from exogibbs.optimize.pipm_gie_cond import solve_gibbs_iteration_equations_cond
from exogibbs.optimize.pipm_gie_cond import trace_minimize_gibbs_cond_iterations


def test_compute_iteration_step_metrics_preserves_raw_condensate_direction_before_clipping():
    formula_matrix = jnp.array([[1.0, 1.0], [0.0, 1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[1.0], [0.0]], dtype=jnp.float64)
    epsilon = jnp.asarray(-3.0, dtype=jnp.float64)
    temperature = jnp.asarray(1000.0, dtype=jnp.float64)
    ln_normalized_pressure = jnp.asarray(0.0, dtype=jnp.float64)
    b = jnp.array([1.5, 0.5], dtype=jnp.float64)
    hvector = jnp.array([0.0, 0.1], dtype=jnp.float64)
    hvector_cond = jnp.array([-0.3], dtype=jnp.float64)

    ln_nk0 = jnp.array([0.0, 0.0], dtype=jnp.float64)
    ln_mk0 = jnp.array([0.5], dtype=jnp.float64)
    ln_ntot0 = jnp.asarray(0.1, dtype=jnp.float64)
    gk0 = _compute_gk(temperature, ln_nk0, ln_ntot0, hvector, ln_normalized_pressure)

    metrics = _compute_iteration_step_metrics(
        ln_nk0,
        ln_mk0,
        ln_ntot0,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk0,
        hvector_cond,
        epsilon,
    )

    assert float(metrics["max_abs_raw_delta_ln_mk"]) > 0.1
    assert float(metrics["max_abs_clipped_delta_ln_mk"]) == 0.1
    assert not jnp.allclose(metrics["raw_delta_ln_mk"], metrics["delta_ln_mk"])


def test_update_all_reports_post_update_residual_with_fresh_pi():
    formula_matrix = jnp.array([[1.0, 1.0], [0.0, 1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[1.0], [0.0]], dtype=jnp.float64)
    epsilon = jnp.asarray(-3.0, dtype=jnp.float64)
    temperature = jnp.asarray(1000.0, dtype=jnp.float64)
    ln_normalized_pressure = jnp.asarray(0.0, dtype=jnp.float64)
    b = jnp.array([1.5, 0.5], dtype=jnp.float64)
    hvector = jnp.array([0.0, 0.1], dtype=jnp.float64)
    hvector_cond = jnp.array([-0.3], dtype=jnp.float64)

    ln_nk0 = jnp.array([0.0, 0.0], dtype=jnp.float64)
    ln_mk0 = jnp.array([0.5], dtype=jnp.float64)
    ln_ntot0 = jnp.asarray(0.1, dtype=jnp.float64)
    gk0 = _compute_gk(temperature, ln_nk0, ln_ntot0, hvector, ln_normalized_pressure)
    An0 = formula_matrix @ jnp.exp(ln_nk0)
    Am0 = formula_matrix_cond @ jnp.exp(ln_mk0)

    ln_nk1, ln_mk1, ln_ntot1, gk1, An1, Am1, residual1 = _update_all(
        ln_nk0,
        ln_mk0,
        ln_ntot0,
        formula_matrix,
        formula_matrix_cond,
        b,
        temperature,
        ln_normalized_pressure,
        hvector,
        hvector_cond,
        gk0,
        An0,
        Am0,
        jnp.asarray(jnp.inf, dtype=jnp.float64),
        epsilon,
        iter_count=0,
        debug_nan=False,
    )

    nk1 = jnp.exp(ln_nk1)
    mk1 = jnp.exp(ln_mk1)
    ntot1 = jnp.exp(ln_ntot1)
    pi_resid = _recompute_pi_for_residual(
        nk1,
        mk1,
        ntot1,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk1,
        hvector_cond,
        epsilon,
    )
    expected_residual = _compute_residuals(
        nk1,
        mk1,
        ntot1,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk1,
        hvector_cond,
        jnp.exp(epsilon),
        An1,
        Am1,
        pi_resid,
    )

    bk0 = formula_matrix @ jnp.exp(ln_nk0)
    _delta_ln_nk0, _delta_ln_mk0, pi_stale, _delta_ln_ntot0 = solve_gibbs_iteration_equations_cond(
        jnp.exp(ln_nk0),
        jnp.exp(ln_mk0),
        jnp.exp(ln_ntot0),
        formula_matrix,
        formula_matrix_cond,
        b,
        gk0,
        hvector_cond,
        bk0,
        jnp.exp(epsilon),
    )
    stale_residual = _compute_residuals(
        nk1,
        mk1,
        ntot1,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk1,
        hvector_cond,
        jnp.exp(epsilon),
        An1,
        Am1,
        pi_stale,
    )

    assert jnp.isclose(residual1, expected_residual, rtol=1.0e-12, atol=1.0e-12)
    assert not jnp.isclose(residual1, stale_residual, rtol=1.0e-6, atol=1.0e-6)


def test_choose_lambda_by_residual_backtracking_accepts_first_monotone_trial(monkeypatch):
    def stub_evaluate(*args, **kwargs):
        lam = jnp.asarray(args[3], dtype=jnp.float64)
        residual = jnp.where(
            jnp.isclose(lam, 0.8),
            jnp.asarray(5.0, dtype=jnp.float64),
            jnp.asarray(jnp.inf, dtype=jnp.float64),
        )
        return {
            "lam": lam,
            "ln_nk": jnp.asarray([10.0], dtype=jnp.float64) + lam,
            "ln_mk": jnp.asarray([20.0], dtype=jnp.float64) + lam,
            "ln_ntot": jnp.asarray(30.0, dtype=jnp.float64) + lam,
            "gk": jnp.asarray([40.0], dtype=jnp.float64) + lam,
            "An": jnp.asarray([50.0], dtype=jnp.float64) + lam,
            "Am": jnp.asarray([60.0], dtype=jnp.float64) + lam,
            "fresh_residual": residual,
            "all_finite": jnp.asarray(True),
        }

    monkeypatch.setattr("exogibbs.optimize.pipm_gie_cond._evaluate_trial_step", stub_evaluate)

    selected = _choose_lambda_by_residual_backtracking(
        ln_nk=jnp.asarray([0.0], dtype=jnp.float64),
        ln_mk=jnp.asarray([0.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        current_gk=jnp.asarray([1.0], dtype=jnp.float64),
        current_An=jnp.asarray([2.0], dtype=jnp.float64),
        current_Am=jnp.asarray([3.0], dtype=jnp.float64),
        current_residual=jnp.asarray(6.0, dtype=jnp.float64),
        lam_init=jnp.asarray(0.8, dtype=jnp.float64),
        delta_ln_nk=jnp.asarray([0.0], dtype=jnp.float64),
        delta_ln_mk=jnp.asarray([0.0], dtype=jnp.float64),
        delta_ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        b=jnp.asarray([1.0], dtype=jnp.float64),
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        hvector=jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond=jnp.asarray([0.0], dtype=jnp.float64),
        epsilon=jnp.asarray(-5.0, dtype=jnp.float64),
    )

    assert float(selected["lam"]) == 0.8
    assert float(selected["fresh_residual"]) == 5.0


def test_choose_lambda_by_residual_backtracking_falls_back_to_best_finite(monkeypatch):
    def stub_evaluate(*args, **kwargs):
        lam = jnp.asarray(args[3], dtype=jnp.float64)
        residual = jnp.select(
            [
                jnp.isclose(lam, 0.8),
                jnp.isclose(lam, 0.4),
                jnp.isclose(lam, 0.2),
                jnp.isclose(lam, 0.1),
                jnp.isclose(lam, 0.05),
                jnp.isclose(lam, 0.025),
                jnp.isclose(lam, 0.0125),
                jnp.isclose(lam, 0.00625),
                jnp.isclose(lam, 0.003125),
            ],
            [
                jnp.asarray(8.0, dtype=jnp.float64),
                jnp.asarray(7.5, dtype=jnp.float64),
                jnp.asarray(7.0, dtype=jnp.float64),
                jnp.asarray(7.2, dtype=jnp.float64),
                jnp.asarray(jnp.inf, dtype=jnp.float64),
                jnp.asarray(7.1, dtype=jnp.float64),
                jnp.asarray(jnp.inf, dtype=jnp.float64),
                jnp.asarray(7.4, dtype=jnp.float64),
                jnp.asarray(7.3, dtype=jnp.float64),
            ],
            default=jnp.asarray(jnp.inf, dtype=jnp.float64),
        )
        return {
            "lam": lam,
            "ln_nk": jnp.asarray([10.0], dtype=jnp.float64) + lam,
            "ln_mk": jnp.asarray([20.0], dtype=jnp.float64) + lam,
            "ln_ntot": jnp.asarray(30.0, dtype=jnp.float64) + lam,
            "gk": jnp.asarray([40.0], dtype=jnp.float64) + lam,
            "An": jnp.asarray([50.0], dtype=jnp.float64) + lam,
            "Am": jnp.asarray([60.0], dtype=jnp.float64) + lam,
            "fresh_residual": residual,
            "all_finite": jnp.isfinite(residual),
        }

    monkeypatch.setattr("exogibbs.optimize.pipm_gie_cond._evaluate_trial_step", stub_evaluate)

    selected = _choose_lambda_by_residual_backtracking(
        ln_nk=jnp.asarray([0.0], dtype=jnp.float64),
        ln_mk=jnp.asarray([0.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        current_gk=jnp.asarray([1.0], dtype=jnp.float64),
        current_An=jnp.asarray([2.0], dtype=jnp.float64),
        current_Am=jnp.asarray([3.0], dtype=jnp.float64),
        current_residual=jnp.asarray(6.0, dtype=jnp.float64),
        lam_init=jnp.asarray(0.8, dtype=jnp.float64),
        delta_ln_nk=jnp.asarray([0.0], dtype=jnp.float64),
        delta_ln_mk=jnp.asarray([0.0], dtype=jnp.float64),
        delta_ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        b=jnp.asarray([1.0], dtype=jnp.float64),
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        hvector=jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond=jnp.asarray([0.0], dtype=jnp.float64),
        epsilon=jnp.asarray(-5.0, dtype=jnp.float64),
    )

    assert float(selected["lam"]) == 0.2
    assert float(selected["fresh_residual"]) == 7.0


def test_choose_lambda_by_residual_backtracking_returns_zero_step_when_all_trials_invalid(monkeypatch):
    def stub_evaluate(*args, **kwargs):
        return {
            "lam": jnp.asarray(args[3], dtype=jnp.float64),
            "ln_nk": jnp.asarray([99.0], dtype=jnp.float64),
            "ln_mk": jnp.asarray([98.0], dtype=jnp.float64),
            "ln_ntot": jnp.asarray(97.0, dtype=jnp.float64),
            "gk": jnp.asarray([96.0], dtype=jnp.float64),
            "An": jnp.asarray([95.0], dtype=jnp.float64),
            "Am": jnp.asarray([94.0], dtype=jnp.float64),
            "fresh_residual": jnp.asarray(jnp.inf, dtype=jnp.float64),
            "all_finite": jnp.asarray(False),
        }

    monkeypatch.setattr("exogibbs.optimize.pipm_gie_cond._evaluate_trial_step", stub_evaluate)

    selected = _choose_lambda_by_residual_backtracking(
        ln_nk=jnp.asarray([0.0], dtype=jnp.float64),
        ln_mk=jnp.asarray([1.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(2.0, dtype=jnp.float64),
        current_gk=jnp.asarray([3.0], dtype=jnp.float64),
        current_An=jnp.asarray([4.0], dtype=jnp.float64),
        current_Am=jnp.asarray([5.0], dtype=jnp.float64),
        current_residual=jnp.asarray(6.0, dtype=jnp.float64),
        lam_init=jnp.asarray(0.8, dtype=jnp.float64),
        delta_ln_nk=jnp.asarray([0.0], dtype=jnp.float64),
        delta_ln_mk=jnp.asarray([0.0], dtype=jnp.float64),
        delta_ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        b=jnp.asarray([1.0], dtype=jnp.float64),
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        hvector=jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond=jnp.asarray([0.0], dtype=jnp.float64),
        epsilon=jnp.asarray(-5.0, dtype=jnp.float64),
    )

    assert float(selected["lam"]) == 0.0
    assert float(selected["fresh_residual"]) == 6.0
    assert jnp.allclose(selected["ln_nk"], jnp.asarray([0.0], dtype=jnp.float64))
    assert jnp.allclose(selected["ln_mk"], jnp.asarray([1.0], dtype=jnp.float64))
    assert jnp.allclose(selected["ln_ntot"], jnp.asarray(2.0, dtype=jnp.float64))


def test_minimize_gibbs_cond_core_api_still_works():
    formula_matrix = jnp.array([[1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[1.0]], dtype=jnp.float64)
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )

    ln_nk, ln_mk, ln_ntot, counter = minimize_gibbs_cond_core(
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
    assert int(counter) == 0


def test_trace_minimize_gibbs_cond_iterations_reports_line_search_fields():
    formula_matrix = jnp.array([[1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[1.0]], dtype=jnp.float64)
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )

    trace = trace_minimize_gibbs_cond_iterations(
        state,
        ln_nk_init=jnp.array([0.0], dtype=jnp.float64),
        ln_mk_init=jnp.array([0.0], dtype=jnp.float64),
        ln_ntot_init=jnp.asarray(0.0, dtype=jnp.float64),
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=lambda temperature: jnp.array([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.array([2.0], dtype=jnp.float64),
        epsilon=-5.0,
        max_iter=1,
    )

    assert len(trace["history"]) == 1
    record = trace["history"][0]
    expected_fields = {
        "lam_heuristic",
        "lam_selected",
        "n_backtracks",
        "residual_before",
        "residual_after",
        "line_search_used",
        "line_search_accept_kind",
    }
    assert expected_fields.issubset(record.keys())
