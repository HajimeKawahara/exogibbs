import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.minimize_cond import CondensateEquilibriumInit
from exogibbs.optimize.minimize_cond import trace_condensate_sk_stage_feasibility
from exogibbs.optimize.core import _compute_gk
from exogibbs.optimize.pipm_rgie_cond import _compute_residuals
from exogibbs.optimize.pipm_rgie_cond import _recompute_pi_for_residual
from exogibbs.optimize.pipm_rgie_cond import _update_all
from exogibbs.optimize.pipm_rgie_cond import minimize_gibbs_cond_with_diagnostics
from exogibbs.optimize.pipm_rgie_cond import solve_reduced_gibbs_iteration_equations_cond
from exogibbs.optimize.pipm_rgie_cond import trace_minimize_gibbs_cond_epsilon_sweep


def test_minimize_gibbs_cond_with_diagnostics_smoke():
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

    expected_fields = {
        "n_iter",
        "converged",
        "hit_max_iter",
        "final_residual",
        "residual_crit",
        "max_iter",
        "epsilon",
        "final_step_size",
        "invalid_numbers_detected",
        "debug_nan",
    }
    assert expected_fields.issubset(diagnostics.keys())

    assert diagnostics["n_iter"].shape == ()
    assert diagnostics["final_residual"].shape == ()
    assert diagnostics["residual_crit"].shape == ()
    assert diagnostics["epsilon"].shape == ()
    assert diagnostics["final_step_size"].shape == ()

    assert int(diagnostics["n_iter"]) == 0
    assert int(diagnostics["max_iter"]) == 0
    assert not bool(diagnostics["converged"])
    assert bool(diagnostics["hit_max_iter"])
    assert not bool(diagnostics["invalid_numbers_detected"])
    assert not bool(diagnostics["debug_nan"])
    assert not (bool(diagnostics["converged"]) and bool(diagnostics["hit_max_iter"]))


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

    ln_nk1, ln_mk1, ln_ntot1, gk1, An1, Am1, residual1, _lam = _update_all(
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
    sk0 = jnp.exp(2.0 * ln_mk0 - epsilon)
    pi_stale, _delta_ln_ntot0 = solve_reduced_gibbs_iteration_equations_cond(
        jnp.exp(ln_nk0),
        jnp.exp(ln_mk0),
        jnp.exp(ln_ntot0),
        formula_matrix,
        formula_matrix_cond,
        b,
        gk0,
        bk0,
        hvector_cond,
        sk0,
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


def test_trace_minimize_gibbs_cond_epsilon_sweep_smoke():
    formula_matrix = jnp.array([[1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[1.0]], dtype=jnp.float64)
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )

    trace = trace_minimize_gibbs_cond_epsilon_sweep(
        state,
        ln_nk_init=jnp.array([0.0], dtype=jnp.float64),
        ln_mk_init=jnp.array([0.0], dtype=jnp.float64),
        ln_ntot_init=jnp.asarray(0.0, dtype=jnp.float64),
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=lambda temperature: jnp.array([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.array([2.0], dtype=jnp.float64),
        epsilons=[-5.0],
        max_iter=2,
    )

    assert trace["epsilons"] == [-5.0]
    assert len(trace["traces"]) == 1
    first = trace["traces"][0]
    assert "history" in first
    assert len(first["history"]) >= 1
    record = first["history"][0]
    expected_fields = {
        "residual",
        "lam",
        "lam1_gas",
        "lam1_cond",
        "lam2_cond",
        "max_abs_delta_ln_nk",
        "max_abs_raw_delta_ln_mk",
        "max_abs_clipped_delta_ln_mk",
        "delta_ln_ntot",
        "pi_norm",
        "reduced_row_scale_min",
        "reduced_row_scale_max",
        "reduced_row_scale_ratio",
    }
    assert expected_fields.issubset(record.keys())


def test_trace_condensate_sk_stage_feasibility_reports_pre_iteration_violation():
    formula_matrix = jnp.array([[1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[1.0]], dtype=jnp.float64)
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )

    trace = trace_condensate_sk_stage_feasibility(
        state,
        init=CondensateEquilibriumInit(
            ln_nk=jnp.array([0.0], dtype=jnp.float64),
            ln_mk=jnp.array([10.0], dtype=jnp.float64),
            ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        ),
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=lambda temperature: jnp.array([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.array([2.0], dtype=jnp.float64),
        epsilon_start=0.0,
        epsilon_crit=-1.0,
        n_step=1,
        max_iter=0,
        condensate_species=["test_cond"],
    )

    assert len(trace["stages"]) == 2
    first = trace["stages"][0]
    assert first["has_pre_iteration_sk_infeasibility"]
    assert first["worst_infeasible_indices"] == [0]
    assert first["worst_infeasible_names"] == ["test_cond"]
    assert first["condition"] == "log_s_max + epsilon - 2*ln_mk >= 0"
