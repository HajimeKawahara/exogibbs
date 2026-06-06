import numpy as np
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.minimize_cond import CondensateEquilibriumInit
from exogibbs.optimize.minimize_cond import build_lnnk_constructor_source_trace
from exogibbs.optimize.minimize_cond import trace_condensate_full_vs_reduced_gie_direction
from exogibbs.optimize.minimize_cond import trace_condensate_pdipm_vs_pipm_direction
from exogibbs.optimize.minimize_cond import trace_condensate_pdipm_vs_pipm_fixed_epsilon_trajectories
from exogibbs.optimize.minimize_cond import trace_condensate_gas_limiter_diagnostics
from exogibbs.optimize.minimize_cond import trace_condensate_iteration_lambda_trials
from exogibbs.optimize.minimize_cond import trace_condensate_reduced_solver_backends
from exogibbs.optimize.minimize_cond import trace_condensate_sk_stage_feasibility
from exogibbs.optimize.minimize_cond import solve_gas_equilibrium_with_duals
from exogibbs.optimize.core import _compute_gk
from exogibbs.optimize.pipm_rgie_cond import _compute_residuals
from exogibbs.optimize.pipm_rgie_cond import _choose_lambda_by_residual_backtracking
from exogibbs.optimize.pipm_rgie_cond import _recompute_pi_for_residual
from exogibbs.optimize.pipm_rgie_cond import _update_all
from exogibbs.optimize.pipm_rgie_cond import diagnose_full_vs_reduced_gie_direction
from exogibbs.optimize.pipm_rgie_cond import diagnose_pdipm_vs_pipm_direction
from exogibbs.optimize.pipm_rgie_cond import diagnose_pdipm_vs_pipm_fixed_epsilon_trajectories
from exogibbs.optimize.pipm_rgie_cond import diagnose_gas_step_limiter_and_direction
from exogibbs.optimize.pipm_rgie_cond import diagnose_iteration_lambda_trials
from exogibbs.optimize.pipm_rgie_cond import diagnose_reduced_solver_backend_experiments
from exogibbs.optimize.pipm_rgie_cond import minimize_gibbs_cond_with_diagnostics
from exogibbs.optimize.pipm_rgie_cond import solve_reduced_gibbs_iteration_equations_cond
from exogibbs.optimize.pipm_rgie_cond import trace_minimize_gibbs_cond_epsilon_sweep
from exogibbs.optimize.pipm_rgie_cond import trace_minimize_gibbs_cond_iterations


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
        "lam_heuristic",
        "lam_selected",
        "n_backtracks",
        "residual_before",
        "residual_after",
        "line_search_used",
        "line_search_accept_kind",
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


def test_diagnose_iteration_lambda_trials_reports_trial_grid_metrics():
    formula_matrix = jnp.array([[1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[1.0]], dtype=jnp.float64)
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )

    diagnostics = diagnose_iteration_lambda_trials(
        state,
        ln_nk=jnp.array([0.0], dtype=jnp.float64),
        ln_mk=jnp.array([0.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=lambda temperature: jnp.array([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.array([2.0], dtype=jnp.float64),
        epsilon=-5.0,
        lambda_trials=[1.0, 0.5, 0.1],
    )

    assert diagnostics["trial_lambdas"] == [1.0, 0.5, 0.1]
    assert diagnostics["heuristic_lambda"] <= 1.0
    assert len(diagnostics["trials"]) == 3
    for trial in diagnostics["trials"]:
        expected_fields = {
            "lambda_trial",
            "relative_to_heuristic",
            "fresh_residual",
            "element_balance_residual_norm",
            "ntot_residual",
            "zero_charge_residual",
            "sk_feasibility_margin_min",
            "invalid_numbers_detected",
            "max_abs_delta_ln_nk",
            "max_abs_delta_ln_mk",
            "abs_delta_ln_ntot",
            "normalized_gibbs_energy",
        }
        assert expected_fields.issubset(trial.keys())
        assert trial["zero_charge_residual"] is None


def test_diagnose_iteration_lambda_trials_reports_zero_charge_residual_when_requested():
    formula_matrix = jnp.array([[1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[0.0]], dtype=jnp.float64)
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([0.0], dtype=jnp.float64),
    )

    diagnostics = diagnose_iteration_lambda_trials(
        state,
        ln_nk=jnp.array([-2.0], dtype=jnp.float64),
        ln_mk=jnp.array([0.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=lambda temperature: jnp.array([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.array([2.0], dtype=jnp.float64),
        epsilon=-5.0,
        lambda_trials=[0.0],
        charge_row_index=0,
    )

    trial = diagnostics["trials"][0]
    assert trial["zero_charge_residual"] is not None
    assert trial["zero_charge_residual"] > 0.0


def test_trace_minimize_gibbs_cond_iterations_can_attach_trial_lambda_diagnostics():
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
        trial_lambda_iterations=[0],
        trial_lambda_values=[1.0, 0.5],
    )

    assert len(trace["history"]) == 1
    record = trace["history"][0]
    assert "trial_lambda_diagnostics" in record
    assert record["trial_lambda_diagnostics"]["trial_lambdas"] == [1.0, 0.5]


def test_trace_condensate_iteration_lambda_trials_structured_wrapper(monkeypatch):
    captured = {}

    def stub_raw(state, **kwargs):
        captured["ln_nk"] = kwargs["ln_nk"]
        captured["ln_mk"] = kwargs["ln_mk"]
        captured["ln_ntot"] = kwargs["ln_ntot"]
        captured["epsilon"] = kwargs["epsilon"]
        return {"heuristic_lambda": 0.25, "trial_lambdas": [1.0]}

    monkeypatch.setattr(
        "exogibbs.optimize.minimize_cond._diagnose_iteration_lambda_trials_raw",
        stub_raw,
    )

    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )
    init = CondensateEquilibriumInit(
        ln_nk=jnp.array([0.0], dtype=jnp.float64),
        ln_mk=jnp.array([1.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(2.0, dtype=jnp.float64),
    )

    result = trace_condensate_iteration_lambda_trials(
        state,
        init=init,
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        epsilon=-5.0,
    )

    assert result["heuristic_lambda"] == 0.25
    assert jnp.allclose(captured["ln_nk"], init.ln_nk)
    assert jnp.allclose(captured["ln_mk"], init.ln_mk)
    assert jnp.allclose(captured["ln_ntot"], init.ln_ntot)
    assert captured["epsilon"] == -5.0


def test_diagnose_gas_step_limiter_and_direction_reports_expected_fields():
    formula_matrix = jnp.array([[1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[1.0]], dtype=jnp.float64)
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )

    diagnostics = diagnose_gas_step_limiter_and_direction(
        state,
        ln_nk=jnp.array([0.0], dtype=jnp.float64),
        ln_mk=jnp.array([0.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=lambda temperature: jnp.array([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.array([2.0], dtype=jnp.float64),
        epsilon=-5.0,
        gas_species_names=["g0"],
        top_k=1,
    )

    assert "gas_limiter" in diagnostics
    assert "direction_comparison" in diagnostics
    top_species = diagnostics["gas_limiter"]["top_species"]
    assert len(top_species) == 1
    assert top_species[0]["species_name"] == "g0"
    assert "species_candidate_lambda" in top_species[0]
    assert "lam1_gas_ref" in diagnostics["direction_comparison"]
    assert "cosine_similarity" in diagnostics["direction_comparison"]


def test_trace_minimize_gibbs_cond_iterations_can_attach_gas_limiter_diagnostics():
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
        gas_species_names=["g0"],
        gas_limiter_iterations=[0],
        gas_limiter_top_k=1,
    )

    assert len(trace["history"]) == 1
    record = trace["history"][0]
    assert "gas_limiter_diagnostics" in record
    assert record["gas_limiter_diagnostics"]["gas_limiter"]["top_species"][0]["species_name"] == "g0"


def test_trace_condensate_gas_limiter_diagnostics_structured_wrapper(monkeypatch):
    captured = {}

    def stub_raw(state, **kwargs):
        captured["ln_nk"] = kwargs["ln_nk"]
        captured["ln_mk"] = kwargs["ln_mk"]
        captured["ln_ntot"] = kwargs["ln_ntot"]
        captured["epsilon"] = kwargs["epsilon"]
        captured["gas_species_names"] = kwargs["gas_species_names"]
        return {"gas_limiter": {"top_species": []}, "direction_comparison": {}}

    monkeypatch.setattr(
        "exogibbs.optimize.minimize_cond._diagnose_gas_step_limiter_and_direction_raw",
        stub_raw,
    )

    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )
    init = CondensateEquilibriumInit(
        ln_nk=jnp.array([0.0], dtype=jnp.float64),
        ln_mk=jnp.array([1.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(2.0, dtype=jnp.float64),
    )

    result = trace_condensate_gas_limiter_diagnostics(
        state,
        init=init,
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        epsilon=-5.0,
        gas_species_names=["g0"],
    )

    assert "gas_limiter" in result
    assert jnp.allclose(captured["ln_nk"], init.ln_nk)
    assert jnp.allclose(captured["ln_mk"], init.ln_mk)
    assert jnp.allclose(captured["ln_ntot"], init.ln_ntot)
    assert captured["epsilon"] == -5.0
    assert captured["gas_species_names"] == ["g0"]


def test_choose_lambda_by_residual_backtracking_accepts_first_monotone_trial(monkeypatch):
    def trial_metrics(residual):
        return {
            "element_balance_residual_norm": jnp.asarray(residual, dtype=jnp.float64),
            "ntot_residual": jnp.asarray(0.0, dtype=jnp.float64),
            "gas_stationarity_residual_norm": jnp.asarray(0.0, dtype=jnp.float64),
            "cond_stationarity_residual_norm": jnp.asarray(0.0, dtype=jnp.float64),
        }

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
            "fresh_residual": jnp.asarray(residual, dtype=jnp.float64),
            "all_finite": jnp.asarray(True),
            **trial_metrics(residual),
        }

    monkeypatch.setattr("exogibbs.optimize.pipm_rgie_cond._evaluate_trial_step", stub_evaluate)

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
    assert int(selected["n_backtracks"]) == 0
    assert int(selected["accept_code"]) == 0


def test_choose_lambda_by_residual_backtracking_falls_back_to_best_finite(monkeypatch):
    def trial_metrics(residual):
        return {
            "element_balance_residual_norm": jnp.asarray(residual, dtype=jnp.float64),
            "ntot_residual": jnp.asarray(0.0, dtype=jnp.float64),
            "gas_stationarity_residual_norm": jnp.asarray(0.0, dtype=jnp.float64),
            "cond_stationarity_residual_norm": jnp.asarray(0.0, dtype=jnp.float64),
        }

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
        finite = jnp.isfinite(residual)
        return {
            "lam": lam,
            "ln_nk": jnp.asarray([10.0], dtype=jnp.float64) + lam,
            "ln_mk": jnp.asarray([20.0], dtype=jnp.float64) + lam,
            "ln_ntot": jnp.asarray(30.0, dtype=jnp.float64) + lam,
            "gk": jnp.asarray([40.0], dtype=jnp.float64) + lam,
            "An": jnp.asarray([50.0], dtype=jnp.float64) + lam,
            "Am": jnp.asarray([60.0], dtype=jnp.float64) + lam,
            "fresh_residual": jnp.asarray(residual, dtype=jnp.float64),
            "all_finite": finite,
            **trial_metrics(residual),
        }

    monkeypatch.setattr("exogibbs.optimize.pipm_rgie_cond._evaluate_trial_step", stub_evaluate)

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
    assert int(selected["n_backtracks"]) == 2
    assert int(selected["accept_code"]) == 1


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
            "element_balance_residual_norm": jnp.asarray(jnp.inf, dtype=jnp.float64),
            "ntot_residual": jnp.asarray(0.0, dtype=jnp.float64),
            "gas_stationarity_residual_norm": jnp.asarray(0.0, dtype=jnp.float64),
            "cond_stationarity_residual_norm": jnp.asarray(0.0, dtype=jnp.float64),
        }

    monkeypatch.setattr("exogibbs.optimize.pipm_rgie_cond._evaluate_trial_step", stub_evaluate)

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
    assert int(selected["accept_code"]) == 2


def test_diagnose_reduced_solver_backend_experiments_smoke():
    formula_matrix = jnp.array([[1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[1.0]], dtype=jnp.float64)
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )

    diagnostics = diagnose_reduced_solver_backend_experiments(
        state,
        ln_nk=jnp.array([0.0], dtype=jnp.float64),
        ln_mk=jnp.array([0.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=lambda temperature: jnp.array([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.array([2.0], dtype=jnp.float64),
        epsilon=-5.0,
        backend_configs=[
            {"reduced_solver": "augmented_lu_row_scaled"},
            {"reduced_solver": "augmented_lu_rowcol_scaled"},
            {
                "reduced_solver": "schur_cholesky_reg",
                "regularization_mode": "diag_shift",
                "regularization_strength": 1.0e-12,
            },
        ],
    )

    assert diagnostics["baseline_backend"] == "augmented_lu_row_scaled"
    assert len(diagnostics["comparisons"]) == 3
    assert diagnostics["comparisons"][0]["backend"] == "augmented_lu_row_scaled"
    assert "fresh_post_update_residual" in diagnostics["comparisons"][1]
    assert "cosine_similarity_vs_baseline" in diagnostics["comparisons"][2]


def test_diagnose_reduced_solver_backend_exact_input_bundle_default_off():
    formula_matrix = jnp.array([[1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[1.0]], dtype=jnp.float64)
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )

    diagnostics = diagnose_reduced_solver_backend_experiments(
        state,
        ln_nk=jnp.array([0.0], dtype=jnp.float64),
        ln_mk=jnp.array([0.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=lambda temperature: jnp.array([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.array([2.0], dtype=jnp.float64),
        epsilon=-5.0,
        backend_configs=[{"reduced_solver": "augmented_lu_row_scaled"}],
    )
    assert diagnostics["exact_input_bundle_emitter"]["implemented"] is True
    assert diagnostics["exact_input_bundle_emitter"]["default_off"] is True
    assert diagnostics["exact_input_bundle_emitter"]["active"] is False
    assert "exact_input_bundle" not in diagnostics["comparisons"][0]


def test_diagnose_reduced_solver_backend_exact_input_bundle_emits_fields():
    formula_matrix = jnp.array([[1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[1.0]], dtype=jnp.float64)
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )

    diagnostics = diagnose_reduced_solver_backend_experiments(
        state,
        ln_nk=jnp.array([0.0], dtype=jnp.float64),
        ln_mk=jnp.array([0.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=lambda temperature: jnp.array([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.array([2.0], dtype=jnp.float64),
        epsilon=-5.0,
        backend_configs=[{"reduced_solver": "augmented_lu_row_scaled"}],
        case_key="30:-10",
        newton_iter=0,
        condensates_jac_indices=[7],
        condensate_labels_jac_order=["H2O[s]"],
        element_labels_reduced_order=["H"],
        emit_exact_input_bundle=True,
    )
    bundle = diagnostics["comparisons"][0]["exact_input_bundle"]
    assert bundle["case_key"] == "30:-10"
    assert bundle["same_iteration_newton_iter"] == 0
    assert bundle["ln_nk"] == [0.0]
    assert bundle["ln_mk"] == [0.0]
    assert bundle["ln_ntot"] == 0.0
    assert bundle["nk"] == [1.0]
    assert bundle["mk"] == [1.0]
    assert bundle["ntotk"] == 1.0
    assert bundle["formula_matrix"] == [[1.0]]
    assert bundle["formula_matrix_cond"] == [[1.0]]
    assert bundle["b"] == [1.0]
    assert bundle["gk"] == [0.0]
    assert bundle["bk"] == [1.0]
    assert bundle["hvector_cond"] == [2.0]
    assert bundle["condensates_jac_indices"] == [7]
    assert bundle["condensate_labels_jac_order"] == ["H2O[s]"]
    assert bundle["element_labels_reduced_order"] == ["H"]
    carrier = bundle["reduced_assembly_owner_density_denominator_carrier"]
    assert carrier["diagnostic_only"] is True
    assert carrier["default_off"] is True
    assert carrier["used_as_KL_constructor_input"] is False
    assert carrier["sum_nk_particle_denominator"] == 1.0
    assert carrier["ntotk_denominator"] == 1.0
    assert carrier["resn_sum_nk_minus_ntotk"] == 0.0
    assert carrier["ngas_over_sum_nk_cgs"] == bundle[
        "gas_species_count_density_cgs_candidate"
    ]
    correction = bundle["row_scaled_jec_owner_scalar_correction_carrier"]
    assert correction["diagnostic_only"] is True
    assert correction["default_off"] is True
    assert correction["used_as_KL_constructor_input"] is False
    assert correction["reference_trace_input"] is False
    assert correction["baseline_owner_scalar_cgs"] == carrier["ngas_over_sum_nk_cgs"]
    assert correction["candidate_correction_factors"]["unity"] == 1.0
    assert (
        correction["candidate_correction_factors"]["sum_nk_over_ntotk"]
        == 1.0
    )
    source_audit = bundle["reduced_system_condensate_coupling_source_audit"]
    assert source_audit["diagnostic_only"] is True
    assert source_audit["default_off"] is True
    assert source_audit["used_as_KL_constructor_input"] is False
    assert source_audit["reference_trace_input"] is False
    assert np.allclose(source_audit["q_cond_slot_scale_vector"], [np.exp(5.0)])
    assert np.allclose(
        source_audit["rhs_cond_slot_scale_vector"],
        [2.0 * np.exp(5.0) - 1.0],
    )
    assert (
        source_audit[
            "cgs_row_scaled_J_ec_owner_block_materialized_by_python_reduced_assembly"
        ]
        is False
    )
    scalarization = bundle["fixed_bridge_budget_cap_scalarization_carrier"]
    assert scalarization["diagnostic_only"] is True
    assert scalarization["default_off"] is True
    assert scalarization["used_as_KL_constructor_input"] is False
    assert scalarization["available"] is False
    assert scalarization["selected_slot_count"] == 1
    assert scalarization["valid_slot_count"] == 0
    assert "fixed_84_condensate_slot_mapping_indices_within_local_condensate_vector" in scalarization[
        "missing_inputs"
    ]
    seeding = bundle["fastchem_style_maxdensity_seeding_total_density_carrier"]
    assert seeding["diagnostic_only"] is True
    assert seeding["default_off"] is True
    assert seeding["used_as_KL_constructor_input"] is False
    assert seeding["FastChem_trace_values_used_as_inputs"] is False
    assert seeding["available"] is True
    assert seeding["exact_owner_verified"] is False
    assert seeding["candidate_total_density_cgs"] == bundle[
        "gas_species_total_element_density_cgs_candidate"
    ]
    assert (
        "FastChem-style total_element_density at post_calculate_entry_seeding timing"
        in seeding["missing_exact_inputs"]
    )
    post_initial = bundle["fastchem_post_initial_gas_total_density_carrier"]
    assert post_initial["diagnostic_only"] is True
    assert post_initial["default_off"] is True
    assert post_initial["used_as_KL_constructor_input"] is False
    assert post_initial["FastChem_trace_values_used_as_inputs"] is False
    assert post_initial["available"] is True
    assert post_initial["exact_owner_verified"] is False
    assert post_initial["candidate_total_density_cgs"] == bundle[
        "gas_species_total_element_density_cgs_candidate"
    ]
    assert (
        "KL-owned FastChem-style gas-only totalElementDensity replay at post_initial_gas_total_element_density timing"
        in post_initial["missing_exact_inputs"]
    )
    species_replay = bundle[
        "fastchem_post_initial_gas_species_density_replay_carrier"
    ]
    assert species_replay["diagnostic_only"] is True
    assert species_replay["default_off"] is True
    assert species_replay["used_as_KL_constructor_input"] is False
    assert species_replay["FastChem_trace_values_used_as_inputs"] is False
    assert species_replay["available"] is True
    assert species_replay["exact_owner_verified"] is False
    assert species_replay["current_species_density_cgs_candidate_vector"] == [
        bundle["gas_species_number_density_cgs_candidate_vector"][0]
    ]
    assert species_replay["current_total_element_density_cgs_candidate"] == bundle[
        "gas_species_total_element_density_cgs_candidate"
    ]
    assert species_replay["recovered_hvector_gas"] == [0.0]
    assert species_replay["atomic_prefix_replay_available"] is True
    assert species_replay["atomic_prefix_species_density_cgs_candidate_vector"] == [
        bundle["gas_species_number_density_cgs_candidate_vector"][0]
    ]
    assert species_replay["atomic_prefix_total_element_density_cgs_candidate"] == bundle[
        "gas_species_total_element_density_cgs_candidate"
    ]
    assert species_replay["missing_exact_inputs"] == []
    lifecycle = bundle[
        "fastchem_gas_phase_calculate_lifecycle_replay_contract_carrier"
    ]
    assert lifecycle["diagnostic_only"] is True
    assert lifecycle["default_off"] is True
    assert lifecycle["used_as_KL_constructor_input"] is False
    assert lifecycle["FastChem_trace_values_used_as_inputs"] is False
    assert lifecycle["available"] is True
    assert lifecycle["exact_owner_verified"] is False
    assert "loop element_calculation_order and calculateElementDensities" in lifecycle[
        "lifecycle_stage_order"
    ]
    assert "electron_solver" in lifecycle["solver_dispatch_paths"]
    assert "recovered_hvector_gas" in lifecycle["KL_available_inputs"]
    assert (
        "FastChem element_calculation_order in species/source ordering"
        in lifecycle["missing_exact_inputs"]
    )
    assert (
        "FastChem number_density_old lifecycle and backup/Newton branch decisions"
        in lifecycle["missing_exact_inputs"]
    )
    assert "element_calculation_order" in lifecycle["available_lifecycle_field_ports"]
    assert "element_phi_vector" in lifecycle["available_lifecycle_field_ports"]
    assert "element_epsilon_vector" in lifecycle["available_lifecycle_field_ports"]
    assert "fixed_by_condensation_flags" in lifecycle[
        "available_lifecycle_field_ports"
    ]
    assert "fastchem_options" in lifecycle["available_lifecycle_field_ports"]
    assert "branch_decisions" in lifecycle["available_lifecycle_field_ports"]
    assert "electron_old_density" in lifecycle["missing_lifecycle_field_ports"]
    assert (
        lifecycle["field_port_status"]
        == "lifecycle ports emitted; exact replay inputs still missing"
    )
    assert lifecycle["lifecycle_field_ports"]["element_calculation_order"][
        "available"
    ] is True
    assert lifecycle["lifecycle_field_ports"]["element_calculation_order"][
        "value"
    ] == {"indices": [0], "labels": ["H"]}
    assert lifecycle["lifecycle_field_ports"]["element_calculation_order"][
        "exactness"
    ] == "KL_owned_candidate"
    assert lifecycle["lifecycle_field_ports"]["element_calculation_order"][
        "used_as_KL_constructor_input"
    ] is False
    assert lifecycle["lifecycle_field_ports"]["fastchem_options"]["value"][
        "chem_accuracy"
    ] == 1.0e-5
    assert lifecycle["lifecycle_field_ports"]["branch_decisions"]["value"][
        "actual_converged_iteration"
    ] is None
    assert (
        lifecycle["lifecycle_field_ports"]["branch_decisions"]["exactness"]
        == "partial_runtime_candidate_actual_branch_state_missing"
    )
    assert (
        lifecycle["exact_output_target"]
        == "post-initial gas species number_density vector after gas_phase.calculate"
    )
    timing = bundle["gas_phase_calculate_runtime_timing_result_carrier"]
    assert timing["diagnostic_only"] is True
    assert timing["default_off"] is True
    assert timing["used_as_KL_constructor_input"] is False
    assert timing["FastChem_trace_values_used_as_inputs"] is False
    assert timing["available"] is True
    assert timing["missing_result_ports"] == []
    assert timing["result_ports"]["actual_converged_iteration"]["available"] is True
    assert timing["result_ports"]["actual_converged_iteration"][
        "used_as_KL_constructor_input"
    ] is False
    assert timing["result_ports"]["actual_electron_old_density_cgs"]["available"] is True
    assert timing["result_ports"]["actual_electron_old_density_cgs"][
        "used_as_KL_constructor_input"
    ] is False
    replay = bundle["fastchem_post_initial_gas_total_element_density_replay_carrier"]
    assert replay["diagnostic_only"] is True
    assert replay["default_off"] is True
    assert replay["used_as_KL_constructor_input"] is False
    assert replay["FastChem_trace_values_used_as_inputs"] is False
    assert replay["available"] is True
    assert replay["exact_owner_verified"] is False
    assert replay["total_element_density_cgs_candidate"] == bundle[
        "gas_species_total_element_density_cgs_candidate"
    ]
    assert replay["non_electron_total_element_density_cgs_candidate"] == bundle[
        "gas_species_total_element_density_cgs_candidate"
    ]
    assert replay["positive_stoich_total_element_density_cgs_candidate"] == bundle[
        "gas_species_total_element_density_cgs_candidate"
    ]
    assert replay["electron_row_index"] is None
    assert replay["electron_row_contribution_cgs"] is None
    assert replay["element_density_cgs_candidate_vector"] == [
        bundle["gas_species_total_element_density_cgs_candidate"]
    ]
    assert (
        "post-initial FastChem gas species number_density vector after gas_phase.calculate"
        in replay["missing_exact_inputs"]
    )
    budget_cap = bundle["fastchem_style_maxdensity_budget_cap_carrier"]
    assert budget_cap["diagnostic_only"] is True
    assert budget_cap["default_off"] is True
    assert budget_cap["used_as_KL_constructor_input"] is False
    assert budget_cap["FastChem_trace_values_used_as_inputs"] is False
    assert budget_cap["available"] is False
    assert budget_cap["exact_owner_verified"] is False
    assert (
        "FastChem element epsilon vector at Condensate::maxDensity call timing"
        in budget_cap["missing_exact_inputs"]
    )
    assert "fixed_84_condensate_slot_mapping_indices_within_local_condensate_vector" in budget_cap[
        "missing_inputs"
    ]
    epsilon_budget = bundle["fastchem_style_element_epsilon_budget_cap_carrier"]
    assert epsilon_budget["diagnostic_only"] is True
    assert epsilon_budget["default_off"] is True
    assert epsilon_budget["used_as_KL_constructor_input"] is False
    assert epsilon_budget["FastChem_trace_values_used_as_inputs"] is False
    assert epsilon_budget["available"] is True
    assert epsilon_budget["exact_owner_verified"] is False
    assert epsilon_budget["element_epsilon_candidate_source"] == "b"
    assert epsilon_budget["element_epsilon_candidate_vector"] == bundle["b"]
    assert bundle["fastchem_style_element_epsilon_budget_cap_vector"] == bundle[
        "condensate_budget_cap_vector"
    ]
    assert (
        "FastChem element.epsilon vector at Condensate::maxDensity call timing"
        in epsilon_budget["missing_exact_inputs"]
    )
    normalized_epsilon_budget = bundle[
        "fastchem_style_normalized_b_element_epsilon_budget_cap_carrier"
    ]
    assert normalized_epsilon_budget["diagnostic_only"] is True
    assert normalized_epsilon_budget["default_off"] is True
    assert normalized_epsilon_budget["used_as_KL_constructor_input"] is False
    assert normalized_epsilon_budget["FastChem_trace_values_used_as_inputs"] is False
    assert normalized_epsilon_budget["available"] is True
    assert normalized_epsilon_budget["element_epsilon_candidate_source"] == "b / sum(b)"
    assert normalized_epsilon_budget["b_sum"] == 1.0
    assert normalized_epsilon_budget["missing_exact_inputs"] == []
    assert bundle["element_epsilon_from_normalized_b_vector"] == bundle["b"]
    assert bundle["normalized_b_maxdensity_budget_cap_vector"] == bundle[
        "condensate_budget_cap_vector"
    ]
    file_epsilon_budget = bundle["fastchem_file_element_epsilon_budget_cap_carrier"]
    assert file_epsilon_budget["diagnostic_only"] is True
    assert file_epsilon_budget["default_off"] is True
    assert file_epsilon_budget["used_as_KL_constructor_input"] is False
    assert file_epsilon_budget["FastChem_trace_values_used_as_inputs"] is False
    assert file_epsilon_budget["available"] is True
    assert file_epsilon_budget["missing_exact_inputs"] == []
    assert bundle["fastchem_file_element_epsilon_vector"] == bundle["b"]
    assert bundle["fastchem_file_maxdensity_budget_cap_vector"] == bundle[
        "condensate_budget_cap_vector"
    ]
    file_owner = bundle["fastchem_file_budget_maxdensity_owner_density_carrier"]
    assert file_owner["diagnostic_only"] is True
    assert file_owner["default_off"] is True
    assert file_owner["used_as_KL_constructor_input"] is False
    assert file_owner["FastChem_trace_values_used_as_inputs"] is False
    assert file_owner["available"] is True
    assert file_owner["budget_cap_source"] == "fastchem_file_maxdensity_budget_cap_vector"
    assert (
        file_owner["total_density_source"]
        == "fastchem_post_initial_gas_total_element_density_replay_carrier.total_element_density_cgs_candidate"
    )
    assert file_owner["missing_exact_inputs"] == []
    assert file_owner["total_density_cgs_candidate"] == bundle[
        "gas_species_total_element_density_cgs_candidate"
    ]
    assert bundle["fastchem_file_budget_maxdensity_owner_density_vector"] == [
        bundle["gas_species_total_element_density_cgs_candidate"]
    ]
    verifier = bundle["row_scaled_jec_owner_scalar_verifier"]
    assert verifier["diagnostic_only"] is True
    assert verifier["default_off"] is True
    assert verifier["used_as_KL_constructor_input"] is False
    assert verifier["available"] is False
    assert verifier["scalar_cgs"] is None
    assert "row_scaled_element_condensate_J_ec_target_block" in verifier[
        "missing_inputs"
    ]
    assert bundle["source_state_hash"]
    assert bundle["emitted_before_update_all_with_metrics"] is True


def test_trace_condensate_reduced_solver_backends_structured_wrapper(monkeypatch):
    captured = {}

    def stub_raw(state, **kwargs):
        captured["ln_nk"] = kwargs["ln_nk"]
        captured["ln_mk"] = kwargs["ln_mk"]
        captured["ln_ntot"] = kwargs["ln_ntot"]
        captured["epsilon"] = kwargs["epsilon"]
        captured["case_key"] = kwargs["case_key"]
        captured["emit_exact_input_bundle"] = kwargs["emit_exact_input_bundle"]
        return {"comparisons": []}

    monkeypatch.setattr(
        "exogibbs.optimize.minimize_cond._diagnose_reduced_solver_backend_experiments_raw",
        stub_raw,
    )

    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )
    init = CondensateEquilibriumInit(
        ln_nk=jnp.array([0.0], dtype=jnp.float64),
        ln_mk=jnp.array([1.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(2.0, dtype=jnp.float64),
    )

    result = trace_condensate_reduced_solver_backends(
        state,
        init=init,
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        epsilon=-5.0,
    )

    assert "comparisons" in result
    assert jnp.allclose(captured["ln_nk"], init.ln_nk)
    assert jnp.allclose(captured["ln_mk"], init.ln_mk)
    assert jnp.allclose(captured["ln_ntot"], init.ln_ntot)
    assert captured["epsilon"] == -5.0
    assert captured["case_key"] == "diagnostic"
    assert captured["emit_exact_input_bundle"] is False


def test_exact_input_bundle_materializes_row_scaled_jec_owner_verifier():
    formula_matrix = jnp.array([[1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[1.0]], dtype=jnp.float64)
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )

    diagnostics = diagnose_reduced_solver_backend_experiments(
        state,
        ln_nk=jnp.array([0.0], dtype=jnp.float64),
        ln_mk=jnp.array([0.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=lambda temperature: jnp.array([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.array([2.0], dtype=jnp.float64),
        epsilon=-5.0,
        backend_configs=[{"reduced_solver": "augmented_lu_row_scaled"}],
        case_key="30:-10",
        newton_iter=0,
        condensates_jac_indices=[0],
        condensate_labels_jac_order=["H2O[s]"],
        element_labels_reduced_order=["H"],
        emit_exact_input_bundle=True,
        row_scaled_element_condensate_jec_target_block=[[6.0]],
        selected_element_row_scaling_vector=[2.0],
    )

    verifier = diagnostics["comparisons"][0]["exact_input_bundle"][
        "row_scaled_jec_owner_scalar_verifier"
    ]
    assert verifier["available"] is True
    assert verifier["reference_trace_input"] is True
    assert verifier["used_as_KL_constructor_input"] is False
    assert verifier["scalar_cgs"] == 12.0
    assert verifier["sample_count"] == 1
    assert verifier["missing_inputs"] == []
    candidate = diagnostics["comparisons"][0]["exact_input_bundle"][
        "kl_native_row_scaled_jec_block_candidate"
    ]
    assert candidate["available"] is True
    assert candidate["reference_trace_input"] is False
    assert candidate["used_as_KL_constructor_input"] is False
    assert (
        candidate["row_scaled_element_condensate_J_ec_candidate_block"][0][0]
        == candidate["owner_scalar_cgs"] / 2.0
    )
    assert candidate["candidate_block_shape"] == [1, 1]


def test_exact_input_bundle_accepts_default_off_gas_lifecycle_context():
    formula_matrix = jnp.array([[1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[1.0]], dtype=jnp.float64)
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )

    diagnostics = diagnose_reduced_solver_backend_experiments(
        state,
        ln_nk=jnp.array([0.0], dtype=jnp.float64),
        ln_mk=jnp.array([0.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=lambda temperature: jnp.array([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.array([2.0], dtype=jnp.float64),
        epsilon=-5.0,
        backend_configs=[{"reduced_solver": "augmented_lu_row_scaled"}],
        case_key="30:-10",
        newton_iter=0,
        element_labels_reduced_order=["H"],
        emit_exact_input_bundle=True,
        gas_phase_calculate_lifecycle_context={
            "element_calculation_order": ["H"],
            "element_solver_order_vector": [0],
            "fastchem_options": {"chem_accuracy": 1.0e-4},
            "branch_decisions": {"use_backup_solver": False},
            "runtime_timing_results": {
                "actual_converged_iteration": 2,
                "actual_newtonSolMult_used": False,
                "actual_electron_old_density_cgs": 0.0,
            },
        },
    )

    lifecycle = diagnostics["comparisons"][0]["exact_input_bundle"][
        "fastchem_gas_phase_calculate_lifecycle_replay_contract_carrier"
    ]
    assert lifecycle["diagnostic_only"] is True
    assert lifecycle["default_off"] is True
    assert "element_calculation_order" in lifecycle["available_lifecycle_field_ports"]
    assert "element_solver_order_vector" in lifecycle["available_lifecycle_field_ports"]
    assert "fastchem_options" in lifecycle["available_lifecycle_field_ports"]
    assert "branch_decisions" in lifecycle["available_lifecycle_field_ports"]
    assert "element_phi_vector" in lifecycle["available_lifecycle_field_ports"]
    assert lifecycle["lifecycle_field_ports"]["element_calculation_order"][
        "value"
    ] == ["H"]
    assert lifecycle["lifecycle_field_ports"]["branch_decisions"]["value"] == {
        "use_backup_solver": False
    }
    assert "major_molecules_inc" in lifecycle["missing_lifecycle_field_ports"]
    assert lifecycle["lifecycle_field_ports"]["branch_decisions"][
        "used_as_KL_constructor_input"
    ] is False
    timing = diagnostics["comparisons"][0]["exact_input_bundle"][
        "gas_phase_calculate_runtime_timing_result_carrier"
    ]
    assert timing["available"] is True
    assert timing["result_ports"]["actual_converged_iteration"]["available"] is True
    assert timing["result_ports"]["actual_converged_iteration"]["value"] == 2
    assert timing["result_ports"]["actual_newtonSolMult_used"]["value"] is False
    assert timing["result_ports"]["actual_electron_old_density_cgs"]["value"] == 0.0
    assert timing["result_ports"]["actual_converged_iteration"][
        "used_as_KL_constructor_input"
    ] is False


def test_trace_condensate_reduced_solver_backends_passes_exact_context(monkeypatch):
    captured = {}

    def stub_raw(state, **kwargs):
        captured.update(kwargs)
        return {"comparisons": []}

    monkeypatch.setattr(
        "exogibbs.optimize.minimize_cond._diagnose_reduced_solver_backend_experiments_raw",
        stub_raw,
    )

    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )
    init = CondensateEquilibriumInit(
        ln_nk=jnp.array([0.0], dtype=jnp.float64),
        ln_mk=jnp.array([1.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(2.0, dtype=jnp.float64),
    )

    trace_condensate_reduced_solver_backends(
        state,
        init=init,
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        epsilon=-5.0,
        exact_input_bundle_context={
            "case_key": "30:-10",
            "newton_iter": 0,
            "condensates_jac_indices": [3],
            "condensate_labels_jac_order": ["Al(s)"],
            "element_labels_reduced_order": ["Al"],
            "row_scaled_element_condensate_jec_target_block": [[6.0]],
            "selected_element_row_scaling_vector": [2.0],
            "gas_phase_calculate_lifecycle_context": {
                "element_calculation_order": ["Al"]
            },
            "emit_exact_input_bundle": True,
        },
    )

    assert captured["case_key"] == "30:-10"
    assert captured["newton_iter"] == 0
    assert captured["condensates_jac_indices"] == [3]
    assert captured["condensate_labels_jac_order"] == ["Al(s)"]
    assert captured["element_labels_reduced_order"] == ["Al"]
    assert captured["row_scaled_element_condensate_jec_target_block"] == [[6.0]]
    assert captured["selected_element_row_scaling_vector"] == [2.0]
    assert captured["gas_phase_calculate_lifecycle_context"] == {
        "element_calculation_order": ["Al"]
    }
    assert captured["emit_exact_input_bundle"] is True


def test_trace_condensate_reduced_solver_backends_passes_lnnk_constructor_trace(monkeypatch):
    captured = {}

    def stub_raw(state, **kwargs):
        captured.update(kwargs)
        return {"comparisons": []}

    monkeypatch.setattr(
        "exogibbs.optimize.minimize_cond._diagnose_reduced_solver_backend_experiments_raw",
        stub_raw,
    )

    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )
    source_trace = build_lnnk_constructor_source_trace(
        np.asarray([0.0], dtype=np.longdouble),
        case_key="30:-10",
        newton_iter=0,
        source_stage="test native ln_nk constructor",
        producer_function="tests::test_trace_condensate_reduced_solver_backends_passes_lnnk_constructor_trace",
    )
    init = CondensateEquilibriumInit(
        ln_nk=jnp.array([0.0], dtype=jnp.float64),
        ln_mk=jnp.array([1.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(2.0, dtype=jnp.float64),
        ln_nk_source_trace=source_trace,
    )

    trace_condensate_reduced_solver_backends(
        state,
        init=init,
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        epsilon=-5.0,
        exact_input_bundle_context={
            "case_key": "30:-10",
            "newton_iter": 0,
            "emit_exact_input_bundle": True,
        },
    )

    trace = captured["ln_nk_init_source_trace"]
    assert trace["diagnostic_only"] is True
    assert trace["default_off"] is True
    assert trace["constructor_input"] is False
    assert trace["source_stage"] == "test native ln_nk constructor"
    assert trace["native_longdouble_provenance_available"] is True
    assert trace["used_as_KL_constructor_input"] is False


def test_solve_gas_equilibrium_with_duals_can_emit_lnnk_source_trace():
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )
    result = solve_gas_equilibrium_with_duals(
        state,
        jnp.asarray([[1.0]], dtype=jnp.float64),
        lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        emit_lnnk_source_trace=True,
        source_trace_case_key="30:-10",
    )

    trace = result["ln_nk_source_trace"]
    assert trace["diagnostic_only"] is True
    assert trace["default_off"] is True
    assert trace["constructor_input"] is False
    assert trace["source_stage"] == "minimize_gibbs_core while_loop final carry ln_nk source"
    assert trace["producer_function"].endswith("minimize_gibbs_core_with_source_trace")
    assert trace["native_longdouble_provenance_available"] is False
    assert trace["used_as_KL_constructor_input"] is False
    assert trace["n_iter"] >= 0
    assert trace["trace_boundary"] == "lax.while_loop final carry"
    assert trace["final_carry_tuple_field"] == "ln_nk"
    assert "body/update_all ln_nk_new source" in trace["next_required_field"]
    update_trace = trace["update_all_lnnk_new_source_trace"]
    assert update_trace["diagnostic_only"] is True
    assert update_trace["default_off"] is True
    assert update_trace["constructor_input"] is False
    assert update_trace["source_stage"] == "update_all/_apply_iteration_step ln_nk_new source"
    assert update_trace["update_all_source_components_available"] is True
    assert update_trace["update_all_source_components"]["formula"] == (
        "ln_nk_new = ln_nk + lambda * delta_ln_nk"
    )
    assert "_finish_iteration_solve pi_vector" in update_trace["next_required_field"]
    delta_trace = update_trace["delta_ln_nk_source_components_trace"]
    assert delta_trace["diagnostic_only"] is True
    assert delta_trace["default_off"] is True
    assert delta_trace["constructor_input"] is False
    assert delta_trace["source_stage"] == (
        "_finish_iteration_solve/_compute_gk delta_ln_nk source"
    )
    assert delta_trace["source_formula"] == "delta_ln_nk = at_pi + delta_ln_ntot - gk"
    assert "pi_vector_sample" in delta_trace["component_fields"]
    assert "_solve_iteration_system binv_rhs" in delta_trace["next_required_field"]
    linear_trace = update_trace["linear_solve_and_gk_source_trace"]
    assert linear_trace["diagnostic_only"] is True
    assert linear_trace["default_off"] is True
    assert linear_trace["constructor_input"] is False
    assert linear_trace["source_stage"] == "_solve_iteration_system/_compute_gk source inputs"
    assert "cho_solve" in linear_trace["solve_iteration_system_formula"]
    assert linear_trace["gk_source_formula"] == (
        "gk = hvector + ln_nk - ln_ntot + ln_normalized_pressure"
    )
    assert "hvector_func thermochemical source" in linear_trace["next_required_field"]
    precision_trace = update_trace["hvector_and_linear_precision_source_trace"]
    assert precision_trace["diagnostic_only"] is True
    assert precision_trace["default_off"] is True
    assert precision_trace["constructor_input"] is False
    assert precision_trace["source_stage"] == (
        "hvector_func thermochemical source and cho_solve precision"
    )
    assert precision_trace["linear_solver_factorization"] == "jax.scipy.linalg.cho_factor"
    assert precision_trace["linear_solver_solve"] == "jax.scipy.linalg.cho_solve"
    assert precision_trace["hvector_native_longdouble_provenance_available"] is False
    assert precision_trace["linear_solver_native_longdouble_provenance_available"] is False
    assert "long-double linear algebra comparator" in precision_trace["next_required_field"]
    provider_trace = update_trace["hvector_provider_source_trace"]
    assert provider_trace["diagnostic_only"] is True
    assert provider_trace["default_off"] is True
    assert provider_trace["constructor_input"] is False
    assert provider_trace["source_stage"] == (
        "concrete hvector_func thermochemical provider boundary"
    )
    assert provider_trace["native_longdouble_provenance_available"] is False
    assert "provider-specific hvector construction trace" in provider_trace["next_required_field"]
    comparator = update_trace["longdouble_linear_solve_comparator_trace"]
    assert comparator["diagnostic_only"] is True
    assert comparator["default_off"] is True
    assert comparator["constructor_input"] is False
    assert comparator["attempted"] is True
    assert comparator["source_stage"] == "long-double linear solve comparator"


def test_diagnose_full_vs_reduced_gie_direction_smoke():
    formula_matrix = jnp.array([[1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[1.0]], dtype=jnp.float64)
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )

    diagnostics = diagnose_full_vs_reduced_gie_direction(
        state,
        ln_nk=jnp.array([0.0], dtype=jnp.float64),
        ln_mk=jnp.array([0.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=lambda temperature: jnp.array([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.array([2.0], dtype=jnp.float64),
        epsilon=-5.0,
    )

    assert "raw_direction_comparison" in diagnostics
    assert "reduced_direction" in diagnostics
    assert "full_gie_direction" in diagnostics
    assert "gas_cosine_similarity" in diagnostics["raw_direction_comparison"]
    assert "lam1_gas" in diagnostics["reduced_direction"]
    assert "lam1_gas" in diagnostics["full_gie_direction"]
    assert "fresh_post_update_residual" in diagnostics["full_gie_direction"]


def test_trace_condensate_full_vs_reduced_gie_direction_structured_wrapper(monkeypatch):
    captured = {}

    def stub_raw(state, **kwargs):
        captured["ln_nk"] = kwargs["ln_nk"]
        captured["ln_mk"] = kwargs["ln_mk"]
        captured["ln_ntot"] = kwargs["ln_ntot"]
        captured["epsilon"] = kwargs["epsilon"]
        return {"full_gie_materially_better": False}

    monkeypatch.setattr(
        "exogibbs.optimize.minimize_cond._diagnose_full_vs_reduced_gie_direction_raw",
        stub_raw,
    )

    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )
    init = CondensateEquilibriumInit(
        ln_nk=jnp.array([0.0], dtype=jnp.float64),
        ln_mk=jnp.array([1.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(2.0, dtype=jnp.float64),
    )

    result = trace_condensate_full_vs_reduced_gie_direction(
        state,
        init=init,
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        epsilon=-5.0,
    )

    assert result["full_gie_materially_better"] is False
    assert jnp.allclose(captured["ln_nk"], init.ln_nk)
    assert jnp.allclose(captured["ln_mk"], init.ln_mk)
    assert jnp.allclose(captured["ln_ntot"], init.ln_ntot)
    assert captured["epsilon"] == -5.0


def test_diagnose_pdipm_vs_pipm_direction_smoke():
    formula_matrix = jnp.array([[1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[1.0]], dtype=jnp.float64)
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )

    diagnostics = diagnose_pdipm_vs_pipm_direction(
        state,
        ln_nk=jnp.array([0.0], dtype=jnp.float64),
        ln_mk=jnp.array([0.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=lambda temperature: jnp.array([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.array([2.0], dtype=jnp.float64),
        epsilon=-5.0,
        lambda_trials=[1.0e-4, 5.0e-5],
    )

    assert "direction_comparison" in diagnostics
    assert "pipm_direction" in diagnostics
    assert "pdipm_direction" in diagnostics
    assert diagnostics["lambda_grid"] == [0.0001, 5e-05]
    assert "gas_cosine_similarity" in diagnostics["direction_comparison"]
    assert "lambda_trials" in diagnostics["pipm_direction"]
    assert "pdipm_barrier_residuals_current" in diagnostics["pdipm_direction"]


def test_trace_condensate_pdipm_vs_pipm_direction_structured_wrapper(monkeypatch):
    captured = {}

    def stub_raw(state, **kwargs):
        captured["ln_nk"] = kwargs["ln_nk"]
        captured["ln_mk"] = kwargs["ln_mk"]
        captured["ln_ntot"] = kwargs["ln_ntot"]
        captured["epsilon"] = kwargs["epsilon"]
        captured["lambda_trials"] = kwargs["lambda_trials"]
        return {"pdipm_materially_better": False}

    monkeypatch.setattr(
        "exogibbs.optimize.minimize_cond._diagnose_pdipm_vs_pipm_direction_raw",
        stub_raw,
    )

    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )
    init = CondensateEquilibriumInit(
        ln_nk=jnp.array([0.0], dtype=jnp.float64),
        ln_mk=jnp.array([1.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(2.0, dtype=jnp.float64),
    )

    result = trace_condensate_pdipm_vs_pipm_direction(
        state,
        init=init,
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        epsilon=-5.0,
        lambda_trials=[1.0e-4],
    )

    assert result["pdipm_materially_better"] is False
    assert jnp.allclose(captured["ln_nk"], init.ln_nk)
    assert jnp.allclose(captured["ln_mk"], init.ln_mk)
    assert jnp.allclose(captured["ln_ntot"], init.ln_ntot)
    assert captured["epsilon"] == -5.0
    assert captured["lambda_trials"] == [1.0e-4]


def test_diagnose_pdipm_vs_pipm_fixed_epsilon_trajectories_smoke():
    formula_matrix = jnp.array([[1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.array([[1.0]], dtype=jnp.float64)
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )

    diagnostics = diagnose_pdipm_vs_pipm_fixed_epsilon_trajectories(
        state,
        ln_nk=jnp.array([0.0], dtype=jnp.float64),
        ln_mk=jnp.array([0.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=lambda temperature: jnp.array([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.array([2.0], dtype=jnp.float64),
        epsilon=-5.0,
        rho_offsets=(0.0, 1.0),
        max_iter=2,
    )

    assert "previous_one_step_center_path_note" in diagnostics
    assert "pipm_trace" in diagnostics
    assert "pdipm_runs" in diagnostics
    assert len(diagnostics["pdipm_runs"]) == 2
    assert "initial_fc_norm" in diagnostics["pdipm_runs"][0]
    assert "history" in diagnostics["pdipm_runs"][1]


def test_trace_condensate_pdipm_vs_pipm_fixed_epsilon_trajectories_wrapper(monkeypatch):
    captured = {}

    def stub_raw(state, **kwargs):
        captured["ln_nk"] = kwargs["ln_nk"]
        captured["ln_mk"] = kwargs["ln_mk"]
        captured["ln_ntot"] = kwargs["ln_ntot"]
        captured["epsilon"] = kwargs["epsilon"]
        captured["rho_offsets"] = kwargs["rho_offsets"]
        return {"pdipm_runs": []}

    monkeypatch.setattr(
        "exogibbs.optimize.minimize_cond._diagnose_pdipm_vs_pipm_fixed_epsilon_trajectories_raw",
        stub_raw,
    )

    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.array([1.0], dtype=jnp.float64),
    )
    init = CondensateEquilibriumInit(
        ln_nk=jnp.array([0.0], dtype=jnp.float64),
        ln_mk=jnp.array([1.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(2.0, dtype=jnp.float64),
    )

    result = trace_condensate_pdipm_vs_pipm_fixed_epsilon_trajectories(
        state,
        init=init,
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        epsilon=-5.0,
        rho_offsets=(0.0, 1.0, -1.0),
    )

    assert "pdipm_runs" in result
    assert jnp.allclose(captured["ln_nk"], init.ln_nk)
    assert jnp.allclose(captured["ln_mk"], init.ln_mk)
    assert jnp.allclose(captured["ln_ntot"], init.ln_ntot)
    assert captured["epsilon"] == -5.0
    assert captured["rho_offsets"] == (0.0, 1.0, -1.0)
