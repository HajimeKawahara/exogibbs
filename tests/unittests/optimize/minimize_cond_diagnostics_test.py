import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.minimize_cond import CondensateEquilibriumInit
from exogibbs.optimize.minimize_cond import trace_condensate_full_vs_reduced_gie_direction
from exogibbs.optimize.minimize_cond import trace_condensate_pdipm_vs_pipm_direction
from exogibbs.optimize.minimize_cond import trace_condensate_pdipm_vs_pipm_fixed_epsilon_trajectories
from exogibbs.optimize.minimize_cond import trace_condensate_gas_limiter_diagnostics
from exogibbs.optimize.minimize_cond import trace_condensate_iteration_lambda_trials
from exogibbs.optimize.minimize_cond import trace_condensate_reduced_solver_backends
from exogibbs.optimize.minimize_cond import trace_condensate_sk_stage_feasibility
from exogibbs.optimize.core import _compute_gk
from exogibbs.optimize.pipm_rgie_cond import _compute_residuals
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
            "sk_feasibility_margin_min",
            "invalid_numbers_detected",
            "max_abs_delta_ln_nk",
            "max_abs_delta_ln_mk",
            "abs_delta_ln_ntot",
            "normalized_gibbs_energy",
        }
        assert expected_fields.issubset(trial.keys())


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


def test_trace_condensate_reduced_solver_backends_structured_wrapper(monkeypatch):
    captured = {}

    def stub_raw(state, **kwargs):
        captured["ln_nk"] = kwargs["ln_nk"]
        captured["ln_mk"] = kwargs["ln_mk"]
        captured["ln_ntot"] = kwargs["ln_ntot"]
        captured["epsilon"] = kwargs["epsilon"]
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
