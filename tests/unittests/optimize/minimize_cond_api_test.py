from types import SimpleNamespace

import jax.numpy as jnp
import pytest
from jax import config

config.update("jax_enable_x64", True)

from exogibbs.api.chemistry import ThermoState
import exogibbs.optimize.minimize_cond as condmod
from exogibbs.optimize.pipm_rgie_cond import minimize_gibbs_cond_with_diagnostics as raw_minimize_gibbs_cond_with_diagnostics
from exogibbs.optimize.pipm_rgie_cond import build_hybrid_candidate_masks
from exogibbs.optimize.pipm_rgie_cond import build_internal_complementarity_tau
from exogibbs.optimize.pipm_rgie_cond import build_kl_atomic_candidate_masks
from exogibbs.optimize.pipm_rgie_cond import compute_condensed_element_gas_recoupling_terms
from exogibbs.optimize.pipm_rgie_cond import compute_hybrid_candidate_log_activity_proxy
from exogibbs.optimize.pipm_rgie_cond import compute_internal_complementarity_residual
from exogibbs.optimize.pipm_rgie_cond import compute_kl_atomic_complementarity_residual
from exogibbs.optimize.pipm_rgie_cond import compute_kl_condensate_log_activity
from exogibbs.optimize.pipm_rgie_cond import reconstruct_kl_atomic_gas_from_u
from exogibbs.optimize.pipm_rgie_cond import (
    solve_hybrid_candidate_selected_reduced_coupling_direction,
)


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


def test_minimize_gibbs_cond_default_startup_keeps_existing_ln_mk(monkeypatch):
    captured = {}

    def stub_raw(state, ln_nk_init, ln_mk_init, ln_ntot_init, **kwargs):
        del state, ln_nk_init, ln_ntot_init, kwargs
        captured["ln_mk_init"] = ln_mk_init
        return (
            jnp.asarray([0.1], dtype=jnp.float64),
            jnp.asarray([0.2, 0.3], dtype=jnp.float64),
            jnp.asarray(0.4, dtype=jnp.float64),
            {
                "n_iter": jnp.asarray(0, dtype=jnp.int32),
                "converged": jnp.asarray(False),
                "hit_max_iter": jnp.asarray(True),
                "final_residual": jnp.asarray(1.0, dtype=jnp.float64),
                "residual_crit": jnp.asarray(1.0e-8, dtype=jnp.float64),
                "max_iter": jnp.asarray(0, dtype=jnp.int32),
                "epsilon": jnp.asarray(-5.0, dtype=jnp.float64),
                "final_step_size": jnp.asarray(0.0, dtype=jnp.float64),
                "invalid_numbers_detected": jnp.asarray(False),
                "debug_nan": jnp.asarray(False),
            },
        )

    monkeypatch.setattr(condmod, "_minimize_gibbs_cond_with_diagnostics_raw", stub_raw)

    init = condmod.CondensateEquilibriumInit(
        ln_nk=jnp.asarray([1.0], dtype=jnp.float64),
        ln_mk=jnp.asarray([-11.0, -7.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
    )
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.asarray([1.0], dtype=jnp.float64),
    )

    condmod.minimize_gibbs_cond(
        state,
        init=init,
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0, 0.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        epsilon=-5.0,
        residual_crit=1.0e-8,
        max_iter=0,
    )

    assert jnp.allclose(captured["ln_mk_init"], init.ln_mk)


def test_classify_rgie_support_proxies_uses_r_s_d_kappa():
    result = condmod.classify_rgie_support_proxies(
        ln_mk=jnp.log(jnp.asarray([1.0e-2, 1.0e-20, 1.0e-8], dtype=jnp.float64)),
        driving=jnp.asarray([1.0, -1.0e-4, -1.0e-2], dtype=jnp.float64),
        epsilon=-10.0,
        classifier_config=condmod.CondensateRGIESupportClassifierConfig(
            on_ratio_min=1.0e1,
            off_ratio_max=1.0e-6,
            on_s_min=1.0e-4,
            off_s_max=1.0e-12,
            driving_positive_tol=1.0e-6,
            driving_negative_tol=1.0e-6,
            kappa_on_min_multiple_of_nu=1.0,
            kappa_off_max_multiple_of_nu=1.0 + 1.0e-6,
        ),
    )

    assert result["labels"] == [
        "on_support_proxy",
        "off_support_proxy",
        "ambiguous",
    ]


def test_minimize_gibbs_cond_support_method_smoothed_dispatches(monkeypatch):
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.asarray([1.0], dtype=jnp.float64),
    )
    init = condmod.CondensateEquilibriumInit(
        ln_nk=jnp.asarray([0.0], dtype=jnp.float64),
        ln_mk=jnp.asarray([-10.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
    )
    called = {}

    def stub_experimental(*args, **kwargs):
        del args
        called["support_method"] = kwargs.get("classifier_config", "seen")
        return (
            condmod.CondensateEquilibriumResult(
                ln_nk=jnp.asarray([1.0], dtype=jnp.float64),
                ln_mk=jnp.asarray([-5.0], dtype=jnp.float64),
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
            ),
            {"accepted": True},
        )

    monkeypatch.setattr(condmod, "_run_experimental_smoothed_semismooth_outer", stub_experimental)

    result = condmod.minimize_gibbs_cond(
        state,
        init=init,
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        epsilon=-5.0,
        support_method="smoothed_semismooth_outer",
    )

    assert isinstance(result, condmod.CondensateEquilibriumResult)
    assert "support_method" in called


def test_minimize_gibbs_cond_default_support_method_stays_legacy(monkeypatch):
    captured = {}

    def stub_legacy(*args, **kwargs):
        del args
        captured["called"] = True
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

    monkeypatch.setattr(condmod, "_minimize_gibbs_cond_legacy", stub_legacy)

    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.asarray([1.0], dtype=jnp.float64),
    )
    init = condmod.CondensateEquilibriumInit(
        ln_nk=jnp.asarray([0.0], dtype=jnp.float64),
        ln_mk=jnp.asarray([-10.0], dtype=jnp.float64),
        ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
    )

    condmod.minimize_gibbs_cond(
        state,
        init=init,
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        epsilon=-5.0,
    )

    assert captured["called"] is True


def test_restricted_support_payload_reports_condensate_amount_gauge(monkeypatch):
    def stub_legacy(*args, **kwargs):
        del args, kwargs
        return condmod.CondensateEquilibriumResult(
            ln_nk=jnp.asarray([0.0], dtype=jnp.float64),
            ln_mk=jnp.log(jnp.asarray([0.2], dtype=jnp.float64)),
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

    monkeypatch.setattr(condmod, "_minimize_gibbs_cond_legacy", stub_legacy)
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.asarray([1.0], dtype=jnp.float64),
    )

    result = condmod.solve_restricted_support_condensate_layer(
        state,
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        support_indices=(0,),
        condensate_species=("X(s)",),
        support_amounts_init=jnp.asarray([0.2], dtype=jnp.float64),
    )

    assert result["condensate_amount_gauge"] == "element_inventory_target_fraction"
    assert (
        result["fastchem4_first_step_equivalent_gauge"]
        == "number_density_divided_by_initial_gas_phase_total_element_density"
    )
    assert result["ln_ntot_gauge"] == "gas_species_total_in_element_inventory_target_fraction"


def test_restricted_support_accepts_pdipm_rgie_v11_activity_correction_mode():
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.asarray([1.0], dtype=jnp.float64),
    )

    result = condmod.solve_restricted_support_condensate_layer(
        state,
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        support_indices=(0,),
        condensate_species=("X(s)",),
        support_amounts_init=jnp.asarray([0.5], dtype=jnp.float64),
        max_iter=2,
        reduced_coupling_config=condmod.CondensateRGIEReducedCouplingConfig(
            reduced_coupling_mode="pdipm_rgie_v11_activity_correction",
        ),
    )

    assert (
        result["restricted_reduced_coupling_config_mode"]
        == "pdipm_rgie_v11_activity_correction"
    )
    pdipm_payload = result["diagnostics"]["pdipm_rgie_v11_activity_correction"]
    assert pdipm_payload["activity_correction_state"]["rho_initialization"] == (
        "rho0 = 0, eta0 = 1"
    )
    assert pdipm_payload["activity_correction_state"]["fastchem4_constructor_values_used"] is False
    assert pdipm_payload["activity_correction_state"]["tau_formula"].startswith(
        "condTau * reference_element_budget"
    )
    assert pdipm_payload["activity_correction_state"]["cond_tau"] == pytest.approx(1.0e-15)
    assert (
        pdipm_payload["activity_correction_state"]["paired_density_activity_update"]
        is False
    )
    assert (
        pdipm_payload["activity_correction_state"][
            "activity_correction_update_policy"
        ]
        == "tce_v1_2_pdipm_newton_reconstruction"
    )
    assert (
        pdipm_payload["activity_correction_state"]["jacobian_selection_policy"]
        == "fastchem4_log_activity_jacobian_with_rem_schur_rhs"
    )
    assert (
        "reduced Qhat/RHS Schur contribution"
        in pdipm_payload["activity_correction_state"]["rem_rhs_update_policy"]
    )
    assert result["diagnostics"]["post_solver_gas_refresh"]["policy"] == (
        "post_solver_depleted_gas_refresh_trial"
    )
    assert (
        result["diagnostics"]["post_solver_gas_refresh"][
            "fastchem4_trace_public_runtime_constructor_inputs_used"
        ]
        is False
    )
    assert len(pdipm_payload["history"]) >= 1


def test_pdipm_rgie_v11_activity_correction_uses_tce_gas_source_for_initial_pi(
    monkeypatch,
):
    captured = {}
    q = jnp.log(jnp.asarray([0.25, 0.75], dtype=jnp.float64))
    qtot = jnp.log(jnp.asarray(1.25, dtype=jnp.float64))
    hvector = jnp.asarray([0.2, -0.1], dtype=jnp.float64)
    ln_pressure = jnp.log(jnp.asarray(2.0, dtype=jnp.float64))

    def fake_solve_gas_equilibrium_with_duals(*args, **kwargs):
        del args, kwargs
        return {
            "ln_nk": q,
            "ln_ntot": qtot,
            "pi_vector": jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        }

    def fake_reduced_step(**kwargs):
        captured["state"] = kwargs["state"]
        captured["gas_stationarity_source"] = kwargs["gas_stationarity_source"]
        return SimpleNamespace(
            trial_step_accepted=False,
            alpha=0.0,
            initial_combined_residual_l2=1.0,
            candidate_combined_residual_l2=1.0,
            candidate_budget_l2=1.0,
            candidate_condensate_stationarity_l2=1.0,
            candidate_barrier_complementarity_l2=1.0,
            delta_r=(0.0,),
            delta_rho=(0.0,),
            candidate_state=kwargs["state"],
        )

    import exogibbs.optimize.pdipm_rgie_cond as pdipm_rgie_cond

    monkeypatch.setattr(
        condmod,
        "solve_gas_equilibrium_with_duals",
        fake_solve_gas_equilibrium_with_duals,
    )
    monkeypatch.setattr(
        pdipm_rgie_cond,
        "solve_pdipm_rgie_algorithm_v11_reduced_step",
        fake_reduced_step,
    )
    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=ln_pressure,
        element_vector=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
    )

    condmod.solve_restricted_support_condensate_layer(
        state,
        formula_matrix=jnp.eye(2, dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0], [0.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: hvector,
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        support_indices=(0,),
        support_amounts_init=jnp.asarray([0.5], dtype=jnp.float64),
        initial_log_state_override=condmod.CondensateEquilibriumInit(
            ln_nk=q,
            ln_mk=jnp.log(jnp.asarray([0.5], dtype=jnp.float64)),
            ln_ntot=qtot,
        ),
        max_iter=1,
        reduced_coupling_config=condmod.CondensateRGIEReducedCouplingConfig(
            reduced_coupling_mode="pdipm_rgie_v11_activity_correction",
        ),
    )

    expected_source = hvector + ln_pressure - qtot
    assert captured["gas_stationarity_source"] == pytest.approx(expected_source.tolist())
    assert captured["state"].element_potential == pytest.approx(
        (q + expected_source).tolist()
    )


def test_pdipm_rgie_v11_fixed_support_batch_matches_layer_core(monkeypatch):
    env_names = (
        "EXOGIBBS_FIXED_SUPPORT_BATCH_SECOND_ORDER_CORRECTION",
        "EXOGIBBS_FIXED_SUPPORT_BATCH_ADAPTIVE_REGULARIZATION",
        "EXOGIBBS_FIXED_SUPPORT_BATCH_REGULARIZATION_BASE",
        "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_MAX_ABS_STEP",
        "EXOGIBBS_FIXED_SUPPORT_BATCH_USE_LEGACY_CAPACITY_EPSILON",
        "EXOGIBBS_FIXED_SUPPORT_BATCH_STEP_CONTROL",
        "EXOGIBBS_FIXED_SUPPORT_BATCH_SELECTION_POLICY",
    )
    for env_name in env_names:
        monkeypatch.delenv(env_name, raising=False)
    formula_matrix = jnp.asarray([[1.0]], dtype=jnp.float64)
    formula_matrix_cond_active = jnp.asarray([[1.0]], dtype=jnp.float64)
    hvector = jnp.asarray([[0.0], [0.0]], dtype=jnp.float64)
    hvector_cond_active = jnp.asarray([[0.0], [0.0]], dtype=jnp.float64)
    element_inventory_target = jnp.asarray([[1.0], [1.0]], dtype=jnp.float64)
    ln_normalized_pressure = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
    ln_nk_init = jnp.log(jnp.asarray([[0.8], [0.7]], dtype=jnp.float64))
    ln_mk_init = jnp.log(jnp.asarray([[0.2], [0.3]], dtype=jnp.float64))
    ln_ntot_init = jnp.log(jnp.asarray([0.8, 0.7], dtype=jnp.float64))

    batch_result, batch_extra = (
        condmod._solve_pdipm_rgie_v11_activity_correction_fixed_support_batch(
            ln_nk_init=ln_nk_init,
            ln_mk_init=ln_mk_init,
            ln_ntot_init=ln_ntot_init,
            formula_matrix=formula_matrix,
            formula_matrix_cond_active=formula_matrix_cond_active,
            element_inventory_target=element_inventory_target,
            hvector=hvector,
            hvector_cond_active=hvector_cond_active,
            ln_normalized_pressure=ln_normalized_pressure,
            epsilon=-10.0,
            max_iter=2,
            lambda_initialization="gas_lstsq",
        )
    )

    budget_residual = (
        jnp.exp(batch_result.ln_nk) + jnp.exp(batch_result.ln_mk) - element_inventory_target
    )
    assert jnp.all(jnp.isfinite(batch_result.ln_nk))
    assert jnp.all(jnp.isfinite(batch_result.ln_mk))
    assert float(jnp.max(jnp.abs(budget_residual))) <= 1.0e-3

    payload = batch_extra["pdipm_rgie_v11_activity_correction_fixed_support_batch"]
    assert payload["experimental"] is True
    assert payload["production_route_wiring"] is False
    assert payload["accepted_iteration_count"].shape == (2,)
    assert payload["stationarity_restoration_accepted_iteration_count"].shape == (2,)
    assert payload["second_order_correction_accepted_iteration_count"].shape == (2,)
    assert payload["adaptive_regularization_selected_iteration_count"].shape == (2,)
    assert payload["second_order_correction_enabled"] is False
    assert payload["adaptive_regularization_enabled"] is False
    assert (
        payload["line_search_candidate_selection_policy"]
        == "ipopt_vectorized_max_alpha"
    )
    reconstructed_final_residual = jnp.sqrt(
        payload["gas_residual_norm"] * payload["gas_residual_norm"]
        + payload["condensate_stationarity_residual_norm"]
        * payload["condensate_stationarity_residual_norm"]
        + payload["budget_residual_norm"] * payload["budget_residual_norm"]
        + payload["complementarity_residual_norm"]
        * payload["complementarity_residual_norm"]
        + payload["total_density_residual_norm"]
        * payload["total_density_residual_norm"]
    )
    assert jnp.allclose(
        reconstructed_final_residual,
        batch_result.diagnostics.final_residual,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    complementarity_vector = (
        batch_result.ln_mk
        + payload["final_log_activity_correction"]
        - batch_result.diagnostics.epsilon[:, None]
    )
    assert jnp.allclose(
        jnp.linalg.norm(complementarity_vector, axis=1),
        payload["complementarity_residual_norm"],
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    best_result, best_extra = (
        condmod._solve_pdipm_rgie_v11_activity_correction_fixed_support_batch(
            ln_nk_init=ln_nk_init,
            ln_mk_init=ln_mk_init,
            ln_ntot_init=ln_ntot_init,
            formula_matrix=formula_matrix,
            formula_matrix_cond_active=formula_matrix_cond_active,
            element_inventory_target=element_inventory_target,
            hvector=hvector,
            hvector_cond_active=hvector_cond_active,
            ln_normalized_pressure=ln_normalized_pressure,
            epsilon=-10.0,
            max_iter=2,
            lambda_initialization="best_residual",
        )
    )
    best_payload = best_extra["pdipm_rgie_v11_activity_correction_fixed_support_batch"]
    assert best_result.diagnostics.final_residual.shape == (2,)
    assert best_payload["lambda_selection_index"].shape == (2,)
    assert best_payload["lambda_candidate_labels"] == (
        "provided",
        "gas_lstsq",
        "gas_cond_lstsq",
        "damped_gas_lstsq",
        "damped_gas_cond_lstsq",
    )


def test_fixed_support_batch_persistent_filter_smoke(monkeypatch):
    monkeypatch.setenv("EXOGIBBS_FIXED_SUPPORT_BATCH_IPOPT_FILTER_ACCEPTANCE", "1")
    monkeypatch.setenv("EXOGIBBS_FIXED_SUPPORT_BATCH_SECOND_ORDER_CORRECTION", "1")
    monkeypatch.setenv("EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_BUDGET_PASSES", "2")
    monkeypatch.setenv("EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_DUAL_REPAIR", "0")
    monkeypatch.setenv("EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION", "1")
    monkeypatch.setenv(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_BUDGET_RESTORATION_PASSES", "2"
    )
    monkeypatch.setenv(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_SELECTION_POLICY",
        "ipopt_vectorized_max_alpha",
    )
    result, extra = (
        condmod._solve_pdipm_rgie_v11_activity_correction_fixed_support_batch(
            ln_nk_init=jnp.log(jnp.asarray([[0.8]], dtype=jnp.float64)),
            ln_mk_init=jnp.log(jnp.asarray([[0.2]], dtype=jnp.float64)),
            ln_ntot_init=jnp.log(jnp.asarray([0.8], dtype=jnp.float64)),
            formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
            formula_matrix_cond_active=jnp.asarray([[1.0]], dtype=jnp.float64),
            element_inventory_target=jnp.asarray([[1.0]], dtype=jnp.float64),
            hvector=jnp.asarray([[0.0]], dtype=jnp.float64),
            hvector_cond_active=jnp.asarray([[0.0]], dtype=jnp.float64),
            ln_normalized_pressure=jnp.asarray([0.0], dtype=jnp.float64),
            epsilon=-10.0,
            max_iter=2,
            lambda_initialization="gas_lstsq",
        )
    )
    payload = extra["pdipm_rgie_v11_activity_correction_fixed_support_batch"]
    candidates = payload["line_search_candidate_diagnostics"]

    assert payload["ipopt_filter_acceptance_enabled"] is True
    assert payload["ipopt_filter_policy"] == "persistent_phi_theta"
    assert payload["second_order_correction_policy"] == "reduced_kkt"
    assert (
        payload["budget_restoration_policy"]
        == "positive_negative_slack_proximity_v1"
    )
    assert payload["restoration_phase_entry_count"].shape == (1,)
    assert payload["restoration_phase_exit_count"].shape == (1,)
    assert payload["amount_restoration_accepted_iteration_count"].shape == (1,)
    assert int(payload["restoration_phase_entry_count"][0]) == 0
    assert int(payload["restoration_phase_exit_count"][0]) == 0
    assert not bool(payload["restoration_phase_active_at_stop"][0])
    assert float(payload["restoration_last_dual_alpha"][0]) == 1.0
    assert payload["restoration_entry_residual_vector"].shape == (1, 6)
    assert payload["restoration_best_residual_vector"].shape == (1, 6)
    assert payload["restoration_last_exit_predual_residual_vector"].shape == (
        1,
        6,
    )
    assert payload["restoration_last_exit_postdual_residual_vector"].shape == (
        1,
        6,
    )
    assert payload["restoration_first_normal_residual_vector"].shape == (1, 6)
    assert payload["restoration_first_normal_attempted"].shape == (1,)
    assert payload["restoration_first_normal_accepted"].shape == (1,)
    assert payload["restoration_first_normal_selected_type"].shape == (1,)
    assert payload["restoration_return_probe_pending"].shape == (1,)
    assert payload["restoration_last_active_accepted_iteration_count"].shape == (
        1,
    )
    assert bool(
        jnp.all(
            payload["stationarity_restoration_accepted_iteration_count"]
            <= payload["accepted_iteration_count"]
        )
    )
    assert bool(
        jnp.all(
            payload["second_order_correction_accepted_iteration_count"]
            <= payload["accepted_iteration_count"]
        )
    )
    assert bool(jnp.all(payload["line_search_soc_candidate_count"] <= 1))
    assert result.diagnostics.final_residual.shape == (1,)
    assert candidates["filter_f_type"].shape == candidates["alpha"].shape
    assert candidates["filter_armijo"].shape == candidates["alpha"].shape
    assert candidates["filter_history_accepted"].shape == candidates["alpha"].shape
    assert candidates["filter_entry_count_before"].shape == candidates["alpha"].shape
    assert candidates["soft_restoration_accepted"].shape == candidates["alpha"].shape


def test_fixed_support_batch_legacy_globalization_control_smoke(monkeypatch):
    monkeypatch.setenv("EXOGIBBS_FIXED_SUPPORT_BATCH_IPOPT_FILTER_ACCEPTANCE", "1")
    monkeypatch.setenv(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_IPOPT_FILTER_POLICY", "current_iterate"
    )
    monkeypatch.setenv("EXOGIBBS_FIXED_SUPPORT_BATCH_SECOND_ORDER_CORRECTION", "1")
    monkeypatch.setenv(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_POLICY", "legacy_budget_projection"
    )
    monkeypatch.setenv("EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_BUDGET_PASSES", "2")
    monkeypatch.setenv("EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_DUAL_REPAIR", "1")
    _, extra = condmod._solve_pdipm_rgie_v11_activity_correction_fixed_support_batch(
        ln_nk_init=jnp.log(jnp.asarray([[0.8]], dtype=jnp.float64)),
        ln_mk_init=jnp.log(jnp.asarray([[0.2]], dtype=jnp.float64)),
        ln_ntot_init=jnp.log(jnp.asarray([0.8], dtype=jnp.float64)),
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond_active=jnp.asarray([[1.0]], dtype=jnp.float64),
        element_inventory_target=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector=jnp.asarray([[0.0]], dtype=jnp.float64),
        hvector_cond_active=jnp.asarray([[0.0]], dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray([0.0], dtype=jnp.float64),
        epsilon=-10.0,
        max_iter=1,
        lambda_initialization="gas_lstsq",
    )
    payload = extra["pdipm_rgie_v11_activity_correction_fixed_support_batch"]

    assert payload["ipopt_filter_policy"] == "current_iterate"
    assert payload["second_order_correction_policy"] == "legacy_budget_projection"
    assert payload["second_order_correction_dual_repair"] is True
    assert int(payload["line_search_soc_candidate_count"][0]) == 26


def test_fixed_support_batch_ipopt_first_trial_soc_smoke(monkeypatch):
    monkeypatch.setenv("EXOGIBBS_FIXED_SUPPORT_BATCH_IPOPT_FILTER_ACCEPTANCE", "1")
    monkeypatch.setenv(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_IPOPT_FILTER_POLICY", "persistent_phi_theta"
    )
    monkeypatch.setenv(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_IPOPT_FILTER_THETA_NORM", "l1_scaled"
    )
    monkeypatch.setenv("EXOGIBBS_FIXED_SUPPORT_BATCH_SECOND_ORDER_CORRECTION", "1")
    monkeypatch.setenv("EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_POLICY", "ipopt_first_trial")
    monkeypatch.setenv("EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_BUDGET_PASSES", "4")
    monkeypatch.setenv("EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_KAPPA", "0.99")
    monkeypatch.setenv(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_SOC_ALPHA_Y_POLICY", "min_dual_infeas"
    )
    _, extra = condmod._solve_pdipm_rgie_v11_activity_correction_fixed_support_batch(
        ln_nk_init=jnp.log(jnp.asarray([[0.8]], dtype=jnp.float64)),
        ln_mk_init=jnp.log(jnp.asarray([[0.2]], dtype=jnp.float64)),
        ln_ntot_init=jnp.log(jnp.asarray([0.8], dtype=jnp.float64)),
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond_active=jnp.asarray([[1.0]], dtype=jnp.float64),
        element_inventory_target=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector=jnp.asarray([[0.0]], dtype=jnp.float64),
        hvector_cond_active=jnp.asarray([[0.0]], dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray([0.0], dtype=jnp.float64),
        epsilon=-10.0,
        max_iter=1,
        lambda_initialization="gas_lstsq",
    )
    payload = extra["pdipm_rgie_v11_activity_correction_fixed_support_batch"]

    assert payload["second_order_correction_policy"] == "ipopt_first_trial"
    assert payload["ipopt_filter_theta_norm"] == "l1_scaled"
    assert payload["second_order_correction_kappa_soc"] == 0.99
    assert payload["second_order_correction_alpha_y_policy"] == "min_dual_infeas"
    assert int(payload["line_search_soc_candidate_count"][0]) <= 1
    candidates = payload["line_search_candidate_diagnostics"]
    lifecycle_keys = (
        "ipopt_soc_eligible_iteration_count",
        "ipopt_soc_correction_count",
        "ipopt_soc_finite_direction_iteration_count",
        "ipopt_soc_filter_accepted_iteration_count",
        "ipopt_soc_kappa_stopped_iteration_count",
        "ipopt_soc_last_normal_theta",
        "ipopt_soc_last_final_theta",
        "ipopt_soc_last_normal_phi",
        "ipopt_soc_last_final_phi",
        "ipopt_soc_last_alpha",
        "ipopt_soc_max_solve_linear_residual",
        "ipopt_soc_max_solve_solution_norm",
        "ipopt_soc_min_solve_singular_value",
        "ipopt_soc_max_solve_condition_estimate",
        "ipopt_soc_max_scaled_solve_condition_estimate",
        "ipopt_soc_max_relative_solve_linear_residual",
        "ipopt_filter_consecutive_rejection_count",
        "ipopt_filter_reset_count",
        "ipopt_soc_first_selected_recorded",
        "ipopt_soc_first_selected_solution_norm",
        "ipopt_soc_first_selected_relative_linear_residual",
        "ipopt_soc_first_selected_condition_estimate",
        "ipopt_soc_first_selected_scaled_condition_estimate",
        "ipopt_soc_first_selected_smallest_singular_value",
        "ipopt_soc_first_selected_null_lambda_norm",
        "ipopt_soc_first_selected_null_qtot_abs",
        "ipopt_soc_first_selected_null_dominant_lambda_index",
        "ipopt_soc_first_selected_null_dominant_lambda_abs",
    )
    for key in lifecycle_keys:
        assert candidates[key].shape == candidates["alpha"].shape
    assert candidates["ipopt_soc_first_selected_diagnostic_mode_vector"].shape == (
        *candidates["alpha"].shape,
        25,
    )
    assert int(candidates["ipopt_soc_correction_count"][0, 0]) <= 4


def test_pdipm_rgie_v11_fixed_support_batch_residual_contract_two_element(monkeypatch):
    env_names = (
        "EXOGIBBS_FIXED_SUPPORT_BATCH_SECOND_ORDER_CORRECTION",
        "EXOGIBBS_FIXED_SUPPORT_BATCH_ADAPTIVE_REGULARIZATION",
        "EXOGIBBS_FIXED_SUPPORT_BATCH_USE_LEGACY_CAPACITY_EPSILON",
        "EXOGIBBS_FIXED_SUPPORT_BATCH_STEP_CONTROL",
        "EXOGIBBS_FIXED_SUPPORT_BATCH_SELECTION_POLICY",
    )
    for env_name in env_names:
        monkeypatch.delenv(env_name, raising=False)
    formula_matrix = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=jnp.float64,
    )
    formula_matrix_cond_active = jnp.asarray(
        [
            [1.0],
            [1.0],
        ],
        dtype=jnp.float64,
    )
    hvector = jnp.asarray([[0.1, -0.2]], dtype=jnp.float64)
    hvector_cond_active = jnp.asarray([[0.3]], dtype=jnp.float64)
    ln_normalized_pressure = jnp.asarray([0.05], dtype=jnp.float64)
    element_inventory_target = jnp.asarray([[1.0, 1.0]], dtype=jnp.float64)
    ln_nk_init = jnp.log(jnp.asarray([[0.75, 0.8]], dtype=jnp.float64))
    ln_mk_init = jnp.log(jnp.asarray([[0.2]], dtype=jnp.float64))
    ln_ntot_init = jnp.log(jnp.asarray([1.55], dtype=jnp.float64))

    result, extra = (
        condmod._solve_pdipm_rgie_v11_activity_correction_fixed_support_batch(
            ln_nk_init=ln_nk_init,
            ln_mk_init=ln_mk_init,
            ln_ntot_init=ln_ntot_init,
            formula_matrix=formula_matrix,
            formula_matrix_cond_active=formula_matrix_cond_active,
            element_inventory_target=element_inventory_target,
            hvector=hvector,
            hvector_cond_active=hvector_cond_active,
            ln_normalized_pressure=ln_normalized_pressure,
            epsilon=-10.0,
            max_iter=4,
            lambda_initialization="best_residual",
        )
    )

    payload = extra["pdipm_rgie_v11_activity_correction_fixed_support_batch"]
    q = result.ln_nk[0]
    r = result.ln_mk[0]
    qtot = result.ln_ntot[0]
    lam = payload["final_element_potential"][0]
    rho = payload["final_log_activity_correction"][0]
    epsilon = result.diagnostics.epsilon[0]
    gas_source = hvector[0] + ln_normalized_pressure[0] - qtot
    gas_component = q + gas_source - formula_matrix.T @ lam
    cond_component = (
        hvector_cond_active[0]
        - formula_matrix_cond_active.T @ lam
        - jnp.exp(rho)
    )
    budget_component = (
        formula_matrix @ jnp.exp(q)
        + formula_matrix_cond_active @ jnp.exp(r)
        - element_inventory_target[0]
    )
    complementarity_component = r + rho - epsilon
    total_density_component = jnp.asarray(
        [jnp.sum(jnp.exp(q)) - jnp.exp(qtot)],
        dtype=jnp.float64,
    )
    assert jnp.allclose(
        jnp.linalg.norm(gas_component),
        payload["gas_residual_norm"][0],
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert jnp.allclose(
        jnp.linalg.norm(cond_component),
        payload["condensate_stationarity_residual_norm"][0],
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert jnp.allclose(
        jnp.linalg.norm(budget_component),
        payload["budget_residual_norm"][0],
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert jnp.allclose(
        jnp.linalg.norm(complementarity_component),
        payload["complementarity_residual_norm"][0],
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert jnp.allclose(
        jnp.linalg.norm(total_density_component),
        payload["total_density_residual_norm"][0],
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_pdipm_rgie_v11_fixed_support_batch_legacy_epsilon_is_opt_in(monkeypatch):
    for env_name in (
        "EXOGIBBS_FIXED_SUPPORT_BATCH_SECOND_ORDER_CORRECTION",
        "EXOGIBBS_FIXED_SUPPORT_BATCH_ADAPTIVE_REGULARIZATION",
        "EXOGIBBS_FIXED_SUPPORT_BATCH_STEP_CONTROL",
        "EXOGIBBS_FIXED_SUPPORT_BATCH_SELECTION_POLICY",
    ):
        monkeypatch.delenv(env_name, raising=False)
    formula_matrix = jnp.asarray([[1.0]], dtype=jnp.float64)
    formula_matrix_cond_active = jnp.asarray([[1.0]], dtype=jnp.float64)
    hvector = jnp.asarray([[0.0]], dtype=jnp.float64)
    hvector_cond_active = jnp.asarray([[0.0]], dtype=jnp.float64)
    element_inventory_target = jnp.asarray([[2.0]], dtype=jnp.float64)
    ln_normalized_pressure = jnp.asarray([0.0], dtype=jnp.float64)
    ln_nk_init = jnp.log(jnp.asarray([[1.5]], dtype=jnp.float64))
    ln_mk_init = jnp.log(jnp.asarray([[0.4]], dtype=jnp.float64))
    ln_ntot_init = jnp.log(jnp.asarray([1.5], dtype=jnp.float64))

    monkeypatch.delenv(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_USE_LEGACY_CAPACITY_EPSILON",
        raising=False,
    )
    default_result, default_extra = (
        condmod._solve_pdipm_rgie_v11_activity_correction_fixed_support_batch(
            ln_nk_init=ln_nk_init,
            ln_mk_init=ln_mk_init,
            ln_ntot_init=ln_ntot_init,
            formula_matrix=formula_matrix,
            formula_matrix_cond_active=formula_matrix_cond_active,
            element_inventory_target=element_inventory_target,
            hvector=hvector,
            hvector_cond_active=hvector_cond_active,
            ln_normalized_pressure=ln_normalized_pressure,
            epsilon=-10.0,
            max_iter=1,
            lambda_initialization="best_residual",
        )
    )
    default_payload = default_extra[
        "pdipm_rgie_v11_activity_correction_fixed_support_batch"
    ]
    default_comp = (
        default_result.ln_mk
        + default_payload["final_log_activity_correction"]
        - default_result.diagnostics.epsilon[:, None]
    )
    assert default_payload["use_legacy_capacity_epsilon"] is False
    assert jnp.allclose(
        jnp.linalg.norm(default_comp, axis=1),
        default_payload["complementarity_residual_norm"],
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    monkeypatch.setenv(
        "EXOGIBBS_FIXED_SUPPORT_BATCH_USE_LEGACY_CAPACITY_EPSILON",
        "1",
    )
    legacy_result, legacy_extra = (
        condmod._solve_pdipm_rgie_v11_activity_correction_fixed_support_batch(
            ln_nk_init=ln_nk_init,
            ln_mk_init=ln_mk_init,
            ln_ntot_init=ln_ntot_init,
            formula_matrix=formula_matrix,
            formula_matrix_cond_active=formula_matrix_cond_active,
            element_inventory_target=element_inventory_target,
            hvector=hvector,
            hvector_cond_active=hvector_cond_active,
            ln_normalized_pressure=ln_normalized_pressure,
            epsilon=-10.0,
            max_iter=1,
            lambda_initialization="best_residual",
        )
    )
    legacy_payload = legacy_extra[
        "pdipm_rgie_v11_activity_correction_fixed_support_batch"
    ]
    legacy_epsilon = jnp.log(
        jnp.maximum(1.0e-15 * element_inventory_target[:, 0], 1.0e-300)
    )
    legacy_comp = (
        legacy_result.ln_mk
        + legacy_payload["final_log_activity_correction"]
        - legacy_epsilon[:, None]
    )
    requested_comp = (
        legacy_result.ln_mk
        + legacy_payload["final_log_activity_correction"]
        - legacy_result.diagnostics.epsilon[:, None]
    )
    assert legacy_payload["use_legacy_capacity_epsilon"] is True
    assert jnp.allclose(
        jnp.linalg.norm(legacy_comp, axis=1),
        legacy_payload["complementarity_residual_norm"],
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert not jnp.allclose(
        jnp.linalg.norm(requested_comp, axis=1),
        legacy_payload["complementarity_residual_norm"],
        rtol=1.0e-6,
        atol=1.0e-8,
    )


def test_pdipm_rgie_v11_fixed_support_batch_continuation_smoke(monkeypatch):
    for env_name in (
        "EXOGIBBS_FIXED_SUPPORT_BATCH_SECOND_ORDER_CORRECTION",
        "EXOGIBBS_FIXED_SUPPORT_BATCH_ADAPTIVE_REGULARIZATION",
        "EXOGIBBS_FIXED_SUPPORT_BATCH_USE_LEGACY_CAPACITY_EPSILON",
        "EXOGIBBS_FIXED_SUPPORT_BATCH_STEP_CONTROL",
        "EXOGIBBS_FIXED_SUPPORT_BATCH_SELECTION_POLICY",
    ):
        monkeypatch.delenv(env_name, raising=False)
    result, extra = (
        condmod._solve_pdipm_rgie_v11_activity_correction_fixed_support_batch_continuation(
            ln_nk_init=jnp.log(jnp.asarray([[0.8]], dtype=jnp.float64)),
            ln_mk_init=jnp.log(jnp.asarray([[0.2]], dtype=jnp.float64)),
            ln_ntot_init=jnp.log(jnp.asarray([0.8], dtype=jnp.float64)),
            formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
            formula_matrix_cond_active=jnp.asarray([[1.0]], dtype=jnp.float64),
            element_inventory_target=jnp.asarray([[1.0]], dtype=jnp.float64),
            hvector=jnp.asarray([[0.0]], dtype=jnp.float64),
            hvector_cond_active=jnp.asarray([[0.0]], dtype=jnp.float64),
            ln_normalized_pressure=jnp.asarray([0.0], dtype=jnp.float64),
            epsilon_schedule=(-2.0, -4.0, -6.0, -8.0, -10.0),
            max_iter=1,
            lambda_initialization="best_residual",
        )
    )
    continuation = extra[
        "pdipm_rgie_v11_activity_correction_fixed_support_batch_continuation"
    ]
    stage_epsilons = tuple(stage["epsilon"] for stage in continuation["stages"])
    assert continuation["stage_count"] == 5
    assert continuation["completed_stage_count"] == 1
    assert continuation["stopped_early"] is True
    assert continuation["blocked_stage_index"] == 0
    assert continuation["blocked_epsilon"] == -2.0
    assert stage_epsilons == (-2.0, -4.0, -6.0, -8.0, -10.0)
    assert tuple(stage["attempted"] for stage in continuation["stages"]) == (
        True,
        False,
        False,
        False,
        False,
    )
    assert tuple(
        bool(jnp.all(stage["centered_for_continuation"]))
        for stage in continuation["stages"][1:]
    ) == (False, False, False, False)
    assert continuation["continuation_tolerance_multiplier"] == 3.0
    assert continuation["effective_continuation_tolerance_multiplier"] == 3.0
    assert "continuation_log_tolerance" in continuation["stages"][0]
    assert jnp.all(result.diagnostics.epsilon == -2.0)
    assert result.ln_nk.shape == (1, 1)
    assert result.ln_mk.shape == (1, 1)
    assert (
        extra["pdipm_rgie_v11_activity_correction_fixed_support_batch"][
            "rho_initialization"
        ]
        == "unit_activity"
    )


def test_fixed_support_continuation_advances_layers_independently(monkeypatch):
    calls = []

    def stub_stage_solver(*, ln_nk_init, epsilon, **_kwargs):
        calls.append(float(epsilon))
        batch_size = ln_nk_init.shape[0]
        if float(epsilon) == -2.0:
            gas_norm = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
            converged = jnp.asarray([True, False])
        else:
            gas_norm = jnp.zeros((batch_size,), dtype=jnp.float64)
            converged = jnp.ones((batch_size,), dtype=bool)
        state_value = jnp.full_like(ln_nk_init, float(epsilon))
        diagnostics = condmod.CondensateEquilibriumDiagnostics.from_mapping(
            {
                "n_iter": jnp.ones((batch_size,), dtype=jnp.int32),
                "converged": converged,
                "hit_max_iter": ~converged,
                "final_residual": gas_norm,
                "residual_crit": jnp.full((batch_size,), 0.1),
                "max_iter": jnp.full((batch_size,), 1, dtype=jnp.int32),
                "epsilon": jnp.full((batch_size,), float(epsilon)),
                "final_step_size": jnp.ones((batch_size,)),
                "invalid_numbers_detected": jnp.zeros((batch_size,), dtype=bool),
                "debug_nan": jnp.zeros((batch_size,), dtype=bool),
            }
        )
        payload = {
            "convergence_log_tolerance": 0.1,
            "convergence_budget_relative_tolerance": 0.1,
            "convergence_total_density_tolerance": 0.1,
            "gas_residual_norm": gas_norm,
            "condensate_stationarity_residual_norm": jnp.zeros((batch_size,)),
            "budget_relative_residual_max": jnp.zeros((batch_size,)),
            "complementarity_residual_norm": jnp.zeros((batch_size,)),
            "total_density_residual_norm": jnp.zeros((batch_size,)),
            "accepted_iteration_count": jnp.ones((batch_size,), dtype=jnp.int32),
            "rejected_trial_count": jnp.zeros((batch_size,), dtype=jnp.int32),
            "final_step_size": jnp.ones((batch_size,)),
            "stop_reason_code": jnp.where(converged, 0, 1),
            "dominant_residual_component_index": jnp.zeros(
                (batch_size,), dtype=jnp.int32
            ),
            "final_element_potential": jnp.full((batch_size, 1), float(epsilon)),
            "final_log_activity_correction": jnp.full(
                (batch_size, 1), float(epsilon)
            ),
        }
        return (
            condmod.CondensateEquilibriumResult(
                ln_nk=state_value,
                ln_mk=state_value,
                ln_ntot=state_value[:, 0],
                diagnostics=diagnostics,
            ),
            {"pdipm_rgie_v11_activity_correction_fixed_support_batch": payload},
        )

    monkeypatch.setattr(
        condmod,
        "_solve_pdipm_rgie_v11_activity_correction_fixed_support_batch",
        stub_stage_solver,
    )
    result, extra = (
        condmod._solve_pdipm_rgie_v11_activity_correction_fixed_support_batch_continuation(
            ln_nk_init=jnp.zeros((2, 1)),
            ln_mk_init=jnp.zeros((2, 1)),
            ln_ntot_init=jnp.zeros((2,)),
            formula_matrix=jnp.ones((1, 1)),
            formula_matrix_cond_active=jnp.ones((1, 1)),
            element_inventory_target=jnp.ones((2, 1)),
            hvector=jnp.zeros((2, 1)),
            hvector_cond_active=jnp.zeros((2, 1)),
            ln_normalized_pressure=jnp.zeros((2,)),
            epsilon_schedule=(-2.0, -4.0),
            max_iter=1,
        )
    )
    continuation = extra[
        "pdipm_rgie_v11_activity_correction_fixed_support_batch_continuation"
    ]
    final_payload = extra[
        "pdipm_rgie_v11_activity_correction_fixed_support_batch"
    ]

    assert calls == [-2.0, -4.0]
    assert continuation["stages"][0]["attempted"].tolist() == [True, True]
    assert continuation["stages"][1]["attempted"].tolist() == [True, False]
    assert continuation["layer_completed_stage_count"].tolist() == [2, 1]
    assert continuation["reached_final_epsilon"].tolist() == [True, False]
    assert result.ln_nk[:, 0].tolist() == [-4.0, -2.0]
    assert result.diagnostics.epsilon.tolist() == [-4.0, -2.0]
    assert result.diagnostics.converged.tolist() == [True, False]
    assert result.diagnostics.reached_requested_epsilon.tolist() == [True, False]
    assert final_payload["gas_residual_norm"].tolist() == [0.0, 1.0]


def test_pdipm_rgie_v11_prepared_profile_buckets_return_budget_consistent_layers():
    states = tuple(
        ThermoState(
            temperature=jnp.asarray(1000.0 + 10.0 * index, dtype=jnp.float64),
            ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
            element_vector=jnp.asarray([1.0], dtype=jnp.float64),
        )
        for index in range(3)
    )
    init_states = (
        condmod.CondensateEquilibriumInit(
            ln_nk=jnp.log(jnp.asarray([0.8], dtype=jnp.float64)),
            ln_mk=jnp.log(jnp.asarray([0.2], dtype=jnp.float64)),
            ln_ntot=jnp.log(jnp.asarray(0.8, dtype=jnp.float64)),
        ),
        condmod.CondensateEquilibriumInit(
            ln_nk=jnp.log(jnp.asarray([0.7], dtype=jnp.float64)),
            ln_mk=jnp.log(jnp.asarray([0.3], dtype=jnp.float64)),
            ln_ntot=jnp.log(jnp.asarray(0.7, dtype=jnp.float64)),
        ),
        condmod.CondensateEquilibriumInit(
            ln_nk=jnp.log(jnp.asarray([0.6], dtype=jnp.float64)),
            ln_mk=jnp.log(jnp.asarray([0.4], dtype=jnp.float64)),
            ln_ntot=jnp.log(jnp.asarray(0.6, dtype=jnp.float64)),
        ),
    )
    formula_matrix = jnp.asarray([[1.0]], dtype=jnp.float64)
    formula_matrix_cond = jnp.asarray([[1.0, 1.0]], dtype=jnp.float64)
    support_indices_by_layer = ((0,), (0,), (1,))
    buckets = condmod._prepare_pdipm_rgie_v11_activity_correction_profile_buckets(
        states=states,
        init_states=init_states,
        support_indices_by_layer=support_indices_by_layer,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray(
            [0.0, 0.0],
            dtype=jnp.float64,
        ),
    )

    bucket_results, trace = (
        condmod._run_pdipm_rgie_v11_activity_correction_prepared_profile_buckets(
            buckets=buckets,
            formula_matrix=formula_matrix,
            epsilon=-10.0,
            max_iter=2,
            lambda_initialization="gas_lstsq",
        )
    )

    result_by_layer = {}
    for bucket, result in zip(buckets, bucket_results):
        for local_index, layer_index in enumerate(bucket.layer_indices):
            result_by_layer[layer_index] = (
                result.ln_nk[local_index],
                result.ln_mk[local_index],
                result.ln_ntot[local_index],
                result.diagnostics.final_residual[local_index],
            )

    assert set(result_by_layer) == {0, 1, 2}
    for index, support_indices in enumerate(support_indices_by_layer):
        support_array = jnp.asarray(support_indices, dtype=jnp.int32)
        ln_nk, ln_mk, ln_ntot, final_residual = result_by_layer[index]
        budget_residual = (
            formula_matrix @ jnp.exp(ln_nk)
            + formula_matrix_cond[:, support_array] @ jnp.exp(ln_mk)
            - states[index].element_vector
        )
        assert jnp.all(jnp.isfinite(ln_nk))
        assert jnp.all(jnp.isfinite(ln_mk))
        assert jnp.all(jnp.isfinite(ln_ntot))
        assert float(jnp.max(jnp.abs(budget_residual))) <= 1.01e-3
        assert float(final_residual) >= 0.0

    payload = trace["pdipm_rgie_v11_activity_correction_prepared_profile_buckets"]
    assert payload["experimental"] is True
    assert payload["production_route_wiring"] is False
    assert payload["bucket_count"] == 2
    assert payload["layer_count"] == 3


def test_hybrid_candidate_log_activity_proxy_bookkeeping():
    formula_matrix_cond = jnp.asarray(
        [[1.0, 0.0], [2.0, 1.0]],
        dtype=jnp.float64,
    )
    pi_g = jnp.asarray([0.5, 1.0], dtype=jnp.float64)
    h_cond = jnp.asarray([1.0, 0.25], dtype=jnp.float64)

    proxy = compute_hybrid_candidate_log_activity_proxy(
        formula_matrix_cond,
        pi_g,
        h_cond,
    )

    assert jnp.allclose(proxy, jnp.asarray([1.5, 0.75], dtype=jnp.float64))


def test_hybrid_candidate_active_and_near_active_masks():
    masks = build_hybrid_candidate_masks(
        jnp.asarray([0.2, 0.0, -0.05, -0.2], dtype=jnp.float64)
    )

    assert masks["active_bool"].tolist() == [True, True, False, False]
    assert masks["near_active_bool"].tolist() == [True, True, True, False]
    assert jnp.allclose(masks["active"], jnp.asarray([1.0, 1.0, 0.0, 0.0]))
    assert jnp.allclose(masks["near_active"], jnp.asarray([1.0, 1.0, 1.0, 0.0]))


def test_hybrid_candidate_rem_inventory_applies_correctvalues_update():
    epsilon = -5.0
    direction = solve_hybrid_candidate_selected_reduced_coupling_direction(
        ln_nk=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        ln_mk=jnp.log(jnp.asarray([0.1, 0.2], dtype=jnp.float64)),
        ln_ntot=jnp.log(jnp.asarray(2.0, dtype=jnp.float64)),
        formula_matrix=jnp.eye(2, dtype=jnp.float64),
        formula_matrix_cond=jnp.eye(2, dtype=jnp.float64),
        b=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        gk=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        hvector_cond=jnp.asarray([0.0, 0.2], dtype=jnp.float64),
        epsilon=epsilon,
        candidate_mode="candidate_selected_active_plus_near_jacobian_with_rem_inventory",
    )

    assert direction["near_active_bool"].tolist() == [True, False]
    assert jnp.allclose(direction["m_rem"], jnp.asarray([0.0, 0.2], dtype=jnp.float64))
    assert jnp.allclose(direction["b_solver"], jnp.asarray([1.0, 0.8], dtype=jnp.float64))
    expected_s_rem = jnp.asarray([0.0, 0.2 * 0.2 / jnp.exp(epsilon)], dtype=jnp.float64)
    assert jnp.allclose(direction["s_rem"], expected_s_rem)
    assert jnp.allclose(direction["s_solve"], direction["s_near"] + direction["s_rem"])
    assert float(direction["raw_delta_ln_mk"][1]) != pytest.approx(0.0)
    assert int(direction["rem_inventory_set_size"]) == 1
    assert bool(direction["rem_correctvalues_update_enabled"]) is True
    assert float(direction["rem_correctvalues_max_abs_delta_ln_mk"]) > 0.0


def test_gas_recoupling_replay_bookkeeping_terms():
    terms = compute_condensed_element_gas_recoupling_terms(
        jnp.asarray([[1.0, 2.0], [0.0, 1.0]], dtype=jnp.float64),
        jnp.asarray([0.1, 0.2], dtype=jnp.float64),
        jnp.asarray([1.0, 0.5], dtype=jnp.float64),
    )

    assert jnp.allclose(terms["d_elem"], jnp.asarray([0.5, 0.2], dtype=jnp.float64))
    assert jnp.allclose(terms["b_eff"], jnp.asarray([0.5, 0.3], dtype=jnp.float64))
    assert jnp.allclose(terms["phi"], jnp.asarray([0.5, 0.4], dtype=jnp.float64))


def test_internal_complementarity_tau_bookkeeping():
    tau = build_internal_complementarity_tau(
        jnp.asarray([0, 2, 4], dtype=jnp.int32),
        epsilon=-5.0,
        tau_scale=2.0,
    )

    assert tau.shape == (3,)
    assert jnp.allclose(tau, 2.0 * jnp.exp(jnp.asarray(-5.0, dtype=jnp.float64)))


def test_internal_complementarity_residual_construction():
    q = jnp.log(jnp.asarray([0.6, 0.4], dtype=jnp.float64))
    r_c = jnp.log(jnp.asarray([0.1], dtype=jnp.float64))
    chi_c = jnp.log(jnp.asarray([0.2], dtype=jnp.float64))
    tau_c = jnp.asarray([0.02], dtype=jnp.float64)
    pi = jnp.asarray([1.0], dtype=jnp.float64)
    q_tot = jnp.asarray(0.0, dtype=jnp.float64)

    residual = compute_internal_complementarity_residual(
        q,
        r_c,
        chi_c,
        pi,
        q_tot,
        formula_matrix=jnp.asarray([[1.0, 1.0]], dtype=jnp.float64),
        formula_matrix_cond_c=jnp.asarray([[1.0]], dtype=jnp.float64),
        b=jnp.asarray([1.1], dtype=jnp.float64),
        hvector_gas=jnp.asarray([1.0, 1.0], dtype=jnp.float64)
        - q
        + q_tot,
        hvector_cond_c=jnp.asarray([1.2], dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        tau_c=tau_c,
    )

    assert jnp.allclose(residual["element_conservation"], jnp.asarray([0.0]))
    assert jnp.allclose(residual["total_number_closure"], jnp.asarray([0.0]))
    assert jnp.allclose(residual["activity_complementarity"], jnp.asarray([0.0]))
    assert jnp.allclose(residual["fixed_tau_complementarity"], jnp.asarray([0.0]))
    assert residual["flat"].shape == (6,)


def test_kl_atomic_gas_reconstruction_uses_element_species_first():
    u = jnp.log(jnp.asarray([0.2, 0.3], dtype=jnp.float64))
    formula_matrix_gas = jnp.asarray(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    hvector_gas = jnp.asarray([0.0, 0.0, -0.5], dtype=jnp.float64)

    gas = reconstruct_kl_atomic_gas_from_u(u, formula_matrix_gas, hvector_gas)

    expected_molecule = jnp.exp(0.5) * 0.2 * 0.3 * 0.3
    assert jnp.allclose(gas["nk"][:2], jnp.asarray([0.2, 0.3], dtype=jnp.float64))
    assert jnp.allclose(gas["nk"][2], expected_molecule)
    assert jnp.allclose(jnp.exp(gas["ln_ntot"]), jnp.sum(gas["nk"]))


def test_kl_condensate_log_activity_and_masks_use_atomic_density():
    u = jnp.log(jnp.asarray([0.4, 0.25], dtype=jnp.float64))
    formula_matrix_cond = jnp.asarray(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=jnp.float64,
    )
    hvector_cond = jnp.asarray([-3.0, -1.0, 2.0], dtype=jnp.float64)

    ell = compute_kl_condensate_log_activity(u, formula_matrix_cond, hvector_cond)
    masks = build_kl_atomic_candidate_masks(ell)

    assert jnp.allclose(
        ell,
        jnp.asarray(
            [
                3.0 + jnp.log(0.4) + jnp.log(0.25),
                1.0 + jnp.log(0.4),
                -2.0 + jnp.log(0.25),
            ],
            dtype=jnp.float64,
        ),
    )
    assert masks["active_bool"].tolist() == [True, True, False]
    assert masks["near_active_bool"].tolist() == [True, True, False]


def test_kl_atomic_complementarity_residual_bookkeeping():
    u = jnp.log(jnp.asarray([0.2], dtype=jnp.float64))
    r_c = jnp.log(jnp.asarray([0.1], dtype=jnp.float64))
    chi_c = jnp.log(jnp.asarray([0.4], dtype=jnp.float64))
    tau_c = jnp.asarray([0.04], dtype=jnp.float64)
    residual = compute_kl_atomic_complementarity_residual(
        u,
        r_c,
        chi_c,
        formula_matrix_gas=jnp.asarray([[1.0, 2.0]], dtype=jnp.float64),
        formula_matrix_cond_c=jnp.asarray([[1.0]], dtype=jnp.float64),
        b=jnp.asarray([0.38], dtype=jnp.float64),
        hvector_gas=jnp.asarray([0.0, -0.5], dtype=jnp.float64),
        hvector_cond_c=jnp.asarray([jnp.log(0.2) + 0.4], dtype=jnp.float64),
        tau_c=tau_c,
    )

    molecule = jnp.exp(0.5) * 0.2 * 0.2
    expected_element = 0.38 - 0.2 - 2.0 * molecule - 0.1
    assert jnp.allclose(residual["element_conservation"], jnp.asarray([expected_element]))
    assert jnp.allclose(residual["activity_slack"], jnp.asarray([0.0]))
    assert jnp.allclose(residual["fixed_tau_complementarity"], jnp.asarray([0.0]))
    assert residual["flat"].shape == (3,)


def test_minimize_gibbs_cond_candidate_selected_branch_dispatches(monkeypatch):
    captured = {}

    def stub_raw(state, ln_nk_init, ln_mk_init, ln_ntot_init, **kwargs):
        del state
        captured["reduced_coupling_mode"] = kwargs["reduced_coupling_mode"]
        captured["gas_step_scale"] = kwargs["gas_step_scale"]
        captured["gas_step_direction_sign"] = kwargs["gas_step_direction_sign"]
        captured["ntot_step_scale"] = kwargs["ntot_step_scale"]
        captured["condensate_step_scale"] = kwargs["condensate_step_scale"]
        captured["initial_residual_policy"] = kwargs["initial_residual_policy"]
        captured["ln_mk_init"] = ln_mk_init
        return (
            ln_nk_init,
            ln_mk_init,
            ln_ntot_init,
            {
                "n_iter": jnp.asarray(0, dtype=jnp.int32),
                "converged": jnp.asarray(True),
                "hit_max_iter": jnp.asarray(False),
                "final_residual": jnp.asarray(0.0, dtype=jnp.float64),
                "residual_crit": jnp.asarray(kwargs["residual_crit"], dtype=jnp.float64),
                "max_iter": jnp.asarray(kwargs["max_iter"], dtype=jnp.int32),
                "epsilon": jnp.asarray(kwargs["epsilon"], dtype=jnp.float64),
                "final_step_size": jnp.asarray(0.0, dtype=jnp.float64),
                "invalid_numbers_detected": jnp.asarray(False),
                "debug_nan": jnp.asarray(False),
            },
        )

    monkeypatch.setattr(condmod, "_minimize_gibbs_cond_with_diagnostics_raw", stub_raw)
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
        reduced_coupling_config=condmod.CondensateRGIEReducedCouplingConfig(
            reduced_coupling_mode="candidate_selected_active_plus_near_jacobian",
            gas_step_scale=0.01,
            gas_step_direction_sign=-1.0,
            ntot_step_scale=0.02,
            condensate_step_scale=0.5,
            initial_residual_policy="computed_fresh",
        ),
    )

    assert captured["reduced_coupling_mode"] == "candidate_selected_active_plus_near_jacobian"
    assert captured["gas_step_scale"] == pytest.approx(0.01)
    assert captured["gas_step_direction_sign"] == pytest.approx(-1.0)
    assert captured["ntot_step_scale"] == pytest.approx(0.02)
    assert captured["condensate_step_scale"] == pytest.approx(0.5)
    assert captured["initial_residual_policy"] == "computed_fresh"


def test_minimize_gibbs_cond_profile_passes_support_method(monkeypatch):
    captured = []

    def stub_minimize_gibbs_cond(state, init, **kwargs):
        del state, init
        captured.append(kwargs["support_method"])
        return condmod.CondensateEquilibriumResult(
            ln_nk=jnp.asarray([0.0], dtype=jnp.float64),
            ln_mk=jnp.asarray([-1.0], dtype=jnp.float64),
            ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
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
                debug_nan=jnp.asarray(False),
            ),
        )

    monkeypatch.setattr(condmod, "minimize_gibbs_cond", stub_minimize_gibbs_cond)

    condmod.minimize_gibbs_cond_profile(
        temperatures=jnp.asarray([1000.0], dtype=jnp.float64),
        ln_normalized_pressures=jnp.asarray([0.0], dtype=jnp.float64),
        element_vector=jnp.asarray([1.0], dtype=jnp.float64),
        init=condmod.CondensateEquilibriumInit(
            ln_nk=jnp.asarray([0.0], dtype=jnp.float64),
            ln_mk=jnp.asarray([-10.0], dtype=jnp.float64),
            ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        ),
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        n_step=1,
        support_method="smoothed_semismooth_outer",
    )

    assert captured == ["smoothed_semismooth_outer", "smoothed_semismooth_outer"]


def test_minimize_gibbs_cond_ratio_uniform_startup_overrides_ln_mk(monkeypatch):
    captured = {}

    def stub_raw(state, ln_nk_init, ln_mk_init, ln_ntot_init, **kwargs):
        del state, ln_nk_init, ln_ntot_init, kwargs
        captured["ln_mk_init"] = ln_mk_init
        return (
            jnp.asarray([0.1], dtype=jnp.float64),
            jnp.asarray([0.2, 0.3], dtype=jnp.float64),
            jnp.asarray(0.4, dtype=jnp.float64),
            {
                "n_iter": jnp.asarray(0, dtype=jnp.int32),
                "converged": jnp.asarray(False),
                "hit_max_iter": jnp.asarray(True),
                "final_residual": jnp.asarray(1.0, dtype=jnp.float64),
                "residual_crit": jnp.asarray(1.0e-8, dtype=jnp.float64),
                "max_iter": jnp.asarray(0, dtype=jnp.int32),
                "epsilon": jnp.asarray(-5.0, dtype=jnp.float64),
                "final_step_size": jnp.asarray(0.0, dtype=jnp.float64),
                "invalid_numbers_detected": jnp.asarray(False),
                "debug_nan": jnp.asarray(False),
            },
        )

    monkeypatch.setattr(condmod, "_minimize_gibbs_cond_with_diagnostics_raw", stub_raw)

    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.asarray([1.0], dtype=jnp.float64),
    )

    condmod.minimize_gibbs_cond(
        state,
        init=condmod.CondensateEquilibriumInit(
            ln_nk=jnp.asarray([1.0], dtype=jnp.float64),
            ln_mk=jnp.asarray([-20.0, -19.0], dtype=jnp.float64),
            ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        ),
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0, 0.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        epsilon=-5.0,
        residual_crit=1.0e-8,
        max_iter=0,
        startup_config=condmod.CondensateRGIEStartupConfig(
            policy="ratio_uniform_r0",
            r0=1.0e-3,
        ),
    )

    expected = -5.0 + jnp.log(jnp.asarray(1.0e-3, dtype=jnp.float64))
    assert jnp.allclose(captured["ln_mk_init"], expected)


def test_minimize_gibbs_cond_warm_previous_with_ratio_floor_applies_floor(monkeypatch):
    captured = {}

    def stub_raw(state, ln_nk_init, ln_mk_init, ln_ntot_init, **kwargs):
        del state, ln_nk_init, ln_ntot_init, kwargs
        captured["ln_mk_init"] = ln_mk_init
        return (
            jnp.asarray([0.1], dtype=jnp.float64),
            ln_mk_init,
            jnp.asarray(0.4, dtype=jnp.float64),
            {
                "n_iter": jnp.asarray(0, dtype=jnp.int32),
                "converged": jnp.asarray(False),
                "hit_max_iter": jnp.asarray(True),
                "final_residual": jnp.asarray(1.0, dtype=jnp.float64),
                "residual_crit": jnp.asarray(1.0e-8, dtype=jnp.float64),
                "max_iter": jnp.asarray(0, dtype=jnp.int32),
                "epsilon": jnp.asarray(-5.0, dtype=jnp.float64),
                "final_step_size": jnp.asarray(0.0, dtype=jnp.float64),
                "invalid_numbers_detected": jnp.asarray(False),
                "debug_nan": jnp.asarray(False),
            },
        )

    monkeypatch.setattr(condmod, "_minimize_gibbs_cond_with_diagnostics_raw", stub_raw)

    state = ThermoState(
        temperature=jnp.asarray(1000.0, dtype=jnp.float64),
        ln_normalized_pressure=jnp.asarray(0.0, dtype=jnp.float64),
        element_vector=jnp.asarray([1.0], dtype=jnp.float64),
    )

    condmod.minimize_gibbs_cond(
        state,
        init=condmod.CondensateEquilibriumInit(
            ln_nk=jnp.asarray([1.0], dtype=jnp.float64),
            ln_mk=jnp.asarray([-20.0, -2.0], dtype=jnp.float64),
            ln_ntot=jnp.asarray(0.0, dtype=jnp.float64),
        ),
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0, 0.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        epsilon=-5.0,
        residual_crit=1.0e-8,
        max_iter=0,
        startup_config=condmod.CondensateRGIEStartupConfig(
            policy="warm_previous_with_ratio_floor",
            r0=1.0e-3,
        ),
    )

    floor_value = -5.0 + jnp.log(jnp.asarray(1.0e-3, dtype=jnp.float64))
    assert jnp.allclose(
        captured["ln_mk_init"],
        jnp.asarray([floor_value, -2.0], dtype=jnp.float64),
    )


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


def test_minimize_gibbs_cond_profile_scan_hot_from_bottom_applies_startup_policy_hook(monkeypatch):
    captured_ln_mk = []

    def stub_minimize_gibbs_cond(state, init, **kwargs):
        del state, kwargs
        captured_ln_mk.append(jnp.asarray(init.ln_mk))
        next_ln_mk = (
            jnp.asarray([-20.0], dtype=jnp.float64)
            if len(captured_ln_mk) == 1
            else jnp.asarray(init.ln_mk)
        )
        return condmod.CondensateEquilibriumResult(
            ln_nk=jnp.asarray(init.ln_nk),
            ln_mk=next_ln_mk,
            ln_ntot=jnp.asarray(init.ln_ntot),
            diagnostics=condmod.CondensateEquilibriumDiagnostics(
                n_iter=jnp.asarray(1, dtype=jnp.int32),
                converged=jnp.asarray(True),
                hit_max_iter=jnp.asarray(False),
                final_residual=jnp.asarray(1.0e-12, dtype=jnp.float64),
                residual_crit=jnp.asarray(1.0e-8, dtype=jnp.float64),
                max_iter=jnp.asarray(1, dtype=jnp.int32),
                epsilon=jnp.asarray(-5.0, dtype=jnp.float64),
                final_step_size=jnp.asarray(0.25, dtype=jnp.float64),
                invalid_numbers_detected=jnp.asarray(False),
                debug_nan=jnp.asarray(False),
            ),
        )

    monkeypatch.setattr(condmod, "minimize_gibbs_cond", stub_minimize_gibbs_cond)

    condmod.minimize_gibbs_cond_profile(
        temperatures=jnp.asarray([1000.0, 1100.0], dtype=jnp.float64),
        ln_normalized_pressures=jnp.asarray([-1.0, 0.0], dtype=jnp.float64),
        element_vector=jnp.asarray([1.0], dtype=jnp.float64),
        init=condmod.CondensateEquilibriumInit(
            ln_nk=jnp.asarray([[10.0], [20.0]], dtype=jnp.float64),
            ln_mk=jnp.asarray([[-30.0], [-30.0]], dtype=jnp.float64),
            ln_ntot=jnp.asarray([100.0, 200.0], dtype=jnp.float64),
        ),
        formula_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        formula_matrix_cond=jnp.asarray([[1.0]], dtype=jnp.float64),
        hvector_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        hvector_cond_func=lambda temperature: jnp.asarray([0.0], dtype=jnp.float64),
        epsilon_start=0.0,
        epsilon_crit=-5.0,
        n_step=1,
        max_iter=1,
        method="scan_hot_from_bottom_final_only",
        epsilon_schedule="adaptive_sk_guard",
        startup_config=condmod.CondensateRGIEStartupConfig(
            policy="warm_previous_with_ratio_floor",
            r0=1.0e-3,
        ),
    )

    startup_floor = jnp.log(jnp.asarray(1.0e-3, dtype=jnp.float64))
    final_only_floor = -5.0 + jnp.log(jnp.asarray(1.0e-3, dtype=jnp.float64))
    assert jnp.allclose(captured_ln_mk[0], jnp.asarray([startup_floor], dtype=jnp.float64))
    assert jnp.allclose(captured_ln_mk[-1], jnp.asarray([final_only_floor], dtype=jnp.float64))


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
