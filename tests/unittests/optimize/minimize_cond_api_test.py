import jax.numpy as jnp
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
        ),
    )

    assert captured["reduced_coupling_mode"] == "candidate_selected_active_plus_near_jacobian"


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
