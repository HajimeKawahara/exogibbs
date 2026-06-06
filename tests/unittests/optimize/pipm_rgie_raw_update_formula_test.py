import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from exogibbs.optimize.pipm_rgie_cond import build_rgie_condensate_direction_variant
from exogibbs.optimize.pipm_rgie_cond import build_rgie_condensate_direction_transform_variant
from exogibbs.optimize.pipm_rgie_cond import build_rgie_gas_direction_variant
from exogibbs.optimize.pipm_rgie_cond import compute_rgie_lam1_gas_ignore_trace_diagnostics
from exogibbs.optimize.pipm_rgie_cond import compute_rgie_lambda_cap_policy
from exogibbs.optimize.pipm_rgie_cond import diagnose_rgie_raw_condensate_update_block


def test_diagnose_rgie_raw_condensate_update_block_matches_stationarity_over_nu():
    ln_mk = jnp.log(jnp.asarray([2.0, 5.0], dtype=jnp.float64))
    epsilon = -2.0
    formula_matrix_cond = jnp.asarray([[1.0, 2.0], [0.5, -1.0]], dtype=jnp.float64)
    pi_vector = jnp.asarray([0.7, -0.3], dtype=jnp.float64)
    hvector_cond = jnp.asarray([0.2, -1.1], dtype=jnp.float64)

    raw = diagnose_rgie_raw_condensate_update_block(
        ln_mk=ln_mk,
        epsilon=epsilon,
        formula_matrix_cond=formula_matrix_cond,
        pi_vector=pi_vector,
        hvector_cond=hvector_cond,
    )

    assert jnp.allclose(
        raw["raw_delta_ln_mk_current"],
        raw["condensate_stationarity_residual_over_nu"],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert float(raw["raw_identity_max_abs_diff"]) <= 1.0e-12


def test_build_rgie_condensate_direction_variant_handles_gas_only_and_correction_only():
    raw_update = {
        "raw_delta_ln_mk_current": jnp.asarray([1.2, -0.7], dtype=jnp.float64),
        "correction": jnp.asarray([0.2, -1.7], dtype=jnp.float64),
    }

    gas_only = build_rgie_condensate_direction_variant(raw_update, "gas_only")
    correction_only = build_rgie_condensate_direction_variant(
        raw_update, "correction_only_no_clip"
    )
    rescaled = build_rgie_condensate_direction_variant(
        raw_update, "correction_only_scalar_rescale_0p1"
    )

    assert jnp.allclose(gas_only["delta_ln_mk"], jnp.zeros(2, dtype=jnp.float64))
    assert jnp.allclose(correction_only["delta_ln_mk"], raw_update["correction"])
    assert float(jnp.max(jnp.abs(rescaled["delta_ln_mk"]))) <= 0.1 + 1.0e-12


def test_build_rgie_condensate_direction_transform_variant_reports_clip_and_rescale():
    raw = jnp.asarray([1.5, -0.2], dtype=jnp.float64)

    clipped = build_rgie_condensate_direction_transform_variant(raw, "current_component_clip_0p1")
    scaled = build_rgie_condensate_direction_transform_variant(raw, "scalar_rescale_inf_0p1")

    assert jnp.allclose(clipped["delta_ln_mk"], jnp.asarray([0.1, -0.1], dtype=jnp.float64))
    assert float(clipped["saturated_fraction"]) > 0.0
    assert float(jnp.max(jnp.abs(scaled["delta_ln_mk"]))) <= 0.1 + 1.0e-12
    assert scaled["saturated_fraction"] is None


def test_compute_rgie_lambda_cap_policy_filters_expected_limiters():
    lam1_gas = jnp.asarray(0.3, dtype=jnp.float64)
    lam1_cond = jnp.asarray(0.2, dtype=jnp.float64)
    lam2_cond = jnp.asarray(0.4, dtype=jnp.float64)

    current = compute_rgie_lambda_cap_policy(
        "current_full_cap",
        lam1_gas=lam1_gas,
        lam1_cond=lam1_cond,
        lam2_cond=lam2_cond,
    )
    no_cond = compute_rgie_lambda_cap_policy(
        "no_cond_cap",
        lam1_gas=lam1_gas,
        lam1_cond=lam1_cond,
        lam2_cond=lam2_cond,
    )
    no_heuristic = compute_rgie_lambda_cap_policy(
        "no_heuristic_cap",
        lam1_gas=lam1_gas,
        lam1_cond=lam1_cond,
        lam2_cond=lam2_cond,
    )

    assert jnp.isclose(current["lam_cap"], 0.2)
    assert current["production_limiting_name"] == "lam1_cond"
    assert jnp.isclose(no_cond["lam_cap"], 0.3)
    assert jnp.isclose(no_heuristic["lam_cap"], 1.0)


def test_build_rgie_gas_direction_variant_handles_ntot_shift_variants():
    current_n = jnp.asarray([3.0, 5.0], dtype=jnp.float64)
    current_t = jnp.asarray(2.0, dtype=jnp.float64)
    ref_n = jnp.asarray([7.0, 11.0], dtype=jnp.float64)
    ref_t = jnp.asarray(-1.0, dtype=jnp.float64)

    no_shift = build_rgie_gas_direction_variant(
        "no_common_ntot_shift",
        delta_ln_nk_current=current_n,
        delta_ln_ntot_current=current_t,
        delta_ln_nk_ref=ref_n,
        delta_ln_ntot_ref=ref_t,
    )
    partial = build_rgie_gas_direction_variant(
        "partial_ntot_shift_0p5",
        delta_ln_nk_current=current_n,
        delta_ln_ntot_current=current_t,
        delta_ln_nk_ref=ref_n,
        delta_ln_ntot_ref=ref_t,
    )

    assert jnp.allclose(no_shift["delta_ln_nk"], jnp.asarray([1.0, 3.0], dtype=jnp.float64))
    assert jnp.isclose(no_shift["delta_ln_ntot"], 0.0)
    assert jnp.allclose(partial["delta_ln_nk"], jnp.asarray([2.0, 4.0], dtype=jnp.float64))
    assert jnp.isclose(partial["delta_ln_ntot"], 1.0)


def test_compute_rgie_lam1_gas_ignore_trace_diagnostics_ignores_small_vmr_limiters():
    ln_ntot = jnp.asarray(0.0, dtype=jnp.float64)
    ln_nk = jnp.log(jnp.asarray([1.0e-40, 1.0], dtype=jnp.float64))
    delta_ln_nk = jnp.asarray([10.0, 0.1], dtype=jnp.float64)
    delta_ln_ntot = jnp.asarray(0.0, dtype=jnp.float64)

    out = compute_rgie_lam1_gas_ignore_trace_diagnostics(
        ln_nk=ln_nk,
        ln_ntot=ln_ntot,
        delta_ln_nk=delta_ln_nk,
        delta_ln_ntot=delta_ln_ntot,
        vmr_floor=1.0e-30,
    )

    assert out["current_top_limiter_species_index"] == 0
    assert not out["current_top_limiter_active_under_floor"]
    assert out["active_top_limiter_species_index"] == 1
    assert out["lam1_gas_ignore_trace"] > 1.0
