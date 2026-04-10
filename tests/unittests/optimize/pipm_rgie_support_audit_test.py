import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from exogibbs.optimize.pipm_rgie_cond import build_rgie_support_candidate_indices
from exogibbs.optimize.pipm_rgie_cond import summarize_rgie_inactive_driving


def test_summarize_rgie_inactive_driving_reports_top_positive_inactive():
    full_driving = jnp.array([-0.5, 0.2, 0.8, -0.1, 0.4], dtype=jnp.float64)
    support_indices = jnp.array([0, 3], dtype=jnp.int32)

    summary = summarize_rgie_inactive_driving(
        full_driving,
        support_indices,
        condensate_species_names=["A", "B", "C", "D", "E"],
        top_k=2,
    )

    assert summary["inactive_positive_count"] == 3
    assert summary["max_positive_inactive_driving"] == 0.8
    assert summary["top_inactive_indices"] == [2, 4]
    assert summary["top_inactive_names"] == ["C", "E"]
    assert summary["top_positive_inactive_indices"] == [2, 4, 1]


def test_build_rgie_support_candidate_indices_semismooth_and_augmented():
    support_indices = jnp.array([0, 3], dtype=jnp.int32)
    full_driving = jnp.array([-0.2, 0.7, -0.4, 0.1, 0.6], dtype=jnp.float64)
    active_ln_mk = jnp.array([-40.0, -2.0], dtype=jnp.float64)
    active_driving = jnp.array([0.3, -0.5], dtype=jnp.float64)

    semismooth = build_rgie_support_candidate_indices(
        support_indices,
        full_driving=full_driving,
        active_ln_mk=active_ln_mk,
        active_driving=active_driving,
        mechanism_name="semismooth_candidate",
    )
    augmented = build_rgie_support_candidate_indices(
        support_indices,
        full_driving=full_driving,
        active_ln_mk=active_ln_mk,
        active_driving=active_driving,
        mechanism_name="augmented_semismooth_candidate",
    )

    assert semismooth["added_indices"] == [1]
    assert semismooth["support_indices"].tolist() == [0, 1, 3]
    assert augmented["added_indices"] == [1]
    assert augmented["dropped_indices"] == [0]
    assert augmented["support_indices"].tolist() == [1, 3]
