"""Small RGIE helpers shared by explicit legacy condensate routes."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np


def build_rgie_condensate_init_from_policy(
    epsilon: float,
    support_indices: jnp.ndarray,
    startup_policy: str,
    *,
    driving: Optional[jnp.ndarray] = None,
    m0: Optional[float] = None,
    r0: Optional[float] = None,
    top_k: Optional[int] = None,
    tiny_fallback: float = 1.0e-30,
    dtype: Optional[jnp.dtype] = None,
) -> jnp.ndarray:
    """Build an RGIE condensate initialization from a startup policy."""

    support_indices = jnp.asarray(support_indices)
    if support_indices.ndim != 1:
        raise ValueError("support_indices must be a one-dimensional index array.")

    if dtype is None:
        dtype = jnp.float64

    eps = jnp.asarray(epsilon, dtype=dtype)
    n_support = int(support_indices.shape[0])
    if n_support == 0:
        return jnp.zeros((0,), dtype=dtype)

    if tiny_fallback <= 0.0:
        raise ValueError("tiny_fallback must be positive.")
    fallback_ln_m0 = jnp.log(jnp.asarray(tiny_fallback, dtype=dtype))

    if startup_policy == "absolute_uniform_m0":
        if m0 is None or m0 <= 0.0:
            raise ValueError("absolute_uniform_m0 requires a positive m0.")
        return jnp.full(
            (n_support,),
            jnp.log(jnp.asarray(m0, dtype=dtype)),
            dtype=dtype,
        )

    target_ln_m0 = None
    if startup_policy in (
        "ratio_uniform_r0",
        "ratio_positive_driving_r0",
        "ratio_topk_positive_driving_r0",
    ):
        if r0 is None or r0 <= 0.0:
            raise ValueError(f"{startup_policy} requires a positive r0.")
        target_ln_m0 = eps + jnp.log(jnp.asarray(r0, dtype=dtype))

    if startup_policy == "ratio_uniform_r0":
        return jnp.full((n_support,), target_ln_m0, dtype=dtype)

    if startup_policy in (
        "ratio_positive_driving_r0",
        "ratio_topk_positive_driving_r0",
    ):
        if driving is None:
            raise ValueError(f"{startup_policy} requires driving values.")
        driving = jnp.asarray(driving, dtype=dtype)
        if driving.shape != (n_support,):
            raise ValueError(
                "driving must have the same shape as the supported condensate block "
                f"(got {driving.shape}, expected {(n_support,)})."
            )
        positive = driving > 0.0

        if startup_policy == "ratio_positive_driving_r0":
            selected = positive
        else:
            if top_k is None:
                raise ValueError("ratio_topk_positive_driving_r0 requires top_k.")
            if top_k < 0:
                raise ValueError("top_k must be non-negative.")
            if top_k == 0:
                selected = jnp.zeros((n_support,), dtype=bool)
            else:
                safe_driving = jnp.where(positive, driving, -jnp.inf)
                ranked = jnp.argsort(-safe_driving)
                top_indices = ranked[: min(top_k, n_support)]
                selected = (
                    jnp.zeros((n_support,), dtype=bool)
                    .at[top_indices]
                    .set(True)
                )
                selected = selected & positive

        return jnp.where(selected, target_ln_m0, fallback_ln_m0)

    raise ValueError(
        "Unknown startup_policy "
        f"'{startup_policy}'. Expected one of "
        "('absolute_uniform_m0', 'ratio_uniform_r0', "
        "'ratio_positive_driving_r0', 'ratio_topk_positive_driving_r0')."
    )


def summarize_rgie_inactive_driving(
    full_driving: jnp.ndarray,
    support_indices: jnp.ndarray,
    *,
    condensate_species_names: Optional[Sequence[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Summarize inactive-driving violations for a current support."""

    driving = np.asarray(jax.device_get(full_driving), dtype=np.float64)
    support = np.asarray(jax.device_get(support_indices), dtype=np.int64)
    n_cond = int(driving.shape[0])
    support_mask = np.zeros((n_cond,), dtype=bool)
    if support.size:
        support_mask[support] = True
    inactive_indices = np.nonzero(~support_mask)[0]

    if inactive_indices.size == 0:
        return {
            "max_positive_inactive_driving": 0.0,
            "inactive_positive_count": 0,
            "top_inactive_indices": [],
            "top_inactive_names": [],
            "top_inactive_driving": [],
            "top_positive_inactive_indices": [],
        }

    inactive_driving = driving[inactive_indices]
    positive_mask = inactive_driving > 0.0
    positive_indices = inactive_indices[positive_mask]
    positive_driving = inactive_driving[positive_mask]

    if positive_indices.size == 0:
        return {
            "max_positive_inactive_driving": 0.0,
            "inactive_positive_count": 0,
            "top_inactive_indices": [],
            "top_inactive_names": [],
            "top_inactive_driving": [],
            "top_positive_inactive_indices": [],
        }

    ranked_order = np.argsort(-positive_driving)
    ranked_positive = positive_indices[ranked_order]
    top_indices = ranked_positive[
        : min(int(top_k), int(ranked_positive.shape[0]))
    ]
    top_driving = driving[top_indices]
    if condensate_species_names is None:
        top_names = [str(int(index)) for index in top_indices]
    else:
        top_names = [
            str(condensate_species_names[int(index)]) for index in top_indices
        ]

    return {
        "max_positive_inactive_driving": float(np.max(positive_driving)),
        "inactive_positive_count": int(positive_indices.shape[0]),
        "top_inactive_indices": [int(index) for index in top_indices],
        "top_inactive_names": top_names,
        "top_inactive_driving": [float(value) for value in top_driving],
        "top_positive_inactive_indices": [
            int(index) for index in ranked_positive
        ],
    }


__all__ = [
    "build_rgie_condensate_init_from_policy",
    "summarize_rgie_inactive_driving",
]
