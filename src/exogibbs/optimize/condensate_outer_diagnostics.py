"""Diagnostic gas-inner / condensate-outer prototype helpers.

These helpers are intentionally isolated from the active production condensate
solver path. They reuse the existing gas-only equilibrium solve and recover the
gas dual vector on the converged state so outer condensate active-set ideas can
be inspected without changing the production PIPM updates.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence

import jax.numpy as jnp
import numpy as np

from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.core import _compute_gk
from exogibbs.optimize.minimize import minimize_gibbs_core
from exogibbs.optimize.minimize import solve_gibbs_iteration_equations

try:
    from scipy.optimize import linprog as _scipy_linprog
except Exception:  # pragma: no cover - fallback only
    _scipy_linprog = None

try:
    from scipy.optimize import least_squares as _scipy_least_squares
except Exception:  # pragma: no cover - fallback only
    _scipy_least_squares = None

try:
    from scipy.optimize import Bounds as _scipy_Bounds
    from scipy.optimize import LinearConstraint as _scipy_LinearConstraint
    from scipy.optimize import minimize as _scipy_minimize
except Exception:  # pragma: no cover - fallback only
    _scipy_Bounds = None
    _scipy_LinearConstraint = None
    _scipy_minimize = None


def _default_gas_init(b_vec: jnp.ndarray, n_species: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    dtype = jnp.result_type(jnp.asarray(b_vec).dtype, jnp.float32)
    ln_nk0 = jnp.zeros((n_species,), dtype=dtype)
    ln_ntot0 = jnp.log(jnp.asarray(n_species, dtype=dtype))
    return ln_nk0, ln_ntot0


def solve_gas_equilibrium_with_duals(
    state: ThermoState,
    formula_matrix: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    *,
    ln_nk_init: Optional[jnp.ndarray] = None,
    ln_ntot_init: Optional[jnp.ndarray] = None,
    epsilon_crit: float = 1.0e-10,
    max_iter: int = 1000,
) -> dict[str, Any]:
    """Solve the existing gas-only problem and recover the gas dual vector."""

    n_species = int(formula_matrix.shape[1])
    if ln_nk_init is None or ln_ntot_init is None:
        ln_nk0, ln_ntot0 = _default_gas_init(state.element_vector, n_species)
    else:
        ln_nk0 = jnp.asarray(ln_nk_init)
        ln_ntot0 = jnp.asarray(ln_ntot_init)

    ln_nk, ln_ntot, n_iter, final_residual = minimize_gibbs_core(
        state,
        ln_nk0,
        ln_ntot0,
        formula_matrix,
        hvector_func,
        epsilon_crit=epsilon_crit,
        max_iter=max_iter,
    )

    nk = jnp.exp(ln_nk)
    ntot = jnp.exp(ln_ntot)
    hvector = hvector_func(state.temperature)
    gk = _compute_gk(
        state.temperature,
        ln_nk,
        ln_ntot,
        hvector,
        state.ln_normalized_pressure,
    )
    An = formula_matrix @ nk
    pi_vector, delta_ln_ntot = solve_gibbs_iteration_equations(
        nk,
        ntot,
        formula_matrix,
        state.element_vector,
        gk,
        An,
    )
    stationarity = formula_matrix.T @ pi_vector - gk

    epsilon_crit_used = jnp.asarray(epsilon_crit, dtype=final_residual.dtype)
    max_iter_used = jnp.asarray(max_iter, dtype=n_iter.dtype)
    converged = final_residual <= epsilon_crit_used

    return {
        "ln_nk": ln_nk,
        "ln_ntot": ln_ntot,
        "nk": nk,
        "ntot": ntot,
        "hvector": hvector,
        "gk": gk,
        "An": An,
        "pi_vector": pi_vector,
        "delta_ln_ntot_recovered": delta_ln_ntot,
        "stationarity": stationarity,
        "stationarity_inf_norm": jnp.max(jnp.abs(stationarity)),
        "diagnostics": {
            "n_iter": n_iter,
            "converged": converged,
            "hit_max_iter": (n_iter >= max_iter_used) & (~converged),
            "final_residual": final_residual,
            "epsilon_crit": epsilon_crit_used,
            "max_iter": max_iter_used,
        },
    }


def _amounts_from_optional_inputs(
    formula_matrix_cond: jnp.ndarray,
    *,
    m: Optional[jnp.ndarray],
    ln_mk: Optional[jnp.ndarray],
) -> jnp.ndarray:
    if m is not None and ln_mk is not None:
        raise ValueError("Pass either m or ln_mk, not both.")
    if m is not None:
        return jnp.asarray(m)
    if ln_mk is not None:
        return jnp.exp(jnp.asarray(ln_mk))
    return jnp.zeros((formula_matrix_cond.shape[1],), dtype=formula_matrix_cond.dtype)


def _top_indices(values: jnp.ndarray, *, descending: bool, top_k: int) -> list[int]:
    if values.size == 0:
        return []
    ranked = jnp.argsort(-values if descending else values)
    limit = min(int(values.shape[0]), top_k)
    return [int(i) for i in ranked[:limit]]


def _max_feasible_increment(column: jnp.ndarray, b_eff: jnp.ndarray) -> float:
    positive = column > 0.0
    if not bool(jnp.any(positive)):
        return 0.0
    ratios = jnp.where(positive, b_eff / jnp.clip(column, 1.0e-300), jnp.inf)
    return float(jnp.min(ratios))


def _normalized_gibbs(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_ntot: jnp.ndarray,
    hvector_gas: jnp.ndarray,
    m: jnp.ndarray,
    hvector_cond: jnp.ndarray,
) -> jnp.ndarray:
    nk = jnp.exp(ln_nk)
    gas_term = jnp.sum(nk * (hvector_gas + ln_nk - ln_ntot + state.ln_normalized_pressure))
    cond_term = jnp.sum(m * hvector_cond)
    return gas_term + cond_term


def _normalize_stoich_columns(formula_matrix_cond: jnp.ndarray) -> jnp.ndarray:
    col_norm = jnp.linalg.norm(formula_matrix_cond, axis=0)
    return formula_matrix_cond / jnp.clip(col_norm, 1.0e-300)


def _column_fingerprints(formula_matrix_cond: jnp.ndarray) -> jnp.ndarray:
    col_sum = jnp.sum(formula_matrix_cond, axis=0)
    return formula_matrix_cond / jnp.clip(col_sum[None, :], 1.0e-300)


def _cluster_positive_condensates(
    *,
    formula_matrix_cond: jnp.ndarray,
    driving: jnp.ndarray,
    positive_indices: list[int],
    condensate_species: Optional[Sequence[str]],
    element_names: Optional[Sequence[str]],
    similarity_threshold: float,
    driving_tie_relative_tol: float,
    driving_tie_absolute_tol: float,
    cluster_leader_top_k: int,
) -> dict[str, Any]:
    normalized_columns = _normalize_stoich_columns(formula_matrix_cond)
    fingerprints = _column_fingerprints(formula_matrix_cond)
    positive_set = jnp.asarray(positive_indices, dtype=jnp.int32)
    clusters: list[dict[str, Any]] = []
    consumed = set()

    def _cluster_threshold(idx: int) -> float:
        return max(
            float(driving_tie_absolute_tol),
            float(driving_tie_relative_tol) * abs(float(driving[idx])),
        )

    for idx in positive_indices:
        if idx in consumed:
            continue
        leader_fp = normalized_columns[:, idx]
        leader_drive = float(driving[idx])
        members = []
        for cand in positive_indices:
            if cand in consumed:
                continue
            similarity = float(jnp.dot(leader_fp, normalized_columns[:, cand]))
            if similarity < similarity_threshold:
                continue
            if abs(float(driving[cand]) - leader_drive) > max(_cluster_threshold(idx), _cluster_threshold(cand)):
                continue
            members.append(cand)

        for cand in members:
            consumed.add(cand)

        member_idx = jnp.asarray(members, dtype=jnp.int32)
        member_weights = jnp.sum(formula_matrix_cond[:, member_idx], axis=1)
        dominant_rank = jnp.argsort(-member_weights)
        dominant_indices = [int(i) for i in dominant_rank[: min(3, member_weights.shape[0])]]
        ordered_members = sorted(members, key=lambda i: float(driving[i]), reverse=True)
        leading = ordered_members[: min(cluster_leader_top_k, len(ordered_members))]
        clusters.append(
            {
                "leader_index": idx,
                "leader_name": None if condensate_species is None else str(condensate_species[idx]),
                "size": len(members),
                "member_indices": members,
                "member_names": None if condensate_species is None else [str(condensate_species[i]) for i in members],
                "similarity_threshold": float(similarity_threshold),
                "driving_min": float(jnp.min(driving[member_idx])),
                "driving_max": float(jnp.max(driving[member_idx])),
                "leading_indices": leading,
                "leading_names": None if condensate_species is None else [str(condensate_species[i]) for i in leading],
                "dominant_element_indices": dominant_indices,
                "dominant_element_names": None if element_names is None else [str(element_names[i]) for i in dominant_indices],
                "dominant_element_weights": [float(member_weights[i]) for i in dominant_indices],
                "mean_fingerprint": [float(x) for x in jnp.mean(fingerprints[:, member_idx], axis=1)],
            }
        )

    similarities = normalized_columns[:, positive_set].T @ normalized_columns[:, positive_set]
    return {
        "positive_count": len(positive_indices),
        "cluster_count": len(clusters),
        "clusters": sorted(clusters, key=lambda rec: (-int(rec["size"]), -float(rec["driving_max"]))),
        "pairwise_similarity_matrix": similarities,
        "fingerprints": fingerprints[:, positive_set] if positive_indices else jnp.zeros((formula_matrix_cond.shape[0], 0)),
        "positive_indices": positive_indices,
    }


def _top_elements_for_condensate(
    formula_matrix_cond: jnp.ndarray,
    idx: int,
    *,
    element_names: Optional[Sequence[str]],
    top_k: int = 3,
) -> dict[str, Any]:
    column = formula_matrix_cond[:, idx]
    ranked = jnp.argsort(-column)
    chosen = [int(i) for i in ranked[: min(top_k, column.shape[0])] if float(column[i]) > 0.0]
    return {
        "element_indices": chosen,
        "element_names": None if element_names is None else [str(element_names[i]) for i in chosen],
        "stoich": [float(column[i]) for i in chosen],
    }


def _element_bottleneck_summary(
    *,
    b_eff: jnp.ndarray,
    pi_vector: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    positive_indices: list[int],
    top_positive_indices: list[int],
    element_names: Optional[Sequence[str]],
    condensate_species: Optional[Sequence[str]],
) -> dict[str, Any]:
    shadow_rank = jnp.argsort(-pi_vector)
    budget_rank = jnp.argsort(b_eff)
    top_shadow = [int(i) for i in shadow_rank[: min(5, pi_vector.shape[0])]]
    top_tight = [int(i) for i in budget_rank[: min(5, b_eff.shape[0])]]

    aggregate_demand = (
        jnp.sum(formula_matrix_cond[:, positive_indices], axis=1)
        if positive_indices
        else jnp.zeros_like(b_eff)
    )
    demand_rank = jnp.argsort(-aggregate_demand)
    top_aggregate = [int(i) for i in demand_rank[: min(5, aggregate_demand.shape[0])]]

    top_condensates = []
    for idx in top_positive_indices:
        signature = _top_elements_for_condensate(
            formula_matrix_cond,
            idx,
            element_names=element_names,
            top_k=4,
        )
        top_condensates.append(
            {
                "index": idx,
                "name": None if condensate_species is None else str(condensate_species[idx]),
                "demand_signature": signature,
                "budget_share_on_signature": [
                    float(formula_matrix_cond[element_idx, idx] / jnp.clip(b_eff[element_idx], 1.0e-300))
                    for element_idx in signature["element_indices"]
                ],
                "shadow_price_on_signature": [float(pi_vector[element_idx]) for element_idx in signature["element_indices"]],
            }
        )

    return {
        "b_eff": b_eff,
        "pi_vector": pi_vector,
        "top_shadow_price_indices": top_shadow,
        "top_shadow_price_names": None if element_names is None else [str(element_names[i]) for i in top_shadow],
        "top_shadow_price_values": [float(pi_vector[i]) for i in top_shadow],
        "tightest_budget_indices": top_tight,
        "tightest_budget_names": None if element_names is None else [str(element_names[i]) for i in top_tight],
        "tightest_budget_values": [float(b_eff[i]) for i in top_tight],
        "aggregate_positive_demand_indices": top_aggregate,
        "aggregate_positive_demand_names": None if element_names is None else [str(element_names[i]) for i in top_aggregate],
        "aggregate_positive_demand_values": [float(aggregate_demand[i]) for i in top_aggregate],
        "top_positive_condensates": top_condensates,
    }


def _greedy_joint_activation(
    *,
    restricted_matrix: jnp.ndarray,
    restricted_driving: jnp.ndarray,
    b_eff: jnp.ndarray,
) -> dict[str, Any]:
    x = jnp.zeros((restricted_matrix.shape[1],), dtype=restricted_matrix.dtype)
    remaining = jnp.asarray(b_eff)
    chosen = []
    order = sorted(range(restricted_matrix.shape[1]), key=lambda i: float(restricted_driving[i]), reverse=True)
    for local_idx in order:
        if float(restricted_driving[local_idx]) <= 0.0:
            continue
        step = _max_feasible_increment(restricted_matrix[:, local_idx], remaining)
        if step <= 0.0:
            continue
        x = x.at[local_idx].set(step)
        remaining = remaining - step * restricted_matrix[:, local_idx]
        chosen.append(local_idx)
    return {
        "status": "fallback_greedy",
        "success": True,
        "x": x,
        "objective": float(jnp.dot(restricted_driving, x)),
        "message": "SciPy linprog unavailable; used greedy packing fallback.",
        "chosen_indices": chosen,
    }


def _joint_activation_lp(
    *,
    formula_matrix_cond: jnp.ndarray,
    driving: jnp.ndarray,
    b_eff: jnp.ndarray,
    candidate_indices: list[int],
    condensate_species: Optional[Sequence[str]],
    clusters: Optional[dict[str, Any]],
    element_names: Optional[Sequence[str]],
    support_tol: float,
) -> dict[str, Any]:
    if not candidate_indices:
        return {
            "candidate_count": 0,
            "candidate_indices": [],
            "candidate_names": [],
            "support_size": 0,
            "support_indices": [],
            "support_names": [],
            "support_amounts": [],
            "support_cluster_labels": [],
            "binding_element_indices": [],
            "binding_element_names": [],
            "binding_element_slacks": [],
            "element_consumption_fraction": [],
            "objective": 0.0,
            "solver_status": "empty",
            "solver_success": True,
            "solver_message": "No positive-driving candidates available.",
        }
    candidate_idx = jnp.asarray(candidate_indices, dtype=jnp.int32)
    restricted_matrix = jnp.asarray(formula_matrix_cond[:, candidate_idx], dtype=jnp.float64).reshape(formula_matrix_cond.shape[0], -1)
    restricted_driving = jnp.asarray(driving[candidate_idx], dtype=jnp.float64).reshape(-1)
    rhs = jnp.asarray(b_eff, dtype=jnp.float64)

    if _scipy_linprog is None:  # pragma: no cover - fallback path only
        lp_result = _greedy_joint_activation(
            restricted_matrix=restricted_matrix,
            restricted_driving=restricted_driving,
            b_eff=rhs,
        )
        x = jnp.asarray(lp_result["x"])
    else:
        scipy_res = _scipy_linprog(
            c=(-restricted_driving).tolist(),
            A_ub=restricted_matrix.tolist(),
            b_ub=rhs.tolist(),
            bounds=[(0.0, None)] * restricted_matrix.shape[1],
            method="highs",
        )
        x = jnp.asarray(
            scipy_res.x if scipy_res.x is not None else jnp.zeros((restricted_matrix.shape[1],), dtype=restricted_matrix.dtype),
            dtype=restricted_matrix.dtype,
        )
        lp_result = {
            "status": str(scipy_res.status),
            "success": bool(scipy_res.success),
            "objective": float(-scipy_res.fun) if scipy_res.fun is not None else float(jnp.dot(restricted_driving, x)),
            "message": str(scipy_res.message),
        }

    support_local = [i for i in range(len(candidate_indices)) if float(x[i]) > support_tol]
    support_global = [candidate_indices[i] for i in support_local]
    consumed = restricted_matrix @ x
    slack = rhs - consumed
    binding = [int(i) for i in range(slack.shape[0]) if float(slack[i]) <= max(1.0e-10, 1.0e-8 * float(jnp.max(rhs) + 1.0))]

    cluster_labels = None
    if clusters is not None:
        cluster_labels = []
        for global_idx in support_global:
            matches = [
                cluster["leader_name"] if cluster["leader_name"] is not None else cluster["leader_index"]
                for cluster in clusters["clusters"]
                if global_idx in cluster["member_indices"]
            ]
            cluster_labels.append(matches[0] if matches else None)

    return {
        "candidate_count": len(candidate_indices),
        "candidate_indices": candidate_indices,
        "candidate_names": None if condensate_species is None else [str(condensate_species[i]) for i in candidate_indices],
        "support_size": len(support_global),
        "support_indices": support_global,
        "support_names": None if condensate_species is None else [str(condensate_species[i]) for i in support_global],
        "support_amounts": [float(x[i]) for i in support_local],
        "support_cluster_labels": cluster_labels,
        "binding_element_indices": binding,
        "binding_element_names": None if element_names is None else [str(element_names[i]) for i in binding],
        "binding_element_slacks": [float(slack[i]) for i in binding],
        "element_consumption_fraction": [
            float(consumed[i] / jnp.clip(rhs[i], 1.0e-300))
            for i in binding
        ],
        "objective": float(lp_result["objective"]),
        "solver_status": lp_result["status"],
        "solver_success": bool(lp_result["success"]),
        "solver_message": lp_result["message"],
    }


def _max_feasible_scaling(m: jnp.ndarray, formula_matrix_cond: jnp.ndarray, b: jnp.ndarray) -> float:
    demand = formula_matrix_cond @ m
    positive = demand > 0.0
    if not bool(jnp.any(positive)):
        return 1.0
    ratios = jnp.where(positive, b / jnp.clip(demand, 1.0e-300), jnp.inf)
    return float(jnp.clip(jnp.min(ratios), 0.0, 1.0))


def _support_mask(n_cond: int, support_indices: Sequence[int]) -> jnp.ndarray:
    mask = jnp.zeros((n_cond,), dtype=bool)
    if support_indices:
        mask = mask.at[jnp.asarray(support_indices, dtype=jnp.int32)].set(True)
    return mask


def _assemble_support_amounts(
    n_cond: int,
    support_indices: Sequence[int],
    support_amounts: jnp.ndarray,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    m_full = jnp.zeros((n_cond,), dtype=dtype)
    if support_indices:
        m_full = m_full.at[jnp.asarray(support_indices, dtype=jnp.int32)].set(support_amounts)
    return m_full


def _restricted_support_names(
    support_indices: Sequence[int],
    condensate_species: Optional[Sequence[str]],
) -> Optional[list[str]]:
    if condensate_species is None:
        return None
    return [str(condensate_species[i]) for i in support_indices]


def _top_inactive_violators(
    driving: jnp.ndarray,
    support_mask: jnp.ndarray,
    condensate_species: Optional[Sequence[str]],
    top_k: int,
) -> dict[str, Any]:
    inactive_indices = [int(i) for i in jnp.where(~support_mask)[0]]
    positive_inactive = [idx for idx in inactive_indices if float(driving[idx]) > 0.0]
    ordered = sorted(positive_inactive, key=lambda idx: float(driving[idx]), reverse=True)
    chosen = ordered[: min(top_k, len(ordered))]
    return {
        "max_positive_inactive_driving": float(
            jnp.max(jnp.maximum(jnp.where(~support_mask, driving, -jnp.inf), 0.0))
        )
        if inactive_indices
        else 0.0,
        "positive_inactive_count": len(positive_inactive),
        "top_indices": chosen,
        "top_names": None if condensate_species is None else [str(condensate_species[i]) for i in chosen],
        "top_driving": [float(driving[i]) for i in chosen],
    }


def _restricted_support_initial_guess(
    *,
    support_indices: Sequence[int],
    support_amounts_init: Optional[jnp.ndarray],
    joint_activation: Optional[dict[str, Any]],
    default_amount: float,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    if support_amounts_init is not None:
        values = jnp.asarray(support_amounts_init, dtype=dtype)
        if values.shape != (len(support_indices),):
            raise ValueError("support_amounts_init must have shape (len(support_indices),).")
        return jnp.maximum(values, 0.0)
    if joint_activation is not None and joint_activation.get("support_indices") == list(support_indices):
        return jnp.asarray(joint_activation.get("support_amounts", []), dtype=dtype)
    return jnp.full((len(support_indices),), default_amount, dtype=dtype)


def _support_amounts_from_full(
    support_indices: Sequence[int],
    m_full: jnp.ndarray,
    *,
    default_amount: float,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    if not support_indices:
        return jnp.zeros((0,), dtype=dtype)
    values = []
    for idx in support_indices:
        amount = float(m_full[idx])
        values.append(amount if amount > 0.0 else float(default_amount))
    return jnp.asarray(values, dtype=dtype)


def _rank_inactive_candidates(
    *,
    driving_full: jnp.ndarray,
    current_support: Sequence[int],
    formula_matrix_cond: jnp.ndarray,
    binding_element_indices: Sequence[int],
    add_threshold: float,
    prefer_binding_elements: bool,
) -> list[int]:
    support_set = set(int(i) for i in current_support)
    candidates = [idx for idx in range(int(driving_full.shape[0])) if idx not in support_set and float(driving_full[idx]) > add_threshold]
    if not candidates:
        return []

    def _uses_binding(idx: int) -> bool:
        if not binding_element_indices:
            return False
        return any(float(formula_matrix_cond[element_idx, idx]) > 0.0 for element_idx in binding_element_indices)

    if prefer_binding_elements:
        binding_candidates = [idx for idx in candidates if _uses_binding(idx)]
        if binding_candidates:
            candidates = binding_candidates
    return sorted(candidates, key=lambda idx: float(driving_full[idx]), reverse=True)


def _rank_inactive_candidates_with_tiebreak(
    *,
    driving_full: jnp.ndarray,
    current_support: Sequence[int],
    formula_matrix_cond: jnp.ndarray,
    binding_element_indices: Sequence[int],
    add_threshold: float,
    bottleneck_tiebreak: bool,
) -> list[int]:
    support_set = set(int(i) for i in current_support)
    candidates = [
        idx
        for idx in range(int(driving_full.shape[0]))
        if idx not in support_set and float(driving_full[idx]) > add_threshold
    ]
    if not candidates:
        return []

    def _uses_binding(idx: int) -> bool:
        if not binding_element_indices:
            return False
        return any(float(formula_matrix_cond[element_idx, idx]) > 0.0 for element_idx in binding_element_indices)

    if bottleneck_tiebreak:
        return sorted(
            candidates,
            key=lambda idx: (_uses_binding(idx), float(driving_full[idx])),
            reverse=True,
        )
    return sorted(candidates, key=lambda idx: float(driving_full[idx]), reverse=True)


def _inactive_addition_gate(
    *,
    solve: dict[str, Any],
    add_threshold: float,
    settle_inactive_ratio: float,
    settle_stationarity_abs_tol: float,
    settle_feasibility_tol: float,
    settle_complementarity_abs_tol: float,
) -> dict[str, Any]:
    inactive_max = float(solve["max_positive_inactive_driving"])
    inactive_scale = max(inactive_max, float(add_threshold), 1.0)
    stationarity = float(solve["true_stationarity_residual_inf"])
    feasibility = float(solve["feasibility_residual_inf"])
    complementarity_value = solve.get("complementarity_residual_inf")
    complementarity = 0.0 if complementarity_value is None else float(complementarity_value)

    stationarity_limit = max(settle_stationarity_abs_tol, settle_inactive_ratio * inactive_scale)
    complementarity_limit = max(settle_complementarity_abs_tol, settle_inactive_ratio * inactive_scale)
    settled = (
        feasibility <= settle_feasibility_tol
        and stationarity <= stationarity_limit
        and (
            complementarity_value is None
            or (not bool(solve.get("solver_success")))
            or complementarity <= complementarity_limit
        )
    )
    return {
        "settled_for_addition": bool(settled),
        "stationarity_limit": float(stationarity_limit),
        "complementarity_limit": float(complementarity_limit),
        "stationarity": stationarity,
        "feasibility": feasibility,
        "complementarity": None if complementarity_value is None else complementarity,
    }


def _select_working_set_additions(
    *,
    solve: dict[str, Any],
    current_support: Sequence[int],
    formula_matrix_cond: jnp.ndarray,
    add_rule: str,
    add_threshold: float,
    max_additions_per_iter: int,
    reduced_cost_threshold: float,
    reduced_cost_fraction: float,
    binding_element_indices: Sequence[int],
    bottleneck_tiebreak: bool,
    require_settled: bool,
    settle_inactive_ratio: float,
    settle_stationarity_abs_tol: float,
    settle_feasibility_tol: float,
    settle_complementarity_abs_tol: float,
) -> dict[str, Any]:
    gate = _inactive_addition_gate(
        solve=solve,
        add_threshold=add_threshold,
        settle_inactive_ratio=settle_inactive_ratio,
        settle_stationarity_abs_tol=settle_stationarity_abs_tol,
        settle_feasibility_tol=settle_feasibility_tol,
        settle_complementarity_abs_tol=settle_complementarity_abs_tol,
    )
    ranked = _rank_inactive_candidates_with_tiebreak(
        driving_full=jnp.asarray(solve["driving_full"]),
        current_support=current_support,
        formula_matrix_cond=formula_matrix_cond,
        binding_element_indices=binding_element_indices,
        add_threshold=add_threshold,
        bottleneck_tiebreak=bottleneck_tiebreak,
    )
    if require_settled and not gate["settled_for_addition"]:
        return {
            **gate,
            "eligible_ranked_indices": ranked,
            "eligible_ranked_driving": [float(solve["driving_full"][idx]) for idx in ranked[: min(10, len(ranked))]],
            "selected_indices": [],
            "selection_threshold": None,
        }

    if add_rule == "naive_topk":
        selected = ranked[:max_additions_per_iter]
        selection_threshold = float(add_threshold)
    elif add_rule == "reduced_cost_threshold":
        inactive_max = float(solve["max_positive_inactive_driving"])
        selection_threshold = max(
            float(add_threshold),
            float(reduced_cost_threshold),
            float(reduced_cost_fraction) * max(inactive_max, 0.0),
        )
        selected = [idx for idx in ranked if float(solve["driving_full"][idx]) >= selection_threshold]
        if max_additions_per_iter > 0:
            selected = selected[:max_additions_per_iter]
    else:
        raise ValueError("add_rule must be one of ('naive_topk', 'reduced_cost_threshold').")

    return {
        **gate,
        "eligible_ranked_indices": ranked,
        "eligible_ranked_driving": [float(solve["driving_full"][idx]) for idx in ranked[: min(10, len(ranked))]],
        "selected_indices": selected,
        "selection_threshold": float(selection_threshold) if selection_threshold is not None else None,
    }


def _compute_dynamic_support_merit(
    *,
    solve: dict[str, Any],
    stationarity_weight: float,
    feasibility_weight: float,
    complementarity_weight: float,
    inactive_weight: float,
    inactive_count_weight: float,
    support_size: int = 0,
    support_size_penalty: float = 0.0,
    added_count: int = 0,
    addition_penalty: float = 0.0,
) -> dict[str, Any]:
    stationarity = float(solve["true_stationarity_residual_inf"])
    feasibility = float(solve["feasibility_residual_inf"])
    complementarity_value = solve.get("complementarity_residual_inf")
    complementarity = 0.0 if complementarity_value is None else float(complementarity_value)
    inactive_max = float(solve["max_positive_inactive_driving"])
    inactive_count = float(solve["inactive_positive_count"])
    support_size_value = int(support_size)
    added_count_value = int(added_count)
    sparsity_penalty = (
        float(support_size_penalty) * float(support_size_value)
        + float(addition_penalty) * float(added_count_value)
    )
    value = (
        float(stationarity_weight) * stationarity
        + float(feasibility_weight) * feasibility
        + float(complementarity_weight) * complementarity
        + float(inactive_weight) * inactive_max
        + float(inactive_count_weight) * inactive_count
        + sparsity_penalty
    )
    return {
        "value": float(value),
        "base_value": float(value - sparsity_penalty),
        "stationarity": stationarity,
        "feasibility": feasibility,
        "complementarity": None if complementarity_value is None else complementarity,
        "inactive_max": inactive_max,
        "inactive_count": int(solve["inactive_positive_count"]),
        "support_size": support_size_value,
        "support_size_penalty": float(support_size_penalty),
        "support_size_penalty_value": float(float(support_size_penalty) * float(support_size_value)),
        "added_count": added_count_value,
        "addition_penalty": float(addition_penalty),
        "addition_penalty_value": float(float(addition_penalty) * float(added_count_value)),
        "sparsity_penalty_value": float(sparsity_penalty),
        "weights": {
            "stationarity": float(stationarity_weight),
            "feasibility": float(feasibility_weight),
            "complementarity": float(complementarity_weight),
            "inactive_max": float(inactive_weight),
            "inactive_count": float(inactive_count_weight),
            "support_size": float(support_size_penalty),
            "added_count": float(addition_penalty),
        },
    }


def _build_addition_proposals(
    *,
    solve: dict[str, Any],
    current_support: Sequence[int],
    formula_matrix_cond: jnp.ndarray,
    add_threshold: float,
    reduced_cost_threshold: float,
    reduced_cost_fraction: float,
    binding_element_indices: Sequence[int],
    bottleneck_tiebreak: bool,
    proposal_batch_sizes: Sequence[int],
    current_support_size: int,
    support_size_cap: Optional[int],
    max_batch_size: Optional[int],
) -> list[dict[str, Any]]:
    ranked = _rank_inactive_candidates_with_tiebreak(
        driving_full=jnp.asarray(solve["driving_full"]),
        current_support=current_support,
        formula_matrix_cond=formula_matrix_cond,
        binding_element_indices=binding_element_indices,
        add_threshold=add_threshold,
        bottleneck_tiebreak=bottleneck_tiebreak,
    )
    proposals: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    effective_batch_cap = None if max_batch_size is None else max(0, int(max_batch_size))
    support_room = None if support_size_cap is None else max(0, int(support_size_cap) - int(current_support_size))

    def _apply_caps(indices: tuple[int, ...]) -> tuple[int, ...]:
        capped = indices
        if effective_batch_cap is not None:
            capped = capped[:effective_batch_cap]
        if support_room is not None:
            capped = capped[:support_room]
        return capped

    for batch_size in proposal_batch_sizes:
        chosen = _apply_caps(tuple(ranked[: max(0, int(batch_size))]))
        if not chosen or chosen in seen:
            continue
        seen.add(chosen)
        proposals.append(
            {
                "proposal_kind": f"naive_top_{len(chosen)}",
                "indices": list(chosen),
                "selection_threshold": float(add_threshold),
            }
        )

    inactive_max = float(solve["max_positive_inactive_driving"])
    selection_threshold = max(
        float(add_threshold),
        float(reduced_cost_threshold),
        float(reduced_cost_fraction) * max(inactive_max, 0.0),
    )
    threshold_indices = _apply_caps(tuple(
        idx for idx in ranked if float(solve["driving_full"][idx]) >= selection_threshold
    ))
    if threshold_indices and threshold_indices not in seen:
        proposals.append(
            {
                "proposal_kind": "reduced_cost_threshold",
                "indices": list(threshold_indices),
                "selection_threshold": float(selection_threshold),
            }
        )
    return proposals


def _build_drop_proposals(
    *,
    solve: dict[str, Any],
    current_support: Sequence[int],
    drop_m_threshold: float,
    drop_driving_threshold: float,
    drop_m_hysteresis_factor: float,
    drop_driving_hysteresis: float,
    max_drop_proposals: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    eligible: list[tuple[int, float, float]] = []
    tightened_m_threshold = float(drop_m_threshold) * float(drop_m_hysteresis_factor)
    tightened_driving_threshold = float(drop_driving_threshold) - float(drop_driving_hysteresis)
    for local_idx, global_idx in enumerate(current_support):
        amount = float(solve["m_candidate"][local_idx])
        driving_value = float(solve["driving_full"][global_idx])
        if amount <= tightened_m_threshold and driving_value <= tightened_driving_threshold:
            eligible.append((int(global_idx), amount, driving_value))
    eligible.sort(key=lambda rec: (rec[1], rec[2]))
    proposals: list[dict[str, Any]] = []
    if eligible:
        all_indices = [idx for idx, _amount, _driving in eligible]
        proposals.append(
            {
                "proposal_kind": "drop_all_eligible",
                "indices": all_indices,
            }
        )
    for idx, _amount, _driving in eligible[: max(0, int(max_drop_proposals))]:
        proposals.append(
            {
                "proposal_kind": "drop_single",
                "indices": [idx],
            }
        )
    return [idx for idx, _amount, _driving in eligible], proposals


def _solver_like_support_stabilized(
    *,
    solve: dict[str, Any],
    no_changes: bool,
    add_threshold: float,
    reduced_cost_threshold: float,
    stabilization_inactive_tol: float,
    settle_stationarity_abs_tol: float,
    settle_feasibility_tol: float,
    settle_complementarity_abs_tol: float,
) -> bool:
    if not no_changes:
        return False
    complementarity_value = solve.get("complementarity_residual_inf")
    complementarity_ok = (
        complementarity_value is None
        or float(complementarity_value) <= float(settle_complementarity_abs_tol)
    )
    return bool(
        float(solve["true_stationarity_residual_inf"]) <= float(settle_stationarity_abs_tol)
        and float(solve["feasibility_residual_inf"]) <= float(settle_feasibility_tol)
        and complementarity_ok
        and float(solve["max_positive_inactive_driving"]) <= max(
            float(add_threshold),
            float(reduced_cost_threshold),
            float(stabilization_inactive_tol),
        )
    )


def _fischer_burmeister(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(a * a + b * b) - a - b


def _smoothed_fischer_burmeister(a: jnp.ndarray, b: jnp.ndarray, mu: float) -> jnp.ndarray:
    mu_arr = jnp.asarray(mu, dtype=jnp.result_type(a.dtype, b.dtype))
    return jnp.sqrt(a * a + b * b + mu_arr * mu_arr) - a - b


def solve_restricted_support_condensate_layer(
    state: ThermoState,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    hvector_cond_func: Callable[[float], jnp.ndarray],
    *,
    support_indices: Sequence[int],
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
    support_amounts_init: Optional[jnp.ndarray] = None,
    joint_activation: Optional[dict[str, Any]] = None,
    ln_nk_gas_init: Optional[jnp.ndarray] = None,
    ln_ntot_gas_init: Optional[jnp.ndarray] = None,
    gas_epsilon_crit: float = 1.0e-10,
    gas_max_iter: int = 1000,
    budget_negative_tol: float = 1.0e-14,
    feasibility_penalty_weight: float = 1.0e6,
    active_m_threshold: float = 1.0e-24,
    inactive_top_k: int = 5,
    default_initial_amount: float = 1.0e-30,
    least_squares_ftol: float = 1.0e-10,
    least_squares_xtol: float = 1.0e-10,
    least_squares_gtol: float = 1.0e-10,
    least_squares_max_nfev: int = 100,
) -> dict[str, Any]:
    """Diagnostic-only nonlinear restricted-support outer solve for one layer.

    This uses a minimal restricted KKT residual, ``driving_S(m) ≈ 0``, on a
    support selected elsewhere (for example by the joint-activation LP). That is
    sufficient for the current diagnostic because support membership is fixed by
    the LP and inactive-species optimality is evaluated explicitly afterwards.
    """

    formula_matrix = jnp.asarray(formula_matrix)
    formula_matrix_cond = jnp.asarray(formula_matrix_cond)
    support_indices = [int(i) for i in support_indices]
    n_cond = int(formula_matrix_cond.shape[1])
    support_mask = _support_mask(n_cond, support_indices)
    support_names = _restricted_support_names(support_indices, condensate_species)
    hvector_cond = hvector_cond_func(state.temperature)
    dtype = jnp.result_type(formula_matrix_cond.dtype, state.element_vector.dtype, jnp.float64)
    initial_guess = _restricted_support_initial_guess(
        support_indices=support_indices,
        support_amounts_init=support_amounts_init,
        joint_activation=joint_activation,
        default_amount=default_initial_amount,
        dtype=dtype,
    )

    if _scipy_least_squares is None:  # pragma: no cover - fallback only
        raise RuntimeError("scipy.optimize.least_squares is required for restricted-support diagnostics.")

    evaluations: dict[str, Any] = {"count": 0, "gas_solves": 0, "last": None}

    def _evaluate_support(m_support: jnp.ndarray) -> dict[str, Any]:
        evaluations["count"] += 1
        m_support = jnp.asarray(m_support, dtype=dtype)
        m_full = _assemble_support_amounts(n_cond, support_indices, m_support, dtype)
        b_eff = state.element_vector - formula_matrix_cond @ m_full
        negative_budget = jnp.maximum(-b_eff, 0.0)
        infeasible = bool(jnp.any(b_eff < -budget_negative_tol))
        residual_penalty = feasibility_penalty_weight * negative_budget
        if infeasible:
            driving_full = jnp.full((n_cond,), jnp.nan, dtype=dtype)
            inactive = {
                "max_positive_inactive_driving": float("nan"),
                "positive_inactive_count": 0,
                "top_indices": [],
                "top_names": [] if condensate_species is not None else None,
                "top_driving": [],
            }
            element_bottlenecks = {
                "tightest_budget_indices": [int(i) for i in jnp.argsort(b_eff)[: min(5, b_eff.shape[0])]],
                "tightest_budget_names": None
                if element_names is None
                else [str(element_names[i]) for i in jnp.argsort(b_eff)[: min(5, b_eff.shape[0])]],
                "tightest_budget_values": [
                    float(b_eff[i]) for i in jnp.argsort(b_eff)[: min(5, b_eff.shape[0])]
                ],
            }
            return {
                "m_support": m_support,
                "m_full": m_full,
                "b_eff": b_eff,
                "negative_budget": negative_budget,
                "gas_result": None,
                "driving_support": jnp.zeros((len(support_indices),), dtype=dtype),
                "driving_full": driving_full,
                "inactive": inactive,
                "element_bottlenecks": element_bottlenecks,
                "active_gap_inf": float(jnp.max(negative_budget)) if negative_budget.size else 0.0,
                "combined_inf": float(jnp.max(negative_budget)) if negative_budget.size else 0.0,
                "residual_vector": jnp.concatenate([jnp.zeros((len(support_indices),), dtype=dtype), residual_penalty]),
                "status": "infeasible_b_eff",
            }

        gas_state = ThermoState(
            temperature=state.temperature,
            ln_normalized_pressure=state.ln_normalized_pressure,
            element_vector=b_eff,
        )
        gas_result = solve_gas_equilibrium_with_duals(
            gas_state,
            formula_matrix,
            hvector_func,
            ln_nk_init=ln_nk_gas_init,
            ln_ntot_init=ln_ntot_gas_init,
            epsilon_crit=gas_epsilon_crit,
            max_iter=gas_max_iter,
        )
        evaluations["gas_solves"] += 1
        driving_full = formula_matrix_cond.T @ gas_result["pi_vector"] - hvector_cond
        driving_support = driving_full[jnp.asarray(support_indices, dtype=jnp.int32)] if support_indices else jnp.zeros((0,), dtype=dtype)
        inactive = _top_inactive_violators(
            driving_full,
            support_mask,
            condensate_species,
            inactive_top_k,
        )
        element_bottlenecks = _element_bottleneck_summary(
            b_eff=b_eff,
            pi_vector=gas_result["pi_vector"],
            formula_matrix_cond=formula_matrix_cond,
            positive_indices=[idx for idx in range(n_cond) if float(driving_full[idx]) > 0.0],
            top_positive_indices=_top_indices(driving_full, descending=True, top_k=min(inactive_top_k, n_cond)),
            element_names=element_names,
            condensate_species=condensate_species,
        )
        active_gap_inf = float(jnp.max(jnp.abs(driving_support))) if support_indices else 0.0
        combined_inf = max(active_gap_inf, float(inactive["max_positive_inactive_driving"]))
        return {
            "m_support": m_support,
            "m_full": m_full,
            "b_eff": b_eff,
            "negative_budget": negative_budget,
            "gas_result": gas_result,
            "driving_support": driving_support,
            "driving_full": driving_full,
            "inactive": inactive,
            "element_bottlenecks": element_bottlenecks,
            "active_gap_inf": active_gap_inf,
            "combined_inf": combined_inf,
            "residual_vector": jnp.concatenate([driving_support, residual_penalty]),
            "status": "ok",
        }

    def _residual_numpy(m_support: np.ndarray) -> np.ndarray:
        current = _evaluate_support(jnp.asarray(m_support, dtype=dtype))
        evaluations["last"] = current
        return np.asarray(current["residual_vector"], dtype=np.float64)

    start = time.perf_counter()
    scipy_result = _scipy_least_squares(
        _residual_numpy,
        x0=np.asarray(initial_guess, dtype=np.float64),
        bounds=(0.0, np.inf),
        ftol=least_squares_ftol,
        xtol=least_squares_xtol,
        gtol=least_squares_gtol,
        max_nfev=least_squares_max_nfev,
    )
    elapsed = time.perf_counter() - start
    final_eval = _evaluate_support(jnp.asarray(scipy_result.x, dtype=dtype))
    feasible_projection_alpha = None
    raw_final_status = final_eval["status"]
    if final_eval["status"] == "infeasible_b_eff":
        feasible_projection_alpha = _max_feasible_scaling(final_eval["m_full"], formula_matrix_cond, state.element_vector)
        if feasible_projection_alpha > 0.0:
            projected_support = feasible_projection_alpha * final_eval["m_support"]
            final_eval = _evaluate_support(projected_support)

    return {
        "prototype_family": "restricted_support_condensate_kkt_diagnostic",
        "support_indices": support_indices,
        "support_names": support_names,
        "support_size": len(support_indices),
        "support_from_joint_activation": None if joint_activation is None else joint_activation.get("candidate_count"),
        "initial_guess": initial_guess,
        "status": final_eval["status"],
        "raw_final_status": raw_final_status,
        "feasible_projection_alpha": feasible_projection_alpha,
        "solver_success": bool(scipy_result.success) and final_eval["status"] == "ok",
        "solver_status": int(scipy_result.status),
        "solver_message": str(scipy_result.message),
        "solver_cost": float(scipy_result.cost),
        "solver_optimality": float(scipy_result.optimality),
        "runtime_seconds": float(elapsed),
        "nfev": int(scipy_result.nfev),
        "njev": None if scipy_result.njev is None else int(scipy_result.njev),
        "evaluation_count": int(evaluations["count"]),
        "gas_solve_count": int(evaluations["gas_solves"]),
        "m_support": final_eval["m_support"],
        "m_full": final_eval["m_full"],
        "active_support_count": int(jnp.sum(final_eval["m_support"] > active_m_threshold)),
        "b_eff": final_eval["b_eff"],
        "b_eff_feasible": bool(jnp.all(final_eval["b_eff"] >= -budget_negative_tol)),
        "negative_budget_inf": float(jnp.max(final_eval["negative_budget"])) if final_eval["negative_budget"].size else 0.0,
        "driving_support": final_eval["driving_support"],
        "driving_full": final_eval["driving_full"],
        "restricted_kkt_gap_inf": float(final_eval["active_gap_inf"]),
        "max_positive_inactive_driving": float(final_eval["inactive"]["max_positive_inactive_driving"]),
        "inactive_positive_count": int(final_eval["inactive"]["positive_inactive_count"]),
        "top_inactive_indices": final_eval["inactive"]["top_indices"],
        "top_inactive_names": final_eval["inactive"]["top_names"],
        "top_inactive_driving": final_eval["inactive"]["top_driving"],
        "combined_kkt_inf": float(final_eval["combined_inf"]),
        "support_needs_add_drop": bool(
            (final_eval["status"] != "ok")
            or (float(final_eval["inactive"]["max_positive_inactive_driving"]) > 1.0e-8)
            or bool(jnp.any(final_eval["m_support"] <= active_m_threshold))
        ),
        "binding_element_indices": final_eval["element_bottlenecks"].get("tightest_budget_indices", []),
        "binding_element_names": final_eval["element_bottlenecks"].get("tightest_budget_names"),
        "binding_element_values": final_eval["element_bottlenecks"].get("tightest_budget_values", []),
        "element_bottlenecks": final_eval["element_bottlenecks"],
        "gas_result": final_eval["gas_result"],
        "joint_activation": joint_activation,
    }


def diagnose_support_updating_active_set_layer(
    state: ThermoState,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    hvector_cond_func: Callable[[float], jnp.ndarray],
    *,
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
    m0: Optional[jnp.ndarray] = None,
    ln_nk_gas_init: Optional[jnp.ndarray] = None,
    ln_ntot_gas_init: Optional[jnp.ndarray] = None,
    initial_support_lp_top_k: int = 20,
    gas_epsilon_crit: float = 1.0e-10,
    gas_max_iter: int = 1000,
    budget_negative_tol: float = 1.0e-14,
    active_m_threshold: float = 1.0e-24,
    outer_max_iter: int = 6,
    inner_max_nfev: int = 100,
    add_threshold: float = 1.0e-3,
    drop_threshold: float = -1.0e-3,
    active_gap_tol: float = 1.0e-3,
    max_additions_per_iter: int = 2,
    drop_m_threshold: float = 1.0e-18,
    default_initial_amount: float = 1.0e-30,
    prefer_binding_additions: bool = True,
    refresh_lp_every: int = 0,
) -> dict[str, Any]:
    """Diagnostic-only support-updating active-set loop for one condensate layer."""

    formula_matrix = jnp.asarray(formula_matrix)
    formula_matrix_cond = jnp.asarray(formula_matrix_cond)
    n_cond = int(formula_matrix_cond.shape[1])
    dtype = jnp.result_type(formula_matrix_cond.dtype, state.element_vector.dtype, jnp.float64)
    if m0 is None:
        m0 = jnp.zeros((n_cond,), dtype=dtype)
    else:
        m0 = jnp.asarray(m0, dtype=dtype)

    initial_diag = diagnose_condensate_outer_active_set_layer(
        state,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        m=m0,
        condensate_species=condensate_species,
        element_names=element_names,
        ln_nk_gas_init=ln_nk_gas_init,
        ln_ntot_gas_init=ln_ntot_gas_init,
        gas_epsilon_crit=gas_epsilon_crit,
        gas_max_iter=gas_max_iter,
        budget_negative_tol=budget_negative_tol,
        active_m_threshold=active_m_threshold,
        joint_activation_top_k=(initial_support_lp_top_k,),
    )
    initial_record = initial_diag["initial"]
    initial_lp = (
        initial_record["joint_activation"].get(str(initial_support_lp_top_k))
        if initial_record["status"] == "ok"
        else None
    )
    current_support = [] if initial_lp is None else list(initial_lp["support_indices"])
    current_support_amounts = (
        jnp.zeros((0,), dtype=dtype)
        if initial_lp is None
        else jnp.asarray(initial_lp["support_amounts"], dtype=dtype)
    )

    history: list[dict[str, Any]] = []
    stabilized = False
    start = time.perf_counter()
    last_result: Optional[dict[str, Any]] = None

    for outer_iter in range(int(outer_max_iter)):
        if refresh_lp_every > 0 and outer_iter > 0 and (outer_iter % refresh_lp_every) == 0 and last_result is not None:
            refresh_diag = diagnose_condensate_outer_active_set_layer(
                state,
                formula_matrix,
                formula_matrix_cond,
                hvector_func,
                hvector_cond_func,
                m=last_result["m_full"],
                condensate_species=condensate_species,
                element_names=element_names,
                ln_nk_gas_init=ln_nk_gas_init,
                ln_ntot_gas_init=ln_ntot_gas_init,
                gas_epsilon_crit=gas_epsilon_crit,
                gas_max_iter=gas_max_iter,
                budget_negative_tol=budget_negative_tol,
                active_m_threshold=active_m_threshold,
                joint_activation_top_k=(initial_support_lp_top_k,),
            )
            refresh_lp = refresh_diag["initial"]["joint_activation"].get(str(initial_support_lp_top_k))
            if refresh_lp is not None:
                merged_support = sorted(set(current_support).union(int(i) for i in refresh_lp["support_indices"]))
                if merged_support != current_support:
                    current_support = merged_support
                    current_support_amounts = _support_amounts_from_full(
                        current_support,
                        last_result["m_full"],
                        default_amount=default_initial_amount,
                        dtype=dtype,
                    )

        restricted = solve_restricted_support_condensate_layer(
            state,
            formula_matrix,
            formula_matrix_cond,
            hvector_func,
            hvector_cond_func,
            support_indices=current_support,
            condensate_species=condensate_species,
            element_names=element_names,
            support_amounts_init=current_support_amounts,
            joint_activation=initial_lp if outer_iter == 0 else None,
            ln_nk_gas_init=ln_nk_gas_init,
            ln_ntot_gas_init=ln_ntot_gas_init,
            gas_epsilon_crit=gas_epsilon_crit,
            gas_max_iter=gas_max_iter,
            budget_negative_tol=budget_negative_tol,
            active_m_threshold=active_m_threshold,
            default_initial_amount=default_initial_amount,
            least_squares_max_nfev=inner_max_nfev,
        )
        last_result = restricted
        support_before = list(current_support)
        support_before_names = _restricted_support_names(support_before, condensate_species)

        drop_indices = []
        for local_idx, global_idx in enumerate(support_before):
            amount = float(restricted["m_support"][local_idx])
            drive = float(restricted["driving_support"][local_idx])
            if amount <= drop_m_threshold and drive <= drop_threshold:
                drop_indices.append(global_idx)

        ranked_additions = _rank_inactive_candidates(
            driving_full=jnp.asarray(restricted["driving_full"]),
            current_support=support_before,
            formula_matrix_cond=formula_matrix_cond,
            binding_element_indices=restricted["binding_element_indices"],
            add_threshold=add_threshold,
            prefer_binding_elements=prefer_binding_additions,
        )
        add_indices = ranked_additions[: max_additions_per_iter]

        next_support = sorted(set(support_before).difference(drop_indices).union(add_indices))
        no_changes = next_support == support_before
        merit_small = (
            restricted["b_eff_feasible"]
            and restricted["restricted_kkt_gap_inf"] <= active_gap_tol
            and restricted["max_positive_inactive_driving"] <= add_threshold
        )
        stabilized = no_changes and merit_small

        history.append(
            {
                "outer_iter": outer_iter,
                "support_before_indices": support_before,
                "support_before_names": support_before_names,
                "support_size_before": len(support_before),
                "solve": restricted,
                "drop_indices": drop_indices,
                "drop_names": _restricted_support_names(drop_indices, condensate_species),
                "add_indices": add_indices,
                "add_names": _restricted_support_names(add_indices, condensate_species),
                "support_after_indices": next_support,
                "support_after_names": _restricted_support_names(next_support, condensate_species),
                "support_size_after": len(next_support),
                "combined_merit": max(
                    float(restricted["restricted_kkt_gap_inf"]),
                    float(restricted["max_positive_inactive_driving"]),
                    float(restricted["negative_budget_inf"]),
                ),
                "stabilized": stabilized,
            }
        )

        current_support = next_support
        current_support_amounts = _support_amounts_from_full(
            current_support,
            restricted["m_full"],
            default_amount=default_initial_amount,
            dtype=dtype,
        )

        if stabilized or no_changes:
            break

    elapsed = time.perf_counter() - start
    final_record = history[-1] if history else None
    return {
        "prototype_family": "support_updating_condensate_active_set_diagnostic",
        "initial_diagnostic": initial_diag,
        "initial_lp_top_k": int(initial_support_lp_top_k),
        "initial_lp_support_indices": [] if initial_lp is None else list(initial_lp["support_indices"]),
        "initial_lp_support_names": None if initial_lp is None else initial_lp["support_names"],
        "initial_lp_support_size": 0 if initial_lp is None else int(initial_lp["support_size"]),
        "history": history,
        "final": None if final_record is None else final_record["solve"],
        "final_support_indices": [] if final_record is None else final_record["support_after_indices"],
        "final_support_names": None if final_record is None else final_record["support_after_names"],
        "final_support_size": 0 if final_record is None else int(final_record["support_size_after"]),
        "stabilized": bool(stabilized),
        "outer_iterations_completed": len(history),
        "runtime_seconds": float(elapsed),
    }


def solve_semismooth_candidate_condensate_layer(
    state: ThermoState,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    hvector_cond_func: Callable[[float], jnp.ndarray],
    *,
    candidate_indices: Sequence[int],
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
    candidate_amounts_init: Optional[jnp.ndarray] = None,
    ln_nk_gas_init: Optional[jnp.ndarray] = None,
    ln_ntot_gas_init: Optional[jnp.ndarray] = None,
    gas_epsilon_crit: float = 1.0e-10,
    gas_max_iter: int = 1000,
    budget_negative_tol: float = 1.0e-14,
    feasibility_penalty_weight: float = 1.0e6,
    inactive_top_k: int = 5,
    default_initial_amount: float = 1.0e-30,
    least_squares_ftol: float = 1.0e-10,
    least_squares_xtol: float = 1.0e-10,
    least_squares_gtol: float = 1.0e-10,
    least_squares_max_nfev: int = 100,
) -> dict[str, Any]:
    """Diagnostic-only semismooth complementarity solve on a restricted candidate set."""

    formula_matrix = jnp.asarray(formula_matrix)
    formula_matrix_cond = jnp.asarray(formula_matrix_cond)
    candidate_indices = [int(i) for i in candidate_indices]
    n_cond = int(formula_matrix_cond.shape[1])
    dtype = jnp.result_type(formula_matrix_cond.dtype, state.element_vector.dtype, jnp.float64)
    if _scipy_least_squares is None:  # pragma: no cover - fallback only
        raise RuntimeError("scipy.optimize.least_squares is required for semismooth diagnostics.")

    candidate_mask = _support_mask(n_cond, candidate_indices)
    candidate_names = _restricted_support_names(candidate_indices, condensate_species)
    hvector_cond = jnp.asarray(hvector_cond_func(state.temperature), dtype=dtype)
    candidate_matrix = formula_matrix_cond[:, jnp.asarray(candidate_indices, dtype=jnp.int32)] if candidate_indices else jnp.zeros((formula_matrix_cond.shape[0], 0), dtype=dtype)
    if candidate_amounts_init is None:
        x0 = jnp.full((len(candidate_indices),), default_initial_amount, dtype=dtype)
    else:
        x0 = jnp.maximum(jnp.asarray(candidate_amounts_init, dtype=dtype), 0.0)
        if x0.shape != (len(candidate_indices),):
            raise ValueError("candidate_amounts_init must have shape (len(candidate_indices),).")

    evaluations: dict[str, Any] = {"count": 0, "gas_solves": 0}

    def _evaluate(m_candidate: jnp.ndarray) -> dict[str, Any]:
        evaluations["count"] += 1
        m_candidate = jnp.asarray(m_candidate, dtype=dtype)
        m_full = _assemble_support_amounts(n_cond, candidate_indices, m_candidate, dtype)
        b_eff = state.element_vector - formula_matrix_cond @ m_full
        negative_budget = jnp.maximum(-b_eff, 0.0)
        feasibility_residual = feasibility_penalty_weight * negative_budget
        infeasible = bool(jnp.any(b_eff < -budget_negative_tol))

        if infeasible:
            return {
                "status": "infeasible_b_eff",
                "m_candidate": m_candidate,
                "m_full": m_full,
                "b_eff": b_eff,
                "negative_budget": negative_budget,
                "candidate_slack": jnp.zeros((len(candidate_indices),), dtype=dtype),
                "candidate_fb": _fischer_burmeister(m_candidate, jnp.zeros((len(candidate_indices),), dtype=dtype)),
                "driving_full": jnp.full((n_cond,), jnp.nan, dtype=dtype),
                "gas_result": None,
                "inactive": {
                    "max_positive_inactive_driving": float("nan"),
                    "positive_inactive_count": 0,
                    "top_indices": [],
                    "top_names": [] if condensate_species is not None else None,
                    "top_driving": [],
                },
                "element_bottlenecks": {
                    "tightest_budget_indices": [int(i) for i in jnp.argsort(b_eff)[: min(5, b_eff.shape[0])]],
                    "tightest_budget_names": None if element_names is None else [str(element_names[i]) for i in jnp.argsort(b_eff)[: min(5, b_eff.shape[0])]],
                    "tightest_budget_values": [float(b_eff[i]) for i in jnp.argsort(b_eff)[: min(5, b_eff.shape[0])]],
                },
                "residual_vector": jnp.concatenate([
                    _fischer_burmeister(m_candidate, jnp.zeros((len(candidate_indices),), dtype=dtype)),
                    feasibility_residual,
                ]),
            }

        gas_state = ThermoState(
            temperature=state.temperature,
            ln_normalized_pressure=state.ln_normalized_pressure,
            element_vector=b_eff,
        )
        gas_result = solve_gas_equilibrium_with_duals(
            gas_state,
            formula_matrix,
            hvector_func,
            ln_nk_init=ln_nk_gas_init,
            ln_ntot_init=ln_ntot_gas_init,
            epsilon_crit=gas_epsilon_crit,
            max_iter=gas_max_iter,
        )
        evaluations["gas_solves"] += 1
        driving_full = formula_matrix_cond.T @ gas_result["pi_vector"] - hvector_cond
        candidate_driving = driving_full[jnp.asarray(candidate_indices, dtype=jnp.int32)] if candidate_indices else jnp.zeros((0,), dtype=dtype)
        candidate_slack = -candidate_driving
        candidate_fb = _fischer_burmeister(m_candidate, candidate_slack)
        inactive = _top_inactive_violators(
            driving_full,
            candidate_mask,
            condensate_species,
            inactive_top_k,
        )
        element_bottlenecks = _element_bottleneck_summary(
            b_eff=b_eff,
            pi_vector=gas_result["pi_vector"],
            formula_matrix_cond=formula_matrix_cond,
            positive_indices=[idx for idx in range(n_cond) if float(driving_full[idx]) > 0.0],
            top_positive_indices=_top_indices(driving_full, descending=True, top_k=min(inactive_top_k, n_cond)),
            element_names=element_names,
            condensate_species=condensate_species,
        )
        return {
            "status": "ok",
            "m_candidate": m_candidate,
            "m_full": m_full,
            "b_eff": b_eff,
            "negative_budget": negative_budget,
            "candidate_slack": candidate_slack,
            "candidate_fb": candidate_fb,
            "driving_full": driving_full,
            "gas_result": gas_result,
            "inactive": inactive,
            "element_bottlenecks": element_bottlenecks,
            "residual_vector": jnp.concatenate([candidate_fb, feasibility_residual]),
        }

    def _residual_numpy(m_candidate: np.ndarray) -> np.ndarray:
        current = _evaluate(jnp.asarray(m_candidate, dtype=dtype))
        return np.asarray(current["residual_vector"], dtype=np.float64)

    start = time.perf_counter()
    scipy_result = _scipy_least_squares(
        _residual_numpy,
        x0=np.asarray(x0, dtype=np.float64),
        bounds=(0.0, np.inf),
        ftol=least_squares_ftol,
        xtol=least_squares_xtol,
        gtol=least_squares_gtol,
        max_nfev=least_squares_max_nfev,
    )
    elapsed = time.perf_counter() - start
    final_eval = _evaluate(jnp.asarray(scipy_result.x, dtype=dtype))
    feasible_projection_alpha = None
    raw_final_status = final_eval["status"]
    if final_eval["status"] == "infeasible_b_eff":
        feasible_projection_alpha = _max_feasible_scaling(final_eval["m_full"], formula_matrix_cond, state.element_vector)
        if feasible_projection_alpha > 0.0:
            final_eval = _evaluate(feasible_projection_alpha * final_eval["m_candidate"])

    fb_inf = float(jnp.max(jnp.abs(final_eval["candidate_fb"]))) if candidate_indices else 0.0
    fb_norm = float(jnp.linalg.norm(final_eval["candidate_fb"])) if candidate_indices else 0.0
    neg_budget_inf = float(jnp.max(final_eval["negative_budget"])) if final_eval["negative_budget"].size else 0.0
    combined_inf = max(fb_inf, neg_budget_inf)
    return {
        "prototype_family": "restricted_candidate_semismooth_complementarity_diagnostic",
        "candidate_indices": candidate_indices,
        "candidate_names": candidate_names,
        "candidate_size": len(candidate_indices),
        "status": final_eval["status"],
        "raw_final_status": raw_final_status,
        "feasible_projection_alpha": feasible_projection_alpha,
        "solver_success": bool(scipy_result.success) and final_eval["status"] == "ok",
        "solver_status": int(scipy_result.status),
        "solver_message": str(scipy_result.message),
        "solver_cost": float(scipy_result.cost),
        "solver_optimality": float(scipy_result.optimality),
        "runtime_seconds": float(elapsed),
        "nfev": int(scipy_result.nfev),
        "njev": None if scipy_result.njev is None else int(scipy_result.njev),
        "evaluation_count": int(evaluations["count"]),
        "gas_solve_count": int(evaluations["gas_solves"]),
        "m_candidate": final_eval["m_candidate"],
        "m_full": final_eval["m_full"],
        "candidate_slack": final_eval["candidate_slack"],
        "candidate_fb": final_eval["candidate_fb"],
        "fb_residual_inf": fb_inf,
        "fb_residual_norm": fb_norm,
        "negative_budget_inf": neg_budget_inf,
        "b_eff": final_eval["b_eff"],
        "b_eff_feasible": bool(jnp.all(final_eval["b_eff"] >= -budget_negative_tol)),
        "driving_full": final_eval["driving_full"],
        "max_positive_inactive_driving": float(final_eval["inactive"]["max_positive_inactive_driving"]),
        "inactive_positive_count": int(final_eval["inactive"]["positive_inactive_count"]),
        "top_inactive_indices": final_eval["inactive"]["top_indices"],
        "top_inactive_names": final_eval["inactive"]["top_names"],
        "top_inactive_driving": final_eval["inactive"]["top_driving"],
        "combined_merit_inf": combined_inf,
        "element_bottlenecks": final_eval["element_bottlenecks"],
        "binding_element_indices": final_eval["element_bottlenecks"].get("tightest_budget_indices", []),
        "binding_element_names": final_eval["element_bottlenecks"].get("tightest_budget_names"),
        "binding_element_values": final_eval["element_bottlenecks"].get("tightest_budget_values", []),
        "gas_result": final_eval["gas_result"],
        "candidate_self_consistent": bool(
            final_eval["status"] == "ok"
            and fb_inf <= 1.0e-6
            and neg_budget_inf <= budget_negative_tol
            and float(final_eval["inactive"]["max_positive_inactive_driving"]) <= 1.0e-6
        ),
    }


def diagnose_semismooth_candidate_condensate_layer(
    state: ThermoState,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    hvector_cond_func: Callable[[float], jnp.ndarray],
    *,
    candidate_lp_top_k: int = 20,
    augment_inactive_violators: int = 0,
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
    m0: Optional[jnp.ndarray] = None,
    ln_nk_gas_init: Optional[jnp.ndarray] = None,
    ln_ntot_gas_init: Optional[jnp.ndarray] = None,
    gas_epsilon_crit: float = 1.0e-10,
    gas_max_iter: int = 1000,
    budget_negative_tol: float = 1.0e-14,
    feasibility_penalty_weight: float = 1.0e6,
    inactive_top_k: int = 5,
    default_initial_amount: float = 1.0e-30,
    least_squares_max_nfev: int = 100,
) -> dict[str, Any]:
    """Build an LP-seeded candidate set and run the restricted semismooth solve."""

    dtype = jnp.result_type(jnp.asarray(formula_matrix_cond).dtype, state.element_vector.dtype, jnp.float64)
    if m0 is None:
        m0 = jnp.zeros((int(jnp.asarray(formula_matrix_cond).shape[1]),), dtype=dtype)
    initial_diag = diagnose_condensate_outer_active_set_layer(
        state,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        m=m0,
        condensate_species=condensate_species,
        element_names=element_names,
        ln_nk_gas_init=ln_nk_gas_init,
        ln_ntot_gas_init=ln_ntot_gas_init,
        gas_epsilon_crit=gas_epsilon_crit,
        gas_max_iter=gas_max_iter,
        budget_negative_tol=budget_negative_tol,
        joint_activation_top_k=(candidate_lp_top_k,),
    )
    if initial_diag["initial"]["status"] != "ok":
        return {
            "prototype_family": "restricted_candidate_semismooth_complementarity_diagnostic",
            "initial_diagnostic": initial_diag,
            "status": initial_diag["initial"]["status"],
            "candidate_indices": [],
            "candidate_names": [],
        }

    initial_lp = initial_diag["initial"]["joint_activation"][str(candidate_lp_top_k)]
    candidate_indices = list(initial_lp["support_indices"])
    candidate_amounts = jnp.asarray(initial_lp["support_amounts"], dtype=dtype)

    initial_semismooth = solve_semismooth_candidate_condensate_layer(
        state,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        candidate_indices=candidate_indices,
        candidate_amounts_init=candidate_amounts,
        condensate_species=condensate_species,
        element_names=element_names,
        ln_nk_gas_init=ln_nk_gas_init,
        ln_ntot_gas_init=ln_ntot_gas_init,
        gas_epsilon_crit=gas_epsilon_crit,
        gas_max_iter=gas_max_iter,
        budget_negative_tol=budget_negative_tol,
        feasibility_penalty_weight=feasibility_penalty_weight,
        inactive_top_k=inactive_top_k,
        default_initial_amount=default_initial_amount,
        least_squares_max_nfev=least_squares_max_nfev,
    )

    augmented = None
    if augment_inactive_violators > 0 and initial_semismooth["status"] == "ok":
        additions = initial_semismooth["top_inactive_indices"][:augment_inactive_violators]
        augmented_indices = sorted(set(candidate_indices).union(int(i) for i in additions))
        if augmented_indices != candidate_indices:
            augmented = solve_semismooth_candidate_condensate_layer(
                state,
                formula_matrix,
                formula_matrix_cond,
                hvector_func,
                hvector_cond_func,
                candidate_indices=augmented_indices,
                candidate_amounts_init=_support_amounts_from_full(
                    augmented_indices,
                    initial_semismooth["m_full"],
                    default_amount=default_initial_amount,
                    dtype=dtype,
                ),
                condensate_species=condensate_species,
                element_names=element_names,
                ln_nk_gas_init=ln_nk_gas_init,
                ln_ntot_gas_init=ln_ntot_gas_init,
                gas_epsilon_crit=gas_epsilon_crit,
                gas_max_iter=gas_max_iter,
                budget_negative_tol=budget_negative_tol,
                feasibility_penalty_weight=feasibility_penalty_weight,
                inactive_top_k=inactive_top_k,
                default_initial_amount=default_initial_amount,
                least_squares_max_nfev=least_squares_max_nfev,
            )
            augmented["added_candidate_indices"] = additions
            augmented["added_candidate_names"] = _restricted_support_names(additions, condensate_species)

    return {
        "prototype_family": "restricted_candidate_semismooth_complementarity_diagnostic",
        "initial_diagnostic": initial_diag,
        "initial_lp_top_k": int(candidate_lp_top_k),
        "initial_lp_support_indices": candidate_indices,
        "initial_lp_support_names": initial_lp["support_names"],
        "initial_lp_support_size": int(initial_lp["support_size"]),
        "initial_semismooth": initial_semismooth,
        "augmented_semismooth": augmented,
    }


def solve_smoothed_semismooth_candidate_condensate_layer(
    state: ThermoState,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    hvector_cond_func: Callable[[float], jnp.ndarray],
    *,
    candidate_indices: Sequence[int],
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
    candidate_amounts_init: Optional[jnp.ndarray] = None,
    mu_schedule: Sequence[float] = (1.0e2, 1.0e1, 1.0e0, 1.0e-1, 1.0e-2),
    ln_nk_gas_init: Optional[jnp.ndarray] = None,
    ln_ntot_gas_init: Optional[jnp.ndarray] = None,
    gas_epsilon_crit: float = 1.0e-10,
    gas_max_iter: int = 1000,
    budget_negative_tol: float = 1.0e-14,
    feasibility_penalty_weight: float = 1.0e6,
    inactive_top_k: int = 5,
    default_initial_amount: float = 1.0e-30,
    least_squares_ftol: float = 1.0e-10,
    least_squares_xtol: float = 1.0e-10,
    least_squares_gtol: float = 1.0e-10,
    least_squares_max_nfev: int = 100,
) -> dict[str, Any]:
    """Diagnostic-only smoothed semismooth continuation solve on a restricted candidate set."""

    formula_matrix = jnp.asarray(formula_matrix)
    formula_matrix_cond = jnp.asarray(formula_matrix_cond)
    candidate_indices = [int(i) for i in candidate_indices]
    n_cond = int(formula_matrix_cond.shape[1])
    dtype = jnp.result_type(formula_matrix_cond.dtype, state.element_vector.dtype, jnp.float64)
    if _scipy_least_squares is None:  # pragma: no cover - fallback only
        raise RuntimeError("scipy.optimize.least_squares is required for smoothed semismooth diagnostics.")

    candidate_mask = _support_mask(n_cond, candidate_indices)
    candidate_names = _restricted_support_names(candidate_indices, condensate_species)
    hvector_cond = jnp.asarray(hvector_cond_func(state.temperature), dtype=dtype)
    if candidate_amounts_init is None:
        current_x = jnp.full((len(candidate_indices),), default_initial_amount, dtype=dtype)
    else:
        current_x = jnp.maximum(jnp.asarray(candidate_amounts_init, dtype=dtype), 0.0)
        if current_x.shape != (len(candidate_indices),):
            raise ValueError("candidate_amounts_init must have shape (len(candidate_indices),).")

    evaluations: dict[str, Any] = {"count": 0, "gas_solves": 0}

    def _evaluate(m_candidate: jnp.ndarray, mu: float) -> dict[str, Any]:
        evaluations["count"] += 1
        m_candidate = jnp.asarray(m_candidate, dtype=dtype)
        m_full = _assemble_support_amounts(n_cond, candidate_indices, m_candidate, dtype)
        b_eff = state.element_vector - formula_matrix_cond @ m_full
        negative_budget = jnp.maximum(-b_eff, 0.0)
        feasibility_residual = feasibility_penalty_weight * negative_budget
        infeasible = bool(jnp.any(b_eff < -budget_negative_tol))

        if infeasible:
            zero_slack = jnp.zeros((len(candidate_indices),), dtype=dtype)
            smoothed_fb = _smoothed_fischer_burmeister(m_candidate, zero_slack, mu)
            raw_fb = _fischer_burmeister(m_candidate, zero_slack)
            return {
                "status": "infeasible_b_eff",
                "m_candidate": m_candidate,
                "m_full": m_full,
                "b_eff": b_eff,
                "negative_budget": negative_budget,
                "candidate_slack": zero_slack,
                "candidate_smoothed_fb": smoothed_fb,
                "candidate_raw_fb": raw_fb,
                "driving_full": jnp.full((n_cond,), jnp.nan, dtype=dtype),
                "gas_result": None,
                "inactive": {
                    "max_positive_inactive_driving": float("nan"),
                    "positive_inactive_count": 0,
                    "top_indices": [],
                    "top_names": [] if condensate_species is not None else None,
                    "top_driving": [],
                },
                "element_bottlenecks": {
                    "tightest_budget_indices": [int(i) for i in jnp.argsort(b_eff)[: min(5, b_eff.shape[0])]],
                    "tightest_budget_names": None if element_names is None else [str(element_names[i]) for i in jnp.argsort(b_eff)[: min(5, b_eff.shape[0])]],
                    "tightest_budget_values": [float(b_eff[i]) for i in jnp.argsort(b_eff)[: min(5, b_eff.shape[0])]],
                },
                "residual_vector": jnp.concatenate([smoothed_fb, feasibility_residual]),
            }

        gas_state = ThermoState(
            temperature=state.temperature,
            ln_normalized_pressure=state.ln_normalized_pressure,
            element_vector=b_eff,
        )
        gas_result = solve_gas_equilibrium_with_duals(
            gas_state,
            formula_matrix,
            hvector_func,
            ln_nk_init=ln_nk_gas_init,
            ln_ntot_init=ln_ntot_gas_init,
            epsilon_crit=gas_epsilon_crit,
            max_iter=gas_max_iter,
        )
        evaluations["gas_solves"] += 1
        driving_full = formula_matrix_cond.T @ gas_result["pi_vector"] - hvector_cond
        candidate_driving = driving_full[jnp.asarray(candidate_indices, dtype=jnp.int32)] if candidate_indices else jnp.zeros((0,), dtype=dtype)
        candidate_slack = -candidate_driving
        smoothed_fb = _smoothed_fischer_burmeister(m_candidate, candidate_slack, mu)
        raw_fb = _fischer_burmeister(m_candidate, candidate_slack)
        inactive = _top_inactive_violators(
            driving_full,
            candidate_mask,
            condensate_species,
            inactive_top_k,
        )
        element_bottlenecks = _element_bottleneck_summary(
            b_eff=b_eff,
            pi_vector=gas_result["pi_vector"],
            formula_matrix_cond=formula_matrix_cond,
            positive_indices=[idx for idx in range(n_cond) if float(driving_full[idx]) > 0.0],
            top_positive_indices=_top_indices(driving_full, descending=True, top_k=min(inactive_top_k, n_cond)),
            element_names=element_names,
            condensate_species=condensate_species,
        )
        return {
            "status": "ok",
            "m_candidate": m_candidate,
            "m_full": m_full,
            "b_eff": b_eff,
            "negative_budget": negative_budget,
            "candidate_slack": candidate_slack,
            "candidate_smoothed_fb": smoothed_fb,
            "candidate_raw_fb": raw_fb,
            "driving_full": driving_full,
            "gas_result": gas_result,
            "inactive": inactive,
            "element_bottlenecks": element_bottlenecks,
            "residual_vector": jnp.concatenate([smoothed_fb, feasibility_residual]),
        }

    stage_history = []
    final_eval = None
    start = time.perf_counter()
    for mu in [float(value) for value in mu_schedule]:
        def _residual_numpy(m_candidate: np.ndarray) -> np.ndarray:
            current = _evaluate(jnp.asarray(m_candidate, dtype=dtype), mu)
            return np.asarray(current["residual_vector"], dtype=np.float64)

        scipy_result = _scipy_least_squares(
            _residual_numpy,
            x0=np.asarray(current_x, dtype=np.float64),
            bounds=(0.0, np.inf),
            ftol=least_squares_ftol,
            xtol=least_squares_xtol,
            gtol=least_squares_gtol,
            max_nfev=least_squares_max_nfev,
        )
        current_x = jnp.asarray(scipy_result.x, dtype=dtype)
        stage_eval = _evaluate(current_x, mu)
        feasible_projection_alpha = None
        raw_final_status = stage_eval["status"]
        if stage_eval["status"] == "infeasible_b_eff":
            feasible_projection_alpha = _max_feasible_scaling(stage_eval["m_full"], formula_matrix_cond, state.element_vector)
            if feasible_projection_alpha > 0.0:
                current_x = feasible_projection_alpha * stage_eval["m_candidate"]
                stage_eval = _evaluate(current_x, mu)
        stage_history.append(
            {
                "mu": float(mu),
                "status": stage_eval["status"],
                "raw_final_status": raw_final_status,
                "feasible_projection_alpha": feasible_projection_alpha,
                "solver_success": bool(scipy_result.success) and stage_eval["status"] == "ok",
                "solver_status": int(scipy_result.status),
                "solver_message": str(scipy_result.message),
                "solver_cost": float(scipy_result.cost),
                "solver_optimality": float(scipy_result.optimality),
                "nfev": int(scipy_result.nfev),
                "njev": None if scipy_result.njev is None else int(scipy_result.njev),
                "smoothed_fb_inf": float(jnp.max(jnp.abs(stage_eval["candidate_smoothed_fb"]))) if candidate_indices else 0.0,
                "smoothed_fb_norm": float(jnp.linalg.norm(stage_eval["candidate_smoothed_fb"])) if candidate_indices else 0.0,
                "raw_fb_inf": float(jnp.max(jnp.abs(stage_eval["candidate_raw_fb"]))) if candidate_indices else 0.0,
                "raw_fb_norm": float(jnp.linalg.norm(stage_eval["candidate_raw_fb"])) if candidate_indices else 0.0,
                "negative_budget_inf": float(jnp.max(stage_eval["negative_budget"])) if stage_eval["negative_budget"].size else 0.0,
                "max_positive_inactive_driving": float(stage_eval["inactive"]["max_positive_inactive_driving"]),
                "inactive_positive_count": int(stage_eval["inactive"]["positive_inactive_count"]),
                "binding_element_names": stage_eval["element_bottlenecks"].get("tightest_budget_names"),
            }
        )
        final_eval = stage_eval

    elapsed = time.perf_counter() - start
    assert final_eval is not None
    final_smoothed_inf = float(jnp.max(jnp.abs(final_eval["candidate_smoothed_fb"]))) if candidate_indices else 0.0
    final_smoothed_norm = float(jnp.linalg.norm(final_eval["candidate_smoothed_fb"])) if candidate_indices else 0.0
    final_raw_inf = float(jnp.max(jnp.abs(final_eval["candidate_raw_fb"]))) if candidate_indices else 0.0
    final_raw_norm = float(jnp.linalg.norm(final_eval["candidate_raw_fb"])) if candidate_indices else 0.0
    neg_budget_inf = float(jnp.max(final_eval["negative_budget"])) if final_eval["negative_budget"].size else 0.0

    return {
        "prototype_family": "restricted_candidate_smoothed_semismooth_continuation_diagnostic",
        "candidate_indices": candidate_indices,
        "candidate_names": candidate_names,
        "candidate_size": len(candidate_indices),
        "mu_schedule": [float(value) for value in mu_schedule],
        "status": final_eval["status"],
        "solver_success": bool(stage_history[-1]["solver_success"]),
        "solver_status": int(stage_history[-1]["solver_status"]),
        "solver_message": str(stage_history[-1]["solver_message"]),
        "runtime_seconds": float(elapsed),
        "evaluation_count": int(evaluations["count"]),
        "gas_solve_count": int(evaluations["gas_solves"]),
        "stage_history": stage_history,
        "m_candidate": final_eval["m_candidate"],
        "m_full": final_eval["m_full"],
        "candidate_slack": final_eval["candidate_slack"],
        "smoothed_fb_residual_inf": final_smoothed_inf,
        "smoothed_fb_residual_norm": final_smoothed_norm,
        "raw_fb_residual_inf": final_raw_inf,
        "raw_fb_residual_norm": final_raw_norm,
        "negative_budget_inf": neg_budget_inf,
        "b_eff": final_eval["b_eff"],
        "b_eff_feasible": bool(jnp.all(final_eval["b_eff"] >= -budget_negative_tol)),
        "driving_full": final_eval["driving_full"],
        "max_positive_inactive_driving": float(final_eval["inactive"]["max_positive_inactive_driving"]),
        "inactive_positive_count": int(final_eval["inactive"]["positive_inactive_count"]),
        "top_inactive_indices": final_eval["inactive"]["top_indices"],
        "top_inactive_names": final_eval["inactive"]["top_names"],
        "top_inactive_driving": final_eval["inactive"]["top_driving"],
        "element_bottlenecks": final_eval["element_bottlenecks"],
        "binding_element_indices": final_eval["element_bottlenecks"].get("tightest_budget_indices", []),
        "binding_element_names": final_eval["element_bottlenecks"].get("tightest_budget_names"),
        "binding_element_values": final_eval["element_bottlenecks"].get("tightest_budget_values", []),
        "gas_result": final_eval["gas_result"],
        "candidate_self_consistent": bool(
            final_eval["status"] == "ok"
            and final_raw_inf <= 1.0e-6
            and neg_budget_inf <= budget_negative_tol
            and float(final_eval["inactive"]["max_positive_inactive_driving"]) <= 1.0e-6
        ),
    }


def diagnose_smoothed_semismooth_candidate_condensate_layer(
    state: ThermoState,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    hvector_cond_func: Callable[[float], jnp.ndarray],
    *,
    candidate_lp_top_k: int = 20,
    augment_inactive_violators: int = 0,
    drop_small_threshold: float = 1.0e-18,
    drop_slack_threshold: float = 1.0e-3,
    prefer_binding_additions: bool = True,
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
    m0: Optional[jnp.ndarray] = None,
    mu_schedule: Sequence[float] = (1.0e2, 1.0e1, 1.0e0, 1.0e-1, 1.0e-2),
    ln_nk_gas_init: Optional[jnp.ndarray] = None,
    ln_ntot_gas_init: Optional[jnp.ndarray] = None,
    gas_epsilon_crit: float = 1.0e-10,
    gas_max_iter: int = 1000,
    budget_negative_tol: float = 1.0e-14,
    feasibility_penalty_weight: float = 1.0e6,
    inactive_top_k: int = 5,
    default_initial_amount: float = 1.0e-30,
    least_squares_max_nfev: int = 100,
) -> dict[str, Any]:
    """LP-seeded smoothed semismooth continuation with optional one-pass augmentation/drop."""

    dtype = jnp.result_type(jnp.asarray(formula_matrix_cond).dtype, state.element_vector.dtype, jnp.float64)
    if m0 is None:
        m0 = jnp.zeros((int(jnp.asarray(formula_matrix_cond).shape[1]),), dtype=dtype)
    initial_diag = diagnose_condensate_outer_active_set_layer(
        state,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        m=m0,
        condensate_species=condensate_species,
        element_names=element_names,
        ln_nk_gas_init=ln_nk_gas_init,
        ln_ntot_gas_init=ln_ntot_gas_init,
        gas_epsilon_crit=gas_epsilon_crit,
        gas_max_iter=gas_max_iter,
        budget_negative_tol=budget_negative_tol,
        joint_activation_top_k=(candidate_lp_top_k,),
    )
    if initial_diag["initial"]["status"] != "ok":
        return {
            "prototype_family": "restricted_candidate_smoothed_semismooth_continuation_diagnostic",
            "initial_diagnostic": initial_diag,
            "status": initial_diag["initial"]["status"],
            "candidate_indices": [],
            "candidate_names": [],
        }

    initial_lp = initial_diag["initial"]["joint_activation"][str(candidate_lp_top_k)]
    candidate_indices = list(initial_lp["support_indices"])
    candidate_amounts = jnp.asarray(initial_lp["support_amounts"], dtype=dtype)
    initial_smoothed = solve_smoothed_semismooth_candidate_condensate_layer(
        state,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        candidate_indices=candidate_indices,
        candidate_amounts_init=candidate_amounts,
        condensate_species=condensate_species,
        element_names=element_names,
        mu_schedule=mu_schedule,
        ln_nk_gas_init=ln_nk_gas_init,
        ln_ntot_gas_init=ln_ntot_gas_init,
        gas_epsilon_crit=gas_epsilon_crit,
        gas_max_iter=gas_max_iter,
        budget_negative_tol=budget_negative_tol,
        feasibility_penalty_weight=feasibility_penalty_weight,
        inactive_top_k=inactive_top_k,
        default_initial_amount=default_initial_amount,
        least_squares_max_nfev=least_squares_max_nfev,
    )

    adjusted = None
    if initial_smoothed["status"] == "ok":
        keep_indices = []
        dropped = []
        candidate_index_set = set(candidate_indices)
        for local_idx, global_idx in enumerate(candidate_indices):
            amount = float(initial_smoothed["m_candidate"][local_idx])
            slack = float(initial_smoothed["candidate_slack"][local_idx])
            if amount <= drop_small_threshold and slack >= drop_slack_threshold:
                dropped.append(global_idx)
            else:
                keep_indices.append(global_idx)

        ranked_additions = _rank_inactive_candidates(
            driving_full=jnp.asarray(initial_smoothed["driving_full"]),
            current_support=keep_indices,
            formula_matrix_cond=jnp.asarray(formula_matrix_cond),
            binding_element_indices=initial_smoothed["binding_element_indices"],
            add_threshold=0.0,
            prefer_binding_elements=prefer_binding_additions,
        )
        additions = ranked_additions[:augment_inactive_violators]
        updated_indices = sorted(set(keep_indices).union(additions))
        if updated_indices != candidate_indices:
            adjusted = solve_smoothed_semismooth_candidate_condensate_layer(
                state,
                formula_matrix,
                formula_matrix_cond,
                hvector_func,
                hvector_cond_func,
                candidate_indices=updated_indices,
                candidate_amounts_init=_support_amounts_from_full(
                    updated_indices,
                    initial_smoothed["m_full"],
                    default_amount=default_initial_amount,
                    dtype=dtype,
                ),
                condensate_species=condensate_species,
                element_names=element_names,
                mu_schedule=mu_schedule,
                ln_nk_gas_init=ln_nk_gas_init,
                ln_ntot_gas_init=ln_ntot_gas_init,
                gas_epsilon_crit=gas_epsilon_crit,
                gas_max_iter=gas_max_iter,
                budget_negative_tol=budget_negative_tol,
                feasibility_penalty_weight=feasibility_penalty_weight,
                inactive_top_k=inactive_top_k,
                default_initial_amount=default_initial_amount,
                least_squares_max_nfev=least_squares_max_nfev,
            )
            adjusted["added_candidate_indices"] = additions
            adjusted["added_candidate_names"] = _restricted_support_names(additions, condensate_species)
            adjusted["dropped_candidate_indices"] = dropped
            adjusted["dropped_candidate_names"] = _restricted_support_names(dropped, condensate_species)

    return {
        "prototype_family": "restricted_candidate_smoothed_semismooth_continuation_diagnostic",
        "initial_diagnostic": initial_diag,
        "initial_lp_top_k": int(candidate_lp_top_k),
        "initial_lp_support_indices": candidate_indices,
        "initial_lp_support_names": initial_lp["support_names"],
        "initial_lp_support_size": int(initial_lp["support_size"]),
        "mu_schedule": [float(value) for value in mu_schedule],
        "initial_smoothed": initial_smoothed,
        "adjusted_smoothed": adjusted,
    }


def solve_augmented_semismooth_candidate_condensate_layer(
    state: ThermoState,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    hvector_cond_func: Callable[[float], jnp.ndarray],
    *,
    candidate_indices: Sequence[int],
    inactive_indices: Sequence[int],
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
    candidate_amounts_init: Optional[jnp.ndarray] = None,
    mu_schedule: Sequence[float] = (1.0e2, 1.0e1, 1.0e0, 1.0e-1, 1.0e-2),
    active_weight: float = 1.0,
    inactive_weight: float = 1.0,
    budget_weight: float = 1.0e6,
    ln_nk_gas_init: Optional[jnp.ndarray] = None,
    ln_ntot_gas_init: Optional[jnp.ndarray] = None,
    gas_epsilon_crit: float = 1.0e-10,
    gas_max_iter: int = 1000,
    budget_negative_tol: float = 1.0e-14,
    inactive_top_k: int = 5,
    default_initial_amount: float = 1.0e-30,
    least_squares_ftol: float = 1.0e-10,
    least_squares_xtol: float = 1.0e-10,
    least_squares_gtol: float = 1.0e-10,
    least_squares_max_nfev: int = 100,
) -> dict[str, Any]:
    """Diagnostic-only augmented semismooth solve with active and inactive KKT residuals."""

    formula_matrix = jnp.asarray(formula_matrix)
    formula_matrix_cond = jnp.asarray(formula_matrix_cond)
    candidate_indices = [int(i) for i in candidate_indices]
    inactive_indices = [int(i) for i in inactive_indices if int(i) not in set(candidate_indices)]
    n_cond = int(formula_matrix_cond.shape[1])
    dtype = jnp.result_type(formula_matrix_cond.dtype, state.element_vector.dtype, jnp.float64)
    if _scipy_least_squares is None:  # pragma: no cover - fallback only
        raise RuntimeError("scipy.optimize.least_squares is required for augmented semismooth diagnostics.")

    candidate_mask = _support_mask(n_cond, candidate_indices)
    candidate_names = _restricted_support_names(candidate_indices, condensate_species)
    inactive_names = _restricted_support_names(inactive_indices, condensate_species)
    hvector_cond = jnp.asarray(hvector_cond_func(state.temperature), dtype=dtype)
    sqrt_active = jnp.sqrt(jnp.asarray(active_weight, dtype=dtype))
    sqrt_inactive = jnp.sqrt(jnp.asarray(inactive_weight, dtype=dtype))
    sqrt_budget = jnp.sqrt(jnp.asarray(budget_weight, dtype=dtype))
    if candidate_amounts_init is None:
        current_x = jnp.full((len(candidate_indices),), default_initial_amount, dtype=dtype)
    else:
        current_x = jnp.maximum(jnp.asarray(candidate_amounts_init, dtype=dtype), 0.0)
        if current_x.shape != (len(candidate_indices),):
            raise ValueError("candidate_amounts_init must have shape (len(candidate_indices),).")

    evaluations: dict[str, Any] = {"count": 0, "gas_solves": 0}

    def _evaluate(m_candidate: jnp.ndarray, mu: float) -> dict[str, Any]:
        evaluations["count"] += 1
        m_candidate = jnp.asarray(m_candidate, dtype=dtype)
        m_full = _assemble_support_amounts(n_cond, candidate_indices, m_candidate, dtype)
        b_eff = state.element_vector - formula_matrix_cond @ m_full
        negative_budget = jnp.maximum(-b_eff, 0.0)
        budget_residual = sqrt_budget * negative_budget
        infeasible = bool(jnp.any(b_eff < -budget_negative_tol))

        if infeasible:
            zero_slack = jnp.zeros((len(candidate_indices),), dtype=dtype)
            active_smoothed = sqrt_active * _smoothed_fischer_burmeister(m_candidate, zero_slack, mu)
            active_raw = _fischer_burmeister(m_candidate, zero_slack)
            inactive_residual = jnp.zeros((len(inactive_indices),), dtype=dtype)
            return {
                "status": "infeasible_b_eff",
                "m_candidate": m_candidate,
                "m_full": m_full,
                "b_eff": b_eff,
                "negative_budget": negative_budget,
                "candidate_slack": zero_slack,
                "active_smoothed_fb": active_smoothed,
                "active_raw_fb": active_raw,
                "inactive_residual": inactive_residual,
                "driving_full": jnp.full((n_cond,), jnp.nan, dtype=dtype),
                "gas_result": None,
                "inactive": {
                    "max_positive_inactive_driving": float("nan"),
                    "positive_inactive_count": 0,
                    "top_indices": [],
                    "top_names": [] if condensate_species is not None else None,
                    "top_driving": [],
                },
                "element_bottlenecks": {
                    "tightest_budget_indices": [int(i) for i in jnp.argsort(b_eff)[: min(5, b_eff.shape[0])]],
                    "tightest_budget_names": None if element_names is None else [str(element_names[i]) for i in jnp.argsort(b_eff)[: min(5, b_eff.shape[0])]],
                    "tightest_budget_values": [float(b_eff[i]) for i in jnp.argsort(b_eff)[: min(5, b_eff.shape[0])]],
                },
                "residual_vector": jnp.concatenate([active_smoothed, inactive_residual, budget_residual]),
            }

        gas_state = ThermoState(
            temperature=state.temperature,
            ln_normalized_pressure=state.ln_normalized_pressure,
            element_vector=b_eff,
        )
        gas_result = solve_gas_equilibrium_with_duals(
            gas_state,
            formula_matrix,
            hvector_func,
            ln_nk_init=ln_nk_gas_init,
            ln_ntot_init=ln_ntot_gas_init,
            epsilon_crit=gas_epsilon_crit,
            max_iter=gas_max_iter,
        )
        evaluations["gas_solves"] += 1
        driving_full = formula_matrix_cond.T @ gas_result["pi_vector"] - hvector_cond
        candidate_driving = driving_full[jnp.asarray(candidate_indices, dtype=jnp.int32)] if candidate_indices else jnp.zeros((0,), dtype=dtype)
        candidate_slack = -candidate_driving
        active_smoothed = sqrt_active * _smoothed_fischer_burmeister(m_candidate, candidate_slack, mu)
        active_raw = _fischer_burmeister(m_candidate, candidate_slack)
        inactive_driving = driving_full[jnp.asarray(inactive_indices, dtype=jnp.int32)] if inactive_indices else jnp.zeros((0,), dtype=dtype)
        inactive_residual = sqrt_inactive * jnp.maximum(inactive_driving, 0.0)
        inactive_summary = _top_inactive_violators(
            driving_full,
            candidate_mask,
            condensate_species,
            inactive_top_k,
        )
        element_bottlenecks = _element_bottleneck_summary(
            b_eff=b_eff,
            pi_vector=gas_result["pi_vector"],
            formula_matrix_cond=formula_matrix_cond,
            positive_indices=[idx for idx in range(n_cond) if float(driving_full[idx]) > 0.0],
            top_positive_indices=_top_indices(driving_full, descending=True, top_k=min(inactive_top_k, n_cond)),
            element_names=element_names,
            condensate_species=condensate_species,
        )
        return {
            "status": "ok",
            "m_candidate": m_candidate,
            "m_full": m_full,
            "b_eff": b_eff,
            "negative_budget": negative_budget,
            "candidate_slack": candidate_slack,
            "active_smoothed_fb": active_smoothed,
            "active_raw_fb": active_raw,
            "inactive_residual": inactive_residual,
            "inactive_driving": inactive_driving,
            "driving_full": driving_full,
            "gas_result": gas_result,
            "inactive": inactive_summary,
            "element_bottlenecks": element_bottlenecks,
            "residual_vector": jnp.concatenate([active_smoothed, inactive_residual, budget_residual]),
        }

    stage_history = []
    final_eval = None
    start = time.perf_counter()
    for mu in [float(value) for value in mu_schedule]:
        def _residual_numpy(m_candidate: np.ndarray) -> np.ndarray:
            current = _evaluate(jnp.asarray(m_candidate, dtype=dtype), mu)
            return np.asarray(current["residual_vector"], dtype=np.float64)

        scipy_result = _scipy_least_squares(
            _residual_numpy,
            x0=np.asarray(current_x, dtype=np.float64),
            bounds=(0.0, np.inf),
            ftol=least_squares_ftol,
            xtol=least_squares_xtol,
            gtol=least_squares_gtol,
            max_nfev=least_squares_max_nfev,
        )
        current_x = jnp.asarray(scipy_result.x, dtype=dtype)
        stage_eval = _evaluate(current_x, mu)
        feasible_projection_alpha = None
        raw_final_status = stage_eval["status"]
        if stage_eval["status"] == "infeasible_b_eff":
            feasible_projection_alpha = _max_feasible_scaling(stage_eval["m_full"], formula_matrix_cond, state.element_vector)
            if feasible_projection_alpha > 0.0:
                current_x = feasible_projection_alpha * stage_eval["m_candidate"]
                stage_eval = _evaluate(current_x, mu)
        stage_history.append(
            {
                "mu": float(mu),
                "status": stage_eval["status"],
                "raw_final_status": raw_final_status,
                "feasible_projection_alpha": feasible_projection_alpha,
                "solver_success": bool(scipy_result.success) and stage_eval["status"] == "ok",
                "solver_status": int(scipy_result.status),
                "solver_message": str(scipy_result.message),
                "solver_cost": float(scipy_result.cost),
                "solver_optimality": float(scipy_result.optimality),
                "nfev": int(scipy_result.nfev),
                "njev": None if scipy_result.njev is None else int(scipy_result.njev),
                "active_smoothed_norm": float(jnp.linalg.norm(stage_eval["active_smoothed_fb"])) if candidate_indices else 0.0,
                "active_raw_norm": float(jnp.linalg.norm(stage_eval["active_raw_fb"])) if candidate_indices else 0.0,
                "inactive_norm": float(jnp.linalg.norm(stage_eval["inactive_residual"])) if inactive_indices else 0.0,
                "budget_norm": float(jnp.linalg.norm(sqrt_budget * stage_eval["negative_budget"])) if stage_eval["negative_budget"].size else 0.0,
                "combined_norm": float(jnp.linalg.norm(stage_eval["residual_vector"])),
                "max_positive_inactive_driving": float(stage_eval["inactive"]["max_positive_inactive_driving"]),
                "inactive_positive_count": int(stage_eval["inactive"]["positive_inactive_count"]),
                "binding_element_names": stage_eval["element_bottlenecks"].get("tightest_budget_names"),
            }
        )
        final_eval = stage_eval

    elapsed = time.perf_counter() - start
    assert final_eval is not None
    active_smoothed_norm = float(jnp.linalg.norm(final_eval["active_smoothed_fb"])) if candidate_indices else 0.0
    active_raw_norm = float(jnp.linalg.norm(final_eval["active_raw_fb"])) if candidate_indices else 0.0
    inactive_norm = float(jnp.linalg.norm(final_eval["inactive_residual"])) if inactive_indices else 0.0
    budget_norm = float(jnp.linalg.norm(sqrt_budget * final_eval["negative_budget"])) if final_eval["negative_budget"].size else 0.0
    combined_norm = float(jnp.linalg.norm(final_eval["residual_vector"]))
    neg_budget_inf = float(jnp.max(final_eval["negative_budget"])) if final_eval["negative_budget"].size else 0.0

    return {
        "prototype_family": "augmented_candidate_smoothed_semismooth_diagnostic",
        "candidate_indices": candidate_indices,
        "candidate_names": candidate_names,
        "candidate_size": len(candidate_indices),
        "inactive_indices": inactive_indices,
        "inactive_names": inactive_names,
        "inactive_size": len(inactive_indices),
        "mu_schedule": [float(value) for value in mu_schedule],
        "weights": {
            "active_weight": float(active_weight),
            "inactive_weight": float(inactive_weight),
            "budget_weight": float(budget_weight),
        },
        "status": final_eval["status"],
        "solver_success": bool(stage_history[-1]["solver_success"]),
        "solver_status": int(stage_history[-1]["solver_status"]),
        "solver_message": str(stage_history[-1]["solver_message"]),
        "runtime_seconds": float(elapsed),
        "evaluation_count": int(evaluations["count"]),
        "gas_solve_count": int(evaluations["gas_solves"]),
        "stage_history": stage_history,
        "m_candidate": final_eval["m_candidate"],
        "m_full": final_eval["m_full"],
        "candidate_slack": final_eval["candidate_slack"],
        "active_smoothed_residual_norm": active_smoothed_norm,
        "active_raw_fb_norm": active_raw_norm,
        "inactive_residual_norm": inactive_norm,
        "combined_residual_norm": combined_norm,
        "negative_budget_norm": budget_norm,
        "negative_budget_inf": neg_budget_inf,
        "b_eff": final_eval["b_eff"],
        "b_eff_feasible": bool(jnp.all(final_eval["b_eff"] >= -budget_negative_tol)),
        "driving_full": final_eval["driving_full"],
        "max_positive_inactive_driving": float(final_eval["inactive"]["max_positive_inactive_driving"]),
        "inactive_positive_count": int(final_eval["inactive"]["positive_inactive_count"]),
        "top_inactive_indices": final_eval["inactive"]["top_indices"],
        "top_inactive_names": final_eval["inactive"]["top_names"],
        "top_inactive_driving": final_eval["inactive"]["top_driving"],
        "element_bottlenecks": final_eval["element_bottlenecks"],
        "binding_element_indices": final_eval["element_bottlenecks"].get("tightest_budget_indices", []),
        "binding_element_names": final_eval["element_bottlenecks"].get("tightest_budget_names"),
        "binding_element_values": final_eval["element_bottlenecks"].get("tightest_budget_values", []),
        "gas_result": final_eval["gas_result"],
        "materially_self_consistent": bool(
            final_eval["status"] == "ok"
            and active_raw_norm <= 1.0e-6
            and inactive_norm <= 1.0e-6
            and neg_budget_inf <= budget_negative_tol
        ),
    }


def diagnose_augmented_semismooth_candidate_condensate_layer(
    state: ThermoState,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    hvector_cond_func: Callable[[float], jnp.ndarray],
    *,
    candidate_lp_top_k: int = 20,
    inactive_violator_top_k: int = 3,
    inactive_policy: str = "naive_topk",
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
    m0: Optional[jnp.ndarray] = None,
    mu_schedule: Sequence[float] = (1.0e2, 1.0e1, 1.0e0, 1.0e-1, 1.0e-2),
    active_weight: float = 1.0,
    inactive_weight: float = 1.0,
    budget_weight: float = 1.0e6,
    ln_nk_gas_init: Optional[jnp.ndarray] = None,
    ln_ntot_gas_init: Optional[jnp.ndarray] = None,
    gas_epsilon_crit: float = 1.0e-10,
    gas_max_iter: int = 1000,
    budget_negative_tol: float = 1.0e-14,
    inactive_top_k: int = 5,
    default_initial_amount: float = 1.0e-30,
    least_squares_max_nfev: int = 100,
) -> dict[str, Any]:
    """LP-seeded augmented semismooth diagnostic with explicit inactive-violator residuals."""

    if inactive_policy not in ("naive_topk", "bottleneck_aware"):
        raise ValueError("inactive_policy must be one of ('naive_topk', 'bottleneck_aware').")
    dtype = jnp.result_type(jnp.asarray(formula_matrix_cond).dtype, state.element_vector.dtype, jnp.float64)
    if m0 is None:
        m0 = jnp.zeros((int(jnp.asarray(formula_matrix_cond).shape[1]),), dtype=dtype)
    initial_diag = diagnose_condensate_outer_active_set_layer(
        state,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        m=m0,
        condensate_species=condensate_species,
        element_names=element_names,
        ln_nk_gas_init=ln_nk_gas_init,
        ln_ntot_gas_init=ln_ntot_gas_init,
        gas_epsilon_crit=gas_epsilon_crit,
        gas_max_iter=gas_max_iter,
        budget_negative_tol=budget_negative_tol,
        joint_activation_top_k=(candidate_lp_top_k,),
    )
    if initial_diag["initial"]["status"] != "ok":
        return {
            "prototype_family": "augmented_candidate_smoothed_semismooth_diagnostic",
            "initial_diagnostic": initial_diag,
            "status": initial_diag["initial"]["status"],
            "candidate_indices": [],
            "inactive_indices": [],
        }

    initial_lp = initial_diag["initial"]["joint_activation"][str(candidate_lp_top_k)]
    candidate_indices = list(initial_lp["support_indices"])
    candidate_amounts = jnp.asarray(initial_lp["support_amounts"], dtype=dtype)
    baseline = solve_smoothed_semismooth_candidate_condensate_layer(
        state,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        candidate_indices=candidate_indices,
        candidate_amounts_init=candidate_amounts,
        condensate_species=condensate_species,
        element_names=element_names,
        mu_schedule=mu_schedule,
        ln_nk_gas_init=ln_nk_gas_init,
        ln_ntot_gas_init=ln_ntot_gas_init,
        gas_epsilon_crit=gas_epsilon_crit,
        gas_max_iter=gas_max_iter,
        budget_negative_tol=budget_negative_tol,
        feasibility_penalty_weight=budget_weight,
        inactive_top_k=inactive_top_k,
        default_initial_amount=default_initial_amount,
        least_squares_max_nfev=least_squares_max_nfev,
    )

    inactive_indices = _rank_inactive_candidates(
        driving_full=jnp.asarray(baseline["driving_full"]),
        current_support=candidate_indices,
        formula_matrix_cond=jnp.asarray(formula_matrix_cond),
        binding_element_indices=baseline["binding_element_indices"],
        add_threshold=0.0,
        prefer_binding_elements=(inactive_policy == "bottleneck_aware"),
    )[:inactive_violator_top_k]

    augmented = solve_augmented_semismooth_candidate_condensate_layer(
        state,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        candidate_indices=candidate_indices,
        inactive_indices=inactive_indices,
        candidate_amounts_init=jnp.asarray(baseline["m_candidate"], dtype=dtype),
        condensate_species=condensate_species,
        element_names=element_names,
        mu_schedule=mu_schedule,
        active_weight=active_weight,
        inactive_weight=inactive_weight,
        budget_weight=budget_weight,
        ln_nk_gas_init=ln_nk_gas_init,
        ln_ntot_gas_init=ln_ntot_gas_init,
        gas_epsilon_crit=gas_epsilon_crit,
        gas_max_iter=gas_max_iter,
        budget_negative_tol=budget_negative_tol,
        inactive_top_k=inactive_top_k,
        default_initial_amount=default_initial_amount,
        least_squares_max_nfev=least_squares_max_nfev,
    )
    augmented["inactive_policy"] = inactive_policy

    return {
        "prototype_family": "augmented_candidate_smoothed_semismooth_diagnostic",
        "initial_diagnostic": initial_diag,
        "initial_lp_top_k": int(candidate_lp_top_k),
        "initial_lp_support_indices": candidate_indices,
        "initial_lp_support_names": initial_lp["support_names"],
        "initial_lp_support_size": int(initial_lp["support_size"]),
        "mu_schedule": [float(value) for value in mu_schedule],
        "baseline_smoothed": baseline,
        "augmented": augmented,
    }


def evaluate_outer_objective_on_candidate_support(
    state: ThermoState,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    hvector_cond_func: Callable[[float], jnp.ndarray],
    *,
    candidate_indices: Sequence[int],
    m_candidate: jnp.ndarray,
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
    ln_nk_gas_init: Optional[jnp.ndarray] = None,
    ln_ntot_gas_init: Optional[jnp.ndarray] = None,
    gas_epsilon_crit: float = 1.0e-10,
    gas_max_iter: int = 1000,
    budget_negative_tol: float = 1.0e-14,
    objective_infeasible_penalty_weight: float = 1.0e12,
    inactive_top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate the diagnostic outer objective on a fixed condensate support.

    The code uses the sign convention already present in this module:
    ``driving = A_c^T pi_gas - c``. Therefore the outer objective gradient is
    ``grad Phi = c - A_c^T pi_gas = -driving`` on the candidate support.
    """

    formula_matrix = jnp.asarray(formula_matrix)
    formula_matrix_cond = jnp.asarray(formula_matrix_cond)
    candidate_indices = [int(i) for i in candidate_indices]
    n_cond = int(formula_matrix_cond.shape[1])
    dtype = jnp.result_type(formula_matrix_cond.dtype, state.element_vector.dtype, jnp.float64)
    m_candidate = jnp.asarray(m_candidate, dtype=dtype)
    if m_candidate.shape != (len(candidate_indices),):
        raise ValueError("m_candidate must have shape (len(candidate_indices),).")

    hvector_cond = jnp.asarray(hvector_cond_func(state.temperature), dtype=dtype)
    m_full = _assemble_support_amounts(n_cond, candidate_indices, m_candidate, dtype)
    b_eff = state.element_vector - formula_matrix_cond @ m_full
    negative_budget = jnp.maximum(-b_eff, 0.0)
    infeasible = bool(jnp.any(b_eff < -budget_negative_tol))
    support_mask = _support_mask(n_cond, candidate_indices)
    candidate_names = _restricted_support_names(candidate_indices, condensate_species)

    if infeasible:
        objective_penalty = float(
            objective_infeasible_penalty_weight * jnp.sum(negative_budget * negative_budget)
        )
        gradient = jnp.full((len(candidate_indices),), jnp.nan, dtype=dtype)
        driving_full = jnp.full((n_cond,), jnp.nan, dtype=dtype)
        inactive = {
            "max_positive_inactive_driving": float("nan"),
            "positive_inactive_count": 0,
            "top_indices": [],
            "top_names": [] if condensate_species is not None else None,
            "top_driving": [],
        }
        element_bottlenecks = {
            "tightest_budget_indices": [int(i) for i in jnp.argsort(b_eff)[: min(5, b_eff.shape[0])]],
            "tightest_budget_names": None
            if element_names is None
            else [str(element_names[i]) for i in jnp.argsort(b_eff)[: min(5, b_eff.shape[0])]],
            "tightest_budget_values": [
                float(b_eff[i]) for i in jnp.argsort(b_eff)[: min(5, b_eff.shape[0])]
            ],
        }
        return {
            "status": "infeasible_b_eff",
            "candidate_indices": candidate_indices,
            "candidate_names": candidate_names,
            "m_candidate": m_candidate,
            "m_full": m_full,
            "b_eff": b_eff,
            "negative_budget": negative_budget,
            "gas_result": None,
            "gas_objective": float("nan"),
            "condensate_linear_objective": float(jnp.dot(m_full, hvector_cond)),
            "outer_objective": float("inf"),
            "outer_objective_with_penalty": objective_penalty,
            "gradient": gradient,
            "gradient_full": jnp.full((n_cond,), jnp.nan, dtype=dtype),
            "driving_full": driving_full,
            "inactive": inactive,
            "element_bottlenecks": element_bottlenecks,
            "binding_element_indices": element_bottlenecks.get("tightest_budget_indices", []),
            "binding_element_names": element_bottlenecks.get("tightest_budget_names"),
            "binding_element_values": element_bottlenecks.get("tightest_budget_values", []),
            "active_support_mask": support_mask,
            "active_support_count": int(jnp.sum(m_candidate > 1.0e-24)),
            "active_support_gradient_norm": float("nan"),
            "support_gradient_inf": float("nan"),
        }

    gas_state = ThermoState(
        temperature=state.temperature,
        ln_normalized_pressure=state.ln_normalized_pressure,
        element_vector=b_eff,
    )
    gas_result = solve_gas_equilibrium_with_duals(
        gas_state,
        formula_matrix,
        hvector_func,
        ln_nk_init=ln_nk_gas_init,
        ln_ntot_init=ln_ntot_gas_init,
        epsilon_crit=gas_epsilon_crit,
        max_iter=gas_max_iter,
    )
    driving_full = formula_matrix_cond.T @ gas_result["pi_vector"] - hvector_cond
    support_gradient = -driving_full[jnp.asarray(candidate_indices, dtype=jnp.int32)] if candidate_indices else jnp.zeros((0,), dtype=dtype)
    gradient_full = -driving_full
    inactive = _top_inactive_violators(
        driving_full,
        support_mask,
        condensate_species,
        inactive_top_k,
    )
    positive_indices = [idx for idx in range(n_cond) if float(driving_full[idx]) > 0.0]
    element_bottlenecks = _element_bottleneck_summary(
        b_eff=b_eff,
        pi_vector=gas_result["pi_vector"],
        formula_matrix_cond=formula_matrix_cond,
        positive_indices=positive_indices,
        top_positive_indices=_top_indices(driving_full, descending=True, top_k=min(inactive_top_k, n_cond)),
        element_names=element_names,
        condensate_species=condensate_species,
    )
    gas_objective = float(
        _normalized_gibbs(
            gas_state,
            gas_result["ln_nk"],
            gas_result["ln_ntot"],
            gas_result["hvector"],
            jnp.zeros((n_cond,), dtype=dtype),
            jnp.zeros((n_cond,), dtype=dtype),
        )
    )
    cond_linear = float(jnp.dot(m_full, hvector_cond))
    active_mask = m_candidate > 1.0e-24
    active_support_gradient = jnp.where(active_mask, support_gradient, 0.0)
    return {
        "status": "ok",
        "candidate_indices": candidate_indices,
        "candidate_names": candidate_names,
        "m_candidate": m_candidate,
        "m_full": m_full,
        "b_eff": b_eff,
        "negative_budget": negative_budget,
        "gas_result": gas_result,
        "gas_objective": gas_objective,
        "condensate_linear_objective": cond_linear,
        "outer_objective": gas_objective + cond_linear,
        "outer_objective_with_penalty": gas_objective + cond_linear,
        "gradient": support_gradient,
        "gradient_full": gradient_full,
        "driving_full": driving_full,
        "inactive": inactive,
        "element_bottlenecks": element_bottlenecks,
        "binding_element_indices": element_bottlenecks.get("tightest_budget_indices", []),
        "binding_element_names": element_bottlenecks.get("tightest_budget_names"),
        "binding_element_values": element_bottlenecks.get("tightest_budget_values", []),
        "active_support_mask": support_mask,
        "active_support_count": int(jnp.sum(active_mask)),
        "active_support_gradient_norm": float(jnp.linalg.norm(active_support_gradient)),
        "support_gradient_inf": float(jnp.max(jnp.abs(support_gradient))) if candidate_indices else 0.0,
    }


def _compute_outer_optimizer_kkt(
    *,
    scipy_result: Any,
    gradient: jnp.ndarray,
    m_candidate: jnp.ndarray,
    candidate_matrix: jnp.ndarray,
    element_vector: jnp.ndarray,
    budget_negative_tol: float,
) -> dict[str, Any]:
    """Return constrained-KKT diagnostics for the restricted outer problem."""

    dtype = jnp.result_type(
        jnp.asarray(gradient).dtype,
        jnp.asarray(m_candidate).dtype,
        jnp.asarray(element_vector).dtype,
        jnp.float64,
    )
    gradient = jnp.asarray(gradient, dtype=dtype)
    m_candidate = jnp.asarray(m_candidate, dtype=dtype)
    candidate_matrix = jnp.asarray(candidate_matrix, dtype=dtype)
    element_vector = jnp.asarray(element_vector, dtype=dtype)

    budget_slack = element_vector - candidate_matrix @ m_candidate
    lower_bound_violation = jnp.maximum(-m_candidate, 0.0)
    budget_violation = jnp.maximum(-budget_slack, 0.0)
    feasibility_residual_inf = float(
        jnp.max(
            jnp.concatenate(
                [
                    lower_bound_violation,
                    budget_violation,
                ]
            )
        )
    ) if (m_candidate.size + budget_slack.size) > 0 else 0.0

    lagrangian_grad = getattr(scipy_result, "lagrangian_grad", None)
    stationarity_source = "optimizer_lagrangian_grad"
    if lagrangian_grad is None:
        # Fallback: a projected-gradient test that is exact for bound-active
        # coordinates and still useful when constraint multipliers are absent.
        projected = jnp.where(
            m_candidate > 0.0,
            gradient,
            jnp.minimum(gradient, 0.0),
        )
        lagrangian_grad = projected
        stationarity_source = "projected_gradient_fallback"
    lagrangian_grad = jnp.asarray(lagrangian_grad, dtype=dtype).reshape(-1)

    raw_multipliers = getattr(scipy_result, "v", None)
    budget_multiplier = None
    lower_bound_multiplier = None
    multiplier_source = None
    complementarity_residual_inf = None
    if raw_multipliers is not None and len(raw_multipliers) >= 2:
        budget_multiplier = jnp.maximum(
            jnp.asarray(raw_multipliers[0], dtype=dtype).reshape(-1),
            0.0,
        )
        # `trust-constr` returns bound multipliers with the sign convention used
        # in its internal bound representation; the lower-bound multipliers are
        # the positive part of `-v_bounds`.
        lower_bound_multiplier = jnp.maximum(
            -jnp.asarray(raw_multipliers[1], dtype=dtype).reshape(-1),
            0.0,
        )
        budget_complementarity = budget_multiplier * jnp.maximum(budget_slack, 0.0)
        lower_complementarity = lower_bound_multiplier * jnp.maximum(m_candidate, 0.0)
        complementarity_residual_inf = float(
            jnp.max(jnp.concatenate([budget_complementarity, lower_complementarity]))
        ) if (budget_complementarity.size + lower_complementarity.size) > 0 else 0.0
        multiplier_source = "optimizer_multipliers"

    return {
        "stationarity_source": stationarity_source,
        "multiplier_source": multiplier_source,
        "lagrangian_gradient": lagrangian_grad,
        "stationarity_residual_inf": float(jnp.max(jnp.abs(lagrangian_grad))) if lagrangian_grad.size else 0.0,
        "stationarity_residual_norm": float(jnp.linalg.norm(lagrangian_grad)) if lagrangian_grad.size else 0.0,
        "feasibility_residual_inf": feasibility_residual_inf,
        "lower_bound_violation_inf": float(jnp.max(lower_bound_violation)) if lower_bound_violation.size else 0.0,
        "budget_violation_inf": float(jnp.max(budget_violation)) if budget_violation.size else 0.0,
        "budget_slack": budget_slack,
        "budget_multiplier": budget_multiplier,
        "lower_bound_multiplier": lower_bound_multiplier,
        "complementarity_residual_inf": complementarity_residual_inf,
    }


def optimize_outer_objective_on_candidate_support(
    state: ThermoState,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    hvector_cond_func: Callable[[float], jnp.ndarray],
    *,
    candidate_indices: Sequence[int],
    candidate_amounts_init: Optional[jnp.ndarray] = None,
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
    optimizer_method: str = "SLSQP",
    ln_nk_gas_init: Optional[jnp.ndarray] = None,
    ln_ntot_gas_init: Optional[jnp.ndarray] = None,
    gas_epsilon_crit: float = 1.0e-10,
    gas_max_iter: int = 1000,
    budget_negative_tol: float = 1.0e-14,
    objective_infeasible_penalty_weight: float = 1.0e12,
    inactive_top_k: int = 5,
    default_initial_amount: float = 1.0e-30,
    optimizer_maxiter: int = 100,
    optimizer_ftol: float = 1.0e-10,
    active_m_threshold: float = 1.0e-24,
) -> dict[str, Any]:
    """Diagnostic-only direct outer objective optimization on a fixed support."""

    if _scipy_minimize is None:  # pragma: no cover - fallback only
        raise RuntimeError("scipy.optimize.minimize is required for outer-objective diagnostics.")

    formula_matrix_cond = jnp.asarray(formula_matrix_cond)
    candidate_indices = [int(i) for i in candidate_indices]
    dtype = jnp.result_type(formula_matrix_cond.dtype, state.element_vector.dtype, jnp.float64)
    if candidate_amounts_init is None:
        x0 = jnp.full((len(candidate_indices),), default_initial_amount, dtype=dtype)
    else:
        x0 = jnp.maximum(jnp.asarray(candidate_amounts_init, dtype=dtype), 0.0)
        if x0.shape != (len(candidate_indices),):
            raise ValueError("candidate_amounts_init must have shape (len(candidate_indices),).")

    candidate_matrix = formula_matrix_cond[:, jnp.asarray(candidate_indices, dtype=jnp.int32)] if candidate_indices else jnp.zeros((formula_matrix_cond.shape[0], 0), dtype=dtype)
    evaluations: dict[str, Any] = {"count": 0, "gas_solves": 0, "last": None}

    def _evaluate(x: jnp.ndarray) -> dict[str, Any]:
        evaluations["count"] += 1
        current = evaluate_outer_objective_on_candidate_support(
            state,
            formula_matrix,
            formula_matrix_cond,
            hvector_func,
            hvector_cond_func,
            candidate_indices=candidate_indices,
            m_candidate=x,
            condensate_species=condensate_species,
            element_names=element_names,
            ln_nk_gas_init=ln_nk_gas_init,
            ln_ntot_gas_init=ln_ntot_gas_init,
            gas_epsilon_crit=gas_epsilon_crit,
            gas_max_iter=gas_max_iter,
            budget_negative_tol=budget_negative_tol,
            objective_infeasible_penalty_weight=objective_infeasible_penalty_weight,
            inactive_top_k=inactive_top_k,
        )
        if current["gas_result"] is not None:
            evaluations["gas_solves"] += 1
        evaluations["last"] = current
        return current

    def _objective_numpy(x: np.ndarray) -> float:
        return float(_evaluate(jnp.asarray(x, dtype=dtype))["outer_objective_with_penalty"])

    def _gradient_numpy(x: np.ndarray) -> np.ndarray:
        current = _evaluate(jnp.asarray(x, dtype=dtype))
        gradient = jnp.asarray(current["gradient"], dtype=dtype)
        if current["status"] != "ok":
            gradient = jnp.where(jnp.isnan(gradient), 0.0, gradient)
        return np.asarray(gradient, dtype=np.float64)

    constraints = []
    if candidate_indices:
        def _ineq_fun(x: np.ndarray) -> np.ndarray:
            return np.asarray(state.element_vector - candidate_matrix @ jnp.asarray(x, dtype=dtype), dtype=np.float64)

        def _ineq_jac(x: np.ndarray) -> np.ndarray:
            del x
            return np.asarray(-candidate_matrix, dtype=np.float64)

        constraints.append({"type": "ineq", "fun": _ineq_fun, "jac": _ineq_jac})

    start = time.perf_counter()
    try:
        if optimizer_method == "trust-constr":
            trust_constraints = []
            if candidate_indices:
                if _scipy_Bounds is None or _scipy_LinearConstraint is None:  # pragma: no cover - fallback only
                    raise RuntimeError("trust-constr diagnostics require scipy Bounds and LinearConstraint.")
                trust_constraints.append(
                    _scipy_LinearConstraint(
                        np.asarray(candidate_matrix, dtype=np.float64),
                        -np.inf,
                        np.asarray(state.element_vector, dtype=np.float64),
                    )
                )
            scipy_result = _scipy_minimize(
                _objective_numpy,
                x0=np.asarray(x0, dtype=np.float64),
                jac=_gradient_numpy,
                bounds=None
                if _scipy_Bounds is None
                else _scipy_Bounds(
                    np.zeros((len(candidate_indices),), dtype=np.float64),
                    np.full((len(candidate_indices),), np.inf, dtype=np.float64),
                ),
                constraints=trust_constraints,
                method=optimizer_method,
                options={
                    "maxiter": int(optimizer_maxiter),
                    "gtol": float(optimizer_ftol),
                    "xtol": float(optimizer_ftol),
                    "barrier_tol": float(optimizer_ftol),
                },
            )
        else:
            scipy_result = _scipy_minimize(
                _objective_numpy,
                x0=np.asarray(x0, dtype=np.float64),
                jac=_gradient_numpy,
                bounds=[(0.0, None)] * len(candidate_indices),
                constraints=constraints,
                method=optimizer_method,
                options={"maxiter": int(optimizer_maxiter), "ftol": float(optimizer_ftol)},
            )
    except Exception as exc:  # pragma: no cover - exercised on real chemistry profiles
        scipy_result = SimpleNamespace(
            success=False,
            status=-1,
            message=f"{type(exc).__name__}: {exc}",
            x=np.asarray(x0, dtype=np.float64),
            fun=np.nan,
            optimality=np.nan,
            nfev=0,
            njev=0,
            lagrangian_grad=None,
            v=None,
        )
    elapsed = time.perf_counter() - start
    final_eval = _evaluate(jnp.asarray(scipy_result.x, dtype=dtype))
    feasible_projection_alpha = None
    raw_final_status = final_eval["status"]
    if final_eval["status"] == "infeasible_b_eff":
        feasible_projection_alpha = _max_feasible_scaling(final_eval["m_full"], formula_matrix_cond, state.element_vector)
        if feasible_projection_alpha > 0.0:
            projected_full = feasible_projection_alpha * final_eval["m_full"]
            final_eval = evaluate_outer_objective_on_candidate_support(
                state,
                formula_matrix,
                formula_matrix_cond,
                hvector_func,
                hvector_cond_func,
                candidate_indices=candidate_indices,
                m_candidate=projected_full[jnp.asarray(candidate_indices, dtype=jnp.int32)] if candidate_indices else jnp.zeros((0,), dtype=dtype),
                condensate_species=condensate_species,
                element_names=element_names,
                ln_nk_gas_init=ln_nk_gas_init,
                ln_ntot_gas_init=ln_ntot_gas_init,
                gas_epsilon_crit=gas_epsilon_crit,
                gas_max_iter=gas_max_iter,
                budget_negative_tol=budget_negative_tol,
                objective_infeasible_penalty_weight=objective_infeasible_penalty_weight,
                inactive_top_k=inactive_top_k,
            )

    active_mask = jnp.asarray(final_eval["m_candidate"]) > active_m_threshold
    active_gradient = jnp.where(active_mask, jnp.asarray(final_eval["gradient"]), 0.0)
    kkt = _compute_outer_optimizer_kkt(
        scipy_result=scipy_result,
        gradient=jnp.asarray(final_eval["gradient"], dtype=dtype),
        m_candidate=jnp.asarray(final_eval["m_candidate"], dtype=dtype),
        candidate_matrix=candidate_matrix,
        element_vector=jnp.asarray(state.element_vector, dtype=dtype),
        budget_negative_tol=budget_negative_tol,
    )
    return {
        "prototype_family": "restricted_support_outer_objective_diagnostic",
        "candidate_indices": candidate_indices,
        "candidate_names": _restricted_support_names(candidate_indices, condensate_species),
        "candidate_size": len(candidate_indices),
        "status": final_eval["status"],
        "raw_final_status": raw_final_status,
        "feasible_projection_alpha": feasible_projection_alpha,
        "solver_success": bool(scipy_result.success),
        "solver_status": int(scipy_result.status),
        "solver_message": str(scipy_result.message),
        "optimizer_method": optimizer_method,
        "runtime_seconds": float(elapsed),
        "evaluation_count": int(evaluations["count"]),
        "gas_solve_count": int(evaluations["gas_solves"]),
        "m_candidate": final_eval["m_candidate"],
        "m_full": final_eval["m_full"],
        "outer_objective": float(final_eval["outer_objective"]),
        "gas_objective": float(final_eval["gas_objective"]),
        "condensate_linear_objective": float(final_eval["condensate_linear_objective"]),
        "gradient": final_eval["gradient"],
        "gradient_inf_norm": float(jnp.max(jnp.abs(final_eval["gradient"]))) if candidate_indices else 0.0,
        "gradient_norm": float(jnp.linalg.norm(final_eval["gradient"])) if candidate_indices else 0.0,
        "active_support_gradient_norm": float(jnp.linalg.norm(active_gradient)) if candidate_indices else 0.0,
        "support_gradient_inf": float(jnp.max(jnp.abs(active_gradient))) if candidate_indices else 0.0,
        "active_support_count": int(jnp.sum(active_mask)),
        "true_stationarity_residual_inf": float(kkt["stationarity_residual_inf"]),
        "true_stationarity_residual_norm": float(kkt["stationarity_residual_norm"]),
        "stationarity_source": kkt["stationarity_source"],
        "multiplier_source": kkt["multiplier_source"],
        "lagrangian_gradient": kkt["lagrangian_gradient"],
        "feasibility_residual_inf": float(kkt["feasibility_residual_inf"]),
        "budget_violation_inf": float(kkt["budget_violation_inf"]),
        "lower_bound_violation_inf": float(kkt["lower_bound_violation_inf"]),
        "budget_slack": kkt["budget_slack"],
        "budget_multiplier": kkt["budget_multiplier"],
        "lower_bound_multiplier": kkt["lower_bound_multiplier"],
        "complementarity_residual_inf": kkt["complementarity_residual_inf"],
        "negative_budget_inf": float(jnp.max(final_eval["negative_budget"])) if final_eval["negative_budget"].size else 0.0,
        "b_eff": final_eval["b_eff"],
        "b_eff_feasible": bool(jnp.all(final_eval["b_eff"] >= -budget_negative_tol)),
        "driving_full": final_eval["driving_full"],
        "max_positive_inactive_driving": float(final_eval["inactive"]["max_positive_inactive_driving"]),
        "inactive_positive_count": int(final_eval["inactive"]["positive_inactive_count"]),
        "top_inactive_indices": final_eval["inactive"]["top_indices"],
        "top_inactive_names": final_eval["inactive"]["top_names"],
        "top_inactive_driving": final_eval["inactive"]["top_driving"],
        "element_bottlenecks": final_eval["element_bottlenecks"],
        "binding_element_indices": final_eval["binding_element_indices"],
        "binding_element_names": final_eval["binding_element_names"],
        "binding_element_values": final_eval["binding_element_values"],
        "gas_result": final_eval["gas_result"],
        "candidate_self_consistent": bool(
            final_eval["status"] == "ok"
            and float(kkt["stationarity_residual_inf"]) <= 1.0e-6
            and float(final_eval["inactive"]["max_positive_inactive_driving"]) <= 1.0e-6
            and float(kkt["feasibility_residual_inf"]) <= budget_negative_tol
        ),
    }


def diagnose_outer_objective_candidate_condensate_layer(
    state: ThermoState,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    hvector_cond_func: Callable[[float], jnp.ndarray],
    *,
    candidate_lp_top_k: int = 20,
    augment_inactive_violators: int = 0,
    augment_policy: str = "bottleneck_aware",
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
    m0: Optional[jnp.ndarray] = None,
    ln_nk_gas_init: Optional[jnp.ndarray] = None,
    ln_ntot_gas_init: Optional[jnp.ndarray] = None,
    gas_epsilon_crit: float = 1.0e-10,
    gas_max_iter: int = 1000,
    budget_negative_tol: float = 1.0e-14,
    objective_infeasible_penalty_weight: float = 1.0e12,
    inactive_top_k: int = 5,
    default_initial_amount: float = 1.0e-30,
    optimizer_method: str = "SLSQP",
    optimizer_maxiter: int = 100,
    optimizer_ftol: float = 1.0e-10,
) -> dict[str, Any]:
    """LP-seeded diagnostic outer objective optimization with optional support augmentation."""

    if augment_policy not in ("naive_topk", "bottleneck_aware"):
        raise ValueError("augment_policy must be one of ('naive_topk', 'bottleneck_aware').")
    dtype = jnp.result_type(jnp.asarray(formula_matrix_cond).dtype, state.element_vector.dtype, jnp.float64)
    if m0 is None:
        m0 = jnp.zeros((int(jnp.asarray(formula_matrix_cond).shape[1]),), dtype=dtype)

    initial_diag = diagnose_condensate_outer_active_set_layer(
        state,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        m=m0,
        condensate_species=condensate_species,
        element_names=element_names,
        ln_nk_gas_init=ln_nk_gas_init,
        ln_ntot_gas_init=ln_ntot_gas_init,
        gas_epsilon_crit=gas_epsilon_crit,
        gas_max_iter=gas_max_iter,
        budget_negative_tol=budget_negative_tol,
        joint_activation_top_k=(candidate_lp_top_k,),
    )
    if initial_diag["initial"]["status"] != "ok":
        return {
            "prototype_family": "restricted_support_outer_objective_diagnostic",
            "initial_diagnostic": initial_diag,
            "status": initial_diag["initial"]["status"],
            "candidate_indices": [],
        }

    initial_lp = initial_diag["initial"]["joint_activation"][str(candidate_lp_top_k)]
    candidate_indices = list(initial_lp["support_indices"])
    candidate_amounts = jnp.asarray(initial_lp["support_amounts"], dtype=dtype)
    baseline = optimize_outer_objective_on_candidate_support(
        state,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        candidate_indices=candidate_indices,
        candidate_amounts_init=candidate_amounts,
        condensate_species=condensate_species,
        element_names=element_names,
        optimizer_method=optimizer_method,
        ln_nk_gas_init=ln_nk_gas_init,
        ln_ntot_gas_init=ln_ntot_gas_init,
        gas_epsilon_crit=gas_epsilon_crit,
        gas_max_iter=gas_max_iter,
        budget_negative_tol=budget_negative_tol,
        objective_infeasible_penalty_weight=objective_infeasible_penalty_weight,
        inactive_top_k=inactive_top_k,
        default_initial_amount=default_initial_amount,
        optimizer_maxiter=optimizer_maxiter,
        optimizer_ftol=optimizer_ftol,
    )

    adjusted = None
    if baseline["status"] == "ok" and augment_inactive_violators > 0:
        additions = _rank_inactive_candidates(
            driving_full=jnp.asarray(baseline["driving_full"]),
            current_support=candidate_indices,
            formula_matrix_cond=jnp.asarray(formula_matrix_cond),
            binding_element_indices=baseline["binding_element_indices"],
            add_threshold=0.0,
            prefer_binding_elements=(augment_policy == "bottleneck_aware"),
        )[:augment_inactive_violators]
        updated_indices = sorted(set(candidate_indices).union(additions))
        if updated_indices != candidate_indices:
            adjusted = optimize_outer_objective_on_candidate_support(
                state,
                formula_matrix,
                formula_matrix_cond,
                hvector_func,
                hvector_cond_func,
                candidate_indices=updated_indices,
                candidate_amounts_init=_support_amounts_from_full(
                    updated_indices,
                    jnp.asarray(baseline["m_full"], dtype=dtype),
                    default_amount=default_initial_amount,
                    dtype=dtype,
                ),
                condensate_species=condensate_species,
                element_names=element_names,
                optimizer_method=optimizer_method,
                ln_nk_gas_init=ln_nk_gas_init,
                ln_ntot_gas_init=ln_ntot_gas_init,
                gas_epsilon_crit=gas_epsilon_crit,
                gas_max_iter=gas_max_iter,
                budget_negative_tol=budget_negative_tol,
                objective_infeasible_penalty_weight=objective_infeasible_penalty_weight,
                inactive_top_k=inactive_top_k,
                default_initial_amount=default_initial_amount,
                optimizer_maxiter=optimizer_maxiter,
                optimizer_ftol=optimizer_ftol,
            )
            adjusted["added_candidate_indices"] = additions
            adjusted["added_candidate_names"] = _restricted_support_names(additions, condensate_species)
            adjusted["augment_policy"] = augment_policy

    return {
        "prototype_family": "restricted_support_outer_objective_diagnostic",
        "initial_diagnostic": initial_diag,
        "initial_lp_top_k": int(candidate_lp_top_k),
        "initial_lp_support_indices": candidate_indices,
        "initial_lp_support_names": initial_lp["support_names"],
        "initial_lp_support_size": int(initial_lp["support_size"]),
        "optimizer_method": optimizer_method,
        "baseline_optimization": baseline,
        "adjusted_optimization": adjusted,
    }


def diagnose_dynamic_support_outer_objective_layer(
    state: ThermoState,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    hvector_cond_func: Callable[[float], jnp.ndarray],
    *,
    initial_lp_top_k: int = 20,
    initial_bottleneck_augment: int = 0,
    update_policy: Optional[str] = None,
    add_rule: str = "naive_topk",
    bottleneck_tiebreak: bool = False,
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
    m0: Optional[jnp.ndarray] = None,
    ln_nk_gas_init: Optional[jnp.ndarray] = None,
    ln_ntot_gas_init: Optional[jnp.ndarray] = None,
    gas_epsilon_crit: float = 1.0e-10,
    gas_max_iter: int = 1000,
    budget_negative_tol: float = 1.0e-14,
    objective_infeasible_penalty_weight: float = 1.0e12,
    inactive_top_k: int = 5,
    default_initial_amount: float = 1.0e-30,
    optimizer_method: str = "trust-constr",
    optimizer_maxiter: int = 100,
    optimizer_ftol: float = 1.0e-10,
    outer_max_iter: int = 10,
    active_m_threshold: float = 1.0e-24,
    add_threshold: float = 1.0e-3,
    max_additions_per_iter: int = 4,
    reduced_cost_threshold: float = 1.0,
    reduced_cost_fraction: float = 2.0e-1,
    require_settled_before_add: bool = True,
    settle_inactive_ratio: float = 5.0e-2,
    settle_stationarity_abs_tol: float = 1.0e-2,
    settle_feasibility_tol: float = 2.5e-1,
    settle_complementarity_abs_tol: float = 1.0e-2,
    proposal_batch_sizes: Sequence[int] = (1, 2, 3),
    max_drop_proposals: int = 3,
    merit_stationarity_weight: float = 1.0,
    merit_feasibility_weight: float = 1.0e3,
    merit_complementarity_weight: float = 1.0,
    merit_inactive_weight: float = 1.0,
    merit_inactive_count_weight: float = 1.0e-2,
    support_size_penalty: float = 0.0,
    addition_penalty: float = 0.0,
    merit_improve_abs_tol: float = 1.0e-3,
    merit_improve_rel_tol: float = 1.0e-3,
    stabilization_inactive_tol: float = 1.0,
    support_size_cap: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    drop_m_threshold: float = 1.0e-18,
    drop_driving_threshold: float = -1.0e-1,
    drop_m_hysteresis_factor: float = 1.0,
    drop_driving_hysteresis: float = 0.0,
) -> dict[str, Any]:
    """Diagnostic-only merit-controlled dynamic-support outer objective loop.

    The support-update acceptance uses the combined merit

    ``M = w_stat * stat + w_feas * feas + w_comp * comp + w_inactive * inactive_max
    + w_count * inactive_positive_count + lambda_size * |C|
    + lambda_add * (# newly added species)``

    where ``comp`` is zero when multiplier-based complementarity is unavailable.
    Candidate support changes are accepted only if they reduce this merit by at
    least ``max(merit_improve_abs_tol, merit_improve_rel_tol * current_merit)``.
    """

    if update_policy is not None:
        if update_policy == "naive_topk":
            add_rule = "naive_topk"
            bottleneck_tiebreak = False
        elif update_policy == "bottleneck_aware":
            add_rule = "naive_topk"
            bottleneck_tiebreak = True
        elif update_policy == "reduced_cost_threshold":
            add_rule = "reduced_cost_threshold"
            bottleneck_tiebreak = False
        elif update_policy == "reduced_cost_threshold_bottleneck":
            add_rule = "reduced_cost_threshold"
            bottleneck_tiebreak = True
        else:
            raise ValueError(
                "update_policy must be one of "
                "('naive_topk', 'bottleneck_aware', 'reduced_cost_threshold', "
                "'reduced_cost_threshold_bottleneck')."
            )
    if add_rule not in ("naive_topk", "reduced_cost_threshold"):
        raise ValueError("add_rule must be one of ('naive_topk', 'reduced_cost_threshold').")
    if support_size_cap is not None and int(support_size_cap) < 0:
        raise ValueError("support_size_cap must be non-negative when provided.")
    if max_batch_size is not None and int(max_batch_size) < 0:
        raise ValueError("max_batch_size must be non-negative when provided.")
    if float(drop_m_hysteresis_factor) < 0.0:
        raise ValueError("drop_m_hysteresis_factor must be non-negative.")
    if float(drop_driving_hysteresis) < 0.0:
        raise ValueError("drop_driving_hysteresis must be non-negative.")

    formula_matrix = jnp.asarray(formula_matrix)
    formula_matrix_cond = jnp.asarray(formula_matrix_cond)
    n_cond = int(formula_matrix_cond.shape[1])
    dtype = jnp.result_type(formula_matrix_cond.dtype, state.element_vector.dtype, jnp.float64)
    if m0 is None:
        m0 = jnp.zeros((n_cond,), dtype=dtype)
    else:
        m0 = jnp.asarray(m0, dtype=dtype)

    initial_diag = diagnose_condensate_outer_active_set_layer(
        state,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        m=m0,
        condensate_species=condensate_species,
        element_names=element_names,
        ln_nk_gas_init=ln_nk_gas_init,
        ln_ntot_gas_init=ln_ntot_gas_init,
        gas_epsilon_crit=gas_epsilon_crit,
        gas_max_iter=gas_max_iter,
        budget_negative_tol=budget_negative_tol,
        active_m_threshold=active_m_threshold,
        joint_activation_top_k=(initial_lp_top_k,),
    )
    if initial_diag["initial"]["status"] != "ok":
        return {
            "prototype_family": "dynamic_support_outer_objective_diagnostic",
            "initial_diagnostic": initial_diag,
            "status": initial_diag["initial"]["status"],
            "history": [],
        }

    initial_record = initial_diag["initial"]
    initial_lp = initial_record["joint_activation"][str(initial_lp_top_k)]
    current_support = list(initial_lp["support_indices"])
    current_amounts = jnp.asarray(initial_lp["support_amounts"], dtype=dtype)
    initial_seed_additions: list[int] = []
    if initial_bottleneck_augment > 0:
        initial_seed_additions = _rank_inactive_candidates(
            driving_full=jnp.asarray(initial_record["driving"], dtype=dtype),
            current_support=current_support,
            formula_matrix_cond=formula_matrix_cond,
            binding_element_indices=initial_record["element_bottlenecks"].get("tightest_budget_indices", []),
            add_threshold=0.0,
            prefer_binding_elements=True,
        )[:initial_bottleneck_augment]
        if initial_seed_additions:
            current_support = sorted(set(current_support).union(initial_seed_additions))
            current_amounts = _support_amounts_from_full(
                current_support,
                jnp.zeros((n_cond,), dtype=dtype),
                default_amount=default_initial_amount,
                dtype=dtype,
            )

    history: list[dict[str, Any]] = []
    support_stabilized = False
    start = time.perf_counter()
    final_result: Optional[dict[str, Any]] = None
    repeated_no_change = 0

    for outer_iter in range(int(outer_max_iter)):
        support_before = list(current_support)
        support_before_names = _restricted_support_names(support_before, condensate_species)
        solve = optimize_outer_objective_on_candidate_support(
            state,
            formula_matrix,
            formula_matrix_cond,
            hvector_func,
            hvector_cond_func,
            candidate_indices=support_before,
            candidate_amounts_init=current_amounts,
            condensate_species=condensate_species,
            element_names=element_names,
            optimizer_method=optimizer_method,
            ln_nk_gas_init=ln_nk_gas_init,
            ln_ntot_gas_init=ln_ntot_gas_init,
            gas_epsilon_crit=gas_epsilon_crit,
            gas_max_iter=gas_max_iter,
            budget_negative_tol=budget_negative_tol,
            objective_infeasible_penalty_weight=objective_infeasible_penalty_weight,
            inactive_top_k=inactive_top_k,
            default_initial_amount=default_initial_amount,
            optimizer_maxiter=optimizer_maxiter,
            optimizer_ftol=optimizer_ftol,
            active_m_threshold=active_m_threshold,
        )
        final_result = solve
        current_merit = _compute_dynamic_support_merit(
            solve=solve,
            stationarity_weight=merit_stationarity_weight,
            feasibility_weight=merit_feasibility_weight,
            complementarity_weight=merit_complementarity_weight,
            inactive_weight=merit_inactive_weight,
            inactive_count_weight=merit_inactive_count_weight,
            support_size=len(support_before),
            support_size_penalty=support_size_penalty,
            added_count=0,
            addition_penalty=addition_penalty,
        )
        required_merit_improvement = max(
            float(merit_improve_abs_tol),
            float(merit_improve_rel_tol) * max(float(current_merit["value"]), 1.0),
        )

        naive_additions = _rank_inactive_candidates_with_tiebreak(
            driving_full=jnp.asarray(solve["driving_full"], dtype=dtype),
            current_support=support_before,
            formula_matrix_cond=formula_matrix_cond,
            binding_element_indices=[],
            add_threshold=add_threshold,
            bottleneck_tiebreak=False,
        )[: max(proposal_batch_sizes) if proposal_batch_sizes else 0]
        bottleneck_additions = _rank_inactive_candidates_with_tiebreak(
            driving_full=jnp.asarray(solve["driving_full"], dtype=dtype),
            current_support=support_before,
            formula_matrix_cond=formula_matrix_cond,
            binding_element_indices=solve["binding_element_indices"],
            add_threshold=add_threshold,
            bottleneck_tiebreak=True,
        )[: max(proposal_batch_sizes) if proposal_batch_sizes else 0]
        add_decision = _select_working_set_additions(
            solve=solve,
            current_support=support_before,
            formula_matrix_cond=formula_matrix_cond,
            add_rule=add_rule,
            add_threshold=add_threshold,
            max_additions_per_iter=max_additions_per_iter,
            reduced_cost_threshold=reduced_cost_threshold,
            reduced_cost_fraction=reduced_cost_fraction,
            binding_element_indices=solve["binding_element_indices"],
            bottleneck_tiebreak=bottleneck_tiebreak,
            require_settled=require_settled_before_add,
            settle_inactive_ratio=settle_inactive_ratio,
            settle_stationarity_abs_tol=settle_stationarity_abs_tol,
            settle_feasibility_tol=settle_feasibility_tol,
            settle_complementarity_abs_tol=settle_complementarity_abs_tol,
        )
        candidate_proposals: list[dict[str, Any]] = []
        accepted_proposal: Optional[dict[str, Any]] = None
        accepted_result: Optional[dict[str, Any]] = None

        if (not require_settled_before_add) or add_decision["settled_for_addition"]:
            addition_proposals = _build_addition_proposals(
                solve=solve,
                current_support=support_before,
                formula_matrix_cond=formula_matrix_cond,
                add_threshold=add_threshold,
                reduced_cost_threshold=reduced_cost_threshold,
                reduced_cost_fraction=reduced_cost_fraction,
                binding_element_indices=solve["binding_element_indices"],
                bottleneck_tiebreak=bottleneck_tiebreak,
                proposal_batch_sizes=proposal_batch_sizes,
                current_support_size=len(support_before),
                support_size_cap=support_size_cap,
                max_batch_size=max_batch_size,
            )
        else:
            addition_proposals = []
        eligible_drop_indices, drop_proposals = _build_drop_proposals(
            solve=solve,
            current_support=support_before,
            drop_m_threshold=drop_m_threshold,
            drop_driving_threshold=drop_driving_threshold,
            drop_m_hysteresis_factor=drop_m_hysteresis_factor,
            drop_driving_hysteresis=drop_driving_hysteresis,
            max_drop_proposals=max_drop_proposals,
        )

        warm_start_full = jnp.asarray(solve["m_full"], dtype=dtype)
        for proposal in addition_proposals:
            proposal_indices = list(proposal["indices"])
            if not proposal_indices:
                continue
            proposed_support = sorted(set(support_before).union(proposal_indices))
            if support_size_cap is not None and len(proposed_support) > int(support_size_cap):
                continue
            proposal_result = optimize_outer_objective_on_candidate_support(
                state,
                formula_matrix,
                formula_matrix_cond,
                hvector_func,
                hvector_cond_func,
                candidate_indices=proposed_support,
                candidate_amounts_init=_support_amounts_from_full(
                    proposed_support,
                    warm_start_full,
                    default_amount=default_initial_amount,
                    dtype=dtype,
                ),
                condensate_species=condensate_species,
                element_names=element_names,
                optimizer_method=optimizer_method,
                ln_nk_gas_init=ln_nk_gas_init,
                ln_ntot_gas_init=ln_ntot_gas_init,
                gas_epsilon_crit=gas_epsilon_crit,
                gas_max_iter=gas_max_iter,
                budget_negative_tol=budget_negative_tol,
                objective_infeasible_penalty_weight=objective_infeasible_penalty_weight,
                inactive_top_k=inactive_top_k,
                default_initial_amount=default_initial_amount,
                optimizer_maxiter=optimizer_maxiter,
                optimizer_ftol=optimizer_ftol,
                active_m_threshold=active_m_threshold,
            )
            proposal_merit = _compute_dynamic_support_merit(
                solve=proposal_result,
                stationarity_weight=merit_stationarity_weight,
                feasibility_weight=merit_feasibility_weight,
                complementarity_weight=merit_complementarity_weight,
                inactive_weight=merit_inactive_weight,
                inactive_count_weight=merit_inactive_count_weight,
                support_size=len(proposed_support),
                support_size_penalty=support_size_penalty,
                added_count=len(proposal_indices),
                addition_penalty=addition_penalty,
            )
            merit_delta = float(current_merit["value"] - proposal_merit["value"])
            accepted = merit_delta >= required_merit_improvement
            record = {
                "proposal_type": "addition",
                "proposal_kind": proposal["proposal_kind"],
                "selection_threshold": proposal.get("selection_threshold"),
                "indices": proposal_indices,
                "names": _restricted_support_names(proposal_indices, condensate_species),
                "support_indices": proposed_support,
                "support_names": _restricted_support_names(proposed_support, condensate_species),
                "merit": proposal_merit,
                "merit_delta": merit_delta,
                "accepted": bool(accepted),
                "solve": proposal_result,
            }
            candidate_proposals.append(record)
            if accepted and (
                accepted_proposal is None
                or float(record["merit"]["value"]) < float(accepted_proposal["merit"]["value"])
            ):
                accepted_proposal = record
                accepted_result = proposal_result

        for proposal in drop_proposals:
            proposal_indices = list(proposal["indices"])
            if not proposal_indices:
                continue
            proposed_support = [idx for idx in support_before if idx not in set(proposal_indices)]
            proposal_result = optimize_outer_objective_on_candidate_support(
                state,
                formula_matrix,
                formula_matrix_cond,
                hvector_func,
                hvector_cond_func,
                candidate_indices=proposed_support,
                candidate_amounts_init=_support_amounts_from_full(
                    proposed_support,
                    warm_start_full,
                    default_amount=default_initial_amount,
                    dtype=dtype,
                ),
                condensate_species=condensate_species,
                element_names=element_names,
                optimizer_method=optimizer_method,
                ln_nk_gas_init=ln_nk_gas_init,
                ln_ntot_gas_init=ln_ntot_gas_init,
                gas_epsilon_crit=gas_epsilon_crit,
                gas_max_iter=gas_max_iter,
                budget_negative_tol=budget_negative_tol,
                objective_infeasible_penalty_weight=objective_infeasible_penalty_weight,
                inactive_top_k=inactive_top_k,
                default_initial_amount=default_initial_amount,
                optimizer_maxiter=optimizer_maxiter,
                optimizer_ftol=optimizer_ftol,
                active_m_threshold=active_m_threshold,
            )
            proposal_merit = _compute_dynamic_support_merit(
                solve=proposal_result,
                stationarity_weight=merit_stationarity_weight,
                feasibility_weight=merit_feasibility_weight,
                complementarity_weight=merit_complementarity_weight,
                inactive_weight=merit_inactive_weight,
                inactive_count_weight=merit_inactive_count_weight,
                support_size=len(proposed_support),
                support_size_penalty=support_size_penalty,
                added_count=0,
                addition_penalty=addition_penalty,
            )
            merit_delta = float(current_merit["value"] - proposal_merit["value"])
            accepted = merit_delta >= 0.0
            record = {
                "proposal_type": "drop",
                "proposal_kind": proposal["proposal_kind"],
                "selection_threshold": None,
                "indices": proposal_indices,
                "names": _restricted_support_names(proposal_indices, condensate_species),
                "support_indices": proposed_support,
                "support_names": _restricted_support_names(proposed_support, condensate_species),
                "merit": proposal_merit,
                "merit_delta": merit_delta,
                "accepted": bool(accepted),
                "solve": proposal_result,
            }
            candidate_proposals.append(record)
            if accepted and (
                accepted_proposal is None
                or float(record["merit"]["value"]) < float(accepted_proposal["merit"]["value"])
            ):
                accepted_proposal = record
                accepted_result = proposal_result

        accepted_add_indices: list[int] = []
        accepted_drop_indices: list[int] = []
        next_support = support_before
        if accepted_proposal is not None and accepted_result is not None:
            next_support = list(accepted_proposal["support_indices"])
            final_result = accepted_result
            if accepted_proposal["proposal_type"] == "addition":
                accepted_add_indices = list(accepted_proposal["indices"])
            else:
                accepted_drop_indices = list(accepted_proposal["indices"])

        no_changes = next_support == support_before
        repeated_no_change = repeated_no_change + 1 if no_changes else 0
        reference_result = solve if accepted_result is None else accepted_result
        inactive_clear = float(reference_result["max_positive_inactive_driving"]) <= max(
            add_threshold,
            reduced_cost_threshold,
            stabilization_inactive_tol,
        )
        support_stabilized = bool(
            no_changes
            and (
                inactive_clear
                or repeated_no_change >= 2
                or (accepted_proposal is None and not add_decision["settled_for_addition"] and not eligible_drop_indices)
            )
        )
        solver_like_stabilized = _solver_like_support_stabilized(
            solve=reference_result,
            no_changes=no_changes,
            add_threshold=add_threshold,
            reduced_cost_threshold=reduced_cost_threshold,
            stabilization_inactive_tol=stabilization_inactive_tol,
            settle_stationarity_abs_tol=settle_stationarity_abs_tol,
            settle_feasibility_tol=settle_feasibility_tol,
            settle_complementarity_abs_tol=settle_complementarity_abs_tol,
        )
        history.append(
            {
                "outer_iter": outer_iter,
                "support_before_indices": support_before,
                "support_before_names": support_before_names,
                "support_size_before": len(support_before),
                "solve": solve,
                "add_rule": add_rule,
                "bottleneck_tiebreak": bool(bottleneck_tiebreak),
                "settled_for_addition": add_decision["settled_for_addition"],
                "addition_gate_stationarity": add_decision["stationarity"],
                "addition_gate_stationarity_limit": add_decision["stationarity_limit"],
                "addition_gate_feasibility": add_decision["feasibility"],
                "addition_gate_complementarity": add_decision["complementarity"],
                "addition_gate_complementarity_limit": add_decision["complementarity_limit"],
                "add_selection_threshold": add_decision["selection_threshold"],
                "eligible_add_indices": add_decision["eligible_ranked_indices"],
                "eligible_add_names": _restricted_support_names(add_decision["eligible_ranked_indices"][: min(10, len(add_decision["eligible_ranked_indices"]))], condensate_species),
                "eligible_add_driving": add_decision["eligible_ranked_driving"],
                "naive_add_indices": naive_additions,
                "naive_add_names": _restricted_support_names(naive_additions, condensate_species),
                "bottleneck_add_indices": bottleneck_additions,
                "bottleneck_add_names": _restricted_support_names(bottleneck_additions, condensate_species),
                "current_merit": current_merit,
                "required_merit_improvement": float(required_merit_improvement),
                "eligible_drop_indices": eligible_drop_indices,
                "eligible_drop_names": _restricted_support_names(eligible_drop_indices, condensate_species),
                "proposal_count": len(candidate_proposals),
                "proposals": candidate_proposals,
                "accepted_proposal": accepted_proposal,
                "rejected_proposals": [proposal for proposal in candidate_proposals if not proposal["accepted"]],
                "add_indices": accepted_add_indices,
                "add_names": _restricted_support_names(accepted_add_indices, condensate_species),
                "drop_indices": accepted_drop_indices,
                "drop_names": _restricted_support_names(accepted_drop_indices, condensate_species),
                "support_after_indices": next_support,
                "support_after_names": _restricted_support_names(next_support, condensate_species),
                "support_size_after": len(next_support),
                "support_stabilized": support_stabilized,
                "solver_like_stabilized": solver_like_stabilized,
                "no_support_changes": no_changes,
            }
        )

        if solver_like_stabilized or support_stabilized:
            current_support = next_support
            break

        current_support = next_support
        current_amounts = _support_amounts_from_full(
            current_support,
            jnp.asarray(reference_result["m_full"], dtype=dtype),
            default_amount=default_initial_amount,
            dtype=dtype,
        )

    if final_result is not None and list(final_result["candidate_indices"]) != list(current_support):
        final_result = optimize_outer_objective_on_candidate_support(
            state,
            formula_matrix,
            formula_matrix_cond,
            hvector_func,
            hvector_cond_func,
            candidate_indices=current_support,
            candidate_amounts_init=current_amounts,
            condensate_species=condensate_species,
            element_names=element_names,
            optimizer_method=optimizer_method,
            ln_nk_gas_init=ln_nk_gas_init,
            ln_ntot_gas_init=ln_ntot_gas_init,
            gas_epsilon_crit=gas_epsilon_crit,
            gas_max_iter=gas_max_iter,
            budget_negative_tol=budget_negative_tol,
            objective_infeasible_penalty_weight=objective_infeasible_penalty_weight,
            inactive_top_k=inactive_top_k,
            default_initial_amount=default_initial_amount,
            optimizer_maxiter=optimizer_maxiter,
            optimizer_ftol=optimizer_ftol,
            active_m_threshold=active_m_threshold,
        )

    elapsed = time.perf_counter() - start
    final_merit = None
    if final_result is not None:
        final_merit = _compute_dynamic_support_merit(
            solve=final_result,
            stationarity_weight=merit_stationarity_weight,
            feasibility_weight=merit_feasibility_weight,
            complementarity_weight=merit_complementarity_weight,
            inactive_weight=merit_inactive_weight,
            inactive_count_weight=merit_inactive_count_weight,
            support_size=len(current_support),
            support_size_penalty=support_size_penalty,
            added_count=0,
            addition_penalty=addition_penalty,
        )
    final_solver_like_stabilized = bool(history[-1]["solver_like_stabilized"]) if history else False
    return {
        "prototype_family": "dynamic_support_outer_objective_diagnostic",
        "initial_diagnostic": initial_diag,
        "initial_lp_top_k": int(initial_lp_top_k),
        "initial_lp_support_indices": list(initial_lp["support_indices"]),
        "initial_lp_support_names": initial_lp["support_names"],
        "initial_lp_support_size": int(initial_lp["support_size"]),
        "initial_seed_additions": initial_seed_additions,
        "initial_seed_addition_names": _restricted_support_names(initial_seed_additions, condensate_species),
        "update_policy": "bottleneck_aware" if (add_rule == "naive_topk" and bottleneck_tiebreak) else add_rule,
        "add_rule": add_rule,
        "bottleneck_tiebreak": bool(bottleneck_tiebreak),
        "reduced_cost_threshold": float(reduced_cost_threshold),
        "reduced_cost_fraction": float(reduced_cost_fraction),
        "require_settled_before_add": bool(require_settled_before_add),
        "optimizer_method": optimizer_method,
        "proposal_batch_sizes": [int(size) for size in proposal_batch_sizes],
        "max_drop_proposals": int(max_drop_proposals),
        "support_size_cap": None if support_size_cap is None else int(support_size_cap),
        "max_batch_size": None if max_batch_size is None else int(max_batch_size),
        "merit_definition": {
            "formula": "w_stat*stationarity + w_feas*feasibility + w_comp*complementarity + w_inactive*inactive_max + w_count*inactive_positive_count + lambda_size*support_size + lambda_add*added_count",
            "weights": {
                "stationarity": float(merit_stationarity_weight),
                "feasibility": float(merit_feasibility_weight),
                "complementarity": float(merit_complementarity_weight),
                "inactive_max": float(merit_inactive_weight),
                "inactive_positive_count": float(merit_inactive_count_weight),
                "support_size": float(support_size_penalty),
                "added_count": float(addition_penalty),
            },
            "improvement_rule": {
                "absolute_tol": float(merit_improve_abs_tol),
                "relative_tol": float(merit_improve_rel_tol),
            },
        },
        "history": history,
        "final": final_result,
        "final_merit": final_merit,
        "final_support_indices": [] if not history else history[-1]["support_after_indices"],
        "final_support_names": None if not history else history[-1]["support_after_names"],
        "final_support_size": 0 if not history else int(history[-1]["support_size_after"]),
        "support_stabilized": bool(support_stabilized),
        "solver_like_stabilized": final_solver_like_stabilized,
        "outer_iterations_completed": len(history),
        "runtime_seconds": float(elapsed),
    }


def _summarize_layer(
    *,
    state: ThermoState,
    m: jnp.ndarray,
    gas_result: dict[str, Any],
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    condensate_species: Optional[Sequence[str]],
    element_names: Optional[Sequence[str]],
    top_k: int,
    active_m_threshold: float,
    competitive_relative_tol: float,
    competitive_absolute_tol: float,
    family_similarity_threshold: float,
    driving_tie_relative_tol: float,
    driving_tie_absolute_tol: float,
    joint_activation_top_k: Sequence[int],
    joint_activation_support_tol: float,
) -> dict[str, Any]:
    driving = formula_matrix_cond.T @ gas_result["pi_vector"] - hvector_cond
    b_eff = state.element_vector - formula_matrix_cond @ m
    balance = formula_matrix @ gas_result["nk"] + formula_matrix_cond @ m - state.element_vector
    active_mask = m > active_m_threshold
    inactive_mask = ~active_mask
    max_positive = jnp.maximum(jnp.max(driving), 0.0)
    competitive_threshold = jnp.maximum(
        jnp.asarray(competitive_absolute_tol, dtype=driving.dtype),
        jnp.asarray(competitive_relative_tol, dtype=driving.dtype) * max_positive,
    )
    competitive_mask = driving >= competitive_threshold
    positive_mask = driving > 0.0
    positive_indices = [int(i) for i in jnp.where(positive_mask)[0]]

    positive_indices_ranked = _top_indices(driving, descending=True, top_k=int(driving.shape[0]))
    positive_indices_ranked = [idx for idx in positive_indices_ranked if float(driving[idx]) > 0.0]
    positive_indices_ranked = positive_indices_ranked[: max(top_k, max(joint_activation_top_k, default=0))]

    positive_indices_top = positive_indices_ranked[: min(top_k, len(positive_indices_ranked))]
    negative_indices = _top_indices(driving, descending=False, top_k=top_k)

    def _names(indices: list[int]) -> Optional[list[str]]:
        if condensate_species is None:
            return None
        return [str(condensate_species[i]) for i in indices]

    active_gap = jnp.where(
        jnp.any(active_mask),
        jnp.max(jnp.abs(jnp.where(active_mask, driving, 0.0))),
        jnp.asarray(0.0, dtype=driving.dtype),
    )
    inactive_positive_gap = jnp.where(
        jnp.any(inactive_mask),
        jnp.max(jnp.maximum(jnp.where(inactive_mask, driving, -jnp.inf), 0.0)),
        jnp.asarray(0.0, dtype=driving.dtype),
    )

    clusters = _cluster_positive_condensates(
        formula_matrix_cond=formula_matrix_cond,
        driving=driving,
        positive_indices=positive_indices,
        condensate_species=condensate_species,
        element_names=element_names,
        similarity_threshold=family_similarity_threshold,
        driving_tie_relative_tol=driving_tie_relative_tol,
        driving_tie_absolute_tol=driving_tie_absolute_tol,
        cluster_leader_top_k=3,
    )
    bottlenecks = _element_bottleneck_summary(
        b_eff=b_eff,
        pi_vector=gas_result["pi_vector"],
        formula_matrix_cond=formula_matrix_cond,
        positive_indices=positive_indices,
        top_positive_indices=positive_indices_top,
        element_names=element_names,
        condensate_species=condensate_species,
    )
    joint_activation = {
        str(limit): _joint_activation_lp(
            formula_matrix_cond=formula_matrix_cond,
            driving=driving,
            b_eff=b_eff,
            candidate_indices=positive_indices_ranked[: min(limit, len(positive_indices_ranked))],
            condensate_species=condensate_species,
            clusters=clusters,
            element_names=element_names,
            support_tol=joint_activation_support_tol,
        )
        for limit in joint_activation_top_k
        if limit > 0
    }

    return {
        "m": m,
        "b_eff": b_eff,
        "gas_result": gas_result,
        "driving": driving,
        "hvector_cond": hvector_cond,
        "active_mask": active_mask,
        "competitive_mask": competitive_mask,
        "competitive_threshold": competitive_threshold,
        "element_balance": balance,
        "element_balance_inf_norm": jnp.max(jnp.abs(balance)),
        "normalized_gibbs": _normalized_gibbs(
            state,
            gas_result["ln_nk"],
            gas_result["ln_ntot"],
            gas_result["hvector"],
            m,
            hvector_cond,
        ),
        "kkt_merit": {
            "active_cond_gap_inf": active_gap,
            "inactive_positive_driving_inf": inactive_positive_gap,
            "combined_inf": jnp.maximum(active_gap, inactive_positive_gap),
        },
        "active_count": int(jnp.sum(active_mask)),
        "competitive_count": int(jnp.sum(competitive_mask)),
        "positive_count": int(jnp.sum(positive_mask)),
        "top_positive_indices": positive_indices_top,
        "top_positive_names": _names(positive_indices_top),
        "top_positive_driving": [float(driving[i]) for i in positive_indices_top],
        "top_negative_indices": negative_indices,
        "top_negative_names": _names(negative_indices),
        "top_negative_driving": [float(driving[i]) for i in negative_indices],
        "small_competitive_set": bool(int(jnp.sum(competitive_mask)) <= top_k),
        "family_diagnostics": clusters,
        "element_bottlenecks": bottlenecks,
        "joint_activation": joint_activation,
    }


def diagnose_condensate_outer_active_set_layer(
    state: ThermoState,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func: Callable[[float], jnp.ndarray],
    hvector_cond_func: Callable[[float], jnp.ndarray],
    *,
    m: Optional[jnp.ndarray] = None,
    ln_mk: Optional[jnp.ndarray] = None,
    condensate_species: Optional[Sequence[str]] = None,
    element_names: Optional[Sequence[str]] = None,
    ln_nk_gas_init: Optional[jnp.ndarray] = None,
    ln_ntot_gas_init: Optional[jnp.ndarray] = None,
    gas_epsilon_crit: float = 1.0e-10,
    gas_max_iter: int = 1000,
    budget_negative_tol: float = 1.0e-14,
    active_m_threshold: float = 1.0e-24,
    competitive_relative_tol: float = 1.0e-2,
    competitive_absolute_tol: float = 1.0e-12,
    top_k: int = 5,
    projected_outer_iterations: int = 0,
    projected_activate_top_k: int = 3,
    projected_step_fraction: float = 0.05,
    projected_shrink_fraction: float = 0.5,
    family_similarity_threshold: float = 0.995,
    driving_tie_relative_tol: float = 5.0e-2,
    driving_tie_absolute_tol: float = 1.0e-6,
    joint_activation_top_k: Sequence[int] = (20, 40),
    joint_activation_support_tol: float = 1.0e-12,
) -> dict[str, Any]:
    """Run one diagnostic outer-layer evaluation with optional projected updates."""

    formula_matrix = jnp.asarray(formula_matrix)
    formula_matrix_cond = jnp.asarray(formula_matrix_cond)
    m0 = _amounts_from_optional_inputs(
        formula_matrix_cond,
        m=m,
        ln_mk=ln_mk,
    )

    def _evaluate(m_current: jnp.ndarray) -> dict[str, Any]:
        b_eff = state.element_vector - formula_matrix_cond @ m_current
        infeasible_mask = b_eff < -budget_negative_tol
        if bool(jnp.any(infeasible_mask)):
            alpha = _max_feasible_scaling(m_current, formula_matrix_cond, state.element_vector)
            return {
                "status": "infeasible_b_eff",
                "m": m_current,
                "b_eff": b_eff,
                "negative_b_eff_indices": [int(i) for i in jnp.where(infeasible_mask)[0]],
                "negative_b_eff_values": [float(b_eff[i]) for i in jnp.where(infeasible_mask)[0]],
                "max_feasible_scaling_alpha": alpha,
                "projected_feasible_m": alpha * m_current,
            }

        gas_state = ThermoState(
            temperature=state.temperature,
            ln_normalized_pressure=state.ln_normalized_pressure,
            element_vector=b_eff,
        )
        gas_result = solve_gas_equilibrium_with_duals(
            gas_state,
            formula_matrix,
            hvector_func,
            ln_nk_init=ln_nk_gas_init,
            ln_ntot_init=ln_ntot_gas_init,
            epsilon_crit=gas_epsilon_crit,
            max_iter=gas_max_iter,
        )
        hvector_cond = hvector_cond_func(state.temperature)
        summary = _summarize_layer(
            state=state,
            m=m_current,
            gas_result=gas_result,
            formula_matrix=formula_matrix,
            formula_matrix_cond=formula_matrix_cond,
            hvector_cond=hvector_cond,
            condensate_species=condensate_species,
            element_names=element_names,
            top_k=top_k,
            active_m_threshold=active_m_threshold,
            competitive_relative_tol=competitive_relative_tol,
            competitive_absolute_tol=competitive_absolute_tol,
            family_similarity_threshold=family_similarity_threshold,
            driving_tie_relative_tol=driving_tie_relative_tol,
            driving_tie_absolute_tol=driving_tie_absolute_tol,
            joint_activation_top_k=joint_activation_top_k,
            joint_activation_support_tol=joint_activation_support_tol,
        )
        summary["status"] = "ok"
        return summary

    initial = _evaluate(m0)
    history = [initial]

    if projected_outer_iterations > 0 and initial["status"] == "ok":
        current_m = jnp.asarray(m0)
        for _ in range(projected_outer_iterations):
            current = history[-1]
            driving = jnp.asarray(current["driving"])
            positive_order = [
                idx for idx in _top_indices(driving, descending=True, top_k=driving.shape[0])
                if float(driving[idx]) > 0.0
            ]
            candidate_indices = positive_order[:projected_activate_top_k]
            delta = jnp.zeros_like(current_m)
            if candidate_indices:
                max_positive = max(float(driving[idx]) for idx in candidate_indices)
                b_eff = jnp.asarray(current["b_eff"])
                for idx in candidate_indices:
                    feasible = _max_feasible_increment(formula_matrix_cond[:, idx], b_eff)
                    if feasible <= 0.0:
                        continue
                    scale = float(driving[idx]) / max(max_positive, 1.0e-300)
                    delta = delta.at[idx].set(projected_step_fraction * feasible * scale)

            active_negative = (current_m > active_m_threshold) & (driving < 0.0)
            shrunk = jnp.where(active_negative, (1.0 - projected_shrink_fraction) * current_m, current_m)
            current_m = shrunk + delta
            updated = _evaluate(current_m)
            updated["outer_update"] = {
                "candidate_indices": candidate_indices,
                "candidate_names": None
                if condensate_species is None
                else [str(condensate_species[i]) for i in candidate_indices],
                "delta_m": delta,
                "shrunk_active_negative_count": int(jnp.sum(active_negative)),
            }
            history.append(updated)
            if updated["status"] != "ok":
                break

    return {
        "prototype_family": "gas_inner_condensate_outer_active_set_diagnostic",
        "initial": initial,
        "history": history,
        "final": history[-1],
    }
