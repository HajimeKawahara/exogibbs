"""Frontier refresh selection for condensate HEAD route lifecycle."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from exogibbs.condensates.gas_boundary_refresh_policy import (
    GasBoundaryRefreshPolicyReport,
    build_gas_boundary_refresh_candidate,
    select_gas_boundary_refresh_policy,
)


def select_frontier_refresh_from_metrics(
    *,
    explicit_opt_in: bool,
    case_id: str,
    candidate_metrics: Sequence[Mapping[str, Any]],
    max_accepted_budget: float,
    max_accepted_amount_weighted_gas: float,
) -> GasBoundaryRefreshPolicyReport:
    """Build and select reusable frontier refresh candidates from metrics."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for frontier refresh selection.")
    candidates = tuple(
        build_gas_boundary_refresh_candidate(
            policy_name=str(row.get("policy_name", "adaptive_floor_frontier_repair")),
            candidate_kind=str(row.get("candidate_kind", "hardened_gas_refresh_floor")),
            floor_value=row.get("floor_value"),
            solver_success=bool(row.get("solver_success", True)),
            reached_final_barrier=bool(row.get("reached_final_barrier", True)),
            converged_at_final_barrier=bool(row.get("converged_at_final_barrier", True)),
            budget=float(row["budget"]),
            amount_weighted_gas=float(row["amount_weighted_gas"]),
            complementarity=(
                None if row.get("complementarity") is None else float(row["complementarity"])
            ),
            metadata=dict(row.get("metadata", {})),
        )
        for row in candidate_metrics
    )
    return select_gas_boundary_refresh_policy(
        explicit_opt_in=True,
        case_id=case_id,
        candidates=candidates,
        max_accepted_budget=max_accepted_budget,
        max_accepted_amount_weighted_gas=max_accepted_amount_weighted_gas,
    )


__all__ = ("select_frontier_refresh_from_metrics",)
