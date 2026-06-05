"""Reusable diagnostic policy for gas-boundary refresh candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class GasBoundaryRefreshCandidate:
    """Candidate refresh route and its solver-stage metrics."""

    policy_name: str
    candidate_kind: str
    floor_value: float | None
    solver_success: bool
    reached_final_barrier: bool
    converged_at_final_barrier: bool
    budget: float
    amount_weighted_gas: float
    complementarity: float | None
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GasBoundaryRefreshPolicyReport:
    """Selection report for reusable gas-boundary refresh policy."""

    report_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    case_id: str
    selected_policy: str
    selected_candidate_kind: str
    selected_floor_value: float | None
    accepted: bool
    acceptance_reason: str
    max_accepted_budget: float
    max_accepted_amount_weighted_gas: float
    candidate_count: int
    accepted_candidate_count: int
    candidates: Sequence[GasBoundaryRefreshCandidate]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidates"] = [candidate.as_dict() for candidate in self.candidates]
        return payload


def build_gas_boundary_refresh_candidate(
    *,
    policy_name: str,
    candidate_kind: str,
    budget: float,
    amount_weighted_gas: float,
    floor_value: float | None = None,
    solver_success: bool = True,
    reached_final_barrier: bool = True,
    converged_at_final_barrier: bool = True,
    complementarity: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GasBoundaryRefreshCandidate:
    """Build a refresh candidate from explicit diagnostic metrics."""

    if not str(policy_name):
        raise ValueError("policy_name must not be empty.")
    if not str(candidate_kind):
        raise ValueError("candidate_kind must not be empty.")
    if float(budget) < 0.0:
        raise ValueError("budget must be non-negative.")
    if float(amount_weighted_gas) < 0.0:
        raise ValueError("amount_weighted_gas must be non-negative.")
    if floor_value is not None and float(floor_value) <= 0.0:
        raise ValueError("floor_value must be positive when provided.")
    if complementarity is not None and float(complementarity) < 0.0:
        raise ValueError("complementarity must be non-negative when provided.")
    return GasBoundaryRefreshCandidate(
        policy_name=str(policy_name),
        candidate_kind=str(candidate_kind),
        floor_value=float(floor_value) if floor_value is not None else None,
        solver_success=bool(solver_success),
        reached_final_barrier=bool(reached_final_barrier),
        converged_at_final_barrier=bool(converged_at_final_barrier),
        budget=float(budget),
        amount_weighted_gas=float(amount_weighted_gas),
        complementarity=float(complementarity) if complementarity is not None else None,
        metadata=dict(metadata or {}),
    )


def candidate_is_refresh_accepted(
    candidate: GasBoundaryRefreshCandidate,
    *,
    max_accepted_budget: float,
    max_accepted_amount_weighted_gas: float,
) -> bool:
    """Return whether a candidate satisfies reusable refresh acceptance."""

    if float(max_accepted_budget) < 0.0:
        raise ValueError("max_accepted_budget must be non-negative.")
    if float(max_accepted_amount_weighted_gas) < 0.0:
        raise ValueError("max_accepted_amount_weighted_gas must be non-negative.")
    return (
        bool(candidate.solver_success)
        and bool(candidate.reached_final_barrier)
        and bool(candidate.converged_at_final_barrier)
        and float(candidate.budget) <= float(max_accepted_budget)
        and float(candidate.amount_weighted_gas) <= float(max_accepted_amount_weighted_gas)
    )


def select_gas_boundary_refresh_policy(
    *,
    explicit_opt_in: bool,
    case_id: str,
    candidates: Sequence[GasBoundaryRefreshCandidate],
    max_accepted_budget: float,
    max_accepted_amount_weighted_gas: float,
) -> GasBoundaryRefreshPolicyReport:
    """Select the first accepted gas-boundary refresh candidate.

    The caller owns candidate ordering. This allows one policy to represent both
    T500 depleted-budget tradeoff selection and M4374 adaptive floor selection
    without adding family-specific branches.
    """

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for gas-boundary refresh policy.")
    if not str(case_id):
        raise ValueError("case_id must not be empty.")
    ordered = tuple(candidates)
    if not ordered:
        raise ValueError("candidates must not be empty.")
    accepted = [
        candidate
        for candidate in ordered
        if candidate_is_refresh_accepted(
            candidate,
            max_accepted_budget=max_accepted_budget,
            max_accepted_amount_weighted_gas=max_accepted_amount_weighted_gas,
        )
    ]
    selected = accepted[0] if accepted else ordered[0]
    return GasBoundaryRefreshPolicyReport(
        report_schema="exogibbs_gas_boundary_refresh_policy_report_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        case_id=str(case_id),
        selected_policy=selected.policy_name if accepted else "none",
        selected_candidate_kind=selected.candidate_kind if accepted else "none",
        selected_floor_value=selected.floor_value if accepted else None,
        accepted=bool(accepted),
        acceptance_reason=(
            "The first accepted candidate satisfied final-barrier, budget, and gas metrics."
            if accepted
            else "No candidate satisfied final-barrier, budget, and gas metrics."
        ),
        max_accepted_budget=float(max_accepted_budget),
        max_accepted_amount_weighted_gas=float(max_accepted_amount_weighted_gas),
        candidate_count=len(ordered),
        accepted_candidate_count=len(accepted),
        candidates=ordered,
    )


__all__ = (
    "GasBoundaryRefreshCandidate",
    "GasBoundaryRefreshPolicyReport",
    "build_gas_boundary_refresh_candidate",
    "candidate_is_refresh_accepted",
    "select_gas_boundary_refresh_policy",
)
