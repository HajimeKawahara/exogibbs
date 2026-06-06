"""Residual-balanced coupled RGIE direction diagnostics.

This module is explicit-import diagnostic infrastructure. It does not import
FastChem4, call pyfastchem, call production solvers, or connect to presets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from exogibbs.condensates.native_bundle import validate_native_bundle_provenance


FORBIDDEN_PROVENANCE = {
    "fastchem4_trace",
    "fastchem4_public",
    "fastchem4_runtime",
    "branch_replay",
    "reference_fit",
    "unknown_reference",
}

CASE_SPLIT_STATIONARITY_POLICY = {
    "solar_water_condensation__T300_P1": "expanded_budget_plus_flipped_stationarity_selector",
    "solar_metal_sulfide_or_Fe_Ni_S_region__T700_P1": "budget_projected_stationarity_selector",
    "lowT_strong_condensation_budget_stress__T500_P1": "budget_projected_stationarity_selector",
}

EMPTY_SUPPORT_FLIPPED_PROJECTED_POLICY = {
    "solar_silicate_first_condensation__T1500_P1": "simple_stoich_flipped_projected_empty_support_selector",
    "solar_silicate_first_condensation__T1400_P0p1": "simple_stoich_flipped_projected_empty_support_selector",
    "near_phase_boundary_support_sensitivity__T1490_P1": "simple_stoich_flipped_projected_empty_support_selector",
    "near_phase_boundary_support_sensitivity__T1510_P1": "simple_stoich_flipped_projected_empty_support_selector",
}

NO_CONDENSATE_EMPTY_SUPPORT_POLICY = {
    "solar_highT_no_condensate_gas_regression__T2200_P1": "no_condensate_empty_support_skip",
}

SUPPORT_POSITIVE_JOINT_COMPONENT_POLICY = {
    "solar_water_condensation__T300_P1": "support_positive_joint_component_selector",
    "solar_metal_sulfide_or_Fe_Ni_S_region__T700_P1": "support_positive_joint_component_selector",
    "lowT_strong_condensation_budget_stress__T500_P1": "support_positive_joint_component_selector",
}

COMPONENT_SAFE_STAGNANT_NOOP_CASES = {
    "lowT_strong_condensation_budget_stress__T500_P1",
}


@dataclass(frozen=True)
class ResidualBalancedDirectionCandidate:
    """One candidate direction and its diagnostic component changes."""

    label: str
    alpha_budget: float
    alpha_stationarity: float
    lambda_trial: float
    delta_ln_nk: tuple[float, ...]
    delta_ln_mk: tuple[float, ...]
    delta_ln_ntot: float
    element_balance_residual_norm: float
    abs_ntot_residual: float
    gas_stationarity_residual_norm: float
    cond_stationarity_residual_norm: float
    element_balance_delta: float
    abs_ntot_delta: float
    gas_stationarity_delta: float
    cond_stationarity_delta: float
    budget_pair_improves: bool
    stationarity_pair_improves: bool
    joint_component_descent: bool
    finite: bool


@dataclass(frozen=True)
class ResidualBalancedDirectionReport:
    """Diagnostic report for residual-balanced coupled RGIE direction trials."""

    report_schema: str
    diagnostic_only: bool
    default_off: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    current_components: Mapping[str, float]
    selected_candidate: ResidualBalancedDirectionCandidate | None
    joint_descent_candidate_count: int
    finite_candidate_count: int
    candidates: tuple[ResidualBalancedDirectionCandidate, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResidualBalancedFirstStepCallsitePayload:
    """Explicit opt-in first-step payload for solver-adjacent diagnostics."""

    payload_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    lambda_trial: float
    delta_ln_nk: tuple[float, ...]
    delta_ln_mk: tuple[float, ...]
    delta_ln_ntot: float
    support_size: int
    joint_component_descent: bool
    finite: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResidualBalancedAppliedFirstStepState:
    """State vectors after applying an explicit first-step payload."""

    state_schema: str
    diagnostic_only: bool
    default_off: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    ln_nk: tuple[float, ...]
    ln_mk: tuple[float, ...]
    ln_ntot: float
    applied_lambda: float
    finite: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResidualBalancedFirstStepHardeningCase:
    """Hardening classification for one explicit first-step trace replay."""

    case_id: str
    final_residual_improved: bool
    baseline_final_residual: float
    first_step_final_residual: float
    final_residual_delta: float
    first_step_accept_kind: str
    first_step_hit_max_iter: bool
    internal_update_accepted: bool
    initial_state_improved: bool
    solver_wiring_ready: bool
    hardening_status: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResidualBalancedFirstStepHardeningReport:
    """Hardening report for explicit first-step trace comparisons."""

    report_schema: str
    diagnostic_only: bool
    default_off: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    evaluated_case_count: int
    improved_case_count: int
    internal_update_accepted_count: int
    solver_wiring_ready_count: int
    all_improved: bool
    all_internal_updates_accepted: bool
    solver_wiring_ready: bool
    cases: tuple[ResidualBalancedFirstStepHardeningCase, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IterativeResidualBalancedSelectorRecord:
    """One explicit diagnostic iteration of the residual-balanced selector."""

    iter_index: int
    iteration_status: str
    current_component_merit: float
    trial_component_merit: float | None
    component_merit_delta: float | None
    joint_component_descent: bool
    lambda_trial: float | None
    alpha_budget: float | None
    alpha_stationarity: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IterativeResidualBalancedSelectorReport:
    """Explicit diagnostic report for iterative residual-balanced selector trials."""

    report_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    iteration_count: int
    initial_component_merit: float
    final_component_merit: float
    final_vs_initial_delta: float
    all_trials_monotone: bool
    all_selected_joint_descent: bool
    finite: bool
    records: tuple[IterativeResidualBalancedSelectorRecord, ...]
    final_ln_nk: tuple[float, ...]
    final_ln_mk: tuple[float, ...]
    final_ln_ntot: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JointOnlyPostSelectorCandidatePayload:
    """Explicit diagnostic payload for joint-only post-selector candidates."""

    payload_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    lambda_trial: float
    delta_ln_nk: tuple[float, ...]
    delta_ln_mk: tuple[float, ...]
    delta_ln_ntot: float
    joint_component_descent: bool
    finite: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BudgetProjectedStationarityDirection:
    """Stationarity direction projected onto the linearized budget nullspace."""

    direction_schema: str
    diagnostic_only: bool
    default_off: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    delta_ln_nk: tuple[float, ...]
    delta_ln_mk: tuple[float, ...]
    delta_ln_ntot: float
    linearized_budget_residual_norm: float
    finite: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CaseSplitStationaritySelectorPolicy:
    """Explicit diagnostic selector policy for curated stationarity cases."""

    policy_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    case_id: str
    selected_policy: str
    finite: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IntegratedCuratedCaseSplitPolicy:
    """Explicit diagnostic policy for curated support-positive and empty-support cases."""

    policy_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    case_id: str
    native_support_size: int
    selected_policy: str
    replay_status: str
    finite: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComponentSafeCuratedPolicy:
    """Explicit diagnostic policy for component-safe curated condensate cases."""

    policy_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    case_id: str
    native_support_size: int
    selected_policy: str
    replay_status: str
    component_safe_expected: bool
    improvement_required_for_production_wiring: bool
    stagnant_noop_allowed_for_diagnostics: bool
    finite: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComponentSafePolicyPayload:
    """Explicit diagnostic payload for component-safe policy callsite trials."""

    payload_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    case_id: str
    selected_policy: str
    replay_status: str
    lambda_trial: float
    delta_ln_nk: tuple[float, ...]
    delta_ln_mk: tuple[float, ...]
    delta_ln_ntot: float
    support_size: int
    payload_kind: str
    finite: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TwoStageFlippedBudgetCorrectionReport:
    """Diagnostic report for flipped stationarity followed by budget correction."""

    report_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    initial_components: Mapping[str, float]
    after_flipped_components: Mapping[str, float]
    final_components: Mapping[str, float]
    initial_component_merit: float
    after_flipped_component_merit: float
    final_component_merit: float
    final_vs_initial_delta: float
    component_deltas: Mapping[str, float]
    all_primary_components_nonworsening: bool
    accepted: bool
    budget_correction_status: str
    budget_lambda_trial: float | None
    final_ln_nk: tuple[float, ...]
    final_ln_mk: tuple[float, ...]
    final_ln_ntot: float
    finite: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_case_split_stationarity_selector_policy(
    *,
    case_id: str,
    explicit_opt_in: bool,
    field_provenance: Mapping[str, str] | None = None,
) -> CaseSplitStationaritySelectorPolicy:
    """Return the diagnostic stationarity selector policy for a curated case."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for case-split stationarity policy.")
    _validate_provenance(field_provenance)
    if case_id not in CASE_SPLIT_STATIONARITY_POLICY:
        raise ValueError("case_id is not covered by the curated stationarity selector policy.")
    return CaseSplitStationaritySelectorPolicy(
        policy_schema="exogibbs_case_split_stationarity_selector_policy_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        case_id=case_id,
        selected_policy=CASE_SPLIT_STATIONARITY_POLICY[case_id],
        finite=True,
    )


def build_integrated_curated_case_split_policy(
    *,
    case_id: str,
    native_support_size: int,
    explicit_opt_in: bool,
    field_provenance: Mapping[str, str] | None = None,
) -> IntegratedCuratedCaseSplitPolicy:
    """Return the explicit diagnostic integrated policy for a curated case."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for integrated curated policy.")
    if native_support_size < 0:
        raise ValueError("native_support_size must be non-negative.")
    _validate_provenance(field_provenance)
    if native_support_size > 0:
        if case_id not in CASE_SPLIT_STATIONARITY_POLICY:
            raise ValueError("support-positive case_id is not covered by the integrated curated policy.")
        selected_policy = "support_positive_case_split_selector"
        replay_status = "replayable"
    elif case_id in NO_CONDENSATE_EMPTY_SUPPORT_POLICY:
        selected_policy = NO_CONDENSATE_EMPTY_SUPPORT_POLICY[case_id]
        replay_status = "classified_skip"
    elif case_id in EMPTY_SUPPORT_FLIPPED_PROJECTED_POLICY:
        selected_policy = EMPTY_SUPPORT_FLIPPED_PROJECTED_POLICY[case_id]
        replay_status = "replayable"
    else:
        raise ValueError("empty-support case_id is not covered by the integrated curated policy.")
    return IntegratedCuratedCaseSplitPolicy(
        policy_schema="exogibbs_integrated_curated_case_split_policy_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        case_id=case_id,
        native_support_size=int(native_support_size),
        selected_policy=selected_policy,
        replay_status=replay_status,
        finite=True,
    )


def build_component_safe_curated_policy(
    *,
    case_id: str,
    native_support_size: int,
    explicit_opt_in: bool,
    allow_stagnant_noop: bool = True,
    field_provenance: Mapping[str, str] | None = None,
) -> ComponentSafeCuratedPolicy:
    """Return the explicit diagnostic component-safe policy for a curated case."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for component-safe curated policy.")
    if native_support_size < 0:
        raise ValueError("native_support_size must be non-negative.")
    _validate_provenance(field_provenance)
    stagnant = case_id in COMPONENT_SAFE_STAGNANT_NOOP_CASES
    if native_support_size > 0:
        if case_id not in SUPPORT_POSITIVE_JOINT_COMPONENT_POLICY:
            raise ValueError("support-positive case_id is not covered by the component-safe policy.")
        selected_policy = SUPPORT_POSITIVE_JOINT_COMPONENT_POLICY[case_id]
        replay_status = "replayable_stagnant_noop" if stagnant else "replayable_improving"
    elif case_id in NO_CONDENSATE_EMPTY_SUPPORT_POLICY:
        selected_policy = NO_CONDENSATE_EMPTY_SUPPORT_POLICY[case_id]
        replay_status = "classified_skip"
    elif case_id in EMPTY_SUPPORT_FLIPPED_PROJECTED_POLICY:
        selected_policy = "two_stage_flipped_budget_correction"
        replay_status = "replayable_improving"
    else:
        raise ValueError("case_id is not covered by the component-safe curated policy.")
    if stagnant and not allow_stagnant_noop:
        raise ValueError("stagnant no-op cases require allow_stagnant_noop for diagnostics.")
    return ComponentSafeCuratedPolicy(
        policy_schema="exogibbs_component_safe_curated_policy_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        case_id=case_id,
        native_support_size=int(native_support_size),
        selected_policy=selected_policy,
        replay_status=replay_status,
        component_safe_expected=True,
        improvement_required_for_production_wiring=stagnant,
        stagnant_noop_allowed_for_diagnostics=bool(stagnant and allow_stagnant_noop),
        finite=True,
    )


def build_component_safe_policy_payload(
    *,
    policy: ComponentSafeCuratedPolicy | Mapping[str, Any],
    explicit_opt_in: bool,
    delta_ln_nk: Sequence[float] | None = None,
    delta_ln_mk: Sequence[float] | None = None,
    delta_ln_ntot: float = 0.0,
    lambda_trial: float = 1.0,
    support_size: int | None = None,
    max_abs_delta_ln_nk: float = 1.0e4,
    max_abs_delta_ln_mk: float = 1.0e4,
    max_abs_delta_ln_ntot: float = 1.0e4,
    field_provenance: Mapping[str, str] | None = None,
) -> ComponentSafePolicyPayload:
    """Validate a component-safe diagnostic policy payload."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for component-safe policy payloads.")
    _validate_provenance(field_provenance)
    data = policy.as_dict() if isinstance(policy, ComponentSafeCuratedPolicy) else dict(policy)
    selected_policy = str(data["selected_policy"])
    replay_status = str(data["replay_status"])
    if replay_status == "classified_skip":
        payload_kind = "classified_skip"
        delta_q = np.zeros(0, dtype=np.float64)
        delta_r = np.zeros(0, dtype=np.float64)
        delta_qtot = 0.0
        lam = 0.0
    elif replay_status == "replayable_stagnant_noop":
        payload_kind = "stagnant_noop"
        delta_q = np.zeros(0, dtype=np.float64)
        delta_r = np.zeros(0, dtype=np.float64)
        delta_qtot = 0.0
        lam = 0.0
    else:
        payload_kind = "update"
        if delta_ln_nk is None or delta_ln_mk is None:
            raise ValueError("update payloads require delta_ln_nk and delta_ln_mk.")
        delta_q = _require_vector(delta_ln_nk, "delta_ln_nk")
        delta_r = _require_vector(delta_ln_mk, "delta_ln_mk")
        delta_qtot = float(delta_ln_ntot)
        lam = float(lambda_trial)
    if not np.isfinite(delta_qtot) or not np.isfinite(lam):
        raise ValueError("lambda_trial and delta_ln_ntot must be finite.")
    if delta_q.size and np.max(np.abs(delta_q)) > float(max_abs_delta_ln_nk):
        raise ValueError("delta_ln_nk exceeds max_abs_delta_ln_nk.")
    if delta_r.size and np.max(np.abs(delta_r)) > float(max_abs_delta_ln_mk):
        raise ValueError("delta_ln_mk exceeds max_abs_delta_ln_mk.")
    if abs(delta_qtot) > float(max_abs_delta_ln_ntot):
        raise ValueError("delta_ln_ntot exceeds max_abs_delta_ln_ntot.")
    finite = bool(np.all(np.isfinite(delta_q)) and np.all(np.isfinite(delta_r)))
    return ComponentSafePolicyPayload(
        payload_schema="exogibbs_component_safe_policy_payload_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        case_id=str(data["case_id"]),
        selected_policy=selected_policy,
        replay_status=replay_status,
        lambda_trial=lam,
        delta_ln_nk=tuple(float(value) for value in delta_q),
        delta_ln_mk=tuple(float(value) for value in delta_r),
        delta_ln_ntot=delta_qtot,
        support_size=int(data["native_support_size"] if support_size is None else support_size),
        payload_kind=payload_kind,
        finite=finite,
    )


def _require_matrix(value: Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def _require_vector(value: Sequence[float], name: str, size: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    if size is not None and array.shape[0] != size:
        raise ValueError(f"{name} length mismatch: got {array.shape[0]}, expected {size}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def _validate_provenance(field_provenance: Mapping[str, str] | None) -> None:
    provenance = validate_native_bundle_provenance(field_provenance or {})
    for value in provenance.values():
        if value in FORBIDDEN_PROVENANCE:
            raise ValueError("Forbidden reference provenance cannot enter direction diagnostics.")


def _compute_gk(
    temperature: float,
    ln_nk: np.ndarray,
    ln_ntot: float,
    hvector: np.ndarray,
    ln_normalized_pressure: float,
) -> np.ndarray:
    del temperature
    return hvector + ln_nk - ln_ntot - ln_normalized_pressure


def _estimate_pi(
    formula_matrix: np.ndarray,
    formula_matrix_cond: np.ndarray,
    nk: np.ndarray,
    mk: np.ndarray,
    gk: np.ndarray,
    hvector_cond: np.ndarray,
    nu: float,
) -> np.ndarray:
    gas_weight = np.maximum(nk, 0.0)
    cond_weight = np.maximum(mk, 0.0)
    qmat = (
        formula_matrix @ (gas_weight[:, None] * formula_matrix.T)
        + formula_matrix_cond @ (cond_weight[:, None] * formula_matrix_cond.T)
    )
    rhs = formula_matrix @ (gas_weight * gk)
    rhs += formula_matrix_cond @ (cond_weight * hvector_cond - nu)
    ridge = 1.0e-12 * max(1.0, float(np.max(np.abs(qmat))))
    qmat = qmat + ridge * np.eye(qmat.shape[0], dtype=np.float64)
    return np.linalg.lstsq(qmat, rhs, rcond=None)[0]


def _components(
    ln_nk: np.ndarray,
    ln_mk: np.ndarray,
    ln_ntot: float,
    formula_matrix: np.ndarray,
    formula_matrix_cond: np.ndarray,
    element_budget: np.ndarray,
    hvector: np.ndarray,
    hvector_cond: np.ndarray,
    ln_normalized_pressure: float,
    epsilon: float,
) -> dict[str, Any]:
    nk = np.exp(ln_nk)
    mk = np.exp(ln_mk)
    ntot = float(np.exp(ln_ntot))
    gk = _compute_gk(1.0, ln_nk, ln_ntot, hvector, ln_normalized_pressure)
    nu = float(np.exp(epsilon))
    pi = _estimate_pi(formula_matrix, formula_matrix_cond, nk, mk, gk, hvector_cond, nu)
    gas_stationarity = nk * (formula_matrix.T @ pi - gk)
    cond_stationarity = mk * (formula_matrix_cond.T @ pi - hvector_cond) + nu
    element_balance = formula_matrix @ nk + formula_matrix_cond @ mk - element_budget
    ntot_residual = float(np.sum(nk) - ntot)
    return {
        "nk": nk,
        "mk": mk,
        "ntot": ntot,
        "gk": gk,
        "pi": pi,
        "gas_stationarity": gas_stationarity,
        "cond_stationarity": cond_stationarity,
        "element_balance": element_balance,
        "ntot_residual": ntot_residual,
        "element_balance_residual_norm": float(np.linalg.norm(element_balance)),
        "abs_ntot_residual": abs(ntot_residual),
        "gas_stationarity_residual_norm": float(np.linalg.norm(gas_stationarity)),
        "cond_stationarity_residual_norm": float(np.linalg.norm(cond_stationarity)),
    }


def _budget_direction(
    nk: np.ndarray,
    mk: np.ndarray,
    ntot: float,
    formula_matrix: np.ndarray,
    formula_matrix_cond: np.ndarray,
    element_balance: np.ndarray,
    ntot_residual: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    n_gas = nk.shape[0]
    n_cond = mk.shape[0]
    jac_budget = np.hstack(
        [
            formula_matrix * nk[None, :],
            formula_matrix_cond * mk[None, :],
            np.zeros((formula_matrix.shape[0], 1), dtype=np.float64),
        ]
    )
    jac_ntot = np.concatenate([nk, np.zeros(n_cond, dtype=np.float64), [-ntot]])[None, :]
    mat = np.vstack([jac_budget, jac_ntot])
    rhs = -np.concatenate([element_balance, [ntot_residual]])
    ridge = 1.0e-10 * max(1.0, float(np.max(np.abs(mat))))
    augmented = np.vstack([mat, ridge * np.eye(mat.shape[1], dtype=np.float64)])
    augmented_rhs = np.concatenate([rhs, np.zeros(mat.shape[1], dtype=np.float64)])
    solution = np.linalg.lstsq(augmented, augmented_rhs, rcond=None)[0]
    return solution[:n_gas], solution[n_gas : n_gas + n_cond], float(solution[-1])


def _stationarity_direction(
    nk: np.ndarray,
    mk: np.ndarray,
    gas_stationarity: np.ndarray,
    cond_stationarity: np.ndarray,
    *,
    tiny: float = 1.0e-300,
) -> tuple[np.ndarray, np.ndarray, float]:
    delta_ln_nk = -gas_stationarity / np.maximum(nk, tiny)
    delta_ln_mk = -cond_stationarity / np.maximum(mk, tiny)
    delta_ln_mk = np.clip(delta_ln_mk, -0.1, 0.1)
    return delta_ln_nk, delta_ln_mk, 0.0


def project_stationarity_direction_to_budget_nullspace(
    *,
    ln_nk: Sequence[float],
    ln_mk: Sequence[float],
    ln_ntot: float,
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond: Sequence[Sequence[float]],
    element_budget: Sequence[float],
    hvector: Sequence[float],
    hvector_cond: Sequence[float],
    ln_normalized_pressure: float,
    epsilon: float,
    field_provenance: Mapping[str, str] | None = None,
) -> BudgetProjectedStationarityDirection:
    """Project the local stationarity direction away from linearized budget rows."""

    _validate_provenance(field_provenance)
    ag = _require_matrix(formula_matrix, "formula_matrix")
    ac = _require_matrix(formula_matrix_cond, "formula_matrix_cond")
    q = _require_vector(ln_nk, "ln_nk", ag.shape[1])
    r = _require_vector(ln_mk, "ln_mk", ac.shape[1])
    b = _require_vector(element_budget, "element_budget", ag.shape[0])
    hgas = _require_vector(hvector, "hvector", ag.shape[1])
    hcond = _require_vector(hvector_cond, "hvector_cond", ac.shape[1])
    if ac.shape[0] != ag.shape[0]:
        raise ValueError("formula_matrix and formula_matrix_cond row counts must match.")
    current = _components(q, r, float(ln_ntot), ag, ac, b, hgas, hcond, float(ln_normalized_pressure), float(epsilon))
    stat_q, stat_r, stat_qtot = _stationarity_direction(
        current["nk"],
        current["mk"],
        current["gas_stationarity"],
        current["cond_stationarity"],
    )
    raw = np.concatenate([stat_q, stat_r, [stat_qtot]])
    n_cond = current["mk"].shape[0]
    jac_budget = np.hstack(
        [
            ag * current["nk"][None, :],
            ac * current["mk"][None, :],
            np.zeros((ag.shape[0], 1), dtype=np.float64),
        ]
    )
    jac_ntot = np.concatenate([current["nk"], np.zeros(n_cond, dtype=np.float64), [-current["ntot"]]])[None, :]
    mat = np.vstack([jac_budget, jac_ntot])
    gram = mat @ mat.T
    ridge = 1.0e-12 * max(1.0, float(np.max(np.abs(gram))))
    correction = mat.T @ np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), mat @ raw)
    projected = raw - correction
    residual_norm = float(np.linalg.norm(mat @ projected))
    n_gas = q.shape[0]
    delta_q = projected[:n_gas]
    delta_r = projected[n_gas : n_gas + n_cond]
    delta_qtot = float(projected[-1])
    finite = bool(np.all(np.isfinite(projected)) and np.isfinite(residual_norm))
    return BudgetProjectedStationarityDirection(
        direction_schema="exogibbs_budget_projected_stationarity_direction_v1",
        diagnostic_only=True,
        default_off=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        delta_ln_nk=tuple(float(value) for value in delta_q),
        delta_ln_mk=tuple(float(value) for value in delta_r),
        delta_ln_ntot=delta_qtot,
        linearized_budget_residual_norm=residual_norm,
        finite=finite,
    )


def _component_dict(components: Mapping[str, Any]) -> dict[str, float]:
    return {
        "element_balance_residual_norm": float(
            components["element_balance_residual_norm"]
        ),
        "abs_ntot_residual": float(components["abs_ntot_residual"]),
        "gas_stationarity_residual_norm": float(
            components["gas_stationarity_residual_norm"]
        ),
        "cond_stationarity_residual_norm": float(
            components["cond_stationarity_residual_norm"]
        ),
    }


def _candidate(
    *,
    label: str,
    alpha_budget: float,
    alpha_stationarity: float,
    lambda_trial: float,
    current: Mapping[str, Any],
    trial: Mapping[str, Any],
    delta_ln_nk: np.ndarray,
    delta_ln_mk: np.ndarray,
    delta_ln_ntot: float,
) -> ResidualBalancedDirectionCandidate:
    element_delta = (
        float(trial["element_balance_residual_norm"])
        - float(current["element_balance_residual_norm"])
    )
    ntot_delta = float(trial["abs_ntot_residual"]) - float(current["abs_ntot_residual"])
    gas_delta = (
        float(trial["gas_stationarity_residual_norm"])
        - float(current["gas_stationarity_residual_norm"])
    )
    cond_delta = (
        float(trial["cond_stationarity_residual_norm"])
        - float(current["cond_stationarity_residual_norm"])
    )
    finite = bool(
        np.all(np.isfinite(delta_ln_nk))
        and np.all(np.isfinite(delta_ln_mk))
        and np.isfinite(delta_ln_ntot)
        and np.all(np.isfinite(list(_component_dict(trial).values())))
    )
    budget_pair = element_delta < 0.0 and ntot_delta <= 0.0
    stationarity_pair = gas_delta < 0.0 and cond_delta <= 0.0
    return ResidualBalancedDirectionCandidate(
        label=label,
        alpha_budget=float(alpha_budget),
        alpha_stationarity=float(alpha_stationarity),
        lambda_trial=float(lambda_trial),
        delta_ln_nk=tuple(float(value) for value in delta_ln_nk),
        delta_ln_mk=tuple(float(value) for value in delta_ln_mk),
        delta_ln_ntot=float(delta_ln_ntot),
        element_balance_residual_norm=float(trial["element_balance_residual_norm"]),
        abs_ntot_residual=float(trial["abs_ntot_residual"]),
        gas_stationarity_residual_norm=float(trial["gas_stationarity_residual_norm"]),
        cond_stationarity_residual_norm=float(trial["cond_stationarity_residual_norm"]),
        element_balance_delta=element_delta,
        abs_ntot_delta=ntot_delta,
        gas_stationarity_delta=gas_delta,
        cond_stationarity_delta=cond_delta,
        budget_pair_improves=budget_pair,
        stationarity_pair_improves=stationarity_pair,
        joint_component_descent=budget_pair and stationarity_pair,
        finite=finite,
    )


def build_residual_balanced_coupled_rgie_direction(
    *,
    ln_nk: Sequence[float],
    ln_mk: Sequence[float],
    ln_ntot: float,
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond: Sequence[Sequence[float]],
    element_budget: Sequence[float],
    hvector: Sequence[float],
    hvector_cond: Sequence[float],
    ln_normalized_pressure: float,
    epsilon: float,
    alpha_budget_values: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    alpha_stationarity_values: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    lambda_values: Sequence[float] = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1),
    field_provenance: Mapping[str, str] | None = None,
) -> ResidualBalancedDirectionReport:
    """Evaluate residual-balanced budget/stationarity direction candidates."""

    _validate_provenance(field_provenance)
    ag = _require_matrix(formula_matrix, "formula_matrix")
    ac = _require_matrix(formula_matrix_cond, "formula_matrix_cond")
    q = _require_vector(ln_nk, "ln_nk", ag.shape[1])
    r = _require_vector(ln_mk, "ln_mk", ac.shape[1])
    b = _require_vector(element_budget, "element_budget", ag.shape[0])
    hgas = _require_vector(hvector, "hvector", ag.shape[1])
    hcond = _require_vector(hvector_cond, "hvector_cond", ac.shape[1])
    if ac.shape[0] != ag.shape[0]:
        raise ValueError("formula_matrix and formula_matrix_cond row counts must match.")
    current_full = _components(q, r, ln_ntot, ag, ac, b, hgas, hcond, ln_normalized_pressure, epsilon)
    current = _component_dict(current_full)
    budget_q, budget_r, budget_qtot = _budget_direction(
        current_full["nk"],
        current_full["mk"],
        current_full["ntot"],
        ag,
        ac,
        current_full["element_balance"],
        current_full["ntot_residual"],
    )
    stat_q, stat_r, stat_qtot = _stationarity_direction(
        current_full["nk"],
        current_full["mk"],
        current_full["gas_stationarity"],
        current_full["cond_stationarity"],
    )
    candidates: list[ResidualBalancedDirectionCandidate] = []
    for alpha_budget in alpha_budget_values:
        for alpha_stationarity in alpha_stationarity_values:
            if float(alpha_budget) == 0.0 and float(alpha_stationarity) == 0.0:
                continue
            base_q = float(alpha_budget) * budget_q + float(alpha_stationarity) * stat_q
            base_r = float(alpha_budget) * budget_r + float(alpha_stationarity) * stat_r
            base_qtot = float(alpha_budget) * budget_qtot + float(alpha_stationarity) * stat_qtot
            for lambda_trial in lambda_values:
                lam = float(lambda_trial)
                trial_full = _components(
                    q + lam * base_q,
                    r + lam * base_r,
                    float(ln_ntot) + lam * base_qtot,
                    ag,
                    ac,
                    b,
                    hgas,
                    hcond,
                    ln_normalized_pressure,
                    epsilon,
                )
                candidates.append(
                    _candidate(
                        label="residual_balanced_budget_stationarity_blend",
                        alpha_budget=float(alpha_budget),
                        alpha_stationarity=float(alpha_stationarity),
                        lambda_trial=lam,
                        current=current,
                        trial=_component_dict(trial_full),
                        delta_ln_nk=base_q,
                        delta_ln_mk=base_r,
                        delta_ln_ntot=base_qtot,
                    )
                )
    finite = [candidate for candidate in candidates if candidate.finite]
    joint = [candidate for candidate in finite if candidate.joint_component_descent]
    selected = None
    if joint:
        selected = min(
            joint,
            key=lambda candidate: (
                candidate.element_balance_residual_norm
                + candidate.abs_ntot_residual
                + candidate.gas_stationarity_residual_norm
                + candidate.cond_stationarity_residual_norm
            ),
        )
    elif finite:
        selected = min(
            finite,
            key=lambda candidate: (
                candidate.element_balance_residual_norm
                + candidate.abs_ntot_residual
                + candidate.gas_stationarity_residual_norm
                + candidate.cond_stationarity_residual_norm
            ),
        )
    return ResidualBalancedDirectionReport(
        report_schema="exogibbs_residual_balanced_coupled_rgie_direction_report_v1",
        diagnostic_only=True,
        default_off=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        current_components=current,
        selected_candidate=selected,
        joint_descent_candidate_count=len(joint),
        finite_candidate_count=len(finite),
        candidates=tuple(candidates),
    )


def build_explicit_first_step_callsite_payload(
    candidate: ResidualBalancedDirectionCandidate | Mapping[str, Any],
    *,
    explicit_opt_in: bool,
    max_abs_delta_ln_nk: float = 1.0e4,
    max_abs_delta_ln_mk: float = 1.0,
    max_abs_delta_ln_ntot: float = 1.0e4,
    field_provenance: Mapping[str, str] | None = None,
) -> ResidualBalancedFirstStepCallsitePayload:
    """Validate one residual-balanced candidate for explicit first-step dry-run use."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for first-step callsite payloads.")
    _validate_provenance(field_provenance)
    data = asdict(candidate) if isinstance(candidate, ResidualBalancedDirectionCandidate) else dict(candidate)
    delta_q = _require_vector(data["delta_ln_nk"], "delta_ln_nk")
    delta_r = _require_vector(data["delta_ln_mk"], "delta_ln_mk")
    delta_qtot = float(data["delta_ln_ntot"])
    if not np.isfinite(delta_qtot):
        raise ValueError("delta_ln_ntot must be finite.")
    if np.max(np.abs(delta_q)) > float(max_abs_delta_ln_nk):
        raise ValueError("delta_ln_nk exceeds max_abs_delta_ln_nk.")
    if np.max(np.abs(delta_r)) > float(max_abs_delta_ln_mk):
        raise ValueError("delta_ln_mk exceeds max_abs_delta_ln_mk.")
    if abs(delta_qtot) > float(max_abs_delta_ln_ntot):
        raise ValueError("delta_ln_ntot exceeds max_abs_delta_ln_ntot.")
    finite = bool(data["finite"] and np.all(np.isfinite(delta_q)) and np.all(np.isfinite(delta_r)))
    return ResidualBalancedFirstStepCallsitePayload(
        payload_schema="exogibbs_residual_balanced_first_step_callsite_payload_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        lambda_trial=float(data["lambda_trial"]),
        delta_ln_nk=tuple(float(value) for value in delta_q),
        delta_ln_mk=tuple(float(value) for value in delta_r),
        delta_ln_ntot=delta_qtot,
        support_size=int(delta_r.shape[0]),
        joint_component_descent=bool(data["joint_component_descent"]),
        finite=finite,
    )


def apply_explicit_first_step_payload_to_logs(
    *,
    ln_nk: Sequence[float],
    ln_mk: Sequence[float],
    ln_ntot: float,
    payload: ResidualBalancedFirstStepCallsitePayload | Mapping[str, Any],
    explicit_opt_in: bool,
) -> ResidualBalancedAppliedFirstStepState:
    """Apply a validated explicit first-step payload to log-state arrays."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true to apply a first-step payload.")
    data = asdict(payload) if isinstance(payload, ResidualBalancedFirstStepCallsitePayload) else dict(payload)
    if not bool(data.get("explicit_opt_in", False)):
        raise ValueError("payload must have explicit_opt_in true.")
    if not bool(data.get("finite", False)):
        raise ValueError("payload must be finite.")
    q = _require_vector(ln_nk, "ln_nk", len(data["delta_ln_nk"]))
    r = _require_vector(ln_mk, "ln_mk", len(data["delta_ln_mk"]))
    qtot = float(ln_ntot)
    if not np.isfinite(qtot):
        raise ValueError("ln_ntot must be finite.")
    lam = float(data["lambda_trial"])
    delta_q = _require_vector(data["delta_ln_nk"], "delta_ln_nk", q.shape[0])
    delta_r = _require_vector(data["delta_ln_mk"], "delta_ln_mk", r.shape[0])
    delta_qtot = float(data["delta_ln_ntot"])
    new_q = q + lam * delta_q
    new_r = r + lam * delta_r
    new_qtot = qtot + lam * delta_qtot
    finite = bool(
        np.all(np.isfinite(new_q))
        and np.all(np.isfinite(new_r))
        and np.isfinite(new_qtot)
    )
    return ResidualBalancedAppliedFirstStepState(
        state_schema="exogibbs_residual_balanced_applied_first_step_state_v1",
        diagnostic_only=True,
        default_off=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        ln_nk=tuple(float(value) for value in new_q),
        ln_mk=tuple(float(value) for value in new_r),
        ln_ntot=float(new_qtot),
        applied_lambda=lam,
        finite=finite,
    )


def classify_explicit_first_step_trace_hardening(
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    accepted_internal_update_kinds: Sequence[str] = (
        "monotone",
        "best_finite_fallback",
        "best_finite_opt_in",
        "best_finite_nonincreasing_opt_in",
        "charge_aware_composite_opt_in",
        "charge_aware_composite_nonincreasing_opt_in",
        "component_composite_nonincreasing_opt_in",
    ),
) -> ResidualBalancedFirstStepHardeningReport:
    """Classify whether explicit first-step trace evidence is wiring-ready."""

    accepted_kinds = set(accepted_internal_update_kinds)
    cases: list[ResidualBalancedFirstStepHardeningCase] = []
    for row in comparison_rows:
        if row.get("case_status") != "evaluated":
            continue
        first_accept = str(row["first_step_first_accept_kind"])
        improved = bool(row["final_residual_improved"])
        hit_max_iter = bool(row["first_step_hit_max_iter"])
        internal_accepted = first_accept in accepted_kinds
        wiring_ready = improved and internal_accepted and not hit_max_iter
        if wiring_ready:
            status = "ready_for_explicit_solver_wiring_candidate"
        elif improved and not internal_accepted:
            status = "initial_state_improved_but_internal_update_gap_remains"
        elif improved and hit_max_iter:
            status = "initial_state_improved_but_iteration_limit_gap_remains"
        else:
            status = "first_step_not_improved"
        cases.append(
            ResidualBalancedFirstStepHardeningCase(
                case_id=str(row["case_id"]),
                final_residual_improved=improved,
                baseline_final_residual=float(row["baseline_final_residual"]),
                first_step_final_residual=float(row["first_step_final_residual"]),
                final_residual_delta=float(row["final_residual_delta"]),
                first_step_accept_kind=first_accept,
                first_step_hit_max_iter=hit_max_iter,
                internal_update_accepted=internal_accepted,
                initial_state_improved=improved,
                solver_wiring_ready=wiring_ready,
                hardening_status=status,
            )
        )
    evaluated = len(cases)
    improved_count = sum(case.final_residual_improved for case in cases)
    accepted_count = sum(case.internal_update_accepted for case in cases)
    ready_count = sum(case.solver_wiring_ready for case in cases)
    return ResidualBalancedFirstStepHardeningReport(
        report_schema="exogibbs_residual_balanced_first_step_hardening_report_v1",
        diagnostic_only=True,
        default_off=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        evaluated_case_count=evaluated,
        improved_case_count=improved_count,
        internal_update_accepted_count=accepted_count,
        solver_wiring_ready_count=ready_count,
        all_improved=evaluated > 0 and improved_count == evaluated,
        all_internal_updates_accepted=evaluated > 0 and accepted_count == evaluated,
        solver_wiring_ready=evaluated > 0 and ready_count == evaluated,
        cases=tuple(cases),
    )


def _component_merit(components: Mapping[str, float]) -> float:
    return float(
        components["element_balance_residual_norm"]
        + components["abs_ntot_residual"]
        + components["gas_stationarity_residual_norm"]
        + components["cond_stationarity_residual_norm"]
    )


def run_iterative_residual_balanced_selector(
    *,
    ln_nk: Sequence[float],
    ln_mk: Sequence[float],
    ln_ntot: float,
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond: Sequence[Sequence[float]],
    element_budget: Sequence[float],
    hvector: Sequence[float],
    hvector_cond: Sequence[float],
    ln_normalized_pressure: float,
    epsilon: float,
    explicit_opt_in: bool,
    max_iterations: int = 5,
    field_provenance: Mapping[str, str] | None = None,
) -> IterativeResidualBalancedSelectorReport:
    """Run an explicit diagnostic iterative residual-balanced selector replay."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for iterative selector diagnostics.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    _validate_provenance(field_provenance)
    q = _require_vector(ln_nk, "ln_nk")
    r = _require_vector(ln_mk, "ln_mk")
    qtot = float(ln_ntot)
    if not np.isfinite(qtot):
        raise ValueError("ln_ntot must be finite.")
    records: list[IterativeResidualBalancedSelectorRecord] = []
    initial_merit: float | None = None
    final_merit: float | None = None
    for iter_index in range(int(max_iterations)):
        report = build_residual_balanced_coupled_rgie_direction(
            ln_nk=q,
            ln_mk=r,
            ln_ntot=qtot,
            formula_matrix=formula_matrix,
            formula_matrix_cond=formula_matrix_cond,
            element_budget=element_budget,
            hvector=hvector,
            hvector_cond=hvector_cond,
            ln_normalized_pressure=ln_normalized_pressure,
            epsilon=epsilon,
            field_provenance=field_provenance,
        )
        current_merit = _component_merit(report.current_components)
        if initial_merit is None:
            initial_merit = current_merit
        selected = report.selected_candidate
        if selected is None:
            records.append(
                IterativeResidualBalancedSelectorRecord(
                    iter_index=iter_index,
                    iteration_status="no_selected_candidate",
                    current_component_merit=current_merit,
                    trial_component_merit=None,
                    component_merit_delta=None,
                    joint_component_descent=False,
                    lambda_trial=None,
                    alpha_budget=None,
                    alpha_stationarity=None,
                )
            )
            final_merit = current_merit
            break
        trial_merit = float(
            selected.element_balance_residual_norm
            + selected.abs_ntot_residual
            + selected.gas_stationarity_residual_norm
            + selected.cond_stationarity_residual_norm
        )
        records.append(
            IterativeResidualBalancedSelectorRecord(
                iter_index=iter_index,
                iteration_status="selected",
                current_component_merit=current_merit,
                trial_component_merit=trial_merit,
                component_merit_delta=trial_merit - current_merit,
                joint_component_descent=selected.joint_component_descent,
                lambda_trial=selected.lambda_trial,
                alpha_budget=selected.alpha_budget,
                alpha_stationarity=selected.alpha_stationarity,
            )
        )
        q = q + selected.lambda_trial * np.asarray(selected.delta_ln_nk, dtype=np.float64)
        r = r + selected.lambda_trial * np.asarray(selected.delta_ln_mk, dtype=np.float64)
        qtot = qtot + selected.lambda_trial * selected.delta_ln_ntot
        final_merit = trial_merit
    selected_records = [record for record in records if record.iteration_status == "selected"]
    all_trials_monotone = bool(
        selected_records
        and all(
            record.component_merit_delta is not None
            and record.component_merit_delta < 0.0
            for record in selected_records
        )
    )
    all_joint = bool(
        selected_records and all(record.joint_component_descent for record in selected_records)
    )
    initial = float(initial_merit if initial_merit is not None else np.nan)
    final = float(final_merit if final_merit is not None else initial)
    finite = bool(np.all(np.isfinite(q)) and np.all(np.isfinite(r)) and np.isfinite(qtot))
    return IterativeResidualBalancedSelectorReport(
        report_schema="exogibbs_iterative_residual_balanced_selector_report_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        iteration_count=len(selected_records),
        initial_component_merit=initial,
        final_component_merit=final,
        final_vs_initial_delta=final - initial,
        all_trials_monotone=all_trials_monotone,
        all_selected_joint_descent=all_joint,
        finite=finite,
        records=tuple(records),
        final_ln_nk=tuple(float(value) for value in q),
        final_ln_mk=tuple(float(value) for value in r),
        final_ln_ntot=float(qtot),
    )


def build_joint_only_post_selector_candidate_payload(
    candidate: ResidualBalancedDirectionCandidate | Mapping[str, Any],
    *,
    explicit_opt_in: bool,
    max_abs_delta_ln_nk: float = 1.0e4,
    max_abs_delta_ln_mk: float = 1.0,
    max_abs_delta_ln_ntot: float = 1.0e4,
    field_provenance: Mapping[str, str] | None = None,
) -> JointOnlyPostSelectorCandidatePayload:
    """Validate a post-selector candidate requiring joint component descent."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for post-selector candidates.")
    _validate_provenance(field_provenance)
    data = asdict(candidate) if isinstance(candidate, ResidualBalancedDirectionCandidate) else dict(candidate)
    if not bool(data.get("joint_component_descent", False)):
        raise ValueError("post-selector candidate must have joint_component_descent true.")
    delta_q = _require_vector(data["delta_ln_nk"], "delta_ln_nk")
    delta_r = _require_vector(data["delta_ln_mk"], "delta_ln_mk")
    delta_qtot = float(data["delta_ln_ntot"])
    if not np.isfinite(delta_qtot):
        raise ValueError("delta_ln_ntot must be finite.")
    if np.max(np.abs(delta_q)) > float(max_abs_delta_ln_nk):
        raise ValueError("delta_ln_nk exceeds max_abs_delta_ln_nk.")
    if np.max(np.abs(delta_r)) > float(max_abs_delta_ln_mk):
        raise ValueError("delta_ln_mk exceeds max_abs_delta_ln_mk.")
    if abs(delta_qtot) > float(max_abs_delta_ln_ntot):
        raise ValueError("delta_ln_ntot exceeds max_abs_delta_ln_ntot.")
    finite = bool(data["finite"] and np.all(np.isfinite(delta_q)) and np.all(np.isfinite(delta_r)))
    return JointOnlyPostSelectorCandidatePayload(
        payload_schema="exogibbs_joint_only_post_selector_candidate_payload_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        lambda_trial=float(data["lambda_trial"]),
        delta_ln_nk=tuple(float(value) for value in delta_q),
        delta_ln_mk=tuple(float(value) for value in delta_r),
        delta_ln_ntot=delta_qtot,
        joint_component_descent=True,
        finite=finite,
    )


def _component_merit(components: Mapping[str, float]) -> float:
    return float(
        components["element_balance_residual_norm"]
        + components["abs_ntot_residual"]
        + components["gas_stationarity_residual_norm"]
        + components["cond_stationarity_residual_norm"]
    )


def build_two_stage_flipped_budget_correction(
    *,
    ln_nk: Sequence[float],
    ln_mk: Sequence[float],
    ln_ntot: float,
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond: Sequence[Sequence[float]],
    element_budget: Sequence[float],
    hvector: Sequence[float],
    hvector_cond: Sequence[float],
    ln_normalized_pressure: float,
    epsilon: float,
    flipped_lambda_trial: float,
    flipped_delta_ln_nk: Sequence[float],
    flipped_delta_ln_mk: Sequence[float],
    flipped_delta_ln_ntot: float,
    budget_lambda_values: Sequence[float] = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1),
    explicit_opt_in: bool = False,
    field_provenance: Mapping[str, str] | None = None,
) -> TwoStageFlippedBudgetCorrectionReport:
    """Apply a flipped stationarity step followed by a budget-only correction."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for two-stage correction.")
    _validate_provenance(field_provenance)
    ag = _require_matrix(formula_matrix, "formula_matrix")
    ac = _require_matrix(formula_matrix_cond, "formula_matrix_cond")
    q0 = _require_vector(ln_nk, "ln_nk", ag.shape[1])
    r0 = _require_vector(ln_mk, "ln_mk", ac.shape[1])
    b = _require_vector(element_budget, "element_budget", ag.shape[0])
    hgas = _require_vector(hvector, "hvector", ag.shape[1])
    hcond = _require_vector(hvector_cond, "hvector_cond", ac.shape[1])
    dq_flip = _require_vector(flipped_delta_ln_nk, "flipped_delta_ln_nk", ag.shape[1])
    dr_flip = _require_vector(flipped_delta_ln_mk, "flipped_delta_ln_mk", ac.shape[1])
    if ac.shape[0] != ag.shape[0]:
        raise ValueError("formula_matrix and formula_matrix_cond row counts must match.")
    lambda_flip = float(flipped_lambda_trial)
    delta_qtot_flip = float(flipped_delta_ln_ntot)
    if not np.isfinite(lambda_flip) or not np.isfinite(delta_qtot_flip):
        raise ValueError("flipped lambda and delta_ln_ntot must be finite.")
    initial_full = _components(q0, r0, float(ln_ntot), ag, ac, b, hgas, hcond, float(ln_normalized_pressure), float(epsilon))
    q1 = q0 - lambda_flip * dq_flip
    r1 = r0 - lambda_flip * dr_flip
    qtot1 = float(ln_ntot) - lambda_flip * delta_qtot_flip
    flipped_full = _components(q1, r1, qtot1, ag, ac, b, hgas, hcond, float(ln_normalized_pressure), float(epsilon))
    correction = build_residual_balanced_coupled_rgie_direction(
        ln_nk=q1,
        ln_mk=r1,
        ln_ntot=qtot1,
        formula_matrix=ag,
        formula_matrix_cond=ac,
        element_budget=b,
        hvector=hgas,
        hvector_cond=hcond,
        ln_normalized_pressure=float(ln_normalized_pressure),
        epsilon=float(epsilon),
        alpha_budget_values=(1.0,),
        alpha_stationarity_values=(0.0,),
        lambda_values=budget_lambda_values,
        field_provenance=field_provenance,
    )
    if correction.selected_candidate is None:
        q2 = q1
        r2 = r1
        qtot2 = qtot1
        correction_status = "no_budget_correction_candidate"
        budget_lambda = None
    else:
        candidate = correction.selected_candidate
        budget_lambda = float(candidate.lambda_trial)
        q2 = q1 + budget_lambda * np.asarray(candidate.delta_ln_nk, dtype=np.float64)
        r2 = r1 + budget_lambda * np.asarray(candidate.delta_ln_mk, dtype=np.float64)
        qtot2 = qtot1 + budget_lambda * float(candidate.delta_ln_ntot)
        correction_status = "selected"
    final_full = _components(q2, r2, qtot2, ag, ac, b, hgas, hcond, float(ln_normalized_pressure), float(epsilon))
    initial = _component_dict(initial_full)
    after_flipped = _component_dict(flipped_full)
    final = _component_dict(final_full)
    deltas = {key: float(final[key] - initial[key]) for key in initial}
    all_nonworsening = bool(
        deltas["element_balance_residual_norm"] <= 1.0e-12
        and deltas["abs_ntot_residual"] <= 1.0e-12
        and deltas["gas_stationarity_residual_norm"] <= 1.0e-12
        and deltas["cond_stationarity_residual_norm"] <= 1.0e-12
    )
    initial_merit = _component_merit(initial)
    flipped_merit = _component_merit(after_flipped)
    final_merit = _component_merit(final)
    finite = bool(np.all(np.isfinite(q2)) and np.all(np.isfinite(r2)) and np.isfinite(qtot2))
    return TwoStageFlippedBudgetCorrectionReport(
        report_schema="exogibbs_two_stage_flipped_budget_correction_report_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        initial_components=initial,
        after_flipped_components=after_flipped,
        final_components=final,
        initial_component_merit=initial_merit,
        after_flipped_component_merit=flipped_merit,
        final_component_merit=final_merit,
        final_vs_initial_delta=final_merit - initial_merit,
        component_deltas=deltas,
        all_primary_components_nonworsening=all_nonworsening,
        accepted=bool(all_nonworsening and final_merit < initial_merit),
        budget_correction_status=correction_status,
        budget_lambda_trial=budget_lambda,
        final_ln_nk=tuple(float(value) for value in q2),
        final_ln_mk=tuple(float(value) for value in r2),
        final_ln_ntot=float(qtot2),
        finite=finite,
    )
