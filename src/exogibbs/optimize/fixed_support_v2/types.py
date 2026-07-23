"""Typed mathematical data for the fixed-support v2 solver.

The types in this module contain no solver policy.  They are immutable
``NamedTuple`` instances so that JAX treats every numerical field as a PyTree
leaf without custom flattening code.
"""

from __future__ import annotations

from enum import IntEnum, IntFlag
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, Tuple


Array = Any


class SolverMode(IntEnum):
    """Stable integer labels for the future v2 controller."""

    NORMAL = 0
    RESTORATION = 1
    CONVERGED = 2
    FAILED = 3


class TerminalStatus(IntEnum):
    """Primary terminal status labels reserved by the v2 contract."""

    NOT_TERMINATED = 0
    CONVERGED = 1
    NORMAL_LINE_SEARCH_FAILED = 2
    NORMAL_DUAL_STEP_FAILED = 3
    RESTORATION_FEASIBLE_BUT_UNACCEPTABLE = 4
    RESTORATION_LINEAR_SOLVE_FAILED = 5
    RESTORATION_LINE_SEARCH_FAILED = 6
    RESTORATION_MAX_ITER = 7
    RESTORATION_LOCALLY_INFEASIBLE = 8
    RESTORATION_NONFINITE = 9
    INTERNAL_CONTRACT_ERROR = 10
    NORMAL_LINEAR_SOLVE_FAILED = 11
    RESTORATION_RETURN_ACCEPTED = 12
    NORMAL_MAX_ITER = 13
    RESTORATION_MAX_CALLS = 14
    RETURN_REPRESENTATION_FLOOR_FAILED = 15
    SOC_LINEAR_SOLVE_FAILED = 16


class TrialRejectionReason(IntFlag):
    """Bit flags describing why one ordered normal trial was rejected."""

    NONE = 0
    NONFINITE = 1
    THETA_MAX = 2
    CURRENT_POINT = 4
    FILTER_HISTORY = 8


@dataclass(frozen=True)
class LinearSolverConfig:
    """Numerical policy for solving one reduced R-GIE system."""

    ruiz_iterations: int = 4
    iterative_refinement_steps: int = 2
    relative_residual_tolerance: float = 1.0e-10


@dataclass(frozen=True)
class NormalConfig:
    """Normal line-search policy."""

    backtracking_factor: float = 0.5
    stationarity_tolerance: float = 1.0e-8
    budget_tolerance: float = 1.0e-8
    complementarity_tolerance: float = 1.0e-8
    total_density_tolerance: float = 1.0e-8
    restoration_entry_theta_threshold: float = 1.0e-8


@dataclass(frozen=True)
class FilterConfig:
    """Persistent original-filter policy."""

    gamma_phi: float = 1.0e-8
    gamma_theta: float = 1.0e-5
    theta_max_factor: float = 1.0e4
    theta_min_factor: float = 1.0e-4
    eta_phi: float = 1.0e-8
    switching_delta: float = 1.0
    switching_s_phi: float = 2.3
    switching_s_theta: float = 1.1
    roundoff_tolerance_factor: float = 10.0
    reset_trigger: int = 5
    max_resets: int = 5


@dataclass(frozen=True)
class RestorationConfig:
    """Persistent elastic amount-space restoration policy."""

    elastic_penalty: float = 1.0e3
    proximity_weight: float = 1.0e-3
    amount_scale_floor_fraction: float = 1.0e-12
    interior_push_fraction: float = 1.0e-12
    fraction_to_boundary: float = 0.995
    backtracking_factor: float = 0.5
    armijo_fraction: float = 1.0e-4
    constraint_nonincrease_tolerance: float = 1.0e-12
    required_reduction: float = 0.1
    kkt_tolerance: float = 1.0e-8
    budget_tolerance: float = 1.0e-8
    total_density_tolerance: float = 1.0e-8
    relative_linear_solve_tolerance: float = 1.0e-10
    return_dual_fraction_to_boundary: float = 0.995
    bound_multiplier_reset_threshold: float = 1.0e3
    representation_floor: float = 1.0e-300
    representation_floor_injection_tolerance: float = 1.0e-12


@dataclass(frozen=True)
class SOCConfig:
    """Exact method-0 SOC policy after a rejected normal line search."""

    enabled: bool = True
    max_corrections: int = 4
    kappa_soc: float = 0.99
    fraction_to_boundary: float = 0.995


@dataclass(frozen=True)
class ContinuationConfig:
    """Barrier schedule owned by the outer continuation controller."""

    epsilon_schedule: Tuple[float, ...] = (-11.0, -13.0, -15.0, -17.0)
    initial_state_policy: Literal["center", "provided"] = "center"


@dataclass(frozen=True)
class SolverLimitConfig:
    """Fixed-shape allocation and iteration safeguards."""

    max_normal_iterations: int = 200
    max_line_search_trials: int = 20
    max_restoration_calls: int = 0
    max_restoration_iterations: int = 100
    max_restoration_line_search_trials: int = 20

    @property
    def filter_capacity(self) -> int:
        """Capacity required by the persistent-filter ownership contract."""

        return self.max_normal_iterations + self.max_restoration_calls + 1


@dataclass(frozen=True)
class FixedSupportV2Config:
    """V2 configuration grouped by the component that owns each policy."""

    normal: NormalConfig = field(default_factory=NormalConfig)
    linear_solver: LinearSolverConfig = field(default_factory=LinearSolverConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    restoration: RestorationConfig = field(default_factory=RestorationConfig)
    soc: SOCConfig = field(default_factory=SOCConfig)
    continuation: ContinuationConfig = field(default_factory=ContinuationConfig)
    limits: SolverLimitConfig = field(default_factory=SolverLimitConfig)


class FixedSupportProblem(NamedTuple):
    """Immutable mathematical data for one fixed-support solve.

    ``gamma`` is always ``hgas(T) + log(P / Pref)``.  In particular, it is
    independent of the current iterate.  ``budget_row_scale`` and
    ``total_density_row_scale`` are the fixed multipliers ``Wb`` and ``wt``
    used by the filter for the complete fixed-epsilon solve.
    """

    gas_formula_matrix: Array
    condensate_formula_matrix: Array
    target_inventory: Array
    gamma: Array
    condensate_standard_source: Array
    support_indices: Array
    budget_row_scale: Array
    total_density_row_scale: Array


class OriginalState(NamedTuple):
    """One iterate in the original log-coordinate KKT system."""

    q: Array
    r: Array
    lambda_: Array
    rho: Array
    qtot: Array
    epsilon: Array
    iteration: Array


class OriginalDirection(NamedTuple):
    """A tangent vector in original-state coordinates."""

    q: Array
    r: Array
    lambda_: Array
    rho: Array
    qtot: Array


class PhysicalAmounts(NamedTuple):
    """Positive physical amounts corresponding to log primal variables."""

    gas: Array
    condensate: Array
    total_gas: Array


class ResidualComponents(NamedTuple):
    """Canonical KKT residual blocks in mathematical equation order."""

    gas_stationarity: Array
    condensate_stationarity: Array
    budget: Array
    complementarity: Array
    total_density: Array


class FilterState(NamedTuple):
    """Fixed-capacity persistent filter state."""

    phi_entries: Array
    theta_entries: Array
    valid_entries: Array
    successive_filter_rejections: Array
    reset_count: Array


class LinearSolveDiagnostics(NamedTuple):
    """Audit quantities for the unregularized reduced solve."""

    raw_solution_finite: Array
    residual_norm: Array
    relative_residual: Array
    solution_norm: Array
    smallest_singular_value: Array
    largest_singular_value: Array
    condition_estimate: Array


class NormalDirectionResult(NamedTuple):
    """One normal Newton direction and its solve status."""

    direction: OriginalDirection
    diagnostics: LinearSolveDiagnostics
    status: Array


class NormalTrialBatch(NamedTuple):
    """Parallel values for one sequentially ordered alpha ladder."""

    states: OriginalState
    alphas: Array
    phi: Array
    theta: Array
    linearized_objective_change: Array
    finite: Array
    within_theta_max: Array
    current_acceptable: Array
    history_acceptable: Array
    f_type: Array
    armijo: Array
    accepted: Array
    rejection_reasons: Array


class NormalTrialSelection(NamedTuple):
    """Sequential interpretation of a parallel normal trial batch."""

    accepted: Array
    selected_index: Array
    selected_alpha: Array
    rejected_prefix: Array
    last_rejection_reason: Array
    status: Array


class NormalStepResult(NamedTuple):
    """Complete output of one non-mutating normal kernel call."""

    direction_result: NormalDirectionResult
    trials: NormalTrialBatch
    selection: NormalTrialSelection


class FilterUpdateResult(NamedTuple):
    """A filter update plus explicit fixed-capacity overflow reporting."""

    state: FilterState
    capacity_exhausted: Array


class RestorationState(NamedTuple):
    """Complete persistent state of one restoration NLP."""

    x: Array
    positive_slack: Array
    negative_slack: Array
    equality_dual: Array
    lower_bound_dual_x: Array
    lower_bound_dual_positive: Array
    lower_bound_dual_negative: Array
    restoration_mu: Array
    entry_x: Array
    entry_original_state: OriginalState
    entry_phi: Array
    entry_theta: Array
    variable_scales: Array
    row_scales: Array
    iteration: Array
    accepted_iteration_count: Array


class RestorationResiduals(NamedTuple):
    """Primal-dual KKT blocks for the elastic restoration NLP."""

    dual_x: Array
    dual_positive: Array
    dual_negative: Array
    equality: Array
    complementarity_x: Array
    complementarity_positive: Array
    complementarity_negative: Array


class RestorationDirection(NamedTuple):
    """Primal-dual Newton direction in restoration coordinates."""

    x: Array
    positive_slack: Array
    negative_slack: Array
    equality_dual: Array
    lower_bound_dual_x: Array
    lower_bound_dual_positive: Array
    lower_bound_dual_negative: Array


class RestorationDirectionDiagnostics(NamedTuple):
    """Schur-system and reconstructed KKT solve diagnostics."""

    raw_direction_finite: Array
    schur_residual_norm: Array
    relative_schur_residual: Array
    full_kkt_residual_norm: Array
    relative_full_kkt_residual: Array
    smallest_singular_value: Array
    largest_singular_value: Array
    condition_estimate: Array


class RestorationDirectionResult(NamedTuple):
    """One restoration direction and its typed status."""

    direction: RestorationDirection
    diagnostics: RestorationDirectionDiagnostics
    status: Array


class RestorationTrialBatch(NamedTuple):
    """Ordered restoration trial ladder evaluated in parallel."""

    states: RestorationState
    alphas: Array
    elastic_objective: Array
    barrier_objective: Array
    equality_violation: Array
    finite_and_positive: Array
    objective_acceptable: Array
    constraint_acceptable: Array
    accepted: Array


class RestorationIterationResult(NamedTuple):
    """Result of one persistent restoration iteration."""

    state: RestorationState
    direction_result: RestorationDirectionResult
    trials: RestorationTrialBatch
    accepted: Array
    selected_index: Array
    selected_alpha: Array
    status: Array


class RestorationSolveResult(NamedTuple):
    """Terminal or return-ready result of a standalone restoration solve."""

    state: RestorationState
    status: Array
    return_accepted: Array
    original_phi: Array
    original_theta: Array


class KKTComponentNorms(NamedTuple):
    """Independent original KKT component max norms."""

    gas_stationarity: Array
    condensate_stationarity: Array
    budget_scaled: Array
    complementarity: Array
    total_density_scaled: Array


class RestorationReturnDiagnostics(NamedTuple):
    """Audit information for one accepted restoration return map."""

    alpha_dual: Array
    bound_multiplier_reset: Array
    equality_multiplier_reset: Array
    representation_floor_applied: Array
    scaled_budget_injection_max: Array
    scaled_total_density_injection: Array
    pre_return_norms: KKTComponentNorms
    post_return_norms: KKTComponentNorms


class RestorationReturnResult(NamedTuple):
    """Original-state initializer produced by an accepted restoration exit."""

    original_state: OriginalState
    diagnostics: RestorationReturnDiagnostics
    accepted: Array
    status: Array


class SOCLinearizedResidualNorms(NamedTuple):
    """Five blockwise audits for the generic SOC Newton equation."""

    gas_stationarity: Array
    condensate_stationarity: Array
    budget: Array
    complementarity: Array
    total_density: Array


class SOCTrialBatch(NamedTuple):
    """Fixed-size sequential SOC recurrence diagnostics."""

    states: OriginalState
    attempted: Array
    alpha_test: Array
    alpha_soc: Array
    alpha_y: Array
    alpha_dual: Array
    budget_rhs: Array
    total_density_rhs: Array
    phi: Array
    theta: Array
    finite: Array
    current_acceptable: Array
    history_acceptable: Array
    f_type: Array
    armijo: Array
    accepted: Array
    rejection_reasons: Array
    kappa_continue: Array
    solve_statuses: Array
    linearized_residual_norms: SOCLinearizedResidualNorms


class SOCStepResult(NamedTuple):
    """Outcome of the ordered exact-SOC sequence for one normal failure."""

    eligible: Array
    accepted: Array
    base_trial_index: Array
    selected_index: Array
    selected_state: OriginalState
    correction_count: Array
    trials: SOCTrialBatch


class ControllerState(NamedTuple):
    """Fixed-epsilon NORMAL/RESTORATION controller state."""

    mode: Array
    original_state: OriginalState
    filter_state: FilterState
    restoration_state: RestorationState
    initial_theta: Array
    normal_iteration_count: Array
    restoration_call_count: Array
    restoration_accepted_iteration_count: Array
    soc_attempt_count: Array
    soc_accepted_count: Array
    terminal_status: Array
    last_return_diagnostics: RestorationReturnDiagnostics


class ContinuationState(NamedTuple):
    """Persistent outer state for one complete barrier schedule."""

    controller: ControllerState
    stage_index: Array
    completed_stage_count: Array
    stage_statuses: Array
    stage_normal_iteration_counts: Array
    stage_restoration_call_counts: Array
    stage_restoration_accepted_iteration_counts: Array
    stage_last_return_diagnostics: RestorationReturnDiagnostics
    stage_soc_attempt_counts: Array
    stage_soc_accepted_counts: Array
    terminal_status: Array


__all__ = [
    "Array",
    "FixedSupportProblem",
    "FixedSupportV2Config",
    "FilterConfig",
    "FilterState",
    "FilterUpdateResult",
    "ControllerState",
    "ContinuationState",
    "KKTComponentNorms",
    "LinearSolveDiagnostics",
    "LinearSolverConfig",
    "NormalConfig",
    "NormalDirectionResult",
    "NormalStepResult",
    "NormalTrialBatch",
    "NormalTrialSelection",
    "OriginalDirection",
    "OriginalState",
    "PhysicalAmounts",
    "ResidualComponents",
    "RestorationConfig",
    "ContinuationConfig",
    "SolverLimitConfig",
    "RestorationDirection",
    "RestorationDirectionDiagnostics",
    "RestorationDirectionResult",
    "RestorationIterationResult",
    "RestorationResiduals",
    "RestorationSolveResult",
    "RestorationState",
    "RestorationTrialBatch",
    "RestorationReturnDiagnostics",
    "RestorationReturnResult",
    "SOCConfig",
    "SOCLinearizedResidualNorms",
    "SOCStepResult",
    "SOCTrialBatch",
    "SolverMode",
    "TerminalStatus",
    "TrialRejectionReason",
]
