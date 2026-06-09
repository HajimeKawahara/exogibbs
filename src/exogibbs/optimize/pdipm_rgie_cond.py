"""Explicit experimental PD-IPM R-GIE condensate helpers.

This module is intentionally default-off. It does not call production solvers,
does not change production return signatures, and does not wire into presets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


FORBIDDEN_PROVENANCE = {
    "fastchem4_trace",
    "fastchem4_public",
    "fastchem4_runtime",
    "branch_replay",
    "reference_fit",
    "unknown_reference",
}


@dataclass(frozen=True)
class PdipmRgieCondensateState:
    """Explicit state for experimental coupled PD-IPM R-GIE trials."""

    state_schema: str
    ln_nk: tuple[float, ...]
    ln_mk: tuple[float, ...]
    element_potential: tuple[float, ...]
    ln_ntot: float
    rho: tuple[float, ...] | None
    eta: tuple[float, ...] | None
    diagnostic_only: bool
    default_off: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool
    field_provenance: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PdipmRgieRestrictedTrialReport:
    """Report for one explicit restricted PD-IPM R-GIE trial step."""

    report_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    trial_step_accepted: bool
    alpha: float
    initial_combined_residual_l2: float
    candidate_combined_residual_l2: float
    initial_budget_l2: float
    candidate_budget_l2: float
    initial_gas_stationarity_l2: float
    candidate_gas_stationarity_l2: float
    initial_condensate_stationarity_l2: float
    candidate_condensate_stationarity_l2: float
    initial_complementarity_l2: float | None
    candidate_complementarity_l2: float | None
    merit_component_weights: Mapping[str, float]
    linear_system_component_weights: Mapping[str, float]
    linear_system_row_scaling: str
    linear_system_row_scale_min: float
    linear_system_row_scale_max: float
    linear_system_budget_priority_policy: str
    linear_system_budget_priority: float
    linear_system_budget_priority_effective_weight: float
    linear_system_budget_priority_reference_norm: float
    linear_system_budget_priority_budget_norm: float
    require_budget_nonworsening: bool
    budget_rhs_sign: float
    max_gas_stationarity_worsening_ratio: float | None
    max_condensate_stationarity_worsening_ratio: float | None
    initial_merit_l2: float
    candidate_merit_l2: float
    delta_q: tuple[float, ...]
    delta_r: tuple[float, ...]
    delta_lambda: tuple[float, ...]
    delta_rho: tuple[float, ...] | None
    delta_q_l2: float
    delta_r_l2: float
    delta_lambda_l2: float
    delta_rho_l2: float | None
    finite_trial_step: bool
    initial_state: PdipmRgieCondensateState
    candidate_state: PdipmRgieCondensateState
    fastchem4_trace_public_runtime_constructor_inputs_used: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["initial_state"] = self.initial_state.as_dict()
        payload["candidate_state"] = self.candidate_state.as_dict()
        return payload


@dataclass(frozen=True)
class PdipmRgieReducedStepReport:
    """Report for the algorithm-v1.1 reduced coupled PD-IPM R-GIE step."""

    report_schema: str
    diagnostic_only: bool
    default_off: bool
    explicit_opt_in: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    equation_family: str
    qhat_condition_estimate: float
    qhat_regularization: float
    linear_system_residual_l2: float
    initial_budget_l2: float
    candidate_budget_l2: float
    initial_gas_stationarity_l2: float
    candidate_gas_stationarity_l2: float
    initial_condensate_stationarity_l2: float
    candidate_condensate_stationarity_l2: float
    initial_barrier_complementarity_l2: float
    candidate_barrier_complementarity_l2: float
    initial_total_density_l2: float
    candidate_total_density_l2: float
    initial_combined_residual_l2: float
    candidate_combined_residual_l2: float
    trial_step_accepted: bool
    alpha: float
    require_budget_nonworsening: bool
    delta_q: tuple[float, ...]
    delta_r: tuple[float, ...]
    delta_lambda: tuple[float, ...]
    delta_rho: tuple[float, ...]
    delta_qtot: float
    pi_vector: tuple[float, ...]
    j_vector: tuple[float, ...]
    t_vector: tuple[float, ...]
    delta_q_l2: float
    delta_r_l2: float
    delta_lambda_l2: float
    delta_rho_l2: float
    finite_trial_step: bool
    initial_state: PdipmRgieCondensateState
    candidate_state: PdipmRgieCondensateState
    fastchem4_trace_public_runtime_constructor_inputs_used: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["initial_state"] = self.initial_state.as_dict()
        payload["candidate_state"] = self.candidate_state.as_dict()
        return payload


@dataclass(frozen=True)
class PdipmRgieDualCarrierCallsiteInit:
    """Explicit experimental callsite carrier for PD-IPM R-GIE dual state."""

    carrier_schema: str
    state: PdipmRgieCondensateState
    support_indices: tuple[int, ...]
    support_amounts_init: tuple[float, ...]
    carries_ln_nk: bool
    carries_ln_mk: bool
    carries_ln_ntot: bool
    carries_element_potential: bool
    carries_rho: bool
    carries_eta: bool
    diagnostic_only: bool
    default_off: bool
    production_behavior_change: bool
    production_return_signature_change: bool
    preset_default_wiring_change: bool
    fastchem4_trace_public_runtime_constructor_inputs_used: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.as_dict()
        return payload


def _validate_provenance(field_provenance: Mapping[str, str] | None) -> Mapping[str, str]:
    provenance = {} if field_provenance is None else dict(field_provenance)
    forbidden = sorted(set(provenance.values()) & FORBIDDEN_PROVENANCE)
    if forbidden:
        raise ValueError(f"field_provenance contains forbidden values: {forbidden}")
    return provenance


def _as_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def _as_matrix(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def build_pdipm_rgie_condensate_state(
    *,
    ln_nk: Sequence[float],
    ln_mk: Sequence[float],
    element_potential: Sequence[float],
    ln_ntot: float | None = None,
    rho: Sequence[float] | None = None,
    eta: Sequence[float] | None = None,
    field_provenance: Mapping[str, str] | None = None,
) -> PdipmRgieCondensateState:
    """Build an explicit default-off PD-IPM R-GIE condensate state."""

    provenance = _validate_provenance(field_provenance)
    q = _as_vector(ln_nk, "ln_nk")
    r = _as_vector(ln_mk, "ln_mk")
    lam = _as_vector(element_potential, "element_potential")
    qtot = float(np.log(np.sum(np.exp(q)))) if ln_ntot is None else float(ln_ntot)
    if not np.isfinite(qtot):
        raise ValueError("ln_ntot must be finite.")
    rho_tuple: tuple[float, ...] | None = None
    eta_tuple: tuple[float, ...] | None = None
    if rho is not None:
        rho_array = _as_vector(rho, "rho")
        if rho_array.shape[0] != r.shape[0]:
            raise ValueError("rho length must match ln_mk length.")
        rho_tuple = tuple(float(value) for value in rho_array)
    if eta is not None:
        eta_array = _as_vector(eta, "eta")
        if eta_array.shape[0] != r.shape[0]:
            raise ValueError("eta length must match ln_mk length.")
        eta_tuple = tuple(float(value) for value in eta_array)
    return PdipmRgieCondensateState(
        state_schema="exogibbs_pdipm_rgie_condensate_state_v1",
        ln_nk=tuple(float(value) for value in q),
        ln_mk=tuple(float(value) for value in r),
        element_potential=tuple(float(value) for value in lam),
        ln_ntot=qtot,
        rho=rho_tuple,
        eta=eta_tuple,
        diagnostic_only=True,
        default_off=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
        field_provenance={
            "ln_nk": provenance.get("ln_nk", "exogibbs_native_or_experimental"),
            "ln_mk": provenance.get("ln_mk", "exogibbs_native_or_experimental"),
            "element_potential": provenance.get(
                "element_potential", "exogibbs_native_or_experimental"
            ),
            "rho": provenance.get("rho", "exogibbs_native_or_experimental"),
            "eta": provenance.get("eta", "exogibbs_native_or_experimental"),
        },
    )


def _residuals(
    *,
    formula_matrix: np.ndarray,
    formula_matrix_cond_active: np.ndarray,
    element_inventory_target: np.ndarray,
    mass_action_constants: np.ndarray,
    hvector_cond_active: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    lam: np.ndarray,
    rho: np.ndarray | None,
    barrier_parameter: float | None,
) -> dict[str, np.ndarray]:
    n = np.exp(q)
    m = np.exp(r)
    gas = q - mass_action_constants - formula_matrix.T @ lam
    cond = formula_matrix_cond_active.T @ lam - hvector_cond_active
    budget = element_inventory_target - formula_matrix @ n - formula_matrix_cond_active @ m
    residual_parts = [gas, cond, budget]
    complementarity = None
    if rho is not None and barrier_parameter is not None:
        complementarity = m * rho - float(barrier_parameter)
        residual_parts.append(complementarity)
    return {
        "gas": gas,
        "condensate": cond,
        "budget": budget,
        "complementarity": np.asarray([], dtype=np.float64)
        if complementarity is None
        else complementarity,
        "combined": np.concatenate(residual_parts),
    }


def _algorithm_v11_residuals(
    *,
    formula_matrix: np.ndarray,
    formula_matrix_cond_active: np.ndarray,
    element_inventory_target: np.ndarray,
    gas_stationarity_source: np.ndarray,
    condensate_standard_source: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    lam: np.ndarray,
    rho: np.ndarray,
    qtot: float,
    epsilon: float | Sequence[float],
    qtot_reference: float | None = None,
    condensate_residual_mask: Sequence[bool] | None = None,
) -> dict[str, np.ndarray]:
    n = np.exp(q)
    m = np.exp(r)
    eta = np.exp(rho)
    qtot_value = float(qtot)
    qtot_ref = qtot_value if qtot_reference is None else float(qtot_reference)
    gas = q + gas_stationarity_source + qtot_ref - qtot_value - formula_matrix.T @ lam
    condensate = condensate_standard_source - formula_matrix_cond_active.T @ lam - eta
    if condensate_residual_mask is not None:
        mask = np.asarray(condensate_residual_mask, dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != condensate.shape[0]:
            raise ValueError("condensate_residual_mask must match condensate length.")
        condensate_for_combined = condensate[mask]
    else:
        condensate_for_combined = condensate
    budget = formula_matrix @ n + formula_matrix_cond_active @ m - element_inventory_target
    epsilon_array = np.asarray(epsilon, dtype=np.float64)
    if epsilon_array.ndim == 0:
        epsilon_array = np.full_like(r, float(epsilon_array))
    complementarity = r + rho - epsilon_array
    total_density = np.asarray([np.sum(n) - np.exp(qtot_value)], dtype=np.float64)
    return {
        "gas": gas,
        "condensate": condensate,
        "budget": budget,
        "complementarity": complementarity,
        "total_density": total_density,
        "combined": np.concatenate(
            [gas, condensate_for_combined, budget, complementarity, total_density]
        ),
    }


def solve_pdipm_rgie_algorithm_v11_reduced_step(
    *,
    explicit_opt_in: bool,
    state: PdipmRgieCondensateState,
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond_active: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    gas_stationarity_source: Sequence[float],
    condensate_standard_source: Sequence[float],
    epsilon: float | Sequence[float],
    alpha_candidates: Sequence[float] = (1.0, 0.5, 0.25, 0.125, 0.0625),
    qhat_regularization: float = 0.0,
    max_abs_delta_q: float = 2.0,
    max_abs_delta_r: float = 2.0,
    max_abs_delta_rho: float = 2.0,
    max_abs_delta_lambda: float = 100.0,
    require_budget_nonworsening: bool = False,
    jacobian_mask: Sequence[bool] | None = None,
    paired_density_activity_update: bool = False,
    max_log_condensate_density: Sequence[float] | None = None,
) -> PdipmRgieReducedStepReport:
    """Solve one explicit algorithm-v1.1 reduced coupled PD-IPM R-GIE step.

    This implements the reduced equations documented in
    ``exogibbs_algorithm_v1.1.pdf`` for a fixed active condensate support:

    ``Qhat pi + delta_qtot * b_k =
    A_g(n_k * geff_k) + A_c(j_k * c_k + m_k * t_k - m_k) + delta_bhat_k``

    ``b_k dot pi + delta_ntot,k * delta_qtot =
    n_k dot geff_k - delta_ntot,k``

    with ``j_k = m_k / eta_k`` and ``t_k = r_k + rho_k - epsilon``. The
    implementation uses ``geff_k = q_k + gas_stationarity_source_k`` because
    the public source input excludes the current log-density term. The
    recovered updates are ``delta_q = A_g.T @ pi + delta_qtot - geff_k``,
    ``delta_rho = eta_k^-1 * (c_k - A_c.T @ pi) - 1``, and
    ``delta_r = -delta_rho - t_k``.
    """

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for PD-IPM reduced steps.")
    if not isinstance(state, PdipmRgieCondensateState):
        raise TypeError("state must be a PdipmRgieCondensateState.")
    _validate_provenance(state.field_provenance)
    if state.rho is None:
        raise ValueError("state.rho is required for algorithm-v1.1 PD-IPM steps.")
    if not alpha_candidates:
        raise ValueError("alpha_candidates must not be empty.")
    alphas = tuple(float(value) for value in alpha_candidates)
    if any(value <= 0.0 or value > 1.0 or not np.isfinite(value) for value in alphas):
        raise ValueError("alpha_candidates must be finite values in the interval (0, 1].")
    reg = float(qhat_regularization)
    if not np.isfinite(reg) or reg < 0.0:
        raise ValueError("qhat_regularization must be finite and non-negative.")

    ag = _as_matrix(formula_matrix, "formula_matrix")
    ac = _as_matrix(formula_matrix_cond_active, "formula_matrix_cond_active")
    target = _as_vector(element_inventory_target, "element_inventory_target")
    g = _as_vector(gas_stationarity_source, "gas_stationarity_source")
    c = _as_vector(condensate_standard_source, "condensate_standard_source")
    q = _as_vector(state.ln_nk, "state.ln_nk")
    r = _as_vector(state.ln_mk, "state.ln_mk")
    lam = _as_vector(state.element_potential, "state.element_potential")
    rho = _as_vector(state.rho, "state.rho")
    qtot = float(state.ln_ntot)
    if ag.shape[0] != ac.shape[0] or ag.shape[0] != target.shape[0]:
        raise ValueError("formula matrices and element_inventory_target row counts must match.")
    if lam.shape[0] != target.shape[0]:
        raise ValueError("element_potential length must match element rows.")
    if ag.shape[1] != q.shape[0] or g.shape[0] != q.shape[0]:
        raise ValueError("gas vectors must match formula_matrix columns.")
    if ac.shape[1] != r.shape[0] or c.shape[0] != r.shape[0] or rho.shape[0] != r.shape[0]:
        raise ValueError("condensate vectors must match formula_matrix_cond_active columns.")
    eps = np.asarray(epsilon, dtype=np.float64)
    if eps.ndim == 0:
        eps = np.full_like(r, float(eps))
    if eps.ndim != 1 or eps.shape[0] != r.shape[0]:
        raise ValueError("epsilon must be scalar or match condensate vector length.")
    if not np.all(np.isfinite(eps)):
        raise ValueError("epsilon must be finite.")
    if jacobian_mask is None:
        jac_mask = np.ones_like(r, dtype=bool)
    else:
        jac_mask = np.asarray(jacobian_mask, dtype=bool)
        if jac_mask.ndim != 1 or jac_mask.shape[0] != r.shape[0]:
            raise ValueError("jacobian_mask must match condensate vector length.")
        if not np.any(jac_mask) and jac_mask.shape[0]:
            jac_mask[int(np.argmax(c - ac.T @ lam))] = True
    if max_log_condensate_density is None:
        log_m_cap = None
    else:
        log_m_cap = np.asarray(max_log_condensate_density, dtype=np.float64)
        if log_m_cap.ndim != 1 or log_m_cap.shape[0] != r.shape[0]:
            raise ValueError("max_log_condensate_density must match condensate vector length.")

    n = np.exp(q)
    m = np.exp(r)
    eta = np.exp(rho)
    j_vec = m / np.maximum(eta, 1.0e-300)
    t_vec = r + rho - eps
    geff = q + g
    gas_inventory = ag @ n
    delta_bhat = target - gas_inventory - ac @ m
    delta_ntot = float(np.sum(n) - np.exp(qtot))
    qhat = ag @ (n[:, np.newaxis] * ag.T) + ac @ (j_vec[:, np.newaxis] * ac.T)
    if reg:
        qhat = qhat + reg * np.eye(qhat.shape[0], dtype=np.float64)
    rhs_top = ag @ (n * geff) + ac @ (j_vec * c + m * t_vec - m) + delta_bhat
    rhs_bottom = float(np.dot(n, geff) - delta_ntot)
    reduced_matrix = np.block(
        [
            [qhat, gas_inventory[:, np.newaxis]],
            [gas_inventory[np.newaxis, :], np.asarray([[delta_ntot]], dtype=np.float64)],
        ]
    )
    reduced_rhs = np.concatenate([rhs_top, np.asarray([rhs_bottom], dtype=np.float64)])
    try:
        reduced_solution = np.linalg.lstsq(reduced_matrix, reduced_rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        reduced_solution = np.zeros((target.shape[0] + 1,), dtype=np.float64)
    reduced_solution = np.nan_to_num(
        reduced_solution,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    pi = reduced_solution[:-1]
    delta_qtot = float(reduced_solution[-1])
    raw_delta_q = ag.T @ pi + delta_qtot - geff
    raw_delta_rho = (c - ac.T @ pi) / np.maximum(eta, 1.0e-300) - 1.0
    raw_delta_r = -raw_delta_rho - t_vec
    delta_q = np.clip(raw_delta_q, -float(max_abs_delta_q), float(max_abs_delta_q))
    if paired_density_activity_update:
        delta_r = np.clip(raw_delta_r, -float(max_abs_delta_r), float(max_abs_delta_r))
        if log_m_cap is not None:
            delta_r = np.minimum(delta_r, log_m_cap - r)
        target_delta_rho = eps - rho - r - delta_r
        delta_rho = np.clip(
            target_delta_rho,
            -float(max_abs_delta_rho),
            float(max_abs_delta_rho),
        )
    else:
        delta_r = np.clip(raw_delta_r, -float(max_abs_delta_r), float(max_abs_delta_r))
        delta_rho = np.clip(
            raw_delta_rho,
            -float(max_abs_delta_rho),
            float(max_abs_delta_rho),
        )
    delta_lambda = np.clip(pi - lam, -float(max_abs_delta_lambda), float(max_abs_delta_lambda))

    initial = _algorithm_v11_residuals(
        formula_matrix=ag,
        formula_matrix_cond_active=ac,
        element_inventory_target=target,
        gas_stationarity_source=g,
        condensate_standard_source=c,
        q=q,
        r=r,
        lam=lam,
        rho=rho,
        qtot=qtot,
        epsilon=eps,
        qtot_reference=qtot,
        condensate_residual_mask=jac_mask,
    )
    initial_combined = float(np.linalg.norm(initial["combined"]))
    initial_budget = float(np.linalg.norm(initial["budget"]))
    best_alpha = 0.0
    best_q = q
    best_r = r
    best_lam = lam
    best_rho = rho
    best_qtot = qtot
    best_residuals = initial
    best_combined = initial_combined
    best_fallback_alpha = 0.0
    best_fallback_q = q
    best_fallback_r = r
    best_fallback_lam = lam
    best_fallback_rho = rho
    best_fallback_qtot = qtot
    best_fallback_residuals = initial
    best_fallback_merit = float("inf")
    fallback_accepted = False
    accepted = False
    initial_condensate_accept = float(np.linalg.norm(initial["condensate"][jac_mask]))
    initial_complementarity_accept = float(np.linalg.norm(initial["complementarity"]))
    finite_step = bool(
        np.all(np.isfinite(delta_q))
        and np.all(np.isfinite(delta_r))
        and np.all(np.isfinite(delta_lambda))
        and np.all(np.isfinite(delta_rho))
        and np.isfinite(delta_qtot)
    )
    for alpha in alphas:
        candidate_q = q + float(alpha) * delta_q
        candidate_r = r + float(alpha) * delta_r
        if log_m_cap is not None:
            candidate_r = np.minimum(candidate_r, log_m_cap)
        candidate_lam = lam + float(alpha) * delta_lambda
        candidate_rho = rho + float(alpha) * delta_rho
        candidate_qtot = qtot + float(alpha) * delta_qtot
        candidate_residuals = _algorithm_v11_residuals(
            formula_matrix=ag,
            formula_matrix_cond_active=ac,
            element_inventory_target=target,
            gas_stationarity_source=g,
            condensate_standard_source=c,
            q=candidate_q,
            r=candidate_r,
            lam=candidate_lam,
            rho=candidate_rho,
            qtot=candidate_qtot,
            epsilon=eps,
            qtot_reference=qtot,
            condensate_residual_mask=jac_mask,
        )
        candidate_combined = float(np.linalg.norm(candidate_residuals["combined"]))
        candidate_budget = float(np.linalg.norm(candidate_residuals["budget"]))
        candidate_condensate_accept = float(
            np.linalg.norm(candidate_residuals["condensate"][jac_mask])
        )
        candidate_complementarity_accept = float(
            np.linalg.norm(candidate_residuals["complementarity"])
        )
        fallback_merit = candidate_complementarity_accept
        if (
            paired_density_activity_update
            and finite_step
            and np.isfinite(fallback_merit)
            and candidate_complementarity_accept < initial_complementarity_accept
            and np.isfinite(candidate_combined)
            and candidate_combined <= 1.25 * max(initial_combined, 1.0)
            and fallback_merit < best_fallback_merit
        ):
            best_fallback_alpha = float(alpha)
            best_fallback_q = candidate_q
            best_fallback_r = candidate_r
            best_fallback_lam = candidate_lam
            best_fallback_rho = candidate_rho
            best_fallback_qtot = candidate_qtot
            best_fallback_residuals = candidate_residuals
            best_fallback_merit = fallback_merit
            fallback_accepted = True
        if (
            finite_step
            and np.isfinite(candidate_combined)
            and (not require_budget_nonworsening or candidate_budget <= initial_budget + 1.0e-15)
            and candidate_combined < best_combined
        ):
            best_alpha = float(alpha)
            best_q = candidate_q
            best_r = candidate_r
            best_lam = candidate_lam
            best_rho = candidate_rho
            best_qtot = candidate_qtot
            best_residuals = candidate_residuals
            best_combined = candidate_combined
            accepted = True
            break
    if not accepted and fallback_accepted:
        best_alpha = best_fallback_alpha
        best_q = best_fallback_q
        best_r = best_fallback_r
        best_lam = best_fallback_lam
        best_rho = best_fallback_rho
        best_qtot = best_fallback_qtot
        best_residuals = best_fallback_residuals
        best_combined = float(np.linalg.norm(best_residuals["combined"]))
        accepted = True

    candidate_state = build_pdipm_rgie_condensate_state(
        ln_nk=best_q,
        ln_mk=best_r,
        element_potential=best_lam,
        ln_ntot=best_qtot,
        rho=best_rho,
        eta=np.exp(best_rho),
        field_provenance=state.field_provenance,
    )
    residual_vector = reduced_matrix @ reduced_solution - reduced_rhs
    qhat_cond = (
        float(np.linalg.cond(qhat))
        if qhat.size and np.all(np.isfinite(qhat))
        else float("inf")
    )
    return PdipmRgieReducedStepReport(
        report_schema="exogibbs_pdipm_rgie_algorithm_v11_reduced_step_report_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        equation_family="exogibbs_algorithm_v1_1_pdipm_reduced_rgie",
        qhat_condition_estimate=qhat_cond,
        qhat_regularization=reg,
        linear_system_residual_l2=float(np.linalg.norm(residual_vector)),
        initial_budget_l2=float(np.linalg.norm(initial["budget"])),
        candidate_budget_l2=float(np.linalg.norm(best_residuals["budget"])),
        initial_gas_stationarity_l2=float(np.linalg.norm(initial["gas"])),
        candidate_gas_stationarity_l2=float(np.linalg.norm(best_residuals["gas"])),
        initial_condensate_stationarity_l2=float(np.linalg.norm(initial["condensate"])),
        candidate_condensate_stationarity_l2=float(
            np.linalg.norm(best_residuals["condensate"])
        ),
        initial_barrier_complementarity_l2=float(np.linalg.norm(initial["complementarity"])),
        candidate_barrier_complementarity_l2=float(
            np.linalg.norm(best_residuals["complementarity"])
        ),
        initial_total_density_l2=float(np.linalg.norm(initial["total_density"])),
        candidate_total_density_l2=float(np.linalg.norm(best_residuals["total_density"])),
        initial_combined_residual_l2=initial_combined,
        candidate_combined_residual_l2=float(best_combined),
        trial_step_accepted=accepted,
        alpha=float(best_alpha),
        require_budget_nonworsening=bool(require_budget_nonworsening),
        delta_q=tuple(float(value) for value in delta_q),
        delta_r=tuple(float(value) for value in delta_r),
        delta_lambda=tuple(float(value) for value in delta_lambda),
        delta_rho=tuple(float(value) for value in delta_rho),
        delta_qtot=float(delta_qtot),
        pi_vector=tuple(float(value) for value in pi),
        j_vector=tuple(float(value) for value in j_vec),
        t_vector=tuple(float(value) for value in t_vec),
        delta_q_l2=float(np.linalg.norm(delta_q)),
        delta_r_l2=float(np.linalg.norm(delta_r)),
        delta_lambda_l2=float(np.linalg.norm(delta_lambda)),
        delta_rho_l2=float(np.linalg.norm(delta_rho)),
        finite_trial_step=finite_step,
        initial_state=state,
        candidate_state=candidate_state,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
    )


def _validate_merit_weights(
    merit_component_weights: Mapping[str, float] | None,
) -> dict[str, float]:
    weights = {
        "gas": 1.0,
        "condensate": 1.0,
        "budget": 1.0,
        "complementarity": 1.0,
    }
    if merit_component_weights is None:
        return weights
    for key, value in merit_component_weights.items():
        if key not in weights:
            raise ValueError(f"unknown merit component weight: {key}")
        weight = float(value)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("merit component weights must be finite and non-negative.")
        weights[key] = weight
    return weights


def _weighted_component_merit(
    residuals: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> float:
    parts = [
        weights["gas"] * float(np.linalg.norm(residuals["gas"])),
        weights["condensate"] * float(np.linalg.norm(residuals["condensate"])),
        weights["budget"] * float(np.linalg.norm(residuals["budget"])),
    ]
    if residuals["complementarity"].size:
        parts.append(
            weights["complementarity"] * float(np.linalg.norm(residuals["complementarity"]))
        )
    return float(np.linalg.norm(np.asarray(parts, dtype=np.float64)))


def _apply_linear_system_row_scaling(
    system: np.ndarray,
    rhs: np.ndarray,
    policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if policy == "none":
        scales = np.ones((system.shape[0],), dtype=np.float64)
        return system, rhs, scales
    if policy not in {"fastchem4_row_max_abs", "row_max_abs_floor_1"}:
        raise ValueError(
            "linear_system_row_scaling must be 'none', 'fastchem4_row_max_abs', "
            "or 'row_max_abs_floor_1'."
        )
    scales = np.max(np.abs(system), axis=1)
    scales = np.where(scales == 0.0, 1.0, scales)
    if policy == "row_max_abs_floor_1":
        scales = np.maximum(scales, 1.0)
    return system / scales[:, np.newaxis], rhs / scales, scales


def _row_scaled_block_norm(block: np.ndarray, policy: str) -> float:
    rhs = np.zeros((block.shape[0],), dtype=np.float64)
    scaled, _, _ = _apply_linear_system_row_scaling(block, rhs, policy)
    return float(np.linalg.norm(scaled))


def _budget_priority_multiplier(
    *,
    gas_block: np.ndarray,
    cond_block: np.ndarray,
    budget_block: np.ndarray,
    comp_block: np.ndarray | None,
    row_scaling_policy: str,
    budget_priority_policy: str,
    budget_priority: float,
) -> tuple[float, float, float]:
    if budget_priority_policy == "none":
        return 1.0, 0.0, 0.0
    if not np.isfinite(budget_priority) or budget_priority <= 0.0:
        raise ValueError("linear_system_budget_priority must be finite and positive.")
    if budget_priority_policy not in {
        "match_stationarity_block_norm",
        "match_combined_block_norm",
        "budget_priority_normalized",
    }:
        raise ValueError(
            "linear_system_budget_priority_policy must be 'none', "
            "'match_stationarity_block_norm', 'match_combined_block_norm', "
            "or 'budget_priority_normalized'."
        )
    gas_norm = _row_scaled_block_norm(gas_block, row_scaling_policy)
    cond_norm = _row_scaled_block_norm(cond_block, row_scaling_policy)
    budget_norm = _row_scaled_block_norm(budget_block, row_scaling_policy)
    if budget_priority_policy == "match_combined_block_norm":
        norms = [gas_norm, cond_norm]
        if comp_block is not None:
            norms.append(_row_scaled_block_norm(comp_block, row_scaling_policy))
        reference_norm = float(np.linalg.norm(np.asarray(norms, dtype=np.float64)))
    else:
        reference_norm = max(gas_norm, cond_norm)
    multiplier = float(budget_priority) * reference_norm / max(budget_norm, 1.0e-300)
    return multiplier, reference_norm, budget_norm


def propose_pdipm_rgie_restricted_trial_step(
    *,
    explicit_opt_in: bool,
    state: PdipmRgieCondensateState,
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond_active: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    mass_action_constants: Sequence[float],
    hvector_cond_active: Sequence[float],
    barrier_parameter: float | None = None,
    alpha_candidates: Sequence[float] = (1.0, 0.5, 0.25, 0.125, 0.0625),
    max_abs_delta_q: float = 2.0,
    max_abs_delta_r: float = 2.0,
    max_abs_delta_lambda: float = 100.0,
    merit_component_weights: Mapping[str, float] | None = None,
    linear_system_component_weights: Mapping[str, float] | None = None,
    linear_system_row_scaling: str = "none",
    linear_system_budget_priority_policy: str = "none",
    linear_system_budget_priority: float = 1.0,
    require_budget_nonworsening: bool = False,
    budget_rhs_sign: float = 1.0,
    max_gas_stationarity_worsening_ratio: float | None = None,
    max_condensate_stationarity_worsening_ratio: float | None = None,
) -> PdipmRgieRestrictedTrialReport:
    """Propose one explicit restricted PD-IPM R-GIE trial step."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for PD-IPM R-GIE trial steps.")
    if not isinstance(state, PdipmRgieCondensateState):
        raise TypeError("state must be a PdipmRgieCondensateState.")
    _validate_provenance(state.field_provenance)
    if not alpha_candidates:
        raise ValueError("alpha_candidates must not be empty.")
    alphas = tuple(float(value) for value in alpha_candidates)
    if any(value <= 0.0 or value > 1.0 or not np.isfinite(value) for value in alphas):
        raise ValueError("alpha_candidates must be finite values in the interval (0, 1].")
    if (
        max_gas_stationarity_worsening_ratio is not None
        and float(max_gas_stationarity_worsening_ratio) < 1.0
    ):
        raise ValueError("max_gas_stationarity_worsening_ratio must be at least 1.0.")
    if (
        max_condensate_stationarity_worsening_ratio is not None
        and float(max_condensate_stationarity_worsening_ratio) < 1.0
    ):
        raise ValueError("max_condensate_stationarity_worsening_ratio must be at least 1.0.")

    ag = _as_matrix(formula_matrix, "formula_matrix")
    ac = _as_matrix(formula_matrix_cond_active, "formula_matrix_cond_active")
    q = _as_vector(state.ln_nk, "state.ln_nk")
    r = _as_vector(state.ln_mk, "state.ln_mk")
    lam = _as_vector(state.element_potential, "state.element_potential")
    target = _as_vector(element_inventory_target, "element_inventory_target")
    mac = _as_vector(mass_action_constants, "mass_action_constants")
    hcond = _as_vector(hvector_cond_active, "hvector_cond_active")
    if ag.shape[0] != ac.shape[0] or ag.shape[0] != lam.shape[0]:
        raise ValueError("formula matrix element rows must match element_potential length.")
    if target.shape[0] != lam.shape[0]:
        raise ValueError("element_inventory_target length must match element rows.")
    if ag.shape[1] != q.shape[0] or mac.shape[0] != q.shape[0]:
        raise ValueError("gas vector lengths must match formula_matrix columns.")
    if ac.shape[1] != r.shape[0] or hcond.shape[0] != r.shape[0]:
        raise ValueError("active condensate lengths must match ln_mk.")
    rho = None if state.rho is None else _as_vector(state.rho, "state.rho")
    if barrier_parameter is not None and barrier_parameter <= 0.0:
        raise ValueError("barrier_parameter must be positive when provided.")

    initial = _residuals(
        formula_matrix=ag,
        formula_matrix_cond_active=ac,
        element_inventory_target=target,
        mass_action_constants=mac,
        hvector_cond_active=hcond,
        q=q,
        r=r,
        lam=lam,
        rho=rho,
        barrier_parameter=barrier_parameter,
    )
    nq = q.shape[0]
    nr = r.shape[0]
    nelement = lam.shape[0]
    nvariable = nq + nr + nelement
    gas_block = np.zeros((nq, nvariable), dtype=np.float64)
    gas_block[:, :nq] = np.eye(nq)
    gas_block[:, nq + nr :] = -ag.T
    cond_block = np.zeros((nr, nvariable), dtype=np.float64)
    cond_block[:, nq + nr :] = ac.T
    n = np.exp(q)
    m = np.exp(r)
    budget_block = np.zeros((nelement, nvariable), dtype=np.float64)
    budget_block[:, :nq] = ag * n[np.newaxis, :]
    budget_block[:, nq : nq + nr] = ac * m[np.newaxis, :]
    system_parts = [gas_block, cond_block, budget_block]
    rhs_parts = [-initial["gas"], -initial["condensate"], initial["budget"]]
    comp_block = None
    if rho is not None and barrier_parameter is not None:
        comp_block = np.zeros((nr, nvariable), dtype=np.float64)
        comp_block[:, nq : nq + nr] = np.diag(m * rho)
        system_parts.append(comp_block)
        rhs_parts.append(-initial["complementarity"])
    budget_sign = float(budget_rhs_sign)
    if budget_sign not in (-1.0, 1.0):
        raise ValueError("budget_rhs_sign must be either 1.0 or -1.0.")
    weights = _validate_merit_weights(merit_component_weights)
    linear_weights = _validate_merit_weights(linear_system_component_weights)
    gas_block *= linear_weights["gas"]
    cond_block *= linear_weights["condensate"]
    budget_block *= linear_weights["budget"]
    rhs_parts[0] *= linear_weights["gas"]
    rhs_parts[1] *= linear_weights["condensate"]
    rhs_parts[2] *= linear_weights["budget"]
    if rho is not None and barrier_parameter is not None:
        system_parts[-1] *= linear_weights["complementarity"]
        rhs_parts[-1] *= linear_weights["complementarity"]
    budget_priority_multiplier, budget_priority_reference_norm, budget_priority_budget_norm = (
        _budget_priority_multiplier(
            gas_block=gas_block,
            cond_block=cond_block,
            budget_block=budget_block,
            comp_block=comp_block,
            row_scaling_policy=linear_system_row_scaling,
            budget_priority_policy=linear_system_budget_priority_policy,
            budget_priority=float(linear_system_budget_priority),
        )
    )
    budget_block *= budget_priority_multiplier
    rhs_parts[2] *= budget_priority_multiplier
    system = np.vstack(system_parts)
    rhs_parts[2] = budget_sign * initial["budget"]
    rhs_parts[2] *= linear_weights["budget"]
    rhs_parts[2] *= budget_priority_multiplier
    rhs = np.concatenate(rhs_parts)
    system, rhs, row_scales = _apply_linear_system_row_scaling(
        system,
        rhs,
        linear_system_row_scaling,
    )
    try:
        step = np.linalg.lstsq(system, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        step = np.zeros((nvariable,), dtype=np.float64)
    step = np.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)
    delta_q = np.clip(step[:nq], -float(max_abs_delta_q), float(max_abs_delta_q))
    delta_r = np.clip(step[nq : nq + nr], -float(max_abs_delta_r), float(max_abs_delta_r))
    delta_lambda = np.clip(
        step[nq + nr :],
        -float(max_abs_delta_lambda),
        float(max_abs_delta_lambda),
    )
    delta_rho = None
    if rho is not None and barrier_parameter is not None:
        rho_target = float(barrier_parameter) / np.maximum(m, 1.0e-300)
        delta_rho = rho_target - rho

    best_alpha = 0.0
    best_q = q
    best_r = r
    best_lam = lam
    best_rho = rho
    best_residuals = initial
    best_combined = float(np.linalg.norm(initial["combined"]))
    initial_merit = _weighted_component_merit(initial, weights)
    best_merit = initial_merit
    initial_budget_l2 = float(np.linalg.norm(initial["budget"]))
    initial_gas_l2 = float(np.linalg.norm(initial["gas"]))
    initial_cond_l2 = float(np.linalg.norm(initial["condensate"]))
    accepted = False
    finite_step = bool(
        np.all(np.isfinite(delta_q))
        and np.all(np.isfinite(delta_r))
        and np.all(np.isfinite(delta_lambda))
        and (delta_rho is None or np.all(np.isfinite(delta_rho)))
    )
    for alpha in alphas:
        candidate_q = q + float(alpha) * delta_q
        candidate_r = r + float(alpha) * delta_r
        candidate_lam = lam + float(alpha) * delta_lambda
        candidate_rho = None if rho is None or delta_rho is None else rho + float(alpha) * delta_rho
        candidate_residuals = _residuals(
            formula_matrix=ag,
            formula_matrix_cond_active=ac,
            element_inventory_target=target,
            mass_action_constants=mac,
            hvector_cond_active=hcond,
            q=candidate_q,
            r=candidate_r,
            lam=candidate_lam,
            rho=candidate_rho,
            barrier_parameter=barrier_parameter,
        )
        candidate_combined = float(np.linalg.norm(candidate_residuals["combined"]))
        candidate_merit = _weighted_component_merit(candidate_residuals, weights)
        candidate_budget_l2 = float(np.linalg.norm(candidate_residuals["budget"]))
        budget_allowed = (
            (not require_budget_nonworsening)
            or candidate_budget_l2 <= initial_budget_l2 + 1.0e-15
        )
        gas_allowed = (
            max_gas_stationarity_worsening_ratio is None
            or float(np.linalg.norm(candidate_residuals["gas"]))
            <= float(max_gas_stationarity_worsening_ratio) * max(initial_gas_l2, 1.0e-300)
        )
        cond_allowed = (
            max_condensate_stationarity_worsening_ratio is None
            or float(np.linalg.norm(candidate_residuals["condensate"]))
            <= float(max_condensate_stationarity_worsening_ratio)
            * max(initial_cond_l2, 1.0e-300)
        )
        if (
            finite_step
            and np.all(np.isfinite(candidate_q))
            and np.all(np.isfinite(candidate_r))
            and np.all(np.isfinite(candidate_lam))
            and (candidate_rho is None or np.all(np.isfinite(candidate_rho)))
            and np.isfinite(candidate_combined)
            and np.isfinite(candidate_merit)
            and budget_allowed
            and gas_allowed
            and cond_allowed
            and candidate_merit < best_merit
        ):
            best_alpha = float(alpha)
            best_q = candidate_q
            best_r = candidate_r
            best_lam = candidate_lam
            best_rho = candidate_rho
            best_residuals = candidate_residuals
            best_combined = candidate_combined
            best_merit = candidate_merit
            accepted = True
            break

    candidate_state = build_pdipm_rgie_condensate_state(
        ln_nk=best_q,
        ln_mk=best_r,
        element_potential=best_lam,
        ln_ntot=float(np.log(np.sum(np.exp(best_q)))),
        rho=None if best_rho is None else best_rho,
        eta=state.eta,
        field_provenance=state.field_provenance,
    )
    return PdipmRgieRestrictedTrialReport(
        report_schema="exogibbs_pdipm_rgie_restricted_trial_report_v1",
        diagnostic_only=True,
        default_off=True,
        explicit_opt_in=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        trial_step_accepted=accepted,
        alpha=float(best_alpha),
        initial_combined_residual_l2=float(np.linalg.norm(initial["combined"])),
        candidate_combined_residual_l2=float(best_combined),
        initial_budget_l2=float(np.linalg.norm(initial["budget"])),
        candidate_budget_l2=float(np.linalg.norm(best_residuals["budget"])),
        initial_gas_stationarity_l2=float(np.linalg.norm(initial["gas"])),
        candidate_gas_stationarity_l2=float(np.linalg.norm(best_residuals["gas"])),
        initial_condensate_stationarity_l2=float(np.linalg.norm(initial["condensate"])),
        candidate_condensate_stationarity_l2=float(np.linalg.norm(best_residuals["condensate"])),
        initial_complementarity_l2=(
            None
            if initial["complementarity"].size == 0
            else float(np.linalg.norm(initial["complementarity"]))
        ),
        candidate_complementarity_l2=(
            None
            if best_residuals["complementarity"].size == 0
            else float(np.linalg.norm(best_residuals["complementarity"]))
        ),
        merit_component_weights={key: float(value) for key, value in weights.items()},
        linear_system_component_weights={
            key: float(value) for key, value in linear_weights.items()
        },
        linear_system_row_scaling=str(linear_system_row_scaling),
        linear_system_row_scale_min=float(np.min(row_scales)),
        linear_system_row_scale_max=float(np.max(row_scales)),
        linear_system_budget_priority_policy=str(linear_system_budget_priority_policy),
        linear_system_budget_priority=float(linear_system_budget_priority),
        linear_system_budget_priority_effective_weight=float(budget_priority_multiplier),
        linear_system_budget_priority_reference_norm=float(budget_priority_reference_norm),
        linear_system_budget_priority_budget_norm=float(budget_priority_budget_norm),
        require_budget_nonworsening=bool(require_budget_nonworsening),
        budget_rhs_sign=float(budget_sign),
        max_gas_stationarity_worsening_ratio=(
            None
            if max_gas_stationarity_worsening_ratio is None
            else float(max_gas_stationarity_worsening_ratio)
        ),
        max_condensate_stationarity_worsening_ratio=(
            None
            if max_condensate_stationarity_worsening_ratio is None
            else float(max_condensate_stationarity_worsening_ratio)
        ),
        initial_merit_l2=float(initial_merit),
        candidate_merit_l2=float(best_merit),
        delta_q=tuple(float(value) for value in delta_q),
        delta_r=tuple(float(value) for value in delta_r),
        delta_lambda=tuple(float(value) for value in delta_lambda),
        delta_rho=None if delta_rho is None else tuple(float(value) for value in delta_rho),
        delta_q_l2=float(np.linalg.norm(delta_q)),
        delta_r_l2=float(np.linalg.norm(delta_r)),
        delta_lambda_l2=float(np.linalg.norm(delta_lambda)),
        delta_rho_l2=None if delta_rho is None else float(np.linalg.norm(delta_rho)),
        finite_trial_step=finite_step,
        initial_state=state,
        candidate_state=candidate_state,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
    )


def build_pdipm_rgie_dual_carrier_callsite_init(
    *,
    explicit_opt_in: bool,
    state: PdipmRgieCondensateState,
    support_indices: Sequence[int],
    support_amounts_init: Sequence[float] | None = None,
) -> PdipmRgieDualCarrierCallsiteInit:
    """Build an explicit experimental callsite carrier that preserves duals."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for PD-IPM dual carriers.")
    if not isinstance(state, PdipmRgieCondensateState):
        raise TypeError("state must be a PdipmRgieCondensateState.")
    _validate_provenance(state.field_provenance)
    indices = tuple(int(index) for index in support_indices)
    if len(indices) != len(state.ln_mk):
        raise ValueError("support_indices length must match ln_mk length.")
    if len(set(indices)) != len(indices):
        raise ValueError("support_indices must not contain duplicates.")
    if any(index < 0 for index in indices):
        raise ValueError("support_indices must be non-negative.")
    if support_amounts_init is None:
        amounts = tuple(float(value) for value in np.exp(np.asarray(state.ln_mk, dtype=np.float64)))
    else:
        raw_amounts = _as_vector(support_amounts_init, "support_amounts_init")
        if raw_amounts.shape[0] != len(indices):
            raise ValueError("support_amounts_init length must match support_indices length.")
        amounts = tuple(float(value) for value in raw_amounts)
    if any((not np.isfinite(amount)) or amount <= 0.0 for amount in amounts):
        raise ValueError("support_amounts_init must be finite and positive.")
    return PdipmRgieDualCarrierCallsiteInit(
        carrier_schema="exogibbs_pdipm_rgie_dual_carrier_callsite_init_v1",
        state=state,
        support_indices=indices,
        support_amounts_init=amounts,
        carries_ln_nk=True,
        carries_ln_mk=True,
        carries_ln_ntot=True,
        carries_element_potential=True,
        carries_rho=state.rho is not None,
        carries_eta=state.eta is not None,
        diagnostic_only=True,
        default_off=True,
        production_behavior_change=False,
        production_return_signature_change=False,
        preset_default_wiring_change=False,
        fastchem4_trace_public_runtime_constructor_inputs_used=False,
    )


def run_pdipm_rgie_dual_carrier_solver_step(
    *,
    explicit_opt_in: bool,
    carrier: PdipmRgieDualCarrierCallsiteInit,
    formula_matrix: Sequence[Sequence[float]],
    formula_matrix_cond_active: Sequence[Sequence[float]],
    element_inventory_target: Sequence[float],
    mass_action_constants: Sequence[float],
    hvector_cond_active: Sequence[float],
    barrier_parameter: float | None = None,
    alpha_candidates: Sequence[float] = (1.0, 0.5, 0.25, 0.125, 0.0625),
    merit_component_weights: Mapping[str, float] | None = None,
    linear_system_component_weights: Mapping[str, float] | None = None,
    linear_system_row_scaling: str = "none",
    linear_system_budget_priority_policy: str = "none",
    linear_system_budget_priority: float = 1.0,
    require_budget_nonworsening: bool = False,
    budget_rhs_sign: float = 1.0,
    max_gas_stationarity_worsening_ratio: float | None = None,
    max_condensate_stationarity_worsening_ratio: float | None = None,
) -> PdipmRgieRestrictedTrialReport:
    """Run one explicit experimental solver step from a dual carrier."""

    if not explicit_opt_in:
        raise ValueError("explicit_opt_in must be true for PD-IPM carrier solver steps.")
    if not isinstance(carrier, PdipmRgieDualCarrierCallsiteInit):
        raise TypeError("carrier must be a PdipmRgieDualCarrierCallsiteInit.")
    if not carrier.carries_element_potential:
        raise ValueError("carrier must preserve element_potential.")
    if carrier.state.rho is not None and not carrier.carries_rho:
        raise ValueError("carrier rho flag is inconsistent with state.rho.")
    return propose_pdipm_rgie_restricted_trial_step(
        explicit_opt_in=True,
        state=carrier.state,
        formula_matrix=formula_matrix,
        formula_matrix_cond_active=formula_matrix_cond_active,
        element_inventory_target=element_inventory_target,
        mass_action_constants=mass_action_constants,
        hvector_cond_active=hvector_cond_active,
        barrier_parameter=barrier_parameter,
        alpha_candidates=alpha_candidates,
        merit_component_weights=merit_component_weights,
        linear_system_component_weights=linear_system_component_weights,
        linear_system_row_scaling=linear_system_row_scaling,
        linear_system_budget_priority_policy=linear_system_budget_priority_policy,
        linear_system_budget_priority=linear_system_budget_priority,
        require_budget_nonworsening=require_budget_nonworsening,
        budget_rhs_sign=budget_rhs_sign,
        max_gas_stationarity_worsening_ratio=max_gas_stationarity_worsening_ratio,
        max_condensate_stationarity_worsening_ratio=max_condensate_stationarity_worsening_ratio,
    )


__all__ = (
    "PdipmRgieDualCarrierCallsiteInit",
    "PdipmRgieCondensateState",
    "PdipmRgieReducedStepReport",
    "PdipmRgieRestrictedTrialReport",
    "build_pdipm_rgie_dual_carrier_callsite_init",
    "build_pdipm_rgie_condensate_state",
    "propose_pdipm_rgie_restricted_trial_step",
    "run_pdipm_rgie_dual_carrier_solver_step",
    "solve_pdipm_rgie_algorithm_v11_reduced_step",
)
