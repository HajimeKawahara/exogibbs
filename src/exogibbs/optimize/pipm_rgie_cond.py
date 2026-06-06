import jax
import jax.numpy as jnp
import hashlib
import json
import math
import numpy as np
from dataclasses import asdict, dataclass
from jax import debug as jdebug
from jax.lax import while_loop
from jax.lax import cond
from jax.lax import fori_loop
from jax.scipy.linalg import cho_factor, cho_solve
from jax.scipy.linalg import lu_factor, lu_solve
from typing import Dict, Tuple, Optional, Sequence, Any
from time import perf_counter

from exogibbs.api.chemistry import ThermoState
from exogibbs.optimize.core import _A_diagn_At
from exogibbs.optimize.core import _compute_gk
from exogibbs.optimize.minimize import solve_gibbs_iteration_equations
from exogibbs.optimize.pipm_gie_cond import (
    solve_gibbs_iteration_equations_cond as solve_full_gibbs_iteration_equations_cond,
)

# heuristic step size functions for condensates
from exogibbs.optimize.stepsize import stepsize_cea_gas
from exogibbs.optimize.stepsize import stepsize_cond_heurstic
from exogibbs.optimize.stepsize import stepsize_sk
from exogibbs.optimize.stepsize import LOG_S_MAX

DEFAULT_REDUCED_SOLVER = "augmented_lu_row_scaled"
DEFAULT_REGULARIZATION_MODE = "none"
DEFAULT_REGULARIZATION_STRENGTH = 0.0
KL_DENSITY_GAUGE_P0_CGS = 1.0e6
KL_DENSITY_GAUGE_K_B_CGS = 1.380649e-16


@dataclass(frozen=True)
class FixedHighGainSourceStateCarrierRow:
    """One default-off fixed/high-gain source-state diagnostic row."""

    case_key: str
    element_label: str
    element_index: int
    fixed_high_gain_classification: str
    source_stage: str
    old_element_density: float
    overwrite_new_element_density_candidate: float
    row_scaling: float
    source_vector_contribution: float
    unscaled_numerator_contribution: float
    row_scaled_RHS_contribution: float
    hidden_source: bool
    reference_only: bool
    KL_native_constructible: bool
    metric_lineage: tuple[str, ...]


def build_fixed_high_gain_source_state_lifecycle_carrier(
    *,
    case_key: str,
    element_labels: Sequence[str],
    old_element_density: Sequence[float],
    overwrite_new_element_density_candidate: Sequence[float],
    row_scaling: Sequence[float],
    source_vector_contribution: Sequence[float],
    unscaled_numerator_contribution: Optional[Sequence[float]] = None,
    row_scaled_RHS_contribution: Optional[Sequence[float]] = None,
    fixed_element_labels: Sequence[str] = (),
    high_gain_element_labels: Sequence[str] = (),
    source_stage: str = "default-off fixed/high-gain source-state lifecycle carrier",
    metric_lineage: Sequence[str] = ("M41", "M55", "M56"),
    source_artifact: str = "KL diagnostic source-state carrier",
) -> Dict[str, Any]:
    """Build a default-off fixed/high-gain source-state lifecycle carrier.

    This helper is intentionally opt-in and has no call site in the production
    solver path.  It records the fixed/high-gain element rows and source/N/R
    propagation fields needed by milestone diagnostics without changing the
    RGIE/PIPM equations, presets, or defaults.
    """

    labels = [str(label) for label in element_labels]
    n_labels = len(labels)
    fixed = {str(label) for label in fixed_element_labels}
    high_gain = {str(label) for label in high_gain_element_labels}

    def _array(values: Sequence[float], name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != n_labels:
            raise ValueError(
                f"{name} must be a one-dimensional vector with one value per element "
                f"(got {arr.shape}, expected ({n_labels},))."
            )
        return arr

    old_density = _array(old_element_density, "old_element_density")
    overwrite_density = _array(
        overwrite_new_element_density_candidate,
        "overwrite_new_element_density_candidate",
    )
    scaling = _array(row_scaling, "row_scaling")
    source = _array(source_vector_contribution, "source_vector_contribution")
    numerator = (
        overwrite_density - old_density
        if unscaled_numerator_contribution is None
        else _array(unscaled_numerator_contribution, "unscaled_numerator_contribution")
    )
    rhs = (
        numerator / np.maximum(np.abs(scaling), 1.0)
        if row_scaled_RHS_contribution is None
        else _array(row_scaled_RHS_contribution, "row_scaled_RHS_contribution")
    )

    rows = []
    for index, label in enumerate(labels):
        if label in fixed and label in high_gain:
            classification = "fixed_and_high_gain"
        elif label in fixed:
            classification = "fixed"
        elif label in high_gain:
            classification = "high_gain"
        else:
            classification = "free"
        rows.append(
            asdict(
                FixedHighGainSourceStateCarrierRow(
                    case_key=case_key,
                    element_label=label,
                    element_index=index,
                    fixed_high_gain_classification=classification,
                    source_stage=source_stage,
                    old_element_density=float(old_density[index]),
                    overwrite_new_element_density_candidate=float(overwrite_density[index]),
                    row_scaling=float(scaling[index]),
                    source_vector_contribution=float(source[index]),
                    unscaled_numerator_contribution=float(numerator[index]),
                    row_scaled_RHS_contribution=float(rhs[index]),
                    hidden_source=False,
                    reference_only=False,
                    KL_native_constructible=True,
                    metric_lineage=tuple(str(item) for item in metric_lineage),
                )
            )
        )

    return {
        "carrier_schema": "default_off_fixed_high_gain_source_state_lifecycle_carrier_v1",
        "case_key": case_key,
        "diagnostic_only": True,
        "default_off": True,
        "active_only_when_explicitly_requested": True,
        "source_artifact": source_artifact,
        "source_stage": source_stage,
        "hidden_source": False,
        "reference_only": False,
        "KL_native_constructible": True,
        "production_behavior_change_required": False,
        "metric_lineage": [str(item) for item in metric_lineage],
        "fixed_element_labels": sorted(fixed),
        "high_gain_element_labels": sorted(high_gain),
        "rows": rows,
    }


def density_gauge_log_p0_over_kbt(
    temperature: float,
    *,
    p0_cgs: float = KL_DENSITY_GAUGE_P0_CGS,
    k_b_cgs: float = KL_DENSITY_GAUGE_K_B_CGS,
) -> jnp.ndarray:
    """Return ``ln(p0 / (k_B T))`` for KL audit-only pressure-density bridges."""

    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive.")
    return jnp.log(
        jnp.asarray(p0_cgs, dtype=jnp.float64)
        / (jnp.asarray(k_b_cgs, dtype=jnp.float64) * jnp.asarray(temperature, dtype=jnp.float64))
    )


def gas_molecule_density_gauge_bridge(
    formula_matrix_molecule: jnp.ndarray,
    temperature: float,
) -> jnp.ndarray:
    """Return ``(sum_j nu_ij - 1) ln(p0/(k_B T))`` for gas molecules."""

    formula_mol = jnp.asarray(formula_matrix_molecule, dtype=jnp.float64)
    delta_nu = jnp.sum(formula_mol, axis=0) - 1.0
    return delta_nu * density_gauge_log_p0_over_kbt(temperature)


def condensate_density_gauge_bridge(
    formula_matrix_cond: jnp.ndarray,
    temperature: float,
) -> jnp.ndarray:
    """Return ``sum_j nu_cj ln(p0/(k_B T))`` for condensate activities."""

    formula_cond = jnp.asarray(formula_matrix_cond, dtype=jnp.float64)
    return jnp.sum(formula_cond, axis=0) * density_gauge_log_p0_over_kbt(temperature)


def compute_condensate_budget_limits(
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    m: Optional[jnp.ndarray] = None,
    *,
    element_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Return inventory-based per-condensate budget limits.

    For condensate ``c`` this computes ``min_j b_j / nu_cj`` over positive
    stoichiometric entries. The helper is diagnostic/correction
    infrastructure; it does not alter the RGIE/PIPM equations.
    """

    formula = jnp.asarray(formula_matrix_cond, dtype=jnp.float64)
    budget = jnp.asarray(b, dtype=formula.dtype)
    if formula.ndim != 2:
        raise ValueError("formula_matrix_cond must be a two-dimensional array.")
    if budget.ndim != 1 or budget.shape[0] != formula.shape[0]:
        raise ValueError(
            "b must be a one-dimensional vector with one entry per element row."
        )

    positive = formula > 0.0
    ratios = jnp.where(positive, budget[:, None] / formula, jnp.inf)
    m_c_max_budget = jnp.min(ratios, axis=0)
    has_positive = jnp.any(positive, axis=0)
    limiting_element_index = jnp.where(
        has_positive,
        jnp.argmin(ratios, axis=0).astype(jnp.int32),
        jnp.asarray(-1, dtype=jnp.int32),
    )
    m_c_max_budget = jnp.where(has_positive, m_c_max_budget, jnp.inf)

    out: Dict[str, Any] = {
        "m_c_max_budget": m_c_max_budget,
        "limiting_element_index": limiting_element_index,
    }
    if element_names is not None:
        limiting_host = jax.device_get(limiting_element_index)
        out["limiting_element_name"] = [
            None if int(index) < 0 else str(element_names[int(index)])
            for index in limiting_host
        ]
    if m is not None:
        amount = jnp.asarray(m, dtype=formula.dtype)
        if amount.shape != m_c_max_budget.shape:
            raise ValueError(
                "m must have one entry per condensate column "
                f"(got {amount.shape}, expected {m_c_max_budget.shape})."
            )
        out["budget_ratio"] = amount / m_c_max_budget
    return out


def build_inventory_capped_rgie_startup_ln_mk(
    *,
    epsilon: float,
    r0: float,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    alpha_init: float,
    dtype: Optional[jnp.dtype] = None,
) -> jnp.ndarray:
    """Build ``min(nu * r0, alpha_init * m_c_max_budget)`` in log space."""

    if r0 <= 0.0:
        raise ValueError("r0 must be positive.")
    if alpha_init <= 0.0:
        raise ValueError("alpha_init must be positive.")
    if dtype is None:
        dtype = jnp.float64
    eps = jnp.asarray(epsilon, dtype=dtype)
    base_m0 = jnp.exp(eps) * jnp.asarray(r0, dtype=dtype)
    limits = compute_condensate_budget_limits(formula_matrix_cond, b)["m_c_max_budget"]
    capped = jnp.minimum(base_m0, jnp.asarray(alpha_init, dtype=dtype) * limits)
    capped = jnp.maximum(capped, jnp.asarray(1.0e-300, dtype=dtype))
    return jnp.log(capped.astype(dtype))


def budget_guard_accepts_condensate_burden(
    formula_matrix_cond: jnp.ndarray,
    m: jnp.ndarray,
    b: jnp.ndarray,
    *,
    budget_margin: float = 0.0,
) -> jnp.ndarray:
    """Return whether the aggregate condensate burden fits the element budget."""

    formula = jnp.asarray(formula_matrix_cond)
    amount = jnp.asarray(m, dtype=formula.dtype)
    budget = jnp.asarray(b, dtype=formula.dtype)
    margin = jnp.asarray(budget_margin, dtype=formula.dtype)
    burden = formula @ amount
    return jnp.all(burden <= (1.0 - margin) * budget)


def apply_emergency_budget_projection(
    formula_matrix_cond: jnp.ndarray,
    m: jnp.ndarray,
    b: jnp.ndarray,
    *,
    budget_margin: float = 0.0,
) -> Dict[str, jnp.ndarray]:
    """Fallback global scaling for budget violations.

    This is intentionally a diagnostic emergency mechanism, not the primary
    inventory-aware step control.
    """

    formula = jnp.asarray(formula_matrix_cond)
    amount = jnp.asarray(m, dtype=formula.dtype)
    budget = jnp.asarray(b, dtype=formula.dtype)
    margin = jnp.asarray(budget_margin, dtype=formula.dtype)
    burden = formula @ amount
    target = (1.0 - margin) * budget
    alpha_candidates = jnp.where(burden > 0.0, target / burden, jnp.inf)
    alpha = jnp.minimum(1.0, jnp.min(alpha_candidates))
    alpha = jnp.where(jnp.isfinite(alpha), jnp.clip(alpha, 0.0, 1.0), 1.0)
    projected = amount * alpha
    return {
        "m": projected,
        "alpha": alpha,
        "projection_used": alpha < 1.0,
        "burden_before": burden,
        "burden_after": formula @ projected,
    }


def assemble_inventory_capped_reduced_coupling_variant(
    nk: jnp.ndarray,
    mk: jnp.ndarray,
    ntotk: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
    *,
    variant_name: str = "current_reduced_coupling",
    alpha_coupling: float = 1.0,
) -> Dict[str, jnp.ndarray]:
    """Assemble diagnostic reduced RGIE terms with optional inventory capping.

    This helper is diagnostic-only. It changes neither the production equations
    nor the default solver path.
    """

    if alpha_coupling <= 0.0:
        raise ValueError("alpha_coupling must be positive.")
    valid = (
        "current_reduced_coupling",
        "capped_s_only",
        "capped_rhs_only",
        "capped_both",
    )
    if variant_name not in valid:
        raise ValueError(f"Unknown reduced-coupling variant '{variant_name}'. Expected one of {valid}.")

    nk = jnp.asarray(nk, dtype=jnp.float64)
    mk = jnp.asarray(mk, dtype=jnp.float64)
    ntotk = jnp.asarray(ntotk, dtype=jnp.float64)
    formula_matrix = jnp.asarray(formula_matrix, dtype=jnp.float64)
    formula_matrix_cond = jnp.asarray(formula_matrix_cond, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    gk = jnp.asarray(gk, dtype=jnp.float64)
    hvector_cond = jnp.asarray(hvector_cond, dtype=jnp.float64)
    nu = jnp.exp(jnp.asarray(epsilon, dtype=jnp.float64))

    limits = compute_condensate_budget_limits(formula_matrix_cond, b, mk)
    m_cap = jnp.minimum(
        mk,
        jnp.asarray(alpha_coupling, dtype=mk.dtype) * limits["m_c_max_budget"],
    )
    s_current = (mk * mk) / nu
    s_cap = (m_cap * m_cap) / nu
    use_cap_s = variant_name in ("capped_s_only", "capped_both")
    use_cap_rhs = variant_name in ("capped_rhs_only", "capped_both")
    sk_q = jnp.where(use_cap_s, s_cap, s_current)
    sk_rhs = jnp.where(use_cap_rhs, s_cap, s_current)
    m_rhs = jnp.where(use_cap_rhs, m_cap, mk)

    resn = jnp.sum(nk) - ntotk
    bk = formula_matrix @ nk
    q_gas = _A_diagn_At(nk, formula_matrix)
    q_cond = _A_diagn_At(sk_q, formula_matrix_cond)
    q_block = q_gas + q_cond
    Angk = formula_matrix @ (gk * nk)
    ngk = jnp.dot(nk, gk)
    delta_b_hat = b - (bk + formula_matrix_cond @ m_rhs)
    condvec = formula_matrix_cond @ (sk_rhs * hvector_cond - m_rhs)
    rhs = Angk + condvec + delta_b_hat
    scalar_rhs = ngk - resn
    assemble_mat = jnp.block([[q_block, bk[:, None]], [bk[None, :], jnp.array([[resn]])]])
    assemble_vec = jnp.concatenate([rhs, jnp.array([scalar_rhs])])
    return {
        "variant_name": variant_name,
        "alpha_coupling": jnp.asarray(alpha_coupling, dtype=jnp.float64),
        "m_c_max_budget": limits["m_c_max_budget"],
        "budget_ratio": limits["budget_ratio"],
        "limiting_element_index": limits["limiting_element_index"],
        "m_cap": m_cap,
        "s_current": s_current,
        "s_cap": s_cap,
        "sk_q": sk_q,
        "sk_rhs": sk_rhs,
        "m_rhs": m_rhs,
        "q_gas": q_gas,
        "q_cond": q_cond,
        "q_block": q_block,
        "condvec": condvec,
        "delta_b_hat": delta_b_hat,
        "rhs": rhs,
        "resn": resn,
        "bk": bk,
        "scalar_rhs": scalar_rhs,
        "assemble_mat": assemble_mat,
        "assemble_vec": assemble_vec,
        "uses_capped_s": jnp.asarray(use_cap_s),
        "uses_capped_rhs": jnp.asarray(use_cap_rhs),
        "capped_count": jnp.sum(m_cap < mk).astype(jnp.int32),
    }


def compute_hybrid_candidate_log_activity_proxy(
    formula_matrix_cond: jnp.ndarray,
    pi_g: jnp.ndarray,
    hvector_cond: jnp.ndarray,
) -> jnp.ndarray:
    """Return the FastChem-like condensate activity proxy ``A_cond.T @ pi_g - h``.

    The current ExoGibbs condensate RGIE state is expressed with ``ln_mk`` and
    the complementarity barrier, not FastChem's atomic-density activity state.
    Until an exact atomic-density activity proxy is carried through the solver,
    this opt-in experimental branch uses the recovered gas-only dual vector as
    the branch definition and records that choice in audit diagnostics.
    """

    return (
        jnp.asarray(formula_matrix_cond, dtype=jnp.float64).T
        @ jnp.asarray(pi_g, dtype=jnp.float64)
        - jnp.asarray(hvector_cond, dtype=jnp.float64)
    )


def build_hybrid_candidate_masks(
    log_activity_proxy: jnp.ndarray,
    *,
    near_margin: float = -0.1,
    weighted: bool = False,
) -> Dict[str, jnp.ndarray]:
    """Build FastChem-like active and near-active condensate masks.

    ``active`` follows the structural audit rule ``log_activity >= 0``.
    ``near_active`` follows the optional Jacobian margin
    ``log_activity > -0.1``.  The weighted diagnostic variant replaces the hard
    near-active mask with a narrow linear ramp over ``[-0.1, 0]`` while keeping a
    separate hard active indicator for reporting.
    """

    proxy = jnp.asarray(log_activity_proxy, dtype=jnp.float64)
    active_bool = proxy >= 0.0
    near_bool = proxy > near_margin
    active = active_bool.astype(proxy.dtype)
    if weighted:
        width = jnp.maximum(jnp.asarray(-near_margin, dtype=proxy.dtype), 1.0e-12)
        near = jnp.clip((proxy - near_margin) / width, 0.0, 1.0)
        active_for_rhs = near
    else:
        near = near_bool.astype(proxy.dtype)
        active_for_rhs = active
    return {
        "active_bool": active_bool,
        "near_active_bool": near_bool,
        "active": active,
        "near_active": near,
        "active_for_rhs": active_for_rhs,
    }


def compute_condensed_element_gas_recoupling_terms(
    formula_matrix_cond: jnp.ndarray,
    m_active: jnp.ndarray,
    b: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Return condensed-element gas recoupling bookkeeping.

    The diagnostic replay uses the FastChem-style element recoupling view
    ``d_elem = A_cond m_active`` and ``b_eff = b - d_elem``.  ``phi`` is a
    condensed-element fraction proxy, useful for identifying which condensed
    elements control the gas-only replay.
    """

    formula = jnp.asarray(formula_matrix_cond, dtype=jnp.float64)
    m_active = jnp.asarray(m_active, dtype=jnp.float64)
    budget = jnp.asarray(b, dtype=jnp.float64)
    d_elem = formula @ m_active
    b_eff = budget - d_elem
    phi = jnp.where(budget > 0.0, d_elem / budget, 0.0)
    return {"d_elem": d_elem, "b_eff": b_eff, "phi": phi}


def build_internal_complementarity_tau(
    candidate_indices: jnp.ndarray,
    epsilon: float,
    *,
    tau_scale: float = 1.0,
    dtype: Optional[jnp.dtype] = None,
) -> jnp.ndarray:
    """Return fixed-tau bookkeeping for the internal complementarity branch.

    The experimental branch preserves ExoGibbs' gas-side RGIE variables but
    replaces the eliminated condensate barrier equation
    ``m * (A_c.T @ pi - h_c) + nu = 0`` by explicit variables
    ``r = log(m)`` and ``chi = log(zeta)`` satisfying
    ``A_c.T @ pi - h_c + zeta = 0`` and ``r + chi - log(tau) = 0``.
    We use ``tau = tau_scale * exp(epsilon)`` by default so the branch remains
    tied to the current RGIE barrier schedule while exposing the complementarity
    pair internally.  This is still a pi-proxy transplant, not a true
    atomic-density KL/FastChem inner branch.
    """

    if tau_scale <= 0.0:
        raise ValueError("tau_scale must be positive.")
    indices = jnp.asarray(candidate_indices, dtype=jnp.int32)
    if indices.ndim != 1:
        raise ValueError("candidate_indices must be a one-dimensional array.")
    if dtype is None:
        dtype = jnp.float64
    tau = jnp.asarray(tau_scale, dtype=dtype) * jnp.exp(jnp.asarray(epsilon, dtype=dtype))
    return jnp.full((indices.shape[0],), tau, dtype=dtype)


def compute_internal_complementarity_residual(
    q: jnp.ndarray,
    r_c: jnp.ndarray,
    chi_c: jnp.ndarray,
    pi: jnp.ndarray,
    q_tot: jnp.ndarray,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond_c: jnp.ndarray,
    b: jnp.ndarray,
    hvector_gas: jnp.ndarray,
    hvector_cond_c: jnp.ndarray,
    ln_normalized_pressure: jnp.ndarray,
    tau_c: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Build residual blocks for the opt-in internal complementarity branch.

    Preserved from current RGIE: gas variables ``q = ln(n)``, ``pi``, and
    ``q_tot = ln(n_tot)`` plus gas stationarity, element conservation, and
    total-number closure.  Changed only inside the candidate condensate block:
    explicit ``r_c = ln(m_c)`` and ``chi_c = ln(zeta_c)`` replace the
    barrier-eliminated condensate residual.
    """

    q = jnp.asarray(q, dtype=jnp.float64)
    r_c = jnp.asarray(r_c, dtype=jnp.float64)
    chi_c = jnp.asarray(chi_c, dtype=jnp.float64)
    pi = jnp.asarray(pi, dtype=jnp.float64)
    q_tot = jnp.asarray(q_tot, dtype=jnp.float64)
    formula_matrix = jnp.asarray(formula_matrix, dtype=jnp.float64)
    formula_matrix_cond_c = jnp.asarray(formula_matrix_cond_c, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    hvector_gas = jnp.asarray(hvector_gas, dtype=jnp.float64)
    hvector_cond_c = jnp.asarray(hvector_cond_c, dtype=jnp.float64)
    tau_c = jnp.asarray(tau_c, dtype=jnp.float64)

    n = jnp.exp(q)
    m_c = jnp.exp(r_c)
    zeta_c = jnp.exp(chi_c)
    n_tot = jnp.exp(q_tot)
    g = hvector_gas + q - q_tot + ln_normalized_pressure
    gas_stationarity = n * (formula_matrix.T @ pi - g)
    element_conservation = formula_matrix @ n + formula_matrix_cond_c @ m_c - b
    total_number_closure = jnp.asarray([jnp.sum(n) - n_tot], dtype=jnp.float64)
    activity_complementarity = formula_matrix_cond_c.T @ pi - hvector_cond_c + zeta_c
    fixed_tau_complementarity = r_c + chi_c - jnp.log(tau_c)
    flat = jnp.concatenate(
        [
            gas_stationarity,
            element_conservation,
            total_number_closure,
            activity_complementarity,
            fixed_tau_complementarity,
        ]
    )
    return {
        "gas_stationarity": gas_stationarity,
        "element_conservation": element_conservation,
        "total_number_closure": total_number_closure,
        "activity_complementarity": activity_complementarity,
        "fixed_tau_complementarity": fixed_tau_complementarity,
        "flat": flat,
        "max_abs_activity_complementarity": jnp.max(jnp.abs(activity_complementarity))
        if activity_complementarity.size
        else jnp.asarray(0.0, dtype=jnp.float64),
        "max_abs_fixed_tau_complementarity": jnp.max(jnp.abs(fixed_tau_complementarity))
        if fixed_tau_complementarity.size
        else jnp.asarray(0.0, dtype=jnp.float64),
    }


def reconstruct_kl_atomic_gas_from_u(
    u: jnp.ndarray,
    formula_matrix_gas: jnp.ndarray,
    hvector_gas: jnp.ndarray,
    *,
    temperature: Optional[float] = None,
    apply_density_gauge_bridge: bool = False,
) -> Dict[str, jnp.ndarray]:
    """Reconstruct gas densities from KL/FastChem-style atomic variables.

    This helper is diagnostic-only.  It intentionally does not enter the
    current RGIE/PIPM production algebra.  The FastChem presets list element
    gas species first and store ``h = -logK``.  The KL-like branch therefore
    keeps ``u = log(n_atom)`` for those element species and reconstructs
    molecules with ``log(n_mol) = logK + A_mol.T @ u``.  The optional density
    bridge is guarded for audit-only KL branches and leaves the legacy
    diagnostic path unchanged by default.
    """

    u = jnp.asarray(u, dtype=jnp.float64)
    formula = jnp.asarray(formula_matrix_gas, dtype=jnp.float64)
    hvector = jnp.asarray(hvector_gas, dtype=jnp.float64)
    n_elements = int(u.shape[0])
    if formula.shape[0] != n_elements:
        raise ValueError(
            "u must have one entry per element row "
            f"(got {n_elements}, expected {formula.shape[0]})."
        )
    if hvector.shape[0] != formula.shape[1]:
        raise ValueError("hvector_gas must have one entry per gas species column.")

    log_atom = u
    formula_mol = formula[:, n_elements:]
    h_mol = hvector[n_elements:]
    bridge_mol = jnp.zeros_like(h_mol)
    if apply_density_gauge_bridge:
        if temperature is None:
            raise ValueError("temperature is required when apply_density_gauge_bridge=True.")
        bridge_mol = gas_molecule_density_gauge_bridge(formula_mol, temperature)
    h_mol_density = h_mol + bridge_mol
    log_mol = formula_mol.T @ u - h_mol_density
    ln_nk = jnp.concatenate([log_atom, log_mol])
    nk = jnp.exp(ln_nk)
    return {
        "ln_nk": ln_nk,
        "nk": nk,
        "ln_ntot": jnp.log(jnp.maximum(jnp.sum(nk), jnp.asarray(1.0e-300, dtype=nk.dtype))),
        "atom_n": jnp.exp(log_atom),
        "molecule_n": jnp.exp(log_mol),
        "formula_matrix_molecule": formula_mol,
        "h_molecule_raw": h_mol,
        "density_gauge_bridge_molecule": bridge_mol,
        "h_molecule_density_gauge": h_mol_density,
        "density_gauge_bridge_applied": jnp.asarray(apply_density_gauge_bridge),
    }


def compute_kl_condensate_log_activity(
    u: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    *,
    temperature: Optional[float] = None,
    apply_density_gauge_bridge: bool = False,
) -> jnp.ndarray:
    """Return true KL/FastChem-like condensate activity ``logK + A_C.T @ u``.

    This differs from the earlier RGIE-preserving pi-proxy branch, which used
    ``A_C.T @ pi_g - h_C``.  Here ``u`` is the atomic-density state itself.
    """

    formula_cond = jnp.asarray(formula_matrix_cond, dtype=jnp.float64)
    hcond = jnp.asarray(hvector_cond, dtype=jnp.float64)
    if apply_density_gauge_bridge:
        if temperature is None:
            raise ValueError("temperature is required when apply_density_gauge_bridge=True.")
        hcond = hcond + condensate_density_gauge_bridge(formula_cond, temperature)
    return formula_cond.T @ jnp.asarray(u, dtype=jnp.float64) - hcond


def build_kl_atomic_candidate_masks(
    log_activity: jnp.ndarray,
    *,
    near_margin: float = -0.1,
) -> Dict[str, jnp.ndarray]:
    """Build active and near-active sets from true KL-like ``ell_c(u)``."""

    ell = jnp.asarray(log_activity, dtype=jnp.float64)
    active_bool = ell >= 0.0
    near_active_bool = ell > near_margin
    return {
        "active_bool": active_bool,
        "near_active_bool": near_active_bool,
        "active": active_bool.astype(ell.dtype),
        "near_active": near_active_bool.astype(ell.dtype),
    }


def compute_kl_atomic_complementarity_residual(
    u: jnp.ndarray,
    r_c: jnp.ndarray,
    chi_c: jnp.ndarray,
    formula_matrix_gas: jnp.ndarray,
    formula_matrix_cond_c: jnp.ndarray,
    b: jnp.ndarray,
    hvector_gas: jnp.ndarray,
    hvector_cond_c: jnp.ndarray,
    tau_c: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Build residual blocks for the opt-in KL-like atomic branch.

    Shared ExoGibbs infrastructure: FastChem thermodynamic tables, formula
    matrices, parity-fixed abundances, and gas replay tooling.  Mathematically
    different from RGIE: gas densities are reconstructed from atomic density
    variables ``u`` via mass action, and condensates use true
    ``ell_C(u) = logK_C + A_C.T @ u`` rather than a recovered RGIE dual proxy.
    """

    u = jnp.asarray(u, dtype=jnp.float64)
    r_c = jnp.asarray(r_c, dtype=jnp.float64)
    chi_c = jnp.asarray(chi_c, dtype=jnp.float64)
    formula = jnp.asarray(formula_matrix_gas, dtype=jnp.float64)
    formula_cond_c = jnp.asarray(formula_matrix_cond_c, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    hvector_cond_c = jnp.asarray(hvector_cond_c, dtype=jnp.float64)
    tau_c = jnp.asarray(tau_c, dtype=jnp.float64)

    gas = reconstruct_kl_atomic_gas_from_u(u, formula, hvector_gas)
    m_c = jnp.exp(r_c)
    zeta_c = jnp.exp(chi_c)
    ell_c = compute_kl_condensate_log_activity(u, formula_cond_c, hvector_cond_c)
    element_conservation = b - gas["atom_n"] - gas["formula_matrix_molecule"] @ gas["molecule_n"] - formula_cond_c @ m_c
    activity_slack = ell_c + zeta_c
    fixed_tau_complementarity = r_c + chi_c - jnp.log(tau_c)
    flat = jnp.concatenate(
        [element_conservation, activity_slack, fixed_tau_complementarity]
    )
    return {
        "element_conservation": element_conservation,
        "activity_slack": activity_slack,
        "fixed_tau_complementarity": fixed_tau_complementarity,
        "flat": flat,
        "ell_c": ell_c,
        "ln_nk": gas["ln_nk"],
        "ln_ntot": gas["ln_ntot"],
        "nk": gas["nk"],
        "max_abs_activity_slack": jnp.max(jnp.abs(activity_slack))
        if activity_slack.size
        else jnp.asarray(0.0, dtype=jnp.float64),
        "max_abs_fixed_tau_complementarity": jnp.max(jnp.abs(fixed_tau_complementarity))
        if fixed_tau_complementarity.size
        else jnp.asarray(0.0, dtype=jnp.float64),
        "max_abs_element_conservation": jnp.max(jnp.abs(element_conservation))
        if element_conservation.size
        else jnp.asarray(0.0, dtype=jnp.float64),
    }


def _recover_gas_dual_from_state(
    nk: jnp.ndarray,
    formula_matrix: jnp.ndarray,
    gk: jnp.ndarray,
) -> jnp.ndarray:
    q_gas = _A_diagn_At(nk, formula_matrix)
    rhs = formula_matrix @ (gk * nk)
    return jnp.linalg.lstsq(q_gas, rhs)[0]


def solve_hybrid_candidate_selected_reduced_coupling_direction(
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
    *,
    candidate_mode: str = "candidate_selected_active_plus_near_jacobian",
    near_margin: float = -0.1,
) -> Dict[str, jnp.ndarray]:
    """Solve one opt-in hybrid candidate-selected RGIE direction.

    This is a diagnostic/experimental transplant of the FastChem Cond
    structural audit:

    * recover a gas-only dual ``pi_g`` from the current gas state,
    * define ``log_activity_proxy_c = A_cond[:, c].T @ pi_g - h_cond[c]``,
    * select active condensates with ``log_activity_proxy >= 0``,
    * optionally include ``log_activity_proxy > -0.1`` in the Jacobian only,
    * assemble ``Q_hybrid``, ``delta_b_hat_hybrid``, and ``condvec_hybrid``
      without changing ExoGibbs state variables or support handling.
    """

    valid = (
        "candidate_selected_active_only",
        "candidate_selected_active_plus_near_jacobian",
        "candidate_selected_weighted_mask",
    )
    if candidate_mode not in valid:
        raise ValueError(f"Unknown candidate_mode '{candidate_mode}'. Expected one of {valid}.")

    nk = jnp.exp(jnp.asarray(ln_nk, dtype=jnp.float64))
    mk = jnp.exp(jnp.asarray(ln_mk, dtype=jnp.float64))
    ntot = jnp.exp(jnp.asarray(ln_ntot, dtype=jnp.float64))
    formula_matrix = jnp.asarray(formula_matrix, dtype=jnp.float64)
    formula_matrix_cond = jnp.asarray(formula_matrix_cond, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    gk = jnp.asarray(gk, dtype=jnp.float64)
    hvector_cond = jnp.asarray(hvector_cond, dtype=jnp.float64)
    nu = jnp.exp(jnp.asarray(epsilon, dtype=jnp.float64))

    pi_g = _recover_gas_dual_from_state(nk, formula_matrix, gk)
    log_activity_proxy = compute_hybrid_candidate_log_activity_proxy(
        formula_matrix_cond,
        pi_g,
        hvector_cond,
    )
    masks = build_hybrid_candidate_masks(
        log_activity_proxy,
        near_margin=near_margin,
        weighted=candidate_mode == "candidate_selected_weighted_mask",
    )
    if candidate_mode == "candidate_selected_active_only":
        jacobian_mask = masks["active"]
    else:
        jacobian_mask = masks["near_active"]
    rhs_mask = masks["active_for_rhs"]

    # Hybrid reduced system:
    #   Q_hybrid = A_g diag(n) A_g.T + A_cond diag(s_near) A_cond.T
    #   delta_b_hat_hybrid = b - (A_g n + A_cond m_active)
    #   condvec_hybrid = A_cond (s_near * h_cond - m_active)
    # ``s_near`` is Jacobian-gated, while ``m_active`` is active-gated for the
    # hard-mask branches.  The weighted branch is diagnostic-only and replaces
    # both gates with a smooth ramp around the FastChem activity boundary.
    bk = formula_matrix @ nk
    m_active = rhs_mask * mk
    s_near = jacobian_mask * (mk * mk / nu)
    q_gas = _A_diagn_At(nk, formula_matrix)
    q_cond = _A_diagn_At(s_near, formula_matrix_cond)
    q_block = q_gas + q_cond
    resn = jnp.sum(nk) - ntot
    Angk = formula_matrix @ (gk * nk)
    ngk = jnp.dot(nk, gk)
    delta_b_hat_hybrid = b - (bk + formula_matrix_cond @ m_active)
    condvec_hybrid = formula_matrix_cond @ (s_near * hvector_cond - m_active)
    rhs = Angk + condvec_hybrid + delta_b_hat_hybrid
    scalar_rhs = ngk - resn
    assemble_mat = jnp.block([[q_block, bk[:, None]], [bk[None, :], jnp.array([[resn]])]])
    assemble_vec = jnp.concatenate([rhs, jnp.array([scalar_rhs])])
    row_scale = jnp.maximum(jnp.max(jnp.abs(assemble_mat), axis=1, keepdims=True), 1.0)
    solve_mat = assemble_mat / row_scale
    solve_vec = assemble_vec / row_scale[:, 0]
    lu, piv = lu_factor(solve_mat)
    assemble_variable = lu_solve((lu, piv), solve_vec)
    pi_vector = assemble_variable[:-1]
    delta_ln_ntot = assemble_variable[-1]
    delta_ln_nk = formula_matrix.T @ pi_vector + delta_ln_ntot - gk
    raw_delta_ln_mk = jnp.exp(jnp.asarray(ln_mk, dtype=jnp.float64) - epsilon) * (
        formula_matrix_cond.T @ pi_vector - hvector_cond
    ) + 1.0
    return {
        "candidate_mode": candidate_mode,
        "activity_proxy_source": "pi_g_dual_proxy",
        "atomic_density_proxy_available": jnp.asarray(False),
        "near_margin": jnp.asarray(near_margin, dtype=jnp.float64),
        "pi_g": pi_g,
        "log_activity_proxy": log_activity_proxy,
        "active_mask": masks["active"],
        "near_active_mask": masks["near_active"],
        "active_bool": masks["active_bool"],
        "near_active_bool": masks["near_active_bool"],
        "jacobian_mask": jacobian_mask,
        "m_active": m_active,
        "s_near": s_near,
        "q_gas": q_gas,
        "q_cond": q_cond,
        "q_block": q_block,
        "delta_b_hat_hybrid": delta_b_hat_hybrid,
        "condvec_hybrid": condvec_hybrid,
        "rhs": rhs,
        "resn": resn,
        "bk": bk,
        "scalar_rhs": scalar_rhs,
        "assemble_mat": assemble_mat,
        "assemble_vec": assemble_vec,
        "pi_vector": pi_vector,
        "delta_ln_ntot": delta_ln_ntot,
        "delta_ln_nk": delta_ln_nk,
        "raw_delta_ln_mk": raw_delta_ln_mk,
        "delta_ln_mk": jnp.clip(raw_delta_ln_mk, -0.1, 0.1),
        "factorization_succeeded": jnp.all(jnp.isfinite(assemble_variable)),
        "candidate_set_size": jnp.sum(masks["active_bool"]).astype(jnp.int32),
        "near_active_set_size": jnp.sum(masks["near_active_bool"]).astype(jnp.int32),
        "weighted_mask": jnp.asarray(candidate_mode == "candidate_selected_weighted_mask"),
    }


def solve_inventory_capped_reduced_coupling_direction(
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
    *,
    variant_name: str = "current_reduced_coupling",
    alpha_coupling: float = 1.0,
) -> Dict[str, jnp.ndarray]:
    """Solve one diagnostic reduced-coupling variant direction."""

    nk = jnp.exp(jnp.asarray(ln_nk, dtype=jnp.float64))
    mk = jnp.exp(jnp.asarray(ln_mk, dtype=jnp.float64))
    ntot = jnp.exp(jnp.asarray(ln_ntot, dtype=jnp.float64))
    terms = assemble_inventory_capped_reduced_coupling_variant(
        nk,
        mk,
        ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        epsilon,
        variant_name=variant_name,
        alpha_coupling=alpha_coupling,
    )
    row_scale = jnp.maximum(jnp.max(jnp.abs(terms["assemble_mat"]), axis=1, keepdims=True), 1.0)
    solve_mat = terms["assemble_mat"] / row_scale
    solve_vec = terms["assemble_vec"] / row_scale[:, 0]
    lu, piv = lu_factor(solve_mat)
    assemble_variable = lu_solve((lu, piv), solve_vec)
    pi_vector = assemble_variable[:-1]
    delta_ln_ntot = assemble_variable[-1]
    delta_ln_nk = jnp.asarray(formula_matrix).T @ pi_vector + delta_ln_ntot - gk
    raw_delta_ln_mk = jnp.exp(jnp.asarray(ln_mk) - epsilon) * (
        jnp.asarray(formula_matrix_cond).T @ pi_vector - hvector_cond
    ) + 1.0
    return {
        **terms,
        "pi_vector": pi_vector,
        "delta_ln_ntot": delta_ln_ntot,
        "delta_ln_nk": delta_ln_nk,
        "raw_delta_ln_mk": raw_delta_ln_mk,
        "delta_ln_mk": jnp.clip(raw_delta_ln_mk, -0.1, 0.1),
        "factorization_succeeded": jnp.all(jnp.isfinite(assemble_variable)),
        "max_abs_delta_ln_nk": jnp.max(jnp.abs(delta_ln_nk)),
        "max_abs_raw_delta_ln_mk": jnp.max(jnp.abs(raw_delta_ln_mk)),
    }


def select_conditional_capped_s_reduced_coupling_mode(
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    temperature: float,
    ln_normalized_pressure: float,
    hvector: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
    *,
    alpha_candidates: Sequence[float] = (1.0e-2, 1.0e-1, 1.0),
    mode_selection_margin: float = 0.05,
    shadow_lambda: float = 0.1,
) -> Dict[str, Any]:
    """Choose an opt-in capped-s reduced coupling mode from one shadow step.

    This is deliberately a per-layer/per-run selector. The returned mode and
    alpha are meant to be frozen for the subsequent first-pass run.
    """

    gk = _compute_gk(
        temperature,
        ln_nk,
        ln_ntot,
        hvector,
        ln_normalized_pressure,
    )
    current = solve_inventory_capped_reduced_coupling_direction(
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        epsilon,
        variant_name="current_reduced_coupling",
        alpha_coupling=1.0,
    )

    def score(direction: Dict[str, jnp.ndarray]) -> float:
        trial = _evaluate_trial_step(
            ln_nk,
            ln_mk,
            ln_ntot,
            shadow_lambda,
            direction["delta_ln_nk"],
            jnp.clip(direction["raw_delta_ln_mk"], -0.1, 0.1),
            direction["delta_ln_ntot"],
            formula_matrix,
            formula_matrix_cond,
            b,
            temperature,
            ln_normalized_pressure,
            hvector,
            hvector_cond,
            epsilon,
        )
        return float(trial["fresh_residual"])

    current_score = score(current)
    candidates = []
    for alpha in alpha_candidates:
        direction = solve_inventory_capped_reduced_coupling_direction(
            ln_nk,
            ln_mk,
            ln_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            gk,
            hvector_cond,
            epsilon,
            variant_name="capped_s_only",
            alpha_coupling=float(alpha),
        )
        candidates.append(
            {
                "mode": "capped_s_only",
                "alpha_s": float(alpha),
                "fresh_residual": score(direction),
            }
        )
    best = min(candidates, key=lambda row: row["fresh_residual"]) if candidates else None
    required = current_score * (1.0 - float(mode_selection_margin))
    escalation = bool(best is not None and best["fresh_residual"] < required)
    selected_mode = "capped_s_only" if escalation else "current"
    selected_alpha = float(best["alpha_s"]) if escalation and best is not None else 1.0
    return {
        "selected_mode": selected_mode,
        "selected_alpha_s": selected_alpha,
        "shadow_best_fresh_residual": float(best["fresh_residual"]) if best is not None else float("inf"),
        "shadow_current_fresh_residual": float(current_score),
        "mode_selection_margin": float(mode_selection_margin),
        "escalation_triggered": escalation,
        "shadow_lambda": float(shadow_lambda),
        "shadow_scores": [{"mode": "current", "alpha_s": 1.0, "fresh_residual": float(current_score)}]
        + candidates,
    }


def diagnose_rgie_raw_condensate_update_block(
    ln_mk: jnp.ndarray,
    epsilon: float,
    formula_matrix_cond: jnp.ndarray,
    pi_vector: jnp.ndarray,
    hvector_cond: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Return the current RGIE raw condensate update anatomy.

    This is diagnostic-only. It exposes the decomposition behind the current
    production raw update

        raw_delta_ln_m = 1 + (m / nu) * (A_cond^T pi - h_cond)

    without changing the solver path.
    """

    ln_mk = jnp.asarray(ln_mk)
    epsilon = jnp.asarray(epsilon, dtype=ln_mk.dtype)
    nu = jnp.exp(epsilon)
    mk = jnp.exp(ln_mk)
    driving = formula_matrix_cond.T @ jnp.asarray(pi_vector) - jnp.asarray(hvector_cond)
    factor = mk / nu
    correction = factor * driving
    raw_delta_ln_mk_current = 1.0 + correction
    condensate_stationarity_residual = mk * driving + nu
    condensate_stationarity_residual_over_nu = condensate_stationarity_residual / nu
    return {
        "nu": nu,
        "mk": mk,
        "factor": factor,
        "driving": driving,
        "correction": correction,
        "raw_delta_ln_mk_current": raw_delta_ln_mk_current,
        "condensate_stationarity_residual": condensate_stationarity_residual,
        "condensate_stationarity_residual_over_nu": condensate_stationarity_residual_over_nu,
        "raw_identity_max_abs_diff": jnp.max(
            jnp.abs(raw_delta_ln_mk_current - condensate_stationarity_residual_over_nu)
        ),
    }


def build_rgie_condensate_direction_variant(
    raw_update: Dict[str, jnp.ndarray],
    variant_name: str,
) -> Dict[str, jnp.ndarray]:
    """Build a diagnostic condensate-direction variant from the raw RGIE update block."""

    raw_delta_ln_mk_current = jnp.asarray(raw_update["raw_delta_ln_mk_current"])
    correction = jnp.asarray(raw_update["correction"])

    if variant_name == "production_clipped_current":
        delta_ln_mk = jnp.clip(raw_delta_ln_mk_current, -0.1, 0.1)
    elif variant_name == "raw_current_no_clip":
        delta_ln_mk = raw_delta_ln_mk_current
    elif variant_name == "correction_only_no_clip":
        delta_ln_mk = correction
    elif variant_name == "gas_only":
        delta_ln_mk = jnp.zeros_like(raw_delta_ln_mk_current)
    elif variant_name == "correction_only_scalar_rescale_0p1":
        max_abs_correction = jnp.max(jnp.abs(correction))
        alpha = jnp.where(max_abs_correction <= 0.1, 1.0, 0.1 / max_abs_correction)
        delta_ln_mk = alpha * correction
    else:
        raise ValueError(
            "Unknown RGIE condensate direction variant "
            f"'{variant_name}'. Expected one of "
            "('production_clipped_current', 'raw_current_no_clip', "
            "'correction_only_no_clip', 'gas_only', "
            "'correction_only_scalar_rescale_0p1')."
        )

    return {
        "variant_name": variant_name,
        "delta_ln_mk": delta_ln_mk,
        "max_abs_delta_ln_mk": jnp.max(jnp.abs(delta_ln_mk)),
    }


def build_rgie_condensate_direction_transform_variant(
    raw_delta_ln_mk: jnp.ndarray,
    variant_name: str,
) -> Dict[str, jnp.ndarray]:
    """Build a diagnostic condensate direction transform from the raw RGIE update."""

    raw = jnp.asarray(raw_delta_ln_mk)
    if variant_name == "current_component_clip_0p1":
        limit = 0.1
        delta = jnp.clip(raw, -limit, limit)
        saturated = jnp.abs(raw) > limit + 1.0e-15
    elif variant_name == "component_clip_0p5":
        limit = 0.5
        delta = jnp.clip(raw, -limit, limit)
        saturated = jnp.abs(raw) > limit + 1.0e-15
    elif variant_name == "scalar_rescale_inf_0p1":
        limit = 0.1
        max_abs = jnp.max(jnp.abs(raw))
        alpha = jnp.where(max_abs <= limit, 1.0, limit / max_abs)
        delta = alpha * raw
        saturated = None
    elif variant_name == "scalar_rescale_inf_0p5":
        limit = 0.5
        max_abs = jnp.max(jnp.abs(raw))
        alpha = jnp.where(max_abs <= limit, 1.0, limit / max_abs)
        delta = alpha * raw
        saturated = None
    elif variant_name == "raw_no_clip":
        delta = raw
        saturated = None
    else:
        raise ValueError(
            "Unknown RGIE condensate direction-transform variant "
            f"'{variant_name}'. Expected one of "
            "('current_component_clip_0p1', 'component_clip_0p5', "
            "'scalar_rescale_inf_0p1', 'scalar_rescale_inf_0p5', 'raw_no_clip')."
        )

    return {
        "variant_name": variant_name,
        "delta_ln_mk": delta,
        "cosine_raw_vs_variant": jnp.where(
            jnp.linalg.norm(raw) * jnp.linalg.norm(delta) > 1.0e-300,
            jnp.clip(
                jnp.dot(raw, delta)
                / jnp.maximum(jnp.linalg.norm(raw) * jnp.linalg.norm(delta), 1.0e-300),
                -1.0,
                1.0,
            ),
            jnp.nan,
        ),
        "max_abs_variant_delta_ln_mk": jnp.max(jnp.abs(delta)),
        "saturated_fraction": None
        if saturated is None
        else jnp.mean(saturated.astype(jnp.float64)),
    }


def compute_rgie_lambda_cap_policy(
    policy_name: str,
    *,
    lam1_gas: jnp.ndarray,
    lam1_cond: jnp.ndarray,
    lam2_cond: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Compute a diagnostic lambda cap policy from existing RGIE heuristic ceilings."""

    if policy_name == "current_full_cap":
        lam_cap = jnp.minimum(1.0, jnp.minimum(lam1_gas, jnp.minimum(lam1_cond, lam2_cond)))
    elif policy_name == "no_cond_cap":
        lam_cap = jnp.minimum(1.0, jnp.minimum(lam1_gas, lam2_cond))
    elif policy_name == "no_sk_cap":
        lam_cap = jnp.minimum(1.0, jnp.minimum(lam1_gas, lam1_cond))
    elif policy_name == "gas_only_cap":
        lam_cap = jnp.minimum(1.0, lam1_gas)
    elif policy_name == "no_heuristic_cap":
        lam_cap = jnp.asarray(1.0, dtype=jnp.asarray(lam1_gas).dtype)
    else:
        raise ValueError(
            "Unknown RGIE lambda-cap policy "
            f"'{policy_name}'. Expected one of "
            "('current_full_cap', 'no_cond_cap', 'no_sk_cap', 'gas_only_cap', 'no_heuristic_cap')."
        )

    limiter_values = jnp.asarray([1.0, lam1_gas, lam1_cond, lam2_cond], dtype=jnp.asarray(lam1_gas).dtype)
    limiter_names = ("unit", "lam1_gas", "lam1_cond", "lam2_cond")
    limiting_index = int(jnp.argmin(limiter_values))
    return {
        "policy_name": policy_name,
        "lam_cap": jnp.clip(lam_cap, 0.0, 1.0),
        "lam1_gas": lam1_gas,
        "lam1_cond": lam1_cond,
        "lam2_cond": lam2_cond,
        "production_limiting_index": jnp.asarray(limiting_index, dtype=jnp.int32),
        "production_limiting_name": limiter_names[limiting_index],
    }


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
    """Build a diagnostic-only RGIE condensate initialization from a startup policy.

    The returned vector is defined only on the currently supported condensates.
    This helper does not change production defaults; callers must explicitly opt
    into using it.
    """

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
        return jnp.full((n_support,), jnp.log(jnp.asarray(m0, dtype=dtype)), dtype=dtype)

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

    if startup_policy in ("ratio_positive_driving_r0", "ratio_topk_positive_driving_r0"):
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
                selected = jnp.zeros((n_support,), dtype=bool).at[top_indices].set(True)
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

    full_driving = jnp.asarray(full_driving, dtype=jnp.float64)
    support_indices = jnp.asarray(support_indices, dtype=jnp.int32)
    n_cond = int(full_driving.shape[0])
    support_mask = jnp.zeros((n_cond,), dtype=bool).at[support_indices].set(True)
    inactive_indices = jnp.nonzero(~support_mask, size=n_cond, fill_value=-1)[0]
    inactive_indices = inactive_indices[inactive_indices >= 0]

    if inactive_indices.shape[0] == 0:
        return {
            "max_positive_inactive_driving": 0.0,
            "inactive_positive_count": 0,
            "top_inactive_indices": [],
            "top_inactive_names": [],
            "top_inactive_driving": [],
            "top_positive_inactive_indices": [],
        }

    inactive_driving = full_driving[inactive_indices]
    positive_mask = inactive_driving > 0.0
    positive_indices = inactive_indices[positive_mask]
    positive_driving = inactive_driving[positive_mask]

    if positive_indices.shape[0] == 0:
        return {
            "max_positive_inactive_driving": 0.0,
            "inactive_positive_count": 0,
            "top_inactive_indices": [],
            "top_inactive_names": [],
            "top_inactive_driving": [],
            "top_positive_inactive_indices": [],
        }

    ranked_order = jnp.argsort(-positive_driving)
    ranked_positive = positive_indices[ranked_order]
    top_indices = ranked_positive[: min(int(top_k), int(ranked_positive.shape[0]))]
    top_driving = full_driving[top_indices]
    if condensate_species_names is None:
        top_names = [str(int(index)) for index in jax.device_get(top_indices)]
    else:
        top_names = [
            str(condensate_species_names[int(index)]) for index in jax.device_get(top_indices)
        ]

    return {
        "max_positive_inactive_driving": float(jnp.max(positive_driving)),
        "inactive_positive_count": int(positive_indices.shape[0]),
        "top_inactive_indices": [int(index) for index in jax.device_get(top_indices)],
        "top_inactive_names": top_names,
        "top_inactive_driving": [float(value) for value in jax.device_get(top_driving)],
        "top_positive_inactive_indices": [int(index) for index in jax.device_get(ranked_positive)],
    }


def build_rgie_support_candidate_indices(
    support_indices: jnp.ndarray,
    *,
    full_driving: jnp.ndarray,
    active_ln_mk: jnp.ndarray,
    active_driving: jnp.ndarray,
    mechanism_name: str,
    inactive_positive_ranked: Optional[Sequence[int]] = None,
    semismooth_add_top_k: int = 1,
    smoothed_add_top_k: int = 3,
) -> Dict[str, Any]:
    """Build diagnostic-only support candidates from active/inactive scores."""

    support_indices = jnp.asarray(support_indices, dtype=jnp.int32)
    full_driving = jnp.asarray(full_driving, dtype=jnp.float64)
    active_ln_mk = jnp.asarray(active_ln_mk, dtype=jnp.float64)
    active_driving = jnp.asarray(active_driving, dtype=jnp.float64)

    if active_ln_mk.shape != support_indices.shape or active_driving.shape != support_indices.shape:
        raise ValueError("active_ln_mk and active_driving must match support_indices shape.")

    if inactive_positive_ranked is None:
        inactive_positive_ranked = summarize_rgie_inactive_driving(
            full_driving,
            support_indices,
            top_k=full_driving.shape[0],
        )["top_positive_inactive_indices"]

    support_set = {int(index) for index in jax.device_get(support_indices)}
    ranked_add = [int(index) for index in inactive_positive_ranked if int(index) not in support_set]

    weak_active_order = jnp.argsort(-(active_driving + jnp.maximum(-active_ln_mk, 0.0)))
    weak_active_ranked = [int(index) for index in jax.device_get(support_indices[weak_active_order])]

    candidate = sorted(support_set)
    added_indices: list[int] = []
    dropped_indices: list[int] = []

    if mechanism_name == "current_support_updating_active_set":
        pass
    elif mechanism_name == "semismooth_candidate":
        added_indices = ranked_add[: max(0, int(semismooth_add_top_k))]
        candidate = sorted(set(candidate).union(added_indices))
    elif mechanism_name == "smoothed_semismooth_candidate":
        added_indices = ranked_add[: max(0, int(smoothed_add_top_k))]
        candidate = sorted(set(candidate).union(added_indices))
    elif mechanism_name == "augmented_semismooth_candidate":
        added_indices = ranked_add[:1]
        candidate = sorted(set(candidate).union(added_indices))
        if len(candidate) > 1 and weak_active_ranked:
            drop_index = weak_active_ranked[0]
            if drop_index in candidate and drop_index not in added_indices:
                candidate = [index for index in candidate if index != drop_index]
                dropped_indices = [drop_index]
    else:
        raise ValueError(
            "Unknown RGIE support candidate mechanism "
            f"'{mechanism_name}'. Expected one of "
            "('current_support_updating_active_set', 'semismooth_candidate', "
            "'smoothed_semismooth_candidate', 'augmented_semismooth_candidate')."
        )

    return {
        "mechanism_name": mechanism_name,
        "support_indices": jnp.asarray(candidate, dtype=jnp.int32),
        "added_indices": added_indices,
        "dropped_indices": dropped_indices,
        "inactive_positive_ranked": ranked_add,
        "weak_active_ranked": weak_active_ranked,
    }


def _assemble_reduced_system_terms(
    nk: jnp.ndarray,
    mk: jnp.ndarray,
    ntotk: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    bk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    sk: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Assemble the reduced condensed system once for backend experiments."""

    resn = jnp.sum(nk) - ntotk
    Qk = _A_diagn_At(nk, formula_matrix) + _A_diagn_At(sk, formula_matrix_cond)
    Angk = formula_matrix @ (gk * nk)
    ngk = jnp.dot(nk, gk)
    delta_bk_hat = b - (bk + formula_matrix_cond @ mk)
    condvec = formula_matrix_cond @ (sk * hvector_cond - mk)
    rhs = Angk + condvec + delta_bk_hat
    scalar_rhs = ngk - resn
    assemble_mat = jnp.block([[Qk, bk[:, None]], [bk[None, :], jnp.array([[resn]])]])
    assemble_vec = jnp.concatenate([rhs, jnp.array([scalar_rhs])])
    return {
        "resn": resn,
        "Qk": Qk,
        "rhs": rhs,
        "scalar_rhs": scalar_rhs,
        "assemble_mat": assemble_mat,
        "assemble_vec": assemble_vec,
    }


def _regularize_q_block(
    q_block: jnp.ndarray,
    regularization_mode: str,
    regularization_strength: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply explicit optional regularization to the Q block only."""

    dtype = q_block.dtype
    reg_used = (
        jnp.asarray(regularization_strength, dtype=dtype)
        if regularization_mode == "diag_shift"
        else jnp.asarray(0.0, dtype=dtype)
    )
    if regularization_mode == "none":
        return q_block, reg_used
    if regularization_mode != "diag_shift":
        raise ValueError(
            f"Unknown regularization_mode '{regularization_mode}'. Expected 'none' or 'diag_shift'."
        )
    eye = jnp.eye(q_block.shape[0], dtype=dtype)
    return q_block + reg_used * eye, reg_used


def _diagnostic_json_array(value: Any) -> list[Any]:
    """Return a JSON-safe copy of a diagnostic array."""

    return np.asarray(jax.device_get(value), dtype=np.float64).tolist()


def _build_ln_nk_producer_trace(
    *,
    ln_nk: jnp.ndarray,
    source_stage: str,
    producer_function: str,
    case_key: str,
    newton_iter: int,
) -> Dict[str, Any]:
    """Describe the diagnostic source boundary for an ln_nk handoff."""

    ln_nk_array = jnp.asarray(ln_nk)
    host_float64 = np.asarray(jax.device_get(ln_nk_array), dtype=np.float64)
    finite = np.isfinite(host_float64)
    double_min_log = float(np.log(float.fromhex("0x1p-1022")))
    return {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "case_key": case_key,
        "newton_iter": int(newton_iter),
        "source_stage": source_stage,
        "producer_function": producer_function,
        "consumer_function": "src/exogibbs/optimize/pipm_rgie_cond.py::_build_reduced_solver_exact_input_bundle",
        "ln_nk_dtype": str(ln_nk_array.dtype),
        "ln_nk_shape": [int(value) for value in ln_nk_array.shape],
        "native_longdouble_provenance_available": False,
        "preserves_native_longdouble_bits": False,
        "reconstructed_from_float64": True,
        "finite_count": int(np.count_nonzero(finite)),
        "below_double_normal_log_count": int(
            np.count_nonzero(finite & (host_float64 < double_min_log))
        ),
        "source_density_cgs_before_exp_or_normalization_available": False,
        "density_domain_scale_available": False,
        "floor_policy": "no pre-float64 source floor policy available at this boundary",
        "sample_log_values": host_float64[: min(16, host_float64.size)].tolist(),
    }


def _with_lnnk_source_state_trace(
    context: Optional[Dict[str, Any]],
    *,
    ln_nk: jnp.ndarray,
    source_stage: str,
    producer_function: str,
    iter_count: int,
) -> Optional[Dict[str, Any]]:
    """Attach a default-off ln_nk source trace to an exact-bundle context."""

    if context is None or not bool(context.get("emit_exact_input_bundle", False)):
        return context
    traced_context = dict(context)
    case_key = str(traced_context.get("case_key", "diagnostic"))
    target_iter = int(traced_context.get("newton_iter", iter_count))
    traced_context["ln_nk_producer_trace"] = _build_ln_nk_producer_trace(
        ln_nk=ln_nk,
        source_stage=source_stage,
        producer_function=producer_function,
        case_key=case_key,
        newton_iter=target_iter,
    )
    traced_context["ln_nk_source_state_trace"] = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "source_stage": source_stage,
        "producer_function": producer_function,
        "iter_count": int(iter_count),
        "native_longdouble_provenance_available": False,
        "preserves_native_longdouble_bits": False,
        "next_required_field": "caller-provided ln_nk_init or accepted line-search result before JAX float64 materialization",
    }
    return traced_context


def _diagnostic_source_state_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _build_kl_gas_phase_calculate_replay_results(
    *,
    formula_matrix: jnp.ndarray,
    initial_species_density_cgs: jnp.ndarray,
    recovered_hvector_gas: jnp.ndarray,
    element_labels: Optional[Sequence[str]],
    lifecycle_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Replay the observable GasPhase::calculate result boundary for diagnostics.

    This is deliberately a default-off diagnostic reconstruction.  It uses only
    KL-owned state at the reduced-solver boundary and static FastChem lifecycle
    rules; it does not consume FastChem trace values as constructor inputs.
    """

    formula = np.asarray(jax.device_get(formula_matrix), dtype=np.float64)
    initial_species_density_source_longdouble_cgs = np.asarray(
        jax.device_get(initial_species_density_cgs), dtype=np.longdouble
    ).copy()
    species_density = np.asarray(
        initial_species_density_source_longdouble_cgs, dtype=np.float64
    ).copy()
    hvector = np.asarray(jax.device_get(recovered_hvector_gas), dtype=np.float64)
    n_elements = int(formula.shape[0])
    n_species = int(formula.shape[1])
    labels = [] if element_labels is None else [str(label) for label in element_labels]
    options = dict(lifecycle_context.get("fastchem_options", {}))
    chem_accuracy = float(options.get("chem_accuracy", 1.0e-5))
    max_iter = int(options.get("nb_max_fastchem_iter", 3000))
    nb_switch_to_newton = int(options.get("nb_switch_to_newton", 400))
    use_backup_solver = bool(options.get("chem_use_backup_solver", False))
    density_floor = float(options.get("molecule_density_minlimit", 1.0e-155))
    if species_density.shape[0] != n_species:
        species_density = np.resize(species_density, n_species).astype(np.float64)
    if initial_species_density_source_longdouble_cgs.shape[0] != n_species:
        initial_species_density_source_longdouble_cgs = np.resize(
            initial_species_density_source_longdouble_cgs,
            n_species,
        ).astype(np.longdouble)
    density_domain_scale_cgs = lifecycle_context.get(
        "molecule_number_density_gauge_cgs",
        lifecycle_context.get("gas_number_density_cgs"),
    )
    pressure_bar_context = lifecycle_context.get("pressure_bar")
    temperature_context = lifecycle_context.get("temperature_K")
    if density_domain_scale_cgs is None:
        density_domain_scale = 1.0
        density_domain = "cgs_number_density"
    else:
        density_domain_scale = max(float(density_domain_scale_cgs), 1.0e-300)
        density_domain = "KL_dimensionless_density_scaled_by_gas_number_density_cgs"
    species_density_cgs_initial = species_density.copy()
    initial_species_density_source_longdouble_internal = np.where(
        np.isfinite(initial_species_density_source_longdouble_cgs)
        & (initial_species_density_source_longdouble_cgs > 0.0),
        initial_species_density_source_longdouble_cgs,
        np.longdouble(density_floor),
    ) / np.longdouble(density_domain_scale)
    initial_species_nonpositive = ~(
        np.isfinite(species_density_cgs_initial) & (species_density_cgs_initial > 0.0)
    )
    species_density = np.where(
        np.isfinite(species_density) & (species_density > 0.0),
        species_density,
        density_floor,
    ) / density_domain_scale
    initial_species_density_internal = species_density.copy()
    density_floor_internal = density_floor / density_domain_scale
    old_density = species_density.copy()
    electron_index = None
    for index, label in enumerate(labels):
        if label == "e-":
            electron_index = index
            break
    actual_electron_old_density = (
        None
        if electron_index is None
        else float(species_density_cgs_initial[electron_index])
    )
    backup_switch_iteration = None
    newton_iteration = None
    converged_iteration = None
    max_replay_iter = min(max_iter, max(nb_switch_to_newton + 2, 3))
    disable_replay_convergence_break = bool(
        lifecycle_context.get("disable_replay_convergence_break", False)
    )
    molecule_mass_action_constants = np.asarray(
        lifecycle_context.get(
            "molecule_mass_action_constants",
            (-hvector[n_elements:]).tolist(),
        ),
        dtype=np.float64,
    )
    if molecule_mass_action_constants.shape[0] != max(n_species - n_elements, 0):
        molecule_mass_action_constants = -hvector[n_elements:]
    mass_action_source = str(
        lifecycle_context.get(
            "molecule_mass_action_source",
            "explicit_or_raw_negative_recovered_hvector",
        )
    )
    if mass_action_source == "fastchem_pressure_scaled_from_hvector":
        molecule_formula = formula[:, n_elements:]
        sigma = 1.0 - np.sum(molecule_formula, axis=0)
        raw_lnK = -hvector[n_elements:]
        if density_domain_scale_cgs is not None and pressure_bar_context is not None:
            correction_log = np.log(max(float(pressure_bar_context), 1.0e-300))
            mass_action_correction_source = "raw_lnK - sigma * log(pressure_bar)"
        elif temperature_context is not None:
            correction_log = np.log(
                max(
                    1.0e-6 * float(KL_DENSITY_GAUGE_K_B_CGS) * float(temperature_context),
                    1.0e-300,
                )
            )
            mass_action_correction_source = (
                "raw_lnK - sigma * log(1e-6 * k_B * temperature)"
            )
        else:
            correction_log = 0.0
            mass_action_correction_source = (
                "raw_lnK; pressure correction unavailable in lifecycle context"
            )
        molecule_mass_action_constants = raw_lnK - sigma * correction_log
    else:
        mass_action_correction_source = mass_action_source
    element_solver_coefficient_mass_action_source = str(
        lifecycle_context.get(
            "element_solver_coefficient_mass_action_source",
            "molecule_mass_action_constants",
        )
    )
    coefficient_mass_action_constants = np.asarray(
        lifecycle_context.get(
            "element_solver_coefficient_mass_action_constants",
            molecule_mass_action_constants.tolist(),
        ),
        dtype=np.float64,
    )
    if coefficient_mass_action_constants.shape[0] != max(n_species - n_elements, 0):
        coefficient_mass_action_constants = molecule_mass_action_constants.copy()
    element_order_context = lifecycle_context.get("element_calculation_order", {})
    if isinstance(element_order_context, dict):
        element_order = [
            int(index)
            for index in element_order_context.get("indices", [])
            if 0 <= int(index) < n_elements
        ]
    else:
        element_order = []
    if not element_order:
        element_order = [
            index
            for index in range(n_elements)
            if index != electron_index
        ]
    element_order_positions = {
        int(index): int(position) for position, index in enumerate(element_order)
    }
    major_context = lifecycle_context.get("major_molecules_inc", {})
    minor_context = lifecycle_context.get("minor_molecules", {})

    def _molecule_index(raw_index: Any) -> Optional[int]:
        index = int(raw_index)
        if n_elements <= index < n_species:
            return index - n_elements
        if 0 <= index < n_species - n_elements:
            return index
        return None

    def _list_for_element(source: Any, element_index: int) -> list[int]:
        label = labels[element_index] if element_index < len(labels) else str(element_index)
        raw_values: Sequence[Any]
        if isinstance(source, dict):
            raw_values = source.get(label, source.get(str(element_index), []))
        else:
            raw_values = []
        out = []
        seen = set()
        for value in raw_values:
            mol_index = _molecule_index(value)
            if mol_index is not None and mol_index not in seen:
                out.append(mol_index)
                seen.add(mol_index)
        return out

    molecule_list_by_element = {
        element_index: [
            int(mol_index)
            for mol_index in np.where(formula[element_index, n_elements:] != 0.0)[0]
        ]
        for element_index in range(n_elements)
    }
    molecule_refresh_list_source = str(
        lifecycle_context.get("molecule_refresh_list_source", "lifecycle_context")
    )
    if molecule_refresh_list_source == "selected_source_from_element_abundance":
        major_molecules_by_element = {index: [] for index in range(n_elements)}
        minor_molecules_by_element = {index: [] for index in range(n_elements)}
        abundance_seed = lifecycle_context.get("element_abundance_vector")
        if abundance_seed is None:
            abundance_seed = lifecycle_context.get(
                "element_epsilon_vector",
                np.maximum(species_density[:n_elements], 0.0),
            )
        element_abundance_for_lists = np.asarray(abundance_seed, dtype=np.float64)
        if element_abundance_for_lists.shape[0] != n_elements:
            element_abundance_for_lists = np.resize(
                element_abundance_for_lists,
                n_elements,
            ).astype(np.float64)
        electron_excluded = (
            electron_index is not None
            and str(
                lifecycle_context.get(
                    "molecule_abundance_electron_policy",
                    "exclude_zero_abundance_electron",
                )
            )
            == "exclude_zero_abundance_electron"
        )
        for mol_index in range(max(n_species - n_elements, 0)):
            stoich = formula[:, n_elements + mol_index]
            active = np.where(stoich != 0.0)[0]
            if electron_excluded:
                active = active[active != electron_index]
            if active.size == 0:
                continue
            active_values = element_abundance_for_lists[active]
            if not np.any(np.isfinite(active_values)):
                continue
            molecule_abundance_value = float(np.nanmin(active_values))
            for element_index in active.tolist():
                element_value = float(element_abundance_for_lists[element_index])
                if not np.isfinite(element_value):
                    continue
                if element_value <= molecule_abundance_value:
                    major_molecules_by_element[element_index].append(int(mol_index))
                else:
                    minor_molecules_by_element[element_index].append(int(mol_index))
    else:
        major_molecules_by_element = {
            element_index: _list_for_element(major_context, element_index)
            for element_index in range(n_elements)
        }
        minor_molecules_by_element = {
            element_index: _list_for_element(minor_context, element_index)
            for element_index in range(n_elements)
        }
    molecule_input_trace_records: list[dict[str, Any]] = []
    electron_donor_trace_records: list[dict[str, Any]] = []
    element_solver_trace_records: list[dict[str, Any]] = []
    coefficient_source_value_trace_records: list[dict[str, Any]] = []
    minor_density_trace_records: list[dict[str, Any]] = []
    last_element_slot_write_lineage: list[Optional[dict[str, Any]]] = [
        None for _ in range(n_elements)
    ]
    latest_element_solver_log_materialization: list[Optional[dict[str, Any]]] = [
        None for _ in range(n_elements)
    ]
    molecule_refresh_position = 0
    current_replay_iteration = -1
    previous_iteration_electron_log_density_cgs: Optional[float] = None
    trace_product_element_order_context = lifecycle_context.get(
        "molecule_product_element_order_by_molecule",
        {},
    )

    def _molecule_refresh_electron_density_internal() -> Optional[float]:
        if electron_index is None:
            return None
        if molecule_refresh_electron_density_source == "current_no_floor":
            return float(species_density[electron_index])
        if molecule_refresh_electron_density_source == "old_density":
            return float(old_density[electron_index])
        if molecule_refresh_electron_density_source == "iteration_start":
            return float(electron_iteration_start_density[electron_index])
        if molecule_refresh_electron_density_source == "initial_species_density":
            return float(initial_species_density_internal[electron_index])
        if molecule_refresh_electron_density_source == "post_calculateElementDensities":
            return float(electron_post_element_density[electron_index])
        if molecule_refresh_electron_density_source == "post_molecule_refresh":
            return float(electron_post_molecule_density[electron_index])
        if molecule_refresh_electron_density_source == "post_minor_boundary":
            return float(electron_post_minor_boundary_density[electron_index])
        if molecule_refresh_electron_density_source == "post_electron_self_consistent":
            return float(species_density[electron_index])
        if molecule_refresh_electron_density_source == "lifecycle_entry_old_density":
            return float(actual_electron_old_density / density_domain_scale)
        return None

    def _logsumexp(values: list[float]) -> Optional[float]:
        finite = [float(value) for value in values if np.isfinite(value)]
        if not finite:
            return None
        top = max(finite)
        return float(top + np.log(np.sum(np.exp(np.asarray(finite) - top))))

    def _molecule_refresh_electron_log_density_cgs_internal(
        element_density: np.ndarray,
    ) -> Optional[float]:
        if (
            electron_index is None
            or molecule_refresh_electron_log_density_source
            not in {
                "analytic_singly_ion_from_product_elements",
                "refreshed_electron_log_density",
                "current_with_initial_longdouble_minlimit",
                "previous_iteration_refreshed_or_initial_longdouble",
            }
        ):
            return None
        if molecule_refresh_electron_log_density_source == "refreshed_electron_log_density":
            return electron_current_log_density_cgs
        if (
            molecule_refresh_electron_log_density_source
            == "previous_iteration_refreshed_or_initial_longdouble"
        ):
            return previous_iteration_electron_log_density_cgs
        if (
            molecule_refresh_electron_log_density_source
            == "current_with_initial_longdouble_minlimit"
        ):
            electron_density_cgs = float(
                element_density[electron_index] * density_domain_scale
            )
            initial_nonpositive = (
                0 <= electron_index < initial_species_nonpositive.shape[0]
                and bool(initial_species_nonpositive[electron_index])
            )
            initial_floor_placeholder = (
                initial_nonpositive
                and np.isfinite(element_density[electron_index])
                and abs(float(element_density[electron_index]) - density_floor_internal)
                <= max(abs(density_floor_internal), 1.0e-300) * 1.0e-12
            )
            if initial_floor_placeholder:
                return fastchem_longdouble_minlimit_log_cgs
            if np.isfinite(electron_density_cgs) and electron_density_cgs > 0.0:
                return float(np.log(electron_density_cgs))
            return fastchem_longdouble_minlimit_log_cgs
        charge = formula[electron_index, n_elements:]
        alpha_terms: list[float] = []
        beta_terms: list[float] = []
        for mol_index in molecule_list_by_element.get(electron_index, []):
            charge_number = int(round(charge[mol_index]))
            if abs(charge_number) != 1:
                continue
            stoich = molecule_formula[:, mol_index]
            active = (stoich != 0.0) & (np.arange(n_elements) != electron_index)
            if not np.any(active):
                continue
            values = element_density[active] * density_domain_scale
            if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
                values = np.maximum(values, 1.0e-300)
            with np.errstate(divide="ignore", invalid="ignore"):
                donor_logs = np.log(values)
            exponent = float(
                molecule_mass_action_constants[mol_index]
                + np.dot(stoich[active], donor_logs)
            )
            if charge_number == 1:
                beta_terms.append(exponent)
            elif charge_number == -1:
                alpha_terms.append(exponent)
        log_alpha = _logsumexp(alpha_terms)
        if log_alpha is None:
            return None
        log_beta = _logsumexp(beta_terms)
        log_denominator = 0.0 if log_beta is None else float(np.logaddexp(0.0, log_beta))
        log_electron = 0.5 * (log_alpha - log_denominator)
        return float(log_electron) if np.isfinite(log_electron) else None

    def _molecule_refresh_element_log_density_cgs_internal(
        element_density: np.ndarray,
        *,
        element_context: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        source = molecule_refresh_element_log_density_source
        if source == "density":
            return None
        if source == "current_density_log":
            return _log_from_internal_density_vector(species_density)
        if source == "native_current_vector_product_log_carrier":
            return _log_from_internal_density_vector(species_density)
        if source == "old_density_log":
            return _log_from_internal_density_vector(old_density)
        if source == "post_minor_refresh_density_log":
            return _log_from_internal_density_vector(post_minor_refresh_density)
        if source == "pre_major_refresh_density_log":
            return _log_from_internal_density_vector(element_density)
        if source == "cumulative_n_major_before_update_log":
            return _hhe_lifecycle_scalar_log_vector(
                molecule_refresh_n_major_before_update,
                element_density,
            )
        if source == "cumulative_n_major_after_update_log":
            return _hhe_lifecycle_scalar_log_vector(
                molecule_refresh_n_major_after_update,
                element_density,
            )
        if source == "returned_n_major_delta_log":
            return _hhe_lifecycle_scalar_log_vector(
                molecule_refresh_returned_n_major_delta,
                element_density,
            )
        if source == "pressure_scaled_hvector_density_log":
            return _log_from_internal_density_vector(element_density)
        if source == "element_solver_log_carrier":
            return element_solver_log_density_cgs.copy()
        if source == "element_solver_candidate_log_carrier":
            return element_solver_candidate_log_density_cgs.copy()
        if source == "context_element_solver_log_carrier":
            logs = _log_from_internal_density_vector(element_density)
            if element_context is not None and 0 <= int(element_context) < n_elements:
                logs[int(element_context)] = element_solver_log_density_cgs[
                    int(element_context)
                ]
            return logs
        if source == "context_element_solver_candidate_log_carrier":
            logs = _log_from_internal_density_vector(element_density)
            if element_context is not None and 0 <= int(element_context) < n_elements:
                logs[int(element_context)] = element_solver_candidate_log_density_cgs[
                    int(element_context)
                ]
            return logs
        if source == "abundance_gas_density":
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(
                    element_abundance > 0.0,
                    np.log(element_abundance) + np.log(density_domain_scale),
                    fastchem_longdouble_minlimit_log_cgs,
                ).astype(np.float64)
        if source == "max_density_element_solver_log_carrier":
            with np.errstate(divide="ignore", invalid="ignore"):
                density_logs = np.where(
                    element_density * density_domain_scale > 0.0,
                    np.log(element_density * density_domain_scale),
                    fastchem_longdouble_minlimit_log_cgs,
                )
            return np.maximum(density_logs, element_solver_log_density_cgs).astype(
                np.float64
            )
        if source == "max_density_element_solver_candidate_log_carrier":
            with np.errstate(divide="ignore", invalid="ignore"):
                density_logs = np.where(
                    element_density * density_domain_scale > 0.0,
                    np.log(element_density * density_domain_scale),
                    fastchem_longdouble_minlimit_log_cgs,
                )
            return np.maximum(
                density_logs,
                element_solver_candidate_log_density_cgs,
            ).astype(np.float64)
        return None

    def _molecule_refresh_product_element_log_density_cgs_internal(
        element_density: np.ndarray,
        element_log_density_cgs_override: Optional[np.ndarray],
        *,
        element_context: Optional[int] = None,
        source_override: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        source = (
            molecule_refresh_product_element_log_density_source
            if source_override is None
            else str(source_override)
        )
        if source in {"element_log_density_source", "disabled"}:
            return None
        if source == "current_density_log":
            return _log_from_internal_density_vector(species_density)
        if source == "native_current_vector_product_log_carrier":
            return _log_from_internal_density_vector(species_density)
        if source == "old_density_log":
            return _log_from_internal_density_vector(old_density)
        if source == "initial_species_density_log":
            return _log_from_internal_density_vector(initial_species_density_internal)
        if source == "fastchem_longdouble_minlimit_log":
            return np.full(
                n_elements,
                fastchem_longdouble_minlimit_log_cgs,
                dtype=np.float64,
            )
        if source == "min_current_initial_species_density_log":
            return np.minimum(
                _log_from_internal_density_vector(species_density),
                _log_from_internal_density_vector(initial_species_density_internal),
            ).astype(np.float64)
        if source == "min_current_old_density_log":
            return np.minimum(
                _log_from_internal_density_vector(species_density),
                _log_from_internal_density_vector(old_density),
            ).astype(np.float64)
        if source == "min_current_initial_old_density_log":
            return np.minimum(
                np.minimum(
                    _log_from_internal_density_vector(species_density),
                    _log_from_internal_density_vector(initial_species_density_internal),
                ),
                _log_from_internal_density_vector(old_density),
            ).astype(np.float64)
        if source == "current_or_initial_longdouble_minlimit_log":
            current_logs = _log_from_internal_density_vector(species_density)
            initial_logs = _log_from_internal_density_vector(
                initial_species_density_internal
            )
            return np.where(
                current_logs <= initial_logs,
                current_logs,
                fastchem_longdouble_minlimit_log_cgs,
            ).astype(np.float64)
        if source == "post_minor_refresh_density_log":
            return _log_from_internal_density_vector(post_minor_refresh_density)
        if source == "pre_major_refresh_density_log":
            return _log_from_internal_density_vector(element_density)
        if source == "element_log_density_override":
            if element_log_density_cgs_override is None:
                return None
            return np.asarray(element_log_density_cgs_override, dtype=np.float64).copy()
        if source == "current_element_solver_log_carrier":
            return element_solver_log_density_cgs.copy()
        if source == "candidate_element_solver_log_carrier":
            return element_solver_candidate_log_density_cgs.copy()
        if source == "previous_iteration_element_solver_log_carrier":
            return previous_iteration_element_solver_log_density_cgs.copy()
        if source == "iteration_start_element_solver_log_carrier":
            return iteration_start_element_solver_log_density_cgs.copy()
        if source == "post_element_solver_log_carrier":
            return post_element_solver_log_density_cgs.copy()
        if source == "post_element_solver_candidate_log_carrier":
            return post_element_solver_candidate_log_density_cgs.copy()
        if source == "post_element_solver_species_density_log":
            return post_element_solver_species_log_density_cgs.copy()
        if source == "post_molecule_feedback_log_carrier":
            return _log_from_internal_density_vector(species_density)
        if source == "last_solved_element_handoff_log_carrier":
            return last_solved_element_handoff_log_density_cgs.copy()
        if source == "last_solved_candidate_handoff_log_carrier":
            return last_solved_candidate_handoff_log_density_cgs.copy()
        if source == "max_current_previous_element_solver_log_carrier":
            return np.maximum(
                element_solver_log_density_cgs,
                previous_iteration_element_solver_log_density_cgs,
            ).astype(np.float64)
        if source in {
            "context_current_product_previous_element_solver_log_carrier",
            "context_candidate_product_previous_element_solver_log_carrier",
        }:
            logs = previous_iteration_element_solver_log_density_cgs.copy()
            if element_context is not None and 0 <= int(element_context) < n_elements:
                context_source = (
                    element_solver_candidate_log_density_cgs
                    if source
                    == "context_candidate_product_previous_element_solver_log_carrier"
                    else element_solver_log_density_cgs
                )
                logs[int(element_context)] = context_source[int(element_context)]
            return logs
        if source in {
            "context_post_element_product_last_solved_log_carrier",
            "context_post_candidate_product_last_solved_log_carrier",
            "v41_inferred_kl_owned_handoff_lifecycle_stage",
            "best_kl_owned_later_fs_product_element_handoff_candidate",
        }:
            logs = last_solved_element_handoff_log_density_cgs.copy()
            if element_context is not None and 0 <= int(element_context) < n_elements:
                context_source = (
                    post_element_solver_candidate_log_density_cgs
                    if source
                    == "context_post_candidate_product_last_solved_log_carrier"
                    else post_element_solver_log_density_cgs
                )
                logs[int(element_context)] = context_source[int(element_context)]
            return logs
        return None

    def _molecule_formula_label(mol_index: int) -> str:
        species_index = n_elements + int(mol_index)
        if not (n_elements <= species_index < n_species):
            return str(mol_index)
        stoich = formula[:, species_index]
        parts = []
        for element_index in np.where(stoich != 0.0)[0].tolist():
            label = labels[int(element_index)] if int(element_index) < len(labels) else str(element_index)
            parts.append(f"{label}{float(stoich[int(element_index)]):g}")
        return "".join(parts)

    def _refresh_molecule(
        mol_index: int,
        *,
        element_context: Optional[int] = None,
        refresh_kind: str = "unknown",
        n_major_before_update: Optional[float] = None,
        n_major_after_update: Optional[float] = None,
        returned_n_major_delta: Optional[float] = None,
        local_refresh_order_position: Optional[int] = None,
    ) -> float:
        nonlocal molecule_refresh_position
        nonlocal molecule_refresh_n_major_before_update
        nonlocal molecule_refresh_n_major_after_update
        nonlocal molecule_refresh_returned_n_major_delta
        nonlocal molecule_refresh_last_returned_n_major_delta
        nonlocal molecule_refresh_h_coefficient_log_consumed
        species_index = n_elements + mol_index
        if not (n_elements <= species_index < n_species):
            return 0.0
        previous_density = float(species_density[species_index])
        stoich = formula[:, species_index]
        active = stoich != 0.0
        if not np.any(active):
            return previous_density
        molecule_refresh_n_major_before_update = n_major_before_update
        molecule_refresh_n_major_after_update = n_major_after_update
        molecule_refresh_returned_n_major_delta = returned_n_major_delta
        current_density_snapshot = species_density.copy()
        if molecule_refresh_element_density_source == "old_density":
            element_density = old_density[:n_elements].copy()
        elif molecule_refresh_element_density_source == "initial_species_density":
            element_density = initial_species_density_internal[:n_elements].copy()
        elif molecule_refresh_element_density_source == "max_current_old_density":
            element_density = np.maximum(
                species_density[:n_elements],
                old_density[:n_elements],
            )
        elif molecule_refresh_element_density_source == "external_vector_cgs":
            element_density = molecule_refresh_external_element_density_internal.copy()
        else:
            element_density = species_density[:n_elements].copy()
        if element_solver_nonpositive_candidate_policy == "fastchem_zero_checkN":
            element_density = np.where(
                np.isfinite(element_density),
                element_density,
                density_floor_internal,
            )
        else:
            element_density = np.maximum(element_density, density_floor_internal)
        electron_density_override = _molecule_refresh_electron_density_internal()
        if (
            electron_density_override is not None
            and electron_index is not None
            and electron_index < element_density.shape[0]
        ):
            if molecule_refresh_electron_floor_policy == "allow_subfloor_positive":
                element_density[electron_index] = (
                    electron_density_override
                    if electron_density_override > 0.0
                    else density_floor_internal
                )
            else:
                element_density[electron_index] = max(
                    electron_density_override,
                    density_floor_internal,
                )
        electron_log_density_cgs_override = (
            _molecule_refresh_electron_log_density_cgs_internal(element_density)
        )
        element_log_density_cgs_override = (
            _molecule_refresh_element_log_density_cgs_internal(
                element_density,
                element_context=element_context,
            )
        )
        if (
            element_log_density_cgs_override is not None
            and electron_log_density_cgs_override is not None
            and electron_index is not None
            and electron_index < element_log_density_cgs_override.shape[0]
        ):
            element_log_density_cgs_override = element_log_density_cgs_override.copy()
            element_log_density_cgs_override[electron_index] = (
                electron_log_density_cgs_override
            )
        entry_mutable_product_sources = {
            "calculateMoleculeDensities_entry_mutable_element_vector_log",
            "entry_mutable_vector_context_solver_log",
            "entry_mutable_vector_context_candidate_log",
            "entry_mutable_vector_context_last_solved_log",
        }

        def _entry_mutable_product_log_override(source: str) -> np.ndarray:
            logs = _log_from_internal_density_vector(current_density_snapshot)
            if element_context is not None and 0 <= int(element_context) < n_elements:
                context_index = int(element_context)
                if source == "entry_mutable_vector_context_solver_log":
                    logs[context_index] = post_element_solver_log_density_cgs[
                        context_index
                    ]
                elif source == "entry_mutable_vector_context_candidate_log":
                    logs[context_index] = post_element_solver_candidate_log_density_cgs[
                        context_index
                    ]
                elif source == "entry_mutable_vector_context_last_solved_log":
                    logs[context_index] = last_solved_element_handoff_log_density_cgs[
                        context_index
                    ]
            return logs

        def _product_log_override_for_source(source: str) -> Optional[np.ndarray]:
            if source in entry_mutable_product_sources:
                return _entry_mutable_product_log_override(source)
            return _molecule_refresh_product_element_log_density_cgs_internal(
                element_density,
                element_log_density_cgs_override,
                element_context=element_context,
                source_override=source,
            )

        product_element_log_density_cgs_override = _product_log_override_for_source(
            molecule_refresh_product_element_log_density_source
        )
        product_log_carrier_selection_sources = [
            (
                molecule_refresh_product_element_log_density_source
                if product_element_log_density_cgs_override is not None
                else "element_log_density_source"
            )
            for _ in range(n_elements)
        ]
        product_log_carrier_selection_events: list[dict[str, Any]] = []

        def _trace_float_vector(values: Optional[np.ndarray]) -> Optional[list[float]]:
            if values is None:
                return None
            return [float(value) for value in np.asarray(values, dtype=np.float64)]

        def _trace_selected_values(
            values: Optional[np.ndarray],
            selected_indices: list[int],
        ) -> Optional[list[float]]:
            if values is None:
                return None
            return [
                float(values[int(index)])
                for index in selected_indices
                if 0 <= int(index) < len(values)
            ]

        def _trace_source_vector_bundle(
            selected_indices: list[int],
        ) -> dict[str, dict[str, Optional[list[float]]]]:
            source_map = {
                "native_current_vector_candidate_source": (
                    "native_current_vector_product_log_carrier"
                ),
                "initial_species_density_log_baseline_source": (
                    "initial_species_density_log"
                ),
                "entry_mutable_source": "entry_mutable_vector_context_solver_log",
                "element_solver_handoff_source": "post_element_solver_log_carrier",
                "molecule_feedback_source": "post_molecule_feedback_log_carrier",
            }
            bundle = {}
            for label, source_name in source_map.items():
                vector = _product_log_override_for_source(source_name)
                bundle[label] = {
                    "source_stage_label": source_name,
                    "full_vector_cgs": _trace_float_vector(vector),
                    "selected_values_cgs": _trace_selected_values(
                        vector,
                        selected_indices,
                    ),
                }
            return bundle

        def _record_product_log_carrier_selection(
            *,
            source: str,
            scope_type: str,
            selected_indices: list[int],
        ) -> Optional[dict[str, Any]]:
            if not selected_indices:
                return None
            for selected_index in selected_indices:
                if 0 <= int(selected_index) < n_elements:
                    product_log_carrier_selection_sources[int(selected_index)] = source
            event = {
                "source": source,
                "scope_type": scope_type,
                "selected_element_indices": [
                    int(index)
                    for index in selected_indices
                    if 0 <= int(index) < n_elements
                ],
                "selected_element_labels": [
                    (
                        labels[int(index)]
                        if int(index) < len(labels)
                        else str(index)
                    )
                    for index in selected_indices
                    if 0 <= int(index) < n_elements
                ],
                "KL_owned": True,
                "diagnostic_only": True,
                "default_off": True,
            }
            product_log_carrier_selection_events.append(event)
            return event

        def _ensure_product_element_log_override() -> np.ndarray:
            nonlocal product_element_log_density_cgs_override
            if product_element_log_density_cgs_override is not None:
                return product_element_log_density_cgs_override
            if element_log_density_cgs_override is not None:
                product_element_log_density_cgs_override = (
                    element_log_density_cgs_override.copy()
                )
            else:
                product_element_log_density_cgs_override = (
                    _log_from_internal_density_vector(element_density)
                )
                if (
                    electron_log_density_cgs_override is not None
                    and electron_index is not None
                    and electron_index
                    < product_element_log_density_cgs_override.shape[0]
                ):
                    product_element_log_density_cgs_override[electron_index] = (
                        electron_log_density_cgs_override
                    )
            return product_element_log_density_cgs_override

        def _apply_scoped_product_log_source(
            *,
            scoped_source: str,
            scoped_elements: set[str],
            scoped_iteration: Any,
        ) -> None:
            scoped_iteration_matches = (
                scoped_iteration is None
                or int(current_replay_iteration) == int(scoped_iteration)
            )
            if not scoped_iteration_matches:
                return
            scoped_logs = _product_log_override_for_source(str(scoped_source))
            if scoped_logs is None:
                return
            override = _ensure_product_element_log_override()
            selected_indices = []
            for scoped_index, scoped_label in enumerate(element_labels):
                scoped_element_matches = (
                    not scoped_elements
                    or str(scoped_label) in scoped_elements
                    or str(scoped_index) in scoped_elements
                )
                if scoped_element_matches:
                    override[scoped_index] = scoped_logs[scoped_index]
                    selected_indices.append(int(scoped_index))
            _record_product_log_carrier_selection(
                source=str(scoped_source),
                scope_type="element_scope",
                selected_indices=selected_indices,
            )

        def _scope_values(scope: dict[str, Any], key: str) -> set[str]:
            return {str(value) for value in scope.get(key, []) or []}

        def _family_scope_matches(scope: dict[str, Any]) -> bool:
            scoped_iteration = scope.get("iteration")
            if scoped_iteration is not None and int(current_replay_iteration) != int(
                scoped_iteration
            ):
                return False
            molecule_label = _molecule_formula_label(mol_index)
            molecule_labels = _scope_values(scope, "molecule_labels")
            if molecule_labels and molecule_label not in molecule_labels:
                return False
            molecule_indices = _scope_values(scope, "molecule_indices")
            if molecule_indices and str(int(mol_index)) not in molecule_indices:
                return False
            active_labels = {
                str(element_labels[int(index)]) for index in np.where(active)[0].tolist()
            }
            contains_any = _scope_values(scope, "molecule_contains_any")
            if contains_any and active_labels.isdisjoint(contains_any):
                return False
            contains_all = _scope_values(scope, "molecule_contains_all")
            if contains_all and not contains_all.issubset(active_labels):
                return False
            return True

        def _apply_family_scoped_product_log_source(scope: dict[str, Any]) -> None:
            scoped_source = scope.get("source")
            if scoped_source is None or not _family_scope_matches(scope):
                return
            scoped_logs = _product_log_override_for_source(str(scoped_source))
            if scoped_logs is None:
                return
            scoped_elements = _scope_values(scope, "elements")
            override = _ensure_product_element_log_override()
            pre_override_vector = override.copy()
            pre_source_vector = list(product_log_carrier_selection_sources)
            selected_indices = []
            for scoped_index, scoped_label in enumerate(element_labels):
                scoped_element_matches = (
                    not scoped_elements
                    or str(scoped_label) in scoped_elements
                    or str(scoped_index) in scoped_elements
                )
                if scoped_element_matches:
                    override[scoped_index] = scoped_logs[scoped_index]
                    selected_indices.append(int(scoped_index))
            event = _record_product_log_carrier_selection(
                source=str(scoped_source),
                scope_type="family_scope",
                selected_indices=selected_indices,
            )
            if event is not None:
                post_override_vector = override.copy()
                event.update(
                    {
                        "trace_marker": (
                            "exact_fixed_row_subspace_trace_m362_"
                            "kl_family_scope_product_log_source_application"
                        ),
                        "source_stage_label": (
                            "KL_family_scope_product_log_source_application"
                        ),
                        "producer_stage_label": (
                            "family_scope_override_before_product_log_consumption"
                        ),
                        "family_scope_application_order_position": int(
                            len(product_log_carrier_selection_events) - 1
                        ),
                        "selected_row_scope": {
                            "iteration": int(current_replay_iteration),
                            "molecule_index": int(mol_index),
                            "molecule_label": _molecule_formula_label(mol_index),
                            "element_context_index": (
                                None
                                if element_context is None
                                else int(element_context)
                            ),
                            "element_context_label": (
                                None
                                if element_context is None
                                or int(element_context) >= len(labels)
                                else labels[int(element_context)]
                            ),
                        },
                        "selected_family_scope": dict(scope),
                        "active_product_element_indices": [
                            int(index) for index in np.where(active)[0].tolist()
                        ],
                        "active_product_element_labels": [
                            (
                                labels[int(index)]
                                if int(index) < len(labels)
                                else str(index)
                            )
                            for index in np.where(active)[0].tolist()
                        ],
                        "pre_override_source_vector": pre_source_vector,
                        "post_override_source_vector": list(
                            product_log_carrier_selection_sources
                        ),
                        "pre_override_vector_cgs": _trace_float_vector(
                            pre_override_vector
                        ),
                        "post_override_vector_cgs": _trace_float_vector(
                            post_override_vector
                        ),
                        "pre_override_selected_values_cgs": _trace_selected_values(
                            pre_override_vector,
                            selected_indices,
                        ),
                        "post_override_selected_values_cgs": _trace_selected_values(
                            post_override_vector,
                            selected_indices,
                        ),
                        "candidate_source_vectors": _trace_source_vector_bundle(
                            selected_indices
                        ),
                        "per_element_application_order": [
                            {
                                "order": int(position),
                                "element_index": int(index),
                                "element_label": (
                                    labels[int(index)]
                                    if int(index) < len(labels)
                                    else str(index)
                                ),
                                "source_stage_label": str(scoped_source),
                                "producer_stage_label": (
                                    "family_scope_override_assignment"
                                ),
                            }
                            for position, index in enumerate(selected_indices)
                        ],
                        "FastChem_trace_values_used_as_KL_constructor_inputs": False,
                    }
                )

        if (
            molecule_refresh_product_element_log_density_source_scoped_source
            is not None
        ):
            scoped_iteration_matches = (
                molecule_refresh_product_element_log_density_source_iteration is None
                or int(current_replay_iteration)
                == int(molecule_refresh_product_element_log_density_source_iteration)
            )
            scoped_source = str(
                molecule_refresh_product_element_log_density_source_scoped_source
            )
            if scoped_iteration_matches:
                scoped_logs = _product_log_override_for_source(scoped_source)
                if scoped_logs is not None:
                    if product_element_log_density_cgs_override is None:
                        if element_log_density_cgs_override is not None:
                            product_element_log_density_cgs_override = (
                                element_log_density_cgs_override.copy()
                            )
                        else:
                            product_element_log_density_cgs_override = (
                                _log_from_internal_density_vector(element_density)
                            )
                            if (
                                electron_log_density_cgs_override is not None
                                and electron_index is not None
                                and electron_index
                                < product_element_log_density_cgs_override.shape[0]
                            ):
                                product_element_log_density_cgs_override[
                                    electron_index
                                ] = electron_log_density_cgs_override
                    selected_indices = []
                    for scoped_index, scoped_label in enumerate(element_labels):
                        scoped_element_matches = (
                            not molecule_refresh_product_element_log_density_source_elements
                            or str(scoped_label)
                            in molecule_refresh_product_element_log_density_source_elements
                            or str(scoped_index)
                            in molecule_refresh_product_element_log_density_source_elements
                        )
                        if scoped_element_matches:
                            product_element_log_density_cgs_override[
                                scoped_index
                            ] = scoped_logs[scoped_index]
                            selected_indices.append(int(scoped_index))
                    _record_product_log_carrier_selection(
                        source=scoped_source,
                        scope_type="single_scoped_source",
                        selected_indices=selected_indices,
                    )
        for product_log_scope in molecule_refresh_product_element_log_density_source_scopes:
            if not isinstance(product_log_scope, dict):
                continue
            scoped_source = product_log_scope.get("source")
            if scoped_source is None:
                continue
            scoped_elements = {
                str(value) for value in product_log_scope.get("elements", []) or []
            }
            _apply_scoped_product_log_source(
                scoped_source=str(scoped_source),
                scoped_elements=scoped_elements,
                scoped_iteration=product_log_scope.get("iteration"),
            )
        for product_log_scope in (
            molecule_refresh_product_element_log_density_family_scopes
        ):
            if isinstance(product_log_scope, dict):
                _apply_family_scoped_product_log_source(product_log_scope)
        h_coefficient_route_consumed = False
        h_coefficient_route_scope = molecule_refresh_h_coefficient_log_route
        active_indices = np.where(active)[0]
        h_index = (
            int(element_labels.index("H"))
            if "H" in element_labels and int(element_labels.index("H")) < n_elements
            else None
        )
        if (
            h_index is not None
            and bool(active[h_index])
            and h_coefficient_route_scope != "disabled"
        ):
            route_applies = False
            if h_coefficient_route_scope in {
                "all_h",
                "pre_checkn_all_h",
                "post_checkn_coefficient_preserving_all_h",
                "n_major_aware_all_h",
                "hhe_ordered_all_h",
            }:
                route_applies = True
            elif (
                h_coefficient_route_scope == "first_h"
                and not molecule_refresh_h_coefficient_log_consumed
            ):
                route_applies = True
            elif (
                h_coefficient_route_scope == "h2_only"
                and int(round(stoich[h_index])) == 2
                and np.count_nonzero(active) == 1
            ):
                route_applies = True
            if route_applies:
                candidate_log = float(element_solver_candidate_log_density_cgs[h_index])
                if np.isfinite(candidate_log):
                    if element_log_density_cgs_override is None:
                        element_log_density_cgs_override = _log_from_internal_density_vector(
                            element_density
                        )
                    else:
                        element_log_density_cgs_override = (
                            element_log_density_cgs_override.copy()
                        )
                    element_log_density_cgs_override[h_index] = candidate_log
                    h_coefficient_route_consumed = True
                    if h_coefficient_route_scope == "first_h":
                        molecule_refresh_h_coefficient_log_consumed = True
        electron_active_position = (
            np.where(active_indices == electron_index)[0]
            if electron_index is not None
            else np.asarray([], dtype=int)
        )
        active_log_density_source_switch_labels = [
            "unresolved" for _ in active_indices.tolist()
        ]
        if product_element_log_density_cgs_override is not None:
            active_log_density_internal_values = (
                product_element_log_density_cgs_override[active]
                - np.log(density_domain_scale)
            )
            active_log_density_source_switch_labels = [
                "product_element_log_density_cgs_override"
                for _ in active_indices.tolist()
            ]
        elif element_log_density_cgs_override is not None:
            active_log_density_internal_values = (
                element_log_density_cgs_override[active] - np.log(density_domain_scale)
            )
            active_log_density_source_switch_labels = [
                "element_log_density_cgs_override" for _ in active_indices.tolist()
            ]
        else:
            active_log_density_internal_values = np.log(
                np.maximum(element_density[active], density_floor_internal)
            )
            active_log_density_source_switch_labels = [
                "element_density_internal_log" for _ in active_indices.tolist()
            ]
            if (
                electron_log_density_cgs_override is not None
                and electron_index is not None
                and bool(active[electron_index])
            ):
                if electron_active_position.size:
                    active_log_density_internal_values = (
                        active_log_density_internal_values.copy()
                    )
                    active_log_density_internal_values[
                        int(electron_active_position[0])
                    ] = electron_log_density_cgs_override - np.log(
                        density_domain_scale
                    )
                    active_log_density_source_switch_labels[
                        int(electron_active_position[0])
                    ] = "electron_log_density_cgs_override"
        active_log_density_internal_by_index = {
            int(index): float(active_log_density_internal_values[position])
            for position, index in enumerate(active_indices.tolist())
        }
        active_log_density_source_switch_by_index = {
            int(index): active_log_density_source_switch_labels[position]
            for position, index in enumerate(active_indices.tolist())
        }

        def _active_log_density_cgs(index: int) -> float:
            if product_element_log_density_cgs_override is not None:
                return float(product_element_log_density_cgs_override[int(index)])
            if element_log_density_cgs_override is not None:
                return float(element_log_density_cgs_override[int(index)])
            if (
                electron_log_density_cgs_override is not None
                and electron_index is not None
                and int(index) == electron_index
            ):
                return float(electron_log_density_cgs_override)
            value_cgs = float(element_density[int(index)] * density_domain_scale)
            if (
                molecule_refresh_positive_log_floor_policy
                == "allow_positive_subfloor"
                and value_cgs > 0.0
            ):
                return float(np.log(value_cgs))
            return float(np.log(max(value_cgs, 1.0e-300)))
        if (
            element_solver_nonpositive_candidate_policy == "fastchem_zero_checkN"
            or molecule_refresh_positive_log_floor_policy == "allow_positive_subfloor"
            or electron_log_density_cgs_override is not None
            or element_log_density_cgs_override is not None
        ):
            with np.errstate(divide="ignore", invalid="ignore"):
                if (
                    product_element_log_density_cgs_override is not None
                    or electron_log_density_cgs_override is not None
                    or element_log_density_cgs_override is not None
                ):
                    log_density = float(
                        np.dot(stoich[active], active_log_density_internal_values)
                    )
                else:
                    log_density_values = np.where(
                        element_density[active] > 0.0,
                        element_density[active],
                        density_floor_internal,
                    )
                    log_density = float(
                        np.dot(stoich[active], np.log(log_density_values))
                    )
        else:
            log_density = float(
                np.dot(
                    stoich[active],
                    np.log(np.maximum(element_density[active], density_floor_internal)),
                )
            )
        log_density += float(molecule_mass_action_constants[mol_index])
        refreshed_molecule_log_density_cgs = float(
            log_density + np.log(density_domain_scale)
        )
        refreshed_molecule_density = max(
            float(np.exp(np.clip(log_density, -745.0, 709.0))),
            density_floor_internal,
        )
        if molecule_checkN_enabled:
            refreshed_molecule_density = min(
                max(float(refreshed_molecule_density), density_floor_internal),
                1.0,
            )
        post_refresh_returned_delta = float(
            refreshed_molecule_density * molecule_sigma[mol_index]
        )
        molecule_refresh_last_returned_n_major_delta = post_refresh_returned_delta
        effective_returned_n_major_delta = (
            post_refresh_returned_delta
            if molecule_refresh_returned_delta_timing
            == "post_refresh_molecule_density"
            else returned_n_major_delta
        )
        effective_n_major_after_update = (
            None
            if n_major_before_update is None
            or effective_returned_n_major_delta is None
            else float(n_major_before_update) + float(effective_returned_n_major_delta)
        )
        if (
            emit_molecule_input_trace
            and current_replay_iteration >= 0
            and current_replay_iteration < molecule_input_trace_iteration_limit
            and len(molecule_input_trace_records) < molecule_input_trace_max_records
        ):
            trace_active_indices = active_indices
            if isinstance(trace_product_element_order_context, dict):
                order_values = trace_product_element_order_context.get(
                    str(int(mol_index)),
                    trace_product_element_order_context.get(int(mol_index)),
                )
                if order_values is not None:
                    ordered = [
                        int(index)
                        for index in order_values
                        if 0 <= int(index) < n_elements
                        and bool(active[int(index)])
                    ]
                    if set(ordered) == set(active_indices.tolist()):
                        trace_active_indices = np.asarray(ordered, dtype=int)
            if molecule_refresh_positive_log_floor_policy == "allow_positive_subfloor":
                active_log_density = np.where(
                    element_density[active] > 0.0,
                    element_density[active],
                    density_floor_internal,
                )
            else:
                active_log_density = np.maximum(
                    element_density[active],
                    density_floor_internal,
                )
            if (
                electron_log_density_cgs_override is not None
                and electron_index is not None
                and bool(active[electron_index])
                and electron_active_position.size
            ):
                active_log_density = active_log_density.copy()
                active_log_density[int(electron_active_position[0])] = 1.0
            raw_log_density_internal = float(
                np.dot(
                    stoich[active],
                    active_log_density_internal_values,
                )
            )
            cgs_log_density_terms = [
                float(stoich[int(index)] * _active_log_density_cgs(int(index)))
                for index in trace_active_indices.tolist()
            ]
            pressure_scaled_hvector_density_cgs = [
                float(element_density[int(index)] * density_domain_scale)
                for index in trace_active_indices.tolist()
            ]
            abundance_gas_density_cgs = [
                float(element_abundance[int(index)] * density_domain_scale)
                for index in trace_active_indices.tolist()
            ]
            electron_coupled_density_cgs = [
                (
                    float(element_density[int(index)] * density_domain_scale)
                    if electron_index is None or int(index) != electron_index
                    else (
                        float(np.exp(electron_log_density_cgs_override))
                        if electron_log_density_cgs_override is not None
                        and np.isfinite(electron_log_density_cgs_override)
                        else float(element_density[int(index)] * density_domain_scale)
                    )
                )
                for index in trace_active_indices.tolist()
            ]
            current_iteration_element_log_vector_cgs = [
                float(element_solver_log_density_cgs[int(index)])
                for index in trace_active_indices.tolist()
            ]
            previous_iteration_element_log_vector_cgs = [
                float(previous_iteration_element_solver_log_density_cgs[int(index)])
                for index in trace_active_indices.tolist()
            ]
            previous_product_loop_source_vector_cgs = [
                _active_log_density_cgs(int(index))
                for index in trace_active_indices.tolist()
            ]
            post_element_solver_log_vector_cgs = [
                float(post_element_solver_log_density_cgs[int(index)])
                for index in trace_active_indices.tolist()
            ]
            post_molecule_feedback_log_vector_cgs = [
                float(
                    np.log(
                        max(
                            current_density_snapshot[int(index)]
                            * density_domain_scale,
                            1.0e-300,
                        )
                    )
                )
                for index in trace_active_indices.tolist()
            ]
            full_entry_mutable_element_log_vector_cgs = [
                float(
                    np.log(
                        max(
                            current_density_snapshot[int(index)]
                            * density_domain_scale,
                            1.0e-300,
                        )
                    )
                )
                for index in range(n_elements)
            ]
            full_current_mutable_element_log_vector_cgs = [
                float(
                    np.log(
                        max(
                            species_density[int(index)] * density_domain_scale,
                            1.0e-300,
                        )
                    )
                )
                for index in range(n_elements)
            ]
            full_product_element_log_density_vector_cgs = [
                _active_log_density_cgs(int(index)) for index in range(n_elements)
            ]
            full_refresh_entry_current_density_log_vector_cgs = [
                float(value)
                for value in _log_from_internal_density_vector(
                    current_density_snapshot
                )[:n_elements]
            ]
            full_carrier_selection_current_density_log_vector_cgs = [
                float(value)
                for value in _log_from_internal_density_vector(species_density)[
                    :n_elements
                ]
            ]
            native_current_value_construction_trace = []
            native_current_raw_producer_stage_vectors = {
                "initial_species_density": initial_species_density_internal,
                "old_density": old_density,
                "iteration_start_species_density": electron_iteration_start_density,
                "refresh_entry_current_density_snapshot": current_density_snapshot,
                "post_minor_refresh_density": post_minor_refresh_density,
                "current_species_density_before_carrier_construction": species_density,
            }
            native_current_raw_producer_log_vectors = {
                "current_element_solver_log_carrier": element_solver_log_density_cgs,
                "candidate_element_solver_log_carrier": (
                    element_solver_candidate_log_density_cgs
                ),
                "previous_iteration_element_solver_log_carrier": (
                    previous_iteration_element_solver_log_density_cgs
                ),
                "iteration_start_element_solver_log_carrier": (
                    iteration_start_element_solver_log_density_cgs
                ),
                "post_element_solver_log_carrier": post_element_solver_log_density_cgs,
                "post_element_solver_candidate_log_carrier": (
                    post_element_solver_candidate_log_density_cgs
                ),
                "post_element_solver_species_density_log": (
                    post_element_solver_species_log_density_cgs
                ),
                "last_solved_element_handoff_log_carrier": (
                    last_solved_element_handoff_log_density_cgs
                ),
                "last_solved_candidate_handoff_log_carrier": (
                    last_solved_candidate_handoff_log_density_cgs
                ),
            }
            for construction_index in range(n_elements):
                raw_internal_density = float(species_density[int(construction_index)])
                raw_cgs_density = float(raw_internal_density * density_domain_scale)
                finite_raw = bool(np.isfinite(raw_cgs_density))
                positive_raw = bool(finite_raw and raw_cgs_density > 0.0)
                floor_applied = bool(not positive_raw)
                subdouble_positive = bool(
                    positive_raw and raw_cgs_density < float(np.finfo(float).tiny)
                )
                constructed_log = full_carrier_selection_current_density_log_vector_cgs[
                    int(construction_index)
                ]
                raw_producer_stage_values = {}
                for stage_name, stage_vector in (
                    native_current_raw_producer_stage_vectors.items()
                ):
                    stage_internal = float(stage_vector[int(construction_index)])
                    stage_cgs = float(stage_internal * density_domain_scale)
                    stage_log = (
                        float(np.log(stage_cgs))
                        if np.isfinite(stage_cgs) and stage_cgs > 0.0
                        else fastchem_longdouble_minlimit_log_cgs
                    )
                    raw_producer_stage_values[stage_name] = {
                        "raw_internal_density": stage_internal,
                        "raw_cgs_density": stage_cgs,
                        "log_density_cgs": stage_log,
                        "finite_raw_density": bool(np.isfinite(stage_cgs)),
                        "positive_raw_density": bool(
                            np.isfinite(stage_cgs) and stage_cgs > 0.0
                        ),
                    }
                for stage_name, log_vector in (
                    native_current_raw_producer_log_vectors.items()
                ):
                    raw_producer_stage_values[stage_name] = {
                        "raw_internal_density": None,
                        "raw_cgs_density": None,
                        "log_density_cgs": float(log_vector[int(construction_index)]),
                        "finite_raw_density": None,
                        "positive_raw_density": None,
                    }
                native_current_value_construction_trace.append(
                    {
                        "element_index": int(construction_index),
                        "element_label": (
                            labels[int(construction_index)]
                            if int(construction_index) < len(labels)
                            else str(construction_index)
                        ),
                        "raw_internal_density": raw_internal_density,
                        "raw_cgs_density": raw_cgs_density,
                        "density_domain_scale": float(density_domain_scale),
                        "finite_raw_density": finite_raw,
                        "positive_raw_density": positive_raw,
                        "double_subnormal_positive_density": subdouble_positive,
                        "nonpositive_or_nonfinite_floor_applied": floor_applied,
                        "floor_value_cgs": 1.0e-300,
                        "fastchem_longdouble_minlimit_log_cgs": (
                            fastchem_longdouble_minlimit_log_cgs
                        ),
                        "raw_producer_stage_values": raw_producer_stage_values,
                        "selected_raw_producer_stage": (
                            "current_species_density_before_carrier_construction"
                        ),
                        "last_element_slot_write_lineage": (
                            last_element_slot_write_lineage[
                                int(construction_index)
                            ]
                        ),
                        "raw_producer_application_site": (
                            "src/exogibbs/optimize/pipm_rgie_cond.py::_refresh_molecule "
                            "before full_carrier_selection_current_density_log_vector_cgs "
                            "consumption"
                        ),
                        "constructed_log_density_cgs": constructed_log,
                        "construction_rule": (
                            "log(raw_cgs_density) if finite positive else log(1e-300)"
                        ),
                        "trace_marker": (
                            "exact_fixed_row_subspace_trace_m368_"
                            "kl_species_density_element_slot_write_lineage "
                            "exact_fixed_row_subspace_trace_m367_"
                            "kl_raw_native_current_vector_producer "
                            "exact_fixed_row_subspace_trace_m366_"
                            "kl_native_current_vector_value_construction"
                        ),
                        "KL_owned": True,
                        "diagnostic_only": True,
                        "default_off": True,
                        "FastChem_trace_values_used_as_KL_constructor_inputs": False,
                    }
                )
            full_element_density_log_vector_cgs = [
                float(value)
                for value in _log_from_internal_density_vector(element_density)[
                    :n_elements
                ]
            ]
            full_selected_product_log_carrier_value_vector_cgs = list(
                full_product_element_log_density_vector_cgs
            )
            product_log_decomposition_sources = {
                "selected_product_element_log": None,
                "current_density_log": "current_density_log",
                "native_current_vector_product_log_carrier": (
                    "native_current_vector_product_log_carrier"
                ),
                "entry_mutable_vector": (
                    "calculateMoleculeDensities_entry_mutable_element_vector_log"
                ),
                "entry_mutable_vector_context_solver": (
                    "entry_mutable_vector_context_solver_log"
                ),
                "entry_mutable_vector_context_candidate": (
                    "entry_mutable_vector_context_candidate_log"
                ),
                "entry_mutable_vector_context_last_solved": (
                    "entry_mutable_vector_context_last_solved_log"
                ),
                "previous_iteration_element_solver": (
                    "previous_iteration_element_solver_log_carrier"
                ),
                "post_element_solver": "post_element_solver_log_carrier",
                "post_molecule_feedback": "post_molecule_feedback_log_carrier",
                "context_post_element_product_last_solved": (
                    "context_post_element_product_last_solved_log_carrier"
                ),
                "context_post_candidate_product_last_solved": (
                    "context_post_candidate_product_last_solved_log_carrier"
                ),
                "fastchem_longdouble_minlimit": "fastchem_longdouble_minlimit_log",
                "min_current_initial": "min_current_initial_species_density_log",
                "min_current_old": "min_current_old_density_log",
                "min_current_initial_old": "min_current_initial_old_density_log",
                "current_or_initial_longdouble_minlimit": (
                    "current_or_initial_longdouble_minlimit_log"
                ),
            }
            product_log_source_vectors = {
                name: (
                    full_product_element_log_density_vector_cgs
                    if source is None
                    else _product_log_override_for_source(source)
                )
                for name, source in product_log_decomposition_sources.items()
            }
            direct_product_log_source_vectors = {
                "old_density_log": _log_from_internal_density_vector(old_density),
                "initial_species_density_log": _log_from_internal_density_vector(
                    initial_species_density_internal
                ),
                "post_minor_refresh_density_log": _log_from_internal_density_vector(
                    post_minor_refresh_density
                ),
                "pre_major_element_density_log": _log_from_internal_density_vector(
                    element_density
                ),
                "current_density_snapshot_log": _log_from_internal_density_vector(
                    current_density_snapshot
                ),
            }
            product_log_source_vectors.update(direct_product_log_source_vectors)
            native_current_timing_consistency = {
                "entry_snapshot_matches_carrier_selection_current_vector": bool(
                    np.array_equal(
                        np.asarray(
                            full_refresh_entry_current_density_log_vector_cgs,
                            dtype=float,
                        ),
                        np.asarray(
                            full_carrier_selection_current_density_log_vector_cgs,
                            dtype=float,
                        ),
                    )
                ),
                "entry_snapshot_matches_element_density_vector": bool(
                    np.array_equal(
                        np.asarray(
                            full_refresh_entry_current_density_log_vector_cgs,
                            dtype=float,
                        ),
                        np.asarray(
                            full_element_density_log_vector_cgs,
                            dtype=float,
                        ),
                    )
                ),
                "selected_product_carrier_matches_current_vector": bool(
                    np.array_equal(
                        np.asarray(
                            full_selected_product_log_carrier_value_vector_cgs,
                            dtype=float,
                        ),
                        np.asarray(
                            full_carrier_selection_current_density_log_vector_cgs,
                            dtype=float,
                        ),
                    )
                ),
                "selected_product_carrier_matches_entry_snapshot": bool(
                    np.array_equal(
                        np.asarray(
                            full_selected_product_log_carrier_value_vector_cgs,
                            dtype=float,
                        ),
                        np.asarray(
                            full_refresh_entry_current_density_log_vector_cgs,
                            dtype=float,
                        ),
                    )
                ),
                "selected_product_carrier_matches_element_density_vector": bool(
                    np.array_equal(
                        np.asarray(
                            full_selected_product_log_carrier_value_vector_cgs,
                            dtype=float,
                        ),
                        np.asarray(
                            full_element_density_log_vector_cgs,
                            dtype=float,
                        ),
                    )
                ),
            }
            product_element_log_source_decomposition = []
            product_log_carrier_selection_trace = []
            for index in trace_active_indices.tolist():
                source_values = {}
                for name, values in product_log_source_vectors.items():
                    if values is None or int(index) >= len(values):
                        source_values[name] = None
                    else:
                        source_values[name] = float(values[int(index)])
                native_current_timing_source_values = {
                    "refresh_entry_current_density_log": (
                        full_refresh_entry_current_density_log_vector_cgs[int(index)]
                    ),
                    "carrier_selection_current_density_log": (
                        full_carrier_selection_current_density_log_vector_cgs[
                            int(index)
                        ]
                    ),
                    "element_density_log": (
                        full_element_density_log_vector_cgs[int(index)]
                    ),
                    "selected_product_log_carrier": (
                        full_selected_product_log_carrier_value_vector_cgs[int(index)]
                    ),
                    "initial_species_density_log": float(
                        direct_product_log_source_vectors[
                            "initial_species_density_log"
                        ][int(index)]
                    ),
                    "old_density_log": float(
                        direct_product_log_source_vectors["old_density_log"][
                            int(index)
                        ]
                    ),
                    "native_current_value_construction": (
                        native_current_value_construction_trace[int(index)]
                    ),
                }
                native_current_timing_match_flags = {
                    "selected_matches_refresh_entry_current": bool(
                        native_current_timing_source_values[
                            "selected_product_log_carrier"
                        ]
                        == native_current_timing_source_values[
                            "refresh_entry_current_density_log"
                        ]
                    ),
                    "selected_matches_carrier_selection_current": bool(
                        native_current_timing_source_values[
                            "selected_product_log_carrier"
                        ]
                        == native_current_timing_source_values[
                            "carrier_selection_current_density_log"
                        ]
                    ),
                    "selected_matches_element_density": bool(
                        native_current_timing_source_values[
                            "selected_product_log_carrier"
                        ]
                        == native_current_timing_source_values["element_density_log"]
                    ),
                    "entry_current_matches_carrier_selection_current": bool(
                        native_current_timing_source_values[
                            "refresh_entry_current_density_log"
                        ]
                        == native_current_timing_source_values[
                            "carrier_selection_current_density_log"
                        ]
                    ),
                    "entry_current_matches_element_density": bool(
                        native_current_timing_source_values[
                            "refresh_entry_current_density_log"
                        ]
                        == native_current_timing_source_values["element_density_log"]
                    ),
                }
                selected_value_path_source_values = {
                    "selected_product_log_carrier": (
                        full_selected_product_log_carrier_value_vector_cgs[int(index)]
                    ),
                    "selected_carrier_source": product_log_carrier_selection_sources[
                        int(index)
                    ],
                }
                for source_name in (
                    "current_density_log",
                    "initial_species_density_log",
                    "old_density_log",
                    "fastchem_longdouble_minlimit",
                    "min_current_initial",
                    "min_current_old",
                    "min_current_initial_old",
                    "current_or_initial_longdouble_minlimit",
                ):
                    source_vector = product_log_source_vectors.get(source_name)
                    selected_value_path_source_values[source_name] = (
                        None
                        if source_vector is None or int(index) >= len(source_vector)
                        else float(source_vector[int(index)])
                    )
                selected_value_path_match_flags = {}
                selected_value = selected_value_path_source_values[
                    "selected_product_log_carrier"
                ]
                for source_name, source_value in selected_value_path_source_values.items():
                    if source_name in {
                        "selected_product_log_carrier",
                        "selected_carrier_source",
                    }:
                        continue
                    selected_value_path_match_flags[
                        f"selected_matches_{source_name}"
                    ] = bool(source_value is not None and selected_value == source_value)
                context_index = (
                    None if element_context is None else int(element_context)
                )
                product_index = int(index)
                entry_mutable_value_producer_trace = {
                    "trace_marker": (
                        "exact_fixed_row_subspace_trace_m357_"
                        "kl_entry_mutable_value_producer_lifecycle"
                    ),
                    "source_stage_label": (
                        "KL_refresh_molecule_entry_mutable_vector_before_"
                        "selected_product_log_carrier"
                    ),
                    "producer_stage_label": (
                        "entry_mutable_vector_context_resolution_before_"
                        "product_log_consumption"
                    ),
                    "product_element_index": product_index,
                    "product_element_label": (
                        labels[product_index]
                        if product_index < len(labels)
                        else str(product_index)
                    ),
                    "element_context_index": context_index,
                    "element_context_label": (
                        None
                        if context_index is None or context_index >= len(labels)
                        else labels[context_index]
                    ),
                    "selected_carrier_source": product_log_carrier_selection_sources[
                        product_index
                    ],
                    "selected_product_log_carrier": selected_value,
                    "entry_mutable_snapshot_log": (
                        full_entry_mutable_element_log_vector_cgs[product_index]
                    ),
                    "entry_mutable_snapshot_density_cgs": float(
                        current_density_snapshot[product_index] * density_domain_scale
                    ),
                    "entry_mutable_context_solver_log": float(
                        product_log_source_vectors[
                            "entry_mutable_vector_context_solver"
                        ][product_index]
                    ),
                    "entry_mutable_context_candidate_log": float(
                        product_log_source_vectors[
                            "entry_mutable_vector_context_candidate"
                        ][product_index]
                    ),
                    "entry_mutable_context_last_solved_log": float(
                        product_log_source_vectors[
                            "entry_mutable_vector_context_last_solved"
                        ][product_index]
                    ),
                    "carrier_selection_current_density_log": (
                        full_carrier_selection_current_density_log_vector_cgs[
                            product_index
                        ]
                    ),
                    "element_density_log": (
                        full_element_density_log_vector_cgs[product_index]
                    ),
                    "post_element_solver_log": float(
                        post_element_solver_log_density_cgs[product_index]
                    ),
                    "post_element_solver_candidate_log": float(
                        post_element_solver_candidate_log_density_cgs[product_index]
                    ),
                    "last_solved_element_handoff_log": float(
                        last_solved_element_handoff_log_density_cgs[product_index]
                    ),
                    "entry_context_applies_to_product_element": bool(
                        context_index is not None and context_index == product_index
                    ),
                    "selected_matches_entry_mutable_snapshot": bool(
                        selected_value
                        == full_entry_mutable_element_log_vector_cgs[product_index]
                    ),
                    "selected_matches_entry_context_solver": bool(
                        selected_value
                        == product_log_source_vectors[
                            "entry_mutable_vector_context_solver"
                        ][product_index]
                    ),
                    "selected_matches_entry_context_candidate": bool(
                        selected_value
                        == product_log_source_vectors[
                            "entry_mutable_vector_context_candidate"
                        ][product_index]
                    ),
                    "selected_matches_entry_context_last_solved": bool(
                        selected_value
                        == product_log_source_vectors[
                            "entry_mutable_vector_context_last_solved"
                        ][product_index]
                    ),
                    "KL_owned": True,
                    "diagnostic_only": True,
                    "default_off": True,
                    "FastChem_trace_values_used_as_KL_constructor_inputs": False,
                }
                element_solver_handoff_value_producer_trace = {
                    "trace_marker": (
                        "exact_fixed_row_subspace_trace_m358_"
                        "kl_element_solver_handoff_value_producer_lifecycle"
                    ),
                    "source_stage_label": (
                        "KL_element_solver_handoff_before_entry_mutable_vector"
                    ),
                    "producer_stage_label": (
                        "element_solver_current_candidate_post_last_solved_handoff"
                    ),
                    "product_element_index": product_index,
                    "product_element_label": (
                        labels[product_index]
                        if product_index < len(labels)
                        else str(product_index)
                    ),
                    "element_context_index": context_index,
                    "element_context_label": (
                        None
                        if context_index is None or context_index >= len(labels)
                        else labels[context_index]
                    ),
                    "selected_carrier_source": product_log_carrier_selection_sources[
                        product_index
                    ],
                    "selected_product_log_carrier": selected_value,
                    "entry_mutable_context_solver_log": (
                        entry_mutable_value_producer_trace[
                            "entry_mutable_context_solver_log"
                        ]
                    ),
                    "element_solver_log_density_cgs": float(
                        element_solver_log_density_cgs[product_index]
                    ),
                    "element_solver_candidate_log_density_cgs": float(
                        element_solver_candidate_log_density_cgs[product_index]
                    ),
                    "post_element_solver_log_density_cgs": float(
                        post_element_solver_log_density_cgs[product_index]
                    ),
                    "post_element_solver_candidate_log_density_cgs": float(
                        post_element_solver_candidate_log_density_cgs[product_index]
                    ),
                    "previous_iteration_element_solver_log_density_cgs": float(
                        previous_iteration_element_solver_log_density_cgs[
                            product_index
                        ]
                    ),
                    "iteration_start_element_solver_log_density_cgs": float(
                        iteration_start_element_solver_log_density_cgs[product_index]
                    ),
                    "last_solved_element_handoff_log_density_cgs": float(
                        last_solved_element_handoff_log_density_cgs[product_index]
                    ),
                    "last_solved_candidate_handoff_log_density_cgs": float(
                        last_solved_candidate_handoff_log_density_cgs[product_index]
                    ),
                    "context_post_element_product_last_solved_log": float(
                        product_log_source_vectors[
                            "context_post_element_product_last_solved"
                        ][product_index]
                    )
                    if "context_post_element_product_last_solved"
                    in product_log_source_vectors
                    else None,
                    "context_post_candidate_product_last_solved_log": float(
                        product_log_source_vectors[
                            "context_post_candidate_product_last_solved"
                        ][product_index]
                    )
                    if "context_post_candidate_product_last_solved"
                    in product_log_source_vectors
                    else None,
                    "handoff_context_applies_to_product_element": bool(
                        context_index is not None and context_index == product_index
                    ),
                    "selected_matches_element_solver_log": bool(
                        selected_value == element_solver_log_density_cgs[product_index]
                    ),
                    "selected_matches_element_solver_candidate_log": bool(
                        selected_value
                        == element_solver_candidate_log_density_cgs[product_index]
                    ),
                    "selected_matches_post_element_solver_log": bool(
                        selected_value
                        == post_element_solver_log_density_cgs[product_index]
                    ),
                    "selected_matches_post_element_solver_candidate_log": bool(
                        selected_value
                        == post_element_solver_candidate_log_density_cgs[
                            product_index
                        ]
                    ),
                    "selected_matches_last_solved_element_handoff_log": bool(
                        selected_value
                        == last_solved_element_handoff_log_density_cgs[product_index]
                    ),
                    "selected_matches_previous_iteration_element_solver_log": bool(
                        selected_value
                        == previous_iteration_element_solver_log_density_cgs[
                            product_index
                        ]
                    ),
                    "KL_owned": True,
                    "diagnostic_only": True,
                    "default_off": True,
                    "FastChem_trace_values_used_as_KL_constructor_inputs": False,
                }
                molecule_feedback_lifecycle_value_producer_trace = {
                    "trace_marker": (
                        "exact_fixed_row_subspace_trace_m359_"
                        "kl_molecule_feedback_lifecycle_value_producer"
                    ),
                    "source_stage_label": (
                        "KL_molecule_feedback_lifecycle_before_element_solver_handoff"
                    ),
                    "producer_stage_label": (
                        "species_density_current_snapshot_and_post_refresh_molecule_feedback"
                    ),
                    "product_element_index": product_index,
                    "product_element_label": (
                        labels[product_index]
                        if product_index < len(labels)
                        else str(product_index)
                    ),
                    "element_context_index": context_index,
                    "element_context_label": (
                        None
                        if context_index is None or context_index >= len(labels)
                        else labels[context_index]
                    ),
                    "selected_carrier_source": product_log_carrier_selection_sources[
                        product_index
                    ],
                    "selected_product_log_carrier": selected_value,
                    "current_density_snapshot_log": float(
                        direct_product_log_source_vectors[
                            "current_density_snapshot_log"
                        ][product_index]
                    ),
                    "current_species_density_log": float(
                        full_carrier_selection_current_density_log_vector_cgs[
                            product_index
                        ]
                    ),
                    "post_molecule_feedback_log": float(
                        product_log_source_vectors["post_molecule_feedback"][
                            product_index
                        ]
                    ),
                    "post_minor_refresh_density_log": float(
                        direct_product_log_source_vectors[
                            "post_minor_refresh_density_log"
                        ][product_index]
                    ),
                    "old_density_log": float(
                        direct_product_log_source_vectors["old_density_log"][
                            product_index
                        ]
                    ),
                    "initial_species_density_log": float(
                        direct_product_log_source_vectors[
                            "initial_species_density_log"
                        ][product_index]
                    ),
                    "previous_molecule_density": float(previous_density),
                    "refreshed_molecule_density": float(refreshed_molecule_density),
                    "refreshed_molecule_log_density_cgs": float(
                        refreshed_molecule_log_density_cgs
                    ),
                    "old_returned_n_major_delta": float(
                        previous_density * molecule_sigma[mol_index]
                    ),
                    "post_refresh_returned_n_major_delta": float(
                        post_refresh_returned_delta
                    ),
                    "molecule_sigma": float(molecule_sigma[mol_index]),
                    "molecule_feedback_density_source": molecule_feedback_density_source,
                    "molecule_refresh_returned_delta_timing": (
                        molecule_refresh_returned_delta_timing
                    ),
                    "selected_matches_current_density_snapshot_log": bool(
                        selected_value
                        == direct_product_log_source_vectors[
                            "current_density_snapshot_log"
                        ][product_index]
                    ),
                    "selected_matches_current_species_density_log": bool(
                        selected_value
                        == full_carrier_selection_current_density_log_vector_cgs[
                            product_index
                        ]
                    ),
                    "selected_matches_post_molecule_feedback_log": bool(
                        selected_value
                        == product_log_source_vectors["post_molecule_feedback"][
                            product_index
                        ]
                    ),
                    "selected_matches_post_minor_refresh_density_log": bool(
                        selected_value
                        == direct_product_log_source_vectors[
                            "post_minor_refresh_density_log"
                        ][product_index]
                    ),
                    "selected_matches_old_density_log": bool(
                        selected_value
                        == direct_product_log_source_vectors["old_density_log"][
                            product_index
                        ]
                    ),
                    "selected_matches_initial_species_density_log": bool(
                        selected_value
                        == direct_product_log_source_vectors[
                            "initial_species_density_log"
                        ][product_index]
                    ),
                    "KL_owned": True,
                    "diagnostic_only": True,
                    "default_off": True,
                    "FastChem_trace_values_used_as_KL_constructor_inputs": False,
                }
                selected_product_log_source_switch_trace = {
                    "trace_marker": (
                        "exact_fixed_row_subspace_trace_m360_"
                        "kl_selected_product_log_source_switch_before_active_log_density"
                    ),
                    "source_stage_label": (
                        "KL_selected_product_log_carrier_source_switch_immediately_"
                        "before_active_log_density_internal_values"
                    ),
                    "producer_stage_label": (
                        "active_log_density_internal_values_source_switch"
                    ),
                    "product_element_index": product_index,
                    "product_element_label": (
                        labels[product_index]
                        if product_index < len(labels)
                        else str(product_index)
                    ),
                    "element_context_index": context_index,
                    "element_context_label": (
                        None
                        if context_index is None or context_index >= len(labels)
                        else labels[context_index]
                    ),
                    "selected_carrier_source": product_log_carrier_selection_sources[
                        product_index
                    ],
                    "selected_product_log_carrier": selected_value,
                    "switch_selected_source_label": (
                        active_log_density_source_switch_by_index.get(product_index)
                    ),
                    "switch_active_log_density_internal_value": (
                        active_log_density_internal_by_index.get(product_index)
                    ),
                    "switch_active_log_density_cgs_value": (
                        None
                        if product_index not in active_log_density_internal_by_index
                        else float(
                            active_log_density_internal_by_index[product_index]
                            + np.log(density_domain_scale)
                        )
                    ),
                    "product_override_present": (
                        product_element_log_density_cgs_override is not None
                    ),
                    "element_override_present": (
                        element_log_density_cgs_override is not None
                    ),
                    "electron_override_applies": bool(
                        electron_log_density_cgs_override is not None
                        and electron_index is not None
                        and product_index == electron_index
                    ),
                    "product_override_value_cgs": (
                        None
                        if product_element_log_density_cgs_override is None
                        else float(product_element_log_density_cgs_override[product_index])
                    ),
                    "element_override_value_cgs": (
                        None
                        if element_log_density_cgs_override is None
                        else float(element_log_density_cgs_override[product_index])
                    ),
                    "element_density_internal_log_cgs": float(
                        np.log(
                            max(
                                element_density[product_index] * density_domain_scale,
                                1.0e-300,
                            )
                        )
                    ),
                    "active_log_density_matches_selected_carrier": bool(
                        product_index in active_log_density_internal_by_index
                        and selected_value
                        == float(
                            active_log_density_internal_by_index[product_index]
                            + np.log(density_domain_scale)
                        )
                    ),
                    "active_log_density_matches_fastchem_native_current_candidate": bool(
                        product_index in active_log_density_internal_by_index
                        and float(
                            active_log_density_internal_by_index[product_index]
                            + np.log(density_domain_scale)
                        )
                        == full_carrier_selection_current_density_log_vector_cgs[
                            product_index
                        ]
                    ),
                    "KL_owned": True,
                    "diagnostic_only": True,
                    "default_off": True,
                    "FastChem_trace_values_used_as_KL_constructor_inputs": False,
                }
                product_element_log_source_decomposition.append(
                    {
                        "product_element_index": int(index),
                        "product_element_label": (
                            labels[int(index)]
                            if int(index) < len(labels)
                            else str(index)
                        ),
                        "source_values_cgs": source_values,
                        "selected_value_path_source_values_cgs": (
                            selected_value_path_source_values
                        ),
                        "selected_value_path_match_flags": (
                            selected_value_path_match_flags
                        ),
                        "entry_mutable_value_producer_trace": (
                            entry_mutable_value_producer_trace
                        ),
                        "element_solver_handoff_value_producer_trace": (
                            element_solver_handoff_value_producer_trace
                        ),
                        "molecule_feedback_lifecycle_value_producer_trace": (
                            molecule_feedback_lifecycle_value_producer_trace
                        ),
                        "selected_product_log_source_switch_trace": (
                            selected_product_log_source_switch_trace
                        ),
                        "configured_product_element_log_density_source": (
                            molecule_refresh_product_element_log_density_source
                        ),
                        "KL_owned": True,
                        "diagnostic_only": True,
                        "default_off": True,
                    }
                )
                selected_carrier_source = product_log_carrier_selection_sources[
                    int(index)
                ]
                if product_element_log_density_cgs_override is None:
                    if element_log_density_cgs_override is not None:
                        selected_carrier_source = "element_log_density_override"
                    elif (
                        electron_log_density_cgs_override is not None
                        and electron_index is not None
                        and int(index) == electron_index
                    ):
                        selected_carrier_source = "electron_log_density_override"
                    else:
                        selected_carrier_source = "element_density_log"
                product_log_carrier_selection_trace.append(
                    {
                        "product_element_index": int(index),
                        "product_element_label": (
                            labels[int(index)]
                            if int(index) < len(labels)
                            else str(index)
                        ),
                        "selected_carrier_source": selected_carrier_source,
                        "configured_product_element_log_density_source": (
                            molecule_refresh_product_element_log_density_source
                        ),
                        "carrier_selection_stage": (
                            "after_scoped_product_log_source_selection_before_"
                            "product_log_consumption"
                        ),
                        "product_log_consumer_stage": (
                            "mass_action_product_element_log_dot"
                        ),
                        "product_log_override_present": (
                            product_element_log_density_cgs_override is not None
                        ),
                        "native_current_timing_source_values_cgs": (
                            native_current_timing_source_values
                        ),
                        "native_current_timing_match_flags": (
                            native_current_timing_match_flags
                        ),
                        "selected_value_path_source_values_cgs": (
                            selected_value_path_source_values
                        ),
                        "selected_value_path_match_flags": (
                            selected_value_path_match_flags
                        ),
                        "entry_mutable_value_producer_trace": (
                            entry_mutable_value_producer_trace
                        ),
                        "element_solver_handoff_value_producer_trace": (
                            element_solver_handoff_value_producer_trace
                        ),
                        "molecule_feedback_lifecycle_value_producer_trace": (
                            molecule_feedback_lifecycle_value_producer_trace
                        ),
                        "selected_product_log_source_switch_trace": (
                            selected_product_log_source_switch_trace
                        ),
                        "element_log_override_present": (
                            element_log_density_cgs_override is not None
                        ),
                        "electron_log_override_present": (
                            electron_log_density_cgs_override is not None
                        ),
                        "KL_owned": True,
                        "diagnostic_only": True,
                        "default_off": True,
                    }
                )
            previous_n_major_returned_delta_coupled_vector = {
                "n_major_cumulative_before_update": (
                    None
                    if n_major_before_update is None
                    else float(n_major_before_update)
                ),
                "caller_returned_n_major_delta": (
                    None
                    if returned_n_major_delta is None
                    else float(returned_n_major_delta)
                ),
                "old_returned_n_major_delta": float(
                    previous_density * molecule_sigma[mol_index]
                ),
            }
            kl_owned_previous_lifecycle_fields = {
                "current_iteration_element_log_vector_cgs": True,
                "previous_iteration_element_log_vector_cgs": True,
                "previous_product_loop_source_vector_cgs": True,
                "post_element_solver_log_vector_cgs": True,
                "post_molecule_feedback_log_vector_cgs": True,
                "full_entry_mutable_element_log_vector_cgs": True,
                "full_current_mutable_element_log_vector_cgs": True,
                "full_product_element_log_density_vector_cgs": True,
                "full_refresh_entry_current_density_log_vector_cgs": True,
                "full_carrier_selection_current_density_log_vector_cgs": True,
                "full_element_density_log_vector_cgs": True,
                "full_selected_product_log_carrier_value_vector_cgs": True,
                "native_current_timing_consistency": True,
                "selected_value_path_source_values_cgs": True,
                "selected_value_path_match_flags": True,
                "entry_mutable_value_producer_trace": True,
                "element_solver_handoff_value_producer_trace": True,
                "molecule_feedback_lifecycle_value_producer_trace": True,
                "selected_product_log_source_switch_trace": True,
                "product_element_log_source_decomposition": True,
                "product_log_carrier_selection_trace": True,
                "previous_n_major_returned_delta_coupled_vector": True,
                "previous_molecule_input_source": True,
                "active_element_log_vector_cgs": True,
                "mass_action_raw_sum_before_constant_cgs": True,
                "refreshed_molecule_density": True,
            }
            molecule_input_trace_records.append(
                {
                    "iteration": int(current_replay_iteration),
                    "refresh_order_position": int(molecule_refresh_position),
                    "local_molecule_refresh_order_position": (
                        None
                        if local_refresh_order_position is None
                        else int(local_refresh_order_position)
                    ),
                    "refresh_kind": refresh_kind,
                    "element_context_index": (
                        None if element_context is None else int(element_context)
                    ),
                    "element_context_label": (
                        None
                        if element_context is None
                        or element_context >= len(labels)
                        else labels[element_context]
                    ),
                    "molecule_index": int(mol_index),
                    "molecule_label": _molecule_formula_label(mol_index),
                    "molecule_formula": _molecule_formula_label(mol_index),
                    "molecule_table_order": int(mol_index),
                    "species_index": int(species_index),
                    "product_element_indices": [
                        int(index) for index in trace_active_indices.tolist()
                    ],
                    "product_element_labels": [
                        labels[int(index)] if int(index) < len(labels) else str(index)
                        for index in trace_active_indices.tolist()
                    ],
                    "full_element_indices": [int(index) for index in range(n_elements)],
                    "full_element_labels": [
                        labels[int(index)] if int(index) < len(labels) else str(index)
                        for index in range(n_elements)
                    ],
                    "stoichiometric_coefficients": [
                        float(stoich[int(index)])
                        for index in trace_active_indices.tolist()
                    ],
                    "element_number_densities_cgs": [
                        float(element_density[int(index)] * density_domain_scale)
                        for index in trace_active_indices.tolist()
                    ],
                    "current_element_number_densities_cgs": [
                        float(
                            current_density_snapshot[int(index)]
                            * density_domain_scale
                        )
                        for index in trace_active_indices.tolist()
                    ],
                    "old_element_number_densities_cgs": [
                        float(old_density[int(index)] * density_domain_scale)
                        for index in trace_active_indices.tolist()
                    ],
                    "post_minor_element_number_densities_cgs": [
                        float(
                            post_minor_refresh_density[int(index)]
                            * density_domain_scale
                        )
                        for index in trace_active_indices.tolist()
                    ],
                    "pre_major_element_number_densities_cgs": [
                        float(element_density[int(index)] * density_domain_scale)
                        for index in trace_active_indices.tolist()
                    ],
                    "old_molecule_density": float(previous_density),
                    "old_returned_n_major_delta": float(
                        previous_density * molecule_sigma[mol_index]
                    ),
                    "post_refresh_molecule_density": float(
                        refreshed_molecule_density
                    ),
                    "n_major_cumulative_before_update": (
                        None
                        if n_major_before_update is None
                        else float(n_major_before_update)
                    ),
                    "returned_n_major_delta": (
                        None
                        if effective_returned_n_major_delta is None
                        else float(effective_returned_n_major_delta)
                    ),
                    "n_major_cumulative_after_update": (
                        None
                        if effective_n_major_after_update is None
                        else float(effective_n_major_after_update)
                    ),
                    "caller_n_major_cumulative_after_update": (
                        None
                        if n_major_after_update is None
                        else float(n_major_after_update)
                    ),
                    "post_refresh_returned_n_major_delta": (
                        post_refresh_returned_delta
                    ),
                    "molecule_refresh_returned_delta_timing": (
                        molecule_refresh_returned_delta_timing
                    ),
                    "molecule_refresh_n_major_trace_scope": (
                        molecule_refresh_n_major_trace_scope
                    ),
                    "pressure_scaled_hvector_density_cgs": (
                        pressure_scaled_hvector_density_cgs
                    ),
                    "abundance_gas_density_cgs": abundance_gas_density_cgs,
                    "electron_coupled_density_cgs": electron_coupled_density_cgs,
                    "current_iteration_element_log_vector_cgs": (
                        current_iteration_element_log_vector_cgs
                    ),
                    "previous_iteration_element_log_vector_cgs": (
                        previous_iteration_element_log_vector_cgs
                    ),
                    "previous_product_loop_source_vector_cgs": (
                        previous_product_loop_source_vector_cgs
                    ),
                    "post_element_solver_log_vector_cgs": (
                        post_element_solver_log_vector_cgs
                    ),
                    "post_molecule_feedback_log_vector_cgs": (
                        post_molecule_feedback_log_vector_cgs
                    ),
                    "full_entry_mutable_element_log_vector_cgs": (
                        full_entry_mutable_element_log_vector_cgs
                    ),
                    "full_current_mutable_element_log_vector_cgs": (
                        full_current_mutable_element_log_vector_cgs
                    ),
                    "full_product_element_log_density_vector_cgs": (
                        full_product_element_log_density_vector_cgs
                    ),
                    "full_refresh_entry_current_density_log_vector_cgs": (
                        full_refresh_entry_current_density_log_vector_cgs
                    ),
                    "full_carrier_selection_current_density_log_vector_cgs": (
                        full_carrier_selection_current_density_log_vector_cgs
                    ),
                    "full_element_density_log_vector_cgs": (
                        full_element_density_log_vector_cgs
                    ),
                    "full_selected_product_log_carrier_value_vector_cgs": (
                        full_selected_product_log_carrier_value_vector_cgs
                    ),
                    "native_current_timing_consistency": (
                        native_current_timing_consistency
                    ),
                    "full_entry_current_mutable_element_vector_trace": {
                        "available": True,
                        "reference_only": True,
                        "used_as_constructor_input": False,
                        "width": int(n_elements),
                    },
                    "product_element_log_source_decomposition": (
                        product_element_log_source_decomposition
                    ),
                    "full_product_log_carrier_selection_source": list(
                        product_log_carrier_selection_sources
                    ),
                    "product_log_carrier_selection_trace": (
                        product_log_carrier_selection_trace
                    ),
                    "product_log_carrier_selection_events": (
                        product_log_carrier_selection_events
                    ),
                    "previous_n_major_returned_delta_coupled_vector": (
                        previous_n_major_returned_delta_coupled_vector
                    ),
                    "previous_molecule_input_source": (
                        molecule_refresh_product_element_log_density_source
                    ),
                    "active_element_log_vector_cgs": (
                        previous_product_loop_source_vector_cgs
                    ),
                    "kl_owned_previous_lifecycle_fields": (
                        kl_owned_previous_lifecycle_fields
                    ),
                    "log_element_number_densities_cgs": [
                        _active_log_density_cgs(int(index))
                        for index in trace_active_indices.tolist()
                    ],
                    "stoich_log_density_terms_cgs": cgs_log_density_terms,
                    "raw_log_density_sum_before_mass_action_constant_cgs": float(
                        np.sum(cgs_log_density_terms)
                    ),
                    "raw_log_density_sum_before_mass_action_constant_internal": (
                        raw_log_density_internal
                    ),
                    "molecule_mass_action_constant": float(
                        molecule_mass_action_constants[mol_index]
                    ),
                    "molecule_mass_action_source": mass_action_source,
                    "molecule_mass_action_correction_source": (
                        mass_action_correction_source
                    ),
                    "molecule_number_density_gauge_cgs": float(
                        density_domain_scale
                    ),
                    "electron_density_source": (
                        molecule_refresh_electron_density_source
                    ),
                    "electron_log_density_source": (
                        molecule_refresh_electron_log_density_source
                    ),
                    "element_log_density_source": (
                        molecule_refresh_element_log_density_source
                    ),
                    "product_element_log_density_source": (
                        molecule_refresh_product_element_log_density_source
                    ),
                    "h_coefficient_log_route": h_coefficient_route_scope,
                    "h_coefficient_log_route_consumed": bool(
                        h_coefficient_route_consumed
                    ),
                    "h_coefficient_candidate_log_cgs": (
                        None
                        if h_index is None
                        else float(element_solver_candidate_log_density_cgs[h_index])
                    ),
                    "h_post_checkn_log_cgs": (
                        None
                        if h_index is None
                        else float(element_solver_log_density_cgs[h_index])
                    ),
                    "density_domain": density_domain,
                    "cgs_domain_scale": float(density_domain_scale),
                    "internal_domain_scale": 1.0,
                    "density_domain_classification": {
                        "density_domain": density_domain,
                        "cgs_domain": "number_density_cgs",
                        "internal_domain": "density_domain_scaled_internal",
                    },
                    "molecule_refresh_hhe_lifecycle_element_scope": (
                        molecule_refresh_hhe_lifecycle_element_scope
                    ),
                    "diagnostic_only": True,
                    "default_off": True,
                    "FastChem_trace_values_used_as_inputs": False,
                }
            )
        molecule_refresh_position += 1
        species_density[species_index] = refreshed_molecule_density
        if 0 <= int(mol_index) < molecule_log_density_cgs.shape[0]:
            molecule_log_density_cgs[int(mol_index)] = (
                refreshed_molecule_log_density_cgs
            )
        return previous_density

    molecule_formula = formula[:, n_elements:]
    molecule_sigma = 1.0 - np.sum(molecule_formula, axis=0)
    phi_context = lifecycle_context.get(
        "element_epsilon_vector",
        lifecycle_context.get("element_phi_vector"),
    )
    if phi_context is None:
        phi = species_density[:n_elements].copy()
        phi = phi / max(float(np.sum(phi)), 1.0e-300)
    else:
        phi = np.asarray(phi_context, dtype=np.float64)
        if phi.shape[0] != n_elements:
            phi = np.resize(phi, n_elements).astype(np.float64)
        if str(lifecycle_context.get("element_phi_normalization", "sum_normalize")) == "preserve":
            phi = np.asarray(phi, dtype=np.float64)
        else:
            phi = phi / max(float(np.sum(phi)), 1.0e-300)
    element_solver_lifecycle_enabled = bool(
        lifecycle_context.get("element_solver_lifecycle_enabled", False)
    )
    element_solver_mode = str(
        lifecycle_context.get(
            "element_solver_mode",
            "regular_branch_only" if element_solver_lifecycle_enabled else "disabled",
        )
    )
    element_solver_exponent_clip_context = lifecycle_context.get(
        "element_solver_exponent_clip",
        709.0,
    )
    element_solver_exponent_clip = (
        math.inf
        if str(element_solver_exponent_clip_context) in {"none", "native", "inf"}
        else float(element_solver_exponent_clip_context)
    )
    element_solver_exponential_mode = str(
        lifecycle_context.get("element_solver_exponential_mode", "clipped")
    )
    element_solver_quadratic_branch_mode = str(
        lifecycle_context.get("element_solver_quadratic_branch_mode", "guarded")
    )
    element_solver_newton_policy = str(
        lifecycle_context.get("element_solver_newton_policy", "bounded_legacy")
    )
    element_solver_newton_fallback_policy = str(
        lifecycle_context.get("element_solver_newton_fallback_policy", "disabled")
    )
    element_solver_newton_initial_guess_policy = str(
        lifecycle_context.get(
            "element_solver_newton_initial_guess_policy",
            "density_floor_clamped",
        )
    )
    element_solver_newton_assignment_policy = str(
        lifecycle_context.get(
            "element_solver_newton_assignment_policy",
            "always_use_root",
        )
    )
    element_solver_newton_derivative_zero_policy = str(
        lifecycle_context.get(
            "element_solver_newton_derivative_zero_policy",
            "guard_break",
        )
    )
    if element_solver_newton_policy == "fastchem_options":
        element_solver_newton_max_iter = int(options.get("nb_max_newton_iter", 3000))
        element_solver_newton_err = float(options.get("newton_err", 1.0e-5))
        element_solver_bisection_max_iter = int(
            options.get("nb_max_bisection_iter", 3000)
        )
    else:
        element_solver_newton_max_iter = int(
            lifecycle_context.get("element_solver_newton_max_iter", 8)
        )
        element_solver_newton_err = float(
            lifecycle_context.get("element_solver_newton_err", 1.0e-8)
        )
        element_solver_bisection_max_iter = int(
            lifecycle_context.get("element_solver_bisection_max_iter", 0)
        )
    use_solver_scaling_factor = bool(
        options.get(
            "use_scaling_factor",
            lifecycle_context.get("element_solver_use_scaling_factor", False),
        )
    )
    additional_solver_scaling_factor = float(
        options.get(
            "additional_scaling_factor",
            lifecycle_context.get("element_solver_additional_scaling_factor", 0.0),
        )
    )
    element_density_minlimit = float(
        options.get("element_density_minlimit", density_floor_internal)
    )
    element_solver_nonpositive_candidate_policy = str(
        lifecycle_context.get(
            "element_solver_nonpositive_candidate_policy",
            "fallback_to_previous",
        )
    )
    element_checkN_upper_bound_internal = float(
        lifecycle_context.get("element_checkN_upper_bound_internal", 1.0)
    )
    element_solver_gas_density_internal = float(
        lifecycle_context.get("element_solver_gas_density_internal", 1.0)
    )
    element_solver_coefficient_molecule_source = str(
        lifecycle_context.get(
            "element_solver_coefficient_molecule_source",
            "major_molecules_inc",
        )
    )
    element_solver_coefficient_abundance_gate = bool(
        lifecycle_context.get("element_solver_coefficient_abundance_gate", False)
    )
    element_solver_donor_log_policy = str(
        lifecycle_context.get("element_solver_donor_log_policy", "floor_clamped")
    )
    element_solver_coefficient_donor_log_source = str(
        lifecycle_context.get("element_solver_coefficient_donor_log_source", "density")
    )
    element_solver_coefficient_donor_log_source_scopes = list(
        lifecycle_context.get("element_solver_coefficient_donor_log_source_scopes", [])
    )
    element_solver_order_source = str(
        lifecycle_context.get("element_solver_order_source", "major_molecules_inc")
    )
    element_solver_scaling_factor_molecule_source = str(
        lifecycle_context.get(
            "element_solver_scaling_factor_molecule_source",
            "major_molecules_inc",
        )
    )
    element_abundance_context = lifecycle_context.get("element_abundance_vector")
    if element_abundance_context is None:
        element_abundance = np.asarray(phi, dtype=np.float64).copy()
    else:
        element_abundance = np.asarray(element_abundance_context, dtype=np.float64)
        if element_abundance.shape[0] != n_elements:
            element_abundance = np.resize(element_abundance, n_elements).astype(
                np.float64
            )
    molecule_abundance_rule = str(
        lifecycle_context.get("molecule_abundance_rule", "min_nonzero")
    )
    molecule_abundance_electron_policy = str(
        lifecycle_context.get(
            "molecule_abundance_electron_policy",
            "exclude_zero_abundance_electron",
        )
    )
    element_solver_coefficient_density_domain = str(
        lifecycle_context.get(
            "element_solver_coefficient_density_domain",
            "internal",
        )
    )
    coefficient_density_scale = (
        density_domain_scale
        if element_solver_coefficient_density_domain == "cgs"
        else 1.0
    )
    coefficient_density_floor = (
        density_floor
        if element_solver_coefficient_density_domain == "cgs"
        else density_floor_internal
    )
    coefficient_gas_density = (
        density_domain_scale
        if element_solver_coefficient_density_domain == "cgs"
        else element_solver_gas_density_internal
    )
    molecule_checkN_enabled = bool(
        lifecycle_context.get("molecule_checkN_enabled", False)
    )
    electron_refresh_mode = str(
        lifecycle_context.get("electron_refresh_mode", "standard")
    )
    electron_refresh_timing = str(
        lifecycle_context.get(
            "electron_refresh_timing",
            (
                "after_element_molecule_refresh"
                if electron_refresh_mode == "immediate_after_element"
                else "post_minor_boundary"
            ),
        )
    )
    electron_refresh_donor_state_source = str(
        lifecycle_context.get("electron_refresh_donor_state_source", "current")
    )
    electron_refresh_fixed_point_max_iter = int(
        lifecycle_context.get("electron_refresh_fixed_point_max_iter", 32)
    )
    electron_refresh_fixed_point_damping = float(
        lifecycle_context.get("electron_refresh_fixed_point_damping", 1.0)
    )
    electron_refresh_density_floor_policy = str(
        lifecycle_context.get("electron_refresh_density_floor_policy", "floor")
    )
    electron_refresh_donor_log_density_source = str(
        lifecycle_context.get("electron_refresh_donor_log_density_source", "density")
    )
    electron_refresh_output_value_policy = str(
        lifecycle_context.get("electron_refresh_output_value_policy", "sqrt_alpha_beta")
    )
    element_solver_output_log_policy = str(
        lifecycle_context.get("element_solver_output_log_policy", "assigned_density")
    )
    element_solver_subdouble_log_root_policy = str(
        lifecycle_context.get("element_solver_subdouble_log_root_policy", "disabled")
    )
    element_solver_subdouble_log_root_iteration_limit = int(
        lifecycle_context.get("element_solver_subdouble_log_root_iteration_limit", -1)
    )
    element_solver_fixed_by_condensation_policy = str(
        lifecycle_context.get(
            "element_solver_fixed_by_condensation_policy",
            "clamp_current",
        )
    )
    element_solver_fixed_checkn_min_policy = str(
        lifecycle_context.get("element_solver_fixed_checkn_min_policy", "floor")
    )
    element_solver_minor_density_source = str(
        lifecycle_context.get("element_solver_minor_density_source", "minor_molecules")
    )
    element_solver_minor_density_source_scoped_source = lifecycle_context.get(
        "element_solver_minor_density_source_scoped_source"
    )
    element_solver_minor_density_source_elements = {
        str(value)
        for value in lifecycle_context.get(
            "element_solver_minor_density_source_elements",
            [],
        )
    }
    element_solver_minor_density_source_iteration = lifecycle_context.get(
        "element_solver_minor_density_source_iteration"
    )
    element_solver_minor_density_term_override_policy = str(
        lifecycle_context.get(
            "element_solver_minor_density_term_override_policy",
            "disabled",
        )
    )
    element_solver_minor_density_term_override_molecules = {
        str(value)
        for value in lifecycle_context.get(
            "element_solver_minor_density_term_override_molecules",
            [],
        )
    }
    element_solver_minor_density_term_override_elements = {
        str(value)
        for value in lifecycle_context.get(
            "element_solver_minor_density_term_override_elements",
            [],
        )
    }
    element_solver_minor_density_term_override_iteration = lifecycle_context.get(
        "element_solver_minor_density_term_override_iteration"
    )
    molecule_feedback_density_source = str(
        lifecycle_context.get("molecule_feedback_density_source", "density")
    )
    element_solver_quadratic_precision = str(
        lifecycle_context.get("element_solver_quadratic_precision", "float64")
    )
    element_solver_signed_log_term_selection = str(
        lifecycle_context.get("element_solver_signed_log_term_selection", "all")
    )
    element_solver_signed_log_trace_top_n = int(
        lifecycle_context.get("element_solver_signed_log_trace_top_n", 8)
    )
    molecule_refresh_electron_density_source = str(
        lifecycle_context.get("molecule_refresh_electron_density_source", "current")
    )
    molecule_refresh_electron_log_density_source = str(
        lifecycle_context.get("molecule_refresh_electron_log_density_source", "density")
    )
    molecule_refresh_element_log_density_source = str(
        lifecycle_context.get("molecule_refresh_element_log_density_source", "density")
    )
    molecule_refresh_product_element_log_density_source = str(
        lifecycle_context.get(
            "molecule_refresh_product_element_log_density_source",
            "element_log_density_source",
        )
    )
    molecule_refresh_product_element_log_density_source_scoped_source = (
        lifecycle_context.get(
            "molecule_refresh_product_element_log_density_source_scoped_source"
        )
    )
    molecule_refresh_product_element_log_density_source_elements = {
        str(value)
        for value in lifecycle_context.get(
            "molecule_refresh_product_element_log_density_source_elements",
            [],
        )
    }
    molecule_refresh_product_element_log_density_source_iteration = (
        lifecycle_context.get(
            "molecule_refresh_product_element_log_density_source_iteration"
        )
    )
    molecule_refresh_product_element_log_density_source_scopes = list(
        lifecycle_context.get(
            "molecule_refresh_product_element_log_density_source_scopes",
            [],
        )
        or []
    )
    molecule_refresh_product_element_log_density_family_scopes = list(
        lifecycle_context.get(
            "molecule_refresh_product_element_log_density_family_scopes",
            [],
        )
        or []
    )
    molecule_refresh_h_coefficient_log_route = str(
        lifecycle_context.get("molecule_refresh_h_coefficient_log_route", "disabled")
    )
    molecule_refresh_hhe_lifecycle_element_scope = str(
        lifecycle_context.get(
            "molecule_refresh_hhe_lifecycle_element_scope",
            "all",
        )
    )
    molecule_refresh_returned_delta_timing = str(
        lifecycle_context.get("molecule_refresh_returned_delta_timing", "caller")
    )
    molecule_major_delta_source = str(
        lifecycle_context.get("molecule_major_delta_source", "feedback_density")
    )
    molecule_major_delta_source_elements = {
        str(value)
        for value in lifecycle_context.get(
            "molecule_major_delta_source_elements",
            [],
        )
    }
    molecule_major_delta_source_molecules = {
        str(value)
        for value in lifecycle_context.get(
            "molecule_major_delta_source_molecules",
            [],
        )
    }
    molecule_major_delta_source_iteration = lifecycle_context.get(
        "molecule_major_delta_source_iteration"
    )
    molecule_refresh_n_major_trace_scope = str(
        lifecycle_context.get("molecule_refresh_n_major_trace_scope", "outer")
    )
    molecule_refresh_electron_floor_policy = str(
        lifecycle_context.get("molecule_refresh_electron_floor_policy", "floor")
    )
    molecule_refresh_positive_log_floor_policy = str(
        lifecycle_context.get("molecule_refresh_positive_log_floor_policy", "floor")
    )
    replay_old_density_update_timing = str(
        lifecycle_context.get(
            "replay_old_density_update_timing",
            "iteration_start_previous",
        )
    )
    molecule_refresh_element_density_source = str(
        lifecycle_context.get("molecule_refresh_element_density_source", "current")
    )
    external_element_density = lifecycle_context.get(
        "molecule_refresh_element_density_vector_cgs"
    )
    if external_element_density is None:
        molecule_refresh_external_element_density_internal = (
            initial_species_density_internal[:n_elements].copy()
        )
    else:
        molecule_refresh_external_element_density_internal = np.asarray(
            external_element_density,
            dtype=np.float64,
        )
        if molecule_refresh_external_element_density_internal.shape[0] != n_elements:
            molecule_refresh_external_element_density_internal = np.resize(
                molecule_refresh_external_element_density_internal,
                n_elements,
            ).astype(np.float64)
        molecule_refresh_external_element_density_internal = (
            molecule_refresh_external_element_density_internal / density_domain_scale
        )
    n_major_update_timing = str(
        lifecycle_context.get("n_major_update_timing", "post_update")
    )
    n_major_include_policy = str(
        lifecycle_context.get("n_major_include_policy", "all_element_rows")
    )
    element_solver_n_major_source = str(
        lifecycle_context.get("element_solver_n_major_source", "cumulative")
    )
    element_solver_n_major_update_source = str(
        lifecycle_context.get("element_solver_n_major_update_source", "molecule_refresh")
    )
    per_element_n_major_timing_replay_enabled = bool(
        lifecycle_context.get("per_element_n_major_timing_replay_enabled", False)
    )
    per_element_n_major_timing_replay_field = str(
        lifecycle_context.get(
            "per_element_n_major_timing_replay_field",
            "cumulative_input",
        )
    )
    per_element_n_major_timing_replay_elements = {
        str(value)
        for value in lifecycle_context.get(
            "per_element_n_major_timing_replay_elements",
            [],
        )
    }
    per_element_n_major_timing_replay_iteration = lifecycle_context.get(
        "per_element_n_major_timing_replay_iteration"
    )
    emit_molecule_input_trace = bool(
        lifecycle_context.get("emit_molecule_input_trace", False)
    )
    molecule_input_trace_iteration_limit = int(
        lifecycle_context.get("molecule_input_trace_iteration_limit", 0)
    )
    molecule_input_trace_max_records = int(
        lifecycle_context.get("molecule_input_trace_max_records", 0)
    )
    emit_electron_donor_trace = bool(
        lifecycle_context.get("emit_electron_donor_trace", False)
    )
    electron_donor_trace_order_scope = str(
        lifecycle_context.get("electron_donor_trace_order_scope", "call_local")
    )
    electron_donor_trace_iteration_limit = int(
        lifecycle_context.get("electron_donor_trace_iteration_limit", 0)
    )
    electron_donor_trace_max_records = int(
        lifecycle_context.get("electron_donor_trace_max_records", 0)
    )
    emit_element_solver_trace = bool(
        lifecycle_context.get("emit_element_solver_trace", False)
    )
    element_solver_trace_iteration_limit = int(
        lifecycle_context.get("element_solver_trace_iteration_limit", 0)
    )
    element_solver_trace_max_records = int(
        lifecycle_context.get("element_solver_trace_max_records", 0)
    )
    emit_coefficient_source_value_trace = bool(
        lifecycle_context.get("emit_coefficient_source_value_trace", False)
    )
    coefficient_source_value_trace_iteration_limit = int(
        lifecycle_context.get("coefficient_source_value_trace_iteration_limit", 0)
    )
    coefficient_source_value_trace_max_records = int(
        lifecycle_context.get("coefficient_source_value_trace_max_records", 0)
    )
    coefficient_source_value_trace_elements = {
        str(value)
        for value in lifecycle_context.get(
            "coefficient_source_value_trace_elements",
            [],
        )
    }
    emit_minor_density_trace = bool(
        lifecycle_context.get("emit_minor_density_trace", False)
    )
    minor_density_trace_iteration_limit = int(
        lifecycle_context.get("minor_density_trace_iteration_limit", 0)
    )
    minor_density_trace_max_records = int(
        lifecycle_context.get("minor_density_trace_max_records", 0)
    )
    minor_density_trace_elements = {
        str(value)
        for value in lifecycle_context.get(
            "minor_density_trace_elements",
            [],
        )
    }
    fastchem_longdouble_minlimit_log_cgs = float(
        lifecycle_context.get(
            "fastchem_longdouble_minlimit_log_cgs",
            -512.0 * math.log(10.0),
        )
    )
    n_major_solver_dispatched_rows_policy = "solver_dispatched_rows"
    element_solver_scaling_factors: dict[str, float] = {}
    fixed_context = lifecycle_context.get("fixed_by_condensation_flags", [])
    fixed_by_condensation = np.zeros(n_elements, dtype=bool)
    if fixed_context:
        fixed_values = np.asarray(fixed_context, dtype=bool)
        fixed_by_condensation[: min(n_elements, fixed_values.shape[0])] = (
            fixed_values[:n_elements]
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        element_solver_log_density_cgs = np.where(
            species_density[:n_elements] * density_domain_scale > 0.0,
            np.log(species_density[:n_elements] * density_domain_scale),
            fastchem_longdouble_minlimit_log_cgs,
        ).astype(np.float64)
    element_solver_candidate_log_density_cgs = element_solver_log_density_cgs.copy()
    previous_iteration_element_solver_log_density_cgs = (
        element_solver_log_density_cgs.copy()
    )
    iteration_start_element_solver_log_density_cgs = (
        element_solver_log_density_cgs.copy()
    )
    post_element_solver_log_density_cgs = element_solver_log_density_cgs.copy()
    post_element_solver_candidate_log_density_cgs = (
        element_solver_candidate_log_density_cgs.copy()
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        post_element_solver_species_log_density_cgs = np.where(
            species_density[:n_elements] * density_domain_scale > 0.0,
            np.log(species_density[:n_elements] * density_domain_scale),
            fastchem_longdouble_minlimit_log_cgs,
        ).astype(np.float64)
    last_solved_element_handoff_log_density_cgs = (
        previous_iteration_element_solver_log_density_cgs.copy()
    )
    last_solved_candidate_handoff_log_density_cgs = (
        previous_iteration_element_solver_log_density_cgs.copy()
    )
    for _element_index in range(n_elements):
        latest_element_solver_log_materialization[_element_index] = {
            "iteration": int(current_replay_iteration),
            "element_index": int(_element_index),
            "element_label": (
                element_labels[int(_element_index)]
                if int(_element_index) < len(element_labels)
                else str(_element_index)
            ),
            "materialization_stage": "initial_log_vector",
            "materialization_policy": "initial_species_density_log",
            "pre_assignment_element_solver_log": None,
            "post_assignment_element_solver_log": float(
                element_solver_log_density_cgs[_element_index]
            ),
            "candidate_log_cgs_at_materialization": float(
                element_solver_candidate_log_density_cgs[_element_index]
            ),
            "assigned_log_cgs_at_materialization": float(
                element_solver_log_density_cgs[_element_index]
            ),
            "species_density_internal_at_materialization": float(
                species_density[_element_index]
            ),
            "density_domain_scale": float(density_domain_scale),
            "trace_marker": (
                "exact_fixed_row_subspace_trace_m372_"
                "element_solver_log_carrier_materialization"
            ),
        }
    with np.errstate(divide="ignore", invalid="ignore"):
        molecule_log_density_cgs = np.where(
            species_density[n_elements:] * density_domain_scale > 0.0,
            np.log(species_density[n_elements:] * density_domain_scale),
            fastchem_longdouble_minlimit_log_cgs,
        ).astype(np.float64)
    post_minor_refresh_density = species_density.copy()
    molecule_refresh_n_major_before_update: Optional[float] = None
    molecule_refresh_n_major_after_update: Optional[float] = None
    molecule_refresh_returned_n_major_delta: Optional[float] = None
    molecule_refresh_last_returned_n_major_delta: Optional[float] = None
    molecule_refresh_h_coefficient_log_consumed = False
    per_element_n_major_timing_history: dict[int, dict[str, float]] = {}

    def _log_from_internal_density_vector(values: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                values[:n_elements] * density_domain_scale > 0.0,
                np.log(values[:n_elements] * density_domain_scale),
                fastchem_longdouble_minlimit_log_cgs,
            ).astype(np.float64)

    def _hhe_lifecycle_scalar_log_vector(
        scalar: Optional[float],
        reference_density: np.ndarray,
    ) -> Optional[np.ndarray]:
        if scalar is None or not np.isfinite(float(scalar)):
            return None
        scalar_cgs = float(max(float(scalar), 0.0) * density_domain_scale)
        log_value = (
            float(np.log(scalar_cgs))
            if scalar_cgs > 0.0 and np.isfinite(scalar_cgs)
            else fastchem_longdouble_minlimit_log_cgs
        )
        logs = _log_from_internal_density_vector(reference_density)
        for label in ("H", "He"):
            if label in element_labels:
                logs[int(element_labels.index(label))] = log_value
        return logs

    def _element_log_from_cgs(value: float) -> float:
        if value > 0.0 and np.isfinite(value):
            return float(np.log(value))
        return fastchem_longdouble_minlimit_log_cgs

    def _element_log_from_cgs_longdouble(value: Any) -> float:
        value_ld = np.longdouble(value)
        if value_ld > 0.0 and np.isfinite(value_ld):
            return float(np.log(value_ld))
        return fastchem_longdouble_minlimit_log_cgs

    def _post_checkn_log_cgs(
        candidate_log_cgs: float,
        assigned_log_cgs: float,
        element_density_minlimit_internal: float,
    ) -> float:
        if not np.isfinite(candidate_log_cgs):
            return assigned_log_cgs
        log_cgs = float(candidate_log_cgs)
        minlimit_cgs = (
            np.longdouble(element_density_minlimit_internal)
            * np.longdouble(density_domain_scale)
        )
        if minlimit_cgs > 0.0 and np.isfinite(minlimit_cgs):
            log_cgs = max(log_cgs, float(np.log(minlimit_cgs)))
        upper_cgs = (
            np.longdouble(element_checkN_upper_bound_internal)
            * np.longdouble(density_domain_scale)
        )
        if upper_cgs > 0.0 and np.isfinite(upper_cgs):
            log_cgs = min(log_cgs, float(np.log(upper_cgs)))
        return float(log_cgs)

    def _molecule_abundance(mol_index: int) -> float:
        active = np.nonzero(molecule_formula[:, mol_index] != 0.0)[0]
        if (
            electron_index is not None
            and molecule_abundance_electron_policy
            == "exclude_zero_abundance_electron"
        ):
            active = active[active != electron_index]
        if active.size == 0:
            return float("nan")
        if molecule_abundance_rule == "fastchem_min_element":
            selected = int(active[0])
            for raw_index in active[1:]:
                index = int(raw_index)
                if (
                    element_abundance[index] != 0.0
                    and element_abundance[index] < element_abundance[selected]
                ):
                    selected = index
            return float(element_abundance[selected])
        nonzero = active[element_abundance[active] > 0.0]
        source = nonzero if nonzero.size else active
        return float(np.min(element_abundance[source]))

    molecule_abundances = np.asarray(
        [_molecule_abundance(mol_index) for mol_index in range(molecule_formula.shape[1])],
        dtype=np.float64,
    )

    def _molecule_abundance_matches_element(
        element_index: int,
        mol_index: int,
    ) -> bool:
        value = float(molecule_abundances[mol_index])
        if not np.isfinite(value):
            return False
        return bool(value == float(element_abundance[element_index]))

    def _solver_molecule_indices(
        element_index: int,
        *,
        source: str,
    ) -> list[int]:
        if source in {
            "molecule_list",
            "molecule_list_abundance_gate",
            "fastchem_molecule_list_abundance_gate",
        }:
            values = molecule_list_by_element.get(element_index, [])
        else:
            values = major_molecules_by_element.get(element_index, [])
        if source in {
            "molecule_list_abundance_gate",
            "fastchem_molecule_list_abundance_gate",
        }:
            values = [
                mol_index
                for mol_index in values
                if _molecule_abundance_matches_element(element_index, mol_index)
            ]
        return values

    def _element_solver_minor_density_source_for(element_index: int) -> str:
        source = element_solver_minor_density_source
        if element_solver_minor_density_source_scoped_source is None:
            return source
        element_label = (
            element_labels[element_index]
            if 0 <= element_index < len(element_labels)
            else str(element_index)
        )
        iteration_matches = (
            element_solver_minor_density_source_iteration is None
            or int(current_replay_iteration)
            == int(element_solver_minor_density_source_iteration)
        )
        element_matches = (
            not element_solver_minor_density_source_elements
            or str(element_label) in element_solver_minor_density_source_elements
            or str(element_index) in element_solver_minor_density_source_elements
        )
        if iteration_matches and element_matches:
            return str(element_solver_minor_density_source_scoped_source)
        return source

    def _minor_density(element_index: int) -> float:
        minor_density_source = _element_solver_minor_density_source_for(element_index)
        element_label = (
            element_labels[element_index]
            if 0 <= element_index < len(element_labels)
            else str(element_index)
        )
        trace_element_matches = (
            not minor_density_trace_elements
            or str(element_label) in minor_density_trace_elements
            or str(element_index) in minor_density_trace_elements
        )
        trace_enabled = (
            emit_minor_density_trace
            and trace_element_matches
            and current_replay_iteration >= 0
            and current_replay_iteration < minor_density_trace_iteration_limit
            and len(minor_density_trace_records) < minor_density_trace_max_records
        )

        def _append_minor_density_trace(row: dict[str, Any]) -> None:
            if (
                not trace_enabled
                or len(minor_density_trace_records)
                >= minor_density_trace_max_records
            ):
                return
            minor_density_trace_records.append(
                {
                    "iteration": int(current_replay_iteration),
                    "element_index": int(element_index),
                    "element_label": str(element_label),
                    "minor_density_source": str(minor_density_source),
                    "global_minor_density_source": str(
                        element_solver_minor_density_source
                    ),
                    "scoped_minor_density_source": (
                        None
                        if element_solver_minor_density_source_scoped_source is None
                        else str(element_solver_minor_density_source_scoped_source)
                    ),
                    "coefficient_density_scale": float(coefficient_density_scale),
                    "coefficient_density_domain": (
                        element_solver_coefficient_density_domain
                    ),
                    "epsilon": float(phi[element_index]),
                    "diagnostic_only": True,
                    "default_off": True,
                    "KL_owned": True,
                    "FastChem_trace_values_used_as_inputs": False,
                    "used_as_KL_constructor_input": False,
                    **row,
                }
            )

        if minor_density_source == "zero":
            _append_minor_density_trace(
                {
                    "stage": "minor_density_zero_source",
                    "total_minor_density_cgs": 0.0,
                    "term_count": 0,
                }
            )
            return 0.0
        if (
            minor_density_source == "zero_first_iteration"
            and current_replay_iteration == 0
        ):
            _append_minor_density_trace(
                {
                    "stage": "minor_density_zero_first_iteration",
                    "total_minor_density_cgs": 0.0,
                    "term_count": 0,
                }
            )
            return 0.0
        if (
            minor_density_source == "zero_for_H_quadratic_probe"
            and 0 <= element_index < len(element_labels)
            and str(element_labels[element_index]) == "H"
        ):
            _append_minor_density_trace(
                {
                    "stage": "minor_density_zero_for_H_quadratic_probe",
                    "total_minor_density_cgs": 0.0,
                    "term_count": 0,
                }
            )
            return 0.0
        total = 0.0
        epsilon_value = float(phi[element_index])
        use_molecule_log_carrier = minor_density_source in {
            "molecule_log_carrier",
            "molecule_log_carrier_after_first_iteration",
        } and not (
            minor_density_source == "molecule_log_carrier_after_first_iteration"
            and current_replay_iteration == 0
        )
        for mol_index in minor_molecules_by_element.get(element_index, []):
            molecule_label = _molecule_formula_label(mol_index)
            stoich = molecule_formula[element_index, mol_index]
            override_iteration_matches = (
                element_solver_minor_density_term_override_iteration is None
                or int(current_replay_iteration)
                == int(element_solver_minor_density_term_override_iteration)
            )
            override_element_matches = (
                not element_solver_minor_density_term_override_elements
                or str(element_label)
                in element_solver_minor_density_term_override_elements
                or str(element_index)
                in element_solver_minor_density_term_override_elements
            )
            override_molecule_matches = (
                not element_solver_minor_density_term_override_molecules
                or str(molecule_label)
                in element_solver_minor_density_term_override_molecules
                or str(mol_index)
                in element_solver_minor_density_term_override_molecules
            )
            override_applies = (
                element_solver_minor_density_term_override_policy != "disabled"
                and override_iteration_matches
                and override_element_matches
                and override_molecule_matches
            )
            if (
                use_molecule_log_carrier
                and not (
                    override_applies
                    and element_solver_minor_density_term_override_policy
                    in {"density_for_molecule", "zero_for_molecule"}
                )
            ):
                log_cgs = float(molecule_log_density_cgs[int(mol_index)])
                log_value = (
                    log_cgs
                    if element_solver_coefficient_density_domain == "cgs"
                    else log_cgs - np.log(density_domain_scale)
                )
                molecule_density = (
                    float(np.exp(np.clip(log_value, -745.0, 709.0)))
                    if np.isfinite(log_value)
                    else float(species_density[n_elements + mol_index])
                )
            else:
                log_cgs = None
                log_value = None
                molecule_density = float(species_density[n_elements + mol_index])
            if (
                override_applies
                and element_solver_minor_density_term_override_policy
                == "zero_for_molecule"
            ):
                molecule_density = 0.0
            coefficient = stoich + epsilon_value * molecule_sigma[mol_index]
            contribution = (
                coefficient * molecule_density * coefficient_density_scale
            )
            total += contribution
            _append_minor_density_trace(
                {
                    "stage": "minor_density_term",
                    "molecule_index": int(mol_index),
                    "molecule_label": _molecule_formula_label(mol_index),
                    "stoichiometric_coefficient": float(stoich),
                    "molecule_sigma": float(molecule_sigma[mol_index]),
                    "minor_density_coefficient": float(coefficient),
                    "used_molecule_log_carrier": bool(use_molecule_log_carrier),
                    "term_override_applied": bool(override_applies),
                    "term_override_policy": (
                        element_solver_minor_density_term_override_policy
                        if override_applies
                        else "disabled"
                    ),
                    "molecule_log_density_cgs": (
                        None if log_cgs is None else float(log_cgs)
                    ),
                    "molecule_log_density_internal": (
                        None if log_value is None else float(log_value)
                    ),
                    "molecule_density_internal": float(molecule_density),
                    "molecule_density_cgs": float(
                        molecule_density * coefficient_density_scale
                    ),
                    "minor_density_contribution_cgs": float(contribution),
                    "running_minor_density_cgs": float(total),
                }
            )
        _append_minor_density_trace(
            {
                "stage": "minor_density_total",
                "total_minor_density_cgs": float(total),
                "term_count": int(
                    len(minor_molecules_by_element.get(element_index, []))
                ),
            }
        )
        return float(total)

    def _molecule_density_for_feedback(mol_index: int) -> float:
        if molecule_feedback_density_source in {
            "molecule_log_carrier",
            "molecule_log_carrier_after_first_iteration",
        } and not (
            molecule_feedback_density_source
            == "molecule_log_carrier_after_first_iteration"
            and current_replay_iteration == 0
        ):
            log_cgs = float(molecule_log_density_cgs[int(mol_index)])
            log_value = log_cgs - np.log(density_domain_scale)
            if np.isfinite(log_value):
                return float(np.exp(np.clip(log_value, -745.0, 709.0)))
        return float(species_density[n_elements + mol_index])

    def _molecule_log_carrier_density(mol_index: int) -> float:
        log_cgs = float(molecule_log_density_cgs[int(mol_index)])
        log_value = log_cgs - np.log(density_domain_scale)
        if np.isfinite(log_value):
            return float(np.exp(np.clip(log_value, -745.0, 709.0)))
        return float(species_density[n_elements + mol_index])

    def _molecule_major_delta_source_for(
        element_index: int,
        mol_index: int,
    ) -> str:
        if molecule_major_delta_source == "feedback_density":
            return "feedback_density"
        element_label = (
            element_labels[element_index]
            if 0 <= element_index < len(element_labels)
            else str(element_index)
        )
        molecule_label = _molecule_formula_label(mol_index)
        iteration_matches = (
            molecule_major_delta_source_iteration is None
            or int(current_replay_iteration)
            == int(molecule_major_delta_source_iteration)
        )
        element_matches = (
            not molecule_major_delta_source_elements
            or str(element_label) in molecule_major_delta_source_elements
            or str(element_index) in molecule_major_delta_source_elements
        )
        molecule_matches = (
            not molecule_major_delta_source_molecules
            or str(molecule_label) in molecule_major_delta_source_molecules
            or str(mol_index) in molecule_major_delta_source_molecules
        )
        if iteration_matches and element_matches and molecule_matches:
            return molecule_major_delta_source
        return "feedback_density"

    def _major_density_for_element(
        element_index: int,
        density_vector: np.ndarray,
        *,
        source: str = "density",
    ) -> float:
        total = 0.0
        for mol_index in major_molecules_by_element.get(element_index, []):
            molecule_density = (
                _molecule_log_carrier_density(mol_index)
                if source == "molecule_log_carrier"
                else float(density_vector[n_elements + mol_index])
            )
            total += float(
                molecule_density * molecule_sigma[mol_index]
            )
        return float(total)

    def _major_density_for_all_elements(
        density_vector: np.ndarray,
        *,
        source: str = "density",
    ) -> float:
        total = 0.0
        seen: set[int] = set()
        for values in major_molecules_by_element.values():
            for mol_index in values:
                if mol_index in seen:
                    continue
                seen.add(int(mol_index))
                molecule_density = (
                    _molecule_log_carrier_density(mol_index)
                    if source == "molecule_log_carrier"
                    else float(density_vector[n_elements + mol_index])
                )
                total += float(
                    molecule_density * molecule_sigma[mol_index]
                )
        return float(total)

    def _element_solver_n_major(
        element_index: int,
        cumulative_n_major: float,
    ) -> float:
        if per_element_n_major_timing_replay_enabled:
            element_label = (
                element_labels[element_index]
                if 0 <= element_index < len(element_labels)
                else str(element_index)
            )
            iteration_matches = (
                per_element_n_major_timing_replay_iteration is None
                or int(current_replay_iteration)
                == int(per_element_n_major_timing_replay_iteration)
            )
            element_matches = (
                not per_element_n_major_timing_replay_elements
                or str(element_label) in per_element_n_major_timing_replay_elements
                or str(element_index) in per_element_n_major_timing_replay_elements
            )
            timing_snapshot = per_element_n_major_timing_history.get(element_index, {})
            if iteration_matches and element_matches:
                replay_value = timing_snapshot.get(
                    per_element_n_major_timing_replay_field
                )
                if replay_value is not None and np.isfinite(float(replay_value)):
                    return float(replay_value)
        if element_solver_n_major_source == "zero":
            return 0.0
        if element_solver_n_major_source == "local_current_major_sigma":
            return _major_density_for_element(element_index, species_density)
        if element_solver_n_major_source == "local_molecule_log_major_sigma":
            return _major_density_for_element(
                element_index,
                species_density,
                source="molecule_log_carrier",
            )
        if element_solver_n_major_source == "local_old_major_sigma":
            return _major_density_for_element(element_index, old_density)
        if element_solver_n_major_source == "local_initial_major_sigma":
            return _major_density_for_element(
                element_index,
                initial_species_density_internal,
            )
        if element_solver_n_major_source == "local_max_current_old_major_sigma":
            return _major_density_for_element(
                element_index,
                np.maximum(species_density, old_density),
            )
        if element_solver_n_major_source == "global_current_major_sigma":
            return _major_density_for_all_elements(species_density)
        if element_solver_n_major_source == "global_molecule_log_major_sigma":
            return _major_density_for_all_elements(
                species_density,
                source="molecule_log_carrier",
            )
        if element_solver_n_major_source == "global_old_major_sigma":
            return _major_density_for_all_elements(old_density)
        if element_solver_n_major_source == "global_initial_major_sigma":
            return _major_density_for_all_elements(initial_species_density_internal)
        if element_solver_n_major_source == "global_max_current_old_major_sigma":
            return _major_density_for_all_elements(np.maximum(species_density, old_density))
        return float(cumulative_n_major)

    def _element_budget_major_contribution(
        element_index: int,
        cumulative_n_major: float,
    ) -> float:
        number_density_maj_cgs = (
            float(cumulative_n_major)
            * float(phi[element_index])
            * coefficient_density_scale
        )
        atomic_density_cgs = (
            float(species_density[element_index]) * coefficient_density_scale
        )
        residual_cgs = (
            coefficient_gas_density * float(phi[element_index])
            - atomic_density_cgs
            - _minor_density(element_index)
            - number_density_maj_cgs
        )
        return float(residual_cgs / coefficient_density_scale)

    def _element_budget_major_stoich_contribution(
        element_index: int,
        cumulative_n_major: float,
    ) -> float:
        divisor = 1.0
        for mol_index in major_molecules_by_element.get(element_index, []):
            stoich_self = float(molecule_formula[element_index, mol_index])
            if stoich_self > divisor:
                divisor = stoich_self
        return float(
            _element_budget_major_contribution(element_index, cumulative_n_major)
            / max(divisor, 1.0)
        )

    def _solver_scaling_factor(element_index: int, solver_order: int) -> float:
        if not use_solver_scaling_factor or solver_order <= 0:
            return 0.0
        factor = 0.0
        for mol_index in _solver_molecule_indices(
            element_index,
            source=element_solver_scaling_factor_molecule_source,
        ):
            stoich_self = int(round(molecule_formula[element_index, mol_index]))
            if stoich_self < 1 or stoich_self > solver_order:
                continue
            donor = _donor_log_sum(element_index, mol_index)
            term = float(_coefficient_mass_action_log(mol_index) + donor)
            if term > factor:
                factor = term
        return float(factor - additional_solver_scaling_factor)

    def _scaled_exp(term: float, solver_scaling_factor: float) -> float:
        exponent = term - solver_scaling_factor
        if element_solver_exponential_mode == "native_overflow":
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                return float(np.exp(exponent))
        return float(
            np.exp(
                np.clip(
                    exponent,
                    -element_solver_exponent_clip,
                    element_solver_exponent_clip,
                )
            )
        )

    def _coefficient_mass_action_log(mol_index: int) -> float:
        if (
            element_solver_coefficient_mass_action_source
            == "molecule_mass_action_constants"
        ):
            return float(molecule_mass_action_constants[mol_index])
        return float(coefficient_mass_action_constants[mol_index])

    def _element_solver_coefficient_donor_log_source_for(
        element_index: int,
        donor_index: Optional[int] = None,
    ) -> str:
        if not element_solver_coefficient_donor_log_source_scopes:
            return element_solver_coefficient_donor_log_source
        element_label = (
            element_labels[element_index]
            if 0 <= element_index < len(element_labels)
            else str(element_index)
        )
        donor_label = (
            element_labels[donor_index]
            if donor_index is not None and 0 <= donor_index < len(element_labels)
            else None
        )
        for scope in element_solver_coefficient_donor_log_source_scopes:
            if not isinstance(scope, dict):
                continue
            iteration = scope.get("iteration")
            if iteration is not None and int(iteration) != int(current_replay_iteration):
                continue
            elements = {str(value) for value in scope.get("elements", [])}
            match_on = str(scope.get("match_on", "element"))
            if match_on == "donor_element":
                if donor_index is None:
                    continue
                if elements and str(donor_label) not in elements and str(donor_index) not in elements:
                    continue
            elif elements and str(element_label) not in elements and str(element_index) not in elements:
                continue
            source = scope.get("source")
            if source is not None:
                return str(source)
        return element_solver_coefficient_donor_log_source

    def _element_solver_donor_log_values(
        source: str,
        active: np.ndarray,
    ) -> np.ndarray:
        if source == "element_solver_log_carrier":
            return element_solver_log_density_cgs[active]
        if source in {
            "candidate_element_solver_log_carrier",
            "newton_candidate_log_carrier",
            "pre_checkN_donor_carrier",
        }:
            return element_solver_candidate_log_density_cgs[active]
        if source in {
            "current_element_solver_log_carrier",
            "post_checkN_donor_carrier",
            "newton_donor_input_carrier",
        }:
            return element_solver_log_density_cgs[active]
        if source == "previous_iteration_element_solver_log_carrier":
            return previous_iteration_element_solver_log_density_cgs[active]
        if source == "iteration_start_element_solver_log_carrier":
            return iteration_start_element_solver_log_density_cgs[active]
        if source == "post_element_solver_log_carrier":
            return post_element_solver_log_density_cgs[active]
        if source == "post_element_solver_candidate_log_carrier":
            return post_element_solver_candidate_log_density_cgs[active]
        if source == "post_element_solver_species_density_log":
            return post_element_solver_species_log_density_cgs[active]
        if source == "last_solved_element_handoff_log_carrier":
            return last_solved_element_handoff_log_density_cgs[active]
        if source == "last_solved_candidate_handoff_log_carrier":
            return last_solved_candidate_handoff_log_density_cgs[active]
        if source == "post_molecule_feedback_log_carrier":
            return _log_from_internal_density_vector(species_density)[active]
        values = species_density[:n_elements][active] * coefficient_density_scale
        if element_solver_donor_log_policy == "native_raw":
            with np.errstate(divide="ignore", invalid="ignore"):
                logs = np.log(values)
        else:
            logs = np.log(np.maximum(values, coefficient_density_floor))
        if source == "electron_minlimit" and electron_index is not None:
            active_indices = np.nonzero(active)[0]
            logs = np.asarray(logs, dtype=np.float64).copy()
            logs[active_indices == electron_index] = (
                fastchem_longdouble_minlimit_log_cgs
            )
        return logs

    def _element_solver_donor_log_provenance(
        donor_index: int,
        donor_source: str,
        selected_log_density_cgs: float,
    ) -> dict[str, Any]:
        donor_index = int(donor_index)

        def _density_log(vector: np.ndarray) -> float:
            return _element_log_from_cgs(
                float(vector[donor_index]) * density_domain_scale
            )

        raw_density_cgs = float(species_density[donor_index] * density_domain_scale)
        raw_density_log_cgs = _element_log_from_cgs(raw_density_cgs)
        current_element_solver_log = float(element_solver_log_density_cgs[donor_index])
        candidate_element_solver_log = float(
            element_solver_candidate_log_density_cgs[donor_index]
        )
        previous_element_solver_log = float(
            previous_iteration_element_solver_log_density_cgs[donor_index]
        )
        iteration_start_element_solver_log = float(
            iteration_start_element_solver_log_density_cgs[donor_index]
        )
        post_element_solver_log = float(
            post_element_solver_log_density_cgs[donor_index]
        )
        element_slot_write_value = last_element_slot_write_lineage[donor_index] or {}
        materialization = latest_element_solver_log_materialization[donor_index] or {}
        element_slot_write_log = element_slot_write_value.get("assigned_log_cgs")
        species_density_write_log = element_slot_write_value.get(
            "element_solver_log_density_cgs"
        )
        pre_family_scope_override_log = _density_log(
            initial_species_density_internal
        )
        post_family_scope_override_log = raw_density_log_cgs
        if np.isfinite(raw_density_cgs) and raw_density_cgs > 0.0:
            floor_status = "positive_raw_density"
        elif selected_log_density_cgs <= fastchem_longdouble_minlimit_log_cgs:
            floor_status = "fastchem_longdouble_minlimit_log"
        else:
            floor_status = "nonpositive_raw_density_log_carrier"
        native_number_density_cgs_longdouble = (
            np.longdouble(species_density[donor_index])
            * np.longdouble(density_domain_scale)
        )
        native_source_number_density_cgs_longdouble = (
            np.longdouble(
                initial_species_density_source_longdouble_internal[donor_index]
            )
            * np.longdouble(density_domain_scale)
        )
        native_number_density_subdouble_available = bool(
            native_number_density_cgs_longdouble > 0.0
            and np.isfinite(native_number_density_cgs_longdouble)
        )
        native_source_number_density_longdouble_available = bool(
            native_source_number_density_cgs_longdouble > 0.0
            and np.isfinite(native_source_number_density_cgs_longdouble)
        )
        native_number_density_subdouble_log_cgs = (
            _element_log_from_cgs_longdouble(native_number_density_cgs_longdouble)
            if native_number_density_subdouble_available
            else None
        )
        native_source_number_density_longdouble_log_cgs = (
            _element_log_from_cgs_longdouble(
                native_source_number_density_cgs_longdouble
            )
            if native_source_number_density_longdouble_available
            else None
        )
        native_number_density_subdouble_source = (
            "kl_longdouble_from_float64_species_density"
        )
        native_source_number_density_longdouble_source = (
            "kl_boundary_longdouble_from_initial_species_density_handoff"
        )
        return {
            "donor_source_stage": donor_source,
            "donor_producer_stage": donor_source,
            "donor_raw_density_cgs": raw_density_cgs,
            "donor_raw_density_log_cgs": raw_density_log_cgs,
            "donor_native_number_density_cgs": raw_density_cgs,
            "donor_native_std_log_from_double_cgs": raw_density_log_cgs,
            "donor_native_number_density_subdouble_log_cgs": (
                native_number_density_subdouble_log_cgs
            ),
            "donor_native_number_density_subdouble_available": (
                native_number_density_subdouble_available
            ),
            "donor_native_number_density_subdouble_source": (
                native_number_density_subdouble_source
            ),
            "donor_native_number_density_subdouble_preserves_hidden_bits": False,
            "donor_native_number_density_subdouble_reconstructed_from_float64": True,
            "donor_native_source_number_density_longdouble_log_cgs": (
                native_source_number_density_longdouble_log_cgs
            ),
            "donor_native_source_number_density_longdouble_available": (
                native_source_number_density_longdouble_available
            ),
            "donor_native_source_number_density_longdouble_source": (
                native_source_number_density_longdouble_source
            ),
            "donor_native_source_number_density_longdouble_preserves_hidden_bits": (
                False
            ),
            "donor_native_source_number_density_longdouble_reconstructed_from_float64": (
                True
            ),
            "donor_selected_minus_native_double_log_delta": (
                float(selected_log_density_cgs - raw_density_log_cgs)
                if np.isfinite(selected_log_density_cgs)
                and np.isfinite(raw_density_log_cgs)
                else None
            ),
            "current_element_solver_log": current_element_solver_log,
            "candidate_element_solver_log": candidate_element_solver_log,
            "initial_species_density_log": pre_family_scope_override_log,
            "native_current_vector_log": raw_density_log_cgs,
            "element_slot_write_value_log": (
                None if element_slot_write_log is None else float(element_slot_write_log)
            ),
            "species_density_write_value_log": (
                None
                if species_density_write_log is None
                else float(species_density_write_log)
            ),
            "pre_family_scope_override_log": pre_family_scope_override_log,
            "post_family_scope_override_log": post_family_scope_override_log,
            "old_density_log": _density_log(old_density),
            "previous_iteration_element_solver_log": previous_element_solver_log,
            "iteration_start_element_solver_log": iteration_start_element_solver_log,
            "post_element_solver_log": post_element_solver_log,
            "subdouble_longdouble_floor_status": floor_status,
            "element_solver_log_materialization": dict(materialization),
            "element_solver_log_materialization_stage": materialization.get(
                "materialization_stage"
            ),
            "element_solver_log_materialization_policy": materialization.get(
                "materialization_policy"
            ),
            "element_solver_log_materialization_pre_assignment": (
                materialization.get("pre_assignment_element_solver_log")
            ),
            "element_solver_log_materialization_post_assignment": (
                materialization.get("post_assignment_element_solver_log")
            ),
            "selected_log_matches_current_element_solver_log": bool(
                selected_log_density_cgs == current_element_solver_log
            ),
            "selected_log_matches_candidate_element_solver_log": bool(
                selected_log_density_cgs == candidate_element_solver_log
            ),
            "trace_marker": (
                "exact_fixed_row_subspace_trace_m371_"
                "kl_quadratic_donor_log_value_provenance"
            ),
        }

    def _donor_log_terms(element_index: int, mol_index: int) -> tuple[float, list[dict[str, Any]]]:
        stoich = molecule_formula[:, mol_index]
        active = (stoich != 0.0) & (np.arange(n_elements) != element_index)
        active_indices = np.nonzero(active)[0]
        donor_sources = [
            _element_solver_coefficient_donor_log_source_for(
                element_index,
                int(donor_index),
            )
            for donor_index in active_indices
        ]
        if len(set(donor_sources)) > 1:
            logs = np.empty(active_indices.shape[0], dtype=np.float64)
            for source in sorted(set(donor_sources)):
                source_mask = np.asarray(
                    [value == source for value in donor_sources],
                    dtype=bool,
                )
                full_mask = np.zeros(n_elements, dtype=bool)
                full_mask[active_indices[source_mask]] = True
                logs[source_mask] = _element_solver_donor_log_values(
                    source,
                    full_mask,
                )
        else:
            donor_log_source = (
                donor_sources[0]
                if donor_sources
                else _element_solver_coefficient_donor_log_source_for(element_index)
            )
            logs = _element_solver_donor_log_values(donor_log_source, active)
        rows = []
        for local_position, donor_index in enumerate(active_indices.tolist()):
            donor_log = float(logs[int(local_position)])
            donor_stoich = float(stoich[int(donor_index)])
            rows.append(
                {
                    "donor_element_index": int(donor_index),
                    "donor_element_label": (
                        element_labels[int(donor_index)]
                        if int(donor_index) < len(element_labels)
                        else str(donor_index)
                    ),
                    "donor_stoich": donor_stoich,
                    "donor_source": donor_sources[int(local_position)],
                    "donor_log_density_cgs": donor_log,
                    "donor_weighted_log_contribution": float(
                        donor_stoich * donor_log
                    ),
                    **_element_solver_donor_log_provenance(
                        int(donor_index),
                        donor_sources[int(local_position)],
                        donor_log,
                    ),
                }
            )
        return float(np.dot(stoich[active], logs)), rows

    def _donor_log_sum(element_index: int, mol_index: int) -> float:
        donor, _rows = _donor_log_terms(element_index, mol_index)
        return donor

    def _append_coefficient_source_value_trace(
        *,
        element_index: int,
        order: int,
        solver_scaling_factor: float,
        term_row: dict[str, Any],
    ) -> None:
        if not emit_coefficient_source_value_trace:
            return
        if current_replay_iteration < 0 or (
            current_replay_iteration
            >= coefficient_source_value_trace_iteration_limit
        ):
            return
        if (
            len(coefficient_source_value_trace_records)
            >= coefficient_source_value_trace_max_records
        ):
            return
        element_label_local = (
            element_labels[element_index]
            if 0 <= element_index < len(element_labels)
            else str(element_index)
        )
        if coefficient_source_value_trace_elements and (
            str(element_label_local) not in coefficient_source_value_trace_elements
            and str(element_index) not in coefficient_source_value_trace_elements
        ):
            return
        if str(element_label_local) != "K" or order not in {1, 2}:
            return
        if term_row.get("molecule_index") is None:
            return
        donors = list(term_row.get("donor_log_terms") or [])
        element_order_position = element_order_positions.get(int(element_index))
        coefficient_label = f"a{int(order)}"
        coefficient_source_value_trace_records.append(
            {
                "trace_marker": (
                    "exact_fixed_row_subspace_trace_m379_"
                    "kl_k_a1_a2_coefficient_source_value_path"
                ),
                "diagnostic_only": True,
                "default_off": True,
                "KL_owned": True,
                "FastChem_trace_values_used_as_inputs": False,
                "used_as_KL_constructor_input": False,
                "iteration": int(current_replay_iteration),
                "element_index": int(element_index),
                "element_label": str(element_label_local),
                "element_order_position": element_order_position,
                "call_sequence_id": (
                    None
                    if current_replay_iteration < 0
                    else int(20 + 27 * current_replay_iteration)
                ),
                "coefficient_label": coefficient_label,
                "coefficient_order": int(order),
                "molecule_index": int(term_row["molecule_index"]),
                "molecule_label": term_row.get("molecule_label"),
                "donor_labels": [
                    donor.get("donor_element_label") for donor in donors
                ],
                "donor_log_values": [
                    donor.get("donor_log_density_cgs") for donor in donors
                ],
                "donor_stoich_values": [
                    donor.get("donor_stoich") for donor in donors
                ],
                "donor_raw_values": [
                    donor.get("donor_raw_density_cgs") for donor in donors
                ],
                "donor_source_value_candidates": [
                    {
                        "donor_element_label": donor.get("donor_element_label"),
                        "selected": donor.get("donor_log_density_cgs"),
                        "current_element_solver_log": donor.get(
                            "current_element_solver_log"
                        ),
                        "candidate_element_solver_log": donor.get(
                            "candidate_element_solver_log"
                        ),
                        "initial_species_density_log": donor.get(
                            "initial_species_density_log"
                        ),
                        "native_current_vector_log": donor.get(
                            "native_current_vector_log"
                        ),
                        "native_number_density_cgs": donor.get(
                            "donor_native_number_density_cgs"
                        ),
                        "native_std_log_from_double_cgs": donor.get(
                            "donor_native_std_log_from_double_cgs"
                        ),
                        "native_number_density_subdouble_log_cgs": donor.get(
                            "donor_native_number_density_subdouble_log_cgs"
                        ),
                        "native_number_density_subdouble_available": donor.get(
                            "donor_native_number_density_subdouble_available"
                        ),
                        "native_number_density_subdouble_source": donor.get(
                            "donor_native_number_density_subdouble_source"
                        ),
                        "native_number_density_subdouble_preserves_hidden_bits": donor.get(
                            "donor_native_number_density_subdouble_preserves_hidden_bits"
                        ),
                        "native_number_density_subdouble_reconstructed_from_float64": donor.get(
                            "donor_native_number_density_subdouble_reconstructed_from_float64"
                        ),
                        "native_source_number_density_longdouble_log_cgs": donor.get(
                            "donor_native_source_number_density_longdouble_log_cgs"
                        ),
                        "native_source_number_density_longdouble_available": donor.get(
                            "donor_native_source_number_density_longdouble_available"
                        ),
                        "native_source_number_density_longdouble_source": donor.get(
                            "donor_native_source_number_density_longdouble_source"
                        ),
                        "native_source_number_density_longdouble_preserves_hidden_bits": donor.get(
                            "donor_native_source_number_density_longdouble_preserves_hidden_bits"
                        ),
                        "native_source_number_density_longdouble_reconstructed_from_float64": donor.get(
                            "donor_native_source_number_density_longdouble_reconstructed_from_float64"
                        ),
                        "selected_minus_native_double_log_delta": donor.get(
                            "donor_selected_minus_native_double_log_delta"
                        ),
                        "old_density_log": donor.get("old_density_log"),
                        "previous_iteration_element_solver_log": donor.get(
                            "previous_iteration_element_solver_log"
                        ),
                        "iteration_start_element_solver_log": donor.get(
                            "iteration_start_element_solver_log"
                        ),
                        "post_element_solver_log": donor.get(
                            "post_element_solver_log"
                        ),
                        "pre_family_scope_override_log": donor.get(
                            "pre_family_scope_override_log"
                        ),
                        "post_family_scope_override_log": donor.get(
                            "post_family_scope_override_log"
                        ),
                        "element_slot_write_value_log": donor.get(
                            "element_slot_write_value_log"
                        ),
                        "species_density_write_value_log": donor.get(
                            "species_density_write_value_log"
                        ),
                    }
                    for donor in donors
                ],
                "donor_source_identity": [
                    donor.get("donor_source") for donor in donors
                ],
                "donor_source_stage": [
                    donor.get("donor_source_stage") for donor in donors
                ],
                "donor_producer_stage": [
                    donor.get("donor_producer_stage") for donor in donors
                ],
                "donor_log_domain_conversion": [
                    {
                        "donor_element_label": donor.get("donor_element_label"),
                        "raw_density_log_cgs": donor.get("donor_raw_density_log_cgs"),
                        "selected_log_density_cgs": donor.get(
                            "donor_log_density_cgs"
                        ),
                        "floor_status": donor.get(
                            "subdouble_longdouble_floor_status"
                        ),
                    }
                    for donor in donors
                ],
                "source_vector_before_override_log_cgs": [
                    donor.get("pre_family_scope_override_log") for donor in donors
                ],
                "source_vector_after_override_log_cgs": [
                    donor.get("post_family_scope_override_log") for donor in donors
                ],
                "mass_action_log": term_row.get("mass_action_log"),
                "kappa": term_row.get("kappa"),
                "kappa_sign": term_row.get("kappa_sign"),
                "solver_scaling_factor": float(solver_scaling_factor),
                "donor_log_sum": term_row.get("donor_log_sum"),
                "term_log_abs": term_row.get("term_log_abs"),
                "signed_term_contribution": {
                    "sign": term_row.get("kappa_sign"),
                    "log_abs": term_row.get("term_log_abs"),
                    "selected": bool(term_row.get("selected")),
                },
                "selected_product_log_row_identity": {
                    "element_label": str(element_label_local),
                    "iteration": int(current_replay_iteration),
                    "element_order_position": element_order_position,
                    "call_sequence_id": (
                        None
                        if current_replay_iteration < 0
                        else int(20 + 27 * current_replay_iteration)
                    ),
                    "molecule_index": int(term_row["molecule_index"]),
                    "molecule_label": term_row.get("molecule_label"),
                    "coefficient_label": coefficient_label,
                },
            }
        )

    def _signed_log_add_terms(
        terms: list[tuple[float, float]],
    ) -> tuple[int, float]:
        positive_log = -np.inf
        negative_log = -np.inf
        for sign_value, log_abs_value in terms:
            if sign_value == 0.0 or np.isnan(log_abs_value):
                continue
            if sign_value > 0.0:
                positive_log = float(np.logaddexp(positive_log, log_abs_value))
            else:
                negative_log = float(np.logaddexp(negative_log, log_abs_value))
        if np.isneginf(positive_log) and np.isneginf(negative_log):
            return 0, -np.inf
        if positive_log == negative_log:
            return 0, -np.inf
        if positive_log > negative_log:
            if np.isinf(positive_log):
                return 1, float(positive_log)
            return 1, float(
                positive_log + np.log1p(-np.exp(negative_log - positive_log))
            )
        if np.isinf(negative_log):
            return -1, float(negative_log)
        return -1, float(
            negative_log + np.log1p(-np.exp(positive_log - negative_log))
        )

    def _polynomial_sign_logspace(
        coefficients: list[float],
        log_x: float,
    ) -> int:
        terms: list[tuple[float, float]] = []
        for order, coeff in enumerate(coefficients):
            coeff_value = float(coeff)
            if coeff_value == 0.0 or not np.isfinite(coeff_value):
                continue
            terms.append(
                (
                    float(np.sign(coeff_value)),
                    float(np.log(abs(coeff_value)) + order * log_x),
                )
            )
        sign, _log_abs = _signed_log_add_terms(terms)
        return int(sign)

    def _logspace_positive_root(
        coefficients: list[float],
        *,
        lower_log: float,
        upper_log: float,
    ) -> Optional[float]:
        lower = float(lower_log)
        upper = float(upper_log)
        if not (np.isfinite(lower) and np.isfinite(upper)) or lower >= upper:
            return None
        sign_lower = _polynomial_sign_logspace(coefficients, lower)
        sign_upper = _polynomial_sign_logspace(coefficients, upper)
        if sign_lower == 0:
            return lower
        if sign_upper == 0:
            return upper
        if sign_lower == sign_upper:
            previous_log = lower
            previous_sign = sign_lower
            bracket: Optional[tuple[float, float, int, int]] = None
            for grid_index in range(1, 4097):
                current_log = lower + (upper - lower) * grid_index / 4096.0
                current_sign = _polynomial_sign_logspace(coefficients, current_log)
                if current_sign == 0:
                    return float(current_log)
                if current_sign != previous_sign:
                    bracket = (
                        float(previous_log),
                        float(current_log),
                        int(previous_sign),
                        int(current_sign),
                    )
                    break
                previous_log = current_log
                previous_sign = current_sign
            if bracket is None:
                return None
            lower, upper, sign_lower, sign_upper = bracket
        if sign_lower == sign_upper:
            return None
        for _ in range(240):
            midpoint = 0.5 * (lower + upper)
            sign_mid = _polynomial_sign_logspace(coefficients, midpoint)
            if sign_mid == 0:
                return float(midpoint)
            if sign_mid == sign_lower:
                lower = midpoint
                sign_lower = sign_mid
            else:
                upper = midpoint
                sign_upper = sign_mid
            if abs(upper - lower) <= 1.0e-12 * max(abs(midpoint), 1.0):
                break
        return float(0.5 * (lower + upper))

    def _coeff_signed_log_abs(
        element_index: int,
        order: int,
        n_min: float,
        n_maj: float,
        solver_scaling_factor: float,
    ) -> tuple[int, float]:
        sign, log_abs, _terms = _coeff_signed_log_abs_with_terms(
            element_index,
            order,
            n_min,
            n_maj,
            solver_scaling_factor,
        )
        return sign, log_abs

    def _coeff_signed_log_abs_with_terms(
        element_index: int,
        order: int,
        n_min: float,
        n_maj: float,
        solver_scaling_factor: float,
    ) -> tuple[int, float, list[dict[str, Any]]]:
        if order == 0:
            value = float(
                n_maj
                + n_min
                - coefficient_gas_density * phi[element_index]
            )
            if use_solver_scaling_factor:
                sign_scale, log_scale = 1, -solver_scaling_factor
            else:
                sign_scale, log_scale = 1, 0.0
            if value == 0.0:
                return 0, -np.inf, []
            return int(np.sign(value)) * sign_scale, float(
                np.log(abs(value)) + log_scale
            ), []
        term_rows: list[dict[str, Any]] = []
        selected_terms: list[tuple[float, float]] = []
        raw_terms: list[tuple[float, float, int, dict[str, Any]]] = []
        if order == 1:
            identity_log = (
                float(-solver_scaling_factor) if use_solver_scaling_factor else 0.0
            )
            raw_terms.append(
                (
                    1.0,
                    identity_log,
                    -1,
                    {
                        "molecule_index": None,
                        "molecule_label": "identity",
                        "order": int(order),
                        "stoich_self": 1,
                        "kappa": 1.0,
                        "kappa_sign": 1,
                        "donor_log_sum": 0.0,
                        "mass_action_log": 0.0,
                        "term_log_abs": identity_log,
                        "selected": True,
                    },
                )
            )
        coefficient_molecules = (
            molecule_list_by_element.get(element_index, [])
            if element_solver_coefficient_molecule_source == "molecule_list"
            else major_molecules_by_element.get(element_index, [])
        )
        if element_solver_coefficient_abundance_gate:
            coefficient_molecules = [
                mol_index
                for mol_index in coefficient_molecules
                if _molecule_abundance_matches_element(element_index, mol_index)
            ]
        for mol_index in coefficient_molecules:
            stoich_self = int(round(molecule_formula[element_index, mol_index]))
            if stoich_self != order:
                continue
            donor, donor_terms = _donor_log_terms(element_index, mol_index)
            kappa = float(order + phi[element_index] * molecule_sigma[mol_index])
            if kappa == 0.0:
                continue
            mass_action_log = _coefficient_mass_action_log(mol_index)
            coefficient_before_donor_sum_log_abs = float(
                mass_action_log - solver_scaling_factor + np.log(abs(kappa))
            )
            term_log = float(
                coefficient_before_donor_sum_log_abs + donor
            )
            kappa_sign = float(np.sign(kappa))
            raw_terms.append(
                (
                    kappa_sign,
                    term_log,
                    int(mol_index),
                    {
                        "molecule_index": int(mol_index),
                        "molecule_label": _molecule_formula_label(mol_index),
                        "order": int(order),
                        "stoich_self": int(stoich_self),
                        "kappa": float(kappa),
                        "kappa_sign": int(np.sign(kappa)),
                        "donor_log_sum": float(donor),
                        "donor_log_terms": donor_terms,
                        "mass_action_log": mass_action_log,
                        "branch": "quadratic" if order == 2 else "linear",
                        "coefficient_before_donor_sum_log_abs": (
                            coefficient_before_donor_sum_log_abs
                        ),
                        "coefficient_after_donor_sum_log_abs": term_log,
                        "term_log_abs": term_log,
                        "selected": True,
                    },
                )
            )
        if element_solver_signed_log_term_selection == "positive_kappa_only":
            selected_indices = {
                idx for idx, term in enumerate(raw_terms) if term[0] > 0.0
            }
        elif element_solver_signed_log_term_selection == "negative_kappa_only":
            selected_indices = {
                idx for idx, term in enumerate(raw_terms) if term[0] < 0.0
            }
        elif element_solver_signed_log_term_selection == "dominant_positive_only":
            positive_indices = [
                idx for idx, term in enumerate(raw_terms) if term[0] > 0.0
            ]
            selected_indices = (
                {
                    max(
                        positive_indices,
                        key=lambda idx: raw_terms[idx][1],
                    )
                }
                if positive_indices
                else set()
            )
        elif element_solver_signed_log_term_selection == "drop_dominant_positive":
            positive_indices = [
                idx for idx, term in enumerate(raw_terms) if term[0] > 0.0
            ]
            drop_index = (
                max(positive_indices, key=lambda idx: raw_terms[idx][1])
                if positive_indices
                else None
            )
            selected_indices = {
                idx for idx in range(len(raw_terms)) if idx != drop_index
            }
        else:
            selected_indices = set(range(len(raw_terms)))
        for idx, (sign_value, log_abs_value, _mol_index, row) in enumerate(raw_terms):
            selected = idx in selected_indices
            row["selected"] = bool(selected)
            term_rows.append(row)
            if selected:
                _append_coefficient_source_value_trace(
                    element_index=element_index,
                    order=order,
                    solver_scaling_factor=solver_scaling_factor,
                    term_row=row,
                )
            if selected:
                selected_terms.append((sign_value, log_abs_value))
        sign, log_abs = _signed_log_add_terms(selected_terms)
        return sign, log_abs, term_rows

    def _coeff(
        element_index: int,
        order: int,
        n_min: float,
        n_maj: float,
        solver_scaling_factor: float,
    ) -> float:
        if order == 0:
            total = float(
                n_maj
                + n_min
                - coefficient_gas_density * phi[element_index]
            )
            if use_solver_scaling_factor:
                total *= _scaled_exp(0.0, solver_scaling_factor)
            return float(total)
        total = (
            _scaled_exp(0.0, solver_scaling_factor)
            if order == 1 and use_solver_scaling_factor
            else (1.0 if order == 1 else 0.0)
        )
        coefficient_molecules = (
            molecule_list_by_element.get(element_index, [])
            if element_solver_coefficient_molecule_source == "molecule_list"
            else major_molecules_by_element.get(element_index, [])
        )
        if element_solver_coefficient_abundance_gate:
            coefficient_molecules = [
                mol_index
                for mol_index in coefficient_molecules
                if _molecule_abundance_matches_element(element_index, mol_index)
            ]
        for mol_index in coefficient_molecules:
            stoich_self = int(round(molecule_formula[element_index, mol_index]))
            if stoich_self != order:
                continue
            donor = _donor_log_sum(element_index, mol_index)
            kappa = float(order + phi[element_index] * molecule_sigma[mol_index])
            term = float(_coefficient_mass_action_log(mol_index) + donor)
            total += float(_scaled_exp(term, solver_scaling_factor) * kappa)
        return float(total)

    def _solve_element_density(element_index: int, n_maj: float) -> bool:
        nonlocal element_solver_log_density_cgs
        n_maj_effective = _element_solver_n_major(element_index, n_maj)
        element_label = (
            element_labels[element_index]
            if 0 <= element_index < len(element_labels)
            else str(element_index)
        )
        trace_enabled = (
            emit_element_solver_trace
            and current_replay_iteration >= 0
            and current_replay_iteration < element_solver_trace_iteration_limit
            and len(element_solver_trace_records) < element_solver_trace_max_records
        )

        def _record_element_solver_log_materialization(
            *,
            stage: str,
            policy: str,
            pre_assignment_log: float,
            assigned_log_cgs_value: Optional[float] = None,
            candidate_log_cgs_value: Optional[float] = None,
            checkn_input_internal_value: Optional[float] = None,
            checkn_output_internal_value: Optional[float] = None,
        ) -> dict[str, Any]:
            record = {
                "iteration": int(current_replay_iteration),
                "element_index": int(element_index),
                "element_label": str(element_label),
                "materialization_stage": str(stage),
                "materialization_policy": str(policy),
                "pre_assignment_element_solver_log": float(pre_assignment_log),
                "post_assignment_element_solver_log": float(
                    element_solver_log_density_cgs[element_index]
                ),
                "candidate_log_cgs_at_materialization": (
                    None
                    if candidate_log_cgs_value is None
                    else float(candidate_log_cgs_value)
                ),
                "assigned_log_cgs_at_materialization": (
                    None
                    if assigned_log_cgs_value is None
                    else float(assigned_log_cgs_value)
                ),
                "checkN_input_internal_at_materialization": (
                    None
                    if checkn_input_internal_value is None
                    else float(checkn_input_internal_value)
                ),
                "checkN_output_internal_at_materialization": (
                    None
                    if checkn_output_internal_value is None
                    else float(checkn_output_internal_value)
                ),
                "species_density_internal_at_materialization": float(
                    species_density[element_index]
                ),
                "density_domain_scale": float(density_domain_scale),
                "element_solver_output_log_policy": element_solver_output_log_policy,
                "trace_marker": (
                    "exact_fixed_row_subspace_trace_m372_"
                    "element_solver_log_carrier_materialization"
                ),
                "diagnostic_only": True,
                "default_off": True,
                "KL_owned": True,
                "FastChem_trace_values_used_as_inputs": False,
                "used_as_KL_constructor_input": False,
            }
            latest_element_solver_log_materialization[int(element_index)] = record
            return record

        def _append_element_solver_trace(row: dict[str, Any]) -> None:
            event = {
                "iteration": int(current_replay_iteration),
                "element_index": int(element_index),
                "element_label": str(element_label),
                "n_major_input": float(n_maj),
                "n_major_effective": float(n_maj_effective),
                "per_element_n_major_timing_replay_enabled": bool(
                    per_element_n_major_timing_replay_enabled
                ),
                "per_element_n_major_timing_replay_field": (
                    per_element_n_major_timing_replay_field
                ),
                "per_element_n_major_timing_snapshot": dict(
                    per_element_n_major_timing_history.get(element_index, {})
                ),
                "element_solver_n_major_source": element_solver_n_major_source,
                "element_solver_n_major_update_source": (
                    element_solver_n_major_update_source
                ),
                "coefficient_density_scale": float(coefficient_density_scale),
                "coefficient_density_floor": float(coefficient_density_floor),
                "coefficient_gas_density": float(coefficient_gas_density),
                "element_density_minlimit": float(element_density_minlimit),
                "element_checkN_upper_bound_internal": float(
                    element_checkN_upper_bound_internal
                ),
                "element_solver_mode": element_solver_mode,
                "element_solver_coefficient_density_domain": (
                    element_solver_coefficient_density_domain
                ),
                "element_solver_fixed_by_condensation_policy": (
                    element_solver_fixed_by_condensation_policy
                ),
                "element_solver_nonpositive_candidate_policy": (
                    element_solver_nonpositive_candidate_policy
                ),
                "element_solver_output_log_policy": element_solver_output_log_policy,
                "element_solver_subdouble_log_root_policy": (
                    element_solver_subdouble_log_root_policy
                ),
                "element_solver_subdouble_log_root_iteration_limit": (
                    element_solver_subdouble_log_root_iteration_limit
                ),
                "element_solver_coefficient_donor_log_source": (
                    element_solver_coefficient_donor_log_source
                ),
                "element_solver_minor_density_source": (
                    element_solver_minor_density_source
                ),
                "fixed_by_condensation": bool(fixed_by_condensation[element_index]),
                "phi": float(phi[element_index]),
                "diagnostic_only": True,
                "default_off": True,
                "KL_owned": True,
                "FastChem_trace_values_used_as_inputs": False,
                "used_as_KL_constructor_input": False,
                "trace_marker": (
                    "exact_fixed_row_subspace_trace_m370_"
                    "kl_quadratic_coefficient_donor_source_terms "
                    "exact_fixed_row_subspace_trace_m369_"
                    "kl_element_slot_write_candidate_coefficient_terms "
                    "exact_fixed_row_subspace_trace_m368_"
                    "kl_species_density_element_slot_write_lineage"
                ),
                **row,
            }
            last_element_slot_write_lineage[int(element_index)] = {
                key: event.get(key)
                for key in (
                    "iteration",
                    "element_index",
                    "element_label",
                    "branch_selected",
                    "solver_dispatched",
                    "solver_order",
                    "n_major_input",
                    "n_major_effective",
                    "n_minor",
                    "solver_scaling_factor",
                    "coefficient_density_scale",
                    "coefficient_density_floor",
                    "coefficient_gas_density",
                    "element_density_minlimit",
                    "element_checkN_upper_bound_internal",
                    "element_solver_mode",
                    "element_solver_coefficient_density_domain",
                    "element_solver_output_log_policy",
                    "element_solver_coefficient_donor_log_source",
                    "element_solver_minor_density_source",
                    "a0",
                    "a1",
                    "a2",
                    "a1_sign",
                    "a1_log_abs",
                    "a1_signed_log_terms_top",
                    "a2_sign",
                    "a2_log_abs",
                    "a2_signed_log_terms_top",
                    "element_solver_signed_log_term_selection",
                    "quadratic_discriminant",
                    "quadratic_qj",
                    "quadratic_qj_log_abs",
                    "quadratic_logspace_candidate_used",
                    "quadratic_root_reconstruction_trace",
                    "candidate_before_coefficient_scale",
                    "candidate_log_cgs",
                    "candidate_internal",
                    "checkN_input_internal",
                    "checkN_min_internal",
                    "checkN_output_internal",
                    "species_density_write_value",
                    "assigned_log_cgs",
                    "element_solver_log_density_cgs",
                    "element_solver_log_materialization",
                    "nonpositive_candidate_policy_applied",
                    "minlimit_policy",
                    "upper_bound_policy",
                    "fixed_by_condensation",
                    "phi",
                    "trace_marker",
                )
                if key in event
            }
            if trace_enabled:
                element_solver_trace_records.append(event)

        if not element_solver_lifecycle_enabled or element_index == electron_index:
            pre_assignment_log = float(element_solver_log_density_cgs[element_index])
            materialization = _record_element_solver_log_materialization(
                stage="disabled_or_electron_existing_current",
                policy="no_assignment_existing_current",
                pre_assignment_log=pre_assignment_log,
                assigned_log_cgs_value=pre_assignment_log,
                candidate_log_cgs_value=float(
                    element_solver_candidate_log_density_cgs[element_index]
                ),
                checkn_output_internal_value=float(species_density[element_index]),
            )
            _append_element_solver_trace(
                {
                    "branch_selected": "disabled_or_electron",
                    "solver_dispatched": False,
                    "element_solver_log_materialization": materialization,
                    "species_density_write_value": float(
                        species_density[element_index]
                    ),
                    "assigned_log_cgs": _element_log_from_cgs(
                        float(species_density[element_index]) * density_domain_scale
                    ),
                }
            )
            return False
        if (
            fixed_by_condensation[element_index]
            and element_solver_fixed_by_condensation_policy != "solve"
        ):
            candidate_before_clamp = float(species_density[element_index])
            species_density[element_index] = min(
                max(float(species_density[element_index]), element_density_minlimit),
                1.0,
            )
            pre_assignment_log = float(element_solver_log_density_cgs[element_index])
            element_solver_log_density_cgs[element_index] = _element_log_from_cgs(
                float(species_density[element_index]) * density_domain_scale
            )
            materialization = _record_element_solver_log_materialization(
                stage="fixed_by_condensation_checkN_assignment",
                policy="assigned_log_after_fixed_condensation_clamp",
                pre_assignment_log=pre_assignment_log,
                assigned_log_cgs_value=float(
                    element_solver_log_density_cgs[element_index]
                ),
                candidate_log_cgs_value=float(
                    element_solver_candidate_log_density_cgs[element_index]
                ),
                checkn_input_internal_value=candidate_before_clamp,
                checkn_output_internal_value=float(species_density[element_index]),
            )
            _append_element_solver_trace(
                {
                    "branch_selected": "fixed_by_condensation",
                    "solver_dispatched": False,
                    "intertSol": False,
                    "linear": False,
                    "quadratic": False,
                    "Newton": False,
                    "backup": False,
                    "candidate_before_coefficient_scale": candidate_before_clamp,
                    "candidate_internal": candidate_before_clamp,
                    "checkN_input_internal": candidate_before_clamp,
                    "checkN_min_internal": float(element_density_minlimit),
                    "checkN_upper_internal": 1.0,
                    "checkN_output_internal": float(species_density[element_index]),
                    "species_density_write_value": float(
                        species_density[element_index]
                    ),
                    "assigned_log_cgs": float(
                        element_solver_log_density_cgs[element_index]
                    ),
                    "element_solver_log_materialization": materialization,
                    "nonpositive_candidate_policy_applied": False,
                    "minlimit_policy": "element_density_minlimit",
                    "upper_bound_policy": "fixed_by_condensation_upper_1",
                }
            )
            return False
        if phi[element_index] <= 0.0:
            species_density[element_index] = 0.0
            pre_assignment_log = float(element_solver_log_density_cgs[element_index])
            element_solver_log_density_cgs[element_index] = (
                fastchem_longdouble_minlimit_log_cgs
            )
            materialization = _record_element_solver_log_materialization(
                stage="zero_phi_minlimit_assignment",
                policy="fastchem_longdouble_minlimit_log",
                pre_assignment_log=pre_assignment_log,
                assigned_log_cgs_value=fastchem_longdouble_minlimit_log_cgs,
                candidate_log_cgs_value=float(
                    element_solver_candidate_log_density_cgs[element_index]
                ),
                checkn_output_internal_value=0.0,
            )
            _append_element_solver_trace(
                {
                    "branch_selected": "zero_phi",
                    "solver_dispatched": False,
                    "species_density_write_value": 0.0,
                    "assigned_log_cgs": fastchem_longdouble_minlimit_log_cgs,
                    "element_solver_log_materialization": materialization,
                }
            )
            return False
        n_min = _minor_density(element_index)
        order_molecules = _solver_molecule_indices(
            element_index,
            source=element_solver_order_source,
        )
        orders = [
            int(round(molecule_formula[element_index, mol_index]))
            for mol_index in order_molecules
            if molecule_formula[element_index, mol_index] > 0.0
        ]
        solver_order = max(orders) if orders else 0
        solver_scaling_factor = _solver_scaling_factor(element_index, solver_order)
        if 0 <= element_index < len(element_labels):
            element_solver_scaling_factors[str(element_labels[element_index])] = float(
                solver_scaling_factor
            )
        a0 = _coeff(
            element_index,
            0,
            n_min,
            n_maj_effective * float(phi[element_index]) * coefficient_density_scale,
            solver_scaling_factor,
        )
        a1_trace: Optional[float] = None
        a2_trace: Optional[float] = None
        discriminant_trace: Optional[float] = None
        qj_trace: Optional[float] = None
        logspace_candidate_used = False
        a1_log_abs_trace: Optional[float] = None
        a2_log_abs_trace: Optional[float] = None
        a1_sign_trace: Optional[int] = None
        a2_sign_trace: Optional[int] = None
        qj_log_abs_trace: Optional[float] = None
        a1_term_trace: Optional[list[dict[str, Any]]] = None
        a2_term_trace: Optional[list[dict[str, Any]]] = None
        subdouble_log_root: Optional[float] = None
        quadratic_root_reconstruction_trace: Optional[dict[str, Any]] = None
        if element_solver_mode == "intertSol_only" or solver_order <= 0:
            branch_selected = "intertSol"
            candidate = -a0
            candidate_log_cgs = _element_log_from_cgs(float(candidate))
        elif solver_order == 1:
            branch_selected = "linear"
            a1 = _coeff(element_index, 1, n_min, n_maj_effective, solver_scaling_factor)
            a1_trace = float(a1)
            candidate = -a0 / max(a1, 1.0e-300)
            if -a0 > 0.0 and a1 > 0.0 and np.isfinite(a0) and np.isfinite(a1):
                candidate_log_cgs = float(np.log(-a0) - np.log(a1))
            else:
                candidate_log_cgs = _element_log_from_cgs(float(candidate))
        elif solver_order == 2:
            branch_selected = "quadratic"
            a1 = _coeff(element_index, 1, n_min, n_maj_effective, solver_scaling_factor)
            a2 = _coeff(element_index, 2, n_min, n_maj_effective, solver_scaling_factor)
            a1_trace = float(a1)
            a2_trace = float(a2)
            if element_solver_quadratic_precision == "signed_log_overflow_carrier":
                a0_sign, a0_log_abs = _coeff_signed_log_abs(
                    element_index,
                    0,
                    n_min,
                    n_maj_effective
                    * float(phi[element_index])
                    * coefficient_density_scale,
                    solver_scaling_factor,
                )
                a1_sign, a1_log_abs, a1_terms = _coeff_signed_log_abs_with_terms(
                    element_index, 1, n_min, n_maj_effective, solver_scaling_factor
                )
                a2_sign, a2_log_abs, a2_terms = _coeff_signed_log_abs_with_terms(
                    element_index, 2, n_min, n_maj_effective, solver_scaling_factor
                )
                a1_sign_trace = a1_sign
                a2_sign_trace = a2_sign
                a1_log_abs_trace = a1_log_abs
                a2_log_abs_trace = a2_log_abs
                quadratic_root_reconstruction_trace = {
                    "trace_marker": (
                        "exact_fixed_row_subspace_trace_m373_"
                        "quadratic_root_reconstruction"
                    ),
                    "root_branch": "signed_log_overflow_carrier",
                    "a0": float(a0),
                    "a1": float(a1),
                    "a2": float(a2),
                    "a0_sign": int(a0_sign),
                    "a0_log_abs": float(a0_log_abs),
                    "a1_sign": int(a1_sign),
                    "a1_log_abs": float(a1_log_abs),
                    "a2_sign": int(a2_sign),
                    "a2_log_abs": float(a2_log_abs),
                    "discriminant_terms": [],
                    "discriminant_sign": None,
                    "discriminant_log_abs": None,
                    "sqrt_discriminant_log_abs": None,
                    "qj_sign": None,
                    "qj_log_abs": None,
                    "candidate_log_cgs_from_root": None,
                    "diagnostic_only": True,
                    "default_off": True,
                    "KL_owned": True,
                    "FastChem_trace_values_used_as_inputs": False,
                    "used_as_KL_constructor_input": False,
                }
                a1_term_trace = sorted(
                    a1_terms,
                    key=lambda row: float(row.get("term_log_abs", -np.inf)),
                    reverse=True,
                )[:element_solver_signed_log_trace_top_n]
                a2_term_trace = sorted(
                    a2_terms,
                    key=lambda row: float(row.get("term_log_abs", -np.inf)),
                    reverse=True,
                )[:element_solver_signed_log_trace_top_n]
                if a0_sign < 0 and a2_sign > 0 and np.isfinite(a0_log_abs):
                    disc_terms = []
                    if a1_sign != 0:
                        disc_terms.append((1.0, float(2.0 * a1_log_abs)))
                    disc_terms.append(
                        (1.0, float(np.log(4.0) + a2_log_abs + a0_log_abs))
                    )
                    quadratic_root_reconstruction_trace[
                        "discriminant_terms"
                    ] = [
                        {
                            "sign": float(sign),
                            "log_abs": float(log_abs),
                            "term": (
                                "a1_squared"
                                if index == 0 and len(disc_terms) == 2
                                else "minus_4_a2_a0"
                            ),
                        }
                        for index, (sign, log_abs) in enumerate(disc_terms)
                    ]
                    disc_sign, disc_log_abs = _signed_log_add_terms(disc_terms)
                    quadratic_root_reconstruction_trace[
                        "discriminant_sign"
                    ] = int(disc_sign)
                    quadratic_root_reconstruction_trace[
                        "discriminant_log_abs"
                    ] = float(disc_log_abs)
                    if disc_sign > 0:
                        sqrt_disc_log_abs = float(0.5 * disc_log_abs)
                        quadratic_root_reconstruction_trace[
                            "sqrt_discriminant_log_abs"
                        ] = sqrt_disc_log_abs
                        if a1_sign >= 0:
                            qj_log_abs = float(
                                np.log(0.5)
                                + np.logaddexp(a1_log_abs, sqrt_disc_log_abs)
                            )
                            qj_sign = -1
                        else:
                            if sqrt_disc_log_abs > a1_log_abs:
                                qj_log_abs = float(
                                    np.log(0.5)
                                    + sqrt_disc_log_abs
                                    + np.log1p(
                                        -np.exp(a1_log_abs - sqrt_disc_log_abs)
                                    )
                                )
                                qj_sign = -1
                            elif sqrt_disc_log_abs < a1_log_abs:
                                qj_log_abs = float(
                                    np.log(0.5)
                                    + a1_log_abs
                                    + np.log1p(
                                        -np.exp(sqrt_disc_log_abs - a1_log_abs)
                                    )
                                )
                                qj_sign = 1
                            else:
                                qj_log_abs = -np.inf
                                qj_sign = 0
                        if qj_sign < 0 and np.isfinite(qj_log_abs):
                            qj_log_abs_trace = qj_log_abs
                            candidate_log_cgs = float(a0_log_abs - qj_log_abs)
                            quadratic_root_reconstruction_trace[
                                "qj_sign"
                            ] = int(qj_sign)
                            quadratic_root_reconstruction_trace[
                                "qj_log_abs"
                            ] = float(qj_log_abs)
                            quadratic_root_reconstruction_trace[
                                "candidate_log_cgs_from_root"
                            ] = float(candidate_log_cgs)
                            with np.errstate(over="ignore", under="ignore"):
                                candidate = float(np.exp(candidate_log_cgs))
                            logspace_candidate_used = True
                        else:
                            quadratic_root_reconstruction_trace[
                                "qj_sign"
                            ] = int(qj_sign)
                            quadratic_root_reconstruction_trace[
                                "qj_log_abs"
                            ] = float(qj_log_abs)
                            candidate = np.nan
                            candidate_log_cgs = fastchem_longdouble_minlimit_log_cgs
                    else:
                        candidate = np.nan
                        candidate_log_cgs = fastchem_longdouble_minlimit_log_cgs
                else:
                    candidate = np.nan
                    candidate_log_cgs = fastchem_longdouble_minlimit_log_cgs
                discriminant_trace = float("inf")
                qj_trace = (
                    -float("inf")
                    if qj_log_abs_trace is not None
                    and np.isposinf(qj_log_abs_trace)
                    else float("nan")
                )
            elif element_solver_quadratic_precision == "longdouble_log_carrier":
                a0_ld = np.longdouble(a0)
                a1_ld = np.longdouble(a1)
                a2_ld = np.longdouble(a2)
                discriminant_ld = a1_ld * a1_ld - np.longdouble(4.0) * a2_ld * a0_ld
                discriminant_trace = float(discriminant_ld)
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    qj_ld = -np.longdouble(0.5) * (
                        a1_ld + np.sqrt(discriminant_ld)
                    )
                    candidate_ld = a0_ld / qj_ld
                qj_trace = float(qj_ld)
                candidate = float(candidate_ld)
                candidate_log_cgs = _element_log_from_cgs_longdouble(candidate_ld)
                quadratic_root_reconstruction_trace = {
                    "trace_marker": (
                        "exact_fixed_row_subspace_trace_m373_"
                        "quadratic_root_reconstruction"
                    ),
                    "root_branch": "longdouble_log_carrier",
                    "a0": float(a0),
                    "a1": float(a1),
                    "a2": float(a2),
                    "quadratic_discriminant": float(discriminant_ld),
                    "quadratic_qj": float(qj_ld),
                    "candidate_log_cgs_from_root": float(candidate_log_cgs),
                    "diagnostic_only": True,
                    "default_off": True,
                    "KL_owned": True,
                    "FastChem_trace_values_used_as_inputs": False,
                    "used_as_KL_constructor_input": False,
                }
            else:
                discriminant = a1 * a1 - 4.0 * a2 * a0
                discriminant_trace = float(discriminant)
                if element_solver_quadratic_branch_mode == "fastchem_raw":
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        qj = -0.5 * (a1 + np.sqrt(discriminant))
                        candidate = float(np.divide(a0, qj))
                else:
                    discriminant = max(discriminant, 0.0)
                    qj = -0.5 * (a1 + math.sqrt(discriminant))
                    candidate = (
                        a0 / qj if abs(qj) > 1.0e-300
                        else -a0 / max(a1, 1.0e-300)
                    )
                qj_trace = float(qj)
                if (
                    a0 != 0.0
                    and qj != 0.0
                    and np.sign(a0) == np.sign(qj)
                    and np.isfinite(a0)
                    and np.isfinite(qj)
                ):
                    candidate_log_cgs = float(np.log(abs(a0)) - np.log(abs(qj)))
                else:
                    candidate_log_cgs = _element_log_from_cgs(float(candidate))
                quadratic_root_reconstruction_trace = {
                    "trace_marker": (
                        "exact_fixed_row_subspace_trace_m373_"
                        "quadratic_root_reconstruction"
                    ),
                    "root_branch": "double_raw_or_guarded",
                    "a0": float(a0),
                    "a1": float(a1),
                    "a2": float(a2),
                    "quadratic_discriminant": float(discriminant_trace),
                    "quadratic_qj": float(qj),
                    "candidate_log_cgs_from_root": float(candidate_log_cgs),
                    "diagnostic_only": True,
                    "default_off": True,
                    "KL_owned": True,
                    "FastChem_trace_values_used_as_inputs": False,
                    "used_as_KL_constructor_input": False,
                }
        else:
            branch_selected = "Newton"
            coeffs = [
                _coeff(
                    element_index,
                    order,
                    n_min,
                    n_maj_effective,
                    solver_scaling_factor,
                )
                for order in range(solver_order + 1)
            ]
            subdouble_policy_active = (
                element_solver_subdouble_log_root_iteration_limit < 0
                or current_replay_iteration
                < element_solver_subdouble_log_root_iteration_limit
            )
            if (
                subdouble_policy_active
                and element_solver_subdouble_log_root_policy == "logspace_bisection"
            ):
                upper_log = float(np.log(max(coefficient_gas_density, 1.0)))
                subdouble_log_root = _logspace_positive_root(
                    coeffs,
                    lower_log=fastchem_longdouble_minlimit_log_cgs,
                    upper_log=upper_log,
                )
            elif (
                subdouble_policy_active
                and
                element_solver_subdouble_log_root_policy
                == "dominant_linear_signed_log"
            ):
                a0_sign_log, a0_log_abs = _coeff_signed_log_abs(
                    element_index,
                    0,
                    n_min,
                    n_maj_effective
                    * float(phi[element_index])
                    * coefficient_density_scale,
                    solver_scaling_factor,
                )
                a1_sign_log, a1_log_abs, _a1_terms = _coeff_signed_log_abs_with_terms(
                    element_index,
                    1,
                    n_min,
                    n_maj_effective,
                    solver_scaling_factor,
                )
                if (
                    a0_sign_log < 0
                    and a1_sign_log > 0
                    and np.isfinite(a0_log_abs)
                    and np.isfinite(a1_log_abs)
                ):
                    subdouble_log_root = float(a0_log_abs - a1_log_abs)
            raw_x = float(species_density[element_index]) * coefficient_density_scale
            if (
                element_solver_newton_initial_guess_policy
                == "fastchem_species_or_gas_density"
                and raw_x == 0.0
            ):
                x = float(coefficient_gas_density)
            elif element_solver_newton_initial_guess_policy == "native_species_density":
                x = float(raw_x)
            else:
                x = max(raw_x, coefficient_density_floor)

            def _poly(coefficients: list[float], point: float) -> float:
                order_local = len(coefficients) - 1
                value = coefficients[order_local]
                for local_order in range(order_local - 1, -1, -1):
                    value = coefficients[local_order] + point * value
                return float(value)

            def _newton_root(
                coefficients: list[float],
                initial_x: float,
            ) -> tuple[float, bool]:
                order_local = len(coefficients) - 1
                x_value = float(initial_x)
                converged = False
                for _ in range(element_solver_newton_max_iter):
                    p = coefficients[order_local]
                    dp = order_local * coefficients[order_local]
                    for local_order in range(order_local - 1, 0, -1):
                        p = coefficients[local_order] + x_value * p
                        dp = local_order * coefficients[local_order] + x_value * dp
                    p = coefficients[0] + x_value * p
                    if (
                        element_solver_newton_derivative_zero_policy == "guard_break"
                        and abs(dp) <= 1.0e-300
                    ):
                        break
                    if element_solver_newton_derivative_zero_policy == "native_divide":
                        with np.errstate(divide="ignore", invalid="ignore"):
                            x_new = float(x_value - np.divide(p, dp))
                    else:
                        x_new = x_value - p / dp
                    if abs(x_new - x_value) < (
                        element_solver_newton_err * abs(x_new)
                    ):
                        x_value = float(x_new)
                        converged = True
                        break
                    if x_new < 1.0e-8 * x_value:
                        x_new = 1.0e-8 * x_value
                    x_value = float(x_new)
                x_lower = max(0.0, x_value * (1.0 - element_solver_newton_err))
                x_upper = x_value * (1.0 + element_solver_newton_err)
                invalid = (
                    x_value < 0.0
                    or not converged
                    or _poly(coefficients, x_lower) * _poly(coefficients, x_upper)
                    > 0.0
                )
                if (
                    element_solver_newton_assignment_policy
                    == "fastchem_converged_only"
                    and not converged
                ):
                    return float(initial_x), bool(invalid)
                return float(x_value), bool(invalid)

            def _newton_root_longdouble(
                coefficients: list[float],
                initial_x: float,
            ) -> tuple[np.longdouble, bool]:
                coeff_ld = [np.longdouble(value) for value in coefficients]
                order_local = len(coeff_ld) - 1
                x_value = np.longdouble(initial_x)
                converged = False
                for _ in range(element_solver_newton_max_iter):
                    p = coeff_ld[order_local]
                    dp = np.longdouble(order_local) * coeff_ld[order_local]
                    for local_order in range(order_local - 1, 0, -1):
                        p = coeff_ld[local_order] + x_value * p
                        dp = (
                            np.longdouble(local_order) * coeff_ld[local_order]
                            + x_value * dp
                        )
                    p = coeff_ld[0] + x_value * p
                    if (
                        element_solver_newton_derivative_zero_policy == "guard_break"
                        and abs(dp) <= np.longdouble(1.0e-300)
                    ):
                        break
                    with np.errstate(divide="ignore", invalid="ignore"):
                        x_new = x_value - p / dp
                    if abs(x_new - x_value) < (
                        np.longdouble(element_solver_newton_err) * abs(x_new)
                    ):
                        x_value = x_new
                        converged = True
                        break
                    if x_new < np.longdouble(1.0e-8) * x_value:
                        x_new = np.longdouble(1.0e-8) * x_value
                    x_value = x_new
                x_lower = max(np.longdouble(0.0), x_value * (1.0 - element_solver_newton_err))
                x_upper = x_value * (1.0 + element_solver_newton_err)

                def _poly_ld(point: np.longdouble) -> np.longdouble:
                    value = coeff_ld[order_local]
                    for local_order in range(order_local - 1, -1, -1):
                        value = coeff_ld[local_order] + point * value
                    return value

                invalid = (
                    x_value < 0.0
                    or not converged
                    or _poly_ld(x_lower) * _poly_ld(x_upper) > 0.0
                )
                if (
                    element_solver_newton_assignment_policy
                    == "fastchem_converged_only"
                    and not converged
                ):
                    return np.longdouble(initial_x), bool(invalid)
                return x_value, bool(invalid)

            x_longdouble = None
            if element_solver_output_log_policy == "post_checkN_longdouble":
                x_longdouble, _ = _newton_root_longdouble(coeffs, x)
            x, invalid_root = _newton_root(coeffs, x)
            backup_used = False
            if (
                invalid_root
                and element_solver_newton_fallback_policy
                == "fastchem_alternative_bisection"
            ):
                backup_used = True
                alternative_order = 0
                for mol_index in molecule_list_by_element.get(element_index, []):
                    stoich_self = int(round(molecule_formula[element_index, mol_index]))
                    if stoich_self > alternative_order:
                        alternative_order = stoich_self
                if alternative_order > 0:
                    n_exc = 0.0
                    for mol_index in range(molecule_formula.shape[1]):
                        if molecule_formula[element_index, mol_index] == 0.0:
                            n_exc += (
                                molecule_sigma[mol_index]
                                * species_density[n_elements + mol_index]
                                * coefficient_density_scale
                            )
                    n_exc *= float(phi[element_index])
                    alt_coeffs = [
                        float(n_exc - coefficient_gas_density * phi[element_index])
                    ]
                    if use_solver_scaling_factor:
                        alt_coeffs[0] *= _scaled_exp(0.0, solver_scaling_factor)
                    for order in range(1, alternative_order + 1):
                        total = 0.0
                        for mol_index in molecule_list_by_element.get(element_index, []):
                            stoich_self = int(
                                round(molecule_formula[element_index, mol_index])
                            )
                            if stoich_self != order:
                                continue
                            donor = _donor_log_sum(element_index, mol_index)
                            total += float(
                                _scaled_exp(
                                    _coefficient_mass_action_log(mol_index) + donor,
                                    solver_scaling_factor,
                                )
                                * order
                            )
                        if order == 1:
                            total += (
                                _scaled_exp(0.0, solver_scaling_factor)
                                if use_solver_scaling_factor
                                else 1.0
                            )
                        alt_coeffs.append(float(total))
                    x, invalid_root = _newton_root(alt_coeffs, x)
                    if invalid_root and element_solver_bisection_max_iter > 0:
                        lower = (
                            element_density_minlimit
                            if element_solver_coefficient_density_domain == "cgs"
                            else element_density_minlimit * coefficient_density_scale
                        )
                        upper = coefficient_gas_density
                        for _ in range(element_solver_bisection_max_iter):
                            midpoint = (upper - lower) * 0.5 + lower
                            if -_poly(alt_coeffs, midpoint) < 0.0:
                                upper = midpoint
                            else:
                                lower = midpoint
                            if (
                                upper > 0.0
                                and abs(lower - upper) / upper
                                < chem_accuracy * 1.0e-3
                            ):
                                break
                        x = lower
            candidate = x
            if x_longdouble is not None:
                candidate_log_cgs = _element_log_from_cgs_longdouble(x_longdouble)
            else:
                candidate_log_cgs = _element_log_from_cgs(float(candidate))
            if (
                subdouble_log_root is not None
                and np.isfinite(subdouble_log_root)
                and element_solver_subdouble_log_root_policy
                in {"logspace_bisection", "dominant_linear_signed_log"}
            ):
                candidate_log_cgs = float(subdouble_log_root)
        if not np.isfinite(candidate) or (
            candidate <= 0.0
            and element_solver_nonpositive_candidate_policy != "fastchem_zero_checkN"
        ):
            nonpositive_candidate_policy_applied = True
            candidate = (
                float(species_density[element_index]) * coefficient_density_scale
            )
        else:
            nonpositive_candidate_policy_applied = False
        candidate_internal = candidate / coefficient_density_scale
        element_density_minlimit_internal = (
            element_density_minlimit / coefficient_density_scale
            if element_solver_coefficient_density_domain == "cgs"
            else element_density_minlimit
        )
        checkn_input_internal = float(candidate_internal)
        checkn_lower_internal = (
            0.0
            if fixed_by_condensation[element_index]
            and element_solver_fixed_checkn_min_policy
            == "skip_min_for_fixed_by_condensation"
            else element_density_minlimit_internal
        )
        species_density[element_index] = min(
            max(float(candidate_internal), checkn_lower_internal),
            element_checkN_upper_bound_internal,
        )
        assigned_log_cgs = _element_log_from_cgs(
            float(species_density[element_index]) * density_domain_scale
        )
        pre_assignment_log = float(element_solver_log_density_cgs[element_index])
        if (
            element_solver_output_log_policy == "pre_clamp_candidate"
            and np.isfinite(candidate_log_cgs)
        ):
            materialization_policy = "pre_clamp_candidate"
            element_solver_log_density_cgs[element_index] = float(candidate_log_cgs)
        elif (
            element_solver_output_log_policy
            == "subdouble_candidate_else_post_checkN_longdouble"
            and subdouble_log_root is not None
            and np.isfinite(candidate_log_cgs)
        ):
            materialization_policy = "subdouble_candidate_else_post_checkN_longdouble"
            element_solver_log_density_cgs[element_index] = float(candidate_log_cgs)
        elif (
            element_solver_output_log_policy == "post_checkN_longdouble"
            and np.isfinite(candidate_log_cgs)
        ):
            materialization_policy = "post_checkN_longdouble"
            element_solver_log_density_cgs[element_index] = _post_checkn_log_cgs(
                float(candidate_log_cgs),
                assigned_log_cgs,
                element_density_minlimit_internal,
            )
        else:
            materialization_policy = "assigned_log_after_checkN"
            element_solver_log_density_cgs[element_index] = assigned_log_cgs
        element_solver_candidate_log_density_cgs[element_index] = float(
            candidate_log_cgs
        )
        if quadratic_root_reconstruction_trace is not None:
            quadratic_root_reconstruction_trace = {
                **quadratic_root_reconstruction_trace,
                "candidate_before_coefficient_scale": float(candidate),
                "candidate_log_cgs": float(candidate_log_cgs),
                "candidate_internal": float(candidate_internal),
                "assigned_log_cgs": float(assigned_log_cgs),
                "element_solver_log_density_cgs": float(
                    element_solver_log_density_cgs[element_index]
                ),
                "candidate_minus_assigned_log_delta": (
                    float(candidate_log_cgs - assigned_log_cgs)
                    if np.isfinite(candidate_log_cgs)
                    and np.isfinite(assigned_log_cgs)
                    else None
                ),
                "logspace_candidate_used": bool(logspace_candidate_used),
                "nonpositive_candidate_policy_applied": bool(
                    nonpositive_candidate_policy_applied
                ),
                "checkN_input_internal": float(checkn_input_internal),
                "checkN_output_internal": float(species_density[element_index]),
            }
        materialization = _record_element_solver_log_materialization(
            stage=f"{branch_selected}_output_log_assignment",
            policy=materialization_policy,
            pre_assignment_log=pre_assignment_log,
            assigned_log_cgs_value=assigned_log_cgs,
            candidate_log_cgs_value=float(candidate_log_cgs),
            checkn_input_internal_value=checkn_input_internal,
            checkn_output_internal_value=float(species_density[element_index]),
        )
        _append_element_solver_trace(
            {
                "branch_selected": branch_selected,
                "solver_dispatched": True,
                "intertSol": branch_selected == "intertSol",
                "linear": branch_selected == "linear",
                "quadratic": branch_selected == "quadratic",
                "Newton": branch_selected == "Newton",
                "backup": bool(locals().get("backup_used", False)),
                "solver_order": int(solver_order),
                "n_minor": float(n_min),
                "solver_scaling_factor": float(solver_scaling_factor),
                "a0": float(a0),
                "a1": None if a1_trace is None else float(a1_trace),
                "a2": None if a2_trace is None else float(a2_trace),
                "a1_sign": a1_sign_trace,
                "a1_log_abs": a1_log_abs_trace,
                "a1_signed_log_terms_top": a1_term_trace,
                "a2_sign": a2_sign_trace,
                "a2_log_abs": a2_log_abs_trace,
                "a2_signed_log_terms_top": a2_term_trace,
                "element_solver_signed_log_term_selection": (
                    element_solver_signed_log_term_selection
                ),
                "quadratic_discriminant": (
                    None
                    if discriminant_trace is None
                    else float(discriminant_trace)
                ),
                "quadratic_qj": None if qj_trace is None else float(qj_trace),
                "quadratic_qj_log_abs": qj_log_abs_trace,
                "quadratic_logspace_candidate_used": bool(logspace_candidate_used),
                "quadratic_root_reconstruction_trace": (
                    quadratic_root_reconstruction_trace
                ),
                "candidate_before_coefficient_scale": float(candidate),
                "candidate_log_cgs": float(candidate_log_cgs),
                "candidate_internal": float(candidate_internal),
                "checkN_input_internal": checkn_input_internal,
                "checkN_min_internal": float(checkn_lower_internal),
                "checkN_floor_policy": str(element_solver_fixed_checkn_min_policy),
                "checkN_upper_internal": float(element_checkN_upper_bound_internal),
                "checkN_output_internal": float(species_density[element_index]),
                "species_density_write_value": float(species_density[element_index]),
                "assigned_log_cgs": float(assigned_log_cgs),
                "element_solver_log_density_cgs": float(
                    element_solver_log_density_cgs[element_index]
                ),
                "element_solver_log_materialization": materialization,
                "nonpositive_candidate_policy_applied": bool(
                    nonpositive_candidate_policy_applied
                ),
                "minlimit_policy": "element_density_minlimit_internal",
                "upper_bound_policy": "element_checkN_upper_bound_internal",
            }
        )
        return True

    def _refresh_minor_boundary() -> None:
        for element_index in range(n_elements):
            for mol_index in minor_molecules_by_element.get(element_index, []):
                _refresh_molecule(
                    mol_index,
                    element_context=element_index,
                    refresh_kind="minor_boundary",
                )

    electron_iteration_start_density = species_density.copy()
    electron_post_element_density = species_density.copy()
    electron_post_molecule_density = species_density.copy()
    electron_post_minor_boundary_density = species_density.copy()
    electron_current_log_density_cgs: Optional[float] = None
    electron_donor_trace_iteration_order_position = 0
    if electron_index is not None:
        if bool(initial_species_nonpositive[electron_index]):
            previous_iteration_electron_log_density_cgs = (
                fastchem_longdouble_minlimit_log_cgs
            )
        else:
            initial_electron_cgs = float(
                initial_species_density_internal[electron_index]
                * density_domain_scale
            )
            previous_iteration_electron_log_density_cgs = (
                float(np.log(initial_electron_cgs))
                if initial_electron_cgs > 0.0 and np.isfinite(initial_electron_cgs)
                else fastchem_longdouble_minlimit_log_cgs
            )

    def _electron_donor_density_vector() -> np.ndarray:
        if electron_refresh_donor_state_source == "initial":
            return initial_species_density_internal
        if electron_refresh_donor_state_source == "old_density":
            return old_density
        if electron_refresh_donor_state_source == "iteration_start":
            return electron_iteration_start_density
        if electron_refresh_donor_state_source == "post_calculateElementDensities":
            return electron_post_element_density
        if electron_refresh_donor_state_source == "post_molecule_refresh":
            return electron_post_molecule_density
        if electron_refresh_donor_state_source == "post_minor_boundary":
            return electron_post_minor_boundary_density
        if electron_refresh_donor_state_source == "post_electron_self_consistent":
            return species_density
        return species_density

    def _refresh_electron_density() -> None:
        nonlocal electron_current_log_density_cgs
        nonlocal electron_donor_trace_iteration_order_position
        if electron_index is None:
            return
        charge = formula[electron_index, n_elements:]
        donor_species_density = _electron_donor_density_vector()
        if electron_refresh_mode == "fastchem_singly_ion_analytic":
            alpha = 0.0
            beta = 0.0
            alpha_log_terms: list[float] = []
            beta_log_terms: list[float] = []
            pending_electron_donor_trace_rows: list[dict[str, Any]] = []
            input_electron_cgs = float(
                species_density[electron_index] * density_domain_scale
            )
            for mol_index in molecule_list_by_element.get(electron_index, []):
                charge_number = int(round(charge[mol_index]))
                if abs(charge_number) != 1:
                    continue
                stoich = molecule_formula[:, mol_index]
                active = (stoich != 0.0) & (np.arange(n_elements) != electron_index)
                active_indices = np.where(active)[0]
                values = (
                    donor_species_density[:n_elements][active]
                    * density_domain_scale
                )
                if (
                    electron_refresh_donor_log_density_source
                    == "element_solver_log_carrier"
                ):
                    donor_logs = element_solver_log_density_cgs[active]
                elif (
                    electron_refresh_donor_log_density_source
                    == "nonpositive_longdouble_minlimit"
                ):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        donor_logs = np.where(
                            values > 0.0,
                            np.log(values),
                            fastchem_longdouble_minlimit_log_cgs,
                        )
                else:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        donor_logs = np.log(values)
                donor = float(np.dot(stoich[active], donor_logs))
                exponent = float(molecule_mass_action_constants[mol_index] + donor)
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    contribution = float(np.exp(exponent))
                if not np.isfinite(contribution):
                    contribution = float(np.exp(np.clip(exponent, -745.0, 709.0)))
                alpha_before = alpha
                beta_before = beta
                if charge_number == 1:
                    beta += contribution
                    beta_log_terms.append(exponent)
                elif charge_number == -1:
                    alpha += contribution
                    alpha_log_terms.append(exponent)
                if (
                    emit_electron_donor_trace
                    and current_replay_iteration >= 0
                    and current_replay_iteration < electron_donor_trace_iteration_limit
                    and len(electron_donor_trace_records)
                    + len(pending_electron_donor_trace_rows)
                    < electron_donor_trace_max_records
                ):
                    if electron_donor_trace_order_scope == "iteration_global":
                        ion_refresh_order_position = (
                            electron_donor_trace_iteration_order_position
                        )
                        electron_donor_trace_iteration_order_position += 1
                    else:
                        ion_refresh_order_position = len(
                            pending_electron_donor_trace_rows
                        )
                    molecule_density_cgs = float(
                        species_density[n_elements + mol_index] * density_domain_scale
                    )
                    pending_electron_donor_trace_rows.append(
                        {
                            "iteration": int(current_replay_iteration),
                            "ion_refresh_order_position": int(
                                ion_refresh_order_position
                            ),
                            "electron_element_index": int(electron_index),
                            "electron_element_label": (
                                labels[electron_index]
                                if electron_index < len(labels)
                                else str(electron_index)
                            ),
                            "ion_molecule_index": int(mol_index),
                            "ion_molecule_label": _molecule_formula_label(mol_index),
                            "ion_molecule_formula": _molecule_formula_label(mol_index),
                            "ion_charge": int(charge_number),
                            "ion_molecule_number_density": molecule_density_cgs,
                            "log_ion_molecule_number_density": (
                                float(np.log(molecule_density_cgs))
                                if molecule_density_cgs > 0.0
                                and np.isfinite(molecule_density_cgs)
                                else fastchem_longdouble_minlimit_log_cgs
                            ),
                            "ion_molecule_mass_action_constant": float(
                                molecule_mass_action_constants[mol_index]
                            ),
                            "donor_element_indices": [
                                int(index) for index in active_indices.tolist()
                            ],
                            "donor_element_labels": [
                                labels[int(index)]
                                if int(index) < len(labels)
                                else str(index)
                                for index in active_indices.tolist()
                            ],
                            "donor_element_number_densities": [
                                float(
                                    donor_species_density[int(index)]
                                    * density_domain_scale
                                )
                                for index in active_indices.tolist()
                            ],
                            "donor_log_density_terms": [
                                float(value) for value in donor_logs.tolist()
                            ],
                            "donor_stoichiometric_coefficients": [
                                float(stoich[int(index)])
                                for index in active_indices.tolist()
                            ],
                            "donor_stoich_log_density_terms": [
                                float(stoich[int(index)] * donor_logs[position])
                                for position, index in enumerate(
                                    active_indices.tolist()
                                )
                            ],
                            "donor_log_sum": donor,
                            "exponent_argument": exponent,
                            "ion_contribution": contribution,
                            "alpha_total_before_update": float(alpha_before),
                            "alpha_total_after_update": float(alpha),
                            "beta_total_before_update": float(beta_before),
                            "beta_total_after_update": float(beta),
                            "input_electron_number_density": input_electron_cgs,
                            "electron_refresh_donor_state_source": (
                                electron_refresh_donor_state_source
                            ),
                            "electron_refresh_donor_log_density_source": (
                                electron_refresh_donor_log_density_source
                            ),
                            "electron_refresh_output_value_policy": (
                                electron_refresh_output_value_policy
                            ),
                            "element_solver_output_log_policy": (
                                element_solver_output_log_policy
                            ),
                            "electron_refresh_mode": electron_refresh_mode,
                            "electron_refresh_timing": electron_refresh_timing,
                            "electron_donor_trace_order_scope": (
                                electron_donor_trace_order_scope
                            ),
                            "donor_formula": (
                                "mass_action_constant + "
                                "sum(stoich * log(element.number_density))"
                            ),
                            "ion_molecule_number_density_used_in_formula": False,
                            "diagnostic_only": True,
                            "default_off": True,
                            "KL_owned": True,
                            "FastChem_trace_values_used_as_inputs": False,
                            "used_as_KL_constructor_input": False,
                        }
                    )
            log_alpha = _logsumexp(alpha_log_terms)
            log_beta = _logsumexp(beta_log_terms)
            electron_log_cgs = None
            if log_alpha is not None:
                log_denominator = (
                    0.0
                    if log_beta is None
                    else float(np.logaddexp(0.0, log_beta))
                )
                electron_log_cgs = 0.5 * (log_alpha - log_denominator)
            if electron_log_cgs is not None and np.isfinite(electron_log_cgs):
                electron_current_log_density_cgs = float(electron_log_cgs)
            with np.errstate(invalid="ignore", divide="ignore"):
                electron_cgs = float(np.sqrt(alpha / max(1.0 + beta, 1.0e-300)))
            if (
                electron_refresh_output_value_policy == "log_alpha_beta_exp"
                and electron_log_cgs is not None
                and np.isfinite(electron_log_cgs)
            ):
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    electron_cgs = float(np.exp(electron_log_cgs))
            for row in pending_electron_donor_trace_rows:
                row["output_electron_number_density"] = electron_cgs
                row["log_output_electron_number_density"] = (
                    float(np.log(electron_cgs))
                    if electron_cgs > 0.0 and np.isfinite(electron_cgs)
                    else fastchem_longdouble_minlimit_log_cgs
                )
                row["log_alpha"] = log_alpha
                row["log_beta"] = log_beta
                row["log_denominator_1_plus_beta"] = (
                    None
                    if log_alpha is None
                    else (
                        0.0
                        if log_beta is None
                        else float(np.logaddexp(0.0, log_beta))
                    )
                )
                row["log_electron_density_from_log_alpha_beta"] = (
                    None if electron_log_cgs is None else float(electron_log_cgs)
                )
                electron_donor_trace_records.append(row)
            if np.isfinite(electron_cgs) and electron_cgs > 0.0:
                electron_internal = electron_cgs / density_domain_scale
                if electron_refresh_density_floor_policy == "allow_subfloor_positive":
                    species_density[electron_index] = (
                        electron_internal
                        if electron_internal > 0.0
                        else density_floor_internal
                    )
                else:
                    species_density[electron_index] = max(
                        electron_internal,
                        density_floor_internal,
                    )
                return
        if electron_refresh_mode in {
            "fixed_point_one_step",
            "fixed_point_iterated",
            "fixed_point_damped",
            "post_molecule_refresh_self_consistent",
        }:
            if electron_refresh_mode == "post_molecule_refresh_self_consistent":
                donor_species_density = electron_post_molecule_density
            current_electron = max(
                float(donor_species_density[electron_index]),
                density_floor_internal,
            )
            max_fp_iter = 1
            damping = 1.0
            if electron_refresh_mode == "fixed_point_iterated":
                max_fp_iter = max(1, electron_refresh_fixed_point_max_iter)
            elif electron_refresh_mode == "fixed_point_damped":
                max_fp_iter = max(1, electron_refresh_fixed_point_max_iter)
                damping = min(
                    max(electron_refresh_fixed_point_damping, 0.0),
                    1.0,
                )
            for _ in range(max_fp_iter):
                working_density = donor_species_density.copy()
                working_density[electron_index] = current_electron
                molecule_density_fp = np.zeros_like(charge, dtype=np.float64)
                for mol_index in molecule_list_by_element.get(electron_index, []):
                    stoich = molecule_formula[:, mol_index]
                    active = stoich != 0.0
                    values = working_density[:n_elements][active]
                    values = np.maximum(values, density_floor_internal)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        donor_logs = np.log(values * density_domain_scale)
                    exponent = float(
                        molecule_mass_action_constants[mol_index]
                        + np.dot(stoich[active], donor_logs)
                    )
                    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                        density_cgs = float(np.exp(exponent))
                    if not np.isfinite(density_cgs):
                        density_cgs = float(np.exp(np.clip(exponent, -745.0, 709.0)))
                    molecule_density_fp[mol_index] = density_cgs / density_domain_scale
                positive_fp = float(
                    np.sum(np.where(charge < 0.0, -charge, 0.0) * molecule_density_fp)
                )
                negative_fp = float(
                    np.sum(np.where(charge > 0.0, charge, 0.0) * molecule_density_fp)
                )
                updated_electron = max(
                    positive_fp / max(1.0 + negative_fp, 1.0e-300),
                    density_floor_internal,
                )
                if damping < 1.0:
                    updated_electron = max(
                        (1.0 - damping) * current_electron
                        + damping * updated_electron,
                        density_floor_internal,
                    )
                if abs(updated_electron - current_electron) <= chem_accuracy * max(
                    abs(current_electron),
                    density_floor_internal,
                ):
                    current_electron = updated_electron
                    break
                current_electron = updated_electron
            if np.isfinite(current_electron) and current_electron > 0.0:
                species_density[electron_index] = max(
                    current_electron,
                    density_floor_internal,
                )
                return
        molecule_density = species_density[n_elements:]
        positive = float(np.sum(np.where(charge < 0.0, -charge, 0.0) * molecule_density))
        negative = float(np.sum(np.where(charge > 0.0, charge, 0.0) * molecule_density))
        species_density[electron_index] = max(
            positive / max(1.0 + negative, 1.0e-300),
            density_floor_internal,
        )

    _refresh_minor_boundary()
    post_minor_refresh_density = species_density.copy()
    for iter_step in range(max_replay_iter):
        current_replay_iteration = int(iter_step)
        electron_donor_trace_iteration_order_position = 0
        molecule_refresh_position = 0
        previous = species_density.copy()
        electron_iteration_start_density = species_density.copy()
        iteration_start_element_solver_log_density_cgs = (
            element_solver_log_density_cgs.copy()
        )
        last_solved_element_handoff_log_density_cgs = (
            previous_iteration_element_solver_log_density_cgs.copy()
        )
        last_solved_candidate_handoff_log_density_cgs = (
            previous_iteration_element_solver_log_density_cgs.copy()
        )
        n_major = 0.0
        for element_index in element_order:
            solver_dispatched = _solve_element_density(element_index, n_major)
            post_element_solver_log_density_cgs = (
                element_solver_log_density_cgs.copy()
            )
            post_element_solver_candidate_log_density_cgs = (
                element_solver_candidate_log_density_cgs.copy()
            )
            post_element_solver_species_log_density_cgs = (
                _log_from_internal_density_vector(species_density)
            )
            if 0 <= int(element_index) < n_elements:
                last_solved_element_handoff_log_density_cgs[
                    int(element_index)
                ] = element_solver_log_density_cgs[int(element_index)]
                last_solved_candidate_handoff_log_density_cgs[
                    int(element_index)
                ] = element_solver_candidate_log_density_cgs[int(element_index)]
            electron_post_element_density = species_density.copy()
            if electron_refresh_timing == "after_element_solver_before_molecule_refresh":
                _refresh_electron_density()
            before_major = n_major
            local_molecule_log_major_sigma = _major_density_for_element(
                element_index,
                species_density,
                source="molecule_log_carrier",
            )
            global_molecule_log_major_sigma = _major_density_for_all_elements(
                species_density,
                source="molecule_log_carrier",
            )
            molecule_major_delta = 0.0
            for local_order_position, mol_index in enumerate(
                major_molecules_by_element.get(element_index, [])
            ):
                old_delta_candidate = float(
                    species_density[n_elements + mol_index]
                    * molecule_sigma[mol_index]
                )
                trace_n_major_before = (
                    molecule_major_delta
                    if molecule_refresh_n_major_trace_scope
                    == "local_calculateMoleculeDensities"
                    else before_major
                )
                previous_molecule_density = _refresh_molecule(
                    mol_index,
                    element_context=element_index,
                    refresh_kind="major_refresh",
                    n_major_before_update=trace_n_major_before,
                    n_major_after_update=trace_n_major_before + old_delta_candidate,
                    returned_n_major_delta=old_delta_candidate,
                    local_refresh_order_position=local_order_position,
                )
                molecule_density_for_n_major = (
                    previous_molecule_density
                    if n_major_update_timing == "old_molecule_density"
                    else _molecule_density_for_feedback(mol_index)
                )
                if (
                    n_major_include_policy == "all_element_rows"
                    or solver_dispatched
                ):
                    applied_molecule_major_delta_source = (
                        _molecule_major_delta_source_for(element_index, mol_index)
                    )
                    if applied_molecule_major_delta_source == "caller_old_delta":
                        returned_delta_for_n_major = float(old_delta_candidate)
                    elif (
                        applied_molecule_major_delta_source
                        == "post_refresh_returned_delta"
                    ):
                        returned_delta_for_n_major = float(
                            molecule_refresh_last_returned_n_major_delta
                            if molecule_refresh_last_returned_n_major_delta is not None
                            else old_delta_candidate
                        )
                    elif (
                        applied_molecule_major_delta_source
                        == "previous_molecule_density"
                    ):
                        returned_delta_for_n_major = float(
                            previous_molecule_density * molecule_sigma[mol_index]
                        )
                    elif applied_molecule_major_delta_source == "molecule_log_carrier":
                        returned_delta_for_n_major = float(
                            _molecule_log_carrier_density(mol_index)
                            * molecule_sigma[mol_index]
                        )
                    elif applied_molecule_major_delta_source == "post_refresh_density":
                        returned_delta_for_n_major = float(
                            species_density[n_elements + mol_index]
                            * molecule_sigma[mol_index]
                        )
                    else:
                        returned_delta_for_n_major = float(
                            molecule_density_for_n_major * molecule_sigma[mol_index]
                        )
                    molecule_major_delta += returned_delta_for_n_major
                    if molecule_input_trace_records:
                        last_trace = molecule_input_trace_records[-1]
                        if (
                            int(last_trace.get("iteration", -1))
                            == int(current_replay_iteration)
                            and int(last_trace.get("molecule_index", -1))
                            == int(mol_index)
                            and int(last_trace.get("element_context_index", -1))
                            == int(element_index)
                        ):
                            last_trace[
                                "selected_returned_n_major_delta_for_accumulator"
                            ] = float(returned_delta_for_n_major)
                            last_trace["molecule_major_delta_source"] = (
                                molecule_major_delta_source
                            )
                            last_trace["applied_molecule_major_delta_source"] = (
                                applied_molecule_major_delta_source
                            )
                            last_trace[
                                "molecule_major_delta_source_decomposition"
                            ] = {
                                "old_delta_candidate": float(old_delta_candidate),
                                "post_refresh_returned_delta": float(
                                    molecule_refresh_last_returned_n_major_delta
                                    if molecule_refresh_last_returned_n_major_delta
                                    is not None
                                    else old_delta_candidate
                                ),
                                "previous_molecule_density_delta": float(
                                    previous_molecule_density
                                    * molecule_sigma[mol_index]
                                ),
                                "molecule_log_carrier_delta": float(
                                    _molecule_log_carrier_density(mol_index)
                                    * molecule_sigma[mol_index]
                                ),
                                "post_refresh_density_delta": float(
                                    species_density[n_elements + mol_index]
                                    * molecule_sigma[mol_index]
                                ),
                                "feedback_density_delta": float(
                                    molecule_density_for_n_major
                                    * molecule_sigma[mol_index]
                                ),
                                "selected_delta": float(returned_delta_for_n_major),
                                "source_selection_scope": {
                                    "configured_source": molecule_major_delta_source,
                                    "applied_source": applied_molecule_major_delta_source,
                                    "configured_elements": sorted(
                                        molecule_major_delta_source_elements
                                    ),
                                    "configured_molecules": sorted(
                                        molecule_major_delta_source_molecules
                                    ),
                                    "configured_iteration": (
                                        None
                                        if molecule_major_delta_source_iteration is None
                                        else int(molecule_major_delta_source_iteration)
                                    ),
                                    "element_label": (
                                        element_labels[element_index]
                                        if 0 <= element_index < len(element_labels)
                                        else str(element_index)
                                    ),
                                    "molecule_label": _molecule_formula_label(
                                        mol_index
                                    ),
                                    "iteration": int(current_replay_iteration),
                                },
                                "KL_owned": True,
                                "diagnostic_only": True,
                                "default_off": True,
                            }
            if element_solver_n_major_update_source == "element_budget_residual":
                if n_major_include_policy == "all_element_rows" or solver_dispatched:
                    n_major += _element_budget_major_contribution(
                        element_index,
                        before_major,
                    )
            elif (
                element_solver_n_major_update_source
                == "element_budget_residual_major_stoich"
            ):
                if n_major_include_policy == "all_element_rows" or solver_dispatched:
                    n_major += _element_budget_major_stoich_contribution(
                        element_index,
                        before_major,
                    )
            else:
                n_major += molecule_major_delta
            if not np.isfinite(n_major):
                n_major = before_major
            per_element_n_major_timing_history[element_index] = {
                "cumulative_input": float(before_major),
                "n_major_input": float(before_major),
                "n_major_effective": float(
                    _element_solver_n_major(element_index, before_major)
                ),
                "cumulative_before_update": float(before_major),
                "cumulative_after_update": float(n_major),
                "returned_delta_applied": float(molecule_major_delta),
                "molecule_major_delta_source": molecule_major_delta_source,
                "minor_feedback_contribution": float(_minor_density(element_index)),
                "major_feedback_contribution": float(molecule_major_delta),
                "local_molecule_log_major_sigma": float(
                    local_molecule_log_major_sigma
                ),
                "global_molecule_log_major_sigma": float(
                    global_molecule_log_major_sigma
                ),
            }
            electron_post_molecule_density = species_density.copy()
            if electron_refresh_timing == "after_element_molecule_refresh":
                _refresh_electron_density()
        _refresh_minor_boundary()
        electron_post_minor_boundary_density = species_density.copy()
        post_minor_refresh_density = species_density.copy()
        if electron_refresh_timing == "post_minor_boundary":
            _refresh_electron_density()
        changed = np.abs(species_density - old_density)
        active = (species_density > density_floor_internal) & np.isfinite(species_density)
        rel_limit = chem_accuracy * np.maximum(
            np.abs(old_density),
            density_floor_internal,
        )
        if (
            not disable_replay_convergence_break
            and iter_step > 0
            and bool(np.all(changed[active] <= rel_limit[active]))
        ):
            converged_iteration = int(iter_step)
            break
        if iter_step > nb_switch_to_newton and newton_iteration is None:
            newton_iteration = int(iter_step)
        if iter_step == 390 and not use_backup_solver:
            use_backup_solver = True
            backup_switch_iteration = int(iter_step)
        if replay_old_density_update_timing == "post_iteration_current":
            old_density = species_density.copy()
        else:
            old_density = previous
        if electron_current_log_density_cgs is not None:
            previous_iteration_electron_log_density_cgs = electron_current_log_density_cgs
        previous_iteration_element_solver_log_density_cgs = (
            element_solver_log_density_cgs.copy()
        )

    if converged_iteration is None:
        converged_iteration = int(max_replay_iter - 1)
    return {
        "actual_converged_iteration": converged_iteration,
        "actual_backup_switch_iteration": backup_switch_iteration,
        "actual_newtonSolMult_used": newton_iteration is not None,
        "actual_newtonSolMult_iteration": newton_iteration,
        "actual_electron_old_density_cgs": actual_electron_old_density,
        "post_initial_species_density_vector_cgs": (
            species_density * density_domain_scale
        ).tolist(),
        "replay_loop_status": {
            "available": True,
            "diagnostic_only": True,
            "default_off": True,
            "KL_owned": True,
            "FastChem_trace_values_used_as_inputs": False,
            "used_as_KL_constructor_input": False,
            "algorithm": "mass_action_refresh_with_static_FastChem_branch_timing",
            "max_replay_iter": int(max_replay_iter),
            "disable_replay_convergence_break": disable_replay_convergence_break,
            "species_count": int(n_species),
            "element_count": int(n_elements),
            "electron_row_index": electron_index,
            "backup_branch_exercised": backup_switch_iteration is not None,
            "newtonSolMult_branch_exercised": newton_iteration is not None,
            "element_calculation_order_used": element_order,
            "major_molecule_refresh_count": int(
                sum(len(value) for value in major_molecules_by_element.values())
            ),
            "minor_molecule_refresh_count": int(
                sum(len(value) for value in minor_molecules_by_element.values())
            ),
            "molecule_refresh_order_source": (
                "gas_phase_calculate_lifecycle_context major_molecules_inc and "
                "minor_molecules lists when present; otherwise empty lists"
            ),
            "molecule_refresh_list_source": molecule_refresh_list_source,
            "molecule_density_domain": density_domain,
            "molecule_number_density_gauge_cgs": float(density_domain_scale),
            "molecule_mass_action_source": mass_action_source,
            "molecule_mass_action_correction_source": mass_action_correction_source,
            "element_solver_lifecycle_enabled": element_solver_lifecycle_enabled,
            "element_solver_n_major_source": element_solver_n_major_source,
            "element_solver_n_major_update_source": (
                element_solver_n_major_update_source
            ),
            "element_solver_mode": element_solver_mode,
            "element_solver_exponent_clip": element_solver_exponent_clip,
            "element_solver_exponential_mode": element_solver_exponential_mode,
            "element_solver_quadratic_branch_mode": (
                element_solver_quadratic_branch_mode
            ),
            "element_solver_newton_policy": element_solver_newton_policy,
            "element_solver_newton_fallback_policy": (
                element_solver_newton_fallback_policy
            ),
            "element_solver_newton_max_iter": element_solver_newton_max_iter,
            "element_solver_newton_err": element_solver_newton_err,
            "element_solver_bisection_max_iter": element_solver_bisection_max_iter,
            "element_solver_newton_initial_guess_policy": (
                element_solver_newton_initial_guess_policy
            ),
            "element_solver_newton_assignment_policy": (
                element_solver_newton_assignment_policy
            ),
            "element_solver_newton_derivative_zero_policy": (
                element_solver_newton_derivative_zero_policy
            ),
            "element_solver_use_scaling_factor": use_solver_scaling_factor,
            "element_solver_additional_scaling_factor": (
                additional_solver_scaling_factor
            ),
            "element_solver_gas_density_internal": (
                element_solver_gas_density_internal
            ),
            "element_solver_coefficient_molecule_source": (
                element_solver_coefficient_molecule_source
            ),
            "element_solver_coefficient_abundance_gate": (
                element_solver_coefficient_abundance_gate
            ),
            "element_solver_donor_log_policy": element_solver_donor_log_policy,
            "element_solver_coefficient_donor_log_source": (
                element_solver_coefficient_donor_log_source
            ),
            "element_solver_order_source": element_solver_order_source,
            "element_solver_scaling_factor_molecule_source": (
                element_solver_scaling_factor_molecule_source
            ),
            "molecule_abundance_rule": molecule_abundance_rule,
            "molecule_abundance_electron_policy": molecule_abundance_electron_policy,
            "element_solver_coefficient_density_domain": (
                element_solver_coefficient_density_domain
            ),
            "element_solver_nonpositive_candidate_policy": (
                element_solver_nonpositive_candidate_policy
            ),
            "element_checkN_upper_bound_internal": element_checkN_upper_bound_internal,
            "element_solver_coefficient_density_scale": coefficient_density_scale,
            "element_solver_coefficient_gas_density": coefficient_gas_density,
            "element_phi_normalization": str(
                lifecycle_context.get("element_phi_normalization", "sum_normalize")
            ),
            "element_solver_scaling_factors": element_solver_scaling_factors,
            "molecule_checkN_enabled": molecule_checkN_enabled,
            "electron_refresh_mode": electron_refresh_mode,
            "electron_refresh_timing": electron_refresh_timing,
            "electron_refresh_donor_state_source": electron_refresh_donor_state_source,
            "electron_refresh_fixed_point_max_iter": (
                electron_refresh_fixed_point_max_iter
            ),
            "electron_refresh_fixed_point_damping": (
                electron_refresh_fixed_point_damping
            ),
            "electron_refresh_density_floor_policy": (
                electron_refresh_density_floor_policy
            ),
            "electron_refresh_donor_log_density_source": (
                electron_refresh_donor_log_density_source
            ),
            "electron_refresh_output_value_policy": (
                electron_refresh_output_value_policy
            ),
                "element_solver_output_log_policy": element_solver_output_log_policy,
                "element_solver_subdouble_log_root_policy": (
                    element_solver_subdouble_log_root_policy
                ),
                "element_solver_subdouble_log_root_iteration_limit": (
                    element_solver_subdouble_log_root_iteration_limit
                ),
            "element_solver_fixed_by_condensation_policy": (
                element_solver_fixed_by_condensation_policy
            ),
            "element_solver_minor_density_source": (
                element_solver_minor_density_source
            ),
            "molecule_feedback_density_source": molecule_feedback_density_source,
            "molecule_refresh_electron_density_source": (
                molecule_refresh_electron_density_source
            ),
            "molecule_refresh_electron_log_density_source": (
                molecule_refresh_electron_log_density_source
            ),
            "molecule_refresh_element_log_density_source": (
                molecule_refresh_element_log_density_source
            ),
            "molecule_refresh_h_coefficient_log_route": (
                molecule_refresh_h_coefficient_log_route
            ),
            "fastchem_longdouble_minlimit_log_cgs": (
                fastchem_longdouble_minlimit_log_cgs
            ),
            "molecule_refresh_electron_floor_policy": (
                molecule_refresh_electron_floor_policy
            ),
            "molecule_refresh_positive_log_floor_policy": (
                molecule_refresh_positive_log_floor_policy
            ),
            "replay_old_density_update_timing": replay_old_density_update_timing,
            "molecule_refresh_element_density_source": (
                molecule_refresh_element_density_source
            ),
            "molecule_refresh_external_element_density_available": bool(
                external_element_density is not None
            ),
            "n_major_update_timing": n_major_update_timing,
            "n_major_include_policy": n_major_include_policy,
            "emit_molecule_input_trace": emit_molecule_input_trace,
            "molecule_input_trace_iteration_limit": (
                molecule_input_trace_iteration_limit
            ),
            "molecule_input_trace_record_count": len(molecule_input_trace_records),
            "emit_electron_donor_trace": emit_electron_donor_trace,
            "electron_donor_trace_order_scope": electron_donor_trace_order_scope,
            "electron_donor_trace_iteration_limit": (
                electron_donor_trace_iteration_limit
            ),
            "electron_donor_trace_record_count": len(electron_donor_trace_records),
            "emit_element_solver_trace": emit_element_solver_trace,
            "element_solver_trace_iteration_limit": (
                element_solver_trace_iteration_limit
            ),
            "element_solver_trace_record_count": len(element_solver_trace_records),
            "emit_coefficient_source_value_trace": (
                emit_coefficient_source_value_trace
            ),
            "coefficient_source_value_trace_iteration_limit": (
                coefficient_source_value_trace_iteration_limit
            ),
            "coefficient_source_value_trace_record_count": len(
                coefficient_source_value_trace_records
            ),
            "n_major_solver_dispatched_rows_policy": (
                n_major_solver_dispatched_rows_policy
            ),
            "element_solver_lifecycle_source": (
                "diagnostic Python replay of FastChem intertSol/linSol/quadSol/"
                "newtonSol polynomial coefficients using KL-owned lifecycle fields"
            ),
            "post_initial_species_density_source": (
                "KL-owned replay from current reduced-solver gas species density, "
                "recovered molecule mass-action constants, FastChem-style "
                "element order, major/minor molecule refresh lists, and explicit "
                "mass-action density-domain gauge"
            ),
        },
        "molecule_input_trace_records": molecule_input_trace_records,
        "electron_donor_trace_records": electron_donor_trace_records,
        "element_solver_trace_records": element_solver_trace_records,
        "coefficient_source_value_trace_records": (
            coefficient_source_value_trace_records
        ),
        "minor_density_trace_records": minor_density_trace_records,
    }


def _build_element_slot_gas_density_ntot_normalization_carrier(
    *,
    element_labels_reduced_order: Optional[Sequence[str]],
    b: jnp.ndarray,
    temperature: jnp.ndarray,
    ln_normalized_pressure: jnp.ndarray,
    ntotk: jnp.ndarray,
) -> Dict[str, Any]:
    """Build a default-off element-slot density normalization diagnostic carrier."""

    temperature = jnp.asarray(temperature, dtype=jnp.float64)
    ln_normalized_pressure = jnp.asarray(ln_normalized_pressure, dtype=jnp.float64)
    pressure_bar = jnp.exp(ln_normalized_pressure)
    gas_number_density_cgs = (
        pressure_bar
        * jnp.asarray(KL_DENSITY_GAUGE_P0_CGS, dtype=jnp.float64)
        / (
            jnp.asarray(KL_DENSITY_GAUGE_K_B_CGS, dtype=jnp.float64)
            * jnp.maximum(temperature, jnp.asarray(1.0e-300, dtype=jnp.float64))
        )
    )
    element_density_cgs = jnp.abs(jnp.asarray(b, dtype=jnp.float64)) * gas_number_density_cgs
    inverse_element_density_cgs = 1.0 / jnp.maximum(
        element_density_cgs,
        jnp.asarray(1.0e-300, dtype=jnp.float64),
    )
    ln_gas_number_density_cgs = jnp.log(
        jnp.maximum(gas_number_density_cgs, jnp.asarray(1.0e-300, dtype=jnp.float64))
    )
    ntotk = jnp.asarray(ntotk, dtype=jnp.float64)
    b_values = jnp.asarray(b, dtype=jnp.float64)
    h_half = None
    he_atomic = None
    metals_atomic_sum = None
    particle_denominator = None
    total_element_density_cgs_derived = None
    if element_labels_reduced_order is not None:
        label_to_position = {
            str(label): index for index, label in enumerate(element_labels_reduced_order)
        }
        h_value = (
            b_values[label_to_position["H"]]
            if "H" in label_to_position
            else jnp.asarray(0.0, dtype=jnp.float64)
        )
        he_value = (
            b_values[label_to_position["He"]]
            if "He" in label_to_position
            else jnp.asarray(0.0, dtype=jnp.float64)
        )
        metal_values = [
            b_values[index]
            for label, index in label_to_position.items()
            if label not in {"H", "He", "e-"}
        ]
        metals_value = (
            jnp.sum(jnp.asarray(metal_values, dtype=jnp.float64))
            if metal_values
            else jnp.asarray(0.0, dtype=jnp.float64)
        )
        h_half_value = 0.5 * h_value
        particle_denominator_value = h_half_value + he_value + metals_value
        total_element_density_value = gas_number_density_cgs / jnp.maximum(
            particle_denominator_value,
            jnp.asarray(1.0e-300, dtype=jnp.float64),
        )
        h_half = float(jax.device_get(h_half_value))
        he_atomic = float(jax.device_get(he_value))
        metals_atomic_sum = float(jax.device_get(metals_value))
        particle_denominator = float(jax.device_get(particle_denominator_value))
        total_element_density_cgs_derived = float(
            jax.device_get(total_element_density_value)
        )
    return {
        "available": element_labels_reduced_order is not None,
        "diagnostic_only": True,
        "default_off": True,
        "reference_target": "FastChem-style selected-element row scaling density gauge",
        "temperature_K": float(jax.device_get(temperature)),
        "pressure_bar": float(jax.device_get(pressure_bar)),
        "gas_number_density_cgs": float(jax.device_get(gas_number_density_cgs)),
        "ln_gas_number_density_cgs": float(jax.device_get(ln_gas_number_density_cgs)),
        "density_gauge_p0_cgs": float(KL_DENSITY_GAUGE_P0_CGS),
        "density_gauge_k_B_cgs": float(KL_DENSITY_GAUGE_K_B_CGS),
        "ntotk_dimensionless": float(jax.device_get(ntotk)),
        "ln_density_to_ntotk_ratio": float(
            jax.device_get(
                ln_gas_number_density_cgs
                - jnp.log(jnp.maximum(ntotk, jnp.asarray(1.0e-300, dtype=jnp.float64)))
            )
        ),
        "element_labels_reduced_order": (
            None
            if element_labels_reduced_order is None
            else [str(label) for label in element_labels_reduced_order]
        ),
        "abs_b_times_gas_density_cgs": _diagnostic_json_array(element_density_cgs),
        "inverse_abs_b_times_gas_density_cgs": _diagnostic_json_array(
            inverse_element_density_cgs
        ),
        "formula_candidate_name": "abs_b_times_gas_number_density_cgs",
        "total_element_density_cgs_derived": total_element_density_cgs_derived,
        "total_element_density_derivation": (
            "gas_number_density_cgs / (H/2 + He + metals)"
        ),
        "total_element_density_formula_metadata": {
            "diagnostic_only": True,
            "default_off": True,
            "constructor_input": False,
            "reference_trace_input": False,
            "formula": "gas_number_density_cgs / (H/2 + He + metals)",
            "denominator_convention": "H2/He gas-particle approximation",
            "H_half": h_half,
            "He_atomic": he_atomic,
            "metals_atomic_sum": metals_atomic_sum,
            "electron_excluded_from_denominator": True,
            "particle_denominator": particle_denominator,
            "source_fields": [
                "gas_number_density_cgs",
                "element_labels_reduced_order",
                "b",
            ],
        },
    }


def build_fastchem_exact_total_element_density_convention_carrier(
    *,
    gas_species_number_density_cgs: Sequence[float],
    formula_matrix_gas: Sequence[Sequence[float]],
    gas_number_density_cgs: Optional[float] = None,
    condensate_number_density_cgs: Optional[Sequence[float]] = None,
    formula_matrix_cond: Optional[Sequence[Sequence[float]]] = None,
    source_artifact: str = "KL default-off FastChem exact total_element_density diagnostic",
) -> Dict[str, Any]:
    """Build a default-off carrier for FastChem's exact total-element convention.

    FastChem owns this scalar as the sum of all free atomic element densities,
    molecular stoichiometric burdens, and condensed-phase stoichiometric
    burdens.  This helper computes that formula from KL-owned physical-density
    vectors when diagnostic code supplies them; it does not consume FastChem
    trace values and is not called by the production solver.
    """

    gas_species = np.asarray(gas_species_number_density_cgs, dtype=np.float64)
    gas_formula = np.asarray(formula_matrix_gas, dtype=np.float64)
    if gas_formula.ndim != 2:
        raise ValueError("formula_matrix_gas must be a two-dimensional matrix.")
    if gas_species.ndim != 1 or gas_species.shape[0] != gas_formula.shape[1]:
        raise ValueError(
            "gas_species_number_density_cgs must have one value per gas formula column "
            f"(got {gas_species.shape}, expected ({gas_formula.shape[1]},))."
        )
    gas_element_density = gas_formula @ gas_species
    cond_element_density = np.zeros(gas_formula.shape[0], dtype=np.float64)
    if condensate_number_density_cgs is not None or formula_matrix_cond is not None:
        if condensate_number_density_cgs is None or formula_matrix_cond is None:
            raise ValueError(
                "condensate_number_density_cgs and formula_matrix_cond must be supplied together."
            )
        cond_species = np.asarray(condensate_number_density_cgs, dtype=np.float64)
        cond_formula = np.asarray(formula_matrix_cond, dtype=np.float64)
        if cond_formula.ndim != 2:
            raise ValueError("formula_matrix_cond must be a two-dimensional matrix.")
        if cond_formula.shape[0] != gas_formula.shape[0]:
            raise ValueError(
                "formula_matrix_cond must use the same element rows as formula_matrix_gas."
            )
        if cond_species.ndim != 1 or cond_species.shape[0] != cond_formula.shape[1]:
            raise ValueError(
                "condensate_number_density_cgs must have one value per condensate formula column "
                f"(got {cond_species.shape}, expected ({cond_formula.shape[1]},))."
            )
        cond_element_density = cond_formula @ cond_species
    total_element_density = float(np.sum(gas_element_density + cond_element_density))
    particle_denominator = None
    if gas_number_density_cgs is not None:
        particle_denominator = float(gas_number_density_cgs) / max(
            total_element_density, 1.0e-300
        )
    return {
        "carrier_schema": "default_off_fastchem_exact_total_element_density_convention_v1",
        "diagnostic_only": True,
        "default_off": True,
        "active_only_when_explicitly_requested": True,
        "source_artifact": source_artifact,
        "source_formula": (
            "sum(formula_matrix_gas @ gas_species_number_density_cgs) + "
            "sum(formula_matrix_cond @ condensate_number_density_cgs)"
        ),
        "fastchem_source_functions": [
            "fastchem/fastchem_src/gas_phase/calc_species_densities.cpp::GasPhase::totalElementDensity",
            "fastchem/fastchem_src/condensed_phase/condensed_phase.cpp::CondensedPhase::totalElementDensity",
        ],
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "gas_total_element_density_cgs": float(np.sum(gas_element_density)),
        "condensate_total_element_density_cgs": float(np.sum(cond_element_density)),
        "total_element_density_cgs": total_element_density,
        "gas_number_density_cgs": (
            None if gas_number_density_cgs is None else float(gas_number_density_cgs)
        ),
        "fastchem_exact_particle_denominator": particle_denominator,
        "gas_element_density_cgs": gas_element_density.tolist(),
        "condensate_element_density_cgs": cond_element_density.tolist(),
    }


def build_condensate_density_budget_cap_total_element_density_carrier(
    *,
    condensate_number_density_cgs: Sequence[float],
    condensate_budget_cap: Sequence[float],
    condensate_slot_labels: Optional[Sequence[str]] = None,
    source_artifact: str = "KL default-off condensate-density/budget-cap total_element_density diagnostic",
) -> Dict[str, Any]:
    """Build a default-off total-element scalar from KL condensate slot density.

    The element-condensate Jacobian owner multiplies condensate stoichiometry by
    the local condensate number density.  The fixed bridge writes the same block
    as stoichiometry times ``budget_cap * total_element_density``.  Therefore a
    KL-owned diagnostic scalar is available from each active condensate slot as
    ``condensate_number_density_cgs / condensate_budget_cap``.
    """

    density = np.asarray(condensate_number_density_cgs, dtype=np.float64)
    budget_cap = np.asarray(condensate_budget_cap, dtype=np.float64)
    if density.ndim != 1 or budget_cap.ndim != 1:
        raise ValueError("condensate density and budget cap must be one-dimensional.")
    if density.shape[0] != budget_cap.shape[0]:
        raise ValueError(
            "condensate density and budget cap must have the same length "
            f"(got {density.shape[0]} and {budget_cap.shape[0]})."
        )
    if condensate_slot_labels is not None and len(condensate_slot_labels) != density.shape[0]:
        raise ValueError(
            "condensate_slot_labels must have one label per condensate density."
        )
    valid = (
        np.isfinite(density)
        & np.isfinite(budget_cap)
        & (density > 0.0)
        & (budget_cap > 0.0)
    )
    slot_scalars = np.full(density.shape, np.nan, dtype=np.float64)
    slot_scalars[valid] = density[valid] / budget_cap[valid]
    valid_scalars = slot_scalars[valid]
    total_element_density = (
        None if valid_scalars.size == 0 else float(np.median(valid_scalars))
    )
    median_abs_deviation = (
        None
        if valid_scalars.size == 0
        else float(np.median(np.abs(valid_scalars - float(total_element_density))))
    )
    labels = (
        None
        if condensate_slot_labels is None
        else [str(label) for label in condensate_slot_labels]
    )
    return {
        "carrier_schema": "default_off_condensate_density_budget_cap_total_element_density_v1",
        "diagnostic_only": True,
        "default_off": True,
        "active_only_when_explicitly_requested": True,
        "source_artifact": source_artifact,
        "source_formula": "median(condensate_number_density_cgs / condensate_budget_cap)",
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "total_element_density_cgs": total_element_density,
        "aggregation": "median",
        "valid_slot_count": int(np.count_nonzero(valid)),
        "slot_count": int(density.shape[0]),
        "slot_scalar_min": (
            None if valid_scalars.size == 0 else float(np.min(valid_scalars))
        ),
        "slot_scalar_max": (
            None if valid_scalars.size == 0 else float(np.max(valid_scalars))
        ),
        "slot_scalar_median_abs_deviation": median_abs_deviation,
        "condensate_number_density_cgs": density.tolist(),
        "condensate_budget_cap": budget_cap.tolist(),
        "slot_total_element_density_cgs": slot_scalars.tolist(),
        "condensate_slot_labels": labels,
    }


def _build_reduced_solver_exact_input_bundle(
    *,
    case_key: str,
    newton_iter: int,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    temperature: jnp.ndarray,
    ln_normalized_pressure: jnp.ndarray,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
    condensates_jac_indices: Optional[Sequence[int]],
    condensate_labels_jac_order: Optional[Sequence[str]],
    element_labels_reduced_order: Optional[Sequence[str]],
    call_site_provenance: str,
    active: bool,
    row_scaled_element_condensate_jec_target_block: Optional[Sequence[Sequence[float]]] = None,
    selected_element_row_scaling_vector: Optional[Sequence[float]] = None,
    gas_phase_calculate_lifecycle_context: Optional[Dict[str, Any]] = None,
    ln_nk_producer_trace: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the default-off exact reduced-solver input bundle at the call site."""

    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntotk = jnp.exp(ln_ntot)
    bk = formula_matrix @ nk
    sk = jnp.exp(2.0 * ln_mk - epsilon)
    element_slot_gas_density_ntot_normalization_carrier = (
        _build_element_slot_gas_density_ntot_normalization_carrier(
            element_labels_reduced_order=element_labels_reduced_order,
            b=b,
            temperature=temperature,
            ln_normalized_pressure=ln_normalized_pressure,
            ntotk=ntotk,
        )
    )
    positive_cond_budget_terms = jnp.where(
        formula_matrix_cond > 0.0,
        b[:, None] / formula_matrix_cond,
        jnp.inf,
    )
    condensate_budget_cap_vector = jnp.min(
        positive_cond_budget_terms,
        axis=0,
    )
    condensate_budget_cap_vector = jnp.where(
        jnp.isfinite(condensate_budget_cap_vector),
        condensate_budget_cap_vector,
        jnp.nan,
    )
    element_epsilon_from_normalized_b_vector = b / jnp.maximum(
        jnp.sum(b),
        jnp.asarray(1.0e-300, dtype=jnp.float64),
    )
    normalized_b_budget_terms = jnp.where(
        formula_matrix_cond > 0.0,
        element_epsilon_from_normalized_b_vector[:, None] / formula_matrix_cond,
        jnp.inf,
    )
    normalized_b_maxdensity_budget_cap_vector = jnp.min(
        normalized_b_budget_terms,
        axis=0,
    )
    normalized_b_maxdensity_budget_cap_vector = jnp.where(
        jnp.isfinite(normalized_b_maxdensity_budget_cap_vector),
        normalized_b_maxdensity_budget_cap_vector,
        jnp.nan,
    )
    total_element_density_cgs_derived = (
        element_slot_gas_density_ntot_normalization_carrier.get(
            "total_element_density_cgs_derived"
        )
    )
    if total_element_density_cgs_derived is None:
        condensate_jec_owner_density_cgs_vector = jnp.full_like(
            condensate_budget_cap_vector,
            jnp.nan,
        )
    else:
        condensate_jec_owner_density_cgs_vector = (
            condensate_budget_cap_vector * float(total_element_density_cgs_derived)
        )
    gas_number_density_cgs = float(
        element_slot_gas_density_ntot_normalization_carrier.get(
            "gas_number_density_cgs",
            np.nan,
        )
    )
    species_particle_fraction = nk / jnp.maximum(
        jnp.sum(nk),
        jnp.asarray(1.0e-300, dtype=jnp.float64),
    )
    gas_species_number_density_cgs_candidate_vector = (
        species_particle_fraction * gas_number_density_cgs
    )
    recovered_hvector_gas = (
        gk - ln_nk + ln_ntot - ln_normalized_pressure
    )
    n_element_rows = int(formula_matrix.shape[0])
    atomic_prefix_available = int(ln_nk.shape[0]) >= n_element_rows
    atomic_input_state_trace_rows: List[Dict[str, Any]] = []
    if atomic_prefix_available:
        u_trace_float64 = np.asarray(
            jax.device_get(ln_nk[:n_element_rows]),
            dtype=np.float64,
        )
        u_trace_longdouble = np.asarray(u_trace_float64, dtype=np.longdouble)
        element_trace_labels = (
            [str(label) for label in element_labels_reduced_order]
            if element_labels_reduced_order is not None
            else [str(index) for index in range(n_element_rows)]
        )
        for element_index in range(n_element_rows):
            u_value_longdouble = u_trace_longdouble[element_index]
            u_value_float64 = u_trace_float64[element_index]
            finite = bool(np.isfinite(u_value_longdouble))
            atomic_input_state_trace_rows.append(
                {
                    "case_key": case_key,
                    "newton_iter": int(newton_iter),
                    "element_index": int(element_index),
                    "element_label": element_trace_labels[element_index],
                    "element_order_position": int(element_index),
                    "u_log_density_before_jax_cast": (
                        float(u_value_float64) if np.isfinite(u_value_float64) else None
                    ),
                    "u_log_density_longdouble_replay": (
                        float(u_value_longdouble) if finite else None
                    ),
                    "u_source_stage": "_build_reduced_solver_exact_input_bundle ln_nk[:n_element_rows]",
                    "u_producer_function": "src/exogibbs/optimize/pipm_rgie_cond.py::_build_reduced_solver_exact_input_bundle",
                    "u_consumer_function": "src/exogibbs/optimize/pipm_rgie_cond.py::reconstruct_kl_atomic_gas_from_u",
                    "u_producer_dtype": str(ln_nk.dtype),
                    "u_consumer_cast": "jnp.asarray(u, dtype=jnp.float64)",
                    "preserves_native_longdouble_bits": False,
                    "native_longdouble_provenance_available": False,
                    "reconstructed_from_float64": True,
                    "source_density_cgs_before_exp_or_normalization": None,
                    "density_domain_scale": None,
                    "floor_policy": "no pre-cast floor policy available at this trace boundary",
                    "subdouble_status": (
                        "below_double_normal_log"
                        if finite and u_value_longdouble < np.longdouble(np.log(float.fromhex("0x1p-1022")))
                        else "double_representable_or_nonfinite_log"
                    ),
                    "selected_v54_row_identity_available": False,
                }
            )
    if atomic_prefix_available:
        atomic_prefix_replay = reconstruct_kl_atomic_gas_from_u(
            ln_nk[:n_element_rows],
            formula_matrix,
            recovered_hvector_gas,
            temperature=float(jax.device_get(temperature)),
            apply_density_gauge_bridge=False,
        )
        atomic_prefix_particle_fraction = atomic_prefix_replay["nk"] / jnp.maximum(
            jnp.sum(atomic_prefix_replay["nk"]),
            jnp.asarray(1.0e-300, dtype=jnp.float64),
        )
        atomic_prefix_gas_species_density_cgs_candidate_vector = (
            atomic_prefix_particle_fraction * gas_number_density_cgs
        )
    else:
        atomic_prefix_replay = None
        atomic_prefix_gas_species_density_cgs_candidate_vector = jnp.full_like(
            gas_species_number_density_cgs_candidate_vector,
            jnp.nan,
        )
    gas_species_total_element_density_cgs_candidate = jnp.sum(
        formula_matrix @ gas_species_number_density_cgs_candidate_vector
    )
    atomic_prefix_total_element_density_cgs_candidate = jnp.sum(
        formula_matrix @ atomic_prefix_gas_species_density_cgs_candidate_vector
    )
    gas_species_particle_count_denominator = jnp.sum(nk)
    gas_species_ntot_denominator = ntotk
    gas_species_reduced_element_inventory_sum = jnp.sum(b)
    gas_species_count_density_cgs_candidate = (
        gas_number_density_cgs
        / jnp.maximum(
            gas_species_particle_count_denominator,
            jnp.asarray(1.0e-300, dtype=jnp.float64),
        )
    )
    gas_species_ntot_density_cgs_candidate = (
        gas_number_density_cgs
        / jnp.maximum(
            gas_species_ntot_denominator,
            jnp.asarray(1.0e-300, dtype=jnp.float64),
        )
    )
    gas_species_inventory_renormalized_total_element_density_cgs_candidate = (
        gas_species_total_element_density_cgs_candidate
        / jnp.maximum(
            gas_species_reduced_element_inventory_sum,
            jnp.asarray(1.0e-300, dtype=jnp.float64),
        )
    )
    gas_species_element_density_cgs_candidate_vector = (
        formula_matrix @ gas_species_number_density_cgs_candidate_vector
    )
    gas_species_positive_stoich_total_element_density_cgs_candidate = jnp.sum(
        jnp.where(formula_matrix > 0.0, formula_matrix, 0.0)
        @ gas_species_number_density_cgs_candidate_vector
    )
    electron_row_index = None
    if element_labels_reduced_order is not None:
        for index, label in enumerate(element_labels_reduced_order):
            if str(label) == "e-":
                electron_row_index = index
                break
    electron_row_contribution_cgs = (
        None
        if electron_row_index is None
        else float(
            jax.device_get(
                gas_species_element_density_cgs_candidate_vector[electron_row_index]
            )
        )
    )
    gas_species_non_electron_total_element_density_cgs_candidate = (
        gas_species_total_element_density_cgs_candidate
        if electron_row_index is None
        else (
            gas_species_total_element_density_cgs_candidate
            - gas_species_element_density_cgs_candidate_vector[electron_row_index]
        )
    )
    fastchem_post_initial_gas_species_density_replay_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": True,
        "exact_owner_verified": False,
        "gas_number_density_cgs": gas_number_density_cgs,
        "current_species_density_source": (
            "nk / sum(nk) * gas_number_density_cgs at KL reduced-solver entry"
        ),
        "current_species_density_cgs_candidate_vector": _diagnostic_json_array(
            gas_species_number_density_cgs_candidate_vector
        ),
        "current_total_element_density_cgs_candidate": float(
            jax.device_get(gas_species_total_element_density_cgs_candidate)
        ),
        "recovered_hvector_source": (
            "gk - ln_nk + ln_ntot - ln_normalized_pressure"
        ),
        "recovered_hvector_gas": _diagnostic_json_array(recovered_hvector_gas),
        "atomic_prefix_replay_available": atomic_prefix_available,
        "atomic_prefix_replay_source": (
            "reconstruct_kl_atomic_gas_from_u(ln_nk[:n_elements], formula_matrix, recovered_hvector_gas)"
        ),
        "ln_nk_producer_trace": (
            {
                "diagnostic_only": True,
                "default_off": True,
                "constructor_input": False,
                "reference_trace_input": False,
                "FastChem_trace_values_used_as_inputs": False,
                "used_as_KL_constructor_input": False,
                "available": False,
                "missing_field": "ln_nk producer trace was not supplied by the caller",
            }
            if ln_nk_producer_trace is None
            else dict(ln_nk_producer_trace)
        ),
        "ln_nk_source_state_trace": (
            {
                "diagnostic_only": True,
                "default_off": True,
                "constructor_input": False,
                "reference_trace_input": False,
                "FastChem_trace_values_used_as_inputs": False,
                "used_as_KL_constructor_input": False,
                "available": False,
                "missing_field": "ln_nk source-state trace was not supplied by the caller",
            }
            if gas_phase_calculate_lifecycle_context is None
            else dict(
                gas_phase_calculate_lifecycle_context.get(
                    "ln_nk_source_state_trace",
                    {
                        "diagnostic_only": True,
                        "default_off": True,
                        "constructor_input": False,
                        "reference_trace_input": False,
                        "FastChem_trace_values_used_as_inputs": False,
                        "used_as_KL_constructor_input": False,
                        "available": False,
                        "missing_field": "ln_nk source-state trace not present in lifecycle context",
                    },
                )
            )
        ),
        "ln_nk_init_source_trace": (
            {
                "diagnostic_only": True,
                "default_off": True,
                "constructor_input": False,
                "reference_trace_input": False,
                "FastChem_trace_values_used_as_inputs": False,
                "used_as_KL_constructor_input": False,
                "available": False,
                "missing_field": "ln_nk init source trace was not supplied by the caller",
            }
            if gas_phase_calculate_lifecycle_context is None
            else dict(
                gas_phase_calculate_lifecycle_context.get(
                    "ln_nk_init_source_trace",
                    {
                        "diagnostic_only": True,
                        "default_off": True,
                        "constructor_input": False,
                        "reference_trace_input": False,
                        "FastChem_trace_values_used_as_inputs": False,
                        "used_as_KL_constructor_input": False,
                        "available": False,
                        "missing_field": "ln_nk init source trace not present in lifecycle context",
                    },
                )
            )
        ),
        "atomic_input_state_trace": {
            "diagnostic_only": True,
            "default_off": True,
            "constructor_input": False,
            "reference_trace_input": False,
            "FastChem_trace_values_used_as_inputs": False,
            "used_as_KL_constructor_input": False,
            "target": "src/exogibbs/optimize/pipm_rgie_cond.py::reconstruct_kl_atomic_gas_from_u input state construction",
            "producer_boundary": "src/exogibbs/optimize/pipm_rgie_cond.py::_build_reduced_solver_exact_input_bundle ln_nk[:n_element_rows]",
            "consumer_boundary": "src/exogibbs/optimize/pipm_rgie_cond.py::reconstruct_kl_atomic_gas_from_u u",
            "observed_consumer_cast": "u = jnp.asarray(u, dtype=jnp.float64)",
            "native_longdouble_provenance_available": False,
            "preserves_native_longdouble_bits": False,
            "trace_row_count": len(atomic_input_state_trace_rows),
            "trace_rows": atomic_input_state_trace_rows,
        },
        "atomic_prefix_species_density_cgs_candidate_vector": _diagnostic_json_array(
            atomic_prefix_gas_species_density_cgs_candidate_vector
        ),
        "atomic_prefix_total_element_density_cgs_candidate": float(
            jax.device_get(atomic_prefix_total_element_density_cgs_candidate)
        ),
        "atomic_prefix_replay_metadata": (
            None
            if atomic_prefix_replay is None
            else {
                "density_gauge_bridge_applied": bool(
                    jax.device_get(
                        atomic_prefix_replay["density_gauge_bridge_applied"]
                    )
                ),
                "ln_ntot": float(jax.device_get(atomic_prefix_replay["ln_ntot"])),
                "species_count": int(atomic_prefix_replay["nk"].shape[0]),
                "element_count": n_element_rows,
            }
        ),
        "target_fastchem_source": (
            "fastchem/fastchem_src/calc_densities.cpp::gas_phase.calculate "
            "followed by gas_phase.species[i]->number_density copyout"
        ),
        "missing_exact_inputs": [
            "FastChem iterative gas_phase.calculate lifecycle replay",
            "FastChem element solver order and backup/Newton branch state at post-initial timing",
            "post-initial FastChem gas species number_density vector after gas_phase.calculate",
        ],
        "owner_status": (
            "default-off KL gas species density replay boundary; current and "
            "atomic-prefix candidates are diagnostics, not production inputs"
        ),
    }
    lifecycle_context = (
        {} if gas_phase_calculate_lifecycle_context is None
        else dict(gas_phase_calculate_lifecycle_context)
    )
    element_labels_for_lifecycle = (
        None
        if element_labels_reduced_order is None
        else [str(label) for label in element_labels_reduced_order]
    )
    lifecycle_auto_context: Dict[str, Any] = {}
    lifecycle_field_sources: Dict[str, str] = {}
    lifecycle_field_exactness: Dict[str, str] = {}
    if element_labels_for_lifecycle is not None:
        epsilon_host = np.asarray(
            jax.device_get(element_epsilon_from_normalized_b_vector),
            dtype=np.float64,
        )
        element_abundance_context = lifecycle_context.get("element_abundance_vector")
        if element_abundance_context is None:
            element_abundance_host = epsilon_host
            element_abundance_source = "KL normalized element epsilon"
            element_abundance_exactness = "KL_owned_candidate"
        else:
            element_abundance_host = np.asarray(
                element_abundance_context,
                dtype=np.float64,
            )
            if element_abundance_host.shape[0] != n_element_rows:
                element_abundance_host = np.resize(
                    element_abundance_host,
                    n_element_rows,
                ).astype(np.float64)
            element_abundance_source = (
                "explicit gas_phase_calculate_lifecycle_context.element_abundance_vector"
            )
            element_abundance_exactness = "explicit_KL_owned_candidate"
        formula_host = np.asarray(jax.device_get(formula_matrix), dtype=np.float64)
        non_electron_order = [
            index
            for index, label in enumerate(element_labels_for_lifecycle)
            if label != "e-"
        ]
        sorted_order = sorted(
            non_electron_order,
            key=lambda index: float(element_abundance_host[index]),
            reverse=True,
        )
        lifecycle_auto_context["element_calculation_order"] = {
            "indices": sorted_order,
            "labels": [element_labels_for_lifecycle[index] for index in sorted_order],
        }
        lifecycle_field_sources["element_calculation_order"] = (
            "KL normalized element epsilon sorted descending, excluding e-"
        )
        lifecycle_field_exactness["element_calculation_order"] = "KL_owned_candidate"
        molecule_formula = formula_host[:, n_element_rows:]
        if molecule_formula.size:
            molecule_positive = np.where(molecule_formula > 0.0, molecule_formula, 0.0)
            molecule_participates = molecule_positive > 0.0
            molecule_abundance = np.full(
                molecule_formula.shape[1],
                np.nan,
                dtype=np.float64,
            )
            molecule_abundance_electron_policy = str(
                lifecycle_context.get(
                    "molecule_abundance_electron_policy",
                    "exclude_zero_abundance_electron",
                )
            )
            for mol_index in range(molecule_formula.shape[1]):
                active_elements = np.nonzero(molecule_participates[:, mol_index])[0]
                if (
                    electron_index is not None
                    and molecule_abundance_electron_policy
                    == "exclude_zero_abundance_electron"
                ):
                    active_elements = active_elements[active_elements != electron_index]
                if active_elements.size:
                    molecule_abundance[mol_index] = float(
                        np.min(element_abundance_host[active_elements])
                    )
            solver_order = np.max(np.abs(molecule_formula), axis=1).astype(int)
            lifecycle_auto_context["element_solver_order_vector"] = solver_order.tolist()
            lifecycle_auto_context["major_molecules_inc"] = {
                element_labels_for_lifecycle[element_index]: [
                    int(n_element_rows + mol_index)
                    for mol_index in range(molecule_formula.shape[1])
                    if (
                        molecule_participates[element_index, mol_index]
                        and np.isfinite(molecule_abundance[mol_index])
                        and molecule_abundance[mol_index]
                        >= element_abundance_host[element_index]
                    )
                ]
                for element_index in range(n_element_rows)
            }
            lifecycle_auto_context["minor_molecules"] = {
                element_labels_for_lifecycle[element_index]: [
                    int(n_element_rows + mol_index)
                    for mol_index in range(molecule_formula.shape[1])
                    if (
                        np.isfinite(molecule_abundance[mol_index])
                        and element_abundance_host[element_index]
                        > molecule_abundance[mol_index]
                    )
                ]
                for element_index in range(n_element_rows)
            }
            lifecycle_auto_context["molecule_mass_action_constants"] = (
                _diagnostic_json_array(-recovered_hvector_gas[n_element_rows:])
            )
            lifecycle_field_sources["element_solver_order_vector"] = (
                "max(abs(stoichiometry)) across molecule columns"
            )
            lifecycle_field_sources["major_molecules_inc"] = (
                "ported FastChem createMoleculeLists rule using "
                f"{element_abundance_source}"
            )
            lifecycle_field_sources["minor_molecules"] = (
                "ported FastChem createMoleculeLists rule using "
                f"{element_abundance_source}"
            )
            lifecycle_field_sources["molecule_mass_action_constants"] = (
                "-recovered_hvector_gas for molecule columns"
            )
            lifecycle_field_exactness["element_solver_order_vector"] = "KL_owned_candidate"
            lifecycle_field_exactness["major_molecules_inc"] = (
                element_abundance_exactness
            )
            lifecycle_field_exactness["minor_molecules"] = element_abundance_exactness
            lifecycle_field_exactness["molecule_mass_action_constants"] = (
                "KL_owned_hvector_candidate"
            )
        lifecycle_auto_context["element_phi_vector"] = _diagnostic_json_array(b)
        lifecycle_auto_context["element_epsilon_vector"] = _diagnostic_json_array(
            element_epsilon_from_normalized_b_vector
        )
        lifecycle_auto_context["fixed_by_condensation_flags"] = [
            bool(value <= 0.0) for value in np.asarray(jax.device_get(b), dtype=np.float64)
        ]
        lifecycle_field_sources["element_phi_vector"] = "KL reduced b at bundle timing"
        lifecycle_field_sources["element_epsilon_vector"] = "b / sum(b)"
        lifecycle_field_sources["fixed_by_condensation_flags"] = "b <= 0 diagnostic rule"
        lifecycle_field_exactness["element_phi_vector"] = "KL_owned_candidate"
        lifecycle_field_exactness["element_epsilon_vector"] = "KL_owned_candidate"
        lifecycle_field_exactness["fixed_by_condensation_flags"] = "KL_owned_candidate"
    lifecycle_auto_context["fastchem_options"] = {
        "chem_accuracy": 1.0e-5,
        "nb_max_fastchem_iter": 3000,
        "nb_switch_to_newton": 400,
        "chem_use_backup_solver": False,
        "element_density_minlimit": 1.0e-155,
        "molecule_density_minlimit": 1.0e-155,
        "source": "fastchem/fastchem_src/options.h defaults",
    }
    lifecycle_field_sources["fastchem_options"] = (
        "FastChemOptions defaults from fastchem/fastchem_src/options.h"
    )
    lifecycle_field_exactness["fastchem_options"] = "FastChem_default_candidate"
    lifecycle_auto_context["branch_decisions"] = {
        "initial_use_backup_solver": False,
        "newton_branch_threshold_iter_gt": 400,
        "backup_forced_at_iteration": 390,
        "actual_converged_iteration": None,
        "actual_backup_switch_iteration": None,
        "actual_newtonSolMult_used": None,
        "source": (
            "static FastChem branch rules; actual runtime branch choices unavailable"
        ),
    }
    lifecycle_field_sources["branch_decisions"] = (
        "static FastChem branch thresholds with actual runtime decisions unset"
    )
    lifecycle_field_exactness["branch_decisions"] = (
        "partial_runtime_candidate_actual_branch_state_missing"
    )
    if electron_row_index is not None:
        lifecycle_auto_context["electron_old_density"] = {
            "candidate_cgs": float(
                jax.device_get(
                    gas_species_number_density_cgs_candidate_vector[
                        electron_row_index
                    ]
                )
            ),
            "electron_row_index": int(electron_row_index),
            "source": (
                "current KL electron species density candidate; FastChem number_density_old[e_] "
                "at gas_phase.calculate entry remains unverified"
            ),
        }
        lifecycle_field_sources["electron_old_density"] = (
            "current KL electron density candidate"
        )
        lifecycle_field_exactness["electron_old_density"] = (
            "KL_owned_candidate_not_verified_as_FastChem_number_density_old"
        )
    replay_lifecycle_context = {**lifecycle_auto_context, **lifecycle_context}
    replay_lifecycle_context.setdefault("gas_number_density_cgs", gas_number_density_cgs)
    replay_lifecycle_context.setdefault(
        "molecule_number_density_gauge_cgs",
        gas_number_density_cgs,
    )
    replay_lifecycle_context.setdefault(
        "pressure_bar",
        element_slot_gas_density_ntot_normalization_carrier.get("pressure_bar"),
    )
    replay_lifecycle_context.setdefault(
        "temperature_K",
        element_slot_gas_density_ntot_normalization_carrier.get("temperature_K"),
    )
    replay_lifecycle_context.setdefault(
        "molecule_mass_action_source",
        "fastchem_pressure_scaled_from_hvector",
    )
    replay_initial_species_density = (
        atomic_prefix_gas_species_density_cgs_candidate_vector
        if atomic_prefix_available
        else gas_species_number_density_cgs_candidate_vector
    )
    kl_gas_phase_calculate_replay_results = (
        _build_kl_gas_phase_calculate_replay_results(
            formula_matrix=formula_matrix,
            initial_species_density_cgs=replay_initial_species_density,
            recovered_hvector_gas=recovered_hvector_gas,
            element_labels=element_labels_for_lifecycle,
            lifecycle_context=replay_lifecycle_context,
        )
    )
    kl_gas_phase_calculate_replay_status = dict(
        kl_gas_phase_calculate_replay_results.pop("replay_loop_status")
    )
    kl_replay_species_density_vector = jnp.asarray(
        kl_gas_phase_calculate_replay_results[
            "post_initial_species_density_vector_cgs"
        ],
        dtype=jnp.float64,
    )
    kl_replay_total_element_density_cgs = jnp.sum(
        formula_matrix @ kl_replay_species_density_vector
    )
    fastchem_post_initial_gas_species_density_replay_carrier.update(
        {
            "replay_loop_status": kl_gas_phase_calculate_replay_status,
            "replay_loop_available": True,
            "post_initial_species_density_vector_cgs": (
                kl_gas_phase_calculate_replay_results[
                    "post_initial_species_density_vector_cgs"
                ]
            ),
            "post_initial_species_density_source": (
                kl_gas_phase_calculate_replay_status[
                    "post_initial_species_density_source"
                ]
            ),
            "post_initial_species_density_vector_shape": [
                int(kl_replay_species_density_vector.shape[0])
            ],
            "post_initial_total_element_density_cgs_candidate": float(
                jax.device_get(kl_replay_total_element_density_cgs)
            ),
            "missing_exact_inputs": [],
            "owner_status": (
                "default-off KL GasPhase::calculate replay loop fills the "
                "post-initial species density boundary; diagnostic only and "
                "not used as a production constructor input"
            ),
        }
    )
    runtime_timing_context = lifecycle_context.get("runtime_timing_results", {})
    if runtime_timing_context is None:
        runtime_timing_context = {}
    if not runtime_timing_context:
        runtime_timing_context = dict(kl_gas_phase_calculate_replay_results)
    runtime_timing_field_names = [
        "actual_converged_iteration",
        "actual_backup_switch_iteration",
        "actual_newtonSolMult_used",
        "actual_newtonSolMult_iteration",
        "actual_electron_old_density_cgs",
        "post_initial_species_density_vector_cgs",
    ]
    runtime_timing_result_ports = {
        name: {
            "available": name in runtime_timing_context,
            "value": runtime_timing_context.get(name),
            "source": (
                "gas_phase_calculate_lifecycle_context.runtime_timing_results"
                if name in runtime_timing_context
                else None
            ),
            "used_as_KL_constructor_input": False,
            "FastChem_trace_values_used_as_inputs": False,
        }
        for name in runtime_timing_field_names
    }
    missing_runtime_timing_result_ports = [
        name for name in runtime_timing_field_names if name not in runtime_timing_context
    ]
    gas_phase_calculate_runtime_timing_result_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": bool(runtime_timing_context),
        "exact_owner_verified": False,
        "target_fastchem_source": (
            "fastchem/fastchem_src/gas_phase/calculate.cpp::GasPhase::calculate"
        ),
        "result_ports": runtime_timing_result_ports,
        "missing_result_ports": missing_runtime_timing_result_ports,
        "exact_output_target": (
            "post-initial species density vector and branch/electron timing "
            "from a KL-owned replay of GasPhase::calculate"
        ),
        "owner_status": (
            "runtime timing result slots for the default-off lifecycle replay; "
            "unavailable until the replay loop emits actual branch/electron timing"
        ),
    }
    for name, value in lifecycle_auto_context.items():
        lifecycle_context.setdefault(name, value)
    lifecycle_field_names = [
        "element_calculation_order",
        "element_solver_order_vector",
        "major_molecules_inc",
        "minor_molecules",
        "molecule_mass_action_constants",
        "fastchem_options",
        "branch_decisions",
        "element_phi_vector",
        "element_epsilon_vector",
        "fixed_by_condensation_flags",
        "electron_old_density",
    ]
    lifecycle_field_ports = {
        name: {
            "available": name in lifecycle_context,
            "value": lifecycle_context.get(name),
            "source": lifecycle_field_sources.get(
                name,
                "explicit gas_phase_calculate_lifecycle_context",
            )
            if name in lifecycle_context
            else None,
            "exactness": lifecycle_field_exactness.get(
                name,
                "explicit_context_reference_or_candidate",
            )
            if name in lifecycle_context
            else None,
            "used_as_KL_constructor_input": False,
            "FastChem_trace_values_used_as_inputs": False,
        }
        for name in lifecycle_field_names
    }
    missing_lifecycle_field_ports = [
        name for name in lifecycle_field_names if name not in lifecycle_context
    ]
    fastchem_gas_phase_calculate_lifecycle_replay_contract_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": True,
        "exact_owner_verified": False,
        "target_fastchem_source": (
            "fastchem/fastchem_src/gas_phase/calculate.cpp::GasPhase::calculate"
        ),
        "post_initial_copyout_source": (
            "fastchem/fastchem_src/calc_densities.cpp copies gas_phase.species[i]->number_density "
            "after gas_phase.calculate"
        ),
        "lifecycle_stage_order": [
            "reset element.number_density_maj",
            "calcMinorSpeciesDensities seed from current molecule densities",
            "snapshot number_density_old for all species",
            "iterate until convergence or nb_max_fastchem_iter",
            "check n_min + n_maj against phi * gas_density and maybe enable backup solver",
            "loop element_calculation_order and calculateElementDensities",
            "calculate major molecule densities during each element solve",
            "refresh minor molecule densities for all elements",
            "solve electron density when electron species exists",
            "repeat n_min + n_maj backup check",
            "convergence check against number_density_old",
            "optional multidimensional Newton branch after nb_switch_to_newton",
            "optional backup branch at iteration 390",
            "copy current species densities into number_density_old",
            "copy gas_phase.species number_density into output number_densities",
        ],
        "solver_dispatch_paths": [
            "fixed_or_epsilon_zero",
            "intertSol",
            "linSol",
            "quadSol",
            "newtonSol",
            "backupSol",
            "newtonSolMult",
            "electron_solver",
        ],
        "KL_available_inputs": [
            "formula_matrix",
            "ln_nk",
            "gk",
            "recovered_hvector_gas",
            "gas_number_density_cgs",
            "element_labels_reduced_order",
            "current KL species density candidate",
            "atomic-prefix mass-action species density candidate",
        ],
        "auto_populated_lifecycle_field_ports": sorted(lifecycle_auto_context),
        "lifecycle_field_ports": lifecycle_field_ports,
        "available_lifecycle_field_ports": [
            name for name in lifecycle_field_names if name in lifecycle_context
        ],
        "missing_lifecycle_field_ports": missing_lifecycle_field_ports,
        "missing_exact_inputs": [
            "FastChem element_calculation_order in species/source ordering",
            "FastChem per-element solver_order values",
            "FastChem major_molecules_inc and minor_molecules lists",
            "FastChem molecule mass_action_constant values at this temperature",
            "FastChem options: chem_accuracy, nb_max_fastchem_iter, nb_switch_to_newton, chem_use_backup_solver",
            "FastChem element phi, epsilon, fixed_by_condensation lifecycle state before gas_phase.calculate",
            "FastChem number_density_old lifecycle and backup/Newton branch decisions",
            "FastChem electron solver state and old electron density",
        ],
        "field_port_status": (
            "all lifecycle ports populated"
            if not missing_lifecycle_field_ports
            else "lifecycle ports emitted; exact replay inputs still missing"
        ),
        "exact_output_target": (
            "post-initial gas species number_density vector after gas_phase.calculate"
        ),
        "owner_status": (
            "default-off contract for a KL replay of FastChem gas_phase.calculate; "
            "records the lifecycle needed before this can become an exact owner"
        ),
    }
    fastchem_post_initial_gas_total_element_density_replay_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": True,
        "exact_owner_verified": False,
        "source_formula": (
            "FastChem GasPhase::totalElementDensity(): sum molecule "
            "stoichiometric burdens plus free element number densities"
        ),
        "KL_replay_formula": (
            "sum(formula_matrix @ gas_species_number_density_cgs_candidate_vector)"
        ),
        "gas_species_density_source": (
            "fastchem_post_initial_gas_species_density_replay_carrier.current_species_density_cgs_candidate_vector"
        ),
        "total_element_density_cgs_candidate": float(
            jax.device_get(gas_species_total_element_density_cgs_candidate)
        ),
        "non_electron_total_element_density_cgs_candidate": float(
            jax.device_get(
                gas_species_non_electron_total_element_density_cgs_candidate
            )
        ),
        "positive_stoich_total_element_density_cgs_candidate": float(
            jax.device_get(
                gas_species_positive_stoich_total_element_density_cgs_candidate
            )
        ),
        "electron_row_index": electron_row_index,
        "electron_row_contribution_cgs": electron_row_contribution_cgs,
        "element_labels_reduced_order": (
            None
            if element_labels_reduced_order is None
            else [str(label) for label in element_labels_reduced_order]
        ),
        "element_density_cgs_candidate_vector": _diagnostic_json_array(
            gas_species_element_density_cgs_candidate_vector
        ),
        "missing_exact_inputs": [
            "post-initial FastChem gas species number_density vector after gas_phase.calculate"
        ],
        "owner_status": (
            "FastChem totalElementDensity aggregation is replayed from KL-owned "
            "candidate species densities; remaining exactness depends on the "
            "post-initial gas species density timing"
        ),
    }
    fastchem_post_initial_gas_total_density_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": True,
        "exact_owner_verified": False,
        "candidate_total_density_cgs": float(
            jax.device_get(gas_species_total_element_density_cgs_candidate)
        ),
        "candidate_total_density_source": (
            "gas_species_total_element_density_cgs_candidate"
        ),
        "source_formula": (
            "sum(formula_matrix @ (nk / sum(nk) * gas_number_density_cgs))"
        ),
        "replay_carrier": "fastchem_post_initial_gas_total_element_density_replay_carrier",
        "target_fastchem_source": (
            "fastchem/fastchem_src/calc_densities.cpp::gas_phase.totalElementDensity() "
            "after initial gas_phase.calculate and before condensate maxDensity"
        ),
        "missing_exact_inputs": [
            "KL-owned FastChem-style gas-only totalElementDensity replay at post_initial_gas_total_element_density timing"
        ],
        "owner_status": (
            "KL-owned same-boundary candidate for FastChem post-initial gas "
            "total_element_density; diagnostic only until gas-phase replay timing closes"
        ),
    }
    reduced_assembly_owner_density_denominator_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "lifecycle_timing": (
            "KL exact input bundle at reduced-solver entry before _update_all_with_metrics"
        ),
        "source_provenance": call_site_provenance,
        "gas_number_density_cgs": gas_number_density_cgs,
        "sum_nk_particle_denominator": float(
            jax.device_get(gas_species_particle_count_denominator)
        ),
        "ntotk_denominator": float(jax.device_get(gas_species_ntot_denominator)),
        "sum_b_reduced_element_inventory_denominator": float(
            jax.device_get(gas_species_reduced_element_inventory_sum)
        ),
        "resn_sum_nk_minus_ntotk": float(
            jax.device_get(gas_species_particle_count_denominator - gas_species_ntot_denominator)
        ),
        "ngas_over_sum_nk_cgs": float(
            jax.device_get(gas_species_count_density_cgs_candidate)
        ),
        "ngas_over_ntotk_cgs": float(
            jax.device_get(gas_species_ntot_density_cgs_candidate)
        ),
        "formula_matrix_total_element_density_cgs": float(
            jax.device_get(gas_species_total_element_density_cgs_candidate)
        ),
        "inventory_renormalized_total_element_density_cgs": float(
            jax.device_get(
                gas_species_inventory_renormalized_total_element_density_cgs_candidate
            )
        ),
        "owner_status": (
            "KL-owned reduced-assembly density/denominator provenance carrier; "
            "diagnostic candidates only, not a proven J_ec owner scalar"
        ),
    }
    sum_nk_denominator = float(
        reduced_assembly_owner_density_denominator_carrier[
            "sum_nk_particle_denominator"
        ]
    )
    ntotk_denominator = float(
        reduced_assembly_owner_density_denominator_carrier["ntotk_denominator"]
    )
    sum_b_denominator = float(
        reduced_assembly_owner_density_denominator_carrier[
            "sum_b_reduced_element_inventory_denominator"
        ]
    )
    ngas_over_sum_nk_cgs = float(
        reduced_assembly_owner_density_denominator_carrier["ngas_over_sum_nk_cgs"]
    )
    correction_safe_sum_nk = max(sum_nk_denominator, 1.0e-300)
    correction_safe_ntotk = max(ntotk_denominator, 1.0e-300)
    correction_safe_sum_b = max(sum_b_denominator, 1.0e-300)
    row_scaled_jec_owner_scalar_correction_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": True,
        "lifecycle_timing": (
            "KL exact input bundle at reduced-solver entry before _update_all_with_metrics"
        ),
        "source_provenance": call_site_provenance,
        "baseline_owner_scalar_cgs": ngas_over_sum_nk_cgs,
        "baseline_owner_scalar_source": (
            "reduced_assembly_owner_density_denominator_carrier.ngas_over_sum_nk_cgs"
        ),
        "candidate_correction_factors": {
            "unity": 1.0,
            "sum_nk_over_ntotk": sum_nk_denominator / correction_safe_ntotk,
            "ntotk_over_sum_nk": ntotk_denominator / correction_safe_sum_nk,
            "sum_b_reduced_element_inventory": sum_b_denominator,
            "inverse_sum_b_reduced_element_inventory": 1.0 / correction_safe_sum_b,
            "one_plus_resn_over_sum_nk": (
                1.0
                + float(
                    reduced_assembly_owner_density_denominator_carrier[
                        "resn_sum_nk_minus_ntotk"
                    ]
                )
                / correction_safe_sum_nk
            ),
            "one_minus_resn_over_sum_nk": (
                1.0
                - float(
                    reduced_assembly_owner_density_denominator_carrier[
                        "resn_sum_nk_minus_ntotk"
                    ]
                )
                / correction_safe_sum_nk
            ),
            "formula_total_over_baseline_owner_scalar": (
                float(
                    reduced_assembly_owner_density_denominator_carrier[
                        "formula_matrix_total_element_density_cgs"
                    ]
                )
                / max(abs(ngas_over_sum_nk_cgs), 1.0e-300)
            ),
            "inventory_renormalized_total_over_baseline_owner_scalar": (
                float(
                    reduced_assembly_owner_density_denominator_carrier[
                        "inventory_renormalized_total_element_density_cgs"
                    ]
                )
                / max(abs(ngas_over_sum_nk_cgs), 1.0e-300)
            ),
        },
        "candidate_factor_sources": {
            "unity": "baseline same-timing carrier",
            "sum_nk_over_ntotk": "same-timing particle and ntot denominators",
            "ntotk_over_sum_nk": "same-timing particle and ntot denominators",
            "sum_b_reduced_element_inventory": "same-timing reduced element inventory sum",
            "inverse_sum_b_reduced_element_inventory": "same-timing reduced element inventory sum",
            "one_plus_resn_over_sum_nk": "same-timing sum_nk - ntotk residual",
            "one_minus_resn_over_sum_nk": "same-timing sum_nk - ntotk residual",
            "formula_total_over_baseline_owner_scalar": "same-timing formula-matrix total density candidate",
            "inventory_renormalized_total_over_baseline_owner_scalar": "same-timing inventory-renormalized total density candidate",
        },
        "missing_inputs_for_exact_owner_correction": [
            "row_scaled_element_condensate_J_ec_owner_scalar_correction",
            "or a KL-owned denominator/gas-density field at the exact row-scaled J_ec owner timing",
        ],
        "owner_status": (
            "KL-owned same-timing correction-factor carrier for auditing the "
            "small row-scaled J_ec owner scalar gap; diagnostic candidates only"
        ),
    }
    reduced_system_condensate_coupling_source_audit = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": True,
        "lifecycle_timing": (
            "KL exact input bundle at reduced-solver entry before _update_all_with_metrics"
        ),
        "source_provenance": call_site_provenance,
        "q_cond_slot_scale_vector": _diagnostic_json_array(sk),
        "rhs_cond_slot_scale_vector": _diagnostic_json_array(sk * hvector_cond - mk),
        "condensate_state_vector_mk": _diagnostic_json_array(mk),
        "condensate_budget_cap_vector": _diagnostic_json_array(
            condensate_budget_cap_vector
        ),
        "coupling_source_formulas": {
            "Q_cond": "formula_matrix_cond @ diag(sk) @ formula_matrix_cond.T",
            "condvec": "formula_matrix_cond @ (sk * hvector_cond - mk)",
            "sk": "mk * mk / exp(epsilon)",
        },
        "cgs_row_scaled_J_ec_owner_block_materialized_by_python_reduced_assembly": False,
        "owner_status": (
            "source audit for KL reduced-system condensate coupling; these terms "
            "are floor-scale/dimensionless reduced Newton quantities, not the cgs "
            "row-scaled FastChem J_ec owner-density block"
        ),
        "missing_inputs_for_cgs_row_scaled_J_ec_owner_block": [
            "row_scaled_element_condensate_J_ec_owner_scalar_correction",
            "or old-state cgs condensate density vector at FastChem J_ec block construction timing",
        ],
    }
    row_scaled_jec_owner_scalar_verifier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "lifecycle_timing": (
            "KL exact input bundle at reduced-solver entry before _update_all_with_metrics"
        ),
        "source_provenance": call_site_provenance,
        "scalar_cgs": None,
        "owner_status": (
            "placeholder for a row-scaled J_ec element-condensate owner scalar "
            "verifier; unavailable here because the exact input bundle does not "
            "receive the row-scaled J_ec target block"
        ),
        "missing_inputs": [
            "row_scaled_element_condensate_J_ec_target_block",
            "fixed_84_condensate_slot_mapping",
            "selected_element_row_scaling_vector",
        ],
        "factorization_reference_fields": [
            "reduced_assembly_owner_density_denominator_carrier.gas_number_density_cgs",
            "reduced_assembly_owner_density_denominator_carrier.sum_nk_particle_denominator",
            "reduced_assembly_owner_density_denominator_carrier.ngas_over_sum_nk_cgs",
        ],
    }
    kl_native_row_scaled_jec_block_candidate = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "lifecycle_timing": (
            "KL exact input bundle at reduced-solver entry before _update_all_with_metrics"
        ),
        "source_provenance": call_site_provenance,
        "owner_scalar_source": (
            "reduced_assembly_owner_density_denominator_carrier.ngas_over_sum_nk_cgs"
        ),
        "owner_scalar_cgs": None,
        "row_scaled_element_condensate_J_ec_candidate_block": None,
        "missing_inputs": [
            "fixed_84_condensate_slot_mapping",
            "selected_element_row_scaling_vector",
        ],
    }
    if (
        selected_element_row_scaling_vector is not None
        and condensates_jac_indices is not None
    ):
        selected_row_scaling = np.asarray(
            selected_element_row_scaling_vector, dtype=np.float64
        )
        cond_indices = np.asarray([int(index) for index in condensates_jac_indices])
        formula_cond_host = np.asarray(jax.device_get(formula_matrix_cond), dtype=np.float64)
        budget_cap_host = np.asarray(
            jax.device_get(condensate_budget_cap_vector), dtype=np.float64
        )
        selected_formula = formula_cond_host[:, cond_indices]
        selected_budget_cap = budget_cap_host[cond_indices]
        owner_scalar_cgs = float(
            reduced_assembly_owner_density_denominator_carrier["ngas_over_sum_nk_cgs"]
        )
        if selected_row_scaling.shape[0] == selected_formula.shape[0]:
            candidate_block = (
                selected_formula
                * (selected_budget_cap * owner_scalar_cgs)[None, :]
            ) / np.maximum(selected_row_scaling, 1.0e-300)[:, None]
            kl_native_row_scaled_jec_block_candidate = {
                "diagnostic_only": True,
                "default_off": True,
                "constructor_input": False,
                "reference_trace_input": False,
                "FastChem_trace_values_used_as_inputs": False,
                "used_as_KL_constructor_input": False,
                "available": True,
                "lifecycle_timing": (
                    "KL exact input bundle at reduced-solver entry before _update_all_with_metrics"
                ),
                "source_provenance": call_site_provenance,
                "owner_scalar_source": (
                    "reduced_assembly_owner_density_denominator_carrier.ngas_over_sum_nk_cgs"
                ),
                "owner_scalar_cgs": owner_scalar_cgs,
                "row_scaled_element_condensate_J_ec_candidate_block": (
                    candidate_block.tolist()
                ),
                "candidate_block_shape": [int(value) for value in candidate_block.shape],
                "selected_element_row_scaling_vector": selected_row_scaling.tolist(),
                "missing_inputs": [],
                "owner_status": (
                    "KL-native row-scaled element-condensate J_ec candidate block "
                    "materialized from same-timing carrier density; diagnostic only"
                ),
            }
        else:
            kl_native_row_scaled_jec_block_candidate = {
                **kl_native_row_scaled_jec_block_candidate,
                "selected_element_row_scaling_length": int(selected_row_scaling.shape[0]),
                "expected_row_count": int(selected_formula.shape[0]),
                "missing_inputs": [
                    "selected_element_row_scaling_vector_length_matches_rows"
                ],
            }
    if (
        row_scaled_element_condensate_jec_target_block is not None
        and selected_element_row_scaling_vector is not None
        and condensates_jac_indices is not None
    ):
        target_block = np.asarray(
            row_scaled_element_condensate_jec_target_block, dtype=np.float64
        )
        selected_row_scaling = np.asarray(
            selected_element_row_scaling_vector, dtype=np.float64
        )
        cond_indices = np.asarray([int(index) for index in condensates_jac_indices])
        formula_cond_host = np.asarray(jax.device_get(formula_matrix_cond), dtype=np.float64)
        budget_cap_host = np.asarray(
            jax.device_get(condensate_budget_cap_vector), dtype=np.float64
        )
        selected_formula = formula_cond_host[:, cond_indices]
        selected_budget_cap = budget_cap_host[cond_indices]
        unit_block = (
            selected_formula * selected_budget_cap[None, :]
        ) / np.maximum(selected_row_scaling, 1.0e-300)[:, None]
        if target_block.shape == unit_block.shape:
            mask = (
                np.isfinite(target_block)
                & np.isfinite(unit_block)
                & (np.abs(unit_block) > 1.0e-300)
            )
            ratios = target_block[mask] / unit_block[mask]
        else:
            mask = np.zeros(target_block.shape, dtype=bool)
            ratios = np.asarray([], dtype=np.float64)
        if ratios.size:
            scalar_cgs = float(np.median(ratios))
            row_scaled_jec_owner_scalar_verifier = {
                "diagnostic_only": True,
                "default_off": True,
                "constructor_input": False,
                "reference_trace_input": True,
                "FastChem_trace_values_used_as_inputs": False,
                "used_as_KL_constructor_input": False,
                "available": True,
                "lifecycle_timing": (
                    "KL exact input bundle at reduced-solver entry before _update_all_with_metrics"
                ),
                "source_provenance": call_site_provenance,
                "scalar_cgs": scalar_cgs,
                "sample_count": int(ratios.size),
                "ratio_min": float(np.min(ratios)),
                "ratio_max": float(np.max(ratios)),
                "ratio_median_abs_deviation": float(np.median(np.abs(ratios - scalar_cgs))),
                "target_block_shape": [int(value) for value in target_block.shape],
                "unit_block_shape": [int(value) for value in unit_block.shape],
                "row_scaled_element_condensate_jec_target_block": target_block.tolist(),
                "selected_element_row_scaling_vector": selected_row_scaling.tolist(),
                "owner_status": (
                    "reference-only row-scaled J_ec element-condensate owner scalar "
                    "verifier inferred from the attached target block"
                ),
                "missing_inputs": [],
                "factorization_reference_fields": [
                    "reduced_assembly_owner_density_denominator_carrier.gas_number_density_cgs",
                    "reduced_assembly_owner_density_denominator_carrier.sum_nk_particle_denominator",
                    "reduced_assembly_owner_density_denominator_carrier.ngas_over_sum_nk_cgs",
                ],
            }
        else:
            row_scaled_jec_owner_scalar_verifier = {
                **row_scaled_jec_owner_scalar_verifier,
                "target_block_shape": [int(value) for value in target_block.shape],
                "unit_block_shape": [int(value) for value in unit_block.shape],
                "sample_count": 0,
                "missing_inputs": (
                    []
                    if target_block.shape == unit_block.shape
                    else ["target_block_shape_matches_unit_block_shape"]
                ),
            }
    gas_species_total_element_density_metadata = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "source_formula": (
            "sum(formula_matrix @ (nk / sum(nk) * gas_number_density_cgs))"
        ),
        "particle_count_denominator": float(
            jax.device_get(gas_species_particle_count_denominator)
        ),
        "ntot_denominator": float(jax.device_get(gas_species_ntot_denominator)),
        "reduced_element_inventory_sum": float(
            jax.device_get(gas_species_reduced_element_inventory_sum)
        ),
        "particle_count_density_cgs_candidate": float(
            jax.device_get(gas_species_count_density_cgs_candidate)
        ),
        "ntot_density_cgs_candidate": float(
            jax.device_get(gas_species_ntot_density_cgs_candidate)
        ),
        "inventory_renormalized_total_element_density_cgs_candidate": float(
            jax.device_get(
                gas_species_inventory_renormalized_total_element_density_cgs_candidate
            )
        ),
        "owner_status": (
            "KL-owned gas-species physical-density candidate; reject unless gas "
            "solve residual and element conservation are acceptable"
        ),
    }
    old_state_condensate_density_cgs_candidate_vector = mk * gas_number_density_cgs
    finite_budget_cap = jnp.isfinite(condensate_budget_cap_vector)
    floor_threshold = jnp.asarray(1.0e-12, dtype=jnp.float64)
    retained_slot_flags = mk > floor_threshold
    floor_slot_flags = ~retained_slot_flags
    capped_slot_flags = (
        finite_budget_cap
        & retained_slot_flags
        & jnp.isclose(
            mk,
            condensate_budget_cap_vector,
            rtol=jnp.asarray(1.0e-10, dtype=jnp.float64),
            atol=jnp.asarray(0.0, dtype=jnp.float64),
        )
    )
    newly_active_slot_flags = floor_slot_flags
    maxdensity_total_element_density_cgs_candidate = float(
        jax.device_get(gas_species_total_element_density_cgs_candidate)
    )
    maxDensity_value_vector = (
        condensate_budget_cap_vector * maxdensity_total_element_density_cgs_candidate
    )
    maxDensity_slot_flags = floor_slot_flags & jnp.isfinite(maxDensity_value_vector)
    old_state_condensate_density_cgs_candidate_vector = jnp.where(
        maxDensity_slot_flags,
        maxDensity_value_vector,
        old_state_condensate_density_cgs_candidate_vector,
    )
    fixed_bridge_budget_cap_scalarization_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "lifecycle_timing": (
            "KL exact input bundle at reduced-solver entry before _update_all_with_metrics"
        ),
        "source_provenance": call_site_provenance,
        "density_vector_source": "old_state_condensate_density_cgs_candidate_vector",
        "budget_cap_source": "condensate_budget_cap_vector",
        "source_formula": (
            "median(old_state_condensate_density_cgs_candidate_vector[fixed_slots] / "
            "condensate_budget_cap_vector[fixed_slots])"
        ),
        "missing_inputs": ["fixed_84_condensate_slot_mapping"],
        "owner_status": (
            "KL-owned fixed-bridge scalarization of the emitted old-state cgs "
            "condensate density candidate; diagnostic only"
        ),
    }
    if condensates_jac_indices is not None:
        cond_indices = np.asarray([int(index) for index in condensates_jac_indices])
        density_host = np.asarray(
            jax.device_get(old_state_condensate_density_cgs_candidate_vector),
            dtype=np.float64,
        )
        budget_cap_host = np.asarray(
            jax.device_get(condensate_budget_cap_vector),
            dtype=np.float64,
        )
        if (
            cond_indices.size
            and int(np.max(cond_indices)) < density_host.shape[0]
            and int(np.max(cond_indices)) < budget_cap_host.shape[0]
        ):
            selected_density = density_host[cond_indices]
            selected_budget_cap = budget_cap_host[cond_indices]
            slot_scalars = selected_density / np.maximum(selected_budget_cap, 1.0e-300)
            valid_slot_scalars = slot_scalars[
                np.isfinite(slot_scalars) & (slot_scalars > 0.0)
            ]
            if valid_slot_scalars.size:
                scalar_cgs = float(np.median(valid_slot_scalars))
                fixed_bridge_budget_cap_scalarization_carrier = {
                    **fixed_bridge_budget_cap_scalarization_carrier,
                    "available": True,
                    "scalar_cgs": scalar_cgs,
                    "selected_slot_count": int(cond_indices.size),
                    "valid_slot_count": int(valid_slot_scalars.size),
                    "slot_scalar_min": float(np.min(valid_slot_scalars)),
                    "slot_scalar_median": scalar_cgs,
                    "slot_scalar_max": float(np.max(valid_slot_scalars)),
                    "slot_scalar_std": float(np.std(valid_slot_scalars)),
                    "slot_scalar_max_relative_deviation_from_median": float(
                        np.max(
                            np.abs(
                                valid_slot_scalars
                                / np.maximum(abs(scalar_cgs), 1.0e-300)
                                - 1.0
                            )
                        )
                    ),
                    "uniform": bool(
                        np.max(
                            np.abs(
                                valid_slot_scalars
                                / np.maximum(abs(scalar_cgs), 1.0e-300)
                                - 1.0
                            )
                        )
                        < 1.0e-12
                    ),
                    "selected_condensate_indices": cond_indices.tolist(),
                    "selected_old_state_density_cgs_vector": selected_density.tolist(),
                    "selected_budget_cap_vector": selected_budget_cap.tolist(),
                    "selected_slot_scalar_cgs_vector": slot_scalars.tolist(),
                    "missing_inputs": [],
                }
            else:
                fixed_bridge_budget_cap_scalarization_carrier = {
                    **fixed_bridge_budget_cap_scalarization_carrier,
                    "selected_slot_count": int(cond_indices.size),
                    "valid_slot_count": 0,
                    "missing_inputs": ["positive_finite_old_state_density_and_budget_cap"],
                }
        else:
            fixed_bridge_budget_cap_scalarization_carrier = {
                **fixed_bridge_budget_cap_scalarization_carrier,
                "selected_slot_count": int(cond_indices.size),
                "valid_slot_count": 0,
                "available_slot_count": int(density_host.shape[0]),
                "missing_inputs": [
                    "fixed_84_condensate_slot_mapping_indices_within_local_condensate_vector"
                ],
            }
    fastchem_style_maxdensity_seeding_total_density_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": True,
        "exact_owner_verified": False,
        "lifecycle_timing": (
            "KL exact input bundle at reduced-solver entry; candidate for "
            "FastChem post-calculation maxDensity seeding before "
            "cond_densities_old is copied"
        ),
        "source_provenance": call_site_provenance,
        "candidate_total_density_cgs": maxdensity_total_element_density_cgs_candidate,
        "candidate_total_density_source": (
            "gas_species_total_element_density_cgs_candidate"
        ),
        "candidate_cond_densities_old_source_formula": (
            "where(floor_slot, condensate_budget_cap_vector * "
            "candidate_total_density_cgs, mk * gas_number_density_cgs)"
        ),
        "candidate_maxDensity_vector_source": "maxDensity_value_vector",
        "candidate_old_state_density_vector_source": (
            "old_state_condensate_density_cgs_candidate_vector"
        ),
        "fixed_bridge_budget_cap_scalarization_source": (
            "fixed_bridge_budget_cap_scalarization_carrier"
        ),
        "missing_exact_inputs": [
            "FastChem-style total_element_density at post_calculate_entry_seeding timing"
        ],
        "owner_status": (
            "KL-owned maxDensity seeding candidate emitted for audit; exact "
            "FastChem timing has not been reconstructed from existing KL fields"
        ),
    }
    if fixed_bridge_budget_cap_scalarization_carrier.get("available", False):
        fastchem_style_maxdensity_seeding_total_density_carrier = {
            **fastchem_style_maxdensity_seeding_total_density_carrier,
            "fixed_bridge_scalar_cgs": fixed_bridge_budget_cap_scalarization_carrier[
                "scalar_cgs"
            ],
            "fixed_bridge_uniform": fixed_bridge_budget_cap_scalarization_carrier[
                "uniform"
            ],
            "fixed_bridge_slot_scalar_max_relative_deviation_from_median": (
                fixed_bridge_budget_cap_scalarization_carrier[
                    "slot_scalar_max_relative_deviation_from_median"
                ]
            ),
        }
    fastchem_style_maxdensity_budget_cap_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "exact_owner_verified": False,
        "lifecycle_timing": (
            "KL exact input bundle at reduced-solver entry; candidate for "
            "FastChem Condensate::maxDensity budget cap immediately before "
            "post_calculate_entry_seeding"
        ),
        "source_provenance": call_site_provenance,
        "budget_cap_source": "condensate_budget_cap_vector",
        "source_formula": "min_positive_element(b[element] / stoichiometry[element, condensate])",
        "missing_exact_inputs": [
            "FastChem element epsilon vector at Condensate::maxDensity call timing",
            "FastChem condensate stoichiometric reference-element budget cap ordering",
        ],
        "owner_status": (
            "KL-owned budget-cap candidate emitted for audit; exact FastChem "
            "maxDensity cap timing has not been reconstructed from existing KL fields"
        ),
    }
    fastchem_style_element_epsilon_budget_cap_vector = condensate_budget_cap_vector
    fastchem_style_element_epsilon_budget_cap_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": True,
        "exact_owner_verified": False,
        "element_epsilon_candidate_source": "b",
        "element_epsilon_candidate_vector": _diagnostic_json_array(b),
        "budget_cap_source": "fastchem_style_element_epsilon_budget_cap_vector",
        "source_formula": (
            "min_positive_element(element_epsilon_candidate_vector[element] / "
            "stoichiometry[element, condensate])"
        ),
        "missing_exact_inputs": [
            "FastChem element.epsilon vector at Condensate::maxDensity call timing"
        ],
        "owner_status": (
            "KL-owned candidate uses reduced b as element epsilon; M103 audits "
            "whether this equals the FastChem maxDensity caller epsilon vector"
        ),
    }
    fastchem_style_normalized_b_element_epsilon_budget_cap_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": True,
        "exact_owner_verified": False,
        "element_epsilon_candidate_source": "b / sum(b)",
        "b_sum": float(jax.device_get(jnp.sum(b))),
        "element_epsilon_candidate_vector": _diagnostic_json_array(
            element_epsilon_from_normalized_b_vector
        ),
        "budget_cap_source": "normalized_b_maxdensity_budget_cap_vector",
        "source_formula": (
            "min_positive_element((b / sum(b))[element] / "
            "stoichiometry[element, condensate])"
        ),
        "missing_exact_inputs": [],
        "owner_status": (
            "KL-owned normalized-b candidate for FastChem element.epsilon; "
            "diagnostic only until bridge verification closes"
        ),
    }
    fastchem_file_element_epsilon_vector = None
    fastchem_file_maxdensity_budget_cap_vector = None
    fastchem_file_element_epsilon_budget_cap_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "exact_owner_verified": False,
        "element_epsilon_candidate_source": "packaged FastChem abundance file",
        "source_formula": (
            "parse FastChem A(X) abundance file, normalize linear abundances "
            "to sum 1, then min_positive_element(epsilon[element] / "
            "stoichiometry[element, condensate])"
        ),
        "missing_exact_inputs": ["element_labels_reduced_order"],
        "owner_status": (
            "KL-owned file-backed candidate for FastChem element.epsilon; "
            "diagnostic only until bridge verification closes"
        ),
    }
    if element_labels_reduced_order is not None:
        try:
            from exogibbs.utils.fastchem_parity import build_aligned_abundance_vector

            aligned = build_aligned_abundance_vector(
                [str(label) for label in element_labels_reduced_order],
                source="fastchem_asplund_2020",
                normalize=True,
            )
            fastchem_file_element_epsilon_vector = jnp.asarray(
                aligned.vector,
                dtype=jnp.float64,
            )
            fastchem_file_budget_terms = jnp.where(
                formula_matrix_cond > 0.0,
                fastchem_file_element_epsilon_vector[:, None] / formula_matrix_cond,
                jnp.inf,
            )
            fastchem_file_maxdensity_budget_cap_vector = jnp.min(
                fastchem_file_budget_terms,
                axis=0,
            )
            fastchem_file_maxdensity_budget_cap_vector = jnp.where(
                jnp.isfinite(fastchem_file_maxdensity_budget_cap_vector),
                fastchem_file_maxdensity_budget_cap_vector,
                jnp.nan,
            )
            fastchem_file_element_epsilon_budget_cap_carrier = {
                **fastchem_file_element_epsilon_budget_cap_carrier,
                "available": True,
                "source_path": aligned.source_path,
                "element_epsilon_candidate_vector": _diagnostic_json_array(
                    fastchem_file_element_epsilon_vector
                ),
                "budget_cap_source": "fastchem_file_maxdensity_budget_cap_vector",
                "missing_exact_inputs": [],
            }
        except Exception as exc:  # pragma: no cover - diagnostic metadata only
            fastchem_file_element_epsilon_budget_cap_carrier = {
                **fastchem_file_element_epsilon_budget_cap_carrier,
                "available": False,
                "missing_exact_inputs": [
                    "packaged_fastchem_abundance_file_parseable",
                    str(exc),
                ],
            }
    fastchem_file_budget_maxdensity_owner_density_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": fastchem_file_maxdensity_budget_cap_vector is not None,
        "budget_cap_source": "fastchem_file_maxdensity_budget_cap_vector",
        "total_density_source": (
            "fastchem_post_initial_gas_total_element_density_replay_carrier."
            "total_element_density_cgs_candidate"
        ),
        "source_formula": (
            "fastchem_file_maxdensity_budget_cap_vector * "
            "fastchem_post_initial_gas_total_element_density_replay_carrier."
            "total_element_density_cgs_candidate"
        ),
        "owner_status": (
            "KL-owned file-backed maxDensity budget cap combined with current "
            "KL total-density timing candidate; diagnostic only"
        ),
        "missing_exact_inputs": []
        if fastchem_file_maxdensity_budget_cap_vector is not None
        else ["fastchem_file_maxdensity_budget_cap_vector"],
        "total_density_cgs_candidate": float(
            jax.device_get(gas_species_total_element_density_cgs_candidate)
        ),
    }
    fastchem_file_budget_maxdensity_owner_density_vector = (
        None
        if fastchem_file_maxdensity_budget_cap_vector is None
        else fastchem_file_maxdensity_budget_cap_vector
        * gas_species_total_element_density_cgs_candidate
    )
    if condensates_jac_indices is not None:
        cond_indices = np.asarray([int(index) for index in condensates_jac_indices])
        budget_cap_host = np.asarray(
            jax.device_get(condensate_budget_cap_vector),
            dtype=np.float64,
        )
        if cond_indices.size and int(np.max(cond_indices)) < budget_cap_host.shape[0]:
            selected_budget_cap = budget_cap_host[cond_indices]
            valid_budget_cap = selected_budget_cap[
                np.isfinite(selected_budget_cap) & (selected_budget_cap > 0.0)
            ]
            fastchem_style_maxdensity_budget_cap_carrier = {
                **fastchem_style_maxdensity_budget_cap_carrier,
                "available": bool(valid_budget_cap.size),
                "selected_slot_count": int(cond_indices.size),
                "valid_slot_count": int(valid_budget_cap.size),
                "selected_condensate_indices": cond_indices.tolist(),
                "selected_budget_cap_vector": selected_budget_cap.tolist(),
            }
            if valid_budget_cap.size:
                fastchem_style_maxdensity_budget_cap_carrier = {
                    **fastchem_style_maxdensity_budget_cap_carrier,
                    "budget_cap_min": float(np.min(valid_budget_cap)),
                    "budget_cap_median": float(np.median(valid_budget_cap)),
                    "budget_cap_max": float(np.max(valid_budget_cap)),
                    "budget_cap_std": float(np.std(valid_budget_cap)),
                }
        else:
            fastchem_style_maxdensity_budget_cap_carrier = {
                **fastchem_style_maxdensity_budget_cap_carrier,
                "selected_slot_count": int(cond_indices.size),
                "valid_slot_count": 0,
                "available_slot_count": int(budget_cap_host.shape[0]),
                "missing_inputs": [
                    "fixed_84_condensate_slot_mapping_indices_within_local_condensate_vector"
                ],
            }
    element_abundance_create_molecule_lists_timing_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": element_labels_reduced_order is not None,
        "target_fastchem_source": (
            "fastchem/fastchem_src/calc_densities.cpp sets "
            "element_abundance_cond[j] = element_data.elements[j].phi, then "
            "setElementAbundances(element_abundance_cond), "
            "element_data.setRelativeAbundances(), gas_phase.reInitialise(), "
            "and GasPhase::createMoleculeLists consumes elements[j].abundance"
        ),
        "comparison_target_linkage": (
            "v13 createMoleculeLists element_abundance rows are reference-only "
            "owner evidence and are not used as constructor inputs"
        ),
        "electron_row_policy": (
            "FastChem keeps electron abundance at 0 in setElementAbundances; "
            "KL molecule-abundance reconstruction excludes zero-abundance e- "
            "unless an explicit diagnostic policy variant requests inclusion"
        ),
        "charged_species_policy": (
            "charged species are included through their stoichiometric vectors; "
            "no charged species are dropped by this diagnostic carrier"
        ),
        "fixed_condensation_adjustment_metadata": (
            "candidate H clips non-positive reduced b values before "
            "normalization as a KL-owned fixed/condensation adjusted branch"
        ),
        "fastchem_reinitialise_order_tie_perturbation_metadata": {
            "diagnostic_only": True,
            "default_off": True,
            "constructor_input": False,
            "source": "fastchem/fastchem_src/gas_phase/init_solver.cpp::determineElementCalculationOrder",
            "rule": (
                "for each non-electron element pair with exactly equal abundance, "
                "mutate the later compared element by numeric_limits<double_type>::epsilon() "
                "* abundance before sorting in descending abundance order"
            ),
            "molecule_abundance_source": (
                "fastchem/fastchem_src/gas_phase/gas_phase.cpp::setMoleculeAbundances "
                "uses std::min_element with comparator: if elements[a].abundance != 0 "
                "then compare elements[a].abundance < elements[b].abundance else false"
            ),
            "FastChem_trace_values_used_as_inputs": False,
            "used_as_KL_constructor_input": False,
        },
        "source_stage_candidates": [],
        "best_KL_owned_candidate_from_M128": (
            "B_KL_gas_phase_exit_element_abundance"
        ),
        "element_abundance_createMoleculeLists_timing_candidate_vector": None,
        "normalization_gauge_metadata": (
            "all candidate vectors are element-order aligned; normalized "
            "candidates use sum(vector) = 1 when the finite positive sum is nonzero"
        ),
        "missing_exact_inputs": [],
        "owner_status": (
            "KL-owned default-off element.abundance source-stage carrier for "
            "createMoleculeLists timing; diagnostic only"
        ),
    }
    if element_labels_reduced_order is not None:
        labels_host = [str(label) for label in element_labels_reduced_order]
        b_host = np.asarray(jax.device_get(b), dtype=np.float64)
        b_sum = max(float(np.sum(b_host)), 1.0e-300)
        normalized_b_host = b_host / b_sum
        gas_species_density_host = np.asarray(
            jax.device_get(gas_species_number_density_cgs_candidate_vector),
            dtype=np.float64,
        )
        atomic_prefix_host = gas_species_density_host[:n_element_rows]
        atomic_prefix_sum = max(float(np.sum(atomic_prefix_host)), 1.0e-300)
        atomic_prefix_abundance_host = atomic_prefix_host / atomic_prefix_sum
        element_density_host = np.asarray(
            jax.device_get(gas_species_element_density_cgs_candidate_vector),
            dtype=np.float64,
        )
        element_density_sum = max(float(np.sum(element_density_host)), 1.0e-300)
        element_density_abundance_host = element_density_host / element_density_sum
        fixed_adjusted_host = np.where(b_host > 0.0, b_host, 0.0)
        fixed_adjusted_sum = max(float(np.sum(fixed_adjusted_host)), 1.0e-300)
        fixed_adjusted_abundance_host = fixed_adjusted_host / fixed_adjusted_sum
        def _fastchem_tie_perturb(values: np.ndarray) -> np.ndarray:
            adjusted = np.asarray(values, dtype=np.float64).copy()
            eps = np.finfo(np.float64).eps
            for i, label_i in enumerate(labels_host):
                if label_i == "e-":
                    continue
                for j, label_j in enumerate(labels_host):
                    if label_j == "e-" or i == j:
                        continue
                    if adjusted[i] == adjusted[j]:
                        adjusted[j] += eps * adjusted[j]
            return adjusted

        tie_perturbed_atomic_prefix_host = _fastchem_tie_perturb(
            atomic_prefix_abundance_host
        )
        file_abundance_host = (
            None
            if fastchem_file_element_epsilon_vector is None
            else np.asarray(
                jax.device_get(fastchem_file_element_epsilon_vector),
                dtype=np.float64,
            )
        )
        candidate_vectors = [
            {
                "candidate": "A_M128_current_KL_abundance_basis",
                "source_stage": "KL reduced b / sum(b) at exact bundle timing",
                "KL_owned": True,
                "can_emit_default_off": True,
                "normalization": "sum-normalized reduced b",
                "vector": normalized_b_host,
            },
            {
                "candidate": "B_M128_best_gas_phase_exit_element_abundance",
                "source_stage": (
                    "KL gas species vector atomic-prefix element densities / sum"
                ),
                "KL_owned": True,
                "can_emit_default_off": True,
                "normalization": "sum-normalized atomic-prefix cgs densities",
                "vector": atomic_prefix_abundance_host,
            },
            {
                "candidate": "D_normalized_initial_abundance_candidate",
                "source_stage": "element_epsilon_from_normalized_b_vector",
                "KL_owned": True,
                "can_emit_default_off": True,
                "normalization": "sum-normalized reduced b",
                "vector": normalized_b_host,
            },
            {
                "candidate": "E_post_initial_species_density_derived_abundance",
                "source_stage": "formula_matrix @ gas species density cgs candidate",
                "KL_owned": True,
                "can_emit_default_off": True,
                "normalization": "sum-normalized total element density vector",
                "vector": element_density_abundance_host,
            },
            {
                "candidate": "F_post_update_element_density_derived_abundance",
                "source_stage": "same KL-owned element-density vector at bundle timing",
                "KL_owned": True,
                "can_emit_default_off": True,
                "normalization": "sum-normalized total element density vector",
                "vector": element_density_abundance_host,
            },
            {
                "candidate": "G_old_element_density_timing_candidate",
                "source_stage": "pre-replay KL atomic-prefix element density",
                "KL_owned": True,
                "can_emit_default_off": True,
                "normalization": "sum-normalized atomic-prefix cgs densities",
                "vector": atomic_prefix_abundance_host,
            },
            {
                "candidate": "H_condensation_fixed_adjusted_abundance_candidate",
                "source_stage": "positive part of KL reduced b / sum(positive b)",
                "KL_owned": True,
                "can_emit_default_off": True,
                "normalization": "sum-normalized positive reduced b",
                "vector": fixed_adjusted_abundance_host,
            },
            {
                "candidate": "I_charged_electron_policy_variant",
                "source_stage": (
                    "B vector with explicit charged-species/electron policy metadata"
                ),
                "KL_owned": True,
                "can_emit_default_off": True,
                "normalization": "sum-normalized atomic-prefix cgs densities",
                "vector": atomic_prefix_abundance_host,
                "electron_policy_variant": "include_electron_in_molecule_abundance",
            },
            {
                "candidate": "J_best_KL_owned_FastChem_style_element_abundance_vector",
                "source_stage": "M128 best KL-owned source-stage candidate B",
                "KL_owned": True,
                "can_emit_default_off": True,
                "normalization": "sum-normalized atomic-prefix cgs densities",
                "vector": atomic_prefix_abundance_host,
            },
            {
                "candidate": "L_FastChem_reInitialise_tie_perturbed_B_vector",
                "source_stage": (
                    "M128 best B vector after Python reconstruction of "
                    "FastChem determineElementCalculationOrder tie perturbation"
                ),
                "KL_owned": True,
                "can_emit_default_off": True,
                "normalization": "sum-normalized atomic-prefix cgs densities with FastChem tie perturbation",
                "vector": tie_perturbed_atomic_prefix_host,
            },
            {
                "candidate": "K_FastChem_v13_exact_verifier_reference_only",
                "source_stage": "v13 reference target link only",
                "KL_owned": False,
                "can_emit_default_off": True,
                "normalization": "not emitted as a constructor vector",
                "vector": None,
                "reference_only": True,
            },
        ]
        if file_abundance_host is not None:
            candidate_vectors.insert(
                2,
                {
                    "candidate": "C_file_input_abundance_table_candidate",
                    "source_stage": (
                        "packaged FastChem abundance table parsed through "
                        "build_aligned_abundance_vector"
                    ),
                    "KL_owned": True,
                    "can_emit_default_off": True,
                    "normalization": "sum-normalized file abundance vector",
                    "vector": file_abundance_host,
                },
            )
        else:
            candidate_vectors.insert(
                2,
                {
                    "candidate": "C_file_input_abundance_table_candidate",
                    "source_stage": "packaged FastChem abundance table",
                    "KL_owned": True,
                    "can_emit_default_off": False,
                    "normalization": "unavailable",
                    "vector": None,
                    "missing_exact_inputs": [
                        "fastchem_file_element_epsilon_vector"
                    ],
                },
            )
        element_abundance_create_molecule_lists_timing_carrier = {
            **element_abundance_create_molecule_lists_timing_carrier,
            "element_labels": labels_host,
            "element_order": list(range(n_element_rows)),
            "source_stage_candidates": [
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "vector"
                }
                for candidate in candidate_vectors
            ],
            "candidate_vectors": [
                {
                    **{
                        key: value
                        for key, value in candidate.items()
                        if key != "vector"
                    },
                    "vector": (
                        None
                        if candidate.get("vector") is None
                        else np.asarray(
                            candidate["vector"],
                            dtype=np.float64,
                        ).tolist()
                    ),
                }
                for candidate in candidate_vectors
            ],
            "element_abundance_createMoleculeLists_timing_candidate_vector": (
                atomic_prefix_abundance_host.tolist()
            ),
            "best_candidate": "B_M128_best_gas_phase_exit_element_abundance",
            "best_candidate_source_stage": (
                "KL gas species vector atomic-prefix element densities / sum"
            ),
        }
    fastchem_phi_element_abundance_timing_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": element_labels_reduced_order is not None,
        "target_fastchem_source": (
            "CondensedPhase::calculate and calcDensities compute "
            "element.phi = epsilon * (1 - degree_of_condensation), normalize "
            "phi by sum(phi), then copy phi into element.abundance before "
            "gas_phase.reInitialise()/setMoleculeAbundances."
        ),
        "comparison_target_linkage": (
            "v14 setMoleculeAbundances candidate_element_abundance_values are "
            "reference-only owner evidence and are not used as constructor inputs"
        ),
        "normalization_gauge_metadata": (
            "FastChem-style phi candidates clip negative residual abundance, "
            "preserve electron-row zero policy through the available element "
            "order, and sum-normalize finite positive vectors when possible"
        ),
        "electron_row_policy": (
            "FastChem leaves electron abundance at zero in setElementAbundances; "
            "electron-bearing charged species are handled by explicit diagnostic "
            "policy variants in molecule-abundance reconstruction."
        ),
        "source_stage_candidates": [],
        "candidate_vectors": [],
        "best_KL_owned_candidate": None,
        "missing_exact_inputs": [],
        "owner_status": (
            "KL-owned default-off FastChem phi element.abundance timing carrier; "
            "diagnostic only"
        ),
    }
    fastchem_v15_calc_degree_condensation_burden_reconstruction_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": element_labels_reduced_order is not None,
        "target_fastchem_source": (
            "Element::calcDegreeOfCondensation consumes per-element density_cond "
            "and total_element_density immediately before Element::normalisePhi."
        ),
        "trace_marker": "exact_fixed_row_subspace_trace_v15_calcDegreeOfCondensation_burden_pre_normalisePhi",
        "source_stage_candidates": [],
        "candidate_vectors": [],
        "best_KL_owned_candidate": None,
        "reference_only_verifier": {
            "candidate": "G_FastChem_exact_v15_verifier_reference_only",
            "KL_owned": False,
            "trace_values_used_as_KL_constructor_inputs": False,
            "reference_only": True,
        },
        "owner_status": (
            "KL-owned default-off calcDegreeOfCondensation burden reconstruction "
            "carrier; diagnostic only"
        ),
    }
    fastchem_m141_post_solver_support_pruning_surrogate_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "FastChem selects active floor/maxDensity condensates by activity, "
            "solves the condensed-phase reduced system, then prunes support at "
            "the post-solver log_activity < -0.01 removal boundary."
        ),
        "source_stage_candidates": [],
        "candidate_masks": [],
        "best_KL_owned_candidate": None,
        "reference_only_verifier": {
            "candidate": "G_FastChem_post_final_removal_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "owner_status": (
            "KL-owned default-off post-solver support pruning surrogate carrier; "
            "diagnostic only"
        ),
    }
    fastchem_m142_old_log_activity_closure_reconstruction_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "CondPhaseSolver::newtonStep recomputes old_log_activity from the "
            "old all-element density vector and condensate logK before "
            "partitioning/removal."
        ),
        "thresholds": {
            "post_solver_removal_log_activity": -0.01,
        },
        "source_stage_candidates": [],
        "candidate_masks": [],
        "best_KL_owned_candidate": None,
        "reference_only_verifier": {
            "candidate": "F_FastChem_iter1_old_log_activity_closure_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "owner_status": (
            "KL-owned default-off old element-density log-activity closure "
            "reconstruction carrier; diagnostic only"
        ),
    }
    fastchem_m147_second_pass_36row_reduced_system_reconstruction_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "Second-pass condensed-phase 36-row reduced system assembly, solve "
            "result element tail, second correctValues writeback, and old "
            "log-activity support pruning chain."
        ),
        "candidate_names": [
            "A_M146_baseline_no_second_pass_solve",
            "B_KL_current_post_correctValues_36row_assembly",
            "C_KL_M141_to_M145_old_density_writeback_36row_assembly",
            "D_KL_retained14_support_reconstruction_36row_assembly",
            "E_KL_active_floor_slot_selector_36row_assembly",
            "F_best_KL_owned_second_pass_solve_result_candidate",
            "G_FastChem_exact_second_pass_verifier_reference_only",
        ],
        "mandatory_downstream_chain": [
            "solve_or_materialize_second_pass_36row_result",
            "extract_element_tail_slots",
            "rebuild_correctValues_element_components",
            "rebuild_second_writeback_vector",
            "rebuild_old_element_density_mapping",
            "recompute_old_log_activity_support",
            "rerun_v13_v12_key_set_comparison",
        ],
        "reference_only_verifier": {
            "candidate": "G_FastChem_exact_second_pass_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "owner_status": (
            "KL-owned default-off second-pass 36-row reduced-system "
            "reconstruction carrier; diagnostic only"
        ),
    }
    fastchem_m148_second_pass_row_column_scaling_reconstruction_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "Second-pass 36-row reduced-system row/column scaling convention "
            "used before solving the condensed-phase linear system."
        ),
        "candidate_names": [
            "A_M147_inverse_row_scaling_baseline",
            "B_KL_row_max_scaling",
            "C_KL_row_max_rhs_scaling",
            "D_KL_row_sum_scaling",
            "E_best_KL_owned_second_pass_scaling_candidate",
            "F_FastChem_exact_scaling_verifier_reference_only",
        ],
        "reference_only_verifier": {
            "candidate": "F_FastChem_exact_scaling_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "owner_status": (
            "KL-owned default-off second-pass row/column scaling reconstruction "
            "carrier; diagnostic only"
        ),
    }
    fastchem_m149_second_pass_jacobian_block_reconstruction_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "Second-pass 36-row reduced-system Jacobian block terms: "
            "condensate rows, retained-condensate element coupling, "
            "element-element molecule outer-product, atom diagonal, and "
            "removed-condensate fold-in terms."
        ),
        "candidate_names": [
            "A_M148_scaled_baseline",
            "B_KL_stoichiometric_condensate_element_blocks",
            "C_KL_molecule_outer_product_element_block",
            "D_KL_removed_condensate_foldin_proxy_block",
            "E_best_KL_owned_jacobian_block_candidate",
            "F_FastChem_exact_jacobian_subterm_verifier_reference_only",
        ],
        "reference_only_verifier": {
            "candidate": "F_FastChem_exact_jacobian_subterm_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "owner_status": (
            "KL-owned default-off second-pass Jacobian block reconstruction "
            "carrier; diagnostic only"
        ),
    }
    fastchem_m150_second_pass_element_element_jacobian_reconstruction_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "Second-pass element-element Jacobian block subterms, especially "
            "the old gas molecule-density outer-product term consumed by "
            "CondPhaseSolver::assembleJacobian."
        ),
        "candidate_names": [
            "A_M149_current_nk_molecule_outer_product",
            "B_KL_exp_ln_nk_molecule_outer_product",
            "C_KL_epsilon_scaled_nk_molecule_outer_product",
            "D_KL_atom_diagonal_density_variant",
            "E_KL_removed_condensate_foldin_timing_variant",
            "F_best_KL_owned_element_element_block_candidate",
            "G_FastChem_exact_element_element_subterm_verifier_reference_only",
        ],
        "reference_only_verifier": {
            "candidate": "G_FastChem_exact_element_element_subterm_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "owner_status": (
            "KL-owned default-off second-pass element-element Jacobian "
            "subterm reconstruction carrier; diagnostic only"
        ),
    }
    fastchem_m151_old_gas_molecule_density_outer_product_timing_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "Old gas Molecule::number_density vector consumed by the "
            "second-pass element-element Jacobian molecule outer-product."
        ),
        "candidate_names": [
            "A_M150_bundle_nk_tail_molecule_density",
            "B_KL_exp_ln_nk_tail_molecule_density",
            "C_KL_mass_action_from_b_element_density",
            "D_KL_mass_action_from_bk_element_density",
            "E_KL_mass_action_ln_ntot_gauge_variant",
            "F_best_KL_owned_old_molecule_density_candidate",
            "G_FastChem_exact_old_molecule_density_verifier_reference_only",
        ],
        "reference_only_verifier": {
            "candidate": "G_FastChem_exact_old_molecule_density_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "owner_status": (
            "KL-owned default-off old gas molecule-density outer-product "
            "timing carrier; diagnostic only"
        ),
    }
    fastchem_m152_old_full_element_density_gauge_reconstruction_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "Old full-element physical density vector and density gauge used "
            "to reconstruct second-pass gas Molecule::number_density."
        ),
        "candidate_names": [
            "A_M151_b_element_density_vector",
            "B_KL_bk_element_density_vector",
            "C_KL_b_exp_ln_ntot_density_gauge",
            "D_KL_b_exp_minus_ln_ntot_density_gauge",
            "E_KL_sum_b_normalized_density_gauge",
            "F_best_KL_owned_old_full_element_density_candidate",
            "G_FastChem_exact_old_full_element_density_verifier_reference_only",
        ],
        "reference_only_verifier": {
            "candidate": "G_FastChem_exact_old_full_element_density_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "owner_status": (
            "KL-owned default-off old full-element density/gauge reconstruction "
            "carrier; diagnostic only"
        ),
    }
    fastchem_m153_old_total_density_gauge_deep_reconstruction_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "Old full-element total-density gauge scalar used to materialize "
            "physical element densities before second-pass gas molecule "
            "reconstruction and molecule outer-product assembly."
        ),
        "candidate_names": [
            "A_M152_baseline_b",
            "B_b_times_old_total_density_gauge",
            "C_bk_times_old_total_density_gauge",
            "D_normalized_b_times_gas_number_density_cgs",
            "E_normalized_b_times_totalElementDensity_style_scalar",
            "F_post_initial_element_density_derived_old_vector",
            "G_second_pass_old_density_mapped_vector",
            "H_fixed_condensed_override_vector",
            "I_best_KL_owned_old_physical_full_element_density_vector",
            "J_FastChem_exact_verifier_reference_only",
        ],
        "reference_only_verifier": {
            "candidate": "J_FastChem_exact_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "owner_status": (
            "KL-owned default-off old total-density gauge deep reconstruction "
            "carrier; diagnostic only"
        ),
    }
    fastchem_m154_total_element_density_aggregation_timing_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "totalElementDensity aggregation scalar timing used to materialize "
            "old full-element physical densities for second-pass gas molecule "
            "reconstruction."
        ),
        "candidate_names": [
            "A_M153_totalElementDensity_style_scalar",
            "B_KL_ntot_density_cgs_scalar",
            "C_KL_inventory_renormalized_scalar",
            "D_KL_pressure_count_density_scalar",
            "E_KL_reduced_inventory_sum_corrected_scalar",
            "F_best_KL_owned_totalElementDensity_scalar",
            "G_FastChem_exact_totalElementDensity_verifier_reference_only",
        ],
        "reference_only_verifier": {
            "candidate": "G_FastChem_exact_totalElementDensity_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "owner_status": (
            "KL-owned default-off totalElementDensity aggregation timing "
            "carrier; diagnostic only"
        ),
    }
    fastchem_m155_condensed_total_element_density_correction_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "CondensedPhase::totalElementDensity aggregation correction added "
            "to GasPhase::totalElementDensity before the second-pass reduced "
            "system consumes total_element_density."
        ),
        "candidate_names": [
            "A_M154_best_gas_total_only",
            "B_gas_total_plus_retained14_old_density_raw_sum",
            "C_ntot_density_plus_retained14_old_density_raw_sum",
            "D_gas_total_plus_retained14_jec_owner_density_raw_sum",
            "E_gas_total_plus_retained14_maxDensity_raw_sum",
            "F_gas_total_plus_retained14_stoichiometric_burden_sum",
            "G_gas_total_plus_all_condensate_density_proxy_sum",
            "H_best_KL_owned_condensed_totalElementDensity_correction",
            "I_FastChem_exact_totalElementDensity_verifier_reference_only",
        ],
        "reference_only_verifier": {
            "candidate": "I_FastChem_exact_totalElementDensity_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "owner_status": (
            "KL-owned default-off condensed totalElementDensity correction "
            "carrier; diagnostic only"
        ),
    }
    fastchem_m156_retained_support_density_timing_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "Self-consistent retained-support condensed density timing and "
            "paired gas_phase.totalElementDensity timing consumed by the "
            "second-pass reduced system."
        ),
        "candidate_names": [
            "A_M155_best_retained14_jec_owner_raw_sum",
            "B_gas_total_fixed_point_retained14_budget_cap",
            "C_ntot_density_fixed_point_retained14_budget_cap",
            "D_gas_total_plus_self_consistent_retained14_maxDensity_sum",
            "E_gas_total_depleted_by_retained14_jec_plus_fixed_point_retained14",
            "F_gas_total_minus_retained14_stoichiometric_burden_plus_raw_retained14",
            "G_best_KL_owned_retained_support_density_timing",
            "H_FastChem_exact_totalElementDensity_verifier_reference_only",
        ],
        "reference_only_verifier": {
            "candidate": "H_FastChem_exact_totalElementDensity_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "owner_status": (
            "KL-owned default-off retained-support density timing carrier; "
            "diagnostic only"
        ),
    }
    fastchem_m157_gas_phase_total_element_density_timing_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "GasPhase::totalElementDensity timing after the gas solve and before "
            "retained condensed support is added for the second-pass reduced "
            "system total_element_density."
        ),
        "candidate_names": [
            "A_M156_best_gas_total_plus_retained_jec",
            "B_inventory_renormalized_gas_total_plus_retained_jec",
            "C_ntot_density_gas_total_plus_retained_jec",
            "D_particle_count_gas_total_plus_retained_jec",
            "E_positive_stoich_gas_total_plus_retained_jec",
            "F_free_atom_plus_molecule_split_gas_total_plus_retained_jec",
            "G_best_KL_owned_gas_phase_totalElementDensity_timing",
            "H_FastChem_exact_gas_totalElementDensity_verifier_reference_only",
        ],
        "reference_only_verifier": {
            "candidate": "H_FastChem_exact_gas_totalElementDensity_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "owner_status": (
            "KL-owned default-off gas-phase totalElementDensity timing carrier; "
            "diagnostic only"
        ),
    }
    fastchem_m158_post_gas_solve_molecule_density_timing_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "Post-gas-solve molecule density timing feeding "
            "GasPhase::totalElementDensity before the retained condensed "
            "support density is added to the second-pass reduced-system "
            "total_element_density."
        ),
        "candidate_names": [
            "A_M157_baseline_species_density_vector",
            "B_KL_mass_action_from_b_times_M157_best_scalar",
            "C_KL_mass_action_from_b_times_gas_total_scalar",
            "D_KL_mass_action_from_b_times_retained_corrected_scalar",
            "E_KL_atomic_prefix_plus_mass_action_molecule_tail",
            "F_KL_post_checkN_floor_clipped_molecule_tail",
            "G_best_KL_owned_post_gas_solve_molecule_density_timing",
            "H_FastChem_exact_post_gas_solve_molecule_density_verifier_reference_only",
        ],
        "reference_only_verifier": {
            "candidate": (
                "H_FastChem_exact_post_gas_solve_molecule_density_verifier_reference_only"
            ),
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "owner_status": (
            "KL-owned default-off post-gas-solve molecule-density timing carrier; "
            "diagnostic only"
        ),
    }
    fastchem_m159_molecule_density_cache_checkN_timing_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "GasPhase::calculateMoleculeDensities molecule number_density cache "
            "refresh and Molecule::checkN boundary immediately before "
            "GasPhase::totalElementDensity consumes the gas species cache."
        ),
        "candidate_names": [
            "A_M158_baseline_species_density_vector",
            "B_KL_Molecule_checkN_minlimit_floor_cgs",
            "C_KL_Molecule_checkN_gas_density_upper_cap",
            "D_KL_internal_dimensionless_checkN_then_cgs",
            "E_KL_stale_nk_tail_cache_copyout",
            "F_KL_zero_underflowed_checkN_tail",
            "G_best_KL_owned_molecule_density_cache_checkN_timing",
            "H_FastChem_exact_v12_checkN_verifier_reference_only",
        ],
        "reference_only_verifier": {
            "candidate": "H_FastChem_exact_v12_checkN_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "owner_status": (
            "KL-owned default-off molecule-density cache/checkN timing carrier; "
            "diagnostic only"
        ),
    }
    fastchem_m160_calculate_molecule_element_input_timing_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "Element::number_density vector consumed inside "
            "GasPhase::calculateMoleculeDensities before Molecule::checkN and "
            "GasPhase::totalElementDensity cache consumption."
        ),
        "candidate_names": [
            "A_M159_baseline_atomic_prefix_element_input",
            "B_KL_formula_gas_element_density_input",
            "C_KL_b_times_best_total_density_input",
            "D_KL_b_times_gas_total_density_input",
            "E_KL_max_atomic_and_formula_element_density_input",
            "F_KL_fixed_condensed_zero_override_input",
            "G_best_KL_owned_calculateMoleculeDensities_element_input_timing",
            "H_FastChem_exact_v12_element_input_verifier_reference_only",
        ],
        "reference_only_verifier": {
            "candidate": "H_FastChem_exact_v12_element_input_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "owner_status": (
            "KL-owned default-off calculateMoleculeDensities element-input "
            "timing carrier; diagnostic only"
        ),
    }
    fastchem_m161_old_new_element_density_lifecycle_timing_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "Per-refresh old/new Element::number_density lifecycle consumed by "
            "GasPhase::calculateMoleculeDensities product-term assembly."
        ),
        "candidate_names": [
            "A_M160_static_atomic_prefix",
            "B_KL_old_snapshot_all_elements",
            "C_KL_new_formula_all_elements",
            "D_KL_progressive_element_order_refresh",
            "E_KL_product_local_old_current_mix",
            "F_KL_electron_old_density_lifecycle",
            "G_best_KL_owned_old_new_element_density_lifecycle_timing",
            "H_FastChem_exact_v12_element_lifecycle_verifier_reference_only",
        ],
        "reference_only_verifier": {
            "candidate": "H_FastChem_exact_v12_element_lifecycle_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "owner_status": (
            "KL-owned default-off old/new element-density lifecycle timing "
            "carrier; diagnostic only"
        ),
    }
    fastchem_m162_exact_mutable_element_lifecycle_replay_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "KL-owned replay of mutable Element::number_density updates inside "
            "GasPhase::calculate before calculateMoleculeDensities product terms "
            "are consumed."
        ),
        "candidate_names": [
            "A_M161_static_baseline_no_replay",
            "B_KL_replay_disabled_element_solver_trace",
            "C_KL_replay_regular_branch_element_solver",
            "D_KL_replay_intertSol_only_element_solver",
            "E_KL_replay_scaled_old_electron",
            "F_KL_replay_immediate_electron_refresh",
            "G_best_KL_owned_exact_mutable_element_lifecycle_replay",
            "H_FastChem_exact_v12_mutable_lifecycle_verifier_reference_only",
        ],
        "reference_only_verifier": {
            "candidate": "H_FastChem_exact_v12_mutable_lifecycle_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "owner_status": (
            "KL-owned default-off exact mutable element lifecycle replay carrier; "
            "diagnostic only"
        ),
    }
    fastchem_m163_replay_state_initialization_gauge_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "KL-owned initial species-density vector and molecule-density gauge "
            "feeding GasPhase::calculate mutable replay before "
            "calculateMoleculeDensities product-term assembly."
        ),
        "candidate_names": [
            "A_M162_disabled_replay_baseline_gas_species",
            "B_KL_nk_sum_normalized_gas_density_initial_species",
            "C_KL_nk_sum_normalized_ntot_density_initial_species",
            "D_KL_nk_sum_normalized_count_density_initial_species",
            "E_KL_atomic_prefix_with_floor_tail_initial_species",
            "F_KL_b_element_prefix_with_current_tail_initial_species",
            "G_best_KL_owned_replay_state_initialization_gauge",
            "H_FastChem_exact_v12_replay_state_verifier_reference_only",
        ],
        "reference_only_verifier": {
            "candidate": "H_FastChem_exact_v12_replay_state_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "execution_control": {
            "exploratory_trace_record_cap": 4000,
            "expected_campaign_timeout_seconds": 300,
        },
        "owner_status": (
            "KL-owned default-off replay state initialization/gauge carrier; "
            "diagnostic only"
        ),
    }
    fastchem_m164_molecule_mass_action_gauge_correction_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "KL-owned molecule mass-action constant and density-domain gauge "
            "correction consumed by GasPhase::calculateMoleculeDensities during "
            "GasPhase::calculate replay."
        ),
        "candidate_names": [
            "A_M163_pressure_scaled_hvector_baseline",
            "B_KL_raw_negative_hvector_no_pressure_correction",
            "C_KL_temperature_density_gauge_correction",
            "D_KL_gas_number_density_gauge_correction",
            "E_KL_inverse_gas_number_density_gauge_correction",
            "F_KL_pressure_bar_plus_density_domain_correction",
            "G_best_KL_owned_molecule_mass_action_gauge_correction",
            "H_FastChem_exact_v12_mass_action_verifier_reference_only",
        ],
        "reference_only_verifier": {
            "candidate": "H_FastChem_exact_v12_mass_action_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "execution_control": {
            "exploratory_trace_record_cap": 4000,
            "expected_campaign_timeout_seconds": 300,
        },
        "owner_status": (
            "KL-owned default-off molecule mass-action gauge correction carrier; "
            "diagnostic only"
        ),
    }
    fastchem_m165_per_molecule_thermochemical_mass_action_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "KL/file-backed per-molecule logK(T) plus FastChem density-domain "
            "bridge used as the molecule mass-action constant inside "
            "GasPhase::calculateMoleculeDensities replay."
        ),
        "candidate_names": [
            "A_M164_pressure_scaled_hvector_baseline",
            "B_KL_hvector_density_over_pressure_bridge",
            "C_file_backed_logK_density_over_pressure_bridge",
            "D_file_backed_logK_pressure_over_density_bridge",
            "E_file_backed_logK_no_density_bridge",
            "F_best_KL_owned_per_molecule_thermochemical_mass_action",
            "G_FastChem_exact_v12_mass_action_verifier_reference_only",
        ],
        "reference_only_verifier": {
            "candidate": "G_FastChem_exact_v12_mass_action_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "execution_control": {
            "exploratory_trace_record_cap": 4000,
            "expected_campaign_timeout_seconds": 300,
        },
        "owner_status": (
            "KL-owned default-off per-molecule thermochemical mass-action "
            "carrier; diagnostic only"
        ),
    }
    fastchem_m166_calculate_molecule_element_density_timing_replay_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "KL-owned element-number-density vector timing consumed by "
            "GasPhase::calculateMoleculeDensities after M165 mass-action "
            "constant closure."
        ),
        "candidate_names": [
            "A_M165_current_element_density_replay",
            "B_KL_old_density_element_input_replay",
            "C_KL_initial_species_element_input_replay",
            "D_KL_formula_gas_element_external_input_replay",
            "E_KL_max_current_formula_external_input_replay",
            "F_KL_b_total_density_external_input_replay",
            "G_KL_old_electron_current_elements_replay",
            "H_best_KL_owned_calculateMoleculeDensities_element_density_timing",
            "I_FastChem_exact_v12_element_density_input_verifier_reference_only",
        ],
        "reference_only_verifier": {
            "candidate": "I_FastChem_exact_v12_element_density_input_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "execution_control": {
            "exploratory_trace_record_cap": 4000,
            "expected_campaign_timeout_seconds": 300,
        },
        "owner_status": (
            "KL-owned default-off calculateMoleculeDensities element-density "
            "timing replay carrier; diagnostic only"
        ),
    }
    fastchem_m168_calculate_element_densities_write_site_timing_carrier = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "available": False,
        "target_fastchem_source": (
            "KL-owned calculateElementDensities write-site timing for the "
            "Element::number_density vector consumed immediately inside "
            "GasPhase::calculateMoleculeDensities product-loop log terms."
        ),
        "candidate_names": [
            "A_M167_current_element_density_baseline",
            "B_KL_regular_calculateElementDensities_write_site_replay",
            "C_KL_solver_dispatched_n_major_write_site_replay",
            "D_KL_old_molecule_n_major_write_site_replay",
            "E_KL_immediate_electron_refresh_write_site_replay",
            "F_KL_old_electron_write_site_replay",
            "G_KL_molecule_list_coefficient_branch_gauge",
            "H_KL_preserved_phi_coefficient_gauge",
            "M170_A_dispatch_fixed_baseline_current_element_density",
            "M170_B_dispatch_fixed_normalized_b_abundance_phi",
            "M170_C_dispatch_fixed_gas_element_abundance_phi",
            "M172_A_M170_internal_domain_baseline",
            "M172_B_cgs_coefficient_density_domain",
            "M172_C_cgs_coefficient_density_domain_gas_element_phi",
            "M173_A_M172_internal_domain_baseline",
            "M173_B_native_overflow_raw_quad_internal_domain",
            "M173_C_native_overflow_raw_quad_cgs_domain",
            "M173_D_native_overflow_raw_quad_cgs_gas_element_phi",
            "M174_A_M173_internal_dispatch_baseline",
            "M174_B_solver_order_molecule_list_abundance_gate",
            "M174_C_scaling_factor_molecule_list_abundance_gate",
            "M174_D_coefficient_abundance_gate",
            "M174_E_fastchem_exact_dispatch_internal_domain",
            "M174_F_fastchem_exact_dispatch_cgs_domain",
            "M177_A_M174_solver_order_gate_baseline",
            "M177_B_fastchem_zero_element_checkN",
            "M177_C_fastchem_zero_checkN_cgs_domain",
            "M178_A_M177_zero_checkN_baseline",
            "M178_B_fastchem_newton_policy",
            "M178_C_fastchem_newton_alternative_bisection",
            "M178_D_fastchem_newton_alternative_bisection_cgs",
            "G_best_KL_owned_calculateElementDensities_write_site_timing",
            "H_FastChem_exact_v16_write_site_verifier_reference_only",
            "I_FastChem_exact_v16_write_site_verifier_reference_only",
        ],
        "reference_only_verifier": {
            "candidate": "I_FastChem_exact_v16_write_site_verifier_reference_only",
            "KL_owned": False,
            "reference_only": True,
            "trace_values_used_as_KL_constructor_inputs": False,
        },
        "execution_control": {
            "exploratory_trace_record_cap": 4000,
            "expected_campaign_timeout_seconds": 300,
        },
        "owner_status": (
            "KL-owned default-off calculateElementDensities write-site timing "
            "replay carrier; diagnostic only"
        ),
    }
    if element_labels_reduced_order is not None:
        labels_host = [str(label) for label in element_labels_reduced_order]
        b_host = np.asarray(jax.device_get(b), dtype=np.float64)
        formula_cond_host = np.asarray(
            jax.device_get(formula_matrix_cond),
            dtype=np.float64,
        )
        mk_host = np.asarray(jax.device_get(mk), dtype=np.float64)
        old_state_cond_cgs_host = np.asarray(
            jax.device_get(old_state_condensate_density_cgs_candidate_vector),
            dtype=np.float64,
        )
        total_density_cgs_host = float(
            jax.device_get(gas_species_total_element_density_cgs_candidate)
        )
        species_density_host = np.asarray(
            jax.device_get(gas_species_number_density_cgs_candidate_vector),
            dtype=np.float64,
        )
        element_density_host = np.asarray(
            jax.device_get(gas_species_element_density_cgs_candidate_vector),
            dtype=np.float64,
        )

        def _normalize_host(values: np.ndarray) -> np.ndarray:
            arr = np.asarray(values, dtype=np.float64)
            arr = np.where(np.isfinite(arr), arr, 0.0)
            total = float(np.sum(arr))
            if total <= 0.0:
                return np.zeros_like(arr)
            return arr / total

        epsilon_normalized_b_host = _normalize_host(b_host)
        file_epsilon_host = (
            None
            if fastchem_file_element_epsilon_vector is None
            else np.asarray(
                jax.device_get(fastchem_file_element_epsilon_vector),
                dtype=np.float64,
            )
        )
        atomic_prefix_host = species_density_host[: len(labels_host)]
        atomic_prefix_abundance_host = _normalize_host(atomic_prefix_host)
        total_element_abundance_host = _normalize_host(element_density_host)

        def _phi_from_cond_density(
            epsilon: np.ndarray,
            cond_density: np.ndarray,
            *,
            total_density: Optional[float],
            label: str,
        ) -> dict[str, Any]:
            eps = np.asarray(epsilon, dtype=np.float64)
            cond = np.asarray(cond_density, dtype=np.float64)
            denom = eps * float(total_density) if total_density is not None else eps
            degree = np.divide(
                cond,
                denom,
                out=np.zeros_like(cond),
                where=denom > 0.0,
            )
            degree = np.where(eps == 0.0, 0.0, np.minimum(degree, 1.0))
            phi_raw = np.where(np.isfinite(eps * (1.0 - degree)), eps * (1.0 - degree), 0.0)
            phi = _normalize_host(phi_raw)
            return {
                "candidate": label,
                "source_stage": (
                    "KL-owned reconstruction of FastChem phi from epsilon and "
                    "condensate element density"
                ),
                "KL_owned": True,
                "can_emit_default_off": True,
                "normalization": "sum-normalized phi_raw",
                "degree_of_condensation_vector": degree.tolist(),
                "phi_raw_vector": phi_raw.tolist(),
                "vector": phi,
            }

        def _vector_candidate(
            *,
            label: str,
            source_stage: str,
            values: np.ndarray,
            normalization: str,
        ) -> dict[str, Any]:
            return {
                "candidate": label,
                "source_stage": source_stage,
                "KL_owned": True,
                "can_emit_default_off": True,
                "normalization": normalization,
                "vector": _normalize_host(values),
            }

        cond_density_dimensionless = formula_cond_host @ mk_host
        cond_density_old_state_cgs = formula_cond_host @ old_state_cond_cgs_host
        maxdensity_cond_cgs = np.asarray(
            jax.device_get(maxDensity_value_vector),
            dtype=np.float64,
        )
        maxdensity_cond_element_cgs = formula_cond_host @ maxdensity_cond_cgs
        fixed_by_condensation_flags_host = b_host <= 0.0
        fixed_adjusted_epsilon_host = np.where(
            fixed_by_condensation_flags_host,
            0.0,
            epsilon_normalized_b_host,
        )
        degree_adjusted_phi_raw_host = np.maximum(
            epsilon_normalized_b_host
            - cond_density_old_state_cgs / max(total_density_cgs_host, 1.0e-300),
            0.0,
        )
        pre_correct_values_phi_host = np.maximum(
            atomic_prefix_abundance_host
            - cond_density_dimensionless * np.maximum(epsilon_normalized_b_host, 0.0),
            0.0,
        )
        post_correct_values_phi_host = np.maximum(
            total_element_abundance_host
            - cond_density_old_state_cgs / max(total_density_cgs_host, 1.0e-300),
            0.0,
        )
        candidates = [
            {
                "candidate": "A_M133_best_baseline_atomic_prefix_gas_phase_exit",
                "source_stage": (
                    "M133 best baseline: KL gas species atomic-prefix density / sum"
                ),
                "KL_owned": True,
                "can_emit_default_off": True,
                "normalization": "sum-normalized atomic-prefix gas density",
                "vector": atomic_prefix_abundance_host,
            },
            {
                "candidate": "B_KL_current_phi_vector",
                "source_stage": "KL reduced b / sum(b) at exact bundle timing",
                "KL_owned": True,
                "can_emit_default_off": True,
                "normalization": "sum-normalized reduced b",
                "vector": epsilon_normalized_b_host,
            },
            {
                "candidate": "C_KL_post_condensation_phi_vector",
                "source_stage": "formula_matrix @ gas species density cgs candidate",
                "KL_owned": True,
                "can_emit_default_off": True,
                "normalization": "sum-normalized gas element density",
                "vector": total_element_abundance_host,
            },
            _vector_candidate(
                label="D_KL_degree_of_condensation_adjusted_phi",
                source_stage=(
                    "epsilon_normalized_b - old-state condensate element burden / "
                    "KL total element density"
                ),
                values=degree_adjusted_phi_raw_host,
                normalization="clip nonnegative then sum-normalize",
            ),
            _vector_candidate(
                label="E_KL_fixed_by_condensation_adjusted_phi",
                source_stage=(
                    "epsilon_normalized_b with b<=0 fixed-by-condensation rows zeroed"
                ),
                values=fixed_adjusted_epsilon_host,
                normalization="fixed rows zeroed then sum-normalize",
            ),
            _phi_from_cond_density(
                epsilon_normalized_b_host,
                cond_density_dimensionless,
                total_density=None,
                label="F_KL_condensate_burden_normalized_phi_dimensionless",
            ),
            _phi_from_cond_density(
                epsilon_normalized_b_host,
                cond_density_old_state_cgs,
                total_density=total_density_cgs_host,
                label="F_KL_condensate_burden_normalized_phi_old_state_cgs",
            ),
            _phi_from_cond_density(
                epsilon_normalized_b_host,
                maxdensity_cond_element_cgs,
                total_density=total_density_cgs_host,
                label="G_KL_file_backed_maxDensity_epsilon_budget_phi",
            ),
            _vector_candidate(
                label="H_KL_pre_correctValues_phi",
                source_stage=(
                    "atomic-prefix abundance minus dimensionless condensate burden"
                ),
                values=pre_correct_values_phi_host,
                normalization="clip nonnegative then sum-normalize",
            ),
            _vector_candidate(
                label="I_KL_post_correctValues_phi",
                source_stage=(
                    "post-update element-density abundance minus old-state cgs "
                    "condensate burden"
                ),
                values=post_correct_values_phi_host,
                normalization="clip nonnegative then sum-normalize",
            ),
            _vector_candidate(
                label="J_KL_gas_phase_reInitialise_timing_phi",
                source_stage=(
                    "KL gas_phase.reInitialise timing candidate: atomic-prefix "
                    "abundance before setMoleculeAbundances"
                ),
                values=atomic_prefix_abundance_host,
                normalization="sum-normalized atomic-prefix gas density",
            ),
            _phi_from_cond_density(
                atomic_prefix_abundance_host,
                cond_density_old_state_cgs,
                total_density=total_density_cgs_host,
                label="K_best_KL_owned_FastChem_style_phi_element_abundance_vector",
            ),
            {
                "candidate": "L_FastChem_v13_v14_exact_verifier_reference_only",
                "source_stage": "v13/v14 reference target link only",
                "KL_owned": False,
                "can_emit_default_off": True,
                "normalization": "not emitted as a constructor vector",
                "vector": None,
                "reference_only": True,
            },
        ]
        if file_epsilon_host is not None:
            candidates.insert(
                3,
                _phi_from_cond_density(
                    file_epsilon_host,
                    cond_density_old_state_cgs,
                    total_density=total_density_cgs_host,
                    label="G_file_epsilon_phi_from_old_state_cgs",
                ),
            )
        else:
            candidates.insert(
                3,
                {
                    "candidate": "G_file_epsilon_phi_from_old_state_cgs",
                    "source_stage": "packaged FastChem abundance epsilon with KL old-state condensate cgs density",
                    "KL_owned": True,
                    "can_emit_default_off": False,
                    "normalization": "unavailable",
                    "vector": None,
                    "missing_exact_inputs": [
                        "fastchem_file_element_epsilon_vector"
                    ],
                },
            )
        fastchem_phi_element_abundance_timing_carrier = {
            **fastchem_phi_element_abundance_timing_carrier,
            "element_labels": labels_host,
            "element_order": list(range(len(labels_host))),
            "total_element_density_cgs_candidate": total_density_cgs_host,
            "condensate_element_density_dimensionless_vector": (
                cond_density_dimensionless.tolist()
            ),
            "condensate_element_density_old_state_cgs_vector": (
                cond_density_old_state_cgs.tolist()
            ),
            "source_stage_candidates": [
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "vector"
                }
                for candidate in candidates
            ],
            "candidate_vectors": [
                {
                    **{
                        key: value
                        for key, value in candidate.items()
                        if key != "vector"
                    },
                    "vector": (
                        None
                        if candidate.get("vector") is None
                        else np.asarray(candidate["vector"], dtype=np.float64).tolist()
                    ),
                }
                for candidate in candidates
            ],
            "best_KL_owned_candidate": (
                "pending boundary comparison in M133 campaign artifact"
            ),
            "missing_exact_inputs": [],
        }
        burden_total_candidates = [
            (
                "A_M135_baseline_atomic_prefix",
                atomic_prefix_abundance_host,
                cond_density_old_state_cgs,
                total_density_cgs_host,
                "M135 baseline atomic-prefix abundance with old-state condensate burden",
            ),
            (
                "B_v15_informed_KL_owned_density_cond_formula",
                epsilon_normalized_b_host,
                cond_density_old_state_cgs,
                total_density_cgs_host,
                "formula_matrix_cond @ old_state_condensate_density_cgs_candidate_vector",
            ),
            (
                "C_v15_informed_KL_owned_total_element_density_formula",
                epsilon_normalized_b_host,
                cond_density_old_state_cgs,
                float(
                    jax.device_get(
                        gas_species_inventory_renormalized_total_element_density_cgs_candidate
                    )
                ),
                "inventory-renormalized total element density denominator",
            ),
            (
                "D_v15_informed_KL_owned_degree_of_condensation_formula",
                epsilon_normalized_b_host,
                cond_density_old_state_cgs,
                total_density_cgs_host,
                "density_cond / (epsilon * total_element_density), clipped to one",
            ),
            (
                "E_v15_informed_KL_owned_phi_normalization_formula",
                epsilon_normalized_b_host,
                cond_density_old_state_cgs,
                total_density_cgs_host,
                "epsilon * (1 - degree_of_condensation), sum-normalized",
            ),
            (
                "F_best_KL_owned_calcDegreeOfCondensation_burden_reconstruction",
                atomic_prefix_abundance_host,
                cond_density_old_state_cgs,
                total_density_cgs_host,
                "best KL-owned pre-v15 burden reconstruction branch",
            ),
        ]
        m137_slot_density_candidates = [
            (
                "M137_density_cond_old_state_all_slots",
                epsilon_normalized_b_host,
                cond_density_old_state_cgs,
                total_density_cgs_host,
                "formula_cond @ old_state_condensate_density_cgs_candidate_vector",
            ),
            (
                "M137_density_cond_raw_mk_gas_all_slots",
                epsilon_normalized_b_host,
                formula_cond_host @ (mk_host * gas_number_density_cgs),
                total_density_cgs_host,
                "formula_cond @ (mk * gas_number_density_cgs)",
            ),
            (
                "M137_density_cond_raw_mk_gas_retained_slots",
                epsilon_normalized_b_host,
                formula_cond_host
                @ (
                    mk_host
                    * gas_number_density_cgs
                    * np.asarray(jax.device_get(retained_slot_flags), dtype=np.float64)
                ),
                total_density_cgs_host,
                "formula_cond @ (mk * gas_number_density_cgs * retained_slot_flags)",
            ),
            (
                "M137_density_cond_old_state_retained_slots",
                epsilon_normalized_b_host,
                formula_cond_host
                @ (
                    old_state_cond_cgs_host
                    * np.asarray(jax.device_get(retained_slot_flags), dtype=np.float64)
                ),
                total_density_cgs_host,
                "formula_cond @ (old_state_cgs * retained_slot_flags)",
            ),
            (
                "M137_density_cond_maxDensity_floor_slots",
                epsilon_normalized_b_host,
                formula_cond_host
                @ (
                    maxdensity_cond_cgs
                    * np.asarray(jax.device_get(maxDensity_slot_flags), dtype=np.float64)
                ),
                total_density_cgs_host,
                "formula_cond @ (maxDensity_value_vector * maxDensity_slot_flags)",
            ),
            (
                "M137_density_cond_old_state_capped_slots",
                epsilon_normalized_b_host,
                formula_cond_host
                @ (
                    old_state_cond_cgs_host
                    * np.asarray(jax.device_get(capped_slot_flags), dtype=np.float64)
                ),
                total_density_cgs_host,
                "formula_cond @ (old_state_cgs * capped_slot_flags)",
            ),
        ]
        burden_total_candidates.extend(m137_slot_density_candidates)
        v15_candidates = []
        for (
            candidate_name,
            epsilon_values,
            density_cond_values,
            total_density_value,
            source_stage,
        ) in burden_total_candidates:
            eps = np.asarray(epsilon_values, dtype=np.float64)
            density_cond = np.asarray(density_cond_values, dtype=np.float64)
            total_density = float(total_density_value)
            denom = eps * max(total_density, 1.0e-300)
            degree = np.divide(
                density_cond,
                denom,
                out=np.zeros_like(density_cond),
                where=denom > 0.0,
            )
            degree = np.where(eps == 0.0, 0.0, np.minimum(degree, 1.0))
            phi_raw = np.maximum(eps * (1.0 - degree), 0.0)
            v15_candidates.append(
                {
                    "candidate": candidate_name,
                    "source_stage": source_stage,
                    "KL_owned": True,
                    "can_emit_default_off": True,
                    "normalization": "FastChem-style clipped degree then sum-normalized phi",
                    "epsilon_vector": eps.tolist(),
                    "density_cond_vector": density_cond.tolist(),
                    "total_element_density": total_density,
                    "degree_of_condensation_vector": degree.tolist(),
                    "phi_raw_vector": phi_raw.tolist(),
                    "vector": _normalize_host(phi_raw),
                }
            )
        v15_candidates.append(
            {
                "candidate": "G_FastChem_exact_v15_verifier_reference_only",
                "source_stage": "v15 trace verifier link only",
                "KL_owned": False,
                "can_emit_default_off": True,
                "normalization": "not emitted as a constructor vector",
                "vector": None,
                "reference_only": True,
                "trace_values_used_as_KL_constructor_inputs": False,
            }
        )
        fastchem_v15_calc_degree_condensation_burden_reconstruction_carrier = {
            **fastchem_v15_calc_degree_condensation_burden_reconstruction_carrier,
            "element_labels": labels_host,
            "element_order": list(range(len(labels_host))),
            "source_stage_candidates": [
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "vector"
                }
                for candidate in v15_candidates
            ],
            "candidate_vectors": [
                {
                    **{
                        key: value
                        for key, value in candidate.items()
                        if key != "vector"
                    },
                    "vector": (
                        None
                        if candidate.get("vector") is None
                        else np.asarray(candidate["vector"], dtype=np.float64).tolist()
                    ),
                }
                for candidate in v15_candidates
            ],
            "best_KL_owned_candidate": (
                "pending boundary comparison in M136 campaign artifact"
            ),
        }
        kl_log_activity_host = None
        if atomic_prefix_available:
            kl_log_activity_host = np.asarray(
                jax.device_get(
                    compute_kl_condensate_log_activity(
                        ln_nk[:n_element_rows],
                        formula_matrix_cond,
                        hvector_cond,
                        temperature=float(jax.device_get(temperature)),
                        apply_density_gauge_bridge=False,
                    )
                ),
                dtype=np.float64,
            )
        mk_positive_mask = mk_host > 1.0e-12
        maxdensity_floor_mask = np.isfinite(maxdensity_cond_cgs) & (
            old_state_cond_cgs_host <= np.maximum(maxdensity_cond_cgs, 0.0) * 1.0e-6
        )
        activity_initial_mask = (
            np.zeros_like(mk_host, dtype=bool)
            if kl_log_activity_host is None
            else kl_log_activity_host >= 0.0
        )
        activity_prune_mask = (
            np.zeros_like(mk_host, dtype=bool)
            if kl_log_activity_host is None
            else kl_log_activity_host >= -0.01
        )

        def _support_mask_candidate(
            *,
            label: str,
            source_stage: str,
            mask: np.ndarray,
            KL_owned: bool = True,
            can_emit_default_off: bool = True,
            reference_only: bool = False,
            missing_exact_inputs: Optional[Sequence[str]] = None,
        ) -> dict[str, Any]:
            mask_host = np.asarray(mask, dtype=bool)
            selected = np.flatnonzero(mask_host)[:24]
            return {
                "candidate": label,
                "source_stage": source_stage,
                "KL_owned": KL_owned,
                "can_emit_default_off": can_emit_default_off,
                "reference_only": reference_only,
                "mask_true_count": int(np.count_nonzero(mask_host)),
                "selected_slot_indices_sample": selected.tolist(),
                "selected_slot_labels_sample": [
                    str(condensate_labels_jac_order[index]) for index in selected
                ]
                if condensate_labels_jac_order is not None
                else [],
                "mask": mask_host.astype(np.int32),
                "missing_exact_inputs": list(missing_exact_inputs or []),
            }

        m141_candidates = [
            _support_mask_candidate(
                label="A_M139_best_initial_selector_top14_activity_proxy_floor_maxDensity",
                source_stage=(
                    "M139 KL-owned initial selector proxy; does not include "
                    "FastChem post-solver support pruning"
                ),
                mask=activity_initial_mask & maxdensity_floor_mask,
                can_emit_default_off=kl_log_activity_host is not None,
                missing_exact_inputs=[]
                if kl_log_activity_host is not None
                else ["KL_atomic_prefix_condensate_log_activity"],
            ),
            _support_mask_candidate(
                label="B_KL_current_positive_condensate_support",
                source_stage="KL reduced-solver entry exp(ln_mk) > 1e-12",
                mask=mk_positive_mask,
            ),
            _support_mask_candidate(
                label="C_KL_initial_activity_ge_zero_support",
                source_stage=(
                    "compute_kl_condensate_log_activity(atomic_prefix_u) >= 0"
                ),
                mask=activity_initial_mask,
                can_emit_default_off=kl_log_activity_host is not None,
                missing_exact_inputs=[]
                if kl_log_activity_host is not None
                else ["KL_atomic_prefix_condensate_log_activity"],
            ),
            _support_mask_candidate(
                label="D_KL_post_solver_prune_surrogate_activity_ge_minus_0p01",
                source_stage=(
                    "same KL-owned atomic-prefix activity with the FastChem "
                    "post-solver removal threshold; missing exact post-solver "
                    "activity timing until condensed-phase correctValues state "
                    "is reconstructed"
                ),
                mask=activity_prune_mask,
                can_emit_default_off=kl_log_activity_host is not None,
                missing_exact_inputs=[]
                if kl_log_activity_host is not None
                else ["KL_post_solver_condensate_log_activity"],
            ),
            _support_mask_candidate(
                label="E_KL_floor_maxDensity_initial_activity_support",
                source_stage=(
                    "KL maxDensity/floor slot mask intersected with "
                    "atomic-prefix activity >= 0"
                ),
                mask=maxdensity_floor_mask & activity_initial_mask,
                can_emit_default_off=kl_log_activity_host is not None,
                missing_exact_inputs=[]
                if kl_log_activity_host is not None
                else ["KL_atomic_prefix_condensate_log_activity"],
            ),
            _support_mask_candidate(
                label="F_KL_floor_maxDensity_post_solver_prune_surrogate",
                source_stage=(
                    "KL maxDensity/floor slot mask intersected with "
                    "atomic-prefix activity >= -0.01; this is a surrogate for "
                    "the post-solver boundary, not a reference fit"
                ),
                mask=maxdensity_floor_mask & activity_prune_mask,
                can_emit_default_off=kl_log_activity_host is not None,
                missing_exact_inputs=[]
                if kl_log_activity_host is not None
                else ["KL_post_solver_condensate_log_activity"],
            ),
            _support_mask_candidate(
                label="G_FastChem_post_final_removal_verifier_reference_only",
                source_stage="v15 post_final_removal_condensate_state link only",
                mask=np.zeros_like(mk_host, dtype=bool),
                KL_owned=False,
                reference_only=True,
            ),
        ]
        fastchem_m141_post_solver_support_pruning_surrogate_carrier = {
            **fastchem_m141_post_solver_support_pruning_surrogate_carrier,
            "available": kl_log_activity_host is not None,
            "condensate_slot_count": int(mk_host.shape[0]),
            "thresholds": {
                "selectActiveCondensates_log_activity": 0.0,
                "post_solver_removal_log_activity": -0.01,
                "current_positive_condensate_density": 1.0e-12,
            },
            "KL_atomic_prefix_log_activity_available": kl_log_activity_host
            is not None,
            "KL_atomic_prefix_log_activity_vector": None
            if kl_log_activity_host is None
            else kl_log_activity_host.tolist(),
            "source_stage_candidates": [
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "mask"
                }
                for candidate in m141_candidates
            ],
            "candidate_masks": [
                {
                    **{
                        key: value
                        for key, value in candidate.items()
                        if key != "mask"
                    },
                    "mask": np.asarray(candidate["mask"], dtype=np.int32).tolist(),
                }
                for candidate in m141_candidates
            ],
            "best_KL_owned_candidate": (
                "pending boundary comparison in M141 campaign artifact"
            ),
        }
        hvector_cond_host = np.asarray(
            jax.device_get(hvector_cond),
            dtype=np.float64,
        )

        def _condensate_log_activity_from_element_density(
            element_density: np.ndarray,
        ) -> np.ndarray:
            density = np.asarray(element_density, dtype=np.float64)
            log_density = np.log(np.maximum(density, 1.0e-300))
            return formula_cond_host.T @ log_density - hvector_cond_host

        gas_element_log_activity_host = (
            _condensate_log_activity_from_element_density(element_density_host)
        )
        atomic_prefix_log_activity_cgs_host = (
            _condensate_log_activity_from_element_density(atomic_prefix_host)
        )
        total_element_log_activity_cgs_host = (
            _condensate_log_activity_from_element_density(
                np.maximum(element_density_host, atomic_prefix_host)
            )
        )

        def _log_activity_closure_candidate(
            *,
            label: str,
            source_stage: str,
            log_activity: np.ndarray,
            KL_owned: bool = True,
            reference_only: bool = False,
        ) -> dict[str, Any]:
            log_host = np.asarray(log_activity, dtype=np.float64)
            mask_host = log_host >= -0.01
            selected = np.flatnonzero(mask_host)[:24]
            return {
                "candidate": label,
                "source_stage": source_stage,
                "KL_owned": KL_owned,
                "can_emit_default_off": True,
                "reference_only": reference_only,
                "threshold": -0.01,
                "mask_true_count": int(np.count_nonzero(mask_host)),
                "selected_slot_indices_sample": selected.tolist(),
                "selected_slot_labels_sample": [
                    str(condensate_labels_jac_order[index]) for index in selected
                ]
                if condensate_labels_jac_order is not None
                else [],
                "log_activity_vector": log_host.tolist(),
                "mask": mask_host.astype(np.int32),
                "missing_exact_inputs": [],
            }

        m142_candidates = [
            _log_activity_closure_candidate(
                label="A_M141_atomic_prefix_activity_closure",
                source_stage=(
                    "compute_kl_condensate_log_activity(ln_nk atomic-prefix state)"
                ),
                log_activity=kl_log_activity_host
                if kl_log_activity_host is not None
                else np.full_like(mk_host, -np.inf),
            ),
            _log_activity_closure_candidate(
                label="B_KL_gas_species_element_density_cgs_log_activity",
                source_stage=(
                    "formula_cond.T @ log(KL gas-species element density cgs) "
                    "- hvector_cond"
                ),
                log_activity=gas_element_log_activity_host,
            ),
            _log_activity_closure_candidate(
                label="C_KL_atomic_prefix_element_density_cgs_log_activity",
                source_stage=(
                    "formula_cond.T @ log(KL atomic-prefix element density cgs) "
                    "- hvector_cond"
                ),
                log_activity=atomic_prefix_log_activity_cgs_host,
            ),
            _log_activity_closure_candidate(
                label="D_KL_max_gas_atomic_element_density_cgs_log_activity",
                source_stage=(
                    "formula_cond.T @ log(max(gas-species, atomic-prefix) "
                    "element density cgs) - hvector_cond"
                ),
                log_activity=total_element_log_activity_cgs_host,
            ),
            _log_activity_closure_candidate(
                label="F_FastChem_iter1_old_log_activity_closure_verifier_reference_only",
                source_stage="v15 iter1_old_log_activity_closure link only",
                log_activity=np.full_like(mk_host, -np.inf),
                KL_owned=False,
                reference_only=True,
            ),
        ]
        fastchem_m142_old_log_activity_closure_reconstruction_carrier = {
            **fastchem_m142_old_log_activity_closure_reconstruction_carrier,
            "available": True,
            "condensate_slot_count": int(mk_host.shape[0]),
            "source_stage_candidates": [
                {
                    key: value
                    for key, value in candidate.items()
                    if key not in {"mask", "log_activity_vector"}
                }
                for candidate in m142_candidates
            ],
            "candidate_masks": [
                {
                    **{
                        key: value
                        for key, value in candidate.items()
                        if key != "mask"
                    },
                    "mask": np.asarray(candidate["mask"], dtype=np.int32).tolist(),
                }
                for candidate in m142_candidates
            ],
            "best_KL_owned_candidate": (
                "pending boundary comparison in M142 campaign artifact"
            ),
        }
    condensate_lifecycle_cap_metadata = {
        "diagnostic_only": True,
        "default_off": True,
        "constructor_input": False,
        "reference_trace_input": False,
        "FastChem_trace_values_used_as_inputs": False,
        "used_as_KL_constructor_input": False,
        "lifecycle_timing": (
            "KL exact input bundle at reduced-solver entry before _update_all_with_metrics"
        ),
        "old_state_source": "mk = exp(ln_mk) entering the reduced solver",
        "old_state_cgs_candidate_formula": (
            "where(newly_active_or_floor, maxDensity_value_vector, mk * gas_number_density_cgs)"
        ),
        "retained_slot_rule": "mk > 1e-12",
        "newly_active_slot_rule": "floor slot at reduced-solver entry; FastChem seeds zero newly-active condensates to maxDensity before copying cond_densities_old",
        "capped_slot_rule": "isclose(mk, budget_cap, rtol=1e-10)",
        "floor_slot_rule": "mk <= 1e-12",
        "maxDensity_status": "KL diagnostic candidate materialized from budget_cap * gas_species_total_element_density_cgs_candidate",
        "maxDensity_formula": "budget_cap * gas_species_total_element_density_cgs_candidate",
        "source_provenance": call_site_provenance,
    }
    payload = {
        "case_key": str(case_key),
        "newton_iter": int(newton_iter),
        "same_iteration_newton_iter": int(newton_iter),
        "ln_nk": _diagnostic_json_array(ln_nk),
        "ln_mk": _diagnostic_json_array(ln_mk),
        "ln_ntot": float(jax.device_get(ln_ntot)),
        "nk": _diagnostic_json_array(nk),
        "mk": _diagnostic_json_array(mk),
        "jacobian_owner_condensate_density_vector": _diagnostic_json_array(mk),
        "jacobian_owner_condensate_density_ln_vector": _diagnostic_json_array(ln_mk),
        "jacobian_owner_condensate_density_semantics": (
            "KL reduced-system Jacobian construction receives exp(ln_mk) as the "
            "condensate density column factor; no separate old-state condensate "
            "number-density vector is owned by this call-site bundle."
        ),
        "jacobian_owner_condensate_density_same_as_mk": True,
        "condensate_budget_cap_vector": _diagnostic_json_array(
            condensate_budget_cap_vector
        ),
        "gas_species_number_density_cgs_candidate_vector": _diagnostic_json_array(
            gas_species_number_density_cgs_candidate_vector
        ),
        "gas_species_element_density_cgs_candidate_vector": _diagnostic_json_array(
            gas_species_element_density_cgs_candidate_vector
        ),
        "gas_species_total_element_density_cgs_candidate": float(
            jax.device_get(gas_species_total_element_density_cgs_candidate)
        ),
        "gas_species_particle_count_denominator": float(
            jax.device_get(gas_species_particle_count_denominator)
        ),
        "gas_species_ntot_denominator": float(
            jax.device_get(gas_species_ntot_denominator)
        ),
        "gas_species_reduced_element_inventory_sum": float(
            jax.device_get(gas_species_reduced_element_inventory_sum)
        ),
        "gas_species_count_density_cgs_candidate": float(
            jax.device_get(gas_species_count_density_cgs_candidate)
        ),
        "gas_species_ntot_density_cgs_candidate": float(
            jax.device_get(gas_species_ntot_density_cgs_candidate)
        ),
        "gas_species_inventory_renormalized_total_element_density_cgs_candidate": (
            float(
                jax.device_get(
                    gas_species_inventory_renormalized_total_element_density_cgs_candidate
                )
            )
        ),
        "gas_species_total_element_density_metadata": (
            gas_species_total_element_density_metadata
        ),
        "fastchem_post_initial_gas_total_density_carrier": (
            fastchem_post_initial_gas_total_density_carrier
        ),
        "fastchem_post_initial_gas_species_density_replay_carrier": (
            fastchem_post_initial_gas_species_density_replay_carrier
        ),
        "fastchem_gas_phase_calculate_lifecycle_replay_contract_carrier": (
            fastchem_gas_phase_calculate_lifecycle_replay_contract_carrier
        ),
        "gas_phase_calculate_runtime_timing_result_carrier": (
            gas_phase_calculate_runtime_timing_result_carrier
        ),
        "fastchem_post_initial_gas_total_element_density_replay_carrier": (
            fastchem_post_initial_gas_total_element_density_replay_carrier
        ),
        "reduced_assembly_owner_density_denominator_carrier": (
            reduced_assembly_owner_density_denominator_carrier
        ),
        "row_scaled_jec_owner_scalar_verifier": (
            row_scaled_jec_owner_scalar_verifier
        ),
        "row_scaled_jec_owner_scalar_correction_carrier": (
            row_scaled_jec_owner_scalar_correction_carrier
        ),
        "reduced_system_condensate_coupling_source_audit": (
            reduced_system_condensate_coupling_source_audit
        ),
        "fixed_bridge_budget_cap_scalarization_carrier": (
            fixed_bridge_budget_cap_scalarization_carrier
        ),
        "fastchem_style_maxdensity_seeding_total_density_carrier": (
            fastchem_style_maxdensity_seeding_total_density_carrier
        ),
        "fastchem_style_maxdensity_budget_cap_carrier": (
            fastchem_style_maxdensity_budget_cap_carrier
        ),
        "fastchem_style_element_epsilon_budget_cap_vector": _diagnostic_json_array(
            fastchem_style_element_epsilon_budget_cap_vector
        ),
        "fastchem_style_element_epsilon_budget_cap_carrier": (
            fastchem_style_element_epsilon_budget_cap_carrier
        ),
        "element_epsilon_from_normalized_b_vector": _diagnostic_json_array(
            element_epsilon_from_normalized_b_vector
        ),
        "normalized_b_maxdensity_budget_cap_vector": _diagnostic_json_array(
            normalized_b_maxdensity_budget_cap_vector
        ),
        "fastchem_style_normalized_b_element_epsilon_budget_cap_carrier": (
            fastchem_style_normalized_b_element_epsilon_budget_cap_carrier
        ),
        "fastchem_file_element_epsilon_vector": (
            None
            if fastchem_file_element_epsilon_vector is None
            else _diagnostic_json_array(fastchem_file_element_epsilon_vector)
        ),
        "fastchem_file_maxdensity_budget_cap_vector": (
            None
            if fastchem_file_maxdensity_budget_cap_vector is None
            else _diagnostic_json_array(fastchem_file_maxdensity_budget_cap_vector)
        ),
        "fastchem_file_element_epsilon_budget_cap_carrier": (
            fastchem_file_element_epsilon_budget_cap_carrier
        ),
        "element_abundance_create_molecule_lists_timing_carrier": (
            element_abundance_create_molecule_lists_timing_carrier
        ),
        "element_abundance_createMoleculeLists_timing_candidate_vector": (
            element_abundance_create_molecule_lists_timing_carrier.get(
                "element_abundance_createMoleculeLists_timing_candidate_vector"
            )
        ),
        "fastchem_phi_element_abundance_timing_carrier": (
            fastchem_phi_element_abundance_timing_carrier
        ),
        "fastchem_v15_calc_degree_condensation_burden_reconstruction_carrier": (
            fastchem_v15_calc_degree_condensation_burden_reconstruction_carrier
        ),
        "fastchem_m141_post_solver_support_pruning_surrogate_carrier": (
            fastchem_m141_post_solver_support_pruning_surrogate_carrier
        ),
        "fastchem_m142_old_log_activity_closure_reconstruction_carrier": (
            fastchem_m142_old_log_activity_closure_reconstruction_carrier
        ),
        "fastchem_m147_second_pass_36row_reduced_system_reconstruction_carrier": (
            fastchem_m147_second_pass_36row_reduced_system_reconstruction_carrier
        ),
        "fastchem_m148_second_pass_row_column_scaling_reconstruction_carrier": (
            fastchem_m148_second_pass_row_column_scaling_reconstruction_carrier
        ),
        "fastchem_m149_second_pass_jacobian_block_reconstruction_carrier": (
            fastchem_m149_second_pass_jacobian_block_reconstruction_carrier
        ),
        "fastchem_m150_second_pass_element_element_jacobian_reconstruction_carrier": (
            fastchem_m150_second_pass_element_element_jacobian_reconstruction_carrier
        ),
        "fastchem_m151_old_gas_molecule_density_outer_product_timing_carrier": (
            fastchem_m151_old_gas_molecule_density_outer_product_timing_carrier
        ),
        "fastchem_m152_old_full_element_density_gauge_reconstruction_carrier": (
            fastchem_m152_old_full_element_density_gauge_reconstruction_carrier
        ),
        "fastchem_m153_old_total_density_gauge_deep_reconstruction_carrier": (
            fastchem_m153_old_total_density_gauge_deep_reconstruction_carrier
        ),
        "fastchem_m154_total_element_density_aggregation_timing_carrier": (
            fastchem_m154_total_element_density_aggregation_timing_carrier
        ),
        "fastchem_m155_condensed_total_element_density_correction_carrier": (
            fastchem_m155_condensed_total_element_density_correction_carrier
        ),
        "fastchem_m156_retained_support_density_timing_carrier": (
            fastchem_m156_retained_support_density_timing_carrier
        ),
        "fastchem_m157_gas_phase_total_element_density_timing_carrier": (
            fastchem_m157_gas_phase_total_element_density_timing_carrier
        ),
        "fastchem_m158_post_gas_solve_molecule_density_timing_carrier": (
            fastchem_m158_post_gas_solve_molecule_density_timing_carrier
        ),
        "fastchem_m159_molecule_density_cache_checkN_timing_carrier": (
            fastchem_m159_molecule_density_cache_checkN_timing_carrier
        ),
        "fastchem_m160_calculate_molecule_element_input_timing_carrier": (
            fastchem_m160_calculate_molecule_element_input_timing_carrier
        ),
        "fastchem_m161_old_new_element_density_lifecycle_timing_carrier": (
            fastchem_m161_old_new_element_density_lifecycle_timing_carrier
        ),
        "fastchem_m162_exact_mutable_element_lifecycle_replay_carrier": (
            fastchem_m162_exact_mutable_element_lifecycle_replay_carrier
        ),
        "fastchem_m163_replay_state_initialization_gauge_carrier": (
            fastchem_m163_replay_state_initialization_gauge_carrier
        ),
        "fastchem_m164_molecule_mass_action_gauge_correction_carrier": (
            fastchem_m164_molecule_mass_action_gauge_correction_carrier
        ),
        "fastchem_m165_per_molecule_thermochemical_mass_action_carrier": (
            fastchem_m165_per_molecule_thermochemical_mass_action_carrier
        ),
        "fastchem_m166_calculate_molecule_element_density_timing_replay_carrier": (
            fastchem_m166_calculate_molecule_element_density_timing_replay_carrier
        ),
        "fastchem_m168_calculate_element_densities_write_site_timing_carrier": (
            fastchem_m168_calculate_element_densities_write_site_timing_carrier
        ),
        "fastchem_file_budget_maxdensity_owner_density_vector": (
            None
            if fastchem_file_budget_maxdensity_owner_density_vector is None
            else _diagnostic_json_array(
                fastchem_file_budget_maxdensity_owner_density_vector
            )
        ),
        "fastchem_file_budget_maxdensity_owner_density_carrier": (
            fastchem_file_budget_maxdensity_owner_density_carrier
        ),
        "kl_native_row_scaled_jec_block_candidate": (
            kl_native_row_scaled_jec_block_candidate
        ),
        "old_state_condensate_density_cgs_candidate_vector": _diagnostic_json_array(
            old_state_condensate_density_cgs_candidate_vector
        ),
        "retained_slot_flags": [
            bool(value) for value in np.asarray(jax.device_get(retained_slot_flags)).tolist()
        ],
        "newly_active_slot_flags": [
            bool(value) for value in np.asarray(jax.device_get(newly_active_slot_flags)).tolist()
        ],
        "capped_slot_flags": [
            bool(value) for value in np.asarray(jax.device_get(capped_slot_flags)).tolist()
        ],
        "floor_slot_flags": [
            bool(value) for value in np.asarray(jax.device_get(floor_slot_flags)).tolist()
        ],
        "maxDensity_slot_flags": [
            bool(value) for value in np.asarray(jax.device_get(maxDensity_slot_flags)).tolist()
        ],
        "maxDensity_value_vector": _diagnostic_json_array(maxDensity_value_vector),
        "condensate_lifecycle_cap_metadata": condensate_lifecycle_cap_metadata,
        "condensate_jec_owner_density_cgs_vector": _diagnostic_json_array(
            condensate_jec_owner_density_cgs_vector
        ),
        "condensate_jec_owner_density_cgs_metadata": {
            "diagnostic_only": True,
            "default_off": True,
            "constructor_input": False,
            "reference_trace_input": False,
            "formula": (
                "condensate_budget_cap_vector * "
                "element_slot_gas_density_ntot_normalization_carrier.total_element_density_cgs_derived"
            ),
            "owner_status": "KL-owned cgs-scale diagnostic candidate, not a proven J_ec owner",
            "budget_cap_formula": (
                "min_positive_element_stoichiometry(b[element] / formula_matrix_cond[element, condensate])"
            ),
            "total_element_density_source": (
                "gas_number_density_cgs / (H/2 + He + metals)"
            ),
            "FastChem_trace_values_used_as_inputs": False,
            "used_as_KL_constructor_input": False,
        },
        "ntotk": float(jax.device_get(ntotk)),
        "formula_matrix": _diagnostic_json_array(formula_matrix),
        "formula_matrix_cond": _diagnostic_json_array(formula_matrix_cond),
        "b": _diagnostic_json_array(b),
        "gk": _diagnostic_json_array(gk),
        "bk": _diagnostic_json_array(bk),
        "hvector_cond": _diagnostic_json_array(hvector_cond),
        "sk": _diagnostic_json_array(sk),
        "element_slot_gas_density_ntot_normalization_carrier": (
            element_slot_gas_density_ntot_normalization_carrier
        ),
        "condensates_jac_indices": (
            None
            if condensates_jac_indices is None
            else [int(index) for index in condensates_jac_indices]
        ),
        "condensate_labels_jac_order": (
            None
            if condensate_labels_jac_order is None
            else [str(label) for label in condensate_labels_jac_order]
        ),
        "element_labels_reduced_order": (
            None
            if element_labels_reduced_order is None
            else [str(label) for label in element_labels_reduced_order]
        ),
        "call_site_provenance": call_site_provenance,
        "source_provenance": call_site_provenance,
        "diagnostic_only": True,
        "default_off": True,
        "active": bool(active),
        "activation_flag": "emit_exact_input_bundle",
        "emitted_before_update_all_with_metrics": True,
    }
    hash_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"source_state_hash", "active", "activation_flag"}
    }
    payload["source_state_hash"] = _diagnostic_source_state_hash(hash_payload)
    return payload


def _normalize_exact_input_bundle_context(
    context: Optional[Dict[str, Any]],
    iter_count: int,
) -> Optional[Dict[str, Any]]:
    """Return active diagnostic bundle context for this iteration, if requested."""

    if context is None:
        return None
    if not bool(context.get("emit_exact_input_bundle", False)):
        return None
    target_iter = int(context.get("newton_iter", iter_count))
    if int(iter_count) != target_iter:
        return None
    lifecycle_context = context.get("gas_phase_calculate_lifecycle_context")
    lifecycle_context = {} if lifecycle_context is None else dict(lifecycle_context)
    if context.get("ln_nk_source_state_trace") is not None:
        lifecycle_context["ln_nk_source_state_trace"] = context.get(
            "ln_nk_source_state_trace"
        )
    if context.get("ln_nk_init_source_trace") is not None:
        lifecycle_context["ln_nk_init_source_trace"] = context.get(
            "ln_nk_init_source_trace"
        )
    return {
        "case_key": str(context.get("case_key", "diagnostic")),
        "newton_iter": target_iter,
        "condensates_jac_indices": context.get("condensates_jac_indices"),
        "condensate_labels_jac_order": context.get("condensate_labels_jac_order"),
        "element_labels_reduced_order": context.get("element_labels_reduced_order"),
        "row_scaled_element_condensate_jec_target_block": context.get(
            "row_scaled_element_condensate_jec_target_block"
        ),
        "selected_element_row_scaling_vector": context.get(
            "selected_element_row_scaling_vector"
        ),
        "gas_phase_calculate_lifecycle_context": lifecycle_context,
        "source_provenance": str(context.get("source_provenance", "diagnostic_context")),
        "ln_nk_producer_trace": context.get("ln_nk_producer_trace"),
        "ln_nk_source_state_trace": context.get("ln_nk_source_state_trace"),
        "ln_nk_init_source_trace": context.get("ln_nk_init_source_trace"),
        "call_site_provenance": str(
            context.get(
                "call_site_provenance",
                "src/exogibbs/optimize/pipm_rgie_cond.py::_update_all_with_metrics",
            )
        ),
    }


def solve_reduced_gibbs_iteration_equations_cond(
    nk: jnp.ndarray,
    mk: jnp.ndarray,
    ntotk: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    bk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    sk: jnp.ndarray,
    *,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Tuple[jnp.ndarray, float]:
    """
        Solve the reduced Gibbs iteration equations with condensates using the Lagrange multipliers.
        This function computes the matrix and vector to solve the system of equations
        that arises from the Gibbs energy minimization problem.

        Args:
            nk: number of species vector (n_species,) for k-th iteration.
            mk: number of condensed species vector (n_cond,) for k-th iteration.
            ntotk: Total number of species for k-th iteration.
            formula_matrix: Gas Formula matrix for stoichiometric constraints (n_elements, n_species).
            formula_matrix_cond: Condensates Formula matrix for stoichiometric constraints (n_elements, n_cond).
            b: Element abundance vector (n_elements, ).
            gk: gk vector (n_species,) for k-th iteration.
            bk: (gas) formula_matrix @ nk vector (n_elements, ).
            hvector_cond: chemical_potentials for condensates divided by RT (n_cond, )
            sk: mk^2/nu (n_cond, )

        Returns:
            Tuple containing:
                - The pi vector (nelements, ).fastchem_elements = list(gas.elements)
                    element_indices = jnp.array([fastchem_elements.index(e) for e in elements])

                - The update of the  log total number of species (delta_ln_ntot).
    """

    pi_vector, delta_ln_ntot, _metrics = _solve_reduced_gibbs_iteration_equations_cond_with_metrics(
        nk,
        mk,
        ntotk,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        bk,
        hvector_cond,
        sk,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )
    return pi_vector, delta_ln_ntot


def _solve_reduced_gibbs_iteration_equations_cond_with_metrics(
    nk: jnp.ndarray,
    mk: jnp.ndarray,
    ntotk: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    bk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    sk: jnp.ndarray,
    *,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
    include_system_trace: bool = False,
) -> Tuple[jnp.ndarray, float, Dict[str, jnp.ndarray]]:
    """Solve the reduced system and expose scale metrics for debugging."""

    assembled = _assemble_reduced_system_terms(
        nk,
        mk,
        ntotk,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        bk,
        hvector_cond,
        sk,
    )
    resn = assembled["resn"]
    q_block = assembled["Qk"]
    rhs = assembled["rhs"]
    scalar_rhs = assembled["scalar_rhs"]
    assemble_mat = assembled["assemble_mat"]
    assemble_vec = assembled["assemble_vec"]

    reg_q_block, reg_used = _regularize_q_block(
        q_block,
        regularization_mode,
        regularization_strength,
    )

    row_scale = jnp.maximum(jnp.max(jnp.abs(assemble_mat), axis=1, keepdims=True), 1.0)
    col_scale = jnp.maximum(jnp.max(jnp.abs(assemble_mat), axis=0, keepdims=True), 1.0)
    row_scale_flat = row_scale[:, 0]
    col_scale_flat = col_scale[0, :]

    if reduced_solver == "augmented_lu_row_scaled":
        solve_mat = assemble_mat / row_scale
        solve_vec = assemble_vec / row_scale[:, 0]
        lu, piv = lu_factor(solve_mat)
        assemble_variable = lu_solve((lu, piv), solve_vec)
        factorization_succeeded = jnp.all(jnp.isfinite(assemble_variable))
    elif reduced_solver == "augmented_lu_rowcol_scaled":
        reg_assemble_mat = jnp.block(
            [[reg_q_block, bk[:, None]], [bk[None, :], jnp.array([[resn]])]]
        )
        row_scale = jnp.maximum(jnp.max(jnp.abs(reg_assemble_mat), axis=1, keepdims=True), 1.0)
        solve_mat_row = reg_assemble_mat / row_scale
        col_scale = jnp.maximum(jnp.max(jnp.abs(solve_mat_row), axis=0, keepdims=True), 1.0)
        solve_mat = solve_mat_row / col_scale
        solve_vec = assemble_vec / row_scale[:, 0]
        lu, piv = lu_factor(solve_mat)
        scaled_solution = lu_solve((lu, piv), solve_vec)
        assemble_variable = scaled_solution / col_scale[0, :]
        factorization_succeeded = jnp.all(jnp.isfinite(assemble_variable))
        row_scale_flat = row_scale[:, 0]
        col_scale_flat = col_scale[0, :]
        assemble_mat = reg_assemble_mat
    elif reduced_solver == "schur_cholesky_reg":
        cho, lower = cho_factor(reg_q_block)
        rhs_pair = jnp.stack((rhs, bk), axis=1)
        solved_pair = cho_solve((cho, lower), rhs_pair)
        qinv_rhs = solved_pair[:, 0]
        qinv_bk = solved_pair[:, 1]
        schur = resn - jnp.vdot(bk, qinv_bk)
        schur_safe = jnp.where(
            jnp.abs(schur) < 1.0e-300,
            jnp.where(schur < 0.0, -1.0e-300, 1.0e-300),
            schur,
        )
        delta_ln_ntot = (scalar_rhs - jnp.vdot(bk, qinv_rhs)) / schur_safe
        pi_vector = qinv_rhs - qinv_bk * delta_ln_ntot
        assemble_variable = jnp.concatenate([pi_vector, jnp.array([delta_ln_ntot])])
        factorization_succeeded = jnp.all(jnp.isfinite(assemble_variable))
    else:
        raise ValueError(
            "Unknown reduced_solver "
            f"'{reduced_solver}'. Expected one of "
            "('augmented_lu_row_scaled', 'augmented_lu_rowcol_scaled', 'schur_cholesky_reg')."
        )

    metrics = {
        "reduced_solver_backend": reduced_solver,
        "reduced_factorization_succeeded": factorization_succeeded,
        "reduced_regularization_mode": regularization_mode,
        "reduced_regularization_strength": jnp.asarray(regularization_strength, dtype=q_block.dtype),
        "reduced_regularization_used": reg_used,
        "reduced_resn": resn,
        "reduced_row_scale_min": jnp.min(row_scale_flat),
        "reduced_row_scale_max": jnp.max(row_scale_flat),
        "reduced_row_scale_ratio": jnp.max(row_scale_flat)
        / jnp.maximum(jnp.min(row_scale_flat), 1.0e-300),
        "reduced_col_scale_min": jnp.min(col_scale_flat),
        "reduced_col_scale_max": jnp.max(col_scale_flat),
        "reduced_col_scale_ratio": jnp.max(col_scale_flat)
        / jnp.maximum(jnp.min(col_scale_flat), 1.0e-300),
        "reduced_mat_maxabs": jnp.max(jnp.abs(assemble_mat)),
        "reduced_vec_maxabs": jnp.max(jnp.abs(assemble_vec)),
        "reduced_qk_maxabs": jnp.max(jnp.abs(q_block)),
        "reduced_qk_diag_min": jnp.min(jnp.diag(q_block)),
        "reduced_qk_diag_max": jnp.max(jnp.diag(q_block)),
    }
    if include_system_trace:
        metrics.update(
            {
                "diagnostic_only_reduced_system_trace": jnp.asarray(True),
                "reduced_jacobian_matrix_before_scaling": assemble_mat,
                "reduced_rhs_vector_before_scaling": assemble_vec,
                "reduced_row_scaling_vector": row_scale_flat,
                "reduced_col_scaling_vector": col_scale_flat,
                "reduced_solve_matrix": solve_mat,
                "reduced_solve_rhs_vector": solve_vec,
                "reduced_raw_solver_result_vector": assemble_variable,
                "reduced_q_block": q_block,
                "reduced_rhs_element_block": rhs,
                "reduced_scalar_rhs": jnp.asarray(scalar_rhs),
                "reduced_element_condensate_block_before_scaling": (
                    jnp.asarray(formula_matrix_cond, dtype=jnp.float64)
                    * jnp.asarray(mk, dtype=jnp.float64)[None, :]
                ),
                "reduced_element_condensate_block_scaled_by_reduced_row_scaling": (
                    (
                        jnp.asarray(formula_matrix_cond, dtype=jnp.float64)
                        * jnp.asarray(mk, dtype=jnp.float64)[None, :]
                    )
                    / jnp.maximum(row_scale_flat[:-1], 1.0e-300)[:, None]
                ),
                "reduced_element_condensate_owner_mk_vector": jnp.asarray(
                    mk,
                    dtype=jnp.float64,
                ),
                "reduced_system_trace_basis": jnp.asarray(1, dtype=jnp.int32),
            }
        )
    return assemble_variable[:-1], assemble_variable[-1], metrics




def _compute_residuals(
    nk: jnp.ndarray,
    mk: jnp.ndarray,
    ntotk: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    nu: float,
    An: jnp.ndarray,
    Am: jnp.ndarray,
    pi_vector: jnp.ndarray,
) -> float:

    ress = nk * (formula_matrix.T @ pi_vector - gk)
    ress_squared = jnp.dot(ress, ress)

    resc = mk * (formula_matrix_cond.T @ pi_vector - hvector_cond) + nu
    resc_squared = jnp.dot(resc, resc)

    
    deltabhat = An + Am - b
    resj_squared = jnp.dot(deltabhat, deltabhat)

    resn = jnp.sum(nk) - ntotk
    resn_squared = jnp.dot(resn, resn)

    return jnp.sqrt(ress_squared + resc_squared + resj_squared + resn_squared)


def _compute_residual_component_metrics(
    nk: jnp.ndarray,
    mk: jnp.ndarray,
    ntotk: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    nu: float,
    pi_vector: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Return barrier residual components for diagnostics."""

    ress = nk * (formula_matrix.T @ pi_vector - gk)
    resc = mk * (formula_matrix_cond.T @ pi_vector - hvector_cond) + nu
    deltabhat = formula_matrix @ nk + formula_matrix_cond @ mk - b
    resn = jnp.sum(nk) - ntotk
    residual = jnp.sqrt(
        jnp.dot(ress, ress)
        + jnp.dot(resc, resc)
        + jnp.dot(deltabhat, deltabhat)
        + jnp.dot(resn, resn)
    )
    return {
        "fresh_residual": residual,
        "element_balance_residual_norm": jnp.linalg.norm(deltabhat),
        "ntot_residual": resn,
        "gas_stationarity_residual_norm": jnp.linalg.norm(ress),
        "cond_stationarity_residual_norm": jnp.linalg.norm(resc),
    }


def _compute_normalized_gibbs_energy(
    nk: jnp.ndarray,
    mk: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
) -> jnp.ndarray:
    """Return the current physical Gibbs energy normalized by RT."""

    return jnp.sum(nk * gk) + jnp.sum(mk * hvector_cond)


def _build_trial_lambda_grid(
    heuristic_lambda: float,
    lambda_trials: Optional[Sequence[float]] = None,
    lambda_multipliers: Sequence[float] = (1.0, 0.5, 0.2, 0.1, 0.05),
    extra_lambda_trials: Sequence[float] = (1.0, 0.5, 0.2, 0.1, 0.05),
) -> list[float]:
    """Build a de-duplicated trial grid around the current heuristic lambda."""

    if lambda_trials is not None:
        values = [float(x) for x in lambda_trials]
    else:
        values = [heuristic_lambda * float(scale) for scale in lambda_multipliers]
        values.extend(float(x) for x in extra_lambda_trials)
        values.append(float(heuristic_lambda))

    clipped = []
    for value in values:
        value = max(0.0, min(1.0, float(value)))
        if all(abs(value - existing) > 1.0e-12 for existing in clipped):
            clipped.append(value)
    return sorted(clipped, reverse=True)


def _compute_gas_limiter_species_diagnostics(
    ln_nk: jnp.ndarray,
    ln_ntot: float,
    delta_ln_nk: jnp.ndarray,
    delta_ln_ntot: float,
    heuristic_lam1_gas: float,
    *,
    species_names: Optional[Sequence[str]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Decompose the gas heuristic step limiter into per-species candidates."""

    ln_nk = jnp.asarray(ln_nk)
    ln_ntot = jnp.asarray(ln_ntot)
    delta_ln_nk = jnp.asarray(delta_ln_nk)
    delta_ln_ntot = jnp.asarray(delta_ln_ntot)

    nk = jnp.exp(ln_nk)
    vmr = jnp.exp(ln_nk - ln_ntot)
    common_ntot_cap = 2.0 / jnp.maximum(5.0 * jnp.abs(delta_ln_ntot), 1.0e-300)
    abs_delta_cap = 2.0 / jnp.maximum(jnp.abs(delta_ln_nk), 1.0e-300)
    cap_candidate = jnp.minimum(common_ntot_cap, abs_delta_cap)

    ln_xk = ln_nk - ln_ntot
    denom2 = delta_ln_nk - delta_ln_ntot
    small = (ln_xk <= -18.420681) & (delta_ln_nk >= 0.0)
    safe_trace = small & (denom2 > 0.0)
    trace_candidate = jnp.where(
        safe_trace,
        (-9.2103404 - ln_xk) / denom2,
        jnp.inf,
    )
    species_candidate = jnp.minimum(cap_candidate, trace_candidate)

    ranked = jnp.argsort(species_candidate)
    limit = min(int(species_candidate.shape[0]), top_k)
    top_indices = [int(i) for i in ranked[:limit]]

    species_records = []
    for rank, idx in enumerate(top_indices):
        species_records.append(
            {
                "rank": rank,
                "species_index": idx,
                "species_name": None if species_names is None else str(species_names[idx]),
                "ln_nk": float(ln_nk[idx]),
                "nk": float(nk[idx]),
                "delta_ln_nk": float(delta_ln_nk[idx]),
                "vmr": float(vmr[idx]),
                "ln_vmr": float(ln_xk[idx]),
                "common_ntot_cap_candidate": float(common_ntot_cap),
                "abs_delta_cap_candidate": float(abs_delta_cap[idx]),
                "trace_candidate_lambda": float(trace_candidate[idx]),
                "species_candidate_lambda": float(species_candidate[idx]),
                "trace_guard_active": bool(safe_trace[idx]),
                "is_within_top_k_smallest": True,
                "matches_heuristic_lam1_gas": bool(
                    abs(float(species_candidate[idx]) - float(heuristic_lam1_gas)) <= 1.0e-12
                ),
            }
        )

    global_abs_delta_index = int(jnp.argmax(jnp.abs(delta_ln_nk)))
    trace_ranked = jnp.argsort(trace_candidate)
    top_trace_indices = [
        int(i) for i in trace_ranked[: min(int(jnp.sum(safe_trace)), top_k)]
    ]
    return {
        "lam1_gas": float(heuristic_lam1_gas),
        "common_ntot_cap_candidate": float(common_ntot_cap),
        "max_abs_delta_species_index": global_abs_delta_index,
        "max_abs_delta_species_name": None
        if species_names is None
        else str(species_names[global_abs_delta_index]),
        "max_abs_delta_ln_nk": float(jnp.max(jnp.abs(delta_ln_nk))),
        "n_trace_guard_active": int(jnp.sum(safe_trace)),
        "top_trace_guard_indices": top_trace_indices,
        "top_trace_guard_names": None
        if species_names is None
        else [str(species_names[i]) for i in top_trace_indices],
        "top_species": species_records,
    }


def _compute_frozen_condensate_gas_direction_reference(
    nk: jnp.ndarray,
    mk: jnp.ndarray,
    ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Compute a gas-only reference direction with condensates frozen into b_eff."""

    b_eff = b - formula_matrix_cond @ mk
    An = formula_matrix @ nk
    pi_ref, delta_ln_ntot_ref = solve_gibbs_iteration_equations(
        nk,
        ntot,
        formula_matrix,
        b_eff,
        gk,
        An,
    )
    delta_ln_nk_ref = formula_matrix.T @ pi_ref + delta_ln_ntot_ref - gk
    return {
        "b_eff": b_eff,
        "pi_ref": pi_ref,
        "delta_ln_ntot_ref": delta_ln_ntot_ref,
        "delta_ln_nk_ref": delta_ln_nk_ref,
    }


def _compare_gas_directions(
    ln_nk: jnp.ndarray,
    ln_ntot: float,
    delta_ln_nk_full: jnp.ndarray,
    delta_ln_ntot_full: float,
    delta_ln_nk_ref: jnp.ndarray,
    delta_ln_ntot_ref: float,
    lam1_gas_full: float,
    *,
    species_names: Optional[Sequence[str]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Compare the coupled gas direction against a frozen-condensate gas-only reference."""

    delta_ln_nk_full = jnp.asarray(delta_ln_nk_full)
    delta_ln_nk_ref = jnp.asarray(delta_ln_nk_ref)
    diff = delta_ln_nk_full - delta_ln_nk_ref
    full_norm = jnp.linalg.norm(delta_ln_nk_full)
    ref_norm = jnp.linalg.norm(delta_ln_nk_ref)
    denom = jnp.maximum(full_norm * ref_norm, 1.0e-300)
    cosine_similarity = jnp.dot(delta_ln_nk_full, delta_ln_nk_ref) / denom
    cosine_similarity = jnp.clip(cosine_similarity, -1.0, 1.0)
    angle_degrees = jnp.degrees(jnp.arccos(cosine_similarity))
    lam1_gas_ref = stepsize_cea_gas(delta_ln_nk_ref, delta_ln_ntot_ref, ln_nk, ln_ntot)

    ranked = jnp.argsort(-jnp.abs(diff))
    limit = min(int(diff.shape[0]), top_k)
    top_indices = [int(i) for i in ranked[:limit]]
    disagreement_species = []
    for rank, idx in enumerate(top_indices):
        disagreement_species.append(
            {
                "rank": rank,
                "species_index": idx,
                "species_name": None if species_names is None else str(species_names[idx]),
                "delta_ln_nk_full": float(delta_ln_nk_full[idx]),
                "delta_ln_nk_ref": float(delta_ln_nk_ref[idx]),
                "delta_ln_nk_diff": float(diff[idx]),
                "abs_delta_ln_nk_diff": float(jnp.abs(diff[idx])),
            }
        )

    return {
        "norm_full": float(full_norm),
        "norm_ref": float(ref_norm),
        "cosine_similarity": float(cosine_similarity),
        "angle_degrees": float(angle_degrees),
        "max_abs_delta_ln_nk_diff": float(jnp.max(jnp.abs(diff))),
        "delta_ln_ntot_full": float(delta_ln_ntot_full),
        "delta_ln_ntot_ref": float(delta_ln_ntot_ref),
        "abs_delta_ln_ntot_diff": float(jnp.abs(delta_ln_ntot_full - delta_ln_ntot_ref)),
        "lam1_gas_full": float(lam1_gas_full),
        "lam1_gas_ref": float(lam1_gas_ref),
        "top_direction_disagreement_species": disagreement_species,
    }


def build_rgie_gas_direction_variant(
    variant_name: str,
    *,
    delta_ln_nk_current: jnp.ndarray,
    delta_ln_ntot_current: float,
    delta_ln_nk_ref: jnp.ndarray,
    delta_ln_ntot_ref: float,
) -> Dict[str, jnp.ndarray]:
    """Build a diagnostic gas-side direction variant for RGIE."""

    delta_ln_nk_current = jnp.asarray(delta_ln_nk_current)
    delta_ln_nk_ref = jnp.asarray(delta_ln_nk_ref)
    delta_ln_ntot_current = jnp.asarray(delta_ln_ntot_current, dtype=delta_ln_nk_current.dtype)
    delta_ln_ntot_ref = jnp.asarray(delta_ln_ntot_ref, dtype=delta_ln_nk_current.dtype)

    if variant_name == "current_full_direction":
        delta_ln_nk = delta_ln_nk_current
        delta_ln_ntot = delta_ln_ntot_current
    elif variant_name == "frozen_condensate_gas_only_reference":
        delta_ln_nk = delta_ln_nk_ref
        delta_ln_ntot = delta_ln_ntot_ref
    elif variant_name == "no_common_ntot_shift":
        delta_ln_nk = delta_ln_nk_current - delta_ln_ntot_current
        delta_ln_ntot = jnp.asarray(0.0, dtype=delta_ln_nk_current.dtype)
    elif variant_name == "partial_ntot_shift_0p25":
        scale = jnp.asarray(0.25, dtype=delta_ln_nk_current.dtype)
        delta_ln_nk = (delta_ln_nk_current - delta_ln_ntot_current) + scale * delta_ln_ntot_current
        delta_ln_ntot = scale * delta_ln_ntot_current
    elif variant_name == "partial_ntot_shift_0p5":
        scale = jnp.asarray(0.5, dtype=delta_ln_nk_current.dtype)
        delta_ln_nk = (delta_ln_nk_current - delta_ln_ntot_current) + scale * delta_ln_ntot_current
        delta_ln_ntot = scale * delta_ln_ntot_current
    elif variant_name == "gas_only_with_current_condensate_block":
        delta_ln_nk = delta_ln_nk_ref
        delta_ln_ntot = delta_ln_ntot_ref
    else:
        raise ValueError(
            "Unknown RGIE gas-direction variant "
            f"'{variant_name}'. Expected one of "
            "('current_full_direction', 'frozen_condensate_gas_only_reference', "
            "'no_common_ntot_shift', 'partial_ntot_shift_0p25', "
            "'partial_ntot_shift_0p5', 'gas_only_with_current_condensate_block')."
        )

    return {
        "variant_name": variant_name,
        "delta_ln_nk": delta_ln_nk,
        "delta_ln_ntot": delta_ln_ntot,
    }


def compute_rgie_lam1_gas_ignore_trace_diagnostics(
    ln_nk: jnp.ndarray,
    ln_ntot: float,
    delta_ln_nk: jnp.ndarray,
    delta_ln_ntot: float,
    vmr_floor: float,
) -> Dict[str, Any]:
    """Diagnostic-only lam1_gas recomputation that ignores ultra-trace species."""

    ln_nk = jnp.asarray(ln_nk)
    ln_ntot = jnp.asarray(ln_ntot)
    delta_ln_nk = jnp.asarray(delta_ln_nk)
    delta_ln_ntot = jnp.asarray(delta_ln_ntot, dtype=delta_ln_nk.dtype)
    vmr = jnp.exp(ln_nk - ln_ntot)
    active = vmr >= vmr_floor
    common_ntot_cap = 2.0 / jnp.maximum(5.0 * jnp.abs(delta_ln_ntot), 1.0e-300)
    abs_delta_cap = 2.0 / jnp.maximum(jnp.abs(delta_ln_nk), 1.0e-300)
    cap_candidate = jnp.minimum(common_ntot_cap, abs_delta_cap)
    ln_xk = ln_nk - ln_ntot
    denom2 = delta_ln_nk - delta_ln_ntot
    small = (ln_xk <= -18.420681) & (delta_ln_nk >= 0.0)
    safe_trace = small & (denom2 > 0.0)
    trace_candidate = jnp.where(
        safe_trace,
        (-9.2103404 - ln_xk) / denom2,
        jnp.inf,
    )
    species_candidate = jnp.minimum(cap_candidate, trace_candidate)
    active_candidate = jnp.where(active, species_candidate, jnp.inf)
    top_index = int(jnp.argmin(species_candidate))
    active_top_index = int(jnp.argmin(active_candidate))
    lam1_gas_ignore_trace = float(jnp.min(active_candidate))
    if not math.isfinite(lam1_gas_ignore_trace):
        lam1_gas_ignore_trace = float(common_ntot_cap)
    return {
        "vmr_floor": float(vmr_floor),
        "lam1_gas_ignore_trace": lam1_gas_ignore_trace,
        "current_top_limiter_species_index": top_index,
        "current_top_limiter_active_under_floor": bool(active[top_index]),
        "active_top_limiter_species_index": active_top_index,
        "ignored_species_count": int(jnp.sum(~active)),
    }


def diagnose_iteration_lambda_trials(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    *,
    element_indices: Optional[jnp.ndarray] = None,
    lambda_trials: Optional[Sequence[float]] = None,
    lambda_multipliers: Sequence[float] = (1.0, 0.5, 0.2, 0.1, 0.05),
    extra_lambda_trials: Sequence[float] = (1.0, 0.5, 0.2, 0.1, 0.05),
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
    charge_row_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate fresh-residual trial lambdas along one fixed current direction.

    This helper is diagnostic-only. It computes the current reduced-system
    direction once, then evaluates multiple lambda values along that same
    direction without altering the normal solver update rule.
    """

    n_elements = formula_matrix.shape[0]
    b = (
        jnp.asarray(state.element_vector)
        if element_indices is None
        else jnp.asarray(state.element_vector)[jnp.asarray(element_indices)]
    )
    if b.shape[0] != n_elements:
        raise ValueError(
            "ThermoState.element_vector length does not match the number of element rows "
            f"in the formula matrices (got {b.shape[0]}, expected {n_elements}). "
            "Provide element_indices that map the state vector onto the reduced element set."
        )
    if charge_row_index is not None:
        charge_index = int(charge_row_index)
        if charge_index < 0 or charge_index >= n_elements:
            raise ValueError("charge_row_index is out of range for the element rows.")
    else:
        charge_index = None

    hvector = hvector_func(state.temperature)
    hvector_cond = hvector_cond_func(state.temperature)
    gk = _compute_gk(
        state.temperature,
        ln_nk,
        ln_ntot,
        hvector,
        state.ln_normalized_pressure,
    )
    step_metrics = _compute_iteration_step_metrics(
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        epsilon,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )

    heuristic_lambda = float(step_metrics["lam"])
    trial_grid = _build_trial_lambda_grid(
        heuristic_lambda,
        lambda_trials=lambda_trials,
        lambda_multipliers=lambda_multipliers,
        extra_lambda_trials=extra_lambda_trials,
    )

    delta_ln_nk = jnp.asarray(step_metrics["delta_ln_nk"])
    delta_ln_mk = jnp.asarray(step_metrics["delta_ln_mk"])
    delta_ln_ntot = jnp.asarray(step_metrics["delta_ln_ntot"])

    trials = []
    for lambda_trial in trial_grid:
        lambda_trial_arr = jnp.asarray(lambda_trial, dtype=jnp.asarray(ln_ntot).dtype)
        trial_ln_nk = jnp.asarray(ln_nk) + lambda_trial_arr * delta_ln_nk
        trial_ln_mk = jnp.asarray(ln_mk) + lambda_trial_arr * delta_ln_mk
        trial_ln_ntot = jnp.asarray(ln_ntot) + lambda_trial_arr * delta_ln_ntot

        trial_nk = jnp.exp(trial_ln_nk)
        trial_mk = jnp.exp(trial_ln_mk)
        trial_ntot = jnp.exp(trial_ln_ntot)
        trial_gk = _compute_gk(
            state.temperature,
            trial_ln_nk,
            trial_ln_ntot,
            hvector,
            state.ln_normalized_pressure,
        )

        invalid_numbers_detected = bool(
            _contains_invalid_numbers(
                trial_ln_nk,
                trial_ln_mk,
                trial_ln_ntot,
                trial_nk,
                trial_mk,
                trial_ntot,
                trial_gk,
            )
        )
        sk_margin = LOG_S_MAX + epsilon - 2.0 * trial_ln_mk
        min_sk_margin = jnp.min(sk_margin)
        is_sk_feasible = bool(jnp.all(sk_margin >= 0.0))

        if invalid_numbers_detected:
            pi_vector_resid = None
            residual_metrics = {
                "fresh_residual": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
                "element_balance_residual_norm": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
                "ntot_residual": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
                "gas_stationarity_residual_norm": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
                "cond_stationarity_residual_norm": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
            }
            normalized_gibbs = jnp.asarray(jnp.nan, dtype=trial_ntot.dtype)
            zero_charge_residual = None
        else:
            pi_vector_resid = _recompute_pi_for_residual(
                trial_nk,
                trial_mk,
                trial_ntot,
                formula_matrix,
                formula_matrix_cond,
                b,
                trial_gk,
                hvector_cond,
                epsilon,
                reduced_solver=reduced_solver,
                regularization_mode=regularization_mode,
                regularization_strength=regularization_strength,
            )
            residual_metrics = _compute_residual_component_metrics(
                trial_nk,
                trial_mk,
                trial_ntot,
                formula_matrix,
                formula_matrix_cond,
                b,
                trial_gk,
                hvector_cond,
                jnp.exp(epsilon),
                pi_vector_resid,
            )
            normalized_gibbs = _compute_normalized_gibbs_energy(
                trial_nk,
                trial_mk,
                trial_gk,
                hvector_cond,
            )
            if charge_index is None:
                zero_charge_residual = None
            else:
                charge_raw = (
                    formula_matrix[charge_index, :] @ trial_nk
                    + formula_matrix_cond[charge_index, :] @ trial_mk
                    - b[charge_index]
                )
                charge_scale = jnp.maximum(jnp.abs(b[charge_index]), 1.0)
                zero_charge_residual = float(charge_raw / charge_scale)

        if abs(lambda_trial - heuristic_lambda) <= 1.0e-12:
            relative_to_heuristic = "equal"
        elif lambda_trial > heuristic_lambda:
            relative_to_heuristic = "above"
        else:
            relative_to_heuristic = "below"

        trials.append(
            {
                "lambda_trial": float(lambda_trial),
                "relative_to_heuristic": relative_to_heuristic,
                "is_above_heuristic_lambda": bool(lambda_trial > heuristic_lambda + 1.0e-12),
                "is_below_heuristic_lambda": bool(lambda_trial < heuristic_lambda - 1.0e-12),
                "fresh_residual": float(residual_metrics["fresh_residual"]),
                "element_balance_residual_norm": float(
                    residual_metrics["element_balance_residual_norm"]
                ),
                "ntot_residual": float(residual_metrics["ntot_residual"]),
                "gas_stationarity_residual_norm": float(
                    residual_metrics["gas_stationarity_residual_norm"]
                ),
                "cond_stationarity_residual_norm": float(
                    residual_metrics["cond_stationarity_residual_norm"]
                ),
                "zero_charge_residual": zero_charge_residual,
                "sk_feasibility_margin_min": float(min_sk_margin),
                "sk_feasibility_margin_worst": float(min_sk_margin),
                "sk_feasible": is_sk_feasible,
                "n_sk_infeasible": int(jnp.sum(sk_margin < 0.0)),
                "invalid_numbers_detected": invalid_numbers_detected,
                "all_finite": not invalid_numbers_detected,
                "max_abs_delta_ln_nk": float(jnp.max(jnp.abs(lambda_trial_arr * delta_ln_nk))),
                "max_abs_delta_ln_mk": float(jnp.max(jnp.abs(lambda_trial_arr * delta_ln_mk))),
                "abs_delta_ln_ntot": float(jnp.abs(lambda_trial_arr * delta_ln_ntot)),
                "normalized_gibbs_energy": float(normalized_gibbs),
            }
        )

    return {
        "epsilon": float(epsilon),
        "heuristic_lambda": heuristic_lambda,
        "trial_lambdas": [trial["lambda_trial"] for trial in trials],
        "step_metrics": {
            "lam": float(step_metrics["lam"]),
            "lam1_gas": float(step_metrics["lam1_gas"]),
            "lam1_cond": float(step_metrics["lam1_cond"]),
            "lam2_cond": float(step_metrics["lam2_cond"]),
            "limiting_index": int(step_metrics["limiting_index"]),
            "delta_ln_ntot": float(step_metrics["delta_ln_ntot"]),
            "max_abs_delta_ln_nk": float(step_metrics["max_abs_delta_ln_nk"]),
            "max_abs_raw_delta_ln_mk": float(step_metrics["max_abs_raw_delta_ln_mk"]),
            "max_abs_clipped_delta_ln_mk": float(step_metrics["max_abs_clipped_delta_ln_mk"]),
        },
        "trials": trials,
    }


def diagnose_gas_step_limiter_and_direction(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    *,
    element_indices: Optional[jnp.ndarray] = None,
    gas_species_names: Optional[Sequence[str]] = None,
    top_k: int = 10,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Dict[str, Any]:
    """Diagnostic-only gas limiter decomposition and frozen-condensate direction comparison."""

    n_elements = formula_matrix.shape[0]
    b = (
        jnp.asarray(state.element_vector)
        if element_indices is None
        else jnp.asarray(state.element_vector)[jnp.asarray(element_indices)]
    )
    if b.shape[0] != n_elements:
        raise ValueError(
            "ThermoState.element_vector length does not match the number of element rows "
            f"in the formula matrices (got {b.shape[0]}, expected {n_elements}). "
            "Provide element_indices that map the state vector onto the reduced element set."
        )

    hvector = hvector_func(state.temperature)
    hvector_cond = hvector_cond_func(state.temperature)
    gk = _compute_gk(
        state.temperature,
        ln_nk,
        ln_ntot,
        hvector,
        state.ln_normalized_pressure,
    )
    step_metrics = _compute_iteration_step_metrics(
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        epsilon,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )

    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntot = jnp.exp(ln_ntot)
    gas_limiter = _compute_gas_limiter_species_diagnostics(
        ln_nk,
        ln_ntot,
        step_metrics["delta_ln_nk"],
        step_metrics["delta_ln_ntot"],
        step_metrics["lam1_gas"],
        species_names=gas_species_names,
        top_k=top_k,
    )
    gas_ref = _compute_frozen_condensate_gas_direction_reference(
        nk,
        mk,
        ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
    )
    direction_comparison = _compare_gas_directions(
        ln_nk,
        ln_ntot,
        step_metrics["delta_ln_nk"],
        step_metrics["delta_ln_ntot"],
        gas_ref["delta_ln_nk_ref"],
        gas_ref["delta_ln_ntot_ref"],
        step_metrics["lam1_gas"],
        species_names=gas_species_names,
        top_k=top_k,
    )
    return {
        "epsilon": float(epsilon),
        "step_metrics": {
            "lam": float(step_metrics["lam"]),
            "lam1_gas": float(step_metrics["lam1_gas"]),
            "lam1_cond": float(step_metrics["lam1_cond"]),
            "lam2_cond": float(step_metrics["lam2_cond"]),
            "limiting_index": int(step_metrics["limiting_index"]),
        },
        "gas_limiter": gas_limiter,
        "gas_direction_reference": {
            "b_eff_norm": float(jnp.linalg.norm(gas_ref["b_eff"])),
            "delta_ln_ntot_ref": float(gas_ref["delta_ln_ntot_ref"]),
        },
        "direction_comparison": direction_comparison,
    }


def diagnose_reduced_solver_backend_experiments(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    *,
    element_indices: Optional[jnp.ndarray] = None,
    backend_configs: Optional[Sequence[Dict[str, Any]]] = None,
    case_key: str = "diagnostic",
    newton_iter: int = 0,
    condensates_jac_indices: Optional[Sequence[int]] = None,
    condensate_labels_jac_order: Optional[Sequence[str]] = None,
    element_labels_reduced_order: Optional[Sequence[str]] = None,
    emit_exact_input_bundle: bool = False,
    row_scaled_element_condensate_jec_target_block: Optional[Sequence[Sequence[float]]] = None,
    selected_element_row_scaling_vector: Optional[Sequence[float]] = None,
    gas_phase_calculate_lifecycle_context: Optional[Dict[str, Any]] = None,
    ln_nk_init_source_trace: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare one-step reduced-solver backend experiments from the same state."""

    if backend_configs is None:
        backend_configs = (
            {
                "reduced_solver": DEFAULT_REDUCED_SOLVER,
                "regularization_mode": DEFAULT_REGULARIZATION_MODE,
                "regularization_strength": DEFAULT_REGULARIZATION_STRENGTH,
            },
            {
                "reduced_solver": "augmented_lu_rowcol_scaled",
                "regularization_mode": DEFAULT_REGULARIZATION_MODE,
                "regularization_strength": DEFAULT_REGULARIZATION_STRENGTH,
            },
            {
                "reduced_solver": "schur_cholesky_reg",
                "regularization_mode": "diag_shift",
                "regularization_strength": 1.0e-12,
            },
        )

    n_elements = formula_matrix.shape[0]
    b = (
        jnp.asarray(state.element_vector)
        if element_indices is None
        else jnp.asarray(state.element_vector)[jnp.asarray(element_indices)]
    )
    if b.shape[0] != n_elements:
        raise ValueError(
            "ThermoState.element_vector length does not match the number of element rows "
            f"in the formula matrices (got {b.shape[0]}, expected {n_elements}). "
            "Provide element_indices that map the state vector onto the reduced element set."
        )

    hvector = hvector_func(state.temperature)
    hvector_cond = hvector_cond_func(state.temperature)
    gk = _compute_gk(
        state.temperature,
        ln_nk,
        ln_ntot,
        hvector,
        state.ln_normalized_pressure,
    )
    An = formula_matrix @ jnp.exp(ln_nk)
    Am = formula_matrix_cond @ jnp.exp(ln_mk)
    exact_input_bundle = None
    if emit_exact_input_bundle:
        if gas_phase_calculate_lifecycle_context is None:
            lifecycle_context = None
        else:
            lifecycle_context = dict(gas_phase_calculate_lifecycle_context)
        if ln_nk_init_source_trace is not None:
            lifecycle_context = {} if lifecycle_context is None else lifecycle_context
            lifecycle_context["ln_nk_init_source_trace"] = dict(
                ln_nk_init_source_trace
            )
        exact_input_bundle = _build_reduced_solver_exact_input_bundle(
            case_key=case_key,
            newton_iter=newton_iter,
            ln_nk=ln_nk,
            ln_mk=ln_mk,
            ln_ntot=ln_ntot,
            temperature=state.temperature,
            ln_normalized_pressure=state.ln_normalized_pressure,
            formula_matrix=formula_matrix,
            formula_matrix_cond=formula_matrix_cond,
            b=b,
            gk=gk,
            hvector_cond=hvector_cond,
            epsilon=epsilon,
            condensates_jac_indices=condensates_jac_indices,
            condensate_labels_jac_order=condensate_labels_jac_order,
            element_labels_reduced_order=element_labels_reduced_order,
            call_site_provenance=(
                "src/exogibbs/optimize/pipm_rgie_cond.py::"
                "diagnose_reduced_solver_backend_experiments before _update_all_with_metrics"
            ),
            active=True,
            row_scaled_element_condensate_jec_target_block=(
                row_scaled_element_condensate_jec_target_block
            ),
            selected_element_row_scaling_vector=selected_element_row_scaling_vector,
            gas_phase_calculate_lifecycle_context=lifecycle_context,
            ln_nk_producer_trace=_build_ln_nk_producer_trace(
                ln_nk=ln_nk,
                source_stage="diagnose_reduced_solver_backend_experiments argument ln_nk",
                producer_function=(
                    "src/exogibbs/optimize/pipm_rgie_cond.py::"
                    "diagnose_reduced_solver_backend_experiments"
                ),
                case_key=case_key,
                newton_iter=newton_iter,
            ),
        )

    comparisons = []
    baseline_delta_ln_nk = None
    baseline_backend = None
    for config in backend_configs:
        reduced_solver = config.get("reduced_solver", DEFAULT_REDUCED_SOLVER)
        regularization_mode = config.get("regularization_mode", DEFAULT_REGULARIZATION_MODE)
        regularization_strength = config.get(
            "regularization_strength", DEFAULT_REGULARIZATION_STRENGTH
        )
        include_system_trace = bool(config.get("include_system_trace", False))
        start = perf_counter()
        (
            _ln_nk_new,
            _ln_mk_new,
            _ln_ntot_new,
            _gk_new,
            _An_new,
            _Am_new,
            residual,
            _lam,
            metrics,
        ) = _update_all_with_metrics(
            ln_nk,
            ln_mk,
            ln_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            state.temperature,
            state.ln_normalized_pressure,
            hvector,
            hvector_cond,
            gk,
            An,
            Am,
            jnp.inf,
            epsilon,
            iter_count=0,
            debug_nan=False,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
            include_system_trace=include_system_trace,
        )
        runtime_seconds = perf_counter() - start
        delta_ln_nk = jnp.asarray(metrics["delta_ln_nk"])
        if baseline_delta_ln_nk is None:
            baseline_delta_ln_nk = delta_ln_nk
            baseline_backend = reduced_solver
            cosine_vs_baseline = 1.0
            max_abs_diff_vs_baseline = 0.0
        else:
            denom = jnp.maximum(
                jnp.linalg.norm(delta_ln_nk) * jnp.linalg.norm(baseline_delta_ln_nk),
                1.0e-300,
            )
            cosine_vs_baseline = float(
                jnp.clip(jnp.dot(delta_ln_nk, baseline_delta_ln_nk) / denom, -1.0, 1.0)
            )
            max_abs_diff_vs_baseline = float(
                jnp.max(jnp.abs(delta_ln_nk - baseline_delta_ln_nk))
            )
        comparison = {
                "backend": reduced_solver,
                "regularization_mode": regularization_mode,
                "regularization_strength": float(regularization_strength),
                "include_system_trace": include_system_trace,
                "factorization_succeeded": bool(metrics["reduced_factorization_succeeded"]),
                "regularization_used": float(metrics["reduced_regularization_used"]),
                "reduced_row_scale_ratio": float(metrics["reduced_row_scale_ratio"]),
                "reduced_col_scale_ratio": float(metrics["reduced_col_scale_ratio"]),
                "pi_norm": float(metrics["pi_norm"]),
                "delta_ln_ntot": float(metrics["delta_ln_ntot"]),
                "lam1_gas": float(metrics["lam1_gas"]),
                "lam1_cond": float(metrics["lam1_cond"]),
                "lam2_cond": float(metrics["lam2_cond"]),
                "lam": float(metrics["lam"]),
                "fresh_post_update_residual": float(residual),
                "direction_norm": float(jnp.linalg.norm(delta_ln_nk)),
                "runtime_seconds": runtime_seconds,
                "invalid_numbers_detected": bool(
                    _contains_invalid_numbers(
                        metrics["pi_vector"],
                        metrics["delta_ln_nk"],
                        metrics["delta_ln_mk"],
                        metrics["delta_ln_ntot"],
                        residual,
                    )
                ),
                "cosine_similarity_vs_baseline": cosine_vs_baseline,
                "max_abs_delta_ln_nk_diff_vs_baseline": max_abs_diff_vs_baseline,
                "baseline_backend": baseline_backend,
            }
        if exact_input_bundle is not None:
            comparison["exact_input_bundle"] = exact_input_bundle
        if include_system_trace:
            trace_mat = jnp.asarray(metrics["reduced_jacobian_matrix_before_scaling"])
            trace_rhs = jnp.asarray(metrics["reduced_rhs_vector_before_scaling"])
            trace_solve_mat = jnp.asarray(metrics["reduced_solve_matrix"])
            trace_solve_rhs = jnp.asarray(metrics["reduced_solve_rhs_vector"])
            trace_result = jnp.asarray(metrics["reduced_raw_solver_result_vector"])
            comparison["reduced_system_trace"] = {
                "diagnostic_only": True,
                "jacobian_matrix_before_scaling_shape": [int(trace_mat.shape[0]), int(trace_mat.shape[1])],
                "rhs_vector_before_scaling_length": int(trace_rhs.shape[0]),
                "solve_matrix_shape": [int(trace_solve_mat.shape[0]), int(trace_solve_mat.shape[1])],
                "solve_rhs_vector_length": int(trace_solve_rhs.shape[0]),
                "raw_solver_result_vector_length": int(trace_result.shape[0]),
                "row_scaling_vector_length": int(jnp.asarray(metrics["reduced_row_scaling_vector"]).shape[0]),
                "backend": reduced_solver,
            }
        comparisons.append(comparison)

    return {
        "epsilon": float(epsilon),
        "baseline_backend": baseline_backend,
        "comparisons": comparisons,
        "exact_input_bundle_emitter": {
            "implemented": True,
            "diagnostic_only": True,
            "default_off": True,
            "active": bool(emit_exact_input_bundle),
            "case_key": str(case_key),
            "newton_iter": int(newton_iter),
            "call_site_provenance": (
                "src/exogibbs/optimize/pipm_rgie_cond.py::"
                "diagnose_reduced_solver_backend_experiments before _update_all_with_metrics"
            ),
        },
    }


def _evaluate_direction_with_existing_update_rule(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    hvector: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
    delta_ln_nk: jnp.ndarray,
    raw_delta_ln_mk: jnp.ndarray,
    delta_ln_ntot: float,
    *,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Dict[str, Any]:
    """Apply the current clipping, limiter, and fresh-residual evaluation to a direction."""

    clipped_delta_ln_mk = jnp.clip(raw_delta_ln_mk, -0.1, 0.1)
    lam1_gas = stepsize_cea_gas(delta_ln_nk, delta_ln_ntot, ln_nk, ln_ntot)
    lam1_cond = stepsize_cond_heurstic(clipped_delta_ln_mk)
    lam2_cond = stepsize_sk(clipped_delta_ln_mk, ln_mk, epsilon)
    lam = jnp.minimum(1.0, jnp.minimum(lam1_gas, jnp.minimum(lam1_cond, lam2_cond)))
    lam = jnp.clip(lam, 0.0, 1.0)

    trial_ln_nk = jnp.asarray(ln_nk) + lam * jnp.asarray(delta_ln_nk)
    trial_ln_mk = jnp.asarray(ln_mk) + lam * jnp.asarray(clipped_delta_ln_mk)
    trial_ln_ntot = jnp.asarray(ln_ntot) + lam * jnp.asarray(delta_ln_ntot)
    trial_nk = jnp.exp(trial_ln_nk)
    trial_mk = jnp.exp(trial_ln_mk)
    trial_ntot = jnp.exp(trial_ln_ntot)
    trial_gk = _compute_gk(
        state.temperature,
        trial_ln_nk,
        trial_ln_ntot,
        hvector,
        state.ln_normalized_pressure,
    )
    invalid_numbers_detected = bool(
        _contains_invalid_numbers(
            trial_ln_nk,
            trial_ln_mk,
            trial_ln_ntot,
            trial_nk,
            trial_mk,
            trial_ntot,
            trial_gk,
            delta_ln_nk,
            raw_delta_ln_mk,
            clipped_delta_ln_mk,
            delta_ln_ntot,
        )
    )
    if invalid_numbers_detected:
        pi_vector_resid = None
        residual_metrics = {
            "fresh_residual": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
            "element_balance_residual_norm": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
            "ntot_residual": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
            "gas_stationarity_residual_norm": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
            "cond_stationarity_residual_norm": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
        }
    else:
        pi_vector_resid = _recompute_pi_for_residual(
            trial_nk,
            trial_mk,
            trial_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            trial_gk,
            hvector_cond,
            epsilon,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
        )
        residual_metrics = _compute_residual_component_metrics(
            trial_nk,
            trial_mk,
            trial_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            trial_gk,
            hvector_cond,
            jnp.exp(epsilon),
            pi_vector_resid,
        )

    return {
        "raw_direction_norm": float(
            jnp.linalg.norm(
                jnp.concatenate(
                    [
                        jnp.ravel(jnp.asarray(delta_ln_nk)),
                        jnp.ravel(jnp.asarray(raw_delta_ln_mk)),
                        jnp.atleast_1d(jnp.asarray(delta_ln_ntot)),
                    ]
                )
            )
        ),
        "clipped_direction_norm": float(
            jnp.linalg.norm(
                jnp.concatenate(
                    [
                        jnp.ravel(jnp.asarray(delta_ln_nk)),
                        jnp.ravel(jnp.asarray(clipped_delta_ln_mk)),
                        jnp.atleast_1d(jnp.asarray(delta_ln_ntot)),
                    ]
                )
            )
        ),
        "gas_direction_norm": float(jnp.linalg.norm(jnp.asarray(delta_ln_nk))),
        "raw_cond_direction_norm": float(jnp.linalg.norm(jnp.asarray(raw_delta_ln_mk))),
        "clipped_cond_direction_norm": float(jnp.linalg.norm(jnp.asarray(clipped_delta_ln_mk))),
        "delta_ln_ntot": float(delta_ln_ntot),
        "lam1_gas": float(lam1_gas),
        "lam1_cond": float(lam1_cond),
        "lam2_cond": float(lam2_cond),
        "lam": float(lam),
        "fresh_post_update_residual": float(residual_metrics["fresh_residual"]),
        "element_balance_residual_norm": float(
            residual_metrics["element_balance_residual_norm"]
        ),
        "ntot_residual": float(residual_metrics["ntot_residual"]),
        "gas_stationarity_residual_norm": float(
            residual_metrics["gas_stationarity_residual_norm"]
        ),
        "cond_stationarity_residual_norm": float(
            residual_metrics["cond_stationarity_residual_norm"]
        ),
        "invalid_numbers_detected": invalid_numbers_detected,
        "all_finite": not invalid_numbers_detected,
        "max_abs_delta_ln_nk": float(jnp.max(jnp.abs(delta_ln_nk))),
        "max_abs_raw_delta_ln_mk": float(jnp.max(jnp.abs(raw_delta_ln_mk))),
        "max_abs_clipped_delta_ln_mk": float(jnp.max(jnp.abs(clipped_delta_ln_mk))),
    }


def _solve_full_pipm_gie_direction_with_metrics(
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
) -> Dict[str, Any]:
    """Solve the full PIPM GIE linearization directly for diagnostic comparison."""

    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntot = jnp.exp(ln_ntot)
    bk = formula_matrix @ nk
    nu = jnp.exp(epsilon)
    start = perf_counter()
    delta_ln_nk, delta_ln_mk, pi_vector, delta_ln_ntot = (
        solve_full_gibbs_iteration_equations_cond(
            nk,
            mk,
            ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            gk,
            hvector_cond,
            bk,
            nu,
        )
    )
    runtime_seconds = perf_counter() - start
    factorization_succeeded = bool(
        jnp.all(
            jnp.isfinite(
                jnp.concatenate(
                    [
                        jnp.ravel(jnp.asarray(delta_ln_nk)),
                        jnp.ravel(jnp.asarray(delta_ln_mk)),
                        jnp.ravel(jnp.asarray(pi_vector)),
                        jnp.atleast_1d(jnp.asarray(delta_ln_ntot)),
                    ]
                )
            )
        )
    )
    return {
        "delta_ln_nk": delta_ln_nk,
        "raw_delta_ln_mk": delta_ln_mk,
        "pi_vector": pi_vector,
        "delta_ln_ntot": delta_ln_ntot,
        "pi_norm": float(jnp.linalg.norm(jnp.asarray(pi_vector))),
        "runtime_seconds": runtime_seconds,
        "factorization_succeeded": factorization_succeeded,
    }


def _solve_full_pdipm_gie_direction_with_metrics(
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    rho: Optional[jnp.ndarray],
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
) -> Dict[str, Any]:
    """Solve the full PDIPM GIE linearization directly for one diagnostic state."""

    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntot = jnp.exp(ln_ntot)
    rho = epsilon - ln_mk if rho is None else jnp.asarray(rho)
    eta = jnp.exp(rho)

    nspecies = int(nk.shape[0])
    ncond = int(mk.shape[0])
    nelement = int(b.shape[0])

    y_gas = formula_matrix * nk
    y_cond = formula_matrix_cond * mk
    u = jnp.ones((nspecies, 1), dtype=nk.dtype)
    zeros_nm = jnp.zeros((nspecies, ncond), dtype=nk.dtype)
    zeros_en = jnp.zeros((nelement, nspecies), dtype=nk.dtype)
    zeros_eb = jnp.zeros((nelement, nelement), dtype=nk.dtype)
    zeros_em = jnp.zeros((nelement, ncond), dtype=nk.dtype)
    zeros_cn = jnp.zeros((ncond, nspecies), dtype=nk.dtype)
    zeros_cb = jnp.zeros((ncond, nelement), dtype=nk.dtype)
    zeros_c1 = jnp.zeros((ncond, 1), dtype=nk.dtype)
    zeros_1m = jnp.zeros((1, ncond), dtype=nk.dtype)
    zeros_1b = jnp.zeros((1, nelement), dtype=nk.dtype)

    row_fn = jnp.block(
        [
            jnp.eye(nspecies, dtype=nk.dtype),
            zeros_nm,
            -formula_matrix.T,
            jnp.zeros((nspecies, ncond), dtype=nk.dtype),
            -u,
        ]
    )
    row_fm = jnp.block(
        [
            zeros_cn,
            jnp.zeros((ncond, ncond), dtype=nk.dtype),
            formula_matrix_cond.T,
            jnp.diag(eta),
            zeros_c1,
        ]
    )
    row_flambda = jnp.block(
        [
            y_gas,
            y_cond,
            zeros_eb,
            zeros_em,
            jnp.zeros((nelement, 1), dtype=nk.dtype),
        ]
    )
    row_fc = jnp.block(
        [
            zeros_cn,
            jnp.eye(ncond, dtype=nk.dtype),
            zeros_cb,
            jnp.eye(ncond, dtype=nk.dtype),
            zeros_c1,
        ]
    )
    row_ftot = jnp.block(
        [
            nk[jnp.newaxis, :],
            zeros_1m,
            zeros_1b,
            zeros_1m,
            jnp.array([[-ntot]], dtype=nk.dtype),
        ]
    )
    assemble_mat = jnp.block(
        [
            [row_fn],
            [row_fm],
            [row_flambda],
            [row_fc],
            [row_ftot],
        ]
    )

    rhs_fn = -gk
    rhs_fm = hvector_cond - eta
    rhs_flambda = b - formula_matrix @ nk - formula_matrix_cond @ mk
    rhs_fc = jnp.full_like(mk, epsilon) - ln_mk - rho
    rhs_ftot = jnp.array([ntot - jnp.sum(nk)], dtype=nk.dtype)
    assemble_vec = jnp.concatenate(
        [rhs_fn, rhs_fm, rhs_flambda, rhs_fc, rhs_ftot]
    )

    start = perf_counter()
    assemble_variable = jnp.linalg.solve(assemble_mat, assemble_vec)
    runtime_seconds = perf_counter() - start

    delta_ln_nk = assemble_variable[:nspecies]
    delta_ln_mk = assemble_variable[nspecies : nspecies + ncond]
    pi_vector = assemble_variable[nspecies + ncond : nspecies + ncond + nelement]
    delta_rho = assemble_variable[
        nspecies + ncond + nelement : nspecies + 2 * ncond + nelement
    ]
    delta_ln_ntot = assemble_variable[-1]

    factorization_succeeded = bool(
        jnp.all(jnp.isfinite(assemble_variable))
    )
    return {
        "rho": rho,
        "eta": eta,
        "delta_ln_nk": delta_ln_nk,
        "raw_delta_ln_mk": delta_ln_mk,
        "pi_vector": pi_vector,
        "delta_rho": delta_rho,
        "delta_ln_ntot": delta_ln_ntot,
        "pi_norm": float(jnp.linalg.norm(jnp.asarray(pi_vector))),
        "runtime_seconds": runtime_seconds,
        "factorization_succeeded": factorization_succeeded,
    }


def _pdipm_total_residual_norm(residual_components: Dict[str, float]) -> float:
    """Combine PDIPM residual components into a single Euclidean norm."""

    return float(
        jnp.sqrt(
            residual_components["Fn_norm"] ** 2
            + residual_components["Fm_norm"] ** 2
            + residual_components["Flambda_norm"] ** 2
            + residual_components["Fc_norm"] ** 2
            + residual_components["Ftot_abs"] ** 2
        )
    )


def _compute_pdipm_barrier_residual_components(
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_eta: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    pi_vector: jnp.ndarray,
    epsilon: float,
) -> Dict[str, float]:
    """Compute nonlinear PDIPM residual component norms at a state."""

    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntot = jnp.exp(ln_ntot)
    eta = jnp.exp(ln_eta)
    fn = nk * (formula_matrix.T @ pi_vector - gk)
    fm = eta + formula_matrix_cond.T @ pi_vector - hvector_cond
    flambda = formula_matrix @ nk + formula_matrix_cond @ mk - b
    fc = ln_mk + ln_eta - epsilon
    ftot = jnp.sum(nk) - ntot
    return {
        "Fn_norm": float(jnp.linalg.norm(fn)),
        "Fm_norm": float(jnp.linalg.norm(fm)),
        "Flambda_norm": float(jnp.linalg.norm(flambda)),
        "Fc_norm": float(jnp.linalg.norm(fc)),
        "Ftot_abs": float(jnp.abs(ftot)),
    }


def _evaluate_pdipm_state(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    rho: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    hvector: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
    *,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Dict[str, Any]:
    """Evaluate PDIPM and primal residual diagnostics at a given state."""

    gk = _compute_gk(
        state.temperature,
        ln_nk,
        ln_ntot,
        hvector,
        state.ln_normalized_pressure,
    )
    pdipm_metrics = _solve_full_pdipm_gie_direction_with_metrics(
        ln_nk,
        ln_mk,
        rho,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        epsilon,
    )
    pdipm_residuals = _compute_pdipm_barrier_residual_components(
        ln_nk,
        ln_mk,
        rho,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        pdipm_metrics["pi_vector"],
        epsilon,
    )
    pdipm_total_residual = _pdipm_total_residual_norm(pdipm_residuals)

    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntot = jnp.exp(ln_ntot)
    invalid_numbers_detected = bool(
        _contains_invalid_numbers(ln_nk, ln_mk, rho, ln_ntot, nk, mk, ntot, gk)
    )
    if invalid_numbers_detected:
        primal_residual_metrics = {
            "fresh_residual": jnp.asarray(jnp.nan, dtype=ntot.dtype),
            "element_balance_residual_norm": jnp.asarray(jnp.nan, dtype=ntot.dtype),
            "ntot_residual": jnp.asarray(jnp.nan, dtype=ntot.dtype),
            "gas_stationarity_residual_norm": jnp.asarray(jnp.nan, dtype=ntot.dtype),
            "cond_stationarity_residual_norm": jnp.asarray(jnp.nan, dtype=ntot.dtype),
        }
    else:
        pi_vector_resid = _recompute_pi_for_residual(
            nk,
            mk,
            ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            gk,
            hvector_cond,
            epsilon,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
        )
        primal_residual_metrics = _compute_residual_component_metrics(
            nk,
            mk,
            ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            gk,
            hvector_cond,
            jnp.exp(epsilon),
            pi_vector_resid,
        )

    return {
        "gk": gk,
        "pdipm_metrics": pdipm_metrics,
        "pdipm_residuals": pdipm_residuals,
        "pdipm_total_residual": pdipm_total_residual,
        "primal_residual_metrics": primal_residual_metrics,
        "invalid_numbers_detected": invalid_numbers_detected,
    }


def _evaluate_direction_lambda_grid(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    hvector: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
    delta_ln_nk: jnp.ndarray,
    raw_delta_ln_mk: jnp.ndarray,
    delta_ln_ntot: float,
    *,
    delta_aux: Optional[jnp.ndarray] = None,
    aux_name: str = "delta_aux",
    lambda_trials: Sequence[float],
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> list[Dict[str, Any]]:
    """Evaluate a fixed direction on a shared lambda grid."""

    clipped_delta_ln_mk = jnp.clip(raw_delta_ln_mk, -0.1, 0.1)
    lam1_gas = stepsize_cea_gas(delta_ln_nk, delta_ln_ntot, ln_nk, ln_ntot)
    lam1_cond = stepsize_cond_heurstic(clipped_delta_ln_mk)
    lam2_cond = stepsize_sk(clipped_delta_ln_mk, ln_mk, epsilon)
    trials = []
    for lambda_trial in lambda_trials:
        lam = jnp.asarray(lambda_trial, dtype=jnp.asarray(ln_ntot).dtype)
        trial_ln_nk = jnp.asarray(ln_nk) + lam * jnp.asarray(delta_ln_nk)
        trial_ln_mk = jnp.asarray(ln_mk) + lam * jnp.asarray(clipped_delta_ln_mk)
        trial_ln_ntot = jnp.asarray(ln_ntot) + lam * jnp.asarray(delta_ln_ntot)
        trial_aux = None if delta_aux is None else lam * jnp.asarray(delta_aux)
        trial_nk = jnp.exp(trial_ln_nk)
        trial_mk = jnp.exp(trial_ln_mk)
        trial_ntot = jnp.exp(trial_ln_ntot)
        trial_gk = _compute_gk(
            state.temperature,
            trial_ln_nk,
            trial_ln_ntot,
            hvector,
            state.ln_normalized_pressure,
        )
        invalid_numbers_detected = bool(
            _contains_invalid_numbers(
                trial_ln_nk,
                trial_ln_mk,
                trial_ln_ntot,
                trial_nk,
                trial_mk,
                trial_ntot,
                trial_gk,
                delta_ln_nk,
                raw_delta_ln_mk,
                clipped_delta_ln_mk,
                delta_ln_ntot,
                trial_aux if trial_aux is not None else jnp.asarray(0.0),
            )
        )
        if invalid_numbers_detected:
            residual_metrics = {
                "fresh_residual": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
                "element_balance_residual_norm": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
                "ntot_residual": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
                "gas_stationarity_residual_norm": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
                "cond_stationarity_residual_norm": jnp.asarray(jnp.nan, dtype=trial_ntot.dtype),
            }
        else:
            pi_vector_resid = _recompute_pi_for_residual(
                trial_nk,
                trial_mk,
                trial_ntot,
                formula_matrix,
                formula_matrix_cond,
                b,
                trial_gk,
                hvector_cond,
                epsilon,
                reduced_solver=reduced_solver,
                regularization_mode=regularization_mode,
                regularization_strength=regularization_strength,
            )
            residual_metrics = _compute_residual_component_metrics(
                trial_nk,
                trial_mk,
                trial_ntot,
                formula_matrix,
                formula_matrix_cond,
                b,
                trial_gk,
                hvector_cond,
                jnp.exp(epsilon),
                pi_vector_resid,
            )
        record = {
            "lambda_trial": float(lambda_trial),
            "fresh_post_update_residual": float(residual_metrics["fresh_residual"]),
            "element_balance_residual_norm": float(
                residual_metrics["element_balance_residual_norm"]
            ),
            "ntot_residual": float(residual_metrics["ntot_residual"]),
            "invalid_numbers_detected": invalid_numbers_detected,
            "all_finite": not invalid_numbers_detected,
            "max_abs_delta_ln_nk": float(jnp.max(jnp.abs(lam * jnp.asarray(delta_ln_nk)))),
            "max_abs_delta_ln_mk": float(
                jnp.max(jnp.abs(lam * jnp.asarray(clipped_delta_ln_mk)))
            ),
            "max_abs_delta_ln_ntot": float(jnp.abs(lam * jnp.asarray(delta_ln_ntot))),
            "lam1_gas": float(lam1_gas),
            "lam1_cond": float(lam1_cond),
            "lam2_cond": float(lam2_cond),
        }
        if delta_aux is not None:
            record[f"max_abs_{aux_name}"] = float(jnp.max(jnp.abs(trial_aux)))
        trials.append(record)
    return trials


def trace_pdipm_fixed_epsilon_trajectory(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    *,
    rho_offset: float = 0.0,
    max_iter: int = 10,
    min_lambda: float = 1.0e-6,
    backtrack_factor: float = 0.5,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
    element_indices: Optional[jnp.ndarray] = None,
) -> Dict[str, Any]:
    """Diagnostic-only fixed-epsilon full-PDIPM trajectory with independent rho."""

    n_elements = formula_matrix.shape[0]
    b = (
        jnp.asarray(state.element_vector)
        if element_indices is None
        else jnp.asarray(state.element_vector)[jnp.asarray(element_indices)]
    )
    if b.shape[0] != n_elements:
        raise ValueError(
            "ThermoState.element_vector length does not match the number of element rows "
            f"in the formula matrices (got {b.shape[0]}, expected {n_elements}). "
            "Provide element_indices that map the state vector onto the reduced element set."
        )

    hvector = hvector_func(state.temperature)
    hvector_cond = hvector_cond_func(state.temperature)
    rho = epsilon - jnp.asarray(ln_mk) + jnp.asarray(rho_offset, dtype=jnp.asarray(ln_mk).dtype)
    ln_nk_state = jnp.asarray(ln_nk)
    ln_mk_state = jnp.asarray(ln_mk)
    ln_ntot_state = jnp.asarray(ln_ntot)
    history = []
    start = perf_counter()

    for iter_count in range(max_iter):
        eval_current = _evaluate_pdipm_state(
            state,
            ln_nk_state,
            ln_mk_state,
            rho,
            ln_ntot_state,
            formula_matrix,
            formula_matrix_cond,
            b,
            hvector,
            hvector_cond,
            epsilon,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
        )
        current_total = eval_current["pdipm_total_residual"]
        current_primal = float(eval_current["primal_residual_metrics"]["fresh_residual"])
        metrics = eval_current["pdipm_metrics"]

        trial_lambda = 1.0
        accepted = False
        accepted_eval = None
        while trial_lambda >= min_lambda:
            lam = jnp.asarray(trial_lambda, dtype=ln_nk_state.dtype)
            trial_ln_nk = ln_nk_state + lam * jnp.asarray(metrics["delta_ln_nk"])
            trial_ln_mk = ln_mk_state + lam * jnp.asarray(metrics["raw_delta_ln_mk"])
            trial_rho = rho + lam * jnp.asarray(metrics["delta_rho"])
            trial_ln_ntot = ln_ntot_state + lam * jnp.asarray(metrics["delta_ln_ntot"])
            accepted_eval = _evaluate_pdipm_state(
                state,
                trial_ln_nk,
                trial_ln_mk,
                trial_rho,
                trial_ln_ntot,
                formula_matrix,
                formula_matrix_cond,
                b,
                hvector,
                hvector_cond,
                epsilon,
                reduced_solver=reduced_solver,
                regularization_mode=regularization_mode,
                regularization_strength=regularization_strength,
            )
            if (
                not accepted_eval["invalid_numbers_detected"]
                and accepted_eval["pdipm_total_residual"] <= current_total
            ):
                accepted = True
                break
            trial_lambda *= backtrack_factor

        record = {
            "iter": iter_count,
            "pdipm_total_residual": current_total,
            "primal_fresh_residual": current_primal,
            "Fn_norm": eval_current["pdipm_residuals"]["Fn_norm"],
            "Fm_norm": eval_current["pdipm_residuals"]["Fm_norm"],
            "Flambda_norm": eval_current["pdipm_residuals"]["Flambda_norm"],
            "Fc_norm": eval_current["pdipm_residuals"]["Fc_norm"],
            "Ftot_abs": eval_current["pdipm_residuals"]["Ftot_abs"],
            "delta_ln_ntot": float(metrics["delta_ln_ntot"]),
            "pi_norm": float(metrics["pi_norm"]),
            "max_abs_delta_ln_nk": float(jnp.max(jnp.abs(metrics["delta_ln_nk"]))),
            "max_abs_delta_ln_mk": float(jnp.max(jnp.abs(metrics["raw_delta_ln_mk"]))),
            "max_abs_delta_rho": float(jnp.max(jnp.abs(metrics["delta_rho"]))),
            "chosen_lambda": float(trial_lambda if accepted else 0.0),
            "accepted": accepted,
            "accepted_total_residual": float(
                accepted_eval["pdipm_total_residual"] if accepted_eval is not None else jnp.nan
            ),
            "accepted_primal_fresh_residual": float(
                accepted_eval["primal_residual_metrics"]["fresh_residual"]
                if accepted_eval is not None
                else jnp.nan
            ),
            "invalid_numbers_detected": bool(
                eval_current["invalid_numbers_detected"]
                or (accepted_eval["invalid_numbers_detected"] if accepted_eval is not None else False)
            ),
        }
        history.append(record)
        if not accepted:
            break
        lam = jnp.asarray(trial_lambda, dtype=ln_nk_state.dtype)
        ln_nk_state = ln_nk_state + lam * jnp.asarray(metrics["delta_ln_nk"])
        ln_mk_state = ln_mk_state + lam * jnp.asarray(metrics["raw_delta_ln_mk"])
        rho = rho + lam * jnp.asarray(metrics["delta_rho"])
        ln_ntot_state = ln_ntot_state + lam * jnp.asarray(metrics["delta_ln_ntot"])

    runtime_seconds = perf_counter() - start
    total_residuals = [rec["pdipm_total_residual"] for rec in history]
    monotonically_decreasing = all(
        total_residuals[i + 1] <= total_residuals[i] + 1.0e-12
        for i in range(len(total_residuals) - 1)
    )
    return {
        "rho_offset": float(rho_offset),
        "rho_initialization": "rho0 = epsilon - ln_mk + rho_offset",
        "initial_fc_norm": float(history[0]["Fc_norm"]) if history else float("nan"),
        "runtime_seconds": runtime_seconds,
        "monotonically_decreasing_total_residual": monotonically_decreasing,
        "history": history,
        "final_state": {
            "ln_nk": ln_nk_state,
            "ln_mk": ln_mk_state,
            "rho": rho,
            "ln_ntot": ln_ntot_state,
        },
    }


def diagnose_full_vs_reduced_gie_direction(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    *,
    element_indices: Optional[jnp.ndarray] = None,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Dict[str, Any]:
    """Compare the current reduced-GIE direction against a full-GIE direction on one state."""

    n_elements = formula_matrix.shape[0]
    b = (
        jnp.asarray(state.element_vector)
        if element_indices is None
        else jnp.asarray(state.element_vector)[jnp.asarray(element_indices)]
    )
    if b.shape[0] != n_elements:
        raise ValueError(
            "ThermoState.element_vector length does not match the number of element rows "
            f"in the formula matrices (got {b.shape[0]}, expected {n_elements}). "
            "Provide element_indices that map the state vector onto the reduced element set."
        )

    hvector = hvector_func(state.temperature)
    hvector_cond = hvector_cond_func(state.temperature)
    gk = _compute_gk(
        state.temperature,
        ln_nk,
        ln_ntot,
        hvector,
        state.ln_normalized_pressure,
    )
    reduced_metrics = _compute_iteration_step_metrics(
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        epsilon,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )
    full_metrics = _solve_full_pipm_gie_direction_with_metrics(
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        epsilon,
    )

    reduced_delta_ln_nk = jnp.asarray(reduced_metrics["delta_ln_nk"])
    reduced_raw_delta_ln_mk = jnp.asarray(reduced_metrics["raw_delta_ln_mk"])
    reduced_clipped_delta_ln_mk = jnp.asarray(reduced_metrics["delta_ln_mk"])
    full_delta_ln_nk = jnp.asarray(full_metrics["delta_ln_nk"])
    full_raw_delta_ln_mk = jnp.asarray(full_metrics["raw_delta_ln_mk"])
    full_clipped_delta_ln_mk = jnp.clip(full_raw_delta_ln_mk, -0.1, 0.1)

    gas_diff = reduced_delta_ln_nk - full_delta_ln_nk
    raw_cond_diff = reduced_raw_delta_ln_mk - full_raw_delta_ln_mk
    clipped_cond_diff = reduced_clipped_delta_ln_mk - full_clipped_delta_ln_mk
    reduced_gas_norm = jnp.linalg.norm(reduced_delta_ln_nk)
    full_gas_norm = jnp.linalg.norm(full_delta_ln_nk)
    cosine_denom = jnp.maximum(reduced_gas_norm * full_gas_norm, 1.0e-300)
    gas_cosine_similarity = jnp.clip(
        jnp.dot(reduced_delta_ln_nk, full_delta_ln_nk) / cosine_denom,
        -1.0,
        1.0,
    )
    gas_angle_degrees = jnp.degrees(jnp.arccos(gas_cosine_similarity))

    reduced_step_eval = _evaluate_direction_with_existing_update_rule(
        state,
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        hvector,
        hvector_cond,
        epsilon,
        reduced_delta_ln_nk,
        reduced_raw_delta_ln_mk,
        reduced_metrics["delta_ln_ntot"],
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )
    full_step_eval = _evaluate_direction_with_existing_update_rule(
        state,
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        hvector,
        hvector_cond,
        epsilon,
        full_delta_ln_nk,
        full_raw_delta_ln_mk,
        full_metrics["delta_ln_ntot"],
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )

    full_materially_better = bool(
        full_step_eval["all_finite"]
        and reduced_step_eval["all_finite"]
        and full_step_eval["fresh_post_update_residual"]
        <= 0.95 * reduced_step_eval["fresh_post_update_residual"]
    )

    return {
        "epsilon": float(epsilon),
        "reduced_backend": reduced_solver,
        "reduced_regularization_mode": regularization_mode,
        "reduced_regularization_strength": float(regularization_strength),
        "raw_direction_comparison": {
            "reduced_raw_direction_norm": float(
                jnp.linalg.norm(
                    jnp.concatenate(
                        [
                            reduced_delta_ln_nk,
                            reduced_raw_delta_ln_mk,
                            jnp.atleast_1d(jnp.asarray(reduced_metrics["delta_ln_ntot"])),
                        ]
                    )
                )
            ),
            "full_raw_direction_norm": float(
                jnp.linalg.norm(
                    jnp.concatenate(
                        [
                            full_delta_ln_nk,
                            full_raw_delta_ln_mk,
                            jnp.atleast_1d(jnp.asarray(full_metrics["delta_ln_ntot"])),
                        ]
                    )
                )
            ),
            "reduced_clipped_direction_norm": float(
                jnp.linalg.norm(
                    jnp.concatenate(
                        [
                            reduced_delta_ln_nk,
                            reduced_clipped_delta_ln_mk,
                            jnp.atleast_1d(jnp.asarray(reduced_metrics["delta_ln_ntot"])),
                        ]
                    )
                )
            ),
            "full_clipped_direction_norm": float(
                jnp.linalg.norm(
                    jnp.concatenate(
                        [
                            full_delta_ln_nk,
                            full_clipped_delta_ln_mk,
                            jnp.atleast_1d(jnp.asarray(full_metrics["delta_ln_ntot"])),
                        ]
                    )
                )
            ),
            "gas_cosine_similarity": float(gas_cosine_similarity),
            "gas_angle_degrees": float(gas_angle_degrees),
            "max_abs_delta_ln_nk_diff": float(jnp.max(jnp.abs(gas_diff))),
            "max_abs_raw_delta_ln_mk_diff": float(jnp.max(jnp.abs(raw_cond_diff))),
            "max_abs_clipped_delta_ln_mk_diff": float(jnp.max(jnp.abs(clipped_cond_diff))),
        },
        "reduced_direction": {
            "pi_norm": float(reduced_metrics["pi_norm"]),
            "delta_ln_ntot": float(reduced_metrics["delta_ln_ntot"]),
            "raw_direction_norm": reduced_step_eval["raw_direction_norm"],
            "clipped_direction_norm": reduced_step_eval["clipped_direction_norm"],
            "gas_direction_norm": reduced_step_eval["gas_direction_norm"],
            "raw_cond_direction_norm": reduced_step_eval["raw_cond_direction_norm"],
            "clipped_cond_direction_norm": reduced_step_eval["clipped_cond_direction_norm"],
            "lam1_gas": reduced_step_eval["lam1_gas"],
            "lam1_cond": reduced_step_eval["lam1_cond"],
            "lam2_cond": reduced_step_eval["lam2_cond"],
            "lam": reduced_step_eval["lam"],
            "fresh_post_update_residual": reduced_step_eval["fresh_post_update_residual"],
            "invalid_numbers_detected": reduced_step_eval["invalid_numbers_detected"],
            "max_abs_delta_ln_nk": reduced_step_eval["max_abs_delta_ln_nk"],
            "max_abs_raw_delta_ln_mk": reduced_step_eval["max_abs_raw_delta_ln_mk"],
            "max_abs_clipped_delta_ln_mk": reduced_step_eval["max_abs_clipped_delta_ln_mk"],
        },
        "full_gie_direction": {
            "factorization_succeeded": full_metrics["factorization_succeeded"],
            "runtime_seconds": full_metrics["runtime_seconds"],
            "pi_norm": full_metrics["pi_norm"],
            "delta_ln_ntot": float(full_metrics["delta_ln_ntot"]),
            "raw_direction_norm": full_step_eval["raw_direction_norm"],
            "clipped_direction_norm": full_step_eval["clipped_direction_norm"],
            "gas_direction_norm": full_step_eval["gas_direction_norm"],
            "raw_cond_direction_norm": full_step_eval["raw_cond_direction_norm"],
            "clipped_cond_direction_norm": full_step_eval["clipped_cond_direction_norm"],
            "lam1_gas": full_step_eval["lam1_gas"],
            "lam1_cond": full_step_eval["lam1_cond"],
            "lam2_cond": full_step_eval["lam2_cond"],
            "lam": full_step_eval["lam"],
            "fresh_post_update_residual": full_step_eval["fresh_post_update_residual"],
            "invalid_numbers_detected": full_step_eval["invalid_numbers_detected"],
            "max_abs_delta_ln_nk": full_step_eval["max_abs_delta_ln_nk"],
            "max_abs_raw_delta_ln_mk": full_step_eval["max_abs_raw_delta_ln_mk"],
            "max_abs_clipped_delta_ln_mk": full_step_eval["max_abs_clipped_delta_ln_mk"],
        },
        "full_gie_materially_better": full_materially_better,
        "material_better_criterion": "full fresh_post_update_residual <= 0.95 * reduced fresh_post_update_residual with both finite",
    }


def diagnose_pdipm_vs_pipm_direction(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    *,
    element_indices: Optional[jnp.ndarray] = None,
    lambda_trials: Optional[Sequence[float]] = None,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Dict[str, Any]:
    """Compare the current PIPM direction against an experimental full PDIPM direction."""

    n_elements = formula_matrix.shape[0]
    b = (
        jnp.asarray(state.element_vector)
        if element_indices is None
        else jnp.asarray(state.element_vector)[jnp.asarray(element_indices)]
    )
    if b.shape[0] != n_elements:
        raise ValueError(
            "ThermoState.element_vector length does not match the number of element rows "
            f"in the formula matrices (got {b.shape[0]}, expected {n_elements}). "
            "Provide element_indices that map the state vector onto the reduced element set."
        )

    hvector = hvector_func(state.temperature)
    hvector_cond = hvector_cond_func(state.temperature)
    gk = _compute_gk(
        state.temperature,
        ln_nk,
        ln_ntot,
        hvector,
        state.ln_normalized_pressure,
    )
    pipm_metrics = _compute_iteration_step_metrics(
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        epsilon,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )
    pdipm_metrics = _solve_full_pdipm_gie_direction_with_metrics(
        ln_nk,
        ln_mk,
        None,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        epsilon,
    )

    pipm_delta_ln_nk = jnp.asarray(pipm_metrics["delta_ln_nk"])
    pipm_raw_delta_ln_mk = jnp.asarray(pipm_metrics["raw_delta_ln_mk"])
    pdipm_delta_ln_nk = jnp.asarray(pdipm_metrics["delta_ln_nk"])
    pdipm_raw_delta_ln_mk = jnp.asarray(pdipm_metrics["raw_delta_ln_mk"])
    pipm_clipped_delta_ln_mk = jnp.asarray(pipm_metrics["delta_ln_mk"])
    pdipm_clipped_delta_ln_mk = jnp.clip(pdipm_raw_delta_ln_mk, -0.1, 0.1)

    pipm_gas_norm = jnp.linalg.norm(pipm_delta_ln_nk)
    pdipm_gas_norm = jnp.linalg.norm(pdipm_delta_ln_nk)
    gas_cosine_similarity = jnp.clip(
        jnp.dot(pipm_delta_ln_nk, pdipm_delta_ln_nk)
        / jnp.maximum(pipm_gas_norm * pdipm_gas_norm, 1.0e-300),
        -1.0,
        1.0,
    )
    gas_angle_degrees = jnp.degrees(jnp.arccos(gas_cosine_similarity))

    pipm_heuristic_lambda = float(pipm_metrics["lam"])
    if lambda_trials is None:
        lambda_trials = [
            pipm_heuristic_lambda,
            0.5 * pipm_heuristic_lambda,
            0.1 * pipm_heuristic_lambda,
            1.0e-4,
        ]
    lambda_grid = _build_trial_lambda_grid(
        pipm_heuristic_lambda,
        lambda_trials=lambda_trials,
    )

    pipm_trials = _evaluate_direction_lambda_grid(
        state,
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        hvector,
        hvector_cond,
        epsilon,
        pipm_delta_ln_nk,
        pipm_raw_delta_ln_mk,
        pipm_metrics["delta_ln_ntot"],
        lambda_trials=lambda_grid,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )
    pdipm_trials = _evaluate_direction_lambda_grid(
        state,
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        hvector,
        hvector_cond,
        epsilon,
        pdipm_delta_ln_nk,
        pdipm_raw_delta_ln_mk,
        pdipm_metrics["delta_ln_ntot"],
        delta_aux=pdipm_metrics["delta_rho"],
        aux_name="delta_rho",
        lambda_trials=lambda_grid,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )

    ln_eta = epsilon - jnp.asarray(ln_mk)
    pdipm_current_residuals = _compute_pdipm_barrier_residual_components(
        ln_nk,
        ln_mk,
        ln_eta,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        pdipm_metrics["pi_vector"],
        epsilon,
    )

    best_pipm_trial = min(
        pipm_trials, key=lambda rec: float("inf") if not rec["all_finite"] else rec["fresh_post_update_residual"]
    )
    best_pdipm_trial = min(
        pdipm_trials, key=lambda rec: float("inf") if not rec["all_finite"] else rec["fresh_post_update_residual"]
    )
    pdipm_materially_better = bool(
        best_pdipm_trial["all_finite"]
        and best_pipm_trial["all_finite"]
        and best_pdipm_trial["fresh_post_update_residual"]
        <= 0.95 * best_pipm_trial["fresh_post_update_residual"]
    )

    return {
        "epsilon": float(epsilon),
        "rho_initialization": "rho = epsilon - ln_mk, eta = exp(rho) = exp(epsilon - ln_mk)",
        "lambda_grid": [float(x) for x in lambda_grid],
        "direction_comparison": {
            "pipm_raw_direction_norm": float(
                jnp.linalg.norm(
                    jnp.concatenate(
                        [
                            pipm_delta_ln_nk,
                            pipm_raw_delta_ln_mk,
                            jnp.atleast_1d(jnp.asarray(pipm_metrics["delta_ln_ntot"])),
                        ]
                    )
                )
            ),
            "pdipm_raw_direction_norm": float(
                jnp.linalg.norm(
                    jnp.concatenate(
                        [
                            pdipm_delta_ln_nk,
                            pdipm_raw_delta_ln_mk,
                            jnp.ravel(jnp.asarray(pdipm_metrics["delta_rho"])),
                            jnp.atleast_1d(jnp.asarray(pdipm_metrics["delta_ln_ntot"])),
                        ]
                    )
                )
            ),
            "pipm_clipped_direction_norm": float(
                jnp.linalg.norm(
                    jnp.concatenate(
                        [
                            pipm_delta_ln_nk,
                            pipm_clipped_delta_ln_mk,
                            jnp.atleast_1d(jnp.asarray(pipm_metrics["delta_ln_ntot"])),
                        ]
                    )
                )
            ),
            "pdipm_clipped_direction_norm": float(
                jnp.linalg.norm(
                    jnp.concatenate(
                        [
                            pdipm_delta_ln_nk,
                            pdipm_clipped_delta_ln_mk,
                            jnp.ravel(jnp.asarray(pdipm_metrics["delta_rho"])),
                            jnp.atleast_1d(jnp.asarray(pdipm_metrics["delta_ln_ntot"])),
                        ]
                    )
                )
            ),
            "gas_cosine_similarity": float(gas_cosine_similarity),
            "gas_angle_degrees": float(gas_angle_degrees),
            "max_abs_delta_ln_nk_diff": float(
                jnp.max(jnp.abs(pipm_delta_ln_nk - pdipm_delta_ln_nk))
            ),
            "max_abs_raw_delta_ln_mk_diff": float(
                jnp.max(jnp.abs(pipm_raw_delta_ln_mk - pdipm_raw_delta_ln_mk))
            ),
            "max_abs_clipped_delta_ln_mk_diff": float(
                jnp.max(jnp.abs(pipm_clipped_delta_ln_mk - pdipm_clipped_delta_ln_mk))
            ),
            "abs_delta_ln_ntot_diff": float(
                jnp.abs(pipm_metrics["delta_ln_ntot"] - pdipm_metrics["delta_ln_ntot"])
            ),
        },
        "pipm_direction": {
            "pi_norm": float(pipm_metrics["pi_norm"]),
            "delta_ln_ntot": float(pipm_metrics["delta_ln_ntot"]),
            "lam1_gas": float(pipm_metrics["lam1_gas"]),
            "lam1_cond": float(pipm_metrics["lam1_cond"]),
            "lam2_cond": float(pipm_metrics["lam2_cond"]),
            "lam": float(pipm_metrics["lam"]),
            "max_abs_delta_ln_nk": float(jnp.max(jnp.abs(pipm_delta_ln_nk))),
            "max_abs_raw_delta_ln_mk": float(jnp.max(jnp.abs(pipm_raw_delta_ln_mk))),
            "max_abs_clipped_delta_ln_mk": float(jnp.max(jnp.abs(pipm_clipped_delta_ln_mk))),
            "lambda_trials": pipm_trials,
        },
        "pdipm_direction": {
            "factorization_succeeded": pdipm_metrics["factorization_succeeded"],
            "runtime_seconds": pdipm_metrics["runtime_seconds"],
            "pi_norm": pdipm_metrics["pi_norm"],
            "delta_ln_ntot": float(pdipm_metrics["delta_ln_ntot"]),
            "max_abs_delta_ln_nk": float(jnp.max(jnp.abs(pdipm_delta_ln_nk))),
            "max_abs_raw_delta_ln_mk": float(jnp.max(jnp.abs(pdipm_raw_delta_ln_mk))),
            "max_abs_clipped_delta_ln_mk": float(jnp.max(jnp.abs(pdipm_clipped_delta_ln_mk))),
            "max_abs_delta_rho": float(jnp.max(jnp.abs(pdipm_metrics["delta_rho"]))),
            "pdipm_barrier_residuals_current": pdipm_current_residuals,
            "lambda_trials": pdipm_trials,
        },
        "pdipm_materially_better": pdipm_materially_better,
        "material_better_criterion": "best PDIPM fresh_post_update_residual on shared lambda grid <= 0.95 * best PIPM fresh_post_update_residual with both finite",
    }


def diagnose_pdipm_vs_pipm_fixed_epsilon_trajectories(
    state: ThermoState,
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    *,
    rho_offsets: Sequence[float] = (0.0, 1.0, -1.0),
    max_iter: int = 10,
    min_lambda: float = 1.0e-6,
    backtrack_factor: float = 0.5,
    element_indices: Optional[jnp.ndarray] = None,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Dict[str, Any]:
    """Compare fixed-epsilon full-PDIPM trajectories against the current PIPM trajectory."""

    pipm_trace = trace_minimize_gibbs_cond_iterations(
        state,
        ln_nk_init=ln_nk,
        ln_mk_init=ln_mk,
        ln_ntot_init=ln_ntot,
        formula_matrix=formula_matrix,
        formula_matrix_cond=formula_matrix_cond,
        hvector_func=hvector_func,
        hvector_cond_func=hvector_cond_func,
        epsilon=epsilon,
        max_iter=max_iter,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )
    pipm_history = pipm_trace["history"]
    pipm_best_residual = min(
        (rec["residual"] for rec in pipm_history),
        default=float("inf"),
    )

    pdipm_runs = []
    for rho_offset in rho_offsets:
        run = trace_pdipm_fixed_epsilon_trajectory(
            state,
            ln_nk,
            ln_mk,
            ln_ntot,
            formula_matrix,
            formula_matrix_cond,
            hvector_func,
            hvector_cond_func,
            epsilon,
            rho_offset=rho_offset,
            max_iter=max_iter,
            min_lambda=min_lambda,
            backtrack_factor=backtrack_factor,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
            element_indices=element_indices,
        )
        best_primal_residual = min(
            (rec["accepted_primal_fresh_residual"] for rec in run["history"] if rec["accepted"]),
            default=float("inf"),
        )
        run["best_primal_residual"] = best_primal_residual
        run["beats_pipm_within_budget"] = bool(best_primal_residual < pipm_best_residual)
        pdipm_runs.append(run)

    return {
        "epsilon": float(epsilon),
        "previous_one_step_center_path_note": "the earlier one-step PDIPM diagnostic used rho = epsilon - ln_mk, so Fc = ln_mk + rho - epsilon was zero at the initial state",
        "pipm_trace": {
            "n_iter": len(pipm_history),
            "best_residual": pipm_best_residual,
            "history": pipm_history,
        },
        "pdipm_runs": pdipm_runs,
    }


def _recompute_pi_for_residual(
    nk: jnp.ndarray,
    mk: jnp.ndarray,
    ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
    *,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> jnp.ndarray:
    """Re-solve the reduced system on the current state for residual evaluation only.

    This solve is intentionally separate from the earlier solve that produced the
    update direction. It is not fed back into the primal update; it is only used
    to evaluate a post-update barrier residual on a self-consistent state.
    """

    bk = formula_matrix @ nk
    sk = mk * mk * jnp.exp(-epsilon)
    pi_vector, _delta_ln_ntot = solve_reduced_gibbs_iteration_equations_cond(
        nk,
        mk,
        ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        bk,
        hvector_cond,
        sk,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )
    return pi_vector


def _compute_iteration_step_metrics(
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
    *,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
    include_system_trace: bool = False,
    reduced_coupling_mode: str = "current",
    reduced_coupling_alpha_s: float = 1.0,
) -> Dict[str, jnp.ndarray]:
    """Compute the current PIPM step components without changing the update rule."""

    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntot = jnp.exp(ln_ntot)
    ln_sk = 2.0 * ln_mk - epsilon
    bk = formula_matrix @ nk

    if reduced_coupling_mode == "current":
        pi_vector, delta_ln_ntot, reduced_metrics = _solve_reduced_gibbs_iteration_equations_cond_with_metrics(
            nk,
            mk,
            ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            gk,
            bk,
            hvector_cond,
            jnp.exp(ln_sk),
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
            include_system_trace=include_system_trace,
        )
        raw_delta_ln_mk = jnp.exp(ln_mk - epsilon) * (
            formula_matrix_cond.T @ pi_vector - hvector_cond
        ) + 1.0
        reduced_metrics = dict(reduced_metrics)
        reduced_metrics["reduced_coupling_mode"] = "current"
        reduced_metrics["reduced_coupling_alpha_s"] = jnp.asarray(
            reduced_coupling_alpha_s, dtype=jnp.float64
        )
        reduced_metrics["reduced_coupling_uses_capped_s"] = jnp.asarray(False)
    elif reduced_coupling_mode == "capped_s_only":
        direction = solve_inventory_capped_reduced_coupling_direction(
            ln_nk,
            ln_mk,
            ln_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            gk,
            hvector_cond,
            epsilon,
            variant_name="capped_s_only",
            alpha_coupling=reduced_coupling_alpha_s,
        )
        pi_vector = direction["pi_vector"]
        delta_ln_ntot = direction["delta_ln_ntot"]
        raw_delta_ln_mk = direction["raw_delta_ln_mk"]
        reduced_metrics = {
            "reduced_solver_backend": "inventory_capped_capped_s_only",
            "reduced_factorization_succeeded": direction["factorization_succeeded"],
            "reduced_regularization_mode": regularization_mode,
            "reduced_regularization_strength": jnp.asarray(
                regularization_strength, dtype=jnp.float64
            ),
            "reduced_regularization_used": jnp.asarray(0.0, dtype=jnp.float64),
            "reduced_resn": direction["resn"],
            "reduced_row_scale_min": jnp.asarray(jnp.nan, dtype=jnp.float64),
            "reduced_row_scale_max": jnp.asarray(jnp.nan, dtype=jnp.float64),
            "reduced_row_scale_ratio": jnp.asarray(jnp.nan, dtype=jnp.float64),
            "reduced_col_scale_min": jnp.asarray(jnp.nan, dtype=jnp.float64),
            "reduced_col_scale_max": jnp.asarray(jnp.nan, dtype=jnp.float64),
            "reduced_col_scale_ratio": jnp.asarray(jnp.nan, dtype=jnp.float64),
            "reduced_mat_maxabs": jnp.max(jnp.abs(direction["assemble_mat"])),
            "reduced_vec_maxabs": jnp.max(jnp.abs(direction["assemble_vec"])),
            "reduced_qk_maxabs": jnp.max(jnp.abs(direction["q_block"])),
            "reduced_qk_diag_min": jnp.min(jnp.diag(direction["q_block"])),
            "reduced_qk_diag_max": jnp.max(jnp.diag(direction["q_block"])),
            "reduced_coupling_mode": "capped_s_only",
            "reduced_coupling_alpha_s": jnp.asarray(
                reduced_coupling_alpha_s, dtype=jnp.float64
            ),
            "reduced_coupling_uses_capped_s": jnp.asarray(True),
            "reduced_coupling_capped_count": direction["capped_count"],
        }
    elif reduced_coupling_mode in (
        "candidate_selected_active_only",
        "candidate_selected_active_plus_near_jacobian",
        "candidate_selected_weighted_mask",
    ):
        direction = solve_hybrid_candidate_selected_reduced_coupling_direction(
            ln_nk,
            ln_mk,
            ln_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            gk,
            hvector_cond,
            epsilon,
            candidate_mode=reduced_coupling_mode,
        )
        pi_vector = direction["pi_vector"]
        delta_ln_ntot = direction["delta_ln_ntot"]
        raw_delta_ln_mk = direction["raw_delta_ln_mk"]
        reduced_metrics = {
            "reduced_solver_backend": "hybrid_candidate_selected",
            "reduced_factorization_succeeded": direction["factorization_succeeded"],
            "reduced_regularization_mode": regularization_mode,
            "reduced_regularization_strength": jnp.asarray(
                regularization_strength, dtype=jnp.float64
            ),
            "reduced_regularization_used": jnp.asarray(0.0, dtype=jnp.float64),
            "reduced_resn": direction["resn"],
            "reduced_row_scale_min": jnp.asarray(jnp.nan, dtype=jnp.float64),
            "reduced_row_scale_max": jnp.asarray(jnp.nan, dtype=jnp.float64),
            "reduced_row_scale_ratio": jnp.asarray(jnp.nan, dtype=jnp.float64),
            "reduced_col_scale_min": jnp.asarray(jnp.nan, dtype=jnp.float64),
            "reduced_col_scale_max": jnp.asarray(jnp.nan, dtype=jnp.float64),
            "reduced_col_scale_ratio": jnp.asarray(jnp.nan, dtype=jnp.float64),
            "reduced_mat_maxabs": jnp.max(jnp.abs(direction["assemble_mat"])),
            "reduced_vec_maxabs": jnp.max(jnp.abs(direction["assemble_vec"])),
            "reduced_qk_maxabs": jnp.max(jnp.abs(direction["q_block"])),
            "reduced_qk_diag_min": jnp.min(jnp.diag(direction["q_block"])),
            "reduced_qk_diag_max": jnp.max(jnp.diag(direction["q_block"])),
            "reduced_coupling_mode": reduced_coupling_mode,
            "reduced_coupling_alpha_s": jnp.asarray(
                reduced_coupling_alpha_s, dtype=jnp.float64
            ),
            "reduced_coupling_uses_capped_s": jnp.asarray(False),
            "hybrid_candidate_activity_proxy_source": direction[
                "activity_proxy_source"
            ],
            "hybrid_candidate_atomic_density_proxy_available": direction[
                "atomic_density_proxy_available"
            ],
            "hybrid_candidate_set_size": direction["candidate_set_size"],
            "hybrid_candidate_near_active_set_size": direction["near_active_set_size"],
            "hybrid_candidate_weighted_mask": direction["weighted_mask"],
            "hybrid_candidate_max_log_activity_proxy": jnp.max(
                direction["log_activity_proxy"]
            ),
            "hybrid_candidate_min_log_activity_proxy": jnp.min(
                direction["log_activity_proxy"]
            ),
        }
    else:
        raise ValueError(
            "Unknown reduced_coupling_mode "
            f"'{reduced_coupling_mode}'. Expected 'current', 'capped_s_only', "
            "or a candidate-selected hybrid mode."
        )

    delta_ln_nk = formula_matrix.T @ pi_vector + delta_ln_ntot - gk

    max_step_m = 0.1
    delta_ln_mk = jnp.clip(raw_delta_ln_mk, -max_step_m, max_step_m)

    lam1_gas = stepsize_cea_gas(delta_ln_nk, delta_ln_ntot, ln_nk, ln_ntot)
    lam1_cond = stepsize_cond_heurstic(delta_ln_mk)
    lam2_cond = stepsize_sk(delta_ln_mk, ln_mk, epsilon)
    lam = jnp.minimum(1.0, jnp.minimum(lam1_gas, jnp.minimum(lam1_cond, lam2_cond)))
    lam = jnp.clip(lam, 0.0, 1.0)

    limiter_candidates = jnp.asarray([1.0, lam1_gas, lam1_cond, lam2_cond])
    limiting_index = jnp.argmin(limiter_candidates).astype(jnp.int32)

    metrics = {
        "pi_vector": pi_vector,
        "delta_ln_ntot": delta_ln_ntot,
        "delta_ln_nk": delta_ln_nk,
        "raw_delta_ln_mk": raw_delta_ln_mk,
        "delta_ln_mk": delta_ln_mk,
        "lam1_gas": lam1_gas,
        "lam1_cond": lam1_cond,
        "lam2_cond": lam2_cond,
        "lam": lam,
        "limiting_index": limiting_index,
        "pi_norm": jnp.linalg.norm(pi_vector),
        "max_abs_delta_ln_nk": jnp.max(jnp.abs(delta_ln_nk)),
        "max_abs_raw_delta_ln_mk": jnp.max(jnp.abs(raw_delta_ln_mk)),
        "max_abs_clipped_delta_ln_mk": jnp.max(jnp.abs(delta_ln_mk)),
    }
    metrics.update(reduced_metrics)
    return metrics


def _evaluate_trial_step(
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    lam: float,
    delta_ln_nk: jnp.ndarray,
    delta_ln_mk: jnp.ndarray,
    delta_ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    temperature: float,
    ln_normalized_pressure: float,
    hvector: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
    *,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
    budget_guard_enabled: bool = False,
    budget_margin: float = 0.0,
    charge_row_index: Optional[int] = None,
) -> Dict[str, jnp.ndarray]:
    """Evaluate one damped trial step on a fresh self-consistent residual."""

    lam = jnp.asarray(lam, dtype=jnp.asarray(ln_ntot).dtype)
    trial_ln_nk = jnp.asarray(ln_nk) + lam * jnp.asarray(delta_ln_nk)
    trial_ln_mk = jnp.asarray(ln_mk) + lam * jnp.asarray(delta_ln_mk)
    trial_ln_ntot = jnp.asarray(ln_ntot) + lam * jnp.asarray(delta_ln_ntot)

    trial_nk = jnp.exp(trial_ln_nk)
    trial_mk = jnp.exp(trial_ln_mk)
    trial_ntot = jnp.exp(trial_ln_ntot)
    trial_gk = _compute_gk(
        temperature,
        trial_ln_nk,
        trial_ln_ntot,
        hvector,
        ln_normalized_pressure,
    )
    trial_An = formula_matrix @ trial_nk
    trial_Am = formula_matrix_cond @ trial_mk
    budget_guard_passed = jnp.asarray(True)
    if budget_guard_enabled:
        budget_guard_passed = budget_guard_accepts_condensate_burden(
            formula_matrix_cond,
            trial_mk,
            b,
            budget_margin=budget_margin,
        )

    invalid_state = _contains_invalid_numbers(
        trial_ln_nk,
        trial_ln_mk,
        trial_ln_ntot,
        trial_nk,
        trial_mk,
        trial_ntot,
        trial_gk,
        trial_An,
        trial_Am,
    ) | (~budget_guard_passed)

    pi_placeholder = jnp.full_like(b, jnp.nan)
    residual_placeholder = jnp.asarray(jnp.inf, dtype=trial_ntot.dtype)
    component_placeholder = jnp.asarray(jnp.inf, dtype=trial_ntot.dtype)
    charge_placeholder = jnp.asarray(jnp.inf, dtype=trial_ntot.dtype)

    def _eval_valid(_):
        pi_vector_resid = _recompute_pi_for_residual(
            trial_nk,
            trial_mk,
            trial_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            trial_gk,
            hvector_cond,
            epsilon,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
        )
        residual = _compute_residuals(
            trial_nk,
            trial_mk,
            trial_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            trial_gk,
            hvector_cond,
            jnp.exp(epsilon),
            trial_An,
            trial_Am,
            pi_vector_resid,
        )
        component_metrics = _compute_residual_component_metrics(
            trial_nk,
            trial_mk,
            trial_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            trial_gk,
            hvector_cond,
            jnp.exp(epsilon),
            pi_vector_resid,
        )
        charge_residual = charge_placeholder
        if charge_row_index is not None:
            charge_index = jnp.asarray(int(charge_row_index), dtype=jnp.int32)
            charge_raw = trial_An[charge_index] + trial_Am[charge_index] - b[charge_index]
            charge_scale = jnp.maximum(jnp.abs(b[charge_index]), jnp.asarray(1.0, dtype=trial_ntot.dtype))
            charge_residual = charge_raw / charge_scale
        residual_is_finite = jnp.isfinite(residual) & (~_contains_invalid_numbers(pi_vector_resid))
        residual = jnp.where(residual_is_finite, residual, residual_placeholder)
        return (
            pi_vector_resid,
            residual,
            residual_is_finite,
            component_metrics["element_balance_residual_norm"],
            component_metrics["ntot_residual"],
            component_metrics["gas_stationarity_residual_norm"],
            component_metrics["cond_stationarity_residual_norm"],
            charge_residual,
        )

    (
        pi_vector_resid,
        fresh_residual,
        all_finite,
        element_balance_residual_norm,
        ntot_residual,
        gas_stationarity_residual_norm,
        cond_stationarity_residual_norm,
        zero_charge_residual,
    ) = cond(
        invalid_state,
        lambda _: (
            pi_placeholder,
            residual_placeholder,
            jnp.asarray(False),
            component_placeholder,
            component_placeholder,
            component_placeholder,
            component_placeholder,
            charge_placeholder,
        ),
        _eval_valid,
        operand=0,
    )

    return {
        "lam": lam,
        "ln_nk": trial_ln_nk,
        "ln_mk": trial_ln_mk,
        "ln_ntot": trial_ln_ntot,
        "nk": trial_nk,
        "mk": trial_mk,
        "ntot": trial_ntot,
        "gk": trial_gk,
        "An": trial_An,
        "Am": trial_Am,
        "pi_vector_resid": pi_vector_resid,
        "fresh_residual": fresh_residual,
        "element_balance_residual_norm": element_balance_residual_norm,
        "ntot_residual": ntot_residual,
        "gas_stationarity_residual_norm": gas_stationarity_residual_norm,
        "cond_stationarity_residual_norm": cond_stationarity_residual_norm,
        "zero_charge_residual": zero_charge_residual,
        "all_finite": all_finite,
        "budget_guard_passed": budget_guard_passed,
        "budget_guard_rejected": ~budget_guard_passed,
    }


def _choose_lambda_by_residual_backtracking(
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: float,
    current_gk: jnp.ndarray,
    current_An: jnp.ndarray,
    current_Am: jnp.ndarray,
    current_residual: float,
    lam_init: float,
    delta_ln_nk: jnp.ndarray,
    delta_ln_mk: jnp.ndarray,
    delta_ln_ntot: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    temperature: float,
    ln_normalized_pressure: float,
    hvector: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
    *,
    beta: float = 0.5,
    max_backtracks: int = 8,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
    budget_guard_enabled: bool = False,
    budget_margin: float = 0.0,
    line_search_selection_policy: str = "first_monotone_with_best_finite_fallback",
    line_search_charge_row_index: Optional[int] = None,
    line_search_charge_weight: float = 1.0,
) -> Dict[str, jnp.ndarray]:
    """Choose the damped step from fresh residuals on backtracked trial states."""

    allowed_policies = (
        "first_monotone_with_best_finite_fallback",
        "best_finite",
        "best_finite_nonincreasing",
        "charge_aware_composite",
        "charge_aware_composite_nonincreasing",
        "component_composite_nonincreasing",
    )
    if line_search_selection_policy not in allowed_policies:
        raise ValueError(
            "Unknown line_search_selection_policy "
            f"'{line_search_selection_policy}'. Expected one of {allowed_policies}."
        )

    dtype = jnp.asarray(ln_ntot).dtype
    int_dtype = jnp.int32
    lam_init = jnp.asarray(lam_init, dtype=dtype)
    lam_init = jnp.where(jnp.isfinite(lam_init), jnp.clip(lam_init, 0.0, 1.0), 0.0)
    beta = jnp.asarray(beta, dtype=dtype)
    current_residual = jnp.asarray(current_residual, dtype=dtype)
    inf_value = jnp.asarray(jnp.inf, dtype=dtype)
    current_composite_merit = inf_value
    if line_search_selection_policy == "component_composite_nonincreasing":
        current_nk = jnp.exp(jnp.asarray(ln_nk))
        current_mk = jnp.exp(jnp.asarray(ln_mk))
        current_ntot = jnp.exp(jnp.asarray(ln_ntot))
        current_pi_vector_resid = _recompute_pi_for_residual(
            current_nk,
            current_mk,
            current_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            current_gk,
            hvector_cond,
            epsilon,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
        )
        current_components = _compute_residual_component_metrics(
            current_nk,
            current_mk,
            current_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            current_gk,
            hvector_cond,
            jnp.exp(epsilon),
            current_pi_vector_resid,
        )
        current_charge_term = jnp.asarray(0.0, dtype=dtype)
        if line_search_charge_row_index is not None:
            charge_index = int(line_search_charge_row_index)
            charge_raw = current_An[charge_index] + current_Am[charge_index] - b[charge_index]
            charge_scale = jnp.maximum(
                jnp.abs(b[charge_index]), jnp.asarray(1.0, dtype=dtype)
            )
            current_charge_term = (
                jnp.asarray(line_search_charge_weight, dtype=dtype)
                * jnp.abs(charge_raw / charge_scale)
            )
        current_composite_merit = (
            current_components["element_balance_residual_norm"]
            + jnp.abs(current_components["ntot_residual"])
            + current_components["gas_stationarity_residual_norm"]
            + current_components["cond_stationarity_residual_norm"]
            + current_charge_term
        )
        current_composite_merit = jnp.where(
            jnp.isfinite(current_composite_merit)
            & (~_contains_invalid_numbers(current_pi_vector_resid)),
            current_composite_merit,
            inf_value,
        )

    init_carry = {
        "accepted": jnp.asarray(False),
        "accept_index": jnp.asarray(0, dtype=int_dtype),
        "accept_residual": inf_value,
        "accept_lam": jnp.asarray(0.0, dtype=dtype),
        "accept_ln_nk": jnp.asarray(ln_nk),
        "accept_ln_mk": jnp.asarray(ln_mk),
        "accept_ln_ntot": jnp.asarray(ln_ntot),
        "accept_gk": jnp.asarray(current_gk),
        "accept_An": jnp.asarray(current_An),
        "accept_Am": jnp.asarray(current_Am),
        "best_found": jnp.asarray(False),
        "best_index": jnp.asarray(0, dtype=int_dtype),
        "best_residual": inf_value,
        "best_lam": jnp.asarray(0.0, dtype=dtype),
        "best_ln_nk": jnp.asarray(ln_nk),
        "best_ln_mk": jnp.asarray(ln_mk),
        "best_ln_ntot": jnp.asarray(ln_ntot),
        "best_gk": jnp.asarray(current_gk),
        "best_An": jnp.asarray(current_An),
        "best_Am": jnp.asarray(current_Am),
        "best_composite_found": jnp.asarray(False),
        "best_composite_index": jnp.asarray(0, dtype=int_dtype),
        "best_composite_residual": inf_value,
        "best_composite_merit": inf_value,
        "best_composite_lam": jnp.asarray(0.0, dtype=dtype),
        "best_composite_ln_nk": jnp.asarray(ln_nk),
        "best_composite_ln_mk": jnp.asarray(ln_mk),
        "best_composite_ln_ntot": jnp.asarray(ln_ntot),
        "best_composite_gk": jnp.asarray(current_gk),
        "best_composite_An": jnp.asarray(current_An),
        "best_composite_Am": jnp.asarray(current_Am),
        "budget_guard_rejection_count": jnp.asarray(0, dtype=int_dtype),
    }

    def _loop_body(i, carry):
        lam_trial = lam_init * jnp.power(beta, jnp.asarray(i, dtype=dtype))
        trial = _evaluate_trial_step(
            ln_nk,
            ln_mk,
            ln_ntot,
            lam_trial,
            delta_ln_nk,
            delta_ln_mk,
            delta_ln_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            temperature,
            ln_normalized_pressure,
            hvector,
            hvector_cond,
            epsilon,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
            budget_guard_enabled=budget_guard_enabled,
            budget_margin=budget_margin,
            charge_row_index=line_search_charge_row_index,
        )
        budget_guard_rejected = trial.get(
            "budget_guard_rejected",
            jnp.asarray(False, dtype=jnp.bool_),
        )
        carry = {
            **carry,
            "budget_guard_rejection_count": carry["budget_guard_rejection_count"]
            + budget_guard_rejected.astype(int_dtype),
        }

        finite_trial = jnp.isfinite(trial["fresh_residual"]) & trial["all_finite"]
        monotone_accept = finite_trial & (
            cond(
                jnp.isfinite(current_residual),
                lambda _: trial["fresh_residual"] <= current_residual,
                lambda _: jnp.asarray(True),
                operand=0,
            )
        )
        accept_now = (~carry["accepted"]) & monotone_accept
        better_best = finite_trial & (
            (~carry["best_found"]) | (trial["fresh_residual"] < carry["best_residual"])
        )
        if line_search_charge_row_index is None:
            charge_term = jnp.asarray(0.0, dtype=dtype)
        else:
            finite_charge = jnp.isfinite(trial["zero_charge_residual"])
            charge_term = jnp.where(
                finite_charge,
                jnp.asarray(line_search_charge_weight, dtype=dtype)
                * jnp.abs(trial["zero_charge_residual"]),
                inf_value,
            )
        composite_merit = (
            trial["element_balance_residual_norm"]
            + jnp.abs(trial["ntot_residual"])
            + trial["gas_stationarity_residual_norm"]
            + trial["cond_stationarity_residual_norm"]
            + charge_term
        )
        finite_composite = finite_trial & jnp.isfinite(composite_merit)
        better_composite = finite_composite & (
            (~carry["best_composite_found"])
            | (composite_merit < carry["best_composite_merit"])
        )

        carry = cond(
            accept_now,
            lambda c: {
                **c,
                "accepted": jnp.asarray(True),
                "accept_index": jnp.asarray(i, dtype=int_dtype),
                "accept_residual": trial["fresh_residual"],
                "accept_lam": trial["lam"],
                "accept_ln_nk": trial["ln_nk"],
                "accept_ln_mk": trial["ln_mk"],
                "accept_ln_ntot": trial["ln_ntot"],
                "accept_gk": trial["gk"],
                "accept_An": trial["An"],
                "accept_Am": trial["Am"],
            },
            lambda c: c,
            carry,
        )
        carry = cond(
            better_best,
            lambda c: {
                **c,
                "best_found": jnp.asarray(True),
                "best_index": jnp.asarray(i, dtype=int_dtype),
                "best_residual": trial["fresh_residual"],
                "best_lam": trial["lam"],
                "best_ln_nk": trial["ln_nk"],
                "best_ln_mk": trial["ln_mk"],
                "best_ln_ntot": trial["ln_ntot"],
                "best_gk": trial["gk"],
                "best_An": trial["An"],
                "best_Am": trial["Am"],
            },
            lambda c: c,
            carry,
        )
        carry = cond(
            better_composite,
            lambda c: {
                **c,
                "best_composite_found": jnp.asarray(True),
                "best_composite_index": jnp.asarray(i, dtype=int_dtype),
                "best_composite_residual": trial["fresh_residual"],
                "best_composite_merit": composite_merit,
                "best_composite_lam": trial["lam"],
                "best_composite_ln_nk": trial["ln_nk"],
                "best_composite_ln_mk": trial["ln_mk"],
                "best_composite_ln_ntot": trial["ln_ntot"],
                "best_composite_gk": trial["gk"],
                "best_composite_An": trial["An"],
                "best_composite_Am": trial["Am"],
            },
            lambda c: c,
            carry,
        )
        return carry

    carry = fori_loop(0, max_backtracks + 1, _loop_body, init_carry)

    if line_search_selection_policy == "first_monotone_with_best_finite_fallback":
        use_accept = carry["accepted"]
        use_best = (~use_accept) & carry["best_found"]
        accept_code_best = jnp.asarray(1, dtype=int_dtype)
    elif line_search_selection_policy == "best_finite":
        use_accept = jnp.asarray(False)
        use_best = carry["best_found"]
        use_composite = jnp.asarray(False)
        accept_code_best = jnp.asarray(3, dtype=int_dtype)
        accept_code_composite = accept_code_best
    elif line_search_selection_policy == "charge_aware_composite":
        use_accept = jnp.asarray(False)
        use_best = jnp.asarray(False)
        use_composite = carry["best_composite_found"]
        accept_code_best = jnp.asarray(5, dtype=int_dtype)
        accept_code_composite = jnp.asarray(5, dtype=int_dtype)
    elif line_search_selection_policy == "charge_aware_composite_nonincreasing":
        current_is_finite = jnp.isfinite(current_residual)
        composite_nonincreasing = carry["best_composite_found"] & (
            (~current_is_finite) | (carry["best_composite_residual"] <= current_residual)
        )
        use_accept = jnp.asarray(False)
        use_best = jnp.asarray(False)
        use_composite = composite_nonincreasing
        accept_code_best = jnp.asarray(6, dtype=int_dtype)
        accept_code_composite = jnp.asarray(6, dtype=int_dtype)
    elif line_search_selection_policy == "component_composite_nonincreasing":
        component_nonincreasing = carry["best_composite_found"] & (
            carry["best_composite_merit"] <= current_composite_merit
        )
        use_accept = jnp.asarray(False)
        use_best = jnp.asarray(False)
        use_composite = component_nonincreasing
        accept_code_best = jnp.asarray(7, dtype=int_dtype)
        accept_code_composite = jnp.asarray(7, dtype=int_dtype)
    else:
        current_is_finite = jnp.isfinite(current_residual)
        best_nonincreasing = carry["best_found"] & (
            (~current_is_finite) | (carry["best_residual"] <= current_residual)
        )
        use_accept = jnp.asarray(False)
        use_best = best_nonincreasing
        use_composite = jnp.asarray(False)
        accept_code_best = jnp.asarray(4, dtype=int_dtype)
        accept_code_composite = accept_code_best
    if line_search_selection_policy == "first_monotone_with_best_finite_fallback":
        use_composite = jnp.asarray(False)
        accept_code_composite = accept_code_best
    n_backtracks = jnp.where(
        use_accept,
        carry["accept_index"],
        jnp.where(
            use_best,
            carry["best_index"],
            jnp.where(
                use_composite,
                carry["best_composite_index"],
                jnp.asarray(max_backtracks, dtype=int_dtype),
            ),
        ),
    )
    accept_code = jnp.where(
        use_accept,
        jnp.asarray(0, dtype=int_dtype),
        jnp.where(
            use_best,
            accept_code_best,
            jnp.where(use_composite, accept_code_composite, jnp.asarray(2, dtype=int_dtype)),
        ),
    )

    return {
        "lam": jnp.where(
            use_accept,
            carry["accept_lam"],
            jnp.where(
                use_best,
                carry["best_lam"],
                jnp.where(use_composite, carry["best_composite_lam"], 0.0),
            ),
        ),
        "ln_nk": jnp.where(
            use_accept,
            carry["accept_ln_nk"],
            jnp.where(
                use_best,
                carry["best_ln_nk"],
                jnp.where(use_composite, carry["best_composite_ln_nk"], ln_nk),
            ),
        ),
        "ln_mk": jnp.where(
            use_accept,
            carry["accept_ln_mk"],
            jnp.where(
                use_best,
                carry["best_ln_mk"],
                jnp.where(use_composite, carry["best_composite_ln_mk"], ln_mk),
            ),
        ),
        "ln_ntot": jnp.where(
            use_accept,
            carry["accept_ln_ntot"],
            jnp.where(
                use_best,
                carry["best_ln_ntot"],
                jnp.where(use_composite, carry["best_composite_ln_ntot"], ln_ntot),
            ),
        ),
        "gk": jnp.where(
            use_accept,
            carry["accept_gk"],
            jnp.where(
                use_best,
                carry["best_gk"],
                jnp.where(use_composite, carry["best_composite_gk"], current_gk),
            ),
        ),
        "An": jnp.where(
            use_accept,
            carry["accept_An"],
            jnp.where(
                use_best,
                carry["best_An"],
                jnp.where(use_composite, carry["best_composite_An"], current_An),
            ),
        ),
        "Am": jnp.where(
            use_accept,
            carry["accept_Am"],
            jnp.where(
                use_best,
                carry["best_Am"],
                jnp.where(use_composite, carry["best_composite_Am"], current_Am),
            ),
        ),
        "fresh_residual": jnp.where(
            use_accept,
            carry["accept_residual"],
            jnp.where(
                use_best,
                carry["best_residual"],
                jnp.where(use_composite, carry["best_composite_residual"], current_residual),
            ),
        ),
        "line_search_composite_merit": jnp.where(
            use_composite,
            carry["best_composite_merit"],
            jnp.asarray(jnp.nan, dtype=dtype),
        ),
        "line_search_best_composite_found": carry["best_composite_found"],
        "line_search_best_composite_merit": carry["best_composite_merit"],
        "line_search_best_composite_lam": carry["best_composite_lam"],
        "line_search_best_composite_residual": carry["best_composite_residual"],
        "line_search_current_composite_merit": current_composite_merit,
        "n_backtracks": n_backtracks,
        "accept_code": accept_code,
        "line_search_selection_policy": line_search_selection_policy,
        "budget_guard_rejection_count": carry["budget_guard_rejection_count"],
    }


def _debug_array(label, array, iter_count, limit=None):
    arr = jnp.ravel(jnp.asarray(array))
    max_val = jnp.max(arr)
    min_val = jnp.min(arr)
    has_nan = jnp.any(jnp.isnan(arr))
    has_inf = jnp.any(jnp.isinf(arr))
    has_over = False if limit is None else (max_val > limit)
    predicate = has_nan | has_inf | has_over
    max_idx = jnp.argmax(arr)
    max_at = arr[max_idx]
    if limit is None:
        over_count = jnp.array(0, dtype=jnp.int32)
        first_over_idx = jnp.array(0, dtype=jnp.int32)
        first_over_val = jnp.array(0.0)
    else:
        over_mask = arr > limit
        over_count = jnp.sum(over_mask)
        first_over_idx = jnp.argmax(over_mask)
        first_over_val = arr[first_over_idx]

    def _print(_):
        jdebug.print(
            "iter {i} {label}: min {min_val} max {max_val} nan {nan} inf {inf} "
            "over {over} max_idx {max_idx} max_at {max_at} over_count {over_count} "
            "first_over_idx {first_over_idx} first_over_val {first_over_val}",
            i=iter_count,
            label=label,
            min_val=min_val,
            max_val=max_val,
            nan=has_nan,
            inf=has_inf,
            over=has_over,
            max_idx=max_idx,
            max_at=max_at,
            over_count=over_count,
            first_over_idx=first_over_idx,
            first_over_val=first_over_val,
        )
        return 0

    return cond(predicate, _print, lambda _: 0, operand=0)


def _update_all_core(
    ln_nk,
    ln_mk,
    ln_ntot,
    formula_matrix,
    formula_matrix_cond,
    b,
    T,
    ln_normalized_pressure,
    hvector,
    hvector_cond,
    gk,
    An,
    Am,
    current_residual,
    epsilon,
    iter_count,
    debug_nan=False,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
    budget_guard_enabled: bool = False,
    budget_margin: float = 0.0,
    emergency_budget_projection_enabled: bool = False,
    reduced_coupling_mode: str = "current",
    reduced_coupling_alpha_s: float = 1.0,
    gas_step_scale: float = 1.0,
    gas_step_direction_sign: float = 1.0,
    ntot_step_scale: Optional[float] = None,
    condensate_step_scale: float = 1.0,
    include_system_trace: bool = False,
    line_search_selection_policy: str = "first_monotone_with_best_finite_fallback",
    line_search_charge_row_index: Optional[int] = None,
    line_search_charge_weight: float = 1.0,
):
    exp_overflow_limit = 700.0
    if debug_nan:
        _debug_array("ln_nk pre-exp", ln_nk, iter_count, exp_overflow_limit)
        _debug_array("ln_mk pre-exp", ln_mk, iter_count, exp_overflow_limit)
        _debug_array(
            "ln_ntot pre-exp", jnp.array([ln_ntot]), iter_count, exp_overflow_limit
        )

    ln_sk = 2.0 * ln_mk - epsilon
    bk = formula_matrix @ jnp.exp(ln_nk)

    if debug_nan:
        _debug_array("ln_nk_scaled pre-exp", ln_nk, iter_count, exp_overflow_limit)
        _debug_array("ln_mk_scaled pre-exp", ln_mk, iter_count, exp_overflow_limit)
        _debug_array(
            "ln_ntot_scaled pre-exp",
            jnp.array([ln_ntot]),
            iter_count,
            exp_overflow_limit,
        )
        _debug_array("ln_sk_scaled pre-exp", ln_sk, iter_count, exp_overflow_limit)

    step_metrics = _compute_iteration_step_metrics(
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        epsilon,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
        include_system_trace=include_system_trace,
        reduced_coupling_mode=reduced_coupling_mode,
        reduced_coupling_alpha_s=reduced_coupling_alpha_s,
    )
    pi_vector = step_metrics["pi_vector"]
    delta_ln_ntot = step_metrics["delta_ln_ntot"]

    delta_ln_nk = step_metrics["delta_ln_nk"]
    gas_step_scale_value = jnp.asarray(gas_step_scale, dtype=delta_ln_nk.dtype)
    gas_step_direction_sign_value = jnp.asarray(
        gas_step_direction_sign, dtype=delta_ln_nk.dtype
    )
    gas_step_factor_value = gas_step_scale_value * gas_step_direction_sign_value
    ntot_step_scale_value = jnp.asarray(
        gas_step_scale if ntot_step_scale is None else ntot_step_scale,
        dtype=delta_ln_nk.dtype,
    )
    ntot_step_factor_value = ntot_step_scale_value * gas_step_direction_sign_value
    condensate_step_scale_value = jnp.asarray(
        condensate_step_scale, dtype=delta_ln_nk.dtype
    )
    unscaled_delta_ln_nk = delta_ln_nk
    unscaled_delta_ln_ntot = delta_ln_ntot
    if (
        gas_step_scale != 1.0
        or gas_step_direction_sign != 1.0
        or ntot_step_scale is not None
    ):
        delta_ln_nk = delta_ln_nk * gas_step_factor_value
        delta_ln_ntot = delta_ln_ntot * ntot_step_factor_value
        lam1_gas_scaled = stepsize_cea_gas(delta_ln_nk, delta_ln_ntot, ln_nk, ln_ntot)
        lam_scaled = jnp.minimum(
            1.0,
            jnp.minimum(
                lam1_gas_scaled,
                jnp.minimum(step_metrics["lam1_cond"], step_metrics["lam2_cond"]),
            ),
        )
        lam_scaled = jnp.clip(lam_scaled, 0.0, 1.0)
        limiter_candidates = jnp.asarray(
            [1.0, lam1_gas_scaled, step_metrics["lam1_cond"], step_metrics["lam2_cond"]]
        )
        step_metrics = dict(step_metrics)
        step_metrics["delta_ln_nk"] = delta_ln_nk
        step_metrics["delta_ln_ntot"] = delta_ln_ntot
        step_metrics["unscaled_delta_ln_nk"] = unscaled_delta_ln_nk
        step_metrics["unscaled_delta_ln_ntot"] = unscaled_delta_ln_ntot
        step_metrics["unscaled_lam1_gas"] = step_metrics["lam1_gas"]
        step_metrics["unscaled_lam"] = step_metrics["lam"]
        step_metrics["unscaled_max_abs_delta_ln_nk"] = step_metrics[
            "max_abs_delta_ln_nk"
        ]
        step_metrics["lam1_gas"] = lam1_gas_scaled
        step_metrics["lam"] = lam_scaled
        step_metrics["limiting_index"] = jnp.argmin(limiter_candidates).astype(jnp.int32)
        step_metrics["max_abs_delta_ln_nk"] = jnp.max(jnp.abs(delta_ln_nk))
    else:
        step_metrics = dict(step_metrics)
        step_metrics["unscaled_delta_ln_nk"] = unscaled_delta_ln_nk
        step_metrics["unscaled_delta_ln_ntot"] = unscaled_delta_ln_ntot
        step_metrics["unscaled_lam1_gas"] = step_metrics["lam1_gas"]
        step_metrics["unscaled_lam"] = step_metrics["lam"]
        step_metrics["unscaled_max_abs_delta_ln_nk"] = step_metrics[
            "max_abs_delta_ln_nk"
        ]
    step_metrics["gas_step_scale"] = gas_step_scale_value
    step_metrics["gas_step_direction_sign"] = gas_step_direction_sign_value
    step_metrics["gas_step_factor"] = gas_step_factor_value
    step_metrics["ntot_step_scale"] = ntot_step_scale_value
    step_metrics["ntot_step_factor"] = ntot_step_factor_value
    step_metrics["condensate_step_scale"] = condensate_step_scale_value
    # this breaks the results. we cannot clip here.
    # raw_delta_ln_nk = formula_matrix.T @ pi_vector + delta_ln_ntot - gk
    # MAX_STEP_N_UP = 10.0  # do not update larger than ln(n) 0.1e ~ 10%
    # MAX_STEP_N_LOW = 10.0
    # delta_ln_nk = jnp.clip(raw_delta_ln_nk, -MAX_STEP_N_LOW, MAX_STEP_N_UP)

    # log_m_over_nu = jnp.clip(ln_mk - epsilon, LOG_MIN, LOG_MAX)
    log_m_over_nu = ln_mk - epsilon
    if debug_nan:
        _debug_array(
            "log_m_over_nu pre-exp", log_m_over_nu, iter_count, exp_overflow_limit
        )

    raw_delta_ln_mk = step_metrics["raw_delta_ln_mk"]

    MAX_STEP_M_UP = 0.1  # do not update larger than ln(m) 0.1e ~ 10%
    MAX_STEP_M_LOW = 0.1
    delta_ln_mk = step_metrics["delta_ln_mk"]
    if condensate_step_scale != 1.0:
        delta_ln_mk = delta_ln_mk * condensate_step_scale_value
        step_metrics["delta_ln_mk"] = delta_ln_mk
        step_metrics["lam1_cond"] = stepsize_cond_heurstic(delta_ln_mk)
        step_metrics["lam2_cond"] = stepsize_sk(delta_ln_mk, ln_mk, epsilon)
        step_metrics["lam"] = jnp.minimum(
            1.0,
            jnp.minimum(
                step_metrics["lam1_gas"],
                jnp.minimum(step_metrics["lam1_cond"], step_metrics["lam2_cond"]),
            ),
        )
        step_metrics["lam"] = jnp.clip(step_metrics["lam"], 0.0, 1.0)
        limiter_candidates = jnp.asarray(
            [1.0, step_metrics["lam1_gas"], step_metrics["lam1_cond"], step_metrics["lam2_cond"]]
        )
        step_metrics["limiting_index"] = jnp.argmin(limiter_candidates).astype(jnp.int32)
        step_metrics["max_abs_clipped_delta_ln_mk"] = jnp.max(jnp.abs(delta_ln_mk))
    # delta_ln_mk = jnp.exp(ln_mk - epsilon) * (formula_matrix_cond.T @ pi_vector - hvector_cond) + 1.0

    line_search_result = _choose_lambda_by_residual_backtracking(
        ln_nk,
        ln_mk,
        ln_ntot,
        gk,
        An,
        Am,
        current_residual,
        step_metrics["lam"],
        delta_ln_nk,
        delta_ln_mk,
        delta_ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        T,
        ln_normalized_pressure,
        hvector,
        hvector_cond,
        epsilon,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
        budget_guard_enabled=budget_guard_enabled,
        budget_margin=budget_margin,
        line_search_selection_policy=line_search_selection_policy,
        line_search_charge_row_index=line_search_charge_row_index,
        line_search_charge_weight=line_search_charge_weight,
    )

    lam = line_search_result["lam"]
    ln_nk = line_search_result["ln_nk"]
    ln_mk = line_search_result["ln_mk"]
    ln_ntot = line_search_result["ln_ntot"]
    gk = line_search_result["gk"]
    An = line_search_result["An"]
    Am = line_search_result["Am"]
    residual = line_search_result["fresh_residual"]

    projection = apply_emergency_budget_projection(
        formula_matrix_cond,
        jnp.exp(ln_mk),
        b,
        budget_margin=budget_margin,
    )
    projection_used = emergency_budget_projection_enabled & projection["projection_used"]

    def _apply_projection(_):
        projected_mk = projection["m"]
        projected_ln_mk = jnp.log(jnp.maximum(projected_mk, jnp.asarray(1.0e-300, dtype=projected_mk.dtype)))
        projected_Am = formula_matrix_cond @ projected_mk
        pi_vector_resid = _recompute_pi_for_residual(
            jnp.exp(ln_nk),
            projected_mk,
            jnp.exp(ln_ntot),
            formula_matrix,
            formula_matrix_cond,
            b,
            gk,
            hvector_cond,
            epsilon,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
        )
        projected_residual = _compute_residuals(
            jnp.exp(ln_nk),
            projected_mk,
            jnp.exp(ln_ntot),
            formula_matrix,
            formula_matrix_cond,
            b,
            gk,
            hvector_cond,
            jnp.exp(epsilon),
            An,
            projected_Am,
            pi_vector_resid,
        )
        return projected_ln_mk, projected_Am, projected_residual

    ln_mk, Am, residual = cond(
        projection_used,
        _apply_projection,
        lambda _: (ln_mk, Am, residual),
        operand=0,
    )
    if debug_nan:
        _debug_array("residual", jnp.array([residual]), iter_count)
    numeric_metrics = dict(step_metrics)
    numeric_metrics["lam_heuristic"] = step_metrics["lam"]
    numeric_metrics["lam_selected"] = lam
    numeric_metrics["lam"] = lam
    numeric_metrics["n_backtracks"] = line_search_result["n_backtracks"]
    numeric_metrics["budget_guard_rejection_count"] = line_search_result[
        "budget_guard_rejection_count"
    ]
    numeric_metrics["budget_guard_rejected_any"] = (
        line_search_result["budget_guard_rejection_count"] > 0
    )
    numeric_metrics["emergency_budget_projection_used"] = projection_used
    numeric_metrics["emergency_budget_projection_alpha"] = projection["alpha"]
    numeric_metrics["residual_before"] = jnp.asarray(current_residual, dtype=residual.dtype)
    numeric_metrics["residual_after"] = residual
    numeric_metrics["line_search_accept_code"] = line_search_result["accept_code"]
    numeric_metrics["line_search_selection_policy"] = line_search_result[
        "line_search_selection_policy"
    ]
    numeric_metrics["line_search_composite_merit"] = line_search_result[
        "line_search_composite_merit"
    ]
    numeric_metrics["line_search_current_composite_merit"] = line_search_result[
        "line_search_current_composite_merit"
    ]
    numeric_metrics["line_search_best_composite_found"] = line_search_result[
        "line_search_best_composite_found"
    ]
    numeric_metrics["line_search_best_composite_merit"] = line_search_result[
        "line_search_best_composite_merit"
    ]
    numeric_metrics["line_search_best_composite_lam"] = line_search_result[
        "line_search_best_composite_lam"
    ]
    numeric_metrics["line_search_best_composite_residual"] = line_search_result[
        "line_search_best_composite_residual"
    ]
    numeric_metrics["residual"] = residual
    return ln_nk, ln_mk, ln_ntot, gk, An, Am, residual, lam, numeric_metrics


def _update_all(
    ln_nk,
    ln_mk,
    ln_ntot,
    formula_matrix,
    formula_matrix_cond,
    b,
    T,
    ln_normalized_pressure,
    hvector,
    hvector_cond,
    gk,
    An,
    Am,
    current_residual,
    epsilon,
    iter_count,
    debug_nan=False,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
    budget_guard_enabled: bool = False,
    budget_margin: float = 0.0,
    emergency_budget_projection_enabled: bool = False,
    reduced_coupling_mode: str = "current",
    reduced_coupling_alpha_s: float = 1.0,
    gas_step_scale: float = 1.0,
    gas_step_direction_sign: float = 1.0,
    ntot_step_scale: Optional[float] = None,
    condensate_step_scale: float = 1.0,
    line_search_selection_policy: str = "first_monotone_with_best_finite_fallback",
    line_search_charge_row_index: Optional[int] = None,
    line_search_charge_weight: float = 1.0,
):
    (
        ln_nk,
        ln_mk,
        ln_ntot,
        gk,
        An,
        Am,
        residual,
        lam,
        _numeric_metrics,
    ) = _update_all_core(
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        T,
        ln_normalized_pressure,
        hvector,
        hvector_cond,
        gk,
        An,
        Am,
        current_residual,
        epsilon,
        iter_count,
        debug_nan=debug_nan,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
        budget_guard_enabled=budget_guard_enabled,
        budget_margin=budget_margin,
        emergency_budget_projection_enabled=emergency_budget_projection_enabled,
        reduced_coupling_mode=reduced_coupling_mode,
        reduced_coupling_alpha_s=reduced_coupling_alpha_s,
        gas_step_scale=gas_step_scale,
        gas_step_direction_sign=gas_step_direction_sign,
        ntot_step_scale=ntot_step_scale,
        condensate_step_scale=condensate_step_scale,
        line_search_selection_policy=line_search_selection_policy,
        line_search_charge_row_index=line_search_charge_row_index,
        line_search_charge_weight=line_search_charge_weight,
    )
    return ln_nk, ln_mk, ln_ntot, gk, An, Am, residual, lam


def _update_all_with_metrics(
    ln_nk,
    ln_mk,
    ln_ntot,
    formula_matrix,
    formula_matrix_cond,
    b,
    T,
    ln_normalized_pressure,
    hvector,
    hvector_cond,
    gk,
    An,
    Am,
    current_residual,
    epsilon,
    iter_count,
    debug_nan=False,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
    include_system_trace: bool = False,
    budget_guard_enabled: bool = False,
    budget_margin: float = 0.0,
    emergency_budget_projection_enabled: bool = False,
    reduced_coupling_mode: str = "current",
    reduced_coupling_alpha_s: float = 1.0,
    gas_step_scale: float = 1.0,
    gas_step_direction_sign: float = 1.0,
    ntot_step_scale: Optional[float] = None,
    condensate_step_scale: float = 1.0,
    exact_input_bundle_context: Optional[Dict[str, Any]] = None,
    line_search_selection_policy: str = "first_monotone_with_best_finite_fallback",
    line_search_charge_row_index: Optional[int] = None,
    line_search_charge_weight: float = 1.0,
):
    active_bundle_context = _normalize_exact_input_bundle_context(
        exact_input_bundle_context,
        int(iter_count),
    )
    exact_input_bundle = None
    if active_bundle_context is not None:
        exact_input_bundle = _build_reduced_solver_exact_input_bundle(
            case_key=active_bundle_context["case_key"],
            newton_iter=active_bundle_context["newton_iter"],
            ln_nk=ln_nk,
            ln_mk=ln_mk,
            ln_ntot=ln_ntot,
            temperature=T,
            ln_normalized_pressure=ln_normalized_pressure,
            formula_matrix=formula_matrix,
            formula_matrix_cond=formula_matrix_cond,
            b=b,
            gk=gk,
            hvector_cond=hvector_cond,
            epsilon=epsilon,
            condensates_jac_indices=active_bundle_context["condensates_jac_indices"],
            condensate_labels_jac_order=active_bundle_context[
                "condensate_labels_jac_order"
            ],
            element_labels_reduced_order=active_bundle_context[
                "element_labels_reduced_order"
            ],
            call_site_provenance=active_bundle_context["call_site_provenance"],
            active=True,
            row_scaled_element_condensate_jec_target_block=active_bundle_context[
                "row_scaled_element_condensate_jec_target_block"
            ],
            selected_element_row_scaling_vector=active_bundle_context[
                "selected_element_row_scaling_vector"
            ],
            gas_phase_calculate_lifecycle_context=active_bundle_context[
                "gas_phase_calculate_lifecycle_context"
            ],
            ln_nk_producer_trace=(
                active_bundle_context["ln_nk_producer_trace"]
                if active_bundle_context["ln_nk_producer_trace"] is not None
                else _build_ln_nk_producer_trace(
                    ln_nk=ln_nk,
                    source_stage="_update_all_with_metrics argument ln_nk",
                    producer_function="src/exogibbs/optimize/pipm_rgie_cond.py::_update_all_with_metrics",
                    case_key=active_bundle_context["case_key"],
                    newton_iter=active_bundle_context["newton_iter"],
                )
            ),
        )
        exact_input_bundle["source_provenance"] = active_bundle_context[
            "source_provenance"
        ]
    (
        ln_nk,
        ln_mk,
        ln_ntot,
        gk,
        An,
        Am,
        residual,
        lam,
        trace_metrics,
    ) = _update_all_core(
        ln_nk,
        ln_mk,
        ln_ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        T,
        ln_normalized_pressure,
        hvector,
        hvector_cond,
        gk,
        An,
        Am,
        current_residual,
        epsilon,
        iter_count,
        debug_nan=debug_nan,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
        include_system_trace=include_system_trace,
        budget_guard_enabled=budget_guard_enabled,
        budget_margin=budget_margin,
        emergency_budget_projection_enabled=emergency_budget_projection_enabled,
        reduced_coupling_mode=reduced_coupling_mode,
        reduced_coupling_alpha_s=reduced_coupling_alpha_s,
        gas_step_scale=gas_step_scale,
        gas_step_direction_sign=gas_step_direction_sign,
        ntot_step_scale=ntot_step_scale,
        condensate_step_scale=condensate_step_scale,
        line_search_selection_policy=line_search_selection_policy,
        line_search_charge_row_index=line_search_charge_row_index,
        line_search_charge_weight=line_search_charge_weight,
    )
    trace_metrics = dict(trace_metrics)
    trace_metrics["line_search_used"] = True
    if exact_input_bundle is not None:
        trace_metrics["exact_input_bundle"] = exact_input_bundle
        trace_metrics["exact_input_bundle_emitted"] = True
    else:
        trace_metrics["exact_input_bundle_emitted"] = False
    accept_code = int(trace_metrics["line_search_accept_code"])
    if accept_code == 0:
        accept_kind = "monotone"
    elif accept_code == 1:
        accept_kind = "best_finite_fallback"
    elif accept_code == 3:
        accept_kind = "best_finite_opt_in"
    elif accept_code == 4:
        accept_kind = "best_finite_nonincreasing_opt_in"
    elif accept_code == 5:
        accept_kind = "charge_aware_composite_opt_in"
    elif accept_code == 6:
        accept_kind = "charge_aware_composite_nonincreasing_opt_in"
    elif accept_code == 7:
        accept_kind = "component_composite_nonincreasing_opt_in"
    else:
        accept_kind = "zero_step"
    trace_metrics["line_search_accept_kind"] = accept_kind
    return ln_nk, ln_mk, ln_ntot, gk, An, Am, residual, lam, trace_metrics


def _contains_invalid_numbers(*arrays) -> jnp.ndarray:
    invalid_flags = []
    for array in arrays:
        arr = jnp.asarray(array)
        invalid_flags.append(jnp.any(jnp.isnan(arr) | jnp.isinf(arr)))
    return jnp.any(jnp.stack(invalid_flags))


def _compute_current_fresh_residual(
    ln_nk: jnp.ndarray,
    ln_mk: jnp.ndarray,
    ln_ntot: jnp.ndarray,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    b: jnp.ndarray,
    gk: jnp.ndarray,
    hvector_cond: jnp.ndarray,
    epsilon: float,
    *,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> jnp.ndarray:
    """Compute the current-state residual used by explicit first-step guards."""

    nk = jnp.exp(ln_nk)
    mk = jnp.exp(ln_mk)
    ntot = jnp.exp(ln_ntot)
    pi_vector_resid = _recompute_pi_for_residual(
        nk,
        mk,
        ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        epsilon,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )
    residual = _compute_residuals(
        nk,
        mk,
        ntot,
        formula_matrix,
        formula_matrix_cond,
        b,
        gk,
        hvector_cond,
        jnp.exp(epsilon),
        formula_matrix @ nk,
        formula_matrix_cond @ mk,
        pi_vector_resid,
    )
    return jnp.where(
        jnp.isfinite(residual) & (~_contains_invalid_numbers(pi_vector_resid)),
        residual,
        jnp.asarray(jnp.inf, dtype=residual.dtype),
    )


def _minimize_gibbs_cond_core_impl(
    state: ThermoState,
    ln_nk_init: jnp.ndarray,
    ln_mk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    residual_crit: float = 1.0e-11,
    max_iter: int = 1000,
    element_indices: Optional[jnp.ndarray] = None,
    debug_nan: bool = False,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
    budget_guard_enabled: bool = False,
    budget_margin: float = 0.0,
    emergency_budget_projection_enabled: bool = False,
    reduced_coupling_mode: str = "current",
    reduced_coupling_alpha_s: float = 1.0,
    gas_step_scale: float = 1.0,
    gas_step_direction_sign: float = 1.0,
    ntot_step_scale: Optional[float] = None,
    condensate_step_scale: float = 1.0,
    initial_residual_policy: str = "infinite",
    line_search_selection_policy: str = "first_monotone_with_best_finite_fallback",
    line_search_charge_row_index: Optional[int] = None,
    line_search_charge_weight: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Shared implementation for condensate solves and diagnostics wrappers."""

    n_elements = formula_matrix.shape[0]
    if formula_matrix_cond.shape[0] != n_elements:
        raise ValueError(
            "formula_matrix and formula_matrix_cond must have the same number of element rows."
        )

    b = (
        jnp.asarray(state.element_vector)
        if element_indices is None
        else jnp.asarray(state.element_vector)[jnp.asarray(element_indices)]
    )
    if b.shape[0] != n_elements:
        raise ValueError(
            "ThermoState.element_vector length does not match the number of element rows "
            f"in the formula matrices (got {b.shape[0]}, expected {n_elements}). "
            "Provide element_indices that map the state vector onto the reduced element set."
        )

    hvector = hvector_func(state.temperature)
    hvector_cond = hvector_cond_func(state.temperature)

    def cond_fun(carry):
        *_, residual, counter, _last_step_size, _budget_rejections, _projection_count = carry
        return (residual > residual_crit) & (counter < max_iter)

    def body_fun(carry):
        (
            ln_nk,
            ln_mk,
            ln_ntot,
            gk,
            An,
            Am,
            residual,
            counter,
            _last_step_size,
            budget_rejections,
            projection_count,
        ) = carry
        (
            ln_nk_new,
            ln_mk_new,
            ln_ntot_new,
            gk,
            An,
            Am,
            residual,
            last_step_size,
            numeric_metrics,
        ) = _update_all_core(
            ln_nk=ln_nk,
            ln_mk=ln_mk,
            ln_ntot=ln_ntot,
            formula_matrix=formula_matrix,
            formula_matrix_cond=formula_matrix_cond,
            b=b,
            T=state.temperature,
            ln_normalized_pressure=state.ln_normalized_pressure,
            hvector=hvector,
            hvector_cond=hvector_cond,
            gk=gk,
            An=An,
            Am=Am,
            current_residual=residual,
            epsilon=epsilon,
            iter_count=counter,
            debug_nan=debug_nan,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
            budget_guard_enabled=budget_guard_enabled,
            budget_margin=budget_margin,
            emergency_budget_projection_enabled=emergency_budget_projection_enabled,
            reduced_coupling_mode=reduced_coupling_mode,
            reduced_coupling_alpha_s=reduced_coupling_alpha_s,
            gas_step_scale=gas_step_scale,
            gas_step_direction_sign=gas_step_direction_sign,
            ntot_step_scale=ntot_step_scale,
            condensate_step_scale=condensate_step_scale,
            line_search_selection_policy=line_search_selection_policy,
            line_search_charge_row_index=line_search_charge_row_index,
            line_search_charge_weight=line_search_charge_weight,
        )
        return (
            ln_nk_new,
            ln_mk_new,
            ln_ntot_new,
            gk,
            An,
            Am,
            residual,
            counter + 1,
            last_step_size,
            budget_rejections + numeric_metrics["budget_guard_rejection_count"],
            projection_count
            + numeric_metrics["emergency_budget_projection_used"].astype(jnp.int32),
        )

    gk = _compute_gk(
        state.temperature,
        ln_nk_init,
        ln_ntot_init,
        hvector,
        state.ln_normalized_pressure,
    )
    An_in = formula_matrix @ jnp.exp(ln_nk_init)
    Am_in = formula_matrix_cond @ jnp.exp(ln_mk_init)
    init_last_step_size = jnp.asarray(0.0, dtype=ln_nk_init.dtype)
    init_budget_rejections = jnp.asarray(0, dtype=jnp.int32)
    init_projection_count = jnp.asarray(0, dtype=jnp.int32)

    if initial_residual_policy == "computed_fresh":
        initial_residual = _compute_current_fresh_residual(
            ln_nk_init,
            ln_mk_init,
            ln_ntot_init,
            formula_matrix,
            formula_matrix_cond,
            b,
            gk,
            hvector_cond,
            epsilon,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
        )
    else:
        initial_residual = jnp.asarray(jnp.inf, dtype=ln_nk_init.dtype)

    (
        ln_nk,
        ln_mk,
        ln_ntot,
        _gk,
        _An,
        _Am,
        residual,
        counter,
        last_step_size,
        budget_rejections,
        projection_count,
    ) = while_loop(
        cond_fun,
        body_fun,
        (
            ln_nk_init,
            ln_mk_init,
            ln_ntot_init,
            gk,
            An_in,
            Am_in,
            initial_residual,
            0,
            init_last_step_size,
            init_budget_rejections,
            init_projection_count,
        ),
    )
    return (
        ln_nk,
        ln_mk,
        ln_ntot,
        counter,
        residual,
        last_step_size,
        budget_rejections,
        projection_count,
    )


def minimize_gibbs_cond_core(
    state: ThermoState,
    ln_nk_init: jnp.ndarray,
    ln_mk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,  ### new argument
    residual_crit: float = 1.0e-11,
    max_iter: int = 1000,
    element_indices: Optional[jnp.ndarray] = None,
    debug_nan: bool = False,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, int]:
    """Compute log(number of species) by minimizing the Gibbs energy using the Lagrange multipliers method.

    Args:
        state: Thermodynamic state containing temperature, pressure, and element abundances.
        ln_nk_init: Initial log number of species vector (n_species,).
        ln_ntot_init: Initial log total number of species.
        formula_matrix: Stoichiometric formula matrix (n_elements, n_species).
        hvector: Chemical potential over RT vector (n_species,).
        residual_crit: Convergence tolerance for residual norm.
        max_iter: Maximum number of iterations allowed.
        element_indices: Optional indices mapping ``state.element_vector`` onto the
            element ordering used by ``formula_matrix``/``formula_matrix_cond``.
            Use this when ``state.element_vector`` stores a superset of elements.

    Returns:
        Tuple containing:
            - Final log number of species vector (n_species,).
            - Final log number of condensed species vector (n_condensed_species,).
            - Final log eta vector (n_condensed_species,).
            - Final log total number of species.
            - Number of iterations performed.
    """

    (
        ln_nk,
        ln_mk,
        ln_ntot,
        counter,
        _residual,
        _last_step_size,
        _budget_rejections,
        _projection_count,
    ) = _minimize_gibbs_cond_core_impl(
        state,
        ln_nk_init,
        ln_mk_init,
        ln_ntot_init,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        epsilon,
        residual_crit=residual_crit,
        max_iter=max_iter,
        element_indices=element_indices,
        debug_nan=debug_nan,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
    )
    return ln_nk, ln_mk, ln_ntot, counter


def minimize_gibbs_cond_with_diagnostics(
    state: ThermoState,
    ln_nk_init: jnp.ndarray,
    ln_mk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    residual_crit: float = 1.0e-11,
    max_iter: int = 1000,
    element_indices: Optional[jnp.ndarray] = None,
    debug_nan: bool = False,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
    budget_guard_enabled: bool = False,
    budget_margin: float = 0.0,
    emergency_budget_projection_enabled: bool = False,
    reduced_coupling_mode: str = "current",
    reduced_coupling_alpha_s: float = 1.0,
    reduced_coupling_selection: Optional[Dict[str, Any]] = None,
    gas_step_scale: float = 1.0,
    gas_step_direction_sign: float = 1.0,
    ntot_step_scale: Optional[float] = None,
    condensate_step_scale: float = 1.0,
    initial_residual_policy: str = "infinite",
    line_search_selection_policy: str = "first_monotone_with_best_finite_fallback",
    line_search_charge_row_index: Optional[int] = None,
    line_search_charge_weight: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Run the active condensate solver and return lightweight convergence diagnostics."""

    (
        ln_nk,
        ln_mk,
        ln_ntot,
        n_iter,
        final_residual,
        last_step_size,
        budget_guard_rejection_count,
        emergency_budget_projection_count,
    ) = _minimize_gibbs_cond_core_impl(
        state,
        ln_nk_init,
        ln_mk_init,
        ln_ntot_init,
        formula_matrix,
        formula_matrix_cond,
        hvector_func,
        hvector_cond_func,
        epsilon,
        residual_crit=residual_crit,
        max_iter=max_iter,
        element_indices=element_indices,
        debug_nan=debug_nan,
        reduced_solver=reduced_solver,
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength,
        budget_guard_enabled=budget_guard_enabled,
        budget_margin=budget_margin,
        emergency_budget_projection_enabled=emergency_budget_projection_enabled,
        reduced_coupling_mode=reduced_coupling_mode,
        reduced_coupling_alpha_s=reduced_coupling_alpha_s,
        gas_step_scale=gas_step_scale,
        gas_step_direction_sign=gas_step_direction_sign,
        ntot_step_scale=ntot_step_scale,
        condensate_step_scale=condensate_step_scale,
        initial_residual_policy=initial_residual_policy,
        line_search_selection_policy=line_search_selection_policy,
        line_search_charge_row_index=line_search_charge_row_index,
        line_search_charge_weight=line_search_charge_weight,
    )

    residual_crit_used = jnp.asarray(residual_crit, dtype=final_residual.dtype)
    max_iter_used = jnp.asarray(max_iter, dtype=n_iter.dtype)
    epsilon_used = jnp.asarray(epsilon, dtype=final_residual.dtype)
    converged = final_residual <= residual_crit_used
    hit_max_iter = (n_iter >= max_iter_used) & (~converged)
    invalid_numbers_detected = _contains_invalid_numbers(
        ln_nk,
        ln_mk,
        ln_ntot,
        last_step_size,
    )

    diagnostics = {
        "n_iter": n_iter,
        "converged": converged,
        "hit_max_iter": hit_max_iter,
        "final_residual": final_residual,
        "residual_crit": residual_crit_used,
        "max_iter": max_iter_used,
        "epsilon": epsilon_used,
        "final_step_size": last_step_size,
        "invalid_numbers_detected": invalid_numbers_detected,
        "debug_nan": jnp.asarray(debug_nan),
        "reduced_solver": reduced_solver,
        "regularization_mode": regularization_mode,
        "regularization_strength": jnp.asarray(regularization_strength, dtype=final_residual.dtype),
        "inventory_budget_guard_enabled": jnp.asarray(budget_guard_enabled),
        "inventory_budget_margin": jnp.asarray(budget_margin, dtype=final_residual.dtype),
        "budget_guard_rejection_count": budget_guard_rejection_count,
        "budget_guard_rejected_any": budget_guard_rejection_count > 0,
        "emergency_budget_projection_enabled": jnp.asarray(
            emergency_budget_projection_enabled
        ),
        "emergency_budget_projection_count": emergency_budget_projection_count,
        "emergency_budget_projection_used": emergency_budget_projection_count > 0,
        "reduced_coupling_mode": reduced_coupling_mode,
        "reduced_coupling_alpha_s": jnp.asarray(reduced_coupling_alpha_s, dtype=final_residual.dtype),
        "gas_step_scale": jnp.asarray(gas_step_scale, dtype=final_residual.dtype),
        "gas_step_direction_sign": jnp.asarray(
            gas_step_direction_sign, dtype=final_residual.dtype
        ),
        "ntot_step_scale": jnp.asarray(
            gas_step_scale if ntot_step_scale is None else ntot_step_scale,
            dtype=final_residual.dtype,
        ),
        "condensate_step_scale": jnp.asarray(
            condensate_step_scale, dtype=final_residual.dtype
        ),
        "initial_residual_policy": initial_residual_policy,
        "line_search_selection_policy": line_search_selection_policy,
        "line_search_charge_row_index": (
            -1 if line_search_charge_row_index is None else int(line_search_charge_row_index)
        ),
        "line_search_charge_weight": jnp.asarray(
            line_search_charge_weight,
            dtype=final_residual.dtype,
        ),
    }
    if reduced_coupling_selection is not None:
        diagnostics.update(reduced_coupling_selection)
    return ln_nk, ln_mk, ln_ntot, diagnostics


def trace_minimize_gibbs_cond_iterations(
    state: ThermoState,
    ln_nk_init: jnp.ndarray,
    ln_mk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilon: float,
    residual_crit: float = 1.0e-11,
    max_iter: int = 1000,
    element_indices: Optional[jnp.ndarray] = None,
    tiny_step: float = 1.0e-14,
    trial_lambda_every_iter: bool = False,
    trial_lambda_iterations: Optional[Sequence[int]] = None,
    trial_lambda_values: Optional[Sequence[float]] = None,
    trial_lambda_multipliers: Sequence[float] = (1.0, 0.5, 0.2, 0.1, 0.05),
    extra_trial_lambda_values: Sequence[float] = (1.0, 0.5, 0.2, 0.1, 0.05),
    gas_species_names: Optional[Sequence[str]] = None,
    gas_limiter_every_iter: bool = False,
    gas_limiter_iterations: Optional[Sequence[int]] = None,
    gas_limiter_top_k: int = 10,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
    exact_input_bundle_context: Optional[Dict[str, Any]] = None,
    reduced_coupling_mode: str = "current",
    reduced_coupling_alpha_s: float = 1.0,
    gas_step_scale: float = 1.0,
    gas_step_direction_sign: float = 1.0,
    ntot_step_scale: Optional[float] = None,
    condensate_step_scale: float = 1.0,
    initial_residual_policy: str = "infinite",
    line_search_selection_policy: str = "first_monotone_with_best_finite_fallback",
    trial_lambda_charge_row_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one condensate layer with a full per-iteration trace for debugging."""

    n_elements = formula_matrix.shape[0]
    b = (
        jnp.asarray(state.element_vector)
        if element_indices is None
        else jnp.asarray(state.element_vector)[jnp.asarray(element_indices)]
    )
    if b.shape[0] != n_elements:
        raise ValueError(
            "ThermoState.element_vector length does not match the number of element rows "
            f"in the formula matrices (got {b.shape[0]}, expected {n_elements}). "
            "Provide element_indices that map the state vector onto the reduced element set."
        )

    hvector = hvector_func(state.temperature)
    hvector_cond = hvector_cond_func(state.temperature)
    ln_nk = jnp.asarray(ln_nk_init)
    ln_mk = jnp.asarray(ln_mk_init)
    ln_ntot = jnp.asarray(ln_ntot_init)
    gk = _compute_gk(
        state.temperature,
        ln_nk,
        ln_ntot,
        hvector,
        state.ln_normalized_pressure,
    )
    An = formula_matrix @ jnp.exp(ln_nk)
    Am = formula_matrix_cond @ jnp.exp(ln_mk)
    if initial_residual_policy == "computed_fresh":
        residual = _compute_current_fresh_residual(
            ln_nk,
            ln_mk,
            ln_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            gk,
            hvector_cond,
            epsilon,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
        )
    else:
        residual = jnp.inf
    history = []

    for iter_count in range(max_iter):
        if float(residual) <= float(residual_crit):
            break
        traced_exact_input_bundle_context = _with_lnnk_source_state_trace(
            exact_input_bundle_context,
            ln_nk=ln_nk,
            source_stage=(
                "trace_minimize_gibbs_cond_iterations ln_nk_init"
                if iter_count == 0
                else "trace_minimize_gibbs_cond_iterations accepted line-search ln_nk"
            ),
            producer_function=(
                "src/exogibbs/optimize/pipm_rgie_cond.py::"
                "trace_minimize_gibbs_cond_iterations"
            ),
            iter_count=iter_count,
        )

        (
            ln_nk,
            ln_mk,
            ln_ntot,
            gk,
            An,
            Am,
            residual,
            lam,
            metrics,
        ) = _update_all_with_metrics(
            ln_nk,
            ln_mk,
            ln_ntot,
            formula_matrix,
            formula_matrix_cond,
            b,
            state.temperature,
            state.ln_normalized_pressure,
            hvector,
            hvector_cond,
            gk,
            An,
            Am,
            residual,
            epsilon,
            iter_count,
            debug_nan=False,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
            exact_input_bundle_context=traced_exact_input_bundle_context,
            reduced_coupling_mode=reduced_coupling_mode,
            reduced_coupling_alpha_s=reduced_coupling_alpha_s,
            gas_step_scale=gas_step_scale,
            gas_step_direction_sign=gas_step_direction_sign,
            ntot_step_scale=ntot_step_scale,
            condensate_step_scale=condensate_step_scale,
            line_search_selection_policy=line_search_selection_policy,
        )
        record = {
            "iter": iter_count,
            "residual": float(residual),
            "lam": float(metrics["lam"]),
            "lam_heuristic": float(metrics["lam_heuristic"]),
            "lam_selected": float(metrics["lam_selected"]),
            "lam1_gas": float(metrics["lam1_gas"]),
            "lam1_cond": float(metrics["lam1_cond"]),
            "lam2_cond": float(metrics["lam2_cond"]),
            "n_backtracks": int(metrics["n_backtracks"]),
            "residual_before": float(metrics["residual_before"]),
            "residual_after": float(metrics["residual_after"]),
            "line_search_used": bool(metrics["line_search_used"]),
            "line_search_accept_kind": metrics["line_search_accept_kind"],
            "line_search_selection_policy": metrics["line_search_selection_policy"],
            "line_search_composite_merit": float(metrics["line_search_composite_merit"]),
            "line_search_current_composite_merit": float(
                metrics["line_search_current_composite_merit"]
            ),
            "line_search_best_composite_found": bool(
                metrics["line_search_best_composite_found"]
            ),
            "line_search_best_composite_merit": float(
                metrics["line_search_best_composite_merit"]
            ),
            "line_search_best_composite_lam": float(
                metrics["line_search_best_composite_lam"]
            ),
            "line_search_best_composite_residual": float(
                metrics["line_search_best_composite_residual"]
            ),
            "limiting_index": int(metrics["limiting_index"]),
            "max_abs_delta_ln_nk": float(metrics["max_abs_delta_ln_nk"]),
            "unscaled_max_abs_delta_ln_nk": float(
                metrics["unscaled_max_abs_delta_ln_nk"]
            ),
            "max_abs_raw_delta_ln_mk": float(metrics["max_abs_raw_delta_ln_mk"]),
            "max_abs_clipped_delta_ln_mk": float(metrics["max_abs_clipped_delta_ln_mk"]),
            "delta_ln_ntot": float(metrics["delta_ln_ntot"]),
            "unscaled_delta_ln_ntot": float(metrics["unscaled_delta_ln_ntot"]),
            "gas_step_scale": float(metrics["gas_step_scale"]),
            "gas_step_direction_sign": float(metrics["gas_step_direction_sign"]),
            "ntot_step_scale": float(metrics["ntot_step_scale"]),
            "ntot_step_factor": float(metrics["ntot_step_factor"]),
            "condensate_step_scale": float(metrics["condensate_step_scale"]),
            "gas_step_factor": float(metrics["gas_step_factor"]),
            "pi_norm": float(metrics["pi_norm"]),
            "reduced_resn": float(metrics["reduced_resn"]),
            "reduced_row_scale_min": float(metrics["reduced_row_scale_min"]),
            "reduced_row_scale_max": float(metrics["reduced_row_scale_max"]),
            "reduced_row_scale_ratio": float(metrics["reduced_row_scale_ratio"]),
            "reduced_mat_maxabs": float(metrics["reduced_mat_maxabs"]),
            "reduced_vec_maxabs": float(metrics["reduced_vec_maxabs"]),
            "reduced_qk_maxabs": float(metrics["reduced_qk_maxabs"]),
            "reduced_solver_backend": metrics["reduced_solver_backend"],
            "reduced_coupling_mode": metrics["reduced_coupling_mode"],
            "reduced_coupling_alpha_s": float(metrics["reduced_coupling_alpha_s"]),
            "reduced_coupling_uses_capped_s": bool(
                metrics["reduced_coupling_uses_capped_s"]
            ),
            "reduced_factorization_succeeded": bool(metrics["reduced_factorization_succeeded"]),
            "reduced_regularization_mode": metrics["reduced_regularization_mode"],
            "reduced_regularization_used": float(metrics["reduced_regularization_used"]),
        }
        if metrics.get("exact_input_bundle_emitted", False):
            record["exact_input_bundle"] = metrics["exact_input_bundle"]
        should_record_trial_lambdas = trial_lambda_every_iter or (
            trial_lambda_iterations is not None and iter_count in set(trial_lambda_iterations)
        )
        if should_record_trial_lambdas:
            record["trial_lambda_diagnostics"] = diagnose_iteration_lambda_trials(
                state,
                ln_nk=ln_nk - metrics["lam"] * metrics["delta_ln_nk"],
                ln_mk=ln_mk - metrics["lam"] * metrics["delta_ln_mk"],
                ln_ntot=ln_ntot - metrics["lam"] * metrics["delta_ln_ntot"],
                formula_matrix=formula_matrix,
                formula_matrix_cond=formula_matrix_cond,
                hvector_func=hvector_func,
                hvector_cond_func=hvector_cond_func,
                epsilon=epsilon,
                element_indices=element_indices,
                lambda_trials=trial_lambda_values,
                lambda_multipliers=trial_lambda_multipliers,
                extra_lambda_trials=extra_trial_lambda_values,
                reduced_solver=reduced_solver,
                regularization_mode=regularization_mode,
                regularization_strength=regularization_strength,
                charge_row_index=trial_lambda_charge_row_index,
            )
        should_record_gas_limiter = gas_limiter_every_iter or (
            gas_limiter_iterations is not None and iter_count in set(gas_limiter_iterations)
        )
        if should_record_gas_limiter:
            record["gas_limiter_diagnostics"] = diagnose_gas_step_limiter_and_direction(
                state,
                ln_nk=ln_nk - metrics["lam"] * metrics["delta_ln_nk"],
                ln_mk=ln_mk - metrics["lam"] * metrics["delta_ln_mk"],
                ln_ntot=ln_ntot - metrics["lam"] * metrics["delta_ln_ntot"],
                formula_matrix=formula_matrix,
                formula_matrix_cond=formula_matrix_cond,
                hvector_func=hvector_func,
                hvector_cond_func=hvector_cond_func,
                epsilon=epsilon,
                element_indices=element_indices,
                gas_species_names=gas_species_names,
                top_k=gas_limiter_top_k,
                reduced_solver=reduced_solver,
                regularization_mode=regularization_mode,
                regularization_strength=regularization_strength,
            )
        history.append(record)
        if record["lam"] <= tiny_step:
            break

    return {
        "epsilon": float(epsilon),
        "residual_crit": float(residual_crit),
        "n_iter": len(history),
        "final_residual": float(residual),
        "reduced_solver": reduced_solver,
        "regularization_mode": regularization_mode,
        "regularization_strength": float(regularization_strength),
        "reduced_coupling_mode": reduced_coupling_mode,
        "reduced_coupling_alpha_s": float(reduced_coupling_alpha_s),
        "gas_step_scale": float(gas_step_scale),
        "gas_step_direction_sign": float(gas_step_direction_sign),
        "ntot_step_scale": (
            float(gas_step_scale) if ntot_step_scale is None else float(ntot_step_scale)
        ),
        "condensate_step_scale": float(condensate_step_scale),
        "initial_residual_policy": initial_residual_policy,
        "line_search_selection_policy": line_search_selection_policy,
        "converged": bool(float(residual) <= float(residual_crit)),
        "hit_max_iter": bool(len(history) >= max_iter and float(residual) > float(residual_crit)),
        "history": history,
        "ln_nk": ln_nk,
        "ln_mk": ln_mk,
        "ln_ntot": ln_ntot,
    }


def trace_minimize_gibbs_cond_epsilon_sweep(
    state: ThermoState,
    ln_nk_init: jnp.ndarray,
    ln_mk_init: jnp.ndarray,
    ln_ntot_init: float,
    formula_matrix: jnp.ndarray,
    formula_matrix_cond: jnp.ndarray,
    hvector_func,
    hvector_cond_func,
    epsilons: Sequence[float],
    max_iter: int = 1000,
    element_indices: Optional[jnp.ndarray] = None,
    tiny_step: float = 1.0e-14,
    reduced_solver: str = DEFAULT_REDUCED_SOLVER,
    regularization_mode: str = DEFAULT_REGULARIZATION_MODE,
    regularization_strength: float = DEFAULT_REGULARIZATION_STRENGTH,
) -> Dict[str, Any]:
    """Trace one layer over a fixed list of epsilon values and summarize stagnation."""

    traces = []
    limiter_names = {
        0: "none_or_full_step",
        1: "gas_step_limiter",
        2: "condensate_step_limiter",
        3: "sk_limiter",
    }
    for epsilon in epsilons:
        trace = trace_minimize_gibbs_cond_iterations(
            state,
            ln_nk_init,
            ln_mk_init,
            ln_ntot_init,
            formula_matrix,
            formula_matrix_cond,
            hvector_func,
            hvector_cond_func,
            epsilon=float(epsilon),
            residual_crit=float(jnp.exp(jnp.asarray(epsilon))),
            max_iter=max_iter,
            element_indices=element_indices,
            tiny_step=tiny_step,
            reduced_solver=reduced_solver,
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength,
        )
        history = trace["history"]
        first_tiny = next((rec for rec in history if rec["lam"] <= tiny_step), None)
        first_tiny_iter = None if first_tiny is None else first_tiny["iter"]
        first_tiny_limiter = None if first_tiny is None else limiter_names.get(first_tiny["limiting_index"], "unknown")
        residuals = [rec["residual"] for rec in history]
        residual_decreased_before_stagnation = any(
            curr < prev for prev, curr in zip(residuals[:-1], residuals[1:])
        )
        row_scale_ratio = None if first_tiny is None else first_tiny["reduced_row_scale_ratio"]
        trace["summary"] = {
            "made_progress": residual_decreased_before_stagnation,
            "first_tiny_lam_iter": first_tiny_iter,
            "first_tiny_lam_limiter": first_tiny_limiter,
            "residual_decreased_before_stagnation": residual_decreased_before_stagnation,
            "appears_ill_scaled": False if row_scale_ratio is None else bool(row_scale_ratio > 1.0e12),
        }
        traces.append(trace)

    return {"epsilons": [float(eps) for eps in epsilons], "traces": traces}
