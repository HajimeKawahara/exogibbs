"""Checked first-order responses of a supplied, declared equilibrium branch."""

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .kernel import scaled_basis
from .solve import _energy, _geometry, _physical_state
from .types import Options, Problem, Support


class EquilibriumTangent(NamedTuple):
    """Physical directional derivatives in the Problem's original ordering.

    Units are the corresponding Result field's units per input perturbation.
    log_amounts is d(log(n)) on positive components and zero on excluded ones.
    Chemical potentials of excluded components and unsupported elemental
    potentials are undefined in Result; their tangent entries are zero.
    """

    amounts_mol: Any
    phase_amounts_mol: Any
    phase_element_amounts_mol: Any
    gibbs_j: Any
    chemical_potentials_j_mol: Any
    element_potentials_j_mol: Any
    constraint_multipliers_j_mol: Any
    conservation_residual_mol: Any
    log_amounts: Any


@dataclass(frozen=True)
class FixedSupportLinearization:
    """A first-order local operator and its host-side acceptance diagnostics.

    Construction is host-only. jvp and vjp support JAX jit after construction.
    No equilibrium or phase search is performed. Excluded-phase stability is
    not certified; derivative_valid concerns the supplied branch only.
    """

    derivative_valid: bool
    rejection_reasons: tuple
    diagnostics: dict
    support: Support
    _push: Callable
    _residual: Callable
    _zeros: tuple

    def jvp(self, dtemperature_k: Any, dpressure_pa: Any,
            delement_amounts_mol: Any, dparameters: Any = ()) -> EquilibriumTangent:
        """Apply the local response to an input direction in K, Pa and mol atoms.

        dparameters must match the parameters PyTree given at construction.
        Invalid baselines return NaN physical tangents; no floor is inserted.
        """
        return self._push(jnp.asarray(dtemperature_k, dtype=float),
                          jnp.asarray(dpressure_pa, dtype=float),
                          jnp.asarray(delement_amounts_mol, dtype=float), dparameters)

    def linear_residual(self, dtemperature_k: Any, dpressure_pa: Any,
                        delement_amounts_mol: Any, dparameters: Any = ()) -> Any:
        """Return max|K dz-rhs| / max(1,max|rhs|) for this input direction.

        This nonlinear diagnostic is separate from the JVP/VJP output tuple.
        A small linear residual does not replace baseline acceptance checks.
        """
        return self._residual(jnp.asarray(dtemperature_k, dtype=float),
                              jnp.asarray(dpressure_pa, dtype=float),
                              jnp.asarray(delement_amounts_mol, dtype=float), dparameters)

    def vjp(self, cotangent: EquilibriumTangent) -> tuple:
        """Transpose the same operator, returning (dT, dP, db, dparameters)."""
        return jax.vjp(self._push, *self._zeros)[1](cotangent)


def linearize_fixed_support(
    problem: Problem, *, temperature_k: Any, pressure_pa: Any,
    element_amounts_mol: Any, amounts_mol: Any, support: Support,
    parameters: Any = (), options: Options = Options(),
) -> FixedSupportLinearization:
    """Re-audit and linearize a supplied fixed-support equilibrium.

    The supported first version requires positive element budgets and
    independent element/active-domain rows. Component amounts are strictly
    positive on support, exactly zero elsewhere. Active domain multipliers
    must be strictly positive. Material derivative_domain predicates are
    honored for present phases. Use x64 for the default tolerances.

    This computes a local branch response, not a derivative through phase
    selection. Missing competing-phase bounds do not establish stability;
    this API deliberately reports excluded_phase_stability_checked=False.
    Existing solve_fixed_support's stronger derivative gate is unchanged.
    """
    indices, active, _, matrix, _, tangent, rank = _geometry(problem, support)
    t, p = float(temperature_k), float(pressure_pa)
    b, n = np.asarray(element_amounts_mol, float), np.asarray(amounts_mol, float)
    count, elements = problem.atom_matrix.shape[1], len(problem.elements)
    if b.shape != (elements,) or n.shape != (count,):
        raise ValueError("Amounts and budget must match the Problem ordering.")
    if rank != len(matrix):
        raise ValueError("Independent element and active-domain rows are required.")
    if not (np.isfinite(t) and t > 0 and np.isfinite(p) and p > 0
            and np.all(np.isfinite(b)) and np.all(b > 0)
            and np.all(np.isfinite(n)) and np.all(n >= 0)):
        raise ValueError("Positive finite T, P, budgets and nonnegative amounts are required.")
    absent = np.setdiff1d(np.arange(count), indices)
    if np.any(n[indices] <= 0) or np.any(n[absent] != 0):
        raise ValueError("Amounts must be strictly positive exactly on support.")
    for leaf in jax.tree_util.tree_leaves(parameters):
        if not np.issubdtype(np.asarray(leaf).dtype, np.floating):
            raise ValueError("Dynamic parameter leaves must have floating-point dtype.")
    total = b.sum()
    q = jnp.asarray(n[indices]/total)
    if not np.all(np.asarray(q) > 0):
        raise ValueError("Normalized positive amounts must remain representable.")
    aa = np.asarray(problem.atom_matrix)[:, indices]
    cc = np.asarray(problem.domain_matrix)
    equality = np.concatenate((aa, cc[active][:, indices]))
    row_scale = np.sum(np.abs(equality)*np.asarray(q), axis=1)
    energy = _energy(problem, indices)
    gradient = jax.grad(energy)
    value = float(energy(q, t, p, parameters))
    mu = np.asarray(gradient(q, t, p, parameters))
    # The physical dual is unique under the declared independent-row contract.
    dual = np.linalg.lstsq(equality.T, mu, rcond=None)[0]
    stationarity = float(np.max(np.abs(mu-equality.T @ dual)))
    conservation = float(np.max(np.abs(problem.atom_matrix @ n-b)/b))
    hessian = jax.hessian(energy)(q, t, p, parameters)
    log_jac = jax.jacfwd(lambda u: gradient(q*jnp.exp(u), t, p, parameters))(jnp.zeros(len(q)))
    if not (np.isfinite(value) and np.all(np.isfinite(mu)) and np.all(np.isfinite(hessian))
            and np.all(np.isfinite(log_jac))):
        raise ValueError("Material Gibbs value and derivatives must be finite on the declared branch.")
    kkt = jnp.block([[log_jac, -jnp.asarray(equality.T)],
                     [jnp.asarray(equality)*q[None, :]/row_scale[:, None],
                      jnp.zeros((len(equality), len(equality)))]])
    condition = float(np.linalg.cond(np.asarray(kkt)))
    row = np.linalg.svd(matrix, full_matrices=False)[2][:rank]
    scale, basis = scaled_basis(q, jnp.asarray(row))
    curvature = (float(jnp.linalg.eigvalsh(basis.T @ (scale[:, None]*hessian*scale[None, :]) @ basis)[0])
                 if tangent.shape[1] else float('inf'))
    domain_scales = np.array([n[section].sum()*np.linalg.norm(r)
                             for phase, section in zip(problem.phases, problem.component_slices)
                             for r in phase.domain_matrix])
    slack = cc @ n / np.where(domain_scales > 0, domain_scales, 1.)
    relevant = [i for i in range(len(cc)) if i not in active and np.any(cc[i, indices])]
    smooth = True
    for phase, section in zip(problem.phases, problem.component_slices):
        included = set(i-section.start for i in indices if section.start <= i < section.stop)
        if included:
            # A reachable omitted component in a present solution is a boundary.
            smooth &= included == set(range(len(phase.components)))
            if phase.derivative_domain is not None:
                predicate = np.asarray(phase.derivative_domain(t, p, jnp.asarray(n[section]), parameters))
                if predicate.shape or predicate.dtype != np.dtype(bool):
                    raise ValueError("derivative_domain must return a scalar boolean predicate.")
                smooth &= bool(predicate)
    finite_material = bool(np.all(np.isfinite(mu)) and np.all(np.isfinite(hessian)))
    checks = {
        'finite_material': finite_material,
        'stationarity': stationarity <= options.kkt_tolerance,
        'conservation': conservation <= options.conservation_tolerance,
        'strict_minimum': curvature > options.curvature_tolerance,
        'regular_log_kkt': np.isfinite(condition) and condition < 1/options.derivative_rcond,
        'active_domains': bool(np.all(np.abs(slack[active]) <= options.kkt_tolerance)),
        'inactive_domains': bool(np.all(slack[relevant] > options.support_tolerance)),
        'strict_active_multipliers': bool(np.all(dual[elements:] > options.kkt_tolerance)),
        'material_branch': bool(smooth),
    }
    reasons = tuple(key for key, passed in checks.items() if not passed)
    valid = not reasons
    zeros = (jnp.zeros(()), jnp.zeros(()), jnp.zeros_like(jnp.asarray(b)),
             jax.tree_util.tree_map(lambda x: jnp.zeros_like(jnp.asarray(x)), parameters))
    # Keep the normalization fixed at the anchor. db enters the constraints,
    # never an extra change of the amount gauge in the physical output map.
    def physical(u, temperature, pressure, params):
        return _physical_state(problem, support, energy, q*jnp.exp(u),
                               temperature, pressure, jnp.asarray(b), params)

    def solve_direction(dt, dp, db, dparams):
        _, partial = jax.jvp(lambda temperature, pressure, params: gradient(q, temperature, pressure, params),
                             (jnp.asarray(t), jnp.asarray(p), parameters), (dt, dp, dparams))
        rhs = jnp.concatenate((-partial, jnp.concatenate((db/total, jnp.zeros(len(active))))/row_scale))
        return jnp.linalg.solve(kkt, rhs), rhs

    def residual(dt, dp, db, dparams):
        response, rhs = solve_direction(dt, dp, db, dparams)
        return jnp.max(jnp.abs(kkt @ response-rhs))/jnp.maximum(1., jnp.max(jnp.abs(rhs)))

    def push(dt, dp, db, dparams):
        response, _ = solve_direction(dt, dp, db, dparams)
        du = response[:len(q)]
        _, values = jax.jvp(physical, (jnp.zeros_like(q), jnp.asarray(t), jnp.asarray(p), parameters),
                            (du, dt, dp, dparams))
        values = (*values[:-1], values[-1]-db,
                  jnp.zeros(count).at[indices].set(du))
        # Multiplying a NaN coefficient preserves the linear transpose contract.
        coefficient = 1. if valid else jnp.nan
        signal = sum(jnp.sum(x) for x in jax.tree_util.tree_leaves((dt, dp, db, dparams)))
        invalid_signal = (0. if valid else jnp.nan)*signal
        return EquilibriumTangent(*(coefficient*x+invalid_signal for x in values))

    diagnostics = dict(checks, condition_number=condition,
                       stationarity_error_rt=stationarity, conservation_error=conservation,
                       minimum_curvature=curvature,
                       minimum_positive_normalized_amount=float(q.min()),
                       excluded_phase_stability_checked=False)
    return FixedSupportLinearization(valid, reasons, diagnostics, support, push, residual, zeros)
