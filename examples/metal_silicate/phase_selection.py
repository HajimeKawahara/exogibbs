"""Mixed-metal insertion minima and explicit phase selection
=========================================================

Global certificates require supplied global curvature bounds for the actual
phase scalars on their declared domains. Numerical searches alone never
certify absence or a nonconvex assemblage's stability.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Optional

import numpy as np
from scipy.optimize import least_squares, linprog, minimize

from common_gibbs import GibbsMinimumResult, minimize_gibbs
from full_potential import PhaseCallback, PhaseState
from local import _budgets, build_problem


def restrict_phase_callbacks(record, problem, callbacks):
    """Embed active amounts into each provider's complete component order.

    ``build_problem`` removes components containing exact-zero global elements.
    Providers still receive their declared vectors, with zeros in those slots;
    the returned potentials and optional scalar gradients follow active order.
    """
    restricted = {}
    for phase, section in zip(problem.phases, problem.phase_slices):
        positions = np.array([record["phases"][phase].index(name) for name in problem.species[section]])

        def evaluate(t, p, n, phase=phase, positions=positions):
            full = np.zeros(len(record["phases"][phase]))
            full[positions] = n
            state = callbacks[phase](t, p, full)
            return PhaseState(np.asarray(state.mu_rt)[positions], state.gibbs_rt)

        derivative_provider = getattr(callbacks[phase], "energy_value_and_grad_rt", None)
        if derivative_provider is not None:
            def energy_gradient(t, p, n, phase=phase, positions=positions,
                                provider=derivative_provider):
                full = np.zeros(len(record["phases"][phase]))
                full[positions] = n
                energy, gradient = provider(t, p, full)
                return energy, np.asarray(gradient)[positions]

            evaluate.energy_value_and_grad_rt = energy_gradient
        restricted[phase] = evaluate
    return restricted


def local_metal_selection_accepted(selection: dict) -> bool:
    """Separate a completed local metal test from missing global host evidence.

    An accepted absent-branch result alone is insufficient after favorable
    insertion if the metal-bearing solve failed. This gate consumes a saved
    selection mapping and does not establish material or global acceptance.
    """
    result, insertion = selection.get("result") or {}, selection.get("insertion") or {}
    reasons = tuple(selection.get("reasons", ()))
    status = selection.get("status")
    return bool(result.get("accepted") is True and insertion.get("minimum_certified") is True
                and ((status in ("metal_present", "metal_absent") and not reasons)
                     or (status == "unresolved" and reasons == ("Host global stability is not established.",))))


def local_metal_status(selection: dict) -> str:
    """Classify an accepted local branch without changing its global status.

    This is relative to the declared alloy composition domain. A failed
    metal-bearing solve never turns the retained zero-metal reference into
    accepted absence. Existing saved selections need no new diagnostic fields.
    """
    amount = selection.get("metal_amount_mol")
    if (not local_metal_selection_accepted(selection)
            or isinstance(amount, (bool, np.bool_))
            or not isinstance(amount, (int, float, np.integer, np.floating))
            or not np.isfinite(amount) or amount < 0):
        return "unresolved"
    return "metal_absent" if amount == 0 else "metal_present"


@dataclass(frozen=True)
class InsertionMinimum:
    """Insertion energy per component mole, divided by the common RT."""

    composition: np.ndarray
    upper_bound_rt: float
    lower_bound_rt: Optional[float]
    uncertainty_rt: Optional[float]
    minimum_certified: bool
    reason: str


def _linear_minimum(cost, lower, upper):
    """Exactly solve the linear box-simplex subproblem up to roundoff."""
    result = lower.copy()
    remaining = 1 - result.sum()
    for index in np.argsort(cost):
        amount = min(remaining, upper[index] - result[index])
        result[index] += amount
        remaining -= amount
    if abs(remaining) > 1e-12:
        raise ValueError("The composition domain does not intersect the simplex.")
    return result


def minimize_insertion(
    phase: PhaseCallback, temperature_k: float, pressure_bar: float,
    formula: np.ndarray, elemental_potentials_rt: np.ndarray,
    lower: np.ndarray, upper: np.ndarray, *,
    curvature_lower_bound_rt: Optional[float] = None,
    tolerance: float = 1e-8, maxiter: int = 1000,
) -> InsertionMinimum:
    """Search a mixed phase and bound its minimum over a box in the simplex.

    ``curvature_lower_bound_rt`` must bound the Hessian of this exact molar
    phase scalar globally after eliminating one composition coordinate.
    Standards and elemental potentials add linear terms only. Nonnegative
    bounds give a convex supporting plane. Negative bounds give a conservative
    quadratic remainder using the enclosing box diameter. Without a bound,
    a negative trial can reject absence but cannot certify a global minimum.

    All callback evaluations use one mole of phase in the supplied component
    order. Fixed-zero components may have undefined insertion derivatives;
    free-zero components with singular derivatives yield an unresolved bound.
    """
    values = tuple(map(np.asarray, (formula, elemental_potentials_rt, lower, upper)))
    if not all(np.isrealobj(value) for value in values):
        raise ValueError("Formula, potentials, and composition bounds must be real.")
    atoms, potentials, lo, hi = (value.astype(float) for value in values)
    if (lo.ndim != 1 or not lo.size or hi.shape != lo.shape
            or potentials.ndim != 1 or atoms.shape != (potentials.size, lo.size)
            or not all(np.all(np.isfinite(value)) for value in (atoms, potentials, lo, hi))
            or np.any(atoms < 0) or np.any(lo < 0) or np.any(hi > 1) or np.any(lo > hi)
            or lo.sum() > 1 or hi.sum() < 1
            or not all(np.isfinite(v) and v > 0 for v in (temperature_k, pressure_bar, tolerance))):
        raise ValueError("Supply finite potentials and a feasible nonnegative composition domain.")
    if curvature_lower_bound_rt is not None and not np.isfinite(curvature_lower_bound_rt):
        raise ValueError("A supplied global curvature bound must be finite.")
    if not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer.")
    linear = atoms.T @ potentials
    free = hi > lo

    def evaluate(x):
        state = phase(temperature_k, pressure_bar, x)
        mu = np.asarray(state.mu_rt, dtype=float)
        present = x > 0
        if (mu.shape != x.shape or not np.isfinite(state.gibbs_rt)
                or not np.all(np.isfinite(mu[present]))
                or not np.isclose(x[present] @ mu[present], state.gibbs_rt,
                                  rtol=5e-9, atol=1e-10)):
            raise ValueError("The insertion callback must return a finite extensive G and its full potentials.")
        return float(state.gibbs_rt - linear @ x), mu - linear

    def derivative(x):
        value, gradient = evaluate(x)
        gradient[~free] = 0.
        # Only trial derivatives use one-sided differences at singular zero
        # components. The final certificate uses the actual full potentials.
        for index in np.flatnonzero(~np.isfinite(gradient)):
            trial = x.copy()
            step = 1e-7
            trial[index] += step
            gradient[index] = (evaluate(trial)[0] - value) / step
        return gradient

    width = hi - lo
    center = lo + (1 - lo.sum()) * width / width.sum() if np.any(free) else lo.copy()
    # Different starts can find an instability but do not establish stability.
    seeds = [center]
    for index in np.flatnonzero(free):
        cost = np.zeros(lo.size)
        cost[index] = -1
        vertex = _linear_minimum(cost, lo, hi)
        seeds.append(.5 * center + .5 * vertex)
    candidates = [(evaluate(center)[0], center)]
    if np.any(free):
        for seed in seeds:
            result = minimize(lambda x: evaluate(x)[0], seed, jac=derivative, method="SLSQP",
                              bounds=list(zip(lo, hi)),
                              constraints={"type": "eq", "fun": lambda x: x.sum() - 1,
                                           "jac": lambda x: np.ones_like(x)},
                              options={"ftol": min(1e-13, tolerance * .001), "maxiter": maxiter})
            x = np.asarray(result.x)
            if (np.all(x >= lo) and np.all(x <= hi) and abs(x.sum() - 1) < 1e-12):
                candidates.append((evaluate(x)[0], x))
    upper_value, x = min(candidates, key=lambda item: item[0])
    # Energy differences lose sensitivity before trace-component chemical
    # potentials do. Refine the free composition KKT equations in log amounts;
    # the global bound below, not this root, establishes minimum acceptance.
    interior = free & (x > lo + 1e-10 * width) & (x < hi - 1e-10 * width)
    if np.any(interior) and np.all(x[interior] > 0):
        fixed = x.copy()
        fixed[free & ~interior & (x <= lo + 1e-10 * width)] = lo[free & ~interior & (x <= lo + 1e-10 * width)]
        fixed[free & ~interior & (x >= hi - 1e-10 * width)] = hi[free & ~interior & (x >= hi - 1e-10 * width)]

        def stationarity(log_values):
            point = fixed.copy()
            point[interior] = np.exp(log_values)
            gradient = evaluate(point)[1][interior]
            return np.r_[np.log(point.sum()), gradient[1:] - gradient[0]]

        lower_logs = np.full(interior.sum(), -np.inf)
        positive_lower = lo[interior] > 0
        lower_logs[positive_lower] = np.log(lo[interior][positive_lower])
        root = least_squares(stationarity, np.log(x[interior]),
                             bounds=(lower_logs, np.log(hi[interior])),
                             ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=maxiter)
        refined = fixed.copy()
        refined[interior] = np.exp(root.x)
        cost = evaluate(refined)[0]
        allowance = 128 * np.finfo(float).eps * (1 + abs(upper_value))
        if abs(refined.sum() - 1) <= 1e-12 and cost <= upper_value + allowance:
            x = refined
    upper_value, gradient = evaluate(x)  # Fresh provider evaluation.
    if curvature_lower_bound_rt is None:
        return InsertionMinimum(x, upper_value, None, None, False, "No global curvature bound supplied.")
    if not np.all(np.isfinite(gradient[free])):
        return InsertionMinimum(x, upper_value, None, None, False, "A free endpoint has a singular potential.")
    gradient = np.where(free, gradient, 0.)
    vertex = _linear_minimum(gradient, lo, hi)
    radius_squared = np.sum(np.maximum((lo - x)**2, (hi - x)**2))
    roundoff = 128 * np.finfo(float).eps * (1 + abs(upper_value) + np.linalg.norm(gradient, 1))
    lower_value = (upper_value + gradient @ (vertex - x)
                   + .5 * min(0., curvature_lower_bound_rt) * radius_squared - roundoff)
    gap = max(0., upper_value - lower_value)
    certified = bool(gap <= tolerance)
    reason = "Global supporting bound closes the minimum." if certified else "Global minimum uncertainty exceeds tolerance."
    return InsertionMinimum(x, upper_value, float(lower_value), float(gap), certified, reason)


@dataclass(frozen=True)
class MetalSelection:
    """Numerical phase selection within the declared scalar model and domain."""

    status: str
    result: Optional[GibbsMinimumResult]
    insertion: Optional[InsertionMinimum]
    metal_amount_mol: float
    metal_composition: Optional[np.ndarray]
    reasons: tuple[str, ...]
    local_attempts: tuple[dict, ...] = ()
    metal_free_result: Optional[GibbsMinimumResult] = None
    metal_free_insertion: Optional[InsertionMinimum] = None
    metal_composition_domain: Optional[dict] = None


def _domain_position(components, composition, lower, upper, unsupported):
    """Record domain contact separately from exact-zero inventory support."""
    if composition is None:
        return None
    composition = np.asarray(composition)
    lower_slack, upper_slack = composition - lower, upper - composition
    active = []
    for index, component in enumerate(components):
        for bound, slack, imposed in (("lower", lower_slack[index], lower[index] > 0),
                                      ("upper", upper_slack[index], upper[index] < 1)):
            if slack <= 1e-8:
                kind = ("exact_zero_budget" if unsupported[index] else
                        "composition_box" if imposed else "simplex_boundary")
                active.append({"component": component, "bound": bound,
                               "fraction_slack": float(slack), "kind": kind})
    return {"composition": composition.copy(), "lower_fraction_slack": lower_slack,
            "upper_fraction_slack": upper_slack, "active_bounds": tuple(active),
            "composition_box_contact": any(row["kind"] == "composition_box" for row in active)}


def _insertion_seed(record, budget, absent, composition, callbacks, temperature, pressure, fraction):
    """Enter the metal-bearing feasible set without adding atoms or floors."""
    names = [name for values in record["phases"].values() for name in values]
    formula = np.array([[record["component_formulas"][name].get(element, 0.) for name in names]
                        for element in record["elements"]])
    positions = np.array([names.index(name) for name in record["phases"]["metal"]])
    positive, scale = budget > 0, budget.sum()
    allowed = ~np.any(formula[~positive] > 0, axis=0)
    normalized = budget[positive] / scale
    baseline = absent.component_amounts_mol / scale
    limits = np.min(np.where(formula[positive] > 0,
                             normalized[:, None] / np.where(formula[positive] > 0, formula[positive], 1),
                             np.inf), axis=0)
    reference = np.where(allowed, np.where(baseline > 0, baseline, limits), 1.)
    matrix = formula[positive] * reference / normalized[:, None]
    fixed_composition = np.zeros((len(positions), len(names)))
    fixed_composition[:, positions] = np.eye(len(positions)) - composition[:, None]
    cost = np.zeros(len(names))
    cost[positions] = -reference[positions]
    # Keep host/gas trial changes relative to their actual amounts, including
    # traces. This is a starting-point trust region, not an equilibrium bound.
    start_bounds = [(.5, 1.5) if baseline[index] > 0 else (0, None) if allowed[index] else (0, 0)
                    for index in range(len(names))]
    endpoint = linprog(cost, A_eq=np.vstack((matrix, fixed_composition * reference)),
                       b_eq=np.r_[np.ones(positive.sum()), np.zeros(len(positions))],
                       bounds=start_bounds, method="highs")
    if not endpoint.success or endpoint.x[positions].sum() <= 0:
        raise ValueError("No atom-conserving insertion direction supports the incipient composition.")
    direction = scale * reference * endpoint.x - absent.component_amounts_mol
    last_rejection = "No trial was evaluated."
    for _ in range(16):
        candidate = absent.component_amounts_mol + fraction * direction
        try:
            energy, start = 0., 0
            for phase, components in record["phases"].items():
                section = slice(start, start + len(components))
                energy += callbacks[phase](temperature, pressure, candidate[section]).gibbs_rt
                start = section.stop
            if np.isfinite(energy) and energy < absent.gibbs_rt:
                return candidate
            last_rejection = f"G change divided by total atoms: {(energy - absent.gibbs_rt) / scale:.16g}."
        except (ValueError, RuntimeError, FloatingPointError) as error:
            last_rejection = f"{type(error).__name__}: {error}"
        fraction *= .5
    raise ValueError("No evaluated energy-decreasing feasible insertion start was found. " + last_rejection)


def select_metal_phase(
    record: dict, element_amounts_mol: np.ndarray,
    temperature_k: float, pressure_bar: float, callbacks: Mapping[str, PhaseCallback],
    metal_lower: np.ndarray, metal_upper: np.ndarray, *,
    convex_phase_bounds: Optional[Mapping[str, float]] = None,
    maxiter: int = 1000, tolerance: float = 1e-8, allow_metal: bool = True,
) -> MetalSelection:
    """Evaluate metal-free and metal-bearing branches without a metal floor.

    Callbacks follow each phase's complete record order. Components containing
    exactly absent elements are removed before any logarithmic solve. The
    optional bounds are proven global molar curvature lower bounds, not sampled
    Hessians. Certifying the whole assemblage requires nonnegative bounds for
    every included phase on its declared domain. Missing/nonconvex host evidence
    always returns ``unresolved`` even if the metal insertion test passes.

    Present candidates obey the same homogeneous composition inequalities
    used in insertion. Unsupported zero-phase boundaries remain unresolved. Material calibration
    and the liquid-versus-crystal catalog are separate from this model's status.

    ``metal_free_result`` and ``metal_free_insertion`` preserve the absent
    branch and its phase-generation cost, including after a metal-bearing
    solve. ``insertion`` retains its existing selected-state meaning: it is
    approximately zero at coexistence, not the absent-branch boundary cost.
    ``allow_metal=False`` returns that constrained absent branch without a
    present-phase solve. A favorable insertion keeps its status unresolved;
    numerical acceptance of the constrained state does not establish absence.
    """
    if type(allow_metal) is not bool:
        raise ValueError("allow_metal must be a boolean.")
    if "metal" not in record["phases"] or set(callbacks) != set(record["phases"]):
        raise ValueError("Supply the metal phase and exactly one callback per declared phase.")
    budget = _budgets(element_amounts_mol, len(record["elements"]))
    phases = tuple(record["phases"])
    names = tuple(name for phase in phases for name in record["phases"][phase])
    bounds = dict(convex_phase_bounds or {})
    if (set(bounds) - set(phases)
            or any(not np.isrealobj(value) or not np.isfinite(value) for value in bounds.values())):
        raise ValueError("Curvature evidence must name declared phases and contain finite bounds.")
    elements = record["elements"]
    metal_names = record["phases"]["metal"]
    metal_formula = np.array([[record["component_formulas"][name].get(e, 0)
                               for name in metal_names] for e in elements], dtype=float)
    if not np.isrealobj(metal_lower) or not np.isrealobj(metal_upper):
        raise ValueError("Metal composition bounds must be real.")
    lo, hi = np.asarray(metal_lower, dtype=float), np.asarray(metal_upper, dtype=float).copy()
    if (lo.shape != (len(metal_names),) or hi.shape != lo.shape
            or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi))
            or np.any(lo < 0) or np.any(hi > 1) or np.any(lo > hi)
            or lo.sum() > 1 or hi.sum() < 1
            or not all(np.isfinite(v) and v > 0 for v in (temperature_k, pressure_bar, tolerance))
            or not isinstance(maxiter, int) or maxiter < 1):
        raise ValueError("Supply positive conditions and feasible bounds in the full metal component order.")

    declared_upper = hi.copy()
    unsupported = np.any(metal_formula[budget == 0] != 0, axis=0)
    absent, metal_free_trial = None, None

    def finish(*args):
        selected = MetalSelection(*args)
        effective_upper = np.where(unsupported, 0., declared_upper)
        domain = {
            "component_order": tuple(metal_names), "lower_atomic_fractions": lo.copy(),
            "upper_atomic_fractions": declared_upper.copy(),
            "effective_upper_atomic_fractions": effective_upper,
            "zero_budget_components": tuple(name for name, excluded in zip(metal_names, unsupported) if excluded),
            "contact_tolerance": 1e-8,
            "metal_free_incipient": _domain_position(
                metal_names, None if metal_free_trial is None else metal_free_trial.composition,
                lo, effective_upper, unsupported),
            "selected_metal": _domain_position(metal_names, selected.metal_composition,
                                                lo, effective_upper, unsupported),
            "selected_composition_constraints": (() if selected.result is None else tuple(
                row for row in selected.result.composition_constraints if row["phase"] == "metal")),
            "interpretation": "Composition-box contact is a model restriction, not a physical phase boundary or a calibration certificate.",
        }
        return replace(selected, metal_free_result=absent, metal_free_insertion=metal_free_trial,
                       metal_composition_domain=domain)

    def solve(selected, initial=None):
        problem = build_problem(record, budget, lambda t, p: np.zeros(len(names)), phases=selected)
        restricted = restrict_phase_callbacks(record, problem, callbacks)
        composition_bounds = {}
        for phase, section in zip(problem.phases, problem.phase_slices):
            positions = np.array([record["phases"][phase].index(name) for name in problem.species[section]])
            if phase == "metal":
                composition_bounds[phase] = (lo[positions], hi[positions])
        return minimize_gibbs(problem, temperature_k, pressure_bar, budget, restricted, maxiter=maxiter,
                              initial_component_amounts_mol=initial,
                              phase_composition_bounds=composition_bounds)

    failures = (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError)
    try:
        absent = solve(tuple(phase for phase in phases if phase != "metal"))
    except failures as error:
        return finish("unresolved", None, None, 0., None,
                              (f"Metal-free branch unavailable: {type(error).__name__}: {error}",))
    if not absent.accepted:
        return finish("unresolved", absent, None, 0., None, ("Metal-free local minimization failed.", *absent.audit_reasons))
    hi[unsupported] = 0.
    host_certified = all(bounds.get(phase, -np.inf) >= 0 for phase in phases if phase != "metal")
    if np.any(lo > hi) or hi.sum() < 1:
        reasons = () if host_certified else ("Host global stability is not established.",)
        return finish("metal_absent" if not reasons else "unresolved", absent, None, 0., None,
                              reasons + ("The metal domain is excluded by exact-zero element budgets.",))

    def insertion(result):
        return minimize_insertion(callbacks["metal"], temperature_k, pressure_bar,
                                  metal_formula, result.elemental_potentials_rt, lo, hi,
                                  curvature_lower_bound_rt=bounds.get("metal"), tolerance=tolerance,
                                  maxiter=maxiter)

    try:
        trial = metal_free_trial = insertion(absent)
    except failures as error:
        return finish("unresolved", absent, None, 0., None,
                              (f"Metal insertion unavailable: {type(error).__name__}: {error}",))
    if trial.minimum_certified and trial.lower_bound_rt >= -tolerance:
        reasons = () if host_certified else ("Host global stability is not established.",)
        return finish("metal_absent" if not reasons else "unresolved", absent, trial, 0., None, reasons)
    if trial.upper_bound_rt >= -tolerance:
        return finish("unresolved", absent, trial, 0., None, (trial.reason,))
    if not allow_metal:
        return finish("unresolved", absent, trial, 0., None,
                      ("The constrained metal-free branch has a favorable metal insertion.",))
    attempts, candidates = [], []
    for fraction in (.01, .1):
        try:
            seed = _insertion_seed(record, budget, absent, trial.composition, callbacks,
                                   temperature_k, pressure_bar, fraction)
            candidate = solve(phases, seed)
            attempts.append({"insertion_fraction": fraction, "result": candidate})
            candidates.append(candidate)
            if candidate.accepted:
                break
        except failures as error:
            attempts.append({"insertion_fraction": fraction, "error": f"{type(error).__name__}: {error}"})
    if not candidates:
        return finish("unresolved", absent, trial, 0., None,
                              ("Metal-bearing branch unavailable; see local attempts.",), tuple(attempts))
    present = min(candidates, key=lambda result: (not result.accepted, result.gibbs_rt))
    amounts = present.component_amounts_mol[[names.index(name) for name in metal_names]]
    total = float(amounts.sum())
    reasons = list(present.audit_reasons)
    if not present.accepted:
        reasons.append("Metal-bearing local minimization failed.")
    if total <= 0:
        return finish("unresolved", present, None, total, None,
                              tuple(reasons + ["No positive metal candidate was obtained."]), tuple(attempts))
    composition = amounts / total
    if np.any(composition < lo - 1e-10) or np.any(composition > hi + 1e-10):
        return finish("unresolved", present, None, total, composition,
                              tuple(reasons + ["The metal candidate lies outside the declared composition domain."]), tuple(attempts))
    try:
        trial = insertion(present)
    except failures as error:
        return finish("unresolved", present, None, total, composition,
                              tuple(reasons + [f"Metal insertion unavailable: {type(error).__name__}: {error}"]), tuple(attempts))
    if not trial.minimum_certified or max(abs(trial.upper_bound_rt), abs(trial.lower_bound_rt or 0.)) > tolerance:
        reasons.append("The present metal does not attain a certified zero insertion minimum.")
    candidate = callbacks["metal"](temperature_k, pressure_bar, composition)
    actual_cost = candidate.gibbs_rt - present.elemental_potentials_rt @ metal_formula @ composition
    if abs(actual_cost - trial.upper_bound_rt) > tolerance:
        reasons.append("The present metal composition is not the minimizing incipient composition.")
    if abs(total / budget.sum() * trial.upper_bound_rt) > tolerance:
        reasons.append("Metal amount and insertion energy violate complementarity.")
    if present.gibbs_rt > absent.gibbs_rt + tolerance * budget.sum():
        reasons.append("Allowing metal raised the minimum energy.")
    if not host_certified:
        reasons.append("Host global stability is not established.")
    return finish("metal_present" if not reasons else "unresolved", present, trial,
                          total, composition, tuple(reasons), tuple(attempts))
