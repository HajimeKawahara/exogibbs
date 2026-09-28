"""Outward global insertion bounds for the declared associated-alloy scalar.

The physical expression and curvature of the H--O-free reference stay in
ExoEOS. Only the added H--O term is convexified. Numerical optimizers supply
anchors; interval tangent bounds, not optimizer success, certify minima.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
import hashlib
import heapq
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares, minimize

from m2_liquid_global import _I, _outward_float
from phase_selection import InsertionMinimum


class _First:
    """First derivatives in either binary64 trial or outward interval arithmetic."""
    def __init__(self, value, gradient):
        self.value, self.gradient = value, tuple(gradient)

    def lift(self, value):
        return value if isinstance(value, _First) else _First(value, [0]*len(self.gradient))

    def __add__(self, other):
        other = self.lift(other)
        return _First(self.value+other.value, [a+b for a, b in zip(self.gradient, other.gradient)])

    __radd__ = __add__

    def __neg__(self):
        return _First(-self.value, [-a for a in self.gradient])

    def __sub__(self, other):
        return self+-self.lift(other)

    def __rsub__(self, other):
        return self.lift(other)+-self

    def __mul__(self, other):
        other = self.lift(other)
        return _First(self.value*other.value, [a*other.value+self.value*b
                                              for a, b in zip(self.gradient, other.gradient)])

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.lift(other)
        return _First(self.value/other.value,
                      [(a-self.value*b/other.value)/other.value
                       for a, b in zip(self.gradient, other.gradient)])

    def __rtruediv__(self, other):
        return self.lift(other)/self

    def log(self):
        value = self.value.log() if isinstance(self.value, _I) else math.log(self.value)
        return _First(value, [a/self.value for a in self.gradient])


def _scalar_gradient(excess, costs, solutes, *, intervals):
    size = len(solutes)
    cast = _I if intervals else float
    xs = [_First(cast(v), [cast(int(i == j)) for j in range(size)])
          for i, v in enumerate(solutes)]
    fractions = [1-sum(xs), *xs]
    value = excess(xs)
    for amount, cost in zip(fractions, costs):
        value += amount*cost
        # Zero coordinates are fixed faces. Their ideal derivative is not
        # used; the continuous zero energy is sufficient on that face.
        zero = amount.value.lo == amount.value.hi == 0 if intervals else amount.value == 0
        if not zero:
            value += amount*amount.log()
    return value.value, value.gradient


def _quadratic_lower(value, gradient, anchor, lower, upper, curvature):
    """Minimize a separable strong-convexity minorant, outwardly.

    The box may relax the dependent Fe constraint; this only lowers the bound.
    Gradient intervals use their lower/upper endpoint on positive/negative
    displacements respectively. Every minimization candidate is rational.
    """
    result, kappa = value, Fraction.from_float(float(curvature))
    for g, a, lo, hi in zip(gradient, anchor, lower, upper):
        left, right = Fraction.from_float(float(lo))-a, Fraction.from_float(float(hi))-a
        choices = []
        for l, r, slope in ((left, min(right, Fraction(0)), g.hi),
                             (max(left, Fraction(0)), right, g.lo)):
            if l <= r:
                coefficient = Fraction(slope)
                d = max(l, min(r, -coefficient/kappa))
                choices.append((_I(coefficient)*_I(d)+_I(kappa)*_I(d)*_I(d)/2).lo)
        result += _I(min(choices))
    return result.lo


def _validate_domain(lower, upper):
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if (lo.ndim != 1 or hi.shape != lo.shape or lo.size < 2
            or not all(np.all(np.isfinite(v)) for v in (lo, hi))
            or np.any(lo < 0) or np.any(lo > hi) or np.any(hi > 1)
            or lo[0] <= 0 or sum(map(Fraction.from_float, lo)) > 1
            or sum(map(Fraction.from_float, hi)) < 1):
        raise ValueError("Require finite feasible Fe-rich composition bounds.")
    # The present algorithm uses a rectangular independent domain. Do not
    # replace a coupled simplex by an incorrectly smaller rectangle.
    if (1-sum(map(Fraction.from_float, hi[1:])) < Fraction.from_float(lo[0])
            or 1-sum(map(Fraction.from_float, lo[1:])) > Fraction.from_float(hi[0])):
        raise ValueError("The independent box must lie inside the declared Fe bounds.")
    return lo, hi


def certify_associated_insertion(excess, costs, lower, upper, curvature, shifts, *,
                                 tolerance=1e-8, max_nodes=2000, maxiter=500):
    """Bound the whole closed domain using a verified convex reference.

    ``shifts`` is a nonnegative diagonal making the remaining perturbation
    convex. The caller must establish this from the exact declared expression.
    Finite positive trial coordinates never remove zero endpoints from proofs.
    """
    lo, hi = _validate_domain(lower, upper)
    costs = tuple(_I(v) for v in costs)
    shifts = tuple(_I(v) for v in shifts)
    if (len(costs) != len(lo) or len(shifts) != len(lo)-1
            or not all(v.lo.is_finite() and v.hi.is_finite() for v in (*costs, *shifts))
            or any(v.lo < 0 or v.lo != v.hi for v in shifts)
            or not np.isfinite(curvature) or curvature <= 0
            or not np.isfinite(tolerance) or tolerance <= 0
            or type(max_nodes) is not int or max_nodes < 1
            or type(maxiter) is not int or maxiter < 1):
        raise ValueError("Require finite costs, nonnegative exact shifts, and positive proof budgets.")
    float_costs = [float((v.lo+v.hi)/2) for v in costs]
    rho = np.array([float(v.lo) for v in shifts])
    free = hi[1:] > lo[1:]
    branch = [i for i, v in enumerate(rho) if v > 0 and free[i]]
    fixed = lo[1:].copy()
    best = None
    counter = 0
    leaves = []

    def node(lower_y, upper_y, seed):
        nonlocal best, counter
        floor = np.maximum(lower_y, np.minimum(upper_y*.01, 1e-100))
        initial = np.maximum(floor, np.minimum(upper_y, seed))
        initial[~free] = lower_y[~free]

        def numeric(y):
            value, derivative = _scalar_gradient(excess, float_costs, y, intervals=False)
            delta = .5*rho*(y-lower_y)*(y-upper_y)
            grad = np.array(derivative)+rho*(y-(lower_y+upper_y)/2)
            return float(value+delta.sum()), grad

        indices = np.flatnonzero(free)
        scale = upper_y[indices]
        def scaled(z):
            y = fixed.copy()
            y[indices] = z*scale
            value, grad = numeric(y)
            return value, grad[indices]*scale
        y = fixed.copy()
        if indices.size:
            opt = minimize(scaled, initial[indices]/scale, jac=True, method="L-BFGS-B",
                           bounds=list(zip(floor[indices]/scale, np.ones(len(indices)))),
                           options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": maxiter,
                                    "maxls": 50})
            y[indices] = np.clip(opt.x*scale, floor[indices], upper_y[indices])
        # Trace amounts need logarithmic stationarity refinement even when
        # the objective is indistinguishable at ordinary floating precision.
        g = numeric(y)[1]
        active = ((y >= upper_y*(1-1e-9)) & (g < 0)) | (~free)
        active |= (lower_y > 0) & (y <= lower_y*(1+1e-9)) & (g > 0)
        refine = np.flatnonzero(~active)
        if refine.size:
            log_lo, log_hi = np.log(floor[refine]), np.log(upper_y[refine])
            def residual(z):
                trial = y.copy()
                trial[refine] = np.exp(z)
                return numeric(trial)[1][refine]
            root = least_squares(residual, np.clip(np.log(y[refine]), log_lo, log_hi),
                                 bounds=(log_lo, log_hi), ftol=1e-12, xtol=1e-12,
                                 gtol=1e-12, max_nfev=maxiter)
            refined = y.copy()
            refined[refine] = np.exp(root.x)
            if numeric(refined)[0] <= numeric(y)[0]+1e-12:
                y = refined
        y = np.clip(y, lower_y, upper_y)
        anchor = [Fraction.from_float(float(v)) for v in y]
        value, gradient = _scalar_gradient(excess, costs, anchor, intervals=True)
        if best is None or value.hi < best[0]:
            best = (value.hi, anchor, value.lo)
        minorant = value
        minor_gradient = list(gradient)
        for i, shift in enumerate(shifts):
            point, left, right = _I(anchor[i]), _I(float(lower_y[i])), _I(float(upper_y[i]))
            minorant += shift*(point-left)*(point-right)/2
            minor_gradient[i] += shift*(point-(left+right)/2)
        lower_bound = _quadratic_lower(minorant, minor_gradient, anchor,
                                       lower_y, upper_y, curvature)
        counter += 1
        return (lower_bound, counter, lower_y.copy(), upper_y.copy(), y)

    initial = (lo[1:]+hi[1:])/2
    first = node(lo[1:], hi[1:], initial)
    frontier = [first]
    tolerance_d = Decimal.from_float(float(tolerance))
    while frontier and counter+2 <= max_nodes:
        lower_bound, _, left, right, anchor = heapq.heappop(frontier)
        if best[0]-lower_bound <= tolerance_d:
            heapq.heappush(frontier, (lower_bound, counter+1, left, right, anchor))
            break
        if not branch:
            leaves.append((lower_bound, counter+1, left, right, anchor))
            continue
        dimension = max(branch, key=lambda i: rho[i]*(right[i]-left[i])**2)
        middle = float((Fraction.from_float(float(left[dimension]))+
                        Fraction.from_float(float(right[dimension])))/2)
        if middle == left[dimension] or middle == right[dimension]:
            leaves.append((lower_bound, counter+1, left, right, anchor))
            continue
        low, high = left.copy(), right.copy()
        high[dimension] = middle
        heapq.heappush(frontier, node(low, high, anchor))
        # The second closed half overlaps only at the exact split point.
        low, high = left.copy(), right.copy()
        low[dimension] = middle
        heapq.heappush(frontier, node(low, high, anchor))
    retained = frontier+leaves
    lower_value = min(row[0] for row in retained)
    uncertainty = _I(best[0])-_I(lower_value)
    if uncertainty.hi < 0:
        raise ArithmeticError("An inconsistent lower/upper insertion enclosure cannot certify a minimum.")
    fractions = [1-sum(best[1]), *best[1]]
    return {"lower_bound_rt": str(lower_value), "upper_bound_rt": str(best[0]),
            "uncertainty_upper_rt": str(uncertainty.hi),
            "minimum_certified": uncertainty.hi <= tolerance_d,
            "nodes_evaluated": counter, "composition": [float(v) for v in fractions],
            "exact_composition": [str(v) for v in fractions],
            "reference_curvature_lower_bound_rt": curvature,
            "diagonal_convexification_rt": [str(v.lo) for v in shifts],
            "domain_lower": lo.tolist(), "domain_upper": hi.tolist(),
            "frontier": [{"lower_bound_rt": str(row[0]), "lower": row[2].tolist(),
                          "upper": row[3].tolist()} for row in retained],
            "empirical_material_certified": False}


def _load_provider(checkout, potassium, sodium=False):
    kind = "sodium" if sodium else "potassium" if potassium else "associated"
    name = "_m2_global_"+kind
    filename = {"sodium":"sodium_metal.py", "potassium":"potassium_reference.py",
                "associated":"associate_reference.py"}[kind]
    path = Path(checkout).resolve()/"examples/m2_material"/filename
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path:
            raise ValueError("Use one fixed alloy provider checkout per process.")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_associated_expression(metadata, exoeos_checkout):
    """Bind a saved 18/19/20-species expression and its full declared domain."""
    metadata = deepcopy(metadata)
    kind = metadata.get("metal_model")
    if kind not in ("associated", "associated_k", "associated_k_na"):
        raise ValueError("Require a declared associated alloy and its actual callback.")
    sodium = kind == "associated_k_na"
    potassium = kind in ("associated_k", "associated_k_na")
    row = metadata["associated_metal"]
    provider = _load_provider(exoeos_checkout, potassium, sodium)
    root = Path(exoeos_checkout).resolve()
    for path, digest in row["provider_recipe_file_sha256"].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest() != digest:
            raise ValueError("The saved alloy provider recipe has changed: "+path)
        if path.startswith("src/exoeos/"):
            name = path[len("src/"):-3].replace("/", ".")
            module = __import__(name, fromlist=["__file__"])
            if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
                raise ValueError("The imported alloy dependency differs from its saved recipe: "+path)
    temperature = row["standards"]["temperature_K"]
    policy = row["interactions"]["temperature_policy_for_extra_P_H_cross_terms"]
    options = {"temperature_policy": policy}
    if "hydrogen_oxygen" in row["interactions"]:
        options["hydrogen_oxygen_model"] = row["interactions"]["hydrogen_oxygen"]["model"]
    model, interactions = provider.make_associated_model(temperature, **options)
    reference_options = {**options}
    if "hydrogen_oxygen_model" in options:
        reference_options["hydrogen_oxygen_model"] = "omitted"
    reference, _ = provider.make_associated_model(temperature, **reference_options)
    if (model.reference_model_id != row["model_id"] or interactions != row["interactions"]
            or list(provider.COMPONENTS) != row["component_order"]
            or list(provider.FORMULAS) != row["component_formulas"]
            or row["composition_basis"] != "chemical_species_moles"):
        raise ValueError("Saved associated-alloy expression or basis differs from its provider.")
    parent_model = model.host if sodium else model
    host = parent_model.host if potassium else parent_model
    dry = np.asarray(host.host.host.dry_model.interaction_K).tolist()
    epsilon = np.asarray(host.host.epsilon).tolist()
    matrix = np.asarray(host.additional_matrix).tolist()
    if sodium:
        from m2_sodium_global import parent_domain
        saved_lo, saved_hi = (np.asarray(row[key], float) for key in
                              ('lower_species_fractions', 'upper_species_fractions'))
        parent_lo, parent_hi = parent_domain(saved_lo, saved_hi)
    else:
        saved_lo, saved_hi = _validate_domain(row["lower_species_fractions"], row["upper_species_fractions"])
    parent_provider = provider._potassium() if sodium else provider
    parent_reference = reference.host if sodium else reference
    if sodium:
        from m2_sodium_global import parent_domain
        parent_lo, parent_hi = parent_domain(saved_lo, saved_hi)
    else:
        parent_lo, parent_hi = saved_lo, saved_hi
    kappa = parent_provider.associated_curvature_lower_bound(
        parent_reference, temperature, parent_lo, parent_hi)
    if kappa <= 0:
        raise ValueError("The declared H--O-free reference has no positive global curvature bound.")
    standards = [_I(float(value)) for value in row["standards"]["standard_potentials_rt"]]
    if (len(standards) != len(saved_lo)
            or not all(v.lo.is_finite() and v.hi.is_finite() for v in standards)):
        raise ValueError("Require one finite standard for each declared alloy species.")
    offsets = (metadata.get("provider_scenario") or {}).get("standard_offsets_rt", {})
    for i, name in enumerate(provider.COMPONENTS):
        standards[i] += _I(float(offsets.get(name+"_metal", 0.)))
    ho = _I(float(matrix[2][3]))
    if ho.lo < 0:
        raise ValueError("The declared H--O coefficient must be nonnegative.")
    shifts = [_I(0) for _ in range(len(parent_lo)-1)]
    if potassium:
        h = 1-_I(float(parent_hi[-1]))
        shifts[1] = ho/h+ho*_I(float(parent_hi[3]))/h/h
        shifts[2] = ho/h+ho*_I(float(parent_hi[2]))/h/h
        shifts[-1] = ho*(_I(float(parent_hi[2]))+_I(float(parent_hi[3])))/h/h
        shifts = [_I(v.hi) for v in shifts]
    else:
        shifts[1] = shifts[2] = ho
    def excess(values):
        return provider.associated_excess(temperature, dry, epsilon, matrix, values)
    def parent_excess(values):
        return parent_provider.associated_excess(temperature, dry, epsilon, matrix, values)

    return {"metadata": metadata, "row": row, "provider": provider, "model": model, "temperature": temperature, "standards": standards, "saved_lo": saved_lo, "saved_hi": saved_hi, "kappa": kappa, "shifts": shifts, "excess": excess, "offsets": offsets,
            "parent_excess":parent_excess, "sodium_elimination":sodium}


def certify_context_insertion(context, costs, lower, upper, **options):
    """Select only the bound that matches the saved provider expression."""
    if context.get('sodium_elimination', False):
        from m2_sodium_global import certify_sodium_insertion
        return certify_sodium_insertion(context['parent_excess'], costs, lower, upper,
                                        context['kappa'], context['shifts'], **options)
    return certify_associated_insertion(context['excess'], costs, lower, upper,
                                         context['kappa'], context['shifts'], **options)


def _insertion_callback(context, metal_evaluator):
    if not callable(metal_evaluator):
        raise ValueError("The source alloy evaluator must be callable.")
    metadata, row, provider, temperature, standards, saved_lo, saved_hi, kappa, shifts, excess = (context[key] for key in ('metadata', 'row', 'provider', 'temperature', 'standards', 'saved_lo', 'saved_hi', 'kappa', 'shifts', 'excess'))

    offsets = context["offsets"]

    def minimize_actual(t, p, formula, potentials, lower, upper, *, tolerance=1e-8, maxiter=1000):
        if context.get('sodium_elimination', False):
            from m2_sodium_global import parent_domain
            parent_domain(lower, upper)
            lo, hi = np.asarray(lower, float), np.asarray(upper, float)
        else:
            lo, hi = _validate_domain(lower, upper)
        atoms, plane = np.asarray(formula, float), np.asarray(potentials, float)
        if (t != temperature or not np.isfinite(p) or p <= 0
                or lo.shape != saved_lo.shape or np.any(lo < saved_lo) or np.any(hi > saved_hi)
                or atoms.shape != (plane.size, len(standards)) or plane.ndim != 1
                or not np.all(np.isfinite(atoms)) or np.any(atoms < 0)
                or not np.all(np.isfinite(plane))):
            raise ValueError("Insertion state must match the finite saved provider and its domain.")
        expected_atoms = np.array([[formula.get(element, 0.) for formula in provider.FORMULAS]
                                   for element in metadata["input"]["elements"]])
        if not np.array_equal(atoms, expected_atoms):
            raise ValueError("Insertion formula must preserve the saved element/species order.")
        costs = [standard-sum((_I(float(a))*_I(float(l))
                    for a, l in zip(atoms[:, i], plane)), _I(0))
                 for i, standard in enumerate(standards)]
        report = certify_context_insertion(context, costs, lo, hi, tolerance=tolerance,
                                            max_nodes=max(1, 2*maxiter-1), maxiter=maxiter)
        x = np.asarray(report["composition"])
        state = metal_evaluator(t, p, x)
        actual = float(state.gibbs_rt-atoms.T.dot(plane).dot(x))
        potentials_actual = np.asarray(state.mu_rt, float)
        varying = np.flatnonzero(hi[1:] > lo[1:])
        if (potentials_actual.shape != x.shape
                or not np.all(np.isfinite(potentials_actual[np.r_[0, varying+1]]))):
            raise ValueError("The actual alloy callback needs finite potentials on its active face.")
        _, expected_gradient = _scalar_gradient(
            excess, costs, [Fraction(v) for v in report["exact_composition"][1:]], intervals=True)
        gradient_differences = []
        for i in varying:
            observed = _I(float(potentials_actual[i+1]))-_I(float(potentials_actual[0]))
            observed -= sum(((_I(float(atoms[j, i+1]))-_I(float(atoms[j, 0])))*_I(float(plane[j]))
                             for j in range(len(plane))), _I(0))
            difference = observed-expected_gradient[i]
            error = max(abs(difference.lo), abs(difference.hi))
            gradient_differences.append(float(error))
            if error > Decimal('1e-10')*(1+max(abs(expected_gradient[i].lo), abs(expected_gradient[i].hi))):
                raise ValueError("The source potentials do not match the saved insertion gradient.")
        lower_value = _outward_float(Decimal(report["lower_bound_rt"]), True)
        upper_value = _outward_float(Decimal(report["upper_bound_rt"]), False)
        if not np.isfinite(actual) or abs(actual-upper_value) > 1e-10*(1+abs(actual)):
            raise ValueError("The source scalar does not match the saved global insertion expression.")
        gap = _outward_float(Decimal(report["uncertainty_upper_rt"]), False)
        report["binding"] = {
            "model_id": row["model_id"], "temperature_K": t, "pressure_bar": p,
            "element_order": metadata["input"]["elements"],
            "elemental_potentials_rt": plane.tolist(), "formula_matrix": atoms.tolist(),
            "component_order": row["component_order"],
            "maximum_source_reduced_gradient_difference_rt": max(gradient_differences, default=0.),
            "insertion_linear_cost_intervals_rt": [[str(v.lo), str(v.hi)] for v in costs],
            "standard_offsets_rt": offsets,
            "provider_recipe_file_sha256": row["provider_recipe_file_sha256"],
            "verifier_file_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                      for name in ("m2_associated_global.py", "m2_liquid_global.py",
                                          *(("m2_sodium_global.py", "m2_common_plane.py")
                                             if context.get('sodium_elimination', False) else ()))}}
        minimize_actual.last_report = report
        accepted = bool(report["minimum_certified"] and gap <= tolerance
                        and upper_value-lower_value <= tolerance)
        return InsertionMinimum(x, upper_value, lower_value, gap,
                                accepted,
                                "Outward associated-alloy global minimum gap closes." if report["minimum_certified"]
                                else "Associated-alloy global minimum budget is unresolved.", report)
    minimize_actual.last_report = None
    return minimize_actual


def make_associated_insertion_minimizer(metadata, metal_evaluator, exoeos_checkout):
    """Return the source insertion hook for the bound declared expression.

    The callback takes K, bar, formula matrix, elemental potentials, lower
    and upper fractions, and keyword tolerance/maxiter. Its original source
    T/P and scalar/potential validation are retained.
    """
    return _insertion_callback(load_associated_expression(metadata, exoeos_checkout), metal_evaluator)


def alloy_energy_interval(context, amounts):
    """Outward extensive G for exact atom-repaired component amounts."""
    values = [Fraction(v) for v in amounts]
    if (len(values) != len(context["standards"]) or any(v < 0 for v in values)
            or sum(values) <= 0):
        raise ValueError("Require nonnegative declared alloy amounts and a positive phase.")
    total = sum(values)
    fractions = [v/total for v in values]
    if any(x < Fraction(float(lo)) or x > Fraction(float(hi)) for x, lo, hi in
           zip(fractions, context["saved_lo"], context["saved_hi"])):
        raise ValueError("The feasible primal alloy lies outside the declared domain.")
    value, _ = _scalar_gradient(context["excess"], context["standards"], fractions[1:], intervals=True)
    return _I(total)*value
