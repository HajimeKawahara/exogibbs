"""Verified spatial bounds for the declared regular-liquid tangent plane.

The provider supplies the mixing expression. Decimal interval arithmetic
verifies each retained lower bound; SLSQP supplies anchors, never proofs.
Native binary compatibility and empirical applicability remain separate.
"""

from __future__ import annotations

from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction
import hashlib
import heapq
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy


class _I:
    """Outward intervals at 50 decimal digits, used only by the verifier."""

    def __init__(self, value, upper=None):
        if isinstance(value, _I):
            self.lo, self.hi = value.lo, value.hi
        elif isinstance(value, Fraction):
            self.lo = self._op(Decimal(value.numerator), Decimal(value.denominator), "/", ROUND_FLOOR)
            self.hi = self._op(Decimal(value.numerator), Decimal(value.denominator), "/", ROUND_CEILING)
        else:
            self.lo = Decimal.from_float(value) if isinstance(value, float) else Decimal(value)
            self.hi = self.lo if upper is None else upper

    @staticmethod
    def _op(a, b, op, rounding):
        with localcontext() as context:
            context.prec, context.rounding = 50, rounding
            return a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else a / b

    def __add__(self, other):
        other = _I(other)
        return _I(self._op(self.lo, other.lo, "+", ROUND_FLOOR),
                  self._op(self.hi, other.hi, "+", ROUND_CEILING))

    __radd__ = __add__

    def __neg__(self):
        return _I(self.hi.copy_negate(), self.lo.copy_negate())

    def __sub__(self, other):
        return self + -_I(other)

    def __rsub__(self, other):
        return _I(other) + -self

    def __mul__(self, other):
        other = _I(other)
        pairs = [(a, b) for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return _I(min(self._op(a, b, "*", ROUND_FLOOR) for a, b in pairs),
                  max(self._op(a, b, "*", ROUND_CEILING) for a, b in pairs))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = _I(other)
        if other.lo <= 0 <= other.hi:
            raise ArithmeticError("Interval division crosses zero.")
        pairs = [(a, b) for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return _I(min(self._op(a, b, "/", ROUND_FLOOR) for a, b in pairs),
                  max(self._op(a, b, "/", ROUND_CEILING) for a, b in pairs))

    def log(self):
        if self.lo <= 0:
            raise ArithmeticError("A logarithm needs a positive interval.")
        with localcontext() as context:
            context.prec = 50
            # Decimal.ln is correctly rounded to nearest, independently of
            # context.rounding; one representable neighbor encloses it.
            return _I(self.lo.ln().next_minus(), self.hi.ln().next_plus())


def _outward_float(value, lower):
    result = float(value)
    if (Decimal.from_float(result) > value) if lower else (Decimal.from_float(result) < value):
        result = float(np.nextafter(result, -np.inf if lower else np.inf))
    return result


def _tighten(lower, upper):
    """Propagate sum(x)=1 without removing a roundoff-sized feasible slice."""
    lower, upper = lower.copy(), upper.copy()
    with localcontext() as context:
        context.prec = 400
        for _ in range(2):
            for i in range(len(lower)):
                lo = Decimal(1) - sum(Decimal.from_float(float(v)) for j, v in enumerate(upper) if j != i)
                lower[i] = max(lower[i], _outward_float(lo, True))
                hi = Decimal(1) - sum(Decimal.from_float(float(v)) for j, v in enumerate(lower) if j != i)
                upper[i] = min(upper[i], _outward_float(hi, False))
        impossible = (np.any(lower > upper)
                      or sum(Decimal.from_float(float(v)) for v in lower) > 1
                      or sum(Decimal.from_float(float(v)) for v in upper) < 1)
    return lower, upper, impossible


def _positive_definite(matrix):
    """Interval LDL verifies positive definiteness of every enclosed matrix."""
    size = len(matrix)
    factors = [[_I(0) for _ in range(size)] for _ in range(size)]
    pivots = []
    for i in range(size):
        pivot = matrix[i][i] - sum(factors[i][k] * factors[i][k] * pivots[k] for k in range(i))
        if pivot.lo <= 0:
            return False
        pivots.append(pivot)
        factors[i][i] = _I(1)
        for j in range(i + 1, size):
            factors[j][i] = (matrix[j][i] - sum(factors[j][k] * factors[i][k] * pivots[k]
                                               for k in range(i))) / pivot
    return True


def _value_gradient(x, reference, matrix, alpha, water):
    """Intervals enclose the exact source-expression Bregman distance."""
    x = [_I(float(v)) for v in x]
    reference = [_I(v) for v in reference]
    delta = [a - b for a, b in zip(x, reference)]
    energy = _I(0)
    gradient = []
    for i, (amount, parent) in enumerate(zip(x, reference)):
        logarithm = (amount / parent).log()
        energy += alpha * (amount * logarithm - amount + parent)
        gradient.append(alpha * logarithm + sum(matrix[i][j] * delta[j] for j in range(len(x))))
        for j in range(len(x)):
            energy += _I(.5) * delta[i] * matrix[i][j] * delta[j]
    if water is not None:
        for amount, parent in ((x[water], reference[water]), (1-x[water], 1-reference[water])):
            energy += alpha * (amount * (amount / parent).log() - amount + parent)
        gradient[water] += alpha * ((x[water] / reference[water]).log()
                                   - ((1-x[water]) / (1-reference[water])).log())
    return energy, gradient


def _affine_simplex_lower(value, gradient, anchor, lower, upper):
    """Exact greedy box/simplex LP for an outward affine underestimator."""
    with localcontext() as context:
        context.prec = 400
        coordinates = [Decimal.from_float(float(v)) for v in lower]
        remaining = Decimal(1) - sum(coordinates)
        for i in sorted(range(len(gradient)), key=lambda i: gradient[i].lo):
            extra = min(Decimal.from_float(float(upper[i])) - coordinates[i], remaining)
            coordinates[i] += extra
            remaining -= extra
        if remaining != 0:
            raise ArithmeticError("The simplex LP has no feasible point.")
    result = value
    for g, point, origin in zip(gradient, coordinates, anchor):
        result += _I(g.lo) * _I(point) - g * _I(float(origin))
    return result.lo


def certify_liquid_tangent_plane(parameters, component_moles, *, tolerance_rt=1e-8, max_nodes=20000):
    """Bound the full supported liquid simplex, including its closed boundary.

    The exact mathematical coefficients are the supplied binary floating
    values. The parent fractions are exact ratios of the supplied amounts.
    This certifies that declared expression, not an unknown native build.
    """
    amounts = np.asarray(component_moles, dtype=float)
    matrix = np.asarray(parameters["quadratic_matrix_rt"], dtype=float)
    alpha = float(parameters["entropy_coefficient"])
    if (amounts.ndim != 1 or matrix.shape != (amounts.size, amounts.size)
            or not np.all(np.isfinite(amounts)) or np.any(amounts < 0) or amounts.sum() <= 0
            or not np.all(np.isfinite(matrix)) or not np.array_equal(matrix, matrix.T)
            or not np.isfinite(alpha) or alpha <= 0
            or not np.isfinite(tolerance_rt) or tolerance_rt <= 0
            or type(max_nodes) is not int or max_nodes < 1):
        raise ValueError("Invalid finite mixing model, parent, tolerance or node budget.")
    if np.any(amounts[parameters.get("unsupported_positive_indices", [])] > 0):
        raise ValueError("The parent contains an unsupported component.")
    active = np.flatnonzero(amounts > 0)
    excluded = []
    ledger = parameters.get("component_element_matrix")
    if ledger is not None:
        ledger = np.asarray(ledger, dtype=float)
        if ledger.ndim != 2 or ledger.shape[0] != amounts.size or not np.all(np.isfinite(ledger)):
            raise ValueError("Invalid component element ledger.")
        parent_elements = amounts @ ledger
    for index in np.flatnonzero(amounts == 0):
        if index in parameters.get("unsupported_positive_indices", []):
            reason = "unsupported_by_declared_provider_model"
        elif ledger is not None and np.any((ledger[index] > 0) & (parent_elements == 0)):
            reason = "requires_an_element_absent_from_parent"
        else:
            reason = "zero_parent_component_with_no_exclusion_proof"
        excluded.append({"component_index": int(index), "reason": reason})
    complete_domain = all(row["reason"] != "zero_parent_component_with_no_exclusion_proof" for row in excluded)
    original = [Fraction(float(amounts[i])) for i in active]
    reference = [v / sum(original) for v in original]
    x0 = np.asarray([float(v) for v in reference])
    water_index = parameters.get("water_index")
    water = list(active).index(water_index) if water_index in active else None
    matrix = matrix[np.ix_(active, active)]
    n = len(active)
    if n == 1:
        return {"status": "certified_single_composition", "lower_bound_rt": 0.,
                "formal_mixing_bound_certified": True, "active_component_indices": active.tolist(),
                "complete_element_supported_provider_domain": complete_domain,
                "excluded_components": excluded,
                "native_binary_error_bound_certified": False, "nodes_evaluated": 0, "proof_leaves": []}
    imatrix = [[_I(float(v)) for v in row] for row in matrix]
    ialpha = _I(alpha)
    leaves, heap, unavailable = [], [], []
    count = 0
    best = {"value_rt": 0., "coordinates": x0.tolist()}

    def objective(x):
        delta = x-x0
        value = alpha * np.sum(xlogy(x, x/x0)-x+x0) + .5*delta@matrix@delta
        if water is not None:
            u, v = x[water], x0[water]
            value += alpha*(xlogy(u, u/v)+xlogy(1-u, (1-u)/(1-v)))
        return float(value)

    def jacobian(x):
        result = alpha*np.log(x/x0) + matrix@(x-x0)
        if water is not None:
            result[water] += alpha*np.log(x[water]*(1-x0[water])/(x0[water]*(1-x[water])))
        return result

    def assess_box(lower, upper):
        nonlocal count, best
        count += 1
        lower, upper, impossible = _tighten(lower, upper)
        if impossible:
            leaves.append({"lower": lower.tolist(), "upper": upper.tolist(), "status": "empty_simplex"})
            return
        if np.any(upper <= 0) or np.any(lower >= upper):
            raise ArithmeticError("Degenerate box requires a separate boundary evaluation.")
        hessian = matrix + np.diag(alpha / upper)
        if water is not None:
            hessian[water, water] += 4*alpha
        rho = max(0., -float(np.linalg.eigvalsh(hessian)[0]) + 1e-8)

        def verify_curvature(shift):
            checked = [[entry for entry in row] for row in imatrix]
            for i in range(n):
                checked[i][i] = checked[i][i] + ialpha / _I(float(upper[i])) + _I(shift)
            if water is not None:
                checked[water][water] += 4*ialpha
            return _positive_definite(checked)

        while not verify_curvature(rho):
            rho = 1.01*rho + 1e-7
        contains_parent = all(Fraction(float(lo)) <= v <= Fraction(float(hi))
                              for lo, hi, v in zip(lower, upper, reference))
        if rho == 0 and contains_parent:
            leaves.append({"lower": lower.tolist(), "upper": upper.tolist(),
                           "status": "convex_parent_support", "lower_bound_rt": 0., "rho": 0.})
            return
        seed = lower + (upper-lower)*((1-lower.sum())/(upper.sum()-lower.sum()))
        if any(max(float(lo), 1e-15) >= min(float(hi), 1-1e-15) for lo, hi in zip(lower, upper)):
            raise ArithmeticError("No optimizer anchor at the configured float resolution.")

        def relaxed(x):
            return objective(x) + .5*rho*np.sum((x-lower)*(x-upper))

        def relaxed_gradient(x):
            return jacobian(x) + rho*(x-.5*(lower+upper))

        result = minimize(relaxed, seed, jac=relaxed_gradient, method="SLSQP",
                          bounds=[(max(float(lo), 1e-15), min(float(hi), 1-1e-15)) for lo, hi in zip(lower, upper)],
                          constraints={"type": "eq", "fun": lambda x: x.sum()-1,
                                       "jac": lambda x: np.ones_like(x)},
                          options={"ftol": 1e-11, "maxiter": 100})
        # An interior anchor keeps gradients finite. Its position need not be
        # a numerical minimizer, but must lie inside the certified box.
        point = .999999*result.x + .000001*seed
        if not np.all(np.isfinite(point)) or np.any(point <= lower) or np.any(point >= upper):
            point = seed
        if np.any(point <= 0) or np.any(point >= 1):
            raise ArithmeticError("No finite interior anchor for this box.")
        raw = objective(point)
        if raw < best["value_rt"]:
            best = {"value_rt": raw, "coordinates": point.tolist()}
        value, gradient = _value_gradient(point, reference, imatrix, ialpha, water)
        for i in range(n):
            value += _I(.5)*_I(rho)*(_I(float(point[i]))-_I(float(lower[i])))*(_I(float(point[i]))-_I(float(upper[i])))
            gradient[i] += _I(rho)*(_I(float(point[i]))-_I(.5)*(_I(float(lower[i]))+_I(float(upper[i]))))
        bound = _affine_simplex_lower(value, gradient, point, lower, upper)
        lower_bound = _outward_float(bound, True)
        row = {"lower": lower.tolist(), "upper": upper.tolist(), "rho": rho,
               "anchor": point.tolist(), "lower_bound_rt": lower_bound,
               "verified_decimal_lower_bound_rt": str(bound)}
        if bound >= -Decimal.from_float(tolerance_rt):
            leaves.append({"status": "verified_convex_relaxation", **row})
        else:
            heapq.heappush(heap, (lower_bound, count, lower, upper, row))

    def assess(lower, upper):
        try:
            assess_box(lower, upper)
        except ArithmeticError as error:
            unavailable.append({"lower": lower.tolist(), "upper": upper.tolist(),
                                "status": "unresolved_evaluation_domain", "reason": str(error)})

    assess(np.zeros(n), np.ones(n))
    while heap and count + 2 <= max_nodes:
        _, _, lower, upper, _ = heapq.heappop(heap)
        index = int(np.argmax(upper-lower))
        middle = float(.5*(lower[index]+upper[index]))
        if middle <= lower[index] or middle >= upper[index]:
            raise ArithmeticError("The box cannot be bisected at float precision.")
        for side in (0, 1):
            lo, hi = lower.copy(), upper.copy()
            if side:
                lo[index] = middle
            else:
                hi[index] = middle
            assess(lo, hi)
    frontier = [row[-1] for row in heap]
    lower_bound = None if unavailable else min([0.] + [row["lower_bound_rt"] for row in leaves + frontier if "lower_bound_rt" in row])
    return {"assessment_id": "m2_regular_liquid_global_tangent_plane_v1",
            "status": "unresolved_evaluation_domain" if unavailable else
                      "formal_mixing_bound_certified" if not heap else "unresolved_node_budget",
            "formal_mixing_bound_certified": not heap and not unavailable, "native_binary_error_bound_certified": False,
            "lower_bound_rt": lower_bound, "tolerance_rt": tolerance_rt, "max_nodes": max_nodes,
            "nodes_evaluated": count, "best_trial": best,
            "active_component_indices": active.tolist(), "component_moles": amounts.tolist(),
            "complete_element_supported_provider_domain": complete_domain,
            "excluded_components": excluded,
            "proof_leaves": leaves, "unresolved_boxes": frontier + unavailable,
            "verification": "50-digit outward Decimal intervals; interval LDL positive-definiteness; exact box/simplex affine minimization; closed entropy boundary.",
            "scope": "The declared quadratic-plus-ideal mixing expression on the whole nonnegative parent-support simplex. Native build equality and empirical validity are independent requirements.",
            "search_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def assess_liquid_global_tangent_plane(properties, dissolved_h2_moles, *, mixing_model,
                                       tolerance_rt=1e-8, max_nodes=20000):
    """Connect native compatibility, component support and the formal bound.

    For native distance D and parent H2 fraction z0, minimizing over the
    second liquid's H2 fraction gives -log[z0+(1-z0) exp(-D)]. Its sign is
    the sign of D; for D >= L and L <= 0 it is also >= L. Consequently a
    native nonnegative supporting plane excludes every finite two-liquid
    split in this unchanged ideal-H2-augmented mathematical expression.
    """
    if not np.isfinite(dissolved_h2_moles) or dissolved_h2_moles < 0:
        raise ValueError("Dissolved H2 must be finite and nonnegative.")
    if properties["model_id"] == getattr(mixing_model, "PUBLISHED_MODEL_ID", None):
        parameters = mixing_model.liquid_mixing_parameters(
            properties["T_K"], properties["P_Pa"], properties["basis"]["common_R_J_mol_K"])
        if properties.get("mixing_expression") != parameters:
            raise ValueError("The solver and certificate do not declare identical mixing coefficients.")
        comparison = {"status": "same_declared_expression", "parameters": parameters,
                      "native_standard_state_receipt_sha256": properties["provenance"]["native_standard_state_receipt_sha256"],
                      "global_native_error_bound_certified": False}
    else:
        comparison = mixing_model.compare_native_mixing(properties, tolerance_rt=tolerance_rt)
    if comparison["status"] not in {"compatible_at_supplied_state", "same_declared_expression"}:
        return {"status": "native_expression_mismatch", "comparison": comparison,
                "formal_mixing_bound_certified": False, "native_binary_error_bound_certified": False}
    bound = certify_liquid_tangent_plane(comparison["parameters"], properties["component_moles"],
                                         tolerance_rt=tolerance_rt, max_nodes=max_nodes)
    return {**bound, "comparison": comparison, "dissolved_h2_moles": float(dissolved_h2_moles),
            "formal_h2_augmented_lower_bound_rt": bound["lower_bound_rt"],
            "h2_domain": "All nonnegative H2 amounts with the same ideal dilution law; linear H2 standard cancels analytically.",
            "formal_two_liquid_bound_accepted": bool(bound["formal_mixing_bound_certified"]
                                                      and bound["complete_element_supported_provider_domain"]),
            "global_empirical_stability_certified": False}
