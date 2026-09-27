"""Outward bounds over declared mineral site domains, including ordering.

The provider owns each polynomial, site entropy and admissible domain.
This chemical layer subtracts the actual host tangent plane and covers the
whole domain by boxes. Native compatibility is deliberately separate.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
import heapq

import numpy as np
from scipy.optimize import linprog

from m2_liquid_global import _I, _outward_float, _positive_definite


def _power(value, exponent):
    if exponent == 0:
        return _I(1)
    if exponent == 1:
        return value
    if value.lo < 0 < value.hi and exponent % 2 == 0:
        ends = [_power(_I(x), exponent) for x in (value.lo, value.hi)]
        return _I(0, max(x.hi for x in ends))
    result = _I(1)
    for _ in range(exponent):
        result *= value
    return result


def _poly(terms, box):
    result = _I(0)
    for coefficient, powers in terms:
        term = _I(coefficient)
        for value, exponent in zip(box, powers):
            term *= _power(value, exponent)
        result += term
    return result


def _xlogx(value):
    lower, upper = max(Decimal(0), value.lo), min(Decimal(1), value.hi)
    if lower > upper:
        raise ValueError("A site interval has no physical occupation.")
    endpoints = [_I(0) if point == 0 else _I(point) * _I(point).log()
                 for point in (lower, upper)]
    with localcontext() as context:
        context.prec = 50
        # Decimal.exp is correctly rounded; neighbors enclose exp(-1).
        critical = Decimal(-1).exp()
        critical_lo, critical_hi = critical.next_minus(), critical.next_plus()
    minimum = min(x.lo for x in endpoints)
    if lower <= critical_hi and upper >= critical_lo:
        minimum = min(minimum, critical_hi.copy_negate())
    return _I(minimum, max(x.hi for x in endpoints))


def _solve_exact(matrix, rhs):
    """Rational elimination avoids inventing trace-element directions."""
    rows = [[Fraction(float(v)) for v in row] + [Fraction(float(b))]
            for row, b in zip(matrix, rhs)]
    size = len(rows)
    for column in range(size):
        pivot = next((i for i in range(column, size) if rows[i][column]), None)
        if pivot is None:
            raise ValueError("Singular host oxide basis.")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [v / divisor for v in rows[column]]
        for i in range(size):
            if i != column:
                factor = rows[i][column]
                rows[i] = [a - factor*b for a, b in zip(rows[i], rows[column])]
    return [row[-1] for row in rows]


def _linear_domain(parameters):
    """Return declared linear inequalities b + a.x >= 0, including sites."""
    size = len(parameters["coordinate_bounds"])
    rows = list(parameters["nonnegative_polynomials"])
    for site in parameters["entropy_sites"]:
        rows.append(site["polynomial"])
        rows.append([[1., [0]*size]] + [[-c, powers] for c, powers in site["polynomial"]])
    result = []
    for terms in rows:
        coefficients, constant = [Fraction(0)]*size, Fraction(0)
        if any(sum(powers) > 1 for _, powers in terms):
            continue
        for c, powers in terms:
            if sum(powers) == 0:
                constant += Fraction(float(c))
            else:
                coefficients[powers.index(1)] += Fraction(float(c))
        # The numerical LP only receives exactly representable constraints.
        # Other rows remain in the original interval domain check.
        if all(Fraction(float(x)) == x for x in [constant]+coefficients):
            result.append((float(constant), [float(x) for x in coefficients]))
    return result


def _tighten_domain(lower, upper, inequalities):
    """Enclose bound propagation without discarding any feasible boundary."""
    lower, upper = lower.copy(), upper.copy()
    for _ in range(2):
        for constant, coefficients in inequalities:
            for index, coefficient in enumerate(coefficients):
                if coefficient == 0:
                    continue
                remainder = _I(constant)
                for j, c in enumerate(coefficients):
                    if j != index and c:
                        remainder += _I(c)*_I(float(upper[j] if c > 0 else lower[j]))
                edge = -remainder/_I(coefficient)
                if coefficient > 0:
                    lower[index] = max(lower[index], _outward_float(edge.lo, True))
                else:
                    upper[index] = min(upper[index], _outward_float(edge.hi, False))
                if lower[index] > upper[index]:
                    return lower, upper, True
    return lower, upper, False


def _linear_program_lower(terms, box, inequalities):
    """Certify an affine LP lower bound from arbitrary nonnegative duals.

    The numerical LP supplies useful multipliers only. Re-evaluating their
    Lagrangian with intervals and minimizing its residual on the box makes
    the bound independent of the numerical LP feasibility tolerances.
    """
    size = len(box)
    gradient, constant = [_I(0) for _ in box], _I(0)
    for coefficient, powers in terms:
        if sum(powers) == 0:
            constant += coefficient
        else:
            gradient[powers.index(1)] += coefficient
    matrix = np.asarray([row[1] for row in inequalities])
    rhs = np.asarray([row[0] for row in inequalities])
    result = linprog([float(g.lo) for g in gradient], A_ub=-matrix, b_ub=rhs,
                     bounds=[(float(v.lo), float(v.hi)) for v in box], method="highs")
    multipliers = np.maximum(0., -result.ineqlin.marginals) if result.success else np.zeros(len(rhs))
    for multiplier, (b, a) in zip(multipliers, inequalities):
        weight = _I(float(multiplier))
        constant -= weight*_I(b)
        for j, value in enumerate(a):
            gradient[j] -= weight*_I(value)
    return (constant + sum((g*x for g, x in zip(gradient, box)), _I(0))).lo


def _derivative(terms, index):
    result = []
    for coefficient, powers in terms:
        if powers[index]:
            reduced = list(powers)
            reduced[index] -= 1
            result.append((_I(coefficient)*_I(powers[index]), reduced))
    return result


def _curvature_relaxation(terms, sites, box, inequalities, derivatives, hessians):
    """Support a convex polynomial/entropy minorant and verify its affine LP.

    Positive site entropy has curvature at least c/upper. Negative site
    entropy is concave and lies above its endpoint chord. These replacements
    are valid on the full closed site interval, including empty sites.
    """
    size = len(box)
    center = [_I((v.lo+v.hi)/2) for v in box]
    value = _poly(terms, center)
    gradient = [_poly(row, center) for row in derivatives]
    hessian = [[_poly(row, box) for row in rows] for rows in hessians]
    for site in sites:
        occupation = _poly(site["polynomial"], box)
        lo, hi = max(Decimal(0), occupation.lo), min(Decimal(1), occupation.hi)
        at_center = _poly(site["polynomial"], center)
        slope = [_poly(_derivative(site["polynomial"], j), center) for j in range(size)]
        if any(sum(powers) > 1 for _, powers in site["polynomial"]):
            raise ArithmeticError("Curvature relaxation requires affine site occupations.")
        coefficient = _I(site["coefficient_rt"])
        if hi == 0:
            continue
        if coefficient.lo >= 0:
            anchor = max(lo, min(hi, (at_center.lo+at_center.hi)/2))
            if anchor == 0:
                anchor = hi/2
            logarithm = _I(anchor).log()
            curvature = _I(_outward_float((coefficient/_I(hi)).lo, True))
            delta = at_center-_I(anchor)
            value += coefficient*_I(anchor)*logarithm + coefficient*(logarithm+1)*delta + _I(.5)*curvature*delta*delta
            for i in range(size):
                gradient[i] += (coefficient*(logarithm+1)+curvature*delta)*slope[i]
                for j in range(size):
                    hessian[i][j] += curvature*slope[i]*slope[j]
        else:
            first = coefficient*_xlogx(_I(lo))
            last = coefficient*_xlogx(_I(hi))
            if hi == lo:
                value += first
                continue
            # Interpolate lower endpoint values. Convex combination weights
            # are nonnegative on the declared site interval.
            chord = (_I(last.lo)-_I(first.lo))/(_I(hi)-_I(lo))
            value += _I(first.lo)+chord*(at_center-_I(lo))
            for i in range(size):
                gradient[i] += chord*slope[i]
    midpoint = np.array([[float((entry.lo+entry.hi)/2) for entry in row] for row in hessian])
    rho = max(0., -float(np.linalg.eigvalsh(midpoint)[0])+1e-8)
    ceiling = max(Decimal(0), max(sum(max(abs(hessian[i][j].lo), abs(hessian[i][j].hi))
                                    for j in range(size) if j != i)-hessian[i][i].lo for i in range(size)))
    ceiling = _outward_float(ceiling, False)+1e-7
    for _ in range(8):
        shifted = [[entry+(_I(rho) if i == j else _I(0)) for j, entry in enumerate(row)]
                   for i, row in enumerate(hessian)]
        if _positive_definite(shifted):
            break
        rho = max(1e-7, min(ceiling, 2*rho+1e-7))
    else:
        rho = ceiling
        shifted = [[entry+(_I(rho) if i == j else _I(0)) for j, entry in enumerate(row)]
                   for i, row in enumerate(hessian)]
        if not _positive_definite(shifted):
            raise ArithmeticError("The convex minorant curvature was not verified.")
    for i in range(size):
        value += _I(.5)*_I(rho)*(center[i]-_I(box[i].lo))*(center[i]-_I(box[i].hi))
        gradient[i] += _I(rho)*(center[i]-_I(.5)*(_I(box[i].lo)+_I(box[i].hi)))
    constant = value-sum((g*x for g, x in zip(gradient, center)), _I(0))
    affine = [(constant, [0]*size)]
    for j, g in enumerate(gradient):
        powers = [0]*size
        powers[j] = 1
        affine.append((g, powers))
    return _linear_program_lower(affine, box, inequalities)


def _pure_minimum_interval(model, subdivisions=256):
    """Enclose a complete one-dimensional pure-order global minimum.

    Every interval is covered, and point energies provide only upper bounds.
    This defines the declared equilibrium reference independently of any
    local native Newton ordering state.
    """
    if model["coordinate_bounds"] != [[0, 1]]:
        raise ValueError("A complete [0,1] pure-order coordinate is required.")
    terms = model["polynomial_rt"]
    derivatives = [_derivative(terms, 0)]
    hessians = [[_derivative(derivatives[0], 0)]]
    inequalities = _linear_domain({**model, "nonnegative_polynomials": []})

    def energy(box):
        value = _poly(terms, box)
        for site in model["entropy_sites"]:
            occupation = _poly(site["polynomial"], box)
            if occupation.lo < 0 or occupation.hi > 1:
                raise ValueError("A pure-order expression leaves its physical site domain.")
            value += _I(site["coefficient_rt"])*_xlogx(occupation)
        return value

    constant = all(not any(powers) for _, powers in terms) and not model["entropy_sites"]
    count = 1 if constant else subdivisions
    intervals, witnesses = [], []
    for i in range(count):
        lo, hi = Decimal(i)/Decimal(count), Decimal(i+1)/Decimal(count)
        box = [_I(lo, hi)]
        value = energy(box)
        if not constant and inequalities:
            try:
                bound = _curvature_relaxation(terms, model["entropy_sites"], box, inequalities, derivatives, hessians)
                value = _I(max(value.lo, bound), value.hi)
            except ArithmeticError:
                pass
        intervals.append({"lower": str(lo), "upper": str(hi), "energy_lower_rt": str(value.lo)})
        point = (lo+hi)/2
        witnesses.append({"coordinate": str(point), "energy_upper_rt": str(energy([_I(point)]).hi)})
    for point in (Decimal(0), Decimal(1)):
        witnesses.append({"coordinate": str(point), "energy_upper_rt": str(energy([_I(point)]).hi)})
    lower = min(Decimal(row["energy_lower_rt"]) for row in intervals)
    witness = min(witnesses, key=lambda row: Decimal(row["energy_upper_rt"]))
    return _I(lower, Decimal(witness["energy_upper_rt"])), {"subintervals": intervals, "upper_witness": witness}


def certify_solid_insertion(parameters: dict, standard_states: dict,
                            host_properties: dict, dissolved_h2_moles: float,
                            *, max_nodes: int = 100000, tolerance_rt: float = 1e-8) -> dict:
    """Certify a supplied model's insertion sign on every physical site state.

    Every retained interval is outward rounded. Ordering coordinates remain
    free, so their unknown native local minimizer cannot hide a lower state.
    No native binary or empirical error bound is inferred from this proof.
    """
    if (type(max_nodes) is not int or max_nodes < 1 or not np.isfinite(tolerance_rt)
            or tolerance_rt <= 0 or not np.isfinite(dissolved_h2_moles) or dissolved_h2_moles < 0):
        raise ValueError("Positive node/tolerance budgets and nonnegative H2 are required.")
    if (parameters["phase"] != standard_states["phase"]
            or parameters["T_K"] != host_properties["T_K"]
            or parameters["P_Pa"] != host_properties["P_Pa"]
            or standard_states["T_K"] != host_properties["T_K"]
            or standard_states["P_Pa"] != host_properties["P_Pa"]
            or parameters["common_R_J_mol_K"] != host_properties["basis"]["common_R_J_mol_K"]
            or parameters["oxide_order"] != host_properties["oxide_order"]):
        raise ValueError("Phase, T/P/R and oxide ledgers must agree.")
    if (parameters.get("source_kind") == "pinned_native_binary_instruction_transcription"
            and standard_states.get("native_binary_sha256") != parameters["source_sha256"]):
        raise ValueError("The binary-specific declaration requires its pinned native standard provider.")
    if (host_properties.get("model_id") not in {
            "alphamelts_2_3_2_rhyolite_melts_1_0_2_supplied_liquid_v1",
            "melts_v102_published_mixing_native_standard_states_v1"}
            or host_properties.get("status") != "ok_supplied_liquid_properties"
            or host_properties["phase_policy"]["oxygen_buffer"] != "None"
            or host_properties["phase_policy"]["equilibrated"]):
        raise ValueError("The declared unbuffered supplied-liquid host is required.")
    amounts = np.asarray(host_properties["component_moles"], dtype=float)
    mu = np.asarray(host_properties["mu_RT"], dtype=float)
    matrix = np.asarray(host_properties["basis"]["component_oxide_matrix"], dtype=float).T
    formula = np.asarray(host_properties["basis"]["component_element_matrix"], dtype=float)
    if (amounts.ndim != 1 or np.any(amounts < 0) or not np.all(np.isfinite(amounts))
            or amounts.sum() <= 0 or mu.shape != amounts.shape
            or matrix.shape != (amounts.size, amounts.size)):
        raise ValueError("Invalid host amount or oxide ledger.")
    elements = {name: sum(Fraction(float(n))*Fraction(float(c)) for n, c in zip(amounts, formula[:, j]))
                for j, name in enumerate(host_properties["basis"]["element_order"])}
    if any(elements.get(name, 0) != 0 for name in parameters["required_absent_elements"]):
        raise ValueError("The declared face requires absent Mn/Ni/Co elements.")
    bounds = np.asarray(parameters["coordinate_bounds"], dtype=float)
    lower, upper = bounds[:, 0].copy(), bounds[:, 1].copy()
    removed = []
    if parameters["phase"] in {"alloy-solid", "alloy-liquid"}:
        if elements.get("Ni", 0) == 0:
            lower[0] = upper[0] = 0.
            removed.append({"coordinate": 0, "value": 0., "absent_element": "Ni"})
        if elements.get("Fe", 0) == 0:
            lower[0] = upper[0] = 1.
            removed.append({"coordinate": 0, "value": 1., "absent_element": "Fe"})
        if elements.get("Ni", 0) == 0 and elements.get("Fe", 0) == 0:
            raise ValueError("The alloy has no element-supported composition.")
    total = sum((_I(float(n)) for n in amounts), _I(0))
    log_fraction = (total / (total + _I(float(dissolved_h2_moles)))).log()
    rt = _I(parameters["common_R_J_mol_K"]) * _I(parameters["T_K"])
    standards = np.asarray(standard_states["mu0_J_mol"], dtype=float)
    native_basis = np.asarray(standard_states["native_endmember_oxide_mass_g_per_mol"], dtype=float)
    oxide_masses = np.asarray(host_properties["oxide_molar_masses_g_mol"], dtype=float)
    selected = parameters["native_endmember_indices"]
    declared_oxide = np.asarray(parameters["endmember_oxide_moles"], dtype=float).T
    if (native_basis.shape != (len(oxide_masses), len(standards))
            or not np.all(np.isfinite(native_basis))
            or not np.allclose(native_basis[:, selected] / oxide_masses[:, None], declared_oxide,
                               rtol=0., atol=1e-8)):
        raise ValueError("Native standards and the declared stoichiometric basis disagree.")
    terms = [(float(c), powers) for c, powers in parameters["polynomial_rt"]]
    root_box = [_I(float(a), Decimal.from_float(float(b))) for a, b in zip(lower, upper)]
    costs, reactions = [], []
    reference_intervals = {}
    reference_proofs = {}
    for model in parameters.get("pure_reference_models", []):
        reference_intervals[model["endmember_index"]], reference_proofs[str(model["endmember_index"])] = _pure_minimum_interval(model)
    for row in parameters.get("pure_reference_bounds", []):
        entropy = sum((_I(float(multiplicity))*_I(int(categories)).log()
                       for multiplicity, categories in row["entropy_site_groups"]), _I(0))
        alpha = _I(parameters["native_entropy_R_J_mol_K"]) / _I(parameters["common_R_J_mol_K"])
        bottom = _I(row["enthalpy_lower_J_mol"])/rt - alpha*entropy
        top = _I(row["enthalpy_upper_J_mol"])/rt
        reference_intervals[row["endmember_index"]] = _I(bottom.lo, top.hi)
    for index, oxide, fractions in zip(parameters["native_endmember_indices"],
                                     parameters["endmember_oxide_moles"], parameters["endmember_polynomials"]):
        if _poly(fractions, root_box).lo == 0 and _poly(fractions, root_box).hi == 0:
            costs.append(None)
            reactions.append(None)
            continue
        if index >= len(standards) or not np.isfinite(standards[index]):
            raise ValueError("A finite native standard is required.")
        coefficients = _solve_exact(matrix, oxide)
        if any(c and (not np.isfinite(mu[i]) or amounts[i] == 0) for i, c in enumerate(coefficients)):
            raise ValueError("The full domain needs an unavailable host endpoint potential.")
        cost = _I(float(standards[index])) / rt - reference_intervals.get(index, _I(0))
        for c, potential in zip(coefficients, mu):
            if c:
                cost -= _I(c) * (_I(float(potential)) + log_fraction)
        costs.append([str(cost.lo), str(cost.hi)])
        reactions.append([str(c) for c in coefficients])
        terms.extend((cost * _I(float(coefficient)), powers) for coefficient, powers in fractions)
    combined = {}
    for coefficient, powers in terms:
        key = tuple(powers)
        combined[key] = combined.get(key, _I(0)) + _I(coefficient)
    terms = [(coefficient, powers) for powers, coefficient in combined.items()
             if coefficient.lo != 0 or coefficient.hi != 0]
    inequalities = _linear_domain(parameters)
    affine = [(c, powers) for c, powers in terms if sum(powers) <= 1]
    nonlinear = [(c, powers) for c, powers in terms if sum(powers) > 1]
    derivatives = [_derivative(terms, j) for j in range(len(lower))]
    hessians = [[_derivative(row, j) for j in range(len(lower))] for row in derivatives]

    def bound(lo, hi):
        box = [_I(float(a), Decimal.from_float(float(b))) for a, b in zip(lo, hi)]
        for index, constraint in enumerate(parameters["nonnegative_polynomials"]):
            if _poly(constraint, box).hi < 0:
                return None, f"constraint_{index}"
        value = _poly(terms, box)
        for index, site in enumerate(parameters["entropy_sites"]):
            occupation = _poly(site["polynomial"], box)
            if occupation.hi < 0 or occupation.lo > 1:
                return None, f"site_{index}"
            value += _I(site["coefficient_rt"]) * _xlogx(occupation)
        barrier = parameters["barrier"]
        if barrier is not None:
            occupation = _poly(barrier["polynomial"], box)
            if occupation.hi < 0:
                return None, "barrier_negative"
            if occupation.hi == 0:
                return None, "positive_infinite_barrier"
            # A positive singularity can only increase G. Its finite lower
            # bound includes boxes touching the divergent endpoint.
            value += _I(barrier["numerator_rt"]) / _I(occupation.hi)
        if value.lo < 0 and inequalities:
            improved = _linear_program_lower(affine, box, inequalities)
            # Replace only the affine polynomial contribution; all nonlinear
            # and entropy intervals remain independently enclosing.
            residual = _poly(nonlinear, box)
            for site in parameters["entropy_sites"]:
                residual += _I(site["coefficient_rt"])*_xlogx(_poly(site["polynomial"], box))
            if barrier is not None:
                residual += _I(barrier["numerator_rt"])/_I(_poly(barrier["polynomial"], box).hi)
            value = _I(max(value.lo, (residual + _I(improved)).lo), value.hi)
        if value.lo < 0 and inequalities and barrier is None:
            try:
                relaxed = _curvature_relaxation(terms, parameters["entropy_sites"], box,
                                                 inequalities, derivatives, hessians)
                value = _I(max(value.lo, relaxed), value.hi)
            except ArithmeticError:
                # The basic interval partition remains valid when a stronger
                # relaxation cannot be verified on a degenerate face.
                pass
        return value.lo, None

    queue, leaves, excluded = [], [], []
    serial = nodes = 0

    def visit(lo, hi):
        nonlocal serial, nodes
        nodes += 1
        lo, hi, impossible = _tighten_domain(lo, hi, inequalities)
        if impossible:
            excluded.append({"lower": lo.tolist(), "upper": hi.tolist(), "reason": "empty_linear_domain"})
            return
        value, reason = bound(lo, hi)
        row = {"lower": lo.tolist(), "upper": hi.tolist()}
        if reason:
            excluded.append({**row, "reason": reason})
        elif value >= -Decimal.from_float(tolerance_rt):
            leaves.append({**row, "lower_bound_rt": str(value)})
        else:
            serial += 1
            heapq.heappush(queue, (value, serial, lo, hi))

    visit(lower, upper)
    while queue and nodes + 2 <= max_nodes:
        value, _, lo, hi = heapq.heappop(queue)
        index = int(np.argmax(hi-lo))
        middle = float((lo[index]+hi[index])/2.)
        if middle <= lo[index] or middle >= hi[index]:
            heapq.heappush(queue, (value, serial+1, lo, hi))
            break
        left, right = hi.copy(), lo.copy()
        left[index] = right[index] = middle
        visit(lo, left)
        visit(right, hi)
    minimum = min([Decimal(row["lower_bound_rt"]) for row in leaves] + [row[0] for row in queue], default=Decimal("Infinity"))
    return {"assessment_id": "declared_solid_site_global_insertion_v1", "phase": parameters["phase"],
            "formal_global_insertion_bound_accepted": not queue and bool(leaves),
            "lower_bound_rt_per_formula_unit": _outward_float(minimum, True) if minimum.is_finite() else None,
            "node_count": nodes, "leaf_count": len(leaves), "unresolved_box_count": len(queue),
            "max_nodes": max_nodes, "tolerance_rt": tolerance_rt,
            "parameters": parameters, "native_standard_states": standard_states,
            "standard_insertion_cost_intervals_rt": costs, "host_reactions_exact": reactions,
            "pure_reference_intervals_rt": {str(index): [str(v.lo), str(v.hi)]
                                             for index, v in reference_intervals.items()},
            "pure_reference_proofs": reference_proofs,
            "element_support_restrictions": removed,
            "required_absent_elements_verified": parameters["required_absent_elements"],
            "leaves": leaves, "excluded_boxes": excluded,
            "unresolved_boxes": [{"lower": lo.tolist(), "upper": hi.tolist(), "lower_bound_rt": str(value)}
                                 for value, _, lo, hi in queue],
            "native_binary_error_bound_certified": False,
            "global_empirical_stability_certified": False,
            "interpretation": "The declared site expression is bounded over every admitted composition and ordering coordinate. Pure standards are fixed numerical inputs. Native build identity, standard-state uncertainty, source-alloy stability and empirical applicability are separate requirements."}
