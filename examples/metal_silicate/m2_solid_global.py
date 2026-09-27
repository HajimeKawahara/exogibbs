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

from m2_liquid_global import _I, _outward_float


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
        return value.lo, None

    queue, leaves, excluded = [], [], []
    serial = nodes = 0

    def visit(lo, hi):
        nonlocal serial, nodes
        nodes += 1
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
            "element_support_restrictions": removed,
            "required_absent_elements_verified": parameters["required_absent_elements"],
            "leaves": leaves, "excluded_boxes": excluded,
            "unresolved_boxes": [{"lower": lo.tolist(), "upper": hi.tolist(), "lower_bound_rt": str(value)}
                                 for value, _, lo, hi in queue],
            "native_binary_error_bound_certified": False,
            "global_empirical_stability_certified": False,
            "interpretation": "The declared site expression is bounded over every admitted composition and ordering coordinate. Pure standards are fixed numerical inputs. Native build identity, standard-state uncertainty, source-alloy stability and empirical applicability are separate requirements."}
