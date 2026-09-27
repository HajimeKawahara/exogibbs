"""Global bounds for the reconstructed dry-host/water/H2 Gibbs expression.

Positive external insertion constants permit exact analytical elimination of
both volatile amounts. The remaining dry simplex includes every composition;
its rational water-capacity term is bounded with outward Hessian intervals.
The oxygen capacity is relaxed for the lower bound, never for a witness.
"""
from decimal import Decimal
from fractions import Fraction
import heapq

import numpy as np
from scipy.optimize import minimize

from m2_common_plane import _I, _dot, _exp, ideal_energy, interval_json, liquid_standard_intervals
from m2_liquid_global import _affine_simplex_lower, _outward_float, _positive_definite, _tighten


def parameters_from_saved_water(properties: dict, plane: dict, element_budget: dict,
                                h2_standard_rt, *, water_standard_offset_rt=0.) -> tuple:
    """Bind the full element-supported provider domain to a common plane.

    The external scenario shift is kept separate from the unmodified gas
    standard receipt. Missing positive-budget components fail closed; a zero
    parent amount by itself is never an exclusion from the global domain.
    """
    reconstruction = properties["water_reconstruction"]
    if reconstruction.get("standard_offset_rt", 0.) != water_standard_offset_rt:
        raise ValueError("The audited external water shift differs from the source scenario.")
    expression, dry = reconstruction["expression"], reconstruction["dry_properties"]
    names = properties["component_order"]
    size = len(names)
    index = expression["water_index"]
    if (properties["model_id"] != "dry_melts_thompson2025_water_equivalent_v1"
            or expression["schema"] != "dry_melts_water_equivalent_expression_v1"
            or dry["model_id"] != "melts_v102_published_mixing_native_standard_states_v1"
            or expression["component_order"] != names or dry["component_order"] != names
            or type(index) is not int or not 0 <= index < size or names[index] != "h2o"
            or expression["water_mixing_factor"] != 2.
            or expression["water_standard_pressure_Pa"] != 1e5
            or reconstruction["gas_standard_pressure_Pa"] != 1e5
            or reconstruction["native_water_amount_used_mol"] != 0.
            or expression["capacity_formula"] != "log_Cw = log_prefactor + (weights @ dry)/(T_K*(counts @ dry))"
            or expression["gibbs_formula"] != "G_RT = G_dry_RT + h*(gas_standard_RT - 2*log_Cw) + 2*G_mass_fraction_RT"
            or expression["domain"] != "nonnegative amounts; positive dry host; h <= oxygen @ dry"):
        raise ValueError("Unsupported saved reconstructed-water expression.")
    vectors = ("component_masses_kg_mol", "predictor_oxide_counts",
               "predictor_temperature_weights_K", "component_oxygen_counts")
    if any(len(expression[key]) != size for key in vectors):
        raise ValueError("Water-expression vectors must cover every native component.")
    n = np.asarray(properties["component_moles"], dtype=float)
    dn = np.asarray(dry["component_moles"], dtype=float)
    if (n.shape != (size,) or dn.shape != (size,) or np.any(~np.isfinite(n)) or np.any(n < 0)
            or np.any(~np.isfinite(dn)) or np.any(dn < 0) or dn[index] != 0
            or any(n[i] != dn[i] for i in range(size) if i != index)
            or properties["T_K"] != dry["T_K"] or properties["P_Pa"] != dry["P_Pa"]
            or properties["basis"]["common_R_J_mol_K"] != dry["basis"]["common_R_J_mol_K"]):
        raise ValueError("The saved dry-host reference does not match the wet state.")
    elements = properties["basis"]["element_order"]
    columns = np.asarray(properties["basis"]["component_element_matrix"], dtype=float)
    if (columns.shape != (size, len(elements)) or len(set(elements)) != len(elements) or not {"H", "O"} <= set(elements)
            or np.any(~np.isfinite(columns))
            or columns.tolist() != dry["basis"]["component_element_matrix"]
            or dry["basis"]["element_order"] != elements
            or any(e not in element_budget or e not in plane for e in elements)
            or any(not np.isfinite(v) or v < 0 for v in element_budget.values())
            or any(not np.isfinite(v) for v in plane.values())):
        raise ValueError("A complete finite nonnegative elemental ledger and plane are required.")
    if columns[index].tolist() != [2. if e == "H" else 1. if e == "O" else 0. for e in elements]:
        raise ValueError("The water component must have exactly the H2O atom column.")
    mixing = dry["mixing_expression"]
    matrix = np.asarray(mixing["quadratic_matrix_rt"], dtype=float)
    unsupported = mixing["unsupported_positive_indices"]
    if (matrix.shape != (size, size) or np.any(~np.isfinite(matrix)) or not np.array_equal(matrix, matrix.T)
            or mixing["water_index"] != index or mixing["component_order"] != names
            or mixing["T_K"] != properties["T_K"] or mixing["P_Pa"] != properties["P_Pa"]
            or mixing["common_R_J_mol_K"] != dry["basis"]["common_R_J_mol_K"]
            or mixing["element_order"] != elements or mixing["component_element_matrix"] != columns.tolist()
            or any(type(i) is not int or i < 0 or i >= size for i in unsupported)
            or any(len(dry[key]) != size for key in ("mu0_J_mol", "mu0_RT"))):
        raise ValueError("The dry mixing expression has an inconsistent full component domain.")
    excluded, active = [], []
    for i in range(size):
        if i == index:
            continue
        absent = [e for e, coefficient in zip(elements, columns[i]) if coefficient and element_budget[e] == 0]
        if i in unsupported or absent:
            if dn[i] != 0:
                raise ValueError("The reference occupies an excluded dry component.")
            excluded.append({"component_index": i, "reason": "declared_provider_unsupported" if i in unsupported
                             else "requires_zero_inventory_element", "zero_inventory_elements": absent})
        elif dn[i] <= 0:
            raise ValueError("A supported zero-reference component needs its standard state; it cannot be excluded.")
        else:
            if np.any(columns[i] < 0) or not np.any(columns[i] > 0):
                raise ValueError("Every active dry component needs a nonnegative nonzero atom column.")
            active.append(i)
    standards = liquid_standard_intervals(dry)
    parameters = {
        "dissolved_h2_cost_rt": _I(h2_standard_rt)-2*_I(plane["H"]),
        "water_gas_cost_rt": (_I(reconstruction["gas_H2O_standard_RT"])+_I(water_standard_offset_rt)
                              -2*_I(plane["H"])-_I(plane["O"])),
        "log_capacity_prefactor": expression["log_capacity_prefactor"],
        "predictor_oxide_counts": [expression["predictor_oxide_counts"][i] for i in active],
        "predictor_temperature_weights_K": [expression["predictor_temperature_weights_K"][i] for i in active],
        "temperature_K": properties["T_K"], "water_mass_kg_mol": expression["water_mass_kg_mol"],
        "dry_masses_kg_mol": [expression["component_masses_kg_mol"][i] for i in active],
        "dry_oxygen_counts": [expression["component_oxygen_counts"][i] for i in active],
        "dry_standard_costs_rt": [standards[i]-_dot(columns[i].tolist(), [plane[e] for e in elements]) for i in active],
        "entropy_coefficient": mixing["entropy_coefficient"],
        "quadratic_matrix_rt": matrix[np.ix_(active, active)].tolist(),
    }
    eliminate_volatiles(parameters)  # Validate every projected coefficient.
    return parameters, [float(dn[i]) for i in active], {
        "active_dry_component_indices": active, "excluded_dry_components": excluded,
        "complete_element_supported_provider_domain": True, "element_budget_mol": element_budget,
        "supporting_element_potentials_rt": plane, "bound_unit": "one mole of dry native components",
        "minimum_atoms_per_bound_unit_exact": str(min(sum(map(Fraction, columns[i])) for i in active)),
        "gas_H2O_standard_RT": reconstruction["gas_H2O_standard_RT"],
        "external_h2o_melts_standard_offset_rt": float(water_standard_offset_rt),
        "unmodified_provider_expression": expression,
    }


def water_insertion_value(parameters: dict, dry_amounts: list, water_amount, h2_amount) -> _I:
    """Enclose the uneliminated extensive scalar on its original domain."""
    model = eliminate_volatiles(parameters)
    n = list(map(Fraction, dry_amounts))
    h, hydrogen = Fraction(water_amount), Fraction(h2_amount)
    if (len(n) != len(parameters["dry_standard_costs_rt"]) or any(v < 0 for v in n)
            or sum(n) <= 0 or h < 0 or hydrogen < 0
            or _I(h).hi > _dot(parameters["dry_oxygen_counts"], n).lo):
        raise ValueError("The extensive water state is outside the original amount/capacity domain.")
    total = sum(n)
    log_c = _I(parameters["log_capacity_prefactor"])+_dot(model["weights"], n)/(_I(parameters["temperature_K"])*_dot(model["counts"], n))
    energy = _dot(parameters["dry_standard_costs_rt"], n)+_I(parameters["entropy_coefficient"])*ideal_energy(n)
    energy += _dot(n, [_dot(row, n) for row in parameters["quadratic_matrix_rt"]])/(_I(2)*_I(total))
    energy += _I(h)*(_I(parameters["water_gas_cost_rt"])-2*log_c)
    # The mass-ratio conversion is part of the declared scalar, not a
    # conversion of its physical amount ledger or elemental formula.
    s = sum((Fraction(mass)*amount/Fraction(parameters["water_mass_kg_mol"])
             for mass, amount in zip(parameters["dry_masses_kg_mol"], n)), Fraction(0))
    energy += 2*ideal_energy([s, h])+ideal_energy([total+h, hydrogen])+_I(hydrogen)*_I(parameters["dissolved_h2_cost_rt"])
    return energy


def _hull_intersection(value, lower, upper):
    lo, hi = max(value.lo, lower), min(value.hi, upper)
    if lo > hi:
        raise ArithmeticError("Inconsistent simplex interval bounds.")
    return _I(lo, hi)


def eliminate_volatiles(parameters: dict) -> dict:
    """Prepare the unchanged scalar's exact nonnegative-amount minimum."""
    size = len(parameters["dry_standard_costs_rt"])
    vectors = ("predictor_oxide_counts", "predictor_temperature_weights_K", "dry_masses_kg_mol", "dry_oxygen_counts")
    matrix = np.asarray(parameters["quadratic_matrix_rt"], dtype=float)
    if (size < 2 or any(len(parameters[name]) != size for name in vectors)
            or matrix.shape != (size, size) or not np.all(np.isfinite(matrix))
            or not np.array_equal(matrix, matrix.T)):
        raise ValueError("Require matching finite vectors and a symmetric dry mixing matrix.")
    for value in [*parameters["dry_standard_costs_rt"],
                  *(v for name in vectors for v in parameters[name]),
                  *(parameters[name] for name in ("dissolved_h2_cost_rt", "water_gas_cost_rt", "log_capacity_prefactor",
                                                 "temperature_K", "water_mass_kg_mol", "entropy_coefficient"))]:
        interval = _I(value)
        if not interval.lo.is_finite() or not interval.hi.is_finite() or interval.lo > interval.hi:
            raise ValueError("Every declared water-model coefficient must be finite and ordered.")
    if any(_I(parameters[name]).lo <= 0 for name in ("temperature_K", "water_mass_kg_mol", "entropy_coefficient")):
        raise ValueError("Positive temperature, water molar mass and entropy coefficient are required.")
    if any(_I(v).lo < 0 for v in parameters["dry_oxygen_counts"]):
        raise ValueError("Dry-component oxygen capacities must be nonnegative.")
    qh2 = _I(parameters["dissolved_h2_cost_rt"])
    if qh2.lo <= 0:
        raise ValueError("Analytical H2 elimination requires a positive common-plane cost.")
    a = (1-_exp(-qh2)).log()
    counts = [_I(v) for v in parameters["predictor_oxide_counts"]]
    weights = [_I(v) for v in parameters["predictor_temperature_weights_K"]]
    if any(v.lo <= 0 for v in counts):
        raise ValueError("Every supported dry endmember needs a positive oxide count.")
    ratios = [v/c for v, c in zip(weights, counts)]
    beta = _I(2)/_I(parameters["temperature_K"])
    c0 = _I(parameters["water_gas_cost_rt"])-2*_I(parameters["log_capacity_prefactor"])+a
    c = c0-beta*_I(min(v.lo for v in ratios), max(v.hi for v in ratios))
    if c.lo <= 0:
        raise ValueError("The full dry domain does not have a positive water insertion constant.")
    masses = [_I(v)/_I(parameters["water_mass_kg_mol"]) for v in parameters["dry_masses_kg_mol"]]
    if any(v.lo <= 0 for v in masses):
        raise ValueError("Positive declared dry molar masses are required.")
    return {**parameters, "a": a, "c0": c0, "beta": beta, "counts": counts,
            "weights": weights, "masses": masses, "ratio_bounds": (min(v.lo for v in ratios), max(v.hi for v in ratios)),
            "global_water_constant_interval": c}


def _water_quantities(model, coordinates, *, simplex_box=False):
    k = _dot(model["counts"], coordinates)
    s = _dot(model["masses"], coordinates)
    if simplex_box:
        k = _hull_intersection(k, min(v.lo for v in model["counts"]), max(v.hi for v in model["counts"]))
        s = _hull_intersection(s, min(v.lo for v in model["masses"]), max(v.hi for v in model["masses"]))
    u = _dot(model["weights"], coordinates)/k
    if simplex_box:
        u = _hull_intersection(u, *model["ratio_bounds"])
    c = model["c0"]-model["beta"]*u
    exponential = _exp(-c/2)
    remainder = 1-exponential
    if remainder.lo <= 0:
        raise ArithmeticError("Water elimination crossed its positive-cost domain.")
    logarithm = remainder.log()
    first = exponential/(2*remainder)
    second = -exponential/(4*remainder*remainder)
    ui = [(v-u*count)/k for v, count in zip(model["weights"], model["counts"])]
    ci = [-model["beta"]*v for v in ui]
    return k, s, u, c, logarithm, first, second, ui, ci


def value_gradient(model: dict, point: list) -> tuple:
    """Enclose value and gradient at an exact positive simplex point."""
    x = [_I(v) for v in point]
    if len(point) != len(model["dry_standard_costs_rt"]):
        raise ValueError("The anchor dimension differs from the declared dry model.")
    if sum(map(Fraction, point)) != 1 or any(v.lo <= 0 for v in x):
        raise ValueError("The verification anchor must be an exact positive simplex point.")
    _, s, _, c, logarithm, first, _, _, ci = _water_quantities(model, x)
    alpha = _I(model["entropy_coefficient"])
    matrix = model["quadratic_matrix_rt"]
    mx = [_dot(row, x) for row in matrix]
    value = _dot(model["dry_standard_costs_rt"], x)+alpha*ideal_energy(list(map(Fraction, point)))
    value += _dot(x, mx)/2+model["a"]+2*s*logarithm
    gradient = [_I(linear)+alpha*(v.log()+1)+mixed+2*mass*logarithm+2*s*first*dc
                for linear, v, mixed, mass, dc in zip(model["dry_standard_costs_rt"], x, mx, model["masses"], ci)]
    return value, gradient, {"water_per_dry_component": s/(_exp(c/2)-1),
                             "h2_per_wet_component": 1/(_exp(_I(model["dissolved_h2_cost_rt"]))-1)}


def hessian_lower_enclosure(model: dict, lower, upper) -> list:
    """Enclose the smooth Hessian and minorize the positive ideal diagonal."""
    if len(lower) != len(upper) or len(upper) != len(model["dry_standard_costs_rt"]):
        raise ValueError("The Hessian box dimension differs from the declared model.")
    if any(v <= 0 for v in upper):
        raise ArithmeticError("A zero-width component face needs separate treatment.")
    box = [_I(float(lo), Decimal.from_float(float(hi))) for lo, hi in zip(lower, upper)]
    k, s, _, _, _, first, second, ui, ci = _water_quantities(model, box, simplex_box=True)
    hessian = []
    for i, mass_i in enumerate(model["masses"]):
        row = []
        for j, mass_j in enumerate(model["masses"]):
            cij = model["beta"]*(ui[i]*model["counts"][j]+ui[j]*model["counts"][i])/k
            entry = (_I(model["quadratic_matrix_rt"][i][j])+2*mass_i*first*ci[j]
                     +2*mass_j*first*ci[i]+2*s*(second*ci[i]*ci[j]+first*cij))
            if i == j:
                entry += _I(model["entropy_coefficient"])/_I(float(upper[i]))
            row.append(entry)
        hessian.append(row)
    return hessian


def _simplex_tangent_hessian(hessian, pivot):
    """Enclose B.T H B for columns e_i-e_pivot on sum(x)=1.

    The ideal diagonal already minorizes its actual value. Congruence
    preserves that positive-semidefinite remainder. No curvature in the
    infeasible normal direction is required for a simplex supporting plane.
    """
    indices = [i for i in range(len(hessian)) if i != pivot]
    return [[hessian[i][j]-hessian[i][pivot]-hessian[pivot][j]+hessian[pivot][pivot]
             for j in indices] for i in indices]


def _exact_anchor(candidate, lower, upper):
    """Restore sum=1 exactly and fall back to a rational interior center."""
    candidate = list(map(Fraction, map(float, candidate)))
    index = max(range(len(candidate)), key=lambda i: candidate[i])
    candidate[index] += 1-sum(candidate)
    lo, hi = list(map(Fraction, map(float, lower))), list(map(Fraction, map(float, upper)))
    if any(not a < v < b or v <= 0 for v, a, b in zip(candidate, lo, hi)):
        slack, room = 1-sum(lo), sum(b-a for a, b in zip(lo, hi))
        if room <= 0:
            raise ArithmeticError("The box has no interior simplex anchor.")
        candidate = [a+(b-a)*slack/room for a, b in zip(lo, hi)]
    if sum(candidate) != 1 or any(v <= 0 or v < a or v > b for v, a, b in zip(candidate, lo, hi)):
        raise ArithmeticError("Cannot obtain an exact feasible verification anchor.")
    return candidate


def _floating_model(model):
    mid = lambda v: float((_I(v).lo+_I(v).hi)/2)
    alpha = float(model["entropy_coefficient"])
    matrix = np.asarray(model["quadratic_matrix_rt"])
    costs = np.array([mid(v) for v in model["dry_standard_costs_rt"]])
    weights, counts, masses = [np.array([mid(v) for v in model[name]]) for name in ("weights", "counts", "masses")]
    a, c0, beta = [mid(model[name]) for name in ("a", "c0", "beta")]
    def evaluate(x):
        k, s = counts@x, masses@x
        u = weights@x/k
        c = c0-beta*u
        logarithm = np.log(-np.expm1(-c/2))
        first = .5/np.expm1(c/2)
        ci = -beta*(weights-u*counts)/k
        value = costs@x+alpha*np.dot(x, np.log(x))+.5*x@matrix@x+a+2*s*logarithm
        gradient = costs+alpha*(np.log(x)+1)+matrix@x+2*masses*logarithm+2*s*first*ci
        return value, gradient
    return evaluate


def certify_water_common_plane(parameters: dict, reference_amounts: list, *, tolerance_rt: float = 1e-10,
                                max_nodes: int = 20000, progress_callback=None) -> dict:
    """Cover the dry simplex using verified convex alpha-BB minorants."""
    model = eliminate_volatiles(parameters)
    n = len(reference_amounts)
    if (n != len(parameters["dry_standard_costs_rt"])
            or not np.all(np.isfinite(np.asarray(reference_amounts, dtype=float)))
            or not np.isfinite(tolerance_rt) or tolerance_rt <= 0
            or type(max_nodes) is not int or max_nodes < 1
            or (progress_callback is not None and not callable(progress_callback))):
        raise ValueError("Require a matching finite reference, positive tolerance and integer node budget.")
    amounts = list(map(Fraction, reference_amounts))
    if any(v <= 0 for v in amounts) or n < 2:
        raise ValueError("A complete positive supported dry reference is required.")
    reference = [v/sum(amounts) for v in amounts]
    objective = _floating_model(model)
    leaves, heap = [], []
    counter = 0
    initial_value, _, initial_volatiles = value_gradient(model, reference)
    initial_capacity = _dot(parameters["dry_oxygen_counts"], reference)
    best = {"value_upper_rt": str(initial_value.hi), "coordinates": [str(v) for v in reference],
            "water_capacity_satisfied": initial_volatiles["water_per_dry_component"].hi <= initial_capacity.lo,
            "origin": "saved_dry_reference_with_analytically_minimized_volatiles"}
    def assess_box(lower, upper):
        nonlocal counter, best
        counter += 1
        lower, upper, impossible = _tighten(lower, upper)
        if impossible:
            leaves.append({"status": "empty_simplex", "lower": lower.tolist(), "upper": upper.tolist()})
            return
        if np.any(lower >= upper) or np.any(upper <= 0):
            raise ArithmeticError("Degenerate box requires an explicit boundary proof.")
        pivot = int(np.argmax(upper))
        hessian = _simplex_tangent_hessian(hessian_lower_enclosure(model, lower, upper), pivot)
        if _positive_definite(hessian):
            rho = 0.
        else:
            # Interval row sums determine the shift on the tangent basis.
            # The actual ambient alphaBB diagonal contributes rho*B.T*B;
            # B.T*B = I + 11.T >= I, so verifying H_tangent + rho*I is
            # conservative for that actual minorant.
            required = max((sum((_I(max(v.lo.copy_abs(), v.hi.copy_abs())) for j, v in enumerate(row) if i != j), _I(0))
                            -row[i]).hi for i, row in enumerate(hessian))
            rho = _outward_float(max(Decimal(0), required)+Decimal('1e-20'), False)
            if not _positive_definite([[v+(_I(rho) if i == j else 0) for j, v in enumerate(row)] for i, row in enumerate(hessian)]):
                rho = float(np.nextafter(rho*1.01+1e-8, np.inf))
                if not _positive_definite([[v+(_I(rho) if i == j else 0) for j, v in enumerate(row)] for i, row in enumerate(hessian)]):
                    raise ArithmeticError("The outward curvature shift could not be verified.")
        contains = all(Fraction(float(lo)) <= v <= Fraction(float(hi)) for lo, hi, v in zip(lower, upper, reference))
        if rho == 0 and contains:
            point = reference
        else:
            seed = np.array([float(v) for v in _exact_anchor(.5*(lower+upper), lower, upper)])
            def relaxed(x):
                value, gradient = objective(x)
                return value+.5*rho*np.sum((x-lower)*(x-upper)), gradient+rho*(x-.5*(lower+upper))
            trial = minimize(relaxed, seed, jac=True, method="SLSQP",
                             bounds=[(max(float(a), 1e-15), min(float(b), 1-1e-15)) for a, b in zip(lower, upper)],
                             constraints={"type": "eq", "fun": lambda x: x.sum()-1, "jac": lambda x: np.ones_like(x)},
                             options={"ftol": 1e-11, "maxiter": 100})
            point = _exact_anchor(.999999*trial.x+.000001*seed, lower, upper)
        value, gradient, volatiles = value_gradient(model, point)
        if best["value_upper_rt"] is None or value.hi < Decimal(best["value_upper_rt"]):
            capacity = _dot(parameters["dry_oxygen_counts"], point)
            best = {"value_upper_rt": str(value.hi), "coordinates": [str(v) for v in point],
                    "water_capacity_satisfied": volatiles["water_per_dry_component"].hi <= capacity.lo,
                    "origin": "box_minorant_anchor"}
        for i in range(n):
            value += _I(.5)*_I(rho)*(_I(point[i])-_I(float(lower[i])))*(_I(point[i])-_I(float(upper[i])))
            gradient[i] += _I(rho)*(_I(point[i])-_I(.5)*(_I(float(lower[i]))+_I(float(upper[i]))))
        # The existing affine LP uses binary64 anchors; compensate exactly
        # for that representation without rounding the feasible point.
        adjusted = value+sum((g*(_I(float(x))-_I(x)) for g, x in zip(gradient, point)), _I(0))
        bound = _affine_simplex_lower(adjusted, gradient, list(map(float, point)), lower, upper)
        row = {"status": "verified_convex_minorant", "lower": lower.tolist(), "upper": upper.tolist(),
               "rho": rho, "simplex_tangent_pivot": pivot,
               "anchor_exact": [str(v) for v in point], "lower_bound_rt": str(bound)}
        if bound >= Decimal.from_float(tolerance_rt).copy_negate():
            leaves.append(row)
        else:
            heapq.heappush(heap, (bound, counter, lower, upper, row))
    assess_box(np.zeros(n), np.ones(n))
    while heap and counter+2 <= max_nodes:
        if best["water_capacity_satisfied"] and Decimal(best["value_upper_rt"]) < Decimal.from_float(tolerance_rt).copy_negate():
            break
        _, _, lo, hi, _ = heapq.heappop(heap)
        axis = int(np.argmax(hi-lo))
        mid = .5*(lo[axis]+hi[axis])
        if mid == lo[axis] or mid == hi[axis]:
            raise ArithmeticError("Subdivision reached binary64 resolution.")
        left, right = hi.copy(), lo.copy()
        left[axis], right[axis] = mid, mid
        assess_box(lo.copy(), left)
        assess_box(right, hi.copy())
        if progress_callback is not None and counter % 100 == 1:
            complete_lower = min([row["lower_bound_rt"] for row in leaves if "lower_bound_rt" in row]
                                 + [entry[-1]["lower_bound_rt"] for entry in heap], key=Decimal)
            progress_callback(counter, {**best, "global_lower_bound_rt": complete_lower})
    remaining = [entry[-1] for entry in heap]
    lower = min(Decimal(row["lower_bound_rt"]) for row in leaves+remaining if "lower_bound_rt" in row)
    snapshot = {key: [interval_json(_I(v)) for v in value] if key == "dry_standard_costs_rt" else
                interval_json(value) if isinstance(value, _I) else value for key, value in parameters.items()}
    return {"assessment_id": "reconstructed_water_common_plane_global_bound_v1", "lower_bound_rt_per_dry_component": str(lower),
            "complete_simplex_coverage": True, "bound_within_requested_tolerance": lower >= Decimal.from_float(tolerance_rt).copy_negate(),
            "formal_nonnegative_common_plane_certified": lower >= 0, "parameters": snapshot,
            "tolerance_rt": tolerance_rt, "nodes_evaluated": counter, "best_trial": best, "proof_leaves": leaves,
            "unresolved_boxes": remaining, "volatile_elimination": {"h2_host_term_interval_rt": [str(model["a"].lo), str(model["a"].hi)],
            "water_cost_interval_rt": [str(model["global_water_constant_interval"].lo), str(model["global_water_constant_interval"].hi)],
            "oxygen_capacity_relaxed_only_for_lower_bound": True}, "empirical_material_certified": False}
