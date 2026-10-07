"""Primal/dual bounds for a finite elemental inventory at one T/P.

All real coefficients are the recorded binary64 constants. A negative
insertion bound is retained; lowering every elemental potential by a proved
amount produces a different, valid dual plane. Exact rational atom repair
supplies the feasible primal. Empirical validity is outside this certificate.
"""
from decimal import Decimal, localcontext
from fractions import Fraction

from m2_liquid_global import _I
from m2_solid_global import _poly


def interval_json(value: _I) -> dict:
    return {"lower": str(value.lo), "upper": str(value.hi)}


def _exp(value):
    with localcontext() as context:
        context.prec = 50
        return _I(value.lo.exp().next_minus(), value.hi.exp().next_plus())


def _dot(a, b):
    if len(a) != len(b):
        raise ValueError("An interval dot product requires matching dimensions.")
    return sum((_I(x)*_I(y) for x, y in zip(a, b)), _I(0))


def ideal_energy(amounts: list) -> _I:
    """Extensive ideal mixing, including exact zero faces."""
    total = sum(amounts, Fraction(0))
    if total == 0:
        return _I(0)
    return sum((_I(n)*_I(n/total).log() for n in amounts if n), _I(0))


def ideal_minimum(costs: list) -> _I:
    """Exact simplex minimum -log(sum(exp(-cost))), enclosed outwards."""
    if not costs:
        raise ValueError("An ideal simplex requires at least one component.")
    anchor = min(value.lo for value in costs)
    return _I(anchor)-sum((_exp(_I(anchor)-value) for value in costs), _I(0)).log()


def rational_solve(matrix: list, rhs: list) -> list:
    """Solve a square exact system; no binary64 conversion of rational RHS."""
    rows = [[Fraction(v) for v in row]+[Fraction(b)] for row, b in zip(matrix, rhs)]
    size = len(rows)
    if size == 0 or any(len(row) != size+1 for row in rows):
        raise ValueError("A square system is required.")
    for column in range(size):
        pivot = next((i for i in range(column, size) if rows[i][column]), None)
        if pivot is None:
            raise ValueError("The exact atom basis is singular.")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [value/divisor for value in rows[column]]
        for i in range(size):
            if i != column:
                factor = rows[i][column]
                rows[i] = [a-factor*b for a, b in zip(rows[i], rows[column])]
    return [row[-1] for row in rows]


def _repair_equalities(cols: list, n: list, b: list) -> tuple:
    """Select the same abundant positive basis for atom and optional face rows."""
    chosen, reduced = [], []
    for index in sorted(range(len(n)), key=lambda i: n[i], reverse=True):
        if n[index] == 0:
            continue
        row = cols[index].copy()
        for pivot, old in reduced:
            factor = row[pivot]
            row = [x-factor*y for x, y in zip(row, old)]
        pivot = next((i for i, v in enumerate(row) if v), None)
        if pivot is not None:
            divisor = row[pivot]
            reduced.append((pivot, [v/divisor for v in row]))
            chosen.append(index)
        if len(chosen) == len(b):
            break
    if len(chosen) != len(b):
        raise ValueError("Positive primitive amounts do not span the exact equality constraints.")
    residual = [target-sum(col[j]*v for col, v in zip(cols, n)) for j, target in enumerate(b)]
    changes = rational_solve([[cols[i][j] for i in chosen] for j in range(len(b))], residual)
    repaired = n.copy()
    for i, delta in zip(chosen, changes):
        repaired[i] += delta
    if any(v < 0 for v in repaired):
        raise ValueError("The exact atom correction leaves the nonnegative cone.")
    if any(sum(col[j]*v for col, v in zip(cols, repaired)) != b[j] for j in range(len(b))):
        raise ArithmeticError("Exact equality correction failed.")
    return repaired, chosen, changes


def composition_box_rows(size: int, indices: list, names: list, lower: list, upper: list) -> list:
    """Encode every phase fraction bound as an exact homogeneous row C n >= 0."""
    if (len(indices) != len(names) or len(indices) != len(lower) or len(indices) != len(upper)
            or len(set(indices)) != len(indices) or any(i < 0 or i >= size for i in indices)):
        raise ValueError("Require a complete distinct phase support for the composition box.")
    result = []
    for i, name, low, high in zip(indices, names, lower, upper):
        low, high = Fraction(low), Fraction(high)
        if not 0 <= low <= high <= 1:
            raise ValueError("Composition bounds must remain inside the unit interval.")
        for sense, bound, sign in (("lower", low, 1), ("upper", high, -1)):
            row = [Fraction(0)]*size
            for j in indices:
                row[j] = -sign*bound
            row[i] += sign
            result.append({"name": name+":"+sense, "coefficients": row})
    return result


def feasible_primal(columns: list, amounts: list, budget: list, *, composition_rows=None) -> tuple:
    """Repair atoms and declared homogeneous composition bounds exactly.

    Start with the original abundant atom basis. Any violated composition row
    is added as an exact zero face and the augmented system is solved again
    from the original amounts. Every result must satisfy all atoms, nonnegative
    amounts, and every box row without a tolerance. Rank loss or negative repair
    fails closed; this finite face-addition procedure is not a general LP solve.
    """
    cols = [[Fraction(v) for v in col] for col in columns]
    n, b = list(map(Fraction, amounts)), list(map(Fraction, budget))
    if len(cols) != len(n) or any(len(col) != len(b) for col in cols):
        raise ValueError("Inconsistent primitive atom ledger.")
    if any(v < 0 for v in n+b) or any(v < 0 for col in cols for v in col):
        raise ValueError("Nonnegative atom columns, amounts and budgets are required.")
    constraints = [] if composition_rows is None else [
        {"name": row["name"], "coefficients": list(map(Fraction, row["coefficients"]))}
        for row in composition_rows]
    if (any(len(row["coefficients"]) != len(n) or not isinstance(row["name"], str)
            or not row["name"] for row in constraints)
            or len({row["name"] for row in constraints}) != len(constraints)):
        raise ValueError("Composition rows require distinct names and the complete primitive ledger.")
    dot = lambda row, values: sum((c*v for c, v in zip(row["coefficients"], values)), Fraction(0))
    faces, activations = [], {}
    while True:
        augmented = [col+[constraints[j]["coefficients"][i] for j in faces] for i, col in enumerate(cols)]
        repaired, chosen, changes = _repair_equalities(augmented, n, b+[Fraction(0)]*len(faces))
        violated = next((i for i, row in enumerate(constraints) if dot(row, repaired) < 0), None)
        if violated is None:
            break
        if violated in faces:
            raise ArithmeticError("An exact active composition equality was not preserved.")
        row = constraints[violated]
        activations[violated] = {
            "reason": "original_ledger_outside_box" if dot(row, n) < 0 else "equality_repair_left_box",
            "violated_slack_before_activation_mol_exact": str(dot(row, repaired))}
        faces.append(violated)
    residual = [target-sum(col[j]*v for col, v in zip(cols, n)) for j, target in enumerate(b)]
    receipt = {"exactly_feasible": True, "basis_indices": chosen,
                      "original_atom_residual_mol_exact": [str(v) for v in residual],
                      "amount_corrections_mol_exact": [str(v) for v in changes],
                      "maximum_relative_basis_change": float(max(abs(d)/n[i] for i, d in zip(chosen, changes)))}
    if composition_rows is not None:
        receipt["composition_constraints"] = {
            "schema": "exact_homogeneous_composition_repair_v1", "all_satisfied_exactly": True,
            "added_face_indices": faces,
            "rows": [{"index": i, "name": row["name"],
                      "coefficients_exact": [str(v) for v in row["coefficients"]],
                      "original_slack_mol_exact": str(dot(row, n)),
                      "repaired_slack_mol_exact": str(dot(row, repaired)),
                      "activation": activations.get(i)} for i, row in enumerate(constraints)]}
    return repaired, receipt


def liquid_mixing(parameters: dict, amounts: list) -> _I:
    total = sum(amounts, Fraction(0))
    if total <= 0:
        raise ValueError("A positive liquid amount is required.")
    x = [v/total for v in amounts]
    alpha, matrix = _I(parameters["entropy_coefficient"]), parameters["quadratic_matrix_rt"]
    energy = alpha*ideal_energy(amounts)
    energy += _I(total)*sum((_I(Fraction(1, 2))*_I(a)*_I(c)*_I(b)
                             for a, row in zip(x, matrix) for c, b in zip(row, x)), _I(0))
    water = parameters["water_index"]
    if water is not None:
        energy += alpha*ideal_energy([amounts[water], total-amounts[water]])
    return energy


def liquid_standard_intervals(properties: dict) -> list:
    """Enclose the declared J/(R*T) and the provider's binary64 conversion.

    The real J, R and T constants are retained. Their rounded RT and mu0_RT
    evaluation is included as a coefficient error, never substituted for the
    declared standards. The hull bounds either interpretation independently.
    """
    r, t = properties["basis"]["common_R_J_mol_K"], properties["T_K"]
    rt = _I(r)*_I(t)
    result = []
    for standard, rounded, amount in zip(properties["mu0_J_mol"], properties["mu0_RT"], properties["component_moles"]):
        if not amount:
            result.append(None)
            continue
        if standard/(r*t) != rounded:
            raise ValueError("The saved liquid standard conversion has changed.")
        exact = _I(standard)/rt
        rounded = _I(rounded)
        result.append(_I(min(exact.lo, rounded.lo), max(exact.hi, rounded.hi)))
    return result


def liquid_common_plane(properties: dict, proof: dict, plane: dict, h2_standard: _I) -> tuple:
    """Transfer the global self-plane bound to an external element plane."""
    parameters = properties["mixing_expression"]
    if (properties["model_id"] != "melts_v102_published_mixing_native_standard_states_v1"
            or proof["comparison"]["parameters"] != parameters
            or proof["component_moles"] != properties["component_moles"]
            or not proof["complete_element_supported_provider_domain"]
            or proof["lower_bound_rt"] is None or proof["unresolved_boxes"]):
        raise ValueError("Require the same complete declared-liquid proof and reference.")
    n = list(map(Fraction, properties["component_moles"]))
    total = sum(n)
    reference = [v/total for v in n]
    alpha = _I(parameters["entropy_coefficient"])
    matrix = parameters["quadratic_matrix_rt"]
    active = [i for i, value in enumerate(n) if value]
    water = parameters["water_index"]
    gradient = {}
    for i in active:
        gradient[i] = alpha*(_I(reference[i]).log()+1)+_dot(matrix[i], reference)
        if i == water:
            gradient[i] += alpha*(_I(reference[i]).log()-_I(1-reference[i]).log())
    f = liquid_mixing(parameters, reference)
    intercept = f-sum((_I(reference[i])*gradient[i] for i in active), _I(0))
    standards = liquid_standard_intervals(properties)
    order = properties["basis"]["element_order"]
    columns = properties["basis"]["component_element_matrix"]
    costs = [standards[i]+gradient[i]
             -sum((_I(c)*_I(plane.get(e, 0.)) for e, c in zip(order, columns[i])), _I(0))
             for i in active]
    dry_lower = _I(proof["lower_bound_rt"])+intercept+_I(min(c.lo for c in costs))
    hydrogen = _I(h2_standard)-2*_I(plane["H"])
    lower = ideal_minimum([dry_lower, hydrogen])
    minimum_atoms = min([sum(Fraction(v) for v in columns[i]) for i in active]+[Fraction(2)])
    return lower, minimum_atoms, {"dry_lower_rt": str(dry_lower.lo),
                                 "dissolved_h2_cost_rt": interval_json(hydrogen),
                                 "all_nonnegative_dissolved_h2_fractions": True}


def solution_common_plane(proof: dict, properties: dict, dissolved_h2: float, plane: dict) -> tuple:
    """Translate every signed endmember reaction over the full declared box."""
    if proof["unresolved_box_count"] or proof["lower_bound_rt_per_formula_unit"] is None:
        raise ValueError("A complete solution-domain lower bound is required.")
    reference = proof.get('host_potential_reference')
    if ((reference is None and properties.get('model_id') == 'dry_melts_thompson2025_water_equivalent_v1')
            or (reference is not None and (reference['model_id'] != properties['model_id']
                or reference['dissolved_h2_moles'] != dissolved_h2
                or reference['helium_host_correction_applied'] is not False))):
        raise ValueError('The solution proof and common-plane conversion need the identical bare-host reference.')
    parameters = proof["parameters"]
    order = properties["basis"]["element_order"]
    columns = properties["basis"]["component_element_matrix"]
    total = sum(map(Fraction, properties["component_moles"]))
    dilution = _I(total/(total+Fraction(dissolved_h2))).log()
    shifts = [_I(mu)+dilution-sum((_I(c)*_I(plane.get(e, 0.)) for e, c in zip(order, col)), _I(0))
              if n else None for mu, n, col in zip(properties["mu_RT"], properties["component_moles"], columns)]
    bounds = [list(row) for row in parameters["coordinate_bounds"]]
    for row in proof["element_support_restrictions"]:
        bounds[row["coordinate"]] = [row["value"]]*2
    box = [_I(float(a), Decimal.from_float(float(b))) for a, b in bounds]
    correction_terms = {}
    atom_polynomial = []
    for coefficients, polynomial in zip(proof["host_reactions_exact"], parameters["endmember_polynomials"]):
        weight = _poly(polynomial, box)
        if coefficients is None:
            if weight.lo != 0 or weight.hi != 0:
                raise ValueError("An omitted endpoint is active in the full domain.")
            continue
        reaction = list(map(Fraction, coefficients))
        shift = sum((_I(c)*shifts[i] for i, c in enumerate(reaction) if c), _I(0))
        for coefficient, powers in polynomial:
            key = tuple(powers)
            correction_terms[key] = correction_terms.get(key, _I(0))+_I(coefficient)*shift
        atoms = sum(c*sum(map(Fraction, col)) for c, col in zip(reaction, columns))
        atom_polynomial.extend([(_I(atoms)*_I(coefficient), powers) for coefficient, powers in polynomial])
    # Combine equal powers before interval evaluation (e.g. total atoms may
    # be constant although some formal endmember coordinates are signed).
    combined = {}
    for coefficient, powers in atom_polynomial:
        key = tuple(powers)
        combined[key] = combined.get(key, _I(0))+coefficient
    minimum_atoms = _poly([(c, p) for p, c in combined.items()], box)
    # Signed endmember coordinates share monomials. Preserve their
    # cancellations before enclosing the reference shift over the domain.
    correction = _poly([(c, p) for p, c in correction_terms.items()], box)
    lower = _I(proof["lower_bound_rt_per_formula_unit"])+correction
    # Positive insertion costs require no atom normalization or dual shift.
    if lower.lo < 0 and minimum_atoms.lo <= 0:
        raise ValueError("A negative solution bound needs a positive atom-count bound.")
    return lower, minimum_atoms, {"reference_conversion_interval_rt": interval_json(correction),
                                  "original_lower_bound_rt": proof["lower_bound_rt_per_formula_unit"]}


def primal_dual_certificate(budget: list, potentials: list, primal_energy: _I, phase_bounds: list, *, tolerance: float = 1e-9) -> dict:
    """Weak duality with an explicit common-plane correction in RT units."""
    if not phase_bounds or len(budget) != len(potentials) or any(Fraction(v) < 0 for v in budget):
        raise ValueError("A complete nonnegative finite budget and phase catalog are required.")
    if not 0 < tolerance <= 1e-9:
        raise ValueError("The normalized gap tolerance cannot exceed the M2 energy contract.")
    for value in list(budget)+list(potentials)+[primal_energy]:
        interval = _I(value)
        if not interval.lo.is_finite() or not interval.hi.is_finite() or interval.lo > interval.hi:
            raise ValueError("The finite-source bound requires finite ordered intervals.")
    correction = Decimal(0)
    for row in phase_bounds:
        lower, atoms = row["bound"], row["minimum_atoms"]
        if any(not end.is_finite() for value in (lower, atoms) for end in (value.lo, value.hi)):
            raise ValueError("Every phase bound must be finite.")
        if lower.lo > lower.hi or atoms.lo > atoms.hi:
            raise ValueError("Every phase interval must be ordered.")
        if lower.lo < 0:
            if atoms.lo <= 0:
                raise ValueError("A negative phase bound requires positive atoms per unit.")
            correction = max(correction, (-_I(lower.lo)/_I(atoms.lo)).hi)
    atom_total = sum(map(Fraction, budget))
    if atom_total <= 0:
        raise ValueError("The atom budget is empty.")
    original_dual = _dot(budget, potentials)
    dual = original_dual-_I(correction)*_I(atom_total)
    gap = primal_energy-dual
    if gap.hi < 0:
        raise ArithmeticError("A feasible primal cannot lie below a global dual bound.")
    normalized = gap/_I(atom_total)
    return {"original_common_plane_rt_mol": interval_json(original_dual),
            "uniform_element_plane_reduction_rt": str(correction),
            "corrected_dual_lower_rt_mol": str(dual.lo),
            "primal_energy_rt_mol": interval_json(primal_energy),
            "primal_dual_gap_rt_mol": interval_json(gap),
            "normalized_gap_per_inventory_atom": interval_json(normalized),
            "energy_tolerance_rt_per_inventory_atom": tolerance,
            "declared_model_gap_accepted": normalized.hi <= Decimal.from_float(tolerance),
            "original_plane_strict_nonnegative": all(row["bound"].lo >= 0 for row in phase_bounds),
            "original_negative_bounds_preserved": True,
            "empirical_material_certified": False}
