"""Exact primal feasibility and global duality cannot be replaced by KKT flags."""
from decimal import Decimal, localcontext
from fractions import Fraction
import importlib
from pathlib import Path
import sys

import pytest

DIRECTORY = Path(__file__).resolve().parents[3]/"examples/metal_silicate"
sys.path.insert(0, str(DIRECTORY))
try:
    M = importlib.import_module("m2_common_plane")
finally:
    sys.path.pop(0)
I = M._I


def test_exact_rational_repair_preserves_every_atom_without_clipping():
    columns = [[1, 1], [1, 0], [0, 1], [1, 2]]
    amounts = [1e24, 2e20, 3e20, 1e-20]
    budget = [float(1e24+2e20), float(1e24+3e20)]
    repaired, receipt = M.feasible_primal(columns, amounts, budget)
    assert receipt["exactly_feasible"]
    assert any(Fraction(v) for v in receipt["amount_corrections_mol_exact"])
    assert all(v >= 0 for v in repaired)
    assert repaired[-1] == Fraction(amounts[-1])
    assert [sum(n*col[j] for n, col in zip(repaired, columns)) for j in range(2)] == list(map(Fraction, budget))


def test_repair_rejects_rank_loss_or_a_negative_amount():
    with pytest.raises(ValueError, match="span"):
        M.feasible_primal([[1, 1], [2, 2]], [1., 2.], [1., 1.])
    with pytest.raises(ValueError, match="nonnegative cone"):
        M.feasible_primal([[1, 1], [0, 1]], [2., 1.], [3., 1.])


def test_exact_box_repair_removes_roundoff_violation_without_widening_domain():
    columns = [[1, 0], [0, 1], [1, 0], [0, 1]]
    amounts = list(map(Fraction, [1., 1., .98, .020000000000000004]))
    original = amounts.copy()
    budget = [sum(n*col[j] for n, col in zip(amounts, columns)) for j in range(2)]
    rows = M.composition_box_rows(4, [2, 3], ['Fe', 'P'], [.75, 0.], [1., .02])
    repaired, receipt = M.feasible_primal(columns, amounts, budget, composition_rows=rows)
    assert amounts == original
    assert repaired[3]/sum(repaired[2:]) == Fraction(.02)
    assert repaired[2]/sum(repaired[2:]) >= Fraction(.75)
    assert all(value >= 0 for value in repaired)
    assert [sum(n*col[j] for n, col in zip(repaired, columns)) for j in range(2)] == budget
    audit = receipt['composition_constraints']
    assert audit['all_satisfied_exactly'] and audit['added_face_indices'] == [3]
    assert audit['rows'][3]['activation']['reason'] == 'original_ledger_outside_box'
    assert Fraction(audit['rows'][3]['original_slack_mol_exact']) < 0
    assert Fraction(audit['rows'][3]['repaired_slack_mol_exact']) == 0
    assert receipt['maximum_relative_basis_change'] < 1e-14
    # Independently replay the saved augmented equality solve and all row slacks.
    chosen = receipt['basis_indices']
    augmented = [[col[j] for col in columns] for j in range(2)]
    augmented += [list(map(Fraction, audit['rows'][i]['coefficients_exact'])) for i in audit['added_face_indices']]
    residual = list(map(Fraction, receipt['original_atom_residual_mol_exact']))
    residual += [-sum(c*n for c, n in zip(row, original)) for row in augmented[2:]]
    changes = M.rational_solve([[row[i] for i in chosen] for row in augmented], residual)
    assert changes == list(map(Fraction, receipt['amount_corrections_mol_exact']))
    for row in audit['rows']:
        coefficients = list(map(Fraction, row['coefficients_exact']))
        assert sum(c*n for c, n in zip(coefficients, original)) == Fraction(row['original_slack_mol_exact'])
        assert sum(c*n for c, n in zip(coefficients, repaired)) == Fraction(row['repaired_slack_mol_exact']) >= 0


def test_atom_repair_can_activate_a_previously_satisfied_box_face():
    columns = [[1, 0], [0, 1], [1, 0], [0, 1]]
    amounts = [Fraction(1), Fraction(1), Fraction(1, 10), Fraction(1, 10)]
    budget = [Fraction(11, 10)-Fraction(1, 10**16), Fraction(11, 10)]
    rows = M.composition_box_rows(4, [0, 1], ['Fe', 'P'], [0, 0], [1, Fraction(1, 2)])
    repaired, receipt = M.feasible_primal(columns, amounts, budget, composition_rows=rows)
    row = receipt['composition_constraints']['rows'][3]
    assert row['original_slack_mol_exact'] == row['repaired_slack_mol_exact'] == '0'
    assert row['activation']['reason'] == 'equality_repair_left_box'
    assert repaired[1]/sum(repaired[:2]) == Fraction(1, 2)


def test_new_face_correction_rechecks_every_other_composition_row():
    rows = [{'name': 'zero_first', 'coefficients': [-1, 0, 0]},
            {'name': 'second_upper', 'coefficients': [Fraction(1, 2), Fraction(-1, 2), Fraction(1, 2)]}]
    repaired, receipt = M.feasible_primal([[1], [1], [1]], [1, 1, 1], [3], composition_rows=rows)
    assert repaired == [0, Fraction(3, 2), Fraction(3, 2)]
    assert receipt['composition_constraints']['added_face_indices'] == [0, 1]
    assert all(Fraction(row['repaired_slack_mol_exact']) == 0 for row in receipt['composition_constraints']['rows'])


def test_box_repair_rejects_rank_loss_negative_amounts_and_truncated_rows():
    with pytest.raises(ValueError, match='span'):
        M.feasible_primal([[1]], [1], [1], composition_rows=[{'name': 'impossible', 'coefficients': [-1]}])
    with pytest.raises(ValueError, match='nonnegative cone'):
        M.feasible_primal([[1], [1]], [1, 1], [2], composition_rows=[{'name': 'impossible', 'coefficients': [-2, -3]}])
    with pytest.raises(ValueError, match='complete primitive ledger'):
        M.feasible_primal([[1], [1]], [1, 1], [2], composition_rows=[{'name': 'truncated', 'coefficients': [1]}])


def test_inactive_composition_rows_leave_atom_repair_exactly_unchanged():
    columns, amounts, budget = [[1, 0], [0, 1]], [1., 2.], [1.0000000000000002, 2.]
    expected, original = M.feasible_primal(columns, amounts, budget)
    rows = M.composition_box_rows(2, [0, 1], ['A', 'B'], [0, 0], [1, 1])
    actual, receipt = M.feasible_primal(columns, amounts, budget, composition_rows=rows)
    assert actual == expected
    extra = receipt.pop('composition_constraints')
    assert extra['added_face_indices'] == []
    assert receipt == original


def test_interval_ideal_simplex_minimum_encloses_independent_analytic_value():
    result = M.ideal_minimum([I(Fraction(1, 3)), I(Fraction(5, 7))])
    with localcontext() as context:
        context.prec = 130
        exact = -((-Decimal(1)/3).exp()+(-Decimal(5)/7).exp()).ln()
    assert result.lo <= exact <= result.hi
    pure = M.ideal_energy([Fraction(0), Fraction(10)])
    assert pure.lo <= 0 <= pure.hi
    inverse = 1/I(Fraction(3, 7))
    assert inverse.lo <= Fraction(7, 3) <= inverse.hi


def test_negative_fixed_plane_is_preserved_while_a_corrected_dual_is_accepted():
    tiny = -1e-13
    result = M.primal_dual_certificate([100.], [0.], I(0),
                                      [{"bound": I(tiny), "minimum_atoms": I(2)}])
    assert result["declared_model_gap_accepted"]
    assert not result["original_plane_strict_nonnegative"]
    assert Decimal(result["uniform_element_plane_reduction_rt"]) >= -Decimal.from_float(tiny)/2
    assert Decimal(result["normalized_gap_per_inventory_atom"]["upper"]) >= -Decimal.from_float(tiny)/2
    assert not result["empirical_material_certified"]


def test_large_negative_competing_phase_cannot_pass_via_small_kkt_residual():
    result = M.primal_dual_certificate([1.], [0.], I(0),
                                      [{"bound": I(-.1), "minimum_atoms": I(1)}])
    assert not result["declared_model_gap_accepted"]
    with pytest.raises(ValueError, match="positive atoms"):
        M.primal_dual_certificate([1.], [0.], I(0), [{"bound": I(-.1), "minimum_atoms": I(0)}])
    with pytest.raises(ArithmeticError, match="below"):
        M.primal_dual_certificate([1.], [0.], I(-1), [{"bound": I(0), "minimum_atoms": I(1)}])


def liquid_fixture():
    parameters = {"entropy_coefficient": 1., "quadratic_matrix_rt": [[0., 0.], [0., 0.]], "water_index": None}
    properties = {"model_id": "melts_v102_published_mixing_native_standard_states_v1",
                  "mixing_expression": parameters, "component_moles": [1., 1.],
                  "mu0_J_mol": [0., 0.], "mu0_RT": [0., 0.], "T_K": 2.,
                  "basis": {"common_R_J_mol_K": 1., "element_order": ["A", "H"],
                            "component_element_matrix": [[1., 0.], [0., 2.]]}}
    proof = {"comparison": {"parameters": parameters}, "component_moles": [1., 1.],
             "complete_element_supported_provider_domain": True, "lower_bound_rt": 0., "unresolved_boxes": []}
    return properties, proof


def test_liquid_reference_transfer_includes_arbitrary_hydrogen_fraction():
    properties, proof = liquid_fixture()
    lower, atoms, _ = M.liquid_common_plane(properties, proof, {"A": 0., "H": 0.}, I(0))
    with localcontext() as context:
        context.prec = 130
        exact = -Decimal(3).ln()
    assert lower.lo <= exact <= lower.hi
    assert atoms == 1
    proof["complete_element_supported_provider_domain"] = False
    with pytest.raises(ValueError, match="complete"):
        M.liquid_common_plane(properties, proof, {"A": 0., "H": 0.}, I(0))


def test_liquid_conversion_roundoff_is_bounded_without_replacing_real_standard():
    properties, _ = liquid_fixture()
    properties.update(T_K=3., mu0_J_mol=[1., 1.], mu0_RT=[1/3, 1/3])
    interval = M.liquid_standard_intervals(properties)[0]
    assert interval.lo <= Fraction(1, 3) <= interval.hi
    assert interval.lo <= Decimal.from_float(1/3) <= interval.hi
    properties["mu0_RT"][0] = .34
    with pytest.raises(ValueError, match="conversion"):
        M.liquid_standard_intervals(properties)


def test_signed_solution_endmember_coordinates_keep_reference_error_direction():
    properties, _ = liquid_fixture()
    properties["mu_RT"] = [2., 3.]
    proof = {"unresolved_box_count": 0, "lower_bound_rt_per_formula_unit": 1.,
             "element_support_restrictions": [], "host_reactions_exact": [["-1", "1"], ["2", "0"]],
             "parameters": {"coordinate_bounds": [[0., 1.]],
                            "endmember_polynomials": [[[1., [1]]], [[1., [0]], [-1., [1]]]]}}
    lo, _, details = M.solution_common_plane(proof, properties, 0., {"A": 2.1, "H": 1.55})
    assert lo.lo > 0
    assert Decimal(details["reference_conversion_interval_rt"]["lower"]) < 0
    # The omitted endpoint could carry negative weight; an unavailable value
    # cannot be assumed zero merely because some compositions avoid it.
    proof["host_reactions_exact"][0] = None
    with pytest.raises(ValueError, match="omitted endpoint"):
        M.solution_common_plane(proof, properties, 0., {"A": 2.1, "H": 1.55})


@pytest.mark.parametrize("shift", [.3, -.3])
def test_shared_signed_endmember_monomials_preserve_constant_reference_shift(shift):
    properties, _ = liquid_fixture()
    properties["mu_RT"] = [shift, 0.]
    proof = {"unresolved_box_count": 0, "lower_bound_rt_per_formula_unit": 0.,
             "element_support_restrictions": [], "host_reactions_exact": [["1", "0"], ["1", "0"]],
             "parameters": {"coordinate_bounds": [[0., 1.]],
                            "endmember_polynomials": [[[1., [0]], [-2., [1]]], [[2., [1]]]]}}
    lower, atoms, details = M.solution_common_plane(proof, properties, 0., {"A": 0., "H": 0.})
    # (1 - 2*x)*shift + 2*x*shift is constant, even where a weight is negative.
    exact = Decimal.from_float(shift)
    assert lower.lo <= exact <= lower.hi
    assert lower.hi-lower.lo < Decimal("1e-45")
    assert atoms.lo <= 1 <= atoms.hi
    assert (Decimal(details["reference_conversion_interval_rt"]["upper"]) < 0) == (shift < 0)


def test_signed_reference_rebase_keeps_a_real_negative_endpoint():
    properties, _ = liquid_fixture()
    properties["mu_RT"] = [.25, -.25]
    proof = {"unresolved_box_count": 0, "lower_bound_rt_per_formula_unit": 0.,
             "element_support_restrictions": [], "host_reactions_exact": [["1", "0"], ["0", "1"]],
             "parameters": {"coordinate_bounds": [[0., 1.]],
                            "endmember_polynomials": [[[1., [0]], [-2., [1]]], [[2., [1]]]]}}
    lower, _, _ = M.solution_common_plane(proof, properties, 0., {"A": 0., "H": 0.})
    # The exact correction is 1/4 - x; combining terms must not erase its minimum.
    assert lower.lo <= Decimal("-.75") <= lower.hi
    assert lower.lo > Decimal("-.750000000000001")
    certificate = M.primal_dual_certificate([1.], [0.], I(0),
                                          [{"bound": lower, "minimum_atoms": I(1)}])
    assert not certificate["declared_model_gap_accepted"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_common_plane_inputs_fail_closed(value):
    with pytest.raises((ValueError, ArithmeticError)):
        M.primal_dual_certificate([1.], [value], I(0), [{"bound": I(0), "minimum_atoms": I(1)}])
    with pytest.raises((ValueError, ArithmeticError)):
        M.primal_dual_certificate([1.], [0.], I(0), [{"bound": I(value), "minimum_atoms": I(1)}])


def test_dot_product_rejects_truncated_element_vectors():
    with pytest.raises(ValueError, match="dimensions"):
        M._dot([1., 2.], [3.])
