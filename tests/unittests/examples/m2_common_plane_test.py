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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_common_plane_inputs_fail_closed(value):
    with pytest.raises((ValueError, ArithmeticError)):
        M.primal_dual_certificate([1.], [value], I(0), [{"bound": I(0), "minimum_atoms": I(1)}])
    with pytest.raises((ValueError, ArithmeticError)):
        M.primal_dual_certificate([1.], [0.], I(0), [{"bound": I(value), "minimum_atoms": I(1)}])


def test_dot_product_rejects_truncated_element_vectors():
    with pytest.raises(ValueError, match="dimensions"):
        M._dot([1., 2.], [3.])
