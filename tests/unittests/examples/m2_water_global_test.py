"""Analytical volatile elimination and verified whole-simplex bounds."""
from decimal import Decimal, localcontext
from fractions import Fraction
import copy
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parents[3]/"examples/metal_silicate"
sys.path.insert(0, str(DIRECTORY))
try:
    M = importlib.import_module("m2_water_global")
finally:
    sys.path.pop(0)


def parameters():
    return {"dissolved_h2_cost_rt": 4., "water_gas_cost_rt": 2., "log_capacity_prefactor": 0.,
            "predictor_oxide_counts": [1., 1.], "predictor_temperature_weights_K": [0., 0.],
            "temperature_K": 2000., "water_mass_kg_mol": 1., "dry_masses_kg_mol": [1., 1.],
            "dry_oxygen_counts": [2., 3.], "dry_standard_costs_rt": [2., 2.],
            "entropy_coefficient": 1., "quadratic_matrix_rt": [[0., 0.], [0., 0.]]}


def test_global_ideal_control_matches_closed_form_with_both_volatile_minima():
    result = M.certify_water_common_plane(parameters(), [1., 1.])
    with localcontext() as context:
        context.prec = 100
        a = (1-Decimal(-4).exp()).ln()
        exact = 2-Decimal(2).ln()+a+2*(1-(-(2+a)/2).exp()).ln()
    lower = Decimal(result["lower_bound_rt_per_dry_component"])
    upper = Decimal(result["best_trial"]["value_upper_rt"])
    assert lower <= exact <= upper
    assert upper-lower < Decimal('1e-45')
    assert result["nodes_evaluated"] == 1
    assert result["bound_within_requested_tolerance"]
    assert result["best_trial"]["water_capacity_satisfied"]


def test_negative_water_branch_is_a_witness_not_an_accepted_phase_bound():
    model = parameters()
    model["dry_standard_costs_rt"] = [1., 1.]
    result = M.certify_water_common_plane(model, [1., 1.], max_nodes=3)
    assert not result["bound_within_requested_tolerance"]
    assert Decimal(result["best_trial"]["value_upper_rt"]) < -.1
    assert result["unresolved_boxes"]
    assert not result["empirical_material_certified"]


def test_saved_reference_remains_a_trial_when_a_minorant_anchor_is_worse():
    p = parameters()
    p.update(quadratic_matrix_rt=[[0., 4.], [4., 0.]])
    result = M.certify_water_common_plane(p, [1., 999.], max_nodes=1)
    assert result["best_trial"]["origin"] == "saved_dry_reference_with_analytically_minimized_volatiles"
    model = M.eliminate_volatiles(p)
    expected, _, _ = M.value_gradient(model, [Fraction(1, 1000), Fraction(999, 1000)])
    assert Decimal(result["best_trial"]["value_upper_rt"]) == expected.hi


def test_elimination_requires_positive_cost_on_every_dry_composition():
    model = parameters()
    model["dissolved_h2_cost_rt"] = 0.
    with pytest.raises(ValueError, match="positive common-plane"):
        M.eliminate_volatiles(model)
    model = parameters()
    model["predictor_temperature_weights_K"][1] = 10000.
    with pytest.raises(ValueError, match="full dry domain"):
        M.eliminate_volatiles(model)


def test_water_capacity_is_relaxed_only_for_the_lower_bound():
    model = parameters()
    model["dry_oxygen_counts"] = [.0001, .0001]
    result = M.certify_water_common_plane(model, [1., 1.], max_nodes=3)
    assert result["bound_within_requested_tolerance"]
    assert not result["best_trial"]["water_capacity_satisfied"]
    assert result["volatile_elimination"]["oxygen_capacity_relaxed_only_for_lower_bound"]


def test_rational_composition_term_gradient_and_hessian_match_independent_ad():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    p = parameters()
    p.update(predictor_temperature_weights_K=[-1200., 800.], predictor_oxide_counts=[1., 2.],
             dry_masses_kg_mol=[1., 3.], quadratic_matrix_rt=[[0., 2.], [2., 0.]])
    model = M.eliminate_volatiles(p)
    point = [Fraction(1, 3), Fraction(2, 3)]
    _, gradient, _ = M.value_gradient(model, point)
    hessian = M.hessian_lower_enclosure(model, np.array([.3, .6]), np.array([.4, .7]))
    def scalar(x):
        a = jnp.log1p(-jnp.exp(-4.))
        c = 2.+a-2.*jnp.dot(jnp.array([-1200., 800.]), x)/(2000.*jnp.dot(jnp.array([1., 2.]), x))
        return (2.*x.sum()+jnp.sum(x*jnp.log(x))+2.*x[0]*x[1]+a
                +2.*jnp.dot(jnp.array([1., 3.]), x)*jnp.log1p(-jnp.exp(-c/2.)))
    x = jnp.array(list(map(float, point)))
    actual_gradient = np.array(jax.grad(scalar)(x))
    actual_hessian = np.array(jax.hessian(scalar)(x))
    np.testing.assert_allclose([float(g.lo) for g in gradient], actual_gradient, rtol=1e-13, atol=1e-13)
    # The ideal diagonal was minorized at the box upper edge; restore that
    # known difference before comparing the independent point Hessian.
    actual_hessian += np.diag(1/np.array([.4, .7])-1/np.asarray(x))
    for i, row in enumerate(hessian):
        for j, value in enumerate(row):
            assert float(value.lo)-1e-12 <= actual_hessian[i, j] <= float(value.hi)+1e-12


def test_verification_anchor_must_satisfy_the_simplex_exactly():
    model = M.eliminate_volatiles(parameters())
    with pytest.raises(ValueError, match="exact positive"):
        M.value_gradient(model, [.1, .9])
    point = M._exact_anchor([.1, .9], np.array([0., 0.]), np.array([1., 1.]))
    assert sum(point) == 1
    M.value_gradient(model, point)


@pytest.mark.parametrize("change", [
    lambda p: p.update(dry_masses_kg_mol=[1.]),
    lambda p: p.update(predictor_oxide_counts=[1., 1., 1.]),
    lambda p: p.update(temperature_K=0.),
    lambda p: p.update(water_mass_kg_mol=-1.),
    lambda p: p.update(entropy_coefficient=0.),
    lambda p: p.update(log_capacity_prefactor=float("nan")),
    lambda p: p.update(water_gas_cost_rt=float("inf")),
    lambda p: p.update(dry_standard_costs_rt=[float("-inf"), 2.]),
    lambda p: p.update(quadratic_matrix_rt=[[0., 1.], [0., 0.]]),
    lambda p: p.update(quadratic_matrix_rt=[[0., 1.]]),
    lambda p: p.update(dry_oxygen_counts=[-1., 2.]),
])
def test_malformed_provider_coefficients_fail_closed(change):
    model = parameters()
    change(model)
    with pytest.raises((ValueError, ArithmeticError)):
        M.certify_water_common_plane(model, [1., 1.], max_nodes=3)


@pytest.mark.parametrize("reference,options", [
    ([1.], {}), ([1., float("nan")], {}), ([1., -1.], {}),
    ([1., 1.], {"tolerance_rt": 0.}), ([1., 1.], {"tolerance_rt": float("nan")}),
    ([1., 1.], {"max_nodes": 0}), ([1., 1.], {"max_nodes": 3.5}), ([1., 1.], {"max_nodes": True}),
])
def test_invalid_reference_or_proof_budget_is_not_a_certificate(reference, options):
    with pytest.raises((ValueError, ArithmeticError)):
        M.certify_water_common_plane(parameters(), reference, **options)


def saved_water():
    names, elements = ["a", "b", "h2o"], ["X", "O", "H"]
    columns = [[1., 2., 0.], [1., 3., 0.], [0., 1., 2.]]
    mixing = {"component_order": names, "T_K": 2000., "P_Pa": 1e5, "common_R_J_mol_K": 1.,
              "element_order": elements, "component_element_matrix": columns,
              "entropy_coefficient": 1., "quadratic_matrix_rt": [[0.]*3 for _ in range(3)],
              "water_index": 2, "unsupported_positive_indices": []}
    basis = {"element_order": elements, "component_element_matrix": columns, "common_R_J_mol_K": 1.}
    dry = {"model_id": "melts_v102_published_mixing_native_standard_states_v1", "component_order": names,
           "T_K": 2000., "P_Pa": 1e5, "component_moles": [1., 1., 0.], "basis": basis,
           "mu0_J_mol": [4000., 4000., None], "mu0_RT": [2., 2., None], "mixing_expression": mixing}
    expression = {"schema": "dry_melts_water_equivalent_expression_v1", "component_order": names,
                  "water_index": 2, "component_masses_kg_mol": [1., 1., 1.], "water_mass_kg_mol": 1.,
                  "predictor_oxide_counts": [1., 1., 0.], "predictor_temperature_weights_K": [0., 0., 0.],
                  "component_oxygen_counts": [2., 3., 1.], "log_capacity_prefactor": 0.,
                  "capacity_formula": "log_Cw = log_prefactor + (weights @ dry)/(T_K*(counts @ dry))",
                  "gibbs_formula": "G_RT = G_dry_RT + h*(gas_standard_RT - 2*log_Cw) + 2*G_mass_fraction_RT",
                  "water_mixing_factor": 2., "water_standard_pressure_Pa": 1e5,
                  "domain": "nonnegative amounts; positive dry host; h <= oxygen @ dry"}
    return {"model_id": "dry_melts_thompson2025_water_equivalent_v1", "component_order": names,
            "T_K": 2000., "P_Pa": 1e5, "component_moles": [1., 1., .1], "basis": basis,
            "water_reconstruction": {"expression": expression, "dry_properties": dry,
                                     "gas_H2O_standard_RT": 2., "gas_standard_pressure_Pa": 1e5,
                                     "native_water_amount_used_mol": 0.}}


def test_saved_expression_and_external_water_shift_remain_distinct():
    saved = saved_water()
    saved["water_reconstruction"]["standard_offset_rt"] = .5
    unchanged = copy.deepcopy(saved)
    parameters, amounts, binding = M.parameters_from_saved_water(
        saved, {"X": 0., "O": 0., "H": 0.}, {"X": 2., "O": 5.1, "H": .2}, 4.,
        water_standard_offset_rt=.5)
    assert saved == unchanged
    assert parameters["water_gas_cost_rt"].lo == Decimal('2.5')
    assert binding["gas_H2O_standard_RT"] == 2.
    assert binding["external_h2o_melts_standard_offset_rt"] == .5
    assert binding["minimum_atoms_per_bound_unit_exact"] == "3"
    assert M.certify_water_common_plane(parameters, amounts)["bound_within_requested_tolerance"]


def test_zero_reference_does_not_silently_restrict_global_domain():
    saved = saved_water()
    saved["component_moles"][1] = 0.
    saved["water_reconstruction"]["dry_properties"]["component_moles"][1] = 0.
    with pytest.raises(ValueError, match="zero-reference component"):
        M.parameters_from_saved_water(saved, {"X": 0., "O": 0., "H": 0.}, {"X": 2., "O": 5.1, "H": .2}, 4.)


@pytest.mark.parametrize("change", [
    lambda p: p["water_reconstruction"]["expression"].update(water_mixing_factor=1.),
    lambda p: p["water_reconstruction"]["expression"].update(gibbs_formula="different scalar"),
    lambda p: p["water_reconstruction"]["dry_properties"].update(T_K=2001.),
    lambda p: p["water_reconstruction"]["dry_properties"]["mixing_expression"].update(T_K=2001.),
    lambda p: p["water_reconstruction"]["dry_properties"].update(component_moles=[1., 1., .1]),
    lambda p: p["water_reconstruction"]["dry_properties"].update(mu0_RT=[2.1, 2., None]),
    lambda p: p["water_reconstruction"]["expression"].update(component_masses_kg_mol=[1., 1.]),
])
def test_saved_provider_contract_changes_cannot_enter_the_certificate(change):
    saved = saved_water()
    change(saved)
    with pytest.raises(ValueError):
        M.parameters_from_saved_water(saved, {"X": 0., "O": 0., "H": 0.}, {"X": 2., "O": 5.1, "H": .2}, 4.)


def test_uneliminated_scalar_respects_capacity_and_extensivity():
    p = parameters()
    value = M.water_insertion_value(p, [1., 2.], .3, .1)
    scaled = M.water_insertion_value(p, [3., 6.], .9, .3)
    assert abs(float(value.lo)*3-float(scaled.lo)) < 1e-14
    with pytest.raises(ValueError, match="capacity domain"):
        M.water_insertion_value(p, [1., 2.], 9., .1)
    model = M.eliminate_volatiles(p)
    dry = [Fraction(1, 3), Fraction(2, 3)]
    optimum, _, amounts = M.value_gradient(model, dry)
    h = float(amounts["water_per_dry_component"].lo)
    hydrogen = float(amounts["h2_per_wet_component"].lo)*(1+h)
    trial = M.water_insertion_value(p, dry, h, hydrogen)
    assert float(trial.lo) == pytest.approx(float(optimum.lo), abs=1e-14)
    assert trial.hi >= optimum.lo


@pytest.mark.parametrize("change", [
    lambda p: p["water_reconstruction"]["expression"].update(log_capacity_prefactor=.1),
    lambda p: p["water_reconstruction"].update(gas_H2O_standard_RT=2.1),
    lambda p: p["water_reconstruction"].update(standard_offset_rt=.1),
    lambda p: p["water_reconstruction"]["dry_properties"].update(mu0_J_mol=[4000., 5000., None]),
])
def test_fresh_water_receipt_must_keep_the_same_expression_and_standards(change):
    require = importlib.import_module("m2_liquid_global").require_saved_liquid_expression
    saved = saved_water()
    evaluated = copy.deepcopy(saved)
    require(saved, evaluated)
    change(evaluated)
    with pytest.raises(ValueError):
        require(saved, evaluated)


def test_simplex_curvature_does_not_require_positive_normal_curvature():
    # H = 2 I - 20 11.T has a negative normal eigenvalue but is strictly
    # convex on sum(x)=1. The exact congruence removes only that normal.
    hessian = [[M._I(2 if i == j else 0)-20 for j in range(3)] for i in range(3)]
    tangent = M._simplex_tangent_hessian(hessian, 1)
    assert [[float(v.lo) for v in row] for row in tangent] == [[4., 2.], [2., 4.]]
    assert M._positive_definite(tangent)
    assert not M._positive_definite(hessian)


def test_tangent_shift_has_the_ambient_alphabb_positive_remainder():
    # B.T B = I + 11.T: an ambient diagonal shift is at least the
    # conservative rho I checked in the tangent-coordinate certificate.
    hessian = [[M._I(0) for _ in range(4)] for _ in range(4)]
    shifted = [[v+(3 if i == j else 0) for j,v in enumerate(row)]
               for i,row in enumerate(hessian)]
    tangent = M._simplex_tangent_hessian(shifted, 2)
    assert [[float(v.lo) for v in row] for row in tangent] == [[6.,3.,3.],[3.,6.,3.],[3.,3.,6.]]
