"""The optimizer is not trusted by the interval phase-stability verifier."""

from decimal import Decimal, localcontext
from fractions import Fraction
import importlib.util
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[3] / "examples/metal_silicate/m2_liquid_global.py"
spec = importlib.util.spec_from_file_location("global_liquid", PATH)
global_liquid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(global_liquid)


def parameters(interaction=0):
    return {"quadratic_matrix_rt": [[0., interaction], [interaction, 0.]],
            "entropy_coefficient": 1., "water_index": None}


def test_ideal_liquid_is_globally_supported_with_zero_parent_components():
    model = {"quadratic_matrix_rt": np.zeros((3, 3)).tolist(),
             "entropy_coefficient": 1., "water_index": None}
    result = global_liquid.certify_liquid_tangent_plane(model, [2., 0., 7.])
    assert result["formal_mixing_bound_certified"]
    assert result["lower_bound_rt"] == 0
    assert result["active_component_indices"] == [0, 2]
    assert not result["native_binary_error_bound_certified"]
    assert not result["complete_element_supported_provider_domain"]


def test_missing_component_requires_an_absent_element_for_complete_domain():
    model = {"quadratic_matrix_rt": np.zeros((3, 3)).tolist(),
             "entropy_coefficient": 1., "water_index": None,
             "component_element_matrix": [[1., 0.], [0., 1.], [2., 0.]]}
    result = global_liquid.certify_liquid_tangent_plane(model, [2., 0., 7.])
    assert result["complete_element_supported_provider_domain"]
    model["component_element_matrix"][1] = [3., 0.]
    result = global_liquid.certify_liquid_tangent_plane(model, [2., 0., 7.])
    assert not result["complete_element_supported_provider_domain"]


def test_a_nonconvex_regular_liquid_cannot_receive_a_certificate():
    result = global_liquid.certify_liquid_tangent_plane(parameters(8.), [1., 1.], max_nodes=15)
    assert not result["formal_mixing_bound_certified"]
    assert result["lower_bound_rt"] < -.5
    assert result["best_trial"]["value_rt"] < -.5
    assert result["unresolved_boxes"]


def test_stable_nonconvex_mixing_domain_can_be_bounded_by_subdivision():
    # The binary regular solution is nonconvex near x=1/2, but a sufficiently
    # dilute parent lies below the common tangent everywhere.
    result = global_liquid.certify_liquid_tangent_plane(parameters(3.), [1., 999.], max_nodes=300)
    assert result["formal_mixing_bound_certified"]
    assert result["lower_bound_rt"] >= -1e-8
    assert result["nodes_evaluated"] > 1


def test_interval_log_division_and_negation_enclose_exact_values():
    I = global_liquid._I
    value = (I(Fraction(1, 3)) / I(Fraction(7, 11))).log()
    with localcontext() as context:
        context.prec = 150
        truth = (Decimal(11)/Decimal(21)).ln()
    assert value.lo <= truth <= value.hi
    assert (-value).lo == value.hi.copy_negate()
    assert (-value).hi == value.lo.copy_negate()


def test_interval_ldl_rejects_a_negative_direction_and_rounding_ambiguity():
    I = global_liquid._I
    assert global_liquid._positive_definite([[I(2), I(-1)], [I(-1), I(2)]])
    assert not global_liquid._positive_definite([[I(1), I(2)], [I(2), I(1)]])
    assert not global_liquid._positive_definite([[I(-1, Decimal(1))]])


def test_simplex_tightening_preserves_a_fractional_boundary():
    lower = np.array([.1, .2, 0.])
    upper = np.array([.1, .2, 1.])
    lo, hi, impossible = global_liquid._tighten(lower, upper)
    boundary = 1-Fraction(.1)-Fraction(.2)
    assert not impossible
    assert Fraction(float(lo[2])) <= boundary <= Fraction(float(hi[2]))


def test_invalid_model_is_rejected_before_search():
    with pytest.raises(ValueError):
        global_liquid.certify_liquid_tangent_plane(parameters(), [np.nan, 1.])
    with pytest.raises(ValueError):
        global_liquid.certify_liquid_tangent_plane(parameters(), [-1., 2.])
