"""Global mineral bounds must cover site domains and preserve missing evidence."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/metal_silicate"))
SOLID = importlib.import_module("m2_solid_global")


def case():
    host = {"model_id": "alphamelts_2_3_2_rhyolite_melts_1_0_2_supplied_liquid_v1",
            "status": "ok_supplied_liquid_properties", "T_K": 1., "P_Pa": 1e5,
            "phase_policy": {"oxygen_buffer": "None", "equilibrated": False},
            "component_moles": [1., 1.], "mu_RT": [0., 0.], "oxide_order": ["Fe", "Ni"],
            "oxide_molar_masses_g_mol": [1., 1.],
            "basis": {"common_R_J_mol_K": 1., "component_oxide_matrix": np.eye(2).tolist(),
                      "component_element_matrix": np.eye(2).tolist(), "element_order": ["Fe", "Ni"]}}
    model = {"phase": "alloy-solid", "T_K": 1., "P_Pa": 1e5, "common_R_J_mol_K": 1.,
             "oxide_order": ["Fe", "Ni"], "coordinate_bounds": [[0., 1.]],
             "required_absent_elements": [], "native_endmember_indices": [0, 1],
             "endmember_oxide_moles": np.eye(2).tolist(),
             "endmember_polynomials": [[[1., [0]], [-1., [1]]], [[1., [1]]]],
             "polynomial_rt": [], "nonnegative_polynomials": [], "barrier": None,
             "entropy_sites": [{"coefficient_rt": 1., "polynomial": [[1., [1]]]},
                               {"coefficient_rt": 1., "polynomial": [[1., [0]], [-1., [1]]]}]}
    standards = {"phase": "alloy-solid", "T_K": 1., "P_Pa": 1e5, "mu0_J_mol": [3., 3.],
                 "native_endmember_oxide_mass_g_per_mol": np.eye(2).tolist()}
    return model, standards, host


def test_positive_bound_covers_every_binary_composition_and_ordering_state():
    model, standards, host = case()
    result = SOLID.certify_solid_insertion(model, standards, host, 0.)
    assert result["formal_global_insertion_bound_accepted"]
    assert 2.2 < result["lower_bound_rt_per_formula_unit"] <= 3.-np.log(2.)
    assert not result["native_binary_error_bound_certified"]
    assert not result["global_empirical_stability_certified"]


def test_negative_interior_cannot_be_closed_by_a_finite_budget():
    model, standards, host = case()
    standards["mu0_J_mol"] = [0., 0.]
    result = SOLID.certify_solid_insertion(model, standards, host, 0., max_nodes=31)
    assert not result["formal_global_insertion_bound_accepted"]
    assert result["unresolved_box_count"] > 0
    assert result["lower_bound_rt_per_formula_unit"] <= -np.log(2.)


def test_absent_nickel_proves_a_singleton_domain_without_filling_its_potential():
    model, standards, host = case()
    host["component_moles"][1] = 0.
    host["mu_RT"][1] = None
    standards["mu0_J_mol"] = [0.04, -100.]
    result = SOLID.certify_solid_insertion(model, standards, host, 0.)
    assert result["formal_global_insertion_bound_accepted"]
    assert result["lower_bound_rt_per_formula_unit"] == pytest.approx(0.04)
    assert result["standard_insertion_cost_intervals_rt"][1] is None
    assert result["element_support_restrictions"] == [{"coordinate": 0, "value": 0., "absent_element": "Ni"}]
    host["component_moles"][1] = 1.
    with pytest.raises(ValueError, match="endpoint potential"):
        SOLID.certify_solid_insertion(model, standards, host, 0.)


def test_reference_interval_is_subtracted_before_claiming_a_positive_bound():
    model, standards, host = case()
    model["native_entropy_R_J_mol_K"] = 1.
    model["pure_reference_bounds"] = [{"endmember_index": 0, "enthalpy_lower_J_mol": 0.,
                                      "enthalpy_upper_J_mol": 5., "entropy_site_groups": [[1, 2]]}]
    result = SOLID.certify_solid_insertion(model, standards, host, 0., max_nodes=11)
    assert not result["formal_global_insertion_bound_accepted"]
    assert float(result["pure_reference_intervals_rt"]["0"][1]) >= 5.


def test_stale_standard_state_temperature_is_rejected():
    model, standards, host = case()
    standards["T_K"] = 2.
    with pytest.raises(ValueError, match="ledgers must agree"):
        SOLID.certify_solid_insertion(model, standards, host, 0.)


def test_signed_corner_is_not_omitted_by_nonnegative_endmember_search():
    model, standards, host = case()
    for key in ("component_oxide_matrix", "component_element_matrix"):
        host["basis"][key] = np.eye(3).tolist()
    host["basis"]["element_order"] = ["A", "B", "C"]
    host.update(component_moles=[1., 1., 1.], mu_RT=[0., 0., 0.],
                oxide_order=["A", "B", "C"], oxide_molar_masses_g_mol=[1., 1., 1.])
    model.update(phase="signed", oxide_order=["A", "B", "C"], coordinate_bounds=[[0., 1.]]*2,
                 native_endmember_indices=[0, 1, 2], endmember_oxide_moles=np.eye(3).tolist(),
                 endmember_polynomials=[[[1., [0, 0]], [-1., [1, 0]], [-1., [0, 1]]],
                                        [[1., [1, 0]]], [[1., [0, 1]]]], entropy_sites=[])
    standards.update(phase="signed", mu0_J_mol=[3., 1., 1.],
                     native_endmember_oxide_mass_g_per_mol=np.eye(3).tolist())
    result = SOLID.certify_solid_insertion(model, standards, host, 0., max_nodes=1)
    assert not result["formal_global_insertion_bound_accepted"]
    assert result["lower_bound_rt_per_formula_unit"] <= -1.
