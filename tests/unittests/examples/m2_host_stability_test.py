"""One-sided phase insertion controls independent of the native runtime."""

from pathlib import Path
import sys

import numpy as np
import pytest


EXAMPLES = Path(__file__).resolve().parents[3] / "examples/metal_silicate"
sys.path.insert(0, str(EXAMPLES))
from m2_host_stability import assess_host_candidates
from melts_coupled import COMMON_R, PROVIDER_MODEL_ID


def properties(cost=-0.2):
    return {
        "model_id": PROVIDER_MODEL_ID, "status": "ok_supplied_liquid_properties",
        "T_K": 1000., "P_Pa": 1e5, "component_order": ["A", "B"],
        "component_moles": [2., 3.], "mu_RT": [-2., -3.],
        "oxide_molar_masses_g_mol": [1., 1.],
        "basis": {"component_oxide_matrix": [[1., 0.], [0., 1.]],
                  "component_element_matrix": [[1., 0.], [0., 1.]],
                  "element_order": ["A", "B"], "common_R_J_mol_K": COMMON_R},
        "phase_policy": {"oxygen_buffer": "None", "equilibrated": False},
        "saturation": {"candidate_order": ["AB"], "equilibrated": False,
                       "candidates": [{"phase": "AB", "native_affinity_J": cost * COMMON_R * 1000.,
                                       "status": "ok_candidate_properties", "reason": None,
                                       "oxide_mass_g": [1., 1.],
                                       "gibbs_J": (-5. + cost) * COMMON_R * 1000.}]},
    }


def test_negative_feasible_insertion_rejects_but_nonnegative_is_not_certificate():
    rejected = assess_host_candidates(properties(), 0.)
    row = rejected["candidates"][0]
    assert rejected["status"] == "rejected_by_feasible_trial"
    assert row["maximum_feasible_trial_scale"] == 2.
    assert row["insertion_rt_per_mol_atoms"] == pytest.approx(-0.1)
    unresolved = assess_host_candidates(properties(0.2), 0.)
    assert unresolved["status"] == "unresolved"
    assert not unresolved["global_stability_certified"]
    assert not row["minimum_certified"]


def test_h2_dilution_is_the_derivative_of_the_same_full_scalar():
    native = properties()
    h2 = 5.
    row = assess_host_candidates(native, h2)["candidates"][0]
    assert row["h2_dilution_correction_gibbs_rt"] == pytest.approx(2. * np.log(2.))
    # Directly differentiate the extensive ideal host/H2 mixing term while
    # consuming one mole of each host component to insert the candidate.
    def total(step):
        nhost = 5. - 2. * step
        dilution = nhost * np.log(nhost / (nhost + h2)) + h2 * np.log(h2 / (nhost + h2))
        return -0.2 * step + dilution
    step = 1e-5
    derivative = (total(step) - total(-step)) / (2 * step)
    assert row["insertion_rt_per_mol_atoms"] == pytest.approx(derivative / 2., abs=1e-10)
    assert row["status"] == "nonnegative_insertion_trial"


def test_insertion_is_invariant_to_element_gauge_and_amount_scale():
    reference = assess_host_candidates(properties(), 1.)["candidates"][0]
    shifted = properties()
    gauge = np.array([70., -20.])
    shifted["mu_RT"] = (np.asarray(shifted["mu_RT"]) + gauge).tolist()
    shifted["saturation"]["candidates"][0]["gibbs_J"] += gauge.sum() * COMMON_R * 1000.
    shifted["component_moles"] = [20., 30.]
    row = assess_host_candidates(shifted, 10.)["candidates"][0]
    assert row["insertion_rt_per_mol_atoms"] == pytest.approx(reference["insertion_rt_per_mol_atoms"], abs=1e-13)
    assert row["maximum_feasible_trial_scale"] == pytest.approx(10. * reference["maximum_feasible_trial_scale"])


def test_unavailable_and_infeasible_candidates_remain_unresolved():
    native = properties()
    native["component_moles"][0] = 0.
    native["mu_RT"][0] = None
    assert "absent" in assess_host_candidates(native, 0.)["candidates"][0]["reason"]
    native = properties()
    native["saturation"]["candidates"][0].update(status="unavailable", reason="No native estimate.")
    result = assess_host_candidates(native, 0.)
    assert result["unresolved_trial_phases"] == ["AB"]
    assert result["negative_trial_phases"] == []


def test_tiny_consumption_of_absent_component_cannot_create_a_false_certificate():
    native = properties(-100.)
    native["component_moles"][0] = 0.
    native["mu_RT"][0] = None
    native["saturation"]["candidates"][0]["oxide_mass_g"][0] = 1e-15
    result = assess_host_candidates(native, 0.)
    assert result["status"] == "unresolved"
    assert "absent" in result["candidates"][0]["reason"]


@pytest.mark.parametrize("change", ["buffer", "equilibrated", "reordered", "duplicate", "gas_constant"])
def test_changed_state_or_incomplete_catalog_is_rejected(change):
    native = properties()
    if change == "buffer":
        native["phase_policy"]["oxygen_buffer"] = "IW"
    elif change == "equilibrated":
        native["saturation"]["equilibrated"] = True
    elif change == "reordered":
        native["saturation"]["candidate_order"] = []
    elif change == "duplicate":
        native["saturation"]["candidate_order"] *= 2
        native["saturation"]["candidates"] *= 2
    else:
        native["basis"]["common_R_J_mol_K"] = 8.3143
    with pytest.raises(ValueError):
        assess_host_candidates(native, 0.)
