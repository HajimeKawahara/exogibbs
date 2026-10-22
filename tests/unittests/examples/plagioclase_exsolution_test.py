"""Independent coexistence checks for the plagioclase teaching model."""

from pathlib import Path
import runpy

import numpy as np
import pytest
from scipy.optimize import brentq


@pytest.fixture(scope="module")
def example():
    path = Path(__file__).resolve().parents[3] / "examples/plagioclase_exsolution.py"
    namespace = runpy.run_path(str(path))
    return namespace, namespace["run_example"]()


@pytest.mark.parametrize("name", ["hot_single", "cold_single", "cold_split"])
def test_conserves_endmembers_and_all_five_elements(example, name):
    _, results = example
    result = results[name]
    amounts = np.asarray(result.amounts_mol).reshape(2, 2)
    np.testing.assert_allclose(amounts.sum(axis=0), [0.4, 0.6], atol=1.0e-10)
    np.testing.assert_allclose(
        np.asarray(result.phase_element_amounts_mol).sum(axis=0),
        [0.4, 0.6, 1.6, 2.4, 8.0], atol=1.0e-10,
    )
    np.testing.assert_allclose(result.conservation_residual_mol, 0.0, atol=1.0e-10)
    assert result.stability_status == "unchecked"


def test_cold_coexistence_matches_common_tangent_and_lever_rule(example):
    namespace, results = example
    rt = namespace["GAS_CONSTANT_J_MOL_K"] * namespace["COLD_TEMPERATURE_K"]
    w = namespace["INTERACTION_J_MOL"]
    # The noncentral root of g'(x)=0 is the left common-tangent endpoint
    # for this symmetric regular solution. It does not use the Gibbs callback.
    x_alpha = brentq(
        lambda x: rt * np.log(x / (1.0 - x)) + w * (1.0 - 2.0 * x),
        1.0e-8, 0.499999,
    )
    expected_x = np.array([x_alpha, 1.0 - x_alpha])
    beta_fraction = (0.6 - x_alpha) / (1.0 - 2.0 * x_alpha)
    split = results["cold_split"]
    amounts = np.asarray(split.amounts_mol).reshape(2, 2)
    # Alpha/beta labels are interchangeable; compare in composition order.
    order = np.argsort(amounts[:, 1] / amounts.sum(axis=1))
    phase_amounts = amounts.sum(axis=1)[order]
    np.testing.assert_allclose(
        amounts[order, 1] / phase_amounts, expected_x, rtol=1.0e-7,
    )
    np.testing.assert_allclose(
        phase_amounts, [1.0 - beta_fraction, beta_fraction], rtol=1.0e-7,
    )
    expected_mu = np.column_stack((
        rt * np.log(1.0 - expected_x) + w * expected_x**2,
        rt * np.log(expected_x) + w * (1.0 - expected_x)**2,
    ))
    mu = np.asarray(split.chemical_potentials_j_mol).reshape(2, 2)[order]
    np.testing.assert_allclose(mu, expected_mu, rtol=1.0e-7)
    np.testing.assert_allclose(mu[0], mu[1], rtol=1.0e-7)
    expected_g = rt * (
        (1.0 - x_alpha) * np.log(1.0 - x_alpha) + x_alpha * np.log(x_alpha)
    ) + w * x_alpha * (1.0 - x_alpha)
    np.testing.assert_allclose(split.gibbs_j, expected_g, rtol=1.0e-9)
    assert split.gibbs_j < results["cold_single"].gibbs_j


def test_hot_solution_is_strictly_convex_and_keeps_bulk_composition(example):
    namespace, results = example
    rt = namespace["GAS_CONSTANT_J_MOL_K"] * namespace["HOT_TEMPERATURE_K"]
    w = namespace["INTERACTION_J_MOL"]
    # min g''(x) = 4RT - 2W > 0 rules out separation at this temperature.
    assert 4.0 * rt - 2.0 * w > 0.0
    np.testing.assert_allclose(results["hot_single"].amounts_mol, [0.4, 0.6, 0.0, 0.0])
    expected_g = rt * (0.4 * np.log(0.4) + 0.6 * np.log(0.6)) + w * 0.4 * 0.6
    np.testing.assert_allclose(results["hot_single"].gibbs_j, expected_g, rtol=1.0e-10)
