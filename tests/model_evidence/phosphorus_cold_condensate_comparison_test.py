"""Independent equilibrium checks for the sampled native-phase comparison."""

import importlib.util
from pathlib import Path

import jax
import numpy as np


path = Path(__file__).resolve().parents[2] / "mceg/reviews/phosphorus_cold_condensate_comparison_20261031.py"
spec = importlib.util.spec_from_file_location("cold_condensate_comparison", path)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
jax.config.update("jax_enable_x64", True)


def test_primitive_audit_detects_trace_budget_and_supersaturation():
    # One element: A(g) at p=1 bar coexists with A(s), h_s=0.
    # A2(g) is below float64 underflow but its mass action remains checkable.
    a = np.array([[1., 2.]])
    ac = np.array([[1., 1.]])
    ln_n = np.array([np.log(0.4), np.log(0.4) - 1000.])
    checks = review.primitive_checks(ln_n, np.array([0.6, 0.]), a, ac,
                                     np.array([1.]), np.array([0., -1000.]),
                                     np.array([0., 1.]), 1., [0])
    assert checks["element_relative_residual"] < 1e-14
    assert checks["mass_action_max_abs_ln"] < 1e-12
    assert checks["active_affinity_max_abs_ln"] == 0.
    assert checks["inactive_affinity_positive_max_ln"] == 0.
    unstable = review.primitive_checks(ln_n, np.array([0.6, 0.]), a, ac,
                                       np.array([1.]), np.array([0., -1000.]),
                                       np.array([0., -1.]), 1., [0])
    assert unstable["inactive_affinity_positive_max_ln"] == 1.
    # Tiny inventories must receive the same relative accuracy as abundant ones.
    trace = review.primitive_checks(ln_n + np.log(1e-10), np.array([0.5e-10, 0.]), a, ac,
                                    np.array([1e-10]), np.array([0., -1000.]),
                                    np.array([0., 1.]), 1., [0])
    assert np.isclose(trace["element_relative_residual"], 0.1)


def test_sampled_potential_refuses_unsampled_temperatures():
    data = {"temperature_K": [400., 200.], "gas_lnK_bar": [[0., 4.], [0., 8.]],
            "gas_species": ["A", "A2"], "gas_formula_matrix": [[1., 2.]],
            "elements": ["A"], "budget": [1.]}
    setup = review.sampled_setup(data, "gas")
    np.testing.assert_array_equal(setup.hvector_func(200.), [0., -8.])
    assert np.isnan(setup.hvector_func(300.)).all()
    assert np.isnan(setup.hvector_func(100.)).all()
