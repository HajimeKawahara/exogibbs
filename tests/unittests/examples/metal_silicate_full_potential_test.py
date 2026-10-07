"""Independent finite-budget acceptance for example-local full potentials."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest


DIRECTORY = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate"
NAMES = ("local", "full_potential")
previous = {name: sys.modules.get(name) for name in NAMES}
sys.path.insert(0, str(DIRECTORY))
try:
    for name in NAMES:
        sys.modules.pop(name, None)
    LOCAL, FULL = [importlib.import_module(name) for name in NAMES]
finally:
    sys.path.pop(0)
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def analytic_case():
    """Construct an exact ideal mechanism root, without a fitted physical claim."""
    phases = {
        "silicate": ["MgO_l", "SiO2_l", "FeO_l", "H2_l", "H2O_l"],
        "metal": ["Fe_m", "Si_m", "O_m", "H_m"],
        "gas": ["H2_g", "He_g", "H2O_g", "SiO_g", "SiH4_g", "H_g", "OH_g", "O2_g"],
    }
    formulas = {
        "MgO": {"Mg": 1, "O": 1}, "SiO2": {"Si": 1, "O": 2},
        "FeO": {"Fe": 1, "O": 1}, "H2": {"H": 2}, "H2O": {"H": 2, "O": 1},
        "Fe": {"Fe": 1}, "Si": {"Si": 1}, "O": {"O": 1}, "H": {"H": 1},
        "He": {"He": 1}, "SiO": {"Si": 1, "O": 1}, "SiH4": {"Si": 1, "H": 4},
        "OH": {"O": 1, "H": 1}, "O2": {"O": 2},
    }
    species = [name for phase in phases.values() for name in phase]
    record = {
        "elements": ["H", "He", "Mg", "Si", "Fe", "O", "C"], "phases": phases,
        "component_formulas": {name: formulas[name.split("_")[0]] for name in species},
        "reactions": [],
    }
    formula = np.array([[record["component_formulas"][s].get(e, 0) for s in species] for e in record["elements"]])
    amounts = np.array([1., .8, .1, .03, .02, .1, .01, .01, .02, .08, .01, .02, .005, .002, .001, .001, .001])
    standards = []
    callbacks = {}
    offset = 0
    for phase, names in phases.items():
        phase_n = amounts[offset:offset + len(names)]
        mu0 = -np.log(phase_n / phase_n.sum())
        callbacks[phase] = FULL.ideal_phase(lambda t, p, v=mu0: v, gas=phase == "gas")
        standards.extend(mu0)
        offset += len(names)
    budgets = formula @ amounts
    problem = LOCAL.build_problem(record, budgets, lambda t, p: np.array(standards), phases=tuple(phases))
    return record, problem, budgets, amounts, callbacks


@pytest.mark.parametrize("scale", [1e-6, 1., 1e6])
def test_finite_h_he_mg_si_fe_o_full_potential_root_and_rescaling(scale):
    _, problem, budgets, expected, callbacks = analytic_case()
    result = FULL.solve_full_potentials(
        problem, 1873., 1., scale * budgets, callbacks,
        initial_component_amounts_mol=scale * expected * np.linspace(.5, 1.7, len(expected)),
    )
    assert result.accepted
    np.testing.assert_allclose(result.component_amounts_mol / scale, expected, rtol=2e-8, atol=1e-11)
    np.testing.assert_allclose(np.asarray(problem.full_formula_matrix) @ result.component_amounts_mol,
                               scale * budgets, rtol=1e-9, atol=0)
    assert result.element_residual[-1] == 0
    np.testing.assert_allclose(result.phase_element_amounts_mol.sum(axis=0), scale * budgets, rtol=1e-9)


def test_absent_metal_is_exactly_removed_and_checked_over_ideal_simplex():
    record, _, budgets, initial, callbacks = analytic_case()
    problem = LOCAL.build_problem(record, budgets, lambda t, p: np.zeros(len(initial)), phases=("silicate", "gas"))
    initial[5:9] = 0
    result = FULL.solve_full_potentials(problem, 1873., 1., budgets,
                                      {p: callbacks[p] for p in problem.phases},
                                      initial_component_amounts_mol=initial)
    assert result.accepted
    assert np.all(result.component_amounts_mol[5:9] == 0)
    metal_formula = np.asarray(problem.full_formula_matrix)[:, 5:9]
    metal_mu0 = -np.log(np.array([.1, .01, .01, .02]) / .14)
    insertion, x = FULL.ideal_solution_insertion_rt(metal_mu0, metal_formula, result.elemental_potentials_rt)
    assert insertion < 0  # Local root closes, but omitted metal must appear.
    assert np.isclose(x.sum(), 1.)
    costs = metal_mu0 - metal_formula.T @ result.elemental_potentials_rt
    np.testing.assert_allclose(costs + np.log(x), insertion, atol=1e-12)


def test_provider_is_recomputed_at_final_amounts_and_failed_roots_are_rejected():
    _, problem, budgets, initial, callbacks = analytic_case()
    seen = []
    original = callbacks["silicate"]
    def phase(t, p, n):
        seen.append(n.copy())
        return original(t, p, n)
    callbacks["silicate"] = phase
    result = FULL.solve_full_potentials(problem, 1873., 1., budgets, callbacks,
                                      initial_component_amounts_mol=initial * 3, max_nfev=1)
    assert not result.accepted
    np.testing.assert_array_equal(seen[-1], result.component_amounts_mol[:5])


def test_inconsistent_phase_energy_is_rejected():
    _, problem, budgets, initial, callbacks = analytic_case()
    callbacks["silicate"] = lambda t, p, n: FULL.PhaseState(np.ones(len(n)), 0.)
    with pytest.raises(ValueError, match="Euler"):
        FULL.solve_full_potentials(problem, 1873., 1., budgets, callbacks,
                                  initial_component_amounts_mol=initial)


def test_pure_phase_insertion_sign():
    assert FULL.pure_phase_insertion_rt(3., np.array([1., 1.]), np.array([1., 1.])) == 1.
    assert FULL.pure_phase_insertion_rt(1., np.array([1., 1.]), np.array([1., 1.])) == -1.
