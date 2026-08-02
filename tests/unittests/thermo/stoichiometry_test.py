from exogibbs.thermo.stoichiometry import build_formula_matrix
import numpy as np

def test_build_formula_matrix():
    components = {
        "H2O": {"H": 2, "O": 1},
        "CO2": {"C": 1, "O": 2},
        "CH4": {"C": 1, "H": 4},
        "e-": {"e-": 1}
    }
    elements = ["H", "O", "C", "e-"]
    expected_matrix = np.array([
        [2, 0, 4, 0],  # H
        [1, 2, 0, 0],  # O
        [0, 1, 1, 0],  # C
        [0, 0, 0, 1]   # e-
    ])
    formula_matrix = build_formula_matrix(components, elements)
    assert np.array_equal(formula_matrix, expected_matrix), "Formula matrix does not match expected output."