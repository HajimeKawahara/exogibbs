from exogibbs.utils.nameparser import parse_simple_formula
from exogibbs.utils.nameparser import sanitize_formula
from exogibbs.utils.nameparser import parse_formula_with_charge
from exogibbs.utils.nameparser import set_elements_from_species

def test_set_elements_from_species():
    species = {
        "H2O": {"H": 2, "O": 1},
        "CO2": {"C": 1, "O": 2},
        "CH4": {"C": 1, "H": 4},
        "e-": {"e-": 1}
    }
    elements = set_elements_from_species(species)
    assert elements == {'H', 'O', 'C', 'e-'}, "Element set does not match expected output."


def test_parse_simple_formula():
    assert parse_simple_formula("CH4") == {"C": 1, "H": 4}
    assert parse_simple_formula("C1O2") == {"C": 1, "O": 2}
    assert parse_simple_formula("H2O") == {"H": 2, "O": 1}
    assert parse_simple_formula("NaCl") == {"Na": 1, "Cl": 1}
    assert parse_simple_formula("C6H12O6") == {"C": 6, "H": 12, "O": 6}

    try:
        parse_simple_formula("")
    except ValueError as e:
        assert str(e) == "Empty formula string."

    try:
        parse_simple_formula("C2H5(OH)")
    except ValueError as e:
        assert "Unsupported token" in str(e)

    try:
        parse_simple_formula("H2O3X")
    except ValueError as e:
        assert "Unsupported trailing token" in str(e)


def test_sanitize_formula():
    assert sanitize_formula("C1N2(CNN)") == "C1N2"
    assert sanitize_formula("H2O(g)") == "H2O"
    assert sanitize_formula("Fe2O3(s)") == "Fe2O3"
    assert sanitize_formula("NaCl") == "NaCl"
    assert sanitize_formula("e-") == "e-"

def test_parse_formula_with_charge():
    assert parse_formula_with_charge("CH4") == {"C": 1, "H": 4}
    assert parse_formula_with_charge("C1O2+") == {"C": 1, "O": 2, "e-": -1}
    assert parse_formula_with_charge("H2O-2") == {"H": 2, "O": 1, "e-": 2}
    assert parse_formula_with_charge("e") == {"e-": 1}
    assert parse_formula_with_charge("e3-") == {"e-": 3}

    try:
        parse_formula_with_charge("")
    except ValueError as e:
        assert str(e) == "Empty formula string."

    try:
        parse_formula_with_charge("C2H5(OH)")
    except ValueError as e:
        assert "Unsupported token" in str(e)

    try:
        parse_formula_with_charge("H2O3X")
    except ValueError as e:
        assert "Unsupported trailing token" in str(e)