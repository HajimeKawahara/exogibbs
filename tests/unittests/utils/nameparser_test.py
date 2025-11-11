from exogibbs.utils.nameparser import parse_simple_formula

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