"""Invented NASA9 functions for offline algorithm tests, not physical data."""


def synthetic_data():
    """Keep source licensing and external downloads outside algorithm tests."""
    compositions = {
        "H": {"H": 1}, "O": {"O": 1}, "P": {"P": 1},
        "H2": {"H": 2}, "PH3": {"P": 1, "H": 3},
        "HPO": {"H": 1, "P": 1, "O": 1},
        "HPO2": {"H": 1, "P": 1, "O": 2},
        "H3PO4": {"H": 3, "P": 1, "O": 4},
        "P4O6": {"P": 4, "O": 6}, "P4O9": {"P": 4, "O": 9},
    }
    records = {}
    for name, composition in compositions.items():
        atom = sum(composition.values()) == 1
        coefficients = [2., .2, 2.5 if atom else 3.5, 2e-4, -1e-8,
                        1e-12, 1e-16, -15000. if name == "H2" else 0., .5]
        records[name] = {
            "phase": "gas", "composition": composition, "hf298_J_mol": 0.,
            "intervals": [{"bounds_K": [lo, hi], "coefficients": coefficients.copy()}
                          for lo, hi in ((200., 1000.), (1000., 6000.))],
        }
    return {
        "source_gas_constant_J_mol_K": 8.314510,
        "records": records,
        "groups": {"base": ["H", "O", "P", "H2", "PH3", "P4O6"],
                   "full_p": ["HPO", "P4O9"], "cns": []},
    }
