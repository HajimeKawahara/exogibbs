import re
from typing import Dict

_ELNUM = re.compile(r"([A-Z][a-z]?)(\d*)")

def parse_simple_formula(formula: str) -> Dict[str, int]:
    """
    Parse a simple chemical formula string into a dict of element counts.

    Examples:
        "CH4"   -> {"C": 1, "H": 4}
        "C1O2"  -> {"C": 1, "O": 2}
        "H2O"   -> {"H": 2, "O": 1}

    Notes:
        - Numbers are optional; default is 1.
        - Does not currently handle parentheses, charges, or hydrates.
    """
    if not formula:
        raise ValueError("Empty formula string.")

    pos = 0
    element_counts_dict: Dict[str, int] = {}
    for m in _ELNUM.finditer(formula):
        if m.start() != pos:
            unknown = formula[pos : m.start()]
            raise ValueError(f"Unsupported token '{unknown}' in formula '{formula}'.")
        elem, num = m.groups()
        coeff = int(num) if num else 1
        element_counts_dict[elem] = element_counts_dict.get(elem, 0) + coeff
        pos = m.end()

    if pos != len(formula):
        unknown = formula[pos:]
        raise ValueError(
            f"Unsupported trailing token '{unknown}' in formula '{formula}'."
        )
    return element_counts_dict


