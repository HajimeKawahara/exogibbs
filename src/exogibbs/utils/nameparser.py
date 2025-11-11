#
# Utility functions for parsing chemical species names and formulas.
#


import re
from typing import Dict
# Matches one or more trailing parenthetical annotation groups, e.g. "(CNN)", "(g)", "(NCN)"
_PAREN_ANNOT_TAIL = re.compile(r"(?:\([A-Za-z0-9+\-]*\))+$")
# Add this near the top with the other imports/regexes
_ELECTRON_BASE = re.compile(r"^[eE](\d*)$")  # matches 'e', 'e1', 'E2', etc.
_CHARGE = re.compile(r"^([A-Za-z0-9*]+)([+-]\d*)$")  # unchanged
_ELNUM = re.compile(r"([A-Z][a-z]?)(\d*)")



def sanitize_formula(s: str) -> str:
    """
    Remove leading database markers and trailing parenthetical annotations.

    Examples:
        "*CO2"         -> "CO2"
        "C1N2(CNN)"    -> "C1N2"   # isomer/state tag removed
        "C1N2(NCN)"    -> "C1N2"
        "H2O(g)"       -> "H2O"
        "e1-"          -> "e1-"    # charge kept; handled by parse_formula_with_charge
    """
    # Strip leading DB markers (CEA-style phase flags, etc.)
    s = s.lstrip("*^#@~ ")

    # Remove one or more trailing parenthetical annotation groups.
    # These are non-stoichiometric tags that appear at the end only.
    s = _PAREN_ANNOT_TAIL.sub("", s)

    return s


def parse_formula_with_charge(formula: str) -> Dict[str, int]:
    """
    Parse a chemical formula possibly carrying a net ionic charge.

    Supports:
    - Normal neutral molecules: "H2O", "CO2"
    - Ions with charge suffix: "H3O+", "SO4-2", "Na+"
    - JANAF-style electron species: "e-", "e1-", "e", "e1" (all treated as electrons) -> represented as {"e-": n_electrons}

    NOTE: We do NOT add an extra charge-derived electron count for pure-electron species,
          to avoid double-counting (the base already *is* electrons).
    """
    if not formula:
        raise ValueError("Empty formula string.")

    # Split (base, charge) if a trailing charge suffix exists
    m = _CHARGE.match(formula)
    if m:
        base, charge_str = m.groups()
        # interpret "+", "-" with optional digits; default magnitude is 1
        if len(charge_str) > 1:
            magnitude = int(charge_str[1:])
        else:
            magnitude = 1
        charge = magnitude if charge_str[0] == "+" else -magnitude
    else:
        base, charge = formula, 0

    base = sanitize_formula(base)

    # ---- SPECIAL CASE: pure-electron species like "e", "e1", "E2" ----
    me = _ELECTRON_BASE.fullmatch(base)
    if me:
        n = int(me.group(1)) if me.group(1) else 1  # default 1
        element_count_dict: Dict[str, int] = {"e-": n}

        # Optional consistency check: if a charge suffix exists, it should equal -n
        # e.g., "e1-" => charge = -1 matches n=1
        # We ignore mismatches rather than error, but you can raise if you prefer.
        # if m and charge != -n:
        #     raise ValueError(f"Inconsistent electron charge in '{formula}': base implies {n} e-, "
        #                      f"but suffix implies total charge {charge}.")
        return element_count_dict

    # ---- Normal molecules/ions path ----
    element_count_dict = parse_simple_formula(base)  # may raise on unsupported tokens (parentheses etc.)

    if charge != 0:
        # Positive charge => missing electrons; negative charge => extra electrons
        element_count_dict["e-"] = element_count_dict.get("e-", 0) - charge

    return element_count_dict


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


