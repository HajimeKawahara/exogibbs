"""
Stoichiometry utilities: build a formula matrix from molecular formulae.

This module provides:
- A parser that converts chemical formula strings (e.g., "H2O", "C1O2")
  into dictionaries {element: coefficient}.
- A function to build a formula (stoichiometric) matrix from a DataFrame
  containing molecular names.

Intended to be used as a low-level utility in ExoGibbs, decoupled from
any specific chemical potential source (JANAF, CEA, GGchem, etc.).
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

# Regex to match an element (capital letter + optional lowercase letter)
# followed by an optional integer coefficient
# Add near the top with other regexes

# Matches one or more trailing parenthetical annotation groups, e.g. "(CNN)", "(g)", "(NCN)"
_ELNUM = re.compile(r"([A-Z][a-z]?)(\d*)")
_PAREN_ANNOT_TAIL = re.compile(r"(?:\([A-Za-z0-9+\-]*\))+$")
# Add this near the top with the other imports/regexes
_ELECTRON_BASE = re.compile(r"^[eE](\d*)$")  # matches 'e', 'e1', 'E2', etc.
_CHARGE = re.compile(r"^([A-Za-z0-9*]+)([+-]\d*)$")  # unchanged


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


def build_formula_matrix(
    df_molname: pd.DataFrame,
    species_col: str = "JANAF",
    *,
    element_order: Optional[Sequence[str]] = None,
    sanitize: bool = True,
    species_names: str = "raw",  # NEW: "raw" | "clean" | "both"
) -> Union[
    Tuple[np.ndarray, List[str], List[str]],
    Tuple[np.ndarray, List[str], List[str], List[str]]
]:
    """
    Build a stoichiometric matrix A (elements x species) from df_molname.

    Args:
        species_names : {"raw", "clean", "both"}, default "raw"
        Which species labels to return:
        - "raw": return original strings (e.g., "C1N2(CNN)")
        - "clean": return sanitized/normalized species (e.g., "C1N2")
        - "both": return both (species_raw, species_clean)

    Returns:
        If species_names in {"raw", "clean"}: A, elements, species
        If species_names == "both": A, elements, species_raw, species_clean
    """
    raw_species = df_molname[species_col].astype(str).tolist()

    # Clean names for parsing / normalization (drop leading markers & trailing annotations)
    species_clean = [sanitize_formula(s) if sanitize else s for s in raw_species]

    # Parse using the clean base (never use raw for parsing)
    parsed_list = [parse_formula_with_charge(s) for s in species_clean]

    all_elems = set().union(*parsed_list) if parsed_list else set()
    elements = sorted(all_elems) if element_order is None else list(element_order)

    A = np.zeros((len(elements), len(species_clean)), dtype=np.int64)
    elem_index = {e: i for i, e in enumerate(elements)}

    for j, d in enumerate(parsed_list):
        for e, v in d.items():
            if e not in elem_index:
                raise ValueError(
                    f"Unexpected element '{e}' in species '{species_clean[j]}'."
                )
            A[elem_index[e], j] = int(v)

    # Choose which species names to return (labels only; matrix columns already built)
    if species_names == "raw":
        return A, elements, raw_species
    elif species_names == "clean":
        return A, elements, species_clean
    elif species_names == "both":
        return A, elements, raw_species, species_clean
    else:
        raise ValueError("species_names must be 'raw', 'clean', or 'both'.")


