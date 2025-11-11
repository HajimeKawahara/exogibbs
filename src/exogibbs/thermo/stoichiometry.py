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

from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from exogibbs.utils.nameparser import parse_formula_with_charge, sanitize_formula

def build_formula_matrix_from_JANAF(
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

    all_elements = set().union(*parsed_list) if parsed_list else set()
    elements = sorted(all_elements) if element_order is None else list(element_order)

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



