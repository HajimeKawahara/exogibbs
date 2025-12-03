"""Stoichiometry utilities: build a formula matrices from molecular formulae.

This module provides:
- A parser that converts chemical formula strings (e.g., "H2O", "C1O2")
  into dictionaries {element: coefficient}.
- A function to build a formula (stoichiometric) matrix from a DataFrame
  containing molecular names.

Intended to be used as a low-level utility in ExoGibbs, decoupled from
any specific chemical potential source (JANAF, CEA, GGchem, etc.).
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.linalg import qr
from exogibbs.utils.nameparser import parse_formula_with_charge, sanitize_formula


def build_formula_matrix(
    components: Dict[str, Dict[str, int]], elements: List[str]
) -> np.ndarray:
    """
    Generate the formula matrix from the components dict and elements list.

    Args:
        components: mapping ``species -> {element_symbol: count}``
        elements: list of element symbols

    Returns:
        np.ndarray: formula matrix of shape (num_elements, num_species)
    """
    num_species = len(components)
    num_elements = len(elements)
    formula_matrix = np.zeros((num_elements, num_species), dtype=int)
    species_list = list(components.keys())
    for j, spec in enumerate(species_list):
        comp = components[spec]
        for i, el in enumerate(elements):
            if el in comp:
                formula_matrix[i, j] = comp[el]
    return formula_matrix


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



def contract_formula_matrix(formula_matrix, formula_matrix_cond):
    """Contraction of formula matrices

    Args:
        formula_matrix (ndarray or jax.Array): gas formula matrix
        formula_matrix_cond (ndarray or jax.Array): condensates formula matrix

    Raises:
        ValueError: Incompatible shapes for concatenation.

    Returns:
        tuple: contracted gas formula matrix, contracted condensates formula matrix, and mask of independent elements
    """
    try:
        formula_matrix_all = np.concatenate([formula_matrix, formula_matrix_cond], axis=1)
    except ValueError:
        raise ValueError("Incompatible shapes for concatenation.")
    rank_matrix = np.linalg.matrix_rank(formula_matrix_all)    
    nelements = formula_matrix_all.shape[0]

    # If the rank is full, no contraction is needed
    if rank_matrix >= nelements:
        print("No contraction of the system needed.")
        return formula_matrix, formula_matrix_cond, np.ones(nelements, dtype=bool)
    
    _, _, piv = qr(formula_matrix_all.T, pivoting=True)
    indep_element_mask = np.sort(piv[:rank_matrix])
    formula_matrix_gas_eff = formula_matrix[indep_element_mask, :]
    formula_matrix_cond_eff = formula_matrix_cond[indep_element_mask, :]
    print("Contraction of the system performed.")
    return formula_matrix_gas_eff, formula_matrix_cond_eff, indep_element_mask

