# tests/unittests/io/test_formula_matrix_reference_vs_computed.py
import numpy as np
from typing import Optional, Sequence, Tuple, List

from exogibbs.test.reference_data import load_formula_matrix_reference_v3
from exogibbs.io.load_data import load_molname
from exogibbs.thermo.stoichiometry import build_formula_matrix

def _compute_formula_matrix_from_catalog(
    *,
    species_col: str = "JANAF",
    element_order: Optional[Sequence[str]] = None,
    species_names: str = "raw",
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute A by parsing the current species catalog via build_formula_matrix(load_molname()).
    Used for production code; compare against load_formula_matrix_reference_v3() in tests.
    """
    df = load_molname()
    A, elements, species = build_formula_matrix(
        df, species_col=species_col, element_order=element_order, species_names=species_names
    )
    return A, elements, species


def test_reference_matrix_equals_computed_matrix():
    A_ref = load_formula_matrix_reference_v3() 
    A_new, _, _ = _compute_formula_matrix_from_catalog(species_names="raw")

    assert A_new.shape == A_ref.shape, f"shape mismatch: new={A_new.shape}, ref={A_ref.shape}"
    assert np.array_equal(A_new, A_ref), "computed matrix differs from reference TSV"
