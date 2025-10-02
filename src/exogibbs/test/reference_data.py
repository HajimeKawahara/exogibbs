from __future__ import annotations

import pandas as pd
import numpy as np
from exogibbs.io.load_data import get_data_filepath

FORMULA_MATRIX_V3 = "test/matrix_v3.dat"

def load_formula_matrix_reference_v3() -> np.ndarray:
    """
    Return the *reference* stoichiometric matrix A (elements x species) by reading
    the pinned TSV file used historically in ExoGibbs (FORMULA_MATRIX_V3).

    This function is intentionally separate from the computed path and must remain
    stable for regression testing. Do not replace it with a computed implementation.
    """
    fullpath = get_data_filepath(FORMULA_MATRIX_V3)
    df = pd.read_csv(fullpath, sep="\t", header=None, dtype=int)
    fm_np = np.array(df).T  # legacy orientation: transpose to (elements x species)
    return fm_np