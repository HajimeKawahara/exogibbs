import sympy as sp


def analyze_fmatrix(fm_sp):
    fm_sp_rref, pivots = fm_sp.rref()
    all_cols = list(range(fm_sp.shape[1]))
    free_cols = [col for col in all_cols if col not in pivots]
    new_order = list(pivots) + free_cols
    fm_sp_rref_permuted = fm_sp_rref[:, new_order]
    
    sp.pprint(fm_sp_rref_permuted)

    return fm_sp_rref_permuted

if __name__ == "__main__":
    from exogibbs.io.load_data import load_formula_matrix
    import numpy as np
    fm_np = load_formula_matrix()
    fm_sp = sp.Matrix(fm_np)
    analyze_fmatrix(fm_sp)