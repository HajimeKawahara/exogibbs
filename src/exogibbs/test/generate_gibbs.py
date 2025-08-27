"""Generate gibbs_matrices for JANAF data."""

def generate_gibbs_matrices(path_JANAF_data = "/home/kawahara/thermochemical_equilibrium/Equilibrium/JANAF"):
    import numpy as np
    from exogibbs.io.load_data import load_JANAF_molecules
    from exogibbs.io.load_data import load_molname
    df_molname = load_molname()
    gibbs_matrices = load_JANAF_molecules(df_molname, path_JANAF_data)
    np.savez("gibbs_matrices.npz", gibbs_matrices)
        
if __name__ == "__main__":
    generate_gibbs_matrices()
    print("move gibbs_matrices.npz to the data directory")