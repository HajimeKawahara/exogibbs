from jax import grad
from jax import vmap
import jax.numpy as jnp
from exogibbs.utils.constants import R_gas_constant_si
from exogibbs.equilibrium.gibbs import interpolate_hvector_one
from exogibbs.io.load_data import get_data_filepath
from exogibbs.io.load_data import DEFAULT_JANAF_GIBBS_MATRICES
import numpy as np


class HCOSystem:
    """
    A class to represent the HCO system (CO + 3H2 <-> CH4 + H2O) for chemical equilibrium calculations.
    It provides analytical methods to compute the number densities
    """

    def __init__(self):
        self.species = ["H2", "CO", "CH4", "H2O"]
        self.T_tables, self.mu_tables = (
            self.get_hcosystem_tables()
        )

    def get_hcosystem_tables(self):
        """Load thermochemical data tables for H and H2 from JANAF database.
        
        Returns:
            Tuple containing:
                - T_tables: Temperature tables for H (K).
                - mu_tables: Chemical potential tables for CO, H2, CH4, H2O (J/mol).
        """
        path = get_data_filepath(DEFAULT_JANAF_GIBBS_MATRICES)
        gibbs_matrices = np.load(path, allow_pickle=True)["arr_0"].item()

        kJtoJ = 1000.0  # conversion factor from kJ to J
        for species in self.species:
            T_table = gibbs_matrices[species]["T(K)"].to_numpy()
            mu_table = gibbs_matrices[species]["delta-f G"].to_numpy() * kJtoJ
            self.T_tables[species] = T_table
            self.mu_tables[species] = mu_table
        return self.T_tables, self.mu_tables

    def DeltaT(self, T):
        
        
        return 