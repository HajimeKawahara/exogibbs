import jax.numpy as jnp
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
        self.species = ["H2", "C1O1", "C1H4", "H2O1"]
        self.T_tables, self.mu_tables = self.get_hcosystem_tables()

    def get_hcosystem_tables(self):
        """Load thermochemical data tables for H and H2 from JANAF database.

        Returns:
            Tuple containing:
                - T_tables: Temperature tables for H (K).
                - mu_tables: Chemical potential tables for H2, CO, CH4, H2O (J/mol).
        """
        path = get_data_filepath(DEFAULT_JANAF_GIBBS_MATRICES)
        gibbs_matrices = np.load(path, allow_pickle=True)["arr_0"].item()

        self.T_tables = {}
        self.mu_tables = {}

        kJtoJ = 1000.0  # conversion factor from kJ to J
        for species in self.species:
            T_table = gibbs_matrices[species]["T(K)"].to_numpy()
            mu_table = gibbs_matrices[species]["delta-f G"].to_numpy() * kJtoJ
            self.T_tables[species] = T_table
            self.mu_tables[species] = mu_table
        return self.T_tables, self.mu_tables

    def hv_hco(self, T):
        """Compute chemical potential over RT for HCO system.

        Args:
            T: Temperature in Kelvin.

        Returns:
            Chemical potential divided by RT (dimensionless).
        """
        hv_h2 = interpolate_hvector_one(T, self.T_tables["H2"], self.mu_tables["H2"])
        hv_co = interpolate_hvector_one(
            T, self.T_tables["C1O1"], self.mu_tables["C1O1"]
        )
        hv_ch4 = interpolate_hvector_one(
            T, self.T_tables["C1H4"], self.mu_tables["C1H4"]
        )
        hv_h2o = interpolate_hvector_one(
            T, self.T_tables["H2O1"], self.mu_tables["H2O1"]
        )

        return jnp.array([hv_h2, hv_co, hv_ch4, hv_h2o])

    def deltaT(self, temperature):
        hv_h2, hv_co, hv_ch4, hv_h2o = self.hv_hco(temperature)
        deltaT = - 3.0*hv_h2 - hv_co + hv_ch4 + hv_h2o 
        return deltaT
    
    def equilibrium_constant(self, temperature, normalized_pressure):
        return normalized_pressure**2*jnp.exp(-self.deltaT(temperature))

def function_equilibrium(x_CO, k, bC, bH, bO):
    """Function to compute the equilibrium condition for the HCO system.

    Args:
        x_CO: number of CO over bC, nCO/bC
        k: Equilibrium constant.
        bC: Total number of carbon atoms.
        bH: Total number of hydrogen atoms.
        bO: Total number of oxygen atoms.
    """

    aH = bH / bC
    aO = bO / bC
    x_CH4 = 1.0 - x_CO
    x_H2O = aO - x_CO
    x_H2 = 0.5 * aH - 2.0 * x_CH4 - x_H2O
    x_tot = x_H2 + x_CO + x_CH4 + x_H2O
    return x_CH4 * x_H2O * x_tot**2 - k * x_CO * x_H2**3

