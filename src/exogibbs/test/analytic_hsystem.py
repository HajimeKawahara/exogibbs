from jax import grad
from jax import vmap
import jax.numpy as jnp

from exogibbs.thermo.gibbs import interpolate_hvector_one
from exogibbs.io.load_data import get_data_filepath
from exogibbs.presets.ykb4 import JANAF_GIBBS_MATRICES_YKB4
import numpy as np


class HSystem:
    """
    A class to represent the H system (2H <-> H2) for chemical equilibrium calculations.
    It provides analytical methods to compute the number densities and volume mixing ratios
    of H and H2 based on temperature and pressure. 
    Because it's analytic, it is useful for testing Gibbs minimization etc.
    """

    def __init__(self):
        self.T_h_table, self.mu_h_table, self.T_h2_table, self.mu_h2_table = (
            self.get_hsystem_tables()
        )

    def get_hsystem_tables(self):
        """Load thermochemical data tables for H and H2 from JANAF database.
        
        Returns:
            Tuple containing:
                - T_h_table: Temperature table for H (K).
                - mu_h_table: Chemical potential table for H (J/mol).
                - T_h2_table: Temperature table for H2 (K).
                - mu_h2_table: Chemical potential table for H2 (J/mol).
        """
        path = get_data_filepath(JANAF_GIBBS_MATRICES_YKB4)
        gibbs_matrices = np.load(path, allow_pickle=True)["arr_0"].item()

        kJtoJ = 1000.0  # conversion factor from kJ to J
        T_h_table = gibbs_matrices["H1"]["T(K)"].to_numpy()
        mu_h_table = gibbs_matrices["H1"]["delta-f G"].to_numpy() * kJtoJ
        T_h2_table = gibbs_matrices["H2"]["T(K)"].to_numpy()
        mu_h2_table = gibbs_matrices["H2"]["delta-f G"].to_numpy() * kJtoJ
        return T_h_table, mu_h_table, T_h2_table, mu_h2_table

    def hv_h(self, T):
        """Compute chemical potential over RT for atomic hydrogen.
        
        Args:
            T: Temperature in Kelvin.
            
        Returns:
            Chemical potential divided by RT (dimensionless).
        """
        return interpolate_hvector_one(T, self.T_h_table, self.mu_h_table)

    def hv_h2(self, T):
        """Compute chemical potential over RT for molecular hydrogen.
        
        Args:
            T: Temperature in Kelvin.
            
        Returns:
            Chemical potential divided by RT (dimensionless).
        """
        return interpolate_hvector_one(T, self.T_h2_table, self.mu_h2_table)

    def compute_k(self, ln_normalized_pressure, T):
        """Compute equilibrium constant for H2 dissociation reaction.
        
        Args:
            ln_normalized_pressure: natural log pressure normalized by reference pressure (P/Pref).
            T: Temperature in Kelvin.
            
        Returns:
            Equilibrium constant K = exp(-Δμ/RT) * P/P_ref.
        """
        delta_h = self.hv_h2(T) - 2.0 * self.hv_h(T)
        return jnp.exp(-delta_h) * jnp.exp(ln_normalized_pressure) 

    def nh(self, k):
        """Compute number density of atomic hydrogen.
        
        Args:
            k: Equilibrium constant.
            
        Returns:
            Number density of H normalized by total hydrogen nuclei.
        """
        return 1.0 / jnp.sqrt(4.0 * k + 1.0)

    def nh2(self, k):
        """Compute number density of molecular hydrogen.
        
        Args:
            k: Equilibrium constant.
            
        Returns:
            Number density of H2 normalized by total hydrogen nuclei.
        """
        return 0.5 * (1.0 - self.nh(k))

    def ntotal(self, k):
        """Compute total number density of hydrogen species.
        
        Args:
            k: Equilibrium constant.
            
        Returns:
            Total number density (n_H + n_H2).
        """
        return self.nh(k) + self.nh2(k)

    def vmr_h(self, k):
        """Compute volume mixing ratio of atomic hydrogen.
        
        Args:
            k: Equilibrium constant.
            
        Returns:
            Volume mixing ratio of H (n_H / n_total).
        """
        return self.nh(k) / self.ntotal(k)

    def vmr_h2(self, k):
        """Compute volume mixing ratio of molecular hydrogen.
        
        Args:
            k: Equilibrium constant.
            
        Returns:
            Volume mixing ratio of H2 (n_H2 / n_total).
        """
        return self.nh2(k) / self.ntotal(k)

    def dot_hv_h(self, T):
        """Compute temperature derivative of H chemical potential.
        
        Args:
            T: Temperature in Kelvin.
            
        Returns:
            d(μ_H/RT)/dT.
        """
        return grad(self.hv_h)(T)

    def dot_hv_h2(self, T):
        """Compute temperature derivative of H2 chemical potential.
        
        Args:
            T: Temperature in Kelvin.
            
        Returns:
            d(μ_H2/RT)/dT.
        """
        return grad(self.hv_h2)(T)

    def delta(self, T):
        """Compute chemical potential difference for H2 dissociation.
        
        Args:
            T: Temperature in Kelvin.
            
        Returns:
            Δμ/RT = 2μ_H/RT - μ_H2/RT for reaction H2 ⇌ 2H.
        """
        return 2.0 * self.hv_h(T) - self.hv_h2(T)

    def delta_dT(self, Tarr):
        """Compute temperature derivative of chemical potential difference.
        
        Args:
            Tarr: Array of temperatures in Kelvin.
            
        Returns:
            Array of d(Δμ/RT)/dT values.
        """
        return vmap(grad(self.delta), in_axes=0)(Tarr)

    def ln_nH_dT(self, Tarr, ln_normalized_pressure):
        """Compute temperature derivative of log(n_H).
        
        Args:
            Tarr: Array of temperatures in Kelvin.
            ln_normalized_pressure: natural log pressure normalized by reference pressure (P/Pref).
            
        Returns:
            Array of d(ln n_H)/dT values.
        """
        k = self.compute_k(ln_normalized_pressure, Tarr)
        return -2.0 * self.nh2(k) * self.ntotal(k) * self.delta_dT(Tarr)

    def ln_nH2_dT(self, Tarr, ln_normalized_pressure):
        """Compute temperature derivative of log(n_H2).
        
        Args:
            Tarr: Array of temperatures in Kelvin.
            ln_normalized_pressure: natural log pressure normalized by reference pressure (P/Pref).
            
        Returns:
            Array of d(ln n_H2)/dT values.
        """
        k = self.compute_k(ln_normalized_pressure, Tarr)
        return self.nh(k) * self.ntotal(k) * self.delta_dT(Tarr)

    def ln_nH_dlogp(self, Tarr, ln_normalized_pressure):
        """Compute log pressure derivative of log(n_H).
        
        Args:
            Tarr: Array of temperatures in Kelvin.
            ln_normalized_pressure: natural log pressure normalized by reference pressure (P/Pref).
            
        Returns:
            Array of d(ln n_H)/dT values.
        """
        k = self.compute_k(ln_normalized_pressure, Tarr)
        return -2.0 * self.nh2(k) * self.ntotal(k)

    def ln_nH2_dlogp(self, Tarr, ln_normalized_pressure):
        """Compute pressure derivative of log(n_H2).
        
        Args:
            Tarr: Array of temperatures in Kelvin.
            ln_normalized_pressure: natural log pressure normalized by reference pressure (P/Pref).
            
        Returns:
            Array of d(ln n_H2)/dT values.
        """
        k = self.compute_k(ln_normalized_pressure, Tarr)
        return self.nh(k) * self.ntotal(k)

    def ln_nH_dbH(self, bH):
        """Compute element abundance derivative of log(n_H).
        
        Returns:
            d(ln n_H)/dB_H = 1/bH.
        """

        return 1.0/bH
    
    def ln_nH2_dbH(self, bH):
        """Compute element abundance derivative of log(n_H2).
        
        Returns:
            d(ln n_H2)/dB_H = 1/bH.
        """

        return 1.0/bH