from jax.lax import while_loop
from jax import grad
from jax import custom_vjp
import jax.numpy as jnp
from sympy import root
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
        return normalized_pressure**2*jnp.exp(self.deltaT(temperature))

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
    x_tot = 0.5 * aH + 2.0 * x_CO - 1.0
    return x_CH4 * x_H2O * x_tot**2 - k * x_CO * x_H2**3


def newton_scalar(init_x, *, k, bC, bH, bO, tol=1e-12, maxiter=500):
    def cond_fn(state):
        x, fval, it = state
        return jnp.logical_and(jnp.abs(fval) > tol, it < maxiter)

    def body_fn(state):
        x, fval, it = state
        dfdx = grad(function_equilibrium, 0)(x, k, bC, bH, bO)
        dfdx = jnp.where(dfdx == 0.0, 1e-14, dfdx)
        x_new = x - fval / dfdx
        x_new = jnp.clip(x_new, 0.0, 1.0)
        return (x_new, function_equilibrium(x_new, k, bC, bH, bO), it + 1)

    x0 = jnp.clip(init_x, 0.0, 1.0)
    f0 = function_equilibrium(x0, k, bC, bH, bO)
    x, _, _ = while_loop(cond_fn, body_fn, (x0, f0, 0))
    return x

@custom_vjp
def root_equilibrium(k, bC, bH, bO, x0=0.5):
    return newton_scalar(x0, k=k, bC=bC, bH=bH, bO=bO)

# ── forward pass ─────────────────────────────────────────
def root_equilibrium_fwd(k, bC, bH, bO, x0=0.5):
    x_star = newton_scalar(x0, k=k, bC=bC, bH=bH, bO=bO)
    aux = (x_star, k, bC, bH, bO)
    return x_star, aux

# ── backward pass (implicit function theorem) ────────────
def root_equilibrium_bwd(aux, g):
    x_star, k, bC, bH, bO = aux
    dFdx = grad(function_equilibrium, 0)(x_star, k, bC, bH, bO)
    dFdtheta = grad(function_equilibrium, (1,2,3,4))(x_star, k, bC, bH, bO)
    coef = -g / dFdx
    return tuple(coef * d for d in dFdtheta) + (None,)

root_equilibrium.defvjp(root_equilibrium_fwd, root_equilibrium_bwd)


if __name__ == "__main__":
    # Example usage
    hco_system = HCOSystem()
    T = 298.15
    P = 1.0  # Normalized pressure
    k = hco_system.equilibrium_constant(T, P)
    print(f"Equilibrium constant at T={T} K and P={P}: {k}")
    root_x = root_equilibrium(k, 1.0, 3.0, 1.0, x0=0.5)
    print(f"Root x: {root_x}")
    