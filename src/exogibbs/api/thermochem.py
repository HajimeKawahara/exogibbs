from exogibbs.io.load_data import load_molname
from exogibbs.io.load_data import load_JANAF_molecules
from exogibbs.io.load_data import load_formula_matrix
from exogibbs.io.load_data import load_JANAF_molecules
from exogibbs.equilibrium.gibbs import extract_and_pad_gibbs_data
from exogibbs.equilibrium.gibbs import robust_temperature_range

import jax.numpy as jnp
from jax import tree_util

from dataclasses import dataclass

@tree_util.register_pytree_node_class
@dataclass
class ThermoState:
    temperature: float
    ln_normalized_pressure: float
    b_element_vector: jnp.ndarray

    def tree_flatten(self):
        children = (
            self.temperature,
            self.ln_normalized_pressure,
            self.b_element_vector,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        temperature, ln_normalized_pressure, b_element_vector = children
        return cls(temperature, ln_normalized_pressure, b_element_vector)


class ThermoChem:
    """
    A class to handle thermochemical equilibrium calculations.
    """

    def __init__(self):
        """Initialize the ThermoChem class.
        """
        self.set_equations(path_JANAF_data = "/home/kawahara/thermochemical_equilibrium/Equilibrium/JANAF")
        

    def set_equations(self, path_JANAF_data):
        """Set the molecules, elements, chemical_data (eq_setting in thermochemical_equilibrium package)
        """
        self.df_molname = load_molname()
        self.gibbs_matrices = load_JANAF_molecules(
            self.df_molname,
            path_JANAF_data,
        )
        self.formula_matrix = load_formula_matrix()
        self.molecules, self.T_table, self.G_table, self.grid_lens = extract_and_pad_gibbs_data(self.gibbs_matrices)
        self.Tmin, self.Tmax = robust_temperature_range(self.T_table)
        

    def set_initial_values(self):
        """Set the initial values for the equilibrium calculation.
        """
        return

