from exogibbs.io.load_data import load_molname
from exogibbs.io.load_data import load_JANAF_molecules
from exogibbs.io.load_data import load_formula_matrix


class ThermoChem:
    """
    A class to handle thermochemical equilibrium calculations.
    """

    def __init__(self):
        pass

    def set_equations(self, path_JANAF_data):
        """Set the molecules, elements, chemical_data (eq_setting in thermochemical_equilibrium package)
        """
        self.df_molname = load_molname()
        self.chemical_data = load_JANAF_molecules(
            self.df_molname,
            path_JANAF_data,
        )
        self.formula_matrix = load_formula_matrix()

    def set_initial_values(self):
        """Set the initial values for the equilibrium calculation.
        """
        return

def thermochemical_equilibrium(pressures, temperatures):
    return None
