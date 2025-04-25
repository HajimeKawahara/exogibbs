from exogibbs.io.load_data import load_molname


class ThermoChem:
    """
    A class to handle thermochemical equilibrium calculations.
    """

    def __init__(self):
        pass

    def setting(self):
        """
        Set the molecules, elements, 
        """
        self.df_molname = load_molname()
        


def thermochemical_equilibrium(pressures, temperatures):
    return None

