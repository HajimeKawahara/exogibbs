from exogibbs.io.load_data import get_data_filepath
from exogibbs.io.load_data import load_JANAF_molecules
from exogibbs.io.load_data import JANAF_NAME_KEY
from exogibbs.thermo.gibbs import extract_and_pad_gibbs_data
from exogibbs.thermo.gibbs import interpolate_hvector_all
from exogibbs.thermo.gibbs import robust_temperature_range
import pytest

def _compute_table_gibbs_data():
    import pandas as pd

    df_molecules = pd.DataFrame(
        {
            JANAF_NAME_KEY: ["janaf_raw"],
        }
    )
    filepath = get_data_filepath("test")
    gibbs_matrices = load_JANAF_molecules(df_molecules, filepath, tag="_sample")
    molecules, T_table, G_table, _ = extract_and_pad_gibbs_data(gibbs_matrices)
    return gibbs_matrices, molecules, T_table, G_table


def test_pad_gibbs_data():
    _, molecules, T_table, G_table = _compute_table_gibbs_data()
    assert len(molecules) == 1
    assert T_table.shape == (1, 10)
    assert G_table.shape == (1, 10)


def test_interpolation_gibbs(fig=False):
    gibbs_matrices, molecules, T_table, G_table = _compute_table_gibbs_data()
    T_query = 150.0

    gibbs_vec = interpolate_hvector_all(
        T_query, T_table, G_table
    )  # shape (M,)
    chemical_potential_dict = dict(zip(molecules, gibbs_vec))
    print("chemical_potential_dict", chemical_potential_dict["janaf_raw"])
    #assert chemical_potential_dict["janaf_raw"] == 150.0

    if fig:
        import matplotlib.pyplot as plt
        import numpy as np
        t = gibbs_matrices["janaf_raw"]["T(K)"]
        g = gibbs_matrices["janaf_raw"]["delta-f G"] * 1.0e3
        plt.plot(t, g)
        for T_query in np.linspace(20,1000,50):
        #T_query = 150.0

            gibbs_vec = interpolate_hvector_all(
                T_query, T_table, G_table
            )  # shape (M,)
            chemical_potential_dict = dict(zip(molecules, gibbs_vec))
        
        
            plt.plot(T_query, chemical_potential_dict["janaf_raw"], "o")
        plt.xlabel("T(K)")
        plt.ylabel("h-vector")
        plt.show()


def test_robust_temperature_range():
    _, _, T_table, _ = _compute_table_gibbs_data()
    Tmin, Tmax = robust_temperature_range(T_table)
    assert Tmin == 0.0
    assert Tmax == 500.0



if __name__ == "__main__":
    test_pad_gibbs_data()
    test_interpolation_gibbs(fig=True)
    test_robust_temperature_range()
