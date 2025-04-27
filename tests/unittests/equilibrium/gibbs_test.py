from exogibbs.io.load_data import get_data_filepath
from exogibbs.io.load_data import load_JANAF_molecules
from exogibbs.equilibrium.gibbs import pad_gibbs_data
from exogibbs.equilibrium.gibbs import interpolate_gibbs_all
from exogibbs.equilibrium.gibbs import robust_temperature_range


def _compute_table_gibbs_data():
    import pandas as pd

    df_molecules = pd.DataFrame(
        {
            "Molecule": ["janaf_raw"],
        }
    )
    filepath = get_data_filepath("")
    gibbs_matrices = load_JANAF_molecules(df_molecules, filepath, tag="_sample")
    molecules, T_table, G_table, _ = pad_gibbs_data(gibbs_matrices)
    return gibbs_matrices, molecules, T_table, G_table

def test_pad_gibbs_data():
    _, molecules, T_table, G_table = _compute_table_gibbs_data()
    assert len(molecules) == 1
    assert T_table.shape == (1, 10)
    assert G_table.shape == (1, 10)

def test_interpolation_gibbs(fig=False):
    gibbs_matrices, molecules, T_table, G_table = _compute_table_gibbs_data()
    T_query = 150.0

    gibbs_vec = interpolate_gibbs_all(T_query, T_table, G_table)  # shape (M,)
    Tdict = dict(zip(molecules, gibbs_vec))

    assert Tdict["janaf_raw"] == 0.15

    if fig:
        import matplotlib.pyplot as plt

        t = gibbs_matrices["janaf_raw"]["T(K)"]
        g = gibbs_matrices["janaf_raw"]["delta-f G"]
        plt.plot(t, g)
        plt.plot(T_query, Tdict["janaf_raw"], "o")
        plt.xlabel("T(K)")
        plt.ylabel("delta-f G")
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