from exogibbs.io.load_data import get_data_filepath
from exogibbs.io.load_data import load_JANAF_molecules
from exogibbs.equilibrium.gibbs import pad_gibbs_data
from exogibbs.equilibrium.gibbs import interpolate_gibbs_all


def test_interpolation_gibbs(fig=False):
    import pandas as pd

    df_molecules = pd.DataFrame(
        {
            "Molecule": ["janaf_raw"],
        }
    )
    filepath = get_data_filepath("")
    gibbs_matrices = load_JANAF_molecules(df_molecules, filepath, tag="_sample")
    molecules, T_table, G_table, _ = pad_gibbs_data(gibbs_matrices)
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


if __name__ == "__main__":
    test_interpolation_gibbs(fig=True)
