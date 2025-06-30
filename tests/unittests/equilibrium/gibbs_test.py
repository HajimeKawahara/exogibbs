from exogibbs.io.load_data import get_data_filepath
from exogibbs.io.load_data import load_JANAF_molecules
from exogibbs.equilibrium.gibbs import extract_and_pad_gibbs_data
from exogibbs.equilibrium.gibbs import interpolate_chemical_potential_all
from exogibbs.equilibrium.gibbs import robust_temperature_range
import pytest

def _compute_table_gibbs_data():
    import pandas as pd

    df_molecules = pd.DataFrame(
        {
            "Molecule": ["janaf_raw"],
        }
    )
    filepath = get_data_filepath("")
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

    gibbs_vec = interpolate_chemical_potential_all(
        T_query, T_table, G_table
    )  # shape (M,)
    chemical_potential_dict = dict(zip(molecules, gibbs_vec))
    assert chemical_potential_dict["janaf_raw"] == 150.0

    if fig:
        import matplotlib.pyplot as plt

        t = gibbs_matrices["janaf_raw"]["T(K)"]
        g = gibbs_matrices["janaf_raw"]["delta-f G"] * 1.0e3
        plt.plot(t, g)
        plt.plot(T_query, chemical_potential_dict["janaf_raw"], "o")
        plt.xlabel("T(K)")
        plt.ylabel("delta-f G (J/mol)")
        plt.show()


def test_robust_temperature_range():
    _, _, T_table, _ = _compute_table_gibbs_data()
    Tmin, Tmax = robust_temperature_range(T_table)
    assert Tmin == 0.0
    assert Tmax == 500.0


def test_total_gibbs_energy():
    from exogibbs.equilibrium.gibbs import computes_total_gibbs_energy
    from exogibbs.io.load_data import DEFAULT_JANAF_GIBBS_MATRICES
    from exogibbs.io.load_data import NUMBER_OF_SPECIES_SAMPLE
    import numpy as np
    from jax import config

    config.update("jax_enable_x64", True)

    path = get_data_filepath(DEFAULT_JANAF_GIBBS_MATRICES)
    gibbs_matrices = np.load(path, allow_pickle=True)["arr_0"].item()

    molecules, T_table, mu_table, grid_lens = extract_and_pad_gibbs_data(gibbs_matrices)
    # checks using the results of Kawashima's code.
    import pandas as pd

    npath = get_data_filepath(NUMBER_OF_SPECIES_SAMPLE)
    number_of_species = pd.read_csv(npath, header=None, sep=",").values[0]
    T = 500.0
    P = 10.0
    gibbs_energy = computes_total_gibbs_energy(
        number_of_species, T, P, T_table, mu_table, Pref=1.0
    )
    print("total Gibbs energy ", gibbs_energy, "should be about 8014")
    assert gibbs_energy == pytest.approx(8014.834042286333)


def test_polynomial_interpolation_order2():
    import jax.numpy as jnp
    import numpy as np
    from exogibbs.equilibrium.gibbs import interpolate_chemical_potential_one
    
    # Simple test data: quadratic function y = x^2
    T_vec = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mu_vec = jnp.array([1.0, 4.0, 9.0, 16.0, 25.0])  # x^2 values
    
    # Test scalar input
    T_target = 2.5
    expected = 6.25  # 2.5^2
    
    # Test linear interpolation (should not be exact for quadratic)
    result_linear = interpolate_chemical_potential_one(T_target, T_vec, mu_vec, order=1)
    print(f"Linear interpolation: {result_linear}")
    
    # Test quadratic interpolation (should be exact for quadratic data)
    result_quad = interpolate_chemical_potential_one(T_target, T_vec, mu_vec, order=2)
    print(f"Quadratic interpolation: {result_quad}, expected: {expected}")
    
    # Quadratic should be more accurate than linear for quadratic data
    assert abs(result_quad - expected) < abs(result_linear - expected)
    
    # Test array input (like in tce_two_species.ipynb)
    T_array = np.array([2.0, 2.5, 3.0])
    expected_array = T_array**2  # [4.0, 6.25, 9.0]
    
    result_array_linear = interpolate_chemical_potential_one(T_array, T_vec, mu_vec, order=1)
    result_array_quad = interpolate_chemical_potential_one(T_array, T_vec, mu_vec, order=2)
    
    print(f"Array linear: {result_array_linear}")
    print(f"Array quadratic: {result_array_quad}, expected: {expected_array}")
    
    # Check that array results are close to expected (quadratic should be exact)
    assert np.allclose(result_array_quad, expected_array, atol=1e-10)

if __name__ == "__main__":
    test_pad_gibbs_data()
    test_interpolation_gibbs(fig=True)
    test_robust_temperature_range()
    test_total_gibbs_energy()
    test_polynomial_interpolation_order2()
