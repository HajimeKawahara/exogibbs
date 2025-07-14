def lagrange_newton(T, P, T_table, mu_table, x0, formula_matrix, b, stepsize=1e-2, num_steps=1000
):


if __name__ == "__main__":
    from exogibbs.io.load_data import get_data_filepath
    from exogibbs.equilibrium.gibbs import extract_and_pad_gibbs_data
    from exogibbs.io.load_data import load_formula_matrix
    from exogibbs.io.load_data import DEFAULT_JANAF_GIBBS_MATRICES
    from exogibbs.io.load_data import NUMBER_OF_SPECIES_SAMPLE
    import numpy as np
    import pandas as pd
    from jax import config

    config.update("jax_enable_x64", True)


    ref = pd.read_csv("yk.list", header=None, sep=",").values[0]
    print("ref", ref.shape)


    path = get_data_filepath(DEFAULT_JANAF_GIBBS_MATRICES)
    gibbs_matrices = np.load(path, allow_pickle=True)["arr_0"].item()

    molecules, T_table, mu_table, grid_lens = extract_and_pad_gibbs_data(gibbs_matrices)
    # checks using the results of Kawashima's code.



    npath = get_data_filepath(NUMBER_OF_SPECIES_SAMPLE)
    
    tip = 1.e-20
    number_of_species_init = pd.read_csv(npath, header=None, sep=",").values[0] + tip
    T = 500.0
    P = 10.0

    formula_matrix = load_formula_matrix()

    b = formula_matrix @ number_of_species_init
    #print("init Gibbs:",computes_total_gibbs_energy(number_of_species_init, T, P, T_table, mu_table, Pref=1.0))

    x_opt = optimize_lagrange(
        T,
        P,
        T_table,
        mu_table,
        number_of_species_init,
        formula_matrix,
        b,
        stepsize=1e-5,
        num_steps=1000,
    )
    print("feasible  |Ax-b|:", jnp.linalg.norm(formula_matrix @ x_opt - b))
    print("non-neg   min x :", jnp.min(x_opt))
    print(
        "objective f(x) :",
        computes_total_gibbs_energy(x_opt, T, P, T_table, mu_table, Pref=1.0),
    )
    print(x_opt)

    import matplotlib.pyplot as plt
    plt.plot(number_of_species_init, "^")
    plt.plot(x_opt, "o")
    plt.plot(ref, "x")
    plt.yscale("log")
    plt.legend(["init", "opt", "ref"])
    plt.show()