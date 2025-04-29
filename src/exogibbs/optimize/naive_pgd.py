from exogibbs.equilibrium.gibbs import computes_total_gibbs_energy
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
from jax.lax import fori_loop
from jax.lax import scan
from jax import grad
from jax import jit
from functools import partial
from jaxopt import projection

# def optimize_gibbs_pgd(T, P, T_table, mu_table, formula_matrix, b):
def optimize_gibbs_pgd(
    T, P, T_table, mu_table, x0, formula_matrix, b, stepsize=1e-2, num_steps=1000
):

    # compute cholesky factorization of P
    AAT = formula_matrix @ formula_matrix.T
    Lcho = jnp.linalg.cholesky(AAT)

    @jit
    def proj_nonneg(y):
        return jnp.clip(y, a_min=0.0)

    @jit
    def proj_affine(y):
        lam = cho_solve((Lcho, True), formula_matrix @ y - b)  # lam = (AAT)^-1 (Ay−b)
        return y - formula_matrix.T @ lam


    #def proj_intersection(y):
    #    z, _ = projection.projection_affine_set(
    #            y, formula_matrix, b)
    #    return z

    @jit
    def proj_intersection(y, num_iters=20):
        p = jnp.zeros_like(y)  # dual for nonneg
        q = jnp.zeros_like(y)  # dual for affine
        x = y

        def body(_, state):
            x, p, q = state
            # step 1: non-neg
            y1 = proj_nonneg(x + p)
            p = x + p - y1

            # step 2: affine
            y2 = proj_affine(y1 + q)
            q = y1 + q - y2

            return (y2, p, q)

        x, _, _ = fori_loop(0, num_iters, body, (x, p, q))
        return x

    gibbs_energy = partial(computes_total_gibbs_energy, T=T, P=P, T_table=T_table, mu_table=mu_table)

    grad_f = jit(grad(gibbs_energy))
    #grad_f = grad(gibbs_energy)

    @jit
    def pgd_step(state, _):
        x, t = state  # t = stepsize
        g = grad_f(x)
        y = x - t * g  # unconstrained descent
        x_new = proj_intersection(y)         
        return (x_new, t), x_new

    init_state = (proj_intersection(x0), stepsize)
    
    
    _, traj = scan(pgd_step, init_state, None, length=num_steps)
    return traj  # shape (num_steps, n)


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
    number_of_species_init = pd.read_csv(npath, header=None, sep=",").values[0]
    T = 500.0
    P = 10.0

    formula_matrix = load_formula_matrix()

    b = formula_matrix @ number_of_species_init
    print("init Gibbs:",computes_total_gibbs_energy(number_of_species_init, T, P, T_table, mu_table, Pref=1.0))
    traj = optimize_gibbs_pgd(
        T,
        P,
        T_table,
        mu_table,
        number_of_species_init,
        formula_matrix,
        b,
        stepsize=1e-7,
        num_steps=1000,
    )
    x_opt = traj[-1]
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