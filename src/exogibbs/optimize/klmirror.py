from exogibbs.equilibrium.gibbs import computes_total_gibbs_energy
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
from jax.lax import fori_loop
from jax.lax import scan
from jax import grad
from jax import jit
from functools import partial
import jax, jax.numpy as jnp

def solve_lambda_newton(u, A, b, tol=1e-12, max_iter=20):
    """
    u : log-space vector (shape n)
    returns λ  (shape m)  s.t. A exp(u - Aᵀλ) = b
    """
    m = A.shape[0]
    lam0 = jnp.zeros(m)            # 初期値

    def body(state):
        lam, _ = state
        x  = jnp.exp(u - A.T @ lam)          # x(λ)
        F  = A @ x - b                       # residual (m,)
        # Jacobian: J_ij = ∂F_i/∂λ_j = -Σ_k A_{ik} A_{jk} x_k
        J  = -A @ (x[:, None] * A.T)         # (m,m)
        delta = jnp.linalg.solve(J, F)
        lam_new = lam - delta
        err = jnp.linalg.norm(F, ord=jnp.inf)
        return (lam_new, err)

    def cond(state):
        _, err = state
        return err > tol

    lam_final, _ = jax.lax.while_loop(cond, body, (lam0, jnp.inf))
    return lam_final

# def optimize_gibbs_pgd(T, P, T_table, mu_table, formula_matrix, b):
@jit
def optimize_gibbs_klmirror(
    T, P, T_table, mu_table, x0, formula_matrix, b, stepsize=1e-2, num_steps=1000
):

    gibbs_energy = partial(computes_total_gibbs_energy, T=T, P=P, T_table=T_table, mu_table=mu_table)
    grad_f = jit(grad(gibbs_energy))

    minval = 1.e-20

    def mirror_step(u):
        x   = jnp.exp(u)
        g   = grad_f(x)              
        u_  = u - stepsize * (x * g) 
        lam = solve_lambda_newton(u_, formula_matrix, b)  
        return u_ - formula_matrix.T @ lam  

    u0 = jnp.log(x0)
    def body(i, state):
        u = state
        u = mirror_step(u)
        return u

    u_final = jax.lax.fori_loop(0, num_steps, body, u0)
    return jnp.exp(u_final)  
        

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
    print("init Gibbs:",computes_total_gibbs_energy(number_of_species_init, T, P, T_table, mu_table, Pref=1.0))

    x_opt = optimize_gibbs_klmirror(
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