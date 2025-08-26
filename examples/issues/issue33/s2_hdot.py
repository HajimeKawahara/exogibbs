import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, vmap
from jax import config
config.update("jax_enable_x64", True)

from exogibbs.io.load_data import get_data_filepath, DEFAULT_JANAF_GIBBS_MATRICES
from exogibbs.equilibrium.gibbs import extract_and_pad_gibbs_data, interpolate_hvector_one
from exogibbs.utils.constants import R_gas_constant_si as R

def main():
    # Load JANAF-like tables and coerce to JAX arrays
    path = get_data_filepath(DEFAULT_JANAF_GIBBS_MATRICES)
    gibbs_matrices = np.load(path, allow_pickle=True)["arr_0"].item()
    molecules, T_table, mu_table, grid_lens = extract_and_pad_gibbs_data(gibbs_matrices)

    # Locate S2 and slice its true grid (exclude padding)
    idx = molecules.index("S2")
    L = int(grid_lens[idx])
    T_vec = T_table[idx, :L]
    mu_vec = mu_table[idx, :L]

    print(f"S2 grid length: {L}")
    print(f"S2 T range: [{float(T_vec.min()):.1f}, {float(T_vec.max()):.1f}] K")

    # Define h_S2(T) = mu^o(T) / (R T) using the same interpolator
    def h_S2(T):
        return interpolate_hvector_one(T, T_vec, mu_vec)

    # Sample around 900 K
    #T_points = jnp.array([880.0, 890.0, 900.0, 910.0, 920.0])
    T_points = jnp.linspace(200,1500,130)
    h_vals = vmap(h_S2)(T_points)
    hdot_vals = vmap(grad(h_S2))(T_points)

    # Print numeric view
    print("\nT points [K]:", T_points)
    print("h(T):", h_vals)
    print("hdot(T):", hdot_vals)
    print("isfinite(h):", jnp.isfinite(h_vals))
    print("isfinite(hdot):", jnp.isfinite(hdot_vals))

    # Also show the raw table value nearest 900 K
    idx900 = int(jnp.argmin(jnp.abs(T_vec - 900.0)))
    T_near = float(T_vec[idx900])
    mu_near = float(mu_vec[idx900])
    h_near = mu_near / (R * T_near)
    print(f"\nNearest grid to 900 K: T={T_near:.1f} K, mu={mu_near}")
    print(f"h on grid near 900 K: {h_near}")

    # Optional: plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        T_plot = jnp.linspace(max(800.0, float(T_vec.min())),
                              min(1000.0, float(T_vec.max())), 101)
        h_plot = vmap(h_S2)(T_plot)
        hdot_plot = vmap(grad(h_S2))(T_plot)

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
        ax[0].plot(T_plot, h_plot, label="h(T)")
        ax[0].scatter(T_points, h_vals, color="k", zorder=3, s=15)
        ax[0].axvline(900.0, color="r", ls="--", lw=1, label="900 K")
        ax[0].set_ylabel("h(T) = μ°/(RT)")
        ax[0].legend()

        ax[1].plot(T_plot, hdot_plot, label="ḣ(T)")
        ax[1].scatter(T_points, hdot_vals, color="k", zorder=3, s=15)
        ax[1].axvline(900.0, color="r", ls="--", lw=1)
        ax[1].set_xlabel("Temperature [K]")
        ax[1].set_ylabel("ḣ(T) = d/dT h(T)")
        ax[1].legend()

        fig.tight_layout()
        plt.show()
    except Exception as e:
        print("\n[Plot skipped]", e)

if __name__ == "__main__":
    main()
