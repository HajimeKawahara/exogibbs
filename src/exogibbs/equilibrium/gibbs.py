from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict
import jax
import jax.numpy as jnp
from jax import jit

_INF_STRINGS = {"inf", "Inf", "INFINITE"}
_NINF_STRINGS = {"-inf", "-Inf", "-INFINITE"}


def _coerce_to_float(a):
    """convert "inf" string to np.inf and -inf"""
    a = np.asarray(a, dtype=object)

    is_str = np.vectorize(lambda x: isinstance(x, str))
    str_mask = is_str(a)

    if str_mask.any():
        strs = a[str_mask]
        lower = np.char.lower(strs.astype(str))

        # inf / -inf
        a[str_mask & np.isin(lower, list(_INF_STRINGS))] = np.inf
        a[str_mask & np.isin(lower, list(_NINF_STRINGS))] = -np.inf

        other_mask = str_mask & ~np.isin(lower, list(_INF_STRINGS | _NINF_STRINGS))
        a[other_mask] = np.nan

    return a.astype(np.float64)


def pad_gibbs_data(
    gibbs_matrices: Dict[str, pd.DataFrame],
    temperature_key: str = "T(K)",
    checmical_potential_key: str = "delta-f G",
):
    """padding of gibbs data to the same length

    Args:
        gibbs_matrices (Dict[str, pd.DataFrame]): needs to have the key of temperature_key and checmecal_potential_key
        temperature_key (str): key for temperature
        checmical_potential_key (str): key for chemical potential
    
    Returns:
        molecules (list): list of molecules
        T_table (ndarray): temperature table (Nmolecules, Lmax)
        G_table (ndarray): gibbs energy table (Nmolecules, Lmax)
    """
    molecules = list(gibbs_matrices.keys())
    grid_lens = np.asarray(
        [gibbs_matrices[m][temperature_key].size for m in molecules], dtype=np.int32
    )
    Lmax = int(grid_lens.max())

    def _pad(arr, L):
        arr = _coerce_to_float(arr)
        return np.pad(arr, (0, L - arr.size), mode="edge")

    T_table = np.stack(
        [_pad(gibbs_matrices[m][temperature_key], Lmax) for m in molecules]
    ).astype(np.float64)
    G_table = np.stack(
        [_pad(gibbs_matrices[m][checmical_potential_key], Lmax) for m in molecules]
    ).astype(np.float64)

    return (
        molecules,
        jnp.asarray(T_table),  # shape (M, Lmax), float64
        jnp.asarray(G_table),  # shape (M, Lmax), float64
        tuple(grid_lens),
    )  # shape (M,),      int32


@jit
def _interp_one(T_target, T_vec, G_vec):
    """interpolate one chemical potential at T_target
    Args:
        T_target (scalar): target temperature (K)
        T_vec (1D array): temeprature grid（Lmax)
        G_vec (1D array): chemical potential grid（Lmax)
    """
    n = T_vec.size
    idx = jnp.clip(jnp.searchsorted(T_vec, T_target) - 1, 0, n - 2)
    T0, T1 = T_vec[idx], T_vec[idx + 1]
    G0, G1 = G_vec[idx], G_vec[idx + 1]
    w = (T_target - T0) / (T1 - T0)
    return (1 - w) * G0 + w * G1


def interpolate_gibbs_all(T_target, T_table, G_table)
    """interpolate the chemical potential at T_target for all molecules
    Args:
        T_target (scalar): target temperature (K)
        T_table (ndarray): array of temeprature grid（Lmax)
        G_table (ndarray): array of chemical potential grid（Lmax)
    
    Returns:
        gibbs_vec (ndarray): array of chemical potential at T_target (Nmol,Lmax)
    """
    return jax.lax.map(
        lambda args: _interp_one(T_target, *args),
        (T_table, G_table),
    )


if __name__ == "__main__":

    # Example usage of interpolating the chemical potential, deriving the Gibbs energy at 700K
    from exogibbs.io.load_data import load_JANAF_molecules
    from exogibbs.io.load_data import load_molname

    df_molname = load_molname()
    path_JANAF_data = "/home/kawahara/thermochemical_equilibrium/Equilibrium/JANAF"
    gibbs_matrices = load_JANAF_molecules(df_molname, path_JANAF_data)
    molecules, T_table, G_table, grid_lens = pad_gibbs_data(gibbs_matrices)

    T_query = 700.0
    gibbs_vec = interpolate_gibbs_all(T_query, T_table, G_table)  # shape (M,)
    Tdict = dict(zip(molecules, gibbs_vec))

    import matplotlib.pyplot as plt

    t = gibbs_matrices["C1O2"]["T(K)"]
    g = gibbs_matrices["C1O2"]["delta-f G"]
    plt.plot(t, g)
    plt.plot(T_query, Tdict["C1O2"], "o")
    plt.xlabel("T(K)")
    plt.ylabel("delta-f G")
    plt.show()
