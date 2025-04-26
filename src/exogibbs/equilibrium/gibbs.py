from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict
import jax
import jax.numpy as jnp
from functools import partial

_INF_STRINGS = {"inf", "Inf", "INFINITE"}
_NINF_STRINGS = {"-inf", "-Inf", "-INFINITE"}


def _coerce_to_float(a):
    """convert "inf" string to np.inf and -inf
    """
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
    pad_mode="edge",
):
    """

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

    def _pad(arr, L, mode):
        arr = _coerce_to_float(arr)
        if mode == "edge":
            return np.pad(arr, (0, L - arr.size), mode="edge")
        elif mode == "nan":
            return np.pad(
                arr, (0, L - arr.size), mode="constant", constant_values=np.nan
            )
        else:
            raise ValueError("pad_mode must be 'edge' or 'nan'")

    T_table = np.stack(
        [_pad(gibbs_matrices[m][temperature_key], Lmax, pad_mode) for m in molecules]
    ).astype(np.float64)
    G_table = np.stack(
        [
            _pad(gibbs_matrices[m][checmical_potential_key], Lmax, pad_mode)
            for m in molecules
        ]
    ).astype(np.float64)

    return (
        molecules,
        jnp.asarray(T_table),  # shape (M, Lmax), float64
        jnp.asarray(G_table),  # shape (M, Lmax), float64
        tuple(grid_lens),
    )  # shape (M,),      int32


@partial(jax.jit, static_argnums=(3,))
def _interp_one(T_target, T_vec, G_vec, n):
    """
    T_target : scalar
    T_vec, G_vec : 1-D（Lmax)
    n : actual data number
    """
    idx = jnp.clip(jnp.searchsorted(T_vec[:n], T_target) - 1, 0, n - 2)
    T0, T1 = T_vec[idx], T_vec[idx + 1]
    G0, G1 = G_vec[idx], G_vec[idx + 1]
    w = (T_target - T0) / (T1 - T0)
    return (1 - w) * G0 + w * G1


def interpolate_gibbs_all(T_target, T_table, G_table, grid_lens):
    return jax.lax.map(
        lambda args: _interp_one(T_target, *args),
        (T_table, G_table, grid_lens),
    )


if __name__ == "__main__":
    from exogibbs.io.load_data import load_JANAF_molecules
    from exogibbs.io.load_data import load_molname

    df_molname = load_molname()
    path_JANAF_data = "/home/kawahara/thermochemical_equilibrium/Equilibrium/JANAF"
    gibbs_matrices = load_JANAF_molecules(df_molname, path_JANAF_data)
    molecules, T_table, G_table, grid_lens = pad_gibbs_data(gibbs_matrices)
    print(molecules)
    print(T_table)
    print(G_table)

    T_query = 700.0
    gibbs_vec = interpolate_gibbs_all(T_query, T_table, G_table, grid_lens)  # shape (M,)
    # print(dict(zip(molecules, gibbs_vec)))
