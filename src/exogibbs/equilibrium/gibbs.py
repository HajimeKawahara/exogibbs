from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict
import jax
import jax.numpy as jnp
from jax import jit
from exogibbs.utils.constants import R_gas_constant_si
from jax.scipy.special import xlogy

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


def extract_and_pad_gibbs_data(
    gibbs_matrices: Dict[str, pd.DataFrame],
    temperature_key: str = "T(K)",
    checmical_potential_key: str = "delta-f G",
    unit_conversion_factor = 1.e3
):
    """Extracting molar chemical potential from gibbs_matrices and perform padding to the same length

    Args:
        gibbs_matrices (Dict[str, pd.DataFrame]): chemical potential matrices, needs to have the key of temperature_key and checmecal_potential_key
        temperature_key (str): key for temperature
        checmical_potential_key (str): key for chemical potential, default to "delta-f G" (standard mol chemcical potential, Pst=1 bar IUPAC since 1982)
        unit_conversion_factor (float): conversion factor for chemical potential, default to 1.e3 (kJ/mol -> J/mol), #44 in Kawashima's code (eq_subprog.f90) 

    Notes:
        delta-f G (standard molar Gibbs free energy of formation or standard molar checmical potential) in JANAF is given in kJ/mol, 
        but in the code it is converted to J/mol. "unit_conversion_factor" is used to convert the unit of chemical potential from kJ/mol to J/mol.
    
    Returns:
        molecules (list): list of molecules
        T_table (ndarray): temperature table in K (Nmolecules, Lmax)
        mu_table (ndarray): standard molar chemical potential table in J/mol (Nmolecules, Lmax), standard state: Pref = 1 bar
        grid_length (tuple): tuple of grid lengths for each molecule
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
    mu_table = np.stack(
        [_pad(gibbs_matrices[m][checmical_potential_key], Lmax) for m in molecules]
    ).astype(np.float64)

    mu_table = mu_table*unit_conversion_factor

    return (
        molecules,
        jnp.asarray(T_table),  # shape (M, Lmax), float64
        jnp.asarray(mu_table),  # shape (M, Lmax), float64
        tuple(grid_lens),
    )  # shape (M,),      int32


@jit
def _interp_one(T_target, T_vec, mu_vec):
    """interpolate one chemical potential at T_target
    Args:
        T_target (scalar): target temperature (K)
        T_vec (1D array): temeprature grid（Lmax)
        mu_vec (1D array): chemical potential grid（Lmax)
    """
    n = T_vec.size
    idx = jnp.clip(jnp.searchsorted(T_vec, T_target) - 1, 0, n - 2)
    T0, T1 = T_vec[idx], T_vec[idx + 1]
    mu0, mu1 = mu_vec[idx], mu_vec[idx + 1]
    w = (T_target - T0) / (T1 - T0)
    return (1 - w) * mu0 + w * mu1


def interpolate_chemical_potential_all(T_target, T_table, mu_table):
    """interpolate the chemical potential at T_target for all molecules
    Args:
        T_target (scalar): target temperature (K)
        T_table (ndarray): array of temeprature grid（Lmax)
        mu_table (ndarray): array of chemical potential grid（Lmax)

    Returns:
        chemical_potential_vec (ndarray): array of chemical potential at T_target (Nmol,Lmax)
    """
    return jax.lax.map(
        lambda args: _interp_one(T_target, *args),
        (T_table, mu_table),
    )


def robust_temperature_range(T_table):
    """get the robust temperature range for all molecules
    Args:
        T_table (ndarray): array of temeprature grid（Lmax)
    Returns:
        Tmin (float): minimum temperature (K)
        Tmax (float): maximum temperature (K)
    """
    Tmin = np.max(np.min(T_table, axis=1))
    Tmax = np.min(np.max(T_table, axis=1))
    return Tmin, Tmax


def computes_total_gibbs_energy(number_of_species, T, P, T_table, mu_table, Pref=1.0):
    """computes the total gibbs energy at T and P
    Args:
        number_of_species (ndarray): array of number of species (Nmol)
        T (float): temperature (K)
        P (float): pressure (bar)
        T_table (ndarray): array of temeprature grid（Lmax)
        mu_table (ndarray): array of chemical potential grid（Lmax)
        Pref (float): reference pressure (bar) for the checmical potential data. default to the standard pressure Pst = 1.0 bar (IUPAC since 1982)

    Returns:
        (ndarray): total gibbs energy, Gtot
    """
    chemical_potential_vec = interpolate_chemical_potential_all(
        T, T_table, mu_table
    )  # shape (M,)
    total_number_of_species = jnp.sum(number_of_species)
    x_i = number_of_species / total_number_of_species

    # 3.5-1 (p46) and 3.7-12  (p51) in Smith and Missen (Ideal gas)
    nRT = total_number_of_species * R_gas_constant_si * T
    mui0 = number_of_species * chemical_potential_vec 
    return jnp.sum(mui0) + nRT * (jnp.sum(xlogy(x_i, x_i)) + jnp.log(P / Pref))


if __name__ == "__main__":
    from jax import config

    config.update("jax_enable_x64", True)
    # Example usage of interpolating the chemical potential, deriving the Gibbs energy at 700K
    from exogibbs.io.load_data import load_JANAF_molecules
    from exogibbs.io.load_data import load_molname

    df_molname = load_molname()
    path_JANAF_data = "/home/kawahara/thermochemical_equilibrium/Equilibrium/JANAF"
    mu_matrices = load_JANAF_molecules(df_molname, path_JANAF_data)
    molecules, T_table, mu_table, grid_lens = extract_and_pad_gibbs_data(mu_matrices)

    T_query = 700.0
    mu_vec = interpolate_chemical_potential_all(T_query, T_table, mu_table)  # shape (M,)
    Tdict = dict(zip(molecules, mu_vec))
    Tmin, Tmax = robust_temperature_range(T_table)
    print(f"robust temperature range: {Tmin} - {Tmax}")

    # checks using the results of Kawashima's code.
    import pandas as pd
    number_of_species = pd.read_csv("n.list", header=None, delim_whitespace=True).values[0]
    T = 500.0
    P = 10.0
    gibbs_energy = computes_total_gibbs_energy(
        number_of_species, T, P, T_table, mu_table, Pref=1.0
    )
    print("total Gibbs energy ",gibbs_energy, "should be about 8014")
    
    fig = False
    if fig:
        import matplotlib.pyplot as plt

        t = mu_matrices["C1O2"]["T(K)"]
        mu = mu_matrices["C1O2"]["delta-f G"]
        plt.plot(t, mu)
        plt.plot(T_query, Tdict["C1O2"], "o")
        plt.xlabel("T(K)")
        plt.ylabel("delta-f G")
        plt.title("C1O2 mol standard chemical potential")
        plt.show()
