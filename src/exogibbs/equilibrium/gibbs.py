from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict
import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.special import xlogy
from jax import custom_jvp
from interpax import interp1d
    
from exogibbs.utils.constants import R_gas_constant_si

_INF_STRINGS = {"inf", "Inf", "INFINITE"}
_NINF_STRINGS = {"-inf", "-Inf", "-INFINITE"}
EPS = 1e-20
LOG_EPS = jnp.log(EPS)


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
    unit_conversion_factor=1.0e3,
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

    mu_table = mu_table * unit_conversion_factor  # from kJ/mol to J/mol

    return (
        molecules,
        jnp.asarray(T_table),  # shape (M, Lmax), float64
        jnp.asarray(mu_table),  # shape (M, Lmax), float64
        tuple(grid_lens),
    )  # shape (M,),      int32


def interpolate_chemical_potential_one(T_target, T_vec, mu_vec, method="cubic"):
    """interpolate one chemical potential at T_target
    Args:
        T_target (scalar or array): target temperature(s) (K)
        T_vec (1D array): temeprature grid（Lmax)
        mu_vec (1D array): chemical potential grid（Lmax)
        method (str):  method of interpolation used in interpax.interp1d
            'nearest': nearest neighbor interpolation
            'linear': linear interpolation
            'cubic': C1 cubic splines (aka local splines)
            'cubic2': C2 cubic splines (aka natural splines)
            'catmull-rom': C1 cubic centripetal “tension” splines
            'cardinal': C1 cubic general tension splines. If used, can also pass keyword parameter c in float[0,1] to specify tension
            'monotonic': C1 cubic splines that attempt to preserve monotonicity in the data, and will not introduce new extrema in the interpolated points
            'monotonic-0': same as 'monotonic' but with 0 first derivatives at both endpoints
            'akima': C1 cubic splines that appear smooth and natural
        
    """
    return interp1d(T_target, T_vec, mu_vec, method=method)
    
def interpolate_hvector_one(T_target, T_vec, mu_vec, method="cubic"):
    """interpolate one hvector = (chemical_potential/RT)  at T_target
    Args:
        T_target (scalar or array): target temperature(s) (K)
        T_vec (1D array): temeprature grid（Lmax)
        mu_vec (1D array): chemical potential grid（Lmax)
        method (str):  method of interpolation used in interpax.interp1d
            'nearest': nearest neighbor interpolation
            'linear': linear interpolation
            'cubic': C1 cubic splines (aka local splines)
            'cubic2': C2 cubic splines (aka natural splines)
            'catmull-rom': C1 cubic centripetal “tension” splines
            'cardinal': C1 cubic general tension splines. If used, can also pass keyword parameter c in float[0,1] to specify tension
            'monotonic': C1 cubic splines that attempt to preserve monotonicity in the data, and will not introduce new extrema in the interpolated points
            'monotonic-0': same as 'monotonic' but with 0 first derivatives at both endpoints
            'akima': C1 cubic splines that appear smooth and natural

    """
    RT = R_gas_constant_si * T_vec
    return interp1d(T_target, T_vec, mu_vec/RT, method=method) 
    

def interpolate_chemical_potential_all(T_target, T_table, mu_table):
    """interpolate the chemical potential at T_target for all molecules
    Args:
        T_target (scalar): target temperature (K)
        T_table (ndarray): array of temeprature grid（Lmax)
        mu_table (ndarray): array of chemical potential grid（Nmol, Lmax)

    Returns:
        chemical_potential_vec (ndarray): array of chemical potential at T_target (Nmol,Lmax)
    """
    return jax.lax.map(
        lambda args: interpolate_chemical_potential_one(T_target, *args),
        (T_table, mu_table),
    )

def interpolate_hvector_all(T_target, T_table, mu_table):
    """interpolate hvector = the chemical potential/RT at T_target for all molecules
    Args:
        T_target (scalar): target temperature (K)
        T_table (ndarray): array of temeprature grid（Lmax)
        mu_table (ndarray): array of chemical potential grid（Nmol, Lmax)

    Returns:
        chemical_potential_vec (ndarray): array of chemical potential at T_target (Nmol,Lmax)
    """
    return jax.lax.map(
        lambda args: interpolate_hvector_one(T_target, *args),
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

    Notes:
        Gtot = sum_i^N n_i mu_i (n_i:number of species i, mu_i: chemical potential of species i 3.5-1 p46 in Smith and Missen)
        mu_i = mu_i^0 + RT ln pi (mu_i^0: standard molar chemical potential, pi: partial pressure of species i, 3.7-12 p51)
        x_i = n_i/n_tot is mole fraction of species i, i.e. p_i = x_i P, sum_i x_i = 1, where n_tot is the total number of species
        Then we obtain, Gtot = \sum_i n_i mu_i^0 + n_tot R T (\sum_i x_i ln x_i + ln P ), R: gas constant, T: temperature
    
    """
    chemical_potential_vec = interpolate_chemical_potential_all(
        T, T_table, mu_table
    )  # shape (M,)
    total_number_of_species = jnp.sum(number_of_species)
    x_i = number_of_species / total_number_of_species

    # 3.5-1 (p46) and 3.7-12  (p51) in Smith and Missen (Ideal gas)
    nRT = total_number_of_species * R_gas_constant_si * T
    zero_term = jnp.sum(number_of_species * chemical_potential_vec)
    temperature_factor = nRT * (jnp.sum(safe_xlogx(x_i)) + jnp.log(P / Pref))
    return zero_term + temperature_factor


@custom_jvp
def safe_xlogx(x):
    """"""
    positive = x > EPS
    # positive branch: as is
    branch1 = jax.scipy.special.xlogy(x, x)
    # non-positive: x*log(eps) + (x-eps)  (first-order)
    branch2 = x * LOG_EPS + (x - EPS)
    return jnp.where(positive, branch1, branch2)


@safe_xlogx.defjvp
def safe_xlogx_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    positive = x > EPS
    y = safe_xlogx(x)
    # gradient: 1 + log x   (x>eps) /  log(eps) const  (x<=eps)
    grad_pos = 1.0 + jnp.log(jnp.clip(x, a_min=EPS))
    grad = jnp.where(positive, grad_pos, 1.0 + LOG_EPS)
    return y, grad * dx


if __name__ == "__main__":
    from jax.scipy.special import xlogy
    from jax import grad

    print(xlogy(0.0, 0.0))  # should be 0.0

    def f(x):
        return safe_xlogx(x)

    gf = grad(f)
    print(gf(0.0))  # should be 1.0
