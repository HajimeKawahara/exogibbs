from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.equilibrium.gibbs import extract_and_pad_gibbs_data
from exogibbs.equilibrium.gibbs import interpolate_hvector_all
from exogibbs.equilibrium.gibbs import robust_temperature_range
from exogibbs.io.load_data import get_data_filepath
from exogibbs.thermo.stoichiometry import build_formula_matrix
from typing import Union
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax

JANAF_GIBBS_MATRICES_YKB4 = "gibbs_matrices.npz"
MOLNAME_YKB4 = "molecule_names.csv"
JANAF_NAME_KEY = "JANAF"  # key for JANAF name in the molecule names file


def load_molname_ykb4() -> pd.DataFrame:
    """Load the YKB4 molecular names table.

    Returns:
        pd.DataFrame: Molecule catalog as parsed from the packaged CSV.
    """
    # Lazily import to avoid any potential circular import when other modules
    # import constants from this module.
    from exogibbs.io.load_data import get_data_filepath

    fullpath = get_data_filepath(MOLNAME_YKB4)
    df_molname = pd.read_csv(fullpath, sep=",", dtype=str)
    return df_molname


def prepare_ykb4_setup() -> ChemicalSetup:
    """
    Prepare a JAX-friendly ChemicalSetup from JANAF-like Gibbs matrices.

    Notes
    -----
    * Ensures that all tables (T_table, mu_table) live on device as jnp.arrays.
    * hvector_func(T) stays purely JAX/NumPy to allow grad/jit/vmap through T.
    * formula_matrix is fixed, built from df_molname.
    """
    # Species / formula matrix (fixed)
    df_molname = load_molname_ykb4()
    formula_matrix_np, elems, species = build_formula_matrix(df_molname)
    # Keep the matrix fixed as requested, but move to device
    formula_matrix = jnp.asarray(formula_matrix_np)
    
    # Reference elemental solar abundance b from AAG21 (from exojax.utils.zsol import nsol)
    # AAG21 = Asplund, M., Amarsi, A. M., & Grevesse, N. 2021, arXiv:2105.01661
    b_element_vector_ref = jnp.array(
        [
            2.6627135e-04,
            9.2326087e-01,
            7.5739846e-02,
            1.0847369e-07,
            6.2420098e-05,
            1.5322316e-06,
            4.5219361e-04,
            2.3731458e-07,
            1.2170949e-05,
            8.6163716e-08,
            7.3337216e-09,
            0.0,
        ]
    )
    # Gibbs matrices -> (molecules, T_table, mu_table, grid_lens)
    path = get_data_filepath(JANAF_GIBBS_MATRICES_YKB4)
    gibbs_matrices = np.load(path, allow_pickle=True)["arr_0"].item()
    molecules, T_table_np, mu_table_np, grid_lens = extract_and_pad_gibbs_data(
        gibbs_matrices
    )

    # Move interpolation tables to device so autograd can see/track T properly
    T_table = jnp.asarray(T_table_np)
    mu_table = jnp.asarray(mu_table_np)

    # Compute robust temperature overlap across species to avoid OOB interpolation
    Tmin_np, Tmax_np = robust_temperature_range(T_table_np)
    Tmin = jnp.asarray(Tmin_np)
    Tmax = jnp.asarray(Tmax_np)

    # Define a JAX-differentiable h-vector function
    # hvector_func(T): R -> R^K
    # IMPORTANT:
    #   * No Python-side conditionals on T
    #   * All math stays within JAX space
    #   * Returns a DeviceArray so grad/jit can propagate
    def hvector_func(T: Union[float, jnp.ndarray]) -> jnp.ndarray:
        T = jnp.asarray(T)
        # Clamp to shared valid range to avoid NaN/Inf from OOB
        T_clamped = jnp.clip(T, Tmin, Tmax)
        return interpolate_hvector_all(T_clamped, T_table, mu_table)

    # JIT-compile once (optional but helps in loops)
    hvector_func_jit = jax.jit(hvector_func)

    return ChemicalSetup(
        formula_matrix=formula_matrix,
        hvector_func=hvector_func_jit,
        elems=tuple(elems) if elems is not None else None,
        species=tuple(species) if species is not None else None,
        b_element_vector_reference=b_element_vector_ref,
        metadata={"source": "JANAF"},
    )
