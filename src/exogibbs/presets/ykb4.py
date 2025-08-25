from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.equilibrium.gibbs import extract_and_pad_gibbs_data
from exogibbs.equilibrium.gibbs import interpolate_hvector_all
from exogibbs.io.load_data import load_molname
from exogibbs.io.load_data import get_data_filepath
from exogibbs.io.load_data import DEFAULT_JANAF_GIBBS_MATRICES
from exogibbs.io.load_data import NUMBER_OF_SPECIES_SAMPLE
from exogibbs.thermo.stoichiometry import build_formula_matrix
from typing import Union
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax


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
    df_molname = load_molname()
    formula_matrix_np, elems, species = build_formula_matrix(df_molname)
    # Keep the matrix fixed as requested, but move to device
    formula_matrix = jnp.asarray(formula_matrix_np)

    # Element abundance b from the provided sample number densities
    npath = get_data_filepath(NUMBER_OF_SPECIES_SAMPLE)
    number_of_species_init = pd.read_csv(npath, header=None, sep=",").values[0]
    b_element_vector_np = formula_matrix_np @ number_of_species_init
    b_element_vector = jnp.asarray(b_element_vector_np)

    # Gibbs matrices -> (molecules, T_table, mu_table, grid_lens)
    path = get_data_filepath(DEFAULT_JANAF_GIBBS_MATRICES)
    gibbs_matrices = np.load(path, allow_pickle=True)["arr_0"].item()
    molecules, T_table_np, mu_table_np, grid_lens = extract_and_pad_gibbs_data(
        gibbs_matrices
    )

    # Move interpolation tables to device so autograd can see/track T properly
    T_table = jnp.asarray(T_table_np)
    mu_table = jnp.asarray(mu_table_np)

    # Define a JAX-differentiable h-vector function
    # hvector_func(T): R -> R^K
    # IMPORTANT:
    #   * No Python-side conditionals on T
    #   * All math stays within JAX space
    #   * Returns a DeviceArray so grad/jit can propagate
    def hvector_func(T: Union[float, jnp.ndarray]) -> jnp.ndarray:
        T = jnp.asarray(T)
        return interpolate_hvector_all(T, T_table, mu_table)

    # JIT-compile once (optional but helps in loops)
    hvector_func_jit = jax.jit(hvector_func)

    return ChemicalSetup(
        formula_matrix=formula_matrix,
        b_element_vector=b_element_vector,
        hvector_func=hvector_func_jit,
        elems=tuple(elems) if elems is not None else None,
        species=tuple(species) if species is not None else None,
        metadata={"source": "JANAF"},
    )
