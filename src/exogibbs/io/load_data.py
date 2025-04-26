from __future__ import annotations

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, Optional


TESTDATA_DIR = "data/"
MOLNAME_V3 = "molname_v3.dat"
FORMULA_MATRIX_V3 = "matrix_v3.dat"
JANAF_SAMPLE = "janaf_raw_sample.txt"


def get_data_filepath(filename):
    """get the full path of the data file

    Args:
        filename (str): filename of the test data
        dirname (str): directory name of the test data  (default: "data/testdata/")

    Returns:
        str: full path of the test data file

    """
    from importlib.resources import files

    return files("exogibbs").joinpath(TESTDATA_DIR + filename)


def load_molname():
    """load the molecular name dataframe (df_molname)
    Returns:
        pd.DataFrame: molname dataframe
    """
    fullpath = get_data_filepath(MOLNAME_V3)
    df_molname = pd.read_csv(fullpath, sep="\t", header=None, dtype=str)
    df_molname.columns = [
        "Molecule",
        "color"
    ]
    return df_molname


def load_formula_matrix():
    """loads the formula matrix data
    Returns:
        ndarray: formula matrix
    """
    fullpath = get_data_filepath(FORMULA_MATRIX_V3)
    df = pd.read_csv(fullpath, sep="\t", header=None, dtype=int)
    fm_np = np.array(df).T
    return fm_np


def load_JANAF_rawtxt(filename):
    """loads a JANAF raw text file

    Args:
        filename (str): filename of the JANAF raw text file, e.g. 'H2(g).txt'

    Returns:
        pd.DataFrame: DataFrame containing the data from the JANAF raw text file
    """

    def _convert_value(x):
        if x.strip() == "INFINITE":
            return np.inf
        try:
            return float(x)
        except ValueError:
            return x

    df = pd.read_csv(
        filename,
        sep="\t",
        skiprows=1,
        converters={i: _convert_value for i in range(8)},
    )
    return df





def load_JANAF_molecules(
    df_molname: pd.DataFrame,
    path_JANAF_data: str | Path,
    *,
    tag: str = "(g)",
    save_hdf5: Optional[str] = None,
    hdf5_compression: str = "zlib",
) -> Dict[str, pd.DataFrame]:
    """Load JANAF tables for all molecules listed in ``df_molname`` and return them
    as a dict ``{molecule: DataFrame}``.  Each DataFrame keeps its original
    (row, col) shape, which is convenient for later conversion to JAX arrays.

    Args:
        df_molname : pd.DataFrame
            Must contain a column named ``"Molecule"`` with file prefixes.
        path_JANAF_data : str or Path
            Directory that holds ``<molecule>(g).txt`` files.
        tag : str
            tag for JANAF file default to "(g)" 
        save_hdf5 : str, optional
            If given, the dict is additionally written to an HDF5 file whose
            *key* is the molecule name (e.g. ``/H2O``).  Passing ``None`` skips
            on-disk storage.
        hdf5_compression : {"zlib", "blosc", ...}, default "zlib"
            Compression algorithm for the HDF5 *table* format.

    Returns:
        dict[str, pd.DataFrame] Mapping from molecule name to its JANAF table.

    Notes:
        * Missing files or read errors are skipped with a warning.
        * Use ``jax.numpy.asarray(df.to_numpy())`` when you need a JAX array.

    Examples:
        >>> df_molname = load_molname()
        >>> path_JANAF_data = "/home/kawahara/thermochemical_equilibrium/Equilibrium/JANAF"
        >>> matrices = load_JANAF_molecules(df_molname, path_JANAF_data)
        >>> mat = matrices["C1O2"].to_numpy()
        
    """
    path_JANAF_data = Path(path_JANAF_data).expanduser().resolve()

    matrices: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ #
    # load every molecule into an in-memory dict                         #
    # ------------------------------------------------------------------ #
    for mol in df_molname["Molecule"]:
        file_path = path_JANAF_data / Path(mol+tag+".txt")
        if not file_path.is_file():
            warnings.warn(f"Missing file: {file_path}", RuntimeWarning)
            continue

        try:
            df_single = load_JANAF_rawtxt(file_path)  # -> pd.DataFrame
        except Exception as exc:                      # pragma: no cover
            warnings.warn(f"Failed to load {file_path}: {exc}", RuntimeWarning)
            continue

        matrices[mol] = df_single

    if not matrices:
        raise RuntimeError("No JANAF files were successfully loaded.")

    # ------------------------------------------------------------------ #
    # optional on-disk storage (HDF5, one key per molecule)              #
    # ------------------------------------------------------------------ #
    if save_hdf5 is not None:
        with pd.HDFStore(save_hdf5, mode="w", complevel=9, complib=hdf5_compression) as store:
            for mol, df in matrices.items():
                store.put(f"/{mol}", df, format="table")

    return matrices


        
if __name__ == "__main__":
    df_molname = load_molname()
    path_JANAF_data = "/home/kawahara/thermochemical_equilibrium/Equilibrium/JANAF"
    matrices = load_JANAF_molecules(df_molname, path_JANAF_data)
    mat = matrices["C1O2"].to_numpy()
    