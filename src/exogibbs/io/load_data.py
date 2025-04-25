import pandas as pd
import numpy as np
import pathlib

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
    """load the molecular name data
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
    """loads the JANAF raw text file

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


if __name__ == "__main__":
    # Test the functions
    # df = load_formula_matrix()
    filename = get_data_filepath(JANAF_SAMPLE)
    df = load_JANAF_rawtxt(filename)
    print(df)
