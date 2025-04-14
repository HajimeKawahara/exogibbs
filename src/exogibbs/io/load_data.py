import pandas as pd
import numpy as np

TESTDATA_DIR = "data/"
MOLNAME_V3 = "molname_v3.dat"
FORMULA_MATRIX_V3 = "matrix_v3.dat"

def get_data_filepath(filename):
    """get the full path of the data file

    Args:
        filename (str): filename of the test data
        dirname (str): directory name of the test data  (default: "data/testdata/")

    Returns:
        str: full path of the test data file

    """
    from importlib.resources import files
    return files('exogibbs').joinpath(TESTDATA_DIR + filename)

def load_molname():
    """load the molname data
    Returns:
        pd.DataFrame: molname data
    """
    fullpath = get_data_filepath(MOLNAME_V3)
    df = pd.read_csv(fullpath, sep="\t")
    return df


def load_formula_matrix():
    """load the formula matrix data
    Returns:
        ndarray: formula matrix
    """
    fullpath = get_data_filepath(FORMULA_MATRIX_V3)
    df = pd.read_csv(fullpath, sep="\t", header=None, dtype=int)
    fm_np = np.array(df).T
    return fm_np

if __name__ == "__main__":
    # Test the functions
    df = load_formula_matrix()
        