TESTDATA_DIR = "data/"


def get_data_filename(filename):
    """get the full path of the data file

    Args:
        filename (str): filename of the test data
        dirname (str): directory name of the test data  (default: "data/testdata/")

    Returns:
        str: full path of the test data file

    """
    from importlib.resources import files
    return files('exogibbs').joinpath(TESTDATA_DIR + filename)
