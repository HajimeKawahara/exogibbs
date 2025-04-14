from exogibbs.io.load_data import get_data_filepath
from exogibbs.io.load_data import load_molname
from exogibbs.io.load_data import load_formula_matrix


def get_data_filename_existing_file_test():
    import os
    filename = "testdata.dat"
    fullpath = get_data_filepath(filename)

    assert os.path.exists(fullpath)

def load_molname_test():
    load_molname()

def load_formula_matrix_test():
    load_formula_matrix()


if __name__ == "__main__":
    get_data_filename_existing_file_test()
    load_molname_test()
    load_formula_matrix_test()