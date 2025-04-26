from exogibbs.io.load_data import get_data_filepath
from exogibbs.io.load_data import load_molname
from exogibbs.io.load_data import load_formula_matrix
from exogibbs.io.load_data import load_JANAF_rawtxt
from exogibbs.io.load_data import load_JANAF_molecules
from exogibbs.io.load_data import JANAF_SAMPLE

def get_data_filename_existing_file_test():
    import os
    filename = "testdata.dat"
    fullpath = get_data_filepath(filename)

    assert os.path.exists(fullpath)

def load_molname_test():
    df = load_molname()
    
def load_formula_matrix_test():
    load_formula_matrix()

def load_JANAF_rawtxt_test():
    filename = get_data_filepath(JANAF_SAMPLE)
    load_JANAF_rawtxt(filename)
    
def load_JANAF_molecules_test():
    import pandas as pd
    df_molecules = pd.DataFrame({
        "Molecule": ["janaf_raw"],
    })
    filepath = get_data_filepath("")
    
    matrices = load_JANAF_molecules(df_molecules, filepath, tag="_sample")
    
    assert matrices["janaf_raw"].shape == (10,8)
    
if __name__ == "__main__":
    get_data_filename_existing_file_test()
    load_molname_test()
    load_formula_matrix_test()
    load_JANAF_rawtxt_test()
    load_JANAF_molecules_test()