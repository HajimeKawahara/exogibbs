from exogibbs.io.load_data import get_data_filepath
from exogibbs.io.load_data import load_molname
from exogibbs.io.load_data import load_formula_matrix
from exogibbs.io.load_data import load_JANAF_rawtxt
from exogibbs.io.load_data import load_JANAF_molecules
from exogibbs.io.load_data import JANAF_SAMPLE

def test_get_data_filename_existing_file():
    import os
    filename = "testdata.dat"
    fullpath = get_data_filepath(filename)

    assert os.path.exists(fullpath)

def test_load_molname():
    df = load_molname()
    
def test_load_formula_matrix():
    load_formula_matrix()

def test_load_JANAF_rawtxt():
    filename = get_data_filepath(JANAF_SAMPLE)
    load_JANAF_rawtxt(filename)
    
def test_load_JANAF_molecules():
    import pandas as pd
    df_molecules = pd.DataFrame({
        "Molecule": ["janaf_raw"],
    })
    filepath = get_data_filepath("")
    
    matrices = load_JANAF_molecules(df_molecules, filepath, tag="_sample")
    
    assert matrices["janaf_raw"].shape == (10,8)
    
if __name__ == "__main__":
    test_get_data_filename_existing_file()
    test_load_molname()
    test_load_formula_matrix()
    test_load_JANAF_rawtxt()
    test_load_JANAF_molecules()