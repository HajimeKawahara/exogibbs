from exogibbs.io.load_data import get_data_filepath
from exogibbs.io.load_data import load_JANAF_rawtxt
from exogibbs.io.load_data import load_JANAF_molecules
from exogibbs.io.load_data import JANAF_SAMPLE
from exogibbs.io.load_data import JANAF_NAME_KEY


def test_get_data_filename_existing_file():
    import os

    filename = "test/testdata.dat"
    fullpath = get_data_filepath(filename)

    assert os.path.exists(fullpath)


def test_load_JANAF_rawtxt():
    filename = get_data_filepath(JANAF_SAMPLE)
    load_JANAF_rawtxt(filename)


def test_load_JANAF_molecules():
    import pandas as pd

    df_molecules = pd.DataFrame(
        {
            JANAF_NAME_KEY: ["janaf_raw"],
        }
    )
    filepath = get_data_filepath("")

    matrices = load_JANAF_molecules(df_molecules, filepath, tag="_sample")

    assert matrices["janaf_raw"].shape == (10, 8)
