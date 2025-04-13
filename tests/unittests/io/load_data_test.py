from exogibbs.io.load_data import get_data_filename


def get_data_filename_existing_file_test():
    import os
    filename = "testdata.dat"
    fullpath = get_data_filename(filename)

    assert os.path.exists(fullpath)


#def test_load_matrix():

if __name__ == "__main__":
    get_data_filename_existing_file_test()
    