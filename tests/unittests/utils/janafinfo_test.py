from exogibbs.utils.janafinfo import load_ref_dict
from exogibbs.io.load_data import get_data_filepath


def test_load_ref_dict_parses_reference_tokens():

    path_ref = get_data_filepath("janaf/ref.txt")
    ref, nu = load_ref_dict(path_ref)

    assert ref["Al"] == "Al1"
    assert ref["O"] == "O2"
    assert ref["e-"] == "e1-"
    assert nu["Al"] == 1
    assert nu["O"] == 2
    assert nu["e-"] == 1
