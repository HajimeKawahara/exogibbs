from exogibbs.utils.janafinfo import reference_species_dict


def test_load_ref_dict_parses_reference_tokens():

    ref, nu = reference_species_dict()

    assert ref["Al"] == "Al1"
    assert ref["O"] == "O2"
    assert ref["e-"] == "e1-"
    assert nu["Al"] == 1
    assert nu["O"] == 2
    assert nu["e-"] == 1
