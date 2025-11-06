def test_chemsetup():
    from exogibbs.presets.fastchem_cond import chemsetup
    cond = chemsetup()
    
    assert cond.species is not None
    assert "Al(s)" in cond.species
    assert len(cond.elements) == 24
    assert len(cond.species) == 186


    hvector = cond.hvector_func(1200.0)
    assert hvector.shape[0] == len(cond.species)

if __name__ == "__main__":
    test_chemsetup()