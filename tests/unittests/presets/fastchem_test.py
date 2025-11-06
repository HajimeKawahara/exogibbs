

def test_chemsetup():
    from exogibbs.presets.fastchem import chemsetup
    gas = chemsetup()
    
    assert len(gas.elements) == 28
    assert len(gas.species) == 523

if __name__ == "__main__":
    test_chemsetup()