

def test_chemsetup():
    from exogibbs.presets.fastchem_cond import chemsetup
    cond = chemsetup()
    print(cond)
    
if __name__ == "__main__":
    test_chemsetup()