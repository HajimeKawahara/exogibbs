"""Guard the HCN-only intervention, identifiers and mass-action attribution."""

import hashlib
import json
from pathlib import Path
import runpy
import zipfile

import jax
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
STUDY = runpy.run_path(str(ROOT / "mceg/reviews/fastchem_hcn_20261028.py"))


def test_archive_scope_retains_raw_hoeijmakers_rename():
    audit = STUDY["audit_archives"](STUDY["PREVIOUS"], STUDY["UPDATED"])
    for record in audit.values():
        assert record["changed_coefficients"] == [STUDY["HCN"]]
        assert not record["changed_compositions"]
    assert audit["logK.dat"]["raw_added"] == []
    renamed = audit["logK_hoeijmakers2019.dat"]
    assert renamed["raw_added"] == ["C1H1N1_hcn"]
    assert renamed["raw_removed"] == ["C1H1N1_1"]
    assert renamed["explicit_aliases"] == {"C1H1N1_1": "C1H1N1_hcn"}


def test_reject_unreviewed_catalog_change(tmp_path):
    path = tmp_path / "wrong.zip"
    with zipfile.ZipFile(STUDY["UPDATED"]) as source, zipfile.ZipFile(path, "w") as output:
        for name in source.namelist():
            output.writestr(name, source.read(name).replace(b"C1H1N1_hcn", b"C1H1N1_unreviewed"))
    with pytest.raises(ValueError, match="Unexpected update scope"):
        STUDY["audit_archives"](STUDY["PREVIOUS"], path)


def test_only_hcn_potential_changes_and_bad_attribution_is_rejected():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        models = STUDY["setups"]()
        setup = models["curated"]
        t = np.array([2500., 3400., 4000.])
        h = {m: np.asarray(s.hvector_func(t)) for m, s in models.items()}
        j = setup.species.index(STUDY["HCN"])
        other = np.arange(len(setup.species)) != j
        np.testing.assert_array_equal(h["previous"][:, other], h["updated"][:, other])
        assert np.exp(h["curated"][0, j] - h["updated"][0, j]) == pytest.approx(.9673632060982, rel=1e-11)
        for s in models.values():
            assert s.species == setup.species
            np.testing.assert_array_equal(s.formula_matrix, setup.formula_matrix)
            np.testing.assert_array_equal(s.element_vector_reference, setup.element_vector_reference)
        with pytest.raises(ValueError, match="factorization failed"):
            STUDY["decompose"](setup, np.zeros_like(h["curated"]), h["previous"] - h["updated"])
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_recorded_comparison_inputs_and_numerical_gates():
    report = json.loads((ROOT / "tests/data/validation/new_fastchem_hcn/summary.json").read_text())
    for name, expected in report["selected_file_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    assert report["cases_passed"] == 966
    for audit in report["closure"]:
        STUDY["require_closure"](audit)
    assert report["max_HCN_factorization_residual"] < 1e-8
    sample = report["HCN_at_2500K_0p01bar"][1]
    assert sample["K_ratio"] * sample["atom_product"] == pytest.approx(sample["abundance_ratio"], rel=1e-10)
