"""Check the preliminary-data comparison's catalog, source scope and closure."""

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import jax
import numpy as np
import pytest
from scipy.special import logsumexp

from exogibbs.presets import fastchem4


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "tests/data/validation/new_fastchem_logk"
SPEC = importlib.util.spec_from_file_location(
    "new_fastchem_logk", ROOT / "examples/comparisons/compare_new_fastchem_logk.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("new_fastchem_logk")
    with zipfile.ZipFile(RESULTS / "logK.zip") as archive:
        for name in ("logK.dat", "logK_extended.dat", "logK_hoeijmakers2019.dat"):
            (directory / name).write_bytes(archive.read(name))
    (directory / "logK.zip").write_bytes((RESULTS / "logK.zip").read_bytes())
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield directory, study.comparison_setups(directory / "logK.dat")
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_fixed_catalog_budget_and_original_table(inputs):
    _, setups = inputs
    default = setups["curated"]
    assert len(default.species) == 537
    assert "N1-" not in default.species and "Zn1-" not in default.species
    for setup in setups.values():
        assert setup.species == default.species and setup.elements == default.elements
        np.testing.assert_array_equal(setup.formula_matrix, default.formula_matrix)
        np.testing.assert_array_equal(setup.element_vector_reference, default.element_vector_reference)
        assert setup.metadata["relative_charge_convergence"] is True
    base = fastchem4.chemsetup(path="FastChem4/logK/logK.dat", silent=True)
    columns = [base.species.index(s) for s in default.species]
    temperature = np.array([1500., 2517., 4000.])
    np.testing.assert_allclose(setups["original"].hvector_func(temperature),
                               np.asarray(base.hvector_func(temperature))[:, columns], atol=1e-12, rtol=0)
    matrix = np.asarray(default.formula_matrix)
    charge_row = default.elements.index("e-")
    atomic_ions = (np.delete(matrix, charge_row, axis=0).sum(axis=0) == 1) & (matrix[charge_row] != 0)
    assert np.count_nonzero(atomic_ions) == 47
    np.testing.assert_array_equal(np.asarray(setups["new"].hvector_func(temperature))[:, ~atomic_ions],
                                  np.asarray(setups["original"].hvector_func(temperature))[:, ~atomic_ions])


def test_archived_input_changes_only_atomic_ions(inputs):
    directory, _ = inputs
    audit = study.audit_input_files(directory)
    for name, count in (("logK.dat", 47), ("logK_extended.dat", 79), ("logK_hoeijmakers2019.dat", 228)):
        assert len(audit[name]["changed_coefficients"]) == count
        assert audit[name]["changed_coefficients"] == audit[name]["changed_atomic_ions"]
        assert not any(audit[name][k] for k in ("added", "removed", "changed_compositions", "changed_other_species"))


@pytest.mark.parametrize("profile", ("isothermal", "inverted"))
@pytest.mark.parametrize("model", study.MODELS)
def test_saved_columns_pass_independent_closure(inputs, profile, model):
    _, setups = inputs
    with np.load(RESULTS / f"{profile}.npz") as data:
        assert tuple(data["species"]) == setups[model].species
        assert len(data["pressure_bar"]) == 161
        study.require_closure(study.closure(setups[model], data[f"ln_n_{model}"],
                                            data["temperature_K"], data["pressure_bar"]))
        if model == "curated":
            with np.load(ROOT / "tests/data/validation/curated_atmosphere" / f"{profile}_curated.npz") as old:
                logx = data["ln_n_curated"] - logsumexp(data["ln_n_curated"], axis=-1)[:, None]
                eligible = np.maximum(logx, old["logx"]) >= np.log(study.SIGNIFICANT_X)
                # Historical JAX outputs differ at ppm scale for trace elements;
                # use the independent element-closure gate for populated species.
                np.testing.assert_allclose(logx[eligible], old["logx"][eligible], atol=1e-5, rtol=0)


def test_receipt_matches_archived_inputs_and_results():
    receipt = json.loads((RESULTS / "receipt.json").read_text())
    assert receipt["cases_passed"] == 966
    for path, digest in receipt["selected_file_sha256"].items():
        content = (ROOT / path).read_bytes()
        if path == "examples/comparisons/compare_new_fastchem_logk.py":
            archived = (RESULTS / "source/compare_new_fastchem_logk.py").read_bytes()
            assert content == archived.replace(
                b'"results/new_fastchem_logk/logK.zip"',
                b'"tests/data/validation/new_fastchem_logk/logK.zip"')
            content = archived
        assert hashlib.sha256(content).hexdigest() == digest
    for path, digest in receipt["output_sha256"].items():
        assert hashlib.sha256((RESULTS / path).read_bytes()).hexdigest() == digest
