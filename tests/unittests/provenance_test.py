"""Keep optional evidence discoverable without changing runtime dependencies."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import numpy as np
import pytest

from exogibbs.presets import curated, curated_c2h, curated_co2, curated_h2o, curated_tio, curated_v1
from exogibbs.provenance import describe, evidence_catalog


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def setup():
    return curated.chemsetup()


def test_catalog_is_reproducible_and_retains_archived_records(tmp_path):
    builder = runpy.run_path(str(ROOT / "mceg/build_catalog.py"))
    builder["build_catalog"](tmp_path)
    packaged = evidence_catalog().parent
    for generated in tmp_path.rglob("*"):
        if generated.is_file():
            assert generated.read_bytes() == (packaged / generated.relative_to(tmp_path)).read_bytes()
    catalog = json.loads(evidence_catalog().read_text())
    model = catalog["models"][0]
    assert model["identity"]["settings"] == {"ions": True}
    species = {s["id"]: s["aliases"] for s in model["species"]}
    assert species["C1H1N1_hcn"] == ["HCN"]
    assert species["C1H1N1_hnc"] == ["HNC"]
    for item in catalog["models"]:
        for record in item["records"]:
            for pin in [record, *record.get("artifacts", [])]:
                assert hashlib.sha256((packaged / pin["path"]).read_bytes()).hexdigest() == pin["sha256"]
            assert (packaged / record["path"]).read_bytes() == (ROOT / "mceg" / record["path"]).read_bytes()
    hcn = model["records"][1]
    assert hcn["species"] == ["C1H1N1_hcn"]
    assert "-4.38%" in hcn["summary"] and "-4.67%" in hcn["summary"]
    assert "not accuracy rankings" in hcn["summary"]
    assert "Your conditions have not been evaluated" in hcn["summary"]


def test_c2h_correction_has_separate_evidence_identity(setup):
    corrected = curated_c2h.chemsetup()
    description = describe(corrected)
    assert description["model_id"] == "exogibbs.curated_c2h.v1"
    assert describe(setup)["model_id"] == "exogibbs.curated_h2o.v1"
    assert describe(curated_v1.chemsetup())["model_id"] == "exogibbs.curated_gas.v1"
    catalog = json.loads(evidence_catalog().read_text())
    model = next(m for m in catalog["models"] if m["id"] == description["model_id"])
    assert [r["id"] for r in model["records"]] == ["c2h_correction"]
    assert "whole_mixture_accuracy_review" in model["records"][0]["assessments"]
    assert "unquantified" in model["records"][0]["summary"]
    assert describe(curated_c2h.chemsetup(ions=False))["record_match"] == "no_matching_record"



def test_tio_adoption_retains_its_versioned_evidence_identity():
    setup = curated_tio.chemsetup()
    assert describe(setup)["model_id"] == "exogibbs.curated_tio.v1"
    catalog = json.loads(evidence_catalog().read_text())
    model = next(m for m in catalog["models"] if m["id"] == "exogibbs.curated_tio.v1")
    assert model["identity"]["data_sha256"] == setup.metadata["data_sha256"]
    assert model["dataset"] == "curated-tio-v1"
    record = model["records"][0]
    assert record["id"] == "tio_adoption"
    assert "physical_accuracy_review" in record["assessments"]
    assert "unquantified" in record["summary"]
    assert any(s["id"] == "O1Ti1" and "TiO" in s["aliases"] for s in model["species"])
    assert "Former default" in record["summary"]
    assert describe(curated_tio.chemsetup(ions=False))["record_match"] == "no_matching_record"


def test_co2_adoption_retains_its_versioned_evidence_identity():
    setup = curated_co2.chemsetup()
    assert describe(setup)["model_id"] == "exogibbs.curated_co2.v1"
    catalog = json.loads(evidence_catalog().read_text())
    model = next(m for m in catalog["models"] if m["id"] == "exogibbs.curated_co2.v1")
    assert model["identity"]["data_sha256"] == setup.metadata["data_sha256"]
    assert model["dataset"] == "curated-co2-v1"
    record = model["records"][0]
    assert record["id"] == "co2_adoption"
    assert record["assessments"] == ["scientific_selection_review", "production_adoption_review",
                                     "physical_accuracy_review"]
    assert "physical accuracy remain unknown" in record["summary"]
    assert "Former default" in record["summary"]
    assert any(s["id"] == "C1O2" and "CO2" in s["aliases"] for s in model["species"])
    assert describe(curated_co2.chemsetup(ions=False))["record_match"] == "no_matching_record"


def test_h2o_adoption_has_separate_evidence_identity(setup):
    assert curated.chemsetup is curated_h2o.chemsetup
    catalog = json.loads(evidence_catalog().read_text())
    model = next(m for m in catalog["models"] if m["id"] == "exogibbs.curated_h2o.v1")
    assert model["identity"]["data_sha256"] == setup.metadata["data_sha256"]
    assert model["dataset"] == "curated-h2o-v1"
    record = model["records"][0]
    assert record["id"] == "h2o_adoption"
    assert record["assessments"] == ["scientific_selection_review", "production_adoption_review",
                                     "physical_accuracy_review"]
    assert "physical atmospheric accuracy remain unknown" in record["summary"]
    assert "not relabeled as joint results" in record["summary"]
    assert any(s["id"] == "H2O1" and "H2O" in s["aliases"] for s in model["species"])
    assert describe(curated_h2o.chemsetup(ions=False))["record_match"] == "no_matching_record"


def test_describe_identifies_only_declared_metadata(setup):
    description = describe(setup)
    assert description["model_id"] == "exogibbs.curated_h2o.v1"
    assert description["record_match"] == "declared_metadata"
    assert "does not verify a custom setup" in description["notice"]
    assert json.loads(json.dumps(description)) == description
    # Metadata alone cannot certify a changed callable: the report says so.
    modified = replace(setup, hvector_func=lambda temperature: None)
    assert describe(modified) == description


def test_builder_does_not_rebind_historical_study_to_changed_data(tmp_path, monkeypatch):
    builder = runpy.run_path(str(ROOT / "mceg/build_catalog.py"))["build_catalog"]
    source = tmp_path / "source"
    for name in (
        "mceg/contracts/curated_gas.yaml", "mceg/contracts/fastchem_hcn_followup.yaml",
        "tests/data/validation/new_fastchem_hcn/summary.json",
        "mceg/reviews/fastchem_hcn_sources_20261028.json",
        "src/exogibbs/data/curated/gas_v1.json",
    ):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    data_path = source / "src/exogibbs/data/curated/gas_v1.json"
    data = json.loads(data_path.read_text())
    data["id"] = "unreviewed-replacement"
    data_path.write_text(json.dumps(data))
    monkeypatch.setitem(builder.__globals__, "ROOT", source)
    with pytest.raises(ValueError, match="review new evidence"):
        builder(tmp_path / "output")


@pytest.mark.parametrize("name", ["src/exogibbs/data/curated/co2_v1.json",
                                 "tests/data/validation/curated_co2_adoption/20261104/summary.json",
                                 "src/exogibbs/data/curated/h2o_v1.json",
                                 "tests/data/validation/curated_h2o_adoption/20261104/summary.json"])
def test_builder_rejects_changed_adoption_bytes(tmp_path, monkeypatch, name):
    builder = runpy.run_path(str(ROOT / "mceg/build_catalog.py"))["build_catalog"]
    original = Path.read_bytes

    def changed(path):
        content = original(path)
        return content + b"\n" if path == ROOT / name else content

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError, match="CO2|H2O"):
        builder(tmp_path)


@pytest.mark.parametrize("changes", [
    {"ions": False}, {"ions": 1}, {"data_sha256": "different"}, {"dataset": "custom"},
    {"ions": np.bool_(True)}, {"dataset": Path("custom")},
])
def test_different_network_or_data_does_not_inherit_review(setup, changes):
    description = describe(replace(setup, metadata={**setup.metadata, **changes}))
    assert description["record_match"] == "no_matching_record"
    assert description["model_id"] is None
    assert description["explain_command"] is None
    assert json.loads(json.dumps(description)) == description


def test_missing_metadata_has_no_inferred_reliability(setup):
    description = describe(replace(setup, metadata=None))
    assert description["record_match"] == "no_matching_record"
    assert "does not establish physical reliability" in description["notice"]


def test_preset_solve_and_provenance_work_with_mceg_imports_blocked():
    script = '''
import importlib.abc
import sys
class BlockMCEG(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mceg" or fullname.startswith("mceg."):
            raise AssertionError("MCEG must remain optional")
sys.meta_path.insert(0, BlockMCEG())
from exogibbs.presets import curated
from exogibbs.api.gas import solve
from exogibbs.provenance import describe
import numpy as np
setup = curated.chemsetup()
result, diagnostics = solve(setup, 2500., .01, setup.element_vector_reference,
                            return_diagnostics=True)
assert bool(diagnostics["converged"])
assert np.isclose(np.asarray(result.x).sum(), 1.)
assert describe(setup)["record_match"] == "declared_metadata"
assert "mceg" not in sys.modules
'''
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "JAX_PLATFORMS": "cpu",
           "JAX_ENABLE_X64": "1"}
    subprocess.run([sys.executable, "-c", script], env=env, check=True,
                   capture_output=True, text=True, timeout=120)
