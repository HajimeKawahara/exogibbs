"""Verify pinned upstream source identity without a compiler or network."""

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/data/validation/phosphorus_cold"


def test_vendored_ggchem_sources_are_exact_upstream_blobs():
    manifest = json.loads((FIXTURE / "ggchem_build.json").read_text())
    source = FIXTURE / "ggchem_source.zip"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_zip_sha256"]
    with zipfile.ZipFile(source) as archive:
        assert set(archive.namelist()) == {r["path"] for r in manifest["files"]}
        assert {"LICENSE", "README.md"}.issubset(archive.namelist())
        for record in manifest["files"]:
            content = archive.read(record["path"])
            assert hashlib.sha256(content).hexdigest() == record["sha256"]
            blob = b"blob " + str(len(content)).encode() + b"\0" + content
            assert hashlib.sha1(blob).hexdigest() == record["git_blob_sha1"]


def test_recorded_adapter_matches_the_built_reference():
    manifest = json.loads((FIXTURE / "ggchem_build.json").read_text())
    driver = ROOT / "mceg/reviews/phosphorus_ggchem_driver.f90"
    assert hashlib.sha256(driver.read_bytes()).hexdigest() == manifest["adapter_sha256"]
    assert manifest["revision"] == "65bbb8f9ca3d6c0b4e58cb7dd3862322452d0d4b"
