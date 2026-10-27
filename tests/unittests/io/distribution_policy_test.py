from pathlib import Path
import re

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_unvalidated_sharp_huebner_data_is_excluded_from_distributions():
    manifest_lines = (REPOSITORY_ROOT / "MANIFEST.in").read_text().splitlines()
    assert "recursive-exclude src/exogibbs/data/sharp_huebner *" in manifest_lines

    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text()
    assert "[tool.setuptools.exclude-package-data]" in pyproject
    assert 'exogibbs = ["data/sharp_huebner/*"]' in pyproject


def test_documentation_submodule_is_excluded_from_distributions():
    manifest_lines = (REPOSITORY_ROOT / "MANIFEST.in").read_text().splitlines()
    assert "prune doc_ExoGibbs" in manifest_lines


def test_source_manifest_keeps_fixtures_and_prunes_generated_results():
    file_list = pytest.importorskip("setuptools._distutils.filelist").FileList()
    fixtures = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (REPOSITORY_ROOT / "tests/data/validation").rglob("*")
        if path.is_file()
    ]
    generated = ["results/run.json", "results/nested/plot.png"]
    # Simulate tracked or stale generated files discovered by the build backend.
    file_list.allfiles = fixtures + generated
    file_list.extend(file_list.allfiles)
    for line in (REPOSITORY_ROOT / "MANIFEST.in").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            file_list.process_template_line(line)

    assert fixtures
    assert set(fixtures) <= set(file_list.files)
    assert not set(generated) & set(file_list.files)


def test_documentation_validation_assets_exist():
    documents = REPOSITORY_ROOT / "documents"
    references = re.compile(
        r"(?:\.\./)+tests/data/validation/[^\s>`]+"
        r"|_static/equilibrium_sensitivity/[^\s>`]+"
    )
    assets = []
    for page in [*documents.glob("*.rst"), *documents.glob("presets/*.rst"),
                 *documents.glob("mceg/*.rst")]:
        assets.extend(page.parent / name for name in references.findall(page.read_text()))
    assert assets
    for path in assets:
        assert path.is_file(), path


def test_pyfastchem_is_an_optional_dependency():
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text()
    core_dependencies = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
    optional_dependencies = pyproject.split(
        "[project.optional-dependencies]", 1
    )[1]
    fastchem_dependencies = optional_dependencies.split(
        "fastchem = [", 1
    )[1].split("]", 1)[0]

    assert "pyfastchem" not in core_dependencies
    assert '"pyfastchem"' in fastchem_dependencies
