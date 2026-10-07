from pathlib import Path


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
