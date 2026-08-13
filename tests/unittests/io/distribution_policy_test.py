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
