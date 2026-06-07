from importlib import import_module

import pytest

import exogibbs


def test_package_version_matches_scm_generated_module():
    try:
        exogibbs_version = import_module("exogibbs.exogibbs_version")
    except ImportError:
        pytest.skip("setuptools-scm generated version module is unavailable")

    assert exogibbs.__version__ == exogibbs_version.__version__


def test_load_version_falls_back_to_installed_metadata(monkeypatch):
    def _raise_import_error():
        raise ImportError("generated version module unavailable")

    monkeypatch.setattr(exogibbs, "_load_generated_version", _raise_import_error)
    monkeypatch.setattr(exogibbs, "_pkg_version", lambda name: "9.9.9")

    assert exogibbs._load_version() == "9.9.9"
