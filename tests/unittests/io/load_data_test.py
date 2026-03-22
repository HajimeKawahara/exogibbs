from pathlib import Path

import pytest

from exogibbs.api import get_default_equilibrium_grid_path as get_default_equilibrium_grid_path_api
import exogibbs.io.load_data as load_data_module
from exogibbs.io.load_data import get_data_filepath
from exogibbs.io.load_data import get_default_equilibrium_grid_path
from exogibbs.io.load_data import load_JANAF_rawtxt
from exogibbs.io.load_data import load_JANAF_molecules
from exogibbs.io.load_data import JANAF_SAMPLE
from exogibbs.io.load_data import JANAF_NAME_KEY


def test_get_data_filename_existing_file():
    import os

    filename = "test/testdata.dat"
    fullpath = get_data_filepath(filename)

    assert os.path.exists(fullpath)


def test_load_JANAF_rawtxt():
    filename = get_data_filepath(JANAF_SAMPLE)
    load_JANAF_rawtxt(filename)


def test_load_JANAF_molecules():
    import pandas as pd

    df_molecules = pd.DataFrame(
        {
            JANAF_NAME_KEY: ["janaf_raw"],
        }
    )
    filepath = get_data_filepath("test")
    matrices = load_JANAF_molecules(df_molecules, filepath, tag="_sample")

    assert matrices["janaf_raw"].shape == (10, 8)


@pytest.mark.parametrize(
    "kind,expected_name",
    [
        ("fastchem", "grid_exogibbs.nc"),
        ("fastchem_extended", "grid_exogibbs_extended.nc"),
    ],
)
def test_get_default_equilibrium_grid_path_returns_packaged_grid(kind, expected_name):
    grid_path = get_default_equilibrium_grid_path(kind)

    assert isinstance(grid_path, Path)
    assert grid_path.name == expected_name
    assert grid_path.is_file()


def test_get_default_equilibrium_grid_path_is_exported_via_public_api():
    assert get_default_equilibrium_grid_path_api("fastchem") == get_default_equilibrium_grid_path("fastchem")


def test_get_default_equilibrium_grid_path_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unknown default equilibrium grid kind 'unknown'"):
        get_default_equilibrium_grid_path("unknown")


def test_get_default_equilibrium_grid_path_raises_when_packaged_grid_missing(monkeypatch):
    monkeypatch.setitem(
        load_data_module._DEFAULT_EQUILIBRIUM_GRID_FILES,
        "missing_test_grid",
        "grids/fastchem/does_not_exist.nc",
    )

    with pytest.raises(FileNotFoundError, match="Packaged default equilibrium grid 'missing_test_grid' is missing"):
        get_default_equilibrium_grid_path("missing_test_grid")
