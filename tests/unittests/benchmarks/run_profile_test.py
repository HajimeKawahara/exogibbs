from pathlib import Path
import sys

import pytest

REPO_DIR = Path(__file__).resolve().parents[3]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(REPO_DIR / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "src"))

from benchmarks.run_profile import build_parser
from benchmarks.run_profile import _resolve_default_grid_kind
from benchmarks.run_profile import _resolve_initializer_config
from exogibbs.api import get_default_equilibrium_grid_path


def test_build_parser_accepts_initializer_cli_options():
    parser = build_parser()

    args = parser.parse_args([
        "--method",
        "vmap_cold",
        "--initializer-mode",
        "grid",
        "--grid-path",
        "examples/grids/tmp_grid_check_extended/grid_exogibbs_extended.nc",
        "--grid-preset-name",
        "fastchem",
    ])

    assert args.initializer_mode == "grid"
    assert args.grid_path == Path("examples/grids/tmp_grid_check_extended/grid_exogibbs_extended.nc")
    assert args.grid_preset_name == "fastchem"


def test_resolve_initializer_config_preserves_default_behavior():
    initializer_mode, grid_path, grid_preset_name = _resolve_initializer_config(
        "none",
        None,
        "fastchem",
    )

    assert initializer_mode == "none"
    assert grid_path is None
    assert grid_preset_name is None


def test_resolve_initializer_config_allows_packaged_default_grid_for_grid_mode():
    initializer_mode, grid_path, grid_preset_name = _resolve_initializer_config(
        "grid",
        None,
        "fastchem",
    )

    assert initializer_mode == "grid"
    assert grid_path is None
    assert grid_preset_name == "fastchem"


def test_resolve_initializer_config_rejects_grid_path_for_none_mode():
    with pytest.raises(ValueError, match="--grid-path may be used only"):
        _resolve_initializer_config("none", Path("grid.nc"), "fastchem")


def test_resolve_default_grid_kind_uses_profile_case_preset():
    assert _resolve_default_grid_kind("fastchem_extended_profile_64layer_monotonic_reference") == "fastchem_extended"


def test_public_default_grid_helper_returns_extended_packaged_grid():
    grid_path = get_default_equilibrium_grid_path("fastchem_extended")

    assert grid_path.name == "grid_exogibbs_extended.nc"
