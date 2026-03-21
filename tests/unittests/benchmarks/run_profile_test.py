from pathlib import Path
import sys

import pytest

REPO_DIR = Path(__file__).resolve().parents[3]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(REPO_DIR / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "src"))

from benchmarks.run_profile import build_parser
from benchmarks.run_profile import _resolve_initializer_config


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


def test_resolve_initializer_config_requires_grid_path_for_grid_mode():
    with pytest.raises(ValueError, match="--grid-path is required"):
        _resolve_initializer_config("grid", None, "fastchem")


def test_resolve_initializer_config_rejects_grid_path_for_none_mode():
    with pytest.raises(ValueError, match="--grid-path may be used only"):
        _resolve_initializer_config("none", Path("grid.nc"), "fastchem")
