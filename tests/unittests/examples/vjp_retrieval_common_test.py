"""Dependency-light contracts for the shared gas VJP retrieval demos."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIRECTORY = REPOSITORY_ROOT / "examples" / "retrievals"
COMMON_PATH = EXAMPLE_DIRECTORY / "_exojax_nuts_common.py"


def _load_common():
    spec = importlib.util.spec_from_file_location(
        "exogibbs_vjp_retrieval_common_test_module",
        COMMON_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def common():
    return _load_common()


def test_database_guard_requires_complete_local_sources(common, tmp_path):
    database = tmp_path / "CO" / "12C-16O" / "Li2015"
    database.mkdir(parents=True)
    prefix = "12C-16O__Li2015"

    with pytest.raises(FileNotFoundError, match="offline demo"):
        common.require_local_co_database(database)

    for suffix in (".def", ".pf", ".states.bz2", ".trans.bz2"):
        (database / f"{prefix}{suffix}").write_text("local fixture\n")

    assert common.require_local_co_database(database) == database.resolve()


def test_mock_observation_is_normalized_and_deterministic(common):
    flux = jnp.asarray([2.0, 4.0, 6.0, 8.0])
    first = common.make_mock_observation(
        flux,
        seed=3,
        relative_noise=1.0e-3,
    )
    second = common.make_mock_observation(
        flux,
        seed=3,
        relative_noise=1.0e-3,
    )

    assert float(first.flux_scale) == pytest.approx(5.0)
    np.testing.assert_allclose(first.truth, flux / 5.0)
    np.testing.assert_allclose(first.observed, second.observed)


def test_quick_mode_bounds_sampling_and_problem_shape(common):
    args = argparse.Namespace(
        num_warmup=500,
        num_samples=1000,
        max_tree_depth=10,
        seed=7,
        no_progress_bar=True,
        quick=True,
        nlayer=24,
        nu_points=1024,
    )

    settings = common.resolve_run_settings(args)

    assert settings.num_warmup == 5
    assert settings.num_samples == 10
    assert settings.max_tree_depth == 4
    assert not settings.progress_bar
    assert common.resolve_demo_shape(args) == (8, 256)


def test_gas_wrappers_differ_only_in_initializer_selection():
    no_grid = (
        EXAMPLE_DIRECTORY / "exojax_nuts_gas_no_grid.py"
    ).read_text(encoding="utf-8")
    grid = (
        EXAMPLE_DIRECTORY / "exojax_nuts_gas_grid.py"
    ).read_text(encoding="utf-8")
    common_source = COMMON_PATH.read_text(encoding="utf-8")

    assert "use_grid_initializer=False" in no_grid
    assert "use_grid_initializer=True" in grid
    assert "forward_mode_differentiation=False" in common_source
    assert "initializer=initializer" in common_source
    assert "return_diagnostics=True" in common_source
    assert "jax.value_and_grad" in common_source
