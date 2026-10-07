"""Dependency-light contracts for the shared gas VJP retrieval demos."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

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
    assert settings.quick
    assert not settings.progress_bar
    assert common.resolve_demo_shape(args) == (8, 256)

    gas_settings = common.resolve_run_settings(
        args,
        quick_num_warmup=common.GAS_QUICK_NUM_WARMUP,
        quick_num_samples=common.GAS_QUICK_NUM_SAMPLES,
        quick_max_tree_depth=common.GAS_QUICK_MAX_TREE_DEPTH,
    )
    assert gas_settings.num_warmup == 100
    assert gas_settings.num_samples == 100
    assert gas_settings.max_tree_depth == 8

    args.num_samples = 1
    with pytest.raises(ValueError, match="at least two"):
        common.resolve_run_settings(args)
    args.num_samples = 1000
    with pytest.raises(ValueError, match="sample limit must be at least two"):
        common.resolve_run_settings(args, quick_num_samples=1)


def test_initialize_run_output_removes_only_known_stale_artifacts(
    common,
    tmp_path,
):
    known = (
        "mock_spectrum.npz",
        "posterior_samples.npz",
        "run_summary.json",
        "gas_grid_preflight.json",
    )
    for name in known:
        (tmp_path / name).write_bytes(b"stale")
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("user output\n")

    common.initialize_run_output(
        tmp_path,
        case_name="gas_grid",
        preflight_only=True,
    )

    assert all(not (tmp_path / name).exists() for name in known)
    assert unrelated.read_text() == "user output\n"
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["mode"] == "preflight"
    assert status["state"] == "started"
    assert not any(status["artifacts"].values())

    invalid_output = tmp_path / "invalid"
    invalid_output.mkdir()
    stale_posterior = invalid_output / "posterior_samples.npz"
    stale_posterior.write_bytes(b"stale")
    with pytest.raises(ValueError, match="at least two"):
        common.run_gas_demo(
            use_grid_initializer=False,
            case_name="gas_no_grid",
            argv=[
                "--num-samples",
                "1",
                "--output-dir",
                str(invalid_output),
            ],
        )
    assert stale_posterior.read_bytes() == b"stale"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        common.run_gas_demo(
            use_grid_initializer=False,
            case_name="gas_no_grid",
            argv=[
                "--co-database",
                str(tmp_path / "missing-database"),
                "--output-dir",
                str(invalid_output),
            ],
        )
    assert stale_posterior.read_bytes() == b"stale"

    database = tmp_path / "CO" / "12C-16O" / "Li2015"
    database.mkdir(parents=True)
    prefix = "12C-16O__Li2015"
    for suffix in (".def", ".pf", ".states.bz2", ".trans.bz2"):
        (database / f"{prefix}{suffix}").write_text("local fixture\n")
    with pytest.raises(ValueError, match="relative-noise"):
        common.run_gas_demo(
            use_grid_initializer=False,
            case_name="gas_no_grid",
            argv=[
                "--co-database",
                str(database),
                "--relative-noise",
                "nan",
                "--output-dir",
                str(invalid_output),
            ],
        )
    assert stale_posterior.read_bytes() == b"stale"


def test_run_output_records_config_and_rejects_divergences(common, tmp_path):
    class FakeMCMC:
        def __init__(
            self,
            diverging,
            samples=None,
            *,
            elapsed_seconds=1.25,
            accept_probability=None,
            include_step_size=True,
        ):
            self.diverging = (
                None if diverging is None else np.asarray(diverging)
            )
            self.samples = np.asarray(
                [1159.0, 1160.0, 1161.0] if samples is None else samples
            )
            self.exogibbs_elapsed_seconds = elapsed_seconds
            self.accept_probability = np.asarray(
                [0.7, 0.8, 0.9]
                if accept_probability is None
                else accept_probability
            )
            self.include_step_size = include_step_size

        def get_samples(self):
            return {"T0": self.samples}

        def get_extra_fields(self):
            fields = {
                "accept_prob": self.accept_probability,
                "num_steps": np.asarray([1, 5, 7]),
            }
            if self.include_step_size:
                fields["adapt_state.step_size"] = np.asarray([0.125] * 3)
            if self.diverging is not None:
                fields["diverging"] = self.diverging
            return fields

    context = SimpleNamespace(
        nu_grid=jnp.asarray([1.0, 2.0, 3.0]),
        art=SimpleNamespace(pressure=jnp.asarray([0.1, 1.0])),
    )
    observation = common.MockObservation(
        observed=jnp.ones((3,)),
        truth=jnp.ones((3,)),
        flux_scale=jnp.asarray(1.0),
        noise_std=jnp.asarray(0.01),
    )
    settings = common.RunSettings(
        num_warmup=10,
        num_samples=3,
        seed=2,
        progress_bar=False,
        max_tree_depth=6,
        quick=True,
    )

    common.write_run_outputs(
        tmp_path / "passed",
        case_name="test",
        context=context,
        observation=observation,
        mcmc=FakeMCMC([False, False, False]),
        settings=settings,
    )
    passed = json.loads(
        (tmp_path / "passed" / "run_summary.json").read_text()
    )
    assert passed["run_config"] == {
        "max_tree_depth": 6,
        "num_chains": 1,
        "num_samples": 3,
        "num_warmup": 10,
        "quick": True,
        "seed": 2,
    }
    assert passed["sampling_diagnostics"]["passed"] is True
    assert passed["sampling_diagnostics"][
        "mean_accept_probability"
    ] == pytest.approx(0.8)
    assert passed["sampling_diagnostics"]["maximum_num_steps"] == 7
    assert passed["sampling_diagnostics"]["step_size"] == 0.125

    with pytest.raises(RuntimeError, match="1 divergent transition"):
        common.write_run_outputs(
            tmp_path / "failed",
            case_name="test",
            context=context,
            observation=observation,
            mcmc=FakeMCMC([False, True, False]),
            settings=settings,
        )
    failed = json.loads(
        (tmp_path / "failed" / "run_summary.json").read_text()
    )
    assert failed["divergences"] == 1
    assert failed["sampling_diagnostics"]["passed"] is False
    failed_status = json.loads(
        (tmp_path / "failed" / "run_status.json").read_text()
    )
    assert failed_status["state"] == "failed"

    with pytest.raises(RuntimeError, match="did not vary"):
        common.write_run_outputs(
            tmp_path / "stuck",
            case_name="test",
            context=context,
            observation=observation,
            mcmc=FakeMCMC([False, False, False], samples=[1160.0] * 3),
            settings=settings,
        )

    with pytest.raises(RuntimeError, match="diagnostics are unavailable"):
        common.write_run_outputs(
            tmp_path / "missing_divergence",
            case_name="test",
            context=context,
            observation=observation,
            mcmc=FakeMCMC(None),
            settings=settings,
        )
    missing = json.loads(
        (tmp_path / "missing_divergence" / "run_summary.json").read_text()
    )
    assert missing["divergences"] is None
    assert missing["sampling_diagnostics"]["divergences"] is None

    with pytest.raises(RuntimeError, match="statistics are missing"):
        common.write_run_outputs(
            tmp_path / "missing_sampler_statistics",
            case_name="test",
            context=context,
            observation=observation,
            mcmc=FakeMCMC(
                [False, False, False],
                include_step_size=False,
            ),
            settings=settings,
        )

    with pytest.raises(RuntimeError, match="diagnostics are unavailable"):
        common.write_run_outputs(
            tmp_path / "non_boolean_divergence",
            case_name="test",
            context=context,
            observation=observation,
            mcmc=FakeMCMC([0, 0, 0]),
            settings=settings,
        )
    non_boolean = json.loads(
        (tmp_path / "non_boolean_divergence" / "run_summary.json").read_text()
    )
    assert non_boolean["divergences"] is None

    with pytest.raises(RuntimeError, match="not finite"):
        common.write_run_outputs(
            tmp_path / "nonfinite",
            case_name="test",
            context=context,
            observation=observation,
            mcmc=FakeMCMC(
                [False, False, False],
                samples=[1159.0, np.nan, 1161.0],
                elapsed_seconds=np.inf,
                accept_probability=[0.7, np.nan, 0.9],
            ),
            settings=settings,
        )
    strict_text = (tmp_path / "nonfinite" / "run_summary.json").read_text()

    def reject_nonstandard_json(value):
        raise AssertionError(f"non-standard JSON value: {value}")

    nonfinite = json.loads(
        strict_text,
        parse_constant=reject_nonstandard_json,
    )
    assert nonfinite["elapsed_seconds"] is None
    assert all(
        value is None for value in nonfinite["posterior"]["T0"].values()
    )
    assert nonfinite["sampling_diagnostics"][
        "mean_accept_probability"
    ] is None


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
