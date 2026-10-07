"""Offline numerical and execution checks for the curated demo audit."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from benchmarks.documented_examples.check_curated import run_curated, validate_layer


def _state():
    setup = SimpleNamespace(
        elements=("H", "O"), formula_matrix=np.eye(2),
        formula_matrix_cond=np.asarray([[0.0], [1.0]]),
        gas_setup=SimpleNamespace(element_vector_reference=np.asarray([2.0, 1.0])),
    )
    result = SimpleNamespace(
        gas_n=np.asarray([2.0, 0.5]), gas_x=np.asarray([0.8, 0.2]),
        gas_ntot=2.5, condensate_amounts=np.asarray([0.5]), converged=True,
        status="converged", acceptance_tier="fixed_support_v2_accepted", diagnostics={},
    )
    return setup, result


def test_independent_element_check_rejects_incorrect_success_claim():
    setup, result = _state()
    budget = setup.gas_setup.element_vector_reference
    assert validate_layer(setup, budget, result)["accepted"]
    result.condensate_amounts[:] = 0.8
    report = validate_layer(setup, budget, result)
    assert not report["accepted"]
    assert report["checks"]["converged"]
    assert not report["checks"]["element_budget"]


@pytest.mark.parametrize("failure", ["missing", "nonconverged", "nan", "negative"])
def test_invalid_layer_is_rejected(failure):
    setup, result = _state()
    if failure == "missing":
        result = None
    elif failure == "nonconverged":
        result.converged = False
        result.status = "not_converged"
    elif failure == "nan":
        result.gas_x[0] = np.nan
    else:
        result.condensate_amounts[0] = -0.5
    assert not validate_layer(setup, setup.gas_setup.element_vector_reference, result)["accepted"]


@pytest.mark.parametrize("high_temperature", [False, True])
@pytest.mark.parametrize(("converged", "layer_count"), [(False, 2), (True, 2), (True, 1)])
def test_real_main_and_plot_are_audited_even_if_main_does_not_raise(
    tmp_path, monkeypatch, high_temperature, converged, layer_count
):
    setup, result = _state()
    result.converged = converged
    result.status = "converged" if converged else "not_converged"
    common = ModuleType("_curated_demo_common")
    common.setup = setup
    common.definition = SimpleNamespace(temperatures=(1800.0, 1900.0), pressures=(1.0, 2.0))
    common.results = [result] * layer_count
    common.element_budget_for_profile = lambda *_: setup.gas_setup.element_vector_reference
    common.run_fresh_curated_profile = lambda *_: (common.results, [result.gas_x] * 2, [])
    common.curated_output_path = lambda *_: tmp_path / "wrong.png"
    monkeypatch.setitem(sys.modules, "_curated_demo_common", common)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "cpu")
    script = tmp_path / "demo.py"
    profile = (
        "PRESSURES_BAR = (1., 2.)\n"
        "def _run_profile(setup):\n"
        "    return [(p, r, None) for p, r in zip(PRESSURES_BAR, common.results)]\n"
    ) if high_temperature else ""
    call = "_run_profile(common.setup)" if high_temperature else (
        "common.run_fresh_curated_profile(common.setup, common.definition)"
    )
    script.write_text(
        "import _curated_demo_common as common\n"
        "from _curated_demo_common import curated_output_path\n"
        "from matplotlib.figure import Figure\n" + profile +
        "def main():\n"
        f"    {call}\n"
        "    figure = Figure()\n"
        "    figure.subplots().plot([0, 1], [0, 1])\n"
        "    figure.savefig(curated_output_path(__file__))\n"
    )
    report = run_curated(script, tmp_path / "outputs")
    assert report["accepted"] is (converged and layer_count == 2)
    assert report["expected_layers"] == 2
    assert len(report["layers"]) == layer_count
    assert (tmp_path / "outputs" / "demo.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (tmp_path / "outputs" / "audit.json").is_file()
    assert not (tmp_path / "wrong.png").exists()


def test_gpu_request_cannot_pass_with_cpu_backend(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(default_backend=lambda: "cpu"))
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    with pytest.raises(RuntimeError, match="GPU was requested"):
        run_curated(tmp_path / "unused.py", tmp_path / "outputs")


@pytest.mark.parametrize("filename", [
    "_curated_demo_common.py", "demo_solar_highT_no_condensate_gas_regression.py"
])
def test_curated_startup_preserves_requested_gpu(monkeypatch, filename):
    directory = Path(__file__).resolve().parents[3] / "examples" / "condensates_curated_demo"
    source = directory / filename
    tree = ast.parse(source.read_text())
    # Run actual startup statements, stopping before importing JAX or plotting.
    first_jax_import = next(
        index for index, node in enumerate(tree.body)
        if isinstance(node, ast.ImportFrom) and node.module == "jax"
    )
    tree.body = tree.body[:first_jax_import]
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "gpu")
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = {"__file__": str(source)}
    exec(compile(tree, str(source), "exec"), namespace)
    assert namespace["os"].environ["JAX_PLATFORMS"] == "cuda"
    assert namespace["os"].environ["JAX_PLATFORM_NAME"] == "gpu"
