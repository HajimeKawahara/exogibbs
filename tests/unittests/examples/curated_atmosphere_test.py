"""Regress the controlled atmosphere comparison, abundance ranking and closure gates."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.api.gas import EquilibriumOptions, solve_profile
from exogibbs.presets import fastchem


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/curated_atmosphere"
SPEC = importlib.util.spec_from_file_location("curated_atmosphere", ROOT / "examples/comparisons/demo_curated_atmosphere.py")
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


@pytest.fixture(scope="module")
def models():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield demo.comparison_setups()
    jax.config.update("jax_enable_x64", previous)


def test_thermochemistry_control_and_legacy_catalog_are_distinct(models):
    new, four, old = (models[k] for k in demo.MODELS)
    assert new.species == four.species
    np.testing.assert_array_equal(new.formula_matrix, four.formula_matrix)
    assert (len(new.species), len(old.species)) == (537, 523)
    legacy = fastchem.chemsetup(silent=True)
    assert old.species == legacy.species
    np.testing.assert_array_equal(old.hvector_func(3123.), legacy.hvector_func(3123.))
    for setup in models.values():
        assert setup.elements == new.elements
        np.testing.assert_array_equal(setup.element_vector_reference, new.element_vector_reference)
        assert setup.metadata["relative_charge_convergence"] is True


@pytest.mark.parametrize("model", demo.MODELS)
def test_saved_columns_reproduce_with_independent_closure(models, model):
    setup = models[model]
    temperatures, pressures, expected = [], [], []
    for profile in demo.profiles():
        with np.load(RESULTS / f"{profile}_{model}.npz") as data:
            assert tuple(data["species"]) == setup.species
            demo.require_closure(demo.closure(setup, data["ln_n"], data["temperature_K"], data["pressure_bar"]))
            selected = [0, 59, 80, 120, 127, 160]
            temperatures.extend(data["temperature_K"][selected])
            pressures.extend(data["pressure_bar"][selected])
            expected.extend(data["ln_n"][selected])
    result, diagnostics = solve_profile(
        setup, T=jnp.array(temperatures), P=jnp.array(pressures),
        b=setup.element_vector_reference,
        options=EquilibriumOptions(epsilon_crit=1e-13, method="vmap_cold"), return_diagnostics=True)
    assert np.all(diagnostics["converged"])
    np.testing.assert_allclose(result.ln_n, expected, atol=1e-8, rtol=0)
    demo.require_closure(demo.closure(setup, np.asarray(result.ln_n), np.array(temperatures), np.array(pressures)))


def test_trace_only_ratios_do_not_dominate_the_report():
    setup = SimpleNamespace(species=("Cu1", "trace"))
    old = np.log(np.array([[1e-11, 1e-100], [1e-8, 1e-100]]))
    new = np.log(np.array([[1e-8, 1e-20], [2e-8, 1e-20]]))
    rows = demo.summarize("test", {k: setup for k in demo.MODELS},
                          {"curated": new, "fastchem4": old, "fastchem3": old},
                          np.array([4000., 3500.]), np.array([1e-6, 1.]))
    assert len(rows) == 2 and all(r["species"] == "Cu1" for r in rows)
    assert all(r["curated_over_baseline"] == pytest.approx(1000.) for r in rows)


def test_failed_or_nonfinite_closure_is_rejected():
    good = {"element_relative_residual": 1e-6, "charge_relative_residual": 1e-14,
            "mass_action_max_abs_ln": 1e-12}
    for key in good:
        for bad in (np.nan, .1):
            with pytest.raises(RuntimeError, match="independent closure"):
                demo.require_closure({**good, key: bad})


def test_demo_receipt_pins_data_and_outputs():
    receipt = json.loads((RESULTS / "receipt.json").read_text())
    assert receipt["cases_passed"] == 966
    for path, expected in receipt["selected_file_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    for path, expected in receipt["output_sha256"].items():
        assert hashlib.sha256((RESULTS / path).read_bytes()).hexdigest() == expected, path
