"""Lightweight contracts for the optional ExoEOS gallery example."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PATH = REPOSITORY_ROOT / "examples" / "plot_exoeos_pure_fugacity.py"


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "plot_exoeos_pure_fugacity_test_module",
        EXAMPLE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_is_syntax_valid_and_uses_the_exoeos_adapter() -> None:
    source = EXAMPLE_PATH.read_text(encoding="utf-8")

    compile(source, str(EXAMPLE_PATH), "exec")
    assert "from exoeos import ZhangDuanEOS" in source
    assert (
        "from exogibbs.interop.exoeos import make_pure_lnphi_func" in source
    )
    assert "except ImportError" in source
    assert "lnphi_func=lnphi_func" in source
    assert "eos_by_species=eos_by_species" in source
    assert "def make_pure_lnphi_func" not in source
    assert "state_tp(" not in source


def test_reduced_setup_matches_the_zhang_duan_species_order() -> None:
    example = _load_example()
    setup = example.build_reduced_setup()

    assert setup.species == (
        "CH4",
        "H2O",
        "CO2",
        "H2",
        "CO",
        "O2",
        "C2H6",
    )
    assert setup.elements == ("C", "H", "O")
    assert setup.formula_matrix.shape == (3, 7)
    assert np.linalg.matrix_rank(np.asarray(setup.formula_matrix)) == 3
    np.testing.assert_allclose(
        np.asarray(setup.hvector_func(jnp.asarray(1500.0))),
        np.zeros(7),
    )
