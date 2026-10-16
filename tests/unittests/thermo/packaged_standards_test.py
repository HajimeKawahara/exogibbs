"""Packaged standard functions retain the pinned numerical source convention."""

import importlib.util
from pathlib import Path

import jax
import numpy as np

from exogibbs.thermo import janaf_atomic, young2023

ROOT = Path(__file__).resolve().parents[3]


def test_young_frozen_branch_and_temperature_derivative():
    spec = importlib.util.spec_from_file_location("old_source_standards", ROOT/"examples/metal_silicate/source.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    record = young2023.DATA
    callback = old.make_source_standard_potentials_rt(record, record["cases"][0])
    for t in (2101., 2173.15, 2399.):
        expected = callback(t, 1.)*record["source"]["gas_constant_J_mol_K"]/young2023.R
        np.testing.assert_allclose(young2023.standard_potentials_rt(t), expected, atol=0., rtol=0.)
        assert np.all(np.isfinite(jax.jacfwd(young2023.standard_potentials_rt)(t)))
        assert all(np.isfinite(v) for v in young2023.supplementary_gas_standard_rt(t).values())


def test_janaf_outside_declared_interval_is_unavailable():
    assert np.all(np.isfinite(janaf_atomic.standard_potentials_rt(2173.15)))
    assert np.all(np.isnan(janaf_atomic.standard_potentials_rt(2000.)))
    assert np.all(np.isnan(janaf_atomic.standard_potentials_rt(2500.)))
