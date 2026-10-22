"""Packaged standard functions retain the pinned numerical source convention."""

import json
from pathlib import Path

import jax
import numpy as np

from exogibbs.thermo import janaf_atomic, young2023

REFERENCE = Path(__file__).with_name("data") / "young2023_legacy_standards.json"


def test_young_frozen_branch_and_temperature_derivative():
    reference = json.loads(REFERENCE.read_text())
    assert tuple(reference["components"]) == young2023.COMPONENTS
    for t, expected in zip(reference["temperature_k"], reference["standard_potentials_rt"]):
        np.testing.assert_allclose(young2023.standard_potentials_rt(t), expected, atol=1e-13, rtol=1e-13)
        assert np.all(np.isfinite(jax.jacfwd(young2023.standard_potentials_rt)(t)))
        assert all(np.isfinite(v) for v in young2023.supplementary_gas_standard_rt(t).values())


def test_janaf_outside_declared_interval_is_unavailable():
    assert np.all(np.isfinite(janaf_atomic.standard_potentials_rt(2173.15)))
    assert np.all(np.isnan(janaf_atomic.standard_potentials_rt(2000.)))
    assert np.all(np.isnan(janaf_atomic.standard_potentials_rt(2500.)))
