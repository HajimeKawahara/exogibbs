"""Tests for native FastChem4 thermochemistry presets."""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from exogibbs.presets import fastchem4 as fastchem4_module
from exogibbs.presets import fastchem4_cond as fastchem4_cond_module
from exogibbs.presets.fastchem4 import chemsetup
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup


@pytest.mark.parametrize(
    "preset_module",
    (fastchem4_module, fastchem4_cond_module),
)
def test_fastchem4_preset_closes_thermochemistry_file(
    monkeypatch: pytest.MonkeyPatch,
    preset_module,
) -> None:
    streams = []

    def tracking_open(*args, **kwargs):
        stream = builtins.open(*args, **kwargs)
        streams.append(stream)
        return stream

    monkeypatch.setattr(preset_module, "open", tracking_open, raising=False)

    preset_module.chemsetup(silent=True)

    assert streams
    assert all(stream.closed for stream in streams)


def test_fastchem4_gas_setup_uses_packaged_data() -> None:
    setup = chemsetup(silent=True)

    assert len(setup.species) == 422
    assert setup.formula_matrix.shape == (len(setup.elements), 422)
    assert setup.metadata["source"] == "FastChem4"
    assert setup.metadata["dataset"] == "gas"


def test_fastchem4_condensate_setup_uses_full_standard_species_order() -> None:
    setup = condensate_chemical_setup(silent=True)

    assert len(setup.condensate_species) == 219
    assert setup.formula_matrix_cond.shape == (len(setup.elements), 219)
    assert setup.condensate_species[188] == "Ca3(VO4)2(s)"
    assert setup.condensate_species[186:191] == (
        "Ca(VO3)2(s)",
        "Ca2V2O7(s)",
        "Ca3(VO4)2(s)",
        "CaZn(s)",
        "CaZn2(s)",
    )


def test_fastchem4_preserves_duplicate_condensate_slots() -> None:
    setup = condensate_chemical_setup(silent=True)

    zinc_slots = [
        index for index, species in enumerate(setup.condensate_species) if species == "Zn(s,l)"
    ]

    assert zinc_slots == [167, 202]
    assert setup.formula_matrix_cond.shape[1] == len(setup.condensate_species)


def test_fastchem4_hvectors_are_finite_for_representative_temperature() -> None:
    setup = condensate_chemical_setup(silent=True)

    gas_h = np.asarray(setup.gas_setup.hvector_func(1400.0), dtype=float)
    cond_h = np.asarray(setup.condensate_setup.hvector_func(1400.0), dtype=float)

    assert gas_h.shape == (len(setup.gas_species),)
    assert cond_h.shape == (len(setup.condensate_species),)
    assert np.all(np.isfinite(gas_h))
    assert np.all(np.isfinite(cond_h))
