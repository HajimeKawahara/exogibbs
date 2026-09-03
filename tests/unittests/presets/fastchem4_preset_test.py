"""Tests for native FastChem4 thermochemistry presets."""

from __future__ import annotations

import builtins

import jax.numpy as jnp
import numpy as np
import pytest

from exogibbs.presets import fastchem4 as fastchem4_module
from exogibbs.presets import fastchem4_cond as fastchem4_cond_module
from exogibbs.presets.fastchem4 import chemsetup
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup
from exogibbs.thermo.models import ChemicalSetup


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


def test_fastchem4_default_elements_metadata_ignores_custom_element_file() -> None:
    setup = chemsetup(
        element_file="FastChem4/element_abundances/asplund_2021_extended.dat",
        silent=True,
    )

    assert (
        setup.metadata["fastchem_element_file"]
        == "FastChem4/element_abundances/asplund_2021.dat"
    )


def test_fastchem4_species_derived_elements_metadata_has_no_abundance_file() -> None:
    setup = chemsetup(
        species_defalt_elements=False,
        silent=True,
    )

    assert setup.metadata["fastchem_element_file"] is None


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


def test_fastchem4_condensate_setup_rejects_missing_gas_elements() -> None:
    gas_setup = ChemicalSetup(
        formula_matrix=jnp.eye(2),
        hvector_func=lambda temperature: jnp.zeros((2,)),
        elements=("H", "O"),
        species=("H", "O"),
        element_vector_reference=jnp.asarray([1.0, 1.0e-3]),
    )

    with pytest.raises(ValueError, match="missing elements"):
        fastchem4_cond_module.chemsetup(gas_setup=gas_setup, silent=True)


def test_fastchem4_extended_condensates_reject_default_gas_elements() -> None:
    gas_setup = chemsetup(silent=True)

    with pytest.raises(ValueError, match="missing elements"):
        fastchem4_cond_module.chemsetup(
            path="FastChem4/logK/logK_condensates_extended.dat",
            gas_setup=gas_setup,
            silent=True,
        )


def test_fastchem4_extended_condensates_accept_extended_gas_elements() -> None:
    gas_setup = chemsetup(
        path="FastChem4/logK/logK_extended_wo_ions.dat",
        species_defalt_elements=False,
        element_file="FastChem4/element_abundances/asplund_2021_extended.dat",
        silent=True,
    )

    setup = fastchem4_cond_module.chemsetup(
        path="FastChem4/logK/logK_condensates_extended.dat",
        gas_setup=gas_setup,
        silent=True,
    )

    formula_matrix = np.asarray(setup.formula_matrix)
    assert formula_matrix.shape == (len(gas_setup.elements), 513)
    assert np.all(np.any(formula_matrix != 0.0, axis=0))
