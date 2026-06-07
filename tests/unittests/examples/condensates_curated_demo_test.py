"""Tests for curated condensate demo profile definitions."""

from __future__ import annotations

import sys
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[3] / "examples" / "condensates_curated_demo"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from _curated_demo_common import FRESH_CURATED_PROFILES
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup


def test_fresh_curated_profiles_cover_ten_families() -> None:
    assert len(FRESH_CURATED_PROFILES) == 10
    for family, definition in FRESH_CURATED_PROFILES.items():
        assert definition.family == family
        assert len(definition.temperatures) == len(definition.pressures)
        assert definition.temperatures
        assert all(temperature > 0.0 for temperature in definition.temperatures)
        assert all(pressure > 0.0 for pressure in definition.pressures)


def test_fresh_curated_profile_support_species_exist_in_preset() -> None:
    setup = condensate_chemical_setup(silent=True)
    available = set(setup.condensate_species)
    for definition in FRESH_CURATED_PROFILES.values():
        missing = sorted(set(definition.support_species) - available)
        assert missing == []
