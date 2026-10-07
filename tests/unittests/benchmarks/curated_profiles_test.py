"""Tests for native curated condensate profile definitions."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest


# Benchmark fixtures are intentionally repository-only and are not installed
# with the src-layout package.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPOSITORY_ROOT))
_CURATED_EXAMPLE_ROOT = (
    _REPOSITORY_ROOT / "examples" / "condensates_curated_demo"
)
_CURATED_COMMON_PATH = _CURATED_EXAMPLE_ROOT / "_curated_demo_common.py"

from benchmarks.fixed_support_v2.curated_profiles import (
    FRESH_CURATED_PROFILES,
    case_id_for_profile,
    element_budget_for_profile,
    fresh_profile_definition,
    support_payload_for_profile,
)
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup


def test_fresh_curated_profiles_expose_the_demo_family_set() -> None:
    assert len(FRESH_CURATED_PROFILES) == 10
    assert fresh_profile_definition("solar_water_condensation").support_species == (
        "H2O(s,l)",
    )


def test_case_id_for_profile_matches_existing_demo_labels() -> None:
    definition = fresh_profile_definition("solar_water_condensation")

    assert case_id_for_profile(definition, 300.0, 0.1) == (
        "solar_water_condensation__T300_P0p1"
    )


def test_support_payload_for_profile_uses_native_budget_seed() -> None:
    setup = condensate_chemical_setup(silent=True)
    definition = fresh_profile_definition("solar_water_condensation")
    budget = element_budget_for_profile(setup, definition)

    support_indices, support_amounts = support_payload_for_profile(setup, definition, budget)

    assert tuple(setup.condensate_species[index] for index in support_indices) == (
        "H2O(s,l)",
    )
    assert len(support_amounts) == len(support_indices)
    assert all(amount > 0.0 for amount in support_amounts)
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(support_amounts))))


def _literal_assignments(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }


def _load_curated_common():
    spec = importlib.util.spec_from_file_location(
        "curated_demo_common_test_module",
        _CURATED_COMMON_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seeded_curated_examples_label_their_equilibrium_phases() -> None:
    cases = {
        "demo_carbon_rich_cas_mgs_aln_window.py": {
            "family": "carbon_rich_CaS_MgS_AlN_window",
            "display_name": "Carbon-rich C/Fe/Mg-silicate/sulfide profile",
            "seed_only": {"CaS(s)", "MgS(s)", "AlN(s)"},
            "preferred": {"C(s)", "Fe(s,l)", "MgSiO3(s,l)", "FeS(s,l)"},
        },
        "demo_sio_s_condensate_window.py": {
            "family": "SiO_s_condensate_window",
            "display_name": "Solar Mg-silicate/Fe/feldspar profile",
            "seed_only": {"SiO(s)"},
            "preferred": {
                "MgSiO3(s,l)",
                "Mg2SiO4(s,l)",
                "Fe(s,l)",
                "NaAlSi3O8(s)",
            },
        },
    }

    for filename, expected in cases.items():
        assignments = _literal_assignments(_CURATED_EXAMPLE_ROOT / filename)
        condensates = set(assignments["CONDENSATES"])

        assert assignments["FAMILY"] == expected["family"]
        assert assignments["DISPLAY_NAME"] == expected["display_name"]
        assert condensates.isdisjoint(expected["seed_only"])
        assert condensates.issuperset(expected["preferred"])


def test_plot_curated_family_renders_display_name(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("matplotlib")
    common = _load_curated_common()
    setup = SimpleNamespace(
        gas_species=("H2",),
        condensate_species=("C(s)",),
    )
    definition = SimpleNamespace(
        temperatures=(800.0,),
        pressures=(0.1,),
    )
    result = SimpleNamespace(
        condensate_amounts=np.asarray([1.0e-4]),
        converged=True,
    )
    captured = {}
    real_subplots = common.plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = real_subplots(*args, **kwargs)
        captured["figure"] = figure
        captured["axes"] = axes
        return figure, axes

    monkeypatch.setattr(
        common,
        "condensate_chemical_setup",
        lambda *, silent: setup,
    )
    monkeypatch.setattr(
        common,
        "fresh_profile_definition",
        lambda family: definition,
    )
    monkeypatch.setattr(
        common,
        "run_fresh_curated_profile",
        lambda current_setup, current_definition: (
            [result],
            [np.asarray([1.0])],
            [],
        ),
    )
    monkeypatch.setattr(common.plt, "subplots", capture_subplots)
    output_path = tmp_path / "curated.png"

    returned_path = common.plot_curated_family(
        family="seed_profile_key",
        display_name="Readable equilibrium profile",
        preferred_gas_species=("H2",),
        preferred_condensates=("C(s)",),
        output_path=output_path,
    )

    assert returned_path == output_path
    assert output_path.exists()
    assert captured["axes"][0].get_title() == "Readable equilibrium profile"
