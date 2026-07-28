"""Unit tests for FastChem comparison metrics."""

from __future__ import annotations

import json
from pathlib import Path
import runpy

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_module = runpy.run_path(
    REPOSITORY_ROOT / "benchmarks" / "fastchem4" / "comparison.py"
)
align_species_values = _module["align_species_values"]
condensate_comparison_metrics = _module["condensate_comparison_metrics"]
element_budget_metrics = _module["element_budget_metrics"]
gas_major_species_metrics = _module["gas_major_species_metrics"]
occurrence_keys = _module["occurrence_keys"]
profile_phase_transitions = _module["profile_phase_transitions"]
to_json_safe = _module["to_json_safe"]


def test_occurrence_alignment_preserves_duplicate_zinc_and_leading_axes() -> None:
    target_names = ("Zn(s,l)", "Zn(s,l)", "H")
    source_names = ("Zn(s,l)", "H1", "Zn(s,l)")
    source = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    aligned = align_species_values(target_names, source_names, source)

    assert occurrence_keys(target_names) == (
        ("Zn(s,l)", 0),
        ("Zn(s,l)", 1),
        ("H", 0),
    )
    np.testing.assert_allclose(
        aligned,
        np.asarray(
            [
                [1.0, 3.0, 2.0],
                [4.0, 6.0, 5.0],
            ]
        ),
    )


def test_alignment_handles_electron_position_and_element_aliases() -> None:
    aligned = align_species_values(
        ("e-", "H", "He", "H2", "missing"),
        ("H1", "He1", "H2", "e1-"),
        np.asarray([1.0, 2.0, 3.0, 4.0]),
    )

    np.testing.assert_allclose(aligned, [4.0, 1.0, 2.0, 3.0, 0.0])


def test_element_budget_excludes_electron_and_uses_relative_floor() -> None:
    report = element_budget_metrics(
        gas_formula_matrix=np.asarray(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        ),
        condensate_formula_matrix=np.asarray(
            [
                [0.0],
                [2.0],
                [1.0],
                [1.0e-15],
            ]
        ),
        gas_amounts=np.asarray([5.0, 1.0]),
        condensate_amounts=np.asarray([2.0]),
        target=np.asarray([999.0, 14.0, 2.0, 0.0]),
        element_names=("e1-", "H1", "O", "C"),
        relative_floor=1.0e-12,
    )

    assert report["excluded_element_names"] == ["e1-"]
    assert report["reconstructed"] == pytest.approx([5.0, 14.0, 3.0, 2.0e-15])
    assert report["absolute_relative_residual"] == pytest.approx(
        [994.0 / 999.0, 0.0, 0.5, 0.002]
    )
    assert report["max_absolute_relative_residual"] == pytest.approx(0.5)
    assert report["max_absolute_relative_residual_element"] == "O"
    assert report["finite"] is True


def test_gas_major_species_uses_union_threshold_and_excludes_electron() -> None:
    report = gas_major_species_metrics(
        names=("e1-", "H1", "CO", "H2O"),
        left_values=np.asarray([1.0, 0.9, 1.0e-9, 0.2]),
        right_values=np.asarray([2.0, 0.7, 0.3, 1.0e-9]),
        threshold=0.1,
        ratio_floor=1.0e-12,
        top_n=2,
    )

    assert report["left_major_count"] == 2
    assert report["right_major_count"] == 2
    assert report["intersection_major_count"] == 1
    assert report["major_species_count"] == 3
    assert report["major_set_jaccard"] == pytest.approx(1.0 / 3.0)
    assert [record["name"] for record in report["major_species"]] == [
        "H",
        "CO",
        "H2O",
    ]
    assert [row["name"] for row in report["top_rows"]] == ["CO", "H2O"]


def test_condensate_active_set_comparison_is_slot_aware() -> None:
    report = condensate_comparison_metrics(
        names=("Zn(s,l)", "Zn(s,l)", "MgSiO3(s)"),
        left_values=np.asarray([1.0, 0.0, 2.0]),
        right_values=np.asarray([0.0, 1.0, 2.0]),
        active_floor=0.1,
        ratio_floor=1.0e-12,
        top_n=3,
    )

    assert report["intersection_active_count"] == 1
    assert report["union_active_count"] == 3
    assert report["active_set_jaccard"] == pytest.approx(1.0 / 3.0)
    assert report["left_only_active"] == [
        {
            "name": "Zn(s,l)",
            "occurrence": 0,
            "slot": 0,
            "occurrence_key": ["Zn(s,l)", 0],
        }
    ]
    assert report["right_only_active"] == [
        {
            "name": "Zn(s,l)",
            "occurrence": 1,
            "slot": 1,
            "occurrence_key": ["Zn(s,l)", 1],
        }
    ]


def test_profile_phase_transitions_compare_adjacent_slot_sets() -> None:
    report = profile_phase_transitions(
        names=("Zn(s,l)", "Zn(s,l)", "MgSiO3(s)"),
        amounts=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
            ]
        ),
        threshold=0.5,
    )

    assert report["transition_count"] == 2
    assert report["transition_indices"] == [1, 2]
    assert report["transitions"][0]["entered"][0]["name"] == "MgSiO3(s)"
    second = report["transitions"][1]
    assert second["exited"][0]["occurrence_key"] == ["Zn(s,l)", 0]
    assert second["entered"][0]["occurrence_key"] == ["Zn(s,l)", 1]
    assert report["adjacent"][2]["changed"] is False


def test_json_safe_conversion_replaces_nonfinite_numpy_values() -> None:
    safe = to_json_safe(
        {
            "values": np.asarray([1.0, np.nan, np.inf]),
            "count": np.int64(2),
            "flag": np.bool_(True),
        }
    )

    assert safe == {
        "values": [1.0, None, None],
        "count": 2,
        "flag": True,
    }
    json.dumps(safe, allow_nan=False)
