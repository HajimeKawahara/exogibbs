"""Tests for thermo-valid condensate support filtering."""

from __future__ import annotations

import pytest

from exogibbs.condensates.thermo_valid_support import (
    filter_thermo_valid_condensate_support,
)


def test_filter_thermo_valid_support_removes_sentinel_sources() -> None:
    result = filter_thermo_valid_condensate_support(
        explicit_opt_in=True,
        support_indices=[0, 1, 2, 3],
        condensate_standard_source=[1.0e20, 10.0, 20.0, -1.0e20],
        formula_matrix_cond_active=[[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, 1.0]],
        ln_mk=[-10.0, -9.0, -8.0, -7.0],
        rho=[1.0, 2.0, 3.0, 4.0],
        eta=[2.0, 3.0, 4.0, 5.0],
        species_names=["bad_a", "good_a", "good_b", "bad_b"],
        field_provenance={
            "ln_mk": "exogibbs_native",
            "rho": "exogibbs_native",
            "eta": "exogibbs_native",
        },
    )

    assert result.support_indices == (1, 2)
    assert result.condensate_standard_source == (10.0, 20.0)
    assert result.formula_matrix_cond_active == ((2.0, 3.0), (1.0, 0.0))
    assert result.ln_mk == (-9.0, -8.0)
    assert result.rho == (2.0, 3.0)
    assert result.eta == (3.0, 4.0)
    assert result.report.original_support_count == 4
    assert result.report.filtered_support_count == 2
    assert result.report.removed_support_indices == (0, 3)
    assert result.report.removed_species_names == ("bad_a", "bad_b")
    assert result.report.diagnostic_only is True
    assert result.report.default_off is True
    assert result.report.production_behavior_change is False


def test_filter_thermo_valid_support_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="explicit_opt_in"):
        filter_thermo_valid_condensate_support(
            explicit_opt_in=False,
            support_indices=[0],
            condensate_standard_source=[1.0],
        )


def test_filter_thermo_valid_support_rejects_forbidden_provenance() -> None:
    with pytest.raises(ValueError):
        filter_thermo_valid_condensate_support(
            explicit_opt_in=True,
            support_indices=[0],
            condensate_standard_source=[1.0],
            field_provenance={"ln_mk": "fastchem4_trace"},
        )


def test_filter_thermo_valid_support_rejects_empty_filtered_support() -> None:
    with pytest.raises(ValueError, match="empty"):
        filter_thermo_valid_condensate_support(
            explicit_opt_in=True,
            support_indices=[0, 1],
            condensate_standard_source=[1.0e20, -1.0e20],
        )


def test_filter_thermo_valid_support_validates_shapes() -> None:
    with pytest.raises(ValueError, match="length"):
        filter_thermo_valid_condensate_support(
            explicit_opt_in=True,
            support_indices=[0, 1],
            condensate_standard_source=[1.0],
        )

    with pytest.raises(ValueError, match="column count"):
        filter_thermo_valid_condensate_support(
            explicit_opt_in=True,
            support_indices=[0, 1],
            condensate_standard_source=[1.0, 2.0],
            formula_matrix_cond_active=[[1.0]],
        )

