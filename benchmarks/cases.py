from __future__ import annotations

import math

from benchmarks.models import BenchmarkCase


FASTCHEM_EXTENDED_SINGLE_ANCHOR = BenchmarkCase(
    case_id="fastchem_extended_single_T3000_P1e-8_reference",
    category="single_layer",
    setup_metadata={
        "preset": "fastchem_extended",
        "source": "exogibbs.presets.fastchem.chemsetup",
        "logk_file": "fastchem/logK/logK_extended.dat",
        "element_file": "fastchem/element_abundances/asplund_2020_extended.dat",
        "abundance_source": "normalized_linear_abundance_from_extended_element_file",
    },
    axes={
        "temperature_K": 3000.0,
        "pressure_bar": 1.0e-8,
        "element_range": "fastchem_extended",
        "abundance_pattern": "solar_like",
    },
    solver_options={
        "epsilon_crit": 1.0e-10,
        "max_iter": 1000,
    },
    pref_bar=1.0,
    setup_kwargs={
        "path": "fastchem/logK/logK_extended.dat",
        "species_defalt_elements": False,
        "element_file": "fastchem/element_abundances/asplund_2020_extended.dat",
        "silent": True,
    },
)


SINGLE_LAYER_CASES = {
    FASTCHEM_EXTENDED_SINGLE_ANCHOR.case_id: FASTCHEM_EXTENDED_SINGLE_ANCHOR,
}


def get_single_layer_case(case_id: str) -> BenchmarkCase:
    try:
        return SINGLE_LAYER_CASES[case_id]
    except KeyError as exc:
        raise KeyError(f"Unknown single-layer benchmark case: {case_id!r}") from exc


_PROFILE_LAYER_COUNT = 64
_PROFILE_PRESSURE_MIN_BAR = 1.0e-8
_PROFILE_PRESSURE_MAX_BAR = 1.0e2
_PROFILE_TEMPERATURE_MIN_K = 900.0
_PROFILE_TEMPERATURE_MAX_K = 3000.0
_PROFILE_PRESSURE_STEP = (
    math.log10(_PROFILE_PRESSURE_MAX_BAR) - math.log10(_PROFILE_PRESSURE_MIN_BAR)
) / (_PROFILE_LAYER_COUNT - 1)
_PROFILE_TEMPERATURE_STEP = (
    _PROFILE_TEMPERATURE_MAX_K - _PROFILE_TEMPERATURE_MIN_K
) / (_PROFILE_LAYER_COUNT - 1)


FASTCHEM_EXTENDED_PROFILE_ANCHOR = BenchmarkCase(
    case_id="fastchem_extended_profile_64layer_monotonic_reference",
    category="profile",
    setup_metadata={
        "preset": "fastchem_extended",
        "source": "exogibbs.presets.fastchem.chemsetup",
        "logk_file": "fastchem/logK/logK_extended.dat",
        "element_file": "fastchem/element_abundances/asplund_2020_extended.dat",
        "abundance_source": "normalized_linear_abundance_from_extended_element_file",
        "profile_ordering_convention": "top_to_bottom_increasing_pressure",
    },
    axes={
        "layer_count": _PROFILE_LAYER_COUNT,
        "temperature_profile_kind": "log_pressure_linear_ramp",
        "temperature_min_K": _PROFILE_TEMPERATURE_MIN_K,
        "temperature_max_K": _PROFILE_TEMPERATURE_MAX_K,
        "temperature_step_K": _PROFILE_TEMPERATURE_STEP,
        "pressure_min_bar": _PROFILE_PRESSURE_MIN_BAR,
        "pressure_max_bar": _PROFILE_PRESSURE_MAX_BAR,
        "pressure_log10_step": _PROFILE_PRESSURE_STEP,
        "element_range": "fastchem_extended",
        "abundance_pattern": "solar_like",
        "profile_ordering": "top_to_bottom",
    },
    solver_options={
        "epsilon_crit": 1.0e-10,
        "max_iter": 1000,
    },
    pref_bar=1.0,
    setup_kwargs={
        "path": "fastchem/logK/logK_extended.dat",
        "species_defalt_elements": False,
        "element_file": "fastchem/element_abundances/asplund_2020_extended.dat",
        "silent": True,
    },
)


PROFILE_CASES = {
    FASTCHEM_EXTENDED_PROFILE_ANCHOR.case_id: FASTCHEM_EXTENDED_PROFILE_ANCHOR,
}


def get_profile_case(case_id: str) -> BenchmarkCase:
    try:
        return PROFILE_CASES[case_id]
    except KeyError as exc:
        raise KeyError(f"Unknown profile benchmark case: {case_id!r}") from exc
