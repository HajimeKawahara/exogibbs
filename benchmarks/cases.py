from __future__ import annotations

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
