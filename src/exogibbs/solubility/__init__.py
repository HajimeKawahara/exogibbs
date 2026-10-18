"""Auxiliary volatile-solubility laws for magma-atmosphere boundaries."""

from exogibbs.solubility.sun_yao2026 import (
    SUN_YAO2026_WATER_METADATA,
    h2o_sun_yao2026,
    h2o_sun_yao2026_terms,
)
from exogibbs.solubility.volatile import (
    MELTYQ_SOLUBILITY_METADATA,
    SolubilityMetadata,
    ch4_ardia2013,
    co2_lichtenberg2021,
    co_yoshioka2019,
    h2_hirschmann2012,
    h2o_lichtenberg2021,
    n2_dasgupta2022,
)


__all__ = (
    "SUN_YAO2026_WATER_METADATA",
    "h2o_sun_yao2026",
    "h2o_sun_yao2026_terms",
    "MELTYQ_SOLUBILITY_METADATA",
    "SolubilityMetadata",
    "ch4_ardia2013",
    "co2_lichtenberg2021",
    "co_yoshioka2019",
    "h2_hirschmann2012",
    "h2o_lichtenberg2021",
    "n2_dasgupta2022",
)
