"""Backward-compatible imports for the MELTYQ solubility-law compilation.

The laws live in :mod:`exogibbs.solubility.volatile` because they are useful
outside the experimental MELTYQ-equivalent interface as well.
"""

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
    "MELTYQ_SOLUBILITY_METADATA",
    "SolubilityMetadata",
    "ch4_ardia2013",
    "co2_lichtenberg2021",
    "co_yoshioka2019",
    "h2_hirschmann2012",
    "h2o_lichtenberg2021",
    "n2_dasgupta2022",
)
