"""
Convenience preset for FastChem condensates.

Loads ``fastchem/logK/logK_condensates.dat`` and returns a GasSetup
compatible with ExoGibbs APIs.

This wrapper delegates to ``exogibbs.presets.fastchem.gassetup`` with a
different default data path.
"""

from __future__ import annotations

from exogibbs.presets.fastchem import chemsetup as _chemsetup_base
from exogibbs.api.chemistry import ChemicalSetup


def chemsetup(path: str = "fastchem/logK/logK_condensates.dat") -> ChemicalSetup:
    """Build a GasSetup using the FastChem condensates table.

    Args:
        path: Relative path under ``src/exogibbs/data/`` to a FastChem
              condensates logK file. Defaults to the packaged dataset.

    Returns:
        ChemicalSetup configured from the specified condensates dataset.
    """

    return _chemsetup_base(path=path, add_element_species=False)
