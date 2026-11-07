"""Current curated gas preset, using the reviewed C2H and TiO updates.

The dataset is curated-tio-v1. Earlier datasets remain available through
``curated_c2h`` and ``curated_v1``. All versions cover 1500--4000 K.
"""

from exogibbs.presets.curated_tio import chemsetup


__all__ = ("chemsetup",)
