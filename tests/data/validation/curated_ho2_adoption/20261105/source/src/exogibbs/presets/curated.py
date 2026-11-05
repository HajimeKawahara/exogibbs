"""Current curated gas preset, including reviewed spectroscopic KH energy.

The dataset is curated-kh-v1. Earlier datasets remain available through
curated_oh, curated_h2o, curated_co2, curated_tio, curated_c2h and curated_v1.
All versions cover 1500--4000 K.
"""

from exogibbs.presets.curated_kh import chemsetup


__all__ = ("chemsetup",)
