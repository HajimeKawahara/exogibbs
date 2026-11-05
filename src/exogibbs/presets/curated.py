"""Current curated gas preset, including the reviewed NASA/ATcT HO2 reaction.

The dataset is curated-ho2-v1. Earlier datasets remain available through
curated_kh, curated_oh, curated_h2o, curated_co2, curated_tio, curated_c2h
and curated_v1. All versions cover 1500--4000 K.
"""

from exogibbs.presets.curated_ho2 import chemsetup


__all__ = ("chemsetup",)
