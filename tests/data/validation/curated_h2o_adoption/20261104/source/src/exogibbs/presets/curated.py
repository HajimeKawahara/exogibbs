"""Current curated gas preset, including the reviewed NIST CO2 adoption.

The dataset is curated-co2-v1. Earlier datasets remain available through
curated_tio, curated_c2h and curated_v1. All versions cover 1500--4000 K.
"""

from exogibbs.presets.curated_co2 import chemsetup


__all__ = ("chemsetup",)
