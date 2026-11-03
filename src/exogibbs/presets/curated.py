"""Current curated gas preset, including evaluated H2O and NIST CO2.

The dataset is curated-h2o-v1. Earlier datasets remain available through
curated_co2, curated_tio, curated_c2h and curated_v1. All cover 1500--4000 K.
"""

from exogibbs.presets.curated_h2o import chemsetup


__all__ = ("chemsetup",)
