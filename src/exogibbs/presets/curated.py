"""Current curated gas thermochemistry, with versioned data and evidence.

The dataset is curated-tio2-v1. Earlier datasets remain available through
curated_ps, curated_cn, curated_cos, curated_ho2, curated_kh, curated_oh,
curated_h2o, curated_co2, curated_tio, curated_c2h and curated_v1.
Runtime does not require MCEG.
"""

from exogibbs.presets.curated_tio2 import chemsetup


__all__ = ("chemsetup",)
