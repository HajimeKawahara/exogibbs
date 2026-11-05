"""Current curated gas thermochemistry, with versioned data and evidence.

The dataset is curated-cos-v1. Earlier datasets remain available through
curated_ho2, curated_kh, curated_oh, curated_h2o, curated_co2, curated_tio,
curated_c2h and curated_v1. Runtime does not require MCEG.
"""

from exogibbs.presets.curated_cos import chemsetup


__all__ = ("chemsetup",)
