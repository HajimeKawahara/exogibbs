"""Current curated gas preset, using the reviewed C2H correction.

The dataset is curated-c2h-v1. For the original curated-gas-v1 calculation,
import ``curated_v1`` explicitly. Both versions cover 1500--4000 K.
"""

from exogibbs.presets.curated_c2h import chemsetup


__all__ = ("chemsetup",)
