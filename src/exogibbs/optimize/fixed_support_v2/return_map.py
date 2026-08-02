"""Compatibility alias for fixed-support restoration return mapping."""

import sys

from exogibbs.equilibrium.condensate.fixed_support import (
    return_map as _implementation,
)


sys.modules[__name__] = _implementation
