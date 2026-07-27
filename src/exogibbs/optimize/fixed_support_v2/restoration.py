"""Compatibility alias for fixed-support restoration."""

import sys

from exogibbs.equilibrium.condensate.fixed_support import (
    restoration as _implementation,
)


sys.modules[__name__] = _implementation
