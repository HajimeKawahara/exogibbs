"""Compatibility alias for fixed-support filter operations."""

import sys

from exogibbs.equilibrium.condensate.fixed_support import filter as _implementation


sys.modules[__name__] = _implementation
