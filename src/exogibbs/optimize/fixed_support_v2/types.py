"""Compatibility alias for fixed-support types."""

import sys

from exogibbs.equilibrium.condensate.fixed_support import types as _implementation


sys.modules[__name__] = _implementation
