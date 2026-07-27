"""Compatibility alias for fixed-support problem equations."""

import sys

from exogibbs.equilibrium.condensate.fixed_support import problem as _implementation


sys.modules[__name__] = _implementation
