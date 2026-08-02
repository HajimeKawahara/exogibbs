"""Compatibility alias for fixed-support normal steps."""

import sys

from exogibbs.equilibrium.condensate.fixed_support import normal as _implementation


sys.modules[__name__] = _implementation
