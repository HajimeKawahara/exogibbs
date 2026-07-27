"""Compatibility alias for fixed-support second-order correction."""

import sys

from exogibbs.equilibrium.condensate.fixed_support import soc as _implementation


sys.modules[__name__] = _implementation
