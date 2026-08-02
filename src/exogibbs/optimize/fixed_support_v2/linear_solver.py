"""Compatibility alias for the fixed-support linear solver."""

import sys

from exogibbs.equilibrium.condensate.fixed_support import (
    linear_solver as _implementation,
)


sys.modules[__name__] = _implementation
