"""Compatibility alias for the fixed-support controller."""

import sys

from exogibbs.equilibrium.condensate.fixed_support import (
    controller as _implementation,
)


sys.modules[__name__] = _implementation
