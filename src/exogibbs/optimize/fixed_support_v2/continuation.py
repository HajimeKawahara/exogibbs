"""Compatibility alias for fixed-support continuation."""

import sys

from exogibbs.equilibrium.condensate.fixed_support import (
    continuation as _implementation,
)


sys.modules[__name__] = _implementation
