"""Compatibility alias for the renamed gas autodiff module."""

import sys

from exogibbs.equilibrium.gas.kernel import autodiff as _implementation


sys.modules[__name__] = _implementation
