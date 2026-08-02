"""Compatibility alias for the relocated gas-equilibrium grid service."""

import sys

from exogibbs.equilibrium.gas.grid import service as _implementation


sys.modules[__name__] = _implementation
