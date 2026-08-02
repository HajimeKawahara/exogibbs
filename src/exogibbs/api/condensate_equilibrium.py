"""Compatibility alias for the condensate-equilibrium implementation.

The historical module is kept as an alias, rather than a copying re-export,
so supported objects retain identity and existing diagnostic monkeypatches
continue to address the implementation module during the migration.
"""

import sys

from exogibbs.equilibrium.condensate import solve as _implementation


sys.modules[__name__] = _implementation
