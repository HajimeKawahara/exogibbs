"""Compatibility facade for gas-equilibrium reverse-mode sensitivities."""

from exogibbs.equilibrium.gas.kernel import autodiff as _implementation


for _name, _value in vars(_implementation).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

__all__ = tuple(
    name for name in vars(_implementation) if not name.startswith("_")
)
