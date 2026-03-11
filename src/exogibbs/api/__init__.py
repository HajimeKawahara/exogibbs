from .chemistry import ChemicalSetup, ThermoState

__all__ = [
    "ChemicalSetup",
    "ThermoState",
    "EquilibriumOptions",
    "EquilibriumInit",
    "EquilibriumResult",
]


def __getattr__(name):
    # Delay importing the equilibrium module until one of its public symbols is
    # requested. This avoids import cycles for modules that only need
    # `exogibbs.api.chemistry`.
    if name in {"EquilibriumOptions", "EquilibriumInit", "EquilibriumResult"}:
        from .equilibrium import EquilibriumInit, EquilibriumOptions, EquilibriumResult

        return {
            "EquilibriumOptions": EquilibriumOptions,
            "EquilibriumInit": EquilibriumInit,
            "EquilibriumResult": EquilibriumResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
