"""Fixed-support numerical kernels for condensate equilibrium."""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "DifferentiableFixedSupportResult": (
        ".types",
        "DifferentiableFixedSupportResult",
    ),
    "FixedSupportSourceCotangents": (
        ".types",
        "FixedSupportSourceCotangents",
    ),
    "FixedSupportSolveDiagnostics": (
        ".types",
        "FixedSupportSolveDiagnostics",
    ),
    "fixed_support_source_vjp": (".autodiff", "fixed_support_source_vjp"),
    "minimize_gibbs_fixed_support": (
        ".autodiff",
        "minimize_gibbs_fixed_support",
    ),
    "minimize_gibbs_fixed_support_with_diagnostics": (
        ".autodiff",
        "minimize_gibbs_fixed_support_with_diagnostics",
    ),
}


__all__ = (
    "DifferentiableFixedSupportResult",
    "FixedSupportSolveDiagnostics",
    "FixedSupportSourceCotangents",
    "fixed_support_source_vjp",
    "minimize_gibbs_fixed_support",
    "minimize_gibbs_fixed_support_with_diagnostics",
)


def __getattr__(name: str) -> Any:
    """Load optional fixed-support autodiff objects only when requested."""

    export = _EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = export
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return stable exported names without importing numerical kernels."""

    return sorted(set(globals()) | set(__all__))
