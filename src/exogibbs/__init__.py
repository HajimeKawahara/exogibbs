from importlib import import_module
from importlib.metadata import version as _pkg_version


def _load_generated_version() -> str:
    return import_module(".exogibbs_version", __name__).__version__


def _load_version() -> str:
    try:
        return _load_generated_version()
    except (ImportError, AttributeError):
        return _pkg_version("exogibbs")


__version__ = _load_version()
