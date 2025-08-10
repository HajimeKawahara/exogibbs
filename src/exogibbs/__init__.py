try:
    from .ExoGibbs_version import __version__
except Exception:
    try:
        from importlib.metadata import version as _pkg_version
        __version__ = _pkg_version("ExoGibbs")
    except Exception:
        __version__ = "0.0.0.dev0"