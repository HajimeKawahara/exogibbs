"""Locate original source bytes for receipts predating the C2H default change."""

from pathlib import Path


def historical_source(root: Path, name: str) -> Path:
    """Resolve a recorded path without relabeling old results as current runs.

    The original preset moved byte-for-byte to curated_v1. Other changed
    study scripts are archived before their imports were made explicit.
    Current implementation checks must use their actual paths instead.
    """
    if name == "src/exogibbs/presets/curated.py":
        return root / "src/exogibbs/presets/curated_v1.py"
    archived = root / "tests/data/validation/curated_default_adoption/20261102/source" / name
    return archived if archived.is_file() else root / name
