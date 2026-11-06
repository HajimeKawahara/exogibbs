"""Locate source bytes from receipts predating the TiO default adoption."""

from hashlib import sha256
from pathlib import Path

from curated_history import historical_source as original_source


def historical_source(root: Path, name: str, expected: str) -> Path:
    """Match the recorded hash without rebinding old results to current code.

    New production receipts must check their actual paths directly. This
    resolver is only for immutable records of earlier curated datasets.
    """
    candidates = [root / name,
                  root / "tests/data/validation/curated_tio_adoption/20261103/source" / name,
                  original_source(root, name)]
    for candidate in candidates:
        if candidate.is_file() and sha256(candidate.read_bytes()).hexdigest() == expected:
            return candidate
    raise ValueError(f"No historical source matches the recorded hash: {name}")
