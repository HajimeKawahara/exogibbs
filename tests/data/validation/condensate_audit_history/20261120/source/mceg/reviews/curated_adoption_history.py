"""Resolve immutable evidence sources across versioned curated adoptions."""

from hashlib import sha256
from pathlib import Path

from curated_tio_history import historical_source as earlier_source


def historical_source(root: Path, name: str, expected: str) -> Path:
    """Find the recorded bytes; current production receipts use direct paths."""
    try:
        return earlier_source(root, name, expected)
    except ValueError:
        pass
    snapshots = (root / "tests/data/validation").glob("curated_*_adoption/*/source")
    for snapshot in sorted(snapshots):
        candidate = snapshot / name
        if candidate.is_file() and sha256(candidate.read_bytes()).hexdigest() == expected:
            return candidate
    raise ValueError(f"No historical source matches the recorded hash: {name}")
