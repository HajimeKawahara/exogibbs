"""Historical receipts require exact original bytes, not the current alias."""

from hashlib import sha256
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
from curated_adoption_history import historical_source


def test_adoption_resolver_matches_the_recorded_version_and_rejects_changes(tmp_path):
    name = "src/exogibbs/presets/curated.py"
    current = tmp_path / name
    current.parent.mkdir(parents=True)
    current.write_bytes(b"current default")
    snapshots = []
    for model, content in (("co2", b"previous TiO default"), ("h2o", b"previous CO2 default")):
        path = tmp_path / f"tests/data/validation/curated_{model}_adoption/20261104/source" / name
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        snapshots.append(path)
    for path in [current] + snapshots:
        expected = sha256(path.read_bytes()).hexdigest()
        assert historical_source(tmp_path, name, expected) == path
    expected = sha256(snapshots[0].read_bytes()).hexdigest()
    snapshots[0].write_bytes(b"changed old source")
    with pytest.raises(ValueError, match="recorded hash"):
        historical_source(tmp_path, name, expected)
