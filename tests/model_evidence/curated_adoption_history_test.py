"""Historical receipts require exact original bytes, not the current alias."""

from hashlib import sha256
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
from curated_adoption_history import historical_source


@pytest.mark.parametrize("directories", [
    ("curated_co2_adoption/20261104", "curated_h2o_adoption/20261104"),
    ("condensate_audit_history/20261120", "condensate_audit_history/20990101"),
])
def test_adoption_resolver_matches_the_recorded_version_and_rejects_changes(tmp_path, directories):
    name = "src/exogibbs/presets/curated.py"
    current = tmp_path / name
    current.parent.mkdir(parents=True)
    current.write_bytes(b"current default")
    snapshots = []
    for directory, content in zip(directories, (b"previous TiO default", b"previous CO2 default")):
        path = tmp_path / "tests/data/validation" / directory / "source" / name
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


def test_original_audit_and_resolver_bytes_remain_available():
    import json

    directory = ROOT / "tests/data/validation/condensate_audit_history/20261120"
    receipt = json.loads((directory / "receipt.json").read_text())
    for name, expected in receipt["source_sha256"].items():
        assert historical_source(ROOT, name, expected) == directory / "source" / name
        assert sha256((directory / "source" / name).read_bytes()).hexdigest() == expected
        with pytest.raises(ValueError, match="recorded hash"):
            historical_source(ROOT, name, "0" * 64)
