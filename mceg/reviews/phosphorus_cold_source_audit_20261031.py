"""Inventory neutral P gases in the two pinned NASA9 research source files.

This is a coverage audit, not a claim that every candidate is physically reliable.
The coefficient artifact is written to ignored results, outside packaged data.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_phosphorus_cold_data import (
    ROOT,
    SOURCE_HASHES,
    VULCAN_REVISION,
    digest,
    read_records,
)


ELEMENTS = frozenset("H He C N O P S".split())
ALTERNATE_ANCHORS = {
    "PC": "CP", "HPO": "HPO", "PH2": "PH2", "PH3": "PH3",
    "PO2": "PO2", "PS": "PS", "P4": "P4",
}
ISOTOPE_EXCLUSIONS = {"PT": "Tritium phosphorous; source H3 denotes isotope mass, not three H atoms."}


def classify_record(name: str, selected: bool) -> tuple[str, str | None]:
    """Separate missing identities from alternate anchors and isotope records."""
    if selected:
        return "selected", None
    if name in ISOTOPE_EXCLUSIONS:
        return "excluded_isotope", ISOTOPE_EXCLUSIONS[name]
    if name in ALTERNATE_ANCHORS:
        return "alternate_anchor", ALTERNATE_ANCHORS[name]
    return "candidate", None


def candidate_id(name: str, occurrence: int, count: int) -> str:
    """Retain distinct isomers even when the source uses the same formula label."""
    return f"{name}__occ{occurrence}" if count > 1 else name


def build_inventory(source_directory: Path, fixture: Path) -> dict:
    """Read only P records after verifying the complete pinned source hashes."""
    selected = {(r["source_id"], r["source_line"])
                for r in json.loads(fixture.read_text())["records"].values()}
    records = []
    sources = {"cea2002_vulcan": "nasa9_2002_E.txt",
               "burcat2016_vulcan": "new_nasa9.TXT"}
    for source_id, filename in sources.items():
        path = source_directory / filename
        if digest(path.read_bytes()) != SOURCE_HASHES[filename]:
            raise ValueError(f"Source hash mismatch: {filename}")
        lines = path.read_text().splitlines()
        names = set()
        for index, header in enumerate(lines[:-1]):
            metadata = lines[index + 1]
            if (not header or not header[0].isalpha()
                    or not metadata[:2].strip().isdigit()
                    or len(metadata) < 80 or metadata[51:52] != "0"):
                continue
            elements = {metadata[k:k + 2].strip().capitalize()
                        for k in range(10, 50, 8) if metadata[k:k + 2].strip()}
            if "P" in elements and elements <= ELEMENTS:
                names.add(header.split()[0])
        for name, entries in read_records(path, source_id, names).items():
            for occurrence, entry in enumerate(entries):
                if "P" not in entry["composition"] or not set(entry["composition"]) <= ELEMENTS:
                    continue
                is_selected = (source_id, entry["source_line"]) in selected
                classification, note = classify_record(name, is_selected)
                records.append({
                    "source_name": name, "occurrence": occurrence,
                    "selected": is_selected, "classification": classification,
                    "accepted_candidate": classification == "candidate",
                    "candidate_id": candidate_id(name, occurrence, len(entries)),
                    "classification_note": note, **entry,
                })
    counts = dict(Counter(record["classification"] for record in records))
    return {
        "source_revision": VULCAN_REVISION, "source_sha256": SOURCE_HASHES,
        "fixture_sha256": digest(fixture.read_bytes()),
        "allowed_elements": sorted(ELEMENTS), "counts": counts,
        "identity_notes": {
            "H3PO__occ0": "HOPH2, hydroxyphosphine",
            "H3PO__occ1": "H3PO, phosphine oxide; distinct from HOPH2",
            "C2H7P__occ0": "C2H5PH2", "C2H7P__occ1": "(CH3)2PH",
            "C3H9P__occ0": "C3H7PH2", "C3H9P__occ1": "(CH3)3P",
        },
        "limitations": [
            "A candidate is available for a controlled network test, not recommended for production.",
            "Published fit intervals do not establish physical accuracy throughout those intervals.",
            "P4S3 is absent from both source files; availability is not network completeness.",
        ],
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, default=ROOT.parent / "VULCAN/thermo/NASA9")
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests/data/validation/phosphorus_cold/thermo.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/phosphorus_cold_validation/p_candidate_records.json")
    args = parser.parse_args()
    inventory = build_inventory(args.source_directory, args.fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps(inventory["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
