"""Replay the CN source investigation without production changes."""

import argparse
import json
from pathlib import Path

from carbon_radical_source_review import review as shared_review


def review() -> dict:
    """Return the scoped source comparison."""
    return shared_review("cn")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(review(), indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
