"""End-to-end replay for all curated HEAD route cases."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _head_route_replay import replay_all_curated_rows


def test_all_14_curated_cases_replay_successful_head_route_through_api() -> None:
    results = replay_all_curated_rows()

    assert len(results) == 14
