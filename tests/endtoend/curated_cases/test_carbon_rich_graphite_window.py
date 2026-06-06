"""End-to-end replay for the carbon-rich graphite curated family."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _head_route_replay import replay_family


def test_carbon_rich_graphite_window_replays_successful_head_route_through_api() -> None:
    results = replay_family("carbon_rich_graphite_window")

    assert len(results) == 1
