"""End-to-end replay for the solar water-condensation curated family."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _head_route_replay import replay_family


def test_solar_water_condensation_replays_successful_head_route_through_api() -> None:
    results = replay_family("solar_water_condensation")

    assert len(results) == 2
