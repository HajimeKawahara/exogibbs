"""End-to-end replay for the heavy-element curated family."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _head_route_replay import replay_family


def test_complex_heavy_element_family_replays_successful_head_route_through_api() -> None:
    results = replay_family("complex_heavy_element_or_boron_titanium_zirconium_case")

    assert len(results) == 1
