"""End-to-end replay for the solar metal-sulfide curated family."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _head_route_replay import replay_family


def test_solar_metal_sulfide_or_fe_ni_s_region_replays_successful_head_route_through_api() -> None:
    results = replay_family("solar_metal_sulfide_or_Fe_Ni_S_region")

    assert len(results) == 2
