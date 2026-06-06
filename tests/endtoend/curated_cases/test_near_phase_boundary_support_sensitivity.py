"""End-to-end replay for the near-phase-boundary curated family."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _head_route_replay import replay_family


def test_near_phase_boundary_support_sensitivity_replays_successful_head_route_through_api() -> None:
    results = replay_family("near_phase_boundary_support_sensitivity")

    assert len(results) == 2
