"""End-to-end replay for the low-temperature strong-condensation curated family."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _head_route_replay import replay_family


def test_lowt_strong_condensation_budget_stress_replays_successful_head_route_through_api() -> None:
    results = replay_family("lowT_strong_condensation_budget_stress")

    assert len(results) == 2
