"""End-to-end replay for the carbon-rich CaS/MgS/AlN curated family."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _head_route_replay import replay_family


def test_carbon_rich_cas_mgs_aln_window_replays_successful_head_route_through_api() -> None:
    results = replay_family("carbon_rich_CaS_MgS_AlN_window")

    assert len(results) == 1
