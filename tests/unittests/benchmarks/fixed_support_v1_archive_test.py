"""Integrity checks for the immutable historical v1 evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MATRIX_PATH = (
    REPOSITORY_ROOT
    / "benchmarks"
    / "fixed_support_v2"
    / "fixed_support_v2_gpu_matrix.json"
)


def test_frozen_v1_artifacts_match_the_declared_hashes() -> None:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    artifacts = matrix["frozen_v1_baseline"]["artifacts"]

    assert len(artifacts) == 2
    for declaration in artifacts:
        path = REPOSITORY_ROOT / declaration["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            declaration["sha256"]
        )


def test_frozen_v1_json_remains_a_selected_case_record() -> None:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    json_declaration = next(
        artifact
        for artifact in matrix["frozen_v1_baseline"]["artifacts"]
        if artifact["path"].endswith(".json")
    )
    cases = json.loads(
        (REPOSITORY_ROOT / json_declaration["path"]).read_text(
            encoding="utf-8"
        )
    )

    assert len(cases) == 8
    assert len({case["label"] for case in cases}) == 8
    assert sum(bool(case["solver_success"]) for case in cases) == 3
