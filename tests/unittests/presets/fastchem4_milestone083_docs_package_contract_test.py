"""Regression tests for the FC4-M083 experimental API docs contract."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DOCUMENTS = ROOT / "documents"
RESULTS = ROOT / "results"
JAPANESE_OR_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")

M083_FILES = [
    DOCUMENTS / "diagnostics" / "positive_support_experimental.rst",
    DOCUMENTS / "exogibbs" / "exogibbs.diagnostics.rst",
    DOCUMENTS / "index.rst",
    ROOT / "src" / "exogibbs" / "diagnostics" / "condensate_positive_support_experimental.py",
    RESULTS / "fastchem4_milestone083_docs_package_contract.json",
    RESULTS / "fastchem4_milestone083_docs_package_contract_compact.json",
    RESULTS / "fastchem4_milestone083_docs_package_contract_compact.md",
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_m083_docs_page_defines_release_ready_experimental_contract():
    text = (DOCUMENTS / "diagnostics" / "positive_support_experimental.rst").read_text(
        encoding="utf-8"
    )

    assert "release-ready as an experimental diagnostics API" in text
    assert "enable_experimental_positive_support=True" in text
    assert "seed_fraction <= 1.0e-3" in text
    assert "max_seed_amount <= 1.0e-3" in text
    assert "KKT residual reporting as solver-stage diagnostics" in text
    assert "FastChem4 public, runtime, trace" in text
    assert "production return signature changes" in text


def test_m083_docs_are_linked_from_sphinx_index_and_api_tree():
    index = (DOCUMENTS / "index.rst").read_text(encoding="utf-8")
    api = (DOCUMENTS / "exogibbs" / "exogibbs.rst").read_text(encoding="utf-8")
    diagnostics = (DOCUMENTS / "exogibbs" / "exogibbs.diagnostics.rst").read_text(
        encoding="utf-8"
    )

    assert "diagnostics/positive_support_experimental.rst" in index
    assert "exogibbs.diagnostics" in api
    assert "condensate_positive_support_experimental" in diagnostics


def test_m083_artifact_records_sphinx_build_and_contract_status():
    contract = load_json(RESULTS / "fastchem4_milestone083_docs_package_contract.json")
    compact = load_json(RESULTS / "fastchem4_milestone083_docs_package_contract_compact.json")
    semantic = load_json(RESULTS / "condensate_fastchem_semantic_levers.json")

    assert contract["docs_contract_status"] == "complete"
    assert contract["sphinx_build_status"] == "passed_with_existing_warnings"
    assert compact["decision"] == "FC4_M083_RELEASE_READY_EXPERIMENTAL_DOCS_CONTRACT_COMPLETE"
    assert compact["production_behavior_changed"] is False
    assert compact["preset_default_wiring_changed"] is False
    assert semantic["fastchem4_current_milestone"] == "FC4-M083"
    assert semantic["fastchem4_positive_support_public_experimental_docs_status"] == "complete"


def test_m083_repository_files_are_english_only():
    missing = [path for path in M083_FILES if not path.exists()]
    assert not missing
    violations = []
    for path in M083_FILES:
        if JAPANESE_OR_CJK_RE.search(path.read_text(encoding="utf-8")):
            violations.append(str(path.relative_to(ROOT)))
    assert not violations
