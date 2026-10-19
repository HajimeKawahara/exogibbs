"""Replay one authored literature review without distributing or parsing its PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mceg import assess_conditions, load_contract, restore_v01, trace_provenance
from mceg.check import source_audit
from mceg.migration import write_document
from mceg.schema_v02 import ContractV02
from mceg.validator import read_document


HERE = Path(__file__).resolve().parent
BASELINE = HERE / "baseline.yaml"
REVIEW = HERE / "review.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(baseline: Path = BASELINE, pdf: Path | None = None) -> ContractV02:
    review = read_document(REVIEW)
    if sha256(baseline) != review["baseline_sha256"]:
        raise ValueError("Baseline changed; review it before replaying this literature update")
    if pdf is not None and sha256(pdf) != review["private_source"]["sha256"]:
        raise ValueError("Private PDF bytes differ from the inspected copy; a new source review is required")
    before = load_contract(baseline)
    data = before.model_dump()
    revision = "sha256:" + sha256(REVIEW)
    for item in review["evidence"]:
        if item["id"] in review["private_source_evidence"]:
            digest = review["private_source"]["sha256"]
            item["artifacts"] = [{"source": "urn:sha256:" + digest, "sha256": digest}]
        data["evidence"].append(item)
    data["conditions"].extend({**c, "revision": revision} for c in review["conditions"])
    data["follow_ups"].extend(review["follow_ups"])
    data["follow_up_resolutions"].extend(review["follow_up_resolutions"])
    physical = next(c for c in before.claims if c.id == "physical_accuracy")
    data["assessments"].append({
        "id": "physical_accuracy:full_text:2026-10-19", "kind": "evidence_support",
        "subject": "imported", "context": review["context"],
        "claim": physical.id, "claim_revision": physical.revision,
        "status": "unknown", "origin": "reviewed",
        "method": {"id": "h2_full_text_review", "revision": revision,
                   "description": "Replay the authored 2026-10-19 review; no automatic source inspection or support inference."},
        "evidence": ["implementation_full_text", "experimental_fit_full_text",
                     "experimental_runs_full_text", "calibration_adoption_full_text", "infrared_recalibration"],
        "reasons": review["physical_assessment_reasons"],
    })
    after = ContractV02.model_validate(data)
    context = next(c for c in after.contexts if c.id == review["context"])
    after = assess_conditions(after, context, "imported")
    if restore_v01(after) != restore_v01(before):
        raise ValueError("The update changed imported historical content")
    return after


def summarize(after: ContractV02) -> dict:
    before = load_contract(BASELINE)
    review = read_document(REVIEW)
    comparisons = []
    for old, new in review["trace_comparisons"].items():
        prior, current = trace_provenance(before, old), trace_provenance(after, new)
        comparisons.append({"prior_evidence": old, "reviewed_evidence": new,
                            "prior_audit_complete": prior["recorded_audit_complete"],
                            "reviewed_audit_complete": current["recorded_audit_complete"],
                            "remaining_issues": current["issues"]})
    new_conditions = {c["id"] for c in review["conditions"]}
    selected = next(a for a in after.assessments if a.id == "physical_accuracy:full_text:2026-10-19")
    audit, traces = source_audit(after, selected.evidence)
    return {
        "format": "h2_literature_update_trial_v1",
        "review_sha256": sha256(REVIEW), "generator_sha256": sha256(Path(__file__)),
        "baseline_sha256": sha256(BASELINE),
        "private_pdf_sha256": review["private_source"]["sha256"],
        "source_inspection_repeated_by_replay": False,
        "historical_records_preserved": {
            key: getattr(after, key)[:len(getattr(before, key))] == getattr(before, key)
            for key in ("claims", "evidence", "subjects", "contexts", "conditions", "assessments", "follow_ups",
                        "follow_up_resolutions")
        },
        "v01_restoration_equal": restore_v01(after) == restore_v01(before),
        "trace_comparisons": comparisons,
        "new_range_assessments": [
            {"condition": a.condition, "context": a.context, "classification": a.classification}
            for a in after.assessments if a.kind == "range_membership" and a.condition in new_conditions
        ],
        "physical_assessments": [
            {"id": a.id, "context": a.context, "status": a.status, "origin": a.origin}
            for a in after.assessments if a.kind == "evidence_support" and a.claim == "physical_accuracy"
        ],
        "selected_assessment_audit": {"assessment": selected.id, "check": audit,
                                      "audited_evidence": [trace["evidence"] for trace in traces]},
        "follow_up_resolutions": [r.model_dump() for r in after.follow_up_resolutions],
        "interpretation": "The selected assessment audits only its evidence dependencies, retaining the unresolved calibration and unknown physical support. Authored completion records direct finished inspection tasks to new evidence; historical sources and assessments are unchanged. Material matching and evidence supersession are not inferred. The PDF hash identifies private bytes, not access rights or independent attestation.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New v0.2 contract; never overwritten")
    parser.add_argument("--pdf", type=Path, help="Optional authorized local copy for byte-identity checking only")
    args = parser.parse_args(argv)
    try:
        after = build(pdf=args.pdf)
        report = summarize(after)
        write_document(after.model_dump(exclude_none=True, exclude_defaults=True), args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
