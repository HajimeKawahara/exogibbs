"""Check the frozen release inventory without changing historical evidence."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "mceg/releases/gas_curation/0.1.0.json"


def _load(path):
    return json.loads((ROOT / path).read_text())


def _hash(path):
    return sha256((ROOT / path).read_bytes()).hexdigest()


def _section(text, name):
    """Read a top-level block from these frozen, explicitly authored YAML files."""
    match = re.search(r"(?ms)^" + re.escape(name) + r":\n(.*?)(?=^[A-Za-z_][\w]*:|\Z)", text)
    return match.group(1) if match else ""


def _items(text, name):
    # This is a bounded inspection of the existing block-list layout, not a
    # general YAML parser or a new dependency on optional MCEG/PyYAML.
    block = _section(text, name)
    result = {}
    for item in re.split(r"(?m)^- ", block)[1:]:
        match = re.search(r"(?m)^(?:  )?id: ([^\n]+)$", item)
        if match:
            result[match.group(1).strip("\"'")] = item
    return result


def _statuses(text, name):
    result = {}
    for identifier, block in _items(text, name).items():
        match = re.search(r"(?m)^(?:  )?status: ([^\n]+)$", block)
        if match:
            result[identifier] = match.group(1).strip("\"'")
    return result


def test_release_identity_and_all_base_layers_remain_distinct_from_software_version():
    release = _load(MANIFEST)
    assert release["release_id"] == "gas-curation-v0.1.0"
    assert release["curation_version"] == "0.1.0"
    assert release["scientific_snapshot_revision"] == "f580e66157edbe6f604e2748d8e1a25d46d08b86"
    dataset = release["dataset"]
    assert dataset["id"] == "curated-gas-v2"
    assert dataset["model_id"] == "exogibbs.curated_gas.v2"
    assert dataset["preset"] == "exogibbs.presets.curated_v2.chemsetup"
    assert dataset["temperature_range_K"] == [1500, 4000]
    assert dataset["standard_pressure_Pa"] == 100000
    chain = dataset["base_data_chain"]
    assert len({layer["id"] for layer in chain}) == len(chain) == 16
    assert chain[0]["id"] == "curated-gas-v1"
    for index, layer in enumerate(chain):
        data = _load(layer["path"])
        assert _hash(layer["path"]) == layer["sha256"]
        assert data["id"] == layer["id"]
        assert layer["base_dataset"] == data.get("base_dataset")
        assert layer["base_data_sha256"] == data.get("base_data_sha256")
        if index:
            assert data["base_dataset"] == chain[index - 1]["id"]
            assert data["base_data_sha256"] == chain[index - 1]["sha256"]
    assert chain[-1]["id"] == dataset["id"]
    assert chain[-1]["sha256"] == dataset["data_sha256"]
    for path, expected in dataset["implementation_file_sha256"].items():
        assert path != "src/exogibbs/presets/curated.py"
        assert _hash(path) == expected
    for source in dataset["inherited_baseline_inputs"]:
        assert _hash(source["path"]) == source["sha256"]
    assert "publish_commit" not in release and "self_sha256" not in release


def test_all_108_molecular_identities_are_covered_without_upgrading_nasa_exclusions():
    release = _load(MANIFEST)
    coverage = release["coverage"]
    inventory = coverage["target_inventory"]
    assert _hash(inventory["path"]) == inventory["sha256"]
    original = _load(inventory["path"])
    targets = coverage["targets"]
    assert len(targets) == len({t["species_id"] for t in targets}) == 108
    assert [{key: row[key] for key in old} for old, row in zip(original, targets)] == original
    assert Counter(t["scope"] for t in targets) == coverage["groups"] == {
        "main93": 93, "excluded_from_NASA14": 14, "additional_C3H": 1}
    comparison = _load(release["evidence"]["official_fastchem4_comparison"]["comparison"]["path"])
    by_id = {row["species_id"]: row for row in comparison["targets"]}
    for row in targets:
        assert row["composition"] == by_id[row["species_id"]]["composition"]
        if row["scope"] != "main93":
            assert row["nasa_id"] is None
        assert any(link["role"] == "source_review" for link in row["records"])
        assert any(link["role"] == "current_implementation_layer" for link in row["records"])
        for link in row["records"]:
            assert link["path"] in release["records"]
    by_id = {row["species_id"]: row for row in targets}
    # These distinctions are lost by formula-only matching or a generic source
    # review lookup. Gas H2 must not point at the separate H2 solubility model.
    assert by_id["C1N2_cnn"]["molecule"] == "CNN"
    assert by_id["C1N2_ncn"]["molecule"] == "NCN"
    assert by_id["C2N1"]["molecule"] == "CNC"
    assert by_id["H1N1O2cis"]["molecule"] == "cis-HNO2"
    assert by_id["H1N1O2trans"]["molecule"] == "trans-HNO2"
    assert any(link["path"].endswith("phosphorus_provenance.yaml") for link in by_id["O6P4"]["records"])
    assert all("solubility" not in link["path"] for link in by_id["H2"]["records"])


def test_implemented_sources_and_runtime_policy_are_independent_of_review_completion():
    release = _load(MANIFEST)
    chain = [_load(layer["path"]) for layer in release["dataset"]["base_data_chain"]]
    base = chain[0]
    implemented = set().union(*(set(base[key]) for key in
                                ("nasa9_reactions", "janaf_reactions", "logk_energy_shifts_K", "partition_reactions")))
    policies = dict(base["decisions"])
    layer_for = {sid: base["id"] for sid in policies}
    for data in chain[1:]:
        choices = data.get("decisions") or {data["species"]: data["decision"]}
        implemented.update(choices)
        policies.update(choices)
        layer_for.update({sid: data["id"] for sid in choices})
    coverage = release["coverage"]
    for row in coverage["targets"]:
        sid = row["species_id"]
        expected = "adopted_source" if sid in implemented else "retained_source"
        assert row["decision"] == expected
        assert row["implementation_dataset"] == layer_for.get(sid, base["id"])
        assert row["runtime_source_status"] == policies.get(sid, {"status": "inherited_unreviewed"})["status"]
        # A candidate contract link cannot silently become an adopted variant.
        expected_variants = chain[-1]["candidate_references"].get(sid, [])
        assert row["selected_gas_v2_candidate_variants"] == expected_variants
    assert Counter(row["decision"] for row in coverage["targets"]) == coverage["decision_counts"]
    assert Counter(row["runtime_source_status"] for row in coverage["targets"]) == coverage["runtime_source_status_counts"]
    by_id = {row["species_id"]: row for row in coverage["targets"]}
    assert by_id["C1K1N1"]["decision"] == by_id["C1N1Na1"]["decision"] == "retained_source"
    assert by_id["C2N2"]["selected_gas_v2_candidate_variants"] == []
    assert by_id["C1N2_cnn"]["source_policy_is_provisional"]
    assert by_id["O6P4"]["source_policy_is_provisional"]
    assert any(row["runtime_source_status"] == "inherited_unreviewed" for row in coverage["targets"])


def test_original_contract_statuses_followups_and_partial_resolutions_are_preserved():
    release = _load(MANIFEST)
    counts = Counter()
    for path, record in release["records"].items():
        text = (ROOT / path).read_text()
        assert _hash(path) == record["sha256"], path
        assert record["assessments"] == _statuses(text, "assessments"), path
        assert record["legacy_claim_statuses"] == _statuses(text, "claims"), path
        assert record["follow_ups"] == list(_items(text, "follow_ups")), path
        resolutions = _section(text, "follow_up_resolutions")
        original_ids = re.findall(r"(?m)^- follow_up: ([^\n]+)$", resolutions)
        assert [item["follow_up"] for item in record["follow_up_resolutions"]] == original_ids
        normalized = re.sub(r"\s+", " ", resolutions)
        for item in record["follow_up_resolutions"]:
            assert item["follow_up"] in record["follow_ups"]
            assert re.sub(r"\s+", " ", item["note"]) in normalized
            assert set(item["evidence"]) <= set(_items(text, "evidence"))
        counts.update(contracts=1, assessments=len(record["assessments"]),
                      legacy_claim_statuses=len(record["legacy_claim_statuses"]),
                      follow_ups=len(record["follow_ups"]),
                      follow_up_resolution_records=len(record["follow_up_resolutions"]),
                      unknown_statuses_preserved=sum(value == "unknown" for value in
                          [*record["assessments"].values(), *record["legacy_claim_statuses"].values()]))
    assert dict(counts) == release["record_counts"]
    assert counts["unknown_statuses_preserved"] > 0
    assert release["records"]["mceg/contracts/curated_gas_v2_adoption.yaml"]["assessments"]["physical_accuracy_review"] == "unknown"
    assert release["records"]["mceg/contracts/nh3_source_review.yaml"]["assessments"]["physical_accuracy_review"] == "unknown"
    # A completed finite diagnostic does not resolve VO2 thermochemistry.
    vo2 = release["records"]["mceg/contracts/vo2_source_review.yaml"]
    assert "run_finite_diagnostics" in {r["follow_up"] for r in vo2["follow_up_resolutions"]}
    assert "decide_production_candidate" in vo2["follow_ups"]
    assert "unknown" in vo2["assessments"].values()


def test_existing_production_and_comparison_receipts_are_not_new_or_double_counted_runs():
    release = _load(MANIFEST)
    for path, expected in release["source_artifact_sha256"].items():
        assert _hash(path) == expected, path
    actual = release["evidence"]["current_combined_production"]
    summary = _load(actual["summary"]["path"])
    for field, source in (("final_fresh_endpoints", "new_production_solves"),
                          ("independent_tighter_controls", "tighter_reference_solves"),
                          ("joint_path_nodes", "joint_path_solves"),
                          ("independent_fastchem_comparisons", "independent_fastchem_runs"),
                          ("reused_baseline_endpoints", "reused_baseline_cases")):
        assert actual[field] == summary[source]
    receipt_path = Path(actual["receipt"]["path"])
    receipt = _load(receipt_path)
    assert receipt["result"] == "passed"
    assert receipt["selected_file_sha256"][release["dataset"]["base_data_chain"][-1]["path"]] == release["dataset"]["data_sha256"]
    ids = {row["species_id"] for row in release["coverage"]["targets"]}
    for network, expected in (("ions", 537), ("neutral", 421)):
        path = receipt_path.parent / (network + ".npz")
        assert _hash(path) == receipt["output_sha256"][path.name]
        with np.load(ROOT / path) as archive:
            assert len(archive["species"]) == expected
            assert ids <= set(archive["species"].tolist())
    comparison = release["evidence"]["official_fastchem4_comparison"]
    report = _load(comparison["comparison"]["path"])
    assert comparison["fastchem_ratio_samples"] == report["sample_count"] == 540
    assert comparison["equilibrium_abundance_solves"] == report["equilibrium_abundance_solves"] == 0
    assert comparison["frozen_NASA_ratio_samples"] == sum(row["ln_K_frozen_NASA"] is not None for row in report["samples"]) == 465
    assert comparison["temperatures_K"] == report["temperatures_K"]
    for block in release["evidence"].values():
        for value in block.values():
            if isinstance(value, dict) and set(value) == {"path", "sha256"}:
                assert _hash(value["path"]) == value["sha256"]
