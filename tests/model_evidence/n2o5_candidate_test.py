"""Replay N2O5 source-balanced candidates and finite substitutions offline."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import pending_nitrogen_inputs as inputs
from pending_candidate import formation_h, formation_h_jax, reaction_properties
from pending_equilibrium_impact import replay
from curated_adoption_history import historical_source

SLUG = "n2o5"
DATA = ROOT / f"tests/data/validation/{SLUG}_candidate/20261106"
IMPACT = ROOT / f"tests/data/validation/{SLUG}_equilibrium_impact/20261106"
MODEL = json.loads((DATA / "candidate.json").read_text())


def energy_equal(actual, expected):
    # Dimensional subtraction of ~MJ/mol values has platform-dependent roundoff.
    assert actual == pytest.approx(expected, rel=0, abs=1e-8)


def test_physical_source_anchor_and_single_energy_application():
    record = inputs.source_record(SLUG)
    assert record["janaf_D0_J_mol"] == None
    assert record["atct_D0_J_mol"] == 2151095.0
    assert MODEL == inputs.build(SLUG)
    assert MODEL["baseline_dataset"] == "curated-ch2-v1"
    if record["janaf_D0_J_mol"] is not None:
        delta = record["delta_D0_J_mol"]
        assert abs(record["formal_fit_intercept_Ra_J_mol"]-record["janaf_D0_J_mol"]) > 1
        old = deepcopy(record)
        old["delta_D0_J_mol"] = 0
        for t in (1500., 1937.31, 2500., 3999.91, 4000.):
            a, b = [inputs.independent_properties(r, t, "retained_energy") for r in (old, record)]
            for key in ("G_J_mol", "H_J_mol"):
                energy_equal(b[key]-a[key], -delta)
            assert a["S_J_mol_K"] == b["S_J_mol_K"]
            assert a["Cp_J_mol_K"] == b["Cp_J_mol_K"]
    else:
        with pytest.raises(ValueError, match="missing JANAF"):
            inputs.retained_spec(record)
    if SLUG in inputs.NASA_ALTERNATIVES:
        spec = inputs.nasa_spec(record)
        assert spec["native_energy_removal_D0_over_R_K"] == record["nasa_D0_J_mol"]/inputs.NASA_R
        assert spec["selected_energy_insertion_D0_over_R_K"] == record["atct_D0_J_mol"]/inputs.R
        for t in (1500., 1927.1, 2500., 3999.73, 4000.):
            actual = reaction_properties(t, spec)
            reference = inputs.independent_properties(record, t, "nasa9")
            for key in actual:
                energy_equal(actual[key], reference[key])
        assert abs(record["nasa_D0_J_mol"]*(1-inputs.R/inputs.NASA_R)) > 1
    with pytest.raises(AssertionError):
        energy_equal(1000000.+1e-5, 1000000.)


def test_continuous_calorics_source_fit_residual_and_jax_domain():
    import jax
    previous = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", False)
        report = inputs.validate(MODEL)
        assert not jax.config.x64_enabled
        assert report["temperature_count"] == 2752
        assert max(report["maximum_current_h_errors"].values()) < 1e-12
        for row in report["independent_source_checks"]:
            err = row["independent_source_property_errors"]
            assert max(err["G_J_mol"], err["H_J_mol"]) < 1e-8
            assert max(err["S_J_mol_K"], err["Cp_J_mol_K"]) < 2e-10
            for sample in row.get("retained_source_fit_residuals", []):
                assert abs(sample["preservation_error_ln_K"]) < 1e-12
        jax.config.update("jax_enable_x64", True)
        t = np.array([1500., 1917.53, 2799.37, 4000.])
        for variant in MODEL["variants"]+MODEL.get("selected_variants", []):
            for spec in variant["replacements"].values():
                f = jax.jit(lambda x: formation_h_jax(x, spec))
                expected = [formation_h(float(x), spec) for x in t]
                np.testing.assert_allclose(f(t), expected, rtol=0, atol=2e-12)
                np.testing.assert_allclose(jax.vmap(f)(t), expected, rtol=0, atol=2e-12)
                bad = np.array([1499., 4001., 0., np.nan, np.inf, -np.inf])
                assert np.isnan(f(bad)).all()
                for value in bad:
                    with pytest.raises(ValueError, match="outside"):
                        formation_h(float(value), spec)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_original_execution_receipts_are_not_later_test_receipts():
    for directory in (DATA, IMPACT):
        receipt = json.loads((directory/"receipt.json").read_text())
        assert receipt["result"] == "passed" and receipt["dtype"] == "float64"
        for path, digest in receipt["selected_file_sha256"].items():
            assert sha256(historical_source(ROOT, path, digest).read_bytes()).hexdigest() == digest, path
        for path, digest in receipt["output_sha256"].items():
            assert sha256((directory/path).read_bytes()).hexdigest() == digest, path
        assert str(Path(__file__).relative_to(ROOT)) not in receipt["selected_file_sha256"]


def test_fixed_network_endpoints_controls_and_independent_solver():
    summary = replay(DATA/"candidate.json", IMPACT)
    comparisons = [c for group in summary["independent_solver"] for c in group["comparisons"]]
    assert len(comparisons) == summary["independent_fastchem_runs"]
    assert len({g["case_index"] for g in summary["independent_solver"]}) == 2
    assert max(row["max_abs_delta_log10_x"] for row in comparisons) < 1e-6
    for row in comparisons:
        assert row["element_relative_residual"] < 1e-9
        assert row["charge_relative_residual"] < 1e-9
        assert row["mass_action_max_abs_ln"] < 1e-8
    # Nitrogen is explicitly constrained, in addition to generic H/C/O checks.
    for filename in ("diagnostics.json", "selected_diagnostics.json"):
        for row in json.loads((IMPACT/filename).read_text()):
            assert row["converged"]
    for network in ("ions", "neutral"):
        with np.load(IMPACT/f"{network}.npz") as archive:
            assert archive["ln_n"].shape[0] == len(MODEL["variants"])+1
            assert archive["ln_n"].shape[1] == 75
            elements = list(archive["elements"])
            matrix, budget = archive["formula_matrix"], archive["budget"]
            batches = [(archive["ln_n"], budget),
                       (archive["tight_ln_n"], budget[archive["tight_case_indices"]]),
                       (archive["lambda_ln_n"], budget[int(archive["path_case_index"])])]
            if len(MODEL.get("selected_variants", [])):
                batches.append((archive["selected_ln_n"], budget[archive["selected_case_indices"]]))
            for amounts, reference in batches:
                residual = abs(np.exp(amounts) @ matrix.T-reference)
                for element in MODEL["tracked_elements"]:
                    i = elements.index(element)
                    assert np.max(residual[..., i]/reference[..., i]) < 1e-9
            # Trace abundance can be stable while total-G exchange is unresolved.
            assert not archive["gibbs_ordering_resolved"].all()


def test_no_zero_kelvin_energy_is_fabricated():
    assert [v["name"] for v in MODEL["variants"]] == ["nasa"]
    record = inputs.source_record("n2o5")
    assert record["janaf_D0_J_mol"] is None and record["delta_D0_J_mol"] is None
    energy_equal(record["nasa_D0_J_mol"], 2152710.569)


def test_available_h298_anchor_does_not_invent_a_zero_kelvin_energy():
    import n2o5_h298_candidate as h298
    definition = json.loads((h298.DATA/"candidate.json").read_text())
    assert definition == h298.build()
    ledger = definition["energy_ledger"]
    assert ledger["physical_JANAF_D0_J_mol"] is None
    assert ledger["reference_temperature_K"] == 298.15
    assert ledger["JANAF_atomization_H298_J_mol"] == 2179934
    assert ledger["ATcT_atomization_H298_J_mol"] == 2176321
    assert ledger["delta_formation_GH_J_mol"] == 3613
    assert ledger["ATcT_1_228_Hf298_J_mol"]["N2O5"]-ledger["JANAF_Hf298_J_mol"]["N2O5"] == 3563
    report = h298.validate(definition)
    assert report["temperature_count"] == 2752
    assert report["physical_JANAF_D0_J_mol"] is None
    assert report["maximum_current_h_error"] < 1e-12
    for row in report["retained_source_fit_residuals"]:
        assert abs(row["preservation_error_ln_K"]) < 1e-12
    spec = definition["variants"][0]["replacements"]["N2O5"]
    old = {**spec, "delta_GH_J_mol": 0.}
    for t in (1500., 1927.13, 2500., 3999.17, 4000.):
        before, after = [reaction_properties(t, s) for s in (old, spec)]
        for key in ("G_J_mol", "H_J_mol"):
            energy_equal(after[key]-before[key], 3613)
        assert after["S_J_mol_K"] == before["S_J_mol_K"]
        assert after["Cp_J_mol_K"] == before["Cp_J_mol_K"]


def test_separate_h298_execution_retains_original_nasa_packet():
    cdata = ROOT/"tests/data/validation/n2o5_h298_candidate/20261106"
    edata = ROOT/"tests/data/validation/n2o5_h298_equilibrium_impact/20261106"
    for directory in (cdata, edata):
        receipt = json.loads((directory/"receipt.json").read_text())
        assert receipt["result"] == "passed"
        for path, digest in receipt["selected_file_sha256"].items():
            assert sha256(historical_source(ROOT, path, digest).read_bytes()).hexdigest() == digest
        for path, digest in receipt["output_sha256"].items():
            assert sha256((directory/path).read_bytes()).hexdigest() == digest
    summary = replay(cdata/"candidate.json", edata)
    assert summary["new_endpoint_solves"] == 150
    assert summary["tighter_control_solves"]+summary["path_control_solves"] == 22
    assert summary["independent_fastchem_runs"] == 4
    for group in summary["independent_solver"]:
        for comparison in group["comparisons"]:
            assert comparison["max_abs_delta_log10_x"] < 1e-6
    for network in ("ions", "neutral"):
        with np.load(edata/f"{network}.npz") as a:
            assert not a["gibbs_ordering_resolved"].any()
            elements, matrix, budget = list(a["elements"]), a["formula_matrix"], a["budget"]
            for amounts, ref in [(a["ln_n"], budget), (a["tight_ln_n"], budget[a["tight_case_indices"]]),
                                 (a["lambda_ln_n"], budget[int(a["path_case_index"])])]:
                residual = abs(np.exp(amounts)@matrix.T-ref)
                for element in ("N", "O"):
                    i = elements.index(element)
                    assert np.max(residual[..., i]/ref[..., i]) < 1e-9
