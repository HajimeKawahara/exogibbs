"""Replay both HCP substitutions and quoted-width controls without fresh solves."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import hcp_candidate as candidate
from curated_adoption_history import historical_source

DATA = ROOT / "tests/data/validation/hcp_equilibrium_impact/20261106"
BASELINE = ROOT / "tests/data/validation/curated_tio2_adoption/20261106"


@pytest.fixture(scope="module", params=["ions", "neutral"])
def saved(request):
    with np.load(DATA / f"{request.param}.npz") as archive:
        return request.param, {key: archive[key].copy() for key in archive.files}


def fractions(values):
    return values-logsumexp(values, axis=-1, keepdims=True)


def check_constraints(data, amounts, h, budget, pressure):
    n, matrix = np.exp(amounts), data["formula_matrix"]
    elements, species = data["elements"].tolist(), data["species"].tolist()
    positive = np.asarray(budget).reshape(-1, len(elements))[0] > 0
    residual = abs(n @ matrix.T-budget)
    assert np.isfinite(amounts).all() and np.all(n > 0)
    assert np.max(residual[..., positive]/np.asarray(budget)[..., positive]) < 1e-5
    for element in ("H", "C", "N", "O", "P", "S"):
        j = elements.index(element)
        assert np.max(residual[..., j]/np.asarray(budget)[..., j]) < 1e-9
    if "e-" in elements:
        charge = matrix[elements.index("e-")]
        assert np.max(abs(n @ charge)/(n @ abs(charge))) < 1e-12
    references = [species.index("e1-" if e == "e-" else e+"1") for e in elements]
    logp = fractions(amounts)+np.log(np.asarray(pressure))[..., None]
    assert np.max(abs(logp-logp[..., references] @ matrix+h)) < 1e-8


def test_common_baseline_actual_factory_and_target_only_changes(saved):
    import jax
    from exogibbs.presets import curated_tio2
    from hcp_equilibrium_impact import candidate_setup
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    name, data = saved
    try:
        setup = curated_tio2.chemsetup(ions=name == "ions")
        assert setup.metadata["dataset"] == "curated-tio2-v1"
        assert data["ln_n"].shape == data["h"].shape == (3, 75, 537 if name == "ions" else 421)
        np.testing.assert_array_equal(data["variants"], ["current", *candidate.ANCHORS])
        np.testing.assert_allclose(setup.hvector_func(data["temperature_K"]), data["h"][0], rtol=0, atol=1e-12)
        with np.load(BASELINE / f"{name}.npz") as baseline:
            for key in ("species", "elements", "formula_matrix", "budget", "temperature_K", "pressure_bar", "C_over_O"):
                np.testing.assert_array_equal(data[key], baseline[key])
            np.testing.assert_array_equal(data["ln_n"][0], baseline["ln_n"])
        target = data["species"].tolist().index(candidate.SPECIES)
        for i, anchor in enumerate(candidate.ANCHORS, 1):
            updated = candidate_setup(setup, anchor)
            assert updated.metadata["production_adoption"] is False
            np.testing.assert_allclose(updated.hvector_func(data["temperature_K"]), data["h"][i], rtol=0, atol=1e-12)
            np.testing.assert_array_equal(np.delete(data["h"][i]-data["h"][0], target, axis=-1), 0)
            np.testing.assert_allclose(data["h"][i, :, target], [candidate.formation_h(t, anchor) for t in data["temperature_K"]], atol=2e-12, rtol=0)
        check_constraints(data, data["ln_n"], data["h"], data["budget"], data["pressure_bar"])
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_cold_controls_and_quoted_width_edges(saved):
    _, data = saved
    selected, target = data["tight_case_indices"], data["species"].tolist().index(candidate.SPECIES)
    np.testing.assert_array_equal(selected, [52, 53])
    tight, original = fractions(data["tight_ln_n"]), fractions(data["ln_n"][:, selected])
    delta = tight-original
    assert abs(delta).max() < 1e-4 and abs(delta[..., target]).max() < 1e-8
    assert abs(np.exp(tight)-np.exp(original)).max() < 1e-9
    visible = np.maximum(np.exp(tight), np.exp(original)) >= 1e-12
    assert abs(delta[visible]).max() < 2e-6
    check_constraints(data, data["tight_ln_n"], data["h"][:, selected], data["budget"][selected], data["pressure_bar"][selected])
    for ai, anchor in enumerate(candidate.ANCHORS):
        width = candidate.load_candidate()["anchors"][anchor]["reported_uncertainty_J_mol"]
        for oi, sign in enumerate((-1, 1)):
            difference = data["uncertainty_h"][ai, oi]-data["h"][ai+1, selected]
            np.testing.assert_array_equal(np.delete(difference, target, axis=-1), 0)
            np.testing.assert_allclose(difference[:, target], sign*width/(candidate.R*data["temperature_K"][selected]), rtol=0, atol=1e-12)
    check_constraints(data, data["uncertainty_ln_n"], data["uncertainty_h"], data["budget"][selected], data["pressure_bar"][selected])
    assert np.max(data["uncertainty_ln_n"][:, 1, :, target]-data["uncertainty_ln_n"][:, 0, :, target]) < 0


def test_same_model_gibbs_ordering_is_scoped_not_universally_resolved(saved):
    _, data = saved
    refs = [data["species"].tolist().index("e1-" if e == "e-" else e+"1") for e in data["elements"]]
    for ai in range(2):
        logs, h = [data[key][[0, ai+1]].astype(np.longdouble) for key in ("ln_n", "h")]
        n = np.exp(logs)
        logp = logs-np.log(n.sum(axis=-1))[..., None]+np.log(data["pressure_bar"].astype(np.longdouble))[None, :, None]
        raw = np.asarray([np.sum(n[1]*(h[i]+logp[1])-n[0]*(h[i]+logp[0]), axis=-1) for i in (0, 1)])
        corrected = raw-((n[1]-n[0]) @ data["formula_matrix"].T*(h+logp)[..., refs]).sum(axis=-1)
        np.testing.assert_allclose(raw, data["delta_g_raw"][ai], atol=5e-14, rtol=1e-7)
        np.testing.assert_allclose(corrected, data["delta_g_corrected"][ai], atol=5e-14, rtol=1e-7)
        scale = data["gibbs_numerical_scale"][ai]
        resolved = (data["delta_g_corrected"][ai, 0] > scale) & (data["delta_g_corrected"][ai, 1] < -scale)
        np.testing.assert_array_equal(resolved, data["gibbs_ordering_resolved"][ai])
        assert resolved.sum() == (64 if ai == 0 else 65)


def test_unfloored_mass_action_and_reported_phosphorus_response(saved):
    name, data = saved
    summary = next(row for row in json.loads((DATA / "summary.json").read_text())["networks"] if row["network"] == name)
    species, logx = data["species"].tolist(), fractions(data["ln_n"])
    target, p = species.index(candidate.SPECIES), data["elements"].tolist().index("P")
    share = np.exp(data["ln_n"])*data["formula_matrix"][p]/data["budget"][:, p, None]
    for i, detail in enumerate(summary["anchors"], 1):
        delta = logx[i]-logx[0]
        predicted = sum(delta[:, species.index(a+"1")] for a in ("C", "H", "P"))-(data["h"][i, :, target]-data["h"][0, :, target])
        np.testing.assert_allclose(delta[:, target], predicted, atol=1e-10, rtol=0)
        ratios = np.exp(delta[:, target])
        assert detail["HCP_ratio_min"] == pytest.approx(ratios.min())
        assert detail["HCP_ratio_max"] == pytest.approx(ratios.max())
        assert detail["max_phosphorus_share_after"] == pytest.approx(share[i, :, target].max())
        assert detail["highlighted_case"]["case_index"] == int(np.argmax(abs(share[i, :, target]-share[0, :, target])))
        for point in [detail["highlighted_case"], *detail["phosphorus_carrier_extrema"]]:
            k, j = point["case_index"], species.index(point["species"])
            assert point["x_before"] == pytest.approx(np.exp(logx[0, k, j]), rel=1e-12, abs=0)
            assert point["x_after"] == pytest.approx(np.exp(logx[i, k, j]), rel=1e-12, abs=0)
            assert point["phosphorus_share_after"] == pytest.approx(share[i, k, j], rel=1e-12, abs=0)


def test_execution_counts_convergence_independent_solver_and_hashes():
    summary = json.loads((DATA / "summary.json").read_text())
    assert summary["curated_update"] == "none"
    assert [summary[k] for k in ("new_endpoint_solves", "reused_baseline_cases", "new_control_solves", "new_tighter_solves", "new_quoted_energy_edge_solves", "independent_fastchem_solves")] == [300, 150, 28, 12, 16, 4]
    for name, count in [("diagnostics.json", 300), ("selected_diagnostics.json", 28)]:
        rows = json.loads((DATA / name).read_text())
        assert len(rows) == count and all(row["converged"] for row in rows)
    # Four executions are two candidate comparisons at one condition; the same
    # baseline is intentionally executed twice, not four distinct conditions.
    independent = summary["independent_solver"]
    assert len(independent) == 2 and {r["case_index"] for r in independent} == {53}
    for entry in independent:
        assert len(entry["comparisons"]) == 2
        for row in entry["comparisons"]:
            assert row["max_abs_delta_log10_x"] < 2e-7
            assert row["element_relative_residual"] < 1e-8
            assert row["charge_relative_residual"] < 1e-8
            assert row["mass_action_max_abs_ln"] < 1e-8
    receipt = json.loads((DATA / "receipt.json").read_text())
    assert receipt["result"] == "passed" and receipt["dtype"] == "float64"
    assert receipt["fastchem_binary_sha256"] == "64cb8d525346cb3b68116d53de08f74855744db01b161b27fa753973f95680a2"
    for name, digest in receipt["selected_file_sha256"].items():
        assert sha256(historical_source(ROOT, name, digest).read_bytes()).hexdigest() == digest, name
    for name, digest in receipt["output_sha256"].items():
        assert sha256((DATA / name).read_bytes()).hexdigest() == digest, name
