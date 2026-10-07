"""Fresh archive replay with native alloy chemistry and a recorded host control."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "examples" / "metal_silicate"
ARCHIVE = ROOT / "results" / "subneptune_taxonomy" / "20260911"
LABELS = ("melts_present", "melts_absent")


@pytest.fixture(scope="module")
def replay_modules():
    provider = pytest.importorskip("exoeos")
    if not hasattr(provider, "MaFeSiOHLiquid"):
        pytest.skip("The optional ExoEOS installation predates MaFeSiOHLiquid.")
    names = tuple(path.stem for path in EXAMPLES.glob("*.py"))
    previous = {name: sys.modules.get(name) for name in names}
    old_path = sys.path[:]
    try:
        sys.path.insert(0, str(EXAMPLES))
        for name in names:
            sys.modules.pop(name, None)
        replay = importlib.import_module("revalidate_melts_archive")
        full = importlib.import_module("full_potential")
        return replay, full
    finally:
        sys.path[:] = old_path
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture
def replay_control(replay_modules, monkeypatch, tmp_path):
    replay, full = replay_modules
    baseline = {label: json.loads((ARCHIVE / f"{label}.json").read_text()) for label in LABELS}
    ledger = baseline["melts_present"]["host_ledger"]
    provider_path = tmp_path / "recorded_host.py"
    provider_path.write_text("# Recorded stationary host control; no external runtime.\n")
    reference_path = tmp_path / "recorded_host.json"
    reference_path.write_text(json.dumps({"backend": {"model": "recorded_host_control"}}))
    evaluator = SimpleNamespace(
        __file__=str(provider_path), COMPONENTS=ledger["component_order"][:-1],
        ELEMENTS=ledger["element_order"],
        FORMULA_MATRIX=np.asarray(ledger["formula_matrix_component_rows"][:-1]),
        REFERENCE=json.loads(reference_path.read_text()), REFERENCE_PATH=reference_path,
    )
    factories, calls = [], []

    def host_factory(actual_evaluator, host_names, h2_standard, *, runtime, python_executable):
        assert actual_evaluator is evaluator
        assert host_names == evaluator.COMPONENTS
        assert runtime == tmp_path and python_executable == sys.executable
        label = LABELS[len(factories) % len(LABELS)]
        factories.append(label)
        saved = baseline[label]
        indices = [i for i, name in enumerate(saved["component_order"])
                   if name.endswith("_melts") or name == "H2_dissolved"]
        # Use an immutable baseline; mutated saved residuals cannot change this host.
        mu = (np.asarray(saved["formula_matrix"])[:, indices].T
              @ np.asarray(saved["result"]["elemental_potentials_rt"]))

        def host(temperature, pressure, amounts):
            calls.append((label, temperature, pressure, amounts.copy()))
            assert np.isfinite(h2_standard(temperature, pressure))
            return full.PhaseState(mu, float(amounts @ mu))

        return host

    monkeypatch.setattr(replay, "make_melts_h2_phase", host_factory)

    def run(saved):
        return replay.revalidate_melts_states(
            saved, evaluator=evaluator, runtime=tmp_path, worker_python=sys.executable,
        )

    return SimpleNamespace(saved=copy.deepcopy(baseline), run=run, calls=calls,
                           factories=factories)


def test_fresh_replay_evaluates_each_branch_and_native_alloy(replay_control):
    control = replay_control
    report = control.run(control.saved)
    assert control.factories == list(LABELS)
    assert [call[0] for call in control.calls] == list(LABELS)
    for label, temperature, pressure, host_amounts in control.calls:
        saved = control.saved[label]
        assert (temperature, pressure) == (saved["temperature_k"], saved["pressure_bar"])
        indices = [i for i, name in enumerate(saved["component_order"])
                   if name.endswith("_melts") or name == "H2_dissolved"]
        np.testing.assert_array_equal(host_amounts, np.asarray(saved["result"]["component_amounts_mol"])[indices])
        result = report["cases"][label]
        assert result["accepted"] and result["chemistry"] == "recomputed"
        assert result["max_relative_element_residual"] < 1e-9
        assert result["max_abs_reaction_residual_RT"] < 1e-8
    comparison = report["melts_phase_comparison"]
    assert comparison["absent_alloy_trial"]["disproves_absence"]
    assert comparison["absent_alloy_trial"]["insertion_RT"] == pytest.approx(-2.89230, abs=5e-6)
    assert comparison["same_budget_gibbs_present_minus_absent_RT"] == pytest.approx(-0.225361, abs=5e-7)
    assert report["cases"]["melts_absent"]["phase_interpretation"] == "phase-suppressed diagnostic"


def test_saved_residuals_and_elemental_potentials_do_not_control_fresh_acceptance(replay_control):
    control = replay_control
    expected = control.run(control.saved)
    for saved in control.saved.values():
        result = saved["result"]
        for field in ("reaction_residual", "element_residual", "elemental_potentials_rt"):
            result[field] = [123.0] * len(result[field])
        result["gibbs_rt"] = 123.0
    actual = control.run(control.saved)
    assert actual["cases"] == expected["cases"]
    assert actual["melts_phase_comparison"] == expected["melts_phase_comparison"]
    assert [call[0] for call in control.calls] == list(LABELS) * 2


def test_fresh_gas_chemistry_rejects_element_conserving_displacement(replay_control):
    control = replay_control
    saved = control.saved["melts_present"]
    amounts = np.asarray(saved["result"]["component_amounts_mol"])
    original = amounts.copy()
    names = saved["component_order"]
    amounts[names.index("H2_gas")] -= 1e-6
    amounts[names.index("H_gas")] += 2e-6
    np.testing.assert_allclose(np.asarray(saved["formula_matrix"]) @ amounts,
                               np.asarray(saved["formula_matrix"]) @ original, rtol=1e-14)
    saved["result"]["component_amounts_mol"] = amounts.tolist()
    assert saved["result"]["accepted"]
    with pytest.raises(AssertionError, match="recomputed chemical acceptance"):
        control.run(control.saved)
    assert [call[0] for call in control.calls] == ["melts_present"]


def test_zero_on_declared_support_is_rejected_before_property_callback(replay_control):
    control = replay_control
    saved = control.saved["melts_present"]
    amounts = np.asarray(saved["result"]["component_amounts_mol"])
    original = amounts.copy()
    names = saved["component_order"]
    hydrogen = names.index("H_gas")
    amounts[names.index("H2_gas")] += amounts[hydrogen] / 2
    amounts[hydrogen] = 0
    np.testing.assert_allclose(np.asarray(saved["formula_matrix"]) @ amounts,
                               np.asarray(saved["formula_matrix"]) @ original, rtol=1e-14)
    saved["result"]["component_amounts_mol"] = amounts.tolist()
    with pytest.raises(AssertionError, match="positive on declared support"):
        control.run(control.saved)
    assert not control.factories and not control.calls
