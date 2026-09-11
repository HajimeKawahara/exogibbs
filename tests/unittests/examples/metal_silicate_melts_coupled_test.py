"""Offline full-potential MELTS boundary, basis, and finite-host regressions."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


DIRECTORY = Path(__file__).resolve().parents[3] / "examples" / "metal_silicate"


def _load_examples():
    names = ("local", "reference", "source", "hydrogen", "full_potential", "melts_coupled", "run_melts_reference")
    previous = {name: sys.modules.get(name) for name in names}
    previous_path = sys.path[:]
    try:
        sys.path.insert(0, str(DIRECTORY))
        for name in names:
            sys.modules.pop(name, None)
        return [importlib.import_module(name) for name in names]
    finally:
        sys.path[:] = previous_path
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


LOCAL, _, _, _, FULL, MELTS, RUNNER = _load_examples()
FORMULAS = {
    "sio2": {"Si": 1, "O": 2}, "mg2sio4": {"Mg": 2, "Si": 1, "O": 4},
    "al2o3": {"Al": 2, "O": 3}, "casio3": {"Ca": 1, "Si": 1, "O": 3},
    "h2o": {"H": 2, "O": 1}, "fe2sio4": {"Fe": 2, "Si": 1, "O": 4},
    "co2": {"C": 1, "O": 2},
}
HOST_NAMES = ("casio3", "mg2sio4", "sio2", "al2o3", "h2o")


def fake_evaluator(tmp_path, mutate=None):
    """Use a scalar regular host with a deliberately different backend R."""
    components = list(FORMULAS)
    elements = sorted(set().union(*FORMULAS.values()))
    provider_path = tmp_path / "mock_melts.py"
    provider_path.write_text("# Offline synthetic property evaluator.\n")
    reference_path = tmp_path / "mock_melts_reference.json"
    reference = {"backend": {"version": "offline_mock", "runtime_sha256": {"mock_library": "synthetic"}}}
    reference_path.write_text(json.dumps(reference))
    calls = []

    def properties(temperature, pressure, n, *, runtime, common_R, python_executable, calculation_mode=1):
        n = np.asarray(n, dtype=float)
        calls.append((temperature, pressure, n.copy(), runtime, common_R, python_executable, calculation_mode))
        x = n / n.sum()
        active = n > 0
        standards = np.linspace(-9., -3., len(components))
        ratio, interaction = 8.3143 / common_R, 0.7
        excess = -interaction * x[0] * x[1] * np.ones(len(n))
        excess[0] += interaction * x[1]
        excess[1] += interaction * x[0]
        mu = np.full(len(n), np.nan)
        mu[active] = standards[active] + ratio * np.log(x[active]) + excess[active]
        result = {
            "status": "ok_supplied_liquid_properties", "model_id": MELTS.PROVIDER_MODEL_IDS[calculation_mode],
            "component_order": components.copy(), "component_moles": n.tolist(),
            "returned_component_moles": n.tolist(), "T_K": temperature, "P_Pa": pressure,
            "phase_policy": {"oxygen_buffer": "None"},
            "basis": {"common_R_J_mol_K": common_R, "backend_R_J_mol_K": 8.3143},
            "mu_RT": [float(value) if exists else None for value, exists in zip(mu, active)],
            "gibbs_J": float(n[active] @ mu[active] * common_R * temperature),
            # Full-potential consumers must ignore these separate mixing fields.
            "ln_gamma": [37.] * len(n), "ln_activity": [-53.] * len(n),
        }
        if calculation_mode == 4:
            species_x = np.append(x, 0.25 * x[components.index("co2")])
            for name, sign in (("sio2", 1), ("casio3", -1), ("co2", -1)):
                species_x[components.index(name)] += sign * species_x[-1]
            result["backend_species"] = {"order": components + ["caco3"], "x": species_x.tolist()}
            result["provenance"] = {"backend": {"model": "rhyolite-MELTS 1.2.0", "calculation_mode": 4}}
        if mutate is not None:
            mutate(result)
        return result

    evaluator = SimpleNamespace(
        COMPONENTS=components, ELEMENTS=elements, __file__=str(provider_path),
        REFERENCE=reference, REFERENCE_PATH=reference_path,
        MODEL_ID=MELTS.PROVIDER_MODEL_ID, MODEL_IDS=MELTS.PROVIDER_MODEL_IDS.copy(),
        FORMULA_MATRIX=np.asarray([[FORMULAS[name].get(e, 0) for e in elements] for name in components]),
        evaluate_liquid=properties,
    )
    return evaluator, calls


def test_full_host_mixing_is_added_once_with_explicit_units_and_permutation(tmp_path):
    evaluator, calls = fake_evaluator(tmp_path)
    received_standard = []
    def h2_standard(t, p):
        received_standard.append((t, p))
        return 2.3
    phase = MELTS.make_melts_h2_phase(evaluator, HOST_NAMES, h2_standard,
                                     runtime=tmp_path, python_executable="/chosen/python", common_r=8.3)
    n = np.array([.15, .4, .2, .1, .05, .03])
    state = phase(1873., 12.5, n)
    temperature, pressure, host, runtime, gas_constant, python, mode = calls[0]
    assert (temperature, pressure, runtime, gas_constant, python) == (1873., 1.25e6, tmp_path, 8.3, "/chosen/python")
    assert mode == 1
    assert received_standard == [(1873., 12.5)]
    indices = [evaluator.COMPONENTS.index(name) for name in HOST_NAMES]
    np.testing.assert_array_equal(host[indices], n[:-1])
    np.testing.assert_array_equal(host[[evaluator.COMPONENTS.index(name) for name in ("fe2sio4", "co2")]], 0)
    bare = evaluator.evaluate_liquid(1873., 1.25e6, host, runtime=tmp_path, common_R=8.3, python_executable="/chosen/python")
    host_mu = np.asarray(bare["mu_RT"], dtype=float)[indices]
    expected_mu = np.append(host_mu + np.log(n[:-1].sum() / n.sum()), 2.3 + np.log(n[-1] / n.sum()))
    expected_g = (bare["gibbs_J"] / (8.3 * 1873.) + n[-1] * 2.3
                  + n[:-1].sum() * np.log(n[:-1].sum() / n.sum()) + n[-1] * np.log(n[-1] / n.sum()))
    np.testing.assert_allclose(state.mu_rt, expected_mu, atol=2e-15)
    np.testing.assert_allclose(state.gibbs_rt, expected_g, atol=2e-15)
    np.testing.assert_allclose(n @ state.mu_rt, state.gibbs_rt, atol=2e-15)
    for index in range(len(n)):
        step = np.eye(len(n))[index] * 1e-5
        finite = (phase(1873., 12.5, n + step).gibbs_rt - phase(1873., 12.5, n - step).gibbs_rt) / 2e-5
        np.testing.assert_allclose(finite, state.mu_rt[index], rtol=3e-8, atol=3e-8)


@pytest.mark.parametrize("mutate", [
    lambda result: result.update(status="failed"),
    lambda result: result.update(model_id="different_backend"),
    lambda result: result["component_order"].reverse(),
    lambda result: result["phase_policy"].update(oxygen_buffer="IW"),
    lambda result: result.update(T_K=1874.),
    lambda result: result.update(P_Pa=1.1e5),
    lambda result: result["basis"].update(common_R_J_mol_K=8.3),
    lambda result: result["returned_component_moles"].__setitem__(0, result["returned_component_moles"][0] * 1.001),
    lambda result: result["returned_component_moles"].__setitem__(-1, 1e-20),
    lambda result: result["mu_RT"].__setitem__(0, None),
])
def test_provider_drift_or_missing_present_potentials_are_rejected(tmp_path, mutate):
    evaluator, _ = fake_evaluator(tmp_path, mutate)
    phase = MELTS.make_melts_h2_phase(evaluator, HOST_NAMES, lambda t, p: 0., runtime=tmp_path, python_executable="python")
    with pytest.raises(ValueError):
        phase(1873., 1., np.array([.15, .4, .2, .1, .05, .03]))


@pytest.mark.parametrize("invalid", [np.zeros(6), np.ones(5), np.array([1., 1., 1., 1., 1., -1.]), np.full(6, np.nan)])
def test_invalid_host_inputs_fail_before_provider(tmp_path, invalid):
    evaluator, calls = fake_evaluator(tmp_path)
    phase = MELTS.make_melts_h2_phase(evaluator, HOST_NAMES, lambda t, p: 0., runtime=tmp_path, python_executable="python")
    with pytest.raises(ValueError, match="positive host"):
        phase(1873., 1., invalid)
    assert not calls


def test_finite_background_elements_survive_ledger_and_coupled_h2_exchange(tmp_path):
    evaluator, calls = fake_evaluator(tmp_path)
    ledger = MELTS.provider_ledger(evaluator, HOST_NAMES)
    assert ledger["evaluator_sha256"] == hashlib.sha256(Path(evaluator.__file__).read_bytes()).hexdigest()
    assert ledger["reference_sha256"] == hashlib.sha256(evaluator.REFERENCE_PATH.read_bytes()).hexdigest()
    assert ledger["backend"] == evaluator.REFERENCE["backend"]
    assert ledger["component_order"] == list(HOST_NAMES) + ["H2_dissolved"]
    names = ledger["component_order"] + ["H2_gas", "He_gas"]
    host = np.array([.1, .2, .3, .15, .05])
    expected = np.append(host, [.02, .04, .01])
    elements = list(evaluator.ELEMENTS) + ["He"]
    formulas = {name: FORMULAS[name] for name in HOST_NAMES}
    formulas.update({"H2_dissolved": {"H": 2}, "H2_gas": {"H": 2}, "He_gas": {"He": 1}})
    formula = np.asarray([[formulas[name].get(element, 0) for name in names] for element in elements])
    np.testing.assert_array_equal(np.asarray(ledger["formula_matrix_component_rows"]).T, formula[:-1, :6])
    budgets = formula @ expected
    record = {"elements": elements, "phases": {"silicate": names[:6], "gas": names[6:]},
              "component_formulas": formulas, "reactions": []}
    problem = LOCAL.build_problem(record, budgets, lambda t, p: np.zeros(8), phases=("silicate", "gas"))
    standard = np.log(.04 / .05) - np.log(.02 / (host.sum() + .02))
    melt = MELTS.make_melts_h2_phase(evaluator, HOST_NAMES, lambda t, p: standard,
                                    runtime=tmp_path, python_executable="python")
    result = FULL.solve_full_potentials(problem, 1873., 1., budgets,
                                        {"silicate": melt, "gas": FULL.ideal_phase(lambda t, p: np.zeros(2), gas=True)},
                                        initial_component_amounts_mol=expected * np.linspace(.8, 1.2, 8))
    assert result.accepted
    np.testing.assert_allclose(result.component_amounts_mol, expected, atol=1e-11, rtol=1e-9)
    np.testing.assert_allclose(formula @ result.component_amounts_mol, budgets, atol=1e-12)
    for element in ("Mg", "Al", "Ca"):
        index = elements.index(element)
        assert budgets[index] > 0
        np.testing.assert_allclose(result.phase_element_amounts_mol[0, index], budgets[index], atol=1e-12)
        assert result.phase_element_amounts_mol[1, index] == 0
    assert np.all(result.element_residual[[elements.index("Fe"), elements.index("C")]] == 0)
    indices = [evaluator.COMPONENTS.index(name) for name in HOST_NAMES]
    np.testing.assert_array_equal(calls[-1][2][indices], result.component_amounts_mol[:5])


def test_reduced_gas_common_r_standards_match_independent_pinned_energies():
    fixture = json.loads((DIRECTORY / "reduced_gas_reference.json").read_text())
    actual = RUNNER.reduced_gas_standards_rt(2350.)
    assert set(actual) == {"H", "He", "OH", "SiH4"}
    for name, item in fixture["species"].items():
        np.testing.assert_allclose(actual[name] * MELTS.COMMON_R * 2350., item["mu0_J_mol_2350K"], atol=2e-9, rtol=0)
    assert fixture["species"]["SiH4"]["formula"] == {"Si": 1, "H": 4}
    for temperature in (200., 6000., np.nan):
        with pytest.raises(ValueError, match="200 < T < 6000"):
            RUNNER.reduced_gas_standards_rt(temperature)


@pytest.mark.parametrize("carbon", [0.0, 0.02])
@pytest.mark.parametrize("hydrogen", [0.0, 0.03])
def test_carbon_mode_preserves_independent_potentials_and_exact_zero_limits(tmp_path, carbon, hydrogen):
    evaluator, calls = fake_evaluator(tmp_path)
    names = ("co2",) + HOST_NAMES[::-1]
    amounts = np.array([carbon, .05, .1, .2, .4, .15, hydrogen])
    phase = MELTS.make_melts_h2_phase(
        evaluator, names, lambda t, p: 2.3, runtime=tmp_path,
        python_executable="python", calculation_mode=4,
    )
    state = phase(1473.15, 500., amounts)
    assert calls[-1][-1] == 4
    indices = [evaluator.COMPONENTS.index(name) for name in names]
    host = calls[-1][2]
    np.testing.assert_array_equal(host[indices], amounts[:-1])
    bare = evaluator.evaluate_liquid(
        1473.15, 5e7, host, runtime=tmp_path, common_R=MELTS.COMMON_R,
        python_executable="python", calculation_mode=4,
    )
    active = amounts > 0
    expected_mu = np.asarray(bare["mu_RT"], float)[indices] + np.log(amounts[:-1].sum() / amounts.sum())
    np.testing.assert_allclose(state.mu_rt[:-1][active[:-1]], expected_mu[active[:-1]], atol=1e-14)
    np.testing.assert_allclose(amounts[active] @ state.mu_rt[active], state.gibbs_rt, atol=1e-14)
    if hydrogen == 0:
        assert np.isneginf(state.mu_rt[-1])
        np.testing.assert_allclose(state.gibbs_rt, bare["gibbs_J"] / (MELTS.COMMON_R * 1473.15))
    else:
        np.testing.assert_allclose(state.mu_rt[-1], 2.3 + np.log(hydrogen / amounts.sum()))
    scaled = phase(1473.15, 500., 7. * amounts)
    np.testing.assert_allclose(scaled.mu_rt[active], state.mu_rt[active], atol=1e-14)
    np.testing.assert_allclose(scaled.gibbs_rt, 7. * state.gibbs_rt, atol=1e-13)

    ledger = MELTS.provider_ledger(evaluator, names, calculation_mode=4)
    assert ledger["model_id"] == MELTS.PROVIDER_MODEL_IDS[4]
    assert ledger["backend"]["model"] == "rhyolite-MELTS 1.2.0"
    assert ledger["backend"]["calculation_mode"] == ledger["calculation_mode"] == 4
    assert "caco3" not in ledger["component_order"]
    carbon_formula = dict(zip(ledger["element_order"], ledger["formula_matrix_component_rows"][0]))
    assert {name: value for name, value in carbon_formula.items() if value} == {"C": 1, "O": 2}
    if carbon == 0:
        reduced = MELTS.make_melts_h2_phase(
            evaluator, names[1:], lambda t, p: 2.3, runtime=tmp_path,
            python_executable="python", calculation_mode=4,
        )(1473.15, 500., amounts[1:])
        assert calls[-1][2][evaluator.COMPONENTS.index("co2")] == 0
        assert np.isnan(state.mu_rt[0])
        np.testing.assert_allclose(state.mu_rt[1:], reduced.mu_rt, atol=1e-14)
        np.testing.assert_allclose(state.gibbs_rt, reduced.gibbs_rt, atol=1e-14)
    else:
        assert bare["backend_species"]["x"][-1] > 0
        for index in np.flatnonzero(active):
            step = np.eye(len(amounts))[index] * 1e-5
            finite = (phase(1473.15, 500., amounts + step).gibbs_rt
                      - phase(1473.15, 500., amounts - step).gibbs_rt) / 2e-5
            np.testing.assert_allclose(finite, state.mu_rt[index], rtol=3e-8, atol=3e-8)


@pytest.mark.parametrize("mode", [0, 2, 3, 5, True, 4.0, "4"])
def test_unsupported_modes_fail_before_provider(tmp_path, mode):
    evaluator, calls = fake_evaluator(tmp_path)
    with pytest.raises(ValueError, match="calculation_mode must be"):
        MELTS.make_melts_h2_phase(evaluator, HOST_NAMES, lambda t, p: 0.,
                                  runtime=tmp_path, python_executable="python", calculation_mode=mode)
    with pytest.raises(ValueError, match="calculation_mode must be"):
        MELTS.provider_ledger(evaluator, HOST_NAMES, calculation_mode=mode)
    assert not calls


@pytest.mark.parametrize("mutate", [
    lambda result: result.update(model_id=MELTS.PROVIDER_MODEL_ID),
    lambda result: result["provenance"]["backend"].update(calculation_mode=1),
    lambda result: result["provenance"]["backend"].update(model="rhyolite-MELTS 1.0.2"),
])
def test_carbon_provider_model_drift_is_rejected(tmp_path, mutate):
    evaluator, _ = fake_evaluator(tmp_path, mutate)
    phase = MELTS.make_melts_h2_phase(evaluator, HOST_NAMES + ("co2",), lambda t, p: 0.,
                                     runtime=tmp_path, python_executable="python", calculation_mode=4)
    with pytest.raises(ValueError, match="provider"):
        phase(1473.15, 500., np.array([.15, .4, .2, .1, .05, .02, .03]))


def test_legacy_provider_retains_default_call_and_cannot_claim_carbon(tmp_path):
    evaluator, calls = fake_evaluator(tmp_path)
    del evaluator.MODEL_IDS
    properties = evaluator.evaluate_liquid
    def legacy(temperature, pressure, n, *, runtime, common_R, python_executable):
        return properties(temperature, pressure, n, runtime=runtime, common_R=common_R,
                          python_executable=python_executable)
    evaluator.evaluate_liquid = legacy
    phase = MELTS.make_melts_h2_phase(evaluator, HOST_NAMES, lambda t, p: 0.,
                                     runtime=tmp_path, python_executable="python")
    assert np.isfinite(phase(1473.15, 500., np.ones(6)).gibbs_rt)
    assert len(calls) == 1
    for declared in (None, {4: "unrecognized_carbon_model"}):
        if declared is not None:
            evaluator.MODEL_IDS = declared
        with pytest.raises(ValueError, match="does not declare"):
            MELTS.make_melts_h2_phase(evaluator, HOST_NAMES, lambda t, p: 0.,
                                      runtime=tmp_path, python_executable="python", calculation_mode=4)


def test_installed_exoeos_declares_the_carbon_component_contract():
    exoeos = pytest.importorskip("exoeos")
    checkout = Path(exoeos.__file__).resolve().parents[2]
    evaluator = MELTS.load_melts_evaluator(checkout)
    ledger = MELTS.provider_ledger(evaluator, ("co2", "h2o", "casio3"), calculation_mode=4)
    assert evaluator.MODEL_IDS[4] == MELTS.PROVIDER_MODEL_IDS[4]
    assert len(evaluator.COMPONENTS) == 19
    assert "caco3" not in evaluator.COMPONENTS
    formula = dict(zip(ledger["element_order"], ledger["formula_matrix_component_rows"][0]))
    assert {name: value for name, value in formula.items() if value} == {"C": 1, "O": 2}
    reference = json.loads((checkout / "tests/reference/melts_carbon_v1.json").read_text())
    assert reference["model_id"] == ledger["model_id"]
    assert reference["status"] == "passed_numerical_property_checks"
    assert reference["records"][0]["result"]["component_moles"][evaluator.COMPONENTS.index("co2")] == 0
