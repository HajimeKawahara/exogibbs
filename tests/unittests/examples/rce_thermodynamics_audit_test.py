"""Execute the thermodynamic comparison with real equilibrium calculations."""

import json
from pathlib import Path
import runpy

from exogibbs.api.equilibrium import EquilibriumOptions


SCRIPT = "examples/audit_rce_thermodynamics.py"
RUN_AUDIT = runpy.run_path(str(Path(__file__).resolve().parents[3] / SCRIPT))["run_audit"]


def test_small_audit_executes_all_models_and_checks_conservation():
    report = RUN_AUDIT(
        temperatures_k=(2200.0, 4500.0),
        pressures_bar=(0.01,),
        abundance_points=((0.0, 0.6),),
    )
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report["x64"]
    assert report["excluded_species"] == "C4H6O4"
    assert len(report["fastchem_sha256"]) == len(report["nasa_excerpt_sha256"]) == 64
    for name, count in (("original", 539), ("omission_only", 538), ("hybrid", 538)):
        model = report["models"][name]
        assert model["species_count"] == count
        assert model["converged_points"] == model["valid_points"] == 2
        assert [row["temperature_K"] for row in model["points"]] == [2200.0, 4500.0]
        for row in model["points"]:
            assert row["pressure_bar"] == 0.01
            assert row["relative_element_error"] <= report["comparison_element_rtol"]
            assert row["relative_charge_error"] <= report["comparison_charge_rtol"]
            assert 1.0 < row["mmw_u"] < 3.0
        if name != "original":
            summary = model["comparison_summary"]
            assert summary["common_valid_points"] == 2
            assert summary["excluded_fraction"] < 1e-30
            assert all(row["comparison_to_original"]["accepted"] for row in model["points"])
    omission = report["models"]["omission_only"]["comparison_summary"]
    hybrid = report["models"]["hybrid"]["comparison_summary"]
    assert omission["max_absolute_fraction_change"] < 1e-10
    assert omission["relative_mmw_change"] < 1e-10
    assert 1e-10 < hybrid["max_absolute_fraction_change"] < 1e-3


def test_failed_equilibria_remain_visible_without_comparison(monkeypatch):
    monkeypatch.setitem(
        RUN_AUDIT.__globals__,
        "EquilibriumOptions",
        lambda **kwargs: EquilibriumOptions(**{**kwargs, "max_iter": 1}),
    )
    report = RUN_AUDIT(
        temperatures_k=(2200.0,),
        pressures_bar=(0.01,),
        abundance_points=((0.0, 0.6),),
    )
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report["max_iter"] == 1
    for name, model in report["models"].items():
        assert len(model["points"]) == 1
        assert model["converged_points"] == model["valid_points"] == 0
        assert not model["points"][0]["valid_for_comparison"]
        if name != "original":
            assert model["points"][0]["comparison_to_original"] == {"accepted": False}
            summary = model["comparison_summary"]
            assert summary["common_valid_points"] == 0
            assert all(value is None for key, value in summary.items() if key != "common_valid_points")
