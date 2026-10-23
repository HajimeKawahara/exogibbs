"""Check the optional pressure-to-fugacity solubility comparison."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks.documented_examples import run_all
from examples.comparisons import compare_solubility_laws as example


def test_exoeos_fugacity_units_and_actual_melt_pressure():
    pytest.importorskip("exoeos")
    pressure_bar = np.array([1.0, 1.0e4])
    fugacity_bar = example._exoeos_fugacity_bar(pressure_bar)
    curves = example._solubility_curves(pressure_bar, fugacity_bar)
    ideal = example._solubility_curves(
        pressure_bar, {name: pressure_bar for name in fugacity_bar},
    )

    # Independent 1700 K, 1 GPa reference values also catch bar/GPa confusion
    # and accidental substitution of fugacity for the melt-pressure term.
    np.testing.assert_allclose(curves[0][2][1], 0.154103, rtol=2.0e-5)
    np.testing.assert_allclose(curves[1][2][1], 0.000577945, rtol=2.0e-5)
    assert curves[4][2][1] > ideal[4][2][1]
    for values in fugacity_bar.values():
        np.testing.assert_allclose(values[0], pressure_bar[0], rtol=3.0e-4)
    for index in (2, 3, 5):
        np.testing.assert_array_equal(curves[index][2], ideal[index][2])


def test_exoeos_job_creates_pressure_figures(tmp_path):
    provider = pytest.importorskip("exoeos")
    source = str(Path(provider.__file__).resolve().parents[1])
    job = next(job for job in run_all.build_jobs() if job.name == "solubility_laws_exoeos")
    environment = dict(os.environ, MPLBACKEND="Agg", EXOGIBBS_EXOEOS_SOURCE=source)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(run_all.ROOT / "src"), source, environment.get("PYTHONPATH", "")),
    )
    output = tmp_path / "comparison"
    assert run_all.run_jobs(
        (job,), output=output, resources={"exoeos": source},
        environment=environment, platform="cpu", full_selection=False,
    ) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["jobs"][0]["status"] == "PASS"
    directory = output / job.name
    assert (directory / "solubility_laws_exoeos.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (directory / "solubility_laws_exoeos.pdf").read_bytes().startswith(b"%PDF-")
