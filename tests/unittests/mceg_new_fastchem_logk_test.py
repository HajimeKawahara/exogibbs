"""Replay the scoped molecular attribution without importing MCEG."""

import hashlib
import json
from pathlib import Path
import runpy

import jax
import pytest


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "tests/data/validation/mceg/new_fastchem_logk_20261024/attribution.json"


def test_attribution_selected_inputs_remain_pinned():
    saved = json.loads(REPORT.read_text())
    for name, expected in saved["selected_file_sha256"].items():
        if name.startswith("results/"):
            path = ROOT / "tests/data/validation" / Path(name).relative_to("results")
        elif name in {"examples/comparisons/compare_new_fastchem_logk.py",
                       "mceg/reviews/new_fastchem_logk_20261024.py"}:
            path = ROOT / "tests/data/validation/new_fastchem_logk/source" / Path(name).name
            # Numerical code stays byte-identical; only archived input paths move.
            relocated = path.read_bytes().replace(
                b'"results/new_fastchem_logk/logK.zip"',
                b'"tests/data/validation/new_fastchem_logk/logK.zip"').replace(
                b'RESULTS = ROOT / "results/new_fastchem_logk"',
                b'RESULTS = ROOT / "tests/data/validation/new_fastchem_logk"').replace(
                b'"results/major_molecule_adoption/decisions.csv"',
                b'"tests/data/validation/major_molecule_adoption/decisions.csv"')
            assert (ROOT / name).read_bytes() == relocated, name
        else:
            path = ROOT / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name


def test_five_sample_attribution_replays_with_mass_action():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        review = runpy.run_path(str(ROOT / "mceg/reviews/new_fastchem_logk_20261024.py"))
        actual = review["attribution"]()
    finally:
        jax.config.update("jax_enable_x64", previous)
    saved = json.loads(REPORT.read_text())
    assert len(actual["samples"]) == len(saved["samples"]) == 5
    for row, recorded in zip(actual["samples"], saved["samples"]):
        assert (row["profile"], row["species"]) == (recorded["profile"], recorded["species"])
        for key in ("temperature_K", "pressure_bar", "K_new_over_curated", "atom_product_ratio",
                    "abundance_new_over_curated", "new_x", "curated_x"):
            assert row[key] == pytest.approx(recorded[key], rel=1e-11, abs=0)
        assert abs(row["ln_factorization_residual"]) < 1e-8
    hcn = actual["samples"][1]
    assert hcn["abundance_new_over_curated"] == pytest.approx(.7445271990936414, rel=1e-10)
    assert hcn["K_new_over_curated"] == pytest.approx(.753005681372609, rel=1e-10)
    sis = actual["samples"][4]
    assert sis["K_new_over_curated"] == 1.
    assert sis["atom_product_ratio"] == pytest.approx(sis["abundance_new_over_curated"], rel=1e-11)
    energy = actual["HCN_energy_at_2500_K"]
    assert energy["delta_D0_eV"] == pytest.approx(.06019567816357, abs=1e-12)
    assert energy["energy_only_K_new_over_curated"] == pytest.approx(.7562246920648531, rel=1e-11)
