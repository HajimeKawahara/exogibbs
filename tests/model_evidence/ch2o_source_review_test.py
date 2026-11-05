"""Check formaldehyde energy balance, conventions and archived-grid scope offline."""

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mceg/reviews"))
import ch2o_source_review as study


@pytest.fixture(scope="module")
def report():
    return study.review()


def test_balanced_energies_use_gas_atoms_and_original_nasa_headers(report):
    inputs = json.loads((study.DATA / "inputs.json").read_text())
    assert "C1(g)" in (ROOT / inputs["atomic_janaf_tables"]["C"]).read_text().splitlines()[0]
    assert report["balanced_D0_J_mol"] == pytest.approx(
        {"janaf": 1502104, "nasa": 1494796.222, "atct_1.228": 1495667}, abs=1e-6)
    assert report["ATcT_minus_JANAF_D0_J_mol"] == pytest.approx(-6437)
    tables = {n: study.read_table(ROOT/p) for n, p in inputs["atomic_janaf_tables"].items()}
    direct_d0 = 1000*(sum(c*tables[n][0.][4] for n, c in study.COMPOSITION.items())
                     - study.read_table(study.DATA / "C-061.txt")[0.][4])
    assert direct_d0 == pytest.approx(1502102)
    assert report["balanced_D0_J_mol"]["janaf"]-direct_d0 == pytest.approx(2)
    raw = (study.DATA / "nasa_selected_records.txt").read_text().split("\n\n")
    for record in inputs["nasa_records"].values():
        lines = next(b.splitlines() for b in raw if b.startswith(record["source_name"]+" "))
        assert float(lines[1].split()[-1]) == record["hf298_J_mol"]
        assert float(lines[2].split()[-1]) == record["h298_minus_h0_J_mol"]
        for i, interval in enumerate(record["intervals"]):
            header = lines[2+3*i]
            # The coefficient count immediately follows the second F11.3 field.
            assert [float(header[:11]), float(header[11:22])] == interval["bounds_K"]
    assert inputs["nasa_records"]["CH2O"]["intervals"][0]["bounds_K"] == [300., 1000.]
    with pytest.raises(ValueError, match="outside NASA"):
        study.nasa_properties(inputs["nasa_records"]["CH2O"], 298.15, study.NASA_R)
    for row in report["source_samples"]:
        assert abs(row["decomposition_closure_ln_K"]) < 1e-12
        assert row["ATcT_energy_only_K_over_current"] == pytest.approx(
            math.exp(-6437/(study.R*row["temperature_K"])))


@pytest.mark.parametrize("temperature", [1500., 2500., 4000.])
def test_analytic_fit_calorics_match_independent_runtime_derivatives(temperature):
    import jax
    model = study.setup()
    col = model.species.index(study.SPECIES)
    h = lambda t: model.hvector_func(t)[col]
    dh, d2h = jax.grad(h)(temperature), jax.grad(jax.grad(h))(temperature)
    actual = study.curated_properties(temperature)
    enthalpy = -study.R*temperature**2*float(dh)
    entropy = (enthalpy-study.R*temperature*float(h(temperature)))/temperature
    cp = -study.R*(2*temperature*float(dh)+temperature**2*float(d2h))
    assert actual["ln_K"] == pytest.approx(-float(h(temperature)), abs=2e-12)
    assert actual["H_J_mol"] == pytest.approx(enthalpy, abs=2e-8)
    assert actual["S_J_mol_K"] == pytest.approx(entropy, abs=2e-10)
    assert actual["Cp_J_mol_K"] == pytest.approx(cp, abs=2e-10)


def test_spin_mass_domain_and_partition_derivatives(report):
    rows = {r["temperature_K"]: r for r in report["source_samples"]}
    for t in (300, 1500, 3000):
        p = rows[t]["ayty"]
        assert p["Q_spin_removed"] == p["Q_printed"]/4
        assert p["isotope_translation_K_proxy"] == pytest.approx(
            p["Q_over_nasa"]*(30.01056468/30.02598)**1.5)
    assert rows[3000]["ayty"]["calorics"] is None
    assert rows[3500]["ayty"] is None
    assert rows[4000]["ayty"] is None
    with pytest.raises(ValueError, match="outside reviewed"):
        study.curated_properties(1000)
    t = np.arange(1000., 2001., 10.)
    synthetic = np.column_stack((t, t**3))
    assert study.partition_calorics(synthetic, 1500, 10)["Cp_J_mol_K"] == pytest.approx(5.5*study.R, abs=2e-6)
    with pytest.raises(ValueError, match="outside table"):
        study.partition_calorics(synthetic, 2000, 10)
    for row in rows.values():
        p = row["ayty"]
        if p is not None and p["calorics"] is not None:
            c = p["calorics"]
            # Finite differences have truncation as well as tabulation rounding.
            assert abs(c["Cp_J_mol_K"]-c["Cp_step20_J_mol_K"]) < .001


def test_latest_archived_grid_is_not_a_replacement_solve(report):
    assert report["production_change"] == "none"
    assert report["equilibrium_substitution"] == "not_run"
    for footprint, count in zip(report["footprints"], [537, 421]):
        assert footprint["model"] == "curated-cos-v1"
        assert footprint["all_converged"]
        assert (footprint["cases"], footprint["species_count"]) == (75, count)
        assert footprint["maximum_abs_h_error"] < 2e-12
        assert footprint["maximum_relative_carbon_residual"] < 3e-12
        assert footprint["maximum_carbon_share"] == pytest.approx(4.6427e-5, rel=2e-6)
        assert footprint["case_index"] == 14


def test_replay_and_actual_input_receipt(report):
    expected = json.loads((study.DATA / "comparison.json").read_text())
    def equal(a, b, path=()):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                equal(a[k], b[k], path+(k,))
        elif isinstance(a, list):
            assert len(a) == len(b)
            for i, (x, y) in enumerate(zip(a, b)):
                equal(x, y, path+(i,))
        elif isinstance(a, float):
            # Rounded PF second differences amplify cancellation; source energies
            # and ln K retain tighter checks and do not inherit this tolerance.
            tolerance = 1e-7 if "calorics" in path else 2e-10
            assert a == pytest.approx(b, abs=tolerance, rel=2e-11), path
        else:
            assert a == b
    equal(report, expected)
    receipt = json.loads((study.DATA / "receipt.json").read_text())
    assert "src/exogibbs/data/fastchem/logK/logK.dat" in receipt["selected_input_sha256"]
    for group in ("selected_input_sha256", "output_sha256"):
        for name, digest in receipt[group].items():
            assert sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
