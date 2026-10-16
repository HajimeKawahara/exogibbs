"""Check ionization conventions and independently tabulated discrepancies."""

import importlib.util
from pathlib import Path

from jax.experimental import enable_x64
import numpy as np
import pytest

from exogibbs.equilibrium.gas.kernel.equations import _compute_gk
from exogibbs.io.load_data import get_data_filepath
from exogibbs.presets import fastchem, fastchem4


@pytest.fixture(scope="module")
def comparison():
    path = (Path(__file__).resolve().parents[3] / "examples/comparisons"
            / "compare_ni_co_ionization.py")
    spec = importlib.util.spec_from_file_location("ni_co_ionization", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rows(comparison):
    return {(row["element"], row["temperature_K"]): row
            for row in comparison.compute_comparison()}


@pytest.mark.parametrize("element", ["Ni", "Co"])
def test_pressure_and_electron_conventions_satisfy_reaction_equilibrium(comparison, element):
    temperature = 3000.0
    with enable_x64():
        setup = fastchem4.chemsetup(path=comparison.FASTCHEM_PATH, silent=True)
        indices = [setup.species.index(name) for name in (element + "1", element + "1+", "e1-")]
        hvector = np.asarray(setup.hvector_func(temperature))[indices]
        log_kp = hvector[0] - hvector[1] - hvector[2]
        log_kn = comparison.log_kp_to_log_kn_cm3(log_kp, temperature)
        # A neutral/ion/electron mixture satisfying Kn and charge neutrality.
        neutral_density = 1.0e12  # cm^-3
        ion_density = np.sqrt(np.exp(log_kn) * neutral_density)
        densities = np.array([neutral_density, ion_density, ion_density])
        pressure_bar = densities.sum() * 1.380649e-16 * temperature / 1.0e6
        chemical_potential = np.asarray(_compute_gk(
            temperature, np.log(densities), np.log(densities.sum()),
            hvector, np.log(pressure_bar),
        ))
        assert chemical_potential[1] + chemical_potential[2] - chemical_potential[0] == pytest.approx(0.0, abs=1e-12)


def test_saha_includes_electron_spin_and_uses_one_bar(comparison):
    # Unit internal partition ratio, chi = 7.639878 eV; independent SI evaluation.
    actual = comparison.saha_log_kp(np.array(3000.0), np.array(1.0), 7.639878)
    assert actual / np.log(10.0) == pytest.approx(-10.317653613287533, abs=1e-12)
    # Converting the density unit from m^-3 to cm^-3 removes six decades.
    expected_log_kn_cm3 = actual + np.log(1.0e5 / (1.380649e-23 * 3000.0)) - np.log(1.0e6)
    assert comparison.log_kp_to_log_kn_cm3(actual, 3000.0) == pytest.approx(expected_log_kn_cm3)


@pytest.mark.parametrize(
    "element, temperature, janaf_log_kp, janaf_over_saha",
    [("Ni", 3000.0, -8.36820898986201, 282.6507679510569),
     ("Co", 5000.0, -4.055303352196375, 7.373282466452634)],
)
def test_janaf_discrepancy_survives_correct_reference_conventions(
    rows, element, temperature, janaf_log_kp, janaf_over_saha
):
    row = rows[element, temperature]
    assert row["log10_Kp_janaf"] == pytest.approx(janaf_log_kp, abs=1e-10)
    assert row["janaf_over_saha"] == pytest.approx(janaf_over_saha, rel=1e-9)
    assert row["fastchem_over_janaf"] == pytest.approx(1.0, abs=0.1)
    assert row["log10_Kp_fastchem"] == pytest.approx(row["log10_Kp_exogibbs"], abs=1e-12)


def test_independent_nasa_values_support_saha_scale(rows):
    assert rows["Ni", 3000.0]["log10_Kp_nasa9"] == pytest.approx(-10.81500378586127, abs=1e-10)
    assert rows["Co", 5000.0]["log10_Kp_nasa9"] == pytest.approx(-4.892827788980867, abs=1e-10)
    for (element, _), row in rows.items():
        tolerance = 0.02 if element == "Ni" else 0.12
        assert abs(row["log10_Kp_nasa9"] - row["log10_Kp_saha"]) < tolerance


def test_janaf_partition_anomaly_is_in_ions_while_neutral_atoms_agree(rows):
    for row in rows.values():
        assert row["Q_janaf_neutral"] == pytest.approx(row["Q_reference_neutral"], rel=4e-4)
    nickel = rows["Ni", 3000.0]
    assert nickel["Q_janaf_ion"] == pytest.approx(2323.13, abs=0.01)
    assert nickel["Q_reference_ion"] == pytest.approx(8.29948)
    cobalt = rows["Co", 3000.0]
    assert cobalt["Q_janaf_ion"] == pytest.approx(34.3743, abs=0.001)
    assert cobalt["Q_reference_ion"] == pytest.approx(20.9531)


@pytest.mark.parametrize("element", ["Ni", "Co"])
def test_packaged_fastchem_versions_have_identical_ionization_fits(element):
    fits = []
    for path in ("fastchem/logK/logK.dat", "fastchem/logK/logK_extended.dat",
                 "FastChem4/logK/logK.dat", "FastChem4/logK/logK_extended.dat",
                 "FastChem4/logK/logK_hoeijmakers2019.dat"):
        coefficients, _ = fastchem._parse_fastchem_coeffs(get_data_filepath(path).read_text())
        fits.append(coefficients[element + "1+"])
    assert all(fit == fits[0] for fit in fits[1:])
