"""Analytic coexistence check for the public multiphase example."""

from pathlib import Path
import runpy

import numpy as np


def test_ideal_liquid_and_pure_solid_recover_analytic_coexistence():
    path = Path(__file__).resolve().parents[3] / "examples/multiphase_equilibrium.py"
    example = runpy.run_path(str(path))
    result = example["run_example"]()
    assert result.stability_status == "unchecked"
    assert np.all(np.isfinite(result.element_potentials_j_mol))

    # Equality of mu_X = RT*log(x_X) and g_solid = -RT*log(2) fixes x_X=1/2.
    # All Y stays in the liquid; the remaining two moles of X form the solid.
    np.testing.assert_allclose(result.amounts_mol, [1.0, 1.0, 2.0], rtol=1.0e-7)
    np.testing.assert_allclose(result.phase_amounts_mol, [2.0, 2.0], rtol=1.0e-7)
    np.testing.assert_allclose(result.conservation_residual_mol, 0.0, atol=1.0e-10)
    np.testing.assert_allclose(
        result.chemical_potentials_j_mol[0], result.chemical_potentials_j_mol[2],
        rtol=1.0e-7,
    )
