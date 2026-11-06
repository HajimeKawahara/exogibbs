"""Test the shared runner's archived-array checks without new equilibrium solves."""

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"mceg/reviews"))
import pending_equilibrium_impact as impact


@pytest.mark.parametrize("network", ["ions", "neutral"])
def test_constraints_accept_real_archive_and_reject_formation_perturbation(network):
    path = ROOT/f"tests/data/validation/c2h2_equilibrium_impact/20261106/{network}.npz"
    with np.load(path) as a:
        species, elements = list(a["species"]), list(a["elements"])
        args = (a["formula_matrix"], elements, species, a["budget"], a["ln_n"])
        result = impact.constraints(*args, a["h"], a["pressure_bar"], ["C", "N"])
        assert result["mass_action_max_abs_ln"].max() < 2e-12
        bad = a["h"].copy()
        bad[1, 5, species.index("C2H2")] += .001
        with pytest.raises(AssertionError):
            impact.constraints(*args, bad, a["pressure_bar"], ["C", "N"])


@pytest.mark.parametrize("targets", [["C2H2"], ["C2H2", "C1H2"]])
def test_case_selection_supports_one_and_multiple_replacement_targets(targets):
    path = ROOT/"tests/data/validation/c2h2_equilibrium_impact/20261106/ions.npz"
    with np.load(path) as a:
        species, elements = list(a["species"]), list(a["elements"])
        carbon = elements.index("C")
        share = np.exp(a["ln_n"])*a["formula_matrix"][carbon]/a["budget"][:, carbon, None]
        logx = impact.log_fractions(a["ln_n"])
        effect, relative, highlight, selected, edges = impact.select_cases(
            logx, {"C": share}, [species.index(s) for s in targets], a["temperature_K"])
        assert effect.shape == (75,)
        assert relative.shape == (2, 75, len(targets))
        assert highlight == 55
        assert len(np.unique(selected)) == 3 and len(np.unique(edges)) == 2
        assert highlight in edges
        assert np.max(a["temperature_K"][selected]) == 4000.
