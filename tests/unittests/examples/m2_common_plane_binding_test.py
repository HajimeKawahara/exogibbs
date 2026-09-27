"""A saved lower bound cannot silently change its model or source ledger."""
from copy import deepcopy
import importlib
from pathlib import Path
import sys

import pytest

DIRECTORY = Path(__file__).resolve().parents[3]/"examples/metal_silicate"
sys.path.insert(0, str(DIRECTORY))
try:
    M = importlib.import_module("run_m2_common_plane")
finally:
    sys.path.pop(0)


def fixture():
    parameters = {"phase": "olivine", "T_K": 2100., "P_Pa": 2.5e7,
                  "coordinate_bounds": [[0., 1.]], "polynomial_rt": [[1., [2]]],
                  "source_sha256": "exact-provider-expression"}
    standards = {"phase": "olivine", "mu0_J_mol": [-1e6], "T_K": 2100., "P_Pa": 2.5e7}
    row = {"lower_bound_rt_per_formula_unit": .1, "unresolved_box_count": 0,
           "formal_global_insertion_bound_accepted": True}
    proof = {**row, "phase": "olivine", "parameters": deepcopy(parameters), "native_standard_states": deepcopy(standards)}
    return proof, row, parameters, standards


@pytest.mark.parametrize("change", [
    lambda p: p["parameters"].update(T_K=2200.),
    lambda p: p["parameters"].update(coordinate_bounds=[[0., .9]]),
    lambda p: p["parameters"].update(polynomial_rt=[[1.1, [2]]]),
    lambda p: p["parameters"].update(source_sha256="different-model"),
    lambda p: p["native_standard_states"].update(mu0_J_mol=[-1.1e6]),
    lambda p: p.update(lower_bound_rt_per_formula_unit=.2),
    lambda p: p.update(phase="spinel"),
])
def test_changed_phase_domain_or_standards_do_not_inherit_the_saved_bound(change):
    proof, row, parameters, standards = fixture()
    M.require_solution_proof(proof, row, parameters, standards)
    change(proof)
    with pytest.raises(ValueError):
        M.require_solution_proof(proof, row, parameters, standards)
