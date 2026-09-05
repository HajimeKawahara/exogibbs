"""Replay historical warm boundaries through the public provider API."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from exogibbs.api.condensate import (
    CondensateEquilibriumOptions,
    CondensateEquilibriumPoint,
    regauge_gas_only_warm_start,
    solve_profile,
)


DATA_DIRECTORY = Path(__file__).with_name("data") / "rocky_raccoon_boundary_corpus"
MANIFEST = json.loads((DATA_DIRECTORY / "manifest.json").read_text())
CASES = MANIFEST["cases"]
EXAMPLE_PATH = (
    Path(__file__).resolve().parents[3]
    / "examples/comparisons/demo_rocky_raccoon_trace_mg.py"
)


def _load_case(case):
    path = DATA_DIRECTORY / case["fixture"]
    assert sha256(path.read_bytes()).hexdigest() == case["sha256"]
    with np.load(path, allow_pickle=False) as stored:
        values = {name: stored[name].copy() for name in stored.files}
    assert values["schema"].item() == "rocky_raccoon.warm_parent_case@1"
    assert values["source_snapshot_sha256"].item() == case["source_snapshot_sha256"]
    assert json.loads(values["source_provenance_json"].item())[
        "schema_version"
    ] == "rocky_raccoon.source_provenance@1"
    return values


@pytest.fixture(scope="module")
def setups():
    example = runpy.run_path(str(EXAMPLE_PATH), run_name="rocky_boundary_corpus")
    canonical = example["CONDENSATE_SPECIES"]
    return {
        False: example["build_reduced_setup"](),
        True: example["build_reduced_setup"](
            condensate_species=canonical[:3] + ("SiO(s)",) + canonical[3:]
        ),
    }


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_boundary_fixture_preserves_named_source_state(case):
    stored = _load_case(case)
    assert int(stored["target_step_index"]) == int(stored["parent_layer_index"]) + 1
    assert stored["target_pressure"] < stored["parent_pressure"]
    assert np.all(np.isfinite(stored["parent_gas_ln_n"]))
    assert np.all(np.isfinite(stored["target_inventory"]))
    assert np.all(stored["target_inventory"] >= 0.0)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_public_rainout_certifies_warm_boundary(case, setups):
    stored = _load_case(case)
    setup = setups[case["include_sio"]]
    assert tuple(stored["element_names"].tolist()) == tuple(setup.elements)
    assert tuple(stored["gas_species"].tolist()) == tuple(setup.gas_species)
    inventory = stored["target_inventory"]
    initial = replace(
        regauge_gas_only_warm_start(setup, stored["parent_gas_ln_n"], inventory),
        inventory_bridge_origin=CondensateEquilibriumPoint(
            temperature=float(stored["parent_temperature"]),
            pressure=float(stored["parent_pressure"]),
            element_inventory=stored["parent_inventory"],
        ),
    )
    profile = solve_profile(
        setup,
        T=np.asarray([stored["target_temperature"].item()]),
        P=np.asarray([stored["target_pressure"].item()]),
        b=inventory,
        init=(initial,),
        options=CondensateEquilibriumOptions(
            rainout=True,
            profile_method="scan_hot_from_bottom",
            return_diagnostics=True,
        ),
        return_diagnostics=True,
    )
    assert profile.rainout
    assert len(profile.layers) == 1
    layer = profile.layers[0]
    assert layer.converged
    assert layer.diagnostics["fixed_support_v2"][
        "caller_gauge_zero_barrier_kkt"
    ]["accepted"]
    gas_amounts = np.asarray(layer.gas_n)
    condensate_amounts = np.asarray(layer.condensate_amounts)
    assert np.all(np.isfinite(gas_amounts)) and np.all(gas_amounts >= 0.0)
    assert np.all(np.isfinite(condensate_amounts)) and np.all(condensate_amounts >= 0.0)
    reconstructed = (
        np.asarray(setup.formula_matrix) @ gas_amounts
        + np.asarray(setup.formula_matrix_cond) @ condensate_amounts
    )
    positive = inventory > 0.0
    np.testing.assert_allclose(
        reconstructed[positive], inventory[positive], rtol=1.0e-3, atol=0.0
    )
    np.testing.assert_allclose(
        reconstructed[~positive], inventory[~positive], rtol=0.0, atol=1.0e-9
    )
    np.testing.assert_allclose(np.sum(layer.gas_x), 1.0, rtol=1.0e-10, atol=0.0)
    assert np.all(np.asarray(profile.rainout_element_inventory_out) >= 0.0)
