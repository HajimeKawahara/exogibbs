import pickle
import runpy
from pathlib import Path

import numpy as np
import pytest

from exogibbs.condensates.curated_profiles import CuratedProfileDefinition


_sweep_module = runpy.run_path(
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "fixed_support_v2"
    / "support_atlas_sweep.py"
)
_subset_profile_definition = _sweep_module["_subset_profile_definition"]
_slice_batch_scalar = _sweep_module["_slice_batch_scalar"]
_write_solver_checkpoint = _sweep_module["_write_solver_checkpoint"]


def test_subset_profile_definition_preserves_requested_source_order():
    definition = CuratedProfileDefinition(
        family="test",
        temperatures=(100.0, 200.0, 300.0),
        pressures=(1.0, 2.0, 3.0),
    )

    subset, source_indices = _subset_profile_definition(definition, (2, 0, 2))

    assert source_indices == (2, 0)
    assert subset.temperatures == (300.0, 100.0)
    assert subset.pressures == (3.0, 1.0)


def test_subset_profile_definition_rejects_out_of_range_layer():
    definition = CuratedProfileDefinition(
        family="test",
        temperatures=(100.0,),
        pressures=(1.0,),
    )

    with pytest.raises(ValueError, match="Invalid layer indices"):
        _subset_profile_definition(definition, (1,))


def test_slice_batch_scalar_preserves_vector_diagnostics():
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]])

    assert _slice_batch_scalar(values, 1) == [3.0, 4.0]


def test_solver_checkpoint_atomically_preserves_batch_arrays(tmp_path):
    path = tmp_path / "checkpoint.pkl"

    _write_solver_checkpoint(
        path=path,
        family="test_family",
        variant="test_variant",
        arrays={"diagnostic": np.asarray([[1.0, 2.0]])},
    )

    with path.open("rb") as stream:
        payload = pickle.load(stream)
    assert payload["schema"] == "exogibbs_fixed_support_solver_checkpoint_v1"
    assert payload["arrays"]["diagnostic"].tolist() == [[1.0, 2.0]]
    assert not (tmp_path / ".checkpoint.pkl.tmp").exists()
