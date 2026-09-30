"""The dissolved standard belongs to the original lower source callback."""
import hashlib
import importlib
from pathlib import Path
import sys

import jax
import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parents[3]/"examples/metal_silicate"
sys.path.insert(0, str(DIRECTORY))
try:
    M = importlib.import_module("m2_host_standards")
finally:
    sys.path.pop(0)


def source():
    jax.config.update("jax_enable_x64", True)
    hashes = {name: hashlib.sha256((DIRECTORY/name).read_bytes()).hexdigest()
              for name in ("run_bse_common_gibbs.py", "source.py", "reference.json", "m2_common_gas.py", "hydrogen.py", "m1_chemistry.py")}
    return {"gas_model": "janaf_condensed", "temperature_K": 2173.15, "pressure_bar": 269.6545152505794,
            "source_metadata": {"model_id": "bse_melts_ma_retained_atmosphere_conditional_v1",
                                "liquid_model": "published", "host_ledger": {"model_id": "melts_v102_published_mixing_native_standard_states_v1"},
                                "standards": {"common_gas": {"provenance": M._gas_provenance()}},
                                "provider_scenario": {"standard_offsets_rt": {"H2_dissolved": .5, "h2o_melts": 10.}},
                                "provenance": {"file_sha256": hashes, "jax_version": jax.__version__,
                                               "numpy_version": np.__version__, "exogibbs": {"commit": "test_fixture"}}}}


def test_replay_known_standard_and_keep_scenario_separate():
    result = M.source_hydrogen_standard_receipt(source())
    assert result["common_H2_gas_standard_rt"] == -19.747628372126396
    assert result["dissolved_h2_base_standard_rt"] == -8.324134628967352
    assert result["H2_dissolved_standard_offset_rt"] == .5
    assert result["common_lower_gas_model"] == "m1_expanded"
    assert result["terms_are_additive_separate_constants"]


@pytest.mark.parametrize("change", [
    lambda s: s.update(gas_model="unknown"),
    lambda s: s["source_metadata"].update(metal_model="unknown"),
    lambda s: s["source_metadata"].update(liquid_model="unknown"),
    lambda s: s.update(pressure_bar=float("nan")),
    lambda s: s["source_metadata"]["provenance"]["file_sha256"].update({"hydrogen.py": "wrong"}),
    lambda s: s["source_metadata"]["standards"]["common_gas"]["provenance"].update(package_python_sha256="wrong"),
    lambda s: s["source_metadata"]["provenance"].update(numpy_version="unknown"),
    lambda s: s["source_metadata"]["provider_scenario"]["standard_offsets_rt"].update(H2_dissolved=float("inf")),
])
def test_changed_recipe_or_invalid_source_fails_closed(change):
    saved = source()
    change(saved)
    with pytest.raises(ValueError):
        M.source_hydrogen_standard_receipt(saved)


@pytest.mark.parametrize("metal,key,model_id", [
    ("phosphorus", "phosphorus_metal", "ma_fe_si_o_h_p_dilute_quadratic_continuation_v1"),
    ("associated", "associated_metal", "ma_p_jung_associated_metal_continuation_v1"),
    ("associated_k", "associated_metal", "ma_p_jung_associated_potassium_sensitivity_v1"),
])
def test_explicit_new_metal_ids_preserve_the_hydrogen_recipe(metal, key, model_id):
    saved = source()
    saved["source_metadata"].update(metal_model=metal, **{key: {"model_id": model_id}})
    result = M.source_hydrogen_standard_receipt(saved)
    assert result["dissolved_h2_base_standard_rt"] == -8.324134628967352
    assert result["selected_metal_model"] == metal
