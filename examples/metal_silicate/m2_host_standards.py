"""Replay the saved source builder's dissolved-H2 standard without EOS calls."""
import hashlib
from pathlib import Path

import jax
import numpy as np

from hydrogen import dissolved_h2_standard_rt, hirschmann2012_ln_solubility
from m1_chemistry import provenance as _gas_provenance
from m2_common_gas import anchored_standards_rt, build_common_gas_setup
from run_bse_common_gibbs import source_standards_rt


def source_hydrogen_standard_receipt(source: dict) -> dict:
    """Return the unchanged lower-builder H2 standard and separate shift.

    Retained-atmosphere replacement does not replace the original H2
    solubility callback. In particular its JANAF H2 standard must not silently
    replace the callback's anchored common-gas value.
    """
    metadata = source["source_metadata"]
    if (metadata["model_id"] != "bse_melts_ma_retained_atmosphere_conditional_v1"
            or source["gas_model"] not in {"m1", "janaf", "janaf_condensed"}):
        raise ValueError("Unsupported source builder for the dissolved-H2 replay.")
    liquid_ids = {"native": "alphamelts_2_3_2_rhyolite_melts_1_0_2_supplied_liquid_v1",
                  "published": "melts_v102_published_mixing_native_standard_states_v1",
                  "published_water": "dry_melts_thompson2025_water_equivalent_v1"}
    liquid = metadata.get("liquid_model", "native")
    metal = metadata.get("metal_model", "ma")
    metal_ids = {"phosphorus": ("phosphorus_metal", "ma_fe_si_o_h_p_dilute_quadratic_continuation_v1"),
                 "associated": ("associated_metal", "ma_p_jung_associated_metal_continuation_v1"),
                 "associated_k": ("associated_metal", "ma_p_jung_associated_potassium_sensitivity_v1")}
    if (liquid not in liquid_ids or metadata["host_ledger"]["model_id"] != liquid_ids[liquid]
            or metal not in {"ma", *metal_ids}
            or (metal in metal_ids and metadata[metal_ids[metal][0]]["model_id"] != metal_ids[metal][1])):
        raise ValueError("Unsupported selected liquid or metal model for the saved builder.")
    temperature, pressure = source["temperature_K"], source["pressure_bar"]
    if not np.isfinite(temperature) or temperature <= 0 or not np.isfinite(pressure) or pressure <= 0:
        raise ValueError("A finite positive source T/P is required.")
    provenance = metadata["provenance"]
    if (not jax.config.read("jax_enable_x64") or provenance["jax_version"] != jax.__version__
            or provenance["numpy_version"] != np.__version__):
        raise ValueError("Replay with the saved JAX/NumPy versions and x64 enabled.")
    directory = Path(__file__).resolve().parent
    files = {}
    for name in ("run_bse_common_gibbs.py", "source.py", "reference.json", "m2_common_gas.py", "hydrogen.py", "m1_chemistry.py"):
        path = directory/name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != provenance["file_sha256"].get(name):
            raise ValueError("The saved dissolved-H2 recipe has changed: "+name)
        files[name] = digest
    gas_before = metadata["standards"]["common_gas"]["provenance"]
    gas_now = _gas_provenance()
    def catalog_files(value):
        names = [Path(row["path"]).name for row in value["files"]]
        if len(names) != len(set(names)):
            raise ValueError("Ambiguous common-gas provenance filenames.")
        return {Path(row["path"]).name: row["sha256"] for row in value["files"]}
    if (gas_before["package_python_sha256"] != gas_now["package_python_sha256"]
            or catalog_files(gas_before) != catalog_files(gas_now)):
        raise ValueError("The saved common-gas implementation or thermochemical data have changed.")
    setup = build_common_gas_setup(expanded=True)
    reference, _ = source_standards_rt(temperature, pressure)
    standards, _ = anchored_standards_rt(setup, temperature, reference)
    gas_standard = float(standards[setup.species.index("H2")])
    log_solubility = float(hirschmann2012_ln_solubility(pressure))
    base = float(dissolved_h2_standard_rt(gas_standard, log_solubility))
    scenario = metadata.get("provider_scenario") or {}
    offset = float(scenario.get("standard_offsets_rt", {}).get("H2_dissolved", 0.))
    if not all(np.isfinite(value) for value in (gas_standard, log_solubility, base, offset)):
        raise ValueError("Every replayed hydrogen standard term must be finite.")
    return {"schema": "saved_builder_dissolved_hydrogen_standard_v1",
            "temperature_K": temperature, "pressure_bar": pressure,
            "common_lower_gas_model": "m1_expanded", "gas_species": "H2",
            "selected_liquid_model": liquid, "selected_metal_model": metal,
            "common_H2_gas_standard_rt": gas_standard,
            "ln_solubility_per_fugacity_bar": log_solubility,
            "dissolved_h2_base_standard_rt": base, "H2_dissolved_standard_offset_rt": offset,
            "terms_are_additive_separate_constants": True,
            "recipe_sha256": files, "source_provider_provenance": provenance["exogibbs"],
            "gas_catalog_file_sha256": catalog_files(gas_now),
            "gas_package_python_sha256": gas_now["package_python_sha256"],
            "jax_version": jax.__version__, "numpy_version": np.__version__,
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Replay of the executed lower-builder standard; no replacement by retained-gas H2."}
