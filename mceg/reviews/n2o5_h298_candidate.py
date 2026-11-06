"""Retain N2O5 thermals while updating its balanced 298.15 K energy anchor.

The missing JANAF zero-K row forbids a D0-based correction, not a correction
based on the available reaction enthalpy at 298.15 K. No D0 is inferred here.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

from c2h_source_review import read_table
from gas_source_comparison import ROOT, file_hash
from pending_nitrogen_inputs import source_record

R = 8.31446261815324
SOURCE = ROOT / "tests/data/validation/n2o5_source_review/20261106"
DATA = ROOT / "tests/data/validation/n2o5_h298_candidate/20261106"


def build() -> dict:
    """Balance same-version atoms and molecule at the available source anchor."""
    record = source_record("n2o5")
    assert record["janaf_D0_J_mol"] is None
    source = json.loads((SOURCE / "inputs.json").read_text())
    janaf = {name: round(1000*read_table(SOURCE/path)[298.15][4])
             for name, path in source["tables"].items()}
    atct = {name: round(1000*value["hf298_kJ_mol"])
            for name, value in source["atct"].items()}
    balance = lambda values: math.fsum(count*values[element]
        for element, count in source["composition"].items())-values["N2O5"]
    old, new = balance(janaf), balance(atct)
    assert (old, new) == (2179934, 2176321)
    spec = {"kind": "retained_energy", "coefficients": record["original_coefficients"],
            "delta_GH_J_mol": old-new, "energy_reference_temperature_K": 298.15,
            "physical_JANAF_D0_J_mol": None}
    return {"id": "n2o5_retained_h298_candidate_20261106",
            "baseline_dataset": "curated-ch2-v1", "temperature_range_K": [1500, 4000],
            "tracked_elements": ["N", "O"], "primary_variant": "retained_h298",
            "variants": [{"name": "retained_h298", "replacements": {"N2O5": spec}}],
            "selected_variants": [], "source_record": record,
            "energy_ledger": {"reference_temperature_K": 298.15,
                "JANAF_Hf298_J_mol": janaf, "ATcT_1_228_Hf298_J_mol": atct,
                "JANAF_atomization_H298_J_mol": old,
                "ATcT_atomization_H298_J_mol": new,
                "delta_formation_GH_J_mol": old-new,
                "physical_JANAF_D0_J_mol": None,
                "scope": "Balanced reaction H298 anchor, not D0; unchanged source thermal shape and fit residual."},
            "source_sha256": {**record["source_sha256"],
                str(Path(__file__).resolve().relative_to(ROOT)): file_hash(Path(__file__))},
            "scope": "Retained JANAF-derived reaction thermals with the available H298 energy correction; no production adoption or invented zero-K energy."}


def validate(definition: dict) -> dict:
    """Check source identity, literal thermals, residual and scalar/JAX calorics."""
    import jax
    from exogibbs.presets import curated_ch2
    import pending_candidate as model

    assert definition == build()
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        temperatures = np.unique(np.r_[np.arange(1500., 4001.), np.linspace(1500.137, 3999.813, 251)])
        setup = curated_ch2.chemsetup(ions=True)
        assert setup.metadata["dataset"] == "curated-ch2-v1"
        spec = definition["variants"][0]["replacements"]["N2O5"]
        a, b, c, d, e = spec["coefficients"]
        shift, t = spec["delta_GH_J_mol"], temperatures
        old_g = -R*(a+b*t*np.log(t)+c*t+d*t*t+e*t**3)
        old_h = R*(-a+b*t+d*t*t+2*e*t**3)
        old_s = R*(b*(np.log(t)+1)+c+2*d*t+3*e*t*t)
        old_cp = R*(b+2*d*t+6*e*t*t)
        expected = np.array([old_g+shift, old_h+shift, old_s, old_cp]).T
        current = np.asarray(setup.hvector_func(t))[:, setup.species.index("N2O5")]
        runtime_error = float(abs(current-old_g/(R*t)).max())
        assert runtime_error < 1e-12
        actual = np.array([list(model.reaction_properties(float(temp), spec).values()) for temp in t])
        errors = dict(zip(model.PROPERTY_KEYS, map(float, abs(actual-expected).max(axis=0))))
        assert max(errors["G_J_mol"], errors["H_J_mol"]) < 1e-8
        assert max(errors["S_J_mol_K"], errors["Cp_J_mol_K"]) < 2e-10
        comparison = json.loads((SOURCE/"comparison.json").read_text())
        residuals = []
        old_spec = {**spec, "delta_GH_J_mol": 0.}
        for row in comparison["source_samples"]:
            temp, ln_k = row["temperature_K"], row["native_reactions"]["janaf"]["ln_K"]
            old_residual = -model.formation_h(temp, old_spec)-ln_k
            new_residual = -model.formation_h(temp, spec)-(ln_k-shift/(R*temp))
            assert abs(new_residual-old_residual) < 1e-12
            residuals.append({"temperature_K": temp, "old_fit_minus_JANAF_ln_K": old_residual,
                              "preservation_error_ln_K": new_residual-old_residual})
        return {"temperature_count": len(t), "maximum_current_h_error": runtime_error,
                "independent_source_property_errors": errors, "retained_source_fit_residuals": residuals,
                "jax_derivative_checks": model.validate_definition(definition),
                "energy_reference_temperature_K": 298.15, "physical_JANAF_D0_J_mol": None,
                "production_change": "none", "physical_accuracy": "unknown"}
    finally:
        jax.config.update("jax_enable_x64", previous)


def write_packet() -> dict:
    """Execute source validation and pin the actual inputs and output bytes."""
    DATA.mkdir(parents=True, exist_ok=True)
    if (DATA/"receipt.json").exists():
        raise FileExistsError("Refuse to overwrite executed evidence")
    definition = build()
    result = validate(definition)
    for name, value in (("candidate.json", definition), ("validation.json", result)):
        (DATA/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    paths = {ROOT/p for p in definition["source_sha256"]}
    paths.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                 if getattr(module, "__file__", None) and Path(module.__file__).resolve().is_relative_to(ROOT)
                 and Path(module.__file__).suffix == ".py"
                 and Path(module.__file__).name not in ("curated.py", "exogibbs_version.py"))
    paths.update((ROOT/"src/exogibbs/data/curated").glob("*.json"))
    import jax
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "result": "passed",
               "command": [sys.executable, *sys.argv], "python": platform.python_version(),
               "jax": jax.__version__, "numpy": np.__version__, "dtype": "float64",
               "backend": jax.default_backend(),
               "selected_file_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)},
               "output_sha256": {name: file_hash(DATA/name) for name in ("candidate.json", "validation.json")}}
    (DATA/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return result


if __name__ == "__main__":
    print(json.dumps(write_packet(), indent=2))
