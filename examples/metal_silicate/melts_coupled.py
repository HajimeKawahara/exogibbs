"""Full MELTS host potentials and reduced dissolved hydrogen
==============================================================

This consumer owns basis selection and local chemistry. The provider remains
an optional, explicitly selected source checkout and evaluates properties only.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import numpy as np

from full_potential import PhaseState
from hydrogen import ideal_host_h2_dilution


COMMON_R = 8.31446261815324
PROVIDER_MODEL_ID = "alphamelts_2_3_2_rhyolite_melts_1_0_2_supplied_liquid_v1"
PROVIDER_MODEL_IDS = {
    1: PROVIDER_MODEL_ID,
    4: "alphamelts_2_3_2_rhyolite_melts_1_2_0_supplied_liquid_v1",
}


def _provider_model_id(evaluator: Any, calculation_mode: int) -> str:
    if type(calculation_mode) is not int or calculation_mode not in PROVIDER_MODEL_IDS:
        raise ValueError("calculation_mode must be 1 or 4; carbon requires mode 4.")
    expected = PROVIDER_MODEL_IDS[calculation_mode]
    declared = getattr(evaluator, "MODEL_IDS", {1: getattr(evaluator, "MODEL_ID", None)})
    if declared.get(calculation_mode) != expected:
        raise ValueError("The provider does not declare the selected MELTS model.")
    return expected


def load_melts_evaluator(checkout: Path) -> Any:
    """Load the supplied-composition evaluator from an explicit ExoEOS checkout."""
    path = Path(checkout).resolve() / "examples" / "melts_liquid_evaluator.py"
    spec = importlib.util.spec_from_file_location("_exogibbs_melts_property_provider", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the optional MELTS evaluator at {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path or module.MODEL_ID != PROVIDER_MODEL_ID:
        raise ValueError("The supplied checkout does not provide the declared MELTS model.")
    return module


def make_melts_h2_phase(
    evaluator: Any,
    host_components: Sequence[str],
    h2_standard_rt: Callable[[float, float], float],
    *, runtime: Path,
    python_executable: str,
    common_r: float = COMMON_R,
    calculation_mode: int = 1,
) -> Callable[[float, float, np.ndarray], PhaseState]:
    """Return full potentials in selected host order followed by dissolved H2.

    Inputs use K/bar/mol; the external worker receives K/Pa/mol in its complete
    19-component basis. H2 has a separately supplied absolute standard. MELTS
    H2O is retained without adding another H2O-solubility equation. No arbitrary
    phase-specific standard offset is applied to make an equilibrium fit.
    Cross-phase standard alignment and calibration must be assessed separately.
    Explicit mode 4 selects rhyolite-MELTS 1.2.0 with an independent CO2
    component. Its internal CaCO3 species is not another component amount.
    The default mode 1 retains the original provider calling convention.
    """
    names = tuple(host_components)
    model_id = _provider_model_id(evaluator, calculation_mode)
    if not jax.config.x64_enabled:
        raise RuntimeError("MELTS coupling requires JAX_ENABLE_X64=1 for the residual tolerances.")
    if not names or len(set(names)) != len(names) or not set(names) <= set(evaluator.COMPONENTS):
        raise ValueError("Host components must be unique, declared MELTS endmembers.")
    if not np.isfinite(common_r) or common_r <= 0:
        raise ValueError("The common gas constant must be positive and finite.")
    indices = np.array([evaluator.COMPONENTS.index(name) for name in names])
    mode_arguments = {} if calculation_mode == 1 else {"calculation_mode": calculation_mode}

    def evaluate(temperature: float, pressure: float, amounts: np.ndarray) -> PhaseState:
        n = np.asarray(amounts, dtype=float)
        if n.shape != (len(names) + 1,) or not np.all(np.isfinite(n)) or np.any(n < 0) or n[:-1].sum() <= 0:
            raise ValueError("Supply finite nonnegative host/H2 amounts with a positive host.")
        host = np.zeros(len(evaluator.COMPONENTS))
        host[indices] = n[:-1]
        result = evaluator.evaluate_liquid(
            temperature, pressure * 1e5, host, runtime=runtime,
            common_R=common_r, python_executable=python_executable, **mode_arguments,
        )
        if (result["status"] != "ok_supplied_liquid_properties"
                or result["model_id"] != model_id
                or tuple(result["component_order"]) != tuple(evaluator.COMPONENTS)
                or result["phase_policy"]["oxygen_buffer"] != "None"):
            raise ValueError("Unexpected provider model, component basis, or imposed oxygen buffer.")
        if calculation_mode == 4:
            backend = result.get("provenance", {}).get("backend", {})
            if (backend.get("calculation_mode") != 4
                    or backend.get("model") != "rhyolite-MELTS 1.2.0"):
                raise ValueError("The provider returned a different MELTS calculation mode.")
        if (result["T_K"] != temperature or result["P_Pa"] != pressure * 1e5
                or result["basis"]["common_R_J_mol_K"] != common_r):
            raise ValueError("The provider returned different conditions or a different common R.")
        if not np.allclose(result["returned_component_moles"], host, rtol=5e-9, atol=0):
            raise ValueError("The provider changed the requested finite host composition.")
        host_mu = np.asarray(result["mu_RT"], dtype=float)[indices]
        if not np.all(np.isfinite(host_mu[n[:-1] > 0])):
            raise ValueError("The provider omitted a present host component potential.")
        # Multiplying zero amounts by unavailable endpoint potentials would
        # produce NaNs; the mixing construction only needs the active support.
        diluted = ideal_host_h2_dilution(
            result["gibbs_J"] / (common_r * temperature), host_mu,
            n[:-1], n[-1], h2_standard_rt(temperature, pressure),
        )
        return PhaseState(
            np.append(np.asarray(diluted.host_mu_rt), float(diluted.h2_mu_rt)),
            float(diluted.gibbs_rt),
        )

    return evaluate


def provider_ledger(
    evaluator: Any, host_components: Sequence[str], *, calculation_mode: int = 1,
) -> dict[str, Any]:
    """Record the actual provider file and complete finite host basis."""
    model_id = _provider_model_id(evaluator, calculation_mode)
    path = Path(evaluator.__file__).resolve()
    indices = [evaluator.COMPONENTS.index(name) for name in host_components]
    elements = list(evaluator.ELEMENTS)
    hydrogen = [2 if element == "H" else 0 for element in elements]
    backend = dict(evaluator.REFERENCE["backend"])
    if calculation_mode == 4:
        backend.update(model="rhyolite-MELTS 1.2.0", calculation_mode=4)
    return {
        "model_id": model_id,
        "calculation_mode": calculation_mode,
        "evaluator_path": str(path),
        "evaluator_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "backend": backend,
        "reference_sha256": hashlib.sha256(evaluator.REFERENCE_PATH.read_bytes()).hexdigest(),
        "component_order": list(host_components) + ["H2_dissolved"],
        "element_order": elements,
        "formula_matrix_component_rows": evaluator.FORMULA_MATRIX[indices].tolist() + [hydrogen],
        "amount_basis": "mol of named independent MELTS component; mol of molecular dissolved H2",
        "standard_convention": "full MELTS potentials at T/P; absolute dissolved H2 standard supplied separately",
        "reference_pressure_bar": 1.0,
        "energy_units": "G/(common_R*T), mu/(common_R*T)",
        "mathematical_domain": "positive host total, nonnegative H2; provider must accept the supplied composition",
        "calibration_domain": "not established for the coupled reduced host/alloy/gas model",
        "stable_phase_evidence": "not supplied by a liquid property callback",
        "extrapolation_policy": "conditional mechanism only until common standards and competing phases are validated",
    }
