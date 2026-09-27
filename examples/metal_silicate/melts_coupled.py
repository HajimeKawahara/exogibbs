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
import jax.numpy as jnp
from jax.scipy.special import xlogy
import numpy as np

from full_potential import PhaseState
from hydrogen import ideal_host_h2_dilution


COMMON_R = 8.31446261815324
PROVIDER_MODEL_ID = "alphamelts_2_3_2_rhyolite_melts_1_0_2_supplied_liquid_v1"
PUBLISHED_MODEL_ID = "melts_v102_published_mixing_native_standard_states_v1"


def saved_liquid_model(source: dict) -> str:
    """Read consistent explicit model declarations, defaulting legacy records to native."""
    metadata = source.get("source_metadata", {})
    ledger = metadata.get("host_ledger", {})
    declarations = [item["liquid_model"] for item in
                    (source, source.get("arguments", {}), metadata, ledger)
                    if "liquid_model" in item]
    model = declarations[0] if declarations else "native"
    identities = {"native": PROVIDER_MODEL_ID, "published": PUBLISHED_MODEL_ID}
    if (not isinstance(model, str) or model not in identities or any(value != model for value in declarations)
            or ledger.get("model_id", identities[model]) != identities[model]):
        raise ValueError("The saved liquid model declarations are unknown or inconsistent.")
    return model


def load_melts_evaluator(checkout: Path, *, liquid_model="native", runtime=None,
                         python_executable=None) -> Any:
    """Load the supplied-composition evaluator from an explicit ExoEOS checkout."""
    path = Path(checkout).resolve() / "examples" / "melts_liquid_evaluator.py"
    spec = importlib.util.spec_from_file_location("_exogibbs_melts_property_provider", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the optional MELTS evaluator at {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path or module.MODEL_ID != PROVIDER_MODEL_ID:
        raise ValueError("The supplied checkout does not provide the declared MELTS model.")
    if liquid_model == "native":
        return module
    if liquid_model != "published" or runtime is None or python_executable is None:
        raise ValueError("Select native or published; the published model requires a native standard runtime and Python.")
    mixing_path = path.with_name("melts_liquid_mixing.py")
    mixing_spec = importlib.util.spec_from_file_location("_exogibbs_melts_mixing_provider", mixing_path)
    mixing = importlib.util.module_from_spec(mixing_spec)
    mixing_spec.loader.exec_module(mixing)
    if mixing.PUBLISHED_MODEL_ID != PUBLISHED_MODEL_ID:
        raise ValueError("The published provider has an unexpected model identity.")
    return mixing.make_published_liquid_evaluator(module, runtime=runtime,
                                                 python_executable=python_executable)


def make_melts_h2_phase(
    evaluator: Any,
    host_components: Sequence[str],
    h2_standard_rt: Callable[[float, float], float],
    *, runtime: Path,
    python_executable: str,
    common_r: float = COMMON_R,
) -> Callable[[float, float, np.ndarray], PhaseState]:
    """Return full potentials in selected host order followed by dissolved H2.

    Inputs use K/bar/mol; the external worker receives K/Pa/mol in its complete
    19-component basis. H2 has a separately supplied absolute standard. MELTS
    H2O is retained without adding another H2O-solubility equation. No arbitrary
    phase-specific standard offset is applied to make an equilibrium fit.
    Cross-phase standard alignment and calibration must be assessed separately.
    """
    names = tuple(host_components)
    selected_model = getattr(evaluator, "MODEL_ID", PROVIDER_MODEL_ID)
    if selected_model not in {PROVIDER_MODEL_ID, PUBLISHED_MODEL_ID}:
        raise ValueError("Unsupported supplied-liquid model identity.")
    if not jax.config.x64_enabled:
        raise RuntimeError("MELTS coupling requires JAX_ENABLE_X64=1 for the residual tolerances.")
    if not names or len(set(names)) != len(names) or not set(names) <= set(evaluator.COMPONENTS):
        raise ValueError("Host components must be unique, declared MELTS endmembers.")
    if not np.isfinite(common_r) or common_r <= 0:
        raise ValueError("The common gas constant must be positive and finite.")
    indices = np.array([evaluator.COMPONENTS.index(name) for name in names])

    def evaluate(temperature: float, pressure: float, amounts: np.ndarray) -> PhaseState:
        n = np.asarray(amounts, dtype=float)
        if n.shape != (len(names) + 1,) or not np.all(np.isfinite(n)) or np.any(n < 0) or n[:-1].sum() <= 0:
            raise ValueError("Supply finite nonnegative host/H2 amounts with a positive host.")
        host = np.zeros(len(evaluator.COMPONENTS))
        host[indices] = n[:-1]
        result = evaluator.evaluate_liquid(
            temperature, pressure * 1e5, host, runtime=runtime,
            common_R=common_r, python_executable=python_executable,
        )
        if (result["status"] != "ok_supplied_liquid_properties"
                or result["model_id"] != selected_model
                or tuple(result["component_order"]) != tuple(evaluator.COMPONENTS)
                or result["phase_policy"]["oxygen_buffer"] != "None"):
            raise ValueError("Unexpected provider model, component basis, or imposed oxygen buffer.")
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

    independent_host = getattr(evaluator, "energy_value_and_grad_rt", None)
    if independent_host is not None:
        @jax.jit
        def dilution_scalar(n, standard):
            host, hydrogen = jnp.sum(n[:-1]), n[-1]
            total = host + hydrogen
            return (hydrogen * standard + xlogy(host, host/total)
                    + xlogy(hydrogen, jnp.where(hydrogen > 0, hydrogen/total, 1.)))

        dilution_gradient = jax.jit(jax.value_and_grad(dilution_scalar))

        def energy_value_and_grad_rt(temperature, pressure, amounts):
            n = np.asarray(amounts, dtype=float)
            if (n.shape != (len(names)+1,) or not np.all(np.isfinite(n))
                    or np.any(n < 0) or n[:-1].sum() <= 0):
                raise ValueError("Supply finite nonnegative host/H2 amounts with a positive host.")
            host = np.zeros(len(evaluator.COMPONENTS))
            host[indices] = n[:-1]
            energy, gradient = independent_host(
                temperature, pressure*1e5, host, runtime=runtime,
                common_R=common_r, python_executable=python_executable)
            extra, extra_gradient = dilution_gradient(n, h2_standard_rt(temperature, pressure))
            combined = np.append(np.asarray(gradient)[indices], 0.) + np.asarray(extra_gradient)
            if n[-1] == 0:
                combined[-1] = -np.inf
            return float(energy + extra), combined

        evaluate.energy_value_and_grad_rt = energy_value_and_grad_rt

    return evaluate


def provider_ledger(evaluator: Any, host_components: Sequence[str]) -> dict[str, Any]:
    """Record the actual provider file and complete finite host basis."""
    path = Path(evaluator.__file__).resolve()
    indices = [evaluator.COMPONENTS.index(name) for name in host_components]
    elements = list(evaluator.ELEMENTS)
    hydrogen = [2 if element == "H" else 0 for element in elements]
    return {
        "model_id": getattr(evaluator, "MODEL_ID", PROVIDER_MODEL_ID),
        "evaluator_path": str(path),
        "evaluator_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "backend": evaluator.REFERENCE["backend"],
        "reference_sha256": hashlib.sha256(evaluator.REFERENCE_PATH.read_bytes()).hexdigest(),
        "component_order": list(host_components) + ["H2_dissolved"],
        "element_order": elements,
        "formula_matrix_component_rows": evaluator.FORMULA_MATRIX[indices].tolist() + [hydrogen],
        "amount_basis": "mol of named MELTS endmember; mol of molecular dissolved H2",
        "standard_convention": "full MELTS potentials at T/P; absolute dissolved H2 standard supplied separately",
        "reference_pressure_bar": 1.0,
        "energy_units": "G/(common_R*T), mu/(common_R*T)",
        "mathematical_domain": "positive host total, nonnegative H2; provider must accept the supplied composition",
        "calibration_domain": "not established for the coupled reduced host/alloy/gas model",
        "stable_phase_evidence": "not supplied by a liquid property callback",
        "extrapolation_policy": "conditional mechanism only until common standards and competing phases are validated",
        **({"liquid_model": "published", "mixing_model_id": evaluator.mixing_model_id,
            "mixing_parameter_sha256": evaluator.mixing_parameter_sha256,
            "native_standard_state_receipts": evaluator.standard_state_receipts}
           if getattr(evaluator, "MODEL_ID", PROVIDER_MODEL_ID) == PUBLISHED_MODEL_ID else {"liquid_model": "native"}),
    }
