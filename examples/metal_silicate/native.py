"""Local source and native ExoEOS equilibrium references
======================================================

Reproduce the full source network, then run the explicitly smaller dry
silicate/Fe-Si-O model. ExoEOS is optional and used only by the dry model.
Both calculations assume fixed phases and formal, 1 bar liquid standards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from local import LocalProblem, build_problem, solve
from reference import load_reference
from source import (
    make_source_standard_potentials_rt,
    source_metal_ln_gamma,
    source_reaction_offsets,
)

from exogibbs.interop.exoeos import make_solution_lngamma_func


def load_equilibrium_reference() -> dict[str, Any]:
    """Read independently generated, offline numerical reference cases."""
    return json.loads(Path(__file__).with_name("equilibrium_reference.json").read_text())


def make_reference_problem(
    record: dict[str, Any], case: dict[str, Any], *, model: Any = None,
) -> LocalProblem:
    """Build the full source or explicitly dry, completed-metal calculation.

    The dry model defaults to ExoEOS's pinned ``MaFeSiOLiquid`` family;
    its formal standard shift is taken from the same instance as activities.
    Silicate mixing remains the source's ideal five-component approximation.
    H is never appended to the ternary activity vector.
    """
    thermo = next(item for item in record["cases"] if item["T_K"] == case["T_K"])
    source_standards = make_source_standard_potentials_rt(record, thermo)
    if case["model"] == "gce_young_2023_author_code":
        if model is not None:
            raise ValueError("The full source case uses its prescribed four-component alloy.")
        return build_problem(
            record, case["element_amounts_mol"], source_standards,
            phases=("silicate", "metal", "gas"),
            activity_functions={"metal": source_metal_ln_gamma},
            reaction_offset=source_reaction_offsets,
        )
    if case["model"] != "ma2001_fe_si_o_young2023_printed_v1":
        raise ValueError("Unknown equilibrium reference model.")
    if model is None:
        try:
            from exoeos import MaFeSiOLiquid
        except ImportError as exc:
            raise ImportError("The dry reference requires ExoEOS with MaFeSiOLiquid.") from exc
        model = MaFeSiOLiquid()
    if getattr(model, "reference_model_id", None) != case["model"]:
        raise ValueError("The dry reference requires the declared Ma Fe-Si-O model family.")
    labels = ("Fe", "Si", "O")
    activity = make_solution_lngamma_func(source_components=labels, model=model)
    species = tuple(name for names in record["phases"].values() for name in names)
    metal_indices = jnp.asarray([species.index(name + "_metal") for name in labels])

    def standards(temperature, pressure):
        shift = model.standard_state_shift_RT(temperature)
        # The provider's shift follows model.components, as do its activities.
        shift = shift[jnp.asarray([model.components.index(name) for name in labels])]
        return source_standards(temperature, pressure).at[metal_indices].add(shift)

    return build_problem(
        record, case["element_amounts_mol"], standards,
        activity_functions={"metal": activity},
    )


def reference_initial_amounts(record: dict[str, Any], case: dict[str, Any]) -> np.ndarray:
    """Return the independent solver's starting guess in the full basis."""
    species = tuple(name for names in record["phases"].values() for name in names)
    amounts = np.zeros(len(species))
    indices = [species.index(name) for name in case["active_components"]]
    amounts[indices] = case["initial_component_amounts_mol_active"]
    return amounts


def run_reference(case_id: str) -> dict[str, Any]:
    """Solve one case and report independent elemental and reaction gates."""
    record = load_reference()
    case = next(item for item in load_equilibrium_reference()["cases"] if item["id"] == case_id)
    problem = make_reference_problem(record, case)
    result = solve(
        problem, case["T_K"], case["P_bar"], np.asarray(case["element_amounts_mol"]),
        initial_component_amounts_mol=reference_initial_amounts(record, case),
    )
    converged = bool(result.root_solution.converged)
    element_error = float(jnp.max(jnp.abs(result.element_residual)))
    reaction_error = float(jnp.max(jnp.abs(result.reaction_residual)))
    accepted = converged and element_error <= 1e-9 and reaction_error <= 1e-8
    return {
        "case": case_id, "accepted": accepted,
        "scope": "Fixed-phase formal liquid reference; no phase stability or joint calibration claim.",
        "root_converged": converged,
        "phase_amounts_mol": dict(zip(problem.phases, np.asarray(result.phase_amounts_mol).tolist())),
        "max_relative_element_residual": element_error,
        "max_abs_reaction_residual": reaction_error,
        "component_amounts_mol": np.asarray(result.component_amounts_mol).tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="source_full_2350")
    arguments = parser.parse_args()
    output = run_reference(arguments.case)
    print(json.dumps(output, indent=2, allow_nan=False))
    if not output["accepted"]:
        raise SystemExit(1)
