"""Compare the opt-in hot gas model with the original FastChem4 chemistry.

Run with JAX x64 on CPU; no data download or ExoJAX installation is needed.
The coarse grid is a model comparison, not a certificate for retrieval priors.
The comparison gate is explicitly looser than the RCE adapter's default.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from exogibbs.api.equilibrium import EquilibriumOptions, equilibrium_profile
from exogibbs.presets.fastchem4 import chemsetup
from exogibbs.thermo.models import ChemicalSetup
from exogibbs.thermo.standard import prepare_fastchem_thermodynamics


def run_audit(
    *,
    temperatures_k: tuple = (1500.0, 2200.0, 3000.0, 4500.0),
    pressures_bar: tuple = (1e-5, 0.01, 1.0, 100.0),
    abundance_points: tuple = ((0.0, 0.6), (-1.0, 0.3), (-1.0, 1.2), (1.0, 0.3), (1.0, 1.2)),
) -> dict:
    """Audit the grid product, retaining failed states and absent comparisons.

    Each abundance point is a pair of log10 metal scale and C/O ratio.
    The default grid contains the documented 80 states per model.
    """
    jax.config.update("jax_enable_x64", True)
    thermodynamics = prepare_fastchem_thermodynamics()
    original = chemsetup(path="FastChem4/logK/logK.dat", silent=True)
    excluded_name = "C4H6O4"
    excluded_index = original.species.index(excluded_name)
    retained = np.arange(len(original.species)) != excluded_index
    reduced = ChemicalSetup(
        formula_matrix=original.formula_matrix[:, retained],
        hvector_func=lambda t: original.hvector_func(t)[..., retained],
        species=tuple(s for i, s in enumerate(original.species) if retained[i]),
        elements=original.elements,
        element_vector_reference=original.element_vector_reference,
    )
    models = {
        "original": original,
        "omission_only": reduced,
        "hybrid": thermodynamics.chemical_setup,
    }
    temperature = jnp.repeat(jnp.asarray(temperatures_k), len(pressures_bar))
    pressure = jnp.tile(jnp.asarray(pressures_bar), len(temperatures_k))
    options = EquilibriumOptions(method="vmap_cold", epsilon_crit=1e-14, max_iter=2000)
    conservation_rtol = 3e-6
    elements = original.elements
    physical = np.array([e != "e-" for e in elements])
    metals = jnp.array([e not in ("H", "He", "e-") for e in elements])
    masses = jnp.array([thermodynamics.element_masses_u[e] for e in elements])
    carbon, oxygen = elements.index("C"), elements.index("O")
    values, report = {}, {}
    for label, setup in models.items():

        @jax.jit
        def solve(b):
            return equilibrium_profile(
                setup,
                temperature,
                pressure,
                b,
                options=options,
                return_diagnostics=True,
            )

        rows, fractions = [], []
        for metal_scale, ratio in abundance_points:
            b = original.element_vector_reference * jnp.where(
                metals, 10.0**metal_scale, 1.0
            )
            total_co = b[carbon] + b[oxygen]
            b = (
                b.at[carbon]
                .set(total_co * ratio / (1 + ratio))
                .at[oxygen]
                .set(total_co / (1 + ratio))
            )
            result, diagnostics = solve(b)
            n, x = np.asarray(result.n), np.asarray(result.x)
            error = np.max(
                np.abs(
                    n @ np.asarray(setup.formula_matrix)[physical].T
                    - np.asarray(b)[physical]
                )
                / np.asarray(b)[physical],
                axis=1,
            )
            charge = -np.asarray(setup.formula_matrix)[elements.index("e-")]
            charge_error = np.abs(x @ charge) / np.maximum(
                x @ np.abs(charge), np.finfo(x.dtype).tiny
            )
            mmw = x @ np.asarray(masses @ setup.formula_matrix)
            for i, (t, p) in enumerate(zip(temperature, pressure)):
                converged = bool(diagnostics["converged"][i])
                rows.append(
                    {
                        "temperature_K": float(t),
                        "pressure_bar": float(p),
                        "log_metal_scale": metal_scale,
                        "c_over_o": ratio,
                        "converged": converged,
                        "iterations": int(diagnostics["n_iter"][i]),
                        "relative_element_error": float(error[i]),
                        "relative_charge_error": float(charge_error[i]),
                        "valid_for_comparison": converged
                        and bool(error[i] <= conservation_rtol)
                        and bool(charge_error[i] <= conservation_rtol)
                        and bool(np.all(np.isfinite(x[i]) & (x[i] >= 0))),
                        "mmw_u": float(mmw[i]),
                    }
                )
                fractions.append(x[i])
        values[label] = np.asarray(fractions)
        report[label] = {"species_count": len(setup.species), "points": rows}

    for label in ("omission_only", "hybrid"):
        indices = [original.species.index(s) for s in models[label].species]
        reference = values["original"][:, indices]
        electron = models[label].species.index("e1-")
        common = []
        for i, (before, after) in enumerate(
            zip(report["original"]["points"], report[label]["points"])
        ):
            accepted = before["valid_for_comparison"] and after["valid_for_comparison"]
            comparison = {"accepted": accepted}
            if accepted:
                comparison.update(
                    {
                        "max_absolute_fraction_change": float(
                            np.max(np.abs(values[label][i] - reference[i]))
                        ),
                        "relative_mmw_change": float(
                            abs(after["mmw_u"] / before["mmw_u"] - 1)
                        ),
                        "relative_electron_fraction_change": float(
                            abs(values[label][i, electron] / reference[i, electron] - 1)
                        ),
                        "excluded_fraction": float(
                            values["original"][i, excluded_index]
                        ),
                    }
                )
                common.append(comparison)
            after["comparison_to_original"] = comparison
        report[label]["comparison_summary"] = {
            "common_valid_points": len(common),
            **{
                key: max((row[key] for row in common), default=None)
                for key in (
                    "max_absolute_fraction_change",
                    "relative_mmw_change",
                    "relative_electron_fraction_change",
                    "excluded_fraction",
                )
            },
        }
    for model in report.values():
        model["converged_points"] = sum(row["converged"] for row in model["points"])
        model["valid_points"] = sum(
            row["valid_for_comparison"] for row in model["points"]
        )
    return {
        "jax_version": jax.__version__,
        "platform": jax.default_backend(),
        "x64": bool(jax.config.x64_enabled),
        "epsilon_crit": options.epsilon_crit,
        "max_iter": options.max_iter,
        "comparison_element_rtol": conservation_rtol,
        "comparison_charge_rtol": conservation_rtol,
        "fastchem_sha256": thermodynamics.metadata["fastchem_sha256"],
        "nasa_excerpt_sha256": thermodynamics.metadata["nasa_excerpt_sha256"],
        "excluded_species": excluded_name,
        "models": report,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/rce_thermodynamics_audit.json")
    )
    args = parser.parse_args()
    result = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, model in result["models"].items():
        print(name, model["converged_points"], "/", len(model["points"]), "converged")
        if "comparison_summary" in model:
            print(json.dumps(model["comparison_summary"], sort_keys=True))
