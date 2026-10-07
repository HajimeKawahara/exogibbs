"""Common hydrogen standards and finite inventories
=====================================================

The named older solubility law is a sensitivity model. The revised 2025
experimental tables are not vendored, so no replacement calibration is fitted.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import xlogy

import exogibbs

from local import LocalProblem, build_problem as build_local_problem, solve
from reference import load_reference
from source import make_source_standard_potentials_rt

from exogibbs.interop.exoeos import make_solution_lngamma_func


MODEL_ID = "source_host_ma_fe_si_o_h_hirschmann2012_pressure_control_v1"
H2_CALIBRATION = {
    "model_id": "hirschmann2012_seo2024_mole_fraction_sensitivity",
    "equation": "x_H2 = (f_H2 / bar) exp(-11.403 - 0.76 P_melt/GPa)",
    "concentration_basis": "Model-assumed H2 molecular mole fraction in the full liquid endmember basis; experimental denominator conversion is unverified",
    "denominator_status": "unverified_experimental_to_MELTS_endmember_conversion",
    "fugacity_basis": "H2 fugacity in bar, not total fluid pressure",
    "pressure_basis": "total melt pressure in GPa; pressure correction occurs once in mu0_H2",
    "experimental_doi": "10.1016/j.epsl.2012.06.031",
    "formulation_doi": "10.3847/1538-4357/ad7461",
    "calibration_temperature_k": [1673.0, 1773.0],
    "calibration_total_pressure_gpa": [0.7, 3.0],
    "revised_candidate": {
        "doi": "10.1007/s00410-025-02272-y",
        "status": "host_specific_calibration_and_mole_basis_unestablished",
        "open_tables_url": "https://epub.uni-bayreuth.de/id/eprint/8940/1/s00410-025-02272-y.pdf",
        "reported_concentration": "H2 ppm by weight (micrograms per gram of glass); Table 2",
        "reported_host_basis": "Fe-free experimental compositions; not a BSE calibration",
        "direct_hosts": "Fe-free synthetic basalt and andesite; direct experiments through 1673.15 K",
        "policy": "Published data are not adopted as a BSE calibration; no universal factor-of-ten rescaling or joint valid domain is established.",
    },
}


def hirschmann2012_ln_solubility(pressure_bar: Any) -> jax.Array:
    """Return ln(x_H2 / (f_H2/bar)); pressure work is included in this law."""
    pressure = jnp.asarray(pressure_bar)
    value = -11.403 - 0.76e-4 * pressure
    return jnp.where(jnp.isfinite(pressure) & (pressure >= 0), value, jnp.nan)


def dissolved_h2_standard_rt(
    gas_standard_rt: Any, ln_mole_fraction_per_fugacity_bar: Any,
    *, standard_pressure_bar: float = 1.0,
) -> jax.Array:
    """Convert x_H2 = K f_H2[bar] to one common liquid species standard.

    All inputs and outputs use the same gas constant in mu/(R*T). The liquid
    activity is its full-phase molecular mole fraction. Any pressure work
    already included in K must not be added again. K may depend on T/P; host
    composition dependence requires a separate scalar free-energy treatment.
    """
    if not np.isfinite(standard_pressure_bar) or standard_pressure_bar <= 0:
        raise ValueError("standard_pressure_bar must be finite and positive.")
    return (jnp.asarray(gas_standard_rt) - jnp.asarray(ln_mole_fraction_per_fugacity_bar)
            - jnp.log(standard_pressure_bar))


class HydrogenDilution(NamedTuple):
    """Extensive G/(RT), host mu/(RT), and molecular H2 mu/(RT)."""

    gibbs_rt: jax.Array
    host_mu_rt: jax.Array
    h2_mu_rt: jax.Array


def ideal_host_h2_dilution(
    host_gibbs_rt: Any, host_mu_rt: Any, host_amounts: Any,
    h2_amount: Any, h2_standard_rt: Any,
) -> HydrogenDilution:
    """Add ideal host/H2 dilution to an already mixed full host potential.

    The host amount is the sum of its endmember moles, not oxide or atom moles.
    Convert backend G and mu to one common R*T before calling. This adds only
    dilution by H2, so host ideal/excess mixing and its pressure work are not
    duplicated. The H2 standard must be independent of host composition.
    The positive host and nonnegative H2 domain includes exact zero H2, whose
    insertion chemical potential is -inf. Pure H2 is outside the host domain.
    """
    amounts = jnp.asarray(host_amounts)
    host_mu = jnp.asarray(host_mu_rt)
    hydrogen = jnp.asarray(h2_amount)
    gibbs = jnp.asarray(host_gibbs_rt)
    standard = jnp.asarray(h2_standard_rt)
    if amounts.ndim != 1 or not amounts.size or host_mu.shape != amounts.shape:
        raise ValueError("host_amounts and host_mu_rt must have the same nonempty vector shape.")
    if hydrogen.ndim or gibbs.ndim or standard.ndim:
        raise ValueError("H2 amount, standard and extensive host Gibbs energy must be scalars.")
    host_total = jnp.sum(amounts)
    total = host_total + hydrogen
    host_fraction = host_total / total
    h2_fraction = hydrogen / total
    mixing = xlogy(host_total, host_fraction) + xlogy(hydrogen, h2_fraction)
    valid = (jnp.all(jnp.isfinite(amounts)) & jnp.all(amounts >= 0)
             & jnp.isfinite(host_total) & (host_total > 0)
             & jnp.isfinite(hydrogen) & (hydrogen >= 0)
             & jnp.isfinite(total) & jnp.isfinite(gibbs) & jnp.isfinite(standard))
    return HydrogenDilution(
        jnp.where(valid, gibbs + hydrogen * standard + mixing, jnp.nan),
        jnp.where(valid, host_mu + jnp.log(host_fraction), jnp.nan),
        jnp.where(valid, standard + jnp.log(h2_fraction), jnp.nan),
    )


def make_common_standard_potentials_rt(
    record: dict[str, Any], case: dict[str, Any], *, model: Any,
) -> Callable[[Any, Any], jax.Array]:
    """Replace the dissolved-H2 standard and apply the native alloy shifts.

    Original source records remain untouched. The source's Okuchi absolute H
    potential is inherited; the provider's zero H shift does not replace it.
    Every reaction, including R4/R6/R14, is formed from this species vector.
    No source R14 reaction-specific pressure offset belongs to this branch.
    """
    if tuple(model.components) != ("Fe", "Si", "O", "H"):
        raise ValueError("The hydrogen control requires Fe, Si, O, H model order.")
    base = make_source_standard_potentials_rt(record, case)
    species = tuple(name for names in record["phases"].values() for name in names)
    h2_liquid, h2_gas = (species.index(f"H2_{phase}") for phase in ("silicate", "gas"))
    metal = jnp.asarray([species.index(name + "_metal") for name in model.components])

    def standards(temperature: Any, pressure: Any) -> jax.Array:
        values = base(temperature, pressure)
        h2 = dissolved_h2_standard_rt(values[h2_gas], hirschmann2012_ln_solubility(pressure))
        return values.at[h2_liquid].set(h2).at[metal].add(model.standard_state_shift_RT(temperature))

    return standards


def build_source_control(
    record: dict[str, Any], case: dict[str, Any], element_amounts_mol: Any,
    *, model: Optional[Any] = None, pure_lnphi_func: Optional[Callable] = None,
) -> LocalProblem:
    """Build the formal source-host branch with finite inert He appended.

    Budgets follow the source element order and then He. The gas callback,
    if supplied, returns pure-component ln(phi) in source gas order then He;
    it is called with x=None under a stated ideal-mixture fugacity approximation.
    Exact-zero element support is removed before tracing, including metal H.
    """
    if model is None:
        from exoeos import MaFeSiOHLiquid
        model = MaFeSiOHLiquid()
    standards = make_common_standard_potentials_rt(record, case, model=model)
    extended = copy.deepcopy(record)
    extended["elements"].append("He")
    extended["phases"]["gas"].append("He_gas")
    extended["component_formulas"]["He_gas"] = {"He": 1}
    budget = np.asarray(element_amounts_mol, dtype=float)
    if budget.shape != (len(extended["elements"]),):
        raise ValueError("Supply the source element budgets followed by finite He.")
    full_activity = make_solution_lngamma_func(source_components=model.components, model=model)
    # Static support selection matches local.build_problem, retaining exact zeros.
    supported = lambda name: all(budget[extended["elements"].index(element)] > 0
                                 for element in extended["component_formulas"][name])
    metal_indices = jnp.asarray([i for i, name in enumerate(extended["phases"]["metal"]) if supported(name)])
    gas_indices = jnp.asarray([i for i, name in enumerate(extended["phases"]["gas"]) if supported(name)])

    def metal_activity(temperature, pressure, fractions):
        full_x = jnp.zeros((4,), dtype=fractions.dtype).at[metal_indices].set(fractions)
        return full_activity(temperature, pressure, full_x)[metal_indices]

    def common_standards(temperature, pressure):
        values = standards(temperature, pressure)
        # He occurs in no exchange: zero fixes its elemental reference gauge.
        return jnp.concatenate((values, jnp.zeros((1,), dtype=values.dtype)))

    activities = {"metal": metal_activity}
    if pure_lnphi_func is not None:
        def gas_activity(temperature, pressure, fractions):
            del fractions
            values = jnp.asarray(pure_lnphi_func(temperature, pressure, None))
            if values.shape != (len(extended["phases"]["gas"]),):
                raise ValueError("Pure ln(phi) must follow source gas order then He.")
            return values[gas_indices]
        activities["gas"] = gas_activity
    return build_local_problem(extended, budget, common_standards,
                               phases=("silicate", "metal", "gas"), activity_functions=activities)


def _checkout_provenance(root: Path) -> dict[str, Any]:
    """Record the actual source checkout, including changed tracked content."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    changed = subprocess.run(["git", "diff", "--name-only", "HEAD", "-z"], cwd=root, capture_output=True)
    paths = changed.stdout.decode().split("\0") if changed.returncode == 0 else ()
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              if (root / name).is_file() else None for name in paths if name}
    return {"source_root": str(root), "commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "changed_tracked_file_sha256": hashes}


def run_reference(*, pressure_bar: float = 1.0, hydrogen_scale: float = 1.0) -> dict[str, Any]:
    """Run one conditional 2350 K source-host control and audit physical amounts."""
    import exoeos
    from exoeos import MaFeSiOHLiquid

    if not np.isfinite(hydrogen_scale) or hydrogen_scale < 0:
        raise ValueError("hydrogen_scale must be finite and nonnegative.")
    if not jax.config.x64_enabled:
        raise RuntimeError("Run the numerical reference with JAX_ENABLE_X64=1.")
    record = load_reference()
    fixture = json.loads(Path(__file__).with_name("equilibrium_reference.json").read_text())
    initial = next(item for item in fixture["cases"] if item["id"] == "source_full_2350")
    case = next(item for item in record["cases"] if item["T_K"] == initial["T_K"])
    budget = np.append(initial["element_amounts_mol"], 3.0)
    budget[record["elements"].index("H")] *= hydrogen_scale
    model = MaFeSiOHLiquid()
    problem = build_source_control(record, case, budget, model=model)
    initial_amounts = np.append(initial["component_amounts_mol"], 3.0)
    excluded = np.ones(initial_amounts.size, dtype=bool)
    excluded[problem.species_indices] = False
    initial_amounts[excluded] = 0.0
    result = solve(problem, case["T_K"], pressure_bar, budget,
                   initial_component_amounts_mol=initial_amounts)
    amounts = np.asarray(result.component_amounts_mol)
    active_amounts = amounts[problem.species_indices]
    chemical = np.asarray(problem.standard_potentials_rt(case["T_K"], pressure_bar))[problem.species_indices].copy()
    for phase, section, activity in zip(problem.phases, problem.phase_slices, problem.activity_functions):
        fractions = active_amounts[section] / active_amounts[section].sum()
        chemical[section] += np.log(fractions)
        if activity is not None:
            chemical[section] += np.asarray(activity(case["T_K"], pressure_bar, fractions))
        if phase == "gas":
            chemical[section] += np.log(pressure_bar)
    relative = np.zeros_like(budget)
    totals = np.asarray(problem.full_formula_matrix) @ amounts
    positive = budget > 0
    relative[positive] = totals[positive] / budget[positive] - 1
    reaction = np.asarray(problem.reaction_matrix) @ chemical
    metal_amounts = np.asarray([amounts[problem.full_species.index(name + "_metal")] for name in model.components])
    model.validate_state(case["T_K"], pressure_bar * 1e5, metal_amounts / metal_amounts.sum())
    accepted = (bool(result.root_solution.converged) and np.max(np.abs(relative)) < 1e-9
                and np.max(np.abs(reaction)) < 1e-8 and np.all(totals[~positive] == 0))
    root = Path(__file__).resolve().parents[2]
    files = ("hydrogen.py", "source.py", "reference.json", "local.py")
    return {
        "model_id": MODEL_ID, "accepted": bool(accepted),
        "evidence_level": "conditional_consistent_mechanism_control",
        "domain": {
            "mathematical": "positive T/P, finite nonnegative budgets, positive host and Fe-rich dry alloy",
            "calibration": "No joint calibration or liquid stability evidence; 2350 K extrapolates the older H2 law and source MgO fit.",
            "extrapolation_policy": "formal source-host sensitivity only",
            "stable_phase_evidence": "Not assessed; prescribed liquid/liquid/gas assemblage",
            "exclusions": ["MELTS host", "SiH4/H/OH gas convergence", "phase search", "S/C/N alloy partition", "alloy pressure dependence"],
        },
        "h2_calibration": H2_CALIBRATION,
        "ledger": {
            "elements": list(problem.elements), "components": list(problem.full_species),
            "formula_matrix": np.asarray(problem.full_formula_matrix).tolist(),
            "phases": {name: list(record["phases"][name]) + (["He_gas"] if name == "gas" else []) for name in problem.phases},
            "amount_unit": "mol of listed component; metal H atomic, dissolved/gas H2 molecular",
            "standard_pressure_bar": 1.0, "R_J_mol_K": record["source"]["gas_constant_J_mol_K"],
            "standards": "source Shomate/exchange fits including Okuchi H; native dry shift; pressure-dependent H2 standard; inert He elemental gauge zero",
            "mixing": "ideal source silicate; four-component native alloy; ideal gas",
            "metal_model_id": model.reference_model_id,
        },
        "state": {"T_K": case["T_K"], "P_bar": pressure_bar, "element_amounts_mol": budget.tolist(),
                  "component_amounts_mol": amounts.tolist(), "phase_amounts_mol": np.asarray(result.phase_amounts_mol).tolist(),
                  "phase_element_amounts_mol": np.asarray(result.phase_element_amounts_mol).tolist()},
        "acceptance": {"max_relative_element_residual": float(np.max(np.abs(relative))),
                       "max_abs_reaction_residual_RT": float(np.max(np.abs(reaction))),
                       "exact_zero_budgets_preserved": bool(np.all(totals[~positive] == 0))},
        "provenance": {
            "exogibbs_import_path": exogibbs.__file__, "exoeos_import_path": exoeos.__file__,
            "exogibbs_checkout": _checkout_provenance(root),
            "exoeos_checkout": _checkout_provenance(Path(exoeos.__file__).resolve().parents[2]),
            "file_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in files},
            "jax_version": jax.__version__, "numpy_version": np.__version__, "dtype": str(result.component_amounts_mol.dtype),
            "command": [sys.executable, *sys.argv],
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--hydrogen-scale", type=float, default=1.0)
    args = parser.parse_args()
    output = run_reference(pressure_bar=args.pressure_bar, hydrogen_scale=args.hydrogen_scale)
    print(json.dumps(output, indent=2, allow_nan=False))
    if not output["accepted"]:
        raise SystemExit(1)
