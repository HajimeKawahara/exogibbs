"""Replay TiO energy, thermal and atmospheric-priority evidence offline.

Modern partitions and ground-state energy define diagnostics, not a production
replacement or a new equilibrium solve. All inputs are frozen local artifacts.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from gas_source_comparison import BASELINE, ROOT, file_hash

sys.path.insert(0, str(ROOT / "examples/comparisons"))
import select_major_molecules as major


DATA = ROOT / "tests/data/validation/tio_source_review/20261103"
GRID = ROOT / "tests/data/validation/curated_default_adoption/20261102/ions.npz"
CURATED = ROOT / "src/exogibbs/data/curated/gas_v1.json"
KB = 1.380649e-23
PLANCK = 6.62607015e-34


def curated_ln_k(temperature: float) -> float:
    """Evaluate the actual inherited JANAF quintic plus atomic background."""
    if not math.isfinite(temperature) or not 1500 <= temperature <= 4000:
        raise ValueError("Temperature outside curated TiO domain")
    record = json.loads(CURATED.read_text())["janaf_reactions"]["O1Ti1"]
    knots = record["knots_K"]
    i = min(int(np.searchsorted(knots, temperature, side="right"))-1, len(knots)-2)
    u = (temperature-knots[i])/(knots[i+1]-knots[i])
    free_energy = sum(c*u**power for power, c in enumerate(record["coefficients"][i]))
    background = major.audit.nasa9_g_rt(
        temperature, {"coefficients": record["background_nasa9"], "bounds_K": [1500, 4000]})
    return -free_energy/temperature-background


def internal_partition(temperature: float, family: str, mass_u: float, reference: dict) -> float:
    """Invert molecular G-H(0), separating the source's translational mass."""
    kt = KB*temperature
    log_translation = (1.5*math.log(2*math.pi*mass_u*major.ATOMIC_MASS_KG*kt/PLANCK**2)
                       + math.log(kt/100000))
    return math.exp(-major.thermal_g_rt("TiO", temperature, family, reference)-log_translation)


def footprint() -> dict:
    """Summarize archived gas-only solutions without solving a replacement."""
    with np.load(GRID) as grid:
        molecule = grid["species"].tolist().index("O1Ti1")
        element = grid["elements"].tolist().index("Ti")
        shares = (grid["formula_matrix"][element, molecule]
                  * np.exp(grid["ln_n"][:, molecule])/grid["budget"][:, element])
        i = int(np.argmax(shares))
        return dict(source=str(GRID.relative_to(ROOT)), source_sha256=file_hash(GRID),
                    cases=len(shares), species_count=len(grid["species"]),
                    all_converged=bool(np.all(grid["converged"])),
                    maximum_titanium_fraction=float(shares[i]), case_index=i,
                    temperature_K=float(grid["temperature_K"][i]),
                    pressure_bar=float(grid["pressure_bar"][i]), C_over_O=float(grid["C_over_O"][i]),
                    mole_fraction=float(np.exp(grid["ln_x"][i, molecule])),
                    max_current_potential_residual=max(abs(curated_ln_k(t)+h) for t, h in
                        zip(grid["temperature_K"], grid["h"][:, molecule])),
                    interpretation="Finite gas-only footprint; condensation, rainout and cold trapping absent.")


def review() -> dict:
    """Separate energy and thermal effects with a common neutral-atom gauge."""
    reference = json.loads(major.REFERENCE_PATH.read_text())
    facts = json.loads((DATA / "literature.json").read_text())
    exomol = json.loads((DATA / "exomol_nodes.json").read_text())
    baseline = json.loads((BASELINE / "comparison.json").read_text())
    old_d0 = reference["barklem_collet"]["TiO"]["dissociation"]["dissociation_eV"]
    new_d0 = facts["jain2023"]["D0_TiO_eV"]
    energies = {f: major.atomization_kelvin("TiO", f, reference) for f in ("janaf", "nasa9")}
    m_janaf = facts["janaf"]["mass_u"]
    m_nasa = reference["nasa9_records"]["TiO"]["mass_u"]
    main = exomol["isotopologues"]["48Ti-16O"]
    rows = []
    for t in exomol["temperatures_K"]:
        native = {f: major.formation_log_kp("TiO", t, f, energies[f], reference, f)
                  for f in energies}
        selected = major.formation_log_kp("TiO", t, "janaf", old_d0*major.EV_K, reference)
        q_j = internal_partition(t, "janaf", m_janaf, reference)
        q_n = internal_partition(t, "nasa9", m_nasa, reference)
        normalized = {name: item["partition"][str(t)]/item["nuclear_spin_weight"]
                      for name, item in exomol["isotopologues"].items()}
        q_new = normalized["48Ti-16O"]
        thermal = math.log(q_new/q_j)
        energy = (new_d0-old_d0)*major.EV_K/t
        # Fixed JANAF molecular translation and common NASA atoms isolate Q and D0.
        # This does not turn a monoisotope partition into a natural-isotope mixture.
        current = curated_ln_k(t) if 1500 <= t <= 4000 else None
        old = next((r for r in baseline["samples"]
                    if r["species_id"] == "O1Ti1" and r["temperature_K"] == t), None)
        q_states = exomol["current_states_audit"]["direct_state_sums"][str(t)]
        rows.append(dict(temperature_K=t, nasa_ln_K=native["nasa9"],
                         janaf_native_ln_K=native["janaf"], selected_exact_node_ln_K=selected,
                         curated_ln_K=current,
                         native_janaf_over_nasa=math.exp(native["janaf"]-native["nasa9"]),
                         curated_over_nasa=None if current is None else math.exp(current-native["nasa9"]),
                         selected_minus_nasa_energy_ln_K=(old_d0*major.EV_K-energies["nasa9"])/t,
                         selected_minus_nasa_molecular_thermal_ln_K=
                         major.thermal_g_rt("TiO", t, "nasa9", reference)
                         -major.thermal_g_rt("TiO", t, "janaf", reference),
                         janaf_to_common_atoms_ln_K=sum(
                             major.thermal_g_rt(a, t, "nasa9", reference)
                             -major.thermal_g_rt(a, t, "janaf", reference) for a in ("Ti", "O")),
                         inferred_Q_janaf=q_j, inferred_Q_nasa=q_n,
                         normalized_exomol_Q=normalized,
                         exomol_over_janaf_Q=q_new/q_j, exomol_over_nasa_Q=q_new/q_n,
                         current_states_over_distributed_pf=q_states/q_new,
                         new_energy_only_K_ratio=math.exp(energy),
                         fixed_mass_Q_only_K_ratio=math.exp(thermal),
                         fixed_mass_combined_K_ratio=math.exp(energy+thermal),
                         fixed_mass_diagnostic_over_nasa=math.exp(selected+energy+thermal-native["nasa9"]),
                         new_D0_uncertainty_K_factor=math.exp(facts["jain2023"]["uncertainty_eV"]*major.EV_K/t),
                         baseline_curated_residual=None if old is None or current is None else
                         current-old["ln_Kp"]["curated"]))
    return dict(reaction=facts["identity"]["reaction"], standard_pressure_Pa=100000,
                balanced_native_D0_eV={f:v/major.EV_K for f,v in energies.items()},
                current_D0_eV=old_d0, modern_D0_eV=new_d0,
                mass_u=dict(janaf=m_janaf, nasa=m_nasa, exomol_48=main["mass_u"]),
                molecular_only_translation_factor_48_over_janaf=(main["mass_u"]/m_janaf)**1.5,
                normalization="Q excludes nuclear spin; molecular source H(0) zero. Fixed-mass diagnostic keeps JANAF translation and NASA atomic thermal functions; not a natural-isotope replacement.",
                samples=rows, footprint=footprint(), curated_update="none",
                equilibrium_substitution="not_run", physical_accuracy="unknown")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(review(), indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
