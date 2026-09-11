"""Finite sulfur partition and empirical sulfide appearance
=============================================================

This example has a fixed component/reaction set. The runnable ideal control
tests numerical closure only; its SCSS is synthetic, not a calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable, Optional, Sequence

import numpy as np
import scipy
from scipy.optimize import least_squares


ELEMENTS = ("Mg", "Si", "Fe", "O", "H", "S")
SPECIES = (
    "H2_gas", "H2O_gas", "H2S_gas",
    "MgO_melt", "SiO2_melt", "FeO_melt", "H2_melt", "H2O_melt",
    "FeS_melt", "FeSO4_melt", "Fe_metal", "H_metal", "S_metal",
    "FeS_sulfide",
)
PHASE_SLICES = (slice(0, 3), slice(3, 10), slice(10, 13), slice(13, 14))
FORMULA = np.asarray([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
    [0, 1, 0, 1, 2, 1, 0, 1, 0, 4, 0, 0, 0, 0],
    [2, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
], dtype=np.float64)
MOLAR_MASS_G = np.asarray([24.305, 28.085, 55.845, 15.999, 1.008, 32.06]) @ FORMULA
REACTION_IDS = (
    "FeO+H2=Fe+H2O", "H2_gas=H2_melt", "H2O_gas=H2O_melt",
    "H2_gas=2H_metal", "H2S+FeO=FeS+H2O", "FeS_melt=Fe_metal+S_metal",
    "FeS+4H2O=FeSO4+4H2",
)
REACTIONS = np.zeros((7, len(SPECIES)))
for _row, _terms in enumerate((
    {5: -1, 0: -1, 10: 1, 1: 1}, {0: -1, 6: 1}, {1: -1, 7: 1},
    {0: -1, 11: 2}, {2: -1, 5: -1, 8: 1, 1: 1},
    {8: -1, 10: 1, 12: 1}, {8: -1, 1: -4, 9: 1, 0: 4},
)):
    for _column, _coefficient in _terms.items():
        REACTIONS[_row, _column] = _coefficient


@dataclass(frozen=True)
class SCSS:
    """S mass ppm on a declared melt denominator and oxidation-state basis.

    ``dry`` excludes dissolved H2 and H2O components from melt mass;
    ``hydrous`` includes both. Both retain the mass of FeS/FeSO4 solutes.
    The phase identity is pure FeS; the supplied calibration must justify
    its declared physical state. No FeS chemical-potential equality is added.
    """

    ppm_s: float
    sulfur_state: str
    mass_basis: str
    calibration_id: str
    phase_state: str

    def __post_init__(self):
        if not np.isfinite(self.ppm_s) or self.ppm_s <= 0:
            raise ValueError("SCSS ppm_s must be finite and positive.")
        if self.sulfur_state not in ("sulfide", "total"):
            raise ValueError("SCSS sulfur_state must be 'sulfide' or 'total'.")
        if self.mass_basis not in ("dry", "hydrous"):
            raise ValueError("SCSS mass_basis must be 'dry' or 'hydrous'.")
        if not self.calibration_id or self.phase_state not in ("solid", "liquid", "conditional"):
            raise ValueError("Declare calibration_id and a solid/liquid/conditional FeS state.")


PartitionFunction = Callable[[float, float, np.ndarray], np.ndarray]
SaturationFunction = Callable[[float, float, np.ndarray], SCSS]


@dataclass(frozen=True)
class SulfideResult:
    branch: str
    component_amounts_mol: np.ndarray
    element_residual: np.ndarray
    reaction_residual: np.ndarray
    saturation_log_ratio: float
    scss: SCSS
    accepted: bool
    solver_success: bool


def melt_sulfur_ppm(amounts: np.ndarray, *, sulfur_state: str, mass_basis: str) -> float:
    """Convert the actual FeS/FeSO4 melt amounts to the SCSS concentration."""
    amounts = np.asarray(amounts, dtype=np.float64)
    if amounts.shape != (len(SPECIES),) or not np.all(np.isfinite(amounts)):
        raise ValueError("Amounts must be finite and use SPECIES order.")
    if np.any(amounts[3:10] < 0):
        raise ValueError("Melt amounts must be nonnegative.")
    if sulfur_state not in ("sulfide", "total") or mass_basis not in ("dry", "hydrous"):
        raise ValueError("Declare a sulfide/total sulfur state and dry/hydrous mass basis.")
    indices = [3, 4, 5, 8, 9] if mass_basis == "dry" else list(range(3, 10))
    denominator = MOLAR_MASS_G[indices] @ amounts[indices]
    if denominator <= 0:
        raise ValueError("SCSS comparison requires a positive melt mass.")
    sulfur_mol = amounts[8] + (amounts[9] if sulfur_state == "total" else 0.0)
    return float(1e6 * 32.06 * sulfur_mol / denominator)


def scss_total_ppm(scss: SCSS, sulfate_fraction: float) -> float:
    """Convert sulfide-state SCSS to total S on the SAME mass denominator.

    The melt sulfur set contains only -II (FeS) and +VI (FeSO4).
    A sulfide-only value has no finite total-S conversion at 100% sulfate.
    """
    if not np.isfinite(sulfate_fraction) or not 0 <= sulfate_fraction <= 1:
        raise ValueError("sulfate_fraction must lie between zero and one.")
    if scss.sulfur_state == "total":
        return scss.ppm_s
    if sulfate_fraction == 1:
        raise ValueError("Sulfide-state SCSS requires a nonzero sulfide fraction.")
    return scss.ppm_s / (1 - sulfate_fraction)


def _support(budgets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if budgets.shape != (len(ELEMENTS),) or not np.all(np.isfinite(budgets)):
        raise ValueError("Budgets must be finite and use ELEMENTS order.")
    if np.any(budgets[:5] <= 0) or budgets[5] < 0:
        raise ValueError("This fixed host requires positive Mg/Si/Fe/O/H and nonnegative S.")
    supported = np.all(FORMULA[budgets == 0] == 0, axis=0)
    supported[-1] = False
    reactions = np.all(REACTIONS[:, ~supported] == 0, axis=1)
    return np.flatnonzero(supported), np.flatnonzero(budgets > 0), reactions


def evaluate(
    amounts: np.ndarray, budgets: np.ndarray, temperature_k: float, pressure_bar: float,
    partition_residual: PartitionFunction, saturation: SaturationFunction, *, branch: str,
) -> SulfideResult:
    """Recompute balances, unsaturated exchange, and saturation at returned amounts.

    ``partition_residual`` supplies the seven dimensionless residuals in
    REACTION_IDS order; only the four nonsulfur reactions survive at S=0.
    It must include gas/melt/metal S exchange, not substitute SCSS for it.
    """
    if branch not in ("absent", "present"):
        raise ValueError("branch must be 'absent' or 'present'.")
    if not np.all(np.isfinite([temperature_k, pressure_bar])) or min(temperature_k, pressure_bar) <= 0:
        raise ValueError("Temperature in K and pressure in bar must be finite and positive.")
    amounts = np.asarray(amounts, dtype=np.float64)
    budgets = np.asarray(budgets, dtype=np.float64)
    indices, elements, selected = _support(budgets)
    if amounts.shape != (len(SPECIES),) or not np.all(np.isfinite(amounts)):
        raise ValueError("Amounts must be finite and use SPECIES order.")
    reactions = np.asarray(partition_residual(temperature_k, pressure_bar, amounts))
    if reactions.shape != (len(REACTION_IDS),) or not np.all(np.isfinite(reactions[selected])):
        raise ValueError("Partition callback must return seven residuals, finite on active reactions.")
    scss = saturation(temperature_k, pressure_bar, amounts)
    concentration = melt_sulfur_ppm(amounts, sulfur_state=scss.sulfur_state, mass_basis=scss.mass_basis)
    ratio = float(np.log(concentration / scss.ppm_s)) if concentration > 0 else -np.inf
    balance = FORMULA @ amounts - budgets
    balance[elements] /= budgets[elements]
    zero_support = np.all(FORMULA[budgets == 0] == 0, axis=0)
    valid = np.all(amounts >= 0) and np.all(amounts[~zero_support] == 0)
    valid = valid and np.all(amounts[indices] > 0)
    phase_ok = amounts[-1] == 0 and ratio <= 1e-8 if branch == "absent" else amounts[-1] > 0 and abs(ratio) < 1e-8
    accepted = valid and phase_ok and np.max(np.abs(balance)) < 1e-9 and np.max(np.abs(reactions[selected])) < 1e-8
    return SulfideResult(branch, amounts.copy(), balance, reactions[selected], ratio, scss, bool(accepted), False)


def solve_branches(
    temperature_k: float, pressure_bar: float, element_amounts_mol: Sequence[float],
    partition_residual: PartitionFunction, saturation: SaturationFunction, *,
    initial_amounts: Optional[Sequence[np.ndarray]] = None,
) -> tuple[SulfideResult, ...]:
    """Try both explicit branches from every start, retaining all raw results.

    Active solution amounts use logarithms. FeS amount is an unconstrained
    linear variable in the present solve, so a negative root is rejected
    instead of being disguised by a positive floor. Absent FeS is exactly
    zero. Acceptance uses a fresh callback evaluation, not optimizer status;
    no Gibbs-energy ordering or physical hysteresis is inferred.
    """
    if not np.all(np.isfinite([temperature_k, pressure_bar])) or min(temperature_k, pressure_bar) <= 0:
        raise ValueError("Temperature in K and pressure in bar must be finite and positive.")
    budgets = np.asarray(element_amounts_mol, dtype=np.float64)
    indices, elements, selected = _support(budgets)
    scale = float(budgets.sum())
    default = np.zeros(len(SPECIES))
    limits = np.where(FORMULA[:, indices] > 0, budgets[:, None] / np.maximum(FORMULA[:, indices], 1), np.inf)
    default[indices] = limits.min(axis=0) / len(indices)
    starts = (default,) if initial_amounts is None else tuple(initial_amounts)
    if not starts:
        raise ValueError("Supply at least one initial composition.")
    results = []
    for start in starts:
        start = np.asarray(start, dtype=np.float64)
        if start.shape != default.shape or not np.all(np.isfinite(start)) or np.any(start < 0):
            raise ValueError("Initial amounts must be finite, nonnegative, and use SPECIES order.")
        # Continuation through S=0 needs a fresh positive initial guess for
        # newly active components; this only seeds the solve, not the result.
        positive_start = np.where(start[indices] > 0, start[indices], default[indices])
        for branch in (("absent", "present") if budgets[-1] > 0 else ("absent",)):
            present = branch == "present"

            def unpack(root):
                amounts = np.zeros(len(SPECIES))
                amounts[indices] = scale * np.exp(root[:len(indices)])
                amounts[-1] = scale * root[-1] if present else 0.0
                return amounts

            def residual(root):
                state = evaluate(unpack(root), budgets, temperature_k, pressure_bar, partition_residual, saturation, branch=branch)
                values = np.concatenate((state.element_residual[elements], state.reaction_residual))
                return np.append(values, state.saturation_log_ratio) if present else values

            guess = np.log(positive_start / scale)
            lower = np.full(len(indices), -650.0)
            upper = np.full(len(indices), 10.0)
            if np.any(guess < lower) or np.any(guess > upper):
                raise ValueError("Initial log(amount / total element amount) must lie in [-650, 10].")
            if present:
                guess = np.append(guess, start[-1] / scale)
                lower = np.append(lower, -np.inf)
                upper = np.append(upper, np.inf)
            root = least_squares(residual, guess, bounds=(lower, upper), xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=1000)
            state = evaluate(unpack(root.x), budgets, temperature_k, pressure_bar, partition_residual, saturation, branch=branch)
            results.append(SulfideResult(**{**state.__dict__, "solver_success": bool(root.success)}))
    return tuple(results)


def continue_sulfur(
    total_sulfur_mol: Sequence[float], budgets: Sequence[float], temperature_k: float,
    pressure_bar: float, partition_residual: PartitionFunction, saturation: SaturationFunction,
    *, initial_amounts: Optional[Sequence[np.ndarray]] = None,
) -> tuple[tuple[SulfideResult, ...], ...]:
    """Follow the supplied S order, retaining all starts/roots at every point.

    Call with the reversed S sequence for a backward comparison. Accepted
    states seed the next point in addition to the original starts. No
    acceptable state is selected by optimizer cost or continuation order.
    """
    base = np.asarray(budgets, dtype=np.float64).copy()
    _support(base)
    original = tuple(initial_amounts or ())
    seeds = original
    sequence = []
    for sulfur in total_sulfur_mol:
        base[-1] = sulfur
        states = solve_branches(temperature_k, pressure_bar, base, partition_residual, saturation, initial_amounts=seeds or None)
        sequence.append(states)
        accepted = [state.component_amounts_mol for state in states if state.accepted]
        unique = []
        for amounts in accepted:
            if not any(np.allclose(amounts, prior, rtol=1e-7, atol=1e-12 * base.sum()) for prior in unique):
                unique.append(amounts)
        seeds = original + tuple(unique)
    return tuple(sequence)


def analytic_control() -> tuple[np.ndarray, np.ndarray, PartitionFunction, SaturationFunction]:
    """Return a manufactured Mg-bearing control with a known saturated root.

    Ideal standards are reconstructed from one positive composition at
    1873 K/1 bar. This is a conditional numerical mechanism, with no claim
    of experimental partition calibration or liquid-phase stability.
    """
    known = np.asarray([0.8, 0.2, 0.03, 2.0, 1.0, 0.5, 0.1, 0.2, 0.04, 0.01, 1.0, 0.1, 0.02, 0.15])
    standard = np.zeros(len(SPECIES))
    for section in PHASE_SLICES[:3]:
        standard[section] = -np.log(known[section] / known[section].sum())

    def partition(temperature, pressure, amounts):
        del temperature
        potentials = standard.copy()
        for section in PHASE_SLICES[:3]:
            x = amounts[section] / amounts[section].sum()
            potentials[section] += np.log(np.where(x > 0, x, 1.0))
        potentials[:3] += np.log(pressure / 1.0)
        return REACTIONS @ potentials

    threshold = melt_sulfur_ppm(known, sulfur_state="sulfide", mass_basis="dry")

    def saturation(temperature, pressure, amounts):
        del temperature, pressure, amounts
        return SCSS(threshold, "sulfide", "dry", "analytic_scss_control_v1", "conditional")

    return FORMULA @ known, known, partition, saturation


if __name__ == "__main__":
    _budgets, _known, _partition, _saturation = analytic_control()
    _states = solve_branches(1873.0, 1.0, _budgets, _partition, _saturation, initial_amounts=(_known * 0.7, _known * 1.3))
    print(json.dumps({
        "model_id": "finite_sulfur_empirical_scss_analytic_control_v1",
        "evidence": "conditional numerical control; synthetic SCSS; no phase-stability calibration",
        "T_K": 1873.0, "P_bar": 1.0,
        "elements": ELEMENTS, "element_amounts_mol": _budgets.tolist(),
        "species": SPECIES, "formula_matrix_element_rows": FORMULA.tolist(),
        "amount_basis": "mol of named component; atomic H/S in metal, molecular H2/H2O in gas/melt",
        "standard_convention": "synthetic common mu/(RT) standards reconstructed at the known root; ideal mixing; gas reference 1 bar",
        "domain": "positive Mg/Si/Fe/O/H, nonnegative S; conditional phase assemblage; no calibrated T/P or composition range",
        "scss": _saturation(1873.0, 1.0, _known).__dict__,
        "provenance": {
            "exogibbs_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "dtype": str(_known.dtype), "command": [sys.executable, *sys.argv],
        },
        "states": [{
            "branch": state.branch, "accepted": state.accepted,
            "amounts_mol": state.component_amounts_mol.tolist(),
            "max_relative_element_residual": float(np.max(np.abs(state.element_residual))),
            "max_reaction_residual": float(np.max(np.abs(state.reaction_residual))),
            "saturation_log_ratio": state.saturation_log_ratio,
        } for state in _states],
    }, indent=2, allow_nan=False))
    if not any(state.accepted for state in _states):
        raise SystemExit(1)
