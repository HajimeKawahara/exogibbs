"""FastChem parity helpers for diagnostics and comparison scripts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from exogibbs.io.load_data import get_data_filepath


BOLTZMANN_CGS = 1.380649e-16
FASTCHEM_ASPLUND_2020 = "fastchem/element_abundances/asplund_2020.dat"


@dataclass(frozen=True)
class AlignedAbundanceVector:
    """Element abundance vector aligned to an ExoGibbs element order."""

    elements: tuple[str, ...]
    vector: np.ndarray
    source: str
    source_path: str | None
    normalized: bool
    dropped_elements: tuple[str, ...]
    reordered_elements: tuple[str, ...]
    metadata: dict[str, Any]


def normalize_species_name(name: str) -> str:
    """Normalize simple FastChem/ExoGibbs aliases for diagnostic matching."""

    raw = str(name).strip()
    phase = ""
    match = re.search(r"(\([^)]*\))$", raw)
    if match:
        phase = match.group(1).replace(" ", "")
        raw = raw[: match.start()]
    if raw == "e1-":
        raw = "e-"
    elif re.fullmatch(r"[A-Z][a-z]?1", raw):
        raw = raw[:-1]
    return raw + phase


def parse_fastchem_abundance_file(path: str | Path) -> dict[str, float]:
    """Read a FastChem abundance file in A(X)=log10(n_X/n_H)+12 format."""

    resolved = Path(path)
    values: dict[str, float] = {}
    for raw_line in resolved.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            values[parts[0]] = float(parts[1])
    return values


def abundance_vector_from_log_abundances(
    log_abundances: Mapping[str, float],
    element_order: Sequence[str],
    *,
    normalize: bool = False,
) -> np.ndarray:
    """Convert FastChem log abundances to a vector in ``element_order``."""

    values = []
    for element in element_order:
        if element == "e-":
            values.append(0.0)
        else:
            values.append(10.0 ** (float(log_abundances[element]) - 12.0))
    out = np.asarray(values, dtype=np.float64)
    if normalize:
        total = float(np.sum(out))
        if total > 0.0:
            out = out / total
    return out


def align_named_vectors(
    left_names: Sequence[str],
    left_values: Sequence[float],
    right_names: Sequence[str],
    right_values: Sequence[float],
) -> dict[str, Any]:
    """Return row-wise parity diagnostics for two named vectors."""

    left = {str(k): float(v) for k, v in zip(left_names, left_values)}
    right = {str(k): float(v) for k, v in zip(right_names, right_values)}
    common = sorted(set(left).intersection(right))
    left_only = sorted(set(left) - set(right))
    right_only = sorted(set(right) - set(left))
    diffs = []
    for name in common:
        lv = left[name]
        rv = right[name]
        abs_diff = abs(lv - rv)
        rel_diff = abs_diff / max(abs(rv), 1.0e-300)
        diffs.append(
            {
                "name": name,
                "left": lv,
                "right": rv,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
            }
        )
    return {
        "common_names": common,
        "left_only": left_only,
        "right_only": right_only,
        "rows": diffs,
        "max_abs_diff": max([row["abs_diff"] for row in diffs], default=0.0),
        "max_rel_diff": max([row["rel_diff"] for row in diffs], default=0.0),
        "exactly_identical_after_alignment": bool(
            not left_only and not right_only and all(row["abs_diff"] == 0.0 for row in diffs)
        ),
    }


def build_aligned_abundance_vector(
    element_order: Sequence[str],
    *,
    source: str = "fastchem_asplund_2020",
    normalize: bool = True,
    element_file: str | Path | None = None,
    exogibbs_reference: Sequence[float] | None = None,
) -> AlignedAbundanceVector:
    """Build an explicit abundance vector for parity workflows.

    This helper is intentionally outside the solvers. It lets audits and
    comparison scripts opt into a FastChem file-backed vector without changing
    legacy ``chemsetup()`` defaults.
    """

    elements = tuple(str(e) for e in element_order)
    if source in ("fastchem_asplund_2020", "fastchem_file"):
        if element_file is None:
            path = Path(get_data_filepath(FASTCHEM_ASPLUND_2020))
        else:
            candidate = Path(element_file)
            path = candidate if candidate.is_absolute() or candidate.exists() else Path(get_data_filepath(str(candidate)))
        abundances = parse_fastchem_abundance_file(path)
        vector = abundance_vector_from_log_abundances(abundances, elements, normalize=normalize)
        file_elements = tuple(abundances)
        dropped = tuple(e for e in file_elements if e not in elements)
        reordered = tuple(e for e in elements if e in abundances or e == "e-")
        metadata = {
            "abundance_source": source,
            "format": "FastChem A(X)=log10(n_X/n_H)+12",
            "electron_handling": "electron abundance is set to 0.0 in ExoGibbs element vectors",
            "normalization": "sum-normalized for ExoGibbs solver state" if normalize else "hydrogen-referenced n_H=1",
        }
        return AlignedAbundanceVector(
            elements=elements,
            vector=vector,
            source=source,
            source_path=str(path),
            normalized=normalize,
            dropped_elements=dropped,
            reordered_elements=reordered,
            metadata=metadata,
        )
    if source == "exogibbs_reference":
        if exogibbs_reference is None:
            raise ValueError("exogibbs_reference is required for source='exogibbs_reference'.")
        vector = np.asarray(exogibbs_reference, dtype=np.float64)
        if vector.shape != (len(elements),):
            raise ValueError("exogibbs_reference length must match element_order.")
        return AlignedAbundanceVector(
            elements=elements,
            vector=vector,
            source=source,
            source_path=None,
            normalized=False,
            dropped_elements=(),
            reordered_elements=elements,
            metadata={
                "abundance_source": source,
                "format": "caller-provided ExoGibbs reference vector",
                "normalization": "unchanged caller vector",
            },
        )
    raise ValueError(f"unknown abundance source: {source}")


def fastchem_gas_mass_action_lnK(raw_lnK: float, stoich: Mapping[str, int], temperature: float) -> float:
    """Convert raw gas lnK to FastChem's pressure-scaled mass-action convention."""

    sigma = 1.0 - float(sum(stoich.values()))
    pressure_scaling = 1.0e-6 * BOLTZMANN_CGS * temperature
    return float(raw_lnK) - sigma * math.log(pressure_scaling)


def fastchem_condensate_activity_lnK(raw_lnK: float, stoich: Mapping[str, int], temperature: float) -> float:
    """Convert raw condensate lnK to FastChem's monatomic-reference convention."""

    sigma = float(sum(stoich.values()))
    pressure_scaling = 1.0e6 / (BOLTZMANN_CGS * temperature)
    return float(raw_lnK) - sigma * math.log(pressure_scaling)


def reference_bridge_explanation(kind: str) -> dict[str, Any]:
    """Machine-readable reference-convention explanation for parity reports."""

    if kind == "gas":
        return {
            "reference_state": "FastChem gas mass-action constant with cgs pressure scaling",
            "sign_convention": "ExoGibbs h_g = -raw_lnK; compare FastChem convention after converting -h_g",
            "offset_structure": "raw_lnK - (1 - sum_nu) * ln(1e-6 * k_B * T)",
            "production_scope": "comparison-space conversion only; production hvector is unchanged",
        }
    if kind == "condensate":
        return {
            "reference_state": "FastChem Cond monatomic-reference condensate activity constant",
            "sign_convention": "ExoGibbs h_c = -raw_condensate_lnK; compare FastChem convention after converting -h_c",
            "offset_structure": "raw_lnK - sum_nu * ln(1e6 / (k_B * T))",
            "production_scope": "comparison-space conversion only; production condensate hvector is unchanged",
        }
    raise ValueError(f"unknown reference bridge kind: {kind}")
