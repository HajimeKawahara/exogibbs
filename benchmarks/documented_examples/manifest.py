"""Manifest for the documented condensate example benchmark suite."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DocumentedExampleCase:
    """One documentation entry and its measured ExoGibbs workload."""

    case_id: str
    document: str
    description: str
    source_scripts: tuple[str, ...]
    full_output_layer_count: int
    output_rows_per_condition: int
    expected_phases: tuple[tuple[str, str], ...]
    workload: str
    input_artifacts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible manifest record."""

        return asdict(self)

    def expected_output_layer_count(self, smoke_layers: int | None) -> int:
        """Return the output-row contract for a full or smoke workload."""

        if smoke_layers is None:
            return self.full_output_layer_count
        condition_count = (
            self.full_output_layer_count // self.output_rows_per_condition
        )
        return self.output_rows_per_condition * min(
            smoke_layers, condition_count
        )


CASES = (
    DocumentedExampleCase(
        case_id="visscher_2006_na2s_morley_2012_kcl",
        document="visscher_2006_na2s_morley_2012_kcl",
        description="One-bar KCl and Na2S condensation scans",
        source_scripts=(
            "examples/comparisons/"
            "comparison_with_visscher_2006_na2s_morley_2012_kcl.py",
        ),
        full_output_layer_count=174,
        output_rows_per_condition=2,
        expected_phases=(
            ("build_reduced_setups", "setup"),
            ("solve_kcl", "solver"),
            ("solve_na2s", "solver"),
        ),
        workload="run_visscher_2006",
    ),
    DocumentedExampleCase(
        case_id="visscher_2010_forsterite_enstatite_competition",
        document="visscher_2010_forsterite_enstatite_competition",
        description="Forsterite-enstatite-quartz competition scans",
        source_scripts=(
            "examples/comparisons/"
            "comparison_with_visscher_2010_forsterite_enstatite.py",
        ),
        full_output_layer_count=254,
        output_rows_per_condition=2,
        expected_phases=(
            ("build_reduced_setups", "setup"),
            ("solve_with_enstatite", "solver"),
            ("solve_without_enstatite", "solver"),
        ),
        workload="run_visscher_2010",
    ),
    DocumentedExampleCase(
        case_id="ito_2025_rainout_comparison",
        document="ito_2025_rainout_comparison",
        description="Ito 2025 propagated H/O/Si rainout profile",
        source_scripts=(
            "examples/comparisons/comparison_with_ito_2025_rainout.py",
            "examples/comparisons/comparison_with_ito_2025.py",
        ),
        full_output_layer_count=855,
        output_rows_per_condition=1,
        expected_phases=(
            ("load_ito_profile", "setup"),
            ("solve_propagated_rainout", "solver"),
        ),
        workload="run_ito_2025_rainout",
        input_artifacts=("external_data/Ito_2025.xlsx",),
    ),
    DocumentedExampleCase(
        case_id="fe_fes_rainout_demo",
        document="fe_fes_rainout_demo",
        description="Reduced Fe-FeS local-equilibrium and rainout profiles",
        source_scripts=(
            "examples/comparisons/demo_fe_fes_rainout.py",
        ),
        full_output_layer_count=34,
        output_rows_per_condition=2,
        expected_phases=(
            ("build_reduced_setup", "setup"),
            ("solve_local", "solver"),
            ("solve_rainout", "solver"),
        ),
        workload="run_fe_fes_rainout",
    ),
    DocumentedExampleCase(
        case_id="comparison_example_lineage",
        document="comparison_example_lineage",
        description="Full-catalog L-dwarf condensate and gas profiles",
        source_scripts=(
            "examples/comparisons/comparison_with_fastchem4_condensates.py",
        ),
        full_output_layer_count=26,
        output_rows_per_condition=2,
        expected_phases=(
            ("build_full_catalog_setup", "setup"),
            ("solve_condensate_profile", "solver"),
            ("solve_gas_only_profile", "solver"),
        ),
        workload="run_fastchem4_l_dwarf",
    ),
    DocumentedExampleCase(
        case_id="rocky_raccoon_trace_mg",
        document="rocky_raccoon_trace_mg",
        description="Positive trace-Mg Rocky Raccoon-like rainout layer",
        source_scripts=(
            "examples/comparisons/demo_rocky_raccoon_trace_mg.py",
        ),
        full_output_layer_count=1,
        output_rows_per_condition=1,
        expected_phases=(
            ("build_rocky_raccoon_trace_mg_setup", "setup"),
            ("solve_rocky_raccoon_trace_mg", "solver"),
        ),
        workload="run_rocky_raccoon_trace_mg",
    ),
)

CASES_BY_ID = {case.case_id: case for case in CASES}


def get_case(case_id: str) -> DocumentedExampleCase:
    """Return one registered benchmark case."""

    try:
        return CASES_BY_ID[case_id]
    except KeyError as error:
        raise KeyError(f"Unknown documented example case: {case_id!r}") from error


def example_documents_from_index(path: Path) -> tuple[str, ...]:
    """Return entries from the EXAMPLES toctree in ``documents/index.rst``."""

    lines = path.read_text().splitlines()
    try:
        caption_index = next(
            index
            for index, line in enumerate(lines)
            if line.strip() == ":caption: EXAMPLES:"
        )
    except StopIteration as error:
        raise ValueError("documents/index.rst has no EXAMPLES toctree.") from error

    entries = []
    for line in lines[caption_index + 1 :]:
        stripped = line.strip()
        if stripped.startswith(".. toctree::"):
            break
        if not stripped or stripped.startswith(":") or stripped.startswith(".."):
            continue
        entries.append(stripped.removesuffix(".rst"))
    return tuple(entries)


__all__ = (
    "CASES",
    "CASES_BY_ID",
    "DocumentedExampleCase",
    "example_documents_from_index",
    "get_case",
)
