"""Enforce feature ownership boundaries below the public API."""

from __future__ import annotations

import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "exogibbs"


def _imports_under(
    directory: Path,
    forbidden_prefix: str,
) -> set[tuple[str, str]]:
    imports = set()
    for path in sorted(directory.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == forbidden_prefix or module.startswith(
                    f"{forbidden_prefix}."
                ):
                    imports.add((str(path.relative_to(SOURCE_ROOT)), module))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == forbidden_prefix or alias.name.startswith(
                        f"{forbidden_prefix}."
                    ):
                        imports.add(
                            (str(path.relative_to(SOURCE_ROOT)), alias.name)
                        )
    return imports


def test_internal_packages_do_not_import_public_api_modules() -> None:
    observed = set()
    for package in (
        "condensates",
        "equilibrium",
        "optimize",
        "presets",
        "solubility",
        "thermo",
    ):
        observed |= _imports_under(SOURCE_ROOT / package, "exogibbs.api")

    assert observed == set()


def test_stable_packages_do_not_import_experimental_modules() -> None:
    observed = set()
    for package in (
        "api",
        "condensates",
        "equilibrium",
        "io",
        "math",
        "optimize",
        "presets",
        "solubility",
        "thermo",
        "utils",
    ):
        observed |= _imports_under(
            SOURCE_ROOT / package,
            "exogibbs.experimental",
        )

    assert observed == set()


def test_equilibrium_features_do_not_import_optimize_compatibility_paths() -> None:
    observed = _imports_under(
        SOURCE_ROOT / "equilibrium",
        "exogibbs.optimize",
    )

    assert observed == set()


def test_fixed_support_kernel_does_not_own_support_lifecycle() -> None:
    fixed_support = (
        SOURCE_ROOT / "equilibrium" / "condensate" / "fixed_support"
    )
    observed = set()
    for forbidden in (
        "exogibbs.api",
        "exogibbs.equilibrium.condensate.lifecycle",
        "exogibbs.equilibrium.condensate.support",
    ):
        observed |= _imports_under(fixed_support, forbidden)

    assert observed == set()


def test_result_construction_does_not_run_acceptance_logic() -> None:
    results_module = (
        SOURCE_ROOT / "equilibrium" / "condensate" / "results.py"
    )
    observed = _imports_under(
        results_module.parent,
        "exogibbs.equilibrium.condensate.acceptance",
    )
    observed = {
        item for item in observed if item[0] == "equilibrium/condensate/results.py"
    }

    assert observed == set()
