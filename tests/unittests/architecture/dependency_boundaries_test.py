"""Enforce feature ownership boundaries below the public API."""

from __future__ import annotations

import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "exogibbs"


def _import_from_candidates(
    path: Path,
    node: ast.ImportFrom,
) -> set[str]:
    if node.level:
        package_parts = [
            "exogibbs",
            *path.relative_to(SOURCE_ROOT).parent.parts,
        ]
        keep = len(package_parts) - (node.level - 1)
        module_parts = package_parts[:keep]
        if node.module:
            module_parts.extend(node.module.split("."))
        module = ".".join(module_parts)
    else:
        module = node.module or ""

    candidates = {module}
    candidates.update(
        f"{module}.{alias.name}" if module else alias.name
        for alias in node.names
        if alias.name != "*"
    )
    return candidates


def _imports_under(
    directory: Path,
    forbidden_prefix: str,
) -> set[tuple[str, str]]:
    imports = set()
    for path in sorted(directory.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for candidate in _import_from_candidates(path, node):
                    if candidate == forbidden_prefix or candidate.startswith(
                        f"{forbidden_prefix}."
                    ):
                        imports.add(
                            (
                                str(path.relative_to(SOURCE_ROOT)),
                                candidate,
                            )
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == forbidden_prefix or alias.name.startswith(
                        f"{forbidden_prefix}."
                    ):
                        imports.add(
                            (str(path.relative_to(SOURCE_ROOT)), alias.name)
                        )
    return imports


def test_import_candidates_resolve_application_aliases() -> None:
    path = SOURCE_ROOT / "applications" / "magma_gas" / "module.py"
    cases = {
        "from .. import clouds": "exogibbs.applications.clouds",
        "from exogibbs import applications": "exogibbs.applications",
        "from exogibbs.applications import clouds": (
            "exogibbs.applications.clouds"
        ),
    }
    for statement, expected in cases.items():
        node = ast.parse(statement).body[0]
        assert isinstance(node, ast.ImportFrom)
        assert expected in _import_from_candidates(path, node)


def test_internal_packages_do_not_import_public_api_modules() -> None:
    observed = set()
    for package in (
        "applications",
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
        "applications",
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


def test_foundation_packages_do_not_import_applications() -> None:
    observed = set()
    for package in (
        "condensates",
        "equilibrium",
        "io",
        "math",
        "optimize",
        "solubility",
        "thermo",
        "utils",
    ):
        observed |= _imports_under(
            SOURCE_ROOT / package,
            "exogibbs.applications",
        )

    assert observed == set()


def test_applications_do_not_import_upper_layers() -> None:
    observed = set()
    for forbidden in (
        "exogibbs.api",
        "exogibbs.experimental",
        "exogibbs.presets",
    ):
        observed |= _imports_under(SOURCE_ROOT / "applications", forbidden)

    assert observed == set()


def test_application_packages_do_not_import_siblings() -> None:
    applications_root = SOURCE_ROOT / "applications"
    packages = sorted(
        path.name
        for path in applications_root.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    )
    observed = set()
    for package in packages:
        for sibling in packages:
            if sibling != package:
                observed |= _imports_under(
                    applications_root / package,
                    f"exogibbs.applications.{sibling}",
                )

    assert observed == set()


def test_magma_gas_application_does_not_import_condensate_equilibrium() -> None:
    observed = _imports_under(
        SOURCE_ROOT / "applications" / "magma_gas",
        "exogibbs.equilibrium.condensate",
    )

    assert observed == set()


def test_magma_gas_generic_core_does_not_import_model_physics() -> None:
    observed = set()
    for forbidden in (
        "exogibbs.applications.magma_gas.models",
        "exogibbs.solubility",
        "exogibbs.thermo.oxygen_fugacity",
    ):
        observed |= _imports_under(
            SOURCE_ROOT / "applications" / "magma_gas",
            forbidden,
        )
    generic_observed = {
        item
        for item in observed
        if not item[0].startswith("applications/magma_gas/models/")
    }

    assert generic_observed == set()


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
