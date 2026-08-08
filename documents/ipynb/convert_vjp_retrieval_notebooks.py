#!/usr/bin/env python3
"""Convert the VJP retrieval notebooks to committed RST without execution.

The notebooks are the editable sources.  This script intentionally uses the
``nbconvert`` exporter directly and never starts a Jupyter kernel.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Mapping

try:
    import nbformat
    from nbconvert import RSTExporter
except ImportError as exc:  # pragma: no cover - depends on the docs environment
    raise SystemExit(
        "Notebook conversion requires nbformat and nbconvert. Install them in "
        "the documentation environment before running this script."
    ) from exc


NOTEBOOK_DIRECTORY = Path(__file__).resolve().parent
NOTEBOOK_STEMS = (
    "exojax_nuts_gas_no_grid",
    "exojax_nuts_gas_grid",
    "exojax_nuts_condensate_fixed_support",
)
GENERATED_HEADER = (
    ".. This file is generated from the sibling .ipynb by "
    "convert_vjp_retrieval_notebooks.py.\n"
    ".. Do not edit this RST file directly.\n\n"
)


def _separate_directives(body: str) -> str:
    """Insert the blank line required before top-level RST directives."""

    lines = [line.rstrip() for line in body.splitlines()]
    separated = []
    for line in lines:
        if line.startswith(".. ") and separated and separated[-1].strip():
            separated.append("")
        separated.append(line)
    return "\n".join(separated)


def _export_notebook(notebook_path: Path) -> tuple[str, Mapping[str, bytes]]:
    """Return deterministic RST text and binary resources without execution."""

    notebook = nbformat.read(notebook_path, as_version=4)
    exporter = RSTExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    resources = {
        "unique_key": notebook_path.stem,
        "output_files_dir": f"{notebook_path.stem}_files",
    }
    body, exported_resources = exporter.from_notebook_node(
        notebook,
        resources=resources,
    )
    body = _separate_directives(body)
    rst = GENERATED_HEADER + body.rstrip() + "\n"
    outputs = exported_resources.get("outputs", {})
    return rst, outputs


def _validate_resource_name(name: str, stem: str) -> Path:
    """Reject exporter output paths outside the notebook-specific asset tree."""

    relative_path = Path(name)
    expected_directory = f"{stem}_files"
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or not relative_path.parts
        or relative_path.parts[0] != expected_directory
    ):
        raise ValueError(f"Unsafe nbconvert resource path: {name!r}")
    return relative_path


def _check_notebook(stem: str, rst: str, outputs: Mapping[str, bytes]) -> bool:
    """Return whether committed RST and assets exactly match the notebook."""

    rst_path = NOTEBOOK_DIRECTORY / f"{stem}.rst"
    if not rst_path.is_file() or rst_path.read_text(encoding="utf-8") != rst:
        return False

    expected_resources = {
        _validate_resource_name(name, stem): content
        for name, content in outputs.items()
    }
    asset_directory = NOTEBOOK_DIRECTORY / f"{stem}_files"
    actual_resources = {}
    if asset_directory.is_symlink():
        return False
    if asset_directory.is_dir():
        actual_resources = {
            path.relative_to(NOTEBOOK_DIRECTORY): path.read_bytes()
            for path in sorted(asset_directory.rglob("*"))
            if path.is_file()
        }
    return actual_resources == expected_resources


def _write_notebook(stem: str, rst: str, outputs: Mapping[str, bytes]) -> None:
    """Replace one notebook's generated RST and exact generated asset tree."""

    rst_path = NOTEBOOK_DIRECTORY / f"{stem}.rst"
    with rst_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(rst)

    asset_directory = NOTEBOOK_DIRECTORY / f"{stem}_files"
    if asset_directory.exists():
        if (
            asset_directory.is_symlink()
            or not asset_directory.is_dir()
            or asset_directory.parent != NOTEBOOK_DIRECTORY
        ):
            raise ValueError(f"Refusing to replace unsafe asset path: {asset_directory}")
        shutil.rmtree(asset_directory)

    for name, content in sorted(outputs.items()):
        relative_path = _validate_resource_name(name, stem)
        output_path = NOTEBOOK_DIRECTORY / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stems",
        nargs="*",
        metavar="STEM",
        help="Notebook stems to convert (default: all VJP retrieval tutorials).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check committed generated files without modifying them.",
    )
    args = parser.parse_args()

    selected_stems = args.stems or NOTEBOOK_STEMS
    unknown_stems = sorted(set(selected_stems) - set(NOTEBOOK_STEMS))
    if unknown_stems:
        parser.error(
            "unknown notebook stem(s): "
            + ", ".join(unknown_stems)
            + "; expected one of: "
            + ", ".join(NOTEBOOK_STEMS)
        )
    stale = []
    for stem in selected_stems:
        notebook_path = NOTEBOOK_DIRECTORY / f"{stem}.ipynb"
        if not notebook_path.is_file():
            raise FileNotFoundError(f"Missing notebook source: {notebook_path}")
        rst, outputs = _export_notebook(notebook_path)
        if args.check:
            if not _check_notebook(stem, rst, outputs):
                stale.append(stem)
        else:
            _write_notebook(stem, rst, outputs)
            print(f"converted {notebook_path.name} -> {stem}.rst")

    if stale:
        print("stale generated notebook documentation: " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
