# Repository Guidelines

## Project Structure & Module Organization
Python packages live in `src/exogibbs/` under focused modules (`io/`, `thermo/`, `equilibrium/`, `optimize/`, `api/`, `presets/`, `utils/`). Place shared datasets in `src/exogibbs/data/` and list new assets in `MANIFEST.in`. Tests mirror the package tree at `tests/unittests/`, e.g., `tests/unittests/equilibrium/gibbs_test.py`. Workflow demos sit in `examples/`; long-form background and Sphinx-ready docs belong in `documents/`. Ignore transient build outputs except under `results/`.

## Build, Test, and Development Commands
`python -m pip install -e .` creates an editable install with local extras. Use `pytest tests/unittests` for the CI-aligned deterministic suite; target subpackages with `pytest tests/unittests/thermo`. Run `python -m pip install build && python -m build` before release review to produce sdist and wheel. Execute `./update_doc.sh` whenever documentation sources change.

## Coding Style & Naming Conventions
Support Python 3.9+, using four-space indentation and type annotations on public APIs. Favor small, composable helpers and prune unused code. Imports follow stdlib → third-party → local order. Apply `snake_case` for modules, functions, and variables, `CapWords` for classes, and `UPPER_SNAKE_CASE` for constants. Keep comments purposeful, explaining only non-obvious choices.

## Testing Guidelines
Write deterministic, offline PyTest cases named `*_test.py`, colocated with their modules under `tests/unittests/`. Each bug fix or feature requires a targeted regression test. Run the full suite—`pytest tests/unittests`—before submitting reviews, and note any skipped tests or fixtures you add.

## Commit & Pull Request Guidelines
Use imperative, scoped commit subjects (e.g., `thermo: fix electron parsing`) and reference issues with `#123` when applicable. Pull requests should outline motivation, summarize key changes, link relevant issues, and attach the latest `pytest` output. Update docs, examples, or packaged data alongside behavioral changes, and flag follow-up tasks or known limitations.

## Security & Configuration Tips
Work offline; CI has no network egress. Avoid privileged commands and system-wide installations. Vendor any external resources, pin dependencies when determinism matters, and never commit secrets or credentials.
