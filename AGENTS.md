# Repository Guidelines

## Project Structure & Module Organization
- Source lives under `src/exogibbs/` in focused modules: `io/`, `thermo/`, `equilibrium/`, `optimize/`, `api/`, `presets/`, `utils/`.
- Shared datasets go in `src/exogibbs/data/` and must be listed in `MANIFEST.in`.
- Tests mirror the package tree in `tests/unittests/` (e.g., `tests/unittests/equilibrium/gibbs_test.py`).
- Workflow demos are in `examples/`; long‑form docs (Sphinx‑ready) in `documents/`.
- Ignore transient build outputs; if you must store results, keep them under `results/`.

## Build, Test, and Development Commands
- Install (editable, with local extras): `python -m pip install -e .`
- Run tests (deterministic CI suite): `pytest tests/unittests` or a subpackage, e.g., `pytest tests/unittests/thermo`.
- Build sdist and wheel before release review: `python -m pip install build && python -m build`.
- Update documentation artifacts after doc changes: `./update_doc.sh`.

## Coding Style & Naming Conventions
- Python 3.9+; four‑space indentation; add type annotations on public APIs.
- Imports are ordered: stdlib → third‑party → local.
- Naming: `snake_case` for modules/functions/variables; `CapWords` for classes; `UPPER_SNAKE_CASE` for constants.
- Prefer small, composable helpers; remove unused code. Keep comments purposeful (explain non‑obvious choices).

## Testing Guidelines
- Framework: PyTest; tests are deterministic and offline.
- Name tests `*_test.py` and colocate under `tests/unittests/` mirroring the module path.
- Each feature or bug fix requires a targeted regression test.
- Run the full suite before sending PRs: `pytest tests/unittests`.

## Commit & Pull Request Guidelines
- Commit subjects are imperative and scoped (e.g., `thermo: fix electron parsing`); reference issues with `#123`.
- PRs include motivation, a summary of key changes, linked issues, and the latest `pytest` output.
- Update docs, examples, and packaged data when behavior changes; note any skipped tests/fixtures and follow‑ups.

## Security & Configuration Tips
- Work offline; CI has no network egress.
- Avoid privileged commands and system‑wide installs. Vendor external resources and pin dependencies when determinism matters. Never commit secrets.

