# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/exogibbs/`, grouped by domain: `io/`, `thermo/`, `equilibrium/`, `optimize/`, `api/`, `presets/`, and `utils/`. Shared datasets reside in `src/exogibbs/data/`; add new assets only when necessary and update `MANIFEST.in` so they ship with releases. Deterministic tests mirror this tree under `tests/unittests/` (e.g., `tests/unittests/equilibrium/gibbs_test.py`). Place runnable examples in `examples/` and longer notes or Sphinx docs in `documents/`.

## Build, Test, and Development Commands
- `python -m pip install -e .` sets up an editable install for local development.
- `pytest tests/unittests` runs the deterministic test suite (CI uses the same entry and writes `results/pytest.xml`).
- `python -m pip install build && python -m build` creates sdist and wheel artifacts for release validation.

## Coding Style & Naming Conventions
Use Python 3.9+ with 4-space indentation and type annotations. Prefer small, focused functions. Naming: `snake_case` for modules/functions/variables, `CapWords` for classes, `UPPER_SNAKE_CASE` for constants. Order imports stdlib → third-party → local; remove unused imports. Keep comments brief, only where logic is non-obvious.

## Testing Guidelines
Use `pytest` only; tests must be offline and deterministic. Name files `*_test.py` and colocate with their domain in `tests/unittests/`. When fixing bugs or parsing edges, add a regression test beside the change. Run `pytest tests/unittests` before pushing.

## Commit & Pull Request Guidelines
Write imperative, scoped commit subjects (e.g., `thermo: fix electron parsing`), referencing issues with `#123` when relevant. PRs should explain motivation, list key changes, link issues, and include recent `pytest` output. Update docs, examples, or packaged data when behavior changes.

## Security & Configuration Tips
Work offline; CI has no network access. Avoid privileged/system-wide changes. Pin dependencies when determinism matters. Keep datasets and configs within the repository.
