# Repository Guidelines

## Project Structure & Module Organization
Keep all Python sources in `src/exogibbs/`, grouped by domain modules such as `io/`, `thermo/`, `equilibrium/`, `optimize/`, `api/`, `presets/`, and `utils/`. Shared assets live in `src/exogibbs/data/`; list any new datasets in `MANIFEST.in`. Deterministic tests mirror the package layout under `tests/unittests/`. Place workflow examples inside `examples/` and longer references in `documents/`. Avoid checking in transient artifacts like `results/pytest.xml`.

## Build, Test, and Development Commands
Use `python -m pip install -e .` for an editable install with local dependencies. Run `pytest tests/unittests` before submitting changes; target subsets such as `pytest tests/unittests/thermo` when iterating. Build release artifacts with `python -m pip install build && python -m build`. Refresh docs when needed via `./update_doc.sh`.

## Coding Style & Naming Conventions
Target Python 3.9+, four-space indentation, and concise helpers. Follow import order: stdlib, third-party, then local. Use `snake_case` for modules, functions, and variables; `CapWords` for classes; `UPPER_SNAKE_CASE` for constants. Annotate public APIs with type hints and keep comments succinct, focusing on non-obvious decisions.

## Testing Guidelines
Tests rely on `pytest` and must remain offline and deterministic. Name files `*_test.py` and colocate them with the corresponding module under `tests/unittests/`. Add regression tests when fixing bugs or introducing features, and ensure they pass with `pytest tests/unittests`.

## Commit & Pull Request Guidelines
Write imperative commit subjects scoped to the module (e.g., `thermo: fix electron parsing`) and reference tracker issues like `#123` when applicable. Pull requests should explain motivation, summarize key changes, link issues, and share recent `pytest` output. Update docs, examples, or packaged data when behavior shifts, and note follow-up tasks or known limitations.

## Security & Configuration Tips
Work offline; CI has no network access. Vendor required resources and keep secrets out of the repo. Avoid privileged commands or system-wide installs, and pin dependencies when deterministic behavior matters.
