# Repository Guidelines

## Project Structure & Module Organization
The Python package lives in `src/exogibbs/`, organized into domain subpackages such as `io/`, `thermo/`, `equilibrium/`, `optimize/`, `api/`, `presets/`, and `utils/`. Shared datasets belong in `src/exogibbs/data/`; add new assets there only when necessary and remember to update `MANIFEST.in` for packaging. Deterministic tests mirror the tree under `tests/unittests/`, so prefer paths like `tests/unittests/thermo/matrix_test.py` when expanding coverage. Place runnable tutorials in `examples/` and longer design notes in `documents/`.

## Build, Test, and Development Commands
- `python -m pip install -e .` creates an editable install of the package for iterative development.
- `pytest tests/unittests` runs the deterministic suite; CI expects the same invocation and captures `results/pytest.xml`.
- `python -m pip install build && python -m build` produces both sdist and wheel artifacts before release tags.

## Coding Style & Naming Conventions
Write Python with 4-space indentation, type annotations, and focused functions. Use `snake_case` for modules, functions, and variables, `CapWords` for classes, and `UPPER_SNAKE_CASE` for constants. Group imports by stdlib, third-party, then local modules, removing any unused entries before committing. Keep comments brief and only where logic is non-obvious.

## Testing Guidelines
Rely exclusively on `pytest`; keep tests offline and deterministic. Name test files `*_test.py` and colocate them with their domain areas in `tests/unittests/`. When fixing bugs or adding parsing edge cases, add regression tests alongside the change. Run `pytest tests/unittests` locally before pushing.

## Commit & Pull Request Guidelines
Commits use imperative, scoped subjects such as `thermo: fix electron parsing` and may reference issues with `#123`. Pull requests should outline motivation, enumerate key changes, link related issues, and attach recent `pytest` output. Update docs, examples, or packaged data whenever behavior shifts.

## Security & Configuration Tips
Work offline; network access is disabled in CI. Target Python 3.9 or newer and pin dependencies when determinism matters. Avoid commands that alter system-wide state or require elevated privileges; any dataset or config changes should remain within the repository.
