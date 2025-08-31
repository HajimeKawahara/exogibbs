# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/exogibbs/` with subpackages `io/`, `equilibrium/`, `thermo/`, `optimize/`, `api/`, `presets/`, `utils/`; data files under `src/exogibbs/data/`.
- Tests: `tests/unittests/` organized by domain (e.g., `io/`, `equilibrium/`); files named `*_test.py`.
- Examples & docs: `examples/` and `documents/`.

## Build, Test, and Development Commands
- Install (editable): `python -m pip install -e .` (Python >= 3.9).
- Run tests: `pytest tests/unittests` (CI exports `results/pytest.xml`).
- Optional build: `python -m pip install build && python -m build` (ensure Git tags for correct versioning).

## Coding Style & Naming Conventions
- Language: Python with type hints where practical; 4-space indentation; keep functions short and focused.
- Naming: modules/functions/variables `snake_case`; classes `CapWords`; constants `UPPER_SNAKE_CASE`.
- Imports: standard lib, third-party, then local; avoid unused imports.

## Testing Guidelines
- Framework: `pytest`; keep tests deterministic (no network/GPU). Use small fixtures and cover edge cases (e.g., parsing, interpolation bounds).
- Location/pattern: place tests under `tests/unittests/<area>/`, name `*_test.py`.
- Run locally: `pytest tests/unittests`. Aim to maintain/raise coverage of changed code.

## Commit & Pull Request Guidelines
- Commits: concise, imperative (e.g., "thermo: fix electron parsing"); group related changes; reference issues (`#123`) when applicable.
- PRs: include a clear description, motivation, and testing notes; link issues; add/adjust tests; update data packaging (`MANIFEST.in`) when adding files under `src/exogibbs/data/`.
- CI: PRs to `develop`/`main` must pass the pytest workflow.

## Architecture Overview (Quick)
- `io.load_data`: loads molecule catalogs and JANAF-like tables.
- `thermo.stoichiometry`: builds formula matrices from species names.
- `equilibrium.gibbs`: pads tables and provides JAX-friendly interpolation of h-vectors.
- `optimize.*`: JAX-based core, VJP, and minimization routines.
- `api.chemistry` and `presets/`: typed containers and ready-to-use setups.

