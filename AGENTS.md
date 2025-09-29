# Repository Guidelines

## Project Structure & Module Organization
- Core Python packages live in `src/exogibbs/` with domain-focused subpackages (`io/`, `equilibrium/`, `thermo/`, `optimize/`, `api/`, `presets/`, `utils/`).
- Shared datasets reside under `src/exogibbs/data/`; update `MANIFEST.in` if shipping new files.
- Unit tests sit in `tests/unittests/` mirroring the source layout (e.g., `tests/unittests/io/`); name files `*_test.py`.
- Working examples belong in `examples/`; long-form references or design notes belong in `documents/`.

## Build, Test, and Development Commands
- `python -m pip install -e .` installs the project in editable mode (Python ≥ 3.9).
- `pytest tests/unittests` runs the full deterministic test suite; CI exports results to `results/pytest.xml`.
- `python -m pip install build && python -m build` produces source and wheel artifacts; tag releases before publishing.

## Coding Style & Naming Conventions
- Use Python with type hints where practical, 4-space indentation, and focused functions.
- Follow naming: modules/functions/variables in `snake_case`, classes in `CapWords`, constants in `UPPER_SNAKE_CASE`.
- Order imports as standard library, third-party, then local modules; remove unused imports before committing.
- Favor concise inline comments only for non-obvious logic.

## Testing Guidelines
- Write tests with `pytest`, colocated by domain; prefer small fixtures and deterministic behavior (no network/GPU access).
- Keep high-value edge coverage (parsers, interpolation bounds, stoichiometry balancing) and extend tests when touching those areas.
- Run `pytest tests/unittests` prior to pushing; add regression tests alongside bug fixes.

## Commit & Pull Request Guidelines
- Author commits in imperative mood (e.g., `thermo: fix electron parsing`); group related changes and reference issues (`#123`) when relevant.
- Pull requests require a clear motivation, summary of code changes, and local test evidence; link issues and update docs/examples when behavior shifts.
- Ensure CI passes before requesting review; include data packaging updates if `src/exogibbs/data/` changes.

## Architecture Notes
- `io.load_data` ingests molecule catalogs and JANAF-style tables.
- `thermo.stoichiometry` builds formula matrices from species lists.
- `equilibrium.gibbs` handles interpolation/padding for JAX computations, while `optimize.*` wraps core minimization and VJP routines.
- High-level APIs live in `api/chemistry` and `presets/` for reusable setups.
