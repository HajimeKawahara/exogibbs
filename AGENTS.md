# Repository Guidelines

## Project Structure & Module Organization
Keep Python sources inside `src/exogibbs/`, organized by domain modules such as `io/`, `thermo/`, `equilibrium/`, `optimize/`, `api/`, `presets/`, and `utils/`. Shared assets belong in `src/exogibbs/data/`; only add new datasets when necessary and list them in `MANIFEST.in` so they ship with releases. Deterministic tests mirror the package layout under `tests/unittests/` (e.g., `tests/unittests/equilibrium/gibbs_test.py`). Examples that demonstrate workflows live in `examples/`, while longer notes or Sphinx-ready docs go in `documents/`. CI writes artifacts such as `results/pytest.xml`; avoid committing transient files outside these directories.

## Build, Test, and Development Commands
- `python -m pip install -e .` installs an editable copy with local dependencies.
- `pytest tests/unittests` runs the deterministic suite used by CI.
- `python -m pip install build && python -m build` produces the sdist and wheel for release validation.
- `./update_doc.sh` refreshes generated documentation when docs change.

## Coding Style & Naming Conventions
Target Python 3.9+ and four-space indentation. Annotate functions and public APIs with type hints and prefer small, composable helpers. Use `snake_case` for modules, functions, and variables; `CapWords` for classes; `UPPER_SNAKE_CASE` for constants. Order imports stdlib → third-party → local, and delete unused imports. Keep comments succinct and reserve them for clarifying non-obvious decisions.

## Testing Guidelines
Use `pytest` exclusively, keeping tests offline and deterministic. Name files `*_test.py` and colocate them with the corresponding module inside `tests/unittests/`. When fixing a bug or adding a feature, accompany the change with a focused regression test. Run `pytest tests/unittests` (or a targeted subset such as `pytest tests/unittests/thermo`) before sending a review.

## Commit & Pull Request Guidelines
Write imperative, scoped commit subjects (e.g., `thermo: fix electron parsing`) and reference tracker issues with `#123` when relevant. Pull requests should explain motivation, summarize key changes, link issues, and paste recent `pytest` output. Update docs, examples, or packaged data whenever behavior changes, and call out any follow-up work or known limitations.

## Security & Configuration Tips
Develop offline—CI has no network access—so vendor any required resources. Avoid privileged commands or system-wide installs. Pin dependencies when deterministic behavior matters, and keep credentials or API keys out of the repository.
