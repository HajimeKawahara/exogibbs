# Curated Condensate HEAD Route End-to-End Replays

These tests replay the saved 14-row curated HEAD route evidence through the
public condensate API. They do not discover new supports and do not use
FastChem4 trace, public, or runtime values as constructor inputs.

Run all curated replay tests:

```bash
pytest -q tests/endtoend/curated_cases
```

Run one curated family:

```bash
pytest -q tests/endtoend/curated_cases/test_solar_silicate_first_condensation.py
```

The all-row test checks 14 curated rows. The family-specific files make it
straightforward to re-run only the family under investigation.
