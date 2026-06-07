# Curated Condensate HEAD Route Fresh API Regressions

These tests run curated HEAD route rows through the public condensate API from
a fresh FastChem4 preset setup. They do not read saved `results/` artifacts and
do not use FastChem4 trace, public, or runtime values as constructor inputs.

Run all curated fresh API tests:

```bash
pytest -q tests/endtoend/curated_cases
```

The all-row test checks 14 curated rows using explicit temperatures, pressures,
element budgets, and initial condensate supports encoded in the test file. A
targeted HEAD route v1.1 regression also covers a water-condensation
intermediate layer where the restricted solver succeeds but the lifecycle
selector falls back with a caveat.
