# Pressure-branch initial-amount controls

The code at `f65efac4e0351d613b817ce57e907592a25a79f8` re-solves a small
ideal silicate/metal/gas assemblage at 1.01 bar using either independent
initialization or both accepted branch ledgers from 1 bar. Every result
includes fresh local, atom, scalar derivative and insertion audits.

| Amount scale | Maximum amount difference / scale | G difference / RT / total atom | Maximum relative atom residual |
| --- | ---: | ---: | ---: |
| 1 | 1.11023e-16 | -1.23224e-17 | 2.20940e-16 |
| 1e24 | 2.68435e-16 | +1.32417e-17 | 1.33550e-16 |

[control.json](control.json) retains all donor, cold and continuation results.
[default_compatibility.json](default_compatibility.json) verifies exact equality
of the entire default selection payload against parent `5149a79` on four
independent present/absent controls at 1 and 1.01 bar.
[execution.json](execution.json) records the actual successful processes and
file hashes. The 113 targeted regressions include rejected warm-candidate
fallback and unchanged fresh-insertion admission.

These are numerical controls, not BSE calculations or a measured speedup.
No existing native source, material parameter, physical acceptance flag or
archived scientific result was changed. The full package suite is left to
PR CI; the local run follows the task's explicit targeted-test instruction.
