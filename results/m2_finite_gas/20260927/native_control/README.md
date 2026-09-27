# Finite 76-gas implementation checks, 27 September 2026

This archive preserves the central H = 1e24 mol input, the original empty
stdout/stderr log, and the observed exit status of the cold native contact
attempt. The process ended with exit 137 and produced no source JSON.
No thermodynamic acceptance, convergence failure, or phase boundary follows
from this interrupted calculation. High aggregate memory use was observed;
the exact origin of the kill was not established.

`initial_lp.json` is a later read-only reconstruction of the numerical LP
start, with no native or atmospheric equilibrium evaluation. It is not a
solver iterate or an equilibrium result. Its positive P allocation shows
that the default numerical start did not constrain phosphorus to zero.

`validation.json` pins the executed source and EOS heads, input hash, full
unit-suite result and documentation build. These software checks do not
replace a fresh accepted finite source or global pressure closure.
