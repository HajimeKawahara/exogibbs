# Conserved numerical source initialization

The saved input is the accepted **metal-free** branch of the fresh 35-gas
native source at 2173.15 K / 250 bar. The donor's complete metal selection
was unresolved; this distinction is retained in its provenance. Donor data
are a numerical starting point, not an accepted solution of the new gas or
liquid model. The original Inventory trial hash and inventory-byte hash are
included in `metal_free_seed_250bar.json`.

Component identity, phase membership and elemental formulas are mapped to
the thirteen-carrier record. Adding 1e-4 of the target's feasible metal-free
LP interior preserves the same thirteen budgets (maximum relative residual
8.88e-16), exact-zero global elements and zero initial metal. P starts at
1.9756639132e16 mol, not at an imposed lower bound. No native or equilibrium
evaluation is performed by the mapping.

`mapped_seed_250bar.json` preserves an independent implementation of the
mapping. The public helper at c83bb99 matches its amounts bitwise, as recorded
in `saved_seed_helper_check.json`. With this checkout's `src` on PYTHONPATH
and JAX CPU/x64 enabled, `python check_seed.py` repeats that comparison.
The target provider snapshot in the independent receipt predates the helper
addition; no previous physics result is renamed or assigned to a new head.
