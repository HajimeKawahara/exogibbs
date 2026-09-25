# Bounded native stability searches

These saved searches found no negative feasible witness beyond the declared
`1e-8 RT` per mole of atoms tolerance. Every result remains **unresolved**:
the evaluation budgets were exhausted, no global lower bound was supplied,
and material calibration is not established. These are new property trials
at unchanged saved hosts, not new planetary closures or physical phase
boundaries.

| Search | H inventory (mol atoms) | T (K) | P (bar) | Finite objective evaluations, including fresh final | Best objective (RT per mol atoms) | Failed evaluations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hornblende composition | 2e24 | 2173.15 | 432.29847124 | 9 | 0.5684496831 | 0 |
| Orthoamphibole composition | 2e24 | 2173.15 | 432.29847124 | 9 | 0.4626575341 | 0 |
| Kalsilite composition | 2e24 | 2173.15 | 432.29847124 | 9 | 0.5980603350 | 0 |
| Two liquid daughters | 1e24 | 2173.15 | 267.20283416 | 31 | -3.0397e-15 | 0 |

The solid searches use nonnegative native endmember simplexes and the same
native Gibbs energy plus augmented-host H2 dilution correction. Finite
kalsilite interior properties do not replace its unavailable native
incipient saturation estimate. Each solid has eight budgeted objective
evaluations and one fresh final evaluation.

The two-liquid search partitions every parent component, including dissolved
H2, into two daughters at the same T/P. Identical linear H2 standard terms
cancel; native water remains in MELTS. Daughter fractions range from 0.001 to
0.999, defining a search subset. Exactly absent parent components remain
absent. Thirty budgeted objective evaluations and one fresh final evaluation
each require properties of both daughters. Nonidentical daughter compositions
were sampled: the largest fraction displacement from 0.5 was 0.25305. The
best objective is consistent with roundoff, not a resolved instability.

Independent primitive recount found a maximum component-sum residual of
`5.55e-17 mol` and zero discrepancy in the stored augmented Gibbs energies,
energy differences, and normalized objectives. This checks the saved trial
construction; it does not turn a bounded search into a stability certificate.

## Preserved files and inputs

The two raw result JSON files, original candidate companion receipt, and two
executed orchestration scripts are byte-for-byte copies. Scripts retain their
original machine paths. Use new output locations and explicit matching
checkouts for any future run. No native binary is included.

- [Candidate search raw](native_candidate_search_smoke.json) and
  [original companion receipt](native_candidate_search_smoke_receipt.json).
- [Two-liquid search raw](native_two_liquid_search_smoke.json).
- [Candidate orchestration](check_native_search.py) and
  [two-liquid orchestration](check_native_two_liquids.py).
- [Public input/source references](public_references.json),
  [archive validation receipt](validation_receipt.json), and
  [SHA256 manifest](manifest.json).

The candidate native property input is the
[ExoEOS 147931f supplied-host receipt](https://github.com/HajimeKawahara/exoeos/blob/147931fc382af060d2601fc279a1d65ce73afcee/examples/m2_candidate_basis/native_supplied_host.json).
Its dissolved-H2 amount comes from the
[original high-H physical assessment](https://github.com/HajimeKawahara/exoinventory/blob/15e28809f5bab7e1f20d1b35bdc05f7cea5d836e/examples/subneptune_taxonomy/finite_melt/validation/20260925_m2_hydrogen_pilot/native_h_pilot/case_002/physical_root_000.json).
The two-liquid input is the
[ExoInventory 06077dd middle-H physical assessment](https://github.com/HajimeKawahara/exoinventory/blob/06077dd4d401bd611d69ae308c2c26254f4ff3d8/examples/subneptune_taxonomy/finite_melt/validation/20260925_m2_readiness/physical/physical_reactions_h1e24.json).
All three input blobs were checked against their recorded SHA256 values and
are referenced rather than duplicated here.

The candidate run predates the committed nonnegative-endmember guard. Its
original dirty-checkout provenance is retained; its executed file hashes
match the files committed in Gibbs `f3df59b` and EOS `147931f`. The two-liquid
run used integration commits Gibbs `6329226` and EOS `04c950d`. The listed
executed source files match the public PR files in Gibbs `9ca7c47` and EOS
`b341186` byte-for-byte. These equivalences provide accessible source links;
they do not replace the recorded execution commits or claim that entire
integration checkouts are identical.
