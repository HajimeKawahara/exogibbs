# Five saved finite-source common-plane bounds

These are new mathematical assessments of the unchanged final M1, GAS,
CLOUD, OXYGEN and OH roots. No source equilibrium, pressure root, native
property receipt, standard state or historical strict-sign flag was changed.
All five commands and the full suite actually exited with code 0. The raw
JSON, stdout logs and execution receipts are copied without modification;
`raw_sha256.json` checks every copied file.

The executed proof source was ExoGibbs
`aa12f3c9378dd48763d1fe1d4e5ba8d2e75763f8`. The imported alloy interval provider
was ExoEOS `b041f94d1a6d284db36925e858399856e1cd29b4`. The solution-model
expressions were reconstructed from the pinned ExoEOS git blobs at
`113ae4d306aea81c86f78267b1d9074030bda3f1` and compared to every original
saved phase-proof expression, domain and standard-state receipt.

| Saved case | Phase domains bounded | Certified upper bound `(U-L)/(RT sum(b))` | Declared finite-source numerical acceptance |
| --- | ---: | ---: | --- |
| [M1](M1/result.json) | 53 | 1.4994684346289041e-13 | accepted |
| [GAS](GAS/result.json) | 53 | 1.0847241462381633e-13 | accepted |
| [CLOUD](CLOUD/result.json) | 81 | 1.1102089228423533e-13 | accepted |
| [OXYGEN](OXYGEN/result.json) | 81 | 1.3372917624914841e-13 | accepted |
| [OH](OH/result.json) | 81 | 1.1152130800456598e-13 | accepted |

Here `U` and `L` denote dimensional Gibbs energies; the JSON stores their
dimensionless equivalents in units of RT mol. The normalized error is below
the design energy tolerance of `1e-9` for every case. The unchanged source
atom and chemistry gates and the separate present-alloy composition-search
uncertainty/complementarity gates also pass. The independent interval
evaluation of the repaired feasible state agrees with the saved source
energy to at most `4.489e-15 RT` per inventory atom.

The counts include twenty competing solution models, thirteen recorded pure
native candidates, the selected liquid with dissolved H2, the declared
source-alloy box, the ideal-gas simplex and every eligible pure cloud. A pure
candidate means its exact recorded oxide-mass/G unit and all nonnegative
scales. The negative-bound correction uses the exact atom count of each
declared phase unit. It does not enlarge a pure candidate's composition
domain or assert empirical validity of any provider.

The method is documented in
[the common-plane contract](../../../../examples/metal_silicate/m2_common_plane.md).
An exact rational atom repair supplies a feasible upper bound. A verified
uniform reduction of the saved elemental plane supplies a global lower
bound shared by every included phase. Their interval difference is the
reported primal/dual gap. This is a tolerance-based declared-model result;
it does not turn the original fixed plane into a strictly nonnegative one.
The true tiny negative fixed-plane witnesses for M1 and OXYGEN remain part
of the historical evidence.

The original closure roots, audits, phase proofs, fixed-plane precision
supplement and five-case binding are in the unchanged
[ExoInventory archive at d123579](https://github.com/HajimeKawahara/exoinventory/tree/d123579fad91d2a7f415cf2cf55ad4eddd72cc1e/examples/subneptune_taxonomy/finite_melt/validation/20260927_m2_finite_response).
The executed aggregate hash was
`8e7321bf8729b6b4772dbd89b636976d9a449e608b57f39d7bd610a15a0ef458`.
Each output records all absolute execution paths and input hashes. Reuse at
different paths requires preserving those original bytes, not relabeling a
new execution as the archived one.

Validation: **1990 passed, 1 skipped, 28 warnings**. The sole skip is the
unavailable setuptools-scm generated version module. Twelve overflow and
twelve divide-by-zero warnings originate in the existing rocky-raccoon
least-squares tests; four SLSQP trial-clipping warnings originate in existing
nonconvex-liquid controls. No optional EOS tests were skipped.

The result covers these five finite source states, their declared models and
composition domains at the saved T/P. Empirical material applicability,
native-build equivalence, a metal-free boundary, omitted transport and a
Gibbs minimum for the nonisothermal planet are separate questions. New
reconstructed-water or extended-alloy models require new phase bounds and
a new common-plane assessment.
