# Formal comparison using the fixed column consumer

After the finite-JANAF column-reference correction, the 35-gas source was
closed again with the same final consumer used for expanded scenarios.
This is a new saved root and fresh native phase assessment, kept separately
from the preceding closed root and its offline replay.

The source uses Inventory `4b5255f`, Gibbs `7cf41f1`, EOS `ee21fa1`;
T=2173.15 K and P=267.2028341602758 bar. Its accepted physical audit and
full hash are copied unchanged in both run directories. The proof code
is Gibbs `1af5821`, EOS `113ae4d`. Both actual processes exited 0.

All 20 competing solution phases have strictly positive global insertion
lower bounds; the minimum is orthopyroxene
**0.000370622565611766 RT/formula unit**. The full second-liquid/H2 bound
is **0 RT**, with 4719 nodes. Solid and liquid runs took 253.711 s and
151.469 s respectively. No proof runner solved a new pressure root.

| Phase | Lower bound (RT/formula unit) | Nodes |
| --- | ---: | ---: |
| hornblende | 14.97826319942905 | 1 |
| biotite | 13.35339662045244 | 1 |
| garnet | 1.140485730436018 | 3 |
| alkali-feldspar | 2.429315922798057 | 3 |
| kalsilite | 8.405848619328255 | 1 |
| leucite | 1.177473932506022 | 1 |
| alloy-solid | 0.2306027839422533 | 1 |
| alloy-liquid | 0.04536556585173135 | 1 |
| cummingtonite | 9.778603466507164 | 1 |
| olivine | 0.001562348662674533 | 9 |
| nepheline | 4.667491913622348 | 1 |
| clinoamphibole | 3.933737635185373 | 1 |
| orthoamphibole | 3.564008628189747 | 1 |
| melilite | 1.903911992506217 | 1 |
| ortho-oxide | 0.9416522933615805 | 3 |
| rhm-oxide | 0.09796099625620645 | 59 |
| plagioclase | 1.931935891315485 | 1 |
| clinopyroxene | 0.007455451042878329 | 515 |
| orthopyroxene | 0.000370622565611766 | 731 |
| spinel | 0.01028145072060386 | 313 |

Every domain and uncertainty distinction in the parent README still applies.
The native Fe/Ni alloy candidates do not represent the source Fe/Si/O/H alloy.
The 76-gas, 67-condensate and alternative-standard roots require separate
assessments; this 35-gas result is not a substitute for them.

`execution_receipt.json` records exact input/code hashes and actual outcomes;
`solids` and `liquid` retain complete native receipts and proof partitions.

## Combined evidence at this root

The supplemental proofs resolve specific mathematical questions that the
original physical audit did not assess. Its generic `remaining_requirements`
and source `metal_selection.status=unresolved` remain unchanged historical
fields; they do not mean that the declared 20-model and liquid bounds below
are still missing.

| Evidence | Established scope at this source |
| --- | --- |
| Second liquid and dissolved H2 | Verified lower bound 0 over all 12 supported liquid components and all nonnegative additional ideal-H2 amounts. |
| 20 competing solution models | Strict positive bounds over every declared site and ordering domain, including signed basis coordinates. The count includes the native Fe-only alloy faces. |
| Source Fe–Si–O–H alloy | Verified positive curvature throughout the imposed hard domain D gives an exact nonnegative self-tangent plane and excludes arbitrary splits within D. The stored external-element-potential insertion retains its numerical tolerance. |
| Remaining 13 native pure phases | Every saved insertion value is positive; the minimum is tridymite, 0.39936549021190515 RT/mol atoms. These phases have no composition search, but no strict outward-rounded sign receipt is attached. |
| Registered 35-gas/26-cloud atmosphere | Source, basal parcel and all 16 layers pass the numerical KKT audit for the ideal-gas/pure-cloud convex scalar. At the source, 17 clouds are temperature-eligible and 9 are excluded by the declared temperature limits. This is numerical KKT evidence, not an interval certificate or a physical absence claim for excluded clouds. |
| Empirical applicability and omitted transfer | The common material domain and independent cross-phase calibration remain unestablished. Background gas/cloud effects and Mg/Al/Ca/K/Ti/Cr/P transfer into the deep alloy retain separate response requirements. |

The 13 pure phases are sphene, aenigmatite, muscovite, quartz, tridymite,
cristobalite, corundum, sillimanite, rutile, perovskite, whitlockite, apatite
and native water. The original unavailable kalsilite incipient trial is
superseded only for the declared expression by its positive global bound;
native uniform equivalence remains a separate question. Native alloys and
water retain their own provider and standard conventions.

The source gas stationarity residual is at most 4.974e-14 RT, with no absent
eligible-cloud violation. The source's smallest eligible-cloud insertion is
Fe(s,l), 0.03300840402747607 RT. Across the 16 layers, the maximum gas and
present-cloud residuals are 3.411e-13 and 1.137e-13 RT; absent-cloud violation
is zero. Convexity makes exact catalog KKT sufficient for global optimality;
the saved results apply their recorded numerical tolerances. The combined
evidence is not relabeled a strict shared-potential certificate for the
entire coupled system. Neither native-wide equivalence nor empirical
acceptance follows from it.

Both proof directories preserve the same [physical-audit input](solids/source_physical_audit.json),
SHA256 `1a590ea1128dea2397da942b463da9f01d6676e721489708dae4c6c9195a5ac9`.
It selects run 0, root 0, 16 layers of closure SHA256
`f64e117c5d4c1b32f2de10516d76c699a9a3395e50001b169218e47d4db5adf8`.
The later [catalog-aware physical audit](https://github.com/HajimeKawahara/exoinventory/blob/e9806e9dd1e61c640d0c12b1d65b0c9b7732b720/examples/subneptune_taxonomy/finite_melt/validation/20260927_m2_finite_response/expanded_catalog_audit/matched_m1_physical.json),
SHA256 `6cb55f945a7edb8c5f4ca292506fe44f0885d5368461383f73e89c62c5159f01`,
retains that same source and reconstructed ledger. Its host T/P, component
amounts, dissolved H2, mixing declaration, standards and potentials match
the proof inputs exactly; it is a separate audit, not a rerun of the proof.
Each other closure endpoint needs evidence matched to its own formal inputs.
The separately archived 250-bar expanded-catalog checks cannot be substituted
for a reclosed endpoint merely because its conditions are nearby.
