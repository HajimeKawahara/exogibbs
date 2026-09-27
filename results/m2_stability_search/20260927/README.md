# Whole-catalog and local liquid stability at a saved BSE host

These are new property/search evaluations of the unchanged accepted 16-layer
`H = 1e24 mol atoms` source at **2173.15 K, 267.20283416157343 bar**.
The source pressure and equilibrium were not recomputed. Its original closure
SHA256 is `6b1bdd16ec83f5eeb4f29fe57d55f8f8c37d7167e4f7858525203cc342460940`.
The [physical-audit input](https://github.com/HajimeKawahara/exoinventory/blob/06077dd4d401bd611d69ae308c2c26254f4ff3d8/examples/subneptune_taxonomy/finite_melt/validation/20260925_m2_readiness/physical/physical_reactions_h1e24.json)
is preserved byte for byte in both runs.

The initial run uses ExoGibbs `38c107a` and ExoEOS `4bee4e1`, with whole-simplex
geometrical probes and oxide candidate requests. It exposed roundoff-negative
inferred native coordinates at exact-zero endpoints and impossible Mn/Ni/Co
consumption in the unconstrained olivine interior. The corrected run uses
ExoGibbs `55fcf66` and ExoEOS `44b9778` ([coordinate API PR](https://github.com/HajimeKawahara/exoeos/pull/30)).
It requests explicit native endmember amounts and searches the face generated
by individually feasible endmembers. Both complete raw runs, including every
failure, remain below; no old result is relabeled as a new model calculation.
The initially untracked runner is byte-identical to the runner committed in
`55fcf66`; each execution records its own runner SHA256 and actual model HEAD.

All **33 candidates** have a finite fresh trial after the correction. Every
requested center/vertex/edge-midpoint probe was attempted before optimization.
The 13 one-endmember phases have complete *composition enumeration*, including
a fresh repeat. This is not an interval-certified energy or a global assemblage
stability certificate. The remaining 20 phases are unresolved solution models,
including reduced one-vertex Fe-only faces of the native Fe-Ni alloys. No
negative feasible witness was found. The native alloys and water are alternative
constitutive models, not replacements for the separately selected alloy/gas.

| Measure | Initial oxide requests | Explicit native coordinates |
| --- | ---: | ---: |
| Successful insertion evaluations | 839 | 1097 |
| Failed evaluations retained | 303 | 31 |
| Candidates with no finite fresh trial | olivine, rhm-oxide | none |
| Olivine best fresh insertion, RT/mol atoms | unavailable | 0.0847993423 |
| Rhm-oxide best fresh insertion, RT/mol atoms | unavailable | 0.8806916941 |
| Fixed-composition candidates enumerated | 13 | 13 |

The 31 remaining failed evaluations are actual nonfinite native endpoint or
edge properties. They are not replaced by zero, extrapolated, or treated as
phase absence. The admissible model may extend beyond the nonnegative native
endmember simplex; excluded signed combinations are not certified absent.

| Candidate | Included/native endmembers | Best fresh insertion, RT/mol atoms | Failed evaluations | Coverage meaning |
| --- | ---: | ---: | ---: | --- |
| olivine | 3/6 | 0.0847993423 | 0 | unresolved solution |
| sphene | 1/1 | 1.26274697 | 0 | single composition |
| garnet | 3/3 | 0.343661696 | 0 | unresolved solution |
| melilite | 4/4 | 0.567255499 | 0 | unresolved solution |
| orthopyroxene | 7/7 | 0.150473993 | 3 | unresolved solution |
| clinopyroxene | 7/7 | 0.150403882 | 3 | unresolved solution |
| aenigmatite | 1/1 | 0.778741859 | 0 | single composition |
| cummingtonite | 2/2 | 0.420677339 | 0 | unresolved solution |
| clinoamphibole | 3/3 | 0.419424873 | 4 | unresolved solution |
| orthoamphibole | 3/3 | 0.427376062 | 4 | unresolved solution |
| hornblende | 3/3 | 0.501274252 | 5 | unresolved solution |
| biotite | 2/2 | 0.674200204 | 0 | unresolved solution |
| muscovite | 1/1 | 1.03239247 | 0 | single composition |
| alkali-feldspar | 3/3 | 0.441910579 | 0 | unresolved solution |
| plagioclase | 3/3 | 0.420835609 | 0 | unresolved solution |
| quartz | 1/1 | 0.461376141 | 0 | single composition |
| tridymite | 1/1 | 0.39936549 | 0 | single composition |
| cristobalite | 1/1 | 0.438580682 | 0 | single composition |
| nepheline | 4/4 | 0.551581124 | 6 | unresolved solution |
| kalsilite | 4/4 | 0.573901959 | 6 | unresolved solution |
| leucite | 3/3 | 0.429367417 | 0 | unresolved solution |
| corundum | 1/1 | 0.854602657 | 0 | single composition |
| sillimanite | 1/1 | 0.707609929 | 0 | single composition |
| rutile | 1/1 | 1.90485301 | 0 | single composition |
| perovskite | 1/1 | 1.69893305 | 0 | single composition |
| spinel | 5/5 | 0.308752557 | 0 | unresolved solution |
| rhm-oxide | 4/5 | 0.880691694 | 0 | unresolved solution |
| ortho-oxide | 3/3 | 1.55611092 | 0 | unresolved solution |
| whitlockite | 1/1 | 0.651512989 | 0 | single composition |
| apatite | 1/1 | 0.83740731 | 0 | single composition |
| water | 1/1 | 11.7821113 | 0 | single composition |
| alloy-solid | 1/2 | 0.230602784 | 0 | unresolved solution |
| alloy-liquid | 1/2 | 0.0453655659 | 0 | unresolved solution |

The local liquid test covers **12 independent composition directions** on the
13-positive-component host/H2 support. The smallest scaled projected Hessian
eigenvalue is **0.6757233771**, compared with a step/symmetry/homogeneity sensitivity
margin of **2.48623e-6**. All 59 native property evaluations are stored. Three
finite two-liquid splits along the least-curved direction give positive
energy changes (`2.25990e-9`, `2.25991e-7`, `2.26068e-5` RT/mol atoms), preserving
every component exactly to the recorded precision. This supports local numerical
convexity, not global stability against a distant liquid composition. The
sensitivity margin is not a rigorous Hessian error bound.

`audit_native_archive.py` independently recounts every successful insertion
from native G, supplied host potentials, oxide/element matrices and the H2
dilution term. It also recounts the finite daughter energies and component
conservation. Both runs pass: maximum energy discrepancy `1.42e-14 RT`, maximum
finite-split component residual zero. Actual subprocess exit codes were zero.
Full outputs, source hashes, runtime/provider metadata, protocols, stdout,
process receipts, and validation results are retained. `manifest.json` pins
all archived bytes except itself.

Global stability, empirical model applicability, a shared material calibration
domain, and omitted-transfer error bounds remain unestablished. These results
are evidence about the declared formal model at this one preserved host, not
a BSE physical phase boundary or a new globally closed planetary solution.
