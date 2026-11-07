Na2 measured-energy candidate and thermal audit
===============================================

Prefer the measured D0 energy-only candidate for a separately authorized bounded update; retain the current thermal function until a complete normalized modern construction has been validated.
This review performs an actual fixed-network substitution without changing
production data or the original source-review packet.

Physical energy and invariant thermal functions
-----------------------------------------------

The primary publisher abstract of Jones et al. (1996),
`10.1103/PhysRevA.54.R1006 <https://doi.org/10.1103/PhysRevA.54.R1006>`_,
reports D0 = 5942.6880 +/- 0.0049 cm^-1 for 23Na2 from combined direct
spectroscopic measurements. This is the lowest vibrational level relative
to the free ground-atom limit, not De. The original full paper has not been
obtained; that access limit is recorded rather than converted into a claim
of a complete modern thermal evaluation. Sodium is naturally monoisotopic
for this construction. The selected control moves D0 upward by the reported
0.0049 cm^-1 measurement uncertainty. It is one anchor sensitivity, not a
confidence interval for the retained thermal function or atmosphere.

Using exact SI h, c and Avogadro's constant, the measured D0 is
71090.335610230 J/mol, compared with the physical JANAF source value
70539 J/mol. Therefore deltaD0 is
+551.335610230 J/mol and the candidate changes both formation G and H by
-551.335610230 J/mol. S and Cp are identically unchanged::

    Gcandidate(T) = Gcurrent(T) - (D0(measured) - D0(JANAF))

The literal inherited fit, atomic-reference gauge and complete fit residual
are preserved. The fit intercept is recorded separately and is not the physical
source D0. No ZPE, arbitrary H298 offset, symmetry or nuclear-spin term is added.
Dense scalar/JAX checks cover2752 integer/off-grid temperatures for both main
and selected control potentials. Agreement with the actual curated-ch2-v1
named factory is checked before replacement.

Modern thermal inputs and remaining completeness
------------------------------------------------

Barklem and Collet `10.1051/0004-6361/201526961 <https://doi.org/10.1051/0004-6361/201526961>`_
and its author2022 fixes remain useful node diagnostics. The correction changes
atomic partition/equilibrium tables; it does not replace the molecular Q table.
The pinned state file has eight electronic inputs. Its a3Pi label is not the
missing shallow a3Sigma-u state. Tabulated Q already divides out the product
of nuclear-spin weights; another division by16 or rotational symmetry2 would
be incorrect. Only six nearby table temperatures (1000,1500,2000,3000,4000,
5000K) are available, which do not independently validate continuous G/H/S/Cp.
No spline of those points is adopted as a new physical thermal model.

The official supplements to
`the2025thermal paper <https://doi.org/10.1007/s10765-025-03628-z>`_ are now
inspected: the ODS contains potential parameters on four element sheets,
without tabulated thermochemical or complete rovibrational states. The PDF
explains bound-state and electronic-coverage limitations. The
`correction <https://doi.org/10.1007/s10765-025-03639-w>`_ replaces ground-state
Figures2--7; its publisher-provided first-page preview was read, while complete
correction pages remain unavailable. Figure replacement does not establish
that every coefficient or thermal value was revised. The source pressure is
1atm, so any later use of absolute entropy requires S(1bar)=S(1atm)+Rln(1.01325).
No2025thermal data or pressure conversion is inserted into this energy-only
candidate.

No inspected source supplies a corrected, complete modern continuous
Na2 thermal function ready for adoption. The present energy-only candidate
therefore does not manufacture one from sparse partition-function nodes or
from an incomplete electronic-potential inventory.

Actual substitution and numerical resolution
--------------------------------------------

There are150 fresh endpoints on curated-ch2-v1:75 ion-inclusive537-species and
75 neutral421-species cases, plus12 tighter cold solves,10 path nodes,
4 selected-energy controls and4 independent FastChem runs at two ion-inclusive
conditions. The150 reused baseline endpoints are verified against hashes,
actual named-factory potentials, species, formula matrix, budgets and conditions.
Only Na2 changes. Every element, charge and atomic formation closure is checked.

The ion-inclusive mole-fraction ratio is1.0167157766--1.0451986414.
The largest response is at1500K,100bar,C/O1.5. Maximum Na allocation is
0.00117245757% -> 0.00122544975%; its mole fraction is
1.672874107e-11 -> 1.74848387e-11.
These are solved network responses, distinct from the fixed-atomic-activity
Boltzmann factor. Independent FastChem agrees within1.23e-08dex.
Same-model Gibbs exchanges resolve above the recorded diagnostic scale in
12/75 ion-inclusive and12/75 neutral cases. The remaining comparisons
are unresolved because the response is small; they do not justify either a
false sign claim or a physical source ranking. The diagnostic scale is not a
rigorous optimizer error bound.

Decision, evidence and finite next step
---------------------------------------

Prefer the measured D0 energy-only candidate for a separately authorized bounded update; retain the current thermal function until a complete normalized modern construction has been validated.
The measured energy is preferable to the old source central value because it
is a direct ground-state binding measurement with an explicit asymptote, not
because the small equilibrium response proves complete physical accuracy.
The thermal follow-up needs a normalized coupled state inventory or corrected
continuous caloric data, with explicit treatment of the omitted electronic,
rotational and near-dissociation contributions. Existing source uncertainty,
full reaction covariance and atmospheric observations remain unestablished.
Missing ancestral papers alone are not an unlimited merge gate.

``mceg/contracts/na2_candidate.yaml`` pins the source facts, continuous checks
and execution archive. Offline replay checks input hashes through the historical
resolver, direct output hashes, baseline identity and all endpoint/control
closures. MCEG supports the scoped energy/numerical/decision claims while
retaining unknown full physical accuracy, universal Gibbs resolution and
production adoption. Runtime and older source records remain unchanged.
