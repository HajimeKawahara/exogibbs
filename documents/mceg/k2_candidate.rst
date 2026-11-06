K2 measured-energy candidate and thermal audit
==============================================

Prefer the measured principal-isotope D0 as a bounded central energy update, subject to its explicit aggregate-isotope convention; retain current thermals. The2026potential corroborates the anchor but does not establish a complete high-temperature thermal replacement.
This review performs an actual fixed-network substitution without changing
production data or the original source-review packet.

Physical energy and invariant thermal functions
-----------------------------------------------

The complete Pashov et al. (2006) article in the author thesis,
`10.1063/1.2375120 <https://doi.org/10.1063/1.2375120>`_, gives
D0 = 4404.808 +/- 0.004 cm^-1 for 39K2, relative to the center of gravity
of the atomic ground asymptote. It is a ground-vibrational binding energy,
not the potential well depth. The source and candidate keep the existing
aggregate thermal convention; they do not turn this principal-isotope
measurement into a measured natural-isotope mean or propagate its tiny
experimental uncertainty to all thermodynamic functions.

Using exact SI h, c and Avogadro's constant, the measured D0 is
52693.205333786 J/mol, compared with the physical JANAF source value
52661 J/mol. Therefore deltaD0 is
+32.205333786 J/mol and the candidate changes both formation G and H by
-32.205333786 J/mol. S and Cp are identically unchanged::

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
The pinned state file has eight electronic inputs and no shallow a3Sigma-u
state. The G/F/E/D entries have unspecified electronic labels and degeneracy
metadata, adding a separate completeness limitation. Tabulated Q already divides out the product
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

The official supplement for Brakmane et al. (2026),
`10.1063/5.0343162 <https://doi.org/10.1063/5.0343162>`_, is now obtained from
`the publisher dataset <https://doi.org/10.60893/figshare.jcp.32907533.v1>`_.
Its README, MLR function, potential grid and S10 levels were inspected.
S10 lists 86 bound **J = 0** vibrational levels for 39K39K, with
De = 4450.903 cm^-1 and E(v=0,J=0) = 46.09437349 cm^-1.
Independent subtraction gives D0 = 4404.80862651 cm^-1, only
0.007494724 J/mol above the2006 anchor and within its reported uncertainty.
The largest listed two-grid energy difference is9.095e-12 cm^-1;
this numerical convergence is not an experimental accuracy claim.

The selected control uses that2026D0 with unchanged thermals. At the maximum
allocation condition its mole-fraction ratio to the original is1.0025862122,
compared with1.0025856097 for the main candidate. This agreement corroborates
the chosen energy and does not depend on constructing a new hot partition sum.

The unweighted J0 vibrational sum is13.29480458 at1500K and34.64012301 at4000K;
levels v>=70 supply1.8015% and9.5543%, respectively. These are explicitly
**vibrational J0 diagnostics**, not full molecular partition functions.
Multiplying by a rigid rotor would ignore J-dependent dissociation and
centrifugal effects. Other electronic states, minor isotopologues,
quasibound/continuum treatment and thermal convergence are not supplied by
this86-level list. Accurate fitted spectroscopic lines do not by themselves
establish a complete thermodynamic state inventory.

Actual substitution and numerical resolution
--------------------------------------------

There are150 fresh endpoints on curated-ch2-v1:75 ion-inclusive537-species and
75 neutral421-species cases, plus12 tighter cold solves,10 path nodes,
4 selected-energy controls and4 independent FastChem runs at two ion-inclusive
conditions. The150 reused baseline endpoints are verified against hashes,
actual named-factory potentials, species, formula matrix, budgets and conditions.
Only K2 changes. Every element, charge and atomic formation closure is checked.

The ion-inclusive mole-fraction ratio is1.0009688218--1.0025856112.
The largest response is at1500K,100bar,C/O1.5. Maximum K allocation is
0.00002775497% -> 0.00002782673%; its mole fraction is
2.803540819e-14 -> 2.810789682e-14.
These are solved network responses, distinct from the fixed-atomic-activity
Boltzmann factor. Independent FastChem agrees within1.05e-08dex.
Same-model Gibbs exchanges resolve above the recorded diagnostic scale in
0/75 ion-inclusive and0/75 neutral cases. The remaining comparisons
are unresolved because the response is small; they do not justify either a
false sign claim or a physical source ranking. The diagnostic scale is not a
rigorous optimizer error bound.

Decision, evidence and finite next step
---------------------------------------

Prefer the measured principal-isotope D0 as a bounded central energy update, subject to its explicit aggregate-isotope convention; retain current thermals. The2026potential corroborates the anchor but does not establish a complete high-temperature thermal replacement.
The measured energy is preferable to the old source central value because it
is a direct ground-state binding measurement with an explicit asymptote, not
because the small equilibrium response proves complete physical accuracy.
The thermal follow-up needs a normalized coupled state inventory or corrected
continuous caloric data, with explicit treatment of the omitted electronic,
rotational and near-dissociation contributions. Existing source uncertainty,
full reaction covariance and atmospheric observations remain unestablished.
Missing ancestral papers alone are not an unlimited merge gate.

``mceg/contracts/k2_candidate.yaml`` pins the source facts, continuous checks
and execution archive. Offline replay checks input hashes through the historical
resolver, direct output hashes, baseline identity and all endpoint/control
closures. MCEG supports the scoped energy/numerical/decision claims while
retaining unknown full physical accuracy, universal Gibbs resolution and
production adoption. Runtime and older source records remain unchanged.
