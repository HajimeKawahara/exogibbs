CO2 adoption in curated-co2-v1
==============================

``curated_co2.chemsetup()`` fixes the reviewed NIST 626 CO2 candidate as the
versioned ``curated-co2-v1`` model. The public ``curated`` factory selected this
model at adoption. Only C1O2 changes relative to ``curated-tio-v1``; both the
537-species ion-inclusive and 421-species neutral networks retain their
species, elemental budgets, standard pressure of 1 bar and **1500--4000 K**
domain. The earlier ``curated_tio``, ``curated_c2h`` and ``curated_v1`` factories
remain available for reproducible comparisons. The current public default
is described in :doc:`curated_h2o`; use ``curated_co2.chemsetup()`` to
reproduce the CO2-only adoption recorded here.

Scientific decision
-------------------

We adopt the explicitly normalized :doc:`NIST 626 candidate
<../mceg/co2_candidate>`. The inspected thermodynamic evaluation by Tashkun and
Harvey, `DOI 10.1063/5.0276615 <https://doi.org/10.1063/5.0276615>`__, and its
`tabulated dataset, DOI 10.18434/mds2-2364
<https://doi.org/10.18434/mds2-2364>`__, provide an effective-Hamiltonian
rovibrational model, thermodynamic derivatives and parameter uncertainty.
The distributed Q, H and Cp satisfy the relevant identities; the printed
variance sign issue is distinguished from the internally consistent tables.
This is a more directly justified thermodynamic source than choosing a recent
opacity partition file by its release date. The source review also identifies
the high-state truncation behind the current UCL4000 PF discrepancy.

The adoption keeps the balanced ATcT 1.172 atomization energy of
1598.180 kJ mol\ :sup:`-1` and the original NASA neutral-atom thermal reference.
The change is thermal, not a new estimate of the ground-state binding energy.
No additional historical paper is required to make this scoped choice;
remaining source gaps and the incomplete reaction covariance remain recorded
in :doc:`../mceg/co2_source_review`.

The representation uses the lowest :sup:`12`\ C\ :sup:`16`\ O\ :sub:`2`
rovibrational level as its internal zero, nuclear-spin factor one, and the
existing elemental molecular mass 44.0095 u for translation. Allowed-level
exchange symmetry is already included. No extra symmetry division, ZPE or
mixing entropy is appended. NIST's source-native c2 = 1.43877736 K cm is
preserved in its internal moments; translation and caloric units use current
SI constants and R. NASA's historical R = 8.314510 is retained when recovering
its atomic energy zero. These distinct conventions are handled explicitly.

The fixed natural-isotope thermal-average diagnostic is **not** adopted.
Its geometric-mean Q combines isotope-relative thermal functions at a common
translation mass; it does not include isotope-specific atomization energies,
atom/molecule mixing conventions or equilibrium fractionation. Its roughly
0.43--0.46% change in K is a reference-convention sensitivity, not a physical
correction or uncertainty bound on the representative 626 model.

Size and limits of the change
-----------------------------

The CO2 formation constant increases by 0.0651% at 1500 K and 0.7913% at
4000 K. On the original hydrogen-rich 75-case grid, the largest absolute CO2
mole-fraction increase is only 1.505e-10 because CO2 is trace there. In the
additional hydrogen-poor example at 4000 K and 100 bar, it increases from
0.1345217150 to 0.1351219572: +0.4462%, or +0.00060024 in absolute fraction.
The response therefore depends on elemental conservation and competition
with other species; a change in K is not generally the same abundance change.

In that example, perturbing Q by its quoted one-standard-uncertainty moves
CO2 by approximately +/-0.0567%, smaller than the central adoption shift.
This comparison concerns the source's parameter covariance only. It does not
bound continuum, electronic-state, isotope or high-state systematic errors.
The fixed-isotope-average diagnostic changes the same abundance by +0.2569%,
which illustrates why it must remain separate from complete natural-isotope
chemistry.

The source warns that non-dissociating pure-CO2 heat capacities above about
3500 K become unphysical for the reacting bulk gas. Here the function belongs
to one bound molecular species; the equilibrium network also permits CO, O
and other dissociation products. This distinction permits a bounded species
model without claiming that the warning disappears. The source cutoff,
missing continuum and complete high-temperature uncertainty remain limitations.
No lower-temperature extrapolation, condensation, atmospheric transport,
observational accuracy or ideal-gas validity at every pressure is established.

Implementation and verification
-------------------------------

``co2_v1.json`` copies the reviewed candidate's eleven knots and quintic Gibbs
coefficients exactly, without refitting. One C2 representation of (G-H0)/R
provides H, S and Cp by differentiation. Runtime is JAX-only and reads pinned
packaged data; it needs neither MCEG nor the review scripts. Nonfinite and
out-of-range temperatures return NaN, and float64 is required for the reviewed
numerical precision.

Independent comparison against all 2501 integer-temperature NIST rows gave
maximum errors of 2.38e-8 in ln K, 0.00758 J/mol in H, 4.46e-6 J/(mol K) in S
and 0.000200 J/(mol K) in Cp. Runtime tests recover molecular calorics using
separately evaluated NASA atomic references and automatic differentiation;
this also tests the atom-energy gauge rather than only comparing coefficients.

The production execution cold-solved **150 cases**, 75 per network, through
the public default at CO2 adoption: five temperatures (1500, 2000, 2500, 3000, 4000 K),
five pressures (1e-6, 1e-4, 0.01, 1, 100 bar), and three C/O ratios
(0.5888436554, 1, 1.5). All converge with ``epsilon_crit=5e-14``. Maximum
relative elemental residuals are 1.353e-6 with ions and 1.347e-6 without ions,
dominated by trace species; the maximum mass-action residual is 1.137e-12
in natural-log units and relative charge residual is 4.533e-14. The CO2
mole-fraction logarithm differs from the reviewed candidate by less than
3.72e-12. The all-species maximum, 2.538e-5, reflects numerical sensitivity in
trace species and is not a CO2 response or source uncertainty.

The candidate's tight-solver, same-model Gibbs, hydrogen-poor and independent
FastChem checks remain earlier evidence; they are not claimed as new production
runs. Production checks establish implementation and numerical closure.
Scientific preference is supported within scope; complete physical accuracy
remains unknown in MCEG.

Rebuild and inspect
-------------------

Rebuild the immutable payload offline::

   python mceg/reviews/build_co2_adoption.py --output /tmp/co2_v1.json

The production packet is
``tests/data/validation/curated_co2_adoption/20261104``; its receipt pins the
CO2 default at adoption and every saved numerical output. The ``source`` subdirectory
preserves prior TiO-default source bytes so that its older receipts and
assessments remain independently auditable. The source and candidate packets
remain unchanged. Production claims are in
``mceg/contracts/curated_co2.yaml`` and its packaged mirror; the earlier source
contract records completion of the production-adoption follow-up.

The replay script selects ``curated_co2`` explicitly after later default changes.
Its original default-dependent source and this original guide are preserved in
``tests/data/validation/curated_h2o_adoption/20261104/source``; historical
receipt hashes and numerical results are unchanged.
