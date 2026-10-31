Phosphorus source provenance and default policy
===================================================

The P4O6 discrepancy comes primarily from competing historical energy
evaluations. There is a defensible provisional preference for the
Gurvich-derived P4O6 record used by the inspected Burcat snapshots. This is
a species-specific recommendation, not approval of the entire Burcat
database or validation of cold atmospheric phosphorus chemistry.

This review changes no production constants, preset identifiers or
temperature limits. The existing curated-gas-v1 already inherits the
FastChem4 P4O6 choice, with status ``inherited_unreviewed``. A documented
scientific preference is distinct from a released adoption decision.

Where the difference originates
-----------------------------------

The official `JANAF P4O6 gas table
<https://janaf.nist.gov/tables/O-087.html>`_ gives a formation enthalpy of
-2214.307 kJ/mol at 298.15 K. Its evaluation date is **December 1962**;
inclusion in the 1998 volume does not make it a 1998 reevaluation.
The `evaluation notes, printed p. 1803
<https://janaf.nist.gov/pdf/JANAF-FourthEd-1998-Oxygen.pdf>`_ trace the
liquid formation enthalpy to Koerner and Daniels (1952), who burned red
phosphorus in nitric oxide, and add Farr's (1950) vaporization enthalpy
of 11.46 kcal/mol, or 47.94864 kJ/mol, to obtain the gas value.

The `Hartley and McCoubrey (1963) publisher abstract
<https://www.nature.com/articles/198476a0>`_ questions whether the older
combustion products had been conclusively identified as P4O6 and P4O10.
That is a concrete experimental concern. The original full calorimetric
papers were not accessible here; this review does not reconstruct a
contamination fraction or an exact correction to the measurements.

The inspected `Burcat entry
<https://respecth.elte.hu/burcat/BURCAT.THR.txt>`_ instead selects
**Gurvich (1989), -1606 +/-40 kJ/mol**, and explicitly rejects the JANAF
anchor. The official `NASA polynomial compilation
<https://ntrs.nasa.gov/api/citations/20020085330/downloads/20020085330.pdf>`_,
printed pp. 27 and 172, selects the same value and identifies Gurvich
part 1 p. 413 and part 2 p. 266. The original Gurvich pages remain
uninspected. Agreement between NASA CEA and Burcat is therefore shared
ancestry, not two independent measurements.

`Wang et al. (2016), section 4.2
<https://arxiv.org/pdf/1604.06371>`_ connect the Gurvich branch to the
1963 reevaluation and discuss the earlier disagreement over which
experiment to trust. This link is a later author's account; it does not
replace direct inspection of Gurvich's evaluation.

The pinned GGChem archive contains actual Burcat files dated 5 October
2016 and 3 January 2023. Their selected NASA7 coefficients for P4O6,
H3PO4 and HPO2/HOPO match each other and the inspected public mirror.
This narrows the provenance gap, but does not recover the exact file,
atomic references, grid and weights used to fit FastChem's coefficients.
An older thermochem XML snapshot instead uses the Lewis-family P4O6
value near -2144.5 kJ/mol. A database name alone is not a source identity.

Independent evidence and its limits
---------------------------------------

.. list-table:: P4O6 gas formation enthalpy near 298 K
   :header-rows: 1
   :widths: 30 20 50

   * - Source
     - kJ/mol of P4O6
     - Nature of the evidence
   * - JANAF, evaluated 1962
     - -2214.307
     - Older liquid calorimetry plus vaporization correction
   * - Gurvich through inspected Burcat/CEA
     - -1606
     - Evaluated anchor; Burcat reports +/-40, not verified here as an error distribution
   * - Morgon (2012)
     - -1706.3
     - Modified correlation-consistent composite calculation; abstract inspected
   * - Lee et al. (2024)
     - -1659
     - Independent G4 calculation; Table 2 visually inspected

`Morgon (2012) <https://doi.org/10.1002/qua.24178>`_ and
`Lee et al. (2024), Appendix A.2/Table 2
<https://arxiv.org/html/2409.06802v2>`_ support a substantially less
negative energy than JANAF. They differ from Gurvich by roughly 100 and
53 kJ/mol, respectively: they do not establish that -1606 is exact.
Lee's separate formation Gibbs energy is -1531 kJ/mol; it must not be
confused with the -1659 enthalpy. Network adoption in that paper is not
an independent measurement of the adopted database values.

The `Smoes and Drowart (1973) abstract
<https://doi.org/10.1039/FS9730800139>`_ reports a P4O6 atomization energy,
which requires atomic and thermal reference conversions before comparison
with formation enthalpy. Those conversions were not independently
reconstructed here. `Bains et al. (2023)
<https://arxiv.org/pdf/2305.17405>`_ also argue against JANAF's strong
stabilization while calling for better measurements. Their numerical
reconstruction is not adopted as a calibration: the preprint's phase
labels and entropy-of-formation usage need clarification against JANAF's
explicit liquid origin and absolute entropy column.

Why conventions do not remove this difference
-------------------------------------------------

The two selected anchors refer to gas P4O6 per mole of P4O6, so the
608.307 kJ/mol separation is not a P2O3/P4O6 normalization factor.
The JANAF vaporization increment is only 47.95 kJ/mol. The documented
red-V/white phosphorus correction is about 17.46 kJ per mole of P atoms,
or 69.84 per mole of P4O6; it cannot explain the full separation.
A 1 atm versus 1 bar convention does not change ideal-gas enthalpy.

A consistent elemental reference change also leaves equilibrium invariant.
Let ``A[e,i]`` count atoms, ``b = A n`` be the fixed elemental inventory,
and ``c[e](T)`` be an arbitrary elemental reference shift. If every
species is changed consistently,

.. math::

   g_i' = g_i + \sum_e A_{ei} c_e, \qquad
   G'(n) = G(n) + \sum_e c_e b_e.

The added term is constant over the feasible mixtures, so the minimizing
composition is unchanged. Equivalently, a balanced reaction cancels the
shift because ``A nu = 0``. Inconsistent conversions can create artifacts;
this identity is not permission to combine unconverted source conventions.
The earlier reference-normalized audits and P4O6-only substitution already
establish that the observed source effect survives the reference handling.

H3PO4, HPO2 and P4S3 are distinct questions
-----------------------------------------------

The selected Burcat gas H3PO4 entry cites an Elke Goos G3B3 calculation
(T06/09), and HPO2, structurally HOPO, cites Haworth and Bacskay (2002)
(T11/12). These are computed gas records; their original calculation
outputs/tables were not independently audited. Burcat's liquid and
crystalline H3PO4 entries are different phases and cannot fill a gas
species omission. Including these gases changes species coverage; it
does not validate their energies.

P4S3 uses the same Stock/Kitzmann record in both native comparisons,
with a JANAF source annotation in FastChem4. It is not a changed-source
driver in this experiment. Curated's P-bearing species have mixed
provenance: P4O6 is Burcat-derived, whereas P4O10 and P4S3 have JANAF
annotations. The :doc:`ohno_p3` controls isolate these distinctions.

Recommended software behavior
---------------------------------

* Retain the versioned curated-gas-v1 values and 1500-4000 K evaluation
  domain. The current P4O6 choice is a reasonable provisional continuation;
  do not silently revise old model identities or extend their domain.
* Record a species-specific preference for a pinned Gurvich-derived P4O6
  record, its rationale, alternatives and unresolved accuracy. Preserve
  the JANAF alternative for explicit sensitivity calculations. A future
  released adoption needs an explicit reviewed decision and model version.
* Carry the model identifier, phase, source revision/hash, reference
  convention and applicable interval into calculation provenance.
  Surface the conflict during model inspection, without adding warnings
  inside numerical solver loops.
* Treat the low-temperature network as separate work: review H3PO4/HOPO,
  common source validity intervals and relevant condensed phases before
  offering a production cold preset. The existing research controls
  suffice for source sensitivity; no new runtime switching framework is
  needed for this investigation.
* Do not average incompatible anchors, interpret alternative sources as
  statistical error bars, or tune constants to resemble Ohno's curves.
  Source plausibility, numerical closure and physical accuracy are
  different assessments.

More species, a newer database date and agreement between solvers are not
physical reasons to choose a database wholesale. The positive rationale
here is specific criticism of the old P4O6 experiment together with
independent molecular calculations favoring the less negative branch.

MCEG record and remaining work
----------------------------------

The :download:`source record
<../../mceg/reviews/phosphorus_provenance_20261031.json>` stores inspected
locations, source hashes, numerical anchors, inspection limits and the
provisional selection rationale. The :download:`MCEG contract
<../../mceg/contracts/phosphorus_provenance.yaml>` separates the supported
reference-invariance derivation from unknown physical accuracy. Existing
numerical support remains in :doc:`ohno_p3`; it is not promoted to physical
validation by this follow-up.

The outstanding tasks are to inspect the original Gurvich evaluation,
recover the exact FastChem fit ancestry, establish source-specific cold
validity and phase coverage, and obtain independent physical constraints.
Primary PDFs and coefficient tables remain research downloads under
``results/``; this record contains citations and hashes, not those works.
