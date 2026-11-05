N2: retain NASA thermal functions and record the updated energy
===============================================================

The neutral ``N2`` gas reaction in ``curated-cos-v1`` is NASA CEA thermal
functions plus the balanced ATcT 1.172 atomization energy. This review
reproduces that construction and retains it. The newer evaluated energy
is suitable for a small future candidate; the retrieved ExoMol distribution
needs a version and completeness check before replacing the thermal model.
This gas review is separate from the N2 melt-solubility contract.

Energy and historical sources
-----------------------------

The current D0 is 941.166 kJ/mol. The inspected
`ATcT 1.228 evaluation <https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_
sets N2 Hf(0)=0 exactly and N(4S) Hf(0)=470.655 kJ/mol, giving
D0=941.310 kJ/mol. The +0.144 kJ/mol update alone changes formation K by
+1.1613 percent at 1500 K, +0.5790 percent at 3000 K and +0.4339 percent
at 4000 K. N2 being the elemental reference does not make its balanced
atomization energy exact. The reported N margin is 0.021 kJ/mol; molecule
and atoms must come from the same network, without treating network
correlations as independent measurement errors.

The official nitrogen provenance includes
`Wang et al. (2024) <https://doi.org/10.1063/5.0187003>`_. Its measured
N(2D)+N(2D) channel must be converted to ground N(4S)+N(4S); the excited
threshold is not the required ground-state D0. The official ATcT records
and bibliography were inspected; that paper's full text was not obtained.
The evaluated network value is the energy input here, not a new extraction
of its individual experiments.

NASA's literal header cites Gurvich (1978), part 1 page 280 and part 2
page 207. The original book inputs have not been independently recovered.
`JANAF N-023 <https://janaf.nist.gov/tables/N-023.txt>`_ is compared with
its matching atomic N table, rather than treating the tabulated elemental
formation energy of zero as a molecular association constant.

.. list-table:: Dimensionless atom-to-molecule formation-K ratios
   :header-rows: 1

   * - Temperature (K)
     - Current/native NASA
     - JANAF/native NASA
     - Energy-aligned JANAF/native NASA
   * - 1500
     - 0.963488
     - 1.001378
     - 1.000405
   * - 3000
     - 0.981574
     - 1.000842
     - 1.000356
   * - 4000
     - 0.986149
     - 1.000801
     - 1.000436

Native NASA and JANAF balanced D0 values are 941.635248 and 941.642 kJ/mol.
The small aligned difference supports retaining the established thermal
functions; it is not a complete uncertainty estimate.

Modern partition data and distribution limits
---------------------------------------------

The ExoMol WCCRMT entry covers triplet electronic systems and is not a
complete ground-state thermochemical source. The ground-state TRINITY work
is now formally published as
`Clark-Brown et al. (2026) <https://doi.org/10.1093/mnras/stag1791>`_.
The landing page still says provisional/submitted; that metadata must not
be mistaken for the paper's publication status. Publisher sections on
state filtering and partition functions and its state table were inspected.

The retrieved principal-isotope states file contains 810 levels, v=0--9
and J=0--80. The published table includes state 891 with v=10. The distributed
partition table also differs from a direct sum of these downloaded states
by 0.0438 percent at 4000 K. Thus the paper's broader completeness statement
cannot simply be assigned to this particular file pair.

For 14N2 the even/odd J nuclear weights are 6/3. Divide the state sum by 9
to remove nuclear spin; homonuclear exchange symmetry is already included,
so no additional factor of two is applied. The finite-sum Q/NASA ratios
at 1500/3000/4000 K are 1.000202/0.998026/0.989407. At 4000 K the
finite-state thermal H, S and Cp differ from NASA by -1.750 kJ/mol,
-0.531 J/mol/K and -2.156 J/mol/K. Exact state moments, rather than
derivatives of rounded Q, produce these diagnostics. The isotope mass
correction gives a fixed-atom K proxy of 0.989023 at 4000 K; this is not
a natural-isotope mixture model. Missing high-J and vibrational levels
remain a concern even when the final included shell has a small weight.

Scope, decision and follow-up
-----------------------------

The archived hot-gas grid has 75 converged cases and 537 species. N2 carries
up to 99.99979 percent of the N inventory (1500 K, 1e-6 bar,
C/O=0.588844); its maximum mole fraction is 5.80171e-5. These are existing
``curated-cos-v1`` endpoints, not a replacement response or evidence about
cooler, oxidizing or photochemical atmospheres.

Retain curated in this evidence-only review. An ATcT 1.228 energy-only
candidate and fixed-network substitution can be built from the pinned
values without acquiring another paper. A thermal candidate additionally
needs a reconciled TRINITY release, converged Q/H/S/Cp including relevant
electronic states, and an explicit isotope convention. Historical book
recovery is optional provenance work and does not block merging this review.

All comparisons use one bar, balanced neutral atoms and a common ground
zero. NASA retains its native R=8.314510 J/mol/K; SI tables and production
use R=8.31446261815324. Each source zero is Hf(298.15)-[H(298.15)-H(0)],
including the atomic reference increments. The replay separates energy,
molecular thermal and atomic thermal terms and tests their closure.
MCEG supports attribution, normalized comparison and this scoped decision;
complete physical accuracy and historical recovery remain unknown.

Run ``python mceg/reviews/n2_source_review.py``. Literal selected inputs,
finite states, output and hash receipt are under
``tests/data/validation/n2_source_review/20261106``. No candidate fit,
new equilibrium solve or curated coefficient change is included.
