NH: separate an obsolete JANAF energy from the current fit
==========================================================

The neutral ground-triplet ``H1N1`` reaction in ``curated-cos-v1`` inherits
the FastChem five-coefficient fit labelled Goos/Burcat (2023). It is not
an adopted JANAF or NASA+ATcT reaction. The large JANAF discrepancy is
mainly an outdated formation-energy anchor. Current curated is already
close to a modern evaluated-energy diagnostic; retain it in this review.

Energy and thermal comparison
-----------------------------

The inspected `ATcT 1.228 evaluation
<https://atct.anl.gov/Thermochemical%20Data/version%201.228/species/?species_number=140>`_
has Hf(0)=358.88 and Hf(298.15)=358.93 kJ/mol for ground-triplet NH,
with a reported 0.14 kJ/mol margin. Same-version N and H Hf(0) values
470.655 and 216.034 give balanced D0=327.809 kJ/mol. The singlet NH
entry is a different electronic reference and must not replace this value.
The network combines multiple experimental and theoretical inputs;
its uncertainties and correlated atom references are not independent.

The NASA header cites Anderson (1989) for Hf and Gurvich (1978), part 2
page 223. Its Hf(298.15)=357.032001 kJ/mol and balanced native
D0=329.853074 kJ/mol. `JANAF H-026
<https://janaf.nist.gov/tables/H-026.txt>`_ instead has
Hf(298.15)=376.560 kJ/mol and balanced D0=310.345 kJ/mol.
Those anchors explain most of the disagreement; comparing unaligned K
alone would misidentify the historical energy problem as a thermal error.

.. list-table:: Dimensionless atom-to-molecule formation-K ratios
   :header-rows: 1

   * - Temperature (K)
     - Current/native NASA
     - JANAF/native NASA
     - Energy-aligned JANAF/native NASA
     - NASA thermal+ATcT 1.228/current
   * - 1500
     - 0.868603
     - 0.209285
     - 0.999977
     - 0.977383
   * - 3000
     - 0.930767
     - 0.455555
     - 0.995789
     - 0.989925
   * - 4000
     - 0.947567
     - 0.549911
     - 0.988580
     - 0.992481

The last column changes both the inherited fit construction and its energy
reference. It is not an energy-only correction to that fit: its original
atom basis and fitting objective are not independently recovered. It is
a source-level diagnostic, not a fitted replacement or equilibrium solve.
Original Gurvich/Anderson inputs and the Goos-to-FastChem fitting procedure
remain bounded provenance follow-ups, not a reason to revert to old JANAF.

Modern state sum and caloric checks
-----------------------------------

The latest recommended ExoMol 2kNigHt model is
`Perri et al. (2026) <https://doi.org/10.1093/mnras/stag744>`_.
The author-hosted full paper was inspected. Its formal publication is 2026,
although the data landing page retains 2025 in-press metadata. The numerical
states distribution has 4552 levels, J<=58 and energies below 50000 cm-1,
matching the paper's bound-line-list table. The paper separates bound and
quasibound levels from continuum states using wavefunction extent; the
continuum quadrature states used for photodissociation are not added here.

For principal 14N1H, divide all nuclear statistical weights by 6; retain
electronic/fine-structure degeneracies already present in the individual
J levels. The energy minimum is zero. The direct finite-state partition
sum agrees with the distributed rounded Q table within 0.6 parts per
million at all reviewed 300--4000 K nodes. This verifies the distribution
pair, not complete physical accuracy.

State Q/NASA at 1500/3000/4000 K is 0.999972/0.999736/0.998827.
Exact state moments also give H-H0, S and Cp; at 4000 K their differences
from NASA are -0.187 kJ/mol, -0.061 J/mol/K and -0.241 J/mol/K.
The principal-isotope mass correction makes the fixed-atom K proxy
0.998453 at 4000 K. This is not a natural-isotope mixture calculation.
At 4000 K, 0.0842 percent of Q comes from included levels above the
lowest ground-atom dissociation limit. That fraction diagnoses the chosen
finite metastable-state convention; it is not an estimate of all missing
states or permission to add an arbitrary discretized continuum.

Scope, decision and follow-up
-----------------------------

In the archived 75-case, 537-species ion-inclusive ``curated-cos-v1`` grid,
NH's maximum N inventory fraction is 1.31788 percent and maximum mole
fraction is 1.42501e-6 (4000 K, 100 bar, C/O=0.588844). This existing
footprint is not a response to a replacement and does not validate cooler
or photochemical atmospheric predictions.

Retain curated: the modern energy+NASA diagnostic differs by only
-2.2617 to -0.7519 percent in K across the sampled 1500--4000 K range,
while the independent state-sum thermal comparison is close. A future
NASA-thermal+ATcT candidate can be constructed immediately and assessed
with fixed-network substitutions. A 2kNigHt thermal replacement also needs
explicit state-tail, isotope, metastability and derivative-fit checks.
No unavailable full text is required to merge this bounded source review.

The offline replay keeps the one-bar standard pressure, neutral atom
balance and source-native constants: NASA R=8.314510 J/mol/K; SI tables
and production R=8.31446261815324. Source H(0) gauges consistently subtract
H(298.15)-H(0) from each fixed Hf(298.15), including atomic increments.
Energy, molecular thermal and atomic thermal contributions are separated.
Tests verify closure, exact curated values, caloric identities, nuclear
weights and the frozen grid's stoichiometric inventory.

MCEG supports attribution, normalized diagnostics and the scoped decision.
Complete physical uncertainty and recovery of every original fit input
remain unknown. Run ``python mceg/reviews/nh_source_review.py``; selected
inputs, finite states, output and receipt are in
``tests/data/validation/nh_source_review/20261106``. Production is unchanged.
