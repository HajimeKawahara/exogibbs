Atomic-ion thermodynamic audit
==============================

The FastChem4 ion-inclusive network needs species-specific review before it can
serve as a curated thermodynamic preset. This audit extends :doc:`ni_co_ionization_audit`
to all 49 singly charged atomic ions in the default 27-element set, using
``FastChem4/logK/logK.dat``. It does not audit molecular ions or doubly charged
ions, and does not change a production preset.

The adopted comparison model is recorded in
``mceg/contracts/atomic_ion_reference.yaml`` using the existing MCEG v0.1 schema.
MCEG records assumptions and evidence; the numerical calculation is an ordinary
standalone Python script. ExoGibbs has no MCEG dependency, and no MCEG extension
is needed for this audit.

The follow-up :doc:`atomic_ion_adoption` records species-specific source choices,
independent threshold updates and a completed comparison with the original CDS
table. The results and receipts below describe the original audit run.

Results
-------

The grid is **1000, 1500, 2000, 3000, 4000, 5000 and 6000 K**, with no
partition-function interpolation. There are 27 cation and 22 anion cases.
Of these, 47 have a numerical reference; N- and Zn- are classified separately
because the adopted evaluation reports no stable bound anion.

For each species define ``D = max(abs(log10(Kp_model / Kp_reference)))`` over
the seven nodes. Symmetric factors 1.1 and 2 are review thresholds, not error
bars. FastChem is within factor 1.1 for 37 of the 47 numerical cases; six more
exceed factor 1.1, and four exceed factor 2. All species and temperature values
are in :download:`summary.csv <../tests/data/validation/atomic_ion_audit/summary.csv>` and
:download:`comparison.csv <../tests/data/validation/atomic_ion_audit/comparison.csv>`.

The ten FastChem review candidates are:

.. list-table:: Largest sampled FastChem departures
   :header-rows: 1

   * - Species
     - Kp(FastChem) / Kp(reference)
     - Temperature [K]
   * - Cu+
     - 1665.83
     - 2000
   * - Ni+
     - 293.40
     - 4000
   * - Co+
     - 12.644
     - 6000
   * - Ca-
     - 2.439
     - 6000
   * - Zn+
     - 1.969
     - 1000
   * - Fe+
     - 1.452
     - 1000
   * - Ge-
     - 0.8082
     - 6000
   * - Cr-
     - 0.8950
     - 1000
   * - P-
     - 1.1132
     - 6000
   * - Fe-
     - 1.1084
     - 1000

The Cu+ discrepancy is already present in the public JANAF tables. At these
nodes FastChem reproduces the Cu/Cu+ JANAF formation-energy difference within
about 0.6%, while JANAF itself gives approximately 1548--1666 times the
statistical reference. The original table values are retained in the fixture.
This locates the discrepancy upstream of the five-coefficient fit; it does not
identify the deeper cause in the JANAF evaluation.

NASA9 is a useful comparator, but a blanket replacement is unwarranted:

.. list-table:: Selected NASA9 comparisons
   :header-rows: 1

   * - Species
     - Largest sampled departure from reference
   * - Cu+
     - About 0.052%; strong improvement over FastChem
   * - Ni+
     - About 2.94%; strong improvement over FastChem
   * - Zn+, Fe+
     - About 0.68% and 0.16%; improvements over FastChem
   * - Co+
     - 26.8% at 1000 K; the whole interval is not within factor 1.1
   * - C-
     - Factor 2.516 at 1000 K; substantially worse than FastChem
   * - Ge-
     - Factor 1.923 at 3000 K; worse than FastChem
   * - Ca-
     - No record in the selected NASA source

The source comments identify 46 of the 49 FastChem fits as JANAF-derived,
Ge+ and Ge- as Hoeijmakers et al. (2019), and Ca- as a Kurucz/NIST Saha fit.
Consequently, the Ca- and Ge- results should not be attributed to JANAF.

Model and source choices
------------------------

For cations, the reference uses the first ionization energy from the pinned
January 2022 NIST ASD snapshot and the neutral/cation partitions of Barklem and
Collet (2016), Table 8. The local ExoJAX copies are excerpted without importing
ExoJAX. The original file hashes, source lines, threshold uncertainties and
reference codes are preserved. The original run did not compare the CDS
numerical table independently; :doc:`atomic_ion_adoption` now records that
follow-up comparison and closes this transcription gap.

For anions, the reference uses Ning and Lu (2022), Table 3 electron affinities
and Table 4 bound levels. These selected numerical entries were checked against
the author-hosted PDF images. Older experimental papers cited in the evaluation
were not individually reaudited. The partition function is

.. math::

   U_-(T) = \sum_i (2J_i+1)\exp[-hc\,\widetilde\nu_i/(k_B T)].

The fixture records energies in cm\ :sup:`-1`; the calculation converts them to
m\ :sup:`-1`. The unresolved Si- :math:`{}^2P` levels share their tabulated energy
with combined weight six. The C- fine-structure ordering has theoretical input
as noted in the source. Only the listed bound states are included; completeness
is assumed. Table 5 continuum resonances are excluded. H electron affinity is
the recommended theoretical value, not an experimental measurement.

For N- and Zn-, the evaluation's negative-affinity classification yields no
stable bound-ion reference. Their FastChem values and available formal NASA
values remain visible in the CSV, but reference values and residuals are blank.
These blanks do not mean agreement, zero Kp, or a failed numerical evaluation.

Let

.. math::

   C(T) = \left(\frac{2\pi m_e k_B T}{h^2}\right)^{3/2}
          \frac{k_B T}{p^\circ}, \qquad p^\circ = 10^5\ {\rm Pa}.

The two reaction conventions are

.. math::

   A \rightarrow A^+ + e^-:
   \quad K_p = 2\,\frac{U_+}{U_0}\,C(T)\,e^{-I/(k_B T)},

.. math::

   A + e^- \rightarrow A^-:
   \quad K_p = \frac{U_-}{2U_0 C(T)}\,e^{E_A/(k_B T)}.

All activities are partial pressures divided by 1 bar. The standard pressure
is not an assertion about total gas pressure. Electron spin two is explicit;
nuclear-spin weights are excluded. Heavy-particle mass ratios are set to one,
as in the previous Ni/Co audit. Hyperfine and isotope corrections are neglected;
the results are not precision isotope thermodynamics. Listed energy uncertainties
are not propagated into confidence intervals.

NASA comparisons use a common absolute Gibbs convention, including its electron
record: ``ln Kp = g_A/RT - g_ion/RT - charge*g_e/RT``. The existing NASA excerpt
is reused and 22 missing atomic-ion records are added to the audit fixture,
all from the same pinned NASA CEA source. The original source hash was checked,
and overlapping coefficients were compared directly. NASA has no Ca- or Zn-
record in this selection. Its N- record is retained only as a formal comparator.
NASA and FastChem can share older source material, so their agreement alone is
not independent physical validation.

Recommendation for the next preset step
---------------------------------------

Use this statistical model as the atomic-ion audit reference. Prioritize Cu+
and Ni+ for replacement; NASA9 is a promising numerical candidate for those
ions and for Zn+ and Fe+. Co+ needs a choice that addresses the low-temperature
NASA residual as well as the large FastChem high-temperature residual. Review
Ca-, Ge-, Cr-, P- and Fe- individually. Treat N- and Zn- as exclusion candidates
for a model restricted to stable bound ideal-gas species. Preserve C- when
considering NASA substitutions.

The 37 cases within factor 1.1 have passed this comparison screen only. This
audit does not establish a universal best source or validate heat capacities,
entropy derivatives, pressure applicability or equilibrium abundances. Finite
atomic level lists omit density-dependent dissolution; alkalis are especially
sensitive as temperature rises. Continuous-temperature coefficient generation,
a consistent atomic/electron reference basis, and abundance-impact tests remain
necessary before adopting a production preset.

Reproduction and evidence
-------------------------

The numerical audit and its seven targeted tests require only Python's standard
library and the checked-in data. They make no network requests:

.. code-block:: bash

   python3 examples/comparisons/audit_atomic_ions.py
   python3 -m unittest discover -s tests/unittests/examples -p atomic_ion_audit_test.py -v

Pytest also collects these tests in the ordinary development environment:

.. code-block:: bash

   python -m pytest -q tests/unittests/examples/atomic_ion_audit_test.py

The :download:`audit receipt <../tests/data/validation/atomic_ion_audit/receipt.json>` records
selected input and output hashes, Python version and base Git revision. New
audit files were uncommitted when evaluated, so their hashes identify the actual
implementation. The :download:`test receipt <../tests/data/validation/atomic_ion_audit/tests.json>`
records the executed checks and unavailable optional validation tools. Changing
selected inputs requires rerunning the audit, reviewing the scientific claims,
and updating the MCEG implementation hash and evidence together.

Optional structural validation uses a separate MCEG environment:

.. code-block:: bash

   PYTHONPATH=/path/to/mceg/src python -m mceg mceg/contracts/atomic_ion_reference.yaml

Schema validation establishes structure and local evidence links, not scientific
correctness. The MCEG physical-validation claim remains ``unknown``.

References
----------

* `Barklem and Collet (2016), A&A 588, A96 <https://doi.org/10.1051/0004-6361/201526961>`_,
  Eq. 6, Table 8 and the discussion of finite level sums.
* `NIST ASD ionization energies <https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html>`_,
  historical January 2022 snapshot as identified in the local source header.
* `Ning and Lu (2022), JPCRD 51, 021502 <https://doi.org/10.1063/5.0080243>`_,
  Tables 3 and 4 and Sections 4--6;
  `author-hosted PDF <https://info.phys.tsinghua.edu.cn/henucl/ning/148.pdf>`_.
* JANAF `Cu <https://janaf.nist.gov/tables/Cu-005.html>`_ and
  `Cu+ <https://janaf.nist.gov/tables/Cu-006.html>`_ public tables.
* `Pinned NASA CEA source <https://raw.githubusercontent.com/nasa/cea/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_.
  Its Apache-2.0 license and attribution are retained in
  ``src/exogibbs/data/nasa9/LICENSE.txt`` and ``NOTICE.txt``.
