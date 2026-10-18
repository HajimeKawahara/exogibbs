Atomic-ion source adoption
==========================

Following :doc:`atomic_ion_audit`, the source choices for the next gas-only
preset are **43 NASA9 thermal models with updated spectroscopic thresholds,
four partition-function models, and two exclusions**. These are adoption
decisions for the 49 atomic ions, not a released preset or a claim of a globally
optimal database. Molecular species and higher charge states remain outside
this review.

The decisions and evidence are recorded in
``mceg/contracts/atomic_ion_adoption.yaml``. ExoGibbs does not import or depend on
MCEG. The existing v0.1 schema is sufficient.

Selection policy
----------------

Use NASA9 for its continuous thermal functions where the audit supports the
level statistics, but anchor the ground-state reaction energy to the adopted
spectroscopic evaluation. This also updates small, identifiable energy errors
that the original factor-1.1 screening did not flag. The choice is not made by
selecting the smallest residual at each temperature. NASA and the reference
share some source ancestry, so agreement is not independent physical validation.

Use explicit partition functions when the thermal populations require revision
or NASA lacks a bound species. C- could retain FastChem as an approximation
(maximum sampled factor 1.0089), but the canonical choice here uses the evaluated
bound-level inventory, consistently with Ca- and Ge-. Do not include an unbound
resonance as a stable equilibrium species merely because a polynomial exists.

All species
-----------

``NASA + IE`` uses the pinned NASA9 neutral, cation and electron functions with
the NIST first ionization energy. ``NASA + EA`` uses the corresponding functions
with the Ning and Lu electron affinity. ``Partition`` means NIST IE and CDS
neutral/cation partitions for Co+, or Ning and Lu EA/bound levels and the CDS
neutral partition for anions. ``--`` denotes absence from the audited inventory,
not a new exclusion decision. Both Na and K charge signs require the additional
high-temperature assessment described below.

.. csv-table:: Decisions for the default 27-element atomic-ion inventory
   :header: "Element", "Positive ion", "Negative ion"

   Al, NASA + IE, NASA + EA
   Ar, NASA + IE, --
   C, NASA + IE, Partition
   Ca, NASA + IE, Partition
   Cl, NASA + IE, NASA + EA
   Co, Partition, NASA + EA
   Cr, NASA + IE, NASA + EA
   Cu, NASA + IE, NASA + EA
   F, NASA + IE, NASA + EA
   Fe, NASA + IE, NASA + EA
   Ge, NASA + IE, Partition
   H, NASA + IE, NASA + EA
   He, NASA + IE, --
   K, NASA + IE, NASA + EA
   Mg, NASA + IE, --
   Mn, NASA + IE, --
   N, NASA + IE, Exclude
   Na, NASA + IE, NASA + EA
   Ne, NASA + IE, --
   Ni, NASA + IE, NASA + EA
   O, NASA + IE, NASA + EA
   P, NASA + IE, NASA + EA
   S, NASA + IE, NASA + EA
   Si, NASA + IE, NASA + EA
   Ti, NASA + IE, NASA + EA
   V, NASA + IE, NASA + EA
   Zn, NASA + IE, Exclude

The :download:`49-row decision table <../results/atomic_ion_adoption/decisions.csv>`
includes each threshold, enthalpy-constant shift, sampled comparison and reason.

Energy update and numerical evidence
-------------------------------------

Let :math:`q=+1` for ionization and :math:`q=-1` for attachment, and let
:math:`E` be the positive IE or EA in joules per particle. From the NASA source
headers, compute assigned ground enthalpies
:math:`H_i(0)=\Delta_f H_i(298.15)-[H_i(298.15)-H_i(0)]` and

.. math::

   D_0 = H_{\mathrm{ion}}(0)+qH_e(0)-H_A(0), \qquad
   \Delta b_1 = \frac{qE}{k_B}-\frac{D_0}{R_{\mathrm{NASA}}}.

Here :math:`R_{\mathrm{NASA}}=8.314510\ \mathrm{J\,mol^{-1}\,K^{-1}}`
is the **original coefficient convention**, documented in Appendix A of
NASA/TP-2002-211556. Using the modern gas constant for the old header energies
would introduce a spurious correction. The new threshold uses modern SI
:math:`E/k_B`. The high-temperature integration constant itself is not a
ground-state energy; the header enthalpies are needed.

Update only the ion's :math:`b_1`, so that

.. math::

   \ln K_p^{\mathrm{updated}} = \ln K_p^{\mathrm{NASA}}-\Delta b_1/T.

This changes the enthalpy origin without changing the NASA entropy or heat
capacity. It does not fit the audit residuals. The 43 selected NASA ion curves
have analytic minimum :math:`C_p/R\geq 2.49436` over 1000--6000 K; the minimum
is Ca+. This permits the existing 2.49 polynomial-fit tolerance. It is a check
of these source curves, not of a complete mixed preset's caloric properties.

At the seven audit nodes, the 39 selected NASA models excluding Na and K have
maximum symmetric departure factor **1.01316**. The table below reports
:math:`\max_T\max(K/K_{\rm ref},K_{\rm ref}/K)`; these factors are not
uncertainty bounds, and the maxima need not occur at the same temperature.

.. csv-table:: Selected departures before and after the energy update
   :header: "Ion", "FastChem", "Original NASA9", "Updated NASA9"

   Cu+, 1665.83, 1.00052, 1.00009
   Ni+, 293.40, 1.02939, 1.00088
   Ti+, 1.09103, 1.09443, 1.00245
   Cr-, 1.11727, 1.12217, 1.00510
   Fe-, 1.10842, 1.10541, 1.01316
   Co+, 12.64381, 1.26768, 1.06373
   C-, 1.00890, 2.51606, 2.49527
   Ge-, 1.23735, 1.92300, 2.29469

Co+, C- and Ge- remain partition-function exceptions after the energy update.
For example, the implied NASA C- partition at 1000 K is about 9.97, whereas the
evaluated bound levels give approximately four. The corresponding neutral C
partitions agree closely. Co+ instead retains a growing high-temperature
partition discrepancy. Ca- is bound in the adopted evaluation but absent from
the selected NASA file; its FastChem fit differs by factor 2.44. N- and Zn- have
no stable bound-state reference in the adopted evaluation and are excluded.

Conditions and remaining implementation work
--------------------------------------------

Na+/- and K+/- retain the NASA thermal functions provisionally. Their largest
sampled departures are about 3.83% and 7.02% at 6000 K, respectively, mainly
from the neutral partitions. High excited states become sensitive to the
density-dependent level cutoff in this regime. Replacing their smooth curves
only to match a finite low-density table is not sufficient evidence of an
improvement. Assess the temperature-density range before calling this an
unconditional atmospheric default. The general ideal-gas/LTE assumptions and
finite-level limitations also apply to all other species.

The four partition choices currently evaluate only the seven exact table
temperatures. Their equality to the audit reference is **by construction**,
not a successful independent validation. A production implementation still
needs a defined continuous representation, derivative/caloric checks, and
equilibrium-abundance comparisons. Express each chosen reaction in a common
atomic/electron reference convention before combining it with molecular data;
do not transplant absolute ion Gibbs energies into the FastChem gauge.
The current presets and network remain unchanged by this study.

Provenance and reproduction
----------------------------

The CDS Table 8 original was compared with the ExoJAX copy: all **284 records
times 42 temperatures (11928 values)** agree numerically. Its temperature header
also confirms all 378 values used for the 54 neutral/cation records at the
seven audit nodes. This closes the original audit's CDS transcription gap;
it does not establish the completeness or physical accuracy of the level lists.
The new fixture preserves the original table hash, selected excerpts and line
numbers, together with 75 NASA ground-enthalpy records. The original audit
fixture and execution receipt remain unchanged.

The IE source remains the explicitly pinned January 2022 NIST snapshot; this
study does not claim to have updated every entry to the latest ASD release.
EA and bound-level choices retain the Ning and Lu (2022) evaluation and its
stated theoretical inputs and unresolved levels. Source uncertainties have
not been propagated into confidence intervals for equilibrium abundances.

.. code-block:: bash

   python examples/comparisons/select_atomic_ions.py
   python -m pytest -q tests/unittests/examples/atomic_ion_adoption_test.py
   PYTHONPATH=/path/to/mceg/src python -m mceg mceg/contracts/atomic_ion_adoption.yaml

The :download:`comparison <../results/atomic_ion_adoption/comparison.csv>`,
:download:`execution receipt <../results/atomic_ion_adoption/receipt.json>` and
:download:`test receipt <../results/atomic_ion_adoption/tests.json>` record the
evaluated files and verification scope. MCEG validation checks structure only;
the physical-validation claim remains ``unknown``.

References
----------

* `NASA/TP-2002-211556 <https://ntrs.nasa.gov/api/citations/20020085330/downloads/20020085330.pdf>`_,
  Eqs. 2--3, assigned enthalpies, Appendix A and Table C1;
  `pinned CEA data <https://raw.githubusercontent.com/nasa/cea/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_.
* `Barklem and Collet (2016) <https://doi.org/10.1051/0004-6361/201526961>`_ and
  `CDS Table 8 <https://cdsarc.cds.unistra.fr/ftp/J/A+A/588/A96/table8.dat>`_.
* `NIST ASD ionization energies <https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html>`_.
* `Ning and Lu (2022) <https://doi.org/10.1063/5.0080243>`_, Tables 3--4 and
  the evaluated bound-state classifications.
