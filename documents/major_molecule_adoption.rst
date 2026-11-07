Major-molecule source adoption
==============================

This study records source decisions for **35 neutral gas molecules**, following
the user's approval of :doc:`atomic_ion_adoption`. It is an adoption proposal,
not a production preset. ExoGibbs does not acquire an MCEG dependency.

The result is **14 preferred combinations, six conditional improvements and
15 provisional retentions**. The central decision is to choose the ground-state
atomization energy separately from the temperature-dependent thermal functions.
Neither the NASA9 representation nor a newer compilation date establishes
greater physical accuracy by itself.

Scope and conventions
---------------------

The baseline is the pinned ion-inclusive
``FastChem4/logK/logK.dat``. The current default ``fastchem4.chemsetup()`` uses
``logK_wo_ions.dat``; neither file is changed. The source comments matter:
HNC already uses Goos/Burcat, SH uses Lodders, CaH uses NASA, ten metal hydrides
use Barklem--Collet (BC), and TiH uses Burrows. They are not all JANAF models.

The comparison evaluates 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000 and
6000 K. The temperature column below distinguishes this review window from
the original source intervals. In particular, the TiH review is restricted to
1200--4800 K, the range of the published molecular partition table. This is
not a temperature-density applicability claim or a guarantee between nodes.
Gas-only calculations do not enforce stability against condensation.

All equilibrium constants describe formation from neutral gas atoms,
:math:`\sum_a n_a A_a \rightleftharpoons M`, and are dimensionless at
:math:`p^\circ=10^5` Pa. BC Table 7 instead gives the logarithm of
:math:`p_Ap_B/p_{AB}` in Pa, so its diatomic formation constant satisfies
:math:`\log_{10}K_p=5-\log_{10}(p_Ap_B/p_{AB})`.
It is not a pressure in dyn/cm2. Nuclear-spin products are excluded in BC;
no extra factor of two is applied to a homonuclear reaction.

The ATcT anchor is the explicitly reviewed **version 1.172**, not a claim of
using the latest release. Later 1.228 entries were encountered during the
review; they have not been mixed into this network snapshot. Before publishing
a production preset, the selected atom and molecule records should be reviewed
together against a newer single network version. ATcT provides evaluated
enthalpies, not the full temperature dependence of entropy or heat capacity.
Displayed enthalpy uncertainties are correlated and have not been propagated.

Adoption table
--------------

``prefer`` identifies the preferred source combination in this review.
``prefer_conditionally`` identifies an improvement with a named open issue.
``retain_provisionally`` preserves the present source lineage because the
evidence does not establish a better complete replacement. These labels do
not mean that the resulting abundances have been physically validated.

The full table records each species separately, including the existing source,
selected source, temperature scope, difference mechanism, reason and follow-up.
The short source names refer to the pinned records described below.

.. csv-table:: Species-specific decisions
   :file: ../tests/data/validation/major_molecule_adoption/adoption_table.csv
   :header-rows: 1
   :widths: 5, 13, 14, 12, 9, 18, 18, 15

Download the :download:`adoption table
<../tests/data/validation/major_molecule_adoption/adoption_table.csv>`,
:download:`numerical decision summary
<../tests/data/validation/major_molecule_adoption/decisions.csv>` and
:download:`350 temperature comparisons
<../tests/data/validation/major_molecule_adoption/comparison.csv>`.

What the comparisons establish
-------------------------------

* **H2, H2O, CO, CO2, N2, NH3 and HCN:** prefer the CEA thermal functions
  anchored to balanced ATcT atomization energies. For most of these species,
  the thermal differences are modest; part of the benefit is an explicit
  energy reference and continuous caloric functions. It is not a demonstration
  that every CEA thermal curve is better than JANAF. The fixture also evaluates
  an energy-only correction of the existing JANAF-derived FastChem fit, so the
  energy improvement is not attributed exclusively to NASA9.
* **CH4:** retain JANAF thermal functions provisionally and use the ATcT energy.
  At a common atomization energy and common neutral-atom thermal anchors, the
  CEA/JANAF formation constants differ by a factor of about **2.94 at 6000 K**.
  Their molecular heat capacities are approximately **146.2 versus 106.3
  J/mol/K** there. This is a thermal-model disagreement, not an energy-zero
  correction. Retention is a conservative decision pending an independent
  high-temperature partition audit; it does not establish JANAF as more accurate.
* **HCN and HNC:** preserve distinct isomer identities. HCN's preferred model
  differs from the current FastChem value by a sampled maximum factor of about
  **9.82 at 300 K**. JANAF and CEA HCN thermal functions become much closer
  when the same atomization energy is used. HNC is already a different source
  in FastChem and has no matching JANAF HCN table by identity. Future combined
  HCN/HNC line-list partitions must not count the same states twice.
* **Sulfur species:** update atomic sulfur and molecular energies together.
  Changes in S2 and CS2 can be large at 300 K even when older compilations
  agree with one another. Prefer CEA plus ATcT for the eight reviewed sulfur
  species other than COS; SH and CS2 remain conditional. Retain JANAF for COS
  pending an independently checked energy anchor. Sulfur formation energies
  alone cannot be compared without their elemental references.
* **TiO and VO:** keep JANAF thermal functions provisionally while choosing
  the energy separately. TiO uses BC's evaluated **6.869 +/- 0.061 eV**.
  VO uses the **6.545 +/- 0.002 eV** direct measurement of Merriles et al.
  (2020). The unmodified CEA VO threshold is about 0.20 eV lower; blindly
  adopting it would be a substantial change in the wrong direction relative
  to that measurement. Modern ExoMol partitions require an explicit nuclear-spin,
  isotopologue and state-completeness comparison before substitution.
* **CaH:** retain the existing NASA energy lineage and original continuous
  curves. BC's **2.281 eV** and CEA's approximately **1.686 eV** give radically
  different low-temperature equilibrium constants. The 2017 potential study
  fixes :math:`D_e=14300\ {\rm cm}^{-1}` and explicitly says that its data
  cannot determine an accurate dissociation energy. This supports rejecting an
  automatic BC switch; it is not a precise measurement of :math:`D_0`.
* **MgH:** retain the BC-derived thermal fit and replace **1.271 eV** with
  the inspected **24MgH D0 = 1.28517 eV** measurement. This changes the formation
  constant by about **1.73 at 300 K**. The 2013 refinement and isotope averaging
  remain to be incorporated consistently. Its abstract reports :math:`D_e`,
  which must not be substituted for :math:`D_0`.
* **Other hydrides:** retain the nine already-BC-derived curves provisionally;
  all reproduce BC within about 2% on the sampled grid. This is a fit check,
  not an independent validation of the inherited dissociation energies.
  FeH has a substantial conflicting theoretical threshold. AlH and KH retain
  JANAF provisionally because the inspected evidence does not settle the
  energy choice. TiH retains Burrows within the review window: BC and Burrows
  differ in electronic-state sums and the Ti atomic partition, not just D0.

All factors above are symmetric ratios of formation constants at fixed
temperature. They are **not abundance errors, confidence intervals or bounds
over a continuous temperature interval**.

Energy and reference-state handling
-----------------------------------

For a molecule, use the balanced atomization energy

.. math::

   D_0 = \sum_a n_a\Delta_f H_a(0)-\Delta_f H_M(0).

The CEA source header instead supplies assigned enthalpies in its reference
gauge. Its assigned :math:`H(0)` is obtained from :math:`H_f(298.15)` minus
:math:`H(298.15)-H(0)`, not from the high-temperature coefficient :math:`b_1`.
The source gas constant is **8.314510 J/mol/K**. If :math:`d_{\rm old}` is
the source atomization energy divided by that gas constant and
:math:`d_{\rm new}=D_{0,\rm adopted}/R`, a NASA molecular enthalpy constant is
shifted by :math:`\delta b_1=d_{\rm old}-d_{\rm new}`. Thus

.. math::

   \ln K_{p,\rm new}=\ln K_{p,\rm old}-\delta b_1/T.

This leaves the source molecular entropy and heat capacity unchanged.
ATcT formation enthalpies are not directly inserted into CEA's assigned
enthalpy gauge. JANAF candidates similarly retain the tabulated
:math:`H(T)-H(0)` and :math:`S(T)`, with a separately chosen :math:`D_0`.
All adopted CEA/JANAF candidates use **the same CEA neutral-atom thermal
anchors**, compatible with keeping the preceding atomic-ion policy separate.
The script also records raw JANAF reactions with JANAF atomic anchors as
comparators. Retained FastChem models remain atom-formation-K descriptions;
they have not been certified as accurate caloric models.

The old JANAF HS formation columns do not combine consistently with the
current atom tables. At 6000 K, forming SH from their tabulated formation
Gibbs energies differs by more than eight natural-log units from using the
balanced ``Hf(298) + sensible H - T*S`` construction. This identifies a
reference-table inconsistency; the historical origin has not been traced.
The study therefore uses the latter construction and records the original
columns for review. It does not mistake this discrepancy for a molecular
partition-function error.

The BC table is also reconstructed independently from molecular and atomic
partition functions, masses and D0. The maximum residual is **0.00491 dex**
(about 1.14%) over the 240 sampled diatomic entries. This confirms the
reaction direction and pressure unit at the stated tolerance, not exact
numerical identity. Source constant/rounding conventions behind the residual
have not been fully traced. The published BC Kp table, not this reconstruction,
is the comparator used in the adoption table.

Sources and provenance
----------------------

The :download:`numerical reference fixture
<../examples/comparisons/data/major_molecule_reference.json>` contains the
selected records, original line numbers where available, temperature nodes,
source URLs, full-source hashes for the fetched CEA/CDS files, and inspection
limits. NIST/ATcT HTML values were read through text retrieval; no unobserved
original-HTML hash is asserted. HCN/HNC identity, gas phase, standard pressure,
natural-abundance versus isotope-specific data, and D0 versus De are retained.

* `NASA/TP-2002-211556 <https://ntrs.nasa.gov/citations/20020085330>`_
  defines the NASA9 representation and assigned enthalpy convention. The
  `pinned CEA thermo.inp <https://raw.githubusercontent.com/nasa/cea/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_
  supplies 26 molecule records and their atomic references. Individual source
  headers identify Gurvich, Woolley, East and other evaluations; the underlying
  monographs were not all independently reaudited. NASA is the representation
  and compilation here, not an independent measurement for every species.
* `NIST-JANAF <https://janaf.nist.gov/>`_: 32 identified gas/reference tables,
  including 22 molecules and ten neutral atoms. Each excerpt retains the
  table-specific evaluation date. The numerical audit does not endorse every
  historical reference-state column.
* `ATcT 1.172 <https://atct.anl.gov/Thermochemical%20Data/version%201.172/index.php>`_:
  18 molecules and six constituent atoms, including explicit gas and isomer
  identifiers. The energy evaluation is inspected; the entire contributing
  thermochemical network and covariance matrix are not independently audited.
* `Barklem and Collet (2016) <https://doi.org/10.1051/0004-6361/201526961>`_
  and `CDS Tables 1, 6, 7 and 8 <https://cdsarc.cds.unistra.fr/ftp/J/A+A/588/A96/>`_:
  24 diatomic molecule comparisons. Molecular spin-orbit structure is neglected
  in their general calculation, unlike the resolved atomic state sum. The
  actual table header gives 700 K where the ReadMe column prose says 750 K;
  the numerical header controls this excerpt.
* `Merriles et al. (2020) <https://doi.org/10.1063/5.0014006>`_:
  VO ground-state dissociation threshold, inspected through the author abstract
  indexed at PubMed. The complete experimental systematic-error analysis was
  not reaudited.
* `Shayesteh et al. (2007) <https://doi.org/10.1021/jp075704a>`_ and
  `Henderson et al. (2013) <https://doi.org/10.1021/jp406680r>`_: MgH
  spectroscopy. The first supplies the adopted D0; the latter's De refinement
  is recorded as a follow-up, not silently treated as a D0 measurement.
* `Shayesteh and Ghazizadeh (2017) <https://doi.org/10.22036/pcr.2017.75313.1355>`_:
  CaH potential, especially the fixed well depth and its explicit limitations.
* `DeYonker and Allen (2012) <https://doi.org/10.1063/1.4767771>`_:
  the conflicting FeH theoretical D0, inspected through the author abstract.
* `Burrows et al. (2005) <https://doi.org/10.1086/429366>`_:
  TiH Table 13, Section 6 and Appendix A, including the finite Ti atomic sum.

Reproduction and MCEG
-----------------------

The comparison is offline and uses the Python standard library:

.. code-block:: bash

   python3 examples/comparisons/select_major_molecules.py
   python -m pytest -q tests/unittests/examples/major_molecule_adoption_test.py

The :download:`run receipt <../tests/data/validation/major_molecule_adoption/receipt.json>`
identifies the evaluated working-tree files and outputs. The
:download:`test receipt <../tests/data/validation/major_molecule_adoption/tests.json>`
records the executed verification scope. Optional structural validation uses
the independent MCEG checkout:

.. code-block:: bash

   PYTHONPATH=/home/kawahara/mceg/src python -m mceg \
     mceg/contracts/major_molecule_adoption.yaml

The :download:`MCEG contract <../mceg/contracts/major_molecule_adoption.yaml>`
keeps source inspection, numerical verification and physical validation
distinct. No MCEG schema extension is needed. Physical validation of the full
hybrid network and its abundances remains unknown. JANAF table selections
still need continuous representations; all production candidates need
caloric/interval checks and application-level equilibrium comparisons.
