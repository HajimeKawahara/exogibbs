CO: independent thermal audit with modern state sums
====================================================

This 2026-11-03 investigation addresses Issues #308 and #309 for neutral
``C1O1``. **Retain the current NASA thermal functions plus ATcT 1.172 energy.**
Modern state sums support this scoped choice on the sampled 1500-4000 K
interval. There is no scientific need to replace production CO as part of
this investigation. This conclusion does not establish complete atmospheric
accuracy or make agreement with NASA the selection rule.

CO controls carbon/oxygen allocation and has important infrared bands in
planetary atmospheres. The inspected
`Li et al. (2015) methods <https://doi.org/10.1088/0067-0049/216/1/15>`_
discuss both CO2-dominated and H2-dominated applications. Improved line
positions, intensities, broadening or ultraviolet fluorescence are separate
questions from the equilibrium partition sum.

Energy and native references
----------------------------

The fresh accessible `ATcT 1.172 table
<https://atct.anl.gov/Thermochemical%20Data/version%201.172/index.php>`_
confirms the frozen gas-phase C, O and CO entries. Their balanced atomization
energy is (711.381 + 246.844 - (-113.794)) = **1072.019 kJ/mol**.
The corresponding native NASA and JANAF values are 1071.800005 and
1071.780 kJ/mol. The existing correction gives curated/NASA formation-K
ratios 1.018213 at 1500 K and 1.009065 at 3000 K; it changes the energy
anchor, not molecular S or Cp. ATcT entry uncertainties are correlated;
their quadrature is not adopted as a reaction uncertainty.

NASA assigned H(0) is reconstructed from Hf(298.15) minus its H(298.15)-H(0)
header, separately for the molecule and both atoms. NASA's historical
R=8.314510 is retained in its dimensionless functions. State thermodynamics
and reported caloric differences use R=8.31446261815324 consistently; the
unscaled source NASA Cp is also retained in the numerical packet.
The hybrid shift D0/k is 27.07365 K, including this convention difference;
it is not simply the raw 218.995 J/mol energy difference divided by modern R.

The `JANAF C-093 table <https://janaf.nist.gov/tables/C-093.html>`_ is a
September 1965 evaluation at one bar. Its published Cp values provide a
direct caloric comparison, alongside the frozen C-003/O-001 atomic references.
The native JANAF-minus-NASA atomic thermal term is below 0.000328 in ln K
at the six common nodes. At fixed D0 and fixed NASA atoms, JANAF/NASA thermal
K ratios are 0.999663 at 1500 K and 1.000177 at 4000 K.
The original Gurvich 1979 monograph behind the NASA header was not inspected;
obtaining it is not a prerequisite to retaining the independently audited model.

What the modern data contain
----------------------------

`Tao et al. (2026), 10.1093/mnras/stag985
<https://doi.org/10.1093/mnras/stag985>`_ was inspected in the
`author-hosted accepted full manuscript
<https://www.homepages.ucl.ac.uk/~ucapsy0/pdf/26TaAbEl.pdf>`_, including
Sections 2.2, 3.1 and 3.3 and the partition comparison.
The official 4thplus distribution contains 11264 states of 12C16O:
X1Sigma+ and A1Pi only. The eight electronic potentials discussed in the
paper are not eight states included in this line list. Neighbouring triplets
and their spin-orbit couplings are not included.

Li2015's official 6383-state distribution and published PF provide a second
comparison. Its rendered publisher methods, including Section 4, were read;
its PDF was unavailable. Both models descend from the Coxon-Hajigeorgiou
ground-state potential, with 4thplus refinements and empirical MARVEL levels.
Their agreement is not two independent experimental determinations.
The frozen Barklem-Collet 2016 molecular Q table gives a broader compilation
check, also with shared spectroscopy and its own spin-orbit approximation.

The audit found distribution details that matter more than a blind version
upgrade:

* Both metadata files assign 2 to their symmetry-label nuclear-spin fields,
  but every actual state has g=2J+1. The paper explicitly specifies nuclear
  spin weight 1 for 12C16O, and both ground states are E=0, g=1, J=0.
  No division by two, extra electronic factor or homonuclear symmetry factor
  is applied. A-state parity components are already individual states.
* The 4thplus metadata declares a 1 K grid ending at 9000 K; the PF actually
  contains 9000 rows ending at 9001 K, with printed values such as 1500.2
  and 4000.4 K. Comparisons interpolate the explicit printed temperature
  column, never the row index. Rounded temperature/PF values and source
  conventions leave a residual: state-sum/PF differences are below 0.0030%
  on the six production-domain samples, and below 0.010% including 500 K.
  At 1500 K the 29.5 ppm residual is smaller than the 43.7 ppm Q sensitivity
  to half the printed 0.1 K temperature digit. Across all eight nodes its
  equivalent temperature offset is below 0.05 K. This is consistent with
  temperature rounding, but does not establish the PF generator's cause.
  Li2015 residuals are 6.5-11.0 ppm; Q output rounding alone allows only
  0.02-0.28 ppm. If its declared integer-K grid is exact, the residual needs
  source-generation reconciliation rather than an output-rounding claim.
  Both comparisons reproduce the distributed PF approximately; neither
  validates contradictory metadata or justifies higher PF precision.
* Metadata gives maximum energy 98253.97 cm^-1, whereas the actual 4thplus
  file reaches 98839.941885 cm^-1. The file, its hash and this discrepancy
  are retained separately. Version 20260530 is a metadata declaration.
* The `model page <https://exomol.com/models/CO/12C-16O/4thplus/>`_ gives
  ZPE=1081.77 cm^-1. It is not added to ground-relative state energies or
  subtracted a second time from the adopted D0. A potential De is not D0.

The compact packet retains every state's energy, g, J and X/A flag, so Q,
mean energy and variance are recomputed offline without numerical derivatives
of the rounded PF. The full transition files are unnecessary.

Thermal comparison and isotope scope
------------------------------------

At one bar, Q is converted to thermal Gibbs using ideal-gas molecular
translation. Both caloric comparisons use the same NASA elemental mass
28.0101 u, while 12C16O internal levels serve as an explicit representative
approximation. The isotope mass 27.99491462 u is a separate diagnostic:
its translational multiplier is 0.999186901. Natural-isotope internal
mixtures, isotope-dependent D0 and atomic isotope balance are not established.

.. list-table:: Direct 4thplus state sum versus the adopted NASA thermal model
   :header-rows: 1

   * - T, K
     - Q ratio, common molecular mass
     - Delta H(T)-H(0), J/mol
     - Delta S, J/mol/K
     - Delta Cp, J/mol/K
   * - 1500
     - 1.00014816
     - 0.6077
     - 0.001637
     - 0.001326
   * - 2000
     - 1.00016585
     - 1.1694
     - 0.001964
     - 0.000959
   * - 3000
     - 1.00020364
     - 3.3958
     - 0.002825
     - 0.003724
   * - 4000
     - 1.00025660
     - 7.6179
     - 0.004038
     - 0.003728

The largest Cp difference among all six 1500-4000 K samples is
0.004442 J/mol/K (3500 K). The common-mass K diagnostic changes by only
0.0148-0.0257%; using isotope translation instead gives 0.999335-0.999443
relative to current K. Neither comparison is an abundance response.
Agreement at these nodes is not a uniform error bound between them.

4thplus/Li2015 state-sum Q differences remain below 1.1 parts per million,
and their largest sampled Cp difference is 0.001027 J/mol/K. Direct 4thplus
Cp differs from the independently published JANAF table by less than
0.014 J/mol/K. Barklem-Collet Q differs by less than 0.017% on the shared
nodes. These checks support retention without claiming uncorrelated sources.

At 4000 K the A1Pi contribution is only 2.51e-10 of Q. Thus the newer
ultraviolet transitions are not a reason to change equilibrium thermochemistry.
States above the adopted elemental D0 contribute 3.47e-13 of Q, while
the J>=140 edge contributes 2.69e-6. These are inventory sensitivities;
they do not bound omitted triplet states, unrepresented rotational tails,
quasibound states or the full physical error. The declared 9000 K file range
is not a validation of thermochemical completeness up to that temperature.

Current footprint, MCEG and remaining work
------------------------------------------

The saved **curated-tio-v1** gas-only grid has 75 converged cases and
537 species. All CO potentials reproduce the actual stored reaction
polynomial. Carbon and oxygen conservation residuals are below 2.26e-13
and 1.44e-13. CO contains up to 99.99976% of carbon (2500 K, 1e-6 bar,
reference C/O) and 99.99997% of oxygen (2000 K, 1e-6 bar, C/O=1.5).
These different grid points demonstrate elemental importance despite a
small source discrepancy; they are not new equilibrium solves or atmospheric
validation. Condensation, transport, photochemistry and non-LTE emission
remain separate from this ideal-gas equilibrium audit.

``mceg/contracts/co_source_review.yaml`` records supported source/energy
reconstruction, normalized thermal comparisons and scoped retention, with
complete physical accuracy unknown. The concrete follow-ups are reconciliation
of ExoMol metadata/PF-generation details and a complete isotope/omitted-state
assessment if a future precision requirement warrants it. Neither is a demand
to collect every historical original before merging this investigation.
No production coefficient, runtime module or temperature bound changes.

Reproduce the saved review and, optionally, the compact state extraction::

   PYTHONPATH=src python mceg/reviews/co_source_review.py
   PYTHONPATH=src pytest tests/model_evidence/co_source_review_test.py
   PYTHONPATH=src python mceg/reviews/co_source_review.py --extract /path/to/raw --packet /tmp/co-regenerated

The raw directory needs ``4thplus.states.bz2``, ``4thplus.pf``,
``Li2015.states.bz2`` and ``Li2015.pf`` with the recorded source hashes.
