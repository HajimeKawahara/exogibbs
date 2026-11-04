SO2: energy balance and high-temperature completeness
=====================================================

This 2026-11-04 investigation addresses Issues #308 and #309 for neutral
``O2S1``. **Retain NASA thermal functions plus ATcT 1.172 energy.** The
available modern data do not establish a superior replacement over the
production interval, 1500-4000 K. This is a scoped retention decision with
unresolved thermal accuracy, not a claim that NASA is exact. No production
coefficients, defaults or archived abundance solutions change.

SO2 is a priority because it is observable in planetary atmospheres.
`Tsai et al. (2023) <https://doi.org/10.1038/s41586-023-05902-2>`_
connect the WASP-39b detection to photochemistry involving H2S and H2O;
`the MIRI study (2024) <https://doi.org/10.1038/s41586-024-07040-9>`_
provides additional SO2 measurements. These motivate investigation, but do
not validate a hot equilibrium model or measure its thermodynamic error.

Balanced energy and conventions
-------------------------------

The accessible `ATcT 1.172 table
<https://atct.anl.gov/Thermochemical%20Data/version%201.172/index.php>`_
confirms gas-phase Hf(0) values S = 276.83, O = 246.844 and
SO2 = -294.20 kJ/mol. The same-version atom-balanced D0 is therefore
**1064.718 kJ/mol**, versus native NASA 1062.769902 and JANAF 1062.613.
The molecular ATcT Hf(298.15), -296.74 kJ/mol, is close to NASA's -296.81,
but the sulphur atomic reference also matters. Comparing molecular formation
enthalpies alone misses most of the reaction-energy correction.

NASA assigned H(0) is obtained from Hf(298.15) minus each record's
H(298.15)-H(0), then balanced across S + 2 O - SO2. This assigned energy is
not a separately measured Hf(0). Native NASA R = 8.314510 is retained when
constructing its dimensionless functions; caloric comparisons consistently
use modern R = 8.31446261815324. The hybrid D0/k shift is 235.030752 K,
and curated/native-NASA formation-K is 1.169630 at 1500 K and 1.060518
at 4000 K. The existing ATcT correction changes the energy anchor, not S
or Cp. Reported ATcT species uncertainties are not independent; no reaction
uncertainty is manufactured by quadrature without network covariance.

The NASA header cites Gurvich (1989), part 1 p. 288 and part 2 p. 175.
Its SO2 intervals are 300-1000 and 1000-6000 K. The
`JANAF O-034 gas table <https://janaf.nist.gov/tables/O-034.html>`_
is a June 1961 evaluation at one bar. Fixed-energy, fixed-NASA-atom
JANAF/NASA thermal K is 1.009183 at 1500 K and 1.034227 at 4000 K;
JANAF Cp is respectively 0.279 and 0.231 J/mol/K above the consistently
scaled NASA values. This residual thermal disagreement remains documented.
The historical Gurvich monograph and original JANAF spectroscopy were not
reaudited; they are not blanket prerequisites for this retention PR.

The ground state of 32S16O2 has E = 0 and degeneracy one. All supplied
states have g = 2J+1. Its C2v exchange restrictions already exclude the
zero-weight B1/B2 species; A1/A2 have nuclear-spin weight one. There is no
additional division by the oxygen permutation factor two. The sums describe
one electronic-state rovibrational distribution, not all electronic states.
The representative-isotope mass is 63.96190024 u; the retained NASA
natural-element mass is 64.0638 u. Their translational entropy difference is
0.01985325 J/mol/K (a 0.2385% K effect), separately recorded from internal Q.
ATcT's displayed natural mass, 64.0648, is not silently substituted. This
representative internal distribution with natural translation is an explicit
approximation, not a full isotope-balanced equilibrium model.

The `NIST S I levels
<https://physics.nist.gov/PhysRefData/Handbook/Tables/sulfurtable5_a.htm>`_
place 3P2, 3P1 and 3P0 at 0, 396.055 and 573.640 cm-1 with weights 5, 3
and 1; 1D2 and 1S0 lie at 9238.609 and 22179.954 cm-1. Their five-level
sum agrees with NASA's inferred atomic Q within 0.008% over the sampled
300-4000 K range. Thus the shared atomic term is not a fixed ground weight
5 or an unsplit weight 9. This check does not bound all atomic uncertainties.

What the modern sources establish
---------------------------------

The full author manuscript of
`Underwood et al. (2016), 10.1093/mnras/stw849
<https://doi.org/10.1093/mnras/stw849>`_ was inspected, including Section 3
and Figures 1-3. Its ExoAmes line list considers J <= 165, lower transition
energies below 8000 cm-1 and upper energies below 15000 cm-1. The 2000 K
opacity limit is not automatically the limit of a sum over all upper states.
Indeed, p. 3894 claims approximately 0.1% convergence of the full Q for
3000-5000 K. That statement must be distinguished from what the current
published files actually reproduce.

The hash-pinned current distribution contains **3,270,270 rows**, maximum
energy 14999.996460 cm-1 and J = 165. Its metadata reports 3,270,271 states
and a state maximum of 8000 cm-1, which cannot describe all the supplied
upper states. The paper reports 3,255,954 levels; the current columns include
empirical/calculated source labels. These version/count differences are
recorded without assuming their explanation. The official PF has integer
1-2000 K rows. Direct sums reproduce its sampled values within 1.13 ppm;
the small residual exceeds printed-Q rounding and its cause is unresolved.

At 4000 K, states between 14000 and 15000 cm-1 already contribute 5.8601%
of the supplied Q, and J = 161-165 contributes 0.5267%. The 12000-15000
cm-1 shell contributes 20.2444%. Removing these shells is a convergence
diagnostic, not an estimate of the missing population above the boundary.
The direct full-file Cp falls from 56.43 at 1500 K to 33.42 J/mol/K at
4000 K, whereas NASA gives 56.76 and 60.14. This finite distribution does
not establish the paper's high-temperature completeness claim; its sharp
Cp decline and non-negligible edge population require reconciliation and
completion before adoption. Above 2000 K these are explicitly source
completeness diagnostics outside the distributed PF range, not physical
validation or a measurement of NASA error. Even agreement of Q at 2000 K
does not settle its derivatives: the direct Cp is 3.45 J/mol/K below NASA.

`Gamache et al. (2021), 10.1016/j.jqsrt.2021.107713
<https://doi.org/10.1016/j.jqsrt.2021.107713>`_, Section 3.9, constructs
SO2 TIPS as Watson rotational Q times harmonic-oscillator vibrational Q,
using Pine (1977) rotational constants and Lafferty et al. (1993)
vibrational terms. It tabulates 1-5000 K; that range does not remove the
product approximation or establish full anharmonic completeness.
The full 2025 evaluation,
`10.1016/j.jqsrt.2025.109568 <https://doi.org/10.1016/j.jqsrt.2025.109568>`_,
updates older calculations to newer CODATA constants; SO2 is not among its
new state-based evaluations. The official HAPI TIPS2025 SO2 array differs
from TIPS2021 by at most 2.02 ppm. The inspected
`2025 corrigendum <https://doi.org/10.1016/j.jqsrt.2025.109618>`_
addresses 16O14N18O symmetry labels, not SO2.

Numerical comparison and decision
---------------------------------

All ratios below share the current ATcT D0, NASA atomic functions, one bar
and natural molecular translation. Hence internal Q ratios equal molecular
formation-K ratios. They are not finite abundance responses. H-H(0),
G-H(0), S and Cp, isotope-mass proxies and native-source K are also archived.

.. list-table:: Normalized thermal diagnostics
   :header-rows: 1

   * - T (K)
     - ExoAmes Q/NASA
     - TIPS2025 Q/NASA
     - NASA Cp
     - ExoAmes Cp
     - TIPS Cp estimate
   * - 1500
     - 1.003256
     - 0.997487
     - 56.757
     - 56.430
     - 56.392
   * - 2000
     - 0.999126
     - 0.992069
     - 57.892
     - 54.437
     - 57.338
   * - 3000
     - 0.923989
     - 0.978171
     - 59.150
     - 42.653
     - 58.117
   * - 4000
     - 0.758757
     - 0.961466
     - 60.144
     - 33.416
     - 58.554

Cp units are J/mol/K. ExoAmes calorics use exact energy moments of every
supplied state. TIPS calorics are diagnostic derivatives of its rounded
10 K PF table, using a fourth-order local log-Q versus log-T polynomial
over +/-50 K; repeating with +/-100 K changes sampled TIPS2025 Cp by at most
0.047 J/mol/K (0.052 for TIPS2021). These are not published TIPS Cp measurements or uncertainty
bounds. Exact tabulated Q is used for K, without differentiating a fit.

The harmonic product's lower Q and Cp at high T are consistent with
missing anharmonic/rotation-vibration contributions, but this comparison
does not uniquely assign the entire difference. Nor does JANAF's higher Q
prove a NASA bias. Full-range replacement by the current ExoAmes file would
import a demonstrable completeness problem; replacement by TIPS would
prefer a simpler product approximation without a quantitative accuracy
case. A future completed anharmonic candidate is scientifically worthwhile;
none is adopted or fitted in this source-review stage.

Planetary scope and next work
-----------------------------

The current ``curated-h2o-v1`` archive supplies 75 converged endpoints,
537 gas species, 1500-4000 K, 1e-6-100 bar and three C/O values. The archived
SO2 chemical potentials are exactly reproduced and the sulphur budget
closes within 2.95e-12 relative. Maximum SO2 mole fraction is 1.09251e-10
and its maximum share of sulphur is 4.86763e-6, both at 1500 K, 1e-6 bar
and the baseline C/O. Other carriers can dominate: maxima of the sulphur
share are H2S 0.9140, S 0.99998 and CS 0.9347; their maxima are in different
cases. These endpoints are an archived footprint, not a new substitution
calculation. This hot hydrogen-rich grid does not cover oxidizing, cooler,
high-metallicity or irradiated photochemical atmospheres, and its tiny SO2
abundance does not diminish the observational priority.

Required calculations for a future replacement are (1) extend or correct
the full state distribution and demonstrate Q, H, S and Cp convergence with
both energy and J, (2) compare a completed anharmonic evaluation with NASA,
JANAF and TIPS using the same zero/mass/spin/atomic conventions, including
isotope and omitted electronic contributions if material, and (3) only if
a correction is supported, fit a candidate and run a conserved finite
substitution over a named relevant atmospheric domain. Photochemical SO2
requires a kinetic/irradiation calculation distinct from equilibrium.

The principal document/data follow-up is the complete partition-generation
basis behind Underwood's high-T convergence statement and the current
ExoAmes release (paper DOI above; dataset
`10.5281/zenodo.5716834 <https://doi.org/10.5281/zenodo.5716834>`_).
The full Underwood, Gamache 2021/2025 and corrigendum texts are already
inspected, so no additional paid article blocks this retention PR. Obtain
Gurvich's cited pages or original constants only when needed to resolve a
specific residual, and ATcT covariance only for a claimed quantitative
reaction-energy uncertainty. Full historical provenance collection remains
an MCEG follow-up, not an indefinite merge gate.

Reproduction and MCEG
---------------------

``tests/data/validation/so2_source_review/20261104`` contains the compact
moments, complete small PF/metadata files, exact SO2 TIPS arrays, literature
inspection notes, original SHA-256 hashes and a replay receipt. Full papers
and the 51 MB compressed state file remain outside the committed packet.
The compact replay verifies recorded moments; it does not independently
resum millions of states in CI. The hash-guarded extraction mode does so
when the original source is available.

.. code-block:: console

   python mceg/reviews/so2_source_review.py
   python mceg/reviews/so2_source_review.py --extract results/so2-source-raw --output results/so2-state-replay.json
   python mceg/reviews/so2_source_review.py --extract-tips /path/to/hapi.py --output results/so2-tips-replay.json
   pytest tests/model_evidence/so2_source_review_test.py

``mceg/contracts/so2_source_review.yaml`` supports the source construction,
normalized diagnostics and scoped retention separately. Complete physical
accuracy remains ``unknown``. Historical receipts and curated data remain
unchanged; this review introduces a new dated assessment.
