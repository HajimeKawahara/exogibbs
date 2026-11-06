CH2 energy candidate and thermal reconstruction audit
=====================================================

Prefer the retained-thermal ATcT 1.228 energy candidate for a separately
specified production update. Do not adopt the reconstructed RRHO thermals:
the published triplet-only and singlet-only tables are reproducible, but their
physically weighted combination does not reproduce the published equilibrated
table. This new finding changes the thermal follow-up proposed in the earlier
source review. Existing source packets and production data remain unchanged.

Two candidates each receive 150 fresh fixed-network endpoints. The second is
explicitly a sensitivity built from published molecular constants, not a
reproduction of the IUPAC equilibrated table. No production adoption or merge
is claimed here.

Energy, atomic gauge and retained residual
------------------------------------------

The reaction is C(g) + 2 H(g) -> CH2(g), with a 1 bar standard state. The
same-version ATcT 1.228 neutral-atom cycle is
711.370 + 2(216.034) - 391.046 = 752.392 kJ/mol. The physical JANAF source D0
is 757.340 kJ/mol. The primary candidate therefore adds 4948 J/mol to the
literal current reaction G and H, preserving S and Cp exactly. No additional
zero-point correction, singlet formation energy or fitted reference alignment
is applied. The formal current-fit R*a is 758027.186028 J/mol; it is not
relabeled as the physical D0. Marginal ATcT uncertainties do not specify the
correlated atomization uncertainty.

For the thermal sensitivity, let phi be a molecular or atomic G-H0 function.
NASA coefficients are normalized with the common equilibrium gas constant
R=8.31446261815324 J/mol/K after removing the printed Hf298-(H298-H0) zero
using native NASA R=8.314510. The complete residual is defined as

.. math::

   r(T)=G_{\rm current}(T)-[-D_{0,\rm JANAF}
        +\phi_{\rm NASA,CH_2}(T)-\phi_{\rm NASA,C}(T)-2\phi_{\rm NASA,H}(T)].

The reconstructed candidate uses

.. math::

   G_{\rm RRHO}(T)=r(T)-D_{0,\rm ATcT}
        +\phi_{\rm reconstructed,CH_2}(T)-\phi_{\rm NASA,C}(T)-2\phi_{\rm NASA,H}(T).

Thus the atomic gauge and the complete residual **relative to the NASA thermal
proxy** remain fixed. This is not a direct replacement of a literal JANAF
molecular table. The triplet-only control uses the same proxy and gauge.
Using NASA thermals again under this residual definition is exactly identical
to the energy-only candidate; that identity is tested and is not counted as
another independent candidate or another set of solves.

Reconstruction from inspected molecular data
---------------------------------------------

The state sum uses separate asymmetric-rotor/harmonic-oscillator terms for
X3B1, a1A1 and b1B1. Their T0 origins are 0, 3147 and 11497 cm-1, converted by
hc/k; electronic statistical weights are 3, 1 and 1. Each rotational symmetry
number is 2. Nuclear-spin multiplicity is excluded once, and the translation
uses mean mass 14.02658 u. These are elemental thermodynamics with main-isotope
internal constants, not an isotope mixture or a fixed ortho/para model.

.. list-table:: Constants used in the independent reconstruction
   :header-rows: 1

   * - State
     - A0, B0, C0 (cm-1)
     - Vibrational fundamentals treated as harmonic (cm-1)
   * - X3B1
     - 73.811, 8.450, 7.184
     - 3031, 963.1, 3190
   * - a1A1
     - 20.118, 11.205, 7.069
     - 2806, 1352.6, 2865
   * - b1B1
     - 73.8, 8.59, 7.2
     - 3000, 570, 3000

`IUPAC 2005 <https://doi.org/10.1063/1.1724828>`_ supplies the first two
sets and the electronic origins. The triplet symmetric stretch is scaled
theory; its other two fundamentals and the listed singlet fundamentals are
experimental. The additional b-state constants come from the author-hosted
`Burcat/Ruscic ANL-05/20 report <https://atct.anl.gov/~ruscic/reprints/Burcat%202005.pdf>`_,
printed pages 51-52. They are an approximate uncoupled-state model, not a
Renner-Teller calculation. Its singlet-state record specifies 8350 cm-1 above
a1A1, consistent with 11497-3147. The equilibrium record prints 3147 as the
a-state symmetric stretch and 319.2 kJ/mol as Hf298; these apparent
inconsistencies are recorded, not silently adopted. We use the IUPAC frequency
2806 and the independently evaluated ATcT energy.

The reported Cartesian blocks also disagree with the listed equilibrium
geometries. The experimental rotational constants, rather than those blocks,
define the rotor calculation. No parameter is adjusted to force agreement
with a caloric table.

Two component tables agree; the combined table does not
-------------------------------------------------------

All 16 printed temperature nodes from 150 to 3000 K are pinned for each of
three tables. The independent reconstruction matches the triplet-only table
within 1.076 J/mol in H, 0.000775 J/mol/K in S, and 0.000597 J/mol/K in Cp.
For singlet-only methylene, the corresponding maxima are 0.468 J/mol,
0.000749 J/mol/K and 0.000494 J/mol/K. The small differences are compatible
with the precision of printed constants and caloric values.

An independent calculation uses **only the two printed component tables**,
adds the 3147 cm-1 ground-state gap to the singlet energy, combines their
Gibbs functions by a log-sum, and evaluates the mixed enthalpy and heat-capacity
moments. It agrees with the reconstructed RRHO model but disagrees with the
published equilibrated table. This check does not depend on choosing missing
rovibrational constants or refitting a partition function.

.. list-table:: Direct combination of printed components minus printed equilibrium table
   :header-rows: 1

   * - Temperature (K)
     - Delta G (J/mol)
     - Delta H (J/mol)
     - Delta S (J/mol/K)
     - Delta Cp (J/mol/K)
   * - 1500
     - 1314.377
     - -1915.740
     - -2.153411
     - -1.669894
   * - 2000
     - 2502.208
     - -2577.685
     - -2.539947
     - -0.986250
   * - 3000
     - 5175.220
     - -2974.938
     - -2.716719
     - 0.132984

At 298.15 K the reconstructed singlet contribution to heat content is about
0.0053 J/mol, whereas the printed combined and triplet tables differ by
approximately 5 J/mol. `ATcT 2015 <https://doi.org/10.1021/acs.jpca.5b01346>`_
distinguishes regular and partial triplet/singlet species and states a singlet
contribution below 0.001 kJ/mol with its anharmonic partition functions.
Those complete partition functions are not supplied by that paper and are not
reconstructed here. The precise cause of the 2005 combined-table discrepancy
remains unresolved; it is not attributed to a specific software error.

The earlier source review correctly recorded the printed equilibrium-table
comparison with NASA. That comparison is not reproduced by the molecular-data
model: its thermal K ratio to the NASA proxy is about 0.991-0.995 across the
five candidate temperatures. The difference between these two constructions
must remain explicit when interpreting the original recommendation.

Extension and independent numerical checks
-------------------------------------------

The displayed caloric tables end at 3000 K; the paper also prints polynomial
fits with nominal ranges extending to 6000 K. A printed fit range does not
resolve the component/combined-table inconsistency or establish physical
completeness. The reconstructed model is analytically evaluated on
1500-4000 K solely as a declared finite sensitivity. In particular,
3000-4000 K is an explicit model extension, not an author-validated table.

A single C2 Gibbs function gives scalar/JAX G/H/S/Cp on 2501 temperatures.
The current factory is reproduced within 7.11e-15 in h; automatic and
independent analytic G/H derivatives agree within 1.1e-9 J/mol, with S/Cp
errors below 5e-13 J/mol/K. Explicit oscillator sums through 128 quanta and
an independent asymmetric-rigid-rotor diagonalization through J=100 and 140
check the numerical extension. At 4000 K oscillator truncation changes H by
less than 2.5e-8 J/mol; raising the rotor cutoff changes H by less than
1.0e-6 J/mol. The quantum-rotor versus classical-rotor control changes G by
about -28.4 J/mol at 4000 K. This spin-averaged rotor check does not validate
anharmonicity, dissociation, spin-resolved ortho/para populations or
nonadiabatic coupling.

The full author manuscript and official SI for
`Egorov et al. 2024 <https://doi.org/10.1021/acs.jpca.4c04205>`_ were also
inspected. Their Q convergence is demonstrated at 296 K with Jmax=18
(triplet) and 20 (singlet), not over 1500-4000 K. The quoted Q296 values,
169.347 and 265.028, use conventions including nuclear statistical weights;
the figure uses J=N, S=0. They cannot be imported with an assumed extra
triplet factor. The low triplet barrier to linearity (about 2000 cm-1) and
singlet Renner-Teller mixing are physical limitations of an independent
rigid-rotor approximation. The SI contains 296 K line lists and potential/
dipole data, not a complete high-temperature state/partition archive.
The new paper therefore does not supply a complete replacement thermal model
for this candidate calculation.

Finite fixed-network substitutions
------------------------------------

Both candidates are compared with the named ``curated-tio2-v1`` baseline:
75 ion-inclusive cases with 537 species and 75 neutral cases with 421 species
per candidate, for **300 fresh endpoints**. The 150 unique baseline endpoints
are reused only after archive hashes, species, formula matrices, temperatures,
pressures, elemental budgets and chemical potentials match the named factory.
The grid has 1500, 2000, 2500, 3000 and 4000 K; pressures 1e-6, 1e-4, 0.01,
1 and 100 bar; and C/O=0.5888436554, 1 and 1.5.

.. list-table:: Solved CH2 mole-fraction ratios to the unchanged baseline
   :header-rows: 1

   * - Candidate
     - Ion-inclusive minimum / maximum
     - Neutral minimum / maximum
   * - Retained-thermal ATcT energy
     - 0.672510 / 0.868368
     - 0.672510 / 0.868368
   * - Reconstructed RRHO plus ATcT sensitivity
     - 0.669461 / 0.863749
     - 0.669461 / 0.863749

The largest absolute response and carbon-allocation change occur at 4000 K,
100 bar, C/O=1.5. In the ion-inclusive network the original CH2 mole fraction
3.05859755e-5 becomes 2.65474634e-5 for energy-only and 2.64057717e-5 for the
RRHO sensitivity. The fraction of carbon in CH2 changes from **2.60379173%**
to **2.25999205%** and **2.24792979%**, respectively. These are actual coupled
abundance responses, distinct from the analytic energy-only K factors
0.672510 at 1500 K and 0.861761 at 4000 K. Neutral-network carbon fractions
are 2.60383309%, 2.26002794% and 2.24796550% at the same condition.

There are 34 additional fresh selected cold solves: 6 tighter baseline,
12 tighter candidate, 10 finite-path and 6 triplet-only controls. The triplet
control changes the target into a different partial thermodynamic species;
its selected CH2 ratios to the equilibrated RRHO sensitivity are about
0.8670-0.9771. It is not an interchangeable equilibrium model.

All solves converge. The largest major H/C/O relative residual is 2.69e-13,
charge residual 2.82e-14, and formation-equation residual 1.37e-12 in ln units.
The largest relative residual over all elements, including traces, is 1.40e-6;
all-species tighter-run differences reach 1.40e-5 in ln(x), while the CH2
difference is below 3e-13. Four independent standalone FastChem comparisons
on selected ion-inclusive candidate cases agree within 2.34e-8 dex.
These two solver comparisons use the same species, potentials and budgets.

Feasibility-corrected, same-model Gibbs exchange signs are resolved above the
displayed floating-point scale in 63/75 conditions for each candidate and
network. The remaining 12 are unresolved by this diagnostic. Absolute Gibbs
values of different models do not choose a physical source.

Decision, evidence and remaining work
--------------------------------------

The same-version ATcT energy update is the supported direction: it preserves
the current thermals and has a measurable, numerically verified CH2 response.
Its adoption is a separate production change. Do not recommend the thermal
sensitivity for adoption merely because it lies close to NASA or because the
solver converges. Resolve the inconsistent combined table and quantify the
large-amplitude/high-temperature thermal model before changing the thermals.
No new ancestral full text is a prerequisite to evaluating this bounded
energy update.

``ch2_candidate.yaml`` records supported source-reconstruction diagnostics,
continuous candidate thermodynamics, finite substitutions and the scoped
energy preference. Universal Gibbs resolution, full physical accuracy and
production adoption remain unknown. Numerical closure does not establish
reaction covariance, isotope-mixture accuracy, condensation, transport,
photochemistry or planetary applicability.

Offline checks live in ``ch2_candidate_test.py`` and
``ch2_equilibrium_impact_test.py``. They replay source and execution hashes,
component-table recombination, independent finite sums, scalar/JAX calorics,
archive identity, all conserved constraints and the recorded responses without
new equilibrium solves. Existing source-review evidence and runtime bytes are
unchanged.
