VO2: Balducci equilibrium replay and partial thermal re-reduction
==============================================================

This packet tests what can be recovered from the published equilibrium data
before the Gunina2012 main article becomes available. It does not supply a
recommended replacement energy. In particular, the partial thermal update
breaks the previous agreement between second- and third-law reductions by
about 21 kJ/mol. Its approximately 1175 kJ/mol third-law endpoint must not be
adopted as a validated new atomization energy.

Reproduce from the repository root, offline::

    python mceg/reviews/vo2_balducci_reduction.py --output results/vo2_balducci_reduction.json
    pytest -q tests/model_evidence/vo2_balducci_reduction_test.py

Inputs and limits
-----------------

``table_I.csv`` transcribes the 26 rows of Balducci et al. (1983), Table I,
printed p5617. Columns 1 and 3 report minus ln K; column 2 reports plus ln K.
Small subscript final fives are extra decimal digits: 6.45(subscript 5) is
6.455. Blank entries are missing observations. Every printed ln K and H0
entry is retained without correction. The selected, rounded reaction H0
values are not refitted as observations. These are published equilibrium
estimates after ion-signal corrections, not the unavailable original ion
intensities or pressure-calibration data.

``sources.json`` pins the input files and records the user-provided paper's
hash and access scope. Full copyrighted PDFs and extracted article text are
not redistributed. Only the necessary numerical observations, existing
Table IV inputs, official JANAF table, and authored analysis are included.
``reference_audit.json`` records the independently inspected VO dissociation
reference and the unresolved original EuO reference lineage.

The original VO2 Table IV gives Phi = (H0-G)/T and H-H0. Official current
JANAF V and VO tables supply the other reaction-2 thermal functions. Their
H298 references are first converted to H0. Cubic Hermite interpolation uses
dPhi/dT = (H-H0)/T**2 and does not extrapolate. The exact 1975 JANAF pages
cited in the paper have not been independently compared with these current
tables. Original Eu and EuO thermal functions are not available in this
packet, so full historical reaction-1 and reaction-3 third-law replay remains
open.

JANAF tables use 1 bar. Table IV's standard pressure is not explicitly
stated, so the result retains both a 1 bar assumption and a 1 atm assumption
converted consistently to 1 bar. All three reactions have Delta nu = 0;
pressure conversion cancels if every species uses the same convention.
Mixing conventions introduces a reaction-2 shift of
-R T ln(1.01325), averaging -0.224301 kJ/mol here. The paper's illustrative
pressures in atm do not establish Table IV's thermodynamic convention.

Reproduction results
--------------------

Unweighted OLS of ln K versus 1/T gives the following H_T values and slope
standard errors, in kJ/mol. These fit errors are not a full physical error
budget.

* Reaction 1, 11 observations: 169.519 +/- 4.218; printed 169.5 +/- 4.2.
* Reaction 2, 21 observations: -86.978 +/- 3.088; printed -87.0 +/- 3.1.
* Reaction 3, 16 observations: 81.482 +/- 4.150; printed 82.0 +/- 2.9.

The last discrepancy remains unresolved: weighting, grouping, unrounded
observations or printing differences have not been established. No point
is discarded. Table I also prints 1046 K as reaction 2's mean temperature,
outside its 1913-2186 K observation range; the independent harmonic mean
is 2046.165 K. Both values are retained. The reaction-2 second-law H0
reduction is -73.900 kJ/mol, compared with the printed -73.8.

Third-law reaction-2 replay uses
H0 = -R T ln K + T Delta Phi. Its mean is -73.719920 kJ/mol under the
conditional Table IV = 1 atm interpretation, or -73.495619 kJ/mol under
the 1 bar interpretation. Under the former, 20 of 21 printed H0 entries
close within 0.056 kJ/mol. The visually rechecked 2087 K pair ln K = 5.51
and H0 = -74.8 kJ/mol differs by 1.036528 kJ/mol; this unresolved pair is
preserved, not repaired. Agreement of the other rows does not prove the
historical pressure convention.

Reaction 3 = reaction 1 + reaction 2; the 11 overlapping ln K triplets close
within 0.005. Historically D0(VO) = D0(EuO) + H0(1), so the two VO2 routes
are 2 D0(EuO) + 2 H0(1) + H0(2) and 2 D0(EuO) + H0(1) + H0(3).
Both share EuO twice, VO, and overlapping equilibrium data. No independent
pooling or new combined uncertainty is justified.

Modern anchor and partial thermal calculation
---------------------------------------------

Reaction 2 permits the EuO-independent relation
D0(VO2) = 2 D0(VO) + H0(2). Current curated VO uses Merriles2020
D0 = 6.545(2) eV, referred to V(4F3/2) + O(3P2). Its stated error is a
conservative assigned limit, not explicitly one standard deviation.
Keeping the historical selected H0 = -74 gives 1188.992997 kJ/mol.
Using the replayed ln K instead gives 1189.273078 or 1189.497379 kJ/mol
for the conditional 1 atm or 1 bar Table IV branches, respectively.
No JANAF formation enthalpy or log Kf is used to set this modern energy.

The following limited VO2 thermal modifications keep its zero-K D0 fixed:

* Replace the ground symmetric stretch 983 by 960 cm^-1 and bend 538 by
  291 cm^-1, holding the unobserved antisymmetric stretch at 1034 cm^-1.
  Harmonic vibrational energies are measured relative to v=0, excluding ZPE.
* Replace constant ground electronic weight 2 by Kim2014 doublets at
  0, 0.1845 and 0.813 eV. Their rotational and vibrational factors are
  assumed equal. The additional factor is Q_el/2, not Q_el, because the
  original Table IV already includes ground weight 2.

Geometry, rotation, anharmonicity, higher states and state-specific
rovibrational functions are not re-evaluated. All delta G/H/S/Cp terms
come from the same partition model, and their thermodynamic derivatives
are tested. At 2000 K the bend, symmetric-stretch and electronic delta G
terms are -8.814978, -0.272029 and -5.012219 kJ/mol, respectively.
Combined delta H is +6.470587 kJ/mol, delta S is +10.284906 J/mol/K,
and delta Cp is +2.966317 J/mol/K.

Since VO2 is a reactant, the third-law inferred H0 shift is +delta G(VO2).
Over the 21 observations, the vibrational, electronic and combined mean
shifts are -9.347543, -5.267213 and -14.614756 kJ/mol. The combined
modern-VO third-law endpoint is therefore 1174.658322 or 1174.882623
kJ/mol under the two conditional pressure branches.

However, the second-law reduction shifts by +delta H(VO2), not delta G.
At the harmonic mean temperature, that is +6.607319 kJ/mol, giving
H0(2) = -67.292231 kJ/mol and D0(VO2) = 1195.700766 kJ/mol.
The combined third-law H0 is -88.334676 or -88.110375 kJ/mol; the method
discrepancy is 21.042445 or 20.818144 kJ/mol. Its growth is approximately
T delta S. Thus this partial model does not explain both the observed
temperature slope and absolute equilibrium constant at once. This is an
adoption-blocking consistency diagnostic, not a statistical significance
claim or evidence that the spectroscopic measurements are wrong.

The original equilibrium calibration/corrections and a complete thermal
model, including the Gunina full-text treatment when available, must be
audited before assigning an updated recommended D0 or its uncertainty.
``comparison.json`` retains every observation-level result and both
pressure assumptions. ``receipt.json`` records commands, versions and
file hashes; it is not an execution requirement or a new model contract.
