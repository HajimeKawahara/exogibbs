Full sulfur, nitrogen and carbon source controls
==================================================

``examples/metal_silicate/sulfur_source.py`` reproduces two separately pinned
GCE networks at fixed local temperature and pressure. The sulfur/nitrogen
version retains all 37 components, nine finite elemental budgets and 28
reactions. The carbon version retains all 26 components, seven budgets and
19 reactions, with S absent from that independently defined network. Both
retain gas, silicate and metal as positive-amount phases.

These are **source reproductions**. The references do not establish empirical
calibration, stable liquid phases, a common multicomponent excess Gibbs
energy, or a planetary pressure closure. They preserve the audited Young
``source_full`` and ``completed_dry`` controls as separate model records.

Offline execution and acceptance
----------------------------------

Run the four frozen local cases, with JAX double precision enabled::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
      python examples/metal_silicate/sulfur_source.py

The solver reports absolute component, phase and phase-element amounts in
mol. It independently recomputes relative elemental residuals and all
dimensionless reaction residuals at the final amounts. Acceptance requires
elemental residuals below ``1e-9`` and reaction residuals below ``1e-8``.
The reference comparisons also check changed initial compositions and
amount scaling. A failed root raises an error instead of returning an
accepted physical state.

``sulfur_reference.json`` freezes the original source equations evaluated
with SymPy/NumPy and independently solved with SciPy. Its two temperatures
are 2350 and 3000 K; pressure is 10000 bar for the sulfur/nitrogen roots and
1000 bar for the carbon roots, each using its own original elemental inputs. Additional
unequilibrated compositions at 1 and 1000 bar check the full residual,
including the pressure terms. These temperatures extrapolate the source
MgO liquid fit and do not define a supported planetary domain.

Regenerate the fixture from an existing local checkout, without network
access::

    python examples/metal_silicate/extract_sulfur_reference.py \
      /path/to/GlobalChemicalEquilibrium_Release \
      --output examples/metal_silicate/sulfur_reference.json

Extraction alone requires SymPy. Every executed source file and its inputs
must match their stored SHA256 hashes, at GCE commit
``31558873d8da460c3cd11986b574cb347621b43d``. The optional extractor executes
the unmodified GPL-3.0 source in a temporary directory; that source code is
not vendored or required at runtime. The ordinary example uses only the
committed numerical fixture, JAX, NumPy and SciPy.

Conventions preserved for audit
---------------------------------

* Gas and silicate components count formula-unit moles, while metal
  components count atomic moles. ``FeO15_silicate`` means FeO1.5, including
  1.5 O atoms in every independent finite balance.
* Mole fractions are normalized within the complete source phase. Gas
  activities use ``x * P_bar / 1 bar``. R14 in both of these versions uses
  the supplied pressure; the Young-version fixed 10000 bar correction does
  not apply.
* The GCE Calvo approximation uses the original full silicate association
  fractions in ``logC_S = -5.704 + 3.15*x_FeO + 0.12*x_MgO + 0.75*x_Na2O``.
  It does not renormalize them to oxide fractions. The source correction is
  exactly ``-2.302585093*(-logC_S + ln(x_FeO))``. Its mixed log-base
  convention, omitted pressure/CaO/TiO2/K2O terms, and normalization remain
  explicit audit questions for the original Calvo calibration.
* This host-dependent S correction is a Gibbs exchange equation, not an
  intrinsic metal-only activity. No ExoEOS alloy coefficient is inferred
  from it.
* The carbon coefficient preserves ``-2.303*19.5*ln(1-x_O_metal)``. The
  Carbon version's last reaction uses ``GRT19`` and omits ``ln(gamma_O)``;
  the sulfur/nitrogen version includes that oxygen correction. The two
  version IDs must remain separate when comparing their partition outputs.
* The N dissolution equation preserves its source sign and gas constant,
  including the source's small change of R for the final N reactions.
  Reproducing this expression does not validate its original Bernadou
  concentration convention or replace the existing total-elemental-N
  solubility API with a molecular-N2 interpretation.

Physical integration gates
----------------------------

No SCSS law or sulfide phase is supplied by these source controls. SCSS
cannot replace finite unsaturated S partition. A later empirical saturation
branch must declare sulfide-state versus total S, its dry or hydrous mass
denominator, the sulfide identity, and exact branch disappearance. In
particular, FeS/FeSO4 total melt S must not be compared directly with a
sulfide-only SCSS.

Original Calvo, Blanchard and Fischer calibration tables, a selected SCSS
dataset, graphite/carbide alternatives, and metal-N/nitride constraints are
not present in this fixture. These data and competing phases remain gates
for a physical S+C/N prediction. The separate C and S/N source roots are
useful reproducible controls, but cannot establish calibrated S/C cross
interactions or a quantitative bound on omitting N from the new H-bearing
mechanism model.
