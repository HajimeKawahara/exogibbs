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

Exact-zero C/N/S source branches
--------------------------------

``solve_source`` retains its strictly positive budget contract.
``solve_reduced_source`` is the explicit entry point for zero C in the Carbon
network, or any exact-zero combination of C, N and S in the sulfur/nitrogen
network. Si, Mg, O, Fe, H and Na remain positive, as do gas, silicate and metal
phase totals. A disappearing whole phase requires a different model branch.

For example, select the original S/N host with C=N=0 and finite S::

    network = load_reference()["networks"]["sulfur_nitrogen"]
    budget = np.asarray(network["element_amounts_mol"]).copy()
    budget[[network["elements"].index(element) for element in ("C", "N")]] = 0
    result = solve_reduced_source(
        network, network["cases"][0], element_amounts_mol=budget,
    )

Components containing absent elements and reactions involving those components
are removed before evaluating logarithms. No small abundance replaces zero.
Amounts retain the full source order with excluded components exactly zero.
``active_species``, ``active_elements``, ``zero_budget_elements`` and the
zero-based ``active_reaction_indices`` describe the selected support;
``reaction_residual`` follows those active source rows. Elemental residuals
retain the full element order and are exactly zero for absent elements.
At zero budgets the default initial composition projects the saved converged
source composition onto the active support. Default initial amounts scale
with the ratio of requested to reference atom totals on that support, so
uniformly rescaling the budgets does not change the normalized initial guess.
An explicitly supplied initial composition must be positive on that support
and exactly zero elsewhere; its absolute amounts are used unchanged.

All seven C/N/S absence combinations, Carbon C=0, unchanged positive source
results, and amount-scale invariance are regression controls. The reduced
S/N model preserves its own host components and reaction corrections; even at
S=N=0 it is distinct from the separately pinned Carbon model. These controls
retain the source calibration limitations and do not provide MELTS coupling,
sulfide saturation, or a bound on missing metal N and nitrides.

Ideal, gas-only helium extension
---------------------------------

``build_helium_network`` derives an optional He-bearing model from the S/N
network. It appends ``He`` to the nine-element order and ``He_gas`` to the
37-component order, while retaining the same 28 reactions. He has no reaction
or condensed component; its full inventory remains in the gas. The gas mole
fractions include He, so reactive activities use
``n_i / (n_reactive_gas + n_He) * P_bar / 1 bar``. Here ``P_bar`` is the
total pressure including He. Adding He at fixed total pressure therefore
changes the chemical partition through dilution.

For example, prescribe an illustrative atomic He/H ratio of 0.1 at the first
frozen temperature and 10000 bar, keeping the non-He budgets fixed::

    import numpy as np
    from examples.metal_silicate.sulfur_source import (
        build_helium_network, load_reference, solve_reduced_source,
    )

    network = build_helium_network(
        load_reference()["networks"]["sulfur_nitrogen"],
    )
    budget = np.asarray(network["element_amounts_mol"]).copy()
    budget[network["elements"].index("He")] = (
        0.1 * budget[network["elements"].index("H")]
    )
    result = solve_reduced_source(
        network, network["cases"][0], element_amounts_mol=budget,
        pressure_bar=10000.0,
    )

Budgets and returned amounts are in mol, with the original orders followed
by He. The default He budget is exactly zero; no abundance ratio is assumed.
``solve_reduced_source`` permits any combination of exact-zero C/N/S/He.
``solve_source`` still requires every supplied budget to be positive.
For default initial amounts, the projected source root scales with the
requested non-He atom total, then the He amount is set to its budget.
Explicit initial amounts must follow the full 38-component order and the
declared zero support; they are used unchanged.

The builder leaves the source fixture and input network unchanged. Derived
cases contain thermochemical inputs and initial guesses, without copying
saved source results or probes as He solutions. Derived model IDs and the
``ideal-He extension of source equations`` evidence label distinguish these
calculations; ``source_model_id`` and ``source_case_id`` identify their origins.

Regression controls recover the original roots at He=0, conserve gas-only He,
and verify exact-zero support and uniform amount scaling. A separate analytic
dilution check starts from a source root at pressure ``P0`` with gas amount
``n_gas0``. Adding ``n_He`` while setting
``P = P0 * (1 + n_He / n_gas0)`` preserves every reactive partial pressure and
therefore the original non-He amounts in this frozen source model.

This extension assumes ideal He with no dissolution in silicate or metal.
It does not provide He solubility, nonideal gas fugacity, thermal properties,
or a planetary mass/pressure closure. The original calibration and fixed-phase
limitations continue to apply.

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
