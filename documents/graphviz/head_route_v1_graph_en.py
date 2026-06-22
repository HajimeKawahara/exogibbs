"""Generate an English Graphviz diagram for ExoGibbs HEAD route v1.18."""

from __future__ import annotations

import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
DOT_PATH = HERE / "head_route_v1_en.dot"
PNG_PATH = HERE / "head_route_v1_en.png"

FONT = "DejaVu Sans"

DOT_SOURCE = f"""digraph head_route_v1_en {{
  graph [
    rankdir=TB,
    bgcolor="white",
    pad=0.35,
    nodesep=0.45,
    ranksep=0.62,
    splines=ortho,
    fontname="{FONT}",
    label="ExoGibbs Condensate HEAD Route v1.18",
    labelloc=t,
    fontsize=26
  ];
  node [
    shape=box,
    style="rounded,filled",
    fontname="{FONT}",
    fontsize=13,
    margin="0.12,0.08",
    color="#3f4a5a",
    penwidth=1.4,
    fillcolor="#f8fafc"
  ];
  edge [
    fontname="{FONT}",
    fontsize=11,
    color="#475569",
    arrowsize=0.8,
    penwidth=1.2
  ];

  subgraph cluster_api {{
    label="Public API Boundary";
    fontname="{FONT}";
    fontsize=18;
    color="#2563eb";
    penwidth=1.8;
    style="rounded";

    user_call [
      label="User call\\ncondensate_equilibrium(setup, T, P, b, options)",
      fillcolor="#dbeafe"
    ];
    setup_validation [
      label="Setup / option validation\\nCondensateChemicalSetup\\nCondensateEquilibriumOptions",
      fillcolor="#e0f2fe"
    ];
    support_input_decision [
      shape=diamond,
      label="Are support_indices\\nprovided explicitly?",
      fillcolor="#fef3c7"
    ];
  }}

  subgraph cluster_support {{
    label="Positive Support and Initial Seed";
    fontname="{FONT}";
    fontsize=18;
    color="#16a34a";
    penwidth=1.8;
    style="rounded";

    positive_support [
      label="Activity-driven support outer loop\\ndefault: track all positive candidates\\ntemperature validity + native activity",
      fillcolor="#dcfce7"
    ];
    explicit_support [
      label="Explicit support payload\\ncaller-provided support / seed",
      fillcolor="#f0fdf4"
    ];
    empty_support [
      shape=diamond,
      label="Is positive support\\nempty?",
      fillcolor="#fef3c7"
    ];
    gas_only_exit [
      label="Gas-only result\\nno-condensate boundary\\nstatus = converged",
      fillcolor="#e0f2fe"
    ];
    gas_only_budget_gate [
      shape=diamond,
      label="Full-budget gate\\naccepted?",
      fillcolor="#fef3c7"
    ];
    strict_gas_retry [
      label="empty_support_strict_gas_retry\\nnative gas equilibrium\\nepsilon_crit = 1e-12",
      fillcolor="#bae6fd"
    ];
  }}

  subgraph cluster_warmstart {{
    label="Warm-start Candidate Generation";
    fontname="{FONT}";
    fontsize=18;
    color="#0f766e";
    penwidth=1.8;
    style="rounded";

    warm_start [
      label="HEAD route warm-start\\nhead_route_warm_start.py",
      fillcolor="#ccfbf1"
    ];
    baseline_seed [
      label="baseline_positive_support_seed\\ndefault: max-density seed\\nelement-budget capacity gauge",
      fillcolor="#f0fdfa"
    ];
    depleted_refresh [
      label="depleted_gas_refresh_native_gas_solver\\nsubtract Ac @ m from budget\\nrecompute gas log-density",
      fillcolor="#f0fdfa"
    ];
  }}

  subgraph cluster_solver {{
    label="Restricted Support Solver";
    fontname="{FONT}";
    fontsize=18;
    color="#7c3aed";
    penwidth=1.8;
    style="rounded";

    restricted_solver [
      label="restricted support solver\\nsolve_restricted_support_condensate_layer()\\ndefault: pdipm_rgie_v11_activity_correction",
      fillcolor="#ede9fe"
    ];
    solver_success [
      shape=diamond,
      label="solver_success?",
      fillcolor="#fef3c7"
    ];
    finite_warm_state [
      shape=diamond,
      label="Is a finite warm-start\\nstate available?",
      fillcolor="#fef3c7"
    ];
    no_state_fail [
      label="not_converged\\nno refresh warm-start state",
      fillcolor="#fee2e2"
    ];
  }}

  subgraph cluster_lifecycle {{
    label="HEAD Route Lifecycle";
    fontname="{FONT}";
    fontsize=18;
    color="#ea580c";
    penwidth=1.8;
    style="rounded";

    support_boundary [
      label="Support boundary assembly\\nsupport_boundary.py\\nln_nk, ln_mk, active Ac, budget",
      fillcolor="#ffedd5"
    ];
    continuation_input [
      label="Continuation input assembly\\ncontinuation_input.py\\nq/r/lambda/source/budget frame",
      fillcolor="#ffedd5"
    ];
    primary_continuation [
      label="Primary continuation entry\\nalgorithm-v1.1 high-start policy\\noptimize/condensate_algorithm_v11_callsite.py",
      fillcolor="#fed7aa"
    ];
    pdipm_rgie_core [
      label="RGIE / PD-IPM core\\npdipm_core: reduced direction\\nscalar fraction-to-boundary\\nfilter/restoration + tiny-step handling",
      fillcolor="#fdba74",
      penwidth=2.2,
      color="#c2410c"
    ];
    primary_centered [
      shape=diamond,
      label="Final barrier\\ncentered?",
      fillcolor="#fef3c7"
    ];
    center_fallback [
      label="Center-primary fallback\\ncenter_primary_fallback.py\\ncenter ratio + budget guard",
      fillcolor="#fed7aa"
    ];
    electron_refresh [
      label="Source-convention-safe electron refresh\\nelectron_refresh.py\\nq + source ~= Ag.T @ lambda",
      fillcolor="#fed7aa"
    ];
    frontier_refresh [
      label="Frontier refresh\\nfrontier_refresh.py\\nevaluate adaptive floor candidates",
      fillcolor="#fed7aa"
    ];
    route_selector [
      label="Route selector\\nhead_route_selector.py\\nselect primary / fallback / refresh",
      fillcolor="#fb923c"
    ];
    route_result [
      label="HEAD route result\\nroute_result.py\\nselected_route / integrated_status",
      fillcolor="#fdba74"
    ];
    lifecycle_accepted [
      shape=diamond,
      label="Lifecycle\\naccepted?",
      fillcolor="#fef3c7"
    ];
  }}

  subgraph cluster_retries {{
    label="v1.18 Support / Budget Repair Gates";
    fontname="{FONT}";
    fontsize=18;
    color="#9333ea";
    penwidth=1.8;
    style="rounded";

    retry_selection [
      shape=diamond,
      label="Is retry / repair needed?\\nfull-budget gate / support closure / fallback",
      fillcolor="#fef3c7"
    ];
    lifecycle_final_state_growth [
      label="lifecycle_final_state_support_closure_retry\\ngrow support from PD-IPM final_state\\nsupport-free outer-loop exhaustion",
      fillcolor="#f3e8ff"
    ];
    explicit_support_closure [
      label="explicit_support_closure_retry\\ncaller support is too narrow\\nadd support from inactive driving",
      fillcolor="#f3e8ff"
    ];
    full_budget_restoration [
      label="Full-budget feasibility restoration\\ngas log amounts + active condensate amounts\\napply only if accepted/improved",
      fillcolor="#ede9fe"
    ];
    retry_selection_summary [
      label="support_closure_retry_selection\\nbest ExoGibbs-native closure score\\nFastChem4 constructor inputs: none",
      fillcolor="#ede9fe"
    ];
  }}

  subgraph cluster_public_result {{
    label="Public Result / Standard Gate";
    fontname="{FONT}";
    fontsize=18;
    color="#334155";
    penwidth=1.8;
    style="rounded";

    standard_gate [
      label="Standard gate\\nhead_route_standard_gate.py\\nmetric_status -> tier/status",
      fillcolor="#e2e8f0"
    ];
    result [
      label="CondensateEquilibriumResult\\ngas_ln_n, gas_x, condensate_amounts\\nacceptance_tier, converged, diagnostics",
      fillcolor="#f1f5f9"
    ];
    tier1 [
      label="tier 1\\ntight residual\\nconverged",
      fillcolor="#dcfce7"
    ];
    tier23 [
      label="tier 2/3\\naccepted with caveat\\nconverged_with_caveat",
      fillcolor="#fef9c3"
    ];
    not_converged [
      label="not_converged\\nsolver failed / lifecycle not accepted / full-budget gate rejected",
      fillcolor="#fee2e2"
    ];
    v18_final_gate [
      shape=diamond,
      label="v1.18 final gates\\nstandard status + full-budget gate\\nsupport closure gate + diagnostics",
      fillcolor="#fef3c7"
    ];
    native_seed_fallback [
      label="Native seed fallback\\nfinite warm-start boundary +\\nmax-density seed\\nroute = native_budget_seed_fallback_budget_tradeoff",
      fillcolor="#fef9c3",
      color="#ca8a04",
      penwidth=1.8
    ];
  }}

  user_call -> setup_validation;
  setup_validation -> support_input_decision;
  support_input_decision -> positive_support [label="no"];
  support_input_decision -> explicit_support [label="yes"];
  positive_support -> empty_support;
  explicit_support -> empty_support;
  empty_support -> gas_only_budget_gate [label="yes"];
  gas_only_budget_gate -> gas_only_exit [label="yes"];
  gas_only_budget_gate -> strict_gas_retry [label="no"];
  strict_gas_retry -> gas_only_exit;
  empty_support -> warm_start [label="no"];

  warm_start -> baseline_seed;
  warm_start -> depleted_refresh;
  baseline_seed -> restricted_solver;
  depleted_refresh -> restricted_solver;

  restricted_solver -> solver_success;
  solver_success -> support_boundary [label="yes"];
  solver_success -> finite_warm_state [label="no"];
  finite_warm_state -> no_state_fail [label="no"];
  finite_warm_state -> support_boundary [label="yes"];

  support_boundary -> continuation_input;
  continuation_input -> primary_continuation;
  primary_continuation -> pdipm_rgie_core;
  pdipm_rgie_core -> primary_centered;
  primary_centered -> route_selector [label="yes"];
  primary_centered -> center_fallback [label="no"];
  center_fallback -> electron_refresh;
  electron_refresh -> frontier_refresh;
  frontier_refresh -> route_selector;
  route_selector -> route_result;
  route_result -> retry_selection;
  retry_selection -> lifecycle_final_state_growth [label="support-free"];
  retry_selection -> explicit_support_closure [label="explicit support"];
  retry_selection -> full_budget_restoration [label="budget"];
  lifecycle_final_state_growth -> retry_selection_summary;
  explicit_support_closure -> retry_selection_summary;
  full_budget_restoration -> retry_selection_summary;
  retry_selection_summary -> route_result;
  route_result -> lifecycle_accepted;
  lifecycle_accepted -> standard_gate [label="yes"];
  lifecycle_accepted -> v18_final_gate [label="no"];
  v18_final_gate -> native_seed_fallback [label="fallback"];
  v18_final_gate -> standard_gate [label="accepted/rejected"];
  native_seed_fallback -> standard_gate;

  standard_gate -> tier1 [label="tight_residual_components"];
  standard_gate -> tier23 [label="accepted with caveat"];
  standard_gate -> not_converged [label="not accepted"];
  tier1 -> result;
  tier23 -> result;
  not_converged -> result;
  no_state_fail -> result;
  gas_only_exit -> result;
}}
"""


def main() -> None:
    DOT_PATH.write_text(DOT_SOURCE, encoding="utf-8")
    subprocess.run(
        ["dot", "-Tpng", str(DOT_PATH), "-o", str(PNG_PATH)],
        check=True,
    )
    print(f"wrote {DOT_PATH}")
    print(f"wrote {PNG_PATH}")


if __name__ == "__main__":
    main()
