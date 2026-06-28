"""Generate an English Graphviz diagram for condensate profile execution."""

from __future__ import annotations

import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
DOT_PATH = HERE / "condensate_profile_route_en.dot"
PNG_PATH = HERE / "condensate_profile_route_en.png"

FONT = "DejaVu Sans"

DOT_SOURCE = f"""digraph condensate_profile_route {{
  graph [
    rankdir=TB,
    bgcolor="white",
    pad=0.35,
    nodesep=0.45,
    ranksep=0.62,
    splines=ortho,
    fontname="{FONT}",
    label="ExoGibbs Condensate Profile Execution Route",
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

  subgraph cluster_entry {{
    label="profile API";
    fontname="{FONT}";
    fontsize=18;
    color="#2563eb";
    penwidth=1.8;
    style="rounded";

    user_call [
      label="condensate_equilibrium_profile(setup, T, P, b)",
      fillcolor="#dbeafe"
    ];
    method_auto [
      shape=diamond,
      label="method is auto?",
      fillcolor="#fef3c7"
    ];
    explicit_method [
      label="explicit method\\nscan_hot / vmap_cold",
      fillcolor="#e0f2fe"
    ];
  }}

  subgraph cluster_auto {{
    label="auto selection";
    fontname="{FONT}";
    fontsize=18;
    color="#16a34a";
    penwidth=1.8;
    style="rounded";

    fixed_payload [
      shape=diamond,
      label="complete fixed-support\\npayload available?",
      fillcolor="#fef3c7"
    ];
    conservative_scan [
      label="conservative scan\\nscan_hot_from_top",
      fillcolor="#dcfce7"
    ];
    vmap_fixed [
      label="vmap_cold + fixed-support batch\\nprofile_warm_start_support_policy=explicit_payload",
      fillcolor="#bbf7d0"
    ];
  }}

  subgraph cluster_batch {{
    label="GPU-oriented fixed-support batch";
    fontname="{FONT}";
    fontsize=18;
    color="#7c3aed";
    penwidth=1.8;
    style="rounded";

    build_plan [
      label="fixed-support plan\\nlayer buckets / support masks / initial state",
      fillcolor="#ede9fe"
    ];
    batch_solve [
      label="batched PD-IPM solve\\npdipm_rgie_v11_activity_correction_fixed_support_batch",
      fillcolor="#ddd6fe",
      penwidth=2.2,
      color="#6d28d9"
    ];
    accepted [
      shape=diamond,
      label="all accepted?",
      fillcolor="#fef3c7"
    ];
    rescue_needed [
      shape=diamond,
      label="rescue failed\\nlayers only?",
      fillcolor="#fef3c7"
    ];
    rescue_plan [
      label="fallback-only candidate rescue\\npruned support candidates",
      fillcolor="#f3e8ff"
    ];
    merge_rescue [
      label="merge accepted rescue rows\\nkeep original profile shape",
      fillcolor="#f3e8ff"
    ];
    rescued_accepted [
      shape=diamond,
      label="rescued batch accepted?",
      fillcolor="#fef3c7"
    ];
  }}

  subgraph cluster_scan {{
    label="conservative one-layer path";
    fontname="{FONT}";
    fontsize=18;
    color="#ea580c";
    penwidth=1.8;
    style="rounded";

    scan_loop [
      label="scan_hot layer loop\\ncarry previous accepted layer as initializer",
      fillcolor="#ffedd5"
    ];
    head_route [
      label="single-layer HEAD route v1.18\\ncondensate_equilibrium()",
      fillcolor="#fed7aa"
    ];
    fresh_fallback [
      label="warm-start failure falls back to\\nfresh one-layer route",
      fillcolor="#fed7aa"
    ];
  }}

  subgraph cluster_result {{
    label="public result";
    fontname="{FONT}";
    fontsize=18;
    color="#334155";
    penwidth=1.8;
    style="rounded";

    profile_result [
      label="CondensateEquilibriumProfileResult\\nlayers / method / diagnostics / batched_arrays",
      fillcolor="#e2e8f0"
    ];
    diagnostics [
      label="diagnostics\\nexperimental_profile_fixed_support_batch\\nfallback_rescue metadata",
      fillcolor="#f1f5f9"
    ];
  }}

  user_call -> method_auto;
  method_auto -> fixed_payload;
  method_auto -> explicit_method;
  explicit_method -> conservative_scan;
  explicit_method -> vmap_fixed;
  fixed_payload -> vmap_fixed;
  fixed_payload -> conservative_scan;
  vmap_fixed -> build_plan;
  build_plan -> batch_solve;
  batch_solve -> accepted;
  accepted -> profile_result;
  accepted -> rescue_needed;
  rescue_needed -> rescue_plan;
  rescue_plan -> merge_rescue;
  merge_rescue -> rescued_accepted;
  rescued_accepted -> profile_result;
  rescued_accepted -> conservative_scan;
  rescue_needed -> conservative_scan;
  conservative_scan -> scan_loop;
  scan_loop -> head_route;
  head_route -> fresh_fallback;
  fresh_fallback -> head_route;
  head_route -> profile_result;
  profile_result -> diagnostics;
}}
"""


def main() -> None:
    DOT_PATH.write_text(DOT_SOURCE, encoding="utf-8")
    subprocess.run(["dot", "-Tpng", str(DOT_PATH), "-o", str(PNG_PATH)], check=True)
    print(f"wrote {DOT_PATH}")
    print(f"wrote {PNG_PATH}")


if __name__ == "__main__":
    main()
