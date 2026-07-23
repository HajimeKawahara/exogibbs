# Solver Component Ablation Policy

This note defines how experimental condensate-solver components are evaluated
before they are promoted to the GPU batch default.

## Problem

Sequential experiments can create a false adoption path:

1. component A improves the current baseline;
2. component B improves A;
3. component C improves AB;
4. ABC becomes the apparent candidate.

That is not sufficient.  If B alone gives the same result as ABC within the
accepted tolerance, B is the better default because it is simpler and has a
smaller behavioral surface.

## Component Registry

Every solver experiment should name each independently switchable idea as a
component.  Examples:

- `activity_topk`: ExoGibbs-native activity support expansion with top-k cap.
- `activity_all`: ExoGibbs-native FastChem-style activity support expansion.
- `depleted_refresh`: gas initialization from condensate-depleted budget.
- `main_rem_split`: FastChem-style `log_activity > -0.1` main/rem split.
- `maxdensity_seed`: budget-safe FastChem-style maxDensity seed.
- `post_prune`: post-solver activity pruning.
- `two_candidate`: fixed compact/expanded candidate selection.

Component names describe mechanisms, not historical run names.

## Required Comparisons

A new component is not promoted from an additive run alone.  The minimum useful
comparison set for proposed component C on top of a current candidate B is:

- baseline;
- B;
- C alone when feasible;
- B+C;
- any older superset such as A+B+C if it is still being considered.

For interacting components, add the smallest factorial block that can identify
the interaction.  For example, if `activity_all` and `main_rem_split` are
suspected to interact, run:

- neither;
- `activity_all`;
- `main_rem_split`;
- `activity_all+main_rem_split`.

## Adoption Rule

Promote the smallest Pareto-frontier component set that satisfies all gates.
The default score is:

- primary accuracy: max major-overlap gas log10 error at `>=1e-8`;
- validity gates: convergence count, max budget residual, inactive driving;
- complexity: number of components, then max support size.

A larger component set is rejected as redundant if a strict subset is within the
primary dex tolerance and does not regress convergence, budget, or inactive
driving gates.

## Tool

Historical experiments used the local scratch script
`volatiles_code/summarize_component_ablation.py` on completed comparison JSON
files:

```bash
python volatiles_code/summarize_component_ablation.py \
  --variant base:volatiles_artifacts/run_base.json:base \
  --variant B:volatiles_artifacts/run_b.json:base+activity_all \
  --variant ABC:volatiles_artifacts/run_abc.json:base+activity_all+main_rem_split+maxdensity_seed \
  --output-md volatiles_artifacts/component_ablation_summary.md \
  --output-json volatiles_artifacts/component_ablation_summary.json
```

The scratch script and raw artifacts are intentionally not archived.  Its
report marks Pareto-frontier variants and lists redundant subset candidates.

FastChem4 outputs remain comparison targets only.  Component experiments must
use ExoGibbs-native quantities as solver inputs.
