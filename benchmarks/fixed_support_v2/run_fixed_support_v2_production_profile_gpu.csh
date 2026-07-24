#!/bin/csh -f

set SCRIPT_DIR = `dirname "$0"`
cd "$SCRIPT_DIR/../.."
if ( $status != 0 ) then
  echo "ERROR: could not enter the repository root"
  exit 1
endif

if ( $#argv != 0 && $#argv != 4 ) then
  echo "usage: $0 [max_cold_compile max_cold_wall max_warm_execute max_warm_wall]"
  exit 2
endif

setenv JAX_ENABLE_X64 1
setenv JAX_PLATFORMS cuda,cpu
setenv XLA_PYTHON_CLIENT_PREALLOCATE false
setenv PYTHONHASHSEED 0

set RUNNER = benchmarks/fixed_support_v2/production_profile_gpu_gate.py
set OUTDIR = results/fixed_support_v2_production_profile
set BUDGET_ARGS = ()
if ( $#argv == 4 ) then
  set BUDGET_ARGS = ( \
    --max-cold-compilation-seconds "$1" \
    --max-cold-wall-seconds "$2" \
    --max-warm-execution-seconds "$3" \
    --max-warm-wall-seconds "$4" \
  )
endif
mkdir -p $OUTDIR

echo "== fixed-support v2 production-profile GPU gate =="
date
nvidia-smi
if ( $status != 0 ) then
  echo "ERROR: nvidia-smi failed"
  exit 1
endif

python -m py_compile \
  $RUNNER \
  src/exogibbs/api/condensate_equilibrium.py \
  src/exogibbs/condensates/fixed_support_v2_policy.py \
  src/exogibbs/optimize/fixed_support_v2_profile.py
if ( $status != 0 ) then
  echo "ERROR: Python syntax preflight failed"
  exit 1
endif

env JAX_PLATFORMS=cpu python $RUNNER \
  --preflight-only \
  --output-dir $OUTDIR
if ( $status != 0 ) then
  echo "ERROR: production-profile input preflight failed"
  exit 1
endif

python $RUNNER \
  --output-dir $OUTDIR \
  --require-approved-budgets \
  $BUDGET_ARGS
if ( $status != 0 ) then
  echo "ERROR: production-profile GPU gate failed"
  exit 1
endif

echo "== outputs =="
date
ls -lh \
  $OUTDIR/production_preflight.json \
  $OUTDIR/summary.json \
  $OUTDIR/summary.md
sha256sum \
  $OUTDIR/production_preflight.json \
  $OUTDIR/summary.json \
  $OUTDIR/summary.md

echo "== done =="
date
