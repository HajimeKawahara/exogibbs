#!/bin/csh -f

set SCRIPT_DIR = `dirname "$0"`
cd "$SCRIPT_DIR/../.."
if ( $status != 0 ) then
  echo "ERROR: could not enter the repository root"
  exit 1
endif

setenv JAX_ENABLE_X64 1
setenv JAX_PLATFORMS cuda,cpu
setenv XLA_PYTHON_CLIENT_PREALLOCATE false
setenv PYTHONHASHSEED 0

set RUNNER = benchmarks/fixed_support_v2/fixed_support_v2_unbiased_gpu_experiment.py
set OUTDIR = results/fixed_support_v2_corrected_gpu
mkdir -p $OUTDIR

echo "== fixed-support v2 corrected GPU verification =="
date
nvidia-smi
if ( $status != 0 ) then
  echo "ERROR: nvidia-smi failed"
  exit 1
endif

echo "== syntax, contracts, and frozen-input preflight =="
python -m py_compile \
  $RUNNER \
  src/exogibbs/optimize/fixed_support_v2_profile.py \
  src/exogibbs/optimize/fixed_support_v2/restoration.py \
  src/exogibbs/optimize/fixed_support_v2/types.py
if ( $status != 0 ) then
  echo "ERROR: Python syntax preflight failed"
  exit 1
endif

pytest -q \
  tests/unittests/optimize/fixed_support_v2_problem_test.py \
  tests/unittests/optimize/fixed_support_v2_m1_test.py \
  tests/unittests/optimize/fixed_support_v2_restoration_test.py \
  tests/unittests/optimize/fixed_support_v2_controller_test.py \
  tests/unittests/optimize/fixed_support_v2_continuation_test.py \
  tests/unittests/optimize/fixed_support_v2_soc_test.py \
  tests/unittests/optimize/fixed_support_v2_profile_test.py
if ( $status != 0 ) then
  echo "ERROR: v2 contract tests failed"
  exit 1
endif

python $RUNNER \
  --preflight-only \
  --output-dir $OUTDIR
if ( $status != 0 ) then
  echo "ERROR: frozen baseline or exact-state preflight failed"
  exit 1
endif

echo "== exact-state matrix and corrected support lifecycle through support 128 =="
python $RUNNER \
  --lanes solver lifecycle \
  --cases all \
  --lifecycle-families manifest \
  --output-dir $OUTDIR \
  --epsilon-schedule -11 -13 -15 -17 \
  --max-normal-iterations 1000 \
  --max-line-search-trials 20 \
  --max-restoration-calls 2 \
  --max-restoration-iterations 50 \
  --max-restoration-line-search-trials 20 \
  --v1-residual-tolerance-multiplier 2 \
  --budget-relative-floor 1e-6 \
  --support-closure-tolerance 1e-8 \
  --lifecycle-initial-topk 8 \
  --lifecycle-initial-max-support 16 \
  --lifecycle-add-per-round 8 \
  --lifecycle-max-support 128 \
  --lifecycle-max-rounds 15
if ( $status != 0 ) then
  echo "ERROR: corrected GPU verification failed"
  exit 1
endif

echo "== outputs =="
date
ls -lh $OUTDIR/preflight.json $OUTDIR/summary.json $OUTDIR/summary.md
sha256sum $OUTDIR/preflight.json $OUTDIR/summary.json $OUTDIR/summary.md

echo "== done =="
date
