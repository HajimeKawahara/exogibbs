#!/bin/csh -f

set SCRIPT_DIR = `dirname "$0"`
cd "$SCRIPT_DIR/../.."
if ( $status != 0 ) then
  echo "ERROR: could not enter the repository root"
  exit 1
endif
setenv PYTHONPATH "$cwd/src"

setenv JAX_ENABLE_X64 1
setenv JAX_PLATFORMS cuda,cpu
setenv XLA_PYTHON_CLIENT_PREALLOCATE false
setenv PYTHONHASHSEED 0

set RUNNER = benchmarks/fixed_support_v2/fixed_support_v2_unbiased_gpu_experiment.py
set OUTDIR = results/fixed_support_v2_water128_gpu100
mkdir -p $OUTDIR

echo "== fixed-support v2 water128 restoration-limit verification =="
date
nvidia-smi
if ( $status != 0 ) then
  echo "ERROR: nvidia-smi failed"
  exit 1
endif

echo "== syntax and focused contracts =="
python -m py_compile \
  $RUNNER \
  src/exogibbs/optimize/fixed_support_v2/types.py \
  src/exogibbs/optimize/fixed_support_v2/controller.py \
  src/exogibbs/optimize/fixed_support_v2/continuation.py \
  src/exogibbs/optimize/fixed_support_v2_profile.py
if ( $status != 0 ) then
  echo "ERROR: Python syntax preflight failed"
  exit 1
endif

pytest -q \
  tests/unittests/optimize/fixed_support_v2_restoration_test.py \
  tests/unittests/optimize/fixed_support_v2_controller_test.py \
  tests/unittests/optimize/fixed_support_v2_continuation_test.py \
  tests/unittests/optimize/fixed_support_v2_profile_test.py
if ( $status != 0 ) then
  echo "ERROR: focused v2 contract tests failed"
  exit 1
endif

python $RUNNER \
  --preflight-only \
  --cases large_water_activity128 \
  --lifecycle-families solar_water_condensation \
  --output-dir $OUTDIR
if ( $status != 0 ) then
  echo "ERROR: water128 preflight failed"
  exit 1
endif

echo "== water128 exact-state solve with restoration limit 100 =="
python $RUNNER \
  --lanes solver \
  --cases large_water_activity128 \
  --lifecycle-families solar_water_condensation \
  --output-dir $OUTDIR \
  --epsilon-schedule -11 -13 -15 -17 \
  --max-normal-iterations 1000 \
  --max-line-search-trials 20 \
  --max-restoration-calls 2 \
  --max-restoration-iterations 100 \
  --max-restoration-line-search-trials 20 \
  --v1-residual-tolerance-multiplier 2 \
  --budget-relative-floor 1e-6 \
  --support-closure-tolerance 1e-8
if ( $status != 0 ) then
  echo "ERROR: water128 GPU verification failed"
  exit 1
endif

echo "== outputs =="
date
ls -lh $OUTDIR/preflight.json $OUTDIR/summary.json $OUTDIR/summary.md
sha256sum $OUTDIR/preflight.json $OUTDIR/summary.json $OUTDIR/summary.md

echo "== done =="
date
