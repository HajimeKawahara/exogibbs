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
set OUTDIR = results/fixed_support_v2_final_solver_matrix_gpu
mkdir -p $OUTDIR

echo "== fixed-support v2 final solver-matrix artifact =="
date
nvidia-smi
if ( $status != 0 ) then
  echo "ERROR: nvidia-smi failed"
  exit 1
endif

if ( ! -e results/fixed_support_v2_corrected_gpu/summary.json ) then
  echo "ERROR: corrected lifecycle artifact is missing"
  exit 1
endif
if ( ! -e results/fixed_support_v2_water128_gpu100/summary.json ) then
  echo "ERROR: focused water128 artifact is missing"
  exit 1
endif

echo "== syntax and complete v2 contracts =="
python -m py_compile \
  $RUNNER \
  src/exogibbs/optimize/fixed_support_v2/types.py \
  src/exogibbs/optimize/fixed_support_v2/controller.py \
  src/exogibbs/optimize/fixed_support_v2/continuation.py \
  src/exogibbs/optimize/fixed_support_v2/restoration.py \
  src/exogibbs/optimize/fixed_support_v2_profile.py
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
  --cases all \
  --lifecycle-families manifest \
  --output-dir $OUTDIR
if ( $status != 0 ) then
  echo "ERROR: final matrix preflight failed"
  exit 1
endif

echo "== all 10 exact-state solver cases, restoration limit 100 =="
python $RUNNER \
  --lanes solver \
  --cases all \
  --lifecycle-families manifest \
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
  echo "ERROR: final solver-matrix GPU verification failed"
  exit 1
endif

echo "== outputs =="
date
ls -lh $OUTDIR/preflight.json $OUTDIR/summary.json $OUTDIR/summary.md
sha256sum $OUTDIR/preflight.json $OUTDIR/summary.json $OUTDIR/summary.md

echo "== done =="
date
