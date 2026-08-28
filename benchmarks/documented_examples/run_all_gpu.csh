#!/bin/csh -f

set REPOSITORY_ROOT = ""
if ( $?EXOGIBBS_REPOSITORY_ROOT ) then
  set REPOSITORY_ROOT = "$EXOGIBBS_REPOSITORY_ROOT"
else
  set SCRIPT_DIR = `dirname "$0"`
  set REPOSITORY_ROOT = "$SCRIPT_DIR/../.."
endif
cd "$REPOSITORY_ROOT"
if ( $status != 0 ) then
  echo "ERROR: could not enter the ExoGibbs repository root"
  exit 1
endif

if ( $#argv > 1 ) then
  echo "usage: $0 [OUTPUT_DIRECTORY]"
  exit 2
endif

set PYTHON_COMMAND = "python"
if ( $?EXOGIBBS_PYTHON ) then
  set PYTHON_COMMAND = "$EXOGIBBS_PYTHON"
endif
"$PYTHON_COMMAND" --version
if ( $status != 0 ) then
  echo "ERROR: Python executable failed: $PYTHON_COMMAND"
  exit 2
endif

if ( $?PYTHONPATH ) then
  setenv PYTHONPATH "${cwd}/src:${cwd}:${PYTHONPATH}"
else
  setenv PYTHONPATH "${cwd}/src:${cwd}"
endif
setenv PYTHONHASHSEED 0
setenv XLA_PYTHON_CLIENT_PREALLOCATE false
setenv MPLBACKEND Agg

set STAMP = `date '+%Y%m%dT%H%M%S'`
set OUTPUT_ROOT = "$cwd/results/documented_example_benchmarks/gpu_all_${STAMP}.$$"
if ( $#argv == 1 ) then
  set OUTPUT_ROOT = "$1"
endif
set DOCUMENTED_OUTPUT = "$OUTPUT_ROOT/documented_examples"
mkdir -p "$DOCUMENTED_OUTPUT"
if ( $status != 0 ) then
  echo "ERROR: could not create output directory: $OUTPUT_ROOT"
  exit 1
endif

echo "== ExoGibbs documented GPU benchmarks =="
echo "repository:   $cwd"
echo "python:       $PYTHON_COMMAND"
echo "output:       $OUTPUT_ROOT"
echo "workloads:    six documented examples, then L-dwarf cold + warm 10"
echo "optimizations: default, then disable_most_optimizations"
date

nvidia-smi
if ( $status != 0 ) then
  echo "ERROR: nvidia-smi failed"
  exit 1
endif

env \
  JAX_PLATFORMS=cuda \
  JAX_PLATFORM_NAME=cuda \
  JAX_ENABLE_X64=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$PYTHON_COMMAND" -c 'import jax; devices = jax.devices(); print("JAX backend:", jax.default_backend()); print("JAX devices:", devices); raise SystemExit(0 if jax.default_backend() == "gpu" and devices else 1)'
if ( $status != 0 ) then
  echo "ERROR: JAX did not initialize a CUDA GPU backend"
  exit 1
endif

"$PYTHON_COMMAND" -m py_compile \
  benchmarks/documented_examples/run.py \
  benchmarks/documented_examples/worker.py \
  benchmarks/documented_examples/ldwarf_repeated.py
if ( $status != 0 ) then
  echo "ERROR: benchmark syntax preflight failed"
  exit 1
endif

@ FAILURE_COUNT = 0

echo ""
echo "== [1/3] six documented examples: GPU, both compiler modes =="
date
"$PYTHON_COMMAND" -m benchmarks.documented_examples.run \
  --platform gpu \
  --optimization default \
  --optimization disable_most_optimizations \
  --repeat 1 \
  --output-directory "$DOCUMENTED_OUTPUT"
set DOCUMENTED_RC = $status
if ( $DOCUMENTED_RC != 0 ) then
  echo "ERROR: one or more documented example jobs failed"
  @ FAILURE_COUNT += 1
endif

echo ""
echo "== [2/3] L-dwarf cold + warm 10: default compiler =="
date
"$PYTHON_COMMAND" -m benchmarks.documented_examples.ldwarf_repeated \
  --platform gpu \
  --optimization default \
  --evaluations 10 \
  --output "$OUTPUT_ROOT/ldwarf_repeated_gpu_default.json"
set LDWARF_DEFAULT_RC = $status
if ( $LDWARF_DEFAULT_RC != 0 ) then
  echo "ERROR: default L-dwarf repeated benchmark failed"
  @ FAILURE_COUNT += 1
endif

echo ""
echo "== [3/3] L-dwarf cold + warm 10: compile-light =="
date
"$PYTHON_COMMAND" -m benchmarks.documented_examples.ldwarf_repeated \
  --platform gpu \
  --optimization disable_most_optimizations \
  --evaluations 10 \
  --output "$OUTPUT_ROOT/ldwarf_repeated_gpu_disable_optimizations.json"
set LDWARF_LIGHT_RC = $status
if ( $LDWARF_LIGHT_RC != 0 ) then
  echo "ERROR: compile-light L-dwarf repeated benchmark failed"
  @ FAILURE_COUNT += 1
endif

echo ""
echo "== ExoGibbs GPU benchmark sequence complete =="
echo "documented examples exit code: $DOCUMENTED_RC"
echo "L-dwarf default exit code:     $LDWARF_DEFAULT_RC"
echo "L-dwarf compile-light code:    $LDWARF_LIGHT_RC"
echo "output:                         $OUTPUT_ROOT"
date

if ( $FAILURE_COUNT != 0 ) then
  echo "ERROR: $FAILURE_COUNT benchmark stage(s) failed"
  exit 1
endif
exit 0
