#!/bin/tcsh -f

set SCRIPT_DIR = `dirname "$0"`
cd "$SCRIPT_DIR/../.."
if ( $status != 0 ) then
  echo "ERROR: could not enter the ExoGibbs repository root"
  exit 1
endif

if ( $#argv < 1 ) then
  echo "usage: $0 CASE [CO_DATABASE] [demo options]"
  echo "CASE: gas_no_grid | gas_grid | condensate_fixed_support | condensate_grid"
  echo "CO_DATABASE may instead be set with EXOJAX_CO_DATABASE."
  exit 2
endif

set CASE_NAME = "$1"
shift argv
switch ( "$CASE_NAME" )
case gas_no_grid:
  set RUNNER = examples/retrievals/exojax_nuts_gas_no_grid.py
  breaksw
case gas_grid:
  set RUNNER = examples/retrievals/exojax_nuts_gas_grid.py
  breaksw
case condensate_fixed_support:
  set RUNNER = examples/retrievals/exojax_nuts_condensate_fixed_support.py
  breaksw
case condensate_grid:
  set RUNNER = examples/retrievals/exojax_nuts_condensate_grid.py
  breaksw
default:
  echo "ERROR: unknown CASE '$CASE_NAME'"
  echo "expected gas_no_grid, gas_grid, condensate_fixed_support, or condensate_grid"
  exit 2
endsw

set CO_DATABASE = ""
if ( $?EXOJAX_CO_DATABASE ) then
  set CO_DATABASE = "$EXOJAX_CO_DATABASE"
endif
if ( $#argv > 0 ) then
  if ( "$1" !~ --* ) then
    set CO_DATABASE = "$1"
    shift argv
  endif
endif
if ( "$CO_DATABASE" == "" ) then
  echo "ERROR: provide the exact CO/12C-16O/Li2015 directory as CO_DATABASE"
  echo "or set EXOJAX_CO_DATABASE before launching the job"
  exit 2
endif
if ( ! -d "$CO_DATABASE" ) then
  echo "ERROR: CO database directory does not exist: $CO_DATABASE"
  exit 2
endif
if ( ! -f "$RUNNER" ) then
  echo "ERROR: retrieval demo is missing: $RUNNER"
  exit 2
endif

if ( $?PYTHONPATH ) then
  setenv PYTHONPATH "$cwd/src:$PYTHONPATH"
else
  setenv PYTHONPATH "$cwd/src"
endif
setenv JAX_ENABLE_X64 1
setenv JAX_PLATFORMS cuda
setenv JAX_PLATFORM_NAME cuda
setenv XLA_PYTHON_CLIENT_PREALLOCATE false
setenv PYTHONHASHSEED 0
setenv MPLBACKEND Agg
# The locally installed RADIS package otherwise attempts to cache Numba code
# inside its read-only egg. This does not disable JAX compilation.
setenv NUMBA_DISABLE_JIT 1

set OUTPUT_ROOT = results/vjp_retrieval
if ( $?EXOGIBBS_VJP_OUTPUT_ROOT ) then
  set OUTPUT_ROOT = "$EXOGIBBS_VJP_OUTPUT_ROOT"
endif
set OUTDIR = "$OUTPUT_ROOT/$CASE_NAME"
mkdir -p "$OUTDIR/matplotlib"
if ( $status != 0 ) then
  echo "ERROR: could not create output directory: $OUTDIR"
  exit 1
endif
if ( "$OUTDIR" =~ /* ) then
  setenv MPLCONFIGDIR "$OUTDIR/matplotlib"
else
  setenv MPLCONFIGDIR "$cwd/$OUTDIR/matplotlib"
endif

echo "== ExoJAX + ExoGibbs NUTS VJP retrieval =="
echo "case:        $CASE_NAME"
echo "runner:      $RUNNER"
echo "CO database: $CO_DATABASE"
echo "output:      $OUTDIR"
if ( "$CASE_NAME" == "condensate_fixed_support" || "$CASE_NAME" == "condensate_grid" ) then
  echo "model:       FastChem4 gas + reduced C(s)-only condensate catalog"
  echo "             (not full-catalog condensate equilibrium)"
endif
if ( "$CASE_NAME" == "condensate_grid" ) then
  echo "initializer: runtime FastChem4 fixed-support condensate grids"
endif
date

nvidia-smi
if ( $status != 0 ) then
  echo "ERROR: nvidia-smi failed"
  exit 1
endif

python -c 'import jax; devices=jax.devices(); print("JAX backend:", jax.default_backend()); print("JAX devices:", devices); raise SystemExit(0 if jax.default_backend() == "gpu" and devices else 1)'
if ( $status != 0 ) then
  echo "ERROR: JAX did not initialize a CUDA GPU backend"
  exit 1
endif

python "$RUNNER" \
  --co-database "$CO_DATABASE" \
  --output-dir "$OUTDIR" \
  --preflight-only \
  $argv:q
if ( $status != 0 ) then
  echo "ERROR: retrieval preflight failed"
  exit 1
endif

python "$RUNNER" \
  --co-database "$CO_DATABASE" \
  --output-dir "$OUTDIR" \
  --num-warmup 500 \
  --num-samples 1000 \
  --seed 0 \
  $argv:q
if ( $status != 0 ) then
  echo "ERROR: retrieval run failed"
  exit 1
endif

echo "== completed: $CASE_NAME =="
date
find "$OUTDIR" -maxdepth 1 -type f -ls
