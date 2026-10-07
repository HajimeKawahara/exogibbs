#!/bin/csh -f
# Run fresh documented examples, including external comparisons and full NUTS.
# Scheduler copies should set EXOGIBBS_REPOSITORY_ROOT to the shared checkout.
set REPOSITORY_ROOT = "$0:h/../.."
if ( $?EXOGIBBS_REPOSITORY_ROOT ) then
  set REPOSITORY_ROOT = "$EXOGIBBS_REPOSITORY_ROOT"
endif
cd "$REPOSITORY_ROOT"
if ( $status != 0 ) exit 1
if ( ! -f benchmarks/documented_examples/run_all.py ) then
  echo "ERROR: EXOGIBBS_REPOSITORY_ROOT must identify the ExoGibbs checkout."
  exit 1
endif
set PYTHON_COMMAND = python
if ( $?EXOGIBBS_PYTHON ) then
  set PYTHON_COMMAND = "$EXOGIBBS_PYTHON"
endif
if ( $?PYTHONPATH ) then
  setenv PYTHONPATH "${cwd}/src:${cwd}:${PYTHONPATH}"
else
  setenv PYTHONPATH "${cwd}/src:${cwd}"
endif
setenv PYTHONUNBUFFERED 1
# Keep arguments intact, including paths with spaces. Enforce GPU at the end.
"$PYTHON_COMMAND" -m benchmarks.documented_examples.run_all $argv:q --platform gpu
exit $status
