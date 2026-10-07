#!/usr/bin/env bash

set -euo pipefail

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [path/to/fastchem]" >&2
    exit 2
fi

fastchem_executable=${1:-FastChem/fastchem}
if [[ ! -x "$fastchem_executable" ]]; then
    echo "FastChem executable not found: $fastchem_executable" >&2
    exit 2
fi
co_database=${EXOGIBBS_CO_DATABASE:-.databases/CO/12C-16O/Li2015}
exoeos_source=${EXOGIBBS_EXOEOS_SOURCE:-../exoeos/src}

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/exogibbs_matplotlib}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

while IFS= read -r -d '' example; do
    printf 'Running %s\n' "$example"
    if [[ "$example" == examples/retrievals/* ]]; then
        if [[ ! -d "$co_database" ]]; then
            printf 'Skipping %s: set EXOGIBBS_CO_DATABASE to run it.\n' "$example"
            continue
        fi
        NUMBA_DISABLE_JIT=1 python "$example" \
            --quick --co-database "$co_database"
    elif [[ "$example" == examples/comparisons/comparison_with_fastchem.py \
        || "$example" == examples/comparisons/comparison_with_fastchem_cond.py \
        || "$example" == examples/comparisons/comparison_with_fastchem_extended.py \
        ]] || rg -q -- '"--fastchem-executable"' "$example"; then
        python "$example" --fastchem-executable "$fastchem_executable"
    elif [[ "$example" == examples/plot_exoeos_pure_fugacity.py \
        && -d "$exoeos_source/exoeos" ]]; then
        PYTHONPATH="src:$exoeos_source${PYTHONPATH:+:$PYTHONPATH}" \
            python "$example"
    else
        python "$example"
    fi
done < <(rg --files -0 examples -g '*.py' -g '!**/_*.py' | sort -z)
