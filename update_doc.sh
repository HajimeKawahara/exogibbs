#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIRECTORY}"

python -m mceg docs --config mceg/docs.yaml
python -m sphinx -M clean documents documents/_build
python -m sphinx \
  -M html \
  documents \
  documents/_build \
  -E \
  -a \
  "$@"
