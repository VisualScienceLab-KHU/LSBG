#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$REPO_ROOT/test.py" ]]; then
    if [[ -f "$PWD/test.py" && -f "$PWD/regflow/config.py" ]]; then
        REPO_ROOT="$PWD"
    else
        REPO_ROOT="${SLURM_SUBMIT_DIR:-$REPO_ROOT}"
    fi
fi
if [[ ! -f "$REPO_ROOT/test.py" || ! -f "$REPO_ROOT/regflow/config.py" ]]; then
    printf '%s\n' 'Run from the ReGFLoW repository root, or set sbatch --chdir to it.' >&2
    exit 2
fi
cd "$REPO_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "${PYTHON:-python}" -u test.py "$@"
