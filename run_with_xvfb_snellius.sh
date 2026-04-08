#!/bin/bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command> [args...]"
  exit 1
fi

source /gpfs/home1/rvalperga/miniforge3/etc/profile.d/conda.sh
conda activate icil-rlbench
module load 2024 Xvfb/21.1.14-GCCcore-13.3.0
source "$(dirname "$0")/env_snellius_coppeliasim.sh"

export DISPLAY="${DISPLAY:-:99}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

Xvfb "${DISPLAY}" -screen 0 1280x1024x24 >/tmp/xvfb_${USER}.log 2>&1 &
XVFB_PID=$!
cleanup() {
  kill "${XVFB_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

sleep 2
"$@"
