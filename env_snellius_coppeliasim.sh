#!/bin/bash
# Source after activating the icil-rlbench conda env.

export COPPELIASIM_ROOT="/gpfs/home1/rvalperga/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

if [[ -z "${DISPLAY:-}" ]]; then
  echo "[env_snellius_coppeliasim.sh] DISPLAY is not set."
  echo "[env_snellius_coppeliasim.sh] Set DISPLAY to a valid X server before RLBench eval."
else
  echo "[env_snellius_coppeliasim.sh] DISPLAY=${DISPLAY}"
fi

echo "[env_snellius_coppeliasim.sh] COPPELIASIM_ROOT=${COPPELIASIM_ROOT}"
echo "[env_snellius_coppeliasim.sh] QT_QPA_PLATFORM=${QT_QPA_PLATFORM}"
