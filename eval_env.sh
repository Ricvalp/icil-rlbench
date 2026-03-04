#!/bin/bash
# Environment variables for ICIL RLBench evaluation.
# Source before running eval scripts: source eval_env.sh

export PYTHONUNBUFFERED="1"
export COPPELIASIM_ROOT="${HOME}/CoppeliaSim"
export LD_LIBRARY_PATH="${HOME}/CoppeliaSim:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${HOME}/CoppeliaSim"
export QT_QPA_PLATFORM="xcb"
export QT_XCB_GL_INTEGRATION="xcb_glx"
export __GLX_VENDOR_LIBRARY_NAME="nvidia"
export DISPLAY=":99"

echo "[eval_env.sh] PYTHONUNBUFFERED=${PYTHONUNBUFFERED}"
echo "[eval_env.sh] COPPELIASIM_ROOT=${COPPELIASIM_ROOT}"
echo "[eval_env.sh] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "[eval_env.sh] QT_QPA_PLATFORM_PLUGIN_PATH=${QT_QPA_PLATFORM_PLUGIN_PATH}"
echo "[eval_env.sh] QT_QPA_PLATFORM=${QT_QPA_PLATFORM}"
echo "[eval_env.sh] QT_XCB_GL_INTEGRATION=${QT_XCB_GL_INTEGRATION}"
echo "[eval_env.sh] __GLX_VENDOR_LIBRARY_NAME=${__GLX_VENDOR_LIBRARY_NAME}"
echo "[eval_env.sh] DISPLAY=${DISPLAY}"
