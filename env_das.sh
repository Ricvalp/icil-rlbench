#!/bin/bash
# Environment variables for ICIL perceiver pretraining.
# Source before running training: source env.sh

# =============================================================================
# DATA DIRECTORIES
# =============================================================================

# Root of cached RLBench dense H5 variations.
export ICIL_CACHE_ROOT="/var/scratch/valperga/robotics/rlbench/icil_rlbench/cached/.rlbench_cache_dense_v3/"

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

# Parent directory for run outputs (each run uses a subdirectory named by wandb run id).
export ICIL_OUTPUT_PARENT_DIR="/var/scratch/valperga/robotics/rlbench/icil_runs/outputs"

# Parent directory for checkpoints (each run uses a subdirectory named by wandb run id).
export ICIL_CHECKPOINT_PARENT_DIR="/var/scratch/valperga/robotics/rlbench/icil_runs/checkpoints"

# =============================================================================
# PROFILING OUTPUT DIRECTORIES
# =============================================================================

# Base directory for all profiling traces.
export ICIL_PROFILE_TRACE_DIR="/var/scratch/valperga/robotics/rlbench/icil_runs/profiles"

# Distinct trace file names within the same profiling directory.
export ICIL_PRETRAIN_PROFILE_TRACE_FILE="pretrain_trace.json"
export ICIL_TTT_PROFILE_TRACE_FILE="ttt_trace.json"

# =============================================================================
# WANDB
# =============================================================================

export WANDB_PROJECT="icil-perceiver-pretrain"
export WANDB_ENTITY="ricvalp"
export WANDB_MODE="online"

echo "[env.sh] ICIL_CACHE_ROOT=${ICIL_CACHE_ROOT}"
echo "[env.sh] ICIL_OUTPUT_PARENT_DIR=${ICIL_OUTPUT_PARENT_DIR}"
echo "[env.sh] ICIL_CHECKPOINT_PARENT_DIR=${ICIL_CHECKPOINT_PARENT_DIR}"
echo "[env.sh] ICIL_PROFILE_TRACE_DIR=${ICIL_PROFILE_TRACE_DIR}"
echo "[env.sh] ICIL_PRETRAIN_PROFILE_TRACE_FILE=${ICIL_PRETRAIN_PROFILE_TRACE_FILE}"
echo "[env.sh] ICIL_TTT_PROFILE_TRACE_FILE=${ICIL_TTT_PROFILE_TRACE_FILE}"
echo "[env.sh] WANDB_PROJECT=${WANDB_PROJECT}"
echo "[env.sh] WANDB_ENTITY=${WANDB_ENTITY}"
echo "[env.sh] WANDB_MODE=${WANDB_MODE}"
