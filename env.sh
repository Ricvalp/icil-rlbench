#!/bin/bash
# Environment variables for ICIL perceiver pretraining.
# Source before running training: source env.sh

# =============================================================================
# DATA DIRECTORIES
# =============================================================================

# Root of cached RLBench dense H5 variations.
export ICIL_CACHE_ROOT="/mnt/external_storage/robotics/rlbench/icil_rlbench/.rlbench_cache_dense_v4"

# Root of generated MetaWorld ICIL caches.
export ICIL_METAWORLD_DATA_ROOT="/mnt/external_storage/robotics/metaworld/icil_metaworld"

# Default MetaWorld cache used by the JAX MetaWorld configs.
export ICIL_METAWORLD_CACHE_ROOT="${ICIL_METAWORLD_DATA_ROOT}/button_press_ml1_train"

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

# Parent directory for run outputs (each run uses a subdirectory named by wandb run id).
export ICIL_OUTPUT_PARENT_DIR="/mnt/external_storage/robotics/rlbench/icil_runs/outputs"

# Parent directory for checkpoints (each run uses a subdirectory named by wandb run id).
export ICIL_CHECKPOINT_PARENT_DIR="/mnt/external_storage/robotics/rlbench/icil_runs/checkpoints"

# MetaWorld-specific run outputs and checkpoints.
export ICIL_METAWORLD_OUTPUT_PARENT_DIR="/mnt/external_storage/robotics/metaworld/icil_runs/outputs"
export ICIL_METAWORLD_CHECKPOINT_PARENT_DIR="/mnt/external_storage/robotics/metaworld/icil_runs/checkpoints"

# =============================================================================
# PROFILING OUTPUT DIRECTORIES
# =============================================================================

# Base directory for all profiling traces.
export ICIL_PROFILE_TRACE_DIR="/mnt/external_storage/robotics/rlbench/icil_runs/profiles"

# Distinct trace file names within the same profiling directory.
export ICIL_PRETRAIN_PROFILE_TRACE_FILE="pretrain_trace.json"
export ICIL_TTT_PROFILE_TRACE_FILE="ttt_trace.json"

# =============================================================================
# WANDB
# =============================================================================

export WANDB_PROJECT="icil-perceiver-pretrain"
export WANDB_ENTITY="ricvalp"
export WANDB_MODE="online"

# Used by the JAX MetaWorld configs. This prevents sourcing env.sh from sending
# MetaWorld runs to the RLBench pretraining project above.
export ICIL_METAWORLD_WANDB_PROJECT="icil-jax-metaworld-query-memory"

echo "[env.sh] ICIL_CACHE_ROOT=${ICIL_CACHE_ROOT}"
echo "[env.sh] ICIL_METAWORLD_DATA_ROOT=${ICIL_METAWORLD_DATA_ROOT}"
echo "[env.sh] ICIL_METAWORLD_CACHE_ROOT=${ICIL_METAWORLD_CACHE_ROOT}"
echo "[env.sh] ICIL_OUTPUT_PARENT_DIR=${ICIL_OUTPUT_PARENT_DIR}"
echo "[env.sh] ICIL_CHECKPOINT_PARENT_DIR=${ICIL_CHECKPOINT_PARENT_DIR}"
echo "[env.sh] ICIL_METAWORLD_OUTPUT_PARENT_DIR=${ICIL_METAWORLD_OUTPUT_PARENT_DIR}"
echo "[env.sh] ICIL_METAWORLD_CHECKPOINT_PARENT_DIR=${ICIL_METAWORLD_CHECKPOINT_PARENT_DIR}"
echo "[env.sh] ICIL_PROFILE_TRACE_DIR=${ICIL_PROFILE_TRACE_DIR}"
echo "[env.sh] ICIL_PRETRAIN_PROFILE_TRACE_FILE=${ICIL_PRETRAIN_PROFILE_TRACE_FILE}"
echo "[env.sh] ICIL_TTT_PROFILE_TRACE_FILE=${ICIL_TTT_PROFILE_TRACE_FILE}"
echo "[env.sh] WANDB_PROJECT=${WANDB_PROJECT}"
echo "[env.sh] ICIL_METAWORLD_WANDB_PROJECT=${ICIL_METAWORLD_WANDB_PROJECT}"
echo "[env.sh] WANDB_ENTITY=${WANDB_ENTITY}"
echo "[env.sh] WANDB_MODE=${WANDB_MODE}"
