#!/bin/bash
#SBATCH --job-name=icil_b3_maml
#SBATCH --output=/gpfs/home1/rvalperga/icil-rlbench/sbatch/snellius/logs/icil_b3_maml.%j.out
#SBATCH --error=/gpfs/home1/rvalperga/icil-rlbench/sbatch/snellius/logs/icil_b3_maml.%j.err
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

set -euo pipefail
ulimit -n 4096 || true

set +u
source ~/miniforge3/bin/activate
conda activate jax-icil
set -u

cd /gpfs/home1/rvalperga/icil-rlbench

source env_snellius.sh

if [ ! -d "${ICIL_METAWORLD_ML45_GOAL_CACHE_ROOT}" ]; then
  echo "MetaWorld ML45 goal cache not found: ${ICIL_METAWORLD_ML45_GOAL_CACHE_ROOT}" >&2
  exit 1
fi

export WANDB_PROJECT="${ICIL_OBJECTCENTRIC_WANDB_PROJECT:-icil-metaworld-objectcentric-maml}"
export ICIL_OBJECTCENTRIC_WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_ENTITY="ricvalp"
export WANDB_MODE="online"
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=/gpfs/home1/rvalperga/icil-rlbench
export DISPLAY=:$((100 + SLURM_JOB_ID % 5000))

RUN_NAME="B3-support-encoder-full-maml-optional-ml45-train-settingB-${SLURM_JOB_ID}"

PYTHONPATH=. python icil_jax_query_memory/maml_metaworld_icil_object_centric_direct_regression.py \
  --config=configs/jax_metaworld_icil_b3_support_encoder_full_maml.py \
  --config.wandb.enable=True \
  --config.wandb.name="${RUN_NAME}" \
  --config.data.cache_root="${ICIL_METAWORLD_ML45_GOAL_CACHE_ROOT}" \
  --config.data.tasks='()' \
  --config.data.sample_same_task_name=True \
  --config.data.sample_same_task_instance=False \
  --config.data.support_zero_goal=False \
  --config.data.query_zero_goal=False \
  --config.data.preload_to_memory=True \
  --config.data.num_workers=0 \
  --config.data.persistent_workers=False \
  --config.train.batch_size=16 \
  --config.train.num_steps=100000 \
  --config.train.log_every=10 \
  --config.train.ckpt_every=2000 \
  --config.maml.num_queries_per_step=32 \
  --config.maml.num_query_loss_samples=8 \
  --config.maml.num_inner_batches=0 \
  --config.maml.training_mode_metrics_only=False
