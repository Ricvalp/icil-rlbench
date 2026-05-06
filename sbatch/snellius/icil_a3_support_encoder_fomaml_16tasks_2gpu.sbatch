#!/bin/bash
#SBATCH --job-name=icil_a3_fomaml
#SBATCH --output=/gpfs/home1/rvalperga/icil-rlbench/sbatch/snellius/logs/icil_a3_fomaml.%j.out
#SBATCH --error=/gpfs/home1/rvalperga/icil-rlbench/sbatch/snellius/logs/icil_a3_fomaml.%j.err
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

if [ ! -d "${ICIL_METAWORLD_ML45_FIXED_GOAL_RANDOM_START_CACHE_ROOT}" ]; then
  echo "MetaWorld fixed-goal/random-start cache not found: ${ICIL_METAWORLD_ML45_FIXED_GOAL_RANDOM_START_CACHE_ROOT}" >&2
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

RUN_NAME="A3-support-encoder-fomaml-16tasks-settingA-fixed-goal-random-start-${SLURM_JOB_ID}"

PYTHONPATH=. python icil_jax_query_memory/maml_metaworld_icil_object_centric_direct_regression.py \
  --config=configs/jax_metaworld_icil_a3_support_encoder_fomaml.py \
  --config.wandb.enable=True \
  --config.wandb.name="${RUN_NAME}" \
  --config.data.cache_root="${ICIL_METAWORLD_ML45_FIXED_GOAL_RANDOM_START_CACHE_ROOT}" \
  --config.data.tasks='("basketball-v3","coffee-pull-v3","coffee-push-v3","peg-insert-side-v3","pick-place-v3","pick-place-wall-v3","push-back-v3","push-v3","push-wall-v3","reach-v3","reach-wall-v3","shelf-place-v3","soccer-v3","stick-pull-v3","stick-push-v3","sweep-into-v3")' \
  --config.data.sample_same_task_name=True \
  --config.data.sample_same_task_instance=True \
  --config.data.allow_support_query_same_episode=False \
  --config.data.preload_to_memory=True \
  --config.data.num_workers=0 \
  --config.data.persistent_workers=False \
  --config.train.batch_size=32 \
  --config.train.num_steps=100000 \
  --config.train.log_every=10 \
  --config.train.ckpt_every=2000 \
  --config.maml.num_queries_per_step=32 \
  --config.maml.num_query_loss_samples=8 \
  --config.maml.num_inner_batches=0 \
  --config.maml.training_mode_metrics_only=False
