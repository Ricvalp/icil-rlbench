#!/bin/bash
#SBATCH --job-name=mw_wrong_memctr_maml
#SBATCH --output=/gpfs/home1/rvalperga/icil-rlbench/sbatch/snellius/logs/mw_wrong_memctr_maml.%j.out
#SBATCH --error=/gpfs/home1/rvalperga/icil-rlbench/sbatch/snellius/logs/mw_wrong_memctr_maml.%j.err
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00

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

export WANDB_PROJECT="icil-jax-metaworld-query-memory"
export ICIL_METAWORLD_WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_ENTITY="ricvalp"
export WANDB_MODE="online"

export PYTHONPATH=/gpfs/home1/rvalperga/icil-rlbench
export DISPLAY=:$((100 + SLURM_JOB_ID % 5000))

XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. python icil_jax_query_memory/maml_metaworld_query_memory_write_read_direct_regression.py \
  --config=configs/jax_metaworld_maml_query_memory_write_read_wrong_support_margin_direct_regression.py \
  --config.wandb.name=ml45-query-hidden-wrong-margin-memory-contrast-maml \
  --config.maml.use_memory_contrast=True \
  --config.maml.memory_contrast_weight=0.1 \
  --config.maml.memory_contrast_temperature=0.1 \
  --config.maml.memory_contrast_on_delta=True \
  --config.wandb.enable=True \
  --config.train.batch_size=64 \
  --config.maml.num_queries_per_step=32 \
  --config.train.num_steps=100000 \
  --config.train.log_every=10 \
  --config.train.ckpt_every=1000
