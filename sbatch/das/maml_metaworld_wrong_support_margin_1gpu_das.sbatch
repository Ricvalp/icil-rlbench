#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=mw_wrong_margin_maml
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00
#SBATCH --output=/home/valperga/icil-rlbench/sbatch/das/logs/mw_wrong_margin_maml-%j.out
#SBATCH --error=/home/valperga/icil-rlbench/sbatch/das/logs/mw_wrong_margin_maml-%j.err

set -euo pipefail
ulimit -n 4096 || true

module load cuda12.6/toolkit

set +u
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
elif [ -f "/home/valperga/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/home/valperga/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/home/valperga/miniforge3/etc/profile.d/conda.sh" ]; then
  source "/home/valperga/miniforge3/etc/profile.d/conda.sh"
else
  echo "Could not initialize conda. Checked PATH, ${HOME}/miniconda3, ${HOME}/miniforge3, /home/valperga/miniconda3, and /home/valperga/miniforge3." >&2
  exit 1
fi
conda activate jax-icil
set -u

cd /home/valperga/icil-rlbench

source env_das.sh


if [ ! -d "${ICIL_METAWORLD_ML45_GOAL_CACHE_ROOT}" ]; then
  echo "MetaWorld ML45 goal cache not found: ${ICIL_METAWORLD_ML45_GOAL_CACHE_ROOT}" >&2
  exit 1
fi

export WANDB_PROJECT="icil-jax-metaworld-query-memory"
export ICIL_METAWORLD_WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_ENTITY="ricvalp"
export WANDB_MODE="online"

export PYTHONPATH=/home/valperga/icil-rlbench

XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. python icil_jax_query_memory/maml_metaworld_query_memory_write_read_direct_regression.py \
  --config=configs/jax_metaworld_maml_query_memory_write_read_wrong_support_margin_direct_regression.py \
  --config.wandb.name=ml45-query-hidden-wrong-support-margin-maml \
  --config.wandb.enable=True \
  --config.train.batch_size=64 \
  --config.maml.num_queries_per_step=32 \
  --config.train.num_steps=100000 \
  --config.train.log_every=10 \
  --config.train.ckpt_every=1000
