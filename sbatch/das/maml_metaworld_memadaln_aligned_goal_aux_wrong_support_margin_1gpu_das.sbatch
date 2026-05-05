#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=mw_memadaln_goalaux
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00
#SBATCH --output=/home/valperga/icil-rlbench/sbatch/das/logs/mw_memadaln_goalaux-%j.out
#SBATCH --error=/home/valperga/icil-rlbench/sbatch/das/logs/mw_memadaln_goalaux-%j.err

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
  --config.wandb.name=ml45-aligned-memadaln-query-hidden-goal-aux-wrong-support-margin-maml \
  --config.wandb.enable=True \
  --config.data.cache_root="$ICIL_METAWORLD_ML45_GOAL_CACHE_ROOT" \
  --config.data.tasks='("button-press-v3","button-press-wall-v3","button-press-topdown-v3","button-press-topdown-wall-v3","coffee-button-v3","door-open-v3","door-close-v3","drawer-open-v3","drawer-close-v3","faucet-open-v3","faucet-close-v3","handle-press-v3","handle-press-side-v3","handle-pull-v3","handle-pull-side-v3","dial-turn-v3","lever-pull-v3","window-open-v3","window-close-v3")' \
  --config.data.sample_same_task_name=True \
  --config.data.sample_same_task_instance=False \
  --config.data.support_zero_goal=False \
  --config.data.query_zero_goal=True \
  --config.data.preload_to_memory=True \
  --config.data.num_workers=0 \
  --config.data.persistent_workers=False \
  --config.model.query_memory_direct_regression.write_use_support_obs=True \
  --config.model.query_memory_direct_regression.memory_conditioning_mode=adaln \
  --config.model.query_memory_direct_regression.memory_conditioning_strength=1.0 \
  --config.maml.log_attention_metrics=True \
  --config.maml.inner_lr=1e-1 \
  --config.maml.inner_steps=2 \
  --config.maml.num_queries_per_step=32 \
  --config.maml.num_query_loss_samples=8 \
  --config.maml.num_inner_batches=0 \
  --config.maml.memory_layer_norm_after_update=False \
  --config.model.query_memory_direct_regression.memory_layer_norm_after_update=False \
  --config.maml.use_read_improvement_margin=False \
  --config.maml.use_wrong_support_margin=True \
  --config.maml.wrong_support_margin=0.01 \
  --config.maml.wrong_support_margin_weight=1.0 \
  --config.maml.use_memory_contrast=False \
  --config.maml.goal_prediction_loss_weight=1.0 \
  --config.maml.goal_prediction_loss_type=mse \
  --config.maml.query_goal_dropout_rate=0.0 \
  --config.maml.training_mode_metrics_only=False \
  --config.train.batch_size=64 \
  --config.train.num_steps=200000 \
  --config.train.log_every=10 \
  --config.train.ckpt_every=2000
