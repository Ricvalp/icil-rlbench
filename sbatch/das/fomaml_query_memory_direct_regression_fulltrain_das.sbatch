#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=query_mem_direct_fomaml
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/valperga/icil-rlbench/sbatch/das/logs/query_mem_direct_fomaml-%j.out
#SBATCH --error=/home/valperga/icil-rlbench/sbatch/das/logs/query_mem_direct_fomaml-%j.err

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
conda activate icil-rlbench
set -u

cd /home/valperga/icil-rlbench

source env_das.sh

export WANDB_PROJECT="icil-query-memory-direct-fomaml"
export WANDB_ENTITY="ricvalp"
export WANDB_MODE="online"

export PYTHONPATH=/home/valperga/icil-rlbench

torchrun --standalone --nproc_per_node=1 \
  icil/fomaml_query_memory_direct_regression_ddp.py \
  --config=configs/fomaml_query_memory_direct_regression.py \
  --config.data.task_sampling=variation_uniform \
  --config.data.task_sampling_alpha=0.5 \
  --config.train.num_steps=100000 \
  --config.train.batch_size=2 \
  --config.train.weight_decay=1e-4 \
  --config.train.log_every=10 \
  --config.train.ckpt_every=200 \
  --config.data.num_workers=2 \
  --config.data.pin_memory=False \
  --config.data.persistent_workers=True \
  --config.maml.first_order=True \
  --config.maml.inner_steps=2 \
  --config.maml.inner_lr=1e-2 \
  --config.maml.inner_lr_mode=fixed \
  --config.maml.outer_lr=1e-4 \
  --config.maml.max_grad_norm=1.0 \
  --config.maml.num_queries_per_step=32 \
  --config.maml.num_inner_batches=0 \
  --config.maml.num_query_loss_samples=1 \
  --config.maml.reuse_diffusion_noise=False \
  --config.maml.grad_accum_steps=1 \
  --config.wandb.enable=True \
  --config.wandb.n_loss_steps=1
