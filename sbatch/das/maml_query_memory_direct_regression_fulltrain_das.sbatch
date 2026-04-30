#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=query_mem_direct_maml
#SBATCH --partition=defq
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=/home/valperga/icil-rlbench/sbatch/das/logs/query_mem_direct_maml-%j.out
#SBATCH --error=/home/valperga/icil-rlbench/sbatch/das/logs/query_mem_direct_maml-%j.err

set -euo pipefail

module load cuda12.6/toolkit

if [ -f "${HOME}/miniconda3/bin/activate" ]; then
  source "${HOME}/miniconda3/bin/activate"
elif [ -f "${HOME}/miniforge3/bin/activate" ]; then
  source "${HOME}/miniforge3/bin/activate"
else
  echo "Could not find a conda activation script under ${HOME}/miniconda3 or ${HOME}/miniforge3." >&2
  exit 1
fi

conda activate icil-rlbench

cd /home/valperga/icil-rlbench

source env_das.sh

export WANDB_PROJECT="icil-query-memory-direct-maml"
export WANDB_ENTITY="ricvalp"
export WANDB_MODE="online"

export PYTHONPATH=/home/valperga/icil-rlbench

torchrun --standalone --nproc_per_node=1 \
  icil/maml_query_memory_direct_regression_ddp.py \
  --config=configs/maml_query_memory_direct_regression.py \
  --config.data.task_sampling=variation_uniform \
  --config.data.task_sampling_alpha=0.5 \
  --config.train.num_steps=100000 \
  --config.train.batch_size=8 \
  --config.train.weight_decay=1e-4 \
  --config.train.log_every=10 \
  --config.train.ckpt_every=100 \
  --config.data.num_workers=4 \
  --config.data.pin_memory=True \
  --config.data.persistent_workers=True \
  --config.maml.first_order=False \
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
