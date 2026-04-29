#!/bin/bash
#SBATCH --job-name=query_mem_direct_fomaml
#SBATCH --output=/home/rvalper/icil-rlbench/sbatch/hipster/logs/query_mem_direct_fomaml.%j.out
#SBATCH --error=/home/rvalper/icil-rlbench/sbatch/hipster/logs/query_mem_direct_fomaml.%j.err
#SBATCH --partition=performance
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH -n 32

set -euo pipefail

source ~/miniconda3/bin/activate
conda activate icil-rlbench

cd /home/rvalper/icil-rlbench

source env_hipster.sh

export WANDB_PROJECT="icil-query-memory-direct-fomaml"
export WANDB_ENTITY="ricvalp"
export WANDB_MODE="online"

export PYTHONPATH=/home/rvalper/icil-rlbench

torchrun --standalone --nproc_per_node=1 \
  icil/fomaml_query_memory_direct_regression_ddp.py \
  --config=configs/fomaml_query_memory_direct_regression.py \
  --config.data.task_sampling=variation_uniform \
  --config.data.task_sampling_alpha=0.5 \
  --config.train.num_steps=100000 \
  --config.train.batch_size=2 \
  --config.train.weight_decay=1e-4 \
  --config.train.log_every=10 \
  --config.train.ckpt_every=500 \
  --config.data.num_workers=4 \
  --config.data.pin_memory=True \
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
