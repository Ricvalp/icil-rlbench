#!/bin/bash
#SBATCH --job-name=query_mem_direct_maml_4gpu
#SBATCH --output=/gpfs/home1/rvalperga/icil-rlbench/sbatch/snellius/logs/query_mem_direct_maml_4gpu.%j.out
#SBATCH --error=/gpfs/home1/rvalperga/icil-rlbench/sbatch/snellius/logs/query_mem_direct_maml_4gpu.%j.err
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00

set -euo pipefail

set +u
source ~/miniforge3/bin/activate
conda activate icil-rlbench
set -u

cd /gpfs/home1/rvalperga/icil-rlbench

source env_snellius.sh
source env_snellius_coppeliasim.sh

export WANDB_PROJECT="icil-query-memory-direct-maml"
export WANDB_ENTITY="ricvalp"
export WANDB_MODE="online"

export PYTHONPATH=/gpfs/home1/rvalperga/icil-rlbench
export DISPLAY=:$((100 + SLURM_JOB_ID % 5000))

torchrun --standalone --nproc_per_node=4 \
  /gpfs/home1/rvalperga/icil-rlbench/icil/maml_query_memory_direct_regression_ddp.py \
  --config=configs/maml_query_memory_direct_regression.py \
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
