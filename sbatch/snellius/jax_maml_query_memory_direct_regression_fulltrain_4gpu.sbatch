#!/bin/bash
#SBATCH --job-name=jax_query_mem_maml
#SBATCH --output=/gpfs/home1/rvalperga/icil-rlbench/sbatch/snellius/logs/jax_query_mem_maml.%j.out
#SBATCH --error=/gpfs/home1/rvalperga/icil-rlbench/sbatch/snellius/logs/jax_query_mem_maml.%j.err
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00

set -euo pipefail

set +u
source ~/miniforge3/bin/activate
conda activate jax-icil
set -u

cd /gpfs/home1/rvalperga/icil-rlbench

source env_snellius.sh
source env_snellius_coppeliasim.sh

export WANDB_PROJECT="icil-jax-query-memory-direct-maml"
export WANDB_ENTITY="ricvalp"
export WANDB_MODE="online"

export PYTHONPATH=/gpfs/home1/rvalperga/icil-rlbench
export DISPLAY=:$((100 + SLURM_JOB_ID % 5000))

python /gpfs/home1/rvalperga/icil-rlbench/icil_jax_query_memory/maml_query_memory_direct_regression.py \
  --config=configs/jax_maml_query_memory_direct_regression.py \
  --config.data.task_sampling=variation_uniform \
  --config.data.task_sampling_alpha=0.5 \
  --config.train.num_steps=100000 \
  --config.train.batch_size=4 \
  --config.train.weight_decay=1e-4 \
  --config.train.log_every=10 \
  --config.train.ckpt_every=500 \
  --config.data.num_workers=24 \
  --config.data.pin_memory=True \
  --config.data.persistent_workers=True \
  --config.data.prefetch_factor=8 \
  --config.maml.inner_steps=1 \
  --config.maml.inner_lr=1e-2 \
  --config.maml.inner_lr_mode=fixed \
  --config.maml.outer_lr=1e-4 \
  --config.maml.max_grad_norm=1.0 \
  --config.maml.num_queries_per_step=16 \
  --config.maml.num_inner_batches=0 \
  --config.maml.num_query_loss_samples=1 \
  --config.maml.reuse_diffusion_noise=False \
  --config.maml.grad_accum_steps=1 \
  --config.model.query_memory_direct_regression.d_model=256 \
  --config.model.query_memory_direct_regression.n_heads=4 \
  --config.model.query_memory_direct_regression.decoder_layers=4 \
  --config.model.query_memory_direct_regression.memory_num_tokens=64 \
  --config.model.simple_query_point_encoder.d_model=256 \
  --config.wandb.enable=True \
  --config.wandb.n_loss_steps=1
