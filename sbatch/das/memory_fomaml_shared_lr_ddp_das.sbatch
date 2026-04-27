#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=memory_fomaml_shared_ddp
#SBATCH --partition=defq
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=/home/valperga/icil-rlbench/sbatch/das/logs/memory_fomaml_shared_ddp-%j.out
#SBATCH --error=/home/valperga/icil-rlbench/sbatch/das/logs/memory_fomaml_shared_ddp-%j.err

set -euo pipefail

PRETRAINED_CHECKPOINT="/home/valperga/icil-rlbench/REPLACE_WITH_PRETRAINED_CHECKPOINT.pt"

if [ ! -f "${PRETRAINED_CHECKPOINT}" ]; then
  echo "PRETRAINED_CHECKPOINT does not exist: ${PRETRAINED_CHECKPOINT}" >&2
  exit 1
fi

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

export WANDB_PROJECT="icil-memory-fomaml"
export WANDB_ENTITY="ricvalp"
export WANDB_MODE="online"

export PYTHONPATH=/home/valperga/icil-rlbench

torchrun --standalone --nproc_per_node=4 \
  icil/fomaml_memory_perceiver_encoder_decoder_ddp.py \
  --config=configs/fomaml_memory_perceiver_encoder_decoder.py \
  --config.finetune.pretrained_checkpoint="${PRETRAINED_CHECKPOINT}" \
  --config.finetune.strict_load=True \
  --config.train.resume_path='' \
  --config.dataset.K=0 \
  --config.data.task_sampling=variation_uniform \
  --config.data.task_sampling_alpha=0.5 \
  --config.train.num_steps=100000 \
  --config.train.batch_size=2 \
  --config.train.weight_decay=1e-4 \
  --config.train.log_every=10 \
  --config.train.ckpt_every=500 \
  --config.maml.first_order=True \
  --config.maml.inner_steps=4 \
  --config.maml.inner_lr=1e-2 \
  --config.maml.inner_lr_mode=shared_learned \
  --config.maml.outer_lr=1e-4 \
  --config.maml.max_grad_norm=1.0 \
  --config.maml.num_queries_per_step=32 \
  --config.maml.num_inner_batches=0 \
  --config.maml.num_query_loss_samples=1 \
  --config.maml.holdout_index=-1 \
  --config.maml.reuse_diffusion_noise=False \
  --config.maml.grad_accum_steps=4 \
  --config.outer.train_encoder=True \
  --config.outer.train_decoder=False \
  --config.outer.train_input_projections=False \
  --config.outer.train_output_head=False \
  --config.outer.train_diffusion_conditioning=False \
  --config.wandb.enable=True \
  --config.wandb.n_loss_steps=1
