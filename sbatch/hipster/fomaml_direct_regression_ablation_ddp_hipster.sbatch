#!/bin/bash
#SBATCH --job-name=direct_reg_fomaml_ablation_ddp
#SBATCH --output=/home/rvalper/icil-rlbench/sbatch/hipster/logs/direct_reg_fomaml_ablation_ddp.%j.out
#SBATCH --error=/home/rvalper/icil-rlbench/sbatch/hipster/logs/direct_reg_fomaml_ablation_ddp.%j.err
#SBATCH --partition=performance
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH -n 128

set -euo pipefail

PRETRAINED_CHECKPOINT="/home/rvalper/icil-rlbench/REPLACE_WITH_PRETRAINED_CHECKPOINT.pt"

if [ ! -f "${PRETRAINED_CHECKPOINT}" ]; then
  echo "PRETRAINED_CHECKPOINT does not exist: ${PRETRAINED_CHECKPOINT}" >&2
  exit 1
fi

source ~/miniconda3/bin/activate
conda activate icil-rlbench

cd /home/rvalper/icil-rlbench

source env_hipster.sh

export WANDB_PROJECT="icil-direct-regression-fomaml"
export WANDB_ENTITY="ricvalp"
export WANDB_MODE="online"

export PYTHONPATH=/home/rvalper/icil-rlbench

torchrun --standalone --nproc_per_node=2 \
  icil/fomaml_perceiver_encoder_decoder.py \
  --config=configs/fomaml_perceiver_encoder_decoder.py \
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
  --config.maml.inner_steps=4 \
  --config.maml.inner_lr=1e-2 \
  --config.maml.inner_lr_mode=fixed \
  --config.maml.outer_lr=1e-4 \
  --config.maml.max_grad_norm=1.0 \
  --config.maml.last_frac_fast=1.0 \
  --config.maml.include_decoder_mlp_fast=True \
  --config.maml.include_ada_fast=True \
  --config.maml.include_input_projections_fast=True \
  --config.maml.include_output_head_fast=False \
  --config.maml.include_diffusion_conditioning_fast=False \
  --config.maml.num_loo_per_task=16 \
  --config.maml.outer_context_size=0 \
  --config.maml.reuse_diffusion_noise=True \
  --config.outer.train_encoder=False \
  --config.outer.train_decoder=True \
  --config.outer.train_input_projections=False \
  --config.outer.train_output_head=True \
  --config.outer.train_diffusion_conditioning=False \
  --config.wandb.enable=True \
  --config.wandb.n_loss_steps=1
