#!/bin/bash
#SBATCH --job-name=memory_fomaml_perstep_i4uve44g
#SBATCH --output=/home/rvalper/icil-rlbench/hipster/logs/%x.%j.out
#SBATCH --error=/home/rvalper/icil-rlbench/hipster/logs/%x.%j.err
#SBATCH --partition=performance
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH -n 128

source ~/miniconda3/bin/activate
conda activate icil-rlbench

cd /home/rvalper/icil-rlbench

source env_hipster.sh

export WANDB_PROJECT="icil-memory-fomaml"
export WANDB_ENTITY="ricvalp"
export WANDB_MODE="online"

export PYTHONPATH=/home/rvalper/icil-rlbench

torchrun --standalone --nproc_per_node=2 \
  icil/fomaml_memory_perceiver_encoder_decoder_ddp.py \
  --config=configs/fomaml_memory_perceiver_encoder_decoder.py \
  --config.finetune.pretrained_checkpoint=/home/rvalper/icil-rlbench/pretrained_checkpoints/i4uve44g/step_0561000.pt \
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
  --config.maml.inner_lr_mode=per_step_learned \
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