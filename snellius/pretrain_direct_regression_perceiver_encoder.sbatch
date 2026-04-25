#!/bin/bash
#SBATCH --job-name=direct_regression_perceiver_encoder
#SBATCH --output=/gpfs/home1/rvalperga/icil-rlbench/.smanager/scripts/icil.pretrain_perceiver_encoder_decoder_ddp/logs/direct_regression_perceiver_encoder.%j.out
#SBATCH --error=/gpfs/home1/rvalperga/icil-rlbench/.smanager/scripts/icil.pretrain_perceiver_encoder_decoder_ddp/logs/direct_regression_perceiver_encoder.%j.err
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00

source ~/miniforge3/bin/activate
conda activate icil-rlbench

cd /gpfs/home1/rvalperga/icil-rlbench

source env_snellius.sh
source env_snellius_coppeliasim.sh

export WANDB_PROJECT="icil-direct-regression-pretrain"
export WANDB_ENTITY="ricvalp"
export WANDB_MODE="online"

export PYTHONPATH=/gpfs/home1/rvalperga/icil-rlbench

export DISPLAY=:$((100 + SLURM_JOB_ID % 5000))

cd /gpfs/home1/rvalperga/icil-rlbench

torchrun --standalone --nproc_per_node=4 \
  /gpfs/home1/rvalperga/icil-rlbench/icil/pretrain_direct_regression_perceiver_encoder_decoder_ddp.py \
  --config=configs/pretrain_direct_regression_perceiver_encoder_decoder.py \
  --config.data.task_sampling=variation_uniform \
  --config.model.encoder_name=traj_perceiver \
  --config.dataset.K=4 \
  --config.dataset.L=10 \
  --config.dataset.T_obs=2 \
  --config.dataset.H=16 \
  --config.dataset.stride=2 \
  --config.train.batch_size=32 \
  --config.model.direct_regression.horizon=16 \
  --config.model.direct_regression.context_attention_mode=two_ctx \
  --config.model.direct_regression.loss_type=l1 \
  --config.model.direct_regression.attention_backend=manual \
  --config.model.traj_perceiver.m_frame_tokens=128 \
  --config.model.traj_perceiver.frame_tokenizer_layers=2 \
  --config.model.traj_perceiver.M_demo_latents=256 \
  --config.model.traj_perceiver.demo_perceiver_layers=3 \
  --config.model.traj_perceiver.use_mask_id=False \
  --config.model.traj_perceiver.m_traj_tokens=16 \
  --config.model.traj_perceiver.traj_perceiver_layers=2 \
  --config.model.traj_perceiver.include_traj_tokens=True \
  --config.model.traj_perceiver.use_demo_id_embed=True \
  --config.model.traj_perceiver.use_cond_state_as_traj_fallback=False
