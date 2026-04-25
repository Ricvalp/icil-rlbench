#!/bin/bash
#SBATCH --job-name=direct_regression_traj_perceiver_v2
#SBATCH --output=/home/rvalper/icil-rlbench/hipster/logs/direct_regression_traj_perceiver_v2.%j.out
#SBATCH --error=/home/rvalper/icil-rlbench/hipster/logs/direct_regression_traj_perceiver_v2.%j.err
#SBATCH --partition=performance
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH -n 128

source ~/miniconda3/bin/activate
conda activate icil-rlbench

cd /home/rvalper/icil-rlbench

source env_hipster.sh

export WANDB_PROJECT="icil-direct-regression-pretrain"
export WANDB_ENTITY="ricvalp"
export WANDB_MODE="online"

export PYTHONPATH=/home/rvalper/icil-rlbench

cd /home/rvalper/icil-rlbench

torchrun --standalone --nproc_per_node=4 \
  icil/pretrain_direct_regression_perceiver_encoder_decoder_ddp.py \
  --config=configs/pretrain_direct_regression_perceiver_encoder_decoder.py \
  --config.data.task_sampling=variation_uniform \
  --config.model.encoder_name=traj_perceiver_v2 \
  --config.dataset.K=2 \
  --config.dataset.L=10 \
  --config.dataset.T_obs=2 \
  --config.dataset.H=16 \
  --config.dataset.stride=2 \
  --config.train.batch_size=32 \
  --config.model.direct_regression.horizon=16 \
  --config.model.direct_regression.context_attention_mode=two_ctx \
  --config.model.direct_regression.loss_type=l1 \
  --config.model.direct_regression.attention_backend=manual \
  --config.model.traj_perceiver_v2.use_mask_id=False \
  --config.model.traj_perceiver_v2.include_traj_tokens=True \
  --config.model.traj_perceiver_v2.use_demo_id_embed=True \
  --config.model.traj_perceiver_v2.use_cond_state_as_traj_fallback=False
